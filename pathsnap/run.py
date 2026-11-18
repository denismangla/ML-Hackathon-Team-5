import argparse,csv,json,hashlib,platform,time
from pathlib import Path
import numpy as np
import scipy
from .data import load_trip,load_reference,quality
from .features import build_features,FEATURE_NAMES
from .alignment import align_reference

def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',type=Path,default=Path('results'))
    ap.add_argument('--limit',type=int);ap.add_argument('--k',type=int,default=8);ap.add_argument('--radius',type=float,default=100);ap.add_argument('--gap',type=float,default=120)
    args=ap.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=time.time(); inventory=[];quality_rows=[];coverage=[]; hashes={}; train_ids=[]; test_ids=[]
    for split in ['train','student_test']:
        folders=sorted(p for p in (args.data/split).iterdir() if p.is_dir())
        if args.limit:folders=folders[:args.limit]
        for folder in folders:
            t0=time.time();trip=load_trip(folder);route=load_reference(folder,len(trip.arcs))
            (train_ids if split=='train' else test_ids).append(trip.trip_id)
            for ext in ['nodes','arcs','track','route']:
                p=folder/f'{trip.trip_id}.{ext}';raw=p.read_bytes(); key=str(p.relative_to(args.data))
                inventory.append(dict(split=split,trip_id=trip.trip_id,file_type=ext,bytes=len(raw),rows=len(raw.splitlines())))
                hashes[key]=hashlib.sha256(raw).hexdigest()
            q=quality(trip,route);q['split']=split;quality_rows.append(q)
            features=build_features(trip,args.k,args.radius,args.gap)
            labels=align_reference(trip,route,args.gap)
            oi=features['observation_index'];hit=features['candidate_arc_id']==labels['reference_arc_id'][oi]
            present=np.zeros(len(trip.track),dtype=bool);np.logical_or.at(present,oi,hit)
            usable=labels['alignment_usable']; confident=usable & ~labels['alignment_ambiguous']
            labels['candidate_match']=hit.astype(np.int8);labels['candidate_present']=present
            labels['training_row_usable']=usable[oi]
            destination=out/'features'/split; destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/f'{trip.trip_id}.npz',**features)
            destination=out/'labels'/split;destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/f'{trip.trip_id}.npz',**labels)
            coverage.append(dict(split=split,trip_id=trip.trip_id,observations=len(present),candidate_rows=len(oi),candidate_hits=int(present.sum()),candidate_misses=int((~present).sum()),candidate_miss_rate=float((~present).mean()),usable_alignment=int(usable.sum()),ambiguous_alignment=int(labels['alignment_ambiguous'].sum()),confident_alignment=int(confident.sum()),confident_hits=int((present&confident).sum()),max_alignment_distance_m=float(labels['alignment_distance_m'].max()),fallback_observations=int(len(np.unique(oi[features['X'][:,11]>0]))),seconds=round(time.time()-t0,3)))
            print(f'{split}/{trip.trip_id}: {len(trip.track)} GPS, {len(oi)} candidates, coverage={present.mean():.3f}, {time.time()-t0:.1f}s',flush=True)
    if set(train_ids)&set(test_ids): raise ValueError('Trip leakage between partitions')
    rng=np.random.default_rng(42);shuffled=np.array(train_ids);rng.shuffle(shuffled)
    folds={str(v):int(i%3) for i,v in enumerate(shuffled)}
    split_plan=dict(seed=42,training_trips=train_ids,student_test_trips=test_ids,cv_validation_fold_by_training_trip=folds,evaluator_trips=[])
    csv_write(out/'inventory.csv',inventory);csv_write(out/'data_quality.csv',quality_rows);csv_write(out/'candidate_coverage.csv',coverage)
    for name,value in [('splits.json',split_plan),('input_sha256.json',hashes)]: (out/name).write_text(json.dumps(value,indent=2))
    schema={'version':'1.0','features':FEATURE_NAMES,'feature_dtype':'float32','feature_units':['m','fraction','m','m','m','radian','s','m','m/s','bool','bool','bool','count','count'],'identifiers':['trip_id from filename','observation_index','candidate_arc_id'],'candidate_policy':{'k':args.k,'radius_m':args.radius,'gap_s':args.gap,'fallback':'top K nearest segments only when none lies within radius'},'projection':'spherical azimuthal equidistant centred at median trip longitude/latitude; straight segments in projected space','alignment':'offline monotone occurrence DP; gap resets; 100m usable threshold; alternative distinct arc within residual+5m flagged ambiguous','label_warning':'Derived alignment, not independent per-GPS ground truth. Ambiguous labels require review. Training masks exclude >100m residual but do not automatically remove ambiguity.','preprocessing':'No learned scalers fitted. Fit within training folds in Stage 2.'}
    (out/'feature_schema.json').write_text(json.dumps(schema,indent=2))
    summary={'stage':'1','trips':len(quality_rows),'files':len(inventory),'observations':sum(x['observations'] for x in coverage),'candidate_rows':sum(x['candidate_rows'] for x in coverage),'candidate_miss_rate_derived_reference':sum(x['candidate_misses'] for x in coverage)/sum(x['observations'] for x in coverage),'alignment_usable':sum(x['usable_alignment'] for x in coverage),'alignment_ambiguous':sum(x['ambiguous_alignment'] for x in coverage),'seconds':round(time.time()-started,3),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    make_plots(args.data,out,quality_rows)

def make_plots(data,out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sample=next(iter(sorted((data/'train').iterdir())))
    while not sample.is_dir(): sample=next(p for p in sorted((data/'train').iterdir()) if p.is_dir())
    t=load_trip(sample);route=load_reference(sample,len(t.arcs))
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].hist([r['observations'] for r in rows],bins=20,color='#315b78');ax[0].set(xlabel='GPS observations per trip',ylabel='Trip count',title='All supplied trips')
    ax[1].hist([r['max_gap_s'] for r in rows],bins=np.geomspace(1,200000,20),color='#7c5147');ax[1].set_xscale('log');ax[1].set(xlabel='Maximum timestamp gap (seconds)',ylabel='Trip count',title='Timestamp gap distribution')
    fig.tight_layout();fig.savefig(out/'dataset_overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    from matplotlib.collections import LineCollection
    segments=t.nodes[t.arcs]
    ax.add_collection(LineCollection(segments,colors='#cccccc',linewidths=.25))
    ax.add_collection(LineCollection(segments[route],colors='#278447',linewidths=1.2,label='Supplied reference route'))
    ax.scatter(t.track[:,0],t.track[:,1],s=2,c='#225eab',label='GPS observations')
    ax.set_xlim(t.track[:,0].min()-.01,t.track[:,0].max()+.01);ax.set_ylim(t.track[:,1].min()-.01,t.track[:,1].max()+.01)
    ax.set_aspect(1/np.cos(np.radians(t.track[:,1].mean())));ax.set(xlabel='Longitude',ylabel='Latitude',title=f'Training trip {t.trip_id}: raw GPS and reference')
    ax.legend();fig.tight_layout();fig.savefig(out/'training_trip_map.png',dpi=160);plt.close(fig)
    try:
        from .visualise import folium_map
        folium_map(t,route,out/'training_trip_folium.html')
    except ImportError:
        (out/'FOLIUM_PENDING.txt').write_text('Folium is not installed in this execution environment. Install requirements.txt and run python -m pathsnap.visualise --data PATH --output results/training_trip_folium.html. Static maps are actual generated results; mandatory Folium evidence remains pending until that command runs.')

if __name__=='__main__':main()
