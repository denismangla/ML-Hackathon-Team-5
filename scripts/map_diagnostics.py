"""Plot actual GPS trip distributions, candidate coverage sweeps and local road geometry."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from pathsnap.data import geometry, load_reference, load_trip
from pathsnap.features import build_features

OUT = Path('results/figures')
RADII = (5, 10, 15, 20, 30, 40, 50, 75, 100)
TOP_K = (1, 2, 3, 5, 8, 10, 15, 20)


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distribution_and_coverage(raw):
    distributions, curves = [], []
    for split in ('train','student_test'):
        hits_radius = np.zeros(len(RADII),dtype=int)
        hits_k = np.zeros(len(TOP_K),dtype=int)
        population = 0
        for folder in sorted((raw/split).iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            trip = load_trip(folder)
            _, points, _, _, _ = geometry(trip)
            elapsed = np.diff(trip.track[:,2])
            movement = np.linalg.norm(np.diff(points,axis=0),axis=1)
            distributions.append(dict(split=split,trip_id=trip.trip_id,gps_points=len(points),arcs=len(trip.arcs),
                                      extent_km=np.linalg.norm(np.ptp(points,axis=0))/1000,
                                      regular_1s_pct=100*np.mean(np.isclose(elapsed,1.0)) if len(elapsed) else 0,
                                      jumps_over_60_mps=np.sum((elapsed>0)&(movement>60*elapsed))))
            f = build_features(trip,k=20,radius_m=100.)
            with np.load(Path('data_stage1/labels')/split/(folder.name+'.npz')) as labels:
                obs = f['observation_index']
                starts = np.r_[0,np.flatnonzero(np.diff(obs))+1]
                ranks = np.arange(len(obs)) - starts[obs] + 1
                reference = f['candidate_arc_id']==labels['reference_arc_id'][obs]
                distances = f['X'][:,0]
                hits_radius += [np.sum(reference & (distances<=r)) for r in RADII]
                hits_k += [np.sum(reference & (distances<=100.) & (ranks<=k)) for k in TOP_K]
                population += len(labels['reference_arc_id'])
            print('Geometry diagnostics',split,folder.name,flush=True)
        for kind,sizes,scores in [('radius_m',RADII,hits_radius),('max_k',TOP_K,hits_k)]:
            curves.extend(dict(split=split,kind=kind,setting=value,observations=population,
                               hits=int(hit),miss_pct=100*(1-hit/population)) for value,hit in zip(sizes,scores))
    return distributions,curves


def plot_distributions(records):
    fig,axs=plt.subplots(2,2,figsize=(11,7.5))
    histograms=[('gps_points','GPS observations per trip','GPS observations'),
                ('arcs','Road-network size','Directed arcs'),
                ('extent_km','Geographic extent','Bounding-box diagonal (km)')]
    for split,color in [('train','#126e62'),('student_test','#c05e27')]:
        rows=[r for r in records if r['split']==split]
        for ax,(key,title,xlabel) in zip(axs.flat,histograms):
            values=np.array([r[key] for r in rows])
            overall=np.array([r[key] for r in records])
            bins=np.geomspace(max(1,overall.min()),overall.max(),18) if key=='arcs' else np.linspace(0,overall.max(),18)
            ax.hist(values,bins=bins,alpha=.52,label=split,color=color)
            if key=='arcs':
                ax.set_xscale('log')
            ax.set(title=title,xlabel=xlabel,ylabel='Trips')
        axs[1,1].scatter([r['regular_1s_pct'] for r in rows],[r['jumps_over_60_mps'] for r in rows],s=20,alpha=.72,label=split,color=color)
    axs[1,1].set(title='Sampling regularity and GPS jumps',xlabel='Consecutive timestamps 1 second apart (%)',ylabel='Steps over 60 m/s')
    for ax in axs.flat:
        ax.grid(alpha=.16)
        ax.legend(frameon=False)
    fig.tight_layout(pad=1.9);fig.savefig(OUT/'Dataset_Geometry_and_Sampling.png',dpi=175);plt.close(fig)


def plot_coverage(records):
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,kind,title,xlabel in zip(axs,['radius_m','max_k'],
            ['Miss rate at different search radii','Miss rate at different candidate limits'],
            ['Radius (m) · K fixed at 20','Maximum candidates · radius fixed at 100 m']):
        for split,color in [('train','#126e62'),('student_test','#c05e27')]:
            rows=[r for r in records if r['kind']==kind and r['split']==split]
            ax.plot([r['setting'] for r in rows],[r['miss_pct'] for r in rows],'o-',label=split,color=color)
        ax.set(title=title,xlabel=xlabel,ylabel='Derived reference missing (%)')
        ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.text(.5,.015,'Diagnostic search uses the derived labels and strict distance bounds, with no nearest-road fallback.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'Candidate_Convergence.png',dpi=175);plt.close(fig)


def plot_trip(raw):
    folder=raw/'student_test'/'00000009'
    trip=load_trip(folder)
    reference=load_reference(folder,len(trip.arcs))
    _,points,starts,ends,origin=geometry(trip)
    f=build_features(trip)
    index=min(299,len(points)-1)
    candidates=f['candidate_arc_id'][f['observation_index']==index]
    with np.load('results/decoder/00000009.npz') as prediction:
        selected=int(prediction['predicted_arc_id'][index])
    edges=np.stack([starts,ends],axis=1)
    lo,hi=points.min(axis=0)-65,points.max(axis=0)+65
    visible=((edges.max(axis=1)>=lo)&(edges.min(axis=1)<=hi)).all(axis=1)
    fig,(wide,zoom)=plt.subplots(1,2,figsize=(11,5.4))
    wide.add_collection(LineCollection(edges[visible],colors='#bfc7be',linewidths=.52,alpha=.7))
    gps=wide.scatter(*points.T,c=np.arange(1,len(points)+1),s=5,cmap='viridis',zorder=3)
    wide.scatter(*points[index],s=38,c='#ae35b4',edgecolors='white',linewidth=.6,zorder=4)
    wide.set_xlim(lo[0],hi[0]);wide.set_ylim(lo[1],hi[1])
    wide.set_title('GPS trajectory in observation order')
    fig.colorbar(gps,ax=wide,label='GPS observation')
    centre=points[index]
    zoom.add_collection(LineCollection(edges[reference],colors='#c05e27',linestyles='dashed',linewidths=2.2,label='Reference'))
    zoom.add_collection(LineCollection(edges[candidates],colors='#9a57b6',linewidths=2.5,alpha=.7,label='Candidate roads'))
    zoom.add_collection(LineCollection(edges[[selected]],colors='#08786c',linewidths=3.4,label='Selected road'))
    start,end=max(0,index-13),min(len(points),index+14)
    zoom.scatter(*points[start:end].T,s=12,c='#2856a0',label='Nearby GPS samples',zorder=5)
    zoom.scatter(*centre,s=42,marker='x',c='#161f23',label=f'GPS #{index+1}',zorder=6)
    zoom.set_xlim(centre[0]-115,centre[0]+115);zoom.set_ylim(centre[1]-115,centre[1]+115)
    zoom.set_title(f'Candidate geometry at observation {index+1}')
    zoom.legend(loc='best',framealpha=.92,fontsize=8)
    for ax in (wide,zoom):
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlabel='Local east (m)',ylabel='Local north (m)')
    fig.suptitle(f'Student-test trip {trip.trip_id} · {len(points):,} GPS observations')
    fig.tight_layout();fig.savefig(OUT/'GPS_Trajectory_and_Candidate_Zoom.png',dpi=185);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    plot_trip(args.raw)
    distributions,curves=distribution_and_coverage(args.raw)
    write_csv(OUT/'geometry_statistics.csv',distributions)
    write_csv(OUT/'candidate_convergence.csv',curves)
    plot_distributions(distributions)
    plot_coverage(curves)


if __name__=='__main__':
    main()
