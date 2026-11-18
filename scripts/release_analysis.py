"""Recompute generalisation diagnostics, ordered-route similarity and the map export."""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

from pathsnap.data import load_trip, load_reference, geometry, distances
from pathsnap.models.common import Dataset
from pathsnap.models.definitions import OWNERS

out = Path('results/release')

def changed(values):
    values = np.asarray(values, dtype=int)
    return values[np.r_[True, values[1:] != values[:-1]]].tolist()

def similarity(route, prediction):
    return SequenceMatcher(None, route, prediction, autojunk=False).ratio()

def diagnostics(raw):
    dataset = Dataset('data_stage1', raw)
    train_x, train_y, _ = dataset.training(dataset.splits['training_trips'])
    features, labels, sequences = [], [], []
    for trip in dataset.splits['student_test_trips']:
        f = dataset.get('student_test', trip)
        l = f['labels']
        keep = (l['alignment_usable'] & ~l['alignment_ambiguous'] & l['candidate_present'])[f['observation_index']]
        features.append(f['X'][keep])
        labels.append(l['candidate_match'][keep])
        folder = raw / 'student_test' / trip
        reference = changed(load_reference(folder, len(f['arcs'])))
        record = {'trip': trip}
        for stage, directory in [('classifier','tuning/student_predictions'),('decoder','decoder')]:
            selected = np.load(Path('results') / directory / (trip+'.npz'))['predicted_arc_id']
            record[stage] = similarity(reference, changed(selected))
        sequences.append(record)
    test_x, test_y = np.vstack(features), np.concatenate(labels)
    models = {}
    for name in OWNERS:
        fitted = joblib.load('artifacts/'+name+'.joblib')['model']
        predicted = fitted.predict(test_x)
        models[name] = dict(train_balanced_accuracy=balanced_accuracy_score(train_y, fitted.predict(train_x)),
                            student_balanced_accuracy=balanced_accuracy_score(test_y,predicted),
                            student_confusion_matrix=confusion_matrix(test_y,predicted).tolist())
    align = []
    for trip in ['00000054','00000033','00000011']:
        folder = raw / 'train' / trip
        t = load_trip(folder)
        _, points, start, end, _ = geometry(t)
        route = load_reference(folder,len(t.arcs))
        ix = np.unique(np.linspace(0,len(points)-1,min(len(points),200),dtype=int))
        closest = np.array([distances(points[i],start[route],end[route])[0].min() for i in ix])
        residual = dataset.get('train',trip)['labels']['alignment_distance_m'][ix]
        align.append(dict(trip=trip,sample=len(ix),median_nearest_reference_m=float(np.median(closest)),
                          median_aligned_m=float(np.median(residual)),aligned_above_100m=int((residual>100).sum())))
    return dict(fitted_rows=len(train_y), student_rows=len(test_y), models=models, sequence=sequences, alignment=align)

def export_map(raw):
    from pathsnap.visualise import folium_map
    data = json.loads((out/'sample_response.json').read_text())
    folder = raw/'student_test'/data['trip_id']
    trip = load_trip(folder)
    view = folium_map(trip,load_reference(folder,len(trip.arcs)),out/'PathSnap_Folium.html',data['route_lines_lonlat'],data.get('pointwise_segments_lonlat'))
    groups = {child.layer_name:child for child in view._children.values() if child.__class__.__name__=='FeatureGroup'}
    gps = np.array([p.location[::-1] for p in groups['GPS observations']._children.values()])
    fig, ax = plt.subplots(figsize=(9,5.2))
    for label, color, width in [('Road network','#a9b7ba',.6),('Reference route','#c85815',3.4),('Predicted route','#007d76',1.4)]:
        data = next(iter(groups[label]._children.values())).data
        lines = data['features'][0]['geometry']['coordinates']
        ax.add_collection(LineCollection(lines,colors=color,linewidths=width,label=label))
    ax.scatter(*gps.T,s=3,c='#2856a0',alpha=.75,label='GPS observations')
    extent=np.maximum(np.ptp(gps,axis=0)*.04,.0002)
    ax.set_xlim(gps[:,0].min()-extent[0],gps[:,0].max()+extent[0])
    ax.set_ylim(gps[:,1].min()-extent[1],gps[:,1].max()+extent[1])
    ax.set_aspect(1/np.cos(np.deg2rad(gps[:,1].mean())))
    ax.ticklabel_format(useOffset=False);ax.set_xlabel('Longitude (degrees)');ax.set_ylabel('Latitude (degrees)')
    ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(out/'folium_layers.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True,type=Path);args=parser.parse_args()
    with threadpool_limits(limits=2):
        result=diagnostics(args.raw)
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2))
    export_map(args.raw)
