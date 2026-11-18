"""Select a Viterbi transition scale on a whole-trip training validation fold."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from ..data import load_trip
from ..models.common import Dataset, predict
from .viterbi import decode

def tune(raw):
    data = Dataset('data_stage1', raw)
    folds = data.splits['cv_validation_fold_by_training_trip']
    held_out = sorted(t for t,f in folds.items() if f==0)
    fit_ids = sorted(t for t,f in folds.items() if f!=0)
    x, y, _ = data.training(fit_ids)
    model = clone(joblib.load('artifacts/selected_model.joblib')['model']).fit(x,y)
    scales = [15., 30., 60.]
    scores = {scale:[] for scale in scales}
    for trip in held_out:
        f = data.get('train',trip)
        _, p = predict(model,f)
        t = load_trip(raw/'train'/trip)
        for scale in scales:
            selected = decode(t,f,p,beta_m=scale)['predicted_arc_id']
            scores[scale].append(float(f1_score(f['labels']['reference_arc_id'],selected,average='macro',zero_division=0)))
    best = max(scales,key=lambda scale: np.mean(scores[scale]))
    summary = dict(validation_trips=held_out, classifier_fit_trips=fit_ids, mean_f1={k:float(np.mean(v)) for k,v in scores.items()},
                   per_trip=scores, selected_beta_m=best)
    Path('artifacts/decoder.json').write_text(json.dumps(dict(beta_m=best,gap_s=120.,distance_limit_m=3000.),indent=2))
    Path('results/decoder/search.json').write_text(json.dumps(summary,indent=2))
    print(summary['mean_f1'],best,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    with threadpool_limits(limits=2):
        tune(parser.parse_args().raw)
