"""Tune prediction weighting and same-arc GPS jitter on training fold 0."""
import argparse
import itertools
import json
from time import perf_counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from pathsnap.data import load_trip
from pathsnap.decoder.viterbi import decode
from pathsnap.models.common import Dataset, aggregate, metrics, predict


def main(raw, stage):
    dataset = Dataset('data_stage1', raw)
    folds = dataset.splits['cv_validation_fold_by_training_trip']
    validation = sorted(trip for trip, fold in folds.items() if fold == 0)
    fit = sorted(trip for trip, fold in folds.items() if fold != 0)
    x, y, _ = dataset.training(fit)
    model = clone(joblib.load('artifacts/selected_model.joblib')['model']).fit(x, y)
    reference_path = Path('artifacts/decoder.json')
    current = json.loads(reference_path.read_text())
    reference = {k:current.get(k, v) for k,v in
                 dict(beta_m=15., reverse_tolerance_m=0., emission_weight=1.).items()}
    if stage == 'jitter':
        grid = [dict(beta_m=reference['beta_m'], reverse_tolerance_m=jitter, emission_weight=weight)
                for jitter,weight in itertools.product((0.,10.,25.),(1.,2.,4.))]
        f1_slack,continuity_slack = .005,.015
        filename = 'first_refinement_search.json'
    else:
        grid = [dict(beta_m=beta, reverse_tolerance_m=jitter, emission_weight=1.)
                for beta,jitter in itertools.product((8.,12.,18.,25.),(25.,45.,70.))]
        f1_slack,continuity_slack = .002,.007
        filename = 'refinement_search.json'
    grid = [reference] + [settings for settings in grid if settings != reference]
    trials = [[] for _ in grid]
    for trip in validation:
        features = dataset.get('train', trip)
        _, probabilities = predict(model, features)
        data = load_trip(raw / 'train' / trip)
        for trial, settings in zip(trials, grid):
            started = perf_counter()
            prediction = decode(data, features, probabilities, **settings)['predicted_arc_id']
            trial.append(dict(metrics(features, prediction, probabilities), inference_seconds=perf_counter()-started))
        print('Validation trip complete', trip, flush=True)

    rows = []
    for settings, trial in zip(grid, trials):
        result = aggregate(trial)
        rows.append(dict(settings=settings, **{k:result[k] for k in ('top1_accuracy','macro_f1','direct_continuity')}))
    baseline = rows[0]
    acceptable = [r for r in rows if r['macro_f1'] >= baseline['macro_f1'] - f1_slack
                  and r['direct_continuity'] >= baseline['direct_continuity'] - continuity_slack]
    chosen = max(acceptable, key=lambda r:(r['top1_accuracy'], r['macro_f1']))
    result = dict(fit_trips=fit, validation_trips=validation, baseline=baseline, chosen=chosen, trials=rows,
                  selection=f'Maximise validation Top-1 with Macro F1 within {f1_slack} and continuity within {continuity_slack} of the reference decoder.')
    (Path('results/decoder')/filename).write_text(json.dumps(result, indent=2) + '\n')
    Path('artifacts/decoder.json').write_text(json.dumps(dict(gap_s=120., distance_limit_m=3000., **chosen['settings']), indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--stage', choices=('jitter','scale'), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.raw,args.stage)
