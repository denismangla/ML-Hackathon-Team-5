from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .io import load_trip, list_trip_ids, normalize_trip_id
from .features import build_trip_features, FEATURE_COLUMNS


def make_model(name: str, random_state: int = 42):
    if name == 'logistic':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=500, class_weight='balanced', random_state=random_state))
        ])
    if name == 'naive_bayes':
        return Pipeline([('scaler', StandardScaler()), ('model', GaussianNB(var_smoothing=1e-8))])
    if name == 'knn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1))
        ])
    if name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=2,
            class_weight='balanced_subsample', n_jobs=-1, random_state=random_state
        )
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=random_state
        )
    raise ValueError(f'Unknown model: {name}')


def get_positive_probability(model, X):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    pos_idx = classes.index(1)
    return probs[:, pos_idx]


def evaluate_model(model, val: pd.DataFrame, model_name: str):
    Xv = val[FEATURE_COLUMNS].to_numpy(dtype=float)
    yv = val['label'].to_numpy(dtype=int)
    pv = get_positive_probability(model, Xv)
    pred = (pv >= 0.5).astype(int)
    candidate_acc = accuracy_score(yv, pred) if len(yv) else 0.0
    candidate_precision = precision_score(yv, pred, zero_division=0)
    candidate_recall = recall_score(yv, pred, zero_division=0)
    candidate_f1 = f1_score(yv, pred, zero_division=0)

    scored = val[['trip_id', 'observation_index', 'candidate_arc_id', 'aligned_arc_id',
                  'label_valid', 'candidate_hit']].copy()
    scored['probability'] = pv
    best_rows = scored.sort_values('probability', ascending=False).drop_duplicates(
        ['trip_id', 'observation_index'], keep='first'
    )
    valid = best_rows['label_valid'].astype(bool)
    top1 = float((best_rows.loc[valid, 'candidate_arc_id'].to_numpy() ==
                  best_rows.loc[valid, 'aligned_arc_id'].to_numpy()).mean()) if valid.any() else 0.0
    hit = best_rows.loc[valid, 'candidate_hit'].astype(float).mean() if valid.any() else 0.0
    return {
        'model': model_name,
        'candidate_row_accuracy': float(candidate_acc),
        'candidate_binary_precision': float(candidate_precision),
        'candidate_binary_recall': float(candidate_recall),
        'candidate_binary_f1': float(candidate_f1),
        'candidate_binary_macro_f1': float(f1_score(yv, pred, average='macro', zero_division=0)),
        'observation_top1_accuracy_including_candidate_misses': float(top1),
        'candidate_recall_at_k': float(hit),
        'validation_candidate_rows': int(len(val)),
        'validation_observations_with_valid_alignment': int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description='PathSnap candidate feature generation and baseline ML training')
    parser.add_argument('--data-root', required=True,
                        help='Folder containing train/ and student_test/, e.g. data/raw/Student folder')
    parser.add_argument('--trip-ids', default='',
                        help='Comma-separated training trip IDs. Empty means all training trips.')
    parser.add_argument('--max-trips', type=int, default=5,
                        help='Used only when --trip-ids is empty; limits quick prototype scope.')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--radius-m', type=float, default=100.0)
    parser.add_argument('--label-radius-m', type=float, default=150.0)
    parser.add_argument('--models', default='logistic,random_forest',
                        help='Comma-separated: logistic,naive_bayes,knn,random_forest,hist_gradient_boosting')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.trip_ids.strip():
            trip_ids = [
            normalize_trip_id(x)
            for x in args.trip_ids.split(',')
            if x.strip()
            ]
    else:
        trip_ids = list_trip_ids(data_root, 'train')[:args.max_trips]
    if len(trip_ids) < 2:
        raise SystemExit('Use at least two training trips for a trip-level train/validation split.')

    out_dir = Path('reports')
    model_dir = Path('models')
    out_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    all_frames, summaries = [], []
    total_start = time.perf_counter()
    for i, trip_id in enumerate(trip_ids, 1):
        print(f'[{i}/{len(trip_ids)}] Building features for training trip {trip_id} ...', flush=True)
        start = time.perf_counter()
        trip = load_trip(data_root, 'train', trip_id, include_route=True)
        df, meta, graph, projection, track_xy, by_obs = build_trip_features(
            trip, radius_m=args.radius_m, top_k=args.top_k,
            label_search_radius_m=args.label_radius_m
        )
        all_frames.append(df)
        meta['feature_build_seconds'] = round(time.perf_counter() - start, 3)
        summaries.append(meta)
        print(f"  nodes={meta['n_nodes']:,} arcs={meta['n_arcs']:,} gps={meta['n_observations']:,} "
              f"candidate_rows={meta['candidate_rows']:,} candidate_recall={meta['candidate_recall_at_k']:.3f} "
              f"align_valid={meta['alignment_valid_fraction']:.3f} ({meta['feature_build_seconds']}s)", flush=True)

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv('reports/candidate_features.csv', index=False)
    pd.DataFrame(summaries).to_csv('reports/trip_preprocessing_summary.csv', index=False)
    # Ensure no trip is split across train and validation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(1 / len(trip_ids), 0.2),
                                 random_state=args.random_state)
    train_idx, val_idx = next(splitter.split(full, full['label'], groups=full['trip_id']))
    train = full.iloc[train_idx].copy()
    val = full.iloc[val_idx].copy()
    # Exclude candidate groups that do not contain a positive aligned candidate from
    # classifier fitting: these are candidate-generation misses, not reliable negatives.
    train = train[train['candidate_hit'] == 1].copy()
    val = val[val['label_valid'] == True].copy()
    if train.empty or train['label'].nunique() < 2:
        raise SystemExit('Training data lacks both positive and negative candidates. Increase --max-trips or inspect labels.')
    if val.empty:
        raise SystemExit('Validation fold has no valid aligned observations. Choose different trip IDs.')

    model_names = [x.strip() for x in args.models.split(',') if x.strip()]
    results = []
    best_model, best_name, best_score = None, None, -1.0
    for name in model_names:
        print(f'\nTraining {name} ...', flush=True)
        model = make_model(name, args.random_state)
        t0 = time.perf_counter()
        model.fit(train[FEATURE_COLUMNS].to_numpy(dtype=float), train['label'].to_numpy(dtype=int))
        fit_sec = time.perf_counter() - t0
        metrics = evaluate_model(model, val, name)
        metrics['fit_seconds'] = round(fit_sec, 3)
        metrics['train_candidate_rows'] = int(len(train))
        metrics['train_trips'] = ','.join(sorted(train['trip_id'].unique()))
        metrics['validation_trips'] = ','.join(sorted(val['trip_id'].unique()))
        results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        if metrics['observation_top1_accuracy_including_candidate_misses'] > best_score:
            best_score = metrics['observation_top1_accuracy_including_candidate_misses']
            best_model, best_name = model, name
        joblib.dump(model, model_dir / f'{name}.joblib')

    results_df = pd.DataFrame(results).sort_values(
        'observation_top1_accuracy_including_candidate_misses', ascending=False
    )
    results_df.to_csv('reports/model_comparison.csv', index=False)
    if best_model is not None:
        joblib.dump(best_model, model_dir / 'candidate_model.joblib')
        metadata = {
            'selected_model': best_name,
            'feature_columns': FEATURE_COLUMNS,
            'top_k': args.top_k,
            'candidate_radius_m': args.radius_m,
            'label_search_radius_m': args.label_radius_m,
            'training_trips_requested': trip_ids,
            'training_trips_used': sorted(train['trip_id'].unique().tolist()),
            'validation_trips': sorted(val['trip_id'].unique().tolist()),
            'scope': 'prototype; only selected training trips were processed'
        }
        (model_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('\n=== SUMMARY ===')
    print(results_df.to_string(index=False))
    print(f"\nTotal elapsed: {time.perf_counter()-total_start:.1f}s")
    print('Saved: reports/candidate_features.csv')
    print('Saved: reports/trip_preprocessing_summary.csv')
    print('Saved: reports/model_comparison.csv')
    print('Saved: models/candidate_model.joblib and model artifacts')


if __name__ == '__main__':
    main()
