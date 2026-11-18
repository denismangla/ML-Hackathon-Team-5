"""Display the saved evidence using only the Python standard library."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def load(name):
    return json.loads((root/name).read_text())

summary = load('data_stage1/summary.json')
splits = load('data_stage1/splits.json')
print('PathSnap | Team 5 | 11:30 first evaluation')
print(f"{len(splits['training_trips'])} training trips / {len(splits['student_test_trips'])} student-test trips")
print(f"{summary['observations']:,} GPS points / {summary['candidate_rows']:,} candidate rows")
print('Features:', ', '.join(load('data_stage1/feature_schema.json')['features']))
print('\nModel                   CV Macro F1    Student Top-1   Student Macro F1')
with (root/'results/models/comparison.csv').open() as file:
    for row in csv.DictReader(file):
        print(f"{row['model']:24s} {float(row['cv_macro_f1']):9.4f} {float(row['student_top1']):15.4f} {float(row['student_macro_f1']):17.4f}")
selection = load('results/models/selection.json')
print('\nSelected by trip-grouped CV:', selection['selected']['model'])
print('Evidence: results/models/run.log, results/models/selection.json, artifacts/*.joblib')
