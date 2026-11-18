"""Build offline dashboard data from saved evaluation artifacts."""
import argparse
import base64
import csv
import json
from pathlib import Path

MODELS = (
    ('random_forest','Random Forest','CS161'),
    ('hist_gradient_boosting','HistGradientBoosting','CS173'),
    ('logistic_regression','Logistic Regression','CS151'),
    ('gaussian_nb','Gaussian Naive Bayes','CS151'),
    ('knn','KNN','CS126'),
)


def read(path):
    return json.loads(Path(path).read_text())


def main(charts):
    with (charts/'model_metrics.csv').open() as handle:
        metrics = {row['model']:row for row in csv.DictReader(handle)}
    records = []
    for key, name, srn in MODELS:
        result = read(f'results/models/{key}.json')
        row = {item:result[item] for item in ('owner','configuration','cv_macro_f1','student_test',
                                             'final_fit_seconds','sampling')}
        row['sampling'] = {'rows':row['sampling']['rows']}
        row['key'],row['name'] = key,name
        row['folds'] = [fold['metrics']['macro_f1'] for fold in result['cv']]
        row['candidate'] = {item:float(metrics[name][item]) for item in ('roc_auc','average_precision')}
        image = charts/f'{srn}_{key}.png'
        row['chart'] = 'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
        records.append(row)

    original = read('results/decoder/baseline_summary.json')['decoder']
    latest = read('results/decoder/summary.json')
    sample = read('results/release/sample_response.json')
    sample.pop('route_arc_sections', None)
    sample.pop('predicted_segments_lonlat', None)
    sample.pop('observation_indices', None)
    evidence = dict(models=records, sample=sample,
                    routeEvaluation=dict(original=original,classifier=latest['classifier'],refined=latest['decoder']))
    Path('pathsnap/ui/src/evidence.json').write_text(json.dumps(evidence,separators=(',',':'))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--charts',type=Path,default=Path('results/model_diagnostics'))
    main(parser.parse_args().charts)
