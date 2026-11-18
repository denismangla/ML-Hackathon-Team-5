# PathSnap team checkpoint for the 11:30 evaluation

The first review covers data preprocessing and feature engineering (5 marks), followed by model selection and training (5 marks). This directory contains the team's checkpoint source, processed feature and label files, saved candidate classifiers, per-trip predictions, results and plots. The model experiments ran locally on the Mac in Python 3.14.6 using scikit-learn 1.9.0. AI assistance was used for implementation and experiment reporting.

## What to demonstrate

1. Run `python3 show_review.py`. Explain that 80 whole trips formed the training partition, and 10 student-test trips were held aside. The raw inputs are in the organiser's `Student folder/train` and `Student folder/student_test`. Each trip has a road graph (.nodes and .arcs), GPS observations (.track) and an ordered reference route (.route).
2. Open `data_stage1/training_trip_map.png`, `data_stage1/data_quality.csv` and `data_stage1/feature_schema.json`. Explain how GPS noise can put a point between roads. The candidate generator compares directed arcs by distance and heading, among 14 features. Up to eight nearby arcs within 100 m are considered, with a nearest-candidate fallback. `pathsnap/features.py` contains the feature calculations. `pathsnap/alignment.py` separately aligns the ordered reference to observations to derive training labels.
3. Show `pathsnap/models/train.py` for fit and evaluation, `pathsnap/models/definitions.py` for the five model configurations, and `pathsnap/models/common.py` for training-row selection and metrics. Each model saw the same three whole-trip CV folds. Sampling takes up to 80 unambiguous, usable, candidate-covered observations per training trip, then up to 20,000 candidate rows per fit. One training trip had no eligible rows. Holding out whole trips reduces leakage from adjacent observations.
4. Show `results/models/run.log`, `results/models/comparison.csv`, `results/models/model_comparison.png` and the five files in `artifacts/`. Random Forest won on mean grouped CV Macro F1 (0.6183); HistGradientBoosting was close (0.6174). Random Forest's student Top-1 was 79.31% and Macro F1 was 0.6197. Logistic regression scored higher on student Top-1 (79.76%). The selection criterion was fixed using training-fold performance.
5. If asked about scores, Top-1 counts GPS observations where the top-ranked arc matches the derived reference. Macro F1 is computed over roads separately in each trip, then averaged. Ambiguous observations and candidate misses remain in the primary evaluation. This checkpoint reports candidate classifier results. The afternoon decoder and API results live in the full release.

## Fresh run on the Mac

In this directory (or in project/ in the team ZIP), run:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
RAW="$HOME/ML HACKATHON/Student folder"
python -m pathsnap.models.train --raw "$RAW" --model hist_gradient_boosting --output results_local/hgb --artifacts artifacts_local
```

To repeat all five models, omit `--model hist_gradient_boosting` and set `--output results_local/all`. Fresh results are written separately from the original evaluation. To demonstrate the complete preprocessing step, run `python -m pathsnap.run --data "$RAW" --output results_local/preprocessed`. To train directly on the features from that fresh run, add `--data results_local/preprocessed` to the training command. Run `python -m unittest discover -s tests -v` to verify the pipeline.

The source release used for these results reads the organiser-supplied files on the model laptop. Unzip the original dataset at the path above or set RAW to its actual location.
