# PathSnap — Denis quick-start implementation

This is a **first-review prototype** built against the uploaded `Student folder(1).zip` dataset. It covers:

1. Reading trip data while preserving the original zero-based node and arc row IDs.
2. Trip-local metric projection (longitude/latitude are columns 0/1).
3. Directed graph construction and bounded Dijkstra.
4. Spatially indexed road-candidate generation.
5. Approximate monotonic alignment of GPS observations to *route occurrences* for training labels.
6. Candidate feature generation and candidate-recall reporting.
7. Five baseline candidate classifiers and trip-level train/validation separation.
8. A Folium map for one training trip.
9. A minimal Viterbi skeleton and FastAPI health endpoint for Denis's review evidence.

**Important scope note:** This quick run processes five smaller training trips only. Four trips are used to fit each classifier and one complete trip is held out for validation. It is a prototype, not the final 80-trip model and not final evaluator deployment. The student-test route files are never loaded by the training command.

## 1. Extract the original data

Extract the provided dataset so that the folder structure is:

```text
data/raw/Student folder/
    train/
    student_test/
```

The folder `data/raw/Student folder` must directly contain `train` and `student_test`.

Do not rename trip directories or reorder rows in `.nodes` / `.arcs`.

## 2. Create environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run Denis's graph smoke test

```bash
python -m src.pathsnap.graph_demo --data-root "data/raw/Student folder" --trip-id 00000026
```

Expected evidence: real node/arc counts, sample directed adjacency, a Dijkstra result, and `GRAPH_SMOKE_TEST=PASS`.

## 4. Generate a real trip map

```bash
python -m src.pathsnap.visualize --data-root "data/raw/Student folder" --trip-id 00000026 --output reports/trajectory_00000026.html
```

Open `reports/trajectory_00000026.html` in a browser. It shows a local road-network window, the observed GPS trace, and the reference route. The reference layer is training-only diagnostic evidence, not a model prediction.

## 5. Generate features and train five baseline models

Use this reproducible small-trip prototype set:

```bash
python -m src.pathsnap.train --data-root "data/raw/Student folder" --trip-ids 00000026,00000039,00000018,00000014,00000022 --models logistic,naive_bayes,knn,random_forest,hist_gradient_boosting
```

The script will:

- Process only the named records under `train/`.
- Build the graph and candidate set separately for each trip.
- Align GPS observations monotonically to route **occurrences** for target generation.
- Generate candidate rows (up to 8 nearest candidates within 100 m; if none exist, retain the nearest candidate as a fallback).
- Report candidate recall separately from model accuracy.
- Split by complete trip using `GroupShuffleSplit` (no GPS rows from a trip are split across train and validation).
- Fit the five models on candidate groups whose aligned positive candidate is present. Candidate-miss groups remain counted in candidate-recall reporting rather than being silently treated as valid negative examples.
- Save the best model by held-out observation Top-1 accuracy.

Generated files:

```text
reports/candidate_features.csv
reports/trip_preprocessing_summary.csv
reports/model_comparison.csv
models/logistic.joblib
models/naive_bayes.joblib
models/knn.joblib
models/random_forest.joblib
models/hist_gradient_boosting.joblib
models/candidate_model.joblib
models/metadata.json
```

The model feature columns are defined in `src/pathsnap/features.py` under `FEATURE_COLUMNS`. ID/label columns such as `candidate_arc_id`, `aligned_arc_id`, `label`, `label_valid`, and `trip_id` are not passed into the classifier.

## 6. Run the Viterbi architecture demo (mock only)

```bash
python -m src.pathsnap.viterbi_demo
```

This demonstrates the log-space Viterbi interface with intentionally mocked emissions and transitions. **It is not a real PathSnap route prediction.** Replace the demo transition callback with directed network-distance/connectivity scores from `RoadGraph` before using the decoder in the final pipeline.

## 7. Run the FastAPI health scaffold

```bash
uvicorn src.pathsnap.api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/health`.

Only `/health` is implemented at this stage. Do not claim that `/predict` is ready. Freeze the real prediction endpoint only after the organiser's deployment PDF / evaluator request-response schema is confirmed and the actual candidate → model → HMM/Viterbi → route reconstruction pipeline is integrated.

## 8. What the code does and does not prove

- It uses the reference `.route` only to create training targets and held-out labels. It is never a model feature or a prediction-time input.
- The route alignment is an approximate monotonic nearest-route-occurrence alignment. It preserves repeated route occurrences as separate states, but ambiguous alignments should be reviewed and the method improved before final submission.
- Candidate Recall@K is critical: if the correct reference arc is absent from the candidate set, the classifier cannot recover it.
- The validation metrics from the five-trip quick run are based on a single held-out trip. They are only prototype evidence and are not a reliable estimate of final generalization across all 80 training trips.
- The current HMM/Viterbi module is a generic decoder skeleton; graph-aware transitions and route-arc expansion are still to be implemented.
- The FastAPI app is a health scaffold; the final evaluator contract is not known from the uploaded data ZIP.

## 9. Suggested next steps after the first review

1. Review the route-alignment assumptions with Divyanshi and inspect ambiguous / long-gap observations.
2. Extend the training experiment to all permissible training trips using a more memory-efficient trip-wise or fold-wise feature-generation pipeline.
3. Use grouped cross-validation across complete trips, not one held-out trip only.
4. Compare the candidate classifier with ML-only and ML + graph-aware HMM/Viterbi decoding.
5. Tune candidate radius/top-K and HMM transition parameters using training validation only.
6. Integrate the actual Eshwar model scorer and Divyanshi candidate schema.
7. Confirm the exact evaluator API contract from the organiser deployment PDF before implementing `/predict`.

## 10. Dataset facts confirmed from the uploaded ZIP

- `train/`: 80 trip folders.
- `student_test/`: 10 trip folders.
- Each trip folder has `.nodes`, `.arcs`, `.track`, `.route` files.
- The script uses only `train/` records.
- Coordinates are longitude, latitude; Folium display uses latitude, longitude.
- `.route` values are zero-based row IDs in that trip's `.arcs` file.
