# PathSnap: P1 - Data, candidates, labels, features, maps

This part builds the foundation the rest of the team trains on.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Put the unzipped dataset at `data/Student folder/` (it must contain `train/` and `student_test/`).

## Run (from the repo root, in this order)
```bash
python -m src.eda                 # dataset inventory + data-quality flags + overview plot
python -m src.build_dataset       # candidates + features + labels  (~1 min)
python -m src.coverage            # candidate miss rate vs radius and K
python -m src.maps --split train --trip 00000005 --candidates-at 300   # Folium map
```

## What each file does
| File | Purpose |
|---|---|
| `src/io_utils.py` | Reads the 4 raw files; projects each trip to metres with **its own UTM zone** |
| `src/geometry.py` | Point-to-segment distance |
| `src/candidates.py` | KD-tree candidate search (radius 100 m, up to K=20 arcs per GPS point) |
| `src/labels.py` | Aligns `.route` to GPS points (monotone dynamic programme) |
| `src/features.py` | Features per (point, candidate) pair; `FEATURE_COLS` lists the model inputs |
| `src/build_dataset.py` | Runs everything, writes `data/processed/*.parquet`. `build_trip_tables()` is reusable by the API |
| `src/eda.py`, `src/coverage.py` | EDA and candidate-coverage evidence for the report |
| `src/maps.py` | Folium map; `make_trip_map(trip, pred_arcs=...)` adds the predicted route layer for P4/API |

## Output tables (hand these to P2-P4)
`data/processed/{train,student_test}_candidates.parquet`: one row per (GPS point, candidate arc).
Keys: `trip_id, pt_idx, arc_id`. Model inputs: `FEATURE_COLS` in `src/features.py`.
Target: `label` (1 = candidate is the reference arc). **Group by `trip_id` in cross-validation.**
Filter with `label_ok == 1` for training/evaluation (see DATA_NOTES.md).
`*_points.parquet`: one row per GPS point with `true_arc`, `label_dist`, `label_ok`, `true_rank`.

## Rules to respect
* Never fit scalers/encoders on student_test or evaluator data (fit on train only).
* Candidate generation and features use **no labels**, so identical code runs at inference.
* The evaluator trips are never touched here.
