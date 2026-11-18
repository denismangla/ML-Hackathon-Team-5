# PathSnap | Team 5

The evaluator dashboard matches GPS observations to directed roads using a tuned Random Forest and a refined Viterbi decoder. The dashboard shows a live route map and five baseline model reports.

| Member | SRN | Experiment |
|---|---|---|
| Divyanshi | PES2UG24CS161 | Data preparation, features and Random Forest |
| Denis Mangla | PES2UG24CS151 | Gaussian Naive Bayes, Logistic Regression, decoder and API |
| Gurubelli Yekambar Eshwar Rao | PES2UG24CS173 | HistGradientBoosting, comparison, decoder tuning and UI |
| Ch Sowmya Vaishnavi | PES2UG24CS126 | KNN, maps, report and UI |

## Two-laptop evaluation

Run the API on Laptop 1 from the project root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py server --data "/actual/path/to/organiser-reserved-trips" --mode evaluator
```

For rehearsal use the organiser's `Student folder/student_test` directory and omit `--mode evaluator`. The server uses the bundled weights and settings. An optional `.route` file is used only for metrics after inference. The API sends predicted geometry and aggregate scores to the UI.

On Laptop 2, run `python3 demo.py ui`. Open `http://127.0.0.1:8090`, enter Laptop 1's address with port 8000, then click **Connect**. Select a trip and choose **Reconstruct route**. For a student-test rehearsal, use trip 00000009. **Load sample** opens a clearly labelled saved student run when a server is unavailable. **Model reports** opens the baseline comparison and per-model metrics.

## Review evidence

| Evidence | Location |
|---|---|
| Technical report | `docs/PathSnap_Team5_Technical_Report.pdf` |
| Map diagnostics and accuracy comparison | `docs/PathSnap_Maps_and_Metrics.pdf` |
| Five baseline model metrics, confusion matrices, ROC and precision-recall curves | `docs/PathSnap_Baseline_Model_Metrics.pdf` |
| Interactive student map with predicted, reference, candidate and GPS layers | `results/release/PathSnap_Folium.html` |
| Training map with raw GPS, candidates and supplied reference | `results/release/PathSnap_Training_Trip_00000005.html` |
| Dataset distributions and candidate recall by K and radius | `results/figures/` |
| Selected settings and archived validation trials | `artifacts/decoder.json`, `results/decoder/` |

On the same 19,160 GPS observations from ten student-test trips, the refined decoder improved Top-1 from 78.91% to 89.31%, trip Macro F1 from 0.7419 to 0.8642 and direct continuity from 96.92% to 98.02%. Small backward movements on the same directed road are treated as GPS jitter. Validation used training trips. The organiser's reserved trips remain independent.

Open the Folium map with internet access for street and satellite tiles. Use **GPS #300** to inspect candidates and **Fit trip** to return to the whole trajectory. The live evaluator dashboard draws road geometry with its bundled assets.

## Reproduce the experiments

Supply the organiser's raw student directory in RAW. Preprocessing regenerates the feature and label arrays.

```sh
python -m pathsnap.run --data "$RAW" --output data_stage1
python -m pathsnap.models.train --raw "$RAW"
python -m pathsnap.models.tune --raw "$RAW" --trials 10
python -m pathsnap.decoder.tune --raw "$RAW"
python -m pathsnap.decoder.evaluate --raw "$RAW"
cp results/decoder/summary.json results/decoder/baseline_summary.json
python -m scripts.compare_decoder --raw "$RAW" --stage jitter
python -m scripts.compare_decoder --raw "$RAW" --stage scale
python -m pathsnap.decoder.evaluate --raw "$RAW"
python -m scripts.verify_release --data "$RAW/student_test"
python -m scripts.release_analysis --raw "$RAW"
python -m scripts.map_diagnostics --raw "$RAW"
python -m scripts.plot_decoder_refinement
python -m scripts.build_map_review
python -m scripts.build_report
```

Save the official dashboard and matching server POST screenshots as `results/release/ui_demo.png` and `results/release/receiving_laptop.png`. Rebuild the technical PDF after capturing them. For a source rebuild, use npm install and npm run build from pathsnap/ui.
