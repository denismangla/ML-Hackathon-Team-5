# Results and metrics

The baseline models and their measured scores are the same as the earlier submission. The CSV records three-fold grouped validation and the held-out student-test scores. The baseline diagnostic cards contain precision, recall, F1, log loss, ROC and precision–recall curves, confusion matrices and timing as applicable.

| Pipeline on ten held-out student-test trips | Point Top-1 | Mean trip Macro F1 | Direct continuity |
|---|---:|---:|---:|
| Tuned Random Forest, pointwise selection | 80.43% | 0.6458 | 89.36% |
| Tuned forest with original Viterbi | 78.91% | 0.7419 | 96.92% |
| Tuned forest with refined Viterbi | 89.31% | 0.8642 | 98.02% |

The decoder configuration was selected on training-validation trips. All student-test figures cover 19,160 GPS observations. The reserved evaluator scores are pending the event run.

- Baseline results and model JSONs are in `results/models/`.
- Per-model plots and machine-readable diagnostics are in `results/model_diagnostics/`.
- Tuning trials, validation comparisons and decoder settings are in `results/tuning/`, `results/decoder/` and `artifacts/decoder.json`.
- Geographic analysis and the API check are in `results/figures/` and `results/release/`.
