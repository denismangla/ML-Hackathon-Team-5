# PathSnap — first-review prototype evidence

Generated from the uploaded dataset with the included code. This is an actual prototype run, not a claim that all 80 training trips or the final system are complete.

## Data scope

- Processed training trips: `00000026`, `00000039`, `00000018`, `00000014`, `00000022`.
- Fit trips: `00000014`, `00000022`, `00000026`, `00000039`.
- Held-out validation trip: `00000018`.
- Student-test trips: not loaded by the training script.
- Total candidate feature rows: 15,767.
- Training candidate rows after removing observations without a candidate hit: 12,647.
- Validation candidate rows: 2,984.
- Validation observations with valid route alignment: 379.

## Candidate generation / alignment results

| Trip | GPS observations | Candidate rows | Alignment valid fraction | Candidate Recall@8 |
|---|---:|---:|---:|---:|
| 00000026 | 361 | 2,882 | 100.0% | 99.72% |
| 00000039 | 363 | 2,688 | 100.0% | 100.00% |
| 00000018 | 379 | 2,984 | 100.0% | 99.47% |
| 00000014 | 448 | 3,533 | 100.0% | 98.66% |
| 00000022 | 474 | 3,680 | 100.0% | 97.89% |

Candidate Recall@8 measures whether the aligned reference arc is present among the generated candidates. It is not model accuracy.

## Baseline model comparison

All models used the same generated feature table, same trip-level split and same held-out trip (`00000018`).

| Model | Candidate binary Macro F1 | Observation Top-1 accuracy (candidate misses count as wrong) | Fit time |
|---|---:|---:|---:|
| HistGradientBoosting | 0.9330 | 90.77% | 0.173 s |
| Random Forest | 0.9300 | 90.24% | 1.426 s |
| Gaussian Naive Bayes | 0.7960 | 85.75% | 0.003 s |
| Logistic Regression | 0.7627 | 84.43% | 0.015 s |
| KNN | 0.8361 | 77.57% | 0.014 s |

Prototype model selection: HistGradientBoosting ranked first on this one-trip validation result. Random Forest was close. This is not enough evidence to declare a final winning model; expand grouped validation across the full permitted training set before final selection.

## Important limitations

1. Only five smaller trips were used to make a quick demonstrable prototype.
2. Validation contains one held-out trip only. The percentages must not be presented as full-dataset generalization performance.
3. Reference-to-GPS label alignment is a monotonic nearest-route-occurrence approximation. It preserves repeated route occurrences but requires review for ambiguous locations and the known discontinuous reference transitions.
4. Candidate miss observations are counted in candidate recall and observation-level Top-1 accuracy. They are excluded from binary classifier fitting because no positive candidate is present.
5. HMM/Viterbi is currently a generic log-space decoder skeleton with mock demo transitions. It is not yet connected to real directed shortest-path transition probabilities.
6. FastAPI currently exposes `/health` only. The final `/predict` endpoint must follow the organiser's deployment specification.

## Reproduce

```bash
python -m src.pathsnap.train --data-root "data/raw/Student folder" --trip-ids 00000026,00000039,00000018,00000014,00000022 --models logistic,naive_bayes,knn,random_forest,hist_gradient_boosting
```
