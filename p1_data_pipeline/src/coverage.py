"""Candidate coverage: how big must the radius / K be so the true arc is a candidate?

Usage:  python -m src.coverage     (after python -m src.build_dataset)
Uses TRAIN points with trusted labels only. Outputs results/candidate_coverage.csv
and results/figures/candidate_coverage.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

P = pd.read_parquet("data/processed/train_points.parquet")
P = P[P.label_ok == 1]
radii = [5, 10, 15, 20, 30, 40, 50]
ks = [1, 2, 3, 5, 8, 10, 15, 20]
by_r = pd.DataFrame({"radius_m": radii, "miss_rate": [float((P.label_dist > r).mean()) for r in radii]})
by_k = pd.DataFrame({"K": ks, "miss_rate": [float(((P.true_rank < 0) | (P.true_rank >= k)).mean()) for k in ks]})
Path("results/figures").mkdir(parents=True, exist_ok=True)
pd.concat([by_r.assign(kind="radius"), by_k.rename(columns={"K": "radius_m"}).assign(kind="K")]).to_csv("results/candidate_coverage.csv", index=False)
print("Miss rate vs search radius (true arc farther than R):\n", by_r.to_string(index=False))
print("\nMiss rate vs K (true arc not among the K nearest, radius 100 m):\n", by_k.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(by_r.radius_m, by_r.miss_rate * 100, "o-"); ax[0].set(xlabel="radius (m)", ylabel="candidate miss rate (%)", title="Miss rate vs radius")
ax[1].plot(by_k.K, by_k.miss_rate * 100, "o-"); ax[1].set(xlabel="K nearest arcs", ylabel="candidate miss rate (%)", title="Miss rate vs K")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/figures/candidate_coverage.png", dpi=130)
