"""Build candidate/feature/label tables for every trip in a split.

Usage (from the repo root):
    python -m src.build_dataset                       # train + student_test
    python -m src.build_dataset --splits train --limit 3   # quick test

Outputs (data/processed/):
    <split>_candidates.parquet   one row per (GPS point, candidate arc): features + label
    <split>_points.parquet       one row per GPS point: true arc, label quality, coverage
    <split>_trips.csv            per-trip summary incl. candidate miss rate
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import DEFAULT_K, DEFAULT_RADIUS, build_arc_tree, generate_candidates
from .features import build_features
from .io_utils import DATA_ROOT, SPLITS, Trip, list_trips, load_trip
from .labels import assign_true_arcs

OUT = Path("data/processed")
LABEL_MAX_DIST = 50.0   # metres: label is trusted only if the point is this close to its assigned arc


def build_trip_tables(trip: Trip, radius=DEFAULT_RADIUS, k=DEFAULT_K, with_labels=True):
    """Candidates+features (+labels if the trip has a route). Also used by the API."""
    A, B = trip.A, trip.B
    cands = generate_candidates(trip.pts, A, B, radius=radius, k=k)
    feats = build_features(cands, trip.pts, trip.t, A, B, trip.arcs, len(trip.nodes))
    N = len(trip.pts)
    pts = pd.DataFrame({"pt_idx": np.arange(N), "lon": trip.track["lon"].to_numpy(),
                        "lat": trip.track["lat"].to_numpy(), "t": trip.t,
                        "x": trip.pts[:, 0], "y": trip.pts[:, 1]})
    if with_labels and trip.route is not None:
        true_arc, ldist, rpos = assign_true_arcs(trip.pts, A, B, trip.route)
        pts["true_arc"], pts["label_dist"], pts["route_pos"] = true_arc, ldist, rpos
        feats["label"] = (feats["arc_id"].to_numpy() == true_arc[feats["pt_idx"].to_numpy()]).astype(np.int8)
        hit = feats[feats["label"] == 1].set_index("pt_idx")["rank"]
        pts["true_rank"] = pts["pt_idx"].map(hit).fillna(-1).astype(int)   # -1 = missed
        pts["true_in_cands"] = (pts["true_rank"] >= 0).astype(np.int8)
        pts["label_ok"] = (pts["label_dist"] <= LABEL_MAX_DIST).astype(np.int8)
        feats["label_ok"] = pts["label_ok"].to_numpy()[feats["pt_idx"].to_numpy()]
    return pts, feats


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> int32 (halves file size; features are not precision-critical)."""
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in ("lon", "lat", "x", "y", "t"):
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == np.int64:
            df[c] = df[c].astype(np.int32)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--limit", type=int, default=None, help="only first N trips (testing)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        all_pts, all_c, rows = [], [], []
        ids = list_trips(split, a.root)[: a.limit]
        for n, tid in enumerate(ids, 1):
            t0 = time.time()
            trip = load_trip(split, tid, a.root)
            pts, feats = build_trip_tables(trip, a.radius, a.k)
            pts.insert(0, "trip_id", tid); feats.insert(0, "trip_id", tid)
            all_pts.append(pts); all_c.append(feats)
            rows.append({"trip_id": tid, "n_points": len(pts), "n_arcs": len(trip.arcs),
                         "n_route_arcs": len(trip.route), "n_candidates": len(feats),
                         "label_ok_share": float(pts["label_ok"].mean()),
                         "miss_rate": 1 - pts["true_in_cands"].mean(),
                         "miss_rate_label_ok": 1 - pts.loc[pts["label_ok"] == 1, "true_in_cands"].mean(),
                         "top1_by_distance": float((pts["true_rank"] == 0).mean()),
                         "top1_by_distance_label_ok": float((pts.loc[pts["label_ok"] == 1, "true_rank"] == 0).mean()),
                         "label_dist_median": float(pts["label_dist"].median()),
                         "label_dist_p95": float(pts["label_dist"].quantile(.95)),
                         "seconds": round(time.time() - t0, 1)})
            r = rows[-1]
            print(f"[{split} {n}/{len(ids)}] {tid} pts={r['n_points']} cand={r['n_candidates']} "
                  f"label_ok={r['label_ok_share']:.2f} miss={r['miss_rate']:.3f} "
                  f"miss(ok)={r['miss_rate_label_ok']:.3f} nearest-arc-acc(ok)={r['top1_by_distance_label_ok']:.3f} "
                  f"({r['seconds']}s)", flush=True)
        _compact(pd.concat(all_pts)).to_parquet(OUT / f"{split}_points.parquet", index=False, compression="zstd")
        _compact(pd.concat(all_c)).to_parquet(OUT / f"{split}_candidates.parquet", index=False, compression="zstd")
        summ = pd.DataFrame(rows); summ.to_csv(OUT / f"{split}_trips.csv", index=False)
        P = pd.concat(all_pts); ok = P[P["label_ok"] == 1]
        print(f"\n== {split}: {len(ids)} trips, {len(P)} points "
              f"({len(ok)} with trusted labels = {len(ok)/len(P):.1%})\n"
              f"   candidate miss rate: all {1 - P['true_in_cands'].mean():.4f} | trusted {1 - ok['true_in_cands'].mean():.4f}\n"
              f"   nearest-arc accuracy: all {(P['true_rank'] == 0).mean():.4f} | trusted {(ok['true_rank'] == 0).mean():.4f}\n")


if __name__ == "__main__":
    main()
