"""Structured EDA over every trip (train + student_test).

Usage:  python -m src.eda
Outputs: results/eda_inventory.csv, results/figures/eda_overview.png
Covers the checklist in 'Dataset Exploration.pdf' section 4: trips per partition,
points per trip, missing/invalid points, timestamp gaps, extent, network size,
connectivity, isolated nodes, anomalies, and route validity.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io_utils import DATA_ROOT, SPLITS, list_trips, load_trip

RES = Path("results")
JUMP_SPEED = 60.0   # m/s (~216 km/h): faster implied step = GPS jump


def trip_stats(split: str, tid: str) -> dict:
    base = DATA_ROOT / split / tid / tid
    raw = pd.read_csv(base.with_suffix(".track"), sep=r"\s+", header=None)
    n_raw_nan = int(raw.isna().to_numpy().sum())
    trip = load_trip(split, tid)
    P, t, arcs, n = trip.pts, trip.t, trip.arcs, len(trip.nodes)

    dt = np.diff(t)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    speed = step / np.where(dt > 0, dt, np.nan)

    adj = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    ncomp, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp)
    used = np.zeros(n, bool); used[arcs[:, 0]] = True; used[arcs[:, 1]] = True

    rs = {}
    if trip.route is not None:
        r = trip.route
        valid = (r >= 0) & (r < len(arcs))
        rs = {"n_route_arcs": len(r), "route_ids_valid": bool(valid.all()),
              "route_continuity": float(np.mean(arcs[r[:-1], 1] == arcs[r[1:], 0])) if valid.all() and len(r) > 1 else np.nan,
              "route_has_duplicates": bool(len(set(r.tolist())) != len(r))}
    return {
        "split": split, "trip_id": tid, "epsg": trip.epsg,
        "n_points": len(P), "n_raw_nan_values": n_raw_nan,
        "duration_s": float(t[-1] - t[0]), "dt_median": float(np.median(dt)),
        "share_dt_1s": float(np.mean(dt == 1)), "max_dt_s": float(dt.max()),
        "n_dt_gt_5s": int((dt > 5).sum()), "n_nonpositive_dt": int((dt <= 0).sum()),
        "n_duplicate_points": int((step == 0).sum()), "share_stationary_steps": float(np.mean(step < 0.5)),
        "n_gps_jumps": int(np.nansum(speed > JUMP_SPEED)), "max_speed_ms": float(np.nanmax(speed)),
        "track_length_km": float(step.sum() / 1000),
        "extent_km": float(np.hypot(*np.ptp(P, axis=0)) / 1000),
        "n_nodes": n, "n_arcs": len(arcs), "n_isolated_nodes": int((~used).sum()),
        "n_components": int(ncomp), "largest_component_share": float(sizes.max() / n),
        "n_self_loop_arcs": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "n_duplicate_arcs": int(len(arcs) - len({tuple(a) for a in arcs.tolist()})),
        **rs,
    }


def main():
    RES.mkdir(exist_ok=True); (RES / "figures").mkdir(exist_ok=True)
    rows = []
    for split in SPLITS:
        ids = list_trips(split)
        print(f"{split}: {len(ids)} trips")
        rows += [trip_stats(split, tid) for tid in ids]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "eda_inventory.csv", index=False)

    print("\n--- partition summary ---")
    print(df.groupby("split").agg(trips=("trip_id", "count"), points=("n_points", "sum"),
          pts_median=("n_points", "median"), arcs_median=("n_arcs", "median"),
          nodes_max=("n_nodes", "max")).to_string())
    print("\n--- data-quality flags (count of trips) ---")
    flags = {"missing/NaN values": (df.n_raw_nan_values > 0), "non-positive time steps": (df.n_nonpositive_dt > 0),
             "time gaps > 5 s": (df.n_dt_gt_5s > 0), "GPS jumps (> 60 m/s)": (df.n_gps_jumps > 0),
             "isolated nodes": (df.n_isolated_nodes > 0), "disconnected network (>1 component)": (df.n_components > 1),
             "route has duplicate arcs": df.get("route_has_duplicates", pd.Series(False, index=df.index)).fillna(False).astype(bool),
             "route not fully connected": (df.route_continuity.fillna(1) < 1)}
    for k, m in flags.items():
        print(f"  {k:38s} {int(m.sum()):3d} / {len(df)}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for s, c in zip(SPLITS, ["tab:blue", "tab:orange"]):
        d = df[df.split == s]
        ax[0, 0].hist(d.n_points, bins=25, alpha=.7, label=s, color=c)
        ax[0, 1].hist(np.log10(d.n_arcs), bins=25, alpha=.7, label=s, color=c)
        ax[1, 0].hist(d.extent_km, bins=25, alpha=.7, label=s, color=c)
    ax[0, 0].set(title="GPS points per trip", xlabel="points"); ax[0, 1].set(title="Road-network size", xlabel="log10(arcs)")
    ax[1, 0].set(title="Geographic extent of GPS track", xlabel="km")
    ax[1, 1].scatter(df.share_dt_1s * 100, df.n_gps_jumps, c=df.split.map({"train": "tab:blue", "student_test": "tab:orange"}), s=18)
    ax[1, 1].set(title="Sampling regularity vs GPS jumps", xlabel="% of steps exactly 1 s", ylabel="steps > 60 m/s")
    ax[0, 0].legend(); plt.tight_layout(); plt.savefig(RES / "figures" / "eda_overview.png", dpi=130)
    print("\nSaved results/eda_inventory.csv and results/figures/eda_overview.png")


if __name__ == "__main__":
    main()
