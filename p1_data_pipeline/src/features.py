"""Feature engineering for (GPS point, candidate arc) pairs.

Every feature is computed from the GPS track + road network only (no labels,
no scaling fitted on data), so identical code runs at training and at inference.
Learned transforms (e.g. StandardScaler) must be fitted later on TRAIN only.
"""
import numpy as np
import pandas as pd

from .geometry import unit

MIN_MOVE_M = 2.0   # below this displacement the heading is considered undefined

FEATURE_COLS = [
    "dist", "dist_min", "dist_gap", "rank", "n_cand", "frac", "arc_len",
    "cos_heading", "heading_diff_deg", "heading_valid", "speed",
    "dt_prev", "dt_next", "has_reverse", "end_out_degree",
]


def point_features(P: np.ndarray, t: np.ndarray) -> dict:
    """Per-GPS-point kinematics from neighbouring points."""
    N = len(P)
    i = np.arange(N)
    prev, nxt = np.maximum(i - 1, 0), np.minimum(i + 1, N - 1)
    move = P[nxt] - P[prev]
    dt = np.maximum(t[nxt] - t[prev], 1e-6)
    h, disp = unit(move)
    valid = disp >= MIN_MOVE_M
    return {
        "heading": np.where(valid[:, None], h, 0.0),
        "heading_valid": valid.astype(np.int8),
        "speed": disp / dt,
        "dt_prev": np.where(i > 0, t - t[prev], 0.0),
        "dt_next": np.where(i < N - 1, t[nxt] - t, 0.0),
    }


def network_features(arcs: np.ndarray, n_nodes: int) -> dict:
    """Per-arc topology features."""
    pairs = set(zip(arcs[:, 1].tolist(), arcs[:, 0].tolist()))
    has_rev = np.fromiter(((u, v) in pairs for u, v in arcs.tolist()),
                          dtype=np.int8, count=len(arcs))
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
    return {"has_reverse": has_rev, "end_out_degree": out_deg[arcs[:, 1]]}


def build_features(cands: pd.DataFrame, P, t, A, B, arcs, n_nodes) -> pd.DataFrame:
    pf = point_features(P, t)
    nf = network_features(arcs, n_nodes)
    df = cands.copy()
    pi, ai = df["pt_idx"].to_numpy(), df["arc_id"].to_numpy()

    arc_vec = (B - A)[ai]
    arc_unit, arc_len = unit(arc_vec)
    cos_h = np.einsum("ij,ij->i", pf["heading"][pi], arc_unit)
    valid = pf["heading_valid"][pi].astype(bool)
    cos_h = np.where(valid, np.clip(cos_h, -1, 1), 0.0)

    dmin = df.groupby("pt_idx")["dist"].transform("min")
    df["dist_min"] = dmin
    df["dist_gap"] = df["dist"] - dmin
    df["n_cand"] = df.groupby("pt_idx")["dist"].transform("size")
    df["arc_len"] = arc_len
    df["cos_heading"] = cos_h
    df["heading_diff_deg"] = np.where(valid, np.degrees(np.arccos(cos_h)), 90.0)
    df["heading_valid"] = pf["heading_valid"][pi]
    df["speed"] = pf["speed"][pi]
    df["dt_prev"] = pf["dt_prev"][pi]
    df["dt_next"] = pf["dt_next"][pi]
    df["has_reverse"] = nf["has_reverse"][ai]
    df["end_out_degree"] = nf["end_out_degree"][ai]
    return df
