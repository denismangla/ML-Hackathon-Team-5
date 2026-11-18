"""Candidate generation: for every GPS point, which arcs could it be on?

Method (fast, exact for the final distance):
  1. Densify every arc into sample points at most `spacing` metres apart.
  2. Put the samples in a scipy cKDTree.
  3. For each GPS point, query all samples within radius + spacing/2
     (any arc within `radius` must have a sample that close).
  4. Collapse samples to unique arcs and compute the EXACT point-to-segment
     distance; keep arcs with dist <= radius, then the `k` nearest.

No labels are used here, so this same function is safe to call at inference time.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import point_seg_dist

DEFAULT_RADIUS = 100.0   # metres
DEFAULT_K = 20           # max candidates per GPS point
DEFAULT_SPACING = 25.0   # metres between arc samples


def build_arc_tree(A: np.ndarray, B: np.ndarray, spacing: float = DEFAULT_SPACING):
    L = np.linalg.norm(B - A, axis=1)
    nseg = np.maximum(np.ceil(L / spacing).astype(np.int64), 1)
    counts = nseg + 1
    arc_ids = np.repeat(np.arange(len(A)), counts)
    starts = np.cumsum(counts) - counts
    k = np.arange(counts.sum()) - np.repeat(starts, counts)
    frac = k / np.repeat(nseg, counts)
    samples = A[arc_ids] + frac[:, None] * (B - A)[arc_ids]
    return cKDTree(samples), arc_ids


def generate_candidates(P, A, B, radius=DEFAULT_RADIUS, k=DEFAULT_K,
                        spacing=DEFAULT_SPACING, tree_and_ids=None) -> pd.DataFrame:
    """Return DataFrame[pt_idx, arc_id, dist, frac, rank] (rank 0 = nearest)."""
    tree, sample_arc = tree_and_ids or build_arc_tree(A, B, spacing)
    n_arcs = len(A)
    lists = tree.query_ball_point(P, r=radius + spacing / 2.0, workers=-1)
    lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    if lens.sum() == 0:
        return pd.DataFrame({"pt_idx": [], "arc_id": [], "dist": [], "frac": [], "rank": []})
    pt_idx = np.repeat(np.arange(len(P)), lens)
    arc = sample_arc[np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])]

    key = np.unique(pt_idx * n_arcs + arc)           # unique (point, arc) pairs
    pt_idx, arc = key // n_arcs, key % n_arcs
    d, frac = point_seg_dist(P[pt_idx], A[arc], B[arc])
    keep = d <= radius
    pt_idx, arc, d, frac = pt_idx[keep], arc[keep], d[keep], frac[keep]

    order = np.lexsort((d, pt_idx))                  # by point, then distance
    pt_idx, arc, d, frac = pt_idx[order], arc[order], d[order], frac[order]
    first = np.searchsorted(pt_idx, pt_idx, side="left")
    rank = np.arange(len(pt_idx)) - first
    keep = rank < k
    return pd.DataFrame({"pt_idx": pt_idx[keep], "arc_id": arc[keep],
                         "dist": d[keep], "frac": frac[keep], "rank": rank[keep]})
