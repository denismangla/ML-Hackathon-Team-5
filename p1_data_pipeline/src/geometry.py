"""Vectorised point-to-segment geometry (all inputs in metres)."""
import numpy as np


def point_seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distance from points P to segments A->B (same leading shape, or broadcastable).

    Returns (dist, frac) where frac in [0, 1] is the position of the closest
    point along the segment (0 = at A, 1 = at B).
    """
    AB = B - A
    L2 = np.einsum("...i,...i->...", AB, AB)
    AP = P - A
    t = np.einsum("...i,...i->...", AP, AB) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)
    closest = A + t[..., None] * AB
    d = np.linalg.norm(P - closest, axis=-1)
    return d, t


def unit(v: np.ndarray, eps: float = 1e-9):
    """Row-normalise vectors; returns (unit_vectors, norms)."""
    n = np.linalg.norm(v, axis=-1)
    return v / np.maximum(n, eps)[..., None], n
