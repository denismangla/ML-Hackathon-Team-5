"""Labels: decide which route arc each GPS point truly belongs to.

The .route file is an ORDERED list of arcs for the whole trip, not one arc per
GPS point. We align the two with a monotone dynamic program:

  * point i is assigned route index j(i), and j(i) never decreases with i
    (the vehicle moves forward along its route);
  * cost = distance from point i to route arc j(i) (metres);
  * skipping a route arc costs `skip_pen` metres (short arcs may get no point);
    the penalty is small so a genuinely close arc is never skipped to save cost.

This is the project's labelling PROPOSAL. It is an assumption: .route is
reference information, not verified ground truth, so we also keep the distance
to the assigned arc (label_dist) to flag doubtful labels.
"""
import numpy as np

from .geometry import point_seg_dist


def _dist_to_route_arcs(P, Ra, Rb, chunk=2000):
    N, R = len(P), len(Ra)
    D = np.empty((N, R), dtype=np.float32)
    for s in range(0, N, chunk):
        d, _ = point_seg_dist(P[s:s + chunk, None, :], Ra[None], Rb[None])
        D[s:s + chunk] = d
    return D


def assign_true_arcs(P, A, B, route, skip_pen: float = 2.0):
    """Return (true_arc[N], label_dist[N], route_pos[N])."""
    N, R = len(P), len(route)
    D = _dist_to_route_arcs(P, A[route], B[route])
    j = np.arange(R)
    cost = D[0].astype(np.float64) + skip_pen * j      # free start, small penalty
    back = np.zeros((N, R), dtype=np.int32)
    for i in range(1, N):
        g = cost - skip_pen * j                         # cost[k] - pen*k
        pm = np.minimum.accumulate(g)                   # best over k <= j
        idx = np.maximum.accumulate(np.where(g <= pm, j, 0))  # argmin position
        cost = D[i] + pm + skip_pen * j
        back[i] = idx
    cost = cost + skip_pen * (R - 1 - j)                # penalise unvisited tail
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], D[np.arange(N), pos].astype(np.float32), pos
