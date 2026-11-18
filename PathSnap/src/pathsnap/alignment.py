from __future__ import annotations
import numpy as np
import shapely
from shapely import LineString, Point, STRtree
from .graph import RoadGraph


def align_track_to_route(graph: RoadGraph, track_xy: np.ndarray, timestamps: np.ndarray,
                         route_arc_ids: np.ndarray, search_radius_m: float = 150.0,
                         max_label_distance_m: float = 200.0):
    """Approximate monotonic alignment to route *occurrences* (not just unique arc IDs).

    Returns per-observation arc ID, alignment distance, route progress and validity.
    This is a derived training target; ambiguous/far points are flagged.
    """
    if route_arc_ids is None or len(route_arc_ids) == 0:
        raise ValueError('Training alignment requires a non-empty reference route.')
    route_p1 = graph.arc_p1[route_arc_ids]
    route_p2 = graph.arc_p2[route_arc_ids]
    route_lengths = graph.arc_lengths[route_arc_ids]
    lines = np.empty(len(route_arc_ids), dtype=object)
    for i, (a, b) in enumerate(zip(route_p1, route_p2)):
        lines[i] = LineString([a, b])
    tree = STRtree(lines)
    cumulative = np.concatenate([[0.0], np.cumsum(route_lengths[:-1])])

    # Candidate route occurrences for each GPS observation.
    options = []
    for xy in track_xy:
        point = Point(float(xy[0]), float(xy[1]))
        idx = np.asarray(tree.query(point.buffer(search_radius_m)), dtype=np.int64)
        if len(idx):
            d_all = np.asarray(shapely.distance(lines[idx], point), dtype=float)
            keep = d_all <= search_radius_m
            idx, d_all = idx[keep], d_all[keep]
        else:
            d_all = np.empty(0, dtype=float)
        if len(idx) == 0:
            near_idx, near_dist = tree.query_nearest(point, all_matches=True, return_distance=True)
            idx = np.asarray(near_idx, dtype=np.int64).reshape(-1)
            d_all = np.asarray(near_dist, dtype=float).reshape(-1)
        if len(idx) == 0:
            options.append([])
            continue
        order = np.argsort(d_all)[:10]
        idx, d = idx[order], d_all[order]
        pos = np.asarray(shapely.line_locate_point(lines[idx], point), dtype=float)
        frac = np.divide(pos, route_lengths[idx], out=np.zeros_like(pos), where=route_lengths[idx] > 0)
        frac = np.clip(frac, 0.0, 1.0)
        progress = cumulative[idx] + frac * route_lengths[idx]
        options.append([
            (int(ridx), float(dist), float(prog))
            for ridx, dist, prog in zip(idx, d, progress)
        ])

    # Fill empty observations with the nearest occurrence, while preserving the fact
    # that such observations are low-confidence (distance threshold below will flag them).
    # Usually STRtree.nearest above already avoids empty entries.
    n = len(track_xy)
    if any(len(x) == 0 for x in options):
        for i in range(n):
            if not options[i]:
                options[i] = [(0, float('inf'), 0.0)]

    # Dynamic programming over a small number of nearby route occurrences.
    costs = []
    back = []
    for t, opts in enumerate(options):
        local = np.array([d / 20.0 for _, d, _ in opts], dtype=float)
        if t == 0:
            costs.append(local)
            back.append(np.full(len(opts), -1, dtype=int))
            continue
        prev_opts = options[t - 1]
        prev_cost = costs[t - 1]
        cur_cost = np.full(len(opts), np.inf, dtype=float)
        cur_back = np.full(len(opts), -1, dtype=int)
        gps_disp = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
        dt = float(timestamps[t] - timestamps[t - 1])
        beta = max(25.0, min(150.0, 20.0 + 0.5 * gps_disp))
        for j, (_, _, prog_j) in enumerate(opts):
            for k, (_, _, prog_k) in enumerate(prev_opts):
                delta = prog_j - prog_k
                # Permit small backward movement from GPS noise, but discourage it.
                if delta < -25.0:
                    continue
                if dt > 60.0:
                    trans = 0.15 if delta >= -25.0 else 10.0
                else:
                    trans = abs(max(delta, 0.0) - gps_disp) / beta
                    if delta < 0:
                        trans += abs(delta) / 10.0
                score = prev_cost[k] + trans + local[j]
                if score < cur_cost[j]:
                    cur_cost[j] = score
                    cur_back[j] = k
        if not np.isfinite(cur_cost).any():
            # Restart this alignment segment after a severe ambiguity/disconnection.
            cur_cost = local
            cur_back[:] = -1
        costs.append(cur_cost)
        back.append(cur_back)

    selected = np.zeros(n, dtype=int)
    selected[-1] = int(np.argmin(costs[-1]))
    for t in range(n - 1, 0, -1):
        prev_idx = back[t][selected[t]]
        selected[t - 1] = prev_idx if prev_idx >= 0 else int(np.argmin(costs[t - 1]))

    aligned_arc = np.zeros(n, dtype=np.int64)
    aligned_dist = np.zeros(n, dtype=float)
    aligned_progress = np.zeros(n, dtype=float)
    for t in range(n):
        occurrence, dist, progress = options[t][selected[t]]
        aligned_arc[t] = int(route_arc_ids[occurrence])
        aligned_dist[t] = float(dist)
        aligned_progress[t] = float(progress)
    valid = np.isfinite(aligned_dist) & (aligned_dist <= max_label_distance_m)
    return aligned_arc, aligned_dist, aligned_progress, valid
