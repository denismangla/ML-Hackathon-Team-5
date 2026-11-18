from __future__ import annotations
import numpy as np
import shapely
from shapely import LineString, Point, STRtree
from .graph import RoadGraph
from .geo import movement_features, wrapped_angle_difference


class CandidateGenerator:
    """Spatial-index road candidates with exact point-to-segment distance."""
    def __init__(self, graph: RoadGraph, radius_m: float = 100.0, top_k: int = 8):
        self.graph = graph
        self.radius_m = float(radius_m)
        self.top_k = int(top_k)
        self.lines = np.empty(len(graph.arcs), dtype=object)
        for i, (a, b) in enumerate(zip(graph.arc_p1, graph.arc_p2)):
            self.lines[i] = LineString([a, b])
        self.tree = STRtree(self.lines)

    def generate(self, track_xy: np.ndarray, timestamps: np.ndarray):
        headings, speeds, gaps, heading_valid, gap_flag = movement_features(track_xy, timestamps)
        rows = []
        by_observation = []
        for obs_idx, (xy, ts) in enumerate(zip(track_xy, timestamps)):
            point = Point(float(xy[0]), float(xy[1]))
            indices = np.asarray(self.tree.query(point.buffer(self.radius_m)), dtype=np.int64)
            if len(indices):
                distances_all = np.asarray(shapely.distance(self.lines[indices], point), dtype=float)
                within = distances_all <= self.radius_m
                indices = indices[within]
                distances_all = distances_all[within]
            else:
                distances_all = np.empty(0, dtype=float)
            if len(indices) == 0:
                nearest_idx, nearest_dist = self.tree.query_nearest(
                    point, all_matches=True, return_distance=True
                )
                indices = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
                distances_all = np.asarray(nearest_dist, dtype=float).reshape(-1)
                if len(indices) == 0:
                    by_observation.append([])
                    continue
            order = np.argsort(distances_all)[:self.top_k]
            indices = indices[order]
            distances = distances_all[order]
            fractions = np.asarray(shapely.line_locate_point(self.lines[indices], point), dtype=float)
            lengths = self.graph.arc_lengths[indices]
            fractions = np.divide(fractions, lengths, out=np.zeros_like(fractions), where=lengths > 0)
            fractions = np.clip(fractions, 0.0, 1.0)
            arcs = self.graph.arcs[indices]
            endpoint_dist = np.minimum(
                np.linalg.norm(self.graph.arc_p1[indices] - xy, axis=1),
                np.linalg.norm(self.graph.arc_p2[indices] - xy, axis=1)
            )
            bearing_diff = wrapped_angle_difference(
                np.full(len(indices), headings[obs_idx]), self.graph.arc_bearings[indices]
            )
            for j, arc_id in enumerate(indices):
                row = {
                    'observation_index': int(obs_idx),
                    'timestamp': float(ts),
                    'candidate_arc_id': int(arc_id),
                    'distance_to_segment': float(distances[j]),
                    'projection_fraction': float(fractions[j]),
                    'heading_difference': float(bearing_diff[j]) if heading_valid[obs_idx] else 0.0,
                    'gps_heading': float(headings[obs_idx]) if heading_valid[obs_idx] else 0.0,
                    'heading_valid': float(heading_valid[obs_idx]),
                    'gps_speed': float(min(speeds[obs_idx], 60.0)),
                    'time_delta': float(gaps[obs_idx]),
                    'gps_gap_flag': float(gap_flag[obs_idx]),
                    'segment_length': float(lengths[j]),
                    'endpoint_distance': float(endpoint_dist[j]),
                    'from_in_degree': float(self.graph.in_degree[arcs[j, 0]]),
                    'from_out_degree': float(self.graph.out_degree[arcs[j, 0]]),
                    'to_in_degree': float(self.graph.in_degree[arcs[j, 1]]),
                    'to_out_degree': float(self.graph.out_degree[arcs[j, 1]]),
                }
                rows.append(row)
            by_observation.append([int(x) for x in indices])
        return rows, by_observation
