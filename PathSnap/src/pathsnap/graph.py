from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from .geo import segment_geometry


@dataclass
class RoadGraph:
    nodes_xy: np.ndarray
    arcs: np.ndarray
    arc_p1: np.ndarray
    arc_p2: np.ndarray
    arc_lengths: np.ndarray
    arc_bearings: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    adjacency: list[list[tuple[int, int, float]]]

    @classmethod
    def build(cls, nodes_xy: np.ndarray, arcs: np.ndarray) -> 'RoadGraph':
        p1, p2, lengths, bearings = segment_geometry(nodes_xy, arcs)
        n = len(nodes_xy)
        out_degree = np.bincount(arcs[:, 0], minlength=n).astype(np.int32)
        in_degree = np.bincount(arcs[:, 1], minlength=n).astype(np.int32)
        adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
        for arc_id, (u, v) in enumerate(arcs):
            adjacency[int(u)].append((int(v), int(arc_id), float(lengths[arc_id])))
        return cls(nodes_xy, arcs, p1, p2, lengths, bearings,
                   out_degree, in_degree, adjacency)

    def shortest_path(self, source: int, target: int, max_distance_m: float = 5000.0):
        """Bounded directed Dijkstra. Returns (distance_m, arc_id_list) or (inf, [])."""
        source, target = int(source), int(target)
        if source == target:
            return 0.0, []
        if not (0 <= source < len(self.nodes_xy) and 0 <= target < len(self.nodes_xy)):
            return float('inf'), []
        dist = {source: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist.get(u):
                continue
            if d > max_distance_m:
                break
            if u == target:
                path = []
                cur = target
                while cur != source:
                    parent, arc_id = prev[cur]
                    path.append(arc_id)
                    cur = parent
                path.reverse()
                return d, path
            for v, arc_id, w in self.adjacency[u]:
                nd = d + w
                if nd <= max_distance_m and nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, arc_id)
                    heapq.heappush(heap, (nd, v))
        return float('inf'), []
