import numpy as np
from src.pathsnap.graph import RoadGraph
from src.pathsnap.viterbi import viterbi_decode


def test_directed_shortest_path():
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.]])
    arcs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    graph = RoadGraph.build(nodes, arcs)
    distance, path = graph.shortest_path(0, 2, max_distance_m=100.0)
    assert path == [0, 1]
    assert distance > 0
    reverse_distance, reverse_path = graph.shortest_path(2, 0, max_distance_m=100.0)
    assert not np.isfinite(reverse_distance)
    assert reverse_path == []


def test_viterbi_prefers_connected_sequence():
    states = [[10, 11], [20, 21], [30, 31]]
    emissions = [[0.8, 0.2], [0.4, 0.6], [0.8, 0.2]]
    allowed = {(10, 20), (20, 30), (11, 21), (21, 31)}
    def trans(a, b, t):
        if a == b:
            return -0.2
        return -0.1 if (a, b) in allowed else -10.0
    out = viterbi_decode(states, emissions, trans)
    assert out['predicted_arc_ids_by_observation'] == [10, 20, 30]
    assert out['route_break_observation_indices'] == []
