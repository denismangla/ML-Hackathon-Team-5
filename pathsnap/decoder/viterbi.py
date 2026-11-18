"""Directed-road Viterbi decoding with bounded GPS backtracking on the same arc."""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..data import geometry


def decode(trip, features, probabilities, gap_s=120., distance_limit_m=3000.,
           beta_m=30., reverse_tolerance_m=0., emission_weight=1.):
    nodes, points, start_nodes, end_nodes, _ = geometry(trip)
    lengths = np.linalg.norm(end_nodes - start_nodes, axis=1)
    edges = {}
    arc_for_nodes = {}
    for arc_id, ((u, v), weight) in enumerate(zip(trip.arcs, lengths)):
        pair = (int(u), int(v))
        edges[pair] = min(edges.get(pair, float('inf')), float(weight))
        arc_for_nodes.setdefault(pair, arc_id)
    graph = csr_matrix((list(edges.values()), ([x[0] for x in edges], [x[1] for x in edges])),
                       shape=(len(nodes), len(nodes)))

    @lru_cache(maxsize=64)
    def network(source):
        return dijkstra(graph, directed=True, indices=source, limit=distance_limit_m,
                        return_predecessors=True)

    observations = features['observation_index']
    starts = np.r_[0, np.flatnonzero(np.diff(observations)) + 1]
    ends = np.r_[starts[1:], len(observations)]
    arcs = features['candidate_arc_id']
    fractions = features['X'][:, 1]
    scores, back_pointers, restarts = [], [], [0]
    for i, (start, end) in enumerate(zip(starts, ends)):
        emission = emission_weight * np.log(np.clip(probabilities[start:end], 1e-12, 1.))
        elapsed = float(trip.track[i, 2] - trip.track[i - 1, 2]) if i else 0.
        if i == 0 or elapsed > gap_s:
            scores.append(emission)
            back_pointers.append(np.full(end-start, -1, dtype=int))
            if i:
                restarts.append(i)
            continue

        prev_start, prev_end = starts[i - 1], ends[i - 1]
        previous, current = arcs[prev_start:prev_end], arcs[start:end]
        prev_frac = fractions[prev_start:prev_end]
        curr_frac = fractions[start:end]
        prev_length = lengths[previous]
        destination_nodes = trip.arcs[current, 0]
        network_lengths = np.array([
            network(int(node))[0][destination_nodes] for node in trip.arcs[previous, 1]
        ])
        travel = ((1 - prev_frac) * prev_length)[:, None] + network_lengths + curr_frac * lengths[current]
        signed_progress = (curr_frac[None, :] - prev_frac[:, None]) * prev_length[:, None]
        same_arc = (previous[:, None] == current[None, :]) & (signed_progress >= -reverse_tolerance_m)
        travel[same_arc] = np.maximum(signed_progress[same_arc], 0.)
        displacement = float(np.linalg.norm(points[i] - points[i - 1]))
        valid = np.isfinite(travel) & (travel <= max(200., 60. * max(1., elapsed)))
        transitions = np.where(valid, -np.abs(travel - displacement) / beta_m, -np.inf)
        combined = scores[-1][:, None] + transitions
        if not np.isfinite(combined).any():
            scores.append(emission)
            back_pointers.append(np.full(end-start, -1, dtype=int))
            restarts.append(i)
        else:
            parent = np.argmax(combined, axis=0)
            value = emission + combined[parent, np.arange(end-start)]
            scores.append(value - np.max(value))
            back_pointers.append(parent)

    chosen = np.empty(len(starts), dtype=int)
    state = int(np.argmax(scores[-1]))
    for i in range(len(starts)-1, -1, -1):
        chosen[i] = starts[i] + state
        parent = back_pointers[i][state]
        if i:
            state = int(np.argmax(scores[i - 1])) if parent < 0 else int(parent)

    sections = []
    bridge_count = 0
    breaks = set(restarts)
    for i, row in enumerate(chosen):
        current = int(arcs[row])
        if i in breaks:
            sections.append([current])
            continue
        prev_row = chosen[i - 1]
        previous = int(arcs[prev_row])
        signed_progress = (fractions[row] - fractions[prev_row]) * lengths[previous]
        if previous == current and signed_progress >= -reverse_tolerance_m:
            continue
        source, target = int(trip.arcs[previous, 1]), int(trip.arcs[current, 0])
        bridges = []
        if source != target:
            _, predecessors = network(source)
            while target != source:
                parent = int(predecessors[target])
                if parent < 0:
                    raise ValueError('Chosen transition lacks a directed connecting path')
                bridges.append(arc_for_nodes[parent, target])
                target = parent
        sections[-1].extend(reversed(bridges))
        bridge_count += len(bridges)
        sections[-1].append(current)

    return dict(predicted_arc_id=arcs[chosen], route_sections=sections, bridge_count=bridge_count,
                restart_observations=np.array(restarts, dtype=int),
                settings=dict(gap_s=gap_s, distance_limit_m=distance_limit_m, beta_m=beta_m,
                              reverse_tolerance_m=reverse_tolerance_m, emission_weight=emission_weight))
