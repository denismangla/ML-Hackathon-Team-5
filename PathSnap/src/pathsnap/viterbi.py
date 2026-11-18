from __future__ import annotations
from typing import Callable, Sequence
import numpy as np


def viterbi_decode(
    candidate_arc_ids: Sequence[Sequence[int]],
    emission_probabilities: Sequence[Sequence[float]],
    transition_log_probability: Callable[[int, int, int], float],
    probability_floor: float = 1e-12,
):
    """Decode the most likely arc sequence using log-space Viterbi.

    transition_log_probability(prev_arc_id, curr_arc_id, time_index)
    must return a log probability/score (larger is better; impossible = -inf).
    This is a decoder skeleton: Denis must connect the callback to directed
    network distance/connectivity before using it for real predictions.
    """
    if len(candidate_arc_ids) == 0 or len(candidate_arc_ids) != len(emission_probabilities):
        raise ValueError('Candidate states and emissions must have the same non-zero length.')
    states = [list(map(int, x)) for x in candidate_arc_ids]
    emissions = [np.asarray(x, dtype=float) for x in emission_probabilities]
    for t, (s, e) in enumerate(zip(states, emissions)):
        if len(s) == 0 or len(s) != len(e):
            raise ValueError(f'Observation {t}: candidate IDs and scores must be non-empty and same length.')
        if not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError(f'Observation {t}: emission values must be finite non-negative probabilities.')

    dp = [np.log(np.clip(emissions[0], probability_floor, 1.0))]
    back = [np.full(len(states[0]), -1, dtype=int)]
    breaks = []
    for t in range(1, len(states)):
        emit_log = np.log(np.clip(emissions[t], probability_floor, 1.0))
        cur = np.full(len(states[t]), -np.inf, dtype=float)
        ptr = np.full(len(states[t]), -1, dtype=int)
        for j, curr_arc in enumerate(states[t]):
            trans_scores = np.array([
                dp[t - 1][i] + transition_log_probability(prev_arc, curr_arc, t)
                for i, prev_arc in enumerate(states[t - 1])
            ], dtype=float)
            if np.isfinite(trans_scores).any():
                best_i = int(np.argmax(trans_scores))
                cur[j] = trans_scores[best_i] + emit_log[j]
                ptr[j] = best_i
        if not np.isfinite(cur).any():
            # No feasible transition: explicitly mark a sequence break and restart
            # from the current emissions instead of fabricating network continuity.
            breaks.append(t)
            cur = emit_log.copy()
            ptr[:] = -1
        dp.append(cur)
        back.append(ptr)

    selected = [0] * len(states)
    selected[-1] = int(np.argmax(dp[-1]))
    for t in range(len(states) - 1, 0, -1):
        prev = int(back[t][selected[t]])
        selected[t - 1] = prev if prev >= 0 else int(np.argmax(dp[t - 1]))
    best_arcs = [states[t][selected[t]] for t in range(len(states))]
    total_score = float(dp[-1][selected[-1]])
    return {
        'predicted_arc_ids_by_observation': best_arcs,
        'selected_candidate_indices': selected,
        'route_break_observation_indices': breaks,
        'viterbi_score': total_score,
    }
