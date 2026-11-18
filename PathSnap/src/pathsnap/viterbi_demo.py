from .viterbi import viterbi_decode


def demo_transition(prev_arc: int, curr_arc: int, t: int) -> float:
    # MOCK ONLY: replace this with Denis's directed graph transition score.
    allowed = {(10, 20), (20, 30), (11, 21), (21, 30), (10, 21), (11, 20)}
    if prev_arc == curr_arc:
        return -0.3
    return -0.2 if (prev_arc, curr_arc) in allowed else -6.0


if __name__ == '__main__':
    states = [[10, 11], [20, 21], [30, 31]]
    emissions = [[0.70, 0.30], [0.40, 0.60], [0.85, 0.15]]
    result = viterbi_decode(states, emissions, demo_transition)
    print('MOCK VITERBI DEMO (not real trip predictions)')
    print(result)
