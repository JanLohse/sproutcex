import heapq

from omega_language_modelling import llstr
from sprout_dba import delta
from sprout_dba_optimized import infinity_run_optim, infinity_set_optim, extend_optim, update_cache, escapes_optim
from utils import Automaton


def compute_sccs(graph_dict):
    index = 0
    stack = []
    indices = {}
    lowlinks = {}
    on_stack = set()

    scc_index = 0
    state_to_scc = {}
    scc_to_states = {}

    def strongconnect(state):
        nonlocal index, scc_index
        indices[state] = index
        lowlinks[state] = index
        index += 1
        stack.append(state)
        on_stack.add(state)

        # Explore all successors
        for _, next_state in graph_dict[state].items():
            if next_state not in indices:
                strongconnect(next_state)
                lowlinks[state] = min(lowlinks[state], lowlinks[next_state])
            elif next_state in on_stack:
                lowlinks[state] = min(lowlinks[state], indices[next_state])

        # If state is a root node, pop the stack and generate an SCC
        if lowlinks[state] == indices[state]:
            scc_states = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                state_to_scc[w] = scc_index
                scc_states.add(w)
                if w == state:
                    break
            scc_to_states[scc_index] = scc_states
            scc_index += 1

    # Run Tarjan's algorithm for all unvisited states
    for state in graph_dict:
        if state not in indices:
            strongconnect(state)

    return state_to_scc, scc_to_states


def wdba_consistent(graph_dict, plus, minus, initial_state, infinity_run_cache):
    escapes_negative = {}
    negative_sccs = set()
    cache_update = {}

    state_to_scc, _ = compute_sccs(graph_dict)

    for word in minus:
        success, result, state = infinity_run_optim(graph_dict, word, initial_state, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            negative_sccs.add(state_to_scc[next(iter(result))])
        else:
            escapes_negative.setdefault(state, set()).add(word[result:])

    for word in plus:
        success, result, state = infinity_run_optim(graph_dict, word, initial_state, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            scc_id = state_to_scc[next(iter(result))]
            if scc_id in negative_sccs:
                return False, None
        elif state in escapes_negative and word[result:] in escapes_negative[state]:
            return False, None

    return True, cache_update


def wdba_marking(graph_dict, minus, initial_state, infinity_run_cache):
    negative_states = set()
    state_to_scc, scc_to_states = compute_sccs(graph_dict)

    for word in minus:
        state_set = infinity_set_optim(graph_dict, word, initial_state, infinity_run_cache)
        if state_set is not None:
            negative_states |= scc_to_states[state_to_scc[next(iter(state_set))]]

    return set(graph_dict) - negative_states


def aut_wdba(graph_dict, minus, initial_state, infinity_run_cache):
    accepting_states = wdba_marking(graph_dict, minus, initial_state, infinity_run_cache)
    for state, edges in graph_dict.items():
        graph_dict[state] = [state in accepting_states, edges]

    return Automaton(graph_dict)


def sprout_wdba(plus, minus, square_threshold=False):
    initial_state = llstr("")
    graph_dict = {initial_state: {}}
    samples = {*plus, *minus}
    if square_threshold:
        threshold = max([len(x.prefix) for x in samples] + [0]) + max([len(x.loop) for x in samples] + [0]) ** 2 + 1
    else:
        threshold = max([len(x) for x in samples] + [0]) * 2 - 1
    infinity_run_cache = {}
    escaping_edge_to_words = {}
    for word in plus | minus:
        a = word[0]
        escaping_edge_to_words.setdefault(a, set()).add(word)

    escaping_set = {word[0] for word in plus}
    escaping = list(escaping_set)
    heapq.heapify(escaping)
    while escaping:
        ua = heapq.heappop(escaping)
        escaping_set.remove(ua)
        u = ua[:-1]
        a = ua[-1]

        u_hat = delta(graph_dict, initial_state, u)
        u_hat_a = u_hat + a
        try:
            affected_words = escaping_edge_to_words.pop(u_hat_a)
        except KeyError:
            continue

        if len(u) > threshold:
            return aut_wdba(extend_optim(graph_dict, plus, initial_state, infinity_run_cache), minus, initial_state,
                            infinity_run_cache)

        found_edge = False
        for q in sorted(graph_dict):
            graph_dict[u_hat][a] = q

            consistent, cache_update = wdba_consistent(graph_dict, plus, minus, initial_state, infinity_run_cache)
            if consistent:
                infinity_run_cache = cache_update
                found_edge = True
                break

        if not found_edge:
            graph_dict[u_hat_a] = {}
            graph_dict[u_hat][a] = u_hat_a
            update_cache(graph_dict, affected_words, initial_state, infinity_run_cache)

        escapes_optim(graph_dict, plus, minus, initial_state, infinity_run_cache, affected_words,
                      escaping_edge_to_words, escaping, escaping_set)

    return aut_wdba(graph_dict, minus, initial_state, infinity_run_cache)
