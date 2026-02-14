r"""
Adapts the **Sprout** algorithm by Bohn and Löding from
*Constructing Deterministic $\omega$-Automata from Examples
by an Extension of the RPNI Algorithm* for weak deterministic Büchi automata.
"""

import heapq

from .graph_functions import Automaton, Graph
from .omega_language_modelling import llstr, Omegastr
from .sprout_dba import delta_star
from .sprout_dba_optimized import (
    infinity_run_optim,
    infinity_set_optim,
    extend_optim,
    update_cache,
    escapes_optim,
)


def compute_sccs(graph: Graph) -> tuple[dict[str, int], dict[int, set]]:
    """Computes the sccs in the given graph."""
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
        for _, next_state in graph[state].items():
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
    for state in graph:
        if state not in indices:
            strongconnect(state)

    return state_to_scc, scc_to_states


def wdba_consistent(
    graph: Graph, plus: set[Omegastr], minus: set[Omegastr], infinity_run_cache: dict
) -> tuple[bool, None | dict]:
    """Checks if graph is weakly Büchi consistent with sample."""
    escapes_negative = {}
    negative_sccs = set()
    cache_update = {}

    state_to_scc, _ = compute_sccs(graph)

    for word in minus:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            negative_sccs.add(state_to_scc[next(iter(result))])
        else:
            escapes_negative.setdefault(state, set()).add(word[result:])

    for word in plus:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            scc_id = state_to_scc[next(iter(result))]
            if scc_id in negative_sccs:
                return False, None
        elif state in escapes_negative and word[result:] in escapes_negative[state]:
            return False, None

    return True, cache_update


def wdba_marking(
    graph: Graph, minus: set[Omegastr], infinity_run_cache: dict
) -> set[str]:
    """Computes the accepting states to produce a weak Büchi marking rejecting negative words."""
    negative_states = set()
    state_to_scc, scc_to_states = compute_sccs(graph)

    for word in minus:
        state_set = infinity_set_optim(graph, word, infinity_run_cache)
        if state_set is not None:
            negative_states |= scc_to_states[state_to_scc[next(iter(state_set))]]

    return set(graph) - negative_states


def aut_wdba(graph: Graph, minus: set[Omegastr], infinity_run_cache: dict) -> Automaton:
    """
    Turns graph into a weak deterministic Büchi Automaton that rejects negative words.
    """
    accepting_states = wdba_marking(graph, minus, infinity_run_cache)
    for state, edges in graph.items():
        graph[state] = [state in accepting_states, edges]

    return Automaton(graph)


def sprout_wdba(
    plus: set[Omegastr], minus: set[Omegastr], square_threshold=False
) -> Automaton:
    """
    Computes a weak deterministic Büchi automaton consistent with the sample, if
    possible. Based on **Sprout** algorithm by Bohn and Löding from *Constructing
    Deterministic $omega$-Automata from Examples by an Extension of the RPNI Algorithm*.
    Employs a cache to compute runs faster.

    Args:
        plus: Words that are to be accepted.
        minus: Words that are to be rejected.
        square_threshold: Should the original square threshold from Sprout be used?

    Returns:
        The resulting automaton.
    """
    initial_state = llstr("")
    graph = Graph({initial_state: {}})
    samples = {*plus, *minus}
    if square_threshold:
        threshold = (
            max([len(x.prefix) for x in samples] + [0])
            + max([len(x.loop) for x in samples] + [0]) ** 2
            + 1
        )
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

        u_hat = delta_star(graph, initial_state, u)
        u_hat_a = u_hat + a
        try:
            affected_words = escaping_edge_to_words.pop(u_hat_a)
        except KeyError:
            continue

        if len(u) > threshold:
            return aut_wdba(
                extend_optim(graph, plus, infinity_run_cache),
                minus,
                infinity_run_cache,
            )

        found_edge = False
        for q in sorted(graph):
            graph[u_hat][a] = q

            consistent, cache_update = wdba_consistent(
                graph, plus, minus, infinity_run_cache
            )
            if consistent:
                infinity_run_cache = cache_update
                found_edge = True
                break

        if not found_edge:
            graph[u_hat_a] = {}
            graph[u_hat][a] = u_hat_a
            update_cache(graph, affected_words, infinity_run_cache)

        escapes_optim(
            graph,
            plus,
            minus,
            infinity_run_cache,
            affected_words,
            escaping_edge_to_words,
            escaping,
            escaping_set,
        )

    return aut_wdba(graph, minus, infinity_run_cache)
