r"""
Implements the **Sprout** algorithm by Bohn and Löding from
*Constructing Deterministic $\omega$-Automata from Examples
by an Extension of the RPNI Algorithm* for deterministic Büchi automata.
Uses a cache to speed up run computations.
"""

import heapq

from .graph_functions import Graph, Automaton
from .omega_language_modelling import llstr, Omegastr
from .sprout_dba import extend_state, delta_star


def infinity_run_optim(
    graph: Graph, word: Omegastr, infinity_run_cache: dict
) -> tuple[bool, int | set[str], str | None]:
    """
    Computes the infinity run of a UP word in a graph.

    Args:
        graph: Graph in which the run happens.
        word: Word for which to compute the run.
        infinity_run_cache: Cache or runs in graph.

    Returns:
        A tuple (infinite, infinite_set, escape_state):

        - infinite: Is the run infinite or does the word escape?
        - infinite_set: The index in the word that escapes,
          or the set of infinitely occurring states.
        - escape_state: State from which word escapes if it does.
    """
    initial_state = graph.get_start()

    if word in infinity_run_cache:
        success, a, b = infinity_run_cache[word]
        if success:
            return success, a, b
        else:
            word = word[a:]
            initial_state = b
    else:
        a = 0

    if type(graph[initial_state]) is dict:
        delta = lambda x: graph[x]
    else:
        delta = lambda x: graph[x][1]

    prefix = word.prefix
    loop = word.loop
    loop_len = len(loop)

    current = initial_state
    for i, symbol in enumerate(prefix):
        if symbol in delta(current):
            current = delta(current)[symbol]
        else:
            return False, i + a, current

    index = 0
    sequence = []
    index_map = {}
    count = len(prefix)
    while True:
        symbol = loop[index]
        if symbol in delta(current):
            current = delta(current)[symbol]
        else:
            return False, count + a, current
        sequence.append(current)
        if (current, index) in index_map:
            return True, set(sequence[index_map[(current, index)] :]), None
        index_map[(current, index)] = len(sequence)
        index = (index + 1) % loop_len
        count += 1


def escapes_optim(
    graph: Graph,
    plus: set[Omegastr],
    minus: set[Omegastr],
    infinity_run_cache: dict,
    affected_words: set[Omegastr],
    escaping_edge_to_words: dict,
    escaping_list: list[Omegastr],
    escaping_set: set[Omegastr],
):
    """Updates the data used to compute the minimal escaping prefix."""
    for word in (plus | minus) & affected_words:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        if not success:
            esc_prefix = word[: result + 1]
            esc_edge = state + esc_prefix[-1]
            escaping_edge_to_words.setdefault(esc_edge, set()).add(word)
            if word in plus and esc_prefix not in escaping_set:
                heapq.heappush(escaping_list, esc_prefix)
                escaping_set.add(esc_prefix)


def extend_optim(graph: Graph, plus: set[Omegastr], infinity_run_cache: dict) -> Graph:
    """
    Extends the graph by adding disjunct loops for exit strings.
    Implements **Extend** from **Sprout**.
    """
    escape_strings = {}
    for word in plus:
        success, count, state = infinity_run_optim(graph, word, infinity_run_cache)
        if not success:
            escape_strings.setdefault(state, set()).add(
                word[count : count + len(word.loop)]
            )

    for q0, loops in escape_strings.items():
        extend_state(loops, q0, graph)

    return graph


def infinity_set_optim(
    graph: Graph, word: Omegastr, infinity_run_cache: dict
) -> None | set[str]:
    """Gets the infinite set from the word."""
    success, result, _ = infinity_run_optim(graph, word, infinity_run_cache)
    if success:
        return result
    else:
        return None


def buchi_marking_optim(
    graph: Graph, minus: set[Omegastr], infinity_run_cache: dict
) -> set[str]:
    """
    Computes the accepting states to produce a Büchi marking rejecting negative words.
    Implements **BuchiCons** from **Sprout**.
    """
    negative_states = set()

    for word in minus:
        state_set = infinity_set_optim(graph, word, infinity_run_cache)
        if state_set is not None:
            negative_states |= state_set

    return set(graph) - negative_states


def aut_dba_optim(
    graph: Graph, minus: set[Omegastr], infinity_run_cache: dict
) -> Automaton:
    """Turns graph into a deterministic Büchi Automaton that rejects negative words."""
    accepting_states = buchi_marking_optim(graph, minus, infinity_run_cache)
    for state, edges in graph.items():
        graph[state] = [state in accepting_states, edges]

    return Automaton(graph)


def buchi_consistent_optim(
    graph: Graph, plus: set[Omegastr], minus: set[Omegastr], infinity_run_cache: dict
) -> tuple[bool, None | dict]:
    """
    Checks if graph is Büchi consistent and gives info to update cache efficiently.
    """
    escapes_negative = {}
    negative_states = set()
    cache_update = {}

    for word in minus:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            negative_states |= result
        else:
            if state in escapes_negative:
                escapes_negative[state].add(word[result:])
            else:
                escapes_negative[state] = {word[result:]}

    for word in plus:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            found_state = False
            for q in result:
                if q not in negative_states:
                    found_state = True
                    break
            if not found_state:
                return False, None
        elif state in escapes_negative and word[result:] in escapes_negative[state]:
            return False, None

    return True, cache_update


def update_cache(graph: Graph, affected_words: set[Omegastr], infinity_run_cache: dict):
    """Updates the cache for affected words."""
    for word in affected_words:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        infinity_run_cache[word] = (success, result, state)


def sprout_dba_optim(plus, minus, square_threshold=False):
    """
    Computes a deterministic Büchi automaton consistent with the sample, if possible.
    Based on **Sprout** algorithm by Bohn and Löding from *Constructing Deterministic
    omega-Automata from Examples by an Extension of the RPNI Algorithm*.
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
    escaping_list = list(escaping_set)
    heapq.heapify(escaping_list)
    while escaping_list:
        ua = heapq.heappop(escaping_list)
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
            return aut_dba_optim(
                extend_optim(graph, plus, infinity_run_cache),
                minus,
                infinity_run_cache,
            )

        found_edge = False
        for q in sorted(graph):
            graph[u_hat][a] = q

            consistent, cache_update = buchi_consistent_optim(
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
            escaping_list,
            escaping_set,
        )

    return aut_dba_optim(graph, minus, infinity_run_cache)
