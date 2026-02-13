import heapq

from graph_functions import Graph
from omega_language_modelling import llstr, Omegastr
from sprout_dba import delta_star
from sprout_dba_optimized import (
    infinity_run_optim,
    extend_optim,
    update_cache,
    escapes_optim,
)
from sprout_wdba import aut_wdba


def wdba_consistent_optim(
    graph: Graph,
    plus: set[Omegastr],
    minus: set[Omegastr],
    infinity_run_cache: dict,
    positive_states: set[str],
    negative_states: set[str],
    predecessors: dict[str, set[str]],
    successors: dict[str, set[str]],
    affected_words: set[str],
    escapes_positive: dict[str, set[Omegastr]],
    escapes_negative: dict[str, set[Omegastr]],
    node: str,
    target: str,
):
    """Checks if graph is weakly Büchi consistent with sample."""
    escapes_positive_update = {}
    escapes_negative_update = {}
    positive_states_update = positive_states.copy()
    negative_states_update = negative_states.copy()
    cache_update = {}

    for word in minus & affected_words:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            negative_states_update.update(result)
        else:
            exit_string = word[result:]
            escape_prefix = state + exit_string[0]
            if (
                escape_prefix in escapes_positive
                and exit_string in escapes_positive[escape_prefix]
            ):
                return False, None, None, None, None, None
            escapes_negative_update.setdefault(escape_prefix, set()).add(exit_string)

    for word in plus & affected_words:
        success, result, state = infinity_run_optim(graph, word, infinity_run_cache)
        cache_update[word] = (success, result, state)
        if success:
            positive_states_update.update(result)
        else:
            exit_string = word[result:]
            escape_prefix = state + exit_string[0]
            if (
                (
                    escape_prefix in escapes_negative
                    and exit_string in escapes_negative[escape_prefix]
                )
                or escape_prefix in escapes_negative_update
                and exit_string in escapes_negative_update[escape_prefix]
            ):
                return False, None, None, None, None, None
            escapes_positive_update.setdefault(escape_prefix, set()).add(exit_string)

    if positive_states_update & negative_states_update:
        return False, None, None, None, None, None

    if target not in successors[node] and node in successors[target]:
        common = successors[target] & predecessors[node]
        if common & negative_states_update and common & positive_states_update:
            return False, None, None, None, None, None

    return (
        True,
        cache_update,
        positive_states_update,
        negative_states_update,
        escapes_positive_update,
        escapes_negative_update,
    )


def sprout_wdba_optim(plus, minus, square_threshold=False):
    """
    Computes a weak deterministic Büchi automaton consistent with the sample, if possible.
    Based on Sprout algorithm by Bohn and Löding from Constructing Deterministic
    omega-Automata from Examples by an Extension of the RPNI Algorithm.
    Employs a cache to compute runs faster and stores predecessors and successors
    to check for weakness faster.

    Args:
        plus: Words that are to be accepted.
        minus: Words that are to be rejected.
        square_threshold: Should the original square threshold from Sprout be used?

    Returns:
        The resulting automaton.
    """
    initial_state = llstr("")
    graph_dict = Graph({initial_state: {}})
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
    predecessors = {initial_state: {initial_state}}
    successors = {initial_state: {initial_state}}
    escapes_positive = {}
    for word in plus:
        escapes_positive.setdefault(word[0], set()).add(word)
    escapes_negative = {}
    for word in minus:
        escapes_negative.setdefault(word[0], set()).add(word)
    negative_states = set()
    positive_states = set()
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

        u_hat = delta_star(graph_dict, initial_state, u)
        u_hat_a = u_hat + a
        try:
            affected_words = escaping_edge_to_words.pop(u_hat_a)
        except KeyError:
            continue

        if len(u) > threshold:
            return aut_wdba(
                extend_optim(graph_dict, plus, infinity_run_cache),
                minus,
                infinity_run_cache,
            )

        escapes_positive_u_hat_a = escapes_positive.pop(u_hat_a, set())
        escapes_negative_u_hat_a = escapes_negative.pop(u_hat_a, set())

        found_edge = False
        for q in sorted(graph_dict):
            graph_dict[u_hat][a] = q

            (
                consistent,
                cache_update,
                positive_states_update,
                negative_states_update,
                escapes_positive_update,
                escapes_negative_update,
            ) = wdba_consistent_optim(
                graph_dict,
                plus,
                minus,
                infinity_run_cache,
                positive_states,
                negative_states,
                predecessors,
                successors,
                affected_words,
                escapes_positive,
                escapes_negative,
                u_hat,
                q,
            )
            if consistent:
                infinity_run_cache = cache_update
                positive_states = positive_states_update
                negative_states = negative_states_update

                for escape_prefix, exit_strings in escapes_positive_update.items():
                    escapes_positive.setdefault(escape_prefix, set()).update(
                        exit_strings
                    )

                for escape_prefix, exit_strings in escapes_negative_update.items():
                    escapes_negative.setdefault(escape_prefix, set()).update(
                        exit_strings
                    )

                new_successors = successors[q]
                for new_predeccessor in predecessors[u_hat]:
                    successors[new_predeccessor].update(new_successors)

                new_predecessors = predecessors[u_hat]
                for new_successor in successors[q]:
                    predecessors[new_successor].update(new_predecessors)

                scc_states = successors[q] & predecessors[q]
                if scc_states & positive_states:
                    positive_states.update(scc_states)
                elif scc_states & negative_states:
                    negative_states.update(scc_states)

                found_edge = True
                break

        if not found_edge:
            graph_dict[u_hat_a] = {}
            graph_dict[u_hat][a] = u_hat_a

            successors[u_hat_a] = {u_hat_a}
            predecessors[u_hat_a] = predecessors[u_hat] | {u_hat_a}
            for predecessor in predecessors[u_hat_a]:
                successors[predecessor].add(u_hat_a)

            for word in escapes_positive_u_hat_a:
                escapes_positive.setdefault(u_hat_a + word[1], set()).add(word[1:])

            for word in escapes_negative_u_hat_a:
                escapes_negative.setdefault(u_hat_a + word[1], set()).add(word[1:])

            update_cache(graph_dict, affected_words, infinity_run_cache)

        escapes_optim(
            graph_dict,
            plus,
            minus,
            infinity_run_cache,
            affected_words,
            escaping_edge_to_words,
            escaping,
            escaping_set,
        )

    return aut_wdba(graph_dict, minus, infinity_run_cache)
