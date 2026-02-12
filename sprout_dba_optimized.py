import heapq

from graph_functions import Graph, Automaton
from omega_language_modelling import llstr
from sprout_dba import extend_state, delta


def infinity_run_optim(graph_dict, word, initial_state, infinity_run_cache):
    if word in infinity_run_cache:
        success, a, b = infinity_run_cache[word]
        if success:
            return success, a, b
        else:
            word = word[a:]
            initial_state = b
    else:
        a = 0

    if type(graph_dict[initial_state]) is dict:
        delta = lambda x: graph_dict[x]
    else:
        delta = lambda x: graph_dict[x][1]

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


def escape_prefix_optim(graph_dict, word, initial_state, infinity_run_cache):
    success, result, _ = infinity_run_optim(
        graph_dict, word, initial_state, infinity_run_cache
    )
    if success:
        return None
    else:
        return word[: result + 1]


def escapes_optim(
    graph_dict,
    plus,
    minus,
    initial_state,
    infinity_run_cache,
    affected_words,
    escaping_edge_to_words,
    escaping,
    escaping_set,
):
    for word in (plus | minus) & affected_words:
        success, result, state = infinity_run_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
        if not success:
            esc_prefix = word[: result + 1]
            esc_edge = state + esc_prefix[-1]
            escaping_edge_to_words.setdefault(esc_edge, set()).add(word)
            if word in plus and esc_prefix not in escaping_set:
                heapq.heappush(escaping, esc_prefix)
                escaping_set.add(esc_prefix)


def extend_optim(graph_dict, plus, initial_state, infinity_run_cache):
    escape_strings = {}
    for word in plus:
        success, count, state = infinity_run_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
        if not success:
            escape_strings.setdefault(state, set()).add(
                word[count : count + len(word.loop)]
            )

    for q0, loops in escape_strings.items():
        extend_state(loops, q0, graph_dict)

    return graph_dict


def infinity_set_optim(graph_dict, word, initial_state, infinity_run_cache):
    success, result, _ = infinity_run_optim(
        graph_dict, word, initial_state, infinity_run_cache
    )
    if success:
        return result
    else:
        return None


def büchi_marking_optim(graph_dict, minus, initial_state, infinity_run_cache):
    negative_states = set()

    for word in minus:
        state_set = infinity_set_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
        if state_set is not None:
            negative_states |= state_set

    return set(graph_dict) - negative_states


def aut_optim(graph_dict, minus, initial_state, infinity_run_cache):
    accepting_states = büchi_marking_optim(
        graph_dict, minus, initial_state, infinity_run_cache
    )
    for state, edges in graph_dict.items():
        graph_dict[state] = [state in accepting_states, edges]

    return Automaton(graph_dict)


def büchi_consistent_optim(graph_dict, plus, minus, initial_state, infinity_run_cache):
    escapes_negative = {}
    negative_states = set()
    cache_update = {}

    for word in minus:
        success, result, state = infinity_run_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
        cache_update[word] = (success, result, state)
        if success:
            negative_states |= result
        else:
            if state in escapes_negative:
                escapes_negative[state].add(word[result:])
            else:
                escapes_negative[state] = {word[result:]}

    for word in plus:
        success, result, state = infinity_run_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
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


def update_cache(graph_dict, affected_words, initial_state, infinity_run_cache):
    for word in affected_words:
        success, result, state = infinity_run_optim(
            graph_dict, word, initial_state, infinity_run_cache
        )
        infinity_run_cache[word] = (success, result, state)


def sprout_dba_optim(plus, minus, square_threshold=False):
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
            return aut_optim(
                extend_optim(graph_dict, plus, initial_state, infinity_run_cache),
                minus,
                initial_state,
                infinity_run_cache,
            )

        found_edge = False
        for q in sorted(graph_dict):
            graph_dict[u_hat][a] = q

            consistent, cache_update = büchi_consistent_optim(
                graph_dict, plus, minus, initial_state, infinity_run_cache
            )
            if consistent:
                infinity_run_cache = cache_update
                found_edge = True
                break

        if not found_edge:
            graph_dict[u_hat_a] = {}
            graph_dict[u_hat][a] = u_hat_a
            update_cache(graph_dict, affected_words, initial_state, infinity_run_cache)

        escapes_optim(
            graph_dict,
            plus,
            minus,
            initial_state,
            infinity_run_cache,
            affected_words,
            escaping_edge_to_words,
            escaping,
            escaping_set,
        )

    return aut_optim(graph_dict, minus, initial_state, infinity_run_cache)
