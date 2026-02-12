from graph_functions import Graph, Automaton
from omega_language_modelling import llstr


def extend_state(words, q="", graph=None):
    words_left = set(words)
    if graph is None:
        graph = {q: {}}
    index = 0
    state_to_word = {}
    words_to_prefix = {word: q for word in words}

    # build the minimal prefix tree
    while words_left:
        for word in list(words_left):
            old_prefix = words_to_prefix[word]
            new_symbol = word[index % len(word)]
            new_prefix = old_prefix + new_symbol
            if new_prefix in graph:
                # state already exists thus the word that created it has to still be considered
                words_left.add(state_to_word[new_prefix])
            else:
                # add a new state and mark the current word as done
                graph[new_prefix] = {}
                graph[old_prefix][new_symbol] = new_prefix
                state_to_word[new_prefix] = word
                words_left.remove(word)
            words_to_prefix[word] = new_prefix

        index += 1

    # add loops for every word
    for word, prefix in words_to_prefix.items():
        offset = (len(prefix) - len(q)) % len(word)
        current_state = prefix
        i = 0
        for i in range(len(word) - 1):
            new_symbol = word[(i + offset) % len(word)]
            new_state = current_state + new_symbol
            graph[current_state][new_symbol] = new_state
            graph[new_state] = {}
            current_state = new_state
        new_symbol = word[(i + 1 + offset) % len(word)]
        graph[current_state][new_symbol] = prefix

    return graph


def infinity_run(graph_dict, word, initial_state):
    prefix = word.prefix
    loop = word.loop
    loop_len = len(loop)

    if type(graph_dict[initial_state]) is dict:
        delta = lambda x: graph_dict[x]
    else:
        delta = lambda x: graph_dict[x][1]

    current = initial_state
    for i, symbol in enumerate(prefix):
        if symbol in delta(current):
            current = delta(current)[symbol]
        else:
            return False, i, current

    index = 0
    sequence = []
    index_map = {}
    count = len(prefix)
    while True:
        symbol = loop[index]
        if symbol in delta(current):
            current = delta(current)[symbol]
        else:
            return False, count, current
        sequence.append(current)
        if (current, index) in index_map:
            return True, set(sequence[index_map[(current, index)] :]), None
        index_map[(current, index)] = len(sequence)
        index = (index + 1) % loop_len
        count += 1


def escape_prefix(graph_dict, word, initial_state):
    success, result, _ = infinity_run(graph_dict, word, initial_state)
    if success:
        return None
    else:
        return word[: result + 1]


def escapes(graph_dict, plus, initial_state):
    escaping = set()
    for word in plus:
        esc_prefix = escape_prefix(graph_dict, word, initial_state)
        if esc_prefix is not None:
            escaping.add(esc_prefix)
    return sorted(escaping)


def extend(graph_dict, plus, initial_state):
    escape_strings = {}
    for word in plus:
        success, count, state = infinity_run(graph_dict, word, initial_state)
        if not success:
            escape_strings.setdefault(state, set()).add(
                word[count : count + len(word.loop)]
            )

    for q0, loops in escape_strings.items():
        extend_state(loops, q0, graph_dict)

    return graph_dict


def infinity_set(graph_dict, word, initial_state):
    success, result, _ = infinity_run(graph_dict, word, initial_state)
    if success:
        return result
    else:
        return None


def büchi_marking(graph_dict, minus, initial_state):
    negative_states = set()

    for word in minus:
        state_set = infinity_set(graph_dict, word, initial_state)
        if state_set is not None:
            negative_states |= state_set

    return set(graph_dict) - negative_states


def aut_dba(graph_dict, minus, initial_state):
    accepting_states = büchi_marking(graph_dict, minus, initial_state)
    for state, edges in graph_dict.items():
        graph_dict[state] = [state in accepting_states, edges]

    return Automaton(graph_dict)


def delta(graph_dict, q, w):
    current_state = q

    for a in w:
        try:
            current_state = graph_dict[current_state][a]
        except KeyError:
            return None

    return current_state


def büchi_consistent(graph_dict, plus, minus, initial_state):
    escapes_negative = {}
    negative_states = set()

    for word in minus:
        success, result, state = infinity_run(graph_dict, word, initial_state)
        if success:
            negative_states |= result
        else:
            escapes_negative.setdefault(state, set()).add(word[result:])

    for word in plus:
        success, result, state = infinity_run(graph_dict, word, initial_state)
        if success:
            found_state = False
            for q in result:
                if q not in negative_states:
                    found_state = True
                    break
            if not found_state:
                return False
        elif state in escapes_negative and word[result:] in escapes_negative[state]:
            return False

    return True


def sprout_dba(plus, minus, square_threshold=False):
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

    escaping = escapes(graph_dict, plus, initial_state)
    while escaping:
        ua = escaping[0]
        u = ua[:-1]
        a = ua[-1]

        if len(u) > threshold:
            return aut_dba(
                extend(graph_dict, plus, initial_state), minus, initial_state
            )

        u_hat = delta(graph_dict, initial_state, u)

        found_edge = False
        for q in sorted(graph_dict):
            graph_dict[u_hat][a] = q

            if büchi_consistent(graph_dict, plus, minus, initial_state):
                found_edge = True
                break

        if not found_edge:
            u_hat_a = u_hat + a
            graph_dict[u_hat_a] = {}
            graph_dict[u_hat][a] = u_hat_a

        escaping = escapes(graph_dict, plus, initial_state)

    return aut_dba(graph_dict, minus, initial_state)


def is_accepting(graph_dict, word, initial_state):
    success, state_set, _ = infinity_run(graph_dict, word, initial_state)

    if not success:
        return False

    for state in state_set:
        if graph_dict[state][0]:
            return True
    return False
