import random
from collections import deque

from utils import FastRandomBag


def rename_states_by_prefix(automata, start=""):
    """
    Rename states in a DFA by the smallest prefix reaching them from the start.

    Works with arbitrary (hashable) state names (str, int, etc.).
    """
    if start not in automata:
        start = next(iter(automata))
    # Initialize BFS with (state, prefix)
    queue = deque([(start, "")])
    prefix_for_state = {start: ""}

    while queue:
        state, prefix = queue.popleft()
        _, transitions = automata[state]

        # Explore outgoing transitions in lexicographic order
        for symbol in sorted(transitions.keys()):
            next_state = transitions[symbol]
            if next_state not in prefix_for_state:
                new_prefix = prefix + symbol
                prefix_for_state[next_state] = new_prefix
                queue.append((next_state, new_prefix))

    # Build renamed DFA
    renamed = {}
    for old_state, prefix in prefix_for_state.items():
        accepting, transitions = automata[old_state]
        renamed[prefix] = [accepting, {sym: prefix_for_state[next_state] for sym, next_state in transitions.items()}]

    return renamed


def _generate_wdba_dict(total_states, symbols, prob_acc):
    state_count = 1
    accepting_states = set()
    rejecting_states = set()
    graph_dict = {0: [random.random() < prob_acc, {}]}
    predecessors = {0: {0}}
    successors = {0: {0}}
    if graph_dict[0][0]:
        accepting_states.add(0)
    else:
        rejecting_states.add(0)
    stack = FastRandomBag([(0, symbol) for symbol in symbols])
    while stack:
        node, symbol = next(stack)
        if random.random() < (total_states - state_count) / total_states:
            accepting = random.random() < prob_acc
            if accepting:
                accepting_states.add(state_count)
            else:
                rejecting_states.add(state_count)
            graph_dict[state_count] = [accepting, {}]
            for sym in symbols:
                stack.add((state_count, sym))
            graph_dict[node][1][symbol] = state_count
            successors[state_count] = {state_count}
            predecessors[state_count] = predecessors[node] | {state_count}
            for predecessor in predecessors[state_count]:
                successors[predecessor].add(state_count)
            state_count += 1
        else:
            targets = FastRandomBag(range(state_count))
            found = False
            while not found:
                target = next(targets)
                found = True
                if target in successors[node]:
                    break
                elif node not in successors[target]:
                    break
                else:
                    common = successors[target] & predecessors[node]
                    if common & accepting_states and common & rejecting_states:
                        found = False
                        continue

            graph_dict[node][1][symbol] = target

            new_successors = successors[target]
            for new_predecessor in predecessors[node]:
                successors[new_predecessor].update(new_successors)

            new_predecessors = predecessors[node]
            for new_successor in successors[target]:
                predecessors[new_successor].update(new_predecessors)

    return graph_dict


def _generate_dba_dict(total_states, symbols, prob_acc, new_prob) -> dict[int, list]:
    state_count = 1
    graph_dict = {0: [random.random() < prob_acc, {}]}
    stack = FastRandomBag([(0, symbol) for symbol in symbols])
    while stack:
        node, symbol = next(stack)
        if state_count < total_states and random.random() < new_prob:
            graph_dict[state_count] = [random.random() < prob_acc, {}]
            for sym in symbols:
                stack.add((state_count, sym))
            graph_dict[node][1][symbol] = state_count
            state_count += 1
        else:
            target = random.randrange(state_count)
            graph_dict[node][1][symbol] = target
    return graph_dict


def generate_graph(total_states, weak=True, symbols=None, prob_acc=0.5, new_prob=1., seed=None, rename=True):
    if seed is not None:
        random.seed(seed)
    if symbols is None:
        symbols = 'ab'
    if weak:
        graph_dict = _generate_wdba_dict(total_states, symbols, prob_acc)
    else:
        graph_dict = _generate_dba_dict(total_states, symbols, prob_acc, new_prob)
    if rename:
        graph_dict = rename_states_by_prefix(graph_dict)
    return graph_dict


def get_alphabet(graph):
    alphabet = "".join(sorted({sym for (_, trans) in graph.values() for sym in trans.keys()}))
    return alphabet


def infinity_set(automaton, word, start_state=""):
    if start_state not in automaton:
        start_state = next(iter(automaton))

    prefix = word.prefix
    loop = word.loop

    # Follow the prefix once to reach the state before the cycle starts repeating
    current = start_state
    for symbol in prefix:
        current = automaton[current][1][symbol]

    index = 0
    sequence = []
    index_map = {}
    while True:
        symbol = loop[index]
        current = automaton[current][1][symbol]
        sequence.append(current)
        if (current, index) in index_map:
            return set(sequence[index_map[(current, index)]:])
        index_map[(current, index)] = len(sequence)
        index = (index + 1) % len(loop)


def infinity_set_robust(automaton, word, start_state=""):
    if start_state not in automaton:
        start_state = next(iter(automaton))

    prefix = word.prefix
    loop = word.loop

    # Follow the prefix once to reach the state before the cycle starts repeating
    current = start_state
    for symbol in prefix:
        if symbol in automaton[current][1]:
            current = automaton[current][1][symbol]
        else:
            return None

    index = 0
    sequence = []
    index_map = {}
    while True:
        symbol = loop[index]
        if symbol in automaton[current][1]:
            current = automaton[current][1][symbol]
        else:
            return None
        sequence.append(current)
        if (current, index) in index_map:
            return set(sequence[index_map[(current, index)]:])
        index_map[(current, index)] = len(sequence)
        index = (index + 1) % len(loop)


def _is_accepting(automaton, word, start_state=""):
    # TODO: remove
    state_set = infinity_set(automaton, word, start_state)

    for state in state_set:
        if automaton[state][0]:
            return True
    return False


def strongly_connected_components(automaton, nontrivial=True):
    """
    Compute all strongly connected components (SCCs) of a deterministic Büchi automaton.

    Args:
        automaton: dict of the form {state: [is_accepting, {symbol: next_state, ...}]}

    Returns:
        A list of SCCs, where each SCC is a set of states.
    """
    index = 0
    stack = []
    indices = {}
    lowlinks = {}
    on_stack = set()
    sccs = []

    def strongconnect(state):
        nonlocal index
        indices[state] = index
        lowlinks[state] = index
        index += 1
        stack.append(state)
        on_stack.add(state)

        # Explore all successors
        for _, next_state in automaton[state][1].items():
            if next_state not in indices:
                strongconnect(next_state)
                lowlinks[state] = min(lowlinks[state], lowlinks[next_state])
            elif next_state in on_stack:
                lowlinks[state] = min(lowlinks[state], indices[next_state])

        # If state is a root node, pop the stack and generate an SCC
        if lowlinks[state] == indices[state]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.add(w)
                if w == state:
                    break
            sccs.append(scc)

    # Run Tarjan's algorithm for all unvisited states
    for state in automaton:
        if state not in indices:
            strongconnect(state)

    if nontrivial:
        return [scc for scc in sccs if len(scc) > 1]
    else:
        return sccs


def check_weak(graph):
    sccs = strongly_connected_components(graph)
    for scc in sccs:
        found_positive, found_negative = False, False
        for state in scc:
            if graph[state][0]:
                found_positive = True
            else:
                found_negative = True
            if found_positive and found_negative:
                return False
    return True
