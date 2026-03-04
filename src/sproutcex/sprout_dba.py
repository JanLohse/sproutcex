r"""
Implements the **Sprout** algorithm by Bohn and Löding from
*Constructing Deterministic :math:`\omega`-Automata from Examples
by an Extension of the RPNI Algorithm* for deterministic Büchi automata.
"""

from .graph_functions import Automaton, Graph
from .omega_language_modelling import Omegastr, llstr


def extend_state(loops: set[Omegastr], state="", graph: None | Graph = None) -> Graph:
    """
    Extend graph by adding disjunct loops to a specific state.

    Args:
        loops: Set of loops to add.
        state: State in graph on which to add the loops.
        graph: Graph to extend. Creates single state graph if left empty.

    Returns:
        Extended graph.
    """
    words_left = set(loops)
    if graph is None:
        graph = {state: {}}
    index = 0
    state_to_word = {}
    words_to_prefix = {word: state for word in loops}

    # build the minimal prefix tree
    while words_left:
        for word in list(words_left):
            old_prefix = words_to_prefix[word]
            new_symbol = word[index % len(word)]
            new_prefix = old_prefix + new_symbol
            if new_prefix in graph:
                # state already exists thus the word that created it has to still be
                # considered
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
        offset = (len(prefix) - len(state)) % len(word)
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


def infinity_run(
    graph: Graph, word: Omegastr
) -> tuple[bool, int | set[str], str | None]:
    """
    Computes the infinity run of a UP word in a graph.

    Args:
        graph: Graph in which the run happens.
        word: Word for which to compute the run.

    Returns:
        A tuple (infinite, infinite_set, escape_state):

        - infinite: Is the run infinite or does the word escape?
        - infinite_set: The index in the word that escapes,
          or the set of infinitely occurring states.
        - escape_state: State from which word escapes if it does.
    """
    prefix = word.prefix
    loop = word.loop
    loop_len = len(loop)
    initial_state = graph.get_start()

    if type(graph[initial_state]) is dict:

        def delta(x):
            return graph[x]
    else:

        def delta(x):
            return graph[x][1]

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
            # The word escaped.
            return False, count, current
        sequence.append(current)
        if (current, index) in index_map:
            # The loop has been closed.
            return True, set(sequence[index_map[(current, index)] :]), None
        index_map[(current, index)] = len(sequence)
        index = (index + 1) % loop_len
        count += 1


def escape_prefix(graph: Graph, word: Omegastr) -> None | str:
    """Compute escape prefix if the word escapes from the graph."""
    success, result, _ = infinity_run(graph, word)
    if success:
        return None
    else:
        return word[: result + 1]


def escapes(graph: Graph, plus: set[Omegastr]) -> list[str]:
    """Computes a sorted list of escape prefixes in the graph."""
    escaping = set()
    for word in plus:
        esc_prefix = escape_prefix(graph, word)
        if esc_prefix is not None:
            escaping.add(esc_prefix)
    return sorted(escaping)


def extend(graph: Graph, plus: set[Omegastr]) -> Graph:
    """
    Extends the graph by adding disjunct loops for exit strings.
    Implements **Extend** from **Sprout**.
    """
    escape_strings = {}
    for word in plus:
        success, count, state = infinity_run(graph, word)
        if not success:
            escape_strings.setdefault(state, set()).add(
                word[count : count + len(word.loop)]
            )

    for q0, loops in escape_strings.items():
        extend_state(loops, q0, graph)

    return graph


def infinity_set(graph: Graph, word: Omegastr) -> None | set[str]:
    """Gets the infinite set from the word."""
    success, result, _ = infinity_run(graph, word)
    if success:
        return result
    else:
        return None


def buchi_marking(graph: Graph, minus: set[Omegastr]) -> set[str]:
    """
    Computes the accepting states to produce a Buchi marking rejecting negative words.
    """
    negative_states = set()

    for word in minus:
        state_set = infinity_set(graph, word)
        if state_set is not None:
            negative_states |= state_set

    return set(graph) - negative_states


def aut_dba(graph: Graph, minus: set[Omegastr]) -> Automaton:
    """
    Turns graph into a deterministic Büchi Automaton that rejects negative words.
    Implements the **Aut** function from **Sprout**.
    """
    accepting_states = buchi_marking(graph, minus)
    for state, edges in graph.items():
        graph[state] = [state in accepting_states, edges]

    return Automaton(graph)


def delta_star(graph: Graph, q: str, w: str) -> str | None:
    """
    Computes state that is reached in graph from :math:`q` after reading :math:`w`.
    """
    current_state = q

    for a in w:
        try:
            current_state = graph[current_state][a]
        except KeyError:
            return None

    return current_state


def buchi_consistent(graph: Graph, plus: set[Omegastr], minus: set[Omegastr]) -> bool:
    """Checks if graph is Büchi consistent."""
    escapes_negative = {}
    negative_states = set()

    # Find states that have to be rejecting and exit strings of negative words.
    for word in minus:
        success, result, state = infinity_run(graph, word)
        if success:
            negative_states |= result
        else:
            escapes_negative.setdefault(state, set()).add(word[result:])

    # Test if there are conflicts between positive and negative words.
    for word in plus:
        success, result, state = infinity_run(graph, word)
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


def sprout_dba(
    plus: set[Omegastr], minus: set[Omegastr], square_threshold=False
) -> Automaton:
    """
    Computes a deterministic Büchi automaton consistent with the sample, if possible.
    Based on Sprout algorithm by Bohn and Löding from Constructing Deterministic
    omega-Automata from Examples by an Extension of the RPNI Algorithm.

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

    escaping = escapes(graph, plus)
    while escaping:
        ua = escaping[0]
        u = ua[:-1]
        a = ua[-1]

        # Terminate if threshold reached.
        if len(u) > threshold:
            return aut_dba(extend(graph, plus), minus)

        u_hat = delta_star(graph, initial_state, u)

        # Search for target of edge in existing states.
        found_edge = False
        for q in sorted(graph):
            graph[u_hat][a] = q

            if buchi_consistent(graph, plus, minus):
                found_edge = True
                break

        # Add new state if no target was found.
        if not found_edge:
            u_hat_a = u_hat + a
            graph[u_hat_a] = {}
            graph[u_hat][a] = u_hat_a

        escaping = escapes(graph, plus)

    return aut_dba(graph, minus)


def is_accepting(graph_dict, word):
    success, state_set, _ = infinity_run(graph_dict, word)

    if not success:
        return False

    for state in state_set:
        if graph_dict[state][0]:
            return True
    return False
