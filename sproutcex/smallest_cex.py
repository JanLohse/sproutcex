r"""
Implements the equivalence check and search for smallest counterexamples for
$\omega$-automata. Used to simulate the equivalence queries in learning from smallest
counterexamples.
"""

from collections import deque, defaultdict
from itertools import product
from typing import Any

from .graph_functions import Automaton
from .omega_language_modelling import (
    omegaiter,
    omegaiter_prefix,
    omegaiter_lex,
    omegaiter_expansion,
    llstr,
    OmegastrLoop,
    Omegastr,
    OmegastrPrefix,
    OmegastrLex,
    OmegastrExpansion,
)
from .sprout_dba import is_accepting


def product_of_dba(a: Automaton, b: Automaton) -> Automaton:
    """Computes the product automaton of two automata."""
    init_a = a.get_start()
    init_b = b.get_start()
    stack = [(init_a, init_b)]
    visited = set()
    product_automaton = Automaton(start_node=init_a + "" + init_b)

    while stack:
        # Add a missing product state.
        curr_a, curr_b = stack.pop()
        key = curr_a + "|" + curr_b
        if key in visited:
            continue
        visited.add(key)
        acc_a, trans_a = a.get(curr_a, (False, {}))
        acc_b, trans_b = b.get(curr_b, (False, {}))
        product_automaton[key] = [(acc_a, acc_b), {}]

        for sym in {*trans_a.keys(), *trans_b.keys()}:
            # Add all outgoing edges from the new product state.
            next_a = trans_a.get(sym, "sink")
            next_b = trans_b.get(sym, "sink")
            next_key = next_a + "|" + next_b
            product_automaton[key][1][sym] = next_key
            if next_key not in visited:
                stack.append((next_a, next_b))

    return product_automaton


def find_asymmetric_sccs_product_automaton(automaton):
    """
    Find SCCs in a product automaton that has a run rejected by one and accepted by the
    other automaton.
    """
    # partition states
    state_partition = {
        i: {state for state in automaton if not automaton[state][0][1 - i]}
        for i in (0, 1)
    }
    sccs = []

    for idx in (0, 1):
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()

        def strongconnect(state):
            nonlocal index
            indices[state] = index
            lowlinks[state] = index
            index += 1
            stack.append(state)
            on_stack.add(state)

            # Explore all successors.
            for _, next_state in automaton[state][1].items():
                if next_state not in state_partition[idx]:
                    continue
                if next_state not in indices:
                    strongconnect(next_state)
                    lowlinks[state] = min(lowlinks[state], lowlinks[next_state])
                elif next_state in on_stack:
                    lowlinks[state] = min(lowlinks[state], indices[next_state])

            # If state is a root node, pop the stack and generate an SCC.
            if lowlinks[state] == indices[state]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == state:
                        break
                if (
                    len(scc) > 1
                    or next(iter(scc)) in automaton[next(iter(scc))][1].values()
                ) and any([automaton[s][0][idx] for s in scc]):
                    sccs.append(scc)

        for state in state_partition[idx]:
            if state not in indices and automaton[state][0][idx]:
                strongconnect(state)

    return sccs


def are_dba_equivalent(a: Automaton, b: Automaton) -> bool:
    """Tests whether two deterministic Büchi automata represent the same language."""
    automaton = product_of_dba(a, b)
    sccs = find_asymmetric_sccs_product_automaton(automaton)

    return not bool(sccs)


def smallest_cex(
    a: Automaton, b: Automaton, iterator=omegaiter
) -> tuple[bool, None, None] | tuple[bool, Omegastr, bool]:
    """
    Test for equivalence of two DBA and returns smallest counterexample if they are not.

    Args:
        a: Deterministic Büchi automaton.
        b: Deterministic Büchi automaton.
        iterator: Iterator for Omegastr to specify in what order to test UP words.

    Returns:
        A tuple (equivalent, counterexample, positive):

        - equivalent: True if the automata are equivalent.
        - counterexample: The smallest counterexample, or None if equivalent.
        - positive: True if the counterexample is positive, else None
          when the automata are equivalent.
    """
    # Test for equivalence.
    if are_dba_equivalent(a, b):
        return True, None, None

    # Get combined alphabet and start states.
    alphabet_a = a.get_alphabet()
    alphabet_b = b.get_alphabet()
    alphabet = "".join(sorted(set(alphabet_a + alphabet_b)))

    # Test words one by one if they are in the symmetric difference.
    for word in iterator(alphabet):
        a_result = is_accepting(a, word)
        b_result = is_accepting(b, word)
        if b_result and not a_result:
            return False, word, True
        if a_result and not b_result:
            return False, word, False

    return False, None, None


def smallest_cex_prefix(
    a: Automaton, b: Automaton
) -> tuple[bool, None, None] | tuple[bool, Omegastr, bool]:
    """
    Test for equivalence of two DBA and returns smallest by prefix counterexample if
    they are not.

    Args:
        a: Deterministic Büchi automaton.
        b: Deterministic Büchi automaton.

    Returns:
        A tuple (equivalent, counterexample, positive):

        - equivalent: True if the automata are equivalent.
        - counterexample: The smallest counterexample, or None if equivalent.
        - positive: True if the counterexample is positive, else None
          when the automata are equivalent.
    """
    return smallest_cex(a, b, iterator=omegaiter_prefix)


def smallest_cex_lex(
    a: Automaton, b: Automaton
) -> tuple[bool, None, None] | tuple[bool, Omegastr, bool]:
    """
    Test for equivalence of two DBA and returns smallest by representation
    counterexample if they are not.

    Args:
        a: Deterministic Büchi automaton.
        b: Deterministic Büchi automaton.

    Returns:
        A tuple (equivalent, counterexample, positive):

        - equivalent: True if the automata are equivalent.
        - counterexample: The smallest counterexample, or None if equivalent.
        - positive: True if the counterexample is positive, else None
          when the automata are equivalent.
    """
    return smallest_cex(a, b, iterator=omegaiter_lex)


def smallest_cex_expansion(
    a: Automaton, b: Automaton
) -> tuple[bool, None, None] | tuple[bool, Omegastr, bool]:
    """
    Test for equivalence of two DBA and returns smallest by omega-expansion
    counterexample if they are not.

    Args:
        a: Deterministic Büchi automaton.
        b: Deterministic Büchi automaton.

    Returns:
        A tuple (equivalent, counterexample, positive):

        - equivalent: True if the automata are equivalent.
        - counterexample: The smallest counterexample, or None if equivalent.
        - positive: True if the counterexample is positive, else None
          when the automata are equivalent.
    """
    return smallest_cex(a, b, iterator=omegaiter_expansion)


def smallest_diff_loop_product_automaton(
    automaton: Automaton,
) -> tuple[str, set[str], set[str]] | tuple[None, None, None]:
    """
    Compute the smallest loop rejected by one and accepted by the other automaton of a
    product automaton.
    """
    alphabet = automaton.get_alphabet()

    # Compute SCCs with disagreeing loops and all states in them.
    sccs = find_asymmetric_sccs_product_automaton(automaton)

    scc_states = set().union(*sccs)
    if not scc_states:
        return None, None, None

    # Map each state to its SCC.
    sccs = {i: scc for i, scc in enumerate(sccs)}
    scc_id = {}
    for i, scc in sccs.items():
        for state in scc:
            if state in scc_id and scc_id[state] != i:
                for old_state in sccs[scc_id[state]]:
                    scc_id[old_state] = i
            else:
                scc_id[state] = i

    # Init dp (dynamic programming storage) and words_by_length.
    dp = defaultdict(dict)
    for state in scc_states:
        if not all(automaton[state][0]):
            dp[state][None] = (state, automaton[state][0])
    dp[None] = defaultdict(dict)
    for a in alphabet:
        for state in scc_states:
            target = automaton[state][1].get(a)
            marking = tuple(
                a or b for a, b in zip(automaton[state][0], automaton[target][0])
            )
            if (
                target in scc_states
                and scc_id[target] == scc_id[state]
                and not all(marking)
            ):
                dp[state][a] = (target, marking)
    words_by_length = {}
    k = 0

    while True:
        # Compute all Lyndon words of length k.
        k += 1
        new_words = set()
        decompositions = {}
        words_by_length[k] = set()
        for i in range(1, k // 2 + 1):
            j = k - i
            for a, b in product(words_by_length[i], words_by_length[j]):
                if a != b:
                    if b < a:
                        a, b = b, a
                    ab = a + b
                    if ab not in new_words:
                        new_words.add(ab)
                        decompositions[ab] = a, b
        if k == 1:
            decompositions = {a: (None, a) for a in alphabet}
            new_words = [a for a in alphabet]
        else:
            new_words = sorted(new_words)

        if not new_words:
            continue

        for current_word in new_words:
            left, right = decompositions[current_word]
            accepting_transition = False
            for state in scc_states:
                halfway = dp[state].get(left)
                if halfway is None:
                    continue
                target = dp[halfway[0]].get(right)
                if target is not None:
                    marking = tuple(a or b for a, b in zip(halfway[1], target[1]))
                    dp[state][current_word] = (target[0], marking)
                    words_by_length[k].add(current_word)
                    if marking[0] != marking[1]:
                        accepting_transition = True

            # Only check if any accepting transitions.
            if not accepting_transition:
                continue

            # Search for cycles within SCCs.
            states_left = set(scc_states)
            loop_states_a = set()
            loop_states_b = set()
            while states_left:
                current_state = next(iter(states_left))
                path = set()
                while (
                    current_state is not None
                    and current_state in states_left
                    and current_state not in path
                ):
                    path.add(current_state)
                    states_left.remove(current_state)
                    current_state = dp[current_state].get(current_word)
                    if current_state is None:
                        break
                    else:
                        current_state = current_state[0]
                if current_state in path:
                    loop_path = set()
                    found_a, found_b = False, False
                    while current_state not in loop_path and not (found_a and found_b):
                        loop_path.add(current_state)
                        found_a = found_a or dp[current_state][current_word][1][0]
                        found_b = found_b or dp[current_state][current_word][1][1]
                        current_state = dp[current_state].get(current_word)[0]
                    if found_a and not found_b:
                        loop_states_a.update(loop_path)
                    elif not found_a and found_b:
                        loop_states_b.update(loop_path)

                states_left -= path

            if loop_states_a or loop_states_b:
                return current_word, loop_states_a, loop_states_b

    return None, None, None


def smallest_cex_loop(
    a: Automaton, b: Automaton
) -> tuple[bool, None, None] | tuple[bool, Omegastr, bool]:
    """
    Test for equivalence of two DBA and returns smallest by loop ordering counterexample
    if they are not.

    Args:
        a: Deterministic Büchi automaton.
        b: Deterministic Büchi automaton.

    Returns:
        A tuple (equivalent, counterexample, positive):

        - equivalent: True if the automata are equivalent.
        - counterexample: The smallest counterexample, or None if equivalent.
        - positive: True if the counterexample is positive, else None
          when the automata are equivalent.
    """
    alphabet_a = a.get_alphabet()
    alphabet_b = b.get_alphabet()
    alphabet = "".join(sorted(set(alphabet_a + alphabet_b)))

    product_automaton = product_of_dba(a, b)

    # Find the smallest loop and states from which the loop starts.
    loop, start_points_a, start_points_b = smallest_diff_loop_product_automaton(
        product_automaton
    )
    if loop is None:
        return True, None, None

    # Build mapping from indices of the loop to states from which an accepting run with
    # the loop starts in this state from the index.
    m = len(loop)
    index_states_a = {i: set() for i in range(m)}
    index_states_a[0] = start_points_a
    index_states_b = {i: set() for i in range(m)}
    index_states_b[0] = start_points_b

    # Invert edges to compute predecessors.
    reverse_mapping = {
        state: {a: set() for a in alphabet} for state in product_automaton
    }
    for state in product_automaton:
        for a in alphabet:
            target = product_automaton[state][1][a]
            reverse_mapping[target][a].add(state)

    # Trace back from the loop starting states to complete the mapping from indices.
    queue = deque()
    for state in start_points_a:
        queue.append((0, state))
    while queue:
        index, state = queue.popleft()
        index = (index - 1) % m
        symbol = loop[index]
        for predecessor in reverse_mapping[state][symbol]:
            if predecessor not in index_states_a[index]:
                index_states_a[index].add(predecessor)
                queue.append((index, predecessor))

    queue = deque()
    for state in start_points_b:
        queue.append((0, state))
    while queue:
        index, state = queue.popleft()
        index = (index - 1) % m
        symbol = loop[index]
        for predecessor in reverse_mapping[state][symbol]:
            if predecessor not in index_states_b[index]:
                index_states_b[index].add(predecessor)
                queue.append((index, predecessor))

    # Label states by shortest prefix that reaches them.
    queue = deque()
    initial_state = product_automaton.get_start()
    if initial_state not in product_automaton:
        initial_state = min([llstr(x) for x in product_automaton])

    queue.append(initial_state)
    state_labeling = {initial_state: llstr("")}
    while queue:
        state = queue.popleft()
        state_label = state_labeling[state]
        for symbol, target in product_automaton[state][1].items():
            new_label = state_label + symbol
            if target not in state_labeling or new_label < state_labeling[target]:
                queue.append(target)
                state_labeling[target] = new_label

    # Return state representing the smallest prefix from which an accepting run
    # starts on the smallest cycle.
    min_a = (
        min([state_labeling[x] for x in index_states_a[0]])
        if index_states_a[0]
        else None
    )
    min_b = (
        min([state_labeling[x] for x in index_states_b[0]])
        if index_states_b[0]
        else None
    )

    if min_a is not None and (min_b is None or min_a < min_b):
        return False, OmegastrLoop(min_a, loop, reduce=False), False
    else:
        return False, OmegastrLoop(min_b, loop, reduce=False), True
