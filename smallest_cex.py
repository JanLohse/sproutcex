from collections import deque, defaultdict
from itertools import product

from graph_functions import get_alphabet
from omega_language_modelling import omegaiter, omegaiter_prefix, omegaiter_lex, omegaiter_expansion, llstr, \
    omegastr_loop
from sprout_dba import is_accepting


def product_of_dba(A, B):
    product = {}
    stack = [(next(iter(A)), next(iter(B)))]
    visited = set()

    while stack:
        sA, sB = stack.pop()
        key = sA + '|' + sB
        if key in visited:
            continue
        visited.add(key)
        accA, transA = A.get(sA, (False, {}))
        accB, transB = B.get(sB, (False, {}))
        product[key] = [(accA, accB), {}]
        for sym in {*transA.keys(), *transB.keys()}:
            nextA = transA.get(sym, "sink")
            nextB = transB.get(sym, "sink")
            next_key = nextA + '|' + nextB
            product[key][1][sym] = next_key
            if next_key not in visited:
                stack.append((nextA, nextB))
    return product


def find_asymmetric_sccs_rabin(automaton):
    # partition states
    state_partition = {i: {state for state in automaton if not automaton[state][0][1 - i]} for i in (0, 1)}
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

            # Explore all successors
            for _, next_state in automaton[state][1].items():
                if next_state not in state_partition[idx]:
                    continue
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
                if (len(scc) > 1 or next(iter(scc)) in automaton[next(iter(scc))][1].values()) and any(
                        [automaton[s][0][idx] for s in scc]):
                    sccs.append(scc)

        for state in state_partition[idx]:
            if state not in indices and automaton[state][0][idx]:
                strongconnect(state)

    return sccs


def are_equivalent(A, B):
    automaton = product_of_dba(A, B)
    sccs = find_asymmetric_sccs_rabin(automaton)

    return not bool(sccs)


def smallest_cex(A, B, alphabet=None, iter=omegaiter):
    if are_equivalent(A, B):
        return True, None, None

    if alphabet is None:
        alphabet_A = get_alphabet(A)
        alphabet_B = get_alphabet(B)
        alphabet = "".join(sorted(set(alphabet_A + alphabet_B)))

    A_init = min(A)
    B_init = min(B)

    for word in iter(alphabet):
        A_result = is_accepting(A, word, A_init)
        B_result = is_accepting(B, word, B_init)
        if B_result and not A_result:
            return False, word, True
        if A_result and not B_result:
            return False, word, False


def smallest_cex_prefix(A, B, alphabet=None):
    return smallest_cex(A, B, alphabet=alphabet, iter=omegaiter_prefix)


def smallest_cex_lex(A, B, alphabet=None):
    return smallest_cex(A, B, alphabet=alphabet, iter=omegaiter_lex)


def smallest_cex_expansion(A, B, alphabet=None):
    return smallest_cex(A, B, alphabet=alphabet, iter=omegaiter_expansion)


def smallest_diff_loop_rabin(automaton, alphabet=None):
    if alphabet is None:
        alphabet = get_alphabet(automaton)

    # compute accepting SCCs and all states in them
    sccs = find_asymmetric_sccs_rabin(automaton)

    scc_states = set().union(*sccs)
    if not scc_states:
        return None, None, None

    # map each state to its SCC
    sccs = {i: scc for i, scc in enumerate(sccs)}
    scc_id = {}
    for i, scc in sccs.items():
        for state in scc:
            if state in scc_id and scc_id[state] != i:
                for old_state in sccs[scc_id[state]]:
                    scc_id[old_state] = i
            else:
                scc_id[state] = i

    # init dp and word_list
    dp = defaultdict(dict)
    for state in scc_states:
        if not all(automaton[state][0]):
            dp[state][None] = (state, automaton[state][0])
    dp[None] = defaultdict(dict)
    for a in alphabet:
        for state in scc_states:
            target = automaton[state][1].get(a)
            marking = tuple(a or b for a, b in zip(automaton[state][0], automaton[target][0]))
            if target in scc_states and scc_id[target] == scc_id[state] and not all(marking):
                dp[state][a] = (target, marking)
    words_by_length = {}
    k = 0

    while True:
        # compute all lyndon words of length k
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

            # only check if any accepting transitions
            if not accepting_transition:
                continue

            # search for cycles within SCCs
            states_left = set(scc_states)
            loop_states_A = set()
            loop_states_B = set()
            while states_left:
                current_state = next(iter(states_left))
                path = set()
                while current_state is not None and current_state in states_left and current_state not in path:
                    path.add(current_state)
                    states_left.remove(current_state)
                    current_state = dp[current_state].get(current_word)
                    if current_state is None:
                        break
                    else:
                        current_state = current_state[0]
                if current_state in path:
                    loop_path = set()
                    found_A, found_B = False, False
                    while current_state not in loop_path and not (found_A and found_B):
                        loop_path.add(current_state)
                        found_A = found_A or dp[current_state][current_word][1][0]
                        found_B = found_B or dp[current_state][current_word][1][1]
                        current_state = dp[current_state].get(current_word)[0]
                    if found_A and not found_B:
                        loop_states_A.update(loop_path)
                    elif not found_A and found_B:
                        loop_states_B.update(loop_path)

                states_left -= path

            if loop_states_A or loop_states_B:
                return current_word, loop_states_A, loop_states_B


def smallest_cex_loop(A, B, alphabet=None, initial_state="|"):
    if alphabet is None:
        alphabet_A = get_alphabet(A)
        alphabet_B = get_alphabet(B)
        alphabet = "".join(sorted(set(alphabet_A + alphabet_B)))

    automaton = product_of_dba(A, B)

    # find smallest loop and states from which the loop start
    loop, start_points_A, start_points_B = smallest_diff_loop_rabin(automaton)
    if loop is None:
        return True, None, None

    # build mapping from indices of the loop to states from which an accepting run with the loop
    # starts in this state from the index
    m = len(loop)
    index_states_A = {i: set() for i in range(m)}
    index_states_A[0] = start_points_A
    index_states_B = {i: set() for i in range(m)}
    index_states_B[0] = start_points_B

    # invert edges to compute predecessors
    reverse_mapping = {state: {a: set() for a in alphabet} for state in automaton}
    for state in automaton:
        for a in alphabet:
            target = automaton[state][1][a]
            reverse_mapping[target][a].add(state)

    # trace back from the loop starting states to complete the mapping from indices
    queue = deque()
    for state in start_points_A:
        queue.append((0, state))
    while queue:
        index, state = queue.popleft()
        index = (index - 1) % m
        input = loop[index]
        for predecessor in reverse_mapping[state][input]:
            if predecessor not in index_states_A[index]:
                index_states_A[index].add(predecessor)
                queue.append((index, predecessor))

    queue = deque()
    for state in start_points_B:
        queue.append((0, state))
    while queue:
        index, state = queue.popleft()
        index = (index - 1) % m
        input = loop[index]
        for predecessor in reverse_mapping[state][input]:
            if predecessor not in index_states_B[index]:
                index_states_B[index].add(predecessor)
                queue.append((index, predecessor))

    # label states by shortest prefix that reaches them
    queue = deque()
    initial_state = llstr(initial_state)
    if initial_state not in automaton:
        initial_state = min([llstr(x) for x in automaton])

    queue.append(initial_state)
    state_labeling = {initial_state: llstr("")}
    while queue:
        state = queue.popleft()
        state_label = state_labeling[state]
        for input, target in automaton[state][1].items():
            new_label = state_label + input
            if target not in state_labeling or new_label < state_labeling[target]:
                queue.append(target)
                state_labeling[target] = new_label

    # return state representing smallest prefix from which an accepting run
    # starts on the smallest cycle
    min_A = min([state_labeling[x] for x in index_states_A[0]]) if index_states_A[0] else None
    min_B = min([state_labeling[x] for x in index_states_B[0]]) if index_states_B[0] else None

    if min_A is not None and (min_B is None or min_A < min_B):
        return False, omegastr_loop(min_A, loop, simplify=False), False
    else:
        return False, omegastr_loop(min_B, loop, simplify=False), True
