"""
TODO
"""

import pickle
import random
import string
from typing import Optional

from .graph_functions import Automaton, generate_wdba
from .sproutcex import ConsMethod, Ordering, CONS_METHODS, ORDERINGS

FULL_ALPHABET = string.ascii_lowercase


def reciprocal_distribution(a: int, b: int) -> int:
    r"""
    Returns a value sampled from a discrete reciprocal distribution. This distribution
    is linear in log space and gives a higher likeliness to lower values. Values are
    computed as $(a - 0.5)^{1 - x} \cdot (b + 0.5)^x$, where $x$ is uniformly
    distributed over $[0, 1)$, and the output is rounded.

    Args:
        a: The lower bound $a$ of the distribution.
        b: The upper bound $b$ of the distribution (is included as possible output).

    Returns:
        A discrete value sampled from the reciprocal distribution over $[a, b]$.
    """
    x = random.random()
    y = (a - 0.5) ** (1 - x) * (b + 0.5) ** x
    return round(y)


def sproutcex_silent(
    target: Automaton,
    cons_method: ConsMethod = "dba",
    ordering: Ordering = "total",
    max_steps: Optional[int] = None,
    square_threshold: bool = False,
) -> tuple[Optional[Automaton], int] | tuple[None, None]:
    r"""
    Attempts to learn an $\omega$-automaton from smallest counterexamples. Implements
    **SproutCEX** from *Learning $\omega$-Automata from Smallest Counterexamples* by Jan
    Lohse. This version reduces the output and is made to be used during batch testing.

    Args:
        target: Automaton that is to be learned.
        cons_method: The passive learner to use for constructing automata.
        ordering: How to order the automaton.
        max_steps: How many steps before aborting.
        square_threshold: Whether to use the original square threshold for Sprout.

    Returns:
        An automaton equivalent to the target, if one is found, and the number of
        queries needed to find it.
    """
    sprout_method = CONS_METHODS[cons_method]
    cex_method = ORDERINGS[ordering]

    plus = set()
    minus = set()
    found = False
    query_count = 0

    while not found:
        if max_steps is not None:
            if not max_steps:
                return None, None
            max_steps -= 1
        automaton = sprout_method(plus, minus, square_threshold=square_threshold)
        found, cex, cex_result = cex_method(automaton, target)
        query_count += 1

        if found:
            continue

        cex.reduce()

        if cex_result:
            if cex in plus:
                return None, None
            plus.add(cex)
        else:
            if cex in minus:
                return None, None
            minus.add(cex)

    return automaton, query_count


def generate_automaton(
    alphabet_low=2, alphabet_high=4, state_low=4, state_high=30
) -> Automaton:
    """
    Generates a random wDBA with its size chosen according to a reciprocal
    distribution.

    Args:
        alphabet_low: Minimum length of the alphabet.
        alphabet_high: Maximum length of the alphabet.
        state_low: Minimum upper state bound of the automaton.
        state_high: Maximum state count of the automaton.

    Returns:
        A random weak deterministic Büchi automaton.
    """
    alphabet_size = reciprocal_distribution(alphabet_low, alphabet_high)
    state_count = reciprocal_distribution(
        state_low,
        round(state_high * alphabet_low / alphabet_size),
    )
    automaton = generate_wdba(state_count, FULL_ALPHABET[:alphabet_size])

    return automaton


def generate_automata(
    alphabet_low=2,
    alphabet_high=4,
    state_low=4,
    state_high=30,
    automata_count=100,
    seed=None,
) -> list[Automaton]:
    """
    Generates a set of wDBA with each's size chosen according to a reciprocal
    distribution.

    Args:
        alphabet_low: Minimum length of each automaton's alphabet.
        alphabet_high: Maximum length of each automaton's alphabet.
        state_low: Minimum upper state bound for each automaton.
        state_high: Maximum state count of each automaton.
        automata_count: Number of automata in the set.
        seed: Seed set before generating automata.

    Returns:
        A list of weak deterministic Büchi automata.
    """
    if seed is not None:
        random.seed(seed)

    automata = []

    for _ in range(automata_count):
        automaton = generate_automaton(
            alphabet_low, alphabet_high, state_low, state_high
        )
        automata.append(automaton)

    return automata


def load_automata(
    seed,
    automata_count=100,
    alphabet_low=2,
    alphabet_high=4,
    state_low=4,
    state_high=30,
    path="",
) -> list[Automaton]:
    """
    Loads a set of wDBA with each's size chosen according to a reciprocal
    distribution. If such a set is already stored in the path folder it is loaded,
    and if not a new set is generated at stored in the path folder.

    Args:
        seed: Seed set before generating automata.
        automata_count: Number of automata in the set.
        alphabet_low: Minimum length of each automaton's alphabet.
        alphabet_high: Maximum length of each automaton's alphabet.
        state_low: Minimum upper state bound for each automaton.
        state_high: Maximum state count of each automaton.
        path: Where to look for and store the backup of the automaton set.

    Returns:
        A list of weak deterministic Büchi automata.
    """
    filename = (
        f"automata_{alphabet_low}_{alphabet_high}_{state_low}_{state_high}_"
        f"{automata_count}_{seed}.pkl"
    )
    filepath = path / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            automata = pickle.load(f)
    else:
        automata = generate_automata(
            alphabet_low, alphabet_high, state_low, state_high, automata_count, seed
        )
        with open(filepath, "wb") as f:
            pickle.dump(automata, f)

    return automata
