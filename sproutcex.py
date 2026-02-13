import time
from typing import Literal
from typing import Optional

from IPython.core.display_functions import display

from graph_functions import Automaton
from smallest_cex import (
    smallest_cex,
    smallest_cex_expansion,
    smallest_cex_loop,
    smallest_cex_prefix,
    smallest_cex_lex,
)
from sprout_dba import sprout_dba
from sprout_dba_optimized import sprout_dba_optim
from sprout_wdba import sprout_wdba
from sprout_wdba_optimized import sprout_wdba_optim

CONS_METHODS = {
    "dba": sprout_dba_optim,
    "dba_optim": sprout_dba_optim,
    "dba_legacy": sprout_dba,
    "wdba": sprout_wdba_optim,
    "wdba_optim": sprout_wdba_optim,
    "wdba_legacy": sprout_wdba,
}

ORDERINGS = {
    "total": smallest_cex,
    "loop": smallest_cex_loop,
    "prefix": smallest_cex_prefix,
    "lex": smallest_cex_lex,
    "expansion": smallest_cex_expansion,
}

ConsMethod = Literal[
    "dba", "dba_optim", "dba_legacy", "wdba", "wdba_optim", "wdba_legacy"
]
Ordering = Literal["total", "loop", "prefix", "lex", "expansion"]


def sproutcex(
    target: Automaton,
    cons_method: ConsMethod = "dba",
    ordering: Ordering = "total",
    verbose: bool = False,
    max_steps: Optional[int] = None,
    square_threshold: bool = False,
) -> Optional[Automaton]:
    """
    Attempts to learn an automaton from smallest counterexamples.

    Args:
        target: Automaton that is to be learned.
        cons_method: The passive learner to use for constructing automata.
        ordering: How to order the automaton.
        verbose: Whether to display every the automaton in every step.
        max_steps: How many steps before aborting.
        square_threshold: Whether to use the original square threshold for Sprout.

    Returns:
        An automaton equivalent to the target, if one is found.
    """
    sprout_method = CONS_METHODS[cons_method]
    cex_method = ORDERINGS[ordering]

    alphabet = target.get_alphabet()
    plus = set()
    minus = set()
    found = False
    query_count = 0
    build_time = 0.0
    search_time = 0.0

    while not found:
        if max_steps is not None:
            if not max_steps:
                print(
                    f"Aborted after {query_count} quer{'y' if query_count == 1 else 'ies'}. "
                    f"sprout_time={build_time:.2f}s cex_{search_time=:.2f}s"
                )
                return None
            max_steps -= 1
        start = time.time()
        automaton = sprout_method(plus, minus, square_threshold=square_threshold)
        build_time += time.time() - start
        start = time.time()
        found, cex, cex_result = cex_method(automaton, target)
        search_time += time.time() - start
        query_count += 1

        if found:
            continue

        cex.reduce()

        if cex_result:
            if cex in plus:
                print(
                    f"Failed after {query_count} quer{'y' if query_count == 1 else 'ies'}. "
                    f"sprout_time={build_time:.2f}s cex_{search_time=:.2f}s"
                )
                return None
            plus.add(cex)
        else:
            if cex in minus:
                print(
                    f"Failed after {query_count} quer{'y' if query_count == 1 else 'ies'}. "
                    f"sprout_time={build_time:.2f}s cex_{search_time=:.2f}s"
                )
                return None
            minus.add(cex)

        if verbose and not found and cex is not None:
            display(automaton)
            print(
                f"Received the {'positive' if cex_result else 'negative'} counterexample {cex}."
            )

    display(automaton)
    print(
        f"Found after {query_count} quer{'y' if query_count == 1 else 'ies'}! "
        f"The proportional reference is {len(automaton) ** 2 * len(alphabet)} queries. "
        f"sprout_time={build_time:.2f}s cex_{search_time=:.2f}s"
    )

    return automaton
