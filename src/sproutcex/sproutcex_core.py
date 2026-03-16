r"""
The core implementation of **SproutCEX** from *Learning Deterministic
:math:`\omega`-Automata from Smallest Counterexamples* by Jan Lohse.
"""

import time
from dataclasses import dataclass
from typing import Iterator, Literal

from IPython.core.display_functions import display

from .graph_functions import Automaton
from .omega_language_modelling import Omegastr
from .smallest_cex import (
    smallest_cex,
    smallest_cex_expansion,
    smallest_cex_lex,
    smallest_cex_loop,
    smallest_cex_prefix,
)
from .sprout_dba import sprout_dba
from .sprout_dba_optimized import sprout_dba_optim
from .sprout_wdba import sprout_wdba
from .sprout_wdba_optimized import sprout_wdba_optim

CONS_METHODS = {
    "dba": sprout_dba_optim,
    "dba_optim": sprout_dba_optim,
    "dba_legacy": sprout_dba,
    "wdba": sprout_wdba_optim,
    "wdba_optim": sprout_wdba_optim,
    "wdba_legacy": sprout_wdba,
}

ORDERINGS = {
    "default": smallest_cex,
    "loop": smallest_cex_loop,
    "prefix": smallest_cex_prefix,
    "lex": smallest_cex_lex,
    "expansion": smallest_cex_expansion,
}

ConsMethod = Literal[
    "dba", "dba_optim", "dba_legacy", "wdba", "wdba_optim", "wdba_legacy"
]
Ordering = Literal["default", "loop", "prefix", "lex", "expansion"]


EventType = Literal["iteration", "failed", "aborted", "finished"]


@dataclass
class SproutcexEvent:
    event_type: EventType
    query_count: int
    search_time: float
    build_time: float
    automaton: Automaton | None = None
    cex: Omegastr | None = None
    cex_result: bool | None = False


def sproutcex_iterator(
    target: Automaton,
    cons_method: ConsMethod = "dba",
    ordering: Ordering = "default",
    max_steps: None | int = None,
    square_threshold: bool = False,
) -> Iterator[SproutcexEvent]:
    r"""
    Should not be used directly. Use `sproutcex` under regular circumstances. This
    implements the algorithm and yields the automaton in every step. A wrapper is needed
    for a usable output.

    Attempts to learn an :math:`\omega`-automaton from smallest counterexamples.
    Implements **SproutCEX** from *Learning :math:`\omega`-Automata from Smallest
    Counterexamples* by Jan Lohse.

    Args:
        target: Automaton that is to be learned.
        cons_method: The passive learner to use for constructing automata.
        ordering: How to order the automaton.
        max_steps: How many steps before aborting.
        square_threshold: Whether to use the original square threshold for Sprout.

    Returns:
        Yields automata as they are computed.
    """

    sprout_method = CONS_METHODS[cons_method]
    cex_method = ORDERINGS[ordering]

    plus = set()
    minus = set()
    found = False
    query_count = 0
    build_time = 0.0
    search_time = 0.0

    while not found:
        if max_steps is not None:
            if not max_steps:
                yield SproutcexEvent("aborted", query_count, search_time, build_time)
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
                yield SproutcexEvent("failed", query_count, search_time, build_time)
                return None
            plus.add(cex)
        else:
            if cex in minus:
                yield SproutcexEvent("failed", query_count, search_time, build_time)
                return None
            minus.add(cex)

        if not found and cex is not None:
            yield SproutcexEvent(
                "iteration",
                query_count,
                search_time,
                build_time,
                automaton,
                cex,
                cex_result,
            )

    yield SproutcexEvent("finished", query_count, search_time, build_time, automaton)


def sproutcex(
    target: Automaton,
    cons_method: ConsMethod = "dba",
    ordering: Ordering = "default",
    verbose: bool = False,
    max_steps: None | int = None,
    square_threshold: bool = False,
    typst_output: bool = False,
) -> None | Automaton:
    r"""
    Attempts to learn an :math:`\omega`-automaton from smallest counterexamples.
    Implements **SproutCEX** from *Learning :math:`\omega`-Automata from Smallest
    Counterexamples* by Jan Lohse. Actual algorithm is found in `sproutcex_iterator`.

    Args:
        target: Automaton that is to be learned.
        cons_method: The passive learner to use for constructing automata.
        ordering: How to order the automaton.
        verbose: Whether to display every the automaton in every step.
        max_steps: How many steps before aborting.
        square_threshold: Whether to use the original square threshold for Sprout.
        typst_output: Print for use with diagraph typst package instead of displaying.

    Returns:
        An automaton equivalent to the target, if one is found.
        It will also display the automaton and print the number of equivalence queries
        performed, including the final one with a positive result. As a guide for the
        efficiency of **SproutCEX** a proportional reference is printed. This is
        computed as :math:`|\Sigma| \cdot |Q|^2`, which is the maximum number of
        examples sufficient for identifying all edges of an automaton.
    """
    sproutcex_iter = sproutcex_iterator(
        target, cons_method, ordering, max_steps, square_threshold
    )

    for event in sproutcex_iter:
        match event.event_type:
            case "iteration":
                if verbose:
                    if typst_output:
                        print(event.automaton.to_diagraph())
                    else:
                        display(event.automaton)
                    print(
                        f"Received the "
                        f"{'positive' if event.cex_result else 'negative'} "
                        f"counterexample "
                        f"{event.cex.to_typst() if typst_output else event.cex}."
                    )
            case "aborted":
                print(
                    f"Aborted after {event.query_count} "
                    f"quer{'y' if event.query_count == 1 else 'ies'}. "
                    f"sprout_time={event.build_time:.2f}s cex_{event.search_time=:.2f}s"
                )
            case "failed":
                print(
                    f"Failed after {event.query_count} "
                    f"quer{'y' if event.query_count == 1 else 'ies'}. "
                    f"sprout_time={event.build_time:.2f}s cex_{event.search_time=:.2f}s"
                )
            case "finished":
                if typst_output:
                    print(event.automaton.to_diagraph())
                else:
                    display(event.automaton)
                print(
                    f"Found after {event.query_count} "
                    f"quer{'y' if event.query_count == 1 else 'ies'}! "
                    f"The proportional reference is "
                    f"{len(event.automaton) ** 2 * len(target.get_alphabet())} "
                    f"queries. sprout_time={event.build_time:.2f}s "
                    f"cex_{event.search_time=:.2f}s"
                )
                return event.automaton
