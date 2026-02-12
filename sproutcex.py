import time

from graph_functions import get_alphabet
from smallest_cex import smallest_cex, smallest_cex_expansion, smallest_cex_loop, smallest_cex_prefix, smallest_cex_lex
from sprout_dba import sprout_dba
from sprout_dba_optimized import sprout_dba_optim
from sprout_wdba import sprout_wdba
from sprout_wdba_optimized import sprout_wdba_optim

SPROUT_METHOD_MAPPING = {"dba": sprout_dba_optim, "dba_optim": sprout_dba_optim, "dba_legacy": sprout_dba,
                         "wdba": sprout_wdba_optim, "wdba_optim": sprout_wdba_optim, "wdba_legacy": sprout_wdba, }


def sproutcex(target, method="wdba", ordering="total", verbose=False, silent=False, steps=None, square_threshold=False):
    if method is None or not method:
        method = "wdba"
    assert method in SPROUT_METHOD_MAPPING.keys(), f"method must be one of 'weak', 'büchi', 'optim', got {method!r}"
    sprout_method = SPROUT_METHOD_MAPPING[method]

    if ordering is None or not ordering:
        ordering = "total"
    assert ordering in {"total", "loop", "prefix", "lex", "expansion"}
    cex_method = \
        {"total": smallest_cex, "loop": smallest_cex_loop, "prefix": smallest_cex_prefix, "lex": smallest_cex_lex,
         "expansion": smallest_cex_expansion}[ordering]

    alphabet = get_alphabet(target)
    plus = set()
    minus = set()
    found = False
    count = 0
    build_time = 0.
    search_time = 0.

    while not found:
        if steps is not None:
            if not steps:
                print(
                    f"Aborted after {count} quer{'y' if count == 1 else 'ies'}. sprout_time={build_time:.2f}s cex_{search_time=:.2f}s")
                return None
            steps -= 1
        start = time.time()
        query_dict = sprout_method(plus, minus, square_threshold=square_threshold)
        build_time += time.time() - start
        start = time.time()
        found, cex, cex_result = cex_method(query_dict, target)
        search_time += time.time() - start
        count += 1

        if found:
            continue

        cex.simplify()

        if cex_result:
            if cex in plus:
                print(
                    f"Failed after {count} quer{'y' if count == 1 else 'ies'}. sprout_time={build_time:.2f}s cex_{search_time=:.2f}s")
                return None
            plus.add(cex)
        else:
            if cex in minus:
                print(
                    f"Failed after {count} quer{'y' if count == 1 else 'ies'}. sprout_time={build_time:.2f}s cex_{search_time=:.2f}s")
                return None
            minus.add(cex)

        if verbose and not found and cex is not None:
            display(query_dict)
            print(f"Received the {'positive' if cex_result else 'negative'} counterexample {cex}.")

    if not silent:
        display(query_dict)
        print(
            f"Found after {count} quer{'y' if count == 1 else 'ies'}! The proportional baseline is {len(query_dict) ** 2 * len(alphabet)} queries (with sink state {(len(query_dict) + 1) ** 2 * len(alphabet)}). sprout_time={build_time:.2f}s cex_{search_time=:.2f}s")

    return query_dict
