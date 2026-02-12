import random
from collections import defaultdict
from typing import Optional, Any

from graphviz import Digraph

from omega_language_modelling import llstr
from utils import FastRandomBag


class Graph(dict[str, Any]):
    def __init__(
        self, struct: Optional[dict[str, dict[str, str]]] = None, start_node=None
    ):
        super().__init__(struct or {})

        self._start_node = start_node

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self:
            dot = draw_graph(self)
            return dot._repr_mimebundle_(include=include, exclude=exclude)
        else:
            return dict(self)

    def get_start(self):
        if self._start_node is None and self:
            self._start_node = min(self)
        return self._start_node

    def get_alphabet(self) -> str:
        alphabet = "".join(
            sorted({sym for (_, trans) in self.values() for sym in trans.keys()})
        )
        return alphabet


def draw_graph(graph: Graph):
    dot = Digraph()
    dot.attr(rankdir="LR", fontname="Helvetica", fontsize="14")
    dot.attr(
        "node",
        fontname="Helvetica",
        fontsize="14",
        height="0.25",
        width="0.25",
        shape="plaintext",
    )
    dot.attr(
        "edge", fontname="Helvetica", fontsize="14", arrowhead="vee", arrowsize="0.66"
    )

    # Draw states
    for state in graph:
        sid = state if state else "ε"
        dot.node(sid)

    # Start arrow
    start = graph.get_start()
    dot.node("", shape="none", height="0", width="0")
    dot.edge("", "ε" if start == "" else start)

    # Merge edges
    merged = defaultdict(list)
    for state, transitions in graph.items():
        for symbol, target in transitions.items():
            src = state if state else "ε"
            dst = target if target else "ε"
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


class Automaton(Graph):
    def __init__(
        self,
        struct: Optional[dict[str, tuple[bool, dict[str, str]]]] = None,
        start_node=None,
    ):
        super().__init__(struct, start_node)

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self:
            dot = draw_automaton(self)
            return dot._repr_mimebundle_(include=include, exclude=exclude)
        else:
            return dict(self)


def draw_automaton(automaton: Automaton):
    dot = Digraph()
    dot.attr(size="11,11")
    if max([len(x) for x in automaton]) > 4:
        default_shape = "box"
        small_shape = "box"
    else:
        default_shape = "ellipse"
        small_shape = "circle"
    dot.attr(rankdir="LR", fontname="Helvetica", fontsize="14")
    dot.attr(
        "node",
        fontname="Helvetica",
        fontsize="14",  # fillcolor="#fefeb2",
        shape=default_shape,
        height="0.35",
        width="0.35",
        style="rounded",
    )  # style="rounded, filled"
    dot.attr(
        "edge", fontname="Helvetica", fontsize="14", arrowhead="vee", arrowsize="0.66"
    )

    # Draw states
    for state, (is_final, transitions) in automaton.items():
        sid = state if state else "ε"
        dot.node(
            sid,
            peripheries="2" if is_final else "1",
            shape=small_shape if len(state) <= 1 else default_shape,
        )

    # Start arrow
    start = automaton.get_start()
    dot.node("", shape="none", height="0", width="0")
    dot.edge("", "ε" if start == "" else start)

    # Merge edges
    merged = defaultdict(list)
    for state, (_, transitions) in automaton.items():
        for symbol, target in transitions.items():
            src = state if state else "ε"
            dst = target if target else "ε"
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


def generate_wdba(max_states: int, symbols="ab", prob_acc=0.5, seed=None) -> Automaton:
    if seed is not None:
        random.seed(seed)
    state_count = 1
    accepting_states = set()
    rejecting_states = set()
    initial_state = llstr("")
    initial_accepting = random.random() < prob_acc
    automaton = Automaton(
        {initial_state: (initial_accepting, {})},
    )
    if initial_accepting:
        accepting_states.add(initial_state)
    else:
        rejecting_states.add(initial_state)
    predecessors = {initial_state: {initial_state}}
    successors = {initial_state: {initial_state}}
    stack = FastRandomBag([(initial_state, symbol) for symbol in symbols])
    while stack:
        state, symbol = next(stack)
        new_state = state + symbol
        if random.random() < (max_states - state_count) / max_states:
            accepting = random.random() < prob_acc
            if accepting:
                accepting_states.add(new_state)
            else:
                rejecting_states.add(new_state)
            automaton[new_state] = [accepting, {}]
            for sym in symbols:
                stack.add((new_state, sym))
            automaton[state][1][symbol] = new_state
            successors[new_state] = {new_state}
            predecessors[new_state] = predecessors[state] | {new_state}
            for predecessor in predecessors[new_state]:
                successors[predecessor].add(new_state)
            state_count += 1
        else:
            targets = FastRandomBag(automaton.keys())
            found = False
            while not found:
                target = next(targets)
                found = True
                if target in successors[state]:
                    break
                elif state not in successors[target]:
                    break
                else:
                    common = successors[target] & predecessors[state]
                    if common & accepting_states and common & rejecting_states:
                        found = False
                        continue

            automaton[state][1][symbol] = target

            new_successors = successors[target]
            for new_predecessor in predecessors[state]:
                successors[new_predecessor].update(new_successors)

            new_predecessors = predecessors[state]
            for new_successor in successors[target]:
                predecessors[new_successor].update(new_predecessors)

    return automaton
