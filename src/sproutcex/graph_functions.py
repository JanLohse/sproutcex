"""
Helper classes and functions for handling and displaying graphs and automata.
"""

import random
from collections import defaultdict
from typing import Any

import graphviz
from graphviz import Digraph

from .omega_language_modelling import llstr
from .utils import FastRandomBag


class Graph(dict[str, Any]):
    """Wraps a `dict` to represent a directed deterministic graph."""

    def __init__(
        self,
        struct: None | dict[str, dict[str, str]] = None,
        start_node: None | str = None,
    ):
        """Initialize the graph."""
        super().__init__(struct or {})

        self._start_node = start_node

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Returns a rich display for Jupyter/IPython."""
        if self:
            dot = draw_graph(self)
            return dot._repr_mimebundle_(include=include, exclude=exclude)
        else:
            return dict(self)

    def get_start(self):
        r"""Returns the start node :math:`q_0`. Defaults to smallest node label."""
        if self._start_node is None and self:
            self._start_node = min(self)
        return self._start_node

    def get_alphabet(self) -> str:
        r"""Returns the input alphabet :math:`\Sigma` of the graph."""
        alphabet = "".join(
            sorted({sym for (_, trans) in self.values() for sym in trans.keys()})
        )
        return alphabet

    def to_typst(self):
        """
        Convert graph to a string that can be pasted into the diagraph typst
        package.
        """
        return graph_to_typst(self)


def draw_graph(graph: Graph, for_typst=False) -> Digraph:
    """Converts a graph to a `graphviz.Digraph` for rich display."""
    dot = Digraph()
    if for_typst:

        def format_state(state_name):
            if state_name:
                return " ".join(state_name)
            else:
                return "epsilon"
    else:
        dot.attr(size="11,11")

        def format_state(state_name):
            if state_name:
                return state_name
            else:
                return "ε"

    dot.attr(rankdir="LR", fontname="Helvetica")
    dot.attr("node", height="0.25", width="0.25", shape="plaintext", fontsize="11")
    dot.attr("edge", arrowhead="vee", arrowsize="0.66", fontsize="11")

    # Draw states.
    for state in graph:
        sid = format_state(state)
        dot.node(sid)

    # Start arrow.
    start = graph.get_start()
    dot.node("", shape="none", height="0", width="0")
    dot.edge("", format_state(start))

    # Merge edges.
    merged = defaultdict(list)
    for state, transitions in graph.items():
        for symbol, target in transitions.items():
            src = format_state(state)
            dst = format_state(target)
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


def graph_to_typst(graph: Graph) -> str:
    """
    Convert graph to a string that can be pasted into the diagraph typst package.
    """
    dot = draw_graph(graph, True)
    dot_str = str(dot)
    dot_str = f"""#raw-render(```
{dot_str.strip()}
```)
"""
    return dot_str


class Automaton(Graph):
    """Wraps a `dict` to represent an automaton with state based acceptance."""

    def __init__(
        self,
        struct: None | dict[str, tuple[bool, dict[str, str]]] = None,
        start_node=None,
    ):
        super().__init__(struct, start_node)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Returns a rich display for Jupyter/IPython."""
        if self:
            dot = draw_automaton(self)
            return dot._repr_mimebundle_(include=include, exclude=exclude)
        else:
            return dict(self)

    def to_typst(self):
        """
        Convert automaton to a string that can be pasted into the diagraph typst
        package.
        """
        return automaton_to_typst(self)


def draw_automaton(automaton: Automaton, for_typst=False) -> graphviz.Digraph:
    """Converts an automaton to a `graphviz.Digraph` for rich display."""
    dot = Digraph()
    if for_typst:

        def format_state(state_name):
            if state_name:
                return " ".join(state_name)
            else:
                return "epsilon"
    else:
        dot.attr(size="11,11")

        def format_state(state_name):
            if state_name:
                return state_name
            else:
                return "ε"

    if max([len(x) for x in automaton]) > 4:
        shape = "box"
        fixedsize = "false"
    else:
        shape = "circle"
        fixedsize = "true"
    dot.attr(rankdir="LR")
    dot.attr(
        "node",
        shape=shape,
        height="0.35",
        width="0.35",
        style="rounded",
        fixedsize=fixedsize,
        fontsize="11",
    )
    dot.attr("edge", arrowhead="vee", arrowsize="0.66", fontsize="11")

    # Draw states.
    for state, (is_final, transitions) in automaton.items():
        sid = format_state(state)
        dot.node(sid, peripheries="2" if is_final else "1")

    # Start arrow.
    start = automaton.get_start()
    dot.node("", shape="none", height="0", width="0")
    dot.edge("", format_state(start))

    # Merge edges.
    merged = defaultdict(list)
    for state, (_, transitions) in automaton.items():
        for symbol, target in transitions.items():
            src = format_state(state)
            dst = format_state(target)
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


def automaton_to_typst(automaton: Automaton) -> str:
    """
    Convert automaton to a string that can be pasted into the diagraph typst package.
    """
    dot = draw_automaton(automaton, True)
    dot_str = str(dot)
    dot_str = f"""#raw-render(```
{dot_str.strip()}
```)
"""
    return dot_str


def generate_wdba(max_states: int, symbols="ab", prob_acc=0.5, seed=None) -> Automaton:
    r"""
    Generates a random complete weak deterministic Büchi automaton.

    Args:
        max_states: The maximum number of states allowed.
        symbols: The input alphabet :math:`\Sigma`.
        prob_acc: Probability for each state to be accepting.
        seed: Seed for randomness.

    Returns:
        A weak deterministic Büchi automaton.
    """
    if seed is not None:
        random.seed(seed)
    state_count = 1
    accepting_states = set()
    rejecting_states = set()

    # add initial state
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
        # Select random escaping edge to add.
        state, symbol = next(stack)

        if random.random() < (max_states - state_count) / max_states:
            # Add a new state as the target.
            new_state = state + symbol
            accepting = random.random() < prob_acc
            if accepting:
                accepting_states.add(new_state)
            else:
                rejecting_states.add(new_state)
            automaton[new_state] = [accepting, {}]

            # Update predecessor and successor sets for weakness check.
            for sym in symbols:
                stack.add((new_state, sym))
            automaton[state][1][symbol] = new_state
            successors[new_state] = {new_state}
            predecessors[new_state] = predecessors[state] | {new_state}
            for predecessor in predecessors[new_state]:
                successors[predecessor].add(new_state)
            state_count += 1

        else:
            # Select a random state as the target of the edge.
            targets = FastRandomBag(automaton.keys())
            found = False

            # Search for target state that preserves weakness.
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

            # Update predecessor and successor sets for weakness check.
            new_successors = successors[target]
            for new_predecessor in predecessors[state]:
                successors[new_predecessor].update(new_successors)

            new_predecessors = predecessors[state]
            for new_successor in successors[target]:
                predecessors[new_successor].update(new_predecessors)

    return automaton
