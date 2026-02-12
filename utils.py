import random
from collections import defaultdict

from graphviz import Digraph


class FastRandomBag:
    def __init__(self, items=None):
        self.data = list(items) if items else []

    def add(self, item):
        """Add an item to the bag."""
        self.data.append(item)

    def pop_random(self):
        """Remove and return a random item in O(1) time."""
        if not self.data:
            raise StopIteration("FastRandomBag is empty")
        i = random.randrange(len(self.data))
        # swap and pop for O(1)
        self.data[i], self.data[-1] = self.data[-1], self.data[i]
        return self.data.pop()

    def __len__(self):
        """Number of items left in the bag."""
        return len(self.data)

    def __repr__(self):
        """String representation."""
        return f"FastRandomBag({self.data!r})"

    # Iterator protocol
    def __iter__(self):
        """Iterator that pops random elements until the bag is empty."""
        return self

    def __next__(self):
        """Return next random item (and remove it)."""
        if not self.data:
            raise StopIteration
        return self.pop_random()

    def remove(self, item):
        """Remove an item from the bag."""
        if item in self.data:
            self.data.remove(item)


def compare_dicts(a, b):
    """
    :param a: first dictionary
    :param b: second dictionary
    :return: if there is any disagreement, if there is any intersection with agreement, if a subset b, if b subset a, if identical
    """
    a_in_b = True
    b_in_a = True
    overlap = False
    for key in a:
        if key in b:
            overlap = True
            if a[key] != b[key]:
                return True, False, False, False, False
        else:
            a_in_b = False

    for key in b:
        if key in a:
            overlap = True
            if a[key] != b[key]:
                return True, False, False, False, False
        else:
            b_in_a = False
    return False, overlap, a_in_b, b_in_a, a_in_b and b_in_a


def draw_transition(auto, start=''):
    dot = Digraph()
    dot.attr(rankdir='LR', fontname="Helvetica", fontsize="14")
    dot.attr("node", fontname="Helvetica", fontsize="14", height="0.25", width="0.25", shape="plaintext")
    dot.attr("edge", fontname="Helvetica", fontsize="14", arrowhead="vee", arrowsize="0.66")

    # Draw states
    for state in auto:
        sid = state if state else 'ε'
        dot.node(sid)

    # Start arrow
    if start not in auto:
        start = min(auto.keys())
    dot.node('', shape='none', height="0", width="0")
    dot.edge('', 'ε' if start == '' else start)

    # Merge edges
    merged = defaultdict(list)
    for state, transitions in auto.items():
        for symbol, target in transitions.items():
            src = state if state else 'ε'
            dst = target if target else 'ε'
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


class Graph(dict):
    def _repr_mimebundle_(self, include=None, exclude=None):
        print(exclude, include)
        dot = draw_transition(self)
        return dot._repr_mimebundle_(include=include, exclude=exclude)


def draw_automaton(auto, start=''):
    dot = Digraph()
    dot.attr(size="11,11")
    if max([len(x) for x in auto]) > 4:
        default_shape = "box"
        small_shape = "box"
    else:
        default_shape = "ellipse"
        small_shape = "circle"
    dot.attr(rankdir='LR', fontname="Helvetica", fontsize="14")
    dot.attr("node", fontname="Helvetica", fontsize="14",  # fillcolor="#fefeb2",
             shape=default_shape, height="0.35", width="0.35", style="rounded")  # style="rounded, filled"
    dot.attr("edge", fontname="Helvetica", fontsize="14", arrowhead="vee", arrowsize="0.66")

    # Draw states
    for state, (is_final, transitions) in auto.items():
        sid = state if state else 'ε'
        dot.node(sid, peripheries="2" if is_final else "1", shape=small_shape if len(state) <= 1 else default_shape)

    # Start arrow
    if start not in auto:
        start = min(auto.keys())
    dot.node('', shape='none', height="0", width="0")
    dot.edge('', 'ε' if start == '' else start)

    # Merge edges
    merged = defaultdict(list)
    for state, (_, transitions) in auto.items():
        for symbol, target in transitions.items():
            src = state if state else 'ε'
            dst = target if target else 'ε'
            merged[(src, dst)].append(symbol)

    for (src, dst), symbols in merged.items():
        dot.edge(src, dst, label=", ".join(sorted(symbols)))

    return dot


class Automaton(dict):
    def _repr_mimebundle_(self, include=None, exclude=None):
        dot = draw_automaton(self)
        return dot._repr_mimebundle_(include=include, exclude=exclude)
