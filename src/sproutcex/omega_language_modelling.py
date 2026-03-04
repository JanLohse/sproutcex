r"""
Classes modeling ultimately periodic words with a variety of :math:`\omega`-orderings.
"""

from collections.abc import Iterable
from functools import cache
from itertools import product
from typing import Iterator


class llstr(str):
    """A string subclass that compares by length first, then lexicographically."""

    def __new__(cls, value):
        """Create a new llstr instance from a string value."""
        return super().__new__(cls, value)

    def _compare_key(self):
        """Return a tuple used for comparison: (length, string)."""
        return len(self), str(self)

    def __lt__(self, other):
        """Compare self < other using length-lexicographic order."""
        if isinstance(other, str):
            return self._compare_key() < llstr(other)._compare_key()
        return NotImplemented

    def __le__(self, other):
        """Compare self <= other using length-lexicographic order."""
        if isinstance(other, str):
            return self._compare_key() <= llstr(other)._compare_key()
        return NotImplemented

    def __gt__(self, other):
        """Compare self > other using length-lexicographic order."""
        if isinstance(other, str):
            return self._compare_key() > llstr(other)._compare_key()
        return NotImplemented

    def __ge__(self, other):
        """Compare self >= other using length-lexicographic order."""
        if isinstance(other, str):
            return self._compare_key() >= llstr(other)._compare_key()
        return NotImplemented

    def __eq__(self, other):
        """Compare self == other."""
        if isinstance(other, str):
            return self._compare_key() == llstr(other)._compare_key()
        return NotImplemented

    def __ne__(self, other):
        """Compare self != other using."""
        if isinstance(other, str):
            return self._compare_key() != llstr(other)._compare_key()
        return NotImplemented

    def __add__(self, other):
        """Return a llstr when concatenating with another string."""
        if type(other) is Omegastr:  # type: ignore
            return NotImplemented
        return llstr(super().__add__(str(other)))

    def __radd__(self, other):
        """Return a llstr when a string is added to this llstr."""
        return llstr(str(other) + str(self))

    def __hash__(self):
        """Compute the hash based on the string value."""
        return hash(str(self))

    def __getitem__(self, key):
        """Return a llstr for slices or single-character access."""
        result = super().__getitem__(key)
        if isinstance(result, str):
            return llstr(result)
        return result

    def rstrip(self, chars=None):
        """Return a llstr with trailing white space removed."""
        stripped = super().rstrip(chars)
        return llstr(stripped)


class Omegastr:
    r"""An ultimately periodic word :math:`u v^\omega` sorted by
    :math:`(|uv|, |v|, uv_\text{lex})`."""

    prefix: llstr
    r"""The prefix :math:`u` of :math:`u v^\omega`."""
    loop: llstr
    r"""The loop :math:`v` of :math:`u v^\omega`."""

    def __init__(
        self, prefix: str, loop: str, alphabet: None | Iterable = None, reduce=True
    ):
        """Creates a new UP word from its prefix and loop."""
        # Make prefix and loop llstr for comparison purposes.
        if alphabet is None:
            self.alphabet = "".join(sorted(set(prefix).union(set(loop))))
        else:
            self.alphabet = alphabet
        if not isinstance(prefix, llstr):
            prefix = llstr(prefix)
        if not isinstance(loop, llstr):
            loop = llstr(loop)

        self.prefix = prefix
        self.loop = loop

        # Reduce prefix and loop if asked for.
        if reduce:
            self.reduce()

    @staticmethod
    def _reduce(prefix, loop):
        """Reduces a prefix loop pair to the reduced UP word."""
        m = len(loop)
        # Count how many trailing chars in prefix can be removed,
        # matching loop's trailing char under successive right-rotations.
        i = 0
        lp = len(prefix)
        while i < lp:
            # Compare prefix[-1 - i] with loop[(-1 - i) % m].
            if prefix[-1 - i] != loop[(-1 - i) % m]:
                break
            i += 1

        if i:
            prefix = prefix[:-i]
            # Rotating loop right by i is equivalent to taking i mod m.
            k = i % m
            if k:
                loop = loop[-k:] + loop[:-k]  # if k == 0, loop remains unchanged

        # Reduce loop to its smallest period (if any).
        if loop:
            doubled = loop + loop
            idx = doubled.find(loop, 1)
            if 0 < idx < len(loop):
                loop = loop[:idx]

        return prefix, loop

    def reduce(self):
        """Reduces the representation of the UP word inplace."""
        self.prefix, self.loop = self._reduce(self.prefix, self.loop)

    def reduced(self):
        """Returns a reduced representation of the UP word."""
        prefix, loop = self._reduce(self.prefix, self.loop)
        return self.__class__(prefix, loop, self.alphabet, False)

    def _compare_key(self):
        """
        Returns a tuple for comparison: (combined length, length of loop, prefix + loop)
        """
        prefix_len = len(self.prefix)
        loop_len = len(self.loop)
        return prefix_len + loop_len, loop_len, self.prefix + self.loop

    def __lt__(self, other):
        """Compare self < other based on compare key."""
        if isinstance(other, self.__class__):
            return self._compare_key() < other._compare_key()
        return NotImplemented

    def __le__(self, other):
        """Compare self <= other based on compare key."""
        if isinstance(other, self.__class__):
            return self._compare_key() <= other._compare_key()
        return NotImplemented

    def __gt__(self, other):
        """Compare self > other based on compare key."""
        if isinstance(other, self.__class__):
            return self._compare_key() > other._compare_key()
        return NotImplemented

    def __ge__(self, other):
        """Compare self >= other based on compare key."""
        if isinstance(other, self.__class__):
            return self._compare_key() >= other._compare_key()
        return NotImplemented

    def __eq__(self, other):
        """Compare self == other."""
        if isinstance(other, self.__class__):
            return self._compare_key() == other._compare_key()
        return NotImplemented

    def __ne__(self, other):
        """Compare self != other."""
        if isinstance(other, self.__class__):
            return self._compare_key() != other._compare_key()
        return NotImplemented

    def __repr__(self):
        """Returns a string representation of the UP word."""
        return f"ω({self.prefix if self.prefix else 'ε'}, {self.loop})"

    def __iter__(self):
        """Yield the infinite ultimately periodic ω-word: prefix + loop^ω."""
        # yield prefix once
        for ch in self.prefix:
            yield ch
        # then repeat the loop infinitely
        while True:
            for ch in self.loop:
                yield ch

    def __hash__(self):
        """Compute the hash based on the compare key."""
        return hash(self._compare_key())

    def __len__(self):
        """Return the length of prefix and loop combined."""
        return len(self.prefix) + len(self.loop)

    def __contains__(self, item):
        """Return if string appears in UP word."""
        item = str(item)
        if not item:
            return True

        prefix = self.prefix
        loop = self.loop

        n = (len(item) // len(loop)) + 2

        finite_sample = prefix + loop * n

        return item in finite_sample

    def is_prefix(self, x):
        r"""Returns if :math:`u v^\omega` starts with prefix :math:`x`."""
        return self.__class__.check_prefix(self, x)

    @staticmethod
    def check_prefix(omegaword, x):
        r"""Returns if :math:`x` is prefix of :math:`u v^\omega`."""
        return omegaword[: len(x)] == x

    def subtract_prefix(self, x):
        """Returns a new UP word with the prefix x removed."""
        if not self.is_prefix(x):
            raise ValueError("The string to subtract is not a prefix.")
        return self[len(x) :]

    def __getitem__(self, key):
        """
        Indexing and slicing for UP words.

        - int -> single character
        - finite slice -> llstr (finite prefix)
        - slice with stop=None -> omegastr (infinite suffix, rotated cycle)
        """
        u, v = self.prefix, self.loop
        len_u, len_v = len(u), len(v)

        # --- single character ---
        if isinstance(key, int):
            if key < 0:
                raise IndexError(
                    "Negative indexing not supported for infinite omega-word."
                )
            return u[key] if key < len_u else v[(key - len_u) % len_v]

        # slice
        if not isinstance(key, slice):
            raise TypeError("Index must be int or slice.")

        start = 0 if key.start is None else key.start
        stop = key.stop
        step = 1 if key.step is None else key.step

        if step != 1:
            raise ValueError("Step slicing not supported for omega-words.")
        if start < 0 or (stop is not None and stop < 0):
            raise IndexError("Negative indices not supported for infinite omega-word.")

        # Finite slice → produce a finite llstr prefix
        if stop is not None:

            def char_at(i):
                return u[i] if i < len_u else v[(i - len_u) % len_v]

            return llstr("".join(char_at(i) for i in range(start, stop)))

        # Infinite suffix (omega slice): stop is None
        # Compute where we are inside u·v^ω
        if start < len_u:
            # Still inside prefix
            new_first = u[start:]
            new_second = v
        else:
            # Inside the cycle — rotate it
            offset = (start - len_u) % len_v
            new_first = llstr("")  # no finite prefix left
            new_second = llstr(v[offset:] + v[:offset])
        return self.__class__(new_first, new_second, self.alphabet)

    def __radd__(self, other):
        """Return a UP word with an added prefix."""
        return self.__class__(other + self.prefix, self.loop)

    def get_alphabet(self):
        """Get the alphabet of the UP word."""
        return set(self.prefix) | set(self.loop)


@cache
def _omegaiter_length(alphabet: str, length: int) -> list[Omegastr]:
    """Compute all Omegastrs of a fixed length."""
    length_strings = []
    output_strings = []
    for s in product(alphabet, repeat=length):
        s = "".join(s)
        length_strings.append(s)
        word = Omegastr(s[:-1], s[-1])
        if len(word) == length:
            output_strings.append(word)
    for loop_length in range(2, length + 1):
        for s in length_strings:
            word = Omegastr(s[: length - loop_length], s[-loop_length:])
            if len(word) == length:
                output_strings.append(word)
    return output_strings


def omegaiter(alphabet="ab", limit: None | int = None) -> Iterator[Omegastr]:
    """Iterate over reduced Omegastr in order."""
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_length(alphabet, length)
        length += 1


class OmegastrLoop(Omegastr):
    """UP word sorted by loop then by prefix, length-lexicographically each."""

    def _compare_key(self):
        """
        Length-lexicographic tuple of the comparison keys: loop first, then prefix.
        """
        return (self.loop._compare_key(), self.prefix._compare_key())

    def __repr__(self):
        """Returns a string representation of the UP word."""
        return f"ωˡ({self.prefix if self.prefix else 'ε'}, {self.loop})"


class OmegastrPrefix(Omegastr):
    r"""
    An ultimately periodic word :math:`u v^\omega` sorted by
    :math:`(|uv|, |u|, uv_\text{lex})`.
    """

    def _compare_key(self):
        """
        Return comparison key: (combined length, prefix length, combined representation
        length-lex.).
        """
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        prefix_len = len(self.prefix)
        loop_len = len(self.loop)
        return (prefix_len + loop_len, prefix_len, self.prefix + self.loop)

    def __repr__(self):
        """Return a string representation of the ultimately periodic word."""
        return f"ωᵖ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_prefix_length(alphabet: str, length: int) -> list[OmegastrPrefix]:
    """Iterate over reduced Omegastrs of fixed length in prefix order."""
    length_strings = []
    output_strings = []
    for s in product(alphabet, repeat=length):
        s = "".join(s)
        length_strings.append(s)
        word = OmegastrPrefix("", s)
        if len(word) == length:
            output_strings.append(word)
    for loop_length in range(1, length):
        for s in length_strings:
            word = OmegastrPrefix(s[:loop_length], s[loop_length:])
            if len(word) == length:
                output_strings.append(word)

    return output_strings


def omegaiter_prefix(
    alphabet="ab", limit: None | int = None
) -> Iterator[OmegastrPrefix]:
    """Iterate over reduced Omegastr in prefix order."""
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_prefix_length(alphabet, length)
        length += 1


class OmegastrExpansion(Omegastr):
    r"""
    An ultimately periodic word :math:`u v^\omega` sorted by
    :math:`(|uv|, (uv^\omega)_\text{lex})`.
    """

    __slots__ = ("_key",)

    def _compare_key(self):
        """
        Return compare key: (length of representation, prefix of twice the
        representation length).
        """
        try:
            return self._key
        except AttributeError:
            length = len(self)
            self._key = (length, self[: 2 * length])
            return self._key

    def __repr__(self):
        """String representation of UP word."""
        return f"ωᵉ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_expansion_length(alphabet: str, length: int) -> list[OmegastrExpansion]:
    """Returns all UP words of fixed length in expansion order."""

    length_strings = list()
    for s in map("".join, product(alphabet, repeat=length)):
        for loop_length in range(length):
            word = OmegastrExpansion(s[:loop_length], s[loop_length:])
            if len(word) == length:
                length_strings.append(word)

    return sorted(length_strings)


def omegaiter_expansion(
    alphabet="ab", length_limit: None | int = None
) -> Iterator[OmegastrExpansion]:
    """Iterate over UP words in expansion order."""
    length = 1
    while length_limit is None or length <= length_limit:
        yield from _omegaiter_expansion_length(alphabet, length)
        length += 1


class OmegastrLex(Omegastr):
    r"""
    An ultimately periodic word :math:`u v^\omega` sorted by
    :math:`(|uv|, (uv)_\text{lex}, |v|)`.
    """

    def _compare_key(self):
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        return (self.prefix + self.loop, len(self.loop))

    def __repr__(self):
        """String representation of UP word."""
        return f"ωˡˡ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_lex_length(alphabet: str, length: int) -> list[OmegastrLex]:
    """Returns all UP words of fixed length in representation length-lex order."""
    output_strings = list()
    for s in product(alphabet, repeat=length):
        s = "".join(s)
        for loop_length in range(1, length):
            word = OmegastrLex(s[: length - loop_length], s[-loop_length:])
            if len(word) == length:
                output_strings.append(word)
        word = OmegastrLex("", s)
        if len(word) == length:
            output_strings.append(word)

    return output_strings


def omegaiter_lex(alphabet="ab", limit: None | int = None) -> Iterator[OmegastrLex]:
    """Iterates over UP words in representation length-lex order."""
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_lex_length(alphabet, length)
        length += 1
