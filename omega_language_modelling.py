from functools import cache
from itertools import product


class llstr(str):
    """A string class that compares by length, then lexicographically."""

    def __new__(cls, value):
        # Ensure we actually create a string subclass properly
        return super().__new__(cls, value)

    def _compare_key(self):
        # Return a tuple key for comparison: (length, string)
        return len(self), str(self)

    def __lt__(self, other):
        if isinstance(other, str):
            return self._compare_key() < llstr(other)._compare_key()
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, str):
            return self._compare_key() <= llstr(other)._compare_key()
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, str):
            return self._compare_key() > llstr(other)._compare_key()
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, str):
            return self._compare_key() >= llstr(other)._compare_key()
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, str):
            return self._compare_key() == llstr(other)._compare_key()
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, str):
            return self._compare_key() != llstr(other)._compare_key()
        return NotImplemented

    def __add__(self, other):
        # Return a LengthLexStr instead of plain str
        if type(other) == omegastr:  # type: ignore
            return NotImplemented
        return llstr(super().__add__(str(other)))

    def __radd__(self, other):
        # Handles str + LengthLexStr
        return llstr(str(other) + str(self))

    def __hash__(self):
        return hash(str(self))

    def __getitem__(self, key):
        """Return a llstr for slices and single-character access."""
        result = super().__getitem__(key)
        if isinstance(result, str):
            return llstr(result)
        return result

    def rstrip(self, chars=None):
        stripped = super().rstrip(chars)
        return llstr(stripped)


class omegastr:
    """A pair of LengthLexStrs, compared lexicographically by (first, then second)."""

    def __init__(self, prefix, loop, alphabet=None, simplify=True):
        # Ensure both are LengthLexStrs
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
        if simplify:
            self.simplify()

    def _simplify(self):
        prefix, loop = self.prefix, self.loop

        m = len(loop)
        # count how many trailing chars in prefix can be removed
        # matching loop's trailing char under successive right-rotations
        i = 0
        lp = len(prefix)
        while i < lp:
            # compare prefix[-1 - i] with loop[(-1 - i) % m]
            if prefix[-1 - i] != loop[(-1 - i) % m]:
                break
            i += 1

        if i:
            prefix = prefix[:-i]
            # rotating loop right by i is equivalent to taking i mod m
            k = i % m
            if k:
                loop = loop[-k:] + loop[:-k]  # if k == 0, loop remains unchanged

        # reduce loop to its smallest period (if any)
        if loop:
            doubled = loop + loop
            idx = doubled.find(loop, 1)
            if 0 < idx < len(loop):
                loop = loop[:idx]

        return prefix, loop

    def simplify(self):
        self.prefix, self.loop = self._simplify()

    def simplified(self):
        prefix, loop = self._simplify()
        return self.__class__(prefix, loop, self.alphabet, False)

    @staticmethod
    def _minimize(prefix, loop):
        m = len(loop)

        best_loop = loop
        best_k = 0

        for k in range(1, m):
            rot = loop[-k:] + loop[:-k]
            best_k = k

            if rot < best_loop:
                best_loop = rot
                best_k = k

        best_prefix = prefix + loop[:-best_k]

        return best_prefix, best_loop

    def minimize(self):
        self.prefix, self.loop = self._minimize(self.prefix, self.loop)

    def minimized(self):
        prefix, loop = self._minimize(self.prefix, self.loop)
        return self.__class__(prefix, loop, self.alphabet, False)

    def _compare_key(self):
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        prefix_len = len(self.prefix)
        loop_len = len(self.loop)
        return (prefix_len + loop_len, loop_len, self.prefix + self.loop)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() < other._compare_key()
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() <= other._compare_key()
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() > other._compare_key()
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() >= other._compare_key()
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() == other._compare_key()
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self._compare_key() != other._compare_key()
        return NotImplemented

    def __repr__(self):
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
        return hash(self._compare_key())

    def __len__(self):
        return len(self.prefix) + len(self.loop)

    def __contains__(self, item):
        item = str(item)
        if not item:
            return True

        prefix = self.prefix
        loop = self.loop

        n = (len(item) // len(loop)) + 2

        finite_sample = prefix + loop * n

        return item in finite_sample

    def is_prefix(self, x):
        return self.__class__.check_prefix(self, x)

    @staticmethod
    def check_prefix(omega, x):
        return omega[:len(x)] == x

    def subtract_prefix(self, x):
        if not self.is_prefix(x):
            raise ValueError("The string to subtract is not a prefix.")
        return self[len(x):]

    def __getitem__(self, key):
        """
        Indexing and slicing for omega-words.

        - int  -> single character
        - finite slice -> llstr (finite prefix)
        - slice with stop=None -> omegastr (infinite suffix, rotated cycle)
        """
        u, v = self.prefix, self.loop
        len_u, len_v = len(u), len(v)

        # --- single character ---
        if isinstance(key, int):
            if key < 0:
                raise IndexError("Negative indexing not supported for infinite omega-word.")
            return u[key] if key < len_u else v[(key - len_u) % len_v]

        # --- slice ---
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

    def get_simple_str(self):
        return str(self.prefix) + str(self.loop)

    def __radd__(self, other):
        return self.__class__(other + self.prefix, self.loop, self.alphabet)

    def get_alphabet(self):
        return set(self.prefix) | set(self.loop)


class omegastr_loop(omegastr):
    def _compare_key(self):
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        return (self.loop._compare_key(), self.prefix._compare_key())

    def __repr__(self):
        return f"ωˡ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_length(alphabet, length):
    length_strings = []
    output_strings = []
    for s in product(alphabet, repeat=length):
        s = ''.join(s)
        length_strings.append(s)
        word = omegastr(s[:-1], s[-1])
        if len(word) == length:
            output_strings.append(word)
    for l in range(2, length + 1):
        for s in length_strings:
            word = omegastr(s[:length - l], s[-l:])
            if len(word) == length:
                output_strings.append(word)
    return output_strings


def omegaiter(alphabet="ab", limit=None):
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_length(alphabet, length)
        length += 1


class omegastr_prefix(omegastr):
    def _compare_key(self):
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        prefix_len = len(self.prefix)
        loop_len = len(self.loop)
        return (prefix_len + loop_len, prefix_len, self.prefix + self.loop)

    def __repr__(self):
        return f"ωᵖ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_prefix_length(alphabet, length):
    length_strings = []
    output_strings = []
    for s in product(alphabet, repeat=length):
        s = ''.join(s)
        length_strings.append(s)
        word = omegastr_prefix("", s)
        if len(word) == length:
            output_strings.append(word)
    for l in range(1, length):
        for s in length_strings:
            word = omegastr_prefix(s[:l], s[l:])
            if len(word) == length:
                output_strings.append(word)

    return output_strings


def omegaiter_prefix(alphabet="ab", limit=None):
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_prefix_length(alphabet, length)
        length += 1


class omegastr_expansion(omegastr):
    __slots__ = ("_key",)

    def _compare_key(self):
        try:
            return self._key
        except AttributeError:
            length = len(self)
            self._key = (length, self[:2 * length])
            return self._key

    def __repr__(self):
        return f"ωᵉ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_expansion_length(alphabet, length):
    length_strings = list()
    for s in map(''.join, product(alphabet, repeat=length)):
        for l in range(length):
            word = omegastr_expansion(s[:l], s[l:])
            if len(word) == length:
                length_strings.append(word)

    return sorted(length_strings)


def omegaiter_expansion(alphabet="ab", limit=None):
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_expansion_length(alphabet, length)
        length += 1


class omegastr_lex(omegastr):
    def _compare_key(self):
        # Lexicographic tuple of the comparison keys: loop first, then prefix
        return (self.prefix + self.loop, len(self.loop))

    def __repr__(self):
        return f"ωˡˡ({self.prefix if self.prefix else 'ε'}, {self.loop})"


@cache
def _omegaiter_lex_length(alphabet, length):
    length_strings = list()
    output_strings = list()
    for s in product(alphabet, repeat=length):
        s = ''.join(s)
        for l in range(1, length):
            word = omegastr_lex(s[:length - l], s[-l:])
            if len(word) == length:
                output_strings.append(word)
        word = omegastr_lex("", s)
        if len(word) == length:
            output_strings.append(word)

    return output_strings


def omegaiter_lex(alphabet="ab", limit=None):
    length = 1
    while limit is None or length <= limit:
        yield from _omegaiter_lex_length(alphabet, length)
        length += 1
