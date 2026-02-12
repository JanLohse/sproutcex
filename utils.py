import random


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
