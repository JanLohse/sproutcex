import random
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

plt.style.use("default")


class FastRandomBag:
    """A simple list wrapper to pop random items from a bag."""

    data: list

    def __init__(self, items: None | Iterable = None):
        self.data = list(items) if items else []

    def add(self, item):
        """Add an item to the bag."""
        self.data.append(item)

    def pop_random(self):
        r"""Remove and return a random item in $\mathcal{O}(1)$ time."""
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


def plot_param_with_power_fit(
    df: DataFrame,
    criterion="mean",
    monotonic=False,
    group_by="alphabet_size",
    output_typst=False,
):
    r"""
    Takes a dataframe and plots the data with a power law fit $a \cdot x^b$.
    Expects a dataframe computed by `sproutcex.random_testing.perform_sample_test`.

    Can be configured to print a typst array that can be used as the argument
    `all_figures` in the following function. It will display a figure equivalent to the
    one displayed by this function.
    ```typst
    #import "@preview/lilaq:0.5.0" as lq

    #let regression_figure(
      all_figures,
      width: 12cm,
      height: 8cm,
      criterium: "Average",
      param_name: none,
    ) = {
      lq.diagram(
        width: width,
        height: height,
        xlabel: [Automaton size $abs(Q)$],
        ylabel: [#criterium number of @EQ:pla],
        legend: (position: left + top),
        ..(
          for figure_data in all_figures {
            (
              lq.plot(
                figure_data.x,
                figure_data.y,
                stroke: none,
                label: $#param_name #figure_data.param_value$,
              ),
              lq.plot(
                lq.linspace(calc.min(..figure_data.x), calc.max(..figure_data.x)),
                x => figure_data.a * calc.pow(x, figure_data.b),
                smooth: true,
                mark: none,
                label: $#figure_data.a dot n^#figure_data.b,
                R^2 = #{ calc.round(figure_data.r * 100, digits: 2) }%$,
              ),
            )
          }
        ),
      )
    }
    ```

    Args:
        df: The dataframe to plot.
        criterion: The criterion to aggregate the data by. Default is "mean".
        monotonic: Whether to filter the data to grow monotonically.
        group_by: What parameter to group by.
        output_typst: Whether to print the typst array.
    """
    df_stats = (
        df.groupby([group_by, "automaton_size"])["query_count"]
        .agg([criterion])
        .reset_index()
    )

    if monotonic:
        df_stats = df_stats.sort_values([group_by, "automaton_size"])

        running_max = df_stats.groupby(group_by)[criterion].cummax()

        mask = df_stats[criterion] == running_max
        df_stats = df_stats[mask]

    plt.figure()

    typst_blocks = []

    for group in sorted(df_stats[group_by].unique()):
        subset = df_stats[df_stats[group_by] == group]

        x = subset["automaton_size"].values
        y = subset[criterion].values

        if len(x) < 2:
            continue

        label = f"{group_by}={group}"
        plt.plot(x, y, "o", label=label)

        # --- power-law fit ---
        logx = np.log(x)
        logy = np.log(y)

        b, loga = np.polyfit(logx, logy, 1)
        a_param = np.exp(loga)

        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = a_param * x_fit**b

        y_pred = a_param * x**b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        label = f"{a_param:.4f}·x^{b:.4f}  R²={r2:.4f}"
        plt.plot(x_fit, y_fit, "--", label=label)

        # --- Typst export block ---
        if output_typst:
            x_tuple = ", ".join(map(str, x))
            y_tuple = ", ".join(map(str, y))

            block = f"""  (
    param_value: {group},
    x: ({x_tuple}),
    y: ({y_tuple}),
    a: {a_param:.4f},
    b: {b:.4f},
    r: {r2:.4f},
    )"""
            typst_blocks.append(block)

    plt.xlabel("automaton size")
    plt.ylabel(f"{criterion} query count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if output_typst and typst_blocks:
        typst_string = "(\n" + ",\n".join(typst_blocks) + "\n)"
        print(typst_string)
