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
        r"""Remove and return a random item in :math:`\mathcal{O}(1)` time."""
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
    Takes a dataframe and plots the data with a power law fit :math:`a \cdot x^b`.
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
      criterion: "Average",
      param_name: none,
      color_map: lq.color.map.petroff10,
    ) = {
      lq.diagram(
        width: width,
        height: height,
        xlabel: [Automaton size $abs(Q)$],
        ylabel: [#criterion number of @EQ:pla],
        legend: (position: left + top),
        ..(
          for (i, figure_data) in all_figures.enumerate() {
            (
              lq.plot(
                figure_data.x,
                figure_data.y,
                stroke: none,
                label: $#param_name #figure_data.param_value$,
                color: color_map.at(i),
              ),
              lq.plot(
                lq.linspace(calc.min(..figure_data.x), calc.max(..figure_data.x)),
                x => figure_data.a * calc.pow(x, figure_data.b),
                smooth: true,
                mark: none,
                label: $#figure_data.a dot n^#figure_data.b,
                R^2 = #{ calc.round(figure_data.r * 100, digits: 2) }%$,
                color: color_map.at(i),
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
        (points,) = plt.plot(x, y, "o", label=label)
        color = points.get_color()

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
        plt.plot(x_fit, y_fit, "--", label=label, color=color)

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


def plot_grouped_counts(
    df: DataFrame,
    group_by="alphabet_size",
    output_typst=False,
):
    r"""
    Plot the number of automata in the sample by parameter group.

    If desired it outputs a typst array that can be used as an input for this function:

    ```typst
    #let multi_data_bar_plot(
      x_data,
      y_sets,
      width: 12cm,
      height: 8cm,
      gap_size: 0.2,
      xlabel: [Automaton size $abs(Q)$],
      ylabel: [Number of @wDBA],
      group_label: $abs(Sigma) =$,
    ) = {
      let group_width = 1 - gap_size
      let bar_width = group_width / y_sets.len()
      let base_offset = -(group_width - bar_width) / 2

      lq.diagram(
        width: width,
        height: height,
        ..(
          for (i, y_data) in y_sets.enumerate() {
            (
              lq.bar(
                x_data,
                y_data.at(1),
                width: bar_width,
                offset: base_offset + i * bar_width,
                label: $#group_label #y_data.at(0)$,
              ),
            )
          }
        ),
        grid: none,
        xaxis: (subticks: none),
        yaxis: (subticks: none),
        xlabel: xlabel,
        ylabel: ylabel,
      )
    }
    ```

    Args:
        df: The data frame to plot.
        group_by: What column to group by.
        output_typst: Whether to output the typst array.
    """
    # Count rows per (group, automaton_size)
    df_counts = (
        df.groupby([group_by, "automaton_size"]).size().reset_index(name="count")
    )

    groups = sorted(df_counts[group_by].unique())
    sizes = sorted(df_counts["automaton_size"].unique())

    x = np.arange(len(sizes))
    width = 0.8 / len(groups)

    plt.figure()

    all_group_counts = []

    for i, group in enumerate(groups):
        subset = df_counts[df_counts[group_by] == group]

        counts = [
            subset[subset["automaton_size"] == size]["count"].values[0]
            if size in subset["automaton_size"].values
            else 0
            for size in sizes
        ]

        all_group_counts.append(counts)

        offset = (i - (len(groups) - 1) / 2) * width
        plt.bar(x + offset, counts, width=width, label=f"{group_by}={group}")

    plt.xticks(x, sizes)
    plt.xlabel("automaton size")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Typst output
    if output_typst:
        x_tuple = ", ".join(map(str, sizes))

        group_blocks = []
        for group, counts in zip(groups, all_group_counts):
            group_label = f'"{group}"'  # keep as string
            counts_tuple = ", ".join(map(str, counts))
            group_blocks.append(f"    ({group_label}, ({counts_tuple}))")

        typst_string = f"({x_tuple}),\n(\n" + ",\n".join(group_blocks) + ",\n)"

        print(typst_string)
