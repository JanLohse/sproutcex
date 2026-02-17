"""
Here functions are provided to test SproutCEX on large samples of randomly generated
weak deterministic Büchi automata. With `perform_sample_test` one can easily generate
a sample, perform SproutCEX on that set in parallel, and get results to evaluate the
efficiency of SproutCEX.
"""

import os
import pickle
import random
import sqlite3
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from webbrowser import Opera

import joblib
import pandas as pd
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm

from .graph_functions import Automaton, generate_wdba
from .sproutcex import ConsMethod, Ordering, CONS_METHODS, ORDERINGS

FULL_ALPHABET = string.ascii_lowercase


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager for showing a progress bar during parallel execution."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def reciprocal_distribution(a: int, b: int) -> int:
    r"""
    Returns a value sampled from a discrete reciprocal distribution. This distribution
    is linear in log space and gives a higher likeliness to lower values. Values are
    computed as $(a - 0.5)^{1 - x} \cdot (b + 0.5)^x$, where $x$ is uniformly
    distributed over $[0, 1)$, and the output is rounded.

    Args:
        a: The lower bound $a$ of the distribution.
        b: The upper bound $b$ of the distribution (is included as possible output).

    Returns:
        A discrete value sampled from the reciprocal distribution over $[a, b]$.
    """
    x = random.random()
    y = (a - 0.5) ** (1 - x) * (b + 0.5) ** x
    return round(y)


def sproutcex_silent(
    target: Automaton,
    cons_method: ConsMethod = "dba",
    ordering: Ordering = "total",
    max_steps: Optional[int] = None,
    square_threshold: bool = False,
) -> tuple[Optional[Automaton], int] | tuple[None, None]:
    r"""
    Attempts to learn an $\omega$-automaton from smallest counterexamples. Implements
    **SproutCEX** from *Learning $\omega$-Automata from Smallest Counterexamples* by Jan
    Lohse. This version reduces the output and is made to be used during batch testing.

    Args:
        target: Automaton that is to be learned.
        cons_method: The passive learner to use for constructing automata.
        ordering: How to order the automaton.
        max_steps: How many steps before aborting.
        square_threshold: Whether to use the original square threshold for Sprout.

    Returns:
        An automaton equivalent to the target, if one is found, and the number of
        queries needed to find it.
    """
    sprout_method = CONS_METHODS[cons_method]
    cex_method = ORDERINGS[ordering]

    plus = set()
    minus = set()
    found = False
    query_count = 0

    while not found:
        if max_steps is not None:
            if not max_steps:
                return None, None
            max_steps -= 1
        automaton = sprout_method(plus, minus, square_threshold=square_threshold)
        found, cex, cex_result = cex_method(automaton, target)
        query_count += 1

        if found:
            continue

        cex.reduce()

        if cex_result:
            if cex in plus:
                return None, None
            plus.add(cex)
        else:
            if cex in minus:
                return None, None
            minus.add(cex)

    return automaton, query_count


def generate_automaton(
    alphabet_low=2, alphabet_high=4, state_low=4, state_high=25
) -> Automaton:
    """
    Generates a random wDBA with its size chosen according to a reciprocal
    distribution.

    Args:
        alphabet_low: Minimum length of the alphabet.
        alphabet_high: Maximum length of the alphabet.
        state_low: Minimum upper state bound of the automaton.
        state_high: Maximum state count of the automaton.

    Returns:
        A random weak deterministic Büchi automaton.
    """
    alphabet_size = reciprocal_distribution(alphabet_low, alphabet_high)
    state_count = reciprocal_distribution(
        state_low,
        round(state_high * alphabet_low / alphabet_size),
    )
    automaton = generate_wdba(state_count, FULL_ALPHABET[:alphabet_size])

    return automaton


def generate_automata(
    alphabet_low=2,
    alphabet_high=4,
    state_low=4,
    state_high=25,
    automata_count=100,
    seed=None,
) -> dict[int, Automaton]:
    """
    Generates a set of wDBA with each's size chosen according to a reciprocal
    distribution.

    Args:
        alphabet_low: Minimum length of each automaton's alphabet.
        alphabet_high: Maximum length of each automaton's alphabet.
        state_low: Minimum upper state bound for each automaton.
        state_high: Maximum state count of each automaton.
        automata_count: Number of automata in the set.
        seed: Seed set before generating automata.

    Returns:
        A list of weak deterministic Büchi automata.
    """
    if seed is not None:
        random.seed(seed)

    automata = {}

    for i in range(automata_count):
        automaton = generate_automaton(
            alphabet_low, alphabet_high, state_low, state_high
        )
        automata[i] = automaton

    return automata


def get_file_path(file_name, path=None, folder_name="data") -> Path:
    """Get the path to a file in the specified path and folder."""
    if path is None:
        path = Path()

    folder_path = path / folder_name
    folder_path.mkdir(exist_ok=True)
    file_path = folder_path / file_name

    return file_path


def load_automata(
    seed,
    automata_count=100,
    alphabet_low=2,
    alphabet_high=4,
    state_low=4,
    state_high=25,
    path=None,
    folder_name="data",
    return_filename=False,
) -> dict[int, Automaton] | tuple[dict[int, Automaton], str]:
    """
    Loads a set of wDBA with each's size chosen according to a reciprocal
    distribution. If such a set is already stored in the path folder it is loaded,
    and if not a new set is generated at stored in the path folder.

    Args:
        seed: Seed set before generating automata.
        automata_count: Number of automata in the set.
        alphabet_low: Minimum length of each automaton's alphabet.
        alphabet_high: Maximum length of each automaton's alphabet.
        state_low: Minimum upper state bound for each automaton.
        state_high: Maximum state count of each automaton.
        path: Where to look for and store the backup of the automaton set.
        folder_name: Name of the folder where to store pickle object.
        return_filename: Whether to also return the filename of the automata file.

    Returns:
        A list of weak deterministic Büchi automata and the filename, if specified.
    """
    file_name = (
        f"automata_{alphabet_low}_{alphabet_high}_{state_low}_{state_high}_"
        f"{automata_count}_{seed}.pkl"
    )

    file_path = get_file_path(file_name, path=path, folder_name=folder_name)

    if file_path.exists():
        with open(file_path, "rb") as f:
            automata = pickle.load(f)
    else:
        automata = generate_automata(
            alphabet_low, alphabet_high, state_low, state_high, automata_count, seed
        )
        with open(file_path, "wb") as f:
            pickle.dump(automata, f)

    if return_filename:
        return automata, file_name.split(".")[0]
    else:
        return automata


def init_db(file_path) -> sqlite3.Connection:
    """Create a sqlite database."""
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL;")  # enable concurrency

    # Create table if it doesn't exist
    cursor.execute("""
   CREATE TABLE IF NOT EXISTS automata_results (
                                                   idx INTEGER PRIMARY KEY,
                                                   alphabet_size INTEGER,
                                                   automaton_size INTEGER,
                                                   query_count INTEGER
   )
   """)
    conn.commit()

    return conn


def get_completed_indices(conn: sqlite3.Connection) -> set[int]:
    """Get the already computed indices from the sqlite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT idx FROM automata_results")

    return set(row[0] for row in cursor.fetchall())


def process_single_automaton_worker(idx: int, automaton: Automaton, db_path: Path):
    """Perform SproutCEX and write to sqlite database."""
    reduced_automaton, query_count = sproutcex_silent(automaton)
    alphabet_size = len(automaton.get_alphabet())
    automaton_size = len(reduced_automaton)
    row = {
        "idx": idx,
        "alphabet_size": alphabet_size,
        "automaton_size": automaton_size,
        "query_count": query_count,
    }

    for _ in range(10):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO automata_results
                (idx, alphabet_size, automaton_size, query_count)
                VALUES (:idx, :alphabet_size, :automaton_size, :query_count)
            """,
                row,
            )

            conn.commit()

            conn.close()

            break
        except sqlite3.OperationalError:
            time.sleep(0.1)



def perform_sample_test(
    seed: int,
    automata_count=100,
    alphabet_low=2,
    alphabet_high=4,
    state_low=4,
    state_high=25,
    path=None,
    folder_name="data",
    core_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a sample of weak deterministic Büchi automata and performs SproutCEX on
    all of them. Intermediate results are stored and when calling again it is picked up
    where it was stopped last time. The computation is performed in parallel.

    Args:
        seed: Seed set before generating automata.
        automata_count: Number of automata in the set.
        alphabet_low: Minimum length of each automaton's alphabet.
        alphabet_high: Maximum length of each automaton's alphabet.
        state_low: Minimum upper state bound for each automaton.
        state_high: Maximum state count of each automaton.
        path: Where to look for and store the backup of the automaton set.
        folder_name: Name of the folder where to store pickle object.
        core_count: How many parallel threads of SproutCEX to perform at once.

    Returns:
        The database with the results for all runs.
    """
    automata, file_name = load_automata(
        seed=seed,
        automata_count=automata_count,
        alphabet_low=alphabet_low,
        alphabet_high=alphabet_high,
        state_low=state_low,
        state_high=state_high,
        path=path,
        folder_name=folder_name,
        return_filename=True,
    )
    # Create and connect do database for results.
    db_path = get_file_path(file_name + ".db", path=path, folder_name=folder_name)
    connection = init_db(db_path)

    # Get indices of automata that still need processing.
    completed_indices = get_completed_indices(connection)
    leftover_indices = list(set(automata) - completed_indices)
    total_automata = len(automata)
    already_done = len(completed_indices)
    connection.close()

    # Perform SproutCEX in parallel.
    if core_count is None:
        core_count = os.cpu_count()

    with tqdm_joblib(
        tqdm(total=total_automata, initial=already_done, desc="Processing automata")
    ):
        Parallel(n_jobs=core_count)(
            delayed(process_single_automaton_worker)(idx, automata[idx], db_path)
            for idx in leftover_indices
        )

    # Load result to pandas database.
    connection = sqlite3.connect(db_path)
    results = pd.read_sql("SELECT * FROM automata_results ORDER BY idx", connection)
    connection.close()

    return results
