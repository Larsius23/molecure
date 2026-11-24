"""Splitting a calibration set into training and external (test) compounds.

A split is saved with every run (split.csv), so the same compounds are held out next time.
`split` is a SplitConfig; only its attributes `method`, `test_fraction` and `file` are read.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SUBSETS = ("train", "test")
MAX_SHOWN = 10


class SplitError(Exception):
    """The train/test split cannot be made."""


def _listing(title: str, items: Sequence[str]) -> str:
    shown = "\n".join(f"  - {item}" for item in items[:MAX_SHOWN])
    more = len(items) - MAX_SHOWN
    return f"{title}\n{shown}" + (f"\n  ... and {more} more" if more > 0 else "")


def read_split_file(path: str | Path) -> dict[str, str]:
    """A CSV with the columns compound_id and subset (train or test)."""
    path = Path(path)
    if not path.is_file():
        raise SplitError(f"split file not found: {path}")
    assignment: dict[str, str] = {}
    problems: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        if not {"compound_id", "subset"} <= set(reader.fieldnames):
            raise SplitError(f"{path.name}: needs the columns compound_id and subset")
        for line, row in enumerate(reader, start=2):
            compound = (row["compound_id"] or "").strip()
            subset = (row["subset"] or "").strip().lower()
            if not compound:
                problems.append(f"line {line}: empty compound_id")
            elif subset not in SUBSETS:
                problems.append(f"line {line}: subset '{subset}' is not 'train' or 'test'")
            elif compound in assignment:
                problems.append(f"line {line}: {compound} appears twice")
            else:
                assignment[compound] = subset
    if problems:
        raise SplitError(_listing(f"{path.name}:", problems))
    return assignment


def make_split(split, ids: Sequence[str], root: str | Path, seed: int) -> dict[str, str]:
    """{compound id: 'train' | 'test'} for every id, in the order of `ids`."""
    ids = list(ids)
    if split.method == "from_file":
        assignment = read_split_file(Path(root) / split.file)
        unknown = sorted(set(assignment) - set(ids))
        missing = [i for i in ids if i not in assignment]
        problems = []
        if unknown:
            problems.append(_listing("in the split file but not in the dataset:", unknown))
        if missing:
            problems.append(_listing("in the dataset but not in the split file:", missing))
        if problems:
            raise SplitError("\n".join(problems))
        return {i: assignment[i] for i in ids}

    if split.test_fraction == 0:
        return {i: "train" for i in ids}  # no external set

    if split.method == "random":
        n = len(ids)
        n_test = max(1, round(split.test_fraction * n))
        if n - n_test < 3:
            raise SplitError(f"a test fraction of {split.test_fraction} leaves {n - n_test} of {n}")
        rng = np.random.default_rng([seed, 0])  # derived from the single project seed
        test = set(rng.permutation(n)[:n_test].tolist())
        return {c: ("test" if i in test else "train") for i, c in enumerate(ids)}

    raise SplitError(
        f"split method '{split.method}' with a test fraction above 0 is not implemented yet "
        "(it needs the clustering module); use 'random' or 'from_file'"
    )


def write_split(path: str | Path, assignment: Mapping[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["compound_id", "subset"])
        writer.writerows(assignment.items())
