"""RMSD between poses, measured IN PLACE (the poses are never moved).

Poses live in a shared frame: a docked pose is in the receptor frame, a literature-aligned
set is fitted to a template. Superimposing them (rdMolAlign.AlignMol) would change the
coordinates, which both destroys that frame and hides the very misalignment we want to measure.
"""

import math
from collections.abc import Sequence

import numpy as np


def min_rmsd_over_matches(
    ref_xyz: np.ndarray,
    mol_xyz: np.ndarray,
    ref_match: Sequence[int],
    mol_matches: Sequence[Sequence[int]],
) -> float:
    """RMSD (angstrom) of the matched atoms, taken in place.

    `ref_match[k]` and `match[k]` are the same core atom in the reference and in the molecule.
    A symmetric core has several equivalent matches (a flipped phenyl ring, say): the smallest
    RMSD over all of them is returned, so symmetry is never mistaken for misalignment.
    """
    if not mol_matches:
        raise ValueError("no atom matches to compare")
    ref_points = np.asarray(ref_xyz)[list(ref_match)]
    best = math.inf
    for match in mol_matches:
        if len(match) != len(ref_match):
            raise ValueError("every match must have as many atoms as the reference match")
        difference = np.asarray(mol_xyz)[list(match)] - ref_points
        best = min(best, math.sqrt(float(np.mean(np.sum(difference**2, axis=1)))))
    return best
