"""In-place RMSD: the poses are compared where they are, never superimposed."""

import numpy as np
import pytest

from molecure.align.rmsd import min_rmsd_over_matches

REF = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [9.0, 9.0, 9.0]])


def test_identical_poses_have_zero_rmsd():
    assert min_rmsd_over_matches(REF, REF, (0, 1, 2), [(0, 1, 2)]) == pytest.approx(0.0)


def test_a_uniform_shift_is_measured_not_removed():
    shifted = REF + np.array([1.0, 0.0, 0.0])
    assert min_rmsd_over_matches(REF, shifted, (0, 1, 2), [(0, 1, 2)]) == pytest.approx(1.0)


def test_only_the_matched_core_counts():
    other = REF.copy()
    other[3] = [50.0, 50.0, 50.0]  # an unmatched atom far away must not matter
    assert min_rmsd_over_matches(REF, other, (0, 1, 2), [(0, 1, 2)]) == pytest.approx(0.0)


def test_symmetry_is_not_mistaken_for_misalignment():
    flipped = REF.copy()
    flipped[[1, 2]] = REF[[2, 1]]  # two equivalent atoms swapped places
    one_match = min_rmsd_over_matches(REF, flipped, (0, 1, 2), [(0, 1, 2)])
    both_matches = min_rmsd_over_matches(REF, flipped, (0, 1, 2), [(0, 1, 2), (0, 2, 1)])
    assert one_match > 1.0 and both_matches == pytest.approx(0.0)


def test_the_inputs_are_never_modified():
    mol = REF + 0.3
    before_ref, before_mol = REF.copy(), mol.copy()
    min_rmsd_over_matches(REF, mol, (0, 1, 2), [(0, 1, 2)])
    assert np.array_equal(REF, before_ref) and np.array_equal(mol, before_mol)


def test_bad_matches_are_refused():
    with pytest.raises(ValueError, match="no atom matches"):
        min_rmsd_over_matches(REF, REF, (0, 1, 2), [])
    with pytest.raises(ValueError, match="as many atoms"):
        min_rmsd_over_matches(REF, REF, (0, 1, 2), [(0, 1)])
