"""Check each formula against answers worked out by hand."""

import numpy as np
import pytest

from molecure.fields.kernels import (
    COULOMB_CONSTANT,
    distances,
    electrostatic_coulomb,
    gaussian_field,
    steric_lj,
)

C_X, C_D = 3.851, 0.105  # UFF carbon


def one_atom_distances(*radii):
    """Distance table for ONE atom and grid points at the given distances from it."""
    return np.array(radii, dtype=float).reshape(-1, 1)


def test_distances_shape_and_values():
    points = np.array([[0.0, 0, 0], [3, 4, 0]])
    coords = np.array([[0.0, 0, 0]])
    assert distances(points, coords).tolist() == [[0.0], [5.0]]


def test_lj_minimum_is_minus_well_depth_at_x():
    r = one_atom_distances(C_X)  # carbon atom, carbon probe: x_ij = C_X
    e = steric_lj(r, np.array([C_X]), np.array([C_D]), C_X, C_D, cutoff=30)
    assert e[0] == pytest.approx(-C_D)


def test_lj_is_zero_where_the_two_terms_cancel_and_positive_inside():
    # E = D * (u^2 - 2u) with u = (x/r)^6.  E = 0 at u = 2  ->  r = x / 2^(1/6)
    r0 = C_X / 2 ** (1 / 6)
    e = steric_lj(one_atom_distances(r0, r0 * 0.9), np.array([C_X]), np.array([C_D]), C_X, C_D, 30)
    assert e[0] == pytest.approx(0.0, abs=1e-9)
    assert e[1] > 0


def test_lj_is_truncated_at_the_cutoff_and_survives_r_equal_zero():
    r = one_atom_distances(0.0, 0.5)
    e = steric_lj(r, np.array([C_X]), np.array([C_D]), C_X, C_D, cutoff=30)
    assert e.tolist() == [30.0, 30.0]


def test_lj_decays_to_nearly_zero_far_away():
    e = steric_lj(one_atom_distances(50.0), np.array([C_X]), np.array([C_D]), C_X, C_D, 30)
    assert abs(e[0]) < 1e-6


def test_lj_uses_the_geometric_mean_combining_rule():
    # atom x=4, D=0.04 ; probe x=1, D=1  ->  x_ij = 2, D_ij = 0.2 ; at r = 2 energy = -0.2
    e = steric_lj(one_atom_distances(2.0), np.array([4.0]), np.array([0.04]), 1.0, 1.0, 30)
    assert e[0] == pytest.approx(-0.2)


def test_lj_sums_over_atoms_before_truncating():
    # two atoms each contribute -D at their minimum -> total -2D (no per-atom clipping)
    r = np.array([[C_X, C_X]])
    e = steric_lj(r, np.array([C_X, C_X]), np.array([C_D, C_D]), C_X, C_D, 30)
    assert e[0] == pytest.approx(-2 * C_D)


def test_coulomb_constant_dielectric_by_hand():
    # q = -0.5, probe +1, r = 10 A:  332.0637 * (-0.5) / 10 = -16.603185
    e = electrostatic_coulomb(one_atom_distances(10.0), np.array([-0.5]), 1.0, 30, "constant")
    assert e[0] == pytest.approx(-0.5 * COULOMB_CONSTANT / 10)


def test_coulomb_distance_dielectric_by_hand():
    e = electrostatic_coulomb(one_atom_distances(10.0), np.array([-0.5]), 1.0, 30, "distance")
    assert e[0] == pytest.approx(-0.5 * COULOMB_CONSTANT / 100)


def test_coulomb_sign_follows_the_charges():
    r = one_atom_distances(6.0)
    attract = electrostatic_coulomb(r, np.array([-0.4]), 1.0, 30, "distance")[0]
    repel = electrostatic_coulomb(r, np.array([+0.4]), 1.0, 30, "distance")[0]
    assert attract < 0 < repel


def test_coulomb_is_truncated_on_both_sides():
    r = one_atom_distances(0.5)
    assert electrostatic_coulomb(r, np.array([-1.0]), 1.0, 30, "distance")[0] == -30
    assert electrostatic_coulomb(r, np.array([+1.0]), 1.0, 30, "distance")[0] == +30


def test_coulomb_rejects_unknown_dielectric():
    with pytest.raises(ValueError, match="dielectric"):
        electrostatic_coulomb(one_atom_distances(5.0), np.array([1.0]), 1.0, 30, "fancy")


def test_gaussian_by_hand():
    # w = 1, alpha = 0.3:  r=0 -> -1 ;  r=2 -> -exp(-1.2)
    e = gaussian_field(one_atom_distances(0.0, 2.0), np.array([1.0]), 0.3)
    assert e[0] == pytest.approx(-1.0)
    assert e[1] == pytest.approx(-np.exp(-1.2))


def test_gaussian_is_linear_in_the_atom_weights():
    r = one_atom_distances(1.5)
    one = gaussian_field(r, np.array([1.0]), 0.3)[0]
    assert gaussian_field(r, np.array([-2.5]), 0.3)[0] == pytest.approx(-2.5 * one)
