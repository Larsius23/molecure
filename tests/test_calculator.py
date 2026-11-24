"""Fields for hand-made atoms (no RDKit needed), checked by symmetry and invariance."""

import numpy as np
import pytest

from molecure.config.fields import ComfaFields, ComsiaFields, GridConfig
from molecure.fields.atomdata import AtomData
from molecure.fields.calculator import FieldCalculator
from molecure.fields.errors import FieldError
from molecure.fields.kernels import UFF_VDW


def comfa_settings(**kw):
    return ComfaFields(method="comfa", grid=GridConfig(spacing=1.0, padding=4.0), **kw)


def comsia_settings(**kw):
    return ComsiaFields(method="comsia", grid=GridConfig(spacing=1.0, padding=4.0), **kw)


def make_atoms(coords, charges, symbols=None):
    """A tiny molecule built directly from numbers. Carbon-like unless symbols are given."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    symbols = symbols or ["C"] * n
    return AtomData(
        coords=coords,
        charges=np.asarray(charges, dtype=float),
        vdw_x=np.array([UFF_VDW[s][0] for s in symbols]),
        vdw_d=np.array([UFF_VDW[s][1] for s in symbols]),
        radii=np.full(n, 1.7),
        logp=np.full(n, 0.1441),
        donor=np.zeros(n),
        acceptor=np.zeros(n),
    )


SINGLE = make_atoms([[0.0, 0.0, 0.0]], [-0.4])
PAIR = make_atoms([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [-0.3, 0.3])


def test_single_atom_fields_are_symmetric_about_the_atom():
    calc = FieldCalculator.fit([SINGLE], comfa_settings())
    assert calc.grid.shape == (9, 9, 9)  # padding 4, spacing 1, atom at the centre
    for name, values in calc.compute(SINGLE).items():
        cube = values.reshape(calc.grid.shape)
        assert np.allclose(cube, cube[::-1, :, :]), name
        assert np.allclose(cube, cube[:, ::-1, :]), name
        assert np.allclose(cube, cube[:, :, ::-1]), name
        assert np.allclose(cube, cube.transpose(1, 0, 2)), name


def test_comfa_values_respect_the_truncation_cutoff():
    calc = FieldCalculator.fit([PAIR], comfa_settings())
    blocks = calc.compute(PAIR)
    assert blocks["steric"].max() == pytest.approx(30.0)  # points inside the atoms hit the cap
    assert blocks["electrostatic"].max() <= 30.0
    assert blocks["electrostatic"].min() >= -30.0


def test_fields_do_not_depend_on_where_the_molecule_sits():
    offset = (10.0, -7.0, 3.0)
    a = FieldCalculator.fit([PAIR], comfa_settings())
    b_atoms = PAIR.translated(offset)
    b = FieldCalculator.fit([b_atoms], comfa_settings())
    one, two = a.compute(PAIR), b.compute(b_atoms)
    for name in one:
        assert np.allclose(one[name], two[name], atol=1e-8), name


def test_comsia_blocks_follow_the_include_setting():
    calc = FieldCalculator.fit([PAIR], comsia_settings(include=("steric", "hydrophobic")))
    assert calc.block_names == ("steric", "hydrophobic")
    assert set(calc.compute(PAIR)) == {"steric", "hydrophobic"}


def test_comsia_single_atom_matches_the_formula_at_the_centre():
    calc = FieldCalculator.fit([SINGLE], comsia_settings(include=("electrostatic",)))
    cube = calc.compute(SINGLE)["electrostatic"].reshape(calc.grid.shape)
    centre = tuple(n // 2 for n in calc.grid.shape)
    assert cube[centre] == pytest.approx(0.4)  # -(q * exp(0)) with q = -0.4
    one_step = cube[centre[0] + 1, centre[1], centre[2]]  # 1 A away
    assert one_step == pytest.approx(0.4 * np.exp(-0.3))


def test_comsia_steric_is_negative_everywhere():
    calc = FieldCalculator.fit([PAIR], comsia_settings(include=("steric",)))
    assert np.all(calc.compute(PAIR)["steric"] < 0)


def test_matrix_layout_and_block_accessor():
    calc = FieldCalculator.fit([PAIR], comfa_settings())
    matrix = calc.compute_matrix(["m1", "m2"], [PAIR, PAIR])
    assert matrix.values.shape == (2, 2 * calc.grid.n_points)
    assert np.array_equal(matrix.values[0], matrix.values[1])  # same molecule, same row
    single = calc.compute(PAIR)
    assert np.array_equal(matrix.block("steric")[0], single["steric"])
    assert np.array_equal(matrix.block("electrostatic")[1], single["electrostatic"])


def test_molecule_outside_the_calibration_grid_is_flagged_not_absorbed():
    calc = FieldCalculator.fit([PAIR], comfa_settings())
    grid_before = calc.grid
    far = PAIR.translated((40.0, 0.0, 0.0))
    matrix = calc.compute_matrix(["near", "far"], [PAIR, far])
    assert matrix.outside_grid.tolist() == [False, True]
    assert calc.grid == grid_before  # screening data never moves the grid


def test_grid_is_built_from_all_calibration_molecules():
    other = PAIR.translated((6.0, 0.0, 0.0))
    calc = FieldCalculator.fit([PAIR, other], comfa_settings())
    assert calc.grid.contains(PAIR.coords).all() and calc.grid.contains(other.coords).all()


def test_atomdata_rejects_inconsistent_arrays():
    with pytest.raises(FieldError, match="charges"):
        AtomData(
            coords=np.zeros((2, 3)),
            charges=np.zeros(3),
            vdw_x=np.ones(2),
            vdw_d=np.ones(2),
            radii=np.ones(2),
            logp=np.zeros(2),
            donor=np.zeros(2),
            acceptor=np.zeros(2),
        )


def test_atomdata_rejects_nan():
    with pytest.raises(FieldError, match="not finite"):
        make_atoms([[0.0, 0.0, 0.0]], [float("nan")])


def test_matrix_rejects_mismatched_ids():
    calc = FieldCalculator.fit([PAIR], comfa_settings())
    with pytest.raises(FieldError, match="ids"):
        calc.compute_matrix(["only_one"], [PAIR, PAIR])


def test_a_larger_probe_has_a_larger_repulsive_region():
    small = FieldCalculator.fit([PAIR], comfa_settings(probe_radius=1.5))
    large = FieldCalculator.fit([PAIR], comfa_settings(probe_radius=2.5))

    def capped_points(calc) -> int:
        return int(np.sum(calc.compute(PAIR)["steric"] >= 30.0))

    assert capped_points(large) > capped_points(small)
