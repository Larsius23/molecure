import numpy as np
import pytest

from molecure.fields.grid import Grid


def test_from_coords_encloses_everything_with_padding():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    grid = Grid.from_coords([coords], spacing=1.0, padding=2.0)
    assert grid.origin == (-2.0, -2.0, -2.0)
    assert grid.shape == (8, 6, 5)  # extent 7 x 5 x 4 -> +1 points each
    assert grid.n_points == 8 * 6 * 5
    assert np.all(grid.upper >= coords.max(axis=0) + 2.0 - 1e-9)


def test_spacing_that_does_not_divide_the_extent_still_covers_it():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    grid = Grid.from_coords([coords], spacing=2.0, padding=1.0)  # extent 5 in x
    assert grid.shape[0] == 4  # 0, 2, 4, 6 -> reaches past 4
    assert grid.upper[0] >= 4.0


def test_several_molecules_share_one_grid():
    a = np.array([[0.0, 0, 0]])
    b = np.array([[10.0, 0, 0]])
    grid = Grid.from_coords([a, b], spacing=1.0, padding=0.0)
    assert grid.shape[0] == 11


def test_points_layout_x_slowest_z_fastest():
    grid = Grid(origin=(0.0, 0.0, 0.0), spacing=2.0, shape=(2, 2, 3))
    pts = grid.points()
    assert pts.shape == (12, 3)
    assert pts[0].tolist() == [0, 0, 0]
    assert pts[1].tolist() == [0, 0, 2]  # z moved first
    assert pts[3].tolist() == [0, 2, 0]
    assert pts[-1].tolist() == [2, 2, 4]


def test_contains_flags_atoms_outside_the_box():
    grid = Grid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(5, 5, 5))  # box is 0..4
    coords = np.array([[1.0, 1, 1], [4.0, 4, 4], [4.1, 2, 2], [-0.1, 0, 0]])
    assert grid.contains(coords).tolist() == [True, True, False, False]


def test_shifted_moves_the_origin_only():
    grid = Grid(origin=(0.0, 0.0, 0.0), spacing=2.0, shape=(3, 3, 3))
    moved = grid.shifted((0.5, 0, -1))
    assert moved.origin == (0.5, 0.0, -1.0)
    assert moved.spacing == 2.0 and moved.shape == (3, 3, 3)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="zero molecules"):
        Grid.from_coords([], spacing=1.0, padding=1.0)
