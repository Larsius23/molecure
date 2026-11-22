"""Turn aligned molecules into field values on a grid.

Workflow (this is where the "no leakage" rule becomes structure):

    calc = FieldCalculator.fit(calibration_atoms, settings)   # the ONLY place a grid is built
    matrix = calc.compute_matrix(ids, atom_sets)              # works for calibration or screening

`fit` sees calibration molecules only. Screening molecules are evaluated on the
grid that already exists; they can be flagged as sticking out of it, but they
cannot change it.

`settings` is a ComfaFields or ComsiaFields object from molecure.config. Only its
attributes are used here (duck typing), so this module does not import the config.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .atomdata import AtomData
from .errors import FieldError
from .grid import Grid
from .kernels import (
    UFF_VDW,
    distances,
    electrostatic_coulomb,
    gaussian_field,
    steric_lj,
)

COMFA_BLOCKS = ("steric", "electrostatic")

# Evaluate the grid in slices so a large grid never needs one giant distance table.
CHUNK_POINTS = 20_000


@dataclass(frozen=True, eq=False)
class FieldMatrix:
    """Field values for a set of molecules, ready for PLS.

    values has one row per molecule. Its columns are the blocks one after another:
    [all steric points | all electrostatic points | ...], each block n_points wide,
    in the order of Grid.points().
    """

    ids: tuple[str, ...]
    block_names: tuple[str, ...]
    n_points: int
    values: np.ndarray  # (n_molecules, n_blocks * n_points)
    outside_grid: np.ndarray  # (n_molecules,) True if any atom lies outside the grid box

    def block(self, name: str) -> np.ndarray:
        """The columns of one field, shape (n_molecules, n_points)."""
        i = self.block_names.index(name)
        return self.values[:, i * self.n_points : (i + 1) * self.n_points]


class FieldCalculator:
    def __init__(self, grid: Grid, settings) -> None:
        self.grid = grid
        self.settings = settings
        self._points = grid.points()
        if settings.method == "comfa":
            self.block_names: tuple[str, ...] = COMFA_BLOCKS
        elif settings.method == "comsia":
            self.block_names = tuple(settings.include)
        else:
            raise FieldError(f"unknown field method {settings.method!r}")

    @classmethod
    def fit(cls, calibration_atoms: Sequence[AtomData], settings) -> "FieldCalculator":
        """Build the grid from CALIBRATION molecules only."""
        grid = Grid.from_coords(
            [a.coords for a in calibration_atoms],
            spacing=settings.grid.spacing,
            padding=settings.grid.padding,
        )
        return cls(grid, settings)

    # ---------- one molecule ----------

    def compute(self, atoms: AtomData) -> dict[str, np.ndarray]:
        """Every field block for one molecule: {block name: array of n_points values}."""
        out = {name: np.empty(self.grid.n_points) for name in self.block_names}
        for start in range(0, self.grid.n_points, CHUNK_POINTS):
            part = slice(start, start + CHUNK_POINTS)
            r = distances(self._points[part], atoms.coords)
            for name, values in self._blocks_for_chunk(r, atoms).items():
                out[name][part] = values
        return out

    def _blocks_for_chunk(self, r: np.ndarray, atoms: AtomData) -> dict[str, np.ndarray]:
        s = self.settings
        if s.method == "comfa":
            probe_x, probe_d = UFF_VDW["C"]  # the probe is an sp3 carbon
            return {
                "steric": steric_lj(
                    r, atoms.vdw_x, atoms.vdw_d, probe_x, probe_d, s.energy_cutoff_kcal
                ),
                "electrostatic": electrostatic_coulomb(
                    r, atoms.charges, s.probe_charge, s.energy_cutoff_kcal, s.dielectric
                ),
            }
        weights = {
            # volume of a sphere; the overall scale is arbitrary, only relative weights matter
            "steric": (4.0 / 3.0) * np.pi * atoms.radii**3,
            "electrostatic": atoms.charges,
            "hydrophobic": atoms.logp,
            "donor": atoms.donor,
            "acceptor": atoms.acceptor,
        }
        return {name: gaussian_field(r, weights[name], s.attenuation) for name in self.block_names}

    # ---------- many molecules ----------

    def compute_matrix(self, ids: Sequence[str], atom_sets: Sequence[AtomData]) -> FieldMatrix:
        if len(ids) != len(atom_sets):
            raise FieldError(f"{len(ids)} ids but {len(atom_sets)} molecules")
        if not atom_sets:
            raise FieldError("no molecules to compute fields for")
        rows, outside = [], []
        for atoms in atom_sets:
            blocks = self.compute(atoms)
            rows.append(np.concatenate([blocks[name] for name in self.block_names]))
            outside.append(not bool(self.grid.contains(atoms.coords).all()))
        return FieldMatrix(
            ids=tuple(ids),
            block_names=self.block_names,
            n_points=self.grid.n_points,
            values=np.vstack(rows),
            outside_grid=np.array(outside),
        )
