"""The 3D lattice on which every field is evaluated.

A grid is just three numbers: where it starts (origin), how far apart the points
are (spacing), and how many points it has along x, y and z (shape).

Rule: a grid is built from CALIBRATION molecules only (see FieldCalculator.fit).
Screening molecules are evaluated on that same grid; they never move it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    origin: tuple[float, float, float]  # coordinates of the corner point, angstrom
    spacing: float  # angstrom
    shape: tuple[int, int, int]  # number of points along x, y, z

    @property
    def n_points(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def upper(self) -> np.ndarray:
        """Coordinates of the opposite corner."""
        return np.asarray(self.origin) + (np.asarray(self.shape) - 1) * self.spacing

    def points(self) -> np.ndarray:
        """All grid points as an (n_points, 3) array. x varies slowest, z fastest."""
        axes = [np.arange(n) for n in self.shape]
        ix, iy, iz = np.meshgrid(*axes, indexing="ij")
        index = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
        return np.asarray(self.origin) + index * self.spacing

    def contains(self, coords: np.ndarray) -> np.ndarray:
        """For each atom coordinate: is it inside the grid box? Returns booleans."""
        coords = np.asarray(coords)
        inside = (coords >= np.asarray(self.origin)) & (coords <= self.upper)
        return inside.all(axis=1)

    def shifted(self, offset: Sequence[float]) -> "Grid":
        """The same grid moved by `offset`. Used later to test sensitivity to grid position."""
        new_origin = tuple(float(o + d) for o, d in zip(self.origin, offset, strict=True))
        return Grid(origin=new_origin, spacing=self.spacing, shape=self.shape)

    @classmethod
    def from_coords(
        cls, coord_sets: Sequence[np.ndarray], spacing: float, padding: float
    ) -> "Grid":
        """Smallest regular grid that encloses every atom of every molecule, plus `padding`."""
        if not coord_sets:
            raise ValueError("cannot build a grid from zero molecules")
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        everything = np.vstack([np.asarray(c, dtype=float) for c in coord_sets])
        lower = everything.min(axis=0) - padding
        upper = everything.max(axis=0) + padding
        # The 1e-9 stops floating-point noise from adding a whole extra plane of points.
        n = np.ceil((upper - lower) / spacing - 1e-9).astype(int) + 1
        return cls(
            origin=tuple(float(v) for v in lower),
            spacing=float(spacing),
            shape=tuple(int(v) for v in n),
        )
