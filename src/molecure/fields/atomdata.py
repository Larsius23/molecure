"""Per-atom numbers needed by the field formulas, as plain numpy arrays.

This is the boundary between chemistry and maths: molprops.py (RDKit) fills an
AtomData from a molecule; calculator.py (numpy) consumes it. Everything is in
the SAME order as the atoms of the molecule.
"""

from dataclasses import dataclass, replace

import numpy as np

from .errors import FieldError


@dataclass(frozen=True, eq=False)
class AtomData:
    coords: np.ndarray  # (n, 3) angstrom, already in the common alignment frame
    charges: np.ndarray  # partial charges, in units of e
    vdw_x: np.ndarray  # Lennard-Jones x (angstrom), CoMFA steric
    vdw_d: np.ndarray  # Lennard-Jones D (kcal/mol),  CoMFA steric
    radii: np.ndarray  # van der Waals radius (angstrom), CoMSIA steric
    logp: np.ndarray  # atomic hydrophobicity (Crippen logP contribution), CoMSIA
    donor: np.ndarray  # 1.0 where the atom is an H-bond donor, else 0.0
    acceptor: np.ndarray  # 1.0 where the atom is an H-bond acceptor, else 0.0

    def __post_init__(self) -> None:
        n = len(self.coords)
        if self.coords.shape != (n, 3) or n == 0:
            raise FieldError(f"coords must have shape (n, 3) with n > 0, got {self.coords.shape}")
        for name in ("charges", "vdw_x", "vdw_d", "radii", "logp", "donor", "acceptor"):
            values = getattr(self, name)
            if values.shape != (n,):
                raise FieldError(f"{name} has shape {values.shape}, expected ({n},)")
            if not np.all(np.isfinite(values)):
                raise FieldError(f"{name} contains values that are not finite numbers")
        if not np.all(np.isfinite(self.coords)):
            raise FieldError("coords contain values that are not finite numbers")

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    def translated(self, offset) -> "AtomData":
        """The same atoms moved rigidly by `offset` (used in tests and grid-shift checks)."""
        return replace(self, coords=self.coords + np.asarray(offset, dtype=float))
