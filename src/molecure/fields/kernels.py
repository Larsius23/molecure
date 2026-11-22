"""The physics: three formulas that turn atoms into numbers at grid points.

All functions here are plain numpy. They know nothing about RDKit or config files,
which makes them easy to test against hand-calculated answers.

Shape convention: `r` is a distance table of shape (n_points, n_atoms);
each atom-level array has shape (n_atoms,); results have shape (n_points,).
"""

import numpy as np
from scipy.spatial.distance import cdist

# Coulomb's constant in kcal * angstrom / (mol * e^2): energy of two unit charges 1 A apart.
COULOMB_CONSTANT = 332.0637

# Distances below this are treated as this value. Inside an atom the energies are far
# beyond the truncation cutoff anyway; this just avoids dividing by zero.
MIN_DISTANCE = 0.1

# Lennard-Jones parameters from the UFF force field (Rappe et al. 1992):
#   element -> (x, D)   x = distance of minimum energy (A),  D = well depth (kcal/mol)
# UFF uses the same vdW values for every hybridization of an element, so an
# element-level table is enough. A test cross-checks this table against RDKit's own UFF.
UFF_VDW: dict[str, tuple[float, float]] = {
    "H": (2.886, 0.044),
    "C": (3.851, 0.105),
    "N": (3.660, 0.069),
    "O": (3.500, 0.060),
    "F": (3.364, 0.050),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "Cl": (3.947, 0.227),
    "Br": (4.189, 0.251),
    "I": (4.500, 0.339),
}


def distances(points: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Distance from every grid point to every atom: shape (n_points, n_atoms)."""
    return cdist(points, coords)


def steric_lj(
    r: np.ndarray,
    atom_x: np.ndarray,
    atom_d: np.ndarray,
    probe_x: float,
    probe_d: float,
    cutoff: float,
) -> np.ndarray:
    """CoMFA steric field: Lennard-Jones 6-12 energy between a probe and the molecule.

        E = sum over atoms of  D_ij * [ (x_ij / r)^12 - 2 * (x_ij / r)^6 ]

    with the usual combining rule x_ij = sqrt(x_i * x_j), D_ij = sqrt(D_i * D_j).
    The minimum is -D_ij at r = x_ij. The energy shoots up steeply inside the molecule,
    so the SUM is truncated at +cutoff (kcal/mol).
    """
    x_ij = np.sqrt(atom_x * probe_x)
    d_ij = np.sqrt(atom_d * probe_d)
    ratio6 = (x_ij / np.maximum(r, MIN_DISTANCE)) ** 6
    energy = (d_ij * (ratio6 * ratio6 - 2.0 * ratio6)).sum(axis=1)
    return np.minimum(energy, cutoff)


def electrostatic_coulomb(
    r: np.ndarray,
    atom_q: np.ndarray,
    probe_q: float,
    cutoff: float,
    dielectric: str,
) -> np.ndarray:
    """CoMFA electrostatic field: Coulomb energy between a charged probe and the molecule.

        constant dielectric:   E = 332.0637 * Q * q_i / r
        distance dielectric:   E = 332.0637 * Q * q_i / r^2      (dielectric = r)

    summed over atoms, then truncated to the range [-cutoff, +cutoff].
    """
    r = np.maximum(r, MIN_DISTANCE)
    if dielectric == "distance":
        terms = COULOMB_CONSTANT * probe_q * atom_q / r**2
    elif dielectric == "constant":
        terms = COULOMB_CONSTANT * probe_q * atom_q / r
    else:
        raise ValueError(f"dielectric must be 'distance' or 'constant', not {dielectric!r}")
    return np.clip(terms.sum(axis=1), -cutoff, cutoff)


def gaussian_field(r: np.ndarray, atom_w: np.ndarray, attenuation: float) -> np.ndarray:
    """CoMSIA similarity field (Klebe et al. 1994), with a probe weight of 1:

        A = - sum over atoms of  w_i * exp(-alpha * r^2)

    `w_i` is an atom property (volume, charge, hydrophobicity, ...). The Gaussian
    stays finite at r = 0, so CoMSIA needs no truncation.
    """
    return -(atom_w * np.exp(-attenuation * r**2)).sum(axis=1)
