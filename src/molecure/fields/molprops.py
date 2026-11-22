"""Fill an AtomData from an RDKit molecule.

This is the only file in the fields package that touches RDKit. The molecule must
already be 3D, with explicit hydrogens, and in the common alignment frame.
(Making it so is the job of the 3D-build and alignment stages.)
"""

import os
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors

from .atomdata import AtomData
from .errors import FieldError
from .kernels import UFF_VDW


@lru_cache(maxsize=1)
def _feature_factory():
    """RDKit's built-in definitions of H-bond donors and acceptors (built once, then reused)."""
    return ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )


def _check_ready(mol: Chem.Mol) -> None:
    if mol.GetNumConformers() == 0:
        raise FieldError("molecule has no 3D coordinates")
    probe = Chem.Mol(mol)  # a copy, so we never alter the caller's molecule
    probe.UpdatePropertyCache(strict=False)
    if any(atom.GetNumImplicitHs() > 0 for atom in probe.GetAtoms()):
        raise FieldError(
            "molecule has implicit hydrogens; add explicit hydrogens (Chem.AddHs) "
            "before generating 3D coordinates"
        )


def _lennard_jones_parameters(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unsupported = sorted(set(symbols) - set(UFF_VDW))
    if unsupported:
        raise FieldError(
            f"no Lennard-Jones parameters for element(s) {unsupported}; "
            f"supported: {sorted(UFF_VDW)}"
        )
    x = np.array([UFF_VDW[s][0] for s in symbols])
    d = np.array([UFF_VDW[s][1] for s in symbols])
    return x, d


def _partial_charges(mol: Chem.Mol, model: str) -> np.ndarray:
    n = mol.GetNumAtoms()
    if model == "mmff94":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            raise FieldError(
                "MMFF94 cannot assign parameters to this molecule; "
                "try chemistry.charge_model: gasteiger"
            )
        charges = [props.GetMMFFPartialCharge(i) for i in range(n)]
    elif model == "gasteiger":
        work = Chem.Mol(mol)  # Gasteiger writes charges onto the molecule, so use a copy
        AllChem.ComputeGasteigerCharges(work)
        charges = [work.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(n)]
    elif model == "xtb":
        raise FieldError("charge_model 'xtb' is not implemented yet")
    else:
        raise FieldError(f"unknown charge model {model!r}")
    values = np.asarray(charges, dtype=float)
    if not np.all(np.isfinite(values)):
        raise FieldError(f"{model} produced charges that are not numbers for this molecule")
    return values


def _hbond_flags(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    donor = np.zeros(mol.GetNumAtoms())
    acceptor = np.zeros(mol.GetNumAtoms())
    for feature in _feature_factory().GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family == "Donor":
            target = donor
        elif family == "Acceptor":
            target = acceptor
        else:
            continue
        for index in feature.GetAtomIds():
            target[index] = 1.0
    return donor, acceptor


def atom_data_from_mol(mol: Chem.Mol, charge_model: str = "mmff94") -> AtomData:
    """Collect everything the field formulas need from one molecule (first conformer)."""
    _check_ready(mol)
    vdw_x, vdw_d = _lennard_jones_parameters(mol)
    charges = _partial_charges(mol, charge_model)
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    # Wildman-Crippen logP contribution of each atom, used as its hydrophobicity.
    contributions = rdMolDescriptors._CalcCrippenContribs(mol)  # one (logP, MR) pair per atom
    logp = np.array([logp_value for logp_value, _mr in contributions])
    donor, acceptor = _hbond_flags(mol)
    return AtomData(
        coords=np.array(mol.GetConformer().GetPositions(), dtype=float),
        charges=charges,
        vdw_x=vdw_x,
        vdw_d=vdw_d,
        radii=radii,
        logp=logp,
        donor=donor,
        acceptor=acceptor,
    )
