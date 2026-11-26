"""Core RMSD by maximum common substructure, on small molecules with RDKit."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from molecure.align.mcs import mcs_core_rmsd


def embedded(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def moved(mol: Chem.Mol, offset=(0.0, 0.0, 0.0), swap=()) -> Chem.Mol:
    """A copy with every atom shifted by `offset`, and the positions of some atom pairs swapped."""
    copy = Chem.Mol(mol)
    xyz = np.array(copy.GetConformer().GetPositions()) + np.asarray(offset)
    for a, b in swap:
        xyz[[a, b]] = xyz[[b, a]]
    conf = copy.GetConformer()
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return copy


def test_an_identical_pose_has_zero_rmsd_over_the_whole_molecule():
    ref = embedded("CCCCO")
    rmsd, n_core = mcs_core_rmsd(ref, Chem.Mol(ref), min_atoms=3)
    assert rmsd == pytest.approx(0.0, abs=1e-6) and n_core == 5  # five heavy atoms


def test_a_shifted_pose_is_measured_not_superimposed():
    ref = embedded("CCCCO")
    rmsd, _ = mcs_core_rmsd(ref, moved(ref, offset=(1.5, 0.0, 0.0)), min_atoms=3)
    assert rmsd == pytest.approx(1.5, abs=1e-6)


def test_the_poses_themselves_are_not_moved():
    ref, mol = embedded("CCCCO"), moved(embedded("CCCCO"), offset=(2.0, 0.0, 0.0))
    before = np.array(mol.GetConformer().GetPositions())
    mcs_core_rmsd(ref, mol, min_atoms=3)
    assert np.array_equal(before, np.array(mol.GetConformer().GetPositions()))


def test_different_molecules_share_a_core_and_a_small_core_cannot_be_judged():
    ref, mol = embedded("CCO"), embedded("CCCO")
    rmsd, n_core = mcs_core_rmsd(ref, mol, min_atoms=3)
    assert n_core == 3 and rmsd is not None  # the C-C-O of ethanol sits inside propanol
    assert mcs_core_rmsd(ref, mol, min_atoms=4) == (None, 3)


def test_a_flipped_ring_is_not_mistaken_for_misalignment():
    # toluene heavy atoms: 0 = methyl, 1 = ipso, 2 and 6 = ortho, 3 and 5 = meta, 4 = para.
    # Swapping the ortho pair and the meta pair gives the same molecule, flipped about its axis.
    ref = embedded("Cc1ccccc1")
    rmsd, n_core = mcs_core_rmsd(ref, moved(ref, swap=[(2, 6), (3, 5)]), min_atoms=3)
    assert n_core == 7 and rmsd == pytest.approx(0.0, abs=1e-6)
