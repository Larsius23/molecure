"""Atom mapping by maximum common substructure, for alignment quality control."""

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .rmsd import min_rmsd_over_matches


def mcs_core_rmsd(ref: Chem.Mol, mol: Chem.Mol, min_atoms: int) -> tuple[float | None, int]:
    """(core RMSD in angstrom, number of core atoms) of a pose against a reference pose.

    The core is the maximum common substructure of the two molecules' heavy atoms (elements and
    ring topology must agree; bond orders may differ, so an aromatic and a saturated ring still
    match). The RMSD is None when the core has fewer than `min_atoms` atoms: such a molecule
    cannot be judged against this reference. Neither molecule is moved.
    """
    ref_heavy, mol_heavy = Chem.RemoveHs(ref), Chem.RemoveHs(mol)
    result = rdFMCS.FindMCS(
        [ref_heavy, mol_heavy],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    n_core = int(result.numAtoms)
    if n_core < min_atoms:
        return None, n_core
    pattern = Chem.MolFromSmarts(result.smartsString)
    ref_match = ref_heavy.GetSubstructMatch(pattern)
    mol_matches = mol_heavy.GetSubstructMatches(pattern, uniquify=False, maxMatches=1000)
    if not ref_match or not mol_matches:
        return None, n_core
    rmsd = min_rmsd_over_matches(
        ref_heavy.GetConformer().GetPositions(),
        mol_heavy.GetConformer().GetPositions(),
        ref_match,
        mol_matches,
    )
    return rmsd, n_core
