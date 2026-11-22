"""RDKit layer: molecules -> AtomData -> fields.

These tests need RDKit, so they run on your machine.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from molecure.config.fields import ComfaFields, GridConfig
from molecure.fields.calculator import FieldCalculator
from molecure.fields.errors import FieldError
from molecure.fields.kernels import UFF_VDW
from molecure.fields.molprops import atom_data_from_mol

MODELS = ["mmff94", "gasteiger"]


def embedded(smiles: str, add_hs: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if add_hs:
        mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def test_atom_data_has_consistent_shapes():
    data = atom_data_from_mol(embedded("CCO"))
    assert data.n_atoms == 9  # 3 heavy atoms + 6 hydrogens
    assert data.coords.shape == (9, 3)


@pytest.mark.parametrize("model", MODELS)
def test_neutral_molecule_has_charges_summing_to_zero(model):
    data = atom_data_from_mol(embedded("CCO"), model)
    assert data.charges.sum() == pytest.approx(0.0, abs=1e-3)
    assert data.charges[2] < -0.2  # the oxygen is clearly negative


@pytest.mark.parametrize("model", MODELS)
def test_protonated_amine_sums_to_plus_one(model):
    data = atom_data_from_mol(embedded("C[NH3+]"), model)
    assert data.charges.sum() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("model", MODELS)
def test_zwitterion_sums_to_zero(model):
    # glycine as a zwitterion: the situation our amino-acid-like screening compounds are in
    data = atom_data_from_mol(embedded("[NH3+]CC(=O)[O-]"), model)
    assert data.charges.sum() == pytest.approx(0.0, abs=1e-2)


def test_hbond_flags_for_ethanol():
    data = atom_data_from_mol(embedded("CCO"))
    assert data.donor[2] == 1.0 and data.acceptor[2] == 1.0  # the hydroxyl oxygen
    assert data.donor[[0, 1]].sum() == 0 and data.acceptor[[0, 1]].sum() == 0  # the carbons


def test_hbond_flags_for_acetone():
    data = atom_data_from_mol(embedded("CC(C)=O"))
    assert data.acceptor[3] == 1.0  # the carbonyl oxygen
    assert data.donor.sum() == 0  # nothing to donate


def test_radii_and_hydrophobicity_are_sensible():
    data = atom_data_from_mol(embedded("CCO"))
    assert np.all((data.radii > 1.0) & (data.radii < 2.5))
    assert np.all(np.isfinite(data.logp))


# Same SMILES as a pair of identical atoms, so UFF returns that element's own parameters.
PAIRS = {"H": "CC", "C": "CC", "N": "NN", "O": "OO", "F": "FF", "P": "PP", "S": "SS",
         "Cl": "ClCl", "Br": "BrBr", "I": "II"}  # fmt: skip


@pytest.mark.parametrize("symbol", sorted(UFF_VDW))
def test_uff_table_agrees_with_rdkits_own_uff(symbol):
    get_params = getattr(rdForceFieldHelpers, "GetUFFVdWParams", None)
    if get_params is None:
        pytest.skip("this RDKit version has no GetUFFVdWParams")
    mol = Chem.AddHs(Chem.MolFromSmiles(PAIRS[symbol]))
    first, second = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol][:2]
    x, d = get_params(mol, first, second)
    assert x == pytest.approx(UFF_VDW[symbol][0], rel=2e-3)
    assert d == pytest.approx(UFF_VDW[symbol][1], rel=2e-3)


def test_missing_3d_is_an_error():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # never embedded
    with pytest.raises(FieldError, match="3D"):
        atom_data_from_mol(mol)


def test_implicit_hydrogens_are_an_error():
    with pytest.raises(FieldError, match="implicit hydrogens"):
        atom_data_from_mol(embedded("CCO", add_hs=False))


def test_unsupported_element_is_named():
    with pytest.raises(FieldError, match="'B'"):
        atom_data_from_mol(embedded("OB(O)O"))  # boric acid


def test_xtb_charges_are_not_implemented_yet():
    with pytest.raises(FieldError, match="not implemented"):
        atom_data_from_mol(embedded("CCO"), "xtb")


def test_unknown_charge_model_is_an_error():
    with pytest.raises(FieldError, match="unknown charge model"):
        atom_data_from_mol(embedded("CCO"), "magic")


def test_real_molecules_through_the_whole_field_pipeline():
    # Not aligned to each other (that is Stage 8); this only checks that everything connects.
    atom_sets = [atom_data_from_mol(embedded(s)) for s in ("CCO", "CCCO")]
    calc = FieldCalculator.fit(atom_sets, ComfaFields(method="comfa", grid=GridConfig(spacing=2.0)))
    matrix = calc.compute_matrix(["ethanol", "propanol"], atom_sets)
    assert matrix.values.shape == (2, 2 * calc.grid.n_points)
    assert np.isfinite(matrix.values).all()
    assert not matrix.outside_grid.any()
