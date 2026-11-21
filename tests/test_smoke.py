import pytest
from rdkit import Chem

import molecure


def test_package_imports():
    assert molecure.__version__

def test_rdkit_works():
    mol = Chem.MolFromSmiles("CCO")
    assert mol.GetNumAtoms() == 3

def test_molscrub_optional():
    molscrub = pytest.importorskip("molscrub")
    assert hasattr(molscrub, "Scrub")

def test_meeko_optional():
    meeko = pytest.importorskip("meeko")
    assert hasattr(meeko, "MoleculePreparation")