"""Shared test helpers (pytest fixtures)."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molecure.project import Workspace


@pytest.fixture
def write_sdf():
    """write_sdf(path, [(smiles, {tag: value}), ...], three_d=True) writes a small SDF."""

    def _write(path, records, three_d=True):
        writer = Chem.SDWriter(str(path))
        for smiles, props in records:
            mol = Chem.MolFromSmiles(smiles)
            if three_d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=7)
            else:
                AllChem.Compute2DCoords(mol)  # flat: every z is 0
            for key, value in props.items():
                mol.SetProp(key, str(value))
            writer.write(mol)
        writer.close()

    return _write


@pytest.fixture
def demo_project(tmp_path, write_sdf):
    """A workspace made by `init`, filled with small valid data files."""
    ws = Workspace.create(tmp_path / "demo")
    (ws.root / "data/curated/calibration.csv").write_text(
        "compound_id,smiles,pIC50\nc1,CCO,5.5\nc2,CCCO,6.0\nc3,CCCCO,6.8\n"
    )
    write_sdf(ws.root / "data/reference/reference.sdf", [("CCCO", {})])
    return ws
