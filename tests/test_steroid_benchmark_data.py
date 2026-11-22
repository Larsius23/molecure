"""The steroid benchmark data must agree with the published tables.

Runs benchmarks/steroids/prepare.py into a temporary folder, then loads the result
with the real config and loaders.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typer.testing import CliRunner

from molecure.cli.main import app
from molecure.config import load_config
from molecure.config.datasets import CalibrationDataset
from molecure.data import DataError, load_all, load_dataset

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "steroids" / "prepare.py"
SOURCE = REPO / "benchmarks" / "steroids" / "source" / "Coats_CoMFA_steroidsBenchmarkDataset.sdf"

# Cramer et al. 1988, Table I, in the paper's own sign convention: (TeBG, CBG)
TABLE_I = [
    (-5.322, -6.279), (-9.114, -5.000), (-9.176, -5.000), (-7.462, -5.763), (-7.146, -5.613),
    (-6.342, -7.881), (-6.204, -7.881), (-6.431, -6.892), (-7.819, -5.000), (-7.380, -7.653),
    (-7.204, -7.881), (-9.740, -5.919), (-8.833, -5.000), (-6.633, -5.000), (-8.176, -5.000),
    (-6.146, -5.255), (-7.146, -5.255), (-6.362, -5.000), (-6.944, -7.380), (-6.996, -7.740),
    (-9.204, -6.724),
]  # fmt: skip
# Tables III-V, row "actual": the ten external CBG values (compounds 1-10 of Figure 3)
EXTERNAL = [-7.512, -7.553, -6.779, -7.200, -6.144, -6.247, -7.120, -6.817, -7.688, -5.797]
# Known difference between the source file and Table I (see the README):
# file says 5.225, paper 5.255
KNOWN_CBG_DIFFERENCES = {"S16", "S17"}


@pytest.fixture(scope="module")
def workspaces(tmp_path_factory):
    out = tmp_path_factory.mktemp("steroids")
    subprocess.run([sys.executable, str(SCRIPT), str(out)], check=True)
    return out


def load_set(workspaces: Path, name: str):
    root = workspaces / name
    return load_all(load_config(root / "project.yaml"), root)["steroids"]


def by_code(dataset) -> dict[str, float]:
    """{'S01': activity, ..., 'T10': activity}"""
    return {c.id.split("_")[0]: c.activity for c in dataset.compounds}


def test_cbg_training_values_match_table_i(workspaces):
    values = by_code(load_set(workspaces, "steroids_cbg"))
    for n, (_, cbg) in enumerate(TABLE_I, start=1):
        code = f"S{n:02d}"
        expected = -cbg - 0.03 if code in KNOWN_CBG_DIFFERENCES else -cbg
        assert values[code] == pytest.approx(expected, abs=0.0015), code


def test_tbg_training_values_match_table_i(workspaces):
    values = by_code(load_set(workspaces, "steroids_tbg"))
    assert len(values) == 21
    for n, (tebg, _) in enumerate(TABLE_I, start=1):
        assert values[f"S{n:02d}"] == pytest.approx(-tebg, abs=0.0015), n


def test_external_values_match_the_paper(workspaces):
    values = by_code(load_set(workspaces, "steroids_cbg"))
    for n, actual in enumerate(EXTERNAL, start=1):
        assert values[f"T{n:02d}"] == pytest.approx(-actual, abs=0.0015), n


def test_split_file_matches_the_paper(workspaces):
    rows = (workspaces / "steroids_cbg/splits/paper_split.csv").read_text().splitlines()[1:]
    subsets = [row.split(",")[1] for row in rows]
    assert subsets.count("train") == 21 and subsets.count("test") == 10
    assert all(row.startswith("T") for row in rows if row.endswith(",test"))


def test_every_molecule_is_3d_hydrogenated_and_neutral(workspaces):
    for c in load_set(workspaces, "steroids_cbg").compounds:
        assert c.mol.GetNumConformers() == 1, c.id
        assert any(a.GetAtomicNum() == 1 for a in c.mol.GetAtoms()), c.id
        assert Chem.GetFormalCharge(c.mol) == 0, c.id


def test_the_bad_source_record_is_gone(workspaces):
    dataset = load_set(workspaces, "steroids_cbg")
    acetate = next(c for c in dataset.compounds if c.id.startswith("T02"))
    assert rdMolDescriptors.CalcMolFormula(acetate.mol) == "C23H32O6"


def test_both_workspaces_validate_with_the_cli(workspaces):
    for name in ("steroids_cbg", "steroids_tbg"):
        result = CliRunner().invoke(app, ["validate", str(workspaces / name)])
        assert result.exit_code == 0, result.output


def test_the_raw_file_is_refused_by_the_p_scale_guard():
    # The source stores negative log-K values. Loaded as-is they must be refused, not silently used.
    cfg = CalibrationDataset(
        id="raw",
        role="calibration",
        file=str(SOURCE),
        id_col="name",
        activity_col="pKi_CBG",
        endpoint="pKi",
    )
    with pytest.raises(DataError, match="outside 0.0 to 14.0"):
        load_dataset(cfg, REPO)
