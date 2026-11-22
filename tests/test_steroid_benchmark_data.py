"""The steroid benchmark data must agree with the published tables.

Runs benchmarks/steroids/prepare.py into a temporary folder, then loads the result
with the real config and loaders.
"""

import math
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

# Coats 1998, Table 1: measured affinity constants K x 1e6 (M^-1) of the 21 training steroids.
# None = "< 0.1": no binding could be measured; the dataset lists these as log K = 5.
CBG_K = [
    1.9, None, None, 0.58, 0.41, 76, 76, 7.8, None, 45, 76,
    0.83, None, None, None, 0.18, 0.18, None, 24, 55, 5.3,
]  # fmt: skip
TEBG_K = [
    0.21, 1300, 1500, 29, 14, 2.2, 1.6, 2.7, 66, 24, 16,
    5500, 680, 4.3, 150, 1.4, 14, 2.3, 8.8, 9.9, 1600,
]  # fmt: skip
# The ten external CBG values (log K): Coats Table 2 and Cramer Tables III-V agree.
EXTERNAL = [7.512, 7.553, 6.779, 7.200, 6.144, 6.247, 7.120, 6.817, 7.688, 5.797]

# Compounds whose stereochemistry is known to be wrong or undefined in the source (see README).
KNOWN_STEREO_PROBLEMS = {"S02"}


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


def expected_log_k(k):
    return 5.0 if k is None else math.log10(k * 1e6)


def stereo_problems(compound) -> list[str]:
    """Centres the SMILES tag leaves undefined, or a 3D structure that disagrees with the tag."""
    problems = []
    tag = Chem.MolFromSmiles(compound.mol.GetProp("isomericSMILES"))
    if any(label == "?" for _, label in Chem.FindMolChiralCenters(tag, includeUnassigned=True)):
        problems.append("undefined stereocentre in the SMILES tag")
    if Chem.MolToSmiles(Chem.RemoveHs(compound.mol)) != Chem.MolToSmiles(tag):
        problems.append("3D structure disagrees with the SMILES tag")
    return problems


def test_cbg_training_values_match_the_raw_affinities(workspaces):
    values = by_code(load_set(workspaces, "steroids_cbg"))
    for n, k in enumerate(CBG_K, start=1):
        assert values[f"S{n:02d}"] == pytest.approx(expected_log_k(k), abs=0.0015), n


def test_tbg_training_values_match_the_raw_affinities(workspaces):
    values = by_code(load_set(workspaces, "steroids_tbg"))
    assert len(values) == 21
    for n, k in enumerate(TEBG_K, start=1):
        assert values[f"S{n:02d}"] == pytest.approx(expected_log_k(k), abs=0.0015), n


def test_external_values_match_the_papers(workspaces):
    values = by_code(load_set(workspaces, "steroids_cbg"))
    for n, actual in enumerate(EXTERNAL, start=1):
        assert values[f"T{n:02d}"] == pytest.approx(actual, abs=0.0015), n


def test_corrected_values_keep_their_source_value(workspaces):
    compounds = {c.id.split("_")[0]: c for c in load_set(workspaces, "steroids_cbg").compounds}
    for code in ("S16", "S17"):
        assert compounds[code].activity == pytest.approx(5.255, abs=0.0005)
        source = float(compounds[code].mol.GetProp("source_activity"))
        assert source == pytest.approx(-5.225, abs=0.001)  # what the source file said


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


def test_stereochemistry_problems_are_exactly_the_known_ones(workspaces):
    dataset = load_set(workspaces, "steroids_cbg")
    flagged = {c.id.split("_")[0] for c in dataset.compounds if stereo_problems(c)}
    assert flagged == KNOWN_STEREO_PROBLEMS


def test_both_workspaces_validate_with_the_cli(workspaces):
    for name in ("steroids_cbg", "steroids_tbg"):
        result = CliRunner().invoke(app, ["validate", str(workspaces / name)])
        assert result.exit_code == 0, result.output


def test_exclude_option_leaves_the_compound_out(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--exclude", "S02"], check=True)
    cbg = load_set(tmp_path, "steroids_cbg_no_s02")
    tbg = load_set(tmp_path, "steroids_tbg_no_s02")
    assert len(cbg.compounds) == 30 and len(tbg.compounds) == 20
    assert not any(c.id.startswith("S02") for c in cbg.compounds + tbg.compounds)


def test_the_raw_file_is_refused_by_the_p_scale_guard():
    # The source stores negative log(1/K) values. Loaded as-is they must be refused.
    cfg = CalibrationDataset(
        id="raw",
        role="calibration",
        file=str(SOURCE),
        id_col="name",
        activity_col="pKi_CBG",
        endpoint="pKd",
    )
    with pytest.raises(DataError, match="outside 0.0 to 14.0"):
        load_dataset(cfg, REPO)
