"""The pipeline on real files: the steroid workspaces built by benchmarks/steroids/prepare.py."""

import csv
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from molecure.cli.main import app
from molecure.fields import FieldCalculator, Grid, atom_data_from_mol
from molecure.model.pipeline import PipelineError, fit_project, load_poses
from molecure.project import Workspace


def copy_workspace(source_root, name, tmp_path) -> Workspace:
    """Work on a copy, so the shared workspaces never collect run folders."""
    shutil.copytree(source_root / name, tmp_path / name)
    return Workspace.open(tmp_path / name)


def read_rows(run_root) -> list[dict]:
    with open(run_root / "predictions.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_tbg_runs_end_to_end(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    fit = fit_project(ws, ws.load())
    for name in ("results.json", "predictions.csv", "split.csv", "model.npz"):
        assert (fit.run.root / name).is_file(), name
    summary = json.loads((fit.run.root / "results.json").read_text())
    assert summary["compounds"] == {"train": 21, "test": 0}
    assert "external" not in summary
    assert summary["components"] == {"used": 2, "rule": "fixed", "max": 8}
    assert fit.run.read_manifest()["model"]["n_components"] == 2


def test_cbg_follows_the_split_file_of_the_paper(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_cbg", tmp_path)
    fit = fit_project(ws, ws.load())
    summary = json.loads((fit.run.root / "results.json").read_text())
    assert summary["compounds"] == {"train": 21, "test": 10}
    assert summary["external"]["n"] == 10
    rows = read_rows(fit.run.root)
    assert len(rows) == 31
    assert all(r["compound_id"].startswith("T") == (r["subset"] == "test") for r in rows)
    assert all(r["predicted"] != "" for r in rows if r["subset"] == "test")
    assert all(r["fitted"] != "" and r["cv_loo"] != "" for r in rows if r["subset"] == "train")


def test_the_saved_model_can_be_reapplied_to_the_molecules(steroid_workspaces, tmp_path):
    # This is what screening will do later: saved grid + saved coefficients on fresh field atoms.
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    cfg = ws.load()
    fit = fit_project(ws, cfg)
    saved = np.load(fit.run.root / "model.npz")
    grid = Grid(
        origin=tuple(float(v) for v in saved["grid_origin"]),
        spacing=float(saved["grid_spacing"]),
        shape=tuple(int(v) for v in saved["grid_shape"]),
    )
    rows = read_rows(fit.run.root)
    poses = load_poses(cfg.alignment, ws.root)
    ids = [r["compound_id"] for r in rows]
    atoms = [atom_data_from_mol(poses[i], cfg.chemistry.charge_model) for i in ids]
    X = FieldCalculator(grid, cfg.molecular_fields).compute_matrix(ids, atoms).values
    predicted = X @ saved["coefficients"] + float(saved["intercept"])
    fitted = [float(r["fitted"]) for r in rows]
    assert np.allclose(predicted, fitted, atol=1e-3)  # the CSV keeps four decimals


def test_an_alignment_mode_that_is_not_built_yet_is_a_clear_error(demo_project):
    with pytest.raises(PipelineError, match="Stage 8"):
        fit_project(demo_project, demo_project.load())


def test_missing_poses_are_reported_before_any_run_folder_exists(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    source = (ws.root / "data/curated/steroids_tbg.sdf").read_text()
    few = "$$$$\n".join(source.split("$$$$\n")[:5]) + "$$$$\n"  # the first five records only
    (ws.root / "data/curated/few.sdf").write_text(few)
    text = (ws.root / "project.yaml").read_text()
    old = "  mode: prealigned\n  file: data/curated/steroids_tbg.sdf"
    assert old in text
    (ws.root / "project.yaml").write_text(
        text.replace(old, "  mode: prealigned\n  file: data/curated/few.sdf")
    )
    with pytest.raises(PipelineError, match="no pose for 16"):
        fit_project(ws, ws.load())
    assert list((ws.root / "runs").iterdir()) == []


def test_pose_rules_pick_the_right_pose(write_sdf, tmp_path):
    write_sdf(
        tmp_path / "poses.sdf",
        [
            ("CCO", {"cid": "a", "score": -5.0}),
            ("CCO", {"cid": "a", "score": -7.5}),
            ("CCN", {"cid": "b", "score": -6.0}),
        ],
    )

    def pose_score(rule, better="lower") -> float:
        alignment = SimpleNamespace(
            file="poses.sdf", id_property="cid", pose_rule=rule,
            score_property="score", score_better=better,
        )  # fmt: skip
        return float(load_poses(alignment, tmp_path)["a"].GetProp("score"))

    assert pose_score("first") == -5.0
    assert pose_score("best_score", "lower") == -7.5  # Vina: lower is better
    assert pose_score("best_score", "higher") == -5.0


def test_cli_fit_prints_a_summary(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_cbg", tmp_path)
    result = CliRunner().invoke(app, ["fit", str(ws.root)])
    assert result.exit_code == 0, result.output
    assert "leave-one-out" in result.output and "external (10)" in result.output
    assert "run folder" in result.output


def test_cli_fit_reports_problems_cleanly(demo_project):
    result = CliRunner().invoke(app, ["fit", str(demo_project.root)])
    assert result.exit_code == 1
    assert "Stage 8" in result.output


def test_cli_sensitivity_prints_both_tables(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    result = CliRunner().invoke(app, ["sensitivity", str(ws.root)])
    assert result.exit_code == 0, result.output
    assert "Lattice offsets" in result.output and "range" in result.output
    assert "constant dielectric" in result.output and "grid spacing 1 A" in result.output
    assert list((ws.root / "runs").iterdir()) == []  # nothing is saved


def restrict_poses(ws, name, drop_last):
    """Point the pose file at a copy that lacks the last `drop_last` records."""
    source = (ws.root / f"data/curated/{name}.sdf").read_text()
    records = [r for r in source.split("$$$$\n") if r.strip()]
    kept = "$$$$\n".join(records[: len(records) - drop_last]) + "$$$$\n"
    (ws.root / "data/curated/few.sdf").write_text(kept)
    edit_yaml(
        ws,
        f"  mode: prealigned\n  file: data/curated/{name}.sdf",
        "  mode: prealigned\n  file: data/curated/few.sdf",
    )


def edit_yaml(ws, old, new):
    text = (ws.root / "project.yaml").read_text()
    assert old in text, old
    (ws.root / "project.yaml").write_text(text.replace(old, new))


def append_yaml(ws, block):
    path = ws.root / "project.yaml"
    path.write_text(path.read_text() + "\n" + block)


def test_skipping_invalid_compounds_records_exactly_what_was_dropped(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    restrict_poses(ws, "steroids_tbg", drop_last=2)  # 2 of 21 = 9.5%, under the default 10% cap
    append_yaml(ws, "qc:\n  on_invalid: skip\n")
    fit = fit_project(ws, ws.load())
    summary = json.loads((fit.run.root / "results.json").read_text())
    assert summary["compounds"] == {"train": 19, "test": 0}
    assert [e["stage"] for e in summary["excluded"]] == ["pose", "pose"]
    assert any("skipped" in note for note in summary["warnings"])
    lines = (fit.run.root / "excluded.csv").read_text().splitlines()
    assert lines[0] == "compound_id,stage,reason" and len(lines) == 3
    assert fit.run.read_manifest()["model"]["n_excluded"] == 2


def test_skipping_has_a_cap_so_a_bad_file_cannot_shrink_the_data(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    restrict_poses(ws, "steroids_tbg", drop_last=5)  # 5 of 21 = 24%
    append_yaml(ws, "qc:\n  on_invalid: skip\n")
    with pytest.raises(PipelineError, match="max_skipped_fraction"):
        fit_project(ws, ws.load())
    assert list((ws.root / "runs").iterdir()) == []


def test_the_alignment_check_reports_the_core_rmsd_of_every_steroid(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    edit_yaml(ws, "  mode: prealigned\n", "  mode: prealigned\n  reference: S11_11_deoxycortisol\n")
    fit = fit_project(ws, ws.load())
    rows = read_alignment_qc(fit.run.root)
    assert len(rows) == 21
    assert float(rows["S11_11_deoxycortisol"]["core_rmsd"]) == pytest.approx(0.0, abs=1e-6)
    assert all(int(r["core_atoms"]) >= 8 for r in rows.values())
    assert max(float(r["core_rmsd"]) for r in rows.values()) < 1.5  # Cramer's six-atom fit
    summary = json.loads((fit.run.root / "results.json").read_text())
    assert summary["alignment_qc"]["worst"] in rows and summary["excluded"] == []


def test_the_alignment_gate_stops_compounds_that_are_too_far_off(steroid_workspaces, tmp_path):
    ws = copy_workspace(steroid_workspaces, "steroids_tbg", tmp_path)
    edit_yaml(ws, "  mode: prealigned\n", "  mode: prealigned\n  reference: S11_11_deoxycortisol\n")
    append_yaml(ws, "qc:\n  max_core_rmsd: 0.05\n")  # tighter than any real alignment
    with pytest.raises(PipelineError, match="alignment check failed for"):
        fit_project(ws, ws.load())
    assert list((ws.root / "runs").iterdir()) == []


def test_several_poses_with_the_first_rule_leave_a_note(write_sdf, tmp_path):
    write_sdf(
        tmp_path / "poses.sdf",
        [("CCO", {"cid": "a"}), ("CCO", {"cid": "a"}), ("CCN", {"cid": "b"})],
    )
    alignment = SimpleNamespace(
        file="poses.sdf", id_property="cid", pose_rule="first",
        score_property=None, score_better="lower",
    )  # fmt: skip
    notes: list[str] = []
    load_poses(alignment, tmp_path, notes=notes)
    assert len(notes) == 1 and "several poses" in notes[0] and "a" in notes[0]


def read_alignment_qc(run_root) -> dict[str, dict]:
    with open(run_root / "alignment_qc.csv", newline="", encoding="utf-8") as handle:
        return {row["compound_id"]: row for row in csv.DictReader(handle)}
