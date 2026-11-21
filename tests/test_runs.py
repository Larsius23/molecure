import hashlib

import pytest

from molecure.project import Workspace, WorkspaceError, config_hash, create_run, sha256_file


def test_sha256_matches_the_standard_test_vector(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_config_hash_ignores_formatting_but_notices_changes(demo_project):
    cfg = demo_project.load()
    path = demo_project.root / "project.yaml"
    path.write_text("# an extra comment\n" + path.read_text())
    assert config_hash(demo_project.load()) == config_hash(cfg)
    path.write_text(path.read_text().replace("seed: 42", "seed: 43"))
    assert config_hash(demo_project.load()) != config_hash(cfg)


def test_create_run_writes_config_and_manifest(demo_project):
    cfg = demo_project.load()
    run = create_run(demo_project, cfg)
    assert (run.root / "config.resolved.yaml").is_file()

    manifest = run.read_manifest()
    assert manifest["config_hash"] == config_hash(cfg)
    assert manifest["config_hash"][:8] in run.run_id
    assert manifest["seed"] == 42

    csv = demo_project.root / "data/curated/calibration.csv"
    by_path = {item["path"]: item for item in manifest["inputs"]}
    assert (
        by_path["data/curated/calibration.csv"]["sha256"]
        == hashlib.sha256(csv.read_bytes()).hexdigest()
    )


def test_two_runs_in_the_same_second_get_different_folders(demo_project):
    cfg = demo_project.load()
    first = create_run(demo_project, cfg)
    second = create_run(demo_project, cfg)
    assert first.root != second.root
    assert first.root.is_dir() and second.root.is_dir()


def test_create_run_refuses_when_inputs_are_missing(tmp_path):
    ws = Workspace.create(tmp_path / "empty")
    with pytest.raises(WorkspaceError, match="missing"):
        create_run(ws, ws.load())


def test_add_to_manifest_keeps_existing_entries(demo_project):
    run = create_run(demo_project, demo_project.load())
    run.add_to_manifest("note", {"a": 1})
    manifest = run.read_manifest()
    assert manifest["note"] == {"a": 1}
    assert "config_hash" in manifest
