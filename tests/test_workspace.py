import pytest

from molecure.config import load_config
from molecure.project import Workspace, WorkspaceError


def test_create_makes_folders_and_a_loadable_config(tmp_path):
    ws = Workspace.create(tmp_path / "demo")
    for sub in ("data/raw", "data/curated", "data/screening", "data/reference", "splits", "runs"):
        assert (ws.root / sub).is_dir()
    assert load_config(ws.config_path).project.name == "demo"


def test_create_refuses_to_overwrite(tmp_path):
    Workspace.create(tmp_path / "demo")
    with pytest.raises(WorkspaceError, match="already exists"):
        Workspace.create(tmp_path / "demo")


def test_odd_folder_name_still_gives_a_valid_project_name(tmp_path):
    ws = Workspace.create(tmp_path / "my project!")
    assert load_config(ws.config_path).project.name == "my_project_"


def test_open_accepts_folder_or_file(tmp_path):
    ws = Workspace.create(tmp_path / "demo")
    assert Workspace.open(ws.root).root == ws.root
    assert Workspace.open(ws.config_path).root == ws.root


def test_open_fails_without_project_yaml(tmp_path):
    with pytest.raises(WorkspaceError, match="project.yaml"):
        Workspace.open(tmp_path)


def test_missing_files_are_found_then_resolved(tmp_path):
    ws = Workspace.create(tmp_path / "demo")
    cfg = ws.load()
    assert len(ws.missing_files(cfg)) == 2  # calibration csv + reference sdf
    for _label, path in ws.missing_files(cfg):
        path.touch()
    assert ws.missing_files(cfg) == []
