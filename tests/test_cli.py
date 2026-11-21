from typer.testing import CliRunner

from molecure import __version__
from molecure.cli.main import app
from molecure.project import Workspace

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_init_then_validate_flow(tmp_path):
    folder = tmp_path / "demo"
    result = runner.invoke(app, ["init", str(folder)])
    assert result.exit_code == 0
    assert (folder / "project.yaml").is_file()

    # Fresh project: the config is fine, but the data files are not there yet.
    result = runner.invoke(app, ["validate", str(folder)])
    assert result.exit_code == 2
    assert "config: OK" in result.output
    assert "missing" in result.output

    ws = Workspace.open(folder)
    for _label, path in ws.missing_files(ws.load()):
        path.touch()
    result = runner.invoke(app, ["validate", str(folder)])
    assert result.exit_code == 0


def test_init_refuses_to_overwrite(tmp_path):
    folder = tmp_path / "demo"
    runner.invoke(app, ["init", str(folder)])
    result = runner.invoke(app, ["init", str(folder)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_validate_reports_where_the_problem_is(tmp_path):
    folder = tmp_path / "demo"
    runner.invoke(app, ["init", str(folder)])
    path = folder / "project.yaml"
    path.write_text(path.read_text().replace("    activity_col: pIC50\n", ""))
    result = runner.invoke(app, ["validate", str(folder)])
    assert result.exit_code == 1
    assert "activity_col" in result.output


def test_validate_without_project_yaml(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 1
    assert "project.yaml" in result.output
