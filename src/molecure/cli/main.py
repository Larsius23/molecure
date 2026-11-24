"""Command-line interface: `molecure <command>`."""

from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from molecure import __version__
from molecure.config import ProjectConfig  # NEW
from molecure.data import DataError, load_all  # NEW
from molecure.data.splits import SplitError
from molecure.fields.errors import FieldError
from molecure.model.calibrate import format_summary  # CHANGED
from molecure.model.pipeline import PipelineError, fit_project, load_calibration_inputs
from molecure.model.pls import ModelError
from molecure.project import Workspace, WorkspaceError, create_run
from molecure.validate.sensitivity import format_sweep, lattice_offsets, setting_variations, sweep

app = typer.Typer(
    help="mole.cure: target-agnostic 3D-QSAR for structure-based drug discovery.",
    no_args_is_help=True,
    add_completion=False,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"molecure {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """mole.cure command-line tool."""


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED)
    raise typer.Exit(code)


def _format_errors(err: ValidationError) -> list[str]:
    """Turn pydantic's error list into short 'where: what' lines."""
    lines = []
    for e in err.errors():
        where = ".".join(str(part) for part in e["loc"]) or "(whole file)"
        message = e["msg"].removeprefix("Value error, ")
        lines.append(f"{where}: {message}")
    return lines


def _open_project(project: Path) -> tuple[Workspace, ProjectConfig]:  # NEW
    """Open a workspace and load its config; print a clear message and exit on any problem."""
    try:
        ws = Workspace.open(project)
        return ws, ws.load()
    except WorkspaceError as err:
        _fail(str(err))
    except yaml.YAMLError as err:
        _fail(f"project.yaml is not valid YAML:\n{err}")
    except ValidationError as err:
        typer.secho("project.yaml has problems:", fg=typer.colors.RED)
        for line in _format_errors(err):
            typer.echo(f"  - {line}")
        raise typer.Exit(1) from None


def _require_files(ws: Workspace, cfg: ProjectConfig) -> None:  # NEW
    """Exit with code 2 if any file the config refers to is missing."""
    missing = ws.missing_files(cfg)
    if missing:
        typer.secho("files: missing", fg=typer.colors.YELLOW)
        for label, path in missing:
            typer.echo(f"  - {label}: {path}")
        raise typer.Exit(2)


def _describe(summary: dict) -> str:  # NEW
    return ", ".join(f"{key}={value}" for key, value in summary.items())


@app.command()
def init(folder: Annotated[Path, typer.Argument(help="Folder to create the project in.")]) -> None:
    """Create a new project workspace with a starter project.yaml."""
    try:
        ws = Workspace.create(folder)
    except WorkspaceError as err:
        _fail(str(err))
    typer.echo(f"Created project in {ws.root}")
    typer.echo("Next: put your data in the data/ folders, edit project.yaml, then run:")
    typer.echo(f"  molecure validate {folder}")


@app.command()
def validate(  # CHANGED: uses the two helpers above
    project: Annotated[Path, typer.Argument(help="Project folder (or its project.yaml).")],
) -> None:
    """Check project.yaml, and that every file it refers to exists.

    Exit codes: 0 = all good, 1 = config or folder problem, 2 = config fine but files missing.
    """
    ws, cfg = _open_project(project)

    typer.secho("config: OK", fg=typer.colors.GREEN)
    roles = Counter(d.role for d in cfg.datasets)
    typer.echo(f"  project:   {cfg.project.name}")
    typer.echo("  datasets:  " + ", ".join(f"{n} {role}" for role, n in sorted(roles.items())))
    typer.echo(f"  alignment: {cfg.alignment.mode}")
    typer.echo(f"  fields:    {cfg.molecular_fields.method}")

    _require_files(ws, cfg)
    typer.secho("files: all present", fg=typer.colors.GREEN)


@app.command()  # NEW
def run(
    project: Annotated[Path, typer.Argument(help="Project folder (or its project.yaml).")],
) -> None:
    """Load every dataset, then create a run folder recording exactly what was used.

    Today this is only the first pipeline stage; later stages will be added to it.
    Exit codes: 0 = ok, 1 = config or data problem, 2 = files missing.
    """
    ws, cfg = _open_project(project)
    _require_files(ws, cfg)

    try:
        datasets = load_all(cfg, ws.root)
    except DataError as err:
        _fail(str(err))  # no run folder is created for bad data

    new_run = create_run(ws, cfg)
    summaries = {name: ds.summary() for name, ds in datasets.items()}
    new_run.add_to_manifest("datasets", summaries)

    typer.secho("data: loaded", fg=typer.colors.GREEN)
    for name, summary in summaries.items():
        typer.echo(f"  {name}: {_describe(summary)}")
    typer.echo(f"run folder: {new_run.root}")


FIT_ERRORS = (DataError, PipelineError, SplitError, FieldError, ModelError, ValueError)


@app.command()
def fit(
    project: Annotated[Path, typer.Argument(help="Project folder (or its project.yaml).")],
) -> None:
    """Run the calibration pipeline: load, poses, split, fields, PLS, validation.

    Writes a run folder with results.json, predictions.csv, split.csv and model.npz.
    Exit codes: 0 = ok, 1 = config, data or model problem, 2 = files missing.
    """
    ws, cfg = _open_project(project)
    _require_files(ws, cfg)
    try:
        fit_result = fit_project(ws, cfg)
    except FIT_ERRORS as err:
        _fail(str(err))
    typer.secho("model: fitted", fg=typer.colors.GREEN)
    for line in format_summary(fit_result.result.summary):
        typer.echo(f"  {line}")
    typer.echo(f"run folder: {fit_result.run.root}")


@app.command()
def sensitivity(
    project: Annotated[Path, typer.Argument(help="Project folder (or its project.yaml).")],
) -> None:
    """How much do the results depend on grid position, spacing, scaling and field settings?

    Re-fits with ONE thing changed at a time and reports the leave-one-out q2 of the training
    compounds. Nothing is saved. Exit codes: 0 = ok, 1 = problem, 2 = files missing.
    """
    ws, cfg = _open_project(project)
    _require_files(ws, cfg)
    try:
        inputs = load_calibration_inputs(ws, cfg)
        data = (inputs.ids, inputs.activities, inputs.assignment, inputs.atoms)
        lattice = sweep(cfg, *data, lattice_offsets())
        settings = sweep(cfg, *data, setting_variations(cfg))
    except FIT_ERRORS as err:
        _fail(str(err))
    title = "Lattice offsets (leave-one-out q2, training compounds only)"
    for line in format_sweep(title, lattice, with_range=True):
        typer.echo(line)
    for line in format_sweep("Settings, one at a time", settings):
        typer.echo(line)
