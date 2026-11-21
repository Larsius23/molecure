"""Command-line interface: `molecure <command>`."""

from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from molecure import __version__
from molecure.project import Workspace, WorkspaceError

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
def validate(
    project: Annotated[Path, typer.Argument(help="Project folder (or its project.yaml).")],
) -> None:
    """Check project.yaml, and that every file it refers to exists.

    Exit codes: 0 = all good, 1 = config or folder problem, 2 = config fine but files missing.
    """
    try:
        ws = Workspace.open(project)
        cfg = ws.load()
    except WorkspaceError as err:
        _fail(str(err))
    except yaml.YAMLError as err:
        _fail(f"project.yaml is not valid YAML:\n{err}")
    except ValidationError as err:
        typer.secho("project.yaml has problems:", fg=typer.colors.RED)
        for line in _format_errors(err):
            typer.echo(f"  - {line}")
        raise typer.Exit(1) from None

    typer.secho("config: OK", fg=typer.colors.GREEN)
    roles = Counter(d.role for d in cfg.datasets)
    typer.echo(f"  project:   {cfg.project.name}")
    typer.echo("  datasets:  " + ", ".join(f"{n} {role}" for role, n in sorted(roles.items())))
    typer.echo(f"  alignment: {cfg.alignment.mode}")
    typer.echo(f"  fields:    {cfg.molecular_fields.method}")

    missing = ws.missing_files(cfg)
    if missing:
        typer.secho("files: missing", fg=typer.colors.YELLOW)
        for label, path in missing:
            typer.echo(f"  - {label}: {path}")
        raise typer.Exit(2)
    typer.secho("files: all present", fg=typer.colors.GREEN)
