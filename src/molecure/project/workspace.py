"""A project workspace: one folder holding project.yaml, data, splits and runs.

The workspace lives OUTSIDE the code repository. The repo holds the tool;
a workspace holds one study.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from molecure.config import ProjectConfig, load_config

from .template import STARTER_YAML

CONFIG_NAME = "project.yaml"
SUBDIRS = ("data/raw", "data/curated", "data/screening", "data/reference", "splits", "runs")


class WorkspaceError(Exception):
    """Something is wrong with the project FOLDER (as opposed to the config's contents)."""


def referenced_files(cfg: ProjectConfig) -> list[tuple[str, str]]:
    """Every file the config points to, as (what it is, path relative to the project)."""
    found = [(f"dataset '{d.id}'", d.file) for d in cfg.datasets]

    if cfg.alignment.mode == "prealigned":
        found.append(("alignment.file", cfg.alignment.file))
        if cfg.alignment.receptor_file:
            found.append(("alignment.receptor_file", cfg.alignment.receptor_file))
    if cfg.chemistry.protonation.overrides:
        found.append(("protonation.overrides", cfg.chemistry.protonation.overrides))
    if cfg.validation.split.file:
        found.append(("validation.split.file", cfg.validation.split.file))
    return found


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @classmethod
    def create(cls, folder: str | Path) -> "Workspace":
        """Make the folder layout and a starter project.yaml. Never overwrites."""
        root = Path(folder).expanduser().resolve()
        ws = cls(root)
        if ws.config_path.exists():
            raise WorkspaceError(f"{ws.config_path} already exists; refusing to overwrite it")
        for sub in SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_\-]", "_", root.name) or "project"
        ws.config_path.write_text(STARTER_YAML.replace("__NAME__", name), encoding="utf-8")
        return ws

    @classmethod
    def open(cls, path: str | Path) -> "Workspace":
        """Open an existing workspace, given its folder or its project.yaml."""
        p = Path(path).expanduser().resolve()
        if p.is_file():
            if p.name != CONFIG_NAME:
                raise WorkspaceError(f"expected a file named {CONFIG_NAME}, got {p.name}")
            p = p.parent
        if not (p / CONFIG_NAME).is_file():
            raise WorkspaceError(f"no {CONFIG_NAME} found in {p}")
        return cls(p)

    def load(self) -> ProjectConfig:
        return load_config(self.config_path)

    def missing_files(self, cfg: ProjectConfig) -> list[tuple[str, Path]]:
        """Referenced files that do not exist yet, as (what it is, full path)."""
        paths = [(label, self.root / rel) for label, rel in referenced_files(cfg)]
        return [(label, path) for label, path in paths if not path.is_file()]
