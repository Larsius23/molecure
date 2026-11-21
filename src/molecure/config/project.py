"""The top-level config: one `project.yaml` becomes one ProjectConfig."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from .alignment import AlignmentConfig
from .base import NAME_PATTERN, StrictModel
from .chemistry import ChemistryConfig
from .datasets import Dataset
from .fields import MolecularFields
from .modeling import PLSConfig, ValidationConfig


class ProjectMeta(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""


class RuntimeConfig(StrictModel):
    seed: int = 42  # the ONLY seed; every random step derives from it
    n_jobs: int = Field(1, ge=1)


class ProjectConfig(StrictModel):
    schema_version: Literal[1] = 1
    project: ProjectMeta
    datasets: list[Dataset] = Field(min_length=1)
    chemistry: ChemistryConfig = Field(default_factory=ChemistryConfig)
    alignment: AlignmentConfig
    molecular_fields: MolecularFields
    pls: PLSConfig = Field(default_factory=PLSConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        """Rules that involve more than one section."""
        ids = [d.id for d in self.datasets]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"dataset ids must be unique; repeated: {repeated}")

        roles = {d.id: d.role for d in self.datasets}
        # Relaxed later if we add "reuse a saved model on new screening data".
        if "calibration" not in roles.values():
            raise ValueError("at least one dataset with role 'calibration' is required")

        if self.alignment.mode in ("mcs", "o3a"):
            ref = self.alignment.reference
            if roles.get(ref) != "reference":
                raise ValueError(
                    f"alignment.reference '{ref}' must be the id of a 'reference' dataset"
                )
        return self


def load_config(path: str | Path) -> ProjectConfig:
    """Read and validate a project.yaml. Raises pydantic.ValidationError if invalid."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)  # safe_load: plain data only, never runs code from a file
    return ProjectConfig.model_validate(raw)


def dump_config(cfg: ProjectConfig, path: str | Path) -> None:
    """Write the fully resolved config (all defaults filled in) to YAML."""
    data = cfg.model_dump(mode="json")  # mode="json": tuples become lists, etc.
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
