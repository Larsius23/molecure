"""Dataset roles.

The role decides what a dataset may contain, and therefore what the pipeline
may do with it:

    calibration  structures + measured activity -> splitting, CV, fitting, external test
    screening    structures only                -> prediction only, never touches the fit
    reference    one trusted 3D pose            -> alignment template

For .sdf files, id_col / activity_col name SD properties instead of CSV columns.
"""

from typing import Annotated, Literal

from pydantic import Field

from .base import NAME_PATTERN, StrictModel


class _StructureTable(StrictModel):
    id: str = Field(pattern=NAME_PATTERN)
    file: str  # path relative to the project folder (.csv or .sdf)
    id_col: str = "compound_id"
    smiles_col: str = "smiles"  # ignored for SDF files


class CalibrationDataset(_StructureTable):
    role: Literal["calibration"]
    activity_col: str  # no default: a calibration set MUST say where activity lives
    endpoint: Literal["pIC50", "pKi", "pEC50", "pKd"]


class ScreeningDataset(_StructureTable):
    role: Literal["screening"]
    # Deliberately no activity field. Because of extra="forbid", a screening
    # dataset that tries to declare one is rejected automatically.


class ReferenceDataset(StrictModel):
    role: Literal["reference"]
    id: str = Field(pattern=NAME_PATTERN)
    file: str  # SDF containing the trusted 3D pose


# A "discriminated union": pydantic reads `role` first, then validates against
# the matching class only. Error messages stay short and point at the right rule.
Dataset = Annotated[
    CalibrationDataset | ScreeningDataset | ReferenceDataset,
    Field(discriminator="role"),
]
