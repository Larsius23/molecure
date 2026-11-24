"""Molecular field settings: CoMFA-style or CoMSIA-style."""

from typing import Annotated, Literal

from pydantic import Field

from .base import StrictModel

CoMSIAField = Literal["steric", "electrostatic", "hydrophobic", "donor", "acceptor"]


class GridConfig(StrictModel):
    """Grid size. Its EXTENT is not configurable: it is always derived from the
    calibration molecules plus `padding`, so screening data can never move it."""

    spacing: float = Field(2.0, gt=0.0)  # angstrom
    padding: float = Field(4.0, ge=0.0)  # angstrom


class ComfaFields(StrictModel):
    method: Literal["comfa"]
    grid: GridConfig = Field(default_factory=GridConfig)
    # Standard CoMFA settings as reported in published methods sections:
    # sp3 carbon probe with +1 charge, 30 kcal/mol truncation, distance-dependent dielectric.
    probe_charge: float = 1.0
    # The probe's van der Waals radius in angstrom. The default is UFF sp3 carbon
    # (x = 3.851 A, radius 1.9255 A); the Tripos probe of the 1988 paper was smaller (1.52 A).
    probe_radius: float = Field(1.9255, gt=0.0)
    energy_cutoff_kcal: float = Field(30.0, gt=0.0)
    dielectric: Literal["distance", "constant"] = "distance"


def _comsia_grid() -> GridConfig:
    return GridConfig(spacing=1.0)  # CoMSIA conventionally uses a finer grid


class ComsiaFields(StrictModel):
    method: Literal["comsia"]
    grid: GridConfig = Field(default_factory=_comsia_grid)
    attenuation: float = Field(0.3, gt=0.0)  # Gaussian attenuation factor
    include: tuple[CoMSIAField, ...] = (
        "steric",
        "electrostatic",
        "hydrophobic",
        "donor",
        "acceptor",
    )


MolecularFields = Annotated[ComfaFields | ComsiaFields, Field(discriminator="method")]
