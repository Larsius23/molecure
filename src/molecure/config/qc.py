"""Quality control: what to do with molecules that cannot be used, and how well they must align."""

from typing import Literal

from pydantic import Field

from .base import StrictModel


class QCConfig(StrictModel):
    # What happens when a compound has no pose, cannot get field values, or fails the alignment
    # check. "error" stops with the full list (the safe default). "skip" drops them, records
    # every one in excluded.csv and results.json, and goes on, as long as no more than
    # max_skipped_fraction of the compounds are lost.
    on_invalid: Literal["error", "skip"] = "error"
    max_skipped_fraction: float = Field(0.1, ge=0.0, le=1.0)
    # Alignment check against alignment.reference: the RMSD (angstrom) of the common core, measured
    # in place. None = report only; a number turns it into a gate.
    max_core_rmsd: float | None = Field(None, gt=0.0)
    min_core_atoms: int = Field(8, ge=3)  # a smaller common core means "cannot be judged"
