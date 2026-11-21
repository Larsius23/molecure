"""Alignment: pick ONE mode. Options that belong to another mode don't exist."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .base import StrictModel


class MCSAlignment(StrictModel):
    mode: Literal["mcs"]
    reference: str  # id of a dataset with role "reference"
    smarts: str | None = None  # None = find the MCS against the reference
    min_core_atoms: int = Field(8, ge=3)
    on_fail: Literal["flag", "error"] = "flag"  # flag = mark "unalignable", never force


class O3AAlignment(StrictModel):
    mode: Literal["o3a"]
    reference: str
    variant: Literal["mmff", "crippen"] = "mmff"
    fallback_to_crippen: bool = True  # if MMFF lacks parameters for a molecule


class PrealignedAlignment(StrictModel):
    """Poses already in a common frame: docked SDF, or a literature benchmark."""

    mode: Literal["prealigned"]
    file: str
    id_property: str | None = None  # SD tag holding the compound id (None = title line)
    pose_rule: Literal["first", "best_score"] = "first"
    score_property: str | None = None
    score_better: Literal["lower", "higher"] = "lower"  # Vina: lower is better
    receptor_file: str | None = None  # for overlaying contours on the pocket later
    # Free-form record of how the poses were made (box, seed, exhaustiveness, ...).
    metadata: dict[str, str | int | float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _best_score_needs_a_score(self) -> Self:
        if self.pose_rule == "best_score" and not self.score_property:
            raise ValueError(
                "pose_rule 'best_score' needs score_property (the SD tag with the score)"
            )
        return self


AlignmentConfig = Annotated[
    MCSAlignment | O3AAlignment | PrealignedAlignment,
    Field(discriminator="mode"),
]
