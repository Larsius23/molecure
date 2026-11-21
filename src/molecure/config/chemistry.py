"""Chemistry rules: standardization, protonation, charges, conformers.

There is exactly ONE of each of these per project and no per-dataset overrides.
Calibration and screening molecules go through identical rules, so the model
cannot learn a protocol artifact instead of chemistry.
"""

from typing import Literal

from pydantic import Field

from .base import StrictModel


class StandardizeConfig(StrictModel):
    strip_salts: bool = True  # keep the largest fragment
    undefined_stereo: Literal["flag", "error", "ignore"] = "flag"


class ProtonationConfig(StrictModel):
    backend: Literal["molscrub", "rdkit"] = "molscrub"
    ph: float = Field(7.4, ge=0.0, le=14.0)
    tautomers: Literal["as_given", "enumerate"] = "as_given"
    # QSAR needs ONE state per compound. molscrub can return several.
    state_policy: Literal["first", "require_single"] = "first"
    # Optional CSV (compound_id, smiles) with hand-chosen final states.
    overrides: str | None = None


class ConformerConfig(StrictModel):
    n_conformers: int = Field(50, ge=1)
    minimize: Literal["mmff94", "mmff94s", "none"] = "mmff94"


class ChemistryConfig(StrictModel):
    standardize: StandardizeConfig = Field(default_factory=StandardizeConfig)
    protonation: ProtonationConfig = Field(default_factory=ProtonationConfig)
    # Used for the electrostatic field. Gasteiger handles zwitterions poorly.
    charge_model: Literal["mmff94", "gasteiger", "xtb"] = "mmff94"
    conformers: ConformerConfig = Field(default_factory=ConformerConfig)
