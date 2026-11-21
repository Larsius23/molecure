"""PLS model and validation settings.

Everything fitted from data (variance filter, scaling, number of components)
happens INSIDE the training folds. These settings only choose the recipe.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from .base import StrictModel

Metric = Literal["q2", "r2", "q2_f1", "q2_f2", "q2_f3", "ccc", "rm2"]


class PLSConfig(StrictModel):
    max_components: int = Field(8, ge=1)
    n_components: int | None = Field(None, ge=1)  # fix it to reproduce a published model
    selection: Literal["min_press", "one_se"] = "min_press"
    # none = raw values; autoscale = unit variance per column;
    # block = each field block rescaled to equal total weight.
    scaling: Literal["none", "autoscale", "block"] = "none"
    # Columns with a training-fold std below this are dropped (0 = only constant ones).
    min_sigma: float = Field(0.0, ge=0.0)


class SplitConfig(StrictModel):
    method: Literal["random", "scaffold", "cluster", "from_file"] = "cluster"
    test_fraction: float = Field(0.2, ge=0.0, lt=0.5)  # 0 = no external test set
    file: str | None = None  # for method "from_file", e.g. a published split

    @model_validator(mode="after")
    def _from_file_needs_file(self) -> Self:
        if self.method == "from_file" and not self.file:
            raise ValueError("split method 'from_file' needs `file` (the split to reuse)")
        return self


class CVConfig(StrictModel):
    loo: bool = True
    group_kfold_k: int = Field(5, ge=2)
    group_kfold_repeats: int = Field(20, ge=1)


class ValidationConfig(StrictModel):
    split: SplitConfig = Field(default_factory=SplitConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    cluster_cutoff: float = Field(0.6, gt=0.0, lt=1.0)  # Butina distance = 1 - Tanimoto
    y_randomization_runs: int = Field(100, ge=0)
    metrics: tuple[Metric, ...] = ("q2", "r2", "q2_f2", "ccc")
