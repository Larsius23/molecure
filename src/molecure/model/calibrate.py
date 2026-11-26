"""The numeric core of the calibration pipeline: fields -> PLS -> validation -> files.

No RDKit and no project files in here: the pipeline hands it aligned atoms, activities and the
train/test assignment, so it can be tested with hand-made molecules. `calibrate` computes and
`write_results` saves, so a failed fit never leaves a half-written run folder.

`cfg` is a ProjectConfig. Only these attributes are read:
  cfg.molecular_fields
  cfg.pls.{min_sigma, scaling, max_components, n_components, selection}
  cfg.validation.{cv.loo, cv.group_kfold_k, cv.group_kfold_repeats, y_randomization_runs, metrics}
  cfg.runtime.seed
"""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molecure.data.splits import write_split
from molecure.fields.atomdata import AtomData
from molecure.fields.calculator import FieldCalculator
from molecure.validate.evaluate import EvalSettings, evaluate

from .pls import ModelError, Recipe

SUPPORTED_METRICS = {"q2", "r2", "q2_f1", "q2_f2", "q2_f3", "ccc", "r2_corr"}
PREDICTION_COLUMNS = ["compound_id", "subset", "observed", "fitted", "cv_loo", "predicted"]


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    summary: dict  # JSON-able numbers
    rows: list[dict]  # one per compound, in dataset order
    assignment: dict[str, str]  # compound id -> 'train' | 'test'
    arrays: dict[str, np.ndarray]  # the contents of model.npz
    n_components: int


def calibrate(
    cfg,
    ids: Sequence[str],
    y: Sequence[float],
    assignment: Mapping[str, str],
    atoms: Sequence[AtomData],
) -> CalibrationResult:
    ids = list(ids)
    y = np.asarray(y, dtype=float)
    if not (len(ids) == len(y) == len(atoms)):
        raise ModelError(f"{len(ids)} ids, {len(y)} activities and {len(atoms)} molecules differ")
    unsupported = [m for m in cfg.validation.metrics if m not in SUPPORTED_METRICS]
    if unsupported:
        raise ModelError(
            f"metric(s) {unsupported} are not implemented yet; remove them from validation.metrics"
        )
    train = [i for i, c in enumerate(ids) if assignment[c] == "train"]
    test = [i for i, c in enumerate(ids) if assignment[c] == "test"]
    if len(train) < 3:
        raise ModelError(f"{len(train)} training compounds: at least 3 are needed")

    # The grid comes from the TRAINING compounds only; external compounds just sit on it.
    calculator = FieldCalculator.fit([atoms[i] for i in train], cfg.molecular_fields)
    matrix = calculator.compute_matrix(ids, atoms)
    recipe = Recipe(
        min_sigma=cfg.pls.min_sigma,
        scaling=cfg.pls.scaling,
        blocks=np.repeat(np.arange(len(matrix.block_names)), matrix.n_points),
    )
    settings = EvalSettings(
        max_components=cfg.pls.max_components,
        n_components=cfg.pls.n_components,
        selection=cfg.pls.selection,
        loo=cfg.validation.cv.loo,
        kfold_k=cfg.validation.cv.group_kfold_k,
        kfold_repeats=cfg.validation.cv.group_kfold_repeats,
        y_randomization_runs=cfg.validation.y_randomization_runs,
        seed=cfg.runtime.seed,
    )
    X = matrix.values
    evaluation = evaluate(
        X[train], y[train], recipe, settings,
        X[test] if test else None, y[test] if test else None,
    )  # fmt: skip

    # ---- one row per compound, in dataset order ----
    by_index: dict[int, dict] = {}
    for position, i in enumerate(train):
        cv = evaluation.loo_predictions
        by_index[i] = {
            "fitted": float(evaluation.fit_predictions[position]),
            "cv_loo": float(cv[position]) if cv is not None else None,
            "predicted": None,
        }
    for position, i in enumerate(test):
        by_index[i] = {
            "fitted": None,
            "cv_loo": None,
            "predicted": float(evaluation.external_predictions[position]),
        }
    rows = [
        {
            "compound_id": ids[i],
            "subset": assignment[ids[i]],
            "observed": float(y[i]),
            **by_index[i],
            "outside_grid": bool(matrix.outside_grid[i]),
        }
        for i in range(len(ids))
    ]

    grid = calculator.grid
    summary = dict(evaluation.summary)
    summary["compounds"] = {"train": len(train), "test": len(test)}
    summary["fields"] = {"method": cfg.molecular_fields.method, "blocks": list(matrix.block_names)}
    summary["grid"] = {
        "origin": list(grid.origin),
        "spacing": grid.spacing,
        "shape": list(grid.shape),
        "n_points": grid.n_points,
    }
    summary["outside_grid"] = [r["compound_id"] for r in rows if r["outside_grid"]]
    summary["requested_metrics"] = list(cfg.validation.metrics)
    summary["seed"] = cfg.runtime.seed

    coefficients, intercept = evaluation.model.coefficients(evaluation.n_components)
    arrays = {
        **evaluation.model.to_arrays(),
        "coefficients": coefficients,
        "intercept": np.array(intercept),
        "n_components": np.array(evaluation.n_components),
        "block_names": np.array(matrix.block_names),
        "n_points": np.array(matrix.n_points),
        "grid_origin": np.array(grid.origin),
        "grid_spacing": np.array(grid.spacing),
        "grid_shape": np.array(grid.shape),
    }
    return CalibrationResult(summary, rows, dict(assignment), arrays, evaluation.n_components)


def _json_safe(value):
    """JSON has no NaN or infinity: write them as null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_results(result: CalibrationResult, out_dir: str | Path) -> None:
    """results.json, predictions.csv, split.csv and model.npz."""
    out_dir = Path(out_dir)
    text = json.dumps(_json_safe(result.summary), indent=2)
    (out_dir / "results.json").write_text(text + "\n", encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*PREDICTION_COLUMNS, "outside_grid"])
        for row in result.rows:
            writer.writerow([_cell(row[name]) for name in [*PREDICTION_COLUMNS, "outside_grid"]])
    write_split(out_dir / "split.csv", result.assignment)
    np.savez_compressed(out_dir / "model.npz", **result.arrays)


def _num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def format_summary(summary: dict) -> list[str]:
    """Short human-readable lines for the terminal."""
    n = summary["compounds"]
    comp = summary["components"]
    lines = [
        f"compounds:     {n['train']} training" + (f", {n['test']} external" if n["test"] else ""),
        f"components:    {comp['used']} ({comp['rule']}; at most {comp['max']})",
    ]
    if "loo" in summary:
        loo = summary["loo"]
        lines.append(f"leave-one-out: q2 {_num(loo['q2'])}   spress {_num(loo['spress'])}")
        lines.append("  q2 by components: " + "  ".join(_num(v) for v in loo["q2_by_components"]))
    lines.append(f"fit:           r2 {_num(summary['fit']['r2'])}   s {_num(summary['fit']['s'])}")
    if "nested_loo" in summary:
        nested = summary["nested_loo"]
        lines.append(
            f"nested LOO:    q2 {_num(nested['q2'])} (component count chosen inside each fold)"
        )
    if "kfold" in summary:
        k = summary["kfold"]
        lines.append(
            f"{k['k']}-fold x {k['repeats']}:  q2 {_num(k['q2_mean'])} +/- {_num(k['q2_sd'])} "
            f"(worst {_num(k['q2_min'])})"
        )
    if "y_randomization" in summary:
        y = summary["y_randomization"]
        lines.append(
            f"y-randomization: {y['runs']} runs, q2 mean {_num(y['q2_mean'])}, "
            f"max {_num(y['q2_max'])}; {y['runs_at_or_above_real_q2']} reached the real q2"
        )
    if "external" in summary:
        e = summary["external"]
        lines.append(
            f"external ({e['n']}): Q2F1 {_num(e['q2_f1'])}  Q2F2 {_num(e['q2_f2'])}  "
            f"CCC {_num(e['ccc'])}  r2_corr {_num(e['r2_corr'])}  RMSE {_num(e['rmse'])}"
        )
    qc = summary.get("alignment_qc")
    if qc:
        lines.append(
            f"alignment QC:  core RMSD to the reference: median {_num(qc['median'], 2)}, "
            f"max {_num(qc['max'], 2)} A ({qc['worst']})"
        )
    if summary.get("outside_grid"):
        names = ", ".join(summary["outside_grid"][:5])
        count = len(summary["outside_grid"])
        lines.append(f"WARNING: {count} compound(s) outside the grid: {names}")
    if summary.get("excluded"):
        lines.append(f"WARNING: {len(summary['excluded'])} compound(s) excluded (see excluded.csv)")
    for note in summary.get("warnings", []):
        lines.append(f"WARNING: {note}")
    return lines
