"""The calibration pipeline for one project folder:

    load datasets -> aligned poses -> train/test split -> field atoms -> fields, PLS, validation
    -> a run folder with results.json, predictions.csv, split.csv and model.npz

This is the thin shell around the numeric core in calibrate.py: it reads files and RDKit
molecules, and all of the modelling happens in `calibrate`. Nothing is written until the whole
model has been fitted, so a failure never leaves a half-finished run folder behind.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from molecure.config import ProjectConfig
from molecure.data import CalibrationSet, load_all
from molecure.data.splits import make_split
from molecure.fields.atomdata import AtomData
from molecure.fields.errors import FieldError
from molecure.fields.molprops import atom_data_from_mol
from molecure.project import Run, Workspace, create_run

from .calibrate import CalibrationResult, calibrate, write_results

MAX_SHOWN = 10


class PipelineError(Exception):
    """The project cannot go through the pipeline (reported before any model is fitted)."""


@dataclass(frozen=True, eq=False)
class FitResult:
    run: Run
    result: CalibrationResult


@dataclass(frozen=True, eq=False)
class CalibrationInputs:
    """Everything the numeric core needs, loaded from a project folder."""

    ids: list[str]
    activities: list[float]
    assignment: dict[str, str]  # compound id -> 'train' | 'test'
    atoms: list[AtomData]  # aligned, in dataset order


def _listing(title: str, items: Sequence[str]) -> str:
    shown = "\n".join(f"  - {item}" for item in items[:MAX_SHOWN])
    more = len(items) - MAX_SHOWN
    return f"{title}\n{shown}" + (f"\n  ... and {more} more" if more > 0 else "")


def load_poses(alignment, root: Path) -> dict[str, Chem.Mol]:
    """Read pre-aligned poses from an SDF: {compound id: molecule}.

    The id is the SD property named by `alignment.id_property` (or the title line).
    If a compound has several poses (e.g. from docking), `alignment.pose_rule` picks one:
    the first, or the best by `score_property` (`score_better` says whether lower or higher wins).
    """
    path = root / alignment.file
    if not path.is_file():
        raise PipelineError(f"alignment file not found: {path}")
    where = alignment.id_property or "the title line"
    candidates: dict[str, list[tuple[float, Chem.Mol]]] = {}
    problems: list[str] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for position, mol in enumerate(supplier, start=1):
        if mol is None:
            problems.append(f"record {position}: could not be parsed or sanitized")
            continue
        prop = alignment.id_property or "_Name"
        key = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
        if not key:
            problems.append(f"record {position}: no compound id in {where}")
            continue
        score = 0.0
        if alignment.pose_rule == "best_score":
            try:
                score = float(mol.GetProp(alignment.score_property))
            except (KeyError, ValueError):
                problems.append(f"record {position}: no usable '{alignment.score_property}' score")
                continue
        candidates.setdefault(key, []).append((score, mol))
    if problems:
        raise PipelineError(_listing(f"{path.name}:", problems))

    poses: dict[str, Chem.Mol] = {}
    for key, options in candidates.items():
        if alignment.pose_rule == "first":
            poses[key] = options[0][1]
        else:
            pick = min if alignment.score_better == "lower" else max
            poses[key] = pick(options, key=lambda option: option[0])[1]
    return poses


def load_calibration_inputs(ws: Workspace, cfg: ProjectConfig) -> CalibrationInputs:
    """Datasets -> poses -> split -> field atoms. Raises PipelineError before any fitting."""
    if cfg.alignment.mode != "prealigned":
        raise PipelineError(
            f"alignment mode '{cfg.alignment.mode}' is built in Stage 8; "
            "use mode: prealigned (poses supplied in an SDF) for now"
        )
    datasets = load_all(cfg, ws.root)
    calibration = [d for d in datasets.values() if isinstance(d, CalibrationSet)]
    if len(calibration) != 1:
        raise PipelineError(
            f"exactly one calibration dataset is supported for now, found {len(calibration)}"
        )
    ids = list(calibration[0].ids)
    activities = list(calibration[0].activities)

    assignment = make_split(cfg.validation.split, ids, ws.root, cfg.runtime.seed)

    poses = load_poses(cfg.alignment, ws.root)
    missing = [c for c in ids if c not in poses]
    if missing:
        raise PipelineError(_listing(f"no pose for {len(missing)} compound(s):", missing))

    atoms, problems = [], []
    for compound_id in ids:
        try:
            atoms.append(atom_data_from_mol(poses[compound_id], cfg.chemistry.charge_model))
        except FieldError as err:
            problems.append(f"{compound_id}: {err}")
    if problems:
        raise PipelineError(_listing("fields cannot be computed for:", problems))
    return CalibrationInputs(ids, activities, assignment, atoms)


def fit_project(ws: Workspace, cfg: ProjectConfig) -> FitResult:
    inputs = load_calibration_inputs(ws, cfg)
    # all the modelling happens in calibrate()
    result = calibrate(cfg, inputs.ids, inputs.activities, inputs.assignment, inputs.atoms)

    run = create_run(ws, cfg)  # only now, so a failed fit leaves no run folder
    write_results(result, run.root)
    run.add_to_manifest(
        "model",
        {
            "n_components": result.n_components,
            "n_train": result.summary["compounds"]["train"],
            "n_test": result.summary["compounds"]["test"],
            "files": ["results.json", "predictions.csv", "split.csv", "model.npz"],
        },
    )
    return FitResult(run=run, result=result)
