"""The calibration pipeline for one project folder:

    load datasets -> aligned poses -> train/test split -> field atoms -> alignment check
    -> fields, PLS, validation -> a run folder with the results

This is the thin shell around the numeric core in calibrate.py: it reads files and RDKit
molecules, and all of the modelling happens in `calibrate`. Nothing is written until the whole
model has been fitted, so a failure never leaves a half-finished run folder behind.

Unusable compounds (no pose, no field values, failed alignment check) are handled by the
`qc` settings: by default the pipeline stops and lists every one of them; with
qc.on_invalid: skip it drops them, up to qc.max_skipped_fraction, and records each in
excluded.csv and results.json, so the model is never silently trained on a different set.
"""

import csv
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdkit import Chem

from molecure.align.mcs import mcs_core_rmsd
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


@dataclass(frozen=True)
class Exclusion:
    compound_id: str
    stage: str  # "pose" | "fields" | "alignment"
    reason: str


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
    excluded: list[Exclusion] = field(default_factory=list)  # compounds that were skipped
    warnings: list[str] = field(default_factory=list)
    alignment_qc: list[dict] = field(default_factory=list)  # one row per compound with a pose


def _listing(title: str, items: Sequence[str]) -> str:
    shown = "\n".join(f"  - {item}" for item in items[:MAX_SHOWN])
    more = len(items) - MAX_SHOWN
    return f"{title}\n{shown}" + (f"\n  ... and {more} more" if more > 0 else "")


def load_poses(alignment, root: Path, notes: list[str] | None = None) -> dict[str, Chem.Mol]:
    """Read pre-aligned poses from an SDF: {compound id: molecule}.

    The id is the SD property named by `alignment.id_property` (or the title line).
    If a compound has several poses (e.g. from docking), `alignment.pose_rule` picks one:
    the first (docking programs write the best-scoring pose first), or the best by
    `score_property` (`score_better` says whether lower or higher wins). When several poses
    exist and the rule is 'first', a note is appended to `notes` so the choice is never silent.
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

    several = sorted(key for key, options in candidates.items() if len(options) > 1)
    if several and alignment.pose_rule == "first" and notes is not None:
        examples = ", ".join(several[:3])
        notes.append(
            f"{len(several)} compound(s) have several poses (e.g. {examples}); pose_rule is "
            "'first', so the first pose of each is used. "
            "Set pose_rule: best_score to choose by score."
        )

    poses: dict[str, Chem.Mol] = {}
    for key, options in candidates.items():
        if alignment.pose_rule == "first":
            poses[key] = options[0][1]
        else:
            pick = min if alignment.score_better == "lower" else max
            poses[key] = pick(options, key=lambda option: option[0])[1]
    return poses


def alignment_qc(cfg: ProjectConfig, ids: Sequence[str], poses: dict[str, Chem.Mol]) -> list[dict]:
    """Core RMSD of every pose against the reference pose, measured in place (poses stay put).

    Needs alignment.reference: the id of a compound in the pose file. Without it, no check.
    """
    reference = cfg.alignment.reference
    if reference is None:
        return []
    if reference not in poses:
        raise PipelineError(f"alignment.reference '{reference}' has no pose in the alignment file")
    rows = []
    for compound_id in ids:
        if compound_id not in poses:
            continue
        rmsd, n_core = mcs_core_rmsd(poses[reference], poses[compound_id], cfg.qc.min_core_atoms)
        rows.append({"compound_id": compound_id, "core_atoms": n_core, "core_rmsd": rmsd})
    return rows


def alignment_failures(cfg: ProjectConfig, rows: Sequence[dict]) -> list[Exclusion]:
    failures = []
    for row in rows:
        if row["core_rmsd"] is None:
            reason = (
                f"common core with the reference has {row['core_atoms']} atoms "
                f"(minimum {cfg.qc.min_core_atoms}); it cannot be judged"
            )
            failures.append(Exclusion(row["compound_id"], "alignment", reason))
        elif cfg.qc.max_core_rmsd is not None and row["core_rmsd"] > cfg.qc.max_core_rmsd:
            reason = (
                f"core RMSD {row['core_rmsd']:.2f} A is above qc.max_core_rmsd "
                f"({cfg.qc.max_core_rmsd:g} A)"
            )
            failures.append(Exclusion(row["compound_id"], "alignment", reason))
    return failures


def qc_summary(rows: Sequence[dict]) -> dict | None:
    judged = [r for r in rows if r["core_rmsd"] is not None]
    if not judged:
        return None
    worst = max(judged, key=lambda r: r["core_rmsd"])
    return {
        "reference_compounds": len(judged),
        "median": statistics.median(r["core_rmsd"] for r in judged),
        "max": worst["core_rmsd"],
        "worst": worst["compound_id"],
    }


def _failure_message(excluded: Sequence[Exclusion]) -> str:
    """One listing per stage, in the order the problems arise."""
    parts = []
    by_stage = {s: [e for e in excluded if e.stage == s] for s in ("pose", "fields", "alignment")}
    if by_stage["pose"]:
        ids = [e.compound_id for e in by_stage["pose"]]
        parts.append(_listing(f"no pose for {len(ids)} compound(s):", ids))
    if by_stage["fields"]:
        items = [f"{e.compound_id}: {e.reason}" for e in by_stage["fields"]]
        parts.append(_listing("fields cannot be computed for:", items))
    if by_stage["alignment"]:
        items = [f"{e.compound_id}: {e.reason}" for e in by_stage["alignment"]]
        parts.append(_listing("alignment check failed for:", items))
    return "\n".join(parts)


def load_calibration_inputs(ws: Workspace, cfg: ProjectConfig) -> CalibrationInputs:
    """Datasets -> poses -> split -> field atoms -> alignment check, under the qc policy."""
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

    warnings: list[str] = []
    poses = load_poses(cfg.alignment, ws.root, notes=warnings)

    excluded: dict[str, Exclusion] = {}  # one entry per compound: the first problem found
    atoms_by_id: dict[str, AtomData] = {}
    for compound_id in ids:
        if compound_id not in poses:
            excluded[compound_id] = Exclusion(compound_id, "pose", "no pose in the alignment file")
            continue
        try:
            atoms_by_id[compound_id] = atom_data_from_mol(
                poses[compound_id], cfg.chemistry.charge_model
            )
        except FieldError as err:
            excluded[compound_id] = Exclusion(compound_id, "fields", str(err))

    qc_rows = alignment_qc(cfg, ids, poses)
    for failure in alignment_failures(cfg, qc_rows):
        excluded.setdefault(failure.compound_id, failure)

    if excluded:
        problems = list(excluded.values())
        if cfg.qc.on_invalid == "error":
            raise PipelineError(_failure_message(problems))
        fraction = len(problems) / len(ids)
        if fraction > cfg.qc.max_skipped_fraction:
            raise PipelineError(
                f"{len(problems)} of {len(ids)} compounds ({fraction:.0%}) cannot be used, more "
                f"than qc.max_skipped_fraction ({cfg.qc.max_skipped_fraction:.0%}):\n"
                + _failure_message(problems)
            )
        kept = [c for c in ids if c not in excluded]
        activities = [a for c, a in zip(ids, activities, strict=True) if c not in excluded]
        ids = kept
        assignment = {c: assignment[c] for c in ids}
        warnings.append(
            f"{len(problems)} compound(s) skipped under qc.on_invalid: skip "
            "(each is recorded in excluded.csv)"
        )
    atoms = [atoms_by_id[c] for c in ids]
    return CalibrationInputs(
        ids, activities, assignment, atoms,
        excluded=list(excluded.values()), warnings=warnings, alignment_qc=qc_rows,
    )  # fmt: skip


def _write_qc_files(inputs: CalibrationInputs, out_dir: Path) -> list[str]:
    written = []
    if inputs.excluded:
        with open(out_dir / "excluded.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["compound_id", "stage", "reason"])
            writer.writerows([e.compound_id, e.stage, e.reason] for e in inputs.excluded)
        written.append("excluded.csv")
    if inputs.alignment_qc:
        with open(out_dir / "alignment_qc.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["compound_id", "core_atoms", "core_rmsd"])
            for row in inputs.alignment_qc:
                rmsd = "" if row["core_rmsd"] is None else f"{row['core_rmsd']:.4f}"
                writer.writerow([row["compound_id"], row["core_atoms"], rmsd])
        written.append("alignment_qc.csv")
    return written


def fit_project(ws: Workspace, cfg: ProjectConfig) -> FitResult:
    inputs = load_calibration_inputs(ws, cfg)
    # all the modelling happens in calibrate()
    result = calibrate(cfg, inputs.ids, inputs.activities, inputs.assignment, inputs.atoms)
    result.summary["excluded"] = [asdict(e) for e in inputs.excluded]
    result.summary["warnings"] = list(inputs.warnings)
    qc = qc_summary(inputs.alignment_qc)
    if qc is not None:
        result.summary["alignment_qc"] = qc

    run = create_run(ws, cfg)  # only now, so a failed fit leaves no run folder
    write_results(result, run.root)
    extra = _write_qc_files(inputs, run.root)
    run.add_to_manifest(
        "model",
        {
            "n_components": result.n_components,
            "n_train": result.summary["compounds"]["train"],
            "n_test": result.summary["compounds"]["test"],
            "n_excluded": len(inputs.excluded),
            "files": ["results.json", "predictions.csv", "split.csv", "model.npz", *extra],
        },
    )
    return FitResult(run=run, result=result)
