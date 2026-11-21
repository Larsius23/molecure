"""Read dataset files into role-specific objects.

CSV rows and SDF records both become a _Record (molecule + text properties).
Everything after that is shared, so a calibration set behaves identically
whichever format it came from.

Loaders collect EVERY problem in a file and report them together, so you fix a
file once instead of once per error.
"""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from molecure.config import ProjectConfig
from molecure.config.datasets import CalibrationDataset, ReferenceDataset, ScreeningDataset

from .compounds import (
    CalibrationSet,
    Compound,
    LabeledCompound,
    LoadedDataset,
    ReferenceSet,
    ScreeningSet,
)
from .errors import DataError

# Every endpoint in the schema is a p-value (-log10 of a molar concentration).
# Values outside this range are almost certainly IC50 in nM pasted in by mistake.
P_SCALE = (0.0, 14.0)


@dataclass(frozen=True, eq=False)
class _Record:
    mol: Chem.Mol
    props: dict[str, str]  # CSV: every column of the row. SDF: every SD tag.
    origin: str


# ---------- reading files ----------


def _read_csv(path: Path, smiles_col: str, required: list[str], errors: list[str]) -> list[_Record]:
    try:
        # dtype=str: read everything as text and convert ourselves, so a bad number
        # is reported by row instead of silently becoming NaN.
        # utf-8-sig: transparently strips the invisible marker Excel adds to CSVs.
        table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        errors.append(f"cannot read as CSV: {err}")
        return []

    table.columns = [str(c).strip() for c in table.columns]
    absent = [c for c in required if c not in table.columns]
    if absent:
        errors.append(f"column(s) not found: {absent}; the file has: {list(table.columns)}")
        return []

    records = []
    with rdBase.BlockLogs():  # silence RDKit's own stderr noise; we report cleanly instead
        for i, row in enumerate(table.to_dict("records")):
            origin = f"{path.name} line {i + 2}"  # header is line 1, so data starts at 2
            smiles = row[smiles_col].strip()
            if not smiles:
                errors.append(f"{origin}: empty SMILES")
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors.append(f"{origin}: SMILES not parseable: {smiles}")
                continue
            props = {key: str(value).strip() for key, value in row.items()}
            records.append(_Record(mol, props, origin))
    return records


def _read_sdf(path: Path, errors: list[str]) -> list[_Record]:
    records = []
    with rdBase.BlockLogs():
        # removeHs=False: keep explicit hydrogens. The fields need them.
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        for i, mol in enumerate(supplier, start=1):
            origin = f"{path.name} record {i}"
            if mol is None:
                errors.append(f"{origin}: could not be parsed or sanitized")
                continue
            props = {name: mol.GetProp(name).strip() for name in mol.GetPropNames()}
            records.append(_Record(mol, props, origin))
    return records


def _read_records(
    path: Path, smiles_col: str, required: list[str], errors: list[str]
) -> list[_Record]:
    if not path.is_file():
        errors.append(f"file not found: {path}")
        return []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path, smiles_col, required, errors)
    if suffix == ".sdf":
        return _read_sdf(path, errors)
    errors.append(f"unsupported file type '{path.suffix}' (use .csv or .sdf)")
    return []


# ---------- checks shared by the roles ----------


def _ids(records: list[_Record], id_col: str, errors: list[str]) -> list[str]:
    ids = [rec.props.get(id_col, "") for rec in records]
    for rec, value in zip(records, ids, strict=True):
        if not value:
            errors.append(f"{rec.origin}: no value in '{id_col}'")
    repeated = sorted(i for i, n in Counter(ids).items() if i and n > 1)
    if repeated:
        errors.append(f"duplicate ids (each compound needs a unique id): {repeated}")
    return ids


def _activity(rec: _Record, col: str, endpoint: str, errors: list[str]) -> float:
    raw = rec.props.get(col, "")
    if raw == "":
        errors.append(f"{rec.origin}: no value in activity column '{col}'")
        return math.nan
    try:
        value = float(raw)
    except ValueError:
        errors.append(
            f"{rec.origin}: activity '{raw}' is not a plain number "
            "(censored values such as '>10000' must be removed first)"
        )
        return math.nan
    if not math.isfinite(value):
        errors.append(f"{rec.origin}: activity '{raw}' is not a finite number")
        return math.nan
    lo, hi = P_SCALE
    if not lo <= value <= hi:
        errors.append(
            f"{rec.origin}: {endpoint} = {value} is outside {lo} to {hi}; activity must be "
            "on the p-scale (-log10 of molar), not a raw IC50 in nM"
        )
        return math.nan
    return value


def _is_3d(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    z = mol.GetConformer().GetPositions()[:, 2]
    return bool(np.abs(z).max() > 1e-6)  # a 2D drawing has z = 0 for every atom


# ---------- one loader per role ----------


def load_calibration(cfg: CalibrationDataset, root: Path) -> CalibrationSet:
    source = f"dataset '{cfg.id}' ({cfg.file})"
    errors: list[str] = []
    required = [cfg.id_col, cfg.smiles_col, cfg.activity_col]
    records = _read_records(root / cfg.file, cfg.smiles_col, required, errors)
    if not records:
        raise DataError(source, errors or ["the file contains no compounds"])

    ids = _ids(records, cfg.id_col, errors)
    activities = [_activity(r, cfg.activity_col, cfg.endpoint, errors) for r in records]
    if errors:
        raise DataError(source, errors)

    compounds = tuple(
        LabeledCompound(id=i, mol=r.mol, origin=r.origin, activity=a)
        for r, i, a in zip(records, ids, activities, strict=True)
    )
    return CalibrationSet(id=cfg.id, endpoint=cfg.endpoint, compounds=compounds)


def load_screening(cfg: ScreeningDataset, root: Path) -> ScreeningSet:
    source = f"dataset '{cfg.id}' ({cfg.file})"
    errors: list[str] = []
    records = _read_records(root / cfg.file, cfg.smiles_col, [cfg.id_col, cfg.smiles_col], errors)
    if not records:
        raise DataError(source, errors or ["the file contains no compounds"])

    ids = _ids(records, cfg.id_col, errors)
    if errors:
        raise DataError(source, errors)

    # Whatever activity-like columns the file contains are never read.
    compounds = tuple(
        Compound(id=i, mol=r.mol, origin=r.origin) for r, i in zip(records, ids, strict=True)
    )
    return ScreeningSet(id=cfg.id, compounds=compounds)


def load_reference(cfg: ReferenceDataset, root: Path) -> ReferenceSet:
    source = f"dataset '{cfg.id}' ({cfg.file})"
    path = root / cfg.file
    if path.suffix.lower() != ".sdf":
        raise DataError(source, ["the reference must be an .sdf file with a 3D pose"])

    errors: list[str] = []
    records = _read_records(path, "smiles", [], errors)
    if not records:
        raise DataError(source, errors or ["the file contains no molecules"])
    if len(records) > 1:
        errors.append(f"expected exactly one molecule, found {len(records)}")
    if not _is_3d(records[0].mol):
        errors.append("the molecule has no 3D coordinates (every z value is zero)")
    if errors:
        raise DataError(source, errors)

    compound = Compound(id=cfg.id, mol=records[0].mol, origin=records[0].origin)
    return ReferenceSet(id=cfg.id, compound=compound)


def load_dataset(cfg: CalibrationDataset | ScreeningDataset | ReferenceDataset, root: Path):
    """Load one dataset according to its role."""
    if isinstance(cfg, CalibrationDataset):
        return load_calibration(cfg, root)
    if isinstance(cfg, ScreeningDataset):
        return load_screening(cfg, root)
    return load_reference(cfg, root)


def load_all(cfg: ProjectConfig, root: Path) -> dict[str, LoadedDataset]:
    """Load every dataset in the project. Problems from ALL datasets come back together."""
    loaded: dict[str, LoadedDataset] = {}
    problems: list[str] = []
    for ds in cfg.datasets:
        try:
            loaded[ds.id] = load_dataset(ds, root)
        except DataError as err:
            problems.extend(f"[{ds.id}] {p}" for p in err.problems)
    if problems:
        raise DataError("data files", problems)
    return loaded
