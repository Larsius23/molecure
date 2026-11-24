"""Sensitivity analysis: how much do the results depend on arbitrary choices?

CoMFA models are known to move when the grid is shifted or re-spaced, and with the scaling
choices. Cramer 1988 saw the q2 of one target swing between 0.50 and 0.71 from grid position
alone. A single q2 hides that; a sweep shows it.

Each variation recomputes the fields and the leave-one-out q2 on the TRAINING compounds only
(the external set is not used), changing exactly one thing: the grid position, the grid spacing,
the scaling, the sigma filter, or a field setting.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from molecure.fields.calculator import FieldCalculator
from molecure.model.pls import Recipe

from . import metrics
from .cv import choose_components, cv_predictions, loo_folds, q2_curve

# The probe radii of Cramer 1988, Table III (H, O, N, C, S, P, I, Ca), in angstrom.
PROBE_RADII = (1.08, 1.36, 1.45, 1.52, 1.72, 1.75, 2.05, 2.75)


@dataclass(frozen=True)
class Variation:
    label: str
    grid_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)  # in units of the grid spacing
    grid_spacing: float | None = None  # angstrom
    fields: Mapping[str, object] = field(default_factory=dict)  # other molecular_fields settings
    scaling: str | None = None
    min_sigma: float | None = None


def _replace(settings, **changes):
    """A copy of a settings object with some attributes changed (pydantic model or dataclass)."""
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=changes)
    return dataclasses.replace(settings, **changes)


def run_variation(cfg, ids, y, assignment, atoms, variation: Variation) -> dict:
    """Leave-one-out q2 of one variation. `cfg` supplies everything that is not varied."""
    ids = list(ids)
    y = np.asarray(y, dtype=float)
    train = [i for i, c in enumerate(ids) if assignment[c] == "train"]

    settings = cfg.molecular_fields
    if variation.fields:
        settings = _replace(settings, **variation.fields)
    if variation.grid_spacing is not None:
        settings = _replace(settings, grid=_replace(settings.grid, spacing=variation.grid_spacing))
    train_atoms = [atoms[i] for i in train]
    base = FieldCalculator.fit(train_atoms, settings)
    shift = tuple(fraction * base.grid.spacing for fraction in variation.grid_shift)
    calculator = FieldCalculator(base.grid.shifted(shift), settings)
    matrix = calculator.compute_matrix([ids[i] for i in train], train_atoms)

    recipe = Recipe(
        min_sigma=cfg.pls.min_sigma if variation.min_sigma is None else variation.min_sigma,
        scaling=variation.scaling or cfg.pls.scaling,
        blocks=np.repeat(np.arange(len(matrix.block_names)), matrix.n_points),
    )
    y_train = y[train]
    most = min(cfg.pls.max_components, len(train) - 2)
    predictions = cv_predictions(matrix.values, y_train, recipe, most, loo_folds(len(train)))
    if cfg.pls.n_components is not None:
        k = min(cfg.pls.n_components, most)
    else:
        k = choose_components(y_train, predictions, cfg.pls.selection)
    at_k = predictions[:, k - 1]
    return {
        "label": variation.label,
        "components": int(k),
        "q2": metrics.q2(y_train, at_k),
        "spress": metrics.spress(y_train, at_k, k),
        "q2_by_components": [float(v) for v in q2_curve(y_train, predictions)],
        "n_points": int(matrix.n_points),
        "n_columns": int(matrix.values.shape[1]),
    }


def sweep(cfg, ids, y, assignment, atoms, variations: Sequence[Variation]) -> list[dict]:
    return [run_variation(cfg, ids, y, assignment, atoms, v) for v in variations]


def lattice_offsets() -> list[Variation]:
    """The eight lattice offsets of Cramer 1988 (his Table IV: 0, -0.5 and -1.0 A at a 2 A
    spacing), written as fractions of the spacing so they suit any grid."""
    steps = [
        (0.0, 0.0, 0.0), (-0.25, 0.0, 0.0), (0.0, 0.0, -0.25), (-0.25, -0.25, 0.0),
        (-0.25, -0.25, -0.25), (-0.5, -0.5, 0.0), (-0.5, -0.5, -0.25), (-0.5, -0.5, -0.5),
    ]  # fmt: skip
    return [
        Variation(label=f"offset {x:+.2f} {y:+.2f} {z:+.2f} x spacing", grid_shift=(x, y, z))
        for x, y, z in steps
    ]


def setting_variations(cfg) -> list[Variation]:
    """Spacing, scaling, sigma filter and (for CoMFA) dielectric and cutoff, one at a time."""
    variations = [Variation("as configured")]
    variations += [Variation(f"grid spacing {s:g} A", grid_spacing=s) for s in (1.0, 3.0)]
    for scaling in ("none", "autoscale", "block"):
        for sigma in (0.0, 2.0):
            label = f"scaling {scaling}, sigma >= {sigma:g}"
            variations.append(Variation(label, scaling=scaling, min_sigma=sigma))
    if cfg.molecular_fields.method == "comfa":
        variations.append(Variation("constant dielectric", fields={"dielectric": "constant"}))
        for cutoff in (10.0, 100.0):
            variations.append(
                Variation(f"cutoff {cutoff:g} kcal/mol", fields={"energy_cutoff_kcal": cutoff})
            )
        for radius in PROBE_RADII:
            variations.append(
                Variation(f"probe radius {radius:g} A", fields={"probe_radius": radius})
            )
    return variations


def summarize(rows: Sequence[dict]) -> dict:
    q2 = np.array([r["q2"] for r in rows])
    return {
        "min": float(q2.min()),
        "max": float(q2.max()),
        "mean": float(q2.mean()),
        "sd": float(q2.std(ddof=1)) if len(q2) > 1 else 0.0,
    }


def format_sweep(title: str, rows: Sequence[dict], with_range: bool = False) -> list[str]:
    lines = ["", title]
    for r in rows:
        lines.append(
            f"  {r['label']:36s} {r['components']} comp   q2 {r['q2']:6.3f}   "
            f"spress {r['spress']:6.3f}   ({r['n_columns']} columns)"
        )
    if with_range and rows:
        s = summarize(rows)
        spread = s["max"] - s["min"]
        lines.append(
            f"  range {s['min']:.3f} to {s['max']:.3f}   mean {s['mean']:.3f}   "
            f"sd {s['sd']:.3f}   spread {spread:.3f}"
        )
    return lines
