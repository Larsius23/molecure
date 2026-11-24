"""Sensitivity sweeps, tested with hand-made molecules (no RDKit needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from molecure.config.fields import ComfaFields, ComsiaFields, GridConfig
from molecure.fields.atomdata import AtomData
from molecure.model.calibrate import calibrate
from molecure.validate.sensitivity import (
    Variation,
    format_sweep,
    lattice_offsets,
    run_variation,
    setting_variations,
    summarize,
    sweep,
)


def molecule(x: float) -> AtomData:
    """A fixed three-atom scaffold plus one extra atom at distance x: x sets the activity."""
    coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [x, 0.0, 1.0]]
    n = 4
    return AtomData(
        coords=np.array(coords),
        charges=np.array([-0.3, 0.3, 0.1, -0.1]),
        vdw_x=np.full(n, 3.851),
        vdw_d=np.full(n, 0.105),
        radii=np.full(n, 1.7),
        logp=np.zeros(n),
        donor=np.zeros(n),
        acceptor=np.zeros(n),
    )


def make_cfg(n_components=2, method="comfa"):
    fields = (
        ComfaFields(method="comfa", grid=GridConfig(spacing=2.0, padding=4.0))
        if method == "comfa"
        else ComsiaFields(method="comsia", grid=GridConfig(spacing=2.0, padding=4.0))
    )
    return SimpleNamespace(
        molecular_fields=fields,
        pls=SimpleNamespace(
            min_sigma=0.0, scaling="none", max_components=4,
            n_components=n_components, selection="min_press",
        ),
        validation=SimpleNamespace(
            cv=SimpleNamespace(loo=True, group_kfold_k=4, group_kfold_repeats=2),
            y_randomization_runs=0,
            metrics=("q2",),
        ),
        runtime=SimpleNamespace(seed=7),
    )  # fmt: skip


def make_dataset():
    rng = np.random.default_rng(0)
    xs = np.linspace(2.0, 6.0, 16)
    ids = [f"m{i:02d}" for i in range(16)]
    y = list(0.5 * xs + 0.02 * rng.normal(size=16))
    assignment = {c: ("test" if i in (3, 7, 11, 14) else "train") for i, c in enumerate(ids)}
    return ids, y, assignment, [molecule(x) for x in xs]


def test_the_unvaried_run_agrees_with_the_normal_leave_one_out():
    ids, y, assignment, atoms = make_dataset()
    cfg = make_cfg()
    plain = run_variation(cfg, ids, y, assignment, atoms, Variation("as configured"))
    full = calibrate(cfg, ids, y, assignment, atoms).summary["loo"]["q2"]
    assert plain["q2"] == pytest.approx(full, abs=1e-9)


def test_cramers_eight_lattice_offsets_in_units_of_the_spacing():
    offsets = lattice_offsets()
    assert len(offsets) == 8 and len({v.grid_shift for v in offsets}) == 8
    assert offsets[0].grid_shift == (0.0, 0.0, 0.0)
    assert offsets[-1].grid_shift == (-0.5, -0.5, -0.5)  # his (-1, -1, -1) A at 2 A spacing


def test_shifting_the_grid_moves_the_numbers_but_not_the_signal():
    ids, y, assignment, atoms = make_dataset()
    rows = sweep(make_cfg(), ids, y, assignment, atoms, lattice_offsets())
    q2 = [r["q2"] for r in rows]
    assert min(q2) > 0.7  # a strong signal survives any grid position
    assert len({round(v, 6) for v in q2}) > 1  # but the grid position is not irrelevant
    assert all(r["n_points"] == rows[0]["n_points"] for r in rows)  # same grid, just moved


def test_overrides_take_effect():
    ids, y, assignment, atoms = make_dataset()
    cfg = make_cfg()
    base = run_variation(cfg, ids, y, assignment, atoms, Variation("base"))
    finer = run_variation(cfg, ids, y, assignment, atoms, Variation("fine", grid_spacing=1.0))
    assert finer["n_points"] > base["n_points"]
    constant = Variation("constant", fields={"dielectric": "constant"})
    assert run_variation(cfg, ids, y, assignment, atoms, constant)["q2"] != base["q2"]
    autoscale = Variation("autoscale", scaling="autoscale", min_sigma=1.0)
    scaled = run_variation(cfg, ids, y, assignment, atoms, autoscale)
    assert scaled["n_columns"] == base["n_columns"] and scaled["q2"] != base["q2"]


def test_components_are_chosen_per_variation_when_not_fixed():
    ids, y, assignment, atoms = make_dataset()
    cfg = make_cfg(n_components=None)
    row = run_variation(cfg, ids, y, assignment, atoms, Variation("base"))
    assert 1 <= row["components"] <= 4 and len(row["q2_by_components"]) == 4


def test_setting_variations_depend_on_the_field_method():
    comfa = [v.label for v in setting_variations(make_cfg())]
    comsia = [v.label for v in setting_variations(make_cfg(method="comsia"))]
    assert "constant dielectric" in comfa and "constant dielectric" not in comsia
    assert "grid spacing 1 A" in comfa and "scaling block, sigma >= 2" in comfa
    assert "probe radius 1.52 A" in comfa and "probe radius 1.52 A" not in comsia
    assert len(comfa) == 20 and len(comsia) == 9


def test_summary_and_formatting():
    def row(label, q2, spress):
        return {"label": label, "components": 2, "q2": q2, "spress": spress, "n_columns": 10}

    rows = [row("a", 0.30, 1.0), row("b", 0.50, 0.9)]
    s = summarize(rows)
    assert s["min"] == 0.30 and s["max"] == 0.50 and s["mean"] == pytest.approx(0.40)
    text = "\n".join(format_sweep("Title", rows, with_range=True))
    assert "range 0.300 to 0.500" in text and "spread 0.200" in text
