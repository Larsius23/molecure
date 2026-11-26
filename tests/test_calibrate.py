"""The numeric core of the pipeline, tested with hand-made molecules (no RDKit needed)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from molecure.config.fields import ComfaFields, GridConfig
from molecure.fields.atomdata import AtomData
from molecure.fields.calculator import FieldCalculator
from molecure.fields.grid import Grid
from molecure.model.calibrate import calibrate, format_summary, write_results
from molecure.model.pls import FittedPLS, ModelError


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


def make_cfg(metrics=("q2", "r2", "q2_f1", "ccc", "r2_corr")):
    """The attributes calibrate() reads from a ProjectConfig."""
    return SimpleNamespace(
        molecular_fields=ComfaFields(method="comfa", grid=GridConfig(spacing=2.0, padding=4.0)),
        pls=SimpleNamespace(
            min_sigma=0.0, scaling="none", max_components=4, n_components=2, selection="min_press"
        ),
        validation=SimpleNamespace(
            cv=SimpleNamespace(loo=True, group_kfold_k=4, group_kfold_repeats=3),
            y_randomization_runs=5,
            metrics=metrics,
        ),
        runtime=SimpleNamespace(seed=7),
    )


def make_dataset(test_positions=(3, 7, 11, 14)):
    rng = np.random.default_rng(0)
    xs = np.linspace(2.0, 6.0, 16)
    ids = [f"m{i:02d}" for i in range(16)]
    y = list(0.5 * xs + 0.02 * rng.normal(size=16))
    atoms = [molecule(x) for x in xs]
    assignment = {c: ("test" if i in test_positions else "train") for i, c in enumerate(ids)}
    return ids, y, assignment, atoms


def test_the_model_finds_the_signal_and_reports_every_section():
    ids, y, assignment, atoms = make_dataset()
    result = calibrate(make_cfg(), ids, y, assignment, atoms)
    s = result.summary
    assert s["compounds"] == {"train": 12, "test": 4}
    assert s["loo"]["q2"] > 0.8 and s["fit"]["r2"] > 0.9
    assert s["external"]["n"] == 4 and s["kfold"]["repeats"] == 3
    assert s["y_randomization"]["runs_at_or_above_real_q2"] == 0
    assert s["fields"] == {"method": "comfa", "blocks": ["steric", "electrostatic"]}
    assert s["requested_metrics"] == ["q2", "r2", "q2_f1", "ccc", "r2_corr"] and s["seed"] == 7
    json.dumps(s)


def test_the_grid_comes_from_the_training_compounds_only():
    ids, y, assignment, atoms = make_dataset()
    atoms[14] = molecule(2.0).translated((40.0, 0.0, 0.0))  # an external compound far away
    result = calibrate(make_cfg(), ids, y, assignment, atoms)
    train_coords = [a.coords for a, c in zip(atoms, ids, strict=True) if assignment[c] == "train"]
    expected = Grid.from_coords(train_coords, spacing=2.0, padding=4.0)
    assert tuple(result.summary["grid"]["shape"]) == expected.shape
    assert result.summary["outside_grid"] == ["m14"]  # flagged, and it did not move the grid


def test_rows_follow_the_dataset_order_and_the_roles():
    ids, y, assignment, atoms = make_dataset()
    rows = calibrate(make_cfg(), ids, y, assignment, atoms).rows
    assert [r["compound_id"] for r in rows] == ids
    assert [r["observed"] for r in rows] == pytest.approx(y)
    for row in rows:
        if row["subset"] == "train":
            assert row["fitted"] is not None and row["cv_loo"] is not None
            assert row["predicted"] is None
        else:
            assert row["predicted"] is not None
            assert row["fitted"] is None and row["cv_loo"] is None


def test_files_are_written(tmp_path):
    ids, y, assignment, atoms = make_dataset()
    result = calibrate(make_cfg(), ids, y, assignment, atoms)
    write_results(result, tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["loo"]["q2"] == pytest.approx(result.summary["loo"]["q2"])
    lines = (tmp_path / "predictions.csv").read_text().splitlines()
    assert lines[0] == "compound_id,subset,observed,fitted,cv_loo,predicted,outside_grid"
    assert len(lines) == 17
    assert (tmp_path / "split.csv").read_text().count("test") == 4
    assert set(np.load(tmp_path / "model.npz")) >= {"weights", "coefficients", "grid_origin"}


def test_the_saved_model_reproduces_the_predictions(tmp_path):
    # This is exactly what screening will do later: new molecules, the saved grid, the saved model.
    ids, y, assignment, atoms = make_dataset()
    cfg = make_cfg()
    result = calibrate(cfg, ids, y, assignment, atoms)
    write_results(result, tmp_path)
    saved = np.load(tmp_path / "model.npz")
    grid = Grid(
        origin=tuple(float(v) for v in saved["grid_origin"]),
        spacing=float(saved["grid_spacing"]),
        shape=tuple(int(v) for v in saved["grid_shape"]),
    )
    X = FieldCalculator(grid, cfg.molecular_fields).compute_matrix(ids, atoms).values
    from_coefficients = X @ saved["coefficients"] + float(saved["intercept"])
    model = FittedPLS.from_arrays(saved)
    from_model = model.predict(X, int(saved["n_components"]))
    assert np.allclose(from_coefficients, from_model, atol=1e-8)
    for value, row in zip(from_model, result.rows, strict=True):
        expected = row["fitted"] if row["subset"] == "train" else row["predicted"]
        assert value == pytest.approx(expected)


def test_nan_statistics_are_written_as_null(tmp_path):
    ids, y, _, atoms = make_dataset()
    assignment = {c: ("test" if c in ("m03", "m07") else "train") for c in ids}  # 2 external only
    result = calibrate(make_cfg(), ids, y, assignment, atoms)
    write_results(result, tmp_path)
    text = (tmp_path / "results.json").read_text()
    assert "NaN" not in text and json.loads(text)["external"]["r2_corr"] is None


def test_selecting_components_adds_the_nested_estimate():
    ids, y, assignment, atoms = make_dataset()
    cfg = make_cfg()
    cfg.pls.n_components = None
    result = calibrate(cfg, ids, y, assignment, atoms)
    assert result.summary["components"]["rule"] == "min_press"
    assert "nested_loo" in result.summary


def test_problems_are_refused_clearly():
    ids, y, assignment, atoms = make_dataset()
    with pytest.raises(ModelError, match="rm2"):
        calibrate(make_cfg(metrics=("q2", "rm2")), ids, y, assignment, atoms)
    with pytest.raises(ModelError, match="at least 3"):
        few = {c: ("train" if i < 2 else "test") for i, c in enumerate(ids)}
        calibrate(make_cfg(), ids, y, few, atoms)
    with pytest.raises(ModelError, match="differ"):
        calibrate(make_cfg(), ids, y[:-1], assignment, atoms)


def test_format_summary_mentions_the_key_numbers():
    ids, y, assignment, atoms = make_dataset()
    atoms[14] = molecule(2.0).translated((40.0, 0.0, 0.0))
    text = "\n".join(format_summary(calibrate(make_cfg(), ids, y, assignment, atoms).summary))
    for needle in ("12 training, 4 external", "leave-one-out", "y-randomization", "external (4)"):
        assert needle in text
    assert "WARNING: 1 compound(s) outside the grid: m14" in text


def test_format_summary_shows_warnings_exclusions_and_the_alignment_check():
    ids, y, assignment, atoms = make_dataset()
    summary = calibrate(make_cfg(), ids, y, assignment, atoms).summary
    summary["excluded"] = [{"compound_id": "x1", "stage": "pose", "reason": "no pose"}]
    summary["warnings"] = ["be careful"]
    summary["alignment_qc"] = {"median": 0.25, "max": 0.9, "worst": "m05"}
    text = "\n".join(format_summary(summary))
    assert "WARNING: 1 compound(s) excluded (see excluded.csv)" in text
    assert "WARNING: be careful" in text
    assert "median 0.25, max 0.90 A (m05)" in text
