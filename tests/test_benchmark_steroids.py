"""Benchmark 0: the engine on the four steroid workspaces, compared with the published values.

Run `pytest -s tests/test_benchmark_steroids.py` to see the table of results.

The windows below are deliberately LOOSE. They catch a broken engine, not a small difference
from Sybyl: our probe, charges and truncation differ from the 1988 software. We tighten them
once real numbers exist. Reference values live in benchmarks/steroids/expected.yaml.
"""

import csv
import shutil

import numpy as np
import pytest

from molecure.model.pipeline import fit_project
from molecure.project import Workspace
from molecure.validate import metrics

NAMES = ("steroids_cbg", "steroids_tbg", "steroids_cbg_no_s02", "steroids_tbg_no_s02")
CBG = ("steroids_cbg", "steroids_cbg_no_s02")


def read_rows(run_root) -> list[dict]:
    with open(run_root / "predictions.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def external_without_t10(rows) -> dict:
    """External statistics without steroid 31 (T10, the 9a-fluoro outlier of every method)."""
    train = [float(r["observed"]) for r in rows if r["subset"] == "train"]
    test = [r for r in rows if r["subset"] == "test" and not r["compound_id"].startswith("T10")]
    observed = [float(r["observed"]) for r in test]
    predicted = [float(r["predicted"]) for r in test]
    return metrics.external_metrics(observed, predicted, train)


def _fmt(value) -> str:
    return f"{value:.3f}" if value is not None else "-"


def table(results) -> list[str]:
    head = f"{'workspace':22s} {'comp':>4} {'q2':>6} {'spress':>7} {'r2':>6} {'kfold':>6}"
    head += f" {'yrand max':>9} {'ext Q2F1':>9} {'w/o T10':>8}"
    lines = ["", "Benchmark 0 (leave-one-out q2 at the configured components)", head]
    for name, b in results.items():
        s = b["summary"]
        ext = s.get("external", {}).get("q2_f1")
        ext9 = external_without_t10(b["rows"])["q2_f1"] if "external" in s else None
        lines.append(
            f"{name:22s} {s['components']['used']:>4} {s['loo']['q2']:>6.3f} "
            f"{s['loo']['spress']:>7.3f} {s['fit']['r2']:>6.3f} {s['kfold']['q2_mean']:>6.3f} "
            f"{s['y_randomization']['q2_max']:>9.3f} {_fmt(ext):>9} {_fmt(ext9):>8}"
        )
    return lines


@pytest.fixture(scope="module")
def benchmark(steroid_workspaces, tmp_path_factory):
    work = tmp_path_factory.mktemp("benchmark")
    results = {}
    for name in NAMES:
        shutil.copytree(steroid_workspaces / name, work / name)
        ws = Workspace.open(work / name)
        fit = fit_project(ws, ws.load())
        results[name] = {"summary": fit.result.summary, "rows": read_rows(fit.run.root)}
    print("\n".join(table(results)))
    return results


def test_shuffled_activities_never_reach_the_cbg_models(benchmark):
    for name in CBG:
        s = benchmark[name]["summary"]
        assert s["y_randomization"]["runs_at_or_above_real_q2"] == 0, name
        assert s["loo"]["q2"] > s["y_randomization"]["q2_max"], name


def test_cbg_q2_is_inside_the_range_cramer_observed(benchmark):
    for name in CBG:
        s = benchmark[name]["summary"]
        # Cramer 1988 saw 0.559-0.779 over his lattice offsets; Coats got 0.665 / 0.708 (LOO)
        assert 0.45 <= s["loo"]["q2"] <= 0.85, (name, s["loo"]["q2"])
        best = int(np.argmax(s["loo"]["q2_by_components"])) + 1
        assert best in (2, 3, 4), (name, best)  # Coats: 3 components


def test_t10_is_the_worst_predicted_external_compound(benchmark):
    for name in CBG:
        test = [r for r in benchmark[name]["rows"] if r["subset"] == "test"]
        worst = max(test, key=lambda r: abs(float(r["predicted"]) - float(r["observed"])))
        assert worst["compound_id"].startswith("T10"), name  # every published method misses it


def test_cbg_external_set_is_predictive_without_the_outlier(benchmark):
    for name in CBG:
        stats = external_without_t10(benchmark[name]["rows"])
        assert stats["q2_f1"] > 0.3, (name, stats["q2_f1"])  # Cramer: 0.81 without 3 compounds


def test_tbg_beats_chance_at_the_five_percent_level(benchmark):
    for name in ("steroids_tbg", "steroids_tbg_no_s02"):
        s = benchmark[name]["summary"]
        runs = s["y_randomization"]["runs"]
        # the 1988 paper reports 0.555 and Coats 0.601; steric-only in my sandbox gave 0.22-0.27
        assert s["y_randomization"]["runs_at_or_above_real_q2"] <= 0.05 * runs, name
        assert s["loo"]["q2"] > 0.15, name
