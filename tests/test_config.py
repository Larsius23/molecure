"""Valid configs load; invalid ones fail loudly."""

import pytest
from pydantic import ValidationError

from molecure.config import ProjectConfig, dump_config, load_config


def make_config() -> dict:
    """A minimal valid config, as the plain dict that YAML would give us."""
    return {
        "schema_version": 1,
        "project": {"name": "demo"},
        "datasets": [
            {
                "id": "train",
                "role": "calibration",
                "file": "data/train.csv",
                "activity_col": "pIC50",
                "endpoint": "pIC50",
            },
            {"id": "ref", "role": "reference", "file": "data/ref.sdf"},
            {"id": "screen", "role": "screening", "file": "data/screen.csv"},
        ],
        "alignment": {"mode": "mcs", "reference": "ref"},
        "molecular_fields": {"method": "comfa"},
    }


def test_minimal_config_loads_and_fills_defaults():
    cfg = ProjectConfig.model_validate(make_config())
    assert cfg.chemistry.protonation.ph == 7.4
    assert cfg.chemistry.charge_model == "mmff94"
    assert cfg.molecular_fields.method == "comfa"
    assert cfg.molecular_fields.grid.spacing == 2.0
    assert cfg.runtime.seed == 42


def test_comsia_has_its_own_grid_default():
    raw = make_config()
    raw["molecular_fields"] = {"method": "comsia"}
    cfg = ProjectConfig.model_validate(raw)
    assert cfg.molecular_fields.grid.spacing == 1.0
    assert cfg.molecular_fields.attenuation == 0.3


def test_typo_in_a_key_is_rejected():
    raw = make_config()
    raw["chemistry"] = {"protonation": {"phh": 7.4}}
    with pytest.raises(ValidationError, match="phh"):
        ProjectConfig.model_validate(raw)


def test_screening_dataset_cannot_carry_activity():
    raw = make_config()
    raw["datasets"][2]["activity_col"] = "pIC50"
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_calibration_dataset_requires_activity():
    raw = make_config()
    del raw["datasets"][0]["activity_col"]
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_duplicate_dataset_ids_rejected():
    raw = make_config()
    raw["datasets"][2]["id"] = "train"
    with pytest.raises(ValidationError, match="unique"):
        ProjectConfig.model_validate(raw)


def test_alignment_reference_must_be_a_reference_dataset():
    raw = make_config()
    raw["alignment"]["reference"] = "train"
    with pytest.raises(ValidationError, match="reference"):
        ProjectConfig.model_validate(raw)


def test_prealigned_needs_no_reference():
    raw = make_config()
    raw["alignment"] = {"mode": "prealigned", "file": "data/poses.sdf"}
    cfg = ProjectConfig.model_validate(raw)
    assert cfg.alignment.mode == "prealigned"


def test_best_score_needs_score_property():
    raw = make_config()
    raw["alignment"] = {"mode": "prealigned", "file": "p.sdf", "pose_rule": "best_score"}
    with pytest.raises(ValidationError, match="score_property"):
        ProjectConfig.model_validate(raw)


def test_split_from_file_needs_file():
    raw = make_config()
    raw["validation"] = {"split": {"method": "from_file"}}
    with pytest.raises(ValidationError, match="from_file"):
        ProjectConfig.model_validate(raw)


def test_yaml_round_trip(tmp_path):
    cfg = ProjectConfig.model_validate(make_config())
    path = tmp_path / "project.yaml"
    dump_config(cfg, path)
    assert load_config(path) == cfg


def test_comfa_defaults_follow_the_standard_settings():
    cfg = ProjectConfig.model_validate(make_config())
    fields = cfg.molecular_fields
    assert fields.probe_charge == 1.0
    assert fields.energy_cutoff_kcal == 30.0
    assert fields.dielectric == "distance"


def test_comfa_has_no_force_field_option_any_more():
    raw = make_config()
    raw["molecular_fields"] = {"method": "comfa", "force_field": "mmff94"}
    with pytest.raises(ValidationError, match="force_field"):
        ProjectConfig.model_validate(raw)


def test_squared_correlation_is_an_accepted_metric():
    raw = make_config()
    raw["validation"] = {"metrics": ["q2", "r2_corr"]}
    assert ProjectConfig.model_validate(raw).validation.metrics == ("q2", "r2_corr")
