import pytest

from molecure.config.datasets import CalibrationDataset, ReferenceDataset, ScreeningDataset
from molecure.data import Compound, DataError, load_all, load_dataset


def calibration_cfg(file: str, **kw) -> CalibrationDataset:
    return CalibrationDataset(
        id="train", role="calibration", file=file, activity_col="pIC50", endpoint="pIC50", **kw
    )


def test_calibration_csv_loads(tmp_path):
    (tmp_path / "t.csv").write_text("compound_id,smiles,pIC50\na,CCO,5.5\nb,CCN,6.5\n")
    result = load_dataset(calibration_cfg("t.csv"), tmp_path)
    assert result.ids == ("a", "b")
    assert result.activities == (5.5, 6.5)
    assert result.summary()["activity_range"] == 1.0


def test_every_problem_is_reported_at_once(tmp_path):
    (tmp_path / "bad.csv").write_text(
        "compound_id,smiles,pIC50\na,CCO,6.5\nb,C(C,7.0\nc,CCN,>10000\nd,CCC,650\na,CCCC,5.5\n"
    )
    with pytest.raises(DataError) as info:
        load_dataset(calibration_cfg("bad.csv"), tmp_path)
    text = str(info.value)
    assert "line 3: SMILES not parseable" in text
    assert "'>10000' is not a plain number" in text
    assert "outside 0.0 to 14.0" in text
    assert "duplicate ids" in text


def test_missing_activity_column_is_named(tmp_path):
    (tmp_path / "t.csv").write_text("compound_id,smiles\na,CCO\n")
    with pytest.raises(DataError, match="pIC50"):
        load_dataset(calibration_cfg("t.csv"), tmp_path)


def test_screening_set_has_no_activities(tmp_path):
    (tmp_path / "s.csv").write_text("compound_id,smiles,pIC50\nx1,CCO,9.9\nx2,CCN,9.9\n")
    cfg = ScreeningDataset(id="screen", role="screening", file="s.csv")
    result = load_dataset(cfg, tmp_path)
    assert not hasattr(result, "activities")
    assert all(type(c) is Compound for c in result.compounds)  # plain, never labeled


def test_calibration_from_sdf_reads_tags(tmp_path, write_sdf):
    write_sdf(
        tmp_path / "set.sdf",
        [
            ("CCO", {"compound_id": "s1", "pIC50": 5.2}),
            ("CCCO", {"compound_id": "s2", "pIC50": 6.1}),
        ],
    )
    result = load_dataset(calibration_cfg("set.sdf"), tmp_path)
    assert result.ids == ("s1", "s2")
    assert result.activities == (5.2, 6.1)


def test_sdf_record_without_id_tag_is_reported(tmp_path, write_sdf):
    write_sdf(tmp_path / "set.sdf", [("CCO", {"pIC50": 5})])
    with pytest.raises(DataError, match="no value in 'compound_id'"):
        load_dataset(calibration_cfg("set.sdf"), tmp_path)


def test_reference_loads_with_explicit_hydrogens(tmp_path, write_sdf):
    write_sdf(tmp_path / "ref.sdf", [("CCCO", {})])
    cfg = ReferenceDataset(id="ref", role="reference", file="ref.sdf")
    result = load_dataset(cfg, tmp_path)
    assert result.compound.mol.GetNumConformers() == 1
    assert result.summary()["has_explicit_h"] is True


def test_reference_must_be_3d(tmp_path, write_sdf):
    write_sdf(tmp_path / "flat.sdf", [("CCO", {})], three_d=False)
    cfg = ReferenceDataset(id="ref", role="reference", file="flat.sdf")
    with pytest.raises(DataError, match="3D"):
        load_dataset(cfg, tmp_path)


def test_reference_must_be_a_single_molecule(tmp_path, write_sdf):
    write_sdf(tmp_path / "two.sdf", [("CCO", {}), ("CCC", {})])
    cfg = ReferenceDataset(id="ref", role="reference", file="two.sdf")
    with pytest.raises(DataError, match="exactly one"):
        load_dataset(cfg, tmp_path)


def test_unsupported_file_type(tmp_path):
    (tmp_path / "data.xlsx").write_bytes(b"x")
    with pytest.raises(DataError, match="unsupported"):
        load_dataset(calibration_cfg("data.xlsx"), tmp_path)


def test_load_all_returns_one_object_per_dataset(demo_project):
    loaded = load_all(demo_project.load(), demo_project.root)
    assert set(loaded) == {"calibration_set", "reference_ligand"}
    assert loaded["calibration_set"].summary()["n_compounds"] == 3


def test_load_all_reports_problems_from_every_dataset(demo_project):
    root = demo_project.root
    (root / "data/curated/calibration.csv").write_text("compound_id,smiles,pIC50\nc1,C(C,5.5\n")
    (root / "data/reference/reference.sdf").unlink()
    with pytest.raises(DataError) as info:
        load_all(demo_project.load(), root)
    text = str(info.value)
    assert "[calibration_set]" in text
    assert "[reference_ligand]" in text
