from types import SimpleNamespace

import pytest

from molecure.data.splits import SplitError, make_split, read_split_file, write_split

IDS = [f"c{i:02d}" for i in range(20)]


def cfg(method="random", test_fraction=0.25, file=None):
    """The three attributes make_split reads from a SplitConfig."""
    return SimpleNamespace(method=method, test_fraction=test_fraction, file=file)


def test_write_then_read_round_trip(tmp_path):
    assignment = {"a": "train", "b": "test", "c": "train"}
    write_split(tmp_path / "s.csv", assignment)
    assert read_split_file(tmp_path / "s.csv") == assignment


def test_from_file_follows_the_file_in_dataset_order(tmp_path):
    write_split(tmp_path / "s.csv", {i: ("test" if i in ("c03", "c07") else "train") for i in IDS})
    result = make_split(cfg("from_file", file="s.csv"), IDS, tmp_path, seed=1)
    assert list(result) == IDS  # dataset order, not file order
    assert [i for i, s in result.items() if s == "test"] == ["c03", "c07"]


def test_from_file_must_cover_exactly_the_dataset(tmp_path):
    write_split(tmp_path / "s.csv", {**{i: "train" for i in IDS[:-2]}, "stranger": "test"})
    with pytest.raises(SplitError) as info:
        make_split(cfg("from_file", file="s.csv"), IDS, tmp_path, seed=1)
    text = str(info.value)
    assert "stranger" in text and "c18" in text and "c19" in text


def test_bad_rows_are_all_reported(tmp_path):
    (tmp_path / "s.csv").write_text("compound_id,subset\na,train\nb,validation\n,test\na,test\n")
    with pytest.raises(SplitError) as info:
        read_split_file(tmp_path / "s.csv")
    text = str(info.value)
    assert "'validation'" in text and "empty compound_id" in text and "appears twice" in text


def test_missing_file_and_missing_columns(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        read_split_file(tmp_path / "nope.csv")
    (tmp_path / "s.csv").write_text("id,group\na,train\n")
    with pytest.raises(SplitError, match="needs the columns"):
        read_split_file(tmp_path / "s.csv")


def test_random_split_has_the_requested_size_and_is_reproducible(tmp_path):
    a = make_split(cfg(test_fraction=0.25), IDS, tmp_path, seed=3)
    b = make_split(cfg(test_fraction=0.25), IDS, tmp_path, seed=3)
    c = make_split(cfg(test_fraction=0.25), IDS, tmp_path, seed=4)
    assert a == b and a != c
    assert sum(s == "test" for s in a.values()) == 5


def test_zero_test_fraction_means_no_external_set(tmp_path):
    result = make_split(cfg(method="cluster", test_fraction=0.0), IDS, tmp_path, seed=1)
    assert set(result.values()) == {"train"}


def test_unimplemented_methods_and_silly_fractions_are_refused(tmp_path):
    with pytest.raises(SplitError, match="not implemented yet"):
        make_split(cfg(method="cluster", test_fraction=0.2), IDS, tmp_path, seed=1)
    with pytest.raises(SplitError, match="leaves"):
        make_split(cfg(test_fraction=0.45), IDS[:4], tmp_path, seed=1)  # 2 test, 2 train
