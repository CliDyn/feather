"""Unit tests for the CMIP6 pool DRS-discovery helpers."""

from pathlib import Path

import pytest

from feather.data import pool_discovery as pd


def _make_file(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


@pytest.fixture
def drs_tree(tmp_path):
    """Build a small synthetic CMIP6 DRS tree under tmp_path/CMIP."""
    root = tmp_path / "CMIP"
    # MPI-M / MPI-ESM1-2-LR / historical with two members, gn grid, two versions
    base = root / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
    for mem in ("r1i1p1f1", "r2i1p1f1"):
        for ver in ("v20190101", "v20200101"):
            _make_file(
                base / mem / "Amon" / "tas" / "gn" / ver
                / f"tas_Amon_MPI-ESM1-2-LR_historical_{mem}_gn_185001-201412.nc"
            )
    # AWI model that only has r3 and r10 (numeric ordering check), and both
    # gr (preferred) and gn grids for tas
    awi = root / "AWI" / "AWI-CM-1-1-MR" / "historical"
    for mem in ("r10i1p1f1", "r3i1p1f1"):
        for grid in ("gn", "gr"):
            _make_file(
                awi / mem / "Amon" / "tas" / grid / "v1"
                / f"tas_Amon_AWI-CM-1-1-MR_historical_{mem}_{grid}_x.nc"
            )
    # An institution dir with a model that lacks the experiment
    _make_file(
        root / "NOAA" / "GFDL-NO-HIST" / "amip" / "r1i1p1f1" / "Amon"
        / "tas" / "gn" / "v1" / "x.nc"
    )
    return root


def test_discover_models(drs_tree):
    models = pd.discover_models(drs_tree, "historical")
    assert models == ["AWI-CM-1-1-MR", "MPI-ESM1-2-LR"]
    # Model without the experiment is excluded
    assert "GFDL-NO-HIST" not in models


def test_discover_models_missing_root(tmp_path):
    assert pd.discover_models(tmp_path / "nope", "historical") == []


def test_iter_model_dirs(drs_tree):
    pairs = dict(pd.iter_model_dirs(drs_tree, "historical"))
    assert set(pairs) == {"AWI-CM-1-1-MR", "MPI-ESM1-2-LR"}
    assert pairs["MPI-ESM1-2-LR"].name == "historical"


def test_select_member_prefers_r1(drs_tree):
    exp = drs_tree / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
    assert pd.select_member(exp) == "r1i1p1f1"


def test_select_member_numeric_fallback(drs_tree):
    # AWI has r3 and r10 only — should pick r3 (numeric), not r10 (lexical)
    exp = drs_tree / "AWI" / "AWI-CM-1-1-MR" / "historical"
    assert pd.select_member(exp) == "r3i1p1f1"


def test_select_member_empty(tmp_path):
    assert pd.select_member(tmp_path / "missing") is None


def test_select_grid_prefers_gr(drs_tree):
    var_dir = (
        drs_tree / "AWI" / "AWI-CM-1-1-MR" / "historical"
        / "r3i1p1f1" / "Amon" / "tas"
    )
    assert pd.select_grid(var_dir) == "gr"


def test_select_grid_falls_back(drs_tree):
    var_dir = (
        drs_tree / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
        / "r1i1p1f1" / "Amon" / "tas"
    )
    assert pd.select_grid(var_dir) == "gn"


def test_latest_version(drs_tree):
    grid_dir = (
        drs_tree / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
        / "r1i1p1f1" / "Amon" / "tas" / "gn"
    )
    assert pd.latest_version(grid_dir).name == "v20200101"


def test_variable_files(drs_tree):
    exp = drs_tree / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
    files = pd.variable_files(exp, "r1i1p1f1", "Amon", "tas")
    assert len(files) == 1
    assert files[0].name.endswith(".nc")
    assert "v20200101" in str(files[0])


def test_variable_files_missing(drs_tree):
    exp = drs_tree / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
    assert pd.variable_files(exp, "r1i1p1f1", "Omon", "thetao") == []


# ── Version-union override (split-across-versions publications) ──────────


@pytest.fixture
def split_version_tree(tmp_path):
    """A variable whose record is split across two version directories.

    Mirrors BCC-CSM2-HR hist-1950 Amon: the older version carries the first
    half of the series and the newer one the second, so reading only the
    latest silently drops the early years.
    """
    grid = tmp_path / "Amon" / "tas" / "gn"
    _make_file(grid / "v20200822" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_195001-200012.nc")
    _make_file(grid / "v20200921" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_200101-201412.nc")
    return grid


def _names(paths):
    return [p.name.split("_gn_")[-1].replace(".nc", "") for p in paths]


class TestFileSpan:
    def test_parses_monthly_span(self, tmp_path):
        p = tmp_path / "tas_Amon_M_hist-1950_r1i1p1f1_gn_195001-200012.nc"
        assert pd._file_span(p) == ("195001", "200012")

    def test_parses_daily_span(self, tmp_path):
        p = tmp_path / "tas_day_M_hist-1950_r1i1p1f1_gn_19500101-20001231.nc"
        assert pd._file_span(p) == ("19500101", "20001231")

    def test_fixed_field_has_no_span(self, tmp_path):
        assert pd._file_span(tmp_path / "areacella_fx_M_hist-1950_r1i1p1f1_gn.nc") is None


class TestUnionVersionFiles:
    def test_recovers_the_stranded_segment(self, split_version_tree):
        got = pd.union_version_files(split_version_tree)
        assert _names(got) == ["195001-200012", "200101-201412"]

    def test_default_path_sees_only_the_latest(self, split_version_tree):
        latest = pd.latest_version(split_version_tree)
        assert _names(sorted(latest.glob("*.nc"))) == ["200101-201412"]

    def test_sorted_by_start_date(self, split_version_tree):
        got = pd.union_version_files(split_version_tree)
        assert _names(got) == sorted(_names(got))

    def test_newer_version_wins_on_equal_span(self, tmp_path):
        """A genuine supersede must not be downgraded to its old edition."""
        grid = tmp_path / "Amon" / "tas" / "gn"
        name = "tas_Amon_M_historical_r1i1p1f1_gn_185001-201412.nc"
        _make_file(grid / "v20190101" / name)
        _make_file(grid / "v20200101" / name)
        got = pd.union_version_files(grid)
        assert len(got) == 1
        assert got[0].parent.name == "v20200101"

    def test_single_version_unchanged(self, tmp_path):
        grid = tmp_path / "Amon" / "tas" / "gn"
        _make_file(grid / "v1" / "tas_Amon_M_historical_r1i1p1f1_gn_185001-201412.nc")
        assert pd.union_version_files(grid) == sorted((grid / "v1").glob("*.nc"))

    def test_spanless_files_fall_back_to_latest(self, tmp_path):
        """Without a span there's no way to tell supersede from segment."""
        grid = tmp_path / "fx" / "areacella" / "gn"
        _make_file(grid / "v1" / "areacella_fx_M_historical_r1i1p1f1_gn.nc")
        _make_file(grid / "v2" / "areacella_fx_M_historical_r1i1p1f1_gn.nc")
        got = pd.union_version_files(grid)
        assert len(got) == 1 and got[0].parent.name == "v2"

    def test_missing_dir_returns_empty(self, tmp_path):
        assert pd.union_version_files(tmp_path / "nope") == []


class TestVariableFilesUnionFlag:
    def test_off_by_default(self, tmp_path):
        base = tmp_path / "r1i1p1f1" / "Amon" / "tas" / "gn"
        _make_file(base / "v20200822" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_195001-200012.nc")
        _make_file(base / "v20200921" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_200101-201412.nc")
        got = pd.variable_files(tmp_path, "r1i1p1f1", "Amon", "tas")
        assert _names(got) == ["200101-201412"]

    def test_on_when_requested(self, tmp_path):
        base = tmp_path / "r1i1p1f1" / "Amon" / "tas" / "gn"
        _make_file(base / "v20200822" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_195001-200012.nc")
        _make_file(base / "v20200921" / "tas_Amon_M_hist-1950_r1i1p1f1_gn_200101-201412.nc")
        got = pd.variable_files(tmp_path, "r1i1p1f1", "Amon", "tas",
                                union_versions=True)
        assert _names(got) == ["195001-200012", "200101-201412"]

    def test_union_respects_grid_preference(self, tmp_path):
        """gr is preferred over gn, and the union applies within that grid."""
        for grid in ("gn", "gr"):
            base = tmp_path / "r1i1p1f1" / "Amon" / "tas" / grid
            _make_file(base / "v1" / f"tas_Amon_M_historical_r1i1p1f1_{grid}_195001-200012.nc")
        got = pd.variable_files(tmp_path, "r1i1p1f1", "Amon", "tas",
                                union_versions=True)
        assert all(p.parent.parent.name == "gr" for p in got)
