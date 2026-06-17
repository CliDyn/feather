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
