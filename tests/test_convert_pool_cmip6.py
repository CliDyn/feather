"""Unit tests for the pool→zarr converter logic (no heavy I/O)."""

from pathlib import Path

import pytest

convert = pytest.importorskip("scripts.convert_pool_cmip6")


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


def test_build_var_tables_excludes_day():
    vt = convert.build_var_tables()
    assert "day" not in vt
    # Core eval vars present
    assert "tas" in vt["Amon"]
    assert "rsdt" in vt["Amon"]      # radiation component (extra)
    assert "thetao" in vt["Omon"]    # ocean 3D (extra)
    assert "siconc" in vt["SImon"]


@pytest.fixture
def mini_tree(tmp_path):
    """A model with tas (Amon), tos (Omon) and areacella (fx)."""
    exp = tmp_path / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
    mem = "r1i1p1f1"
    for table, var in [("Amon", "tas"), ("Omon", "tos"), ("fx", "areacella")]:
        _touch(
            exp / mem / table / var / "gn" / "v1"
            / f"{var}_{table}_MPI-ESM1-2-LR_historical_{mem}_gn_x.nc"
        )
    return exp


def test_convert_model_dry_run_counts(mini_tree, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    var_tables = {"Amon": ["tas", "pr"], "Omon": ["tos"]}
    n = convert.convert_model(
        mini_tree, "MPI-ESM1-2-LR", "historical", "r1i1p1f1",
        var_tables, out, ("1980", "2014"),
        skip_existing=True, dry_run=True,
    )
    # tas + tos + areacella present (pr missing) → 3 planned
    assert n == 3


def test_resolve_exp_dir(mini_tree, tmp_path):
    # activity root is tmp_path; model resolves to its historical dir
    found = convert._resolve_exp_dir(tmp_path, "MPI-ESM1-2-LR", "historical")
    assert found is not None and found.name == "historical"
    assert convert._resolve_exp_dir(tmp_path, "NOPE", "historical") is None
