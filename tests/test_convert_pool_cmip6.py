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


# ── Store-validity + atomic write (incomplete-store repair) ──────────


def _tiny_ds(var="tos"):
    import numpy as np
    import xarray as xr
    return xr.Dataset(
        {var: (("y", "x"), np.ones((3, 4)))},
        coords={"nav_lat": (("y", "x"), np.zeros((3, 4))),
                "nav_lon": (("y", "x"), np.zeros((3, 4)))},
    )


def test_store_is_valid_missing(tmp_path):
    assert convert._store_is_valid(tmp_path / "nope.zarr", "tos") is False


def test_store_is_valid_complete(tmp_path):
    pytest.importorskip("zarr")
    store = tmp_path / "ok.zarr"
    convert._write_zarr_atomic(_tiny_ds("tos"), store)
    assert store.exists()
    assert convert._store_is_valid(store, "tos") is True
    # Wrong variable name → not valid for that variable
    assert convert._store_is_valid(store, "so") is False


def test_store_is_valid_missing_variable(tmp_path):
    """An interrupted write (coords only, data var gone) is invalid.

    Simulated by an unconsolidated store whose ``tos`` array was removed,
    leaving only the small coordinate arrays — exactly the corruption seen in
    the real cache.
    """
    pytest.importorskip("zarr")
    import shutil
    store = tmp_path / "partial.zarr"
    _tiny_ds("tos").to_zarr(store, mode="w", consolidated=False)
    shutil.rmtree(store / "tos")  # drop the (large) data variable
    assert convert._store_is_valid(store, "tos") is False


def test_store_is_valid_complete_unconsolidated(tmp_path):
    """A complete store without consolidated metadata is still valid.

    Guards against reconverting good stores merely because consolidation is
    absent (e.g. a newer zarr format) — the variable is present, so skip.
    """
    pytest.importorskip("zarr")
    store = tmp_path / "unconsolidated.zarr"
    _tiny_ds("tos").to_zarr(store, mode="w", consolidated=False)
    assert convert._store_is_valid(store, "tos") is True


def test_atomic_write_leaves_no_tmp(tmp_path):
    pytest.importorskip("zarr")
    store = tmp_path / "a.zarr"
    convert._write_zarr_atomic(_tiny_ds("tos"), store)
    assert not (tmp_path / "a.zarr.tmp").exists()
    # Overwriting an existing (incomplete) store works
    convert._write_zarr_atomic(_tiny_ds("tos"), store)
    assert convert._store_is_valid(store, "tos")


# ── Uniform rechunking (zarr write compatibility) ────────────────────


def test_rechunk_uniform_blocks_ragged_time():
    pytest.importorskip("dask")
    import numpy as np
    import xarray as xr
    ds = xr.Dataset(
        {"tos": (("time", "x"), np.ones((250, 4)))},
        coords={"time": np.arange(250)},
    ).chunk({"time": (4, 1, 7, 238), "x": 4})  # ragged interior time chunks
    out = convert._rechunk_uniform(ds)
    tc = out["tos"].chunks[0]
    assert tc[0] == 120                      # fixed 120-month blocks
    assert len(set(tc[:-1])) <= 1            # uniform except the final chunk


def test_rechunk_uniform_collapses_ragged_space():
    pytest.importorskip("dask")
    import numpy as np
    import xarray as xr
    ds = xr.Dataset(
        {"tos": (("time", "cell"), np.ones((5, 100)))},
        coords={"time": np.arange(5)},
    ).chunk({"time": 5, "cell": (40, 10, 50)})  # non-uniform interior
    out = convert._rechunk_uniform(ds)
    cc = out["tos"].chunks[1]
    assert len(cc) == 1 and cc[0] == 100      # collapsed to one uniform chunk


def test_rechunk_uniform_roundtrips_to_zarr(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("dask")
    import numpy as np
    import xarray as xr
    ds = xr.Dataset(
        {"tos": (("time", "x"), np.ones((30, 4)))},
        coords={"time": np.arange(30)},
    ).chunk({"time": (4, 1, 25), "x": 4})     # ragged → would break plain to_zarr
    store = tmp_path / "rt.zarr"
    convert._write_zarr_atomic(convert._rechunk_uniform(ds), store)
    assert convert._store_is_valid(store, "tos")
