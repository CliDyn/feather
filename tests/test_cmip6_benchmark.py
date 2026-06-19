"""Tests for CMIP6Loader benchmark config + auto model discovery."""

from pathlib import Path

import numpy as np
import xarray as xr

from feather.data.cmip6 import CMIP6Loader


def test_is_griddable_rectilinear():
    da = xr.DataArray(
        np.zeros((2, 3, 4)), dims=("time", "lat", "lon"),
    )
    assert CMIP6Loader._is_griddable(da)


def test_is_griddable_curvilinear_2d_coords():
    da = xr.DataArray(
        np.zeros((2, 3, 4)), dims=("time", "j", "i"),
        coords={
            "nav_lat": (("j", "i"), np.zeros((3, 4))),
            "nav_lon": (("j", "i"), np.zeros((3, 4))),
        },
    )
    assert CMIP6Loader._is_griddable(da)


def test_is_griddable_rejects_unstructured():
    # Single non-time spatial dim 'i' with no 2-D lat/lon → unstructured.
    da = xr.DataArray(np.zeros((2, 5)), dims=("time", "i"))
    assert not CMIP6Loader._is_griddable(da)


def test_is_griddable_accepts_named_lat_dim():
    da = xr.DataArray(np.zeros((2, 5)), dims=("time", "lat"))
    assert CMIP6Loader._is_griddable(da)


def test_is_griddable_accepts_unstructured_with_1d_latlon():
    # ICON triangular grid: single spatial dim 'i' carrying 1-D lat/lon
    # cell-centre coords → regriddable as scattered points.
    n = 8
    da = xr.DataArray(
        np.zeros((2, n)), dims=("time", "i"),
        coords={
            "latitude": (("i",), np.linspace(-89, 89, n)),
            "longitude": (("i",), np.linspace(0, 359, n)),
        },
    )
    assert CMIP6Loader._is_griddable(da)


def test_is_griddable_still_rejects_index_without_latlon():
    # Same shape but no lat/lon coords on 'i' → cannot regrid.
    da = xr.DataArray(np.zeros((2, 8)), dims=("time", "i"))
    assert not CMIP6Loader._is_griddable(da)


def _monthly(start, end):
    t = xr.date_range(start, end, freq="MS", use_cftime=True)
    return xr.DataArray(np.ones(len(t)), dims="time", coords={"time": t})


def test_covers_period_full():
    assert CMIP6Loader._covers_period(_monthly("1980-01", "2014-12"),
                                      ("1980", "2014"))


def test_covers_period_rejects_late_start():
    assert not CMIP6Loader._covers_period(_monthly("2001-01", "2014-12"),
                                          ("1980", "2014"))


def test_covers_period_rejects_early_end():
    assert not CMIP6Loader._covers_period(_monthly("1980-01", "2005-12"),
                                          ("1980", "2014"))


def test_covers_period_no_time_dim_is_true():
    da = xr.DataArray(np.zeros((3, 3)), dims=("lat", "lon"))
    assert CMIP6Loader._covers_period(da, ("1980", "2014"))


def _mk_store(zdir: Path, name: str) -> None:
    (zdir / name).mkdir(parents=True, exist_ok=True)


def test_benchmark_zarr_dir_and_labels(tmp_path):
    cfg = {
        "zarr_dir": str(tmp_path),
        "experiment": "hist-1950",
        "label": "HighResMIP MMM",
        "color": "#9467bd",
        "models": {},
    }
    loader = CMIP6Loader(config=None, cmip6_cfg=cfg)
    assert loader.zarr_dir == str(tmp_path)
    assert loader.label == "HighResMIP MMM"
    assert loader.color == "#9467bd"


def test_auto_discovery_one_variant_per_model(tmp_path):
    # Two models, one with two variants → lowest variant picked.
    _mk_store(tmp_path, "MPI-ESM1-2-LR_historical_r1i1p1f1_Amon_tas.zarr")
    _mk_store(tmp_path, "MPI-ESM1-2-LR_historical_r1i1p1f1_Omon_tos.zarr")
    _mk_store(tmp_path, "CanESM5_historical_r2i1p1f1_Amon_tas.zarr")
    _mk_store(tmp_path, "CanESM5_historical_r1i1p2f1_Amon_tas.zarr")
    # A different experiment must be ignored.
    _mk_store(tmp_path, "OTHER_hist-1950_r1i1p1f1_Amon_tas.zarr")

    cfg = {"zarr_dir": str(tmp_path), "experiment": "historical",
           "models": "auto"}
    loader = CMIP6Loader(config=None, cmip6_cfg=cfg)
    models = loader.models
    assert set(models) == {"MPI-ESM1-2-LR", "CanESM5"}
    # CanESM5 has r1i1p2f1 and r2i1p1f1 → lexicographically lowest variant
    assert models["CanESM5"]["variant"] == "r1i1p2f1"
    assert models["MPI-ESM1-2-LR"]["variant"] == "r1i1p1f1"
    # member pairs reflect one-per-model
    pairs = loader.get_member_pairs()
    assert ("CanESM5", "r1i1p2f1") in pairs


def test_auto_discovery_handles_hyphenated_experiment(tmp_path):
    _mk_store(tmp_path, "ECMWF-IFS-HR_hist-1950_r1i1p1f1_Amon_tas.zarr")
    _mk_store(tmp_path, "MPI-ESM1-2-XR_hist-1950_r1i1p1f1_Amon_pr.zarr")
    cfg = {"zarr_dir": str(tmp_path), "experiment": "hist-1950",
           "models": "auto"}
    loader = CMIP6Loader(config=None, cmip6_cfg=cfg)
    assert set(loader.models) == {"ECMWF-IFS-HR", "MPI-ESM1-2-XR"}


def test_area_path_uses_experiment(tmp_path):
    cfg = {"zarr_dir": str(tmp_path), "experiment": "hist-1950",
           "models": "auto"}
    loader = CMIP6Loader(config=None, cmip6_cfg=cfg)
    atmos = loader._area_zarr_path("ECMWF-IFS-HR", "r1i1p1f1", "Amon")
    ocean = loader._area_zarr_path("ECMWF-IFS-HR", "r1i1p1f1", "Omon")
    assert atmos.endswith("ECMWF-IFS-HR_hist-1950_r1i1p1f1_fx_areacella.zarr")
    assert ocean.endswith("ECMWF-IFS-HR_hist-1950_r1i1p1f1_Ofx_areacello.zarr")


def test_zarr_path_experiment_token(tmp_path):
    cfg = {"zarr_dir": str(tmp_path), "experiment": "hist-1950",
           "models": "auto"}
    loader = CMIP6Loader(config=None, cmip6_cfg=cfg)
    p = loader._zarr_path("ECMWF-IFS-HR", "r1i1p1f1", "Amon", "tas",
                          experiment="hist-1950")
    assert p.endswith("ECMWF-IFS-HR_hist-1950_r1i1p1f1_Amon_tas.zarr")


def test_mmm_handles_mixed_scalar_coords_and_lengths(minimal_config):
    """MMM tolerates members with/without a scalar coord and unequal length."""
    import numpy as np
    import xarray as xr
    from feather.diag.base import DiagnosticBase

    class _Diag(DiagnosticBase):
        name = "t"
        variables = ["tas"]
        def compute(self):
            return {}
        def plot(self, results):
            return []

    def _series(years, with_height):
        t = xr.cftime_range("1980-01-01", periods=len(years), freq="YS")
        da = xr.DataArray(
            np.ones((len(years), 4, 8)) * 287.0,
            dims=("time", "lat", "lon"),
            coords={"time": t,
                    "lat": np.linspace(-80, 80, 4),
                    "lon": np.linspace(0, 315, 8)},
        )
        if with_height:
            da = da.assign_coords(height=2.0)
        return da

    class _Loader:
        label = "HighResMIP MMM"
        color = "#9467bd"
        models = {"A": {}, "B": {}}
        def load_var(self, var, model, *, table=None, period=None,
                     season=None, time_mean=False):
            # A: full 1980-2014 with height; B: shorter 2001-2014 no height
            if model == "A":
                return _series(range(1980, 2015), with_height=True)
            return _series(range(2001, 2015), with_height=False)
        def load_area(self, model, table="Amon"):
            return None

    minimal_config.cmip6 = {"enabled": True}
    diag = _Diag(None, None, minimal_config, benchmarks=[_Loader()])
    ts, info = diag._cmip6_global_mean_timeseries("tas", loader=_Loader())
    assert ts is not None
    # Spans the union (1980-2014), not just the overlap
    assert ts.time.dt.year.min().item() == 1980
    assert ts.time.dt.year.max().item() == 2014
    assert info["n_members"] == 2
