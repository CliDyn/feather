"""Tests for the CORDEX-CORE loader (synthetic directory tree)."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.cordex_loader import CORDEXLoader

# Small rotated-pole-like grid (2-D lat/lon)
_NRLAT, _NRLON = 2, 3
_RLAT = np.array([-1.0, 1.0])
_RLON = np.array([10.0, 11.0, 12.0])
_LAT2D = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
_LON2D = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])


def _write(path, var, times, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((len(times), _NRLAT, _NRLON), value, dtype="float32")
    da = xr.DataArray(
        data, dims=("time", "rlat", "rlon"),
        coords={
            "time": times, "rlat": _RLAT, "rlon": _RLON,
            "lat": (("rlat", "rlon"), _LAT2D),
            "lon": (("rlat", "rlon"), _LON2D),
        },
        name=var,
    )
    da.to_dataset().to_netcdf(path)


def _cfg(tmp_path, model_configs):
    return FeatherConfig(
        model_catalogs={}, models=list(model_configs), obs_root="",
        obs_datasets={}, cmip6={}, dask={}, nereus={},
        output_dir=str(tmp_path),
        data_source={"type": "cordex", "root": str(tmp_path / "AFR-22")},
        model_configs=model_configs,
    )


def _member_dir(root, inst, gcm, exp, rcm, freq, var):
    return (root / "AFR-22" / inst / gcm / exp / "r1i1p1" / rcm
            / "v1" / freq / var / "v20200101")


def test_stitch_historical_rcp85(tmp_path):
    """Reference is assembled from historical + rcp85 segments."""
    root = tmp_path
    base = _member_dir(root, "GERICS", "MPI-M-MPI-ESM-LR", "historical",
                       "GERICS-REMO2015", "mon", "tas")
    hist_t = xr.date_range("2001-01-01", "2005-12-01", freq="MS", calendar="standard")
    _write(base / "tas_hist.nc", "tas", hist_t, 280.0)
    base2 = _member_dir(root, "GERICS", "MPI-M-MPI-ESM-LR", "rcp85",
                        "GERICS-REMO2015", "mon", "tas")
    fut_t = xr.date_range("2006-01-01", "2010-12-01", freq="MS", calendar="standard")
    _write(base2 / "tas_rcp.nc", "tas", fut_t, 282.0)

    mc = {"M1": ModelConfig(
        name="M1", institution="GERICS", gcm="MPI-M-MPI-ESM-LR",
        rcm="GERICS-REMO2015", variant="r1i1p1",
        experiments=["historical", "rcp85"], data_source_type="cordex",
    )}
    loader = CORDEXLoader(_cfg(tmp_path, mc))
    da = loader.load_var("M1", "tas", period=("2004", "2007"))

    years = np.unique(da.time.dt.year.values)
    assert set(years) == {2004, 2005, 2006, 2007}
    assert da.sizes["time"] == 48          # 4 years × 12 months, no dups
    assert da["lat"].ndim == 2             # rotated grid preserved


def test_day_fallback_resampled_to_monthly(tmp_path):
    """A member without monthly output uses daily data resampled to monthly."""
    root = tmp_path
    base = _member_dir(root, "CLMcom-KIT", "MPI-M-MPI-ESM-LR", "historical",
                       "CLMcom-KIT-CCLM5-0-15", "day", "pr")
    days = xr.date_range("2001-01-01", "2001-12-31", freq="D", calendar="standard")
    _write(base / "pr_day.nc", "pr", days, 1.0e-5)

    mc = {"M2": ModelConfig(
        name="M2", institution="CLMcom-KIT", gcm="MPI-M-MPI-ESM-LR",
        rcm="CLMcom-KIT-CCLM5-0-15", variant="r1i1p1",
        experiments=["historical"], data_source_type="cordex",
    )}
    loader = CORDEXLoader(_cfg(tmp_path, mc))
    da = loader.load_var("M2", "pr", period=("2001", "2001"))

    assert da.sizes["time"] == 12          # daily → 12 monthly means
    assert float(da.isel(time=0).mean()) == pytest.approx(1.0e-5, rel=1e-5)


def test_missing_member_raises(tmp_path):
    mc = {"MX": ModelConfig(
        name="MX", institution="NOPE", gcm="GCM", rcm="RCM",
        experiments=["historical"], data_source_type="cordex",
    )}
    loader = CORDEXLoader(_cfg(tmp_path, mc))
    try:
        loader.load_var("MX", "tas")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing member")
