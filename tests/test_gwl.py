"""Tests for the global-warming-level (GWL) window computation."""

import numpy as np
import xarray as xr

from feather.util.gwl import global_mean_annual_tas, warming_level_window


def _annual(years, values):
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=("year",),
        coords={"year": np.asarray(years)},
    )


def test_window_centered_on_crossing():
    # Linear warming: +0.02 K/yr from a 1850-1900 baseline of 0.
    years = np.arange(1850, 2101)
    warming = (years - 1900) * 0.02
    warming[years < 1900] = 0.0
    annual = _annual(years, warming)
    win = warming_level_window(annual, level=2.0, baseline=(1850, 1900), window=30)
    assert win is not None
    start, end = win
    assert end - start + 1 == 30
    # 30-yr-mean reaches +2 °C around the year the annual value passes ~2 °C
    assert 1980 <= start <= 2005


def test_never_reaches_level():
    years = np.arange(1850, 2101)
    annual = _annual(years, np.full(years.shape, 0.5))  # never +2
    assert warming_level_window(annual, level=2.0) is None


def test_global_mean_annual_tas():
    times = xr.date_range("2000-01-01", periods=24, freq="MS", calendar="standard")
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    data = np.zeros((24, 3, 3))
    data[12:] = 2.0  # year 2001 warmer than 2000
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
    )
    annual = global_mean_annual_tas(da)
    assert list(annual["year"].values) == [2000, 2001]
    assert float(annual.sel(year=2000)) == 0.0
    assert float(annual.sel(year=2001)) == 2.0


def test_baseline_offset_respected():
    # Baseline mean = 10; warming relative to it reaches 2 at +2 above 10.
    years = np.arange(1850, 2101)
    vals = 10.0 + np.maximum(0.0, (years - 1900) * 0.05)
    win = warming_level_window(_annual(years, vals), level=2.0, baseline=(1850, 1900))
    assert win is not None and win[1] - win[0] + 1 == 30
