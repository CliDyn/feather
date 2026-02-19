"""Tests for temporal utilities."""

import numpy as np
import xarray as xr

from feather.util.temporal import (
    annual_mean,
    anomaly,
    climatology,
    monthly_climatology,
    seasonal_climatology,
)


def _make_timeseries(n_years=3):
    """Helper: create a simple time series with known values."""
    time = xr.date_range("1990-01", periods=12 * n_years, freq="MS")
    # Values: baseline 280 + seasonal cycle (amplitude 10)
    months = np.tile(np.arange(12), n_years)
    values = 280 + 10 * np.sin(2 * np.pi * months / 12)
    # Add small trend: +0.1 per year
    values += np.linspace(0, 0.1 * n_years, len(values))
    return xr.DataArray(values, dims="time", coords={"time": time}, name="temp")


def test_climatology():
    """Time-mean climatology."""
    da = _make_timeseries(n_years=3)
    clim = climatology(da)
    assert "time" not in clim.dims
    assert 275 < float(clim) < 285


def test_climatology_with_period():
    """Climatology restricted to a period."""
    da = _make_timeseries(n_years=5)
    clim = climatology(da, period=("1990", "1992"))
    assert "time" not in clim.dims


def test_monthly_climatology():
    """Monthly climatological cycle."""
    da = _make_timeseries(n_years=3)
    mc = monthly_climatology(da)
    assert "month" in mc.dims
    assert mc.sizes["month"] == 12


def test_seasonal_climatology():
    """Seasonal climatologies."""
    da = _make_timeseries(n_years=3)
    sc = seasonal_climatology(da)
    for season in ["DJF", "MAM", "JJA", "SON"]:
        assert season in sc.data_vars


def test_anomaly_from_mean():
    """Anomaly relative to time-mean."""
    da = _make_timeseries(n_years=3)
    clim = climatology(da)
    anom = anomaly(da, clim)
    assert anom.sizes["time"] == da.sizes["time"]
    # Mean of anomaly should be ~0
    assert abs(float(anom.mean())) < 1.0


def test_anomaly_from_monthly():
    """Anomaly relative to monthly climatology."""
    da = _make_timeseries(n_years=3)
    mc = monthly_climatology(da)
    anom = anomaly(da, mc)
    assert anom.sizes["time"] == da.sizes["time"]


def test_annual_mean():
    """Annual mean."""
    da = _make_timeseries(n_years=3)
    am = annual_mean(da)
    assert am.sizes["time"] == 3
