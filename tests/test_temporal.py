"""Tests for temporal utilities."""

import numpy as np
import xarray as xr

from feather.util.temporal import (
    annual_mean,
    anomaly,
    climatology,
    deseason,
    detrend,
    linear_trend,
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


# ── deseason tests ──────────────────────────────────────────────────


def test_deseason_removes_seasonal_cycle():
    """After deseason, groupby-month mean should be ~0."""
    da = _make_timeseries(n_years=5)
    deseas = deseason(da)
    monthly_means = deseas.groupby("time.month").mean("time")
    assert float(monthly_means.std()) < 0.1


def test_deseason_with_period():
    """deseason respects period slicing for climatology."""
    da = _make_timeseries(n_years=5)
    deseas = deseason(da, period=("1990", "1992"))
    assert deseas.sizes["time"] == da.sizes["time"]


def test_deseason_preserves_length():
    """deseason output has same time length as input."""
    da = _make_timeseries(n_years=3)
    deseas = deseason(da)
    assert deseas.sizes["time"] == da.sizes["time"]


# ── detrend tests ──────────────────────────────────────────────────


def test_detrend_removes_trend():
    """After detrend, linear_trend should return ~0."""
    da = _make_timeseries(n_years=5)
    detrended = detrend(da)
    residual_trend = linear_trend(detrended)
    assert abs(float(residual_trend)) < 0.01


def test_detrend_preserves_variability():
    """std should be similar after detrending (slightly smaller ok)."""
    da = _make_timeseries(n_years=5)
    original_std = float(da.std())
    detrended = detrend(da)
    detrended_std = float(detrended.std())
    # Should be close — trend is small relative to seasonal cycle
    assert detrended_std > original_std * 0.5


def test_detrend_constant_data():
    """Detrending constant data returns the same values."""
    time = xr.date_range("2000-01", periods=24, freq="MS")
    da = xr.DataArray(
        np.full(24, 5.0), dims="time", coords={"time": time},
    )
    detrended = detrend(da)
    np.testing.assert_allclose(detrended.values, 5.0, atol=1e-10)


def test_detrend_pure_trend():
    """Detrending a pure linear trend should give constant residual."""
    time = xr.date_range("2000-01", periods=36, freq="MS")
    values = np.linspace(0, 3, 36)
    da = xr.DataArray(values, dims="time", coords={"time": time})
    detrended = detrend(da)
    # Residual should have ~zero std
    assert float(detrended.std()) < 0.01


def test_deseason_detrend_pipeline():
    """Chaining deseason + detrend works correctly."""
    da = _make_timeseries(n_years=5)
    deseas = deseason(da)
    result = detrend(deseas)
    # Should have much less variance than original
    assert float(result.std()) < float(da.std())
    # Monthly means should be ~0 (seasonal cycle removed)
    monthly_means = result.groupby("time.month").mean("time")
    assert float(monthly_means.std()) < 0.1


def test_detrend_2d():
    """detrend works on 2D arrays (time x space)."""
    time = xr.date_range("2000-01", periods=36, freq="MS")
    # 3 grid points with different trends
    values = np.column_stack([
        np.linspace(0, 3, 36),        # +1/yr trend
        np.linspace(0, 6, 36),        # +2/yr trend
        np.ones(36) * 10,             # no trend
    ])
    da = xr.DataArray(
        values, dims=("time", "space"),
        coords={"time": time},
    )
    detrended = detrend(da)
    # Each grid point's trend should be ~0
    for i in range(3):
        trend = linear_trend(detrended.isel(space=i))
        assert abs(float(trend)) < 0.01
