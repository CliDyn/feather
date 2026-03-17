"""Tests for spatial utilities."""

import numpy as np
import pytest

from feather.util.spatial import global_mean, zonal_mean


def test_zonal_mean_basic(synth_healpix):
    """Zonal mean of synthetic temperature field."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    lat = ds["latitude"]

    # Use 10° bins for nside=8 (768 cells) to ensure bins are populated
    bins = np.arange(-90, 91, 10.0)
    zm = zonal_mean(t2m, lat, lat_bins=bins)

    assert "lat" in zm.dims
    # Equatorial bins should be warmer than polar bins
    valid = ~np.isnan(zm.values)
    assert valid.any()
    equator_idx = np.argmin(np.abs(zm.lat.values))
    south_idx = np.where(valid)[0][0]
    north_idx = np.where(valid)[0][-1]
    assert zm.values[equator_idx] > zm.values[south_idx]
    assert zm.values[equator_idx] > zm.values[north_idx]


def test_zonal_mean_with_time(synth_healpix):
    """Zonal mean preserves time dimension."""
    ds = synth_healpix
    t2m = ds["avg_2t"]
    lat = ds["latitude"]

    zm = zonal_mean(t2m, lat)

    assert "time" in zm.dims
    assert "lat" in zm.dims
    assert zm.sizes["time"] == 12


def test_zonal_mean_weighted(synth_healpix):
    """Zonal mean with area weights."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    lat = ds["latitude"]
    area = ds["area"]

    zm = zonal_mean(t2m, lat, weights=area)
    assert not np.all(np.isnan(zm.values))


def test_zonal_mean_nan_handling():
    """Zonal mean excludes NaN pixels from denominator.

    Regression test: previously NaN (land) pixels were included in the
    denominator, diluting the zonal mean for ocean-only variables.
    """
    import xarray as xr

    # 10 pixels: 5 in band 0 (0-10°), 5 in band 1 (10-20°)
    lat = np.array([5.0, 5.0, 5.0, 5.0, 5.0,
                     15.0, 15.0, 15.0, 15.0, 15.0])
    # Band 0: 3 ocean (value 20) + 2 land (NaN)
    # Band 1: all ocean (value 10)
    data = np.array([20.0, 20.0, 20.0, np.nan, np.nan,
                      10.0, 10.0, 10.0, 10.0, 10.0])
    da = xr.DataArray(data, dims=("values",))

    bins = np.array([0.0, 10.0, 20.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    # Band 0 should be 20.0 (mean of the 3 valid pixels), NOT 12.0 (60/5)
    assert zm.values[0] == pytest.approx(20.0)
    # Band 1 should be 10.0
    assert zm.values[1] == pytest.approx(10.0)


def test_zonal_mean_nan_with_time():
    """Zonal mean NaN handling works with extra time dimension."""
    import xarray as xr

    lat = np.array([5.0, 5.0, 5.0, 5.0])
    # 2 timesteps × 4 pixels; pixel 3 is always NaN (land)
    data = np.array([
        [10.0, 20.0, 30.0, np.nan],
        [40.0, 50.0, 60.0, np.nan],
    ])
    da = xr.DataArray(data, dims=("time", "values"))
    bins = np.array([0.0, 10.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    # t=0: mean of [10, 20, 30] = 20.0 (not 60/4 = 15.0)
    assert zm.values[0, 0] == pytest.approx(20.0)
    # t=1: mean of [40, 50, 60] = 50.0 (not 150/4 = 37.5)
    assert zm.values[1, 0] == pytest.approx(50.0)


def test_zonal_mean_all_nan_band():
    """Zonal mean returns NaN for latitude band where all pixels are NaN."""
    import xarray as xr

    lat = np.array([5.0, 5.0, 15.0, 15.0])
    data = np.array([np.nan, np.nan, 10.0, 20.0])
    da = xr.DataArray(data, dims=("values",))
    bins = np.array([0.0, 10.0, 20.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    assert np.isnan(zm.values[0])    # all-NaN band → NaN
    assert zm.values[1] == pytest.approx(15.0)


def test_global_mean(synth_healpix):
    """Global mean of synthetic temperature field."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    area = ds["area"]

    gm = global_mean(t2m, area)

    # Should be roughly between 260 and 300 K
    assert 250 < float(gm) < 310


def test_global_mean_with_time(synth_healpix):
    """Global mean preserves time dimension."""
    ds = synth_healpix
    t2m = ds["avg_2t"]
    area = ds["area"]

    gm = global_mean(t2m, area)
    assert "time" in gm.dims
    assert gm.sizes["time"] == 12
