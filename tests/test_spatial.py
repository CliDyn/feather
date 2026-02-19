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
