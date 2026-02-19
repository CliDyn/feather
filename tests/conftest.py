"""Shared test fixtures for feather tests."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synth_healpix():
    """Small synthetic HEALPix dataset (nside=8, 768 cells, 12 timesteps).

    Has a temperature gradient from pole to equator for testing:
    T = 300 - 40 * abs(lat/90)  (warm equator, cold poles).
    """
    nside = 8
    ncells = 12 * nside**2  # 768

    # Generate HEALPix pixel centres
    import healpy as hp
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    # Synthetic temperature: gradient from equator (300K) to poles (260K)
    temp_base = 300 - 40 * np.abs(lat / 90.0)

    # Create monthly time axis (12 months)
    time = xr.cftime_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Add seasonal cycle: +5K in summer, -5K in winter (NH convention)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    # Build 2D field: (time, values)
    temp_2d = temp_base[np.newaxis, :] + seasonal[:, np.newaxis]

    # Cell area (proportional to cos(lat) for simplicity)
    area = np.cos(np.deg2rad(lat))
    area = area / area.sum() * 4 * np.pi  # normalise to sphere

    ds = xr.Dataset(
        {
            "avg_2t": xr.DataArray(
                temp_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(area, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_obs():
    """Synthetic observation on regular lat/lon grid (5-degree resolution).

    Same temperature field as synth_healpix for bias testing.
    """
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.cftime_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Temperature gradient matching synth_healpix
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    ds = xr.Dataset(
        {
            "t2m": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        }
    )
    return ds
