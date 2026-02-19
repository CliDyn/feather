"""Tests for the ObsLoader."""

import os
from pathlib import Path

import pytest

from feather.config import FeatherConfig
from feather.data.obs import ObsLoader


@pytest.fixture
def obs_config(tmp_path):
    """Create a minimal config with a synthetic obs dataset."""
    import numpy as np
    import xarray as xr

    # Create synthetic ERA5-like file
    obs_dir = tmp_path / "ERA5"
    obs_dir.mkdir()

    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.cftime_range("1990-01", periods=24, freq="MS")

    ds = xr.Dataset({
        "t2m": xr.DataArray(
            np.random.rand(24, len(lats), len(lons)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })
    ds.to_netcdf(obs_dir / "ERA5_t2m.nc")

    return FeatherConfig(
        model_catalogs={},
        models=[],
        obs_root=str(tmp_path),
        obs_datasets={
            "ERA5": {
                "path": str(obs_dir),
                "variables": {"t2m": "ERA5_t2m.nc"},
            }
        },
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
    )


def test_load_variable(obs_config):
    """Load a variable from synthetic obs."""
    obs = ObsLoader(obs_config)
    da = obs.load("ERA5", "t2m")
    assert da.dims == ("time", "lat", "lon")
    assert da.sizes["time"] == 24


def test_load_with_period(obs_config):
    """Load with time slicing."""
    obs = ObsLoader(obs_config)
    da = obs.load("ERA5", "t2m", period=("1990", "1990"))
    assert da.sizes["time"] == 12


def test_list_datasets(obs_config):
    """List available datasets."""
    obs = ObsLoader(obs_config)
    assert "ERA5" in obs.list_datasets()


def test_list_variables(obs_config):
    """List variables for a dataset."""
    obs = ObsLoader(obs_config)
    assert "t2m" in obs.list_variables("ERA5")


def test_load_missing_dataset(obs_config):
    """Loading a missing dataset raises KeyError."""
    obs = ObsLoader(obs_config)
    with pytest.raises(KeyError, match="MISSING"):
        obs.load("MISSING", "t2m")


@pytest.mark.integration
def test_load_real_era5():
    """Integration test with real ERA5 data on Levante."""
    default_cfg = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not default_cfg.exists():
        pytest.skip("Default config not found")

    cfg = FeatherConfig.from_yaml(str(default_cfg))
    era5_path = Path(cfg.obs_datasets["ERA5"]["path"])
    if not era5_path.exists():
        pytest.skip("ERA5 data not available")

    obs = ObsLoader(cfg)
    da = obs.load("ERA5", "t2m", period=("1990", "2014"))
    assert "time" in da.dims
