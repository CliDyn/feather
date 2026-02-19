"""Tests for the DataLoader."""

import tempfile

import numpy as np
import pytest
import xarray as xr

from feather.data.loader import DataLoader


def test_make_key_2d():
    """Test catalog key generation for 2D domains."""
    key = DataLoader.make_key("baseline_hist", "ifs-fesom", "sfc")
    assert key == "baseline_hist_2_ifs-fesom_1_0001_clmn_high_sfc"

    key = DataLoader.make_key("baseline_hist", "ifs-nemo", "o2d", member=2)
    assert key == "baseline_hist_2_ifs-nemo_2_0001_clmn_high_o2d"


def test_make_key_3d():
    """Test catalog key generation for 3D domains."""
    key = DataLoader.make_key("baseline_hist", "icon", "pl")
    assert key == "baseline_hist_2_icon_1_0001_clmn_standard_pl"

    key = DataLoader.make_key("baseline_hist", "ifs-fesom", "o3d")
    assert key == "baseline_hist_2_ifs-fesom_1_0001_clmn_standard_o3d"


def test_from_paths_netcdf(tmp_path):
    """Test loading from explicit NetCDF paths."""
    # Create a temporary NetCDF file
    ds = xr.Dataset({
        "temperature": xr.DataArray(
            np.random.rand(10, 50),
            dims=("time", "values"),
        )
    })
    filepath = str(tmp_path / "test.nc")
    ds.to_netcdf(filepath)

    loader = DataLoader.from_paths({"test_entry": filepath})
    assert "test_entry" in loader.list_entries()

    loaded = loader.load("test_entry")
    assert "temperature" in loaded.data_vars


def test_load_var(tmp_path):
    """Test loading a single variable."""
    ds = xr.Dataset({
        "temp": xr.DataArray(np.random.rand(5, 20), dims=("time", "values")),
        "salt": xr.DataArray(np.random.rand(5, 20), dims=("time", "values")),
    })
    filepath = str(tmp_path / "test.nc")
    ds.to_netcdf(filepath)

    loader = DataLoader.from_paths({"entry": filepath})
    da = loader.load_var("entry", "temp")
    assert da.name == "temp"


def test_load_missing_key():
    """Loading a missing key raises KeyError."""
    loader = DataLoader.from_paths({"a": "/tmp/nonexistent.nc"})
    with pytest.raises(KeyError, match="not found"):
        loader.load("nonexistent")


def test_load_missing_variable(tmp_path):
    """Loading a missing variable raises KeyError."""
    ds = xr.Dataset({"temp": xr.DataArray([1, 2, 3], dims="x")})
    filepath = str(tmp_path / "test.nc")
    ds.to_netcdf(filepath)

    loader = DataLoader.from_paths({"entry": filepath})
    with pytest.raises(KeyError, match="salt"):
        loader.load_var("entry", "salt")


@pytest.mark.integration
def test_from_catalog_real():
    """Integration test with real catalog on Levante."""
    import os
    cat_path = "/work/ab0995/a270088/DestinE/GENERATION2_joint/2D/catalog.yaml"
    if not os.path.exists(cat_path):
        pytest.skip("Catalog not available")

    loader = DataLoader.from_catalog(cat_path)
    entries = loader.list_entries()
    assert len(entries) > 0

    key = loader.make_key("baseline_hist", "ifs-fesom", "sfc")
    if key in entries:
        ds = loader.load(key)
        assert "avg_2t" in ds.data_vars
