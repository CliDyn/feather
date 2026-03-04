"""Tests for CMORLoader (feather.data.cmor_loader)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.cmor_loader import CMORLoader


@pytest.fixture
def cmor_tree(tmp_path):
    """Create a minimal CMOR directory tree with synthetic data."""
    root = tmp_path / "CMOR"

    # Model: TestModel, institution: TEST, experiment: hist-1950
    model_dir = root / "TEST" / "TestModel" / "hist-1950" / "r1i1p1f1"

    # Amon/tas
    tas_dir = model_dir / "Amon" / "tas" / "gr" / "v20240101"
    tas_dir.mkdir(parents=True)

    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp = 300 - 40 * np.abs(lat_grid / 90.0)
    temp_3d = temp[np.newaxis, :, :] + np.zeros((12, 1, 1))

    ds = xr.Dataset({
        "tas": xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })
    ds.to_netcdf(tas_dir / "tas_Amon_TestModel_hist-1950_r1i1p1f1_gr_199001-199012.nc")

    # Omon/tos (second table, second variable)
    tos_dir = model_dir / "Omon" / "tos" / "gr" / "v20240101"
    tos_dir.mkdir(parents=True)
    sst = 300 - 30 * np.abs(lat_grid / 90.0)
    sst_3d = sst[np.newaxis, :, :] + np.zeros((12, 1, 1))
    ds_tos = xr.Dataset({
        "tos": xr.DataArray(
            sst_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })
    ds_tos.to_netcdf(tos_dir / "tos_Omon_TestModel_hist-1950_r1i1p1f1_gr_199001-199012.nc")

    return root


@pytest.fixture
def cmor_config(cmor_tree, tmp_path):
    """FeatherConfig pointing to the CMOR tree."""
    return FeatherConfig(
        model_catalogs={},
        models=["TestModel"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor", "root": str(cmor_tree)},
        model_configs={
            "TestModel": ModelConfig(
                name="TestModel",
                institution="TEST",
                experiment="hist-1950",
                variant="r1i1p1f1",
                grids={"sfc": "latlon", "o2d": "latlon"},
                color="#1f77b4",
            ),
        },
    )


class TestCMORLoaderInit:
    def test_init(self, cmor_config):
        loader = CMORLoader(cmor_config)
        assert loader._root.exists()


class TestLoadVar:
    def test_load_tas(self, cmor_config):
        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tas", table="Amon")
        assert da.dims == ("time", "lat", "lon")
        assert len(da.time) == 12
        assert da.lat.values.min() == pytest.approx(-87.5)

    def test_load_infers_table(self, cmor_config):
        """Table inferred from variable registry when not specified."""
        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tas")
        assert da.dims == ("time", "lat", "lon")

    def test_load_tos_ocean(self, cmor_config):
        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tos", table="Omon")
        assert da.dims == ("time", "lat", "lon")

    def test_period_slicing(self, cmor_config):
        loader = CMORLoader(cmor_config)
        da = loader.load_var(
            "TestModel", "tas", table="Amon",
            period=("1990-06", "1990-08"),
        )
        assert len(da.time) == 3

    def test_time_mean(self, cmor_config):
        loader = CMORLoader(cmor_config)
        da = loader.load_var(
            "TestModel", "tas", table="Amon", time_mean=True,
        )
        assert "time" not in da.dims

    def test_caching(self, cmor_config):
        loader = CMORLoader(cmor_config)
        da1 = loader.load_var("TestModel", "tas", table="Amon")
        da2 = loader.load_var("TestModel", "tas", table="Amon")
        assert da1 is da2

    def test_missing_variable_raises(self, cmor_config):
        loader = CMORLoader(cmor_config)
        with pytest.raises(FileNotFoundError):
            loader.load_var("TestModel", "nonexistent", table="Amon")

    def test_missing_model_raises(self, cmor_config):
        loader = CMORLoader(cmor_config)
        with pytest.raises(KeyError):
            loader.load_var("NonExistentModel", "tas")


class TestLoadDataset:
    def test_returns_dataset(self, cmor_config):
        loader = CMORLoader(cmor_config)
        ds = loader.load_dataset("TestModel", "tas", table="Amon")
        assert isinstance(ds, xr.Dataset)
        assert "tas" in ds


class TestAvailableVariables:
    def test_lists_amon_vars(self, cmor_config):
        loader = CMORLoader(cmor_config)
        variables = loader.available_variables("TestModel", table="Amon")
        assert "tas" in variables

    def test_lists_omon_vars(self, cmor_config):
        loader = CMORLoader(cmor_config)
        variables = loader.available_variables("TestModel", table="Omon")
        assert "tos" in variables

    def test_unknown_model(self, cmor_config):
        loader = CMORLoader(cmor_config)
        assert loader.available_variables("Unknown") == []


class TestVersionDir:
    def test_picks_latest_version(self, cmor_tree, cmor_config):
        """When multiple version dirs exist, picks the latest."""
        # Create a second version
        v2_dir = (
            cmor_tree / "TEST" / "TestModel" / "hist-1950" / "r1i1p1f1"
            / "Amon" / "tas" / "gr" / "v20250101"
        )
        v2_dir.mkdir(parents=True)

        # Write a file with different data
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1990-01", periods=12, freq="MS")
        temp = np.ones((12, len(lats), len(lons))) * 999.0
        ds = xr.Dataset({
            "tas": xr.DataArray(
                temp, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        })
        ds.to_netcdf(v2_dir / "tas_Amon_TestModel_hist-1950_r1i1p1f1_gr_199001-199012.nc")

        # Fresh loader should pick v20250101
        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tas", table="Amon")
        assert da.values.mean() == pytest.approx(999.0)


class TestMultiFile:
    def test_loads_multiple_years(self, cmor_tree, cmor_config):
        """CMORLoader opens multi-file datasets across years."""
        # Add a second year file
        data_dir = (
            cmor_tree / "TEST" / "TestModel" / "hist-1950" / "r1i1p1f1"
            / "Amon" / "tas" / "gr" / "v20240101"
        )

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1991-01", periods=12, freq="MS")
        temp = np.ones((12, len(lats), len(lons))) * 280.0
        ds = xr.Dataset({
            "tas": xr.DataArray(
                temp, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        })
        ds.to_netcdf(data_dir / "tas_Amon_TestModel_hist-1950_r1i1p1f1_gr_199101-199112.nc")

        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tas", table="Amon")
        assert len(da.time) == 24  # 12 + 12
