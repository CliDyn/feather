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


# ── Per-model override tests ─────────────────────────────────────────


def _make_synth_nc(path, var_name="tas", time_dim="time", n_months=12):
    """Write a small synthetic NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=n_months, freq="MS")
    data = np.ones((n_months, len(lats), len(lons))) * 300.0
    ds = xr.Dataset({
        var_name: xr.DataArray(
            data, dims=(time_dim, "lat", "lon"),
            coords={time_dim: time, "lat": lats, "lon": lons},
        ),
    })
    ds.to_netcdf(path)
    return ds


@pytest.fixture
def hadgem_tree(tmp_path):
    """Create a HadGEM3-style directory tree with per-model overrides."""
    root = tmp_path / "MOHC" / "HadGEM3"

    # Atmosphere: standard layout with gr1 grid label
    tas_dir = root / "eerie-historical" / "r1i1p1f1" / "Amon" / "tas" / "gr1" / "v20240927"
    _make_synth_nc(
        tas_dir / "tas_UM_u-di356_199001-199012.nc",
        var_name="tas",
    )

    # clt: fraction 0-1 (needs scale_factor=100)
    clt_dir = root / "eerie-historical" / "r1i1p1f1" / "Amon" / "clt" / "gr1" / "v20240927"
    path = clt_dir / "clt_UM_u-di356_199001-199012.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS")
    data = np.ones((12, len(lats), len(lons))) * 0.65  # 65% as fraction
    ds = xr.Dataset({
        "clt": xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })
    ds.to_netcdf(path)

    # Ocean alias: toscon (alias for tos) with gr1 grid label
    tos_dir = root / "eerie-historical" / "r1i1p1f1" / "Omon" / "toscon" / "gr1" / "v20240927"
    _make_synth_nc(
        tos_dir / "toscon_nemo_199001-199012.nc",
        var_name="toscon",
    )

    # Ocean 3D alias: thetao-con (flat layout, no grid_label/version)
    # Dir name has hyphen, NetCDF var name has underscore
    thetao_dir = root / "eerie-historical" / "r1i1p1f1" / "Omon" / "thetao-con"
    _make_synth_nc(
        thetao_dir / "thetao-con_nemo_199001.nc",
        var_name="thetao_con",
        time_dim="time_counter",
    )

    return root


@pytest.fixture
def hadgem_config(hadgem_tree, tmp_path):
    """FeatherConfig for a model with per-model overrides."""
    return FeatherConfig(
        model_catalogs={},
        models=["HadGEM3"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor", "root": ""},
        model_configs={
            "HadGEM3": ModelConfig(
                name="HadGEM3",
                institution="MOHC",
                experiment="eerie-historical",
                variant="r1i1p1f1",
                grids={"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                color="#d62728",
                data_root=str(hadgem_tree),
                grid_label="gr1",
                variable_aliases={
                    "thetao": "thetao-con",
                    "so": "so-abs",
                    "tos": "toscon",
                    "sos": "sosabs",
                },
                scale_factors={"clt": 100},
            ),
        },
    )


class TestDataRootOverride:
    """Tests for per-model data_root path construction."""

    def test_loads_from_data_root(self, hadgem_config):
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "tas", table="Amon")
        assert da.dims == ("time", "lat", "lon")
        assert len(da.time) == 12

    def test_table_dir_uses_data_root(self, hadgem_config, hadgem_tree):
        loader = CMORLoader(hadgem_config)
        td = loader._table_dir("HadGEM3", "Amon")
        expected = hadgem_tree / "eerie-historical" / "r1i1p1f1" / "Amon"
        assert td == expected


class TestGridLabelOverride:
    """Tests for per-model grid_label (gr1 instead of gr)."""

    def test_finds_gr1_dir(self, hadgem_config):
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "tas", table="Amon")
        assert da.values.mean() == pytest.approx(300.0)

    def test_default_grid_label_is_gr(self, cmor_config):
        """Standard models still use 'gr' by default."""
        loader = CMORLoader(cmor_config)
        da = loader.load_var("TestModel", "tas", table="Amon")
        assert da.dims == ("time", "lat", "lon")


class TestVariableAliases:
    """Tests for per-model variable name mapping."""

    def test_alias_tos_to_toscon(self, hadgem_config):
        """Requesting 'tos' loads from 'toscon' directory."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "tos", table="Omon")
        assert da.dims == ("time", "lat", "lon")

    def test_alias_thetao_flat_layout(self, hadgem_config):
        """thetao-con uses flat layout (no grid_label/version subdirs)."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "thetao", table="Omon")
        assert "time" in da.dims  # time_counter renamed to time

    def test_get_alias_returns_mapping(self, hadgem_config):
        loader = CMORLoader(hadgem_config)
        assert loader._get_alias("HadGEM3", "thetao") == "thetao-con"
        assert loader._get_alias("HadGEM3", "tos") == "toscon"
        assert loader._get_alias("HadGEM3", "tas") == "tas"  # no alias

    def test_no_alias_model_returns_variable(self, cmor_config):
        loader = CMORLoader(cmor_config)
        assert loader._get_alias("TestModel", "thetao") == "thetao"


class TestTimeCounterRenaming:
    """Tests for time_counter → time dimension renaming."""

    def test_time_counter_renamed(self, hadgem_config):
        """Ocean 3D files with time_counter dim get renamed to time."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "thetao", table="Omon")
        assert "time" in da.dims
        assert "time_counter" not in da.dims

    def test_standard_time_dim_unchanged(self, hadgem_config):
        """Atmosphere files with standard time dim are unaffected."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "tas", table="Amon")
        assert "time" in da.dims


class TestScaleFactors:
    """Tests for per-model post-load scaling."""

    def test_clt_scaled_to_percentage(self, hadgem_config):
        """clt stored as 0-1 fraction should be scaled to 0-100%."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "clt", table="Amon")
        assert da.values.mean() == pytest.approx(65.0)  # 0.65 * 100

    def test_no_scaling_for_standard_vars(self, hadgem_config):
        """Variables without scale_factor are unchanged."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "tas", table="Amon")
        assert da.values.mean() == pytest.approx(300.0)

    def test_get_scale_factor_default(self, cmor_config):
        loader = CMORLoader(cmor_config)
        assert loader._get_scale_factor("TestModel", "clt") == 1.0


class TestFlatLayout:
    """Tests for flat directory layout (no grid_label/version subdirs)."""

    def test_falls_back_to_flat(self, hadgem_config):
        """When no grid_label/v* dirs exist, reads .nc from var dir."""
        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "thetao", table="Omon")
        assert da.values.mean() == pytest.approx(300.0)

    def test_error_when_nothing_found(self, hadgem_config):
        """Raises FileNotFoundError when neither layout exists."""
        loader = CMORLoader(hadgem_config)
        with pytest.raises(FileNotFoundError, match="tried"):
            loader.load_var("HadGEM3", "pr", table="Amon")


class TestFillValueTime:
    """Tests for files with fill values in time coordinate."""

    def test_loads_despite_fill_value_time(self, hadgem_tree, hadgem_config):
        """Files with 9.97e+36 fill values in time should still load."""
        # Create a SImon/siconc file with a fill value in time
        si_dir = (
            hadgem_tree / "eerie-historical" / "r1i1p1f1"
            / "SImon" / "siconc" / "gr1" / "v20240927"
        )
        si_dir.mkdir(parents=True, exist_ok=True)

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)

        # Write a file with a fill value as time (raw seconds)
        data = np.ones((1, len(lats), len(lons))) * 50.0
        ds = xr.Dataset({
            "siconc": xr.DataArray(
                data, dims=("time", "lat", "lon"),
                coords={
                    "time": [9.969209968386869e+36],  # fill value!
                    "lat": lats,
                    "lon": lons,
                },
            ),
        })
        # Must write with raw float time (no encoding) to reproduce issue
        ds["time"].attrs["units"] = "seconds since 1850-01-01 00:00:00"
        ds["time"].attrs["calendar"] = "gregorian"
        ds["time"].encoding = {"dtype": "float64", "units": "seconds since 1850-01-01 00:00:00", "calendar": "gregorian"}
        ds.to_netcdf(
            si_dir / "siconc_si3_fill.nc",
            encoding={"time": {"dtype": "float64", "_FillValue": None}},
        )

        loader = CMORLoader(hadgem_config)
        da = loader.load_var("HadGEM3", "siconc", table="SImon")
        assert "time" in da.dims

    def test_decode_time_manually_no_bounds(self):
        """Without time_bounds, fill values become NaT."""
        time_vals = np.array([86400.0, 172800.0, 9.97e+36])
        ds = xr.Dataset({
            "x": xr.DataArray([1, 2, 3], dims="time"),
        })
        ds["time"] = ("time", time_vals)
        ds["time"].attrs["units"] = "seconds since 2000-01-01"
        ds["time"].attrs["calendar"] = "gregorian"

        result = CMORLoader._decode_time_manually(ds, "time")
        times = result["time"].values
        assert np.isnat(times[2])  # fill value → NaT
        assert not np.isnat(times[0])  # valid

    def test_decode_time_manually_with_bounds(self):
        """With time_bounds, fill values are recovered from midpoints."""
        time_vals = np.array([86400.0, 9.97e+36])
        bounds = np.array([[0.0, 172800.0], [172800.0, 259200.0]])
        ds = xr.Dataset({
            "x": xr.DataArray([1, 2], dims="time"),
            "time_bounds": xr.DataArray(bounds, dims=("time", "bnds")),
        })
        ds["time"] = ("time", time_vals)
        ds["time"].attrs["units"] = "seconds since 2000-01-01"
        ds["time"].attrs["calendar"] = "gregorian"
        ds["time"].attrs["bounds"] = "time_bounds"

        result = CMORLoader._decode_time_manually(ds, "time")
        times = result["time"].values
        assert not np.isnat(times[0])
        assert not np.isnat(times[1])  # recovered from bounds, not NaT


class TestModelConfigNewFields:
    """Tests for new ModelConfig fields."""

    def test_default_values(self):
        mc = ModelConfig(name="Test")
        assert mc.data_root == ""
        assert mc.grid_label == ""
        assert mc.variable_aliases == {}
        assert mc.scale_factors == {}

    def test_fields_set(self):
        mc = ModelConfig(
            name="Test",
            data_root="/some/path",
            grid_label="gr1",
            variable_aliases={"thetao": "thetao-con"},
            scale_factors={"clt": 100},
        )
        assert mc.data_root == "/some/path"
        assert mc.grid_label == "gr1"
        assert mc.variable_aliases == {"thetao": "thetao-con"}
        assert mc.scale_factors == {"clt": 100}


class TestConfigParsing:
    """Tests that YAML parsing populates the new fields."""

    def test_from_yaml_new_fields(self, tmp_path):
        yaml_content = """
project:
  name: "Test"
  experiment: "hist-test"
  period: ["1980", "2014"]

data_source:
  type: "cmor"
  root: "/data"

models:
  ModelA:
    institution: INST
    experiment: hist-test
    variant: r1i1p1f1
    grids:
      sfc: latlon
    color: "#111111"
    data_root: "/custom/root"
    grid_label: "gr1"
    variable_aliases:
      thetao: "thetao-con"
      tos: "toscon"
    scale_factors:
      clt: 100
  ModelB:
    institution: INST2
    experiment: hist-test
    variant: r1i1p1f1
    grids:
      sfc: latlon
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = FeatherConfig.from_yaml(str(yaml_path))

        mc_a = cfg.model_configs["ModelA"]
        assert mc_a.data_root == "/custom/root"
        assert mc_a.grid_label == "gr1"
        assert mc_a.variable_aliases == {"thetao": "thetao-con", "tos": "toscon"}
        assert mc_a.scale_factors == {"clt": 100}

        # ModelB has no overrides → defaults
        mc_b = cfg.model_configs["ModelB"]
        assert mc_b.data_root == ""
        assert mc_b.grid_label == ""
        assert mc_b.variable_aliases == {}
        assert mc_b.scale_factors == {}
