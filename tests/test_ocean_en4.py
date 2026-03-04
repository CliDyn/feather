"""Tests for the ocean EN4 diagnostic (feather.diag.ocean_en4)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.diag.ocean_en4 import (
    OceanEN4,
    _DEPTH_RANGES,
    _K_TO_C,
    _VAR_CFG,
    _to_plot_time,
)

matplotlib.use("Agg")


# ── Fixtures ─────────────────────────────────────────────────────────


# Depth levels for synthetic model data (5 levels)
_SYNTH_DEPTHS = {
    "ifs-fesom": [5.0, 50.0, 200.0, 1000.0, 3000.0],
    "ifs-nemo": [5.0, 50.0, 200.0, 1000.0, 3000.0],
    "icon": [5.0, 50.0, 200.0, 1000.0, 3000.0],
}
_SYNTH_HALF_LEVELS = {
    "ifs-fesom": [0.0, 10.0, 100.0, 500.0, 2000.0, 4000.0],
    "icon": [0.0, 10.0, 100.0, 500.0, 2000.0, 4000.0],
}


@pytest.fixture
def synth_ocean_3d_healpix():
    """Synthetic 3D HEALPix dataset (nside=8, 768 cells, 5 depth levels).

    Temperature: warm surface (~300K), decreasing with depth.
    Salinity: ~35 PSU surface, increasing slightly with depth.
    """
    import healpy as hp

    nside = 8
    ncells = 12 * nside**2  # 768
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    ndepth = 5
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Temperature: gradient in lat + depth cooling
    temp_base = 300 - 30 * np.abs(lat / 90.0)
    depth_cooling = np.array([0, -5, -15, -30, -50])
    seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    # Build (time, level, values) array
    temp_3d = np.zeros((12, ndepth, ncells))
    for t in range(12):
        for z in range(ndepth):
            temp_3d[t, z, :] = temp_base + depth_cooling[z] + seasonal[t]

    # Salinity: ~35 PSU with slight depth increase
    sal_base = 35.0 + 0.5 * np.abs(lat / 90.0)
    sal_depth = np.array([0, 0.1, 0.3, 0.5, 0.7])
    sal_3d = np.zeros((12, ndepth, ncells))
    for t in range(12):
        for z in range(ndepth):
            sal_3d[t, z, :] = sal_base + sal_depth[z]

    ds = xr.Dataset(
        {
            "avg_thetao": xr.DataArray(
                temp_3d, dims=("time", "level", "values"),
                coords={"time": time},
            ),
            "avg_so": xr.DataArray(
                sal_3d, dims=("time", "level", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_en4():
    """Synthetic EN4-like dataset on 5-degree grid with 5 depth levels.

    Temperature in Kelvin, salinity in PSU.
    Includes lev_bnds for thickness computation.
    """
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    levs = np.array([5.0, 50.0, 200.0, 1000.0, 3000.0])
    lev_bnds = np.array([
        [0.0, 10.0],
        [10.0, 100.0],
        [100.0, 500.0],
        [500.0, 2000.0],
        [2000.0, 4000.0],
    ])
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 30 * np.abs(lat_grid / 90.0)
    depth_cooling = np.array([0, -5, -15, -30, -50])
    seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    temp_4d = np.zeros((12, 5, len(lats), len(lons)))
    for t in range(12):
        for z in range(5):
            temp_4d[t, z, :, :] = temp_base + depth_cooling[z] + seasonal[t]

    sal_base = 35.0 + 0.5 * np.abs(lat_grid / 90.0)
    sal_depth = np.array([0, 0.1, 0.3, 0.5, 0.7])
    sal_4d = np.zeros((12, 5, len(lats), len(lons)))
    for t in range(12):
        for z in range(5):
            sal_4d[t, z, :, :] = sal_base + sal_depth[z]

    thetao_ds = xr.Dataset(
        {
            "thetao": xr.DataArray(
                temp_4d, dims=("time", "lev", "lat", "lon"),
                coords={"time": time, "lev": levs, "lat": lats, "lon": lons},
            ),
            "lev_bnds": xr.DataArray(
                lev_bnds, dims=("lev", "bnds"),
                coords={"lev": levs},
            ),
        }
    )
    so_ds = xr.Dataset(
        {
            "so": xr.DataArray(
                sal_4d, dims=("time", "lev", "lat", "lon"),
                coords={"time": time, "lev": levs, "lat": lats, "lon": lons},
            ),
            "lev_bnds": xr.DataArray(
                lev_bnds, dims=("lev", "bnds"),
                coords={"lev": levs},
            ),
        }
    )
    return {"thetao": thetao_ds, "so": so_ds}


class MockEN4ObsLoader:
    """Mock ObsLoader that returns synthetic EN4 data."""

    def __init__(self, en4_datasets):
        self._datasets = en4_datasets

    def load_en4(self, variable="thetao", period=None):
        ds = self._datasets.get(variable)
        if ds is None:
            raise KeyError(f"EN4 variable {variable!r} not available")
        da = ds[variable]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_en4_dataset(self, variable="thetao", period=None):
        ds = self._datasets.get(variable)
        if ds is None:
            raise KeyError(f"EN4 variable {variable!r} not available")
        if period and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds


class MockOcean3DModelLoader:
    """Mock DataLoader returning synthetic 3D ocean HEALPix data."""

    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset

    def load(self, key: str) -> xr.Dataset:
        return self._ds

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        if variable not in self._ds:
            raise KeyError(f"{variable} not in dataset")
        return self._ds[variable]

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        return DataLoader.make_key(experiment, model, domain, member)


@pytest.fixture
def ocean_3d_config(tmp_path):
    """Config with ocean_3d depth levels."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000, "ocean_influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        ocean_3d={
            "depth_levels": _SYNTH_DEPTHS,
            "half_levels": _SYNTH_HALF_LEVELS,
        },
    )


@pytest.fixture
def ocean_3d_config_multi(tmp_path):
    """Config with ocean_3d depth levels, 3 models."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom", "ifs-nemo", "icon"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000, "ocean_influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        ocean_3d={
            "depth_levels": _SYNTH_DEPTHS,
            "half_levels": _SYNTH_HALF_LEVELS,
        },
    )


@pytest.fixture
def en4_diag(synth_ocean_3d_healpix, synth_en4, ocean_3d_config):
    """OceanEN4 instance with mock loaders."""
    model_loader = MockOcean3DModelLoader(synth_ocean_3d_healpix)
    obs_loader = MockEN4ObsLoader(synth_en4)
    return OceanEN4(
        model_loader, obs_loader, ocean_3d_config,
        experiment="baseline_hist",
        period=("1990", "2014"),
    )


@pytest.fixture
def en4_diag_multi(synth_ocean_3d_healpix, synth_en4, ocean_3d_config_multi):
    """OceanEN4 instance with 3 models."""
    model_loader = MockOcean3DModelLoader(synth_ocean_3d_healpix)
    obs_loader = MockEN4ObsLoader(synth_en4)
    return OceanEN4(
        model_loader, obs_loader, ocean_3d_config_multi,
        experiment="baseline_hist",
        period=("1990", "2014"),
    )


# ── 1. Registration and class attributes ────────────────────────────


class TestRegistration:
    """Test diagnostic registration and class attributes."""

    def test_registered_name(self):
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("ocean_en4")
        assert cls is OceanEN4

    def test_name_attribute(self):
        assert OceanEN4.name == "ocean_en4"

    def test_title_attribute(self):
        assert OceanEN4.title == "Ocean Evaluation (EN4)"

    def test_domain_attribute(self):
        assert OceanEN4.domain == "o3d"

    def test_variables_attribute(self):
        assert "thetao" in OceanEN4.variables
        assert "so" in OceanEN4.variables

    def test_group_attribute(self):
        assert OceanEN4.group == "ocean_3d"


# ── 2. Depth level lookup and thickness computation ─────────────────


class TestDepthLevels:
    """Test depth level configuration and thickness computation."""

    def test_get_depth_levels(self, en4_diag):
        depth = en4_diag._get_depth_levels("ifs-fesom")
        assert len(depth) == 5
        assert depth[0] == 5.0
        assert depth[-1] == 3000.0

    def test_get_depth_levels_missing(self, en4_diag):
        with pytest.raises(KeyError, match="No depth levels configured"):
            en4_diag._get_depth_levels("nonexistent_model")

    def test_get_layer_thickness_from_half_levels(self, en4_diag):
        thickness = en4_diag._get_layer_thickness("ifs-fesom")
        assert len(thickness) == 5
        # First layer: 0-10m = 10m thickness
        assert thickness[0] == pytest.approx(10.0)
        # Second layer: 10-100m = 90m thickness
        assert thickness[1] == pytest.approx(90.0)

    def test_get_layer_thickness_nemo_approximation(self, en4_diag):
        """NEMO doesn't have half levels — should approximate from centres."""
        thickness = en4_diag._get_layer_thickness("ifs-nemo")
        assert len(thickness) == 5
        assert all(t > 0 for t in thickness)

    def test_get_en4_thickness_from_lev_bnds(self, synth_en4):
        ds = synth_en4["thetao"]
        thickness = OceanEN4._get_en4_thickness(ds)
        assert len(thickness) == 5
        assert thickness[0] == pytest.approx(10.0)  # 0-10m
        assert thickness[1] == pytest.approx(90.0)  # 10-100m
        assert thickness[2] == pytest.approx(400.0)  # 100-500m

    def test_get_en4_thickness_fallback_no_bnds(self):
        """When lev_bnds is missing, approximate from lev coordinate."""
        levs = np.array([5.0, 50.0, 200.0, 1000.0, 3000.0])
        ds = xr.Dataset(coords={"lev": levs})
        thickness = OceanEN4._get_en4_thickness(ds)
        assert len(thickness) == 5
        assert all(t > 0 for t in thickness)

    def test_get_en4_thickness_no_lev_raises(self):
        ds = xr.Dataset()
        with pytest.raises(KeyError, match="neither"):
            OceanEN4._get_en4_thickness(ds)

    def test_depth_levels_dtype(self, en4_diag):
        depth = en4_diag._get_depth_levels("ifs-fesom")
        assert depth.dtype == np.float64

    def test_thickness_consistent_with_half_levels(self, en4_diag):
        """Sum of thicknesses should equal last half level."""
        thickness = en4_diag._get_layer_thickness("ifs-fesom")
        total = thickness.sum()
        assert total == pytest.approx(4000.0)  # 0 to 4000m

    def test_thickness_all_positive(self, en4_diag):
        for model in ["ifs-fesom", "ifs-nemo", "icon"]:
            thickness = en4_diag._get_layer_thickness(model)
            assert all(t > 0 for t in thickness), f"Negative thickness for {model}"


# ── 3. Data loading ─────────────────────────────────────────────────


class TestDataLoading:
    """Test model and observation data loading."""

    def test_load_model_data_shape(self, en4_diag):
        model_3d, coords, depths, thickness = en4_diag._load_model_data()
        assert "ifs-fesom" in model_3d
        assert "thetao" in model_3d["ifs-fesom"]
        da = model_3d["ifs-fesom"]["thetao"]
        assert "time" in da.dims
        assert "level" in da.dims

    def test_load_model_coords(self, en4_diag):
        _, coords, _, _ = en4_diag._load_model_data()
        lon, lat = coords["ifs-fesom"]
        assert len(lon) == 768
        assert len(lat) == 768

    def test_load_model_depths(self, en4_diag):
        _, _, depths, thickness = en4_diag._load_model_data()
        assert "ifs-fesom" in depths
        assert len(depths["ifs-fesom"]) == 5
        assert len(thickness["ifs-fesom"]) == 5

    def test_load_en4_data(self, en4_diag):
        en4_data, en4_ds = en4_diag._load_en4_data()
        assert "thetao" in en4_data
        assert "so" in en4_data

    def test_load_en4_shape(self, en4_diag):
        en4_data, _ = en4_diag._load_en4_data()
        da = en4_data["thetao"]
        assert "time" in da.dims
        assert "lev" in da.dims
        assert "lat" in da.dims
        assert "lon" in da.dims

    def test_load_en4_dataset_includes_lev_bnds(self, en4_diag):
        _, en4_ds = en4_diag._load_en4_data()
        ds = en4_ds["thetao"]
        assert "lev_bnds" in ds

    def test_load_model_missing_variable(self, synth_ocean_3d_healpix,
                                          synth_en4, ocean_3d_config):
        """Diagnostic should gracefully skip missing variables."""
        ds = synth_ocean_3d_healpix.drop_vars("avg_so")
        model_loader = MockOcean3DModelLoader(ds)
        obs_loader = MockEN4ObsLoader(synth_en4)
        diag = OceanEN4(model_loader, obs_loader, ocean_3d_config)
        model_3d, _, _, _ = diag._load_model_data()
        assert "thetao" in model_3d["ifs-fesom"]
        assert "so" not in model_3d["ifs-fesom"]

    def test_load_en4_missing_graceful(self, synth_ocean_3d_healpix,
                                       ocean_3d_config):
        """Missing EN4 data should not crash."""
        class EmptyEN4Loader:
            def load_en4(self, *a, **kw):
                raise KeyError("Not configured")
            def load_en4_dataset(self, *a, **kw):
                raise KeyError("Not configured")

        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            EmptyEN4Loader(),
            ocean_3d_config,
        )
        en4_data, en4_ds = diag._load_en4_data()
        assert len(en4_data) == 0


# ── 4. Unit conversion ──────────────────────────────────────────────


class TestConversion:
    """Test K->C and salinity identity conversion."""

    def test_thetao_convert_kelvin_to_celsius(self):
        convert = _VAR_CFG["thetao"]["convert"]
        da = xr.DataArray([300.0, 273.15])
        result = convert(da)
        assert result.values[0] == pytest.approx(300.0 - _K_TO_C)
        assert result.values[1] == pytest.approx(0.0)

    def test_so_convert_identity(self):
        convert = _VAR_CFG["so"]["convert"]
        da = xr.DataArray([35.0, 34.5])
        result = convert(da)
        np.testing.assert_array_equal(result.values, [35.0, 34.5])

    def test_var_cfg_keys(self):
        assert "thetao" in _VAR_CFG
        assert "so" in _VAR_CFG

    def test_var_cfg_has_required_fields(self):
        for var, cfg in _VAR_CFG.items():
            assert "long_name" in cfg
            assert "units" in cfg
            assert "en4_var" in cfg
            assert "convert" in cfg
            assert "cmap" in cfg


# ── 5. Bias map computation ─────────────────────────────────────────


class TestBiasMapComputation:
    """Test surface bias map computation."""

    def test_compute_bias_maps_returns_dict(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        assert "models" in results
        assert "periods" in results

    def test_bias_maps_has_three_periods(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        periods = results["periods"]
        assert "annual" in periods
        assert "djf" in periods
        assert "jja" in periods

    def test_bias_maps_model_results(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        assert "ifs-fesom" in results["models"]
        model_r = results["models"]["ifs-fesom"]
        assert "annual" in model_r
        assert "bias" in model_r["annual"]
        assert "bias_gmean" in model_r["annual"]
        assert "rmse" in model_r["annual"]

    def test_bias_is_finite(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        bias = results["models"]["ifs-fesom"]["annual"]["bias"]
        # Some NaN from ocean masking is OK, but not all NaN
        assert np.isfinite(bias.values).any()

    def test_bias_gmean_is_scalar(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        gmean = results["models"]["ifs-fesom"]["annual"]["bias_gmean"]
        assert isinstance(gmean, float)

    def test_bias_maps_salinity(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "so", model_3d, coords, en4_data)
        assert "ifs-fesom" in results["models"]

    def test_bias_maps_empty_en4(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, {})
        assert results["models"] == {}

    def test_obs_common_coords(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        obs_common = results["periods"]["annual"]["obs_common"]
        assert "lat" in obs_common.coords
        assert "lon" in obs_common.coords


# ── 6. Hovmoller computation ────────────────────────────────────────


class TestHovmollerComputation:
    """Test Hovmoller diagram computation."""

    def test_compute_hovmoller_returns_dict(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        assert "models" in hov
        assert "en4_hov" in hov
        assert "en4_depth" in hov

    def test_en4_hovmoller_shape(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        en4_hov = hov["en4_hov"]
        assert en4_hov is not None
        assert "time" in en4_hov.dims
        assert "depth" in en4_hov.dims
        assert en4_hov.shape == (12, 5)

    def test_hovmoller_salinity(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "so", model_3d, depths, thickness,
            en4_data, en4_ds)
        assert hov["en4_hov"] is not None

    def test_hovmoller_no_en4(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness, {}, {})
        assert hov["en4_hov"] is None

    def test_en4_hovmoller_values_finite(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        en4_hov = hov["en4_hov"]
        assert np.isfinite(en4_hov.values).all()

    def test_en4_hovmoller_first_timestep_accessible(self, en4_diag):
        """First timestep profile should be accessible for anomaly ref."""
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        en4_hov = hov["en4_hov"]
        first_profile = en4_hov.values[0, :]
        assert len(first_profile) == 5
        # Surface should be warmer than deep
        assert first_profile[0] > first_profile[-1]


# ── 7. Depth-layer time series ──────────────────────────────────────


class TestDepthTimeseries:
    """Test depth-layer time series computation."""

    def test_depth_ranges_defined(self):
        assert len(_DEPTH_RANGES) == 3
        assert _DEPTH_RANGES[0][0] == 0
        assert _DEPTH_RANGES[0][1] == 700
        assert _DEPTH_RANGES[2][1] is None  # bottom

    def test_compute_depth_timeseries_structure(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        assert "models" in ts
        assert "en4" in ts

    def test_en4_depth_timeseries_keys(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        en4_ts = ts["en4"]
        # At least some depth ranges should have data
        assert len(en4_ts) > 0

    def test_en4_ts_values_finite(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        for label, da in ts["en4"].items():
            assert np.isfinite(da.values).any(), f"All NaN in EN4 {label}"

    def test_depth_timeseries_salinity(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "so", model_3d, depths, thickness,
            en4_data, en4_ds)
        assert len(ts["en4"]) > 0

    def test_depth_timeseries_no_en4(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness, {}, {})
        assert ts["en4"] == {}


# ── 8. Plotting ─────────────────────────────────────────────────────


class TestPlotting:
    """Test figure creation."""

    def test_plot_bias_maps_creates_figures(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        assert len(figs) == 3  # annual, djf, jja
        for fig, meta in figs:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_hovmoller_anom1_combined(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_hovmoller_anom1("thetao", hov)
        # One combined figure with EN4 + models as subpanels
        assert len(figs) == 1
        fig, meta = figs[0]
        assert isinstance(fig, plt.Figure)
        assert "combined" in meta["figure_id"]
        # Should have subplots: 1 EN4 + 1 model = 2 axes
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_plot_hovmoller_anomref_combined(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_hovmoller_anomref("thetao", hov)
        assert len(figs) == 1
        fig, meta = figs[0]
        assert isinstance(fig, plt.Figure)
        assert "combined" in meta["figure_id"]
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_plot_depth_timeseries_creates_figure(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_depth_timeseries("thetao", ts)
        assert len(figs) == 1
        fig, meta = figs[0]
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_depth_timeseries_has_3_subplots(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_depth_timeseries("thetao", ts)
        fig, _ = figs[0]
        axes = fig.get_axes()
        assert len(axes) == 3
        plt.close(fig)

    def test_plot_hovmoller_anom1_shared_colorbar(self, en4_diag):
        """All panels should share the same symmetric color range."""
        from matplotlib.collections import QuadMesh

        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_hovmoller_anom1("thetao", hov)
        fig, _ = figs[0]
        # Collect clims from QuadMesh objects (the pcolormesh data)
        clims = []
        for ax in fig.get_axes():
            for child in ax.get_children():
                if isinstance(child, QuadMesh):
                    clims.append(child.get_clim())
        if len(clims) > 1:
            assert all(c == clims[0] for c in clims), (
                f"Color ranges differ: {clims}")
        # Symmetric around zero
        if clims:
            vmin, vmax = clims[0]
            assert vmin == pytest.approx(-vmax)
        plt.close(fig)


# ── 9. Metadata generation ──────────────────────────────────────────


class TestMetadata:
    """Test metadata JSON sidecar content."""

    def test_bias_map_metadata_fields(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        _, meta = figs[0]
        assert meta["diagnostic_name"] == "ocean_en4"
        assert meta["plot_type"] == "combined_bias_map"
        assert meta["obs_dataset"] == "EN4 v4.2.2"
        assert "figure_id" in meta
        plt.close("all")

    def test_hovmoller_metadata(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_hovmoller_anom1("thetao", hov)
        _, meta = figs[0]
        assert meta["plot_type"] == "hovmoller"
        assert "hovmoller" in meta["figure_id"]
        plt.close("all")

    def test_depth_timeseries_metadata(self, en4_diag):
        model_3d, _, depths, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        ts = en4_diag._compute_depth_timeseries(
            "thetao", model_3d, depths, thickness,
            en4_data, en4_ds)
        figs = en4_diag._plot_depth_timeseries("thetao", ts)
        _, meta = figs[0]
        assert meta["plot_type"] == "depth_timeseries"
        assert "depth_timeseries" in meta["figure_id"]
        plt.close("all")

    def test_bias_map_figure_id_pattern(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        fig_ids = [meta["figure_id"] for _, meta in figs]
        assert "en4_sst_annual_bias_combined" in fig_ids
        assert "en4_sst_djf_bias_combined" in fig_ids
        assert "en4_sst_jja_bias_combined" in fig_ids
        plt.close("all")

    def test_salinity_bias_map_figure_id(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "so", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("so", results)
        fig_ids = [meta["figure_id"] for _, meta in figs]
        assert "en4_sss_annual_bias_combined" in fig_ids
        plt.close("all")

    def test_metadata_period(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        _, meta = figs[0]
        assert meta["period"] == ["1990", "2014"]
        plt.close("all")

    def test_metadata_models_list(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        _, meta = figs[0]
        assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_metadata_summary_statistics(self, en4_diag):
        model_3d, coords, _, _ = en4_diag._load_model_data()
        en4_data, _ = en4_diag._load_en4_data()
        results = en4_diag._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        figs = en4_diag._plot_bias_maps("thetao", results)
        _, meta = figs[0]
        assert "summary_statistics" in meta
        stats = meta["summary_statistics"]
        assert "ifs-fesom" in stats
        assert "global_mean_bias" in stats["ifs-fesom"]
        assert "rmse" in stats["ifs-fesom"]
        plt.close("all")


# ── 10. run() orchestration ─────────────────────────────────────────


class TestRunOrchestration:
    """Test run() method and skip logic."""

    def test_run_creates_output_dir(self, en4_diag):
        en4_diag.run(skip_existing=False)
        assert en4_diag.output_dir.exists()
        plt.close("all")

    def test_run_produces_figures(self, en4_diag):
        saved = en4_diag.run(skip_existing=False)
        assert len(saved) > 0
        plt.close("all")

    def test_run_creates_png_and_json(self, en4_diag):
        saved = en4_diag.run(skip_existing=False)
        for png_path, json_path in saved:
            assert png_path.suffix == ".png"
            assert json_path.suffix == ".json"
        plt.close("all")

    def test_run_skip_existing(self, en4_diag):
        # First run creates figures
        saved1 = en4_diag.run(skip_existing=False)
        # Second run should skip them
        saved2 = en4_diag.run(skip_existing=True)
        assert len(saved2) == len(saved1)
        plt.close("all")

    def test_run_single_variable(self, synth_ocean_3d_healpix, synth_en4,
                                  ocean_3d_config):
        """Run with only thetao."""
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            ocean_3d_config,
            variables=["thetao"],
        )
        saved = diag.run(skip_existing=False)
        fig_ids = []
        for png_path, json_path in saved:
            fig_ids.append(png_path.stem)
        # Should have thetao figures but not salinity
        assert any("sst" in f for f in fig_ids)
        assert not any("sss" in f for f in fig_ids)
        plt.close("all")

    def test_run_json_readable(self, en4_diag):
        saved = en4_diag.run(skip_existing=False)
        for _, json_path in saved:
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                assert "diagnostic_name" in data
                assert "figure_id" in data
        plt.close("all")


# ── 11. Backward compat compute/plot ────────────────────────────────


class TestBackwardCompat:
    """Test compute() and plot() backward-compat methods."""

    def test_compute_returns_dict(self, en4_diag):
        results = en4_diag.compute()
        assert isinstance(results, dict)

    def test_compute_has_variable_keys(self, en4_diag):
        results = en4_diag.compute()
        assert "thetao_bias" in results
        assert "thetao_hov" in results
        assert "thetao_depth_ts" in results

    def test_plot_returns_figures(self, en4_diag):
        results = en4_diag.compute()
        figures = en4_diag.plot(results)
        assert len(figures) > 0
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
        plt.close("all")


# ── 12. Constructor options ─────────────────────────────────────────


class TestConstructorOptions:
    """Test constructor parameter handling."""

    def test_default_variables(self, en4_diag):
        assert en4_diag.variables == ["thetao", "so"]

    def test_custom_variables(self, synth_ocean_3d_healpix, synth_en4,
                               ocean_3d_config):
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            ocean_3d_config,
            variables=["thetao"],
        )
        assert diag.variables == ["thetao"]

    def test_experiment_kwarg(self, en4_diag):
        assert en4_diag.experiment == "baseline_hist"

    def test_period_kwarg(self, en4_diag):
        assert en4_diag.period == ("1990", "2014")

    def test_cmip6_individual_kwarg(self, synth_ocean_3d_healpix, synth_en4,
                                     ocean_3d_config):
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            ocean_3d_config,
            cmip6_individual=True,
        )
        assert diag.cmip6_individual is True

    def test_cmip6_disabled_by_default(self, en4_diag):
        assert en4_diag.cmip6_enabled is False

    def test_output_dir_property(self, en4_diag):
        expected = Path(en4_diag.config.output_dir) / "figures" / "ocean_en4"
        assert en4_diag.output_dir == expected


# ── 13. Graceful error handling ─────────────────────────────────────


class TestGracefulHandling:
    """Test graceful degradation for missing data."""

    def test_missing_model_in_depth_config(self, synth_ocean_3d_healpix,
                                            synth_en4, tmp_path):
        """Model not in depth_levels config should be skipped."""
        config = FeatherConfig(
            model_catalogs={},
            models=["ifs-fesom"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={"influence_radius": 1_000_000,
                    "ocean_influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
            ocean_3d={"depth_levels": {}, "half_levels": {}},
        )
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            config,
        )
        # Bias maps still work (surface extraction doesn't need depth levels)
        model_3d, coords, depths, thickness = diag._load_model_data()
        # depths should be empty since model not in config
        assert "ifs-fesom" not in depths

    def test_hovmoller_no_model_depths(self, en4_diag):
        """Hovmoller should skip models without depth levels."""
        model_3d, _, _, thickness = en4_diag._load_model_data()
        en4_data, en4_ds = en4_diag._load_en4_data()
        # Pass empty depths
        hov = en4_diag._compute_hovmoller(
            "thetao", model_3d, {}, thickness,
            en4_data, en4_ds)
        assert len(hov["models"]) == 0

    def test_anomref_no_en4_ref(self, en4_diag):
        """No EN4 ref profile should return empty figure list."""
        hov_data = {
            "models": {},
            "en4_hov": None,
            "en4_depth": None,
        }
        figs = en4_diag._plot_hovmoller_anomref("thetao", hov_data)
        assert figs == []


# ── 14. CMIP6 readiness ─────────────────────────────────────────────


class TestCMIP6Readiness:
    """Test CMIP6 integration points (disabled for now)."""

    def test_cmip6_loader_stored(self, synth_ocean_3d_healpix, synth_en4,
                                  ocean_3d_config):
        mock_cmip6 = MagicMock()
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            ocean_3d_config,
            cmip6_loader=mock_cmip6,
        )
        assert diag.cmip6_loader is mock_cmip6

    def test_cmip6_disabled_guard(self, en4_diag):
        """With cmip6 disabled, cmip6_enabled should be False."""
        assert en4_diag.cmip6_enabled is False

    def test_cmip6_individual_stored(self, synth_ocean_3d_healpix, synth_en4,
                                      ocean_3d_config):
        diag = OceanEN4(
            MockOcean3DModelLoader(synth_ocean_3d_healpix),
            MockEN4ObsLoader(synth_en4),
            ocean_3d_config,
            cmip6_individual=True,
        )
        assert diag.cmip6_individual is True


# ── 15. Helper functions ────────────────────────────────────────────


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_to_plot_time_empty(self):
        result = _to_plot_time(np.array([]))
        assert len(result) == 0

    def test_to_plot_time_numpy_datetime(self):
        time = np.array(["1990-01-01", "1990-02-01"], dtype="datetime64")
        result = _to_plot_time(time)
        assert len(result) == 2

    def test_to_plot_time_cftime(self):
        import cftime
        time = np.array([
            cftime.DatetimeGregorian(1990, 1, 1),
            cftime.DatetimeGregorian(1990, 2, 1),
        ])
        result = _to_plot_time(time)
        assert len(result) == 2

    def test_to_plot_time_integers_passthrough(self):
        time = np.array([1, 2, 3])
        result = _to_plot_time(time)
        np.testing.assert_array_equal(result, [1, 2, 3])


# ── 16. Config integration ──────────────────────────────────────────


class TestConfigIntegration:
    """Test FeatherConfig ocean_3d field."""

    def test_config_has_ocean_3d_field(self, ocean_3d_config):
        assert hasattr(ocean_3d_config, "ocean_3d")

    def test_config_depth_levels_accessible(self, ocean_3d_config):
        depths = ocean_3d_config.ocean_3d["depth_levels"]
        assert "ifs-fesom" in depths
        assert len(depths["ifs-fesom"]) == 5

    def test_config_half_levels_accessible(self, ocean_3d_config):
        half = ocean_3d_config.ocean_3d["half_levels"]
        assert "ifs-fesom" in half
        assert len(half["ifs-fesom"]) == 6

    def test_config_default_empty(self, tmp_path):
        """Default ocean_3d should be empty dict."""
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        assert config.ocean_3d == {}


# ── 17. ObsLoader EN4 methods ───────────────────────────────────────


class TestObsLoaderEN4:
    """Test load_en4() and load_en4_dataset() on ObsLoader."""

    def test_load_en4_not_configured(self, tmp_path):
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(KeyError, match="EN4 not configured"):
            loader.load_en4("thetao")

    def test_load_en4_dataset_not_configured(self, tmp_path):
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(KeyError, match="EN4 not configured"):
            loader.load_en4_dataset("thetao")

    def test_load_en4_missing_variable(self, tmp_path):
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={
                "EN4": {"path": str(tmp_path), "variables": {"thetao": "t.nc"}}
            },
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(FileNotFoundError, match="not in EN4 config"):
            loader.load_en4("unknown_var")


# ── 18. Multi-model tests ──────────────────────────────────────────


class TestMultiModel:
    """Test with multiple models."""

    def test_multi_model_bias_maps(self, en4_diag_multi):
        model_3d, coords, _, _ = en4_diag_multi._load_model_data()
        en4_data, _ = en4_diag_multi._load_en4_data()
        results = en4_diag_multi._compute_bias_maps(
            "thetao", model_3d, coords, en4_data)
        # All 3 models should have results
        assert len(results["models"]) == 3

    def test_multi_model_run(self, en4_diag_multi):
        saved = en4_diag_multi.run(skip_existing=False)
        assert len(saved) > 0
        plt.close("all")


# ── 19. Figure existence checking ───────────────────────────────────


class TestFigureExistence:
    """Test _figure_exists logic in run()."""

    def test_figure_exists_false_initially(self, en4_diag):
        assert not en4_diag._figure_exists("en4_sst_annual_bias_combined")

    def test_figure_exists_after_run(self, en4_diag):
        en4_diag.run(skip_existing=False)
        assert en4_diag._figure_exists("en4_sst_annual_bias_combined")
        plt.close("all")

    def test_figure_exists_needs_both_png_and_json(self, en4_diag, tmp_path):
        """Figure only counts as existing if both .png and .json exist."""
        out = en4_diag.output_dir
        out.mkdir(parents=True, exist_ok=True)
        # Only create png, not json
        (out / "test_fig.png").write_bytes(b"png")
        assert not en4_diag._figure_exists("test_fig")
