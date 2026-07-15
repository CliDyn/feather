"""Tests for the ocean SST diagnostic (feather.diag.ocean_sst)."""

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
from feather.diag.ocean_sst import OceanSST, _K_TO_C, _ocean_global_mean, _to_celsius

matplotlib.use("Agg")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synth_ocean_healpix():
    """Synthetic HEALPix ocean dataset (nside=8) for avg_tos.

    Temperature gradient: warm tropics (~300K), cool poles (~270K).
    """
    import healpy as hp

    nside = 8
    ncells = 12 * nside**2  # 768
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    # SST: warm equator (300K), cold poles (270K)
    temp_base = 300 - 30 * np.abs(lat / 90.0)

    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")
    seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_2d = temp_base[np.newaxis, :] + seasonal[:, np.newaxis]

    ds = xr.Dataset(
        {
            "avg_tos": xr.DataArray(
                temp_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_esa_cci_timemean():
    """Synthetic ESA-CCI annual time-mean SST on 5-degree grid."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

    # SST matching ocean model pattern, in Kelvin
    temp_base = 300 - 30 * np.abs(lat_grid / 90.0)
    # Set polar regions to NaN (land/ice)
    temp_base[np.abs(lat_grid) > 60] = np.nan

    da = xr.DataArray(
        temp_base, dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
        name="analysed_sst",
    )
    return da


@pytest.fixture
def synth_esa_cci_ymonmean():
    """Synthetic ESA-CCI monthly climatology (12 months)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 30 * np.abs(lat_grid / 90.0)
    temp_base[np.abs(lat_grid) > 60] = np.nan

    seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    da = xr.DataArray(
        temp_3d, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        name="analysed_sst",
    )
    return da


@pytest.fixture
def synth_esa_cci_monthly():
    """Synthetic ESA-CCI full monthly series."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 30 * np.abs(lat_grid / 90.0)
    temp_base[np.abs(lat_grid) > 60] = np.nan

    seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    da = xr.DataArray(
        temp_3d, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        name="analysed_sst",
    )
    return da


class MockOceanModelLoader:
    """Mock DataLoader for ocean data."""

    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset

    def load(self, key: str) -> xr.Dataset:
        return self._ds

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        return self._ds[variable]

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        return DataLoader.make_key(experiment, model, domain, member)


class MockEsaCciObsLoader:
    """Mock ObsLoader with load_esa_cci() method."""

    def __init__(self, timemean, ymonmean, monthly):
        self._timemean = timemean
        self._ymonmean = ymonmean
        self._monthly = monthly

    def load_esa_cci(self, product="analysed_sst", period=None):
        if product == "timemean":
            return self._timemean
        elif product == "ymonmean":
            return self._ymonmean
        elif product == "analysed_sst":
            da = self._monthly
            if period is not None and "time" in da.dims:
                da = da.sel(time=slice(period[0], period[1]))
            return da
        raise FileNotFoundError(f"Unknown product: {product}")


@pytest.fixture
def mock_ocean_model_loader(synth_ocean_healpix):
    return MockOceanModelLoader(synth_ocean_healpix)


@pytest.fixture
def mock_esa_cci_obs_loader(
    synth_esa_cci_timemean, synth_esa_cci_ymonmean, synth_esa_cci_monthly,
):
    return MockEsaCciObsLoader(
        synth_esa_cci_timemean, synth_esa_cci_ymonmean, synth_esa_cci_monthly,
    )


@pytest.fixture
def ocean_sst_config(tmp_path):
    """Config for ocean SST tests."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={
            "ESA_CCI": {
                "path": "/fake",
                "variables": {
                    "analysed_sst": "monthly.nc",
                    "timemean": "timemean.nc",
                    "ymonmean": "ymonmean.nc",
                },
            },
        },
        cmip6={"enabled": False},
        dask={},
        nereus={
            "influence_radius": 1_000_000,
            "ocean_influence_radius": 1_000_000,
            "resolution": 5.0,
        },
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def ocean_sst_diag(mock_ocean_model_loader, mock_esa_cci_obs_loader,
                   ocean_sst_config):
    """Fully configured OceanSST diagnostic instance."""
    return OceanSST(
        model_loader=mock_ocean_model_loader,
        obs_loader=mock_esa_cci_obs_loader,
        config=ocean_sst_config,
    )


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_ocean_sst_registered(self):
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("ocean_sst")
        assert cls is OceanSST

    def test_ocean_sst_in_list(self):
        from feather.diag.registry import list_diagnostics
        names = [d["name"] for d in list_diagnostics()]
        assert "ocean_sst" in names


# ── Helper functions ─────────────────────────────────────────────────


class TestHelpers:
    def test_to_celsius(self):
        da = xr.DataArray([300.0, 273.15, 0.0])
        result = _to_celsius(da)
        np.testing.assert_allclose(result.values, [26.85, 0.0, -273.15])

    def test_ocean_global_mean_excludes_nan(self):
        """NaN values (land) should be excluded from the mean."""
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0])
        data = np.array([
            [10.0, np.nan],  # -45 lat
            [20.0, 20.0],    # equator
            [np.nan, 10.0],  # +45 lat
        ])
        da = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        result = _ocean_global_mean(da)
        assert np.isfinite(float(result.values))

    def test_ocean_global_mean_uniform(self):
        """Uniform field should return same value."""
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        da = xr.DataArray(
            np.full((len(lats), len(lons)), 15.0),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        result = _ocean_global_mean(da)
        np.testing.assert_allclose(float(result.values), 15.0)


# ── Class attributes ────────────────────────────────────────────────


class TestClassAttributes:
    def test_name(self):
        assert OceanSST.name == "ocean_sst"

    def test_title(self):
        assert OceanSST.title == "Ocean SST Evaluation"

    def test_domain(self):
        assert OceanSST.domain == "o2d"

    def test_variables(self):
        assert OceanSST.variables == ["tos"]

    def test_group(self):
        assert OceanSST.group == "ocean_surface"


# ── Data loading ─────────────────────────────────────────────────────


class TestDataLoading:
    def test_load_model_data(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        assert "ifs-fesom" in model_monthly
        assert "ifs-fesom" in model_coords

    def test_model_data_in_celsius(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        da = model_monthly["ifs-fesom"]
        # Original data is ~270-300K, Celsius should be ~-3 to 27
        assert float(da.mean().values) < 100  # clearly in Celsius

    def test_load_model_data_missing_model(self, mock_esa_cci_obs_loader,
                                            ocean_sst_config):
        """Model missing tos should be skipped."""
        empty_ds = xr.Dataset()
        loader = MockOceanModelLoader(empty_ds)
        # Override load_var to raise KeyError
        loader.load_var = MagicMock(side_effect=KeyError("avg_tos"))

        diag = OceanSST(
            model_loader=loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, model_coords = diag._load_model_data()
        assert len(model_monthly) == 0

    def test_load_obs_timemean(self, ocean_sst_diag):
        da = ocean_sst_diag._load_obs_timemean()
        # Should be in Celsius (original - 273.15)
        assert float(da.mean(skipna=True).values) < 100

    def test_load_obs_ymonmean(self, ocean_sst_diag):
        da = ocean_sst_diag._load_obs_ymonmean()
        assert "time" in da.dims or "month" in da.dims

    def test_load_obs_monthly(self, ocean_sst_diag):
        da = ocean_sst_diag._load_obs_monthly()
        assert "time" in da.dims


# ── Group A: Bias maps ───────────────────────────────────────────────


class TestBiasMaps:
    def test_compute_bias_maps_keys(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        assert "models" in results
        assert "periods" in results
        assert "common_area" in results

    def test_compute_bias_maps_periods(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        for pkey in ["annual", "djf", "jja"]:
            assert pkey in results["periods"]
            assert "obs_common" in results["periods"][pkey]

    def test_compute_bias_maps_model_results(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        assert "ifs-fesom" in results["models"]
        model_data = results["models"]["ifs-fesom"]
        for pkey in ["annual", "djf", "jja"]:
            assert pkey in model_data
            assert "bias" in model_data[pkey]
            assert "bias_gmean" in model_data[pkey]
            assert "rmse" in model_data[pkey]

    def test_bias_gmean_is_float(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        gmean = results["models"]["ifs-fesom"]["annual"]["bias_gmean"]
        assert isinstance(gmean, float)

    def test_default_seasons_exclude_mam_son(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        assert "mam" not in results["periods"]
        assert "son" not in results["periods"]

    def test_configured_seasons_add_mam_son(self, ocean_sst_diag):
        # Opt into all five periods via project.seasons.
        ocean_sst_diag.config.project = {
            "seasons": ["annual", "DJF", "MAM", "JJA", "SON"]}
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        for pkey in ("annual", "djf", "mam", "jja", "son"):
            assert pkey in results["periods"]
            assert "obs_common" in results["periods"][pkey]
            assert pkey in results["models"]["ifs-fesom"]
        # Figures are produced for every configured season (2 per season:
        # per-model + ensemble). Ensemble needs ≥1 evaluated model.
        plot_ids = {
            m["figure_id"] for _, m in ocean_sst_diag._plot_bias_maps(results)}
        assert {"sst_mam_bias_combined", "sst_son_bias_combined"} <= plot_ids

    def test_bias_reasonable_magnitude(self, ocean_sst_diag):
        """Bias should be within reasonable range (< 20K for synth data)."""
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        gmean = results["models"]["ifs-fesom"]["annual"]["bias_gmean"]
        assert abs(gmean) < 20

    def test_plot_bias_maps_produces_figures(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        figures = ocean_sst_diag._plot_bias_maps(results)
        assert len(figures) == 3  # annual, DJF, JJA
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert meta["diagnostic_name"] == "ocean_sst"
            assert meta["plot_type"] == "combined_bias_map"
            plt.close(fig)

    def test_plot_bias_maps_figure_ids(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        figures = ocean_sst_diag._plot_bias_maps(results)
        ids = [meta["figure_id"] for _, meta in figures]
        assert "sst_annual_bias_combined" in ids
        assert "sst_djf_bias_combined" in ids
        assert "sst_jja_bias_combined" in ids
        for fig, _ in figures:
            plt.close(fig)

    def test_plot_bias_maps_obs_dataset(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )
        figures = ocean_sst_diag._plot_bias_maps(results)
        for _, meta in figures:
            assert "ESA-CCI" in meta["obs_dataset"]
        for fig, _ in figures:
            plt.close(fig)

    def test_compute_no_models(self, mock_esa_cci_obs_loader,
                                ocean_sst_config):
        """No models available should produce empty results."""
        empty_ds = xr.Dataset()
        loader = MockOceanModelLoader(empty_ds)
        loader.load_var = MagicMock(side_effect=KeyError("avg_tos"))

        diag = OceanSST(
            model_loader=loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, model_coords = diag._load_model_data()
        results = diag._compute_bias_maps(model_monthly, model_coords)
        assert results["models"] == {}

    def test_plot_bias_maps_no_models(self, mock_esa_cci_obs_loader,
                                      ocean_sst_config):
        """Empty model results should produce no figures."""
        empty_ds = xr.Dataset()
        loader = MockOceanModelLoader(empty_ds)
        loader.load_var = MagicMock(side_effect=KeyError("avg_tos"))

        diag = OceanSST(
            model_loader=loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, model_coords = diag._load_model_data()
        results = diag._compute_bias_maps(model_monthly, model_coords)
        figures = diag._plot_bias_maps(results)
        assert len(figures) == 0


# ── Group B: Time series ─────────────────────────────────────────────


class TestTimeseries:
    def test_compute_timeseries_keys(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        assert "models" in results
        assert "obs" in results

    def test_compute_timeseries_model(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        assert "ifs-fesom" in results["models"]
        ts = results["models"]["ifs-fesom"]
        assert "time" in ts.dims
        assert len(ts.time) == 12

    def test_compute_timeseries_obs(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        assert results["obs"] is not None

    def test_timeseries_values_celsius(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        ts = results["models"]["ifs-fesom"]
        # Should be in Celsius, ~0-30 range
        assert float(ts.mean().values) < 100

    def test_plot_timeseries(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        figures = ocean_sst_diag._plot_timeseries(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert isinstance(fig, plt.Figure)
        assert meta["figure_id"] == "sst_timeseries"
        assert meta["plot_type"] == "timeseries"
        assert meta["diagnostic_name"] == "ocean_sst"
        plt.close(fig)

    def test_plot_timeseries_models_listed(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        figures = ocean_sst_diag._plot_timeseries(results)
        _, meta = figures[0]
        assert "ifs-fesom" in meta["models"]
        assert "ESA-CCI" in meta["models"]
        plt.close(figures[0][0])

    def test_timeseries_obs_failure(self, mock_ocean_model_loader,
                                     ocean_sst_config):
        """Missing obs should not crash -- obs_ts is None."""
        obs_loader = MagicMock()
        obs_loader.load_esa_cci = MagicMock(side_effect=KeyError("missing"))
        diag = OceanSST(
            model_loader=mock_ocean_model_loader,
            obs_loader=obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, _ = diag._load_model_data()
        results = diag._compute_timeseries(model_monthly)
        assert results["obs"] is None

    def test_compute_timeseries_benchmark_ensemble_keys(self, ocean_sst_diag):
        """New keys for benchmark MMM + ensemble stats are present."""
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)
        assert "benchmarks_ts" in results
        assert "ens_mean" in results
        assert "ens_median" in results
        # CMIP6 disabled in the fixture -> no benchmarks
        assert results["benchmarks_ts"] == []
        # single model -> ensemble stats are None
        assert results["ens_mean"] is None
        assert results["ens_median"] is None

    def test_compute_ensemble_stats_single(self):
        """A single-member dict yields (None, None)."""
        ts = xr.DataArray(
            np.arange(12.0), dims="time",
            coords={"time": np.arange(12)},
        )
        mean, median = OceanSST._compute_ensemble_stats({"a": ts})
        assert mean is None and median is None

    def test_compute_ensemble_stats_multi(self):
        """Two members give per-timestep mean/median across members."""
        t = np.arange(12)
        a = xr.DataArray(np.zeros(12), dims="time", coords={"time": t})
        b = xr.DataArray(np.full(12, 4.0), dims="time", coords={"time": t})
        mean, median = OceanSST._compute_ensemble_stats({"a": a, "b": b})
        assert mean is not None and median is not None
        np.testing.assert_allclose(mean.values, np.full(12, 2.0))
        np.testing.assert_allclose(median.values, np.full(12, 2.0))

    def test_plot_timeseries_with_benchmark_and_ensemble(self, ocean_sst_diag):
        """Plot renders benchmark MMM/envelope + ensemble mean/median."""
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_timeseries(model_monthly)

        ts = next(iter(results["models"].values()))
        base = ts.reset_coords(drop=True)
        results["ens_mean"] = base
        results["ens_median"] = base
        results["benchmarks_ts"] = [{
            "label": "CMIP6 MMM",
            "color": "#888888",
            "ts": base,
            "info": {"n_members": 3},
            "env_min": base - 0.5,
            "env_max": base + 0.5,
            "individual": {},
        }]

        figures = ocean_sst_diag._plot_timeseries(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert isinstance(fig, plt.Figure)
        assert "CMIP6 MMM" in meta["models"]
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("CMIP6 MMM" in lbl for lbl in labels)
        assert any("min–max" in lbl for lbl in labels)
        assert any("ensemble mean" in lbl for lbl in labels)
        assert any("ensemble median" in lbl for lbl in labels)
        plt.close(fig)


# ── Group C: Seasonal cycle ──────────────────────────────────────────


class TestSeasonalCycle:
    def test_compute_seasonal_cycle_keys(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_seasonal_cycle(model_monthly)
        assert "models" in results
        assert "obs" in results

    def test_model_cycle_length(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_seasonal_cycle(model_monthly)
        cycle = results["models"]["ifs-fesom"]
        assert len(cycle) == 12

    def test_obs_cycle_length(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_seasonal_cycle(model_monthly)
        obs_cycle = results["obs"]
        assert obs_cycle is not None
        assert len(obs_cycle) == 12

    def test_cycle_months(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_seasonal_cycle(model_monthly)
        cycle = results["models"]["ifs-fesom"]
        months = cycle.month.values
        np.testing.assert_array_equal(months, np.arange(1, 13))

    def test_plot_seasonal_cycle(self, ocean_sst_diag):
        model_monthly, _ = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_seasonal_cycle(model_monthly)
        figures = ocean_sst_diag._plot_seasonal_cycle(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert isinstance(fig, plt.Figure)
        assert meta["figure_id"] == "sst_seasonal_cycle"
        assert meta["plot_type"] == "seasonal_cycle"
        plt.close(fig)

    def test_seasonal_cycle_obs_failure(self, mock_ocean_model_loader,
                                        ocean_sst_config):
        """Missing obs should produce None obs_cycle."""
        obs_loader = MagicMock()
        obs_loader.load_esa_cci = MagicMock(side_effect=KeyError("missing"))
        diag = OceanSST(
            model_loader=mock_ocean_model_loader,
            obs_loader=obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, _ = diag._load_model_data()
        results = diag._compute_seasonal_cycle(model_monthly)
        assert results["obs"] is None


# ── Group D: Zonal mean ──────────────────────────────────────────────


class TestZonalMean:
    def test_compute_zonal_mean_keys(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_zonal_mean(
            model_monthly, model_coords,
        )
        assert "models" in results
        assert "obs" in results

    def test_model_zonal_has_lat(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_zonal_mean(
            model_monthly, model_coords,
        )
        zm = results["models"]["ifs-fesom"]
        assert "lat" in zm.dims

    def test_obs_zonal_has_lat(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_zonal_mean(
            model_monthly, model_coords,
        )
        obs_zm = results["obs"]
        assert obs_zm is not None
        assert "lat" in obs_zm.dims or "latitude" in obs_zm.dims

    def test_zonal_mean_gradient(self, ocean_sst_diag):
        """Equator should be warmer than poles."""
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_zonal_mean(
            model_monthly, model_coords,
        )
        zm = results["models"]["ifs-fesom"]
        lats = zm.lat.values
        # Find equatorial and polar bins
        eq_mask = np.abs(lats) < 15
        pole_mask = np.abs(lats) > 60
        eq_vals = zm.values[eq_mask]
        pole_vals = zm.values[pole_mask]
        eq_mean = np.nanmean(eq_vals)
        pole_mean = np.nanmean(pole_vals)
        assert eq_mean > pole_mean

    def test_plot_zonal_mean(self, ocean_sst_diag):
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_zonal_mean(
            model_monthly, model_coords,
        )
        figures = ocean_sst_diag._plot_zonal_mean(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert isinstance(fig, plt.Figure)
        assert meta["figure_id"] == "sst_zonal_mean"
        assert meta["plot_type"] == "zonal_profile"
        plt.close(fig)

    def test_zonal_mean_obs_failure(self, mock_ocean_model_loader,
                                     ocean_sst_config):
        """Missing obs should produce None obs_zonal."""
        obs_loader = MagicMock()
        obs_loader.load_esa_cci = MagicMock(side_effect=KeyError("missing"))
        diag = OceanSST(
            model_loader=mock_ocean_model_loader,
            obs_loader=obs_loader,
            config=ocean_sst_config,
        )
        model_monthly, model_coords = diag._load_model_data()
        results = diag._compute_zonal_mean(model_monthly, model_coords)
        assert results["obs"] is None


# ── Run / skip-existing ──────────────────────────────────────────────


class TestRun:
    def test_run_creates_figures(self, ocean_sst_diag):
        saved = ocean_sst_diag.run(skip_existing=False)
        # 3 per-model bias + 3 ensemble bias + 1 ts + 1 seasonal + 1 zonal
        assert len(saved) == 9
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
            assert png_path.suffix == ".png"
            assert json_path.suffix == ".json"
        plt.close("all")

    def test_run_figure_ids(self, ocean_sst_diag):
        saved = ocean_sst_diag.run(skip_existing=False)
        stems = [p.stem for p, _ in saved]
        assert "sst_annual_bias_combined" in stems
        assert "sst_djf_bias_combined" in stems
        assert "sst_jja_bias_combined" in stems
        assert "sst_timeseries" in stems
        assert "sst_seasonal_cycle" in stems
        assert "sst_zonal_mean" in stems
        # Ensemble mean/median bias figures
        assert "sst_annual_ens_bias_combined" in stems
        assert "sst_djf_ens_bias_combined" in stems
        assert "sst_jja_ens_bias_combined" in stems
        plt.close("all")

    def test_run_metadata_valid_json(self, ocean_sst_diag):
        saved = ocean_sst_diag.run(skip_existing=False)
        for _, json_path in saved:
            meta = json.loads(json_path.read_text())
            assert meta["diagnostic_name"] == "ocean_sst"
        plt.close("all")

    def test_run_skip_existing(self, ocean_sst_diag):
        """Second run with skip_existing should not regenerate."""
        saved1 = ocean_sst_diag.run(skip_existing=False)
        assert len(saved1) == 9

        saved2 = ocean_sst_diag.run(skip_existing=True)
        assert len(saved2) == 9
        # All paths should match
        for (p1, j1), (p2, j2) in zip(sorted(saved1), sorted(saved2)):
            assert p1 == p2
        plt.close("all")

    def test_run_partial_skip(self, ocean_sst_diag):
        """If only bias maps exist, other groups should still run."""
        # First run to create all
        ocean_sst_diag.run(skip_existing=False)
        plt.close("all")

        # Delete timeseries files
        ts_png = ocean_sst_diag.output_dir / "sst_timeseries.png"
        ts_json = ocean_sst_diag.output_dir / "sst_timeseries.json"
        ts_png.unlink()
        ts_json.unlink()

        # Second run should regenerate timeseries but skip others
        saved = ocean_sst_diag.run(skip_existing=True)
        assert len(saved) == 9
        assert ts_png.exists()
        plt.close("all")

    def test_run_partial_bias_maps(self, ocean_sst_diag):
        """If only some bias maps exist, all 3 should be regenerated."""
        ocean_sst_diag.run(skip_existing=False)
        plt.close("all")

        # Delete one bias map
        annual_png = ocean_sst_diag.output_dir / "sst_annual_bias_combined.png"
        annual_json = ocean_sst_diag.output_dir / "sst_annual_bias_combined.json"
        annual_png.unlink()
        annual_json.unlink()

        saved = ocean_sst_diag.run(skip_existing=True)
        assert len(saved) == 9
        assert annual_png.exists()
        plt.close("all")


# ── Backward compat wrappers ─────────────────────────────────────────


class TestBackwardCompat:
    def test_compute_returns_dict(self, ocean_sst_diag):
        results = ocean_sst_diag.compute()
        assert "bias_maps" in results
        assert "timeseries" in results
        assert "seasonal_cycle" in results
        assert "zonal_mean" in results

    def test_plot_returns_figure_pairs(self, ocean_sst_diag):
        results = ocean_sst_diag.compute()
        figures = ocean_sst_diag.plot(results)
        assert len(figures) == 6
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert isinstance(meta, dict)
            plt.close(fig)


# ── Constructor ──────────────────────────────────────────────────────


class TestConstructor:
    def test_default_ocean_influence_radius(self, mock_ocean_model_loader,
                                            mock_esa_cci_obs_loader, tmp_path):
        """When not configured, uses 20 km default."""
        config = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom"], obs_root="",
            obs_datasets={}, cmip6={"enabled": False}, dask={},
            nereus={},  # no ocean_influence_radius
            output_dir=str(tmp_path / "output"),
        )
        diag = OceanSST(
            model_loader=mock_ocean_model_loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=config,
        )
        assert diag.ocean_influence_radius == 20_000.0

    def test_configured_ocean_influence_radius(self, ocean_sst_diag):
        assert ocean_sst_diag.ocean_influence_radius == 1_000_000

    def test_default_period(self, ocean_sst_diag):
        assert ocean_sst_diag.period == ("1990", "2014")

    def test_default_experiment(self, ocean_sst_diag):
        assert ocean_sst_diag.experiment == "baseline_hist"

    def test_custom_variables(self, mock_ocean_model_loader,
                               mock_esa_cci_obs_loader, ocean_sst_config):
        diag = OceanSST(
            model_loader=mock_ocean_model_loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=ocean_sst_config,
            variables=["tos"],
        )
        assert diag.variables == ["tos"]


# ── Output directory ─────────────────────────────────────────────────


class TestOutputDir:
    def test_output_dir_structure(self, ocean_sst_diag):
        expected = Path(ocean_sst_diag.config.output_dir) / "figures" / "ocean_sst"
        assert ocean_sst_diag.output_dir == expected


# ── Multi-model ──────────────────────────────────────────────────────


class TestMultiModel:
    def test_two_models(self, synth_ocean_healpix, mock_esa_cci_obs_loader,
                        tmp_path):
        """With two models, bias maps should have 2 entries."""
        config = FeatherConfig(
            model_catalogs={},
            models=["ifs-fesom", "ifs-nemo"],
            obs_root="", obs_datasets={},
            cmip6={"enabled": False}, dask={},
            nereus={
                "influence_radius": 1_000_000,
                "ocean_influence_radius": 1_000_000,
                "resolution": 5.0,
            },
            output_dir=str(tmp_path / "output"),
        )
        loader = MockOceanModelLoader(synth_ocean_healpix)
        diag = OceanSST(
            model_loader=loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=config,
        )
        model_monthly, model_coords = diag._load_model_data()
        assert len(model_monthly) == 2
        results = diag._compute_bias_maps(model_monthly, model_coords)
        assert len(results["models"]) == 2
        plt.close("all")


# ── Map land parameter ───────────────────────────────────────────────


class TestMapLandParameter:
    def test_plot_combined_bias_map_accepts_land(self):
        """Verify plot_combined_bias_map accepts land kwarg."""
        from feather.plot.maps import plot_combined_bias_map
        import inspect
        sig = inspect.signature(plot_combined_bias_map)
        assert "land" in sig.parameters

    def test_plot_bias_map_accepts_land(self):
        """Verify plot_bias_map accepts land kwarg."""
        from feather.plot.maps import plot_bias_map
        import inspect
        sig = inspect.signature(plot_bias_map)
        assert "land" in sig.parameters

    def test_bias_maps_pass_land_true(self, ocean_sst_diag):
        """Verify that OceanSST passes land=True to plot_combined_bias_map."""
        model_monthly, model_coords = ocean_sst_diag._load_model_data()
        results = ocean_sst_diag._compute_bias_maps(
            model_monthly, model_coords,
        )

        with patch(
            "feather.plot.maps.plot_combined_bias_map"
        ) as mock_plot:
            mock_fig = MagicMock(spec=plt.Figure)
            mock_plot.return_value = (mock_fig, [])
            ocean_sst_diag._plot_bias_maps(results)

            # Check land=True was passed
            for call in mock_plot.call_args_list:
                assert call.kwargs.get("land") is True


# ── Obs loader integration ───────────────────────────────────────────


class TestObsLoaderIntegration:
    def test_load_esa_cci_method_exists(self):
        """ObsLoader should have load_esa_cci method."""
        from feather.data.obs import ObsLoader
        assert hasattr(ObsLoader, "load_esa_cci")

    def test_load_esa_cci_missing_config(self, tmp_path):
        """load_esa_cci should raise KeyError if ESA_CCI not configured."""
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={}, models=[], obs_root="",
            obs_datasets={},  # no ESA_CCI
            cmip6={"enabled": False}, dask={}, nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(KeyError, match="ESA_CCI"):
            loader.load_esa_cci("timemean")

    def test_load_esa_cci_missing_product(self, tmp_path):
        """load_esa_cci should raise FileNotFoundError for unknown product."""
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={}, models=[], obs_root="",
            obs_datasets={
                "ESA_CCI": {
                    "path": "/fake",
                    "variables": {"timemean": "file.nc"},
                },
            },
            cmip6={"enabled": False}, dask={}, nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(FileNotFoundError, match="unknown_product"):
            loader.load_esa_cci("unknown_product")


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_model(self, synth_ocean_healpix, mock_esa_cci_obs_loader,
                          tmp_path):
        """Single model should work fine."""
        config = FeatherConfig(
            model_catalogs={},
            models=["icon"],
            obs_root="", obs_datasets={},
            cmip6={"enabled": False}, dask={},
            nereus={
                "influence_radius": 1_000_000,
                "ocean_influence_radius": 1_000_000,
                "resolution": 5.0,
            },
            output_dir=str(tmp_path / "output"),
        )
        loader = MockOceanModelLoader(synth_ocean_healpix)
        diag = OceanSST(
            model_loader=loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=config,
        )
        saved = diag.run(skip_existing=False)
        # 3 per-model bias + 3 ensemble bias + ts + seasonal + zonal
        assert len(saved) == 9
        plt.close("all")

    def test_to_plot_time_empty(self):
        """Empty array should return empty."""
        from feather.diag.ocean_sst import _to_plot_time
        result = _to_plot_time(np.array([]))
        assert len(result) == 0

    def test_to_plot_time_numpy(self):
        """Numpy datetime64 should pass through."""
        from feather.diag.ocean_sst import _to_plot_time
        times = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64")
        result = _to_plot_time(times)
        assert len(result) == 2


# ── CMOR K→°C conversion ────────────────────────────────────────────


class TestCMORConversion:
    """CMOR (EERIE) tos is already in °C — no K→°C conversion needed."""

    @pytest.fixture
    def synth_latlon_tos_celsius(self):
        """Synthetic latlon tos data already in °C (as CMOR stores it)."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range(
            "1990-01", periods=12, freq="MS", calendar="standard",
        )
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        # SST in °C: warm equator (~27°C), cold poles (~-3°C)
        temp_base = 27 - 30 * np.abs(lat_grid / 90.0)
        seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
        temp_3d = (
            temp_base[np.newaxis, :, :]
            + seasonal[:, np.newaxis, np.newaxis]
        )
        da = xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )
        return da

    @pytest.fixture
    def cmor_config(self, tmp_path):
        """Config with CMOR data source (like EERIE)."""
        return FeatherConfig(
            model_catalogs={},
            models={
                "IFS-FESOM2-SR": {
                    "institution": "AWI",
                    "experiment": "hist-1950",
                    "variant": "r1i1p1f1",
                    "grids": {"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                    "color": "#1f77b4",
                },
            },
            obs_root="",
            obs_datasets={
                "ESA_CCI": {
                    "path": "/fake",
                    "variables": {
                        "analysed_sst": "monthly.nc",
                        "timemean": "timemean.nc",
                        "ymonmean": "ymonmean.nc",
                    },
                },
            },
            cmip6={"enabled": False},
            dask={},
            nereus={
                "influence_radius": 1_000_000,
                "ocean_influence_radius": 1_000_000,
                "resolution": 5.0,
            },
            output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor", "root": "/fake"},
        )

    def test_cmor_skips_kelvin_conversion(
        self, synth_latlon_tos_celsius, mock_esa_cci_obs_loader, cmor_config,
    ):
        """CMOR tos is already °C — should NOT subtract 273.15."""
        mock_loader = MagicMock()
        mock_loader.load_var.return_value = synth_latlon_tos_celsius

        diag = OceanSST(
            model_loader=mock_loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=cmor_config,
        )
        model_monthly, _ = diag._load_model_data()
        da = model_monthly["IFS-FESOM2-SR"]
        # Mean should be around 12°C (not -261°C from double subtraction)
        assert float(da.mean().values) > -10
        assert float(da.mean().values) < 40

    def test_destine_applies_kelvin_conversion(
        self, ocean_sst_diag,
    ):
        """DestinE tos is in K — should subtract 273.15."""
        model_monthly, _ = ocean_sst_diag._load_model_data()
        da = model_monthly["ifs-fesom"]
        # Original synth data is ~270-300K, Celsius should be ~-3 to 27
        assert float(da.mean().values) < 40
        assert float(da.mean().values) > -10


# ── Kerchunk/mixed-source K→°C conversion ───────────────────────────


class TestKerchunkConversion:
    """Kerchunk parquet tos is in Kelvin — must always apply K→°C.

    Covers the mixed CMOR+kerchunk scenario in eerie_all_members.yaml
    where r1 is CMOR (°C) and r2/r3 are kerchunk_parquet (K).
    """

    @pytest.fixture
    def synth_latlon_tos_kelvin(self):
        """Synthetic latlon tos in Kelvin (as kerchunk data stores it)."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range(
            "1990-01", periods=12, freq="MS", calendar="standard",
        )
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        # SST in K: warm equator (~300K), cold poles (~270K)
        temp_base = 300 - 30 * np.abs(lat_grid / 90.0)
        seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
        temp_3d = (
            temp_base[np.newaxis, :, :]
            + seasonal[:, np.newaxis, np.newaxis]
        )
        da = xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
            attrs={"units": "kelvin"},
        )
        return da

    @pytest.fixture
    def synth_latlon_tos_celsius(self):
        """Synthetic latlon tos in °C (as CMOR stores it)."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range(
            "1990-01", periods=12, freq="MS", calendar="standard",
        )
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        temp_base = 27 - 30 * np.abs(lat_grid / 90.0)
        seasonal = 3 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
        temp_3d = (
            temp_base[np.newaxis, :, :]
            + seasonal[:, np.newaxis, np.newaxis]
        )
        da = xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
            attrs={"units": "degC"},
        )
        return da

    @pytest.fixture
    def mixed_source_config(self, tmp_path):
        """Config with CMOR r1 + kerchunk_parquet r2 (like eerie_all_members)."""
        return FeatherConfig(
            model_catalogs={},
            models={
                "IFS-FESOM2-SR": {
                    "institution": "AWI",
                    "experiment": "hist-1950",
                    "variant": "r1i1p1f1",
                    "grids": {"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                    "color": "#1f77b4",
                    # No data_source_type → inherits global "cmor"
                },
                "IFS-FESOM2-SR-r2": {
                    "institution": "AWI",
                    "experiment": "hist-1950",
                    "variant": "r2i1p1f1",
                    "grids": {"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                    "color": "#5aabdf",
                    "data_source_type": "kerchunk_parquet",
                },
            },
            obs_root="",
            obs_datasets={
                "ESA_CCI": {
                    "path": "/fake",
                    "variables": {
                        "analysed_sst": "monthly.nc",
                        "timemean": "timemean.nc",
                        "ymonmean": "ymonmean.nc",
                    },
                },
            },
            cmip6={"enabled": False},
            dask={},
            nereus={
                "influence_radius": 1_000_000,
                "ocean_influence_radius": 1_000_000,
                "resolution": 5.0,
            },
            output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor", "root": "/fake"},
        )

    def test_needs_celsius_conversion_kelvin_attr(self, synth_latlon_tos_kelvin):
        """units='kelvin' attr forces conversion regardless of data source."""
        from feather.diag.ocean_sst import _needs_celsius_conversion
        assert _needs_celsius_conversion(synth_latlon_tos_kelvin, "cmor") is True

    def test_needs_celsius_conversion_degc_attr(self, synth_latlon_tos_celsius):
        """units='degC' attr suppresses conversion regardless of data source."""
        from feather.diag.ocean_sst import _needs_celsius_conversion
        assert _needs_celsius_conversion(synth_latlon_tos_celsius, "kerchunk_parquet") is False

    def test_needs_celsius_conversion_no_attr_cmor(self):
        """No units attr + CMOR source → heuristic skips conversion."""
        from feather.diag.ocean_sst import _needs_celsius_conversion
        da = xr.DataArray([1.0])  # no units attr
        assert _needs_celsius_conversion(da, "cmor") is False

    def test_needs_celsius_conversion_no_attr_kerchunk(self):
        """No units attr + kerchunk source → heuristic applies conversion."""
        from feather.diag.ocean_sst import _needs_celsius_conversion
        da = xr.DataArray([1.0])  # no units attr
        assert _needs_celsius_conversion(da, "kerchunk_parquet") is True

    def test_kerchunk_applies_kelvin_conversion(
        self,
        synth_latlon_tos_kelvin,
        synth_latlon_tos_celsius,
        mock_esa_cci_obs_loader,
        mixed_source_config,
    ):
        """Kerchunk model (units=kelvin) must be converted; CMOR model must not."""
        def _side_effect(model, variable, **kwargs):
            if model == "IFS-FESOM2-SR":
                return synth_latlon_tos_celsius   # °C
            return synth_latlon_tos_kelvin         # K

        mock_loader = MagicMock()
        mock_loader.load_var.side_effect = _side_effect

        diag = OceanSST(
            model_loader=mock_loader,
            obs_loader=mock_esa_cci_obs_loader,
            config=mixed_source_config,
        )
        model_monthly, _ = diag._load_model_data()

        cmor_da = model_monthly["IFS-FESOM2-SR"]
        kerchunk_da = model_monthly["IFS-FESOM2-SR-r2"]

        # CMOR: already °C — mean should be ~12°C, not ~285°C
        assert float(cmor_da.mean().values) > -10
        assert float(cmor_da.mean().values) < 40

        # Kerchunk: was K, now should be °C — same range
        assert float(kerchunk_da.mean().values) > -10
        assert float(kerchunk_da.mean().values) < 40

        # Kerchunk should NOT be ~285 (unconverted Kelvin)
        assert float(kerchunk_da.mean().values) < 100
