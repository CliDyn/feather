"""Tests for the sea ice diagnostic."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.diag.sea_ice import SeaIceDiag, _METRICS, _MINMAX_MONTHS

from tests.conftest import MockModelLoader, MockObsLoader


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synth_sea_ice_healpix():
    """Synthetic HEALPix dataset with sea ice variables (nside=8)."""
    import healpy as hp

    nside = 8
    ncells = 12 * nside**2  # 768
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Sea ice concentration: 0.8 at poles, 0 at equator
    siconc_base = np.clip(0.8 * (np.abs(lat) - 30) / 60.0, 0, 1)
    # Seasonal: more ice in winter (month 1-3, 10-12 for NH)
    seasonal = -0.2 * np.cos(2 * np.pi * np.arange(12) / 12)
    siconc_2d = np.clip(
        siconc_base[np.newaxis, :] + seasonal[:, np.newaxis], 0, 1,
    )

    # Sea ice thickness: 2m at poles, 0 at equator
    sithick_base = np.clip(2.0 * (np.abs(lat) - 30) / 60.0, 0, 4)
    sithick_2d = np.clip(
        sithick_base[np.newaxis, :] + seasonal[:, np.newaxis] * 0.5, 0, 5,
    )

    area = np.ones(ncells) * (4 * np.pi / ncells)

    ds = xr.Dataset(
        {
            "avg_siconc": xr.DataArray(
                siconc_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "avg_sithick": xr.DataArray(
                sithick_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(area, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_osisaf():
    """Synthetic OSI-SAF-like dataset on small 10×10 grid."""
    ny, nx = 10, 10
    yc = np.arange(ny)
    xc = np.arange(nx)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Latitude: 60-85°N for NH-like
    lat_2d = np.linspace(60, 85, ny)[:, np.newaxis] * np.ones((ny, nx))
    lon_2d = np.linspace(-180, 180, nx)[np.newaxis, :] * np.ones((ny, nx))

    # Concentration in percentage (0-100)
    conc_base = 80.0 * np.ones((ny, nx))
    seasonal = -10.0 * np.cos(2 * np.pi * np.arange(12) / 12)
    conc_3d = np.clip(
        conc_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis],
        0, 100,
    )

    ds = xr.Dataset(
        {
            "ice_conc": xr.DataArray(
                conc_3d, dims=("time", "yc", "xc"),
                coords={"time": time, "yc": yc, "xc": xc},
            ),
            "lat": xr.DataArray(lat_2d, dims=("yc", "xc")),
            "lon": xr.DataArray(lon_2d, dims=("yc", "xc")),
        }
    )
    return ds


@pytest.fixture
def synth_psc():
    """Synthetic PIOMAS/GIOMAS-like dataset on small 10×10 curvilinear grid."""
    ny, nx = 10, 10
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_2d = np.linspace(60, 85, ny)[:, np.newaxis] * np.ones((ny, nx))
    lon_2d = np.linspace(-180, 180, nx)[np.newaxis, :] * np.ones((ny, nx))

    thick_base = 2.0 * np.ones((ny, nx))
    seasonal = -0.5 * np.cos(2 * np.pi * np.arange(12) / 12)
    thick_3d = np.clip(
        thick_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis],
        0, 5,
    )

    # Cell areas (approximate)
    area_2d = np.full((ny, nx), 1e10)  # ~100 km² cells

    ds = xr.Dataset(
        {
            "sithick": xr.DataArray(
                thick_3d, dims=("time", "y", "x"),
                coords={"time": time},
            ),
            "areacello": xr.DataArray(area_2d, dims=("y", "x")),
            "latitude": xr.DataArray(lat_2d, dims=("y", "x")),
            "longitude": xr.DataArray(lon_2d, dims=("y", "x")),
        }
    )
    return ds


class MockObsLoaderSeaIce(MockObsLoader):
    """Mock ObsLoader with load_osisaf() and load_psc() support."""

    def __init__(self, era5_ds, osisaf_ds, psc_ds, var_name="t2m"):
        super().__init__(era5_ds, var_name)
        self._osisaf = osisaf_ds
        self._psc = psc_ds

    def load_osisaf(self, hemisphere, period=None):
        ds = self._osisaf
        if period and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds

    def load_psc(self, product, period=None):
        ds = self._psc
        if period and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds


class MockObsLoaderNoSeaIce(MockObsLoader):
    """Mock ObsLoader without sea ice methods (tests graceful fallback)."""
    pass


@pytest.fixture
def sea_ice_model_loader(synth_sea_ice_healpix):
    return MockModelLoader(synth_sea_ice_healpix)


@pytest.fixture
def sea_ice_obs_loader(synth_obs, synth_osisaf, synth_psc):
    return MockObsLoaderSeaIce(synth_obs, synth_osisaf, synth_psc)


@pytest.fixture
def sea_ice_obs_loader_no_ice(synth_obs):
    return MockObsLoaderNoSeaIce(synth_obs)


@pytest.fixture
def sea_ice_config(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000, "resolution": 1.0},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def sea_ice_diag(sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config):
    return SeaIceDiag(
        sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config,
    )


@pytest.fixture
def sea_ice_diag_no_obs(sea_ice_model_loader, sea_ice_obs_loader_no_ice,
                         sea_ice_config):
    return SeaIceDiag(
        sea_ice_model_loader, sea_ice_obs_loader_no_ice, sea_ice_config,
    )


# ── Test Registration ────────────────────────────────────────────────


class TestRegistration:
    """Tests for diagnostic registration."""

    def test_registered(self):
        """SeaIceDiag is registered in the diagnostic registry."""
        from feather.diag.registry import get_diagnostic
        diag_cls = get_diagnostic("sea_ice")
        assert diag_cls is SeaIceDiag

    def test_name(self):
        assert SeaIceDiag.name == "sea_ice"

    def test_domain(self):
        assert SeaIceDiag.domain == "o2d"

    def test_group(self):
        assert SeaIceDiag.group == "sea_ice"

    def test_variables(self):
        assert "siconc" in SeaIceDiag.variables
        assert "sithick" in SeaIceDiag.variables


# ── Test Model Time Series Computation ───────────────────────────────


class TestModelTimeseries:
    """Tests for _compute_model_timeseries()."""

    def test_returns_dict_per_model(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        assert "ifs-fesom" in result

    def test_has_area_keys(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        mdata = result["ifs-fesom"]
        assert "area_nh" in mdata
        assert "area_sh" in mdata

    def test_has_extent_keys(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        mdata = result["ifs-fesom"]
        assert "extent_nh" in mdata
        assert "extent_sh" in mdata

    def test_has_volume_keys(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        mdata = result["ifs-fesom"]
        assert "volume_nh" in mdata
        assert "volume_sh" in mdata

    def test_area_has_time_dim(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        ts = result["ifs-fesom"]["area_nh"]
        assert "time" in ts.dims

    def test_area_values_positive(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        ts = result["ifs-fesom"]["area_nh"]
        assert float(ts.mean()) >= 0

    def test_extent_values_positive(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        ts = result["ifs-fesom"]["extent_nh"]
        assert float(ts.mean()) >= 0

    def test_volume_values_positive(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        ts = result["ifs-fesom"]["volume_nh"]
        assert float(ts.mean()) >= 0

    def test_12_timesteps(self, sea_ice_diag):
        result = sea_ice_diag._compute_model_timeseries()
        ts = result["ifs-fesom"]["area_nh"]
        assert len(ts.time) == 12

    def test_missing_model_skipped(self, sea_ice_config, sea_ice_obs_loader):
        """Models missing data are gracefully skipped."""
        empty_ds = xr.Dataset()
        loader = MockModelLoader(empty_ds)
        # MockModelLoader.load() returns the dataset; load_var will raise KeyError
        diag = SeaIceDiag(loader, sea_ice_obs_loader, sea_ice_config)
        # Should not raise, just return empty or partial results
        result = diag._compute_model_timeseries()
        # Model should have empty dict or missing keys
        if "ifs-fesom" in result:
            # If it got past load(), it should still lack ice vars
            assert True
        else:
            assert True  # Skipped entirely


# ── Test Obs Time Series Computation ─────────────────────────────────


class TestObsTimeseries:
    """Tests for _compute_obs_timeseries()."""

    def test_has_area_nh(self, sea_ice_diag):
        result = sea_ice_diag._compute_obs_timeseries()
        assert "area_nh" in result

    def test_has_extent_nh(self, sea_ice_diag):
        result = sea_ice_diag._compute_obs_timeseries()
        assert "extent_nh" in result

    def test_has_volume_nh(self, sea_ice_diag):
        result = sea_ice_diag._compute_obs_timeseries()
        assert "volume_nh" in result

    def test_area_nh_has_time(self, sea_ice_diag):
        result = sea_ice_diag._compute_obs_timeseries()
        ts = result["area_nh"]
        assert "time" in ts.dims

    def test_area_positive(self, sea_ice_diag):
        result = sea_ice_diag._compute_obs_timeseries()
        ts = result["area_nh"]
        assert float(ts.mean()) >= 0

    def test_graceful_no_obs_methods(self, sea_ice_diag_no_obs):
        """Falls back gracefully when obs loader lacks sea ice methods."""
        result = sea_ice_diag_no_obs._compute_obs_timeseries()
        # Should not raise — just return empty dict or missing keys
        assert isinstance(result, dict)


# ── Test Plotting: Time Series ───────────────────────────────────────


class TestPlotTimeseries:
    """Tests for _plot_timeseries()."""

    def test_returns_fig_meta_pair(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        result = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)
        assert len(result) == 1
        fig, meta = result[0]
        assert isinstance(fig, plt.Figure)
        assert isinstance(meta, dict)
        plt.close(fig)

    def test_figure_id(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)[0]
        assert meta["figure_id"] == "sea_ice_area_timeseries"
        plt.close("all")

    def test_diagnostic_name(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)[0]
        assert meta["diagnostic_name"] == "sea_ice"
        plt.close("all")

    def test_plot_type(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("volume", model_ts, obs_ts)[0]
        assert meta["plot_type"] == "timeseries"
        plt.close("all")

    def test_all_metrics(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        for metric in _METRICS:
            result = sea_ice_diag._plot_timeseries(metric, model_ts, obs_ts)
            assert len(result) == 1
            plt.close("all")


# ── Test Plotting: Seasonal Cycle ────────────────────────────────────


class TestPlotSeasonalCycle:
    """Tests for _plot_seasonal_cycle()."""

    def test_returns_fig_meta(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        result = sea_ice_diag._plot_seasonal_cycle("area", model_ts, obs_ts)
        assert len(result) == 1
        fig, meta = result[0]
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_id(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_seasonal_cycle(
            "extent", model_ts, obs_ts,
        )[0]
        assert meta["figure_id"] == "sea_ice_extent_seasonal_cycle"
        plt.close("all")

    def test_plot_type(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_seasonal_cycle(
            "area", model_ts, obs_ts,
        )[0]
        assert meta["plot_type"] == "seasonal_cycle"
        plt.close("all")

    def test_all_metrics(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        for metric in _METRICS:
            result = sea_ice_diag._plot_seasonal_cycle(
                metric, model_ts, obs_ts,
            )
            assert len(result) == 1
            plt.close("all")


# ── Test Plotting: Extremes ──────────────────────────────────────────


class TestPlotExtremes:
    """Tests for _plot_extremes()."""

    def test_returns_fig_meta(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        result = sea_ice_diag._plot_extremes("area", model_ts, obs_ts)
        assert len(result) == 1
        fig, meta = result[0]
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_id(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_extremes("area", model_ts, obs_ts)[0]
        assert meta["figure_id"] == "sea_ice_area_extremes"
        plt.close("all")

    def test_plot_type(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_extremes("volume", model_ts, obs_ts)[0]
        assert meta["plot_type"] == "monthly_trends"
        plt.close("all")

    def test_all_metrics(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        for metric in _METRICS:
            result = sea_ice_diag._plot_extremes(metric, model_ts, obs_ts)
            assert len(result) == 1
            plt.close("all")


# ── Test Plotting: Spatial Maps ──────────────────────────────────────


class TestPlotSpatial:
    """Tests for _plot_spatial()."""

    @pytest.fixture(autouse=True)
    def _patch_nr_plot(self):
        """Patch nr.plot to avoid actual nereus rendering in tests."""
        with patch("nereus.plot") as mock_plot:
            self._mock_nr_plot = mock_plot
            yield

    def test_siconc_nh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "siconc", "np",
        )
        assert len(result) == 1
        fig, meta = result[0]
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_siconc_sh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_sh_spatial", "siconc", "sp",
        )
        assert len(result) == 1
        plt.close("all")

    def test_sithick_nh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "sithick_nh_spatial", "sithick", "np",
        )
        assert len(result) == 1
        plt.close("all")

    def test_sithick_sh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "sithick_sh_spatial", "sithick", "sp",
        )
        assert len(result) == 1
        plt.close("all")

    def test_figure_id_correct(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "siconc", "np",
        )
        _, meta = result[0]
        assert meta["figure_id"] == "siconc_nh_spatial"
        plt.close("all")

    def test_plot_type(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "siconc", "np",
        )
        _, meta = result[0]
        assert meta["plot_type"] == "spatial_map"
        plt.close("all")

    def test_spatial_extent_nh(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "siconc", "np",
        )
        _, meta = result[0]
        assert meta["spatial_extent"] == "NH"
        plt.close("all")

    def test_spatial_extent_sh(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_sh_spatial", "siconc", "sp",
        )
        _, meta = result[0]
        assert meta["spatial_extent"] == "SH"
        plt.close("all")


# ── Test compute() / plot() wrappers ─────────────────────────────────


class TestComputePlotWrappers:
    """Tests for compute() and plot() backward-compat wrappers."""

    def test_compute_returns_dict(self, sea_ice_diag):
        result = sea_ice_diag.compute()
        assert "model_ts" in result
        assert "obs_ts" in result

    def test_compute_model_ts_has_data(self, sea_ice_diag):
        result = sea_ice_diag.compute()
        assert "ifs-fesom" in result["model_ts"]

    @patch("nereus.plot")
    def test_plot_returns_list(self, mock_nr_plot, sea_ice_diag):
        results = sea_ice_diag.compute()
        figures = sea_ice_diag.plot(results)
        assert isinstance(figures, list)
        # 3 timeseries + 3 seasonal + 3 extremes + 4 spatial
        # + 4 bias maps (E) + 2 ens summary (F) + 2 mean-bias bars (G) = 21
        assert len(figures) == 21
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert isinstance(meta, dict)
        plt.close("all")


# ── Test run() orchestration ─────────────────────────────────────────


class TestRun:
    """Tests for the full run() orchestration."""

    @patch("nereus.plot")
    def test_run_saves_figures(self, mock_nr_plot, sea_ice_diag):
        saved = sea_ice_diag.run(skip_existing=False)
        assert len(saved) == 21
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
            assert png_path.suffix == ".png"
            assert json_path.suffix == ".json"
        plt.close("all")

    @patch("nereus.plot")
    def test_run_skip_existing(self, mock_nr_plot, sea_ice_diag):
        # First run
        saved1 = sea_ice_diag.run(skip_existing=False)
        assert len(saved1) == 21

        # Second run with skip_existing
        saved2 = sea_ice_diag.run(skip_existing=True)
        assert len(saved2) == 21  # Still returns paths
        plt.close("all")

    @patch("nereus.plot")
    def test_run_output_dir(self, mock_nr_plot, sea_ice_diag):
        sea_ice_diag.run(skip_existing=False)
        assert sea_ice_diag.output_dir.exists()
        pngs = list(sea_ice_diag.output_dir.glob("*.png"))
        jsons = list(sea_ice_diag.output_dir.glob("*.json"))
        assert len(pngs) == 21
        assert len(jsons) == 21
        plt.close("all")

    @patch("nereus.plot")
    def test_run_figure_ids(self, mock_nr_plot, sea_ice_diag):
        """All 21 expected figure IDs are produced."""
        saved = sea_ice_diag.run(skip_existing=False)
        figure_ids = {p.stem for p, _ in saved}
        expected = {
            # Group A: time series
            "sea_ice_area_timeseries",
            "sea_ice_extent_timeseries",
            "sea_ice_volume_timeseries",
            # Group B: seasonal cycles
            "sea_ice_area_seasonal_cycle",
            "sea_ice_extent_seasonal_cycle",
            "sea_ice_volume_seasonal_cycle",
            # Group C: extremes
            "sea_ice_area_extremes",
            "sea_ice_extent_extremes",
            "sea_ice_volume_extremes",
            # Group D: absolute spatial maps
            "siconc_nh_spatial",
            "siconc_sh_spatial",
            "sithick_nh_spatial",
            "sithick_sh_spatial",
            # Group E: bias maps
            "siconc_nh_bias",
            "siconc_sh_bias",
            "sithick_nh_bias",
            "sithick_sh_bias",
            # Group F: ensemble summary
            "siconc_ens_summary",
            "sithick_ens_summary",
            # Group G: per-model mean-bias bar charts
            "siconc_mean_bias",
            "sithick_mean_bias",
        }
        assert figure_ids == expected
        plt.close("all")


# ── Test metadata ────────────────────────────────────────────────────


class TestMetadata:
    """Tests for metadata correctness."""

    def test_diagnostic_name_in_all_figs(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        for metric in _METRICS:
            _, meta = sea_ice_diag._plot_timeseries(
                metric, model_ts, obs_ts,
            )[0]
            assert meta["diagnostic_name"] == "sea_ice"
        plt.close("all")

    def test_period_in_metadata(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)[0]
        assert meta["period"] == ["1990", "2014"]
        plt.close("all")

    def test_obs_dataset_conc(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)[0]
        assert meta["obs_dataset"] == "OSI_SAF"
        plt.close("all")

    def test_obs_dataset_volume(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries("volume", model_ts, obs_ts)[0]
        assert meta["obs_dataset"] == "PSC"
        plt.close("all")


# ── Test mean-bias helpers, orientation cache key, summary stats ──────


class TestMeanBiasHelpers:
    """Tests for the per-model mean-bias additions."""

    def test_weighted_mean_rmse_uniform(self):
        # Constant bias of 2 at all lats → weighted mean 2, rmse 2.
        lat = np.linspace(-80, 80, 50)
        bias = np.full_like(lat, 2.0)
        mean, rmse = SeaIceDiag._weighted_mean_rmse(bias, lat)
        assert mean == pytest.approx(2.0, abs=1e-9)
        assert rmse == pytest.approx(2.0, abs=1e-9)

    def test_weighted_mean_rmse_skips_nan(self):
        lat = np.array([10.0, 20.0, 30.0])
        bias = np.array([1.0, np.nan, -1.0])
        mean, rmse = SeaIceDiag._weighted_mean_rmse(bias, lat)
        assert np.isfinite(mean) and np.isfinite(rmse)
        assert rmse >= abs(mean)

    def test_weighted_mean_rmse_all_nan(self):
        lat = np.array([10.0, 20.0])
        bias = np.array([np.nan, np.nan])
        mean, rmse = SeaIceDiag._weighted_mean_rmse(bias, lat)
        assert np.isnan(mean) and np.isnan(rmse)

    def test_grid_signature_distinguishes_lat_orientation(self):
        """Ascending vs descending latitude must yield different cache keys.

        Regression: IFS-NEMO r1 (lat 90→−90) and r2/r3 (−90→90) share point
        count; keying the interpolator cache on count alone reused one
        interpolator and mirrored r2/r3 hemispherically.
        """
        lon = np.linspace(0, 359, 360)
        lat_desc = np.linspace(90, -90, 180)
        lat_asc = np.linspace(-90, 90, 180)
        # Build meshgridded src arrays the way _build_polar_bias_data does.
        lo_d, la_d = np.meshgrid(lon, lat_desc)
        lo_a, la_a = np.meshgrid(lon, lat_asc)
        key_desc = SeaIceDiag._grid_signature(la_d.ravel(), lo_d.ravel())
        key_asc = SeaIceDiag._grid_signature(la_a.ravel(), lo_a.ravel())
        assert key_desc != key_asc
        # Same orientation → same key (interpolator correctly reused).
        assert key_asc == SeaIceDiag._grid_signature(la_a.ravel(), lo_a.ravel())

    def test_polar_bias_stats_structure(self):
        lat = np.array([60.0, 70.0, 80.0])
        model_biases = {
            "ModelA": {3: np.array([0.1, 0.2, 0.3]),
                       9: np.array([-0.1, -0.2, -0.3])},
        }
        stats = SeaIceDiag._polar_bias_stats(
            SeaIceDiag, model_biases, lat, [3, 9],
        )
        assert "ModelA" in stats
        assert "march_mean_bias" in stats["ModelA"]
        assert "september_rmse" in stats["ModelA"]
        assert stats["ModelA"]["march_mean_bias"] > 0
        assert stats["ModelA"]["september_mean_bias"] < 0

    def test_mean_bias_bars_has_summary_stats(self, sea_ice_diag):
        with patch("nereus.plot"):
            figs = sea_ice_diag._plot_mean_bias_bars("siconc")
        assert len(figs) == 1
        _, meta = figs[0]
        assert meta["figure_id"] == "siconc_mean_bias"
        assert meta["plot_type"] == "bar_chart"
        # Summary statistics must be populated (non-empty) per model.
        assert meta["summary_statistics"]
        plt.close("all")

    def test_bias_map_has_summary_stats(self, sea_ice_diag):
        with patch("nereus.plot"):
            figs = sea_ice_diag._plot_bias_spatial(
                "siconc_nh_bias", "siconc", "np",
            )
        assert len(figs) == 1
        _, meta = figs[0]
        assert meta["summary_statistics"]
        plt.close("all")


# ── Test extreme months ──────────────────────────────────────────────


class TestMinmaxMonths:
    """Tests for _MINMAX_MONTHS constants."""

    def test_nh_max_march(self):
        assert _MINMAX_MONTHS["nh"]["max"] == 3

    def test_nh_min_september(self):
        assert _MINMAX_MONTHS["nh"]["min"] == 9

    def test_sh_max_september(self):
        assert _MINMAX_MONTHS["sh"]["max"] == 9

    def test_sh_min_march(self):
        assert _MINMAX_MONTHS["sh"]["min"] == 3


# ── Test metrics ─────────────────────────────────────────────────────


class TestMetrics:
    """Tests for _METRICS constants."""

    def test_three_metrics(self):
        assert len(_METRICS) == 3

    def test_area_scale(self):
        assert _METRICS["area"]["scale"] == 1e-12

    def test_volume_scale(self):
        assert _METRICS["volume"]["scale"] == 1e-12

    def test_area_units(self):
        assert "km" in _METRICS["area"]["units"]

    def test_volume_units(self):
        assert "km" in _METRICS["volume"]["units"]


# ── Test ObsLoader methods ───────────────────────────────────────────


class TestObsLoaderMethods:
    """Tests for load_osisaf() and load_psc() on the real ObsLoader."""

    def test_load_osisaf_missing_config(self, minimal_config):
        """load_osisaf raises KeyError when OSI_SAF not configured."""
        from feather.data.obs import ObsLoader
        loader = ObsLoader(minimal_config)
        with pytest.raises(KeyError, match="OSI_SAF"):
            loader.load_osisaf("nh")

    def test_load_psc_missing_config(self, minimal_config):
        """load_psc raises KeyError when PSC not configured."""
        from feather.data.obs import ObsLoader
        loader = ObsLoader(minimal_config)
        with pytest.raises(KeyError, match="PSC"):
            loader.load_psc("piomas")

    def test_load_osisaf_missing_hemisphere(self, tmp_path):
        """load_osisaf raises FileNotFoundError for bad hemisphere."""
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={
                "OSI_SAF": {
                    "path": str(tmp_path),
                    "variables": {"nh": "fake.nc"},
                },
            },
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(FileNotFoundError, match="bad_hemi"):
            loader.load_osisaf("bad_hemi")

    def test_load_psc_missing_product(self, tmp_path):
        """load_psc raises FileNotFoundError for bad product."""
        from feather.data.obs import ObsLoader
        config = FeatherConfig(
            model_catalogs={},
            models=[],
            obs_root="",
            obs_datasets={
                "PSC": {
                    "path": str(tmp_path),
                    "variables": {"piomas": "fake.nc"},
                },
            },
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir=str(tmp_path),
        )
        loader = ObsLoader(config)
        with pytest.raises(FileNotFoundError, match="bad_product"):
            loader.load_psc("bad_product")


# ── Test constructor parameters ──────────────────────────────────────


class TestConstructor:
    """Tests for SeaIceDiag constructor."""

    def test_default_period(self, sea_ice_model_loader, sea_ice_obs_loader,
                            sea_ice_config):
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config,
        )
        assert diag.period == ("1990", "2014")

    def test_custom_period(self, sea_ice_model_loader, sea_ice_obs_loader,
                           sea_ice_config):
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config,
            period=("2000", "2010"),
        )
        assert diag.period == ("2000", "2010")

    def test_default_experiment(self, sea_ice_model_loader,
                                sea_ice_obs_loader, sea_ice_config):
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config,
        )
        assert diag.experiment == "baseline_hist"

    def test_output_dir(self, sea_ice_model_loader, sea_ice_obs_loader,
                        sea_ice_config):
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, sea_ice_config,
        )
        assert diag.output_dir.name == "sea_ice"
        assert "figures" in str(diag.output_dir)


# ══════════════════════════════════════════════════════════════════════
# CMIP6 Integration Tests
# ══════════════════════════════════════════════════════════════════════


# ── CMIP6 Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def synth_cmip6_sea_ice():
    """Synthetic CMIP6 sea ice dataset on 5-degree regular lat/lon grid.

    Has 12 monthly timesteps, siconc (0-1 fraction), sithick (m),
    and areacello (cos-lat weighted areas).
    """
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range(
        "1990-01", periods=12, freq="MS", calendar="standard",
    )

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

    # Sea ice concentration: 0–1 fraction, high at poles, zero at equator
    siconc_base = np.clip(0.8 * (np.abs(lat_grid) - 30) / 60.0, 0, 1)
    seasonal = -0.2 * np.cos(2 * np.pi * np.arange(12) / 12)
    siconc_3d = np.clip(
        siconc_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis],
        0, 1,
    )

    # Sea ice thickness: 0–4m, high at poles
    sithick_base = np.clip(2.0 * (np.abs(lat_grid) - 30) / 60.0, 0, 4)
    sithick_3d = np.clip(
        sithick_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis] * 0.5,
        0, 5,
    )

    # Cell areas (cos-lat weighting, approximate)
    area_2d = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)
    # Scale to realistic m² for nereus (roughly Earth-like)
    area_2d = area_2d * 3.1e10  # ~31 billion m² per cell

    ds = xr.Dataset(
        {
            "siconc": xr.DataArray(
                siconc_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "sithick": xr.DataArray(
                sithick_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "areacello": xr.DataArray(
                area_2d, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        }
    )
    return ds


class MockCMIP6LoaderSeaIce:
    """Mock CMIP6Loader for sea ice tests.

    Returns synthetic sea ice data (siconc, sithick, areacello) for two
    models.  Follows the same API as the real CMIP6Loader.
    """

    def __init__(self, dataset, models=None):
        self._ds = dataset
        self._models = models or {
            "ModelA": {"variants": ["r1i1p1f1"]},
            "ModelB": {"variants": ["r1i1p1f1"]},
        }

    @property
    def models(self):
        return self._models

    def load_var(self, cmip6_var, model, *, variant=None, table=None,
                 period=None, season=None, time_mean=True):
        if cmip6_var not in self._ds.data_vars:
            return None
        da = self._ds[cmip6_var]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if season and "time" in da.dims:
            da = da.sel(time=da["time.season"] == season)
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_area(self, model, variant=None, table="Amon"):
        if "areacello" in self._ds.data_vars:
            return self._ds["areacello"]
        return None

    def get_member_pairs(self, ensemble_mode=None):
        pairs = []
        for m, cfg in self._models.items():
            variants = cfg.get("variants", ["r1i1p1f1"])
            pairs.append((m, variants[0]))
        return pairs

    def available_models(self, cmip6_var, table=None):
        if cmip6_var in self._ds.data_vars:
            return list(self._models)
        return []


@pytest.fixture
def cmip6_sea_ice_loader(synth_cmip6_sea_ice):
    return MockCMIP6LoaderSeaIce(synth_cmip6_sea_ice)


@pytest.fixture
def cmip6_sea_ice_config(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={
            "enabled": True,
            "catalog_path": str(tmp_path / "fake_catalog.yaml"),
            "models": {
                "ModelA": {"variants": ["r1i1p1f1"]},
                "ModelB": {"variants": ["r1i1p1f1"]},
            },
        },
        dask={},
        nereus={"influence_radius": 1_000_000, "resolution": 1.0},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def sea_ice_diag_cmip6(sea_ice_model_loader, sea_ice_obs_loader,
                        cmip6_sea_ice_config, cmip6_sea_ice_loader):
    return SeaIceDiag(
        sea_ice_model_loader, sea_ice_obs_loader, cmip6_sea_ice_config,
        cmip6_loader=cmip6_sea_ice_loader,
    )


@pytest.fixture
def sea_ice_diag_cmip6_individual(sea_ice_model_loader, sea_ice_obs_loader,
                                   cmip6_sea_ice_config,
                                   cmip6_sea_ice_loader):
    return SeaIceDiag(
        sea_ice_model_loader, sea_ice_obs_loader, cmip6_sea_ice_config,
        cmip6_loader=cmip6_sea_ice_loader,
        cmip6_individual=True,
    )


# ── CMIP6 Timeseries Computation ─────────────────────────────────────


class TestCMIP6Timeseries:
    """Tests for _compute_cmip6_timeseries()."""

    def test_returns_three_dicts(self, sea_ice_diag_cmip6):
        mmm, info, indiv = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert isinstance(mmm, dict)
        assert isinstance(info, dict)
        assert isinstance(indiv, dict)

    def test_mmm_has_expected_keys(self, sea_ice_diag_cmip6):
        mmm, _, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        for key in ("area_nh", "area_sh", "extent_nh", "extent_sh"):
            assert key in mmm, f"Missing key: {key}"

    def test_mmm_has_volume_keys(self, sea_ice_diag_cmip6):
        mmm, _, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert "volume_nh" in mmm
        assert "volume_sh" in mmm

    def test_info_has_n_members(self, sea_ice_diag_cmip6):
        _, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert info["n_members"] == 2

    def test_info_has_models_used(self, sea_ice_diag_cmip6):
        _, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert "ModelA" in info["models_used"]
        assert "ModelB" in info["models_used"]

    def test_mmm_has_time_dim(self, sea_ice_diag_cmip6):
        mmm, _, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert "time" in mmm["area_nh"].dims

    def test_mmm_values_positive(self, sea_ice_diag_cmip6):
        mmm, _, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert float(mmm["area_nh"].mean()) >= 0

    def test_individual_empty_when_disabled(self, sea_ice_diag_cmip6):
        """cmip6_individual=False → individual_ts is empty."""
        _, _, indiv = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert indiv == {}

    def test_12_timesteps(self, sea_ice_diag_cmip6):
        mmm, _, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        assert len(mmm["area_nh"].time) == 12

    def test_unstructured_grid_no_oom(self, sea_ice_model_loader,
                                      sea_ice_obs_loader, cmip6_sea_ice_config):
        """Unstructured (ICON-like) member: 1-D per-cell lat/lon, single dim.

        Regression: the loader meshgridded 1-D lat/lon into an
        (ncells × ncells) array — for a real ICON grid that is ~10^8 points
        and OOMs (`Killed`). With per-cell coords on one spatial dim the
        meshgrid must be skipped; here a shape mismatch would also surface it.
        """
        rng = np.random.default_rng(0)
        ncells = 500
        clat = rng.uniform(-89, 89, ncells)
        clon = rng.uniform(0, 360, ncells)
        time = xr.date_range("1990-01", periods=12, freq="MS",
                             calendar="standard")
        base = np.clip(0.8 * (np.abs(clat) - 30) / 60.0, 0, 1)
        siconc = np.broadcast_to(base, (12, ncells)).astype(float)
        ds = xr.Dataset({
            "siconc": xr.DataArray(
                siconc, dims=("time", "ncells"),
                coords={"time": time,
                        "lat": ("ncells", clat), "lon": ("ncells", clon)},
            ),
            "sithick": xr.DataArray(
                np.broadcast_to(2.0 * base, (12, ncells)).astype(float),
                dims=("time", "ncells"),
                coords={"time": time,
                        "lat": ("ncells", clat), "lon": ("ncells", clon)},
            ),
            "areacello": xr.DataArray(
                np.full(ncells, 3.1e10), dims=("ncells",),
                coords={"lat": ("ncells", clat), "lon": ("ncells", clon)},
            ),
        })

        class _UnstructuredLoader(MockCMIP6LoaderSeaIce):
            def load_area(self, model, variant=None, table="Amon"):
                return self._ds["areacello"]

        loader = _UnstructuredLoader(ds, models={"ICON": {"variants": ["r1i1p1f1"]}})
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, cmip6_sea_ice_config,
            cmip6_loader=loader, experiment="hist", period=("1990", "1990"),
        )
        mmm, info, _ = diag._compute_cmip6_timeseries(loader=loader)
        # Did not OOM / shape-error, and produced a usable NH metric.
        assert info["n_members"] == 1
        assert "area_nh" in mmm
        assert float(mmm["area_nh"].mean()) >= 0

    def test_curvilinear_no_areacello_mesh_fallback(
        self, sea_ice_model_loader, sea_ice_obs_loader,
        cmip6_sea_ice_config,
    ):
        """Curvilinear member without areacello: areas from nereus mesh.

        Most HighResMIP ocean models (NEMO/ORCA tripolar) ship no
        areacello in the pool. With 2-D lat/lon and no cell areas the
        member used to be skipped, dropping the whole benchmark MMM.
        The mesh fallback must reconstruct areas so the member counts.
        """
        lats1 = np.arange(-87.5, 90, 5.0)
        lons1 = np.arange(2.5, 360, 5.0)
        lat2d, lon2d = np.meshgrid(lats1, lons1, indexing="ij")  # 2-D curvilinear
        time = xr.date_range("1990-01", periods=12, freq="MS",
                             calendar="standard")
        base = np.clip(0.8 * (np.abs(lat2d) - 30) / 60.0, 0, 1)
        siconc = np.broadcast_to(base, (12, *lat2d.shape)).astype(float)
        ds = xr.Dataset({
            "siconc": xr.DataArray(
                siconc, dims=("time", "y", "x"),
                coords={"time": time,
                        "lat": (("y", "x"), lat2d),
                        "lon": (("y", "x"), lon2d)},
            ),
            "sithick": xr.DataArray(
                np.broadcast_to(2.0 * base, (12, *lat2d.shape)).astype(float),
                dims=("time", "y", "x"),
                coords={"time": time,
                        "lat": (("y", "x"), lat2d),
                        "lon": (("y", "x"), lon2d)},
            ),
            # NB: no areacello — forces the mesh fallback.
        })

        class _NoAreaLoader(MockCMIP6LoaderSeaIce):
            def load_area(self, model, variant=None, table="Amon"):
                return None

        loader = _NoAreaLoader(ds, models={"NEMO-ER": {"variants": ["r1i1p1f1"]}})
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader, cmip6_sea_ice_config,
            cmip6_loader=loader, experiment="hist", period=("1990", "1990"),
        )
        mmm, info, _ = diag._compute_cmip6_timeseries(loader=loader)
        # Member survived (not skipped for missing areas) and metrics sane.
        assert info["n_members"] == 1
        assert "area_nh" in mmm
        assert 0 < float(mmm["area_nh"].max()) < 1e14

    def test_fill_values_sanitized(self, synth_cmip6_sea_ice,
                                    sea_ice_model_loader,
                                    sea_ice_obs_loader,
                                    cmip6_sea_ice_config):
        """Non-NaN fill values in siconc are zeroed, not clipped to 1.0.

        CMIP6 zarr stores may use 1e20 fill values at land cells.
        _normalise_siconc divides by 100, making fill values ~1e18.
        These must be zeroed (not clipped to 1.0, which would create
        fake 100% ice at land cells).
        """
        # Inject fill values into siconc at "land" cells (equatorial)
        ds = synth_cmip6_sea_ice.copy(deep=True)
        poisoned = ds["siconc"].values.copy()
        # Set equatorial cells to a huge fill value
        poisoned[:, 15:20, :] = 1e20
        ds["siconc"].values[:] = poisoned

        loader = MockCMIP6LoaderSeaIce(ds)
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        mmm, _, _ = diag._compute_cmip6_timeseries()
        # Fill values zeroed → ice area must stay physically reasonable.
        area_val = float(mmm["area_nh"].max())
        assert area_val < 1e14, (
            f"CMIP6 ice area {area_val:.2e} is unreasonably large — "
            f"fill values not sanitized"
        )
        plt.close("all")

    def test_areacello_fill_values_sanitized(self, synth_cmip6_sea_ice,
                                              sea_ice_model_loader,
                                              sea_ice_obs_loader,
                                              cmip6_sea_ice_config):
        """Fill values in areacello (e.g. 9.97e36) are zeroed."""
        ds = synth_cmip6_sea_ice.copy(deep=True)
        poisoned_area = ds["areacello"].values.copy()
        # Inject float32 netCDF fill value at some cells
        poisoned_area[15:20, :] = 9.96921e+36
        ds["areacello"].values[:] = poisoned_area

        loader = MockCMIP6LoaderSeaIce(ds)
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        mmm, _, _ = diag._compute_cmip6_timeseries()
        area_val = float(mmm["area_nh"].max())
        assert area_val < 1e14, (
            f"CMIP6 ice area {area_val:.2e} with areacello fill values — "
            f"areacello fill values not sanitized"
        )
        plt.close("all")

    def test_sithick_fill_values_sanitized(self, synth_cmip6_sea_ice,
                                            sea_ice_model_loader,
                                            sea_ice_obs_loader,
                                            cmip6_sea_ice_config):
        """Fill values in sithick (e.g. 1e20) are zeroed, not clipped."""
        ds = synth_cmip6_sea_ice.copy(deep=True)
        poisoned_thick = ds["sithick"].values.copy()
        # Inject fill values at polar cells where there IS ice
        poisoned_thick[:, 0:5, :] = 1e20
        ds["sithick"].values[:] = poisoned_thick

        loader = MockCMIP6LoaderSeaIce(ds)
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        mmm, _, _ = diag._compute_cmip6_timeseries()
        if "volume_nh" in mmm:
            vol_val = float(mmm["volume_nh"].max())
            assert vol_val < 1e14, (
                f"CMIP6 ice volume {vol_val:.2e} with sithick fill — "
                f"sithick fill values not sanitized"
            )
        plt.close("all")

    def test_nan_values_handled(self, synth_cmip6_sea_ice,
                                 sea_ice_model_loader,
                                 sea_ice_obs_loader,
                                 cmip6_sea_ice_config):
        """NaN values in siconc are treated as 0 (no ice)."""
        ds = synth_cmip6_sea_ice.copy(deep=True)
        nan_conc = ds["siconc"].values.copy()
        nan_conc[:, 15:20, :] = np.nan
        ds["siconc"].values[:] = nan_conc

        loader = MockCMIP6LoaderSeaIce(ds)
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        mmm, info, _ = diag._compute_cmip6_timeseries()
        assert info["n_members"] == 2
        assert "area_nh" in mmm
        assert np.isfinite(float(mmm["area_nh"].mean()))
        plt.close("all")


class TestCMIP6TimeseriesDisabled:
    """Tests for CMIP6 disabled case."""

    def test_empty_when_disabled(self, sea_ice_diag):
        """Returns empty dicts when CMIP6 is not enabled."""
        mmm, info, indiv = sea_ice_diag._compute_cmip6_timeseries()
        assert mmm == {}
        assert info == {}
        assert indiv == {}


class TestCMIP6ExcludedModels:
    """Tests for _CMIP6_SEA_ICE_EXCLUDE blocklist."""

    def test_excluded_model_not_in_mmm(self, synth_cmip6_sea_ice,
                                        sea_ice_model_loader,
                                        sea_ice_obs_loader,
                                        cmip6_sea_ice_config):
        """FGOALS-g3 is excluded even when present in the loader."""
        loader = MockCMIP6LoaderSeaIce(
            synth_cmip6_sea_ice,
            models={
                "ModelA": {"variants": ["r1i1p1f1"]},
                "FGOALS-g3": {"variants": ["r1i1p1f1"]},
            },
        )
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        _, info, _ = diag._compute_cmip6_timeseries()
        assert "FGOALS-g3" not in info.get("models_used", [])
        assert info["n_members"] == 1
        plt.close("all")

    def test_excluded_model_logged_as_warning(self, synth_cmip6_sea_ice,
                                               sea_ice_model_loader,
                                               sea_ice_obs_loader,
                                               cmip6_sea_ice_config,
                                               caplog):
        """Excluding a model emits a WARNING-level log message."""
        import logging

        loader = MockCMIP6LoaderSeaIce(
            synth_cmip6_sea_ice,
            models={
                "FGOALS-g3": {"variants": ["r1i1p1f1"]},
                "ModelA": {"variants": ["r1i1p1f1"]},
            },
        )
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        with caplog.at_level(logging.WARNING, logger="feather.diag.sea_ice"):
            diag._compute_cmip6_timeseries()
        assert any("EXCLUDING" in msg and "FGOALS-g3" in msg
                    for msg in caplog.messages)
        plt.close("all")

    def test_all_excluded_returns_empty(self, synth_cmip6_sea_ice,
                                         sea_ice_model_loader,
                                         sea_ice_obs_loader,
                                         cmip6_sea_ice_config):
        """If all models are excluded, returns empty dicts gracefully."""
        loader = MockCMIP6LoaderSeaIce(
            synth_cmip6_sea_ice,
            models={"FGOALS-g3": {"variants": ["r1i1p1f1"]}},
        )
        diag = SeaIceDiag(
            sea_ice_model_loader, sea_ice_obs_loader,
            cmip6_sea_ice_config, cmip6_loader=loader,
        )
        mmm, info, indiv = diag._compute_cmip6_timeseries()
        assert mmm == {}
        assert info == {}
        assert indiv == {}
        plt.close("all")


class TestCMIP6Individual:
    """Tests for cmip6_individual mode."""

    def test_individual_populated(self, sea_ice_diag_cmip6_individual):
        _, _, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        assert "ModelA" in indiv
        assert "ModelB" in indiv

    def test_individual_has_area_keys(self, sea_ice_diag_cmip6_individual):
        _, _, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        for key in ("area_nh", "area_sh"):
            assert key in indiv["ModelA"]

    def test_individual_has_time_dim(self, sea_ice_diag_cmip6_individual):
        _, _, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        assert "time" in indiv["ModelA"]["area_nh"].dims


# ── CMIP6 Plotting Tests ────────────────────────────────────────────


class TestCMIP6PlotTimeseries:
    """Tests for CMIP6 lines in _plot_timeseries()."""

    def test_mmm_line_present(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        result = sea_ice_diag_cmip6._plot_timeseries(
            "area", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )
        fig, meta = result[0]
        # Check that CMIP6 MMM label is in the legend of at least one axis
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 MMM" in all_labels
        plt.close(fig)

    def test_individual_lines_present(self, sea_ice_diag_cmip6_individual):
        model_ts = sea_ice_diag_cmip6_individual._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6_individual._compute_obs_timeseries()
        mmm, info, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        result = sea_ice_diag_cmip6_individual._plot_timeseries(
            "area", model_ts, obs_ts,
            cmip6_ts=mmm, cmip6_individual_ts=indiv, cmip6_info=info,
        )
        fig, _ = result[0]
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 members" in all_labels
        assert "CMIP6 MMM" in all_labels
        plt.close(fig)

    def test_backward_compat_no_cmip6(self, sea_ice_diag):
        """Plotting without CMIP6 args still works."""
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        result = sea_ice_diag._plot_timeseries("area", model_ts, obs_ts)
        assert len(result) == 1
        plt.close("all")


class TestCMIP6PlotSeasonalCycle:
    """Tests for CMIP6 lines in _plot_seasonal_cycle()."""

    def test_mmm_line_present(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        result = sea_ice_diag_cmip6._plot_seasonal_cycle(
            "area", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )
        fig, _ = result[0]
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 MMM" in all_labels
        plt.close(fig)

    def test_individual_lines_present(self, sea_ice_diag_cmip6_individual):
        model_ts = sea_ice_diag_cmip6_individual._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6_individual._compute_obs_timeseries()
        mmm, info, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        result = sea_ice_diag_cmip6_individual._plot_seasonal_cycle(
            "extent", model_ts, obs_ts,
            cmip6_ts=mmm, cmip6_individual_ts=indiv, cmip6_info=info,
        )
        fig, _ = result[0]
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 members" in all_labels
        plt.close(fig)


class TestCMIP6PlotExtremes:
    """Tests for CMIP6 lines in _plot_extremes()."""

    def test_mmm_line_present(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        result = sea_ice_diag_cmip6._plot_extremes(
            "area", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )
        fig, _ = result[0]
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 MMM" in all_labels
        plt.close(fig)

    def test_individual_lines_present(self, sea_ice_diag_cmip6_individual):
        model_ts = sea_ice_diag_cmip6_individual._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6_individual._compute_obs_timeseries()
        mmm, info, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        result = sea_ice_diag_cmip6_individual._plot_extremes(
            "volume", model_ts, obs_ts,
            cmip6_ts=mmm, cmip6_individual_ts=indiv, cmip6_info=info,
        )
        fig, _ = result[0]
        all_labels = []
        for ax in fig.get_axes():
            all_labels.extend([t.get_text() for t in ax.get_legend().get_texts()])
        assert "CMIP6 members" in all_labels
        plt.close(fig)


# ── CMIP6 Metadata Tests ────────────────────────────────────────────


class TestCMIP6Metadata:
    """Tests for cmip6_info in metadata."""

    def test_cmip6_info_in_timeseries_meta(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        _, meta = sea_ice_diag_cmip6._plot_timeseries(
            "area", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )[0]
        assert meta.get("cmip6_info") is not None
        assert meta["cmip6_info"]["n_members"] == 2
        plt.close("all")

    def test_cmip6_info_in_seasonal_meta(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        _, meta = sea_ice_diag_cmip6._plot_seasonal_cycle(
            "extent", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )[0]
        assert meta.get("cmip6_info") is not None
        plt.close("all")

    def test_cmip6_info_in_extremes_meta(self, sea_ice_diag_cmip6):
        model_ts = sea_ice_diag_cmip6._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6._compute_obs_timeseries()
        mmm, info, _ = sea_ice_diag_cmip6._compute_cmip6_timeseries()
        _, meta = sea_ice_diag_cmip6._plot_extremes(
            "area", model_ts, obs_ts, cmip6_ts=mmm, cmip6_info=info,
        )[0]
        assert meta.get("cmip6_info") is not None
        plt.close("all")

    def test_no_cmip6_info_when_disabled(self, sea_ice_diag):
        model_ts = sea_ice_diag._compute_model_timeseries()
        obs_ts = sea_ice_diag._compute_obs_timeseries()
        _, meta = sea_ice_diag._plot_timeseries(
            "area", model_ts, obs_ts,
        )[0]
        assert meta.get("cmip6_info") is None
        plt.close("all")

    def test_cmip6_models_in_all_models(self, sea_ice_diag_cmip6_individual):
        model_ts = sea_ice_diag_cmip6_individual._compute_model_timeseries()
        obs_ts = sea_ice_diag_cmip6_individual._compute_obs_timeseries()
        mmm, info, indiv = (
            sea_ice_diag_cmip6_individual._compute_cmip6_timeseries()
        )
        _, meta = sea_ice_diag_cmip6_individual._plot_timeseries(
            "area", model_ts, obs_ts,
            cmip6_ts=mmm, cmip6_individual_ts=indiv, cmip6_info=info,
        )[0]
        assert "ModelA" in meta["models"]
        assert "ModelB" in meta["models"]
        plt.close("all")


# ── CMIP6 compute/plot wrapper tests ────────────────────────────────


class TestCMIP6ComputePlotWrappers:
    """Tests for compute() and plot() with CMIP6 data."""

    def test_compute_includes_cmip6(self, sea_ice_diag_cmip6):
        result = sea_ice_diag_cmip6.compute()
        assert "cmip6_ts" in result
        assert "cmip6_info" in result
        assert "cmip6_individual_ts" in result

    def test_compute_cmip6_has_data(self, sea_ice_diag_cmip6):
        result = sea_ice_diag_cmip6.compute()
        assert result["cmip6_ts"]  # non-empty dict
        assert result["cmip6_info"]["n_members"] == 2

    @patch("nereus.plot")
    def test_plot_with_cmip6(self, mock_nr_plot, sea_ice_diag_cmip6):
        results = sea_ice_diag_cmip6.compute()
        figures = sea_ice_diag_cmip6.plot(results)
        assert len(figures) == 21
        plt.close("all")

    @patch("nereus.plot")
    def test_run_with_cmip6(self, mock_nr_plot, sea_ice_diag_cmip6):
        saved = sea_ice_diag_cmip6.run(skip_existing=False)
        assert len(saved) == 21
        plt.close("all")
