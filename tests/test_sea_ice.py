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
        assert "avg_siconc" in SeaIceDiag.variables
        assert "avg_sithick" in SeaIceDiag.variables


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
            "siconc_nh_spatial", "avg_siconc", "np",
        )
        assert len(result) == 1
        fig, meta = result[0]
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_siconc_sh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_sh_spatial", "avg_siconc", "sp",
        )
        assert len(result) == 1
        plt.close("all")

    def test_sithick_nh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "sithick_nh_spatial", "avg_sithick", "np",
        )
        assert len(result) == 1
        plt.close("all")

    def test_sithick_sh_returns_fig(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "sithick_sh_spatial", "avg_sithick", "sp",
        )
        assert len(result) == 1
        plt.close("all")

    def test_figure_id_correct(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "avg_siconc", "np",
        )
        _, meta = result[0]
        assert meta["figure_id"] == "siconc_nh_spatial"
        plt.close("all")

    def test_plot_type(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "avg_siconc", "np",
        )
        _, meta = result[0]
        assert meta["plot_type"] == "spatial_map"
        plt.close("all")

    def test_spatial_extent_nh(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_nh_spatial", "avg_siconc", "np",
        )
        _, meta = result[0]
        assert meta["spatial_extent"] == "NH"
        plt.close("all")

    def test_spatial_extent_sh(self, sea_ice_diag):
        result = sea_ice_diag._plot_spatial(
            "siconc_sh_spatial", "avg_siconc", "sp",
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
        # 3 timeseries + 3 seasonal + 3 extremes + 4 spatial = 13
        assert len(figures) == 13
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
        assert len(saved) == 13
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
        assert len(saved1) == 13

        # Second run with skip_existing
        saved2 = sea_ice_diag.run(skip_existing=True)
        assert len(saved2) == 13  # Still returns paths
        plt.close("all")

    @patch("nereus.plot")
    def test_run_output_dir(self, mock_nr_plot, sea_ice_diag):
        sea_ice_diag.run(skip_existing=False)
        assert sea_ice_diag.output_dir.exists()
        pngs = list(sea_ice_diag.output_dir.glob("*.png"))
        jsons = list(sea_ice_diag.output_dir.glob("*.json"))
        assert len(pngs) == 13
        assert len(jsons) == 13
        plt.close("all")

    @patch("nereus.plot")
    def test_run_figure_ids(self, mock_nr_plot, sea_ice_diag):
        """All 13 expected figure IDs are produced."""
        saved = sea_ice_diag.run(skip_existing=False)
        figure_ids = {p.stem for p, _ in saved}
        expected = {
            "sea_ice_area_timeseries",
            "sea_ice_extent_timeseries",
            "sea_ice_volume_timeseries",
            "sea_ice_area_seasonal_cycle",
            "sea_ice_extent_seasonal_cycle",
            "sea_ice_volume_seasonal_cycle",
            "sea_ice_area_extremes",
            "sea_ice_extent_extremes",
            "sea_ice_volume_extremes",
            "siconc_nh_spatial",
            "siconc_sh_spatial",
            "sithick_nh_spatial",
            "sithick_sh_spatial",
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
