"""Tests for the climate_variability diagnostic."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.diag.climate_variability import ClimateVariability
from feather.util.temporal import deseason, detrend
from tests.conftest import MockCMIP6Loader


# Large influence radius for nside=8 test data (~815 km spacing)
_TEST_INFLUENCE_RADIUS = 1_000_000


# ============================================================================
# Helper fixtures
# ============================================================================


@pytest.fixture
def synth_healpix_multi():
    """Synthetic HEALPix dataset with 36 months for variability testing."""
    import healpy as hp

    nside = 8
    ncells = 12 * nside**2  # 768
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    temp_base = 300 - 40 * np.abs(lat / 90.0)
    time = xr.date_range("1990-01", periods=36, freq="MS", calendar="standard")

    # Seasonal cycle + small trend + random noise for variability
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(36) % 12 - 3) / 12)
    trend = np.linspace(0, 0.3, 36)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((36, ncells)) * 0.5

    temp_2d = (
        temp_base[np.newaxis, :] + seasonal[:, np.newaxis]
        + trend[:, np.newaxis] + noise
    )
    area = np.ones(ncells) * (4 * np.pi / ncells)

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
def synth_obs_multi():
    """Synthetic observation with 36 months for variability testing."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=36, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(36) % 12 - 3) / 12)
    trend = np.linspace(0, 0.3, 36)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((36, len(lats), len(lons))) * 0.5

    temp_3d = (
        temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]
        + trend[:, np.newaxis, np.newaxis] + noise
    )

    ds = xr.Dataset(
        {
            "t2m": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        }
    )
    return ds


@pytest.fixture
def synth_cmip6_multi():
    """Synthetic CMIP6 dataset with 36 months for variability testing."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=36, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(36) % 12 - 3) / 12)
    trend = np.linspace(0, 0.3, 36)
    rng = np.random.default_rng(123)
    noise = rng.standard_normal((36, len(lats), len(lons))) * 0.5

    temp_3d = (
        temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]
        + trend[:, np.newaxis, np.newaxis] + noise
    )

    area_2d = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "areacella": xr.DataArray(
                area_2d, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        }
    )
    return ds


# Mock loaders using multi-month data

class MockModelLoaderMulti:
    """Mock DataLoader for 36-month data."""

    def __init__(self, dataset):
        self._ds = dataset

    def load(self, key):
        return self._ds

    def load_var(self, key, variable):
        return self._ds[variable]

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        from feather.data.loader import DataLoader
        return DataLoader.make_key(experiment, model, domain, member)


class MockObsLoaderMulti:
    """Mock ObsLoader for 36-month data."""

    def __init__(self, dataset, var_name="t2m"):
        self._ds = dataset
        self._var_name = var_name

    def load(self, dataset, variable, period=None):
        da = self._ds[self._var_name]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_for_model_var(self, model_var, period=None):
        return self.load(None, None, period=period)


@pytest.fixture
def mock_model_loader_multi(synth_healpix_multi):
    return MockModelLoaderMulti(synth_healpix_multi)


@pytest.fixture
def mock_obs_loader_multi(synth_obs_multi):
    return MockObsLoaderMulti(synth_obs_multi)


@pytest.fixture
def mock_cmip6_loader_multi(synth_cmip6_multi):
    return MockCMIP6Loader(synth_cmip6_multi)


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal config for variability tests."""
    from feather.config import FeatherConfig
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": _TEST_INFLUENCE_RADIUS},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def cmip6_config(tmp_path):
    """Config with CMIP6 enabled."""
    from feather.config import FeatherConfig
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={
            "enabled": True,
            "catalog_path": str(tmp_path / "fake_catalog.yaml"),
            "regrid_resolution": 1.0,
            "influence_radius": _TEST_INFLUENCE_RADIUS,
            "ensemble_mode": "one_per_model",
            "models": {
                "ModelA": {"variants": ["r1i1p1f1"]},
                "ModelB": {"variants": ["r1i1p1f1"]},
            },
        },
        dask={},
        nereus={"influence_radius": _TEST_INFLUENCE_RADIUS},
        output_dir=str(tmp_path / "output"),
    )


# ============================================================================
# A. Temporal preprocessing tests
# ============================================================================


class TestDeseasDetrend:
    """Tests for the deseason+detrend pipeline on synthetic data."""

    def test_deseason_removes_seasonal_cycle(self, synth_obs_multi):
        """After deseason, monthly means should be ~0."""
        da = synth_obs_multi["t2m"]
        deseas = deseason(da)
        monthly_means = deseas.groupby("time.month").mean("time")
        # Mean across all months and space should be near zero
        assert float(monthly_means.mean().values) < 0.1

    def test_detrend_removes_trend(self, synth_obs_multi):
        """After detrend, residual trend should be ~0."""
        da = synth_obs_multi["t2m"]
        deseas = deseason(da)
        detrended = detrend(deseas)
        from feather.util.temporal import linear_trend
        trend = linear_trend(detrended)
        assert float(np.abs(trend).mean()) < 0.1

    def test_std_is_positive(self, synth_obs_multi):
        """STD of deseasonalised, detrended data should be positive."""
        da = synth_obs_multi["t2m"]
        deseas = deseason(da)
        detrended = detrend(deseas)
        std = detrended.std("time")
        assert float(std.min()) >= 0.0


# ============================================================================
# B. Compute tests
# ============================================================================


class TestClimateVariabilityCompute:
    """Tests for ClimateVariability._compute_variable()."""

    def test_compute_returns_expected_keys(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """compute() returns expected nested structure."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert "tas" in results
        vr = results["tas"]
        assert "obs_std" in vr
        assert "obs_gmean" in vr
        assert "models" in vr
        assert "var_info" in vr
        assert "colorbar_ranges" in vr
        assert "target_lats" in vr
        assert "target_lons" in vr

    def test_compute_model_results_structure(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """Each model entry has std_regrid, std_diff, stats."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert "std_regrid" in mdata
        assert "std_diff" in mdata
        assert "std_gmean" in mdata
        assert "diff_gmean" in mdata
        assert "rmse" in mdata

    def test_std_values_are_positive(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """All STD values should be >= 0."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        vr = results["tas"]
        assert float(vr["obs_std"].min()) >= 0.0
        for mdata in vr["models"].values():
            assert float(mdata["std_regrid"].min()) >= 0.0

    def test_obs_gmean_is_float(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """obs_gmean is a float."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        assert isinstance(results["tas"]["obs_gmean"], float)

    def test_colorbar_ranges_structure(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """colorbar_ranges has std and diff entries."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        cb = results["tas"]["colorbar_ranges"]
        assert "std" in cb
        assert "diff" in cb
        assert "vmin" in cb["std"]
        assert "vmax" in cb["std"]
        assert "bias_vmax" in cb["diff"]

    def test_std_vmin_le_vmax(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """STD vmin <= vmax."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        cb = results["tas"]["colorbar_ranges"]
        assert cb["std"]["vmin"] <= cb["std"]["vmax"]

    def test_missing_model_variable_skipped(
        self, mock_obs_loader_multi, minimal_config,
    ):
        """Model missing a variable -> warning, skip."""
        class MissingLoader:
            def load(self, key):
                raise KeyError("not found")
            def load_var(self, key, variable):
                raise KeyError("not found")
            @staticmethod
            def make_key(*args, **kwargs):
                from feather.data.loader import DataLoader
                return DataLoader.make_key(*args, **kwargs)

        diag = ClimateVariability(
            MissingLoader(), mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        assert "tas" not in results  # no models -> None -> skipped

    def test_no_models_returns_none(
        self, mock_obs_loader_multi, minimal_config,
    ):
        """Returns None when no models have the variable."""
        class MissingLoader:
            def load_var(self, key, variable):
                raise FileNotFoundError("not found")
            @staticmethod
            def make_key(*args, **kwargs):
                from feather.data.loader import DataLoader
                return DataLoader.make_key(*args, **kwargs)

        diag = ClimateVariability(
            MissingLoader(), mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        result = diag._compute_variable("tas")
        assert result is None

    def test_custom_variables(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """Constructor variables= overrides class default."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        assert diag.variables == ["tas"]

    def test_custom_period(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """Constructor period= is stored."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
            period=("1991", "1992"),
        )
        assert diag.period == ("1991", "1992")

    def test_rmse_is_reasonable(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """RMSE should be a finite positive number."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        mdata = results["tas"]["models"]["ifs-fesom"]
        assert mdata["rmse"] >= 0.0
        assert np.isfinite(mdata["rmse"])


# ============================================================================
# C. Plot tests (mocked)
# ============================================================================


class TestClimateVariabilityPlot:
    """Tests for ClimateVariability._plot_variable()."""

    def _make_var_result(self):
        """Create a minimal var_result dict for plotting."""
        from feather.data.variables import get_var

        var_info = get_var("tas")
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        shape = (len(lats), len(lons))

        rng = np.random.default_rng(42)
        obs_std = xr.DataArray(
            rng.uniform(0.5, 5.0, shape),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        model_std = xr.DataArray(
            rng.uniform(0.5, 5.0, shape),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        std_diff = model_std - obs_std

        return {
            "obs_std": obs_std,
            "obs_gmean": 2.5,
            "models": {
                "ifs-fesom": {
                    "std_regrid": model_std,
                    "std_diff": std_diff,
                    "std_gmean": 2.6,
                    "diff_gmean": 0.1,
                    "rmse": 0.5,
                },
            },
            "var_info": var_info,
            "target_lats": lats,
            "target_lons": lons,
            "colorbar_ranges": {
                "std": {"vmin": 0.0, "vmax": 5.0},
                "diff": {"bias_vmax": 2.0},
            },
            "cmip6_data": {},
            "cmip6_info": {},
            "cmip6_individual_data": {},
        }

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_plot_returns_two_figures(self, mock_bias, mock_map,
                                      mock_model_loader_multi,
                                      mock_obs_loader_multi,
                                      minimal_config):
        """Two figures per variable (STD map + diff map)."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        figures = diag._plot_variable("tas", vr)

        assert len(figures) == 2

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_figure_ids(self, mock_bias, mock_map,
                         mock_model_loader_multi,
                         mock_obs_loader_multi,
                         minimal_config):
        """Figure IDs are {var}_std_combined and {var}_std_diff_combined."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        figures = diag._plot_variable("tas", vr)

        ids = [meta["figure_id"] for _, meta in figures]
        assert "tas_std_combined" in ids
        assert "tas_std_diff_combined" in ids

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_metadata_diagnostic_name(self, mock_bias, mock_map,
                                       mock_model_loader_multi,
                                       mock_obs_loader_multi,
                                       minimal_config):
        """Metadata has diagnostic_name = 'climate_variability'."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        figures = diag._plot_variable("tas", vr)

        for _, meta in figures:
            assert meta["diagnostic_name"] == "climate_variability"

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_metadata_plot_types(self, mock_bias, mock_map,
                                  mock_model_loader_multi,
                                  mock_obs_loader_multi,
                                  minimal_config):
        """Plot types: combined_map and combined_bias_map."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        figures = diag._plot_variable("tas", vr)

        plot_types = {meta["plot_type"] for _, meta in figures}
        assert "combined_map" in plot_types
        assert "combined_bias_map" in plot_types

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_metadata_has_summary_statistics(self, mock_bias, mock_map,
                                              mock_model_loader_multi,
                                              mock_obs_loader_multi,
                                              minimal_config):
        """Metadata includes summary_statistics for models."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        figures = diag._plot_variable("tas", vr)

        for _, meta in figures:
            stats = meta.get("summary_statistics", {})
            assert "ifs-fesom" in stats

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_combined_map_receives_correct_data(
        self, mock_bias, mock_map,
        mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """plot_combined_map is called with obs + model data dict."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        diag._plot_variable("tas", vr)

        # plot_combined_map should have been called once
        assert mock_map.call_count == 1
        call_args = mock_map.call_args
        data_dict = call_args[0][0]
        assert "ERA5" in data_dict  # obs dataset
        assert "ifs-fesom" in data_dict

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_combined_bias_map_receives_correct_data(
        self, mock_bias, mock_map,
        mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """plot_combined_bias_map is called with obs STD + bias dict."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        vr = self._make_var_result()
        diag._plot_variable("tas", vr)

        # plot_combined_bias_map should have been called once
        assert mock_bias.call_count == 1
        call_args = mock_bias.call_args
        # First positional arg is obs_data
        obs = call_args[0][0]
        assert obs.shape == vr["obs_std"].shape
        # Second positional arg is bias_dict
        bias_dict = call_args[0][1]
        assert "ifs-fesom" in bias_dict


# ============================================================================
# D. Run tests
# ============================================================================


class TestClimateVariabilityRun:
    """Tests for ClimateVariability.run()."""

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_run_saves_files(self, mock_bias, mock_map,
                              mock_model_loader_multi,
                              mock_obs_loader_multi,
                              minimal_config):
        """run() saves both PNG and JSON for each figure."""
        def _make_fig():
            fig = MagicMock(spec=plt.Figure)
            fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
            return fig

        mock_map.return_value = (_make_fig(), [None])
        mock_bias.return_value = (_make_fig(), [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        saved = diag.run(skip_existing=False)

        assert len(saved) == 2  # 2 figures per variable
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()

    def test_run_skip_existing(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """run() skips variable when both figure IDs exist."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        # Pre-create both figures
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        for fid in ["tas_std_combined", "tas_std_diff_combined"]:
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        saved = diag.run(skip_existing=True)
        assert len(saved) == 2
        # Files should not have been overwritten
        for png_path, _ in saved:
            assert png_path.read_bytes() == b"fake"

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_run_partial_skip(self, mock_bias, mock_map,
                               mock_model_loader_multi,
                               mock_obs_loader_multi,
                               minimal_config):
        """run() recomputes when only one of two figures exists."""
        def _make_fig():
            fig = MagicMock(spec=plt.Figure)
            fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
            return fig

        mock_map.return_value = (_make_fig(), [None])
        mock_bias.return_value = (_make_fig(), [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        # Only create one figure
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        fid = "tas_std_combined"
        (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
        (diag.output_dir / f"{fid}.json").write_text("{}")

        saved = diag.run(skip_existing=True)
        # Should recompute (2 new figures)
        assert len(saved) == 2

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_run_exception_per_variable(self, mock_bias, mock_map,
                                         mock_model_loader_multi,
                                         mock_obs_loader_multi,
                                         minimal_config):
        """run() catches per-variable exceptions and continues."""
        mock_map.side_effect = RuntimeError("boom")
        mock_bias.return_value = (MagicMock(spec=plt.Figure), [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        # Should not raise
        saved = diag.run(skip_existing=False)
        # Fails during _plot_variable, so nothing saved
        assert len(saved) == 0


# ============================================================================
# E. CMIP6 tests
# ============================================================================


class TestClimateVariabilityCMIP6:
    """Tests for CMIP6 integration."""

    def test_cmip6_mmm_added_to_results(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        cmip6_config, mock_cmip6_loader_multi,
    ):
        """CMIP6 MMM STD appears in results."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, cmip6_config,
            cmip6_loader=mock_cmip6_loader_multi,
            variables=["tas"],
        )
        results = diag.compute()

        vr = results["tas"]
        cmip6_data = vr["cmip6_data"]
        assert "std_regrid" in cmip6_data
        assert "std_diff" in cmip6_data
        assert "diff_gmean" in cmip6_data
        assert "rmse" in cmip6_data

    def test_cmip6_info_populated(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        cmip6_config, mock_cmip6_loader_multi,
    ):
        """CMIP6 info has n_members and models_used."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, cmip6_config,
            cmip6_loader=mock_cmip6_loader_multi,
            variables=["tas"],
        )
        results = diag.compute()

        info = results["tas"]["cmip6_info"]
        assert "n_members" in info
        assert info["n_members"] > 0
        assert "models_used" in info

    def test_cmip6_individual_mode(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        cmip6_config, mock_cmip6_loader_multi,
    ):
        """Individual + MMM both present when cmip6_individual=True."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, cmip6_config,
            cmip6_loader=mock_cmip6_loader_multi,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        vr = results["tas"]
        assert vr["cmip6_data"]  # MMM present
        assert vr["cmip6_individual_data"]  # individual present
        # Individual data keyed by model/variant labels
        for label, cdata in vr["cmip6_individual_data"].items():
            assert "std_regrid" in cdata
            assert "std_diff" in cdata

    def test_cmip6_disabled_no_error(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        minimal_config,
    ):
        """Works fine without CMIP6."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        vr = results["tas"]
        assert vr["cmip6_data"] == {}
        assert vr["cmip6_individual_data"] == {}

    @patch("feather.diag.climate_variability.plot_combined_map")
    @patch("feather.diag.climate_variability.plot_combined_bias_map")
    def test_cmip6_mmm_in_plot_data(
        self, mock_bias, mock_map,
        mock_model_loader_multi, mock_obs_loader_multi,
        cmip6_config, mock_cmip6_loader_multi,
    ):
        """CMIP6 MMM appears in plot data dict."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_map.return_value = (mock_fig, [None])
        mock_bias.return_value = (mock_fig, [None])

        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, cmip6_config,
            cmip6_loader=mock_cmip6_loader_multi,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])

        # Check that CMIP6 MMM is in the model list
        for _, meta in figures:
            assert "CMIP6 MMM" in meta["models"]

    def test_cmip6_std_positive(
        self, mock_model_loader_multi, mock_obs_loader_multi,
        cmip6_config, mock_cmip6_loader_multi,
    ):
        """CMIP6 MMM STD should be positive."""
        diag = ClimateVariability(
            mock_model_loader_multi, mock_obs_loader_multi, cmip6_config,
            cmip6_loader=mock_cmip6_loader_multi,
            variables=["tas"],
        )
        results = diag.compute()
        cmip6_std = results["tas"]["cmip6_data"]["std_regrid"]
        assert float(cmip6_std.min()) >= 0.0


# ============================================================================
# F. Latlon grid tests
# ============================================================================


class TestClimateVariabilityLatlon:
    """Tests for latlon grid dispatch (EERIE-style)."""

    @pytest.fixture
    def latlon_config(self, tmp_path):
        """Config with latlon grid type and CMOR data source."""
        from feather.config import FeatherConfig, ModelConfig
        cfg = FeatherConfig(
            model_catalogs={},
            models=["TestModel"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={"influence_radius": _TEST_INFLUENCE_RADIUS},
            output_dir=str(tmp_path / "output"),
        )
        cfg.model_configs = {
            "TestModel": ModelConfig(
                name="TestModel",
                institution="TEST",
                experiment="hist",
                variant="r1i1p1f1",
                grids={"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                color="#ff0000",
            ),
        }
        cfg.data_source = {"type": "cmor"}
        return cfg

    @pytest.fixture
    def latlon_model_loader(self, synth_obs_multi):
        """Model loader returning latlon data (CMOR-style API)."""
        class LatlonModelLoader:
            def __init__(self, ds):
                self._ds = ds
            def load_var(self, model, variable, **kwargs):
                return self._ds["t2m"]
        return LatlonModelLoader(synth_obs_multi)

    def test_latlon_grid_dispatch(
        self, latlon_model_loader, mock_obs_loader_multi,
        latlon_config,
    ):
        """EERIE-style latlon config works."""
        diag = ClimateVariability(
            latlon_model_loader, mock_obs_loader_multi, latlon_config,
            variables=["tas"],
        )
        results = diag.compute()
        assert "tas" in results
        assert "TestModel" in results["tas"]["models"]

    def test_latlon_std_regrid_has_latlon_dims(
        self, latlon_model_loader, mock_obs_loader_multi,
        latlon_config,
    ):
        """Regridded STD has lat/lon dims."""
        diag = ClimateVariability(
            latlon_model_loader, mock_obs_loader_multi, latlon_config,
            variables=["tas"],
        )
        results = diag.compute()
        std_regrid = results["tas"]["models"]["TestModel"]["std_regrid"]
        assert "lat" in std_regrid.dims
        assert "lon" in std_regrid.dims


# ============================================================================
# G. Registry test
# ============================================================================


class TestClimateVariabilityRegistry:
    """Test diagnostic registration."""

    _saved_registry = None

    @pytest.fixture(autouse=True)
    def _isolate_registry(self):
        """Save and restore the diagnostic registry."""
        from feather.diag.registry import _REGISTRY
        self.__class__._saved_registry = dict(_REGISTRY)
        yield
        _REGISTRY.clear()
        _REGISTRY.update(self.__class__._saved_registry)

    def test_registered(self):
        """get_diagnostic('climate_variability') returns the class."""
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("climate_variability")
        assert cls is ClimateVariability

    def test_in_list(self):
        """climate_variability appears in list_diagnostics()."""
        from feather.diag.registry import list_diagnostics
        entries = list_diagnostics()
        names = [e["name"] for e in entries]
        assert "climate_variability" in names

    def test_class_attributes(self):
        """Class-level attributes are set correctly."""
        assert ClimateVariability.name == "climate_variability"
        assert ClimateVariability.domain == "sfc"
        assert ClimateVariability.group == "evaluation"
        assert "tas" in ClimateVariability.variables
        assert len(ClimateVariability.variables) == 18


# ============================================================================
# H. Colorbar range tests
# ============================================================================


class TestColorbarRanges:
    """Tests for _compute_colorbar_ranges."""

    def test_std_range_positive(self):
        """STD range should have positive values."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        shape = (len(lats), len(lons))

        rng = np.random.default_rng(42)
        obs_std = xr.DataArray(
            rng.uniform(1.0, 5.0, shape),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        model_results = {
            "model1": {
                "std_regrid": xr.DataArray(
                    rng.uniform(1.0, 5.0, shape),
                    dims=("lat", "lon"),
                    coords={"lat": lats, "lon": lons},
                ),
                "std_diff": xr.DataArray(
                    rng.uniform(-2.0, 2.0, shape),
                    dims=("lat", "lon"),
                    coords={"lat": lats, "lon": lons},
                ),
            },
        }

        cb = ClimateVariability._compute_colorbar_ranges(
            model_results, obs_std,
        )
        assert cb["std"]["vmin"] >= 0.0
        assert cb["std"]["vmax"] > cb["std"]["vmin"]
        assert cb["diff"]["bias_vmax"] > 0.0

    def test_includes_cmip6_in_ranges(self):
        """CMIP6 data extends the range computation."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        shape = (len(lats), len(lons))

        rng = np.random.default_rng(42)
        obs_std = xr.DataArray(
            rng.uniform(1.0, 3.0, shape),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        model_results = {
            "model1": {
                "std_regrid": xr.DataArray(
                    rng.uniform(1.0, 3.0, shape),
                    dims=("lat", "lon"),
                    coords={"lat": lats, "lon": lons},
                ),
                "std_diff": xr.DataArray(
                    rng.uniform(-1.0, 1.0, shape),
                    dims=("lat", "lon"),
                    coords={"lat": lats, "lon": lons},
                ),
            },
        }
        # CMIP6 with larger range
        cmip6_data = {
            "std_regrid": xr.DataArray(
                rng.uniform(1.0, 10.0, shape),
                dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
            "std_diff": xr.DataArray(
                rng.uniform(-5.0, 5.0, shape),
                dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        }

        cb = ClimateVariability._compute_colorbar_ranges(
            model_results, obs_std,
            cmip6_data=cmip6_data,
        )
        # Range should be wider due to CMIP6
        assert cb["std"]["vmax"] > 3.0
        assert cb["diff"]["bias_vmax"] > 1.0
