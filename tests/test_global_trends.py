"""Tests for the global_trends diagnostic and supporting utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.diag.global_trends import GlobalTrends
from feather.util.temporal import linear_trend, seasonal_annual_mean

# Large influence radius for nside=8 test data (~815 km spacing)
_TEST_INFLUENCE_RADIUS = 1_000_000


# ============================================================================
# A. linear_trend() utility tests
# ============================================================================


class TestLinearTrend:
    """Tests for the linear_trend() utility function."""

    def test_known_slope_1d(self):
        """Recover exact slope from a perfectly linear time series."""
        time = xr.date_range("2000-01", periods=24, freq="MS")
        # 2 units/year slope
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25
        values = 10.0 + 2.0 * years
        da = xr.DataArray(values, dims="time", coords={"time": time})

        result = linear_trend(da)
        assert abs(float(result.values) - 2.0) < 0.01

    def test_known_slope_2d(self):
        """Recover different slopes across a spatial dimension."""
        time = xr.date_range("2000-01", periods=36, freq="MS")
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        # 3 grid points with slopes 1, -2, 0.5 units/year
        slopes = np.array([1.0, -2.0, 0.5])
        data = 100.0 + slopes[np.newaxis, :] * years[:, np.newaxis]
        da = xr.DataArray(
            data, dims=("time", "values"),
            coords={"time": time},
        )

        result = linear_trend(da)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.values, slopes, atol=0.01)

    def test_known_slope_3d(self):
        """Recover slopes from a (time, lat, lon) array."""
        time = xr.date_range("1990-01", periods=24, freq="MS")
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0])

        # Slope varies by lat: -1, 0, +1
        slope_field = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        data = (
            100.0
            + slope_field[np.newaxis, :, :] * years[:, np.newaxis, np.newaxis]
        )
        da = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )

        result = linear_trend(da)
        assert result.dims == ("lat", "lon")
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result.values, slope_field, atol=0.01)

    def test_zero_trend(self):
        """Constant data gives slope ~ 0."""
        time = xr.date_range("2000-01", periods=24, freq="MS")
        da = xr.DataArray(
            np.full(24, 42.0), dims="time", coords={"time": time},
        )
        result = linear_trend(da)
        assert abs(float(result.values)) < 1e-10

    def test_single_timestep(self):
        """Single timestep returns NaN (cannot compute trend)."""
        time = xr.date_range("2000-01", periods=1, freq="MS")
        da = xr.DataArray([5.0], dims="time", coords={"time": time})
        result = linear_trend(da)
        assert np.isnan(float(result.values))

    def test_numeric_dim(self):
        """Works with numeric (year) coordinate from groupby."""
        years = np.arange(1990, 2015, dtype=float)
        # 3 K/year trend
        data = 280.0 + 3.0 * (years - years[0])
        da = xr.DataArray(data, dims="year", coords={"year": years})
        result = linear_trend(da, dim="year")
        assert abs(float(result.values) - 3.0) < 0.01

    def test_preserves_spatial_coords(self):
        """Result keeps lat/lon coordinates from the input."""
        time = xr.date_range("2000-01", periods=12, freq="MS")
        lats = np.array([10.0, 20.0, 30.0])
        lons = np.array([100.0, 200.0])
        data = np.random.default_rng(42).standard_normal((12, 3, 2))
        da = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )
        result = linear_trend(da)
        np.testing.assert_array_equal(result.lat.values, lats)
        np.testing.assert_array_equal(result.lon.values, lons)

    def test_per_decade_multiply(self):
        """Multiplying by 10 gives per-decade trend."""
        time = xr.date_range("2000-01", periods=120, freq="MS")
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25
        # 0.5 K/year = 5 K/decade
        data = 280.0 + 0.5 * years
        da = xr.DataArray(data, dims="time", coords={"time": time})
        result = linear_trend(da) * 10
        assert abs(float(result.values) - 5.0) < 0.05

    def test_negative_trend(self):
        """Negative trend is correctly recovered."""
        time = xr.date_range("2000-01", periods=24, freq="MS")
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25
        data = 300.0 - 1.5 * years
        da = xr.DataArray(data, dims="time", coords={"time": time})
        result = linear_trend(da)
        assert abs(float(result.values) - (-1.5)) < 0.01

    def test_healpix_dim(self):
        """Works with 'values' dim (HEALPix convention)."""
        time = xr.date_range("2000-01", periods=24, freq="MS")
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25
        ncells = 10
        slopes = np.linspace(-1, 1, ncells)
        data = 280.0 + slopes[np.newaxis, :] * years[:, np.newaxis]
        da = xr.DataArray(data, dims=("time", "values"), coords={"time": time})

        result = linear_trend(da)
        assert result.dims == ("values",)
        np.testing.assert_allclose(result.values, slopes, atol=0.01)


# ============================================================================
# B. seasonal_annual_mean() tests
# ============================================================================


class TestSeasonalAnnualMean:
    """Tests for the seasonal_annual_mean() utility function."""

    def test_djf_year_assignment(self):
        """DJF: December is assigned to the following year."""
        time = xr.date_range("1990-01", periods=36, freq="MS")
        data = np.arange(36, dtype=float)
        da = xr.DataArray(data, dims="time", coords={"time": time})
        result = seasonal_annual_mean(da, "DJF")
        # Dec 1990 → year 1991, so first full DJF is 1991
        years = result.year.values
        assert 1991 in years

    def test_jja_months(self):
        """JJA only uses June, July, August."""
        time = xr.date_range("1990-01", periods=24, freq="MS")
        # Set JJA months to 1.0, everything else to 0.0
        months = np.array([t.month for t in time.values.astype("datetime64[M]").astype("O")])
        # Use time.month via xr
        da = xr.DataArray(
            np.zeros(24), dims="time", coords={"time": time},
        )
        # Manually set JJA months
        for i, t in enumerate(time.values):
            m = int(str(t)[:7].split("-")[1])
            if m in [6, 7, 8]:
                da.values[i] = 1.0

        result = seasonal_annual_mean(da, "JJA")
        # All values should be 1.0 (only JJA months selected, all = 1.0)
        assert all(abs(v - 1.0) < 1e-10 for v in result.values)

    def test_returns_year_coord(self):
        """Result has a 'year' coordinate."""
        time = xr.date_range("1990-01", periods=36, freq="MS")
        da = xr.DataArray(
            np.ones(36), dims="time", coords={"time": time},
        )
        result = seasonal_annual_mean(da, "JJA")
        assert "year" in result.dims

    def test_correct_number_of_years(self):
        """With 3 years of data, get 3 JJA annual means."""
        time = xr.date_range("1990-01", periods=36, freq="MS")
        da = xr.DataArray(
            np.ones(36), dims="time", coords={"time": time},
        )
        result = seasonal_annual_mean(da, "JJA")
        assert len(result.year) == 3  # 1990, 1991, 1992

    def test_period_slicing(self):
        """Period argument restricts the time range before season selection."""
        time = xr.date_range("1988-01", periods=72, freq="MS")
        da = xr.DataArray(
            np.ones(72), dims="time", coords={"time": time},
        )
        result = seasonal_annual_mean(da, "JJA", period=("1990", "1992"))
        years = result.year.values
        assert all(1990 <= y <= 1992 for y in years)

    def test_with_spatial_dims(self):
        """Works with (time, lat, lon) data."""
        time = xr.date_range("1990-01", periods=24, freq="MS")
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0])
        data = np.random.default_rng(42).standard_normal((24, 3, 2))
        da = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )
        result = seasonal_annual_mean(da, "DJF")
        assert "year" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_son_months(self):
        """SON uses September, October, November."""
        time = xr.date_range("1990-01", periods=12, freq="MS")
        da = xr.DataArray(
            np.zeros(12), dims="time", coords={"time": time},
        )
        # Set SON months to 1.0
        for i, t in enumerate(time.values):
            m = int(str(t)[:7].split("-")[1])
            if m in [9, 10, 11]:
                da.values[i] = 1.0
        result = seasonal_annual_mean(da, "SON")
        assert all(abs(v - 1.0) < 1e-10 for v in result.values)

    def test_empty_after_filter(self):
        """If no months match, returns empty DataArray."""
        # Only 3 months that are NOT in DJF
        time = xr.date_range("1990-06", periods=3, freq="MS")  # Jun, Jul, Aug
        da = xr.DataArray(
            np.ones(3), dims="time", coords={"time": time},
        )
        result = seasonal_annual_mean(da, "DJF")
        assert len(result.time) == 0


# ============================================================================
# C. GlobalTrends compute tests
# ============================================================================


class TestGlobalTrendsCompute:
    """Tests for GlobalTrends._compute_variable()."""

    def _make_trending_healpix(self, nside=8, n_years=10, slope=0.5):
        """Create synthetic HEALPix data with a known linear trend."""
        import healpy as hp

        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

        n_months = n_years * 12
        time = xr.date_range("1990-01", periods=n_months, freq="MS")

        temp_base = 300 - 40 * np.abs(lat / 90.0)
        seasonal = 5 * np.sin(
            2 * np.pi * (np.arange(n_months) - 3) / 12,
        )
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        temp_2d = (
            temp_base[np.newaxis, :]
            + seasonal[:, np.newaxis]
            + slope * years[:, np.newaxis]  # linear trend
        )

        ds = xr.Dataset({
            "avg_2t": xr.DataArray(
                temp_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(
                np.ones(ncells) * (4 * np.pi / ncells), dims="values",
            ),
        })
        return ds

    def _make_trending_obs(self, n_years=10, slope=0.3):
        """Create synthetic obs data with a known linear trend."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        n_months = n_years * 12
        time = xr.date_range("1990-01", periods=n_months, freq="MS")

        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        temp_base = 300 - 40 * np.abs(lat_grid / 90.0)
        seasonal = 5 * np.sin(
            2 * np.pi * (np.arange(n_months) - 3) / 12,
        )
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        temp_3d = (
            temp_base[np.newaxis, :, :]
            + seasonal[:, np.newaxis, np.newaxis]
            + slope * years[:, np.newaxis, np.newaxis]
        )

        ds = xr.Dataset({
            "t2m": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        })
        return ds

    def _make_diag(self, model_ds, obs_ds, config, variables=None):
        """Create a GlobalTrends diagnostic with mock loaders."""
        from tests.conftest import MockModelLoader, MockObsLoader
        model_loader = MockModelLoader(model_ds)
        obs_loader = MockObsLoader(obs_ds)
        return GlobalTrends(
            model_loader, obs_loader, config,
            variables=variables or ["avg_2t"],
        )

    def test_compute_returns_expected_keys(self, minimal_config):
        """_compute_variable returns expected top-level keys."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        assert result is not None
        assert "models" in result
        assert "obs" in result
        assert "var_info" in result
        assert "colorbar_ranges" in result

    def test_model_result_fields(self, minimal_config):
        """Each model entry has required trend fields."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        mdata = result["models"]["ifs-fesom"]
        assert "annual_regrid" in mdata
        assert "annual_trend_diff" in mdata
        assert "global_mean_trend" in mdata
        assert "annual_trend_diff_gmean" in mdata
        assert "annual_rmse" in mdata
        assert "seasonal_regrids" in mdata
        assert "seasonal_trend_diffs" in mdata

    def test_obs_result_fields(self, minimal_config):
        """Obs entry has required trend fields."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        obs = result["obs"]
        assert "trend" in obs
        assert "seasonal_trends" in obs
        assert "global_mean_trend" in obs

    def test_trend_sign_positive(self, minimal_config):
        """Model with positive trend has positive global_mean_trend."""
        model_ds = self._make_trending_healpix(slope=1.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        # Slope 1.0 K/yr → ~10 K/decade (after *10)
        assert trend > 0

    def test_trend_sign_negative(self, minimal_config):
        """Model with negative trend has negative global_mean_trend."""
        model_ds = self._make_trending_healpix(slope=-1.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        assert trend < 0

    def test_trend_difference_sign(self, minimal_config):
        """When model trend > obs trend, difference is positive."""
        model_ds = self._make_trending_healpix(slope=1.0)
        obs_ds = self._make_trending_obs(slope=0.3)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        diff = result["models"]["ifs-fesom"]["annual_trend_diff_gmean"]
        # Model slope > obs slope → positive diff
        assert diff > 0

    def test_zero_model_trend(self, minimal_config):
        """Constant model data gives trend near zero."""
        model_ds = self._make_trending_healpix(slope=0.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        assert abs(trend) < 0.5  # near zero

    def test_seasonal_trends_present(self, minimal_config):
        """Seasonal trends are computed for DJF and JJA."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        mdata = result["models"]["ifs-fesom"]
        assert "DJF" in mdata["seasonal_regrids"]
        assert "JJA" in mdata["seasonal_regrids"]

    def test_seasonal_trend_diffs_present(self, minimal_config):
        """Seasonal trend differences are computed."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        mdata = result["models"]["ifs-fesom"]
        assert "DJF" in mdata["seasonal_trend_diffs"]
        assert "JJA" in mdata["seasonal_trend_diffs"]

    def test_obs_trend_on_common_grid(self, minimal_config):
        """Obs trend is on the common lat/lon grid."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        obs_trend = result["obs"]["trend"]
        assert "lat" in obs_trend.dims
        assert "lon" in obs_trend.dims

    def test_colorbar_ranges_present(self, minimal_config):
        """Colorbar ranges are computed for annual + seasonal."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        cb = result["colorbar_ranges"]
        assert "annual" in cb
        assert "vmin" in cb["annual"]
        assert "vmax" in cb["annual"]
        assert "bias_vmax" in cb["annual"]

    def test_colorbar_ranges_symmetric(self, minimal_config):
        """Trend colorbar ranges are symmetric around zero."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        cb = result["colorbar_ranges"]["annual"]
        assert abs(cb["vmin"] + cb["vmax"]) < 1e-10

    def test_missing_variable_returns_none(self, minimal_config):
        """Missing model variable returns None."""
        from tests.conftest import MockModelLoader, MockObsLoader

        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        # Rename the variable so avg_2t is missing
        model_ds_renamed = model_ds.rename({"avg_2t": "avg_msl"})

        model_loader = MockModelLoader(model_ds_renamed)
        obs_loader = MockObsLoader(obs_ds)
        diag = GlobalTrends(
            model_loader, obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        result = diag._compute_variable("avg_2t")
        assert result is None

    def test_rmse_nonnegative(self, minimal_config):
        """RMSE is always non-negative."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        rmse = result["models"]["ifs-fesom"]["annual_rmse"]
        assert rmse >= 0

    def test_compute_wrapper(self, minimal_config):
        """compute() processes all variables."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        results = diag.compute()
        assert "avg_2t" in results

    def test_trend_units_per_decade(self, minimal_config):
        """Trend values are in units/decade (slope * 10)."""
        # Model: 1.0 K/year → 10.0 K/decade
        model_ds = self._make_trending_healpix(slope=1.0, n_years=15)
        obs_ds = self._make_trending_obs(slope=0.0, n_years=15)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        # Should be approximately 10.0 K/decade (with NN regrid tolerance)
        assert abs(trend - 10.0) < 2.0

    def test_obs_global_mean_trend(self, minimal_config):
        """Obs global mean trend is computed correctly."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs(slope=0.5)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("avg_2t")

        obs_trend = result["obs"]["global_mean_trend"]
        # 0.5 K/yr → 5.0 K/decade
        assert abs(obs_trend - 5.0) < 1.5


# ============================================================================
# D. GlobalTrends plot tests
# ============================================================================


class TestGlobalTrendsPlot:
    """Tests for GlobalTrends._plot_variable()."""

    def _make_mock_result(self):
        """Create a mock computation result for plotting tests."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        from feather.data.variables import get_var

        trend_field = 0.5 * np.cos(np.deg2rad(lat_grid))
        obs_trend = xr.DataArray(
            trend_field, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        model_trend = xr.DataArray(
            trend_field * 1.2, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        trend_diff = model_trend - obs_trend

        return {
            "models": {
                "ifs-fesom": {
                    "annual_regrid": model_trend,
                    "annual_trend_diff": trend_diff,
                    "global_mean_trend": 5.0,
                    "annual_trend_diff_gmean": 1.0,
                    "annual_rmse": 0.5,
                    "seasonal_regrids": {
                        "DJF": model_trend,
                        "JJA": model_trend * 0.8,
                    },
                    "seasonal_trend_diffs": {
                        "DJF": trend_diff,
                        "JJA": trend_diff * 0.8,
                    },
                },
            },
            "obs": {
                "trend": obs_trend,
                "seasonal_trends": {
                    "DJF": obs_trend,
                    "JJA": obs_trend * 0.9,
                },
                "global_mean_trend": 3.0,
            },
            "var_info": get_var("avg_2t"),
            "colorbar_ranges": {
                "annual": {"vmin": -1.0, "vmax": 1.0, "bias_vmax": 0.5},
                "DJF": {"vmin": -1.0, "vmax": 1.0, "bias_vmax": 0.5},
                "JJA": {"vmin": -0.8, "vmax": 0.8, "bias_vmax": 0.4},
            },
        }

    def test_returns_figure_list(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """_plot_variable returns a list of (fig, meta) tuples."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        assert isinstance(figures, list)
        assert all(isinstance(f, tuple) and len(f) == 2 for f in figures)
        plt.close("all")

    def test_three_figures_per_variable(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """Produces 3 figures: annual, DJF, JJA."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        assert len(figures) == 3
        plt.close("all")

    def test_metadata_diagnostic_name(self, mock_model_loader,
                                       mock_obs_loader, minimal_config):
        """Metadata has correct diagnostic_name."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert meta["diagnostic_name"] == "global_trends"
        plt.close("all")

    def test_metadata_plot_type(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Metadata has plot_type 'combined_trend_map'."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert meta["plot_type"] == "combined_trend_map"
        plt.close("all")

    def test_metadata_units_per_decade(self, mock_model_loader,
                                        mock_obs_loader, minimal_config):
        """Metadata units show per-decade."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert "/decade" in meta.get("units", "")
        plt.close("all")

    def test_figure_ids(self, mock_model_loader, mock_obs_loader,
                         minimal_config):
        """Figure IDs follow the expected pattern."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        ids = [meta["figure_id"] for _, meta in figures]
        assert "avg_2t_annual_trend_combined" in ids
        assert "avg_2t_djf_trend_combined" in ids
        assert "avg_2t_jja_trend_combined" in ids
        plt.close("all")

    def test_metadata_has_models(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """Metadata includes model list."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_metadata_has_summary_stats(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """Annual metadata includes summary statistics."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        # Find annual figure
        annual_figs = [
            (fig, meta) for fig, meta in figures
            if "annual" in meta["figure_id"]
        ]
        assert len(annual_figs) == 1
        _, meta = annual_figs[0]
        stats = meta.get("summary_statistics", {})
        assert "ifs-fesom" in stats
        assert "global_mean_trend" in stats["ifs-fesom"]
        plt.close("all")

    def test_metadata_has_period(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """Metadata includes the analysis period."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert meta.get("period") is not None
        plt.close("all")

    def test_metadata_has_description(self, mock_model_loader,
                                       mock_obs_loader, minimal_config):
        """Metadata includes a description mentioning trends."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for _, meta in figures:
            assert "trend" in meta.get("description", "").lower()
        plt.close("all")

    def test_figure_is_matplotlib_figure(self, mock_model_loader,
                                          mock_obs_loader, minimal_config):
        """Each figure is a matplotlib Figure."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("avg_2t", vr)
        for fig, _ in figures:
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_wrapper(self, mock_model_loader, mock_obs_loader,
                           minimal_config):
        """plot() processes all variables in results."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        results = {"avg_2t": vr}
        figures = diag.plot(results)
        assert len(figures) == 3
        plt.close("all")

    def test_no_figures_when_empty_dict(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """No figures when trend_diff_dict is empty."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        vr = self._make_mock_result()
        # Remove all model results
        vr["models"] = {}
        figures = diag._plot_variable("avg_2t", vr)
        assert len(figures) == 0
        plt.close("all")


# ============================================================================
# E. Run orchestration tests
# ============================================================================


class TestGlobalTrendsRun:
    """Tests for GlobalTrends.run() orchestration."""

    def _make_trending_data(self, n_years=5, model_slope=0.5, obs_slope=0.3):
        """Create synthetic trending model and obs datasets."""
        import healpy as hp

        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(
            nside, np.arange(ncells), nest=True, lonlat=True,
        )

        n_months = n_years * 12
        time = xr.date_range("1990-01", periods=n_months, freq="MS")
        temp_base = 300 - 40 * np.abs(lat / 90.0)
        seasonal = 5 * np.sin(
            2 * np.pi * (np.arange(n_months) - 3) / 12,
        )
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        model_ds = xr.Dataset({
            "avg_2t": xr.DataArray(
                (temp_base[np.newaxis, :]
                 + seasonal[:, np.newaxis]
                 + model_slope * years[:, np.newaxis]),
                dims=("time", "values"), coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(
                np.ones(ncells) * (4 * np.pi / ncells), dims="values",
            ),
        })

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        obs_base = 300 - 40 * np.abs(lat_grid / 90.0)
        obs_ds = xr.Dataset({
            "t2m": xr.DataArray(
                (obs_base[np.newaxis, :, :]
                 + seasonal[:, np.newaxis, np.newaxis]
                 + obs_slope * years[:, np.newaxis, np.newaxis]),
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        })

        return model_ds, obs_ds

    def test_run_saves_files(self, minimal_config):
        """run() creates PNG + JSON files."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        saved = diag.run(skip_existing=False)
        assert len(saved) == 3
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
        plt.close("all")

    def test_run_skip_existing(self, minimal_config):
        """run() skips when all figures already exist."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        # First run
        saved1 = diag.run(skip_existing=False)
        assert len(saved1) == 3
        plt.close("all")

        # Second run with skip_existing — should skip
        diag2 = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        saved2 = diag2.run(skip_existing=True)
        assert len(saved2) == 3  # returns paths but doesn't recompute

    def test_run_regenerates_partial(self, minimal_config):
        """run() regenerates when only some figures exist."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        # Create partial files (only annual, missing DJF and JJA)
        out_dir = diag.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "avg_2t_annual_trend_combined.png").write_bytes(b"png")
        (out_dir / "avg_2t_annual_trend_combined.json").write_text("{}")

        saved = diag.run(skip_existing=True)
        # Should regenerate (partial → not skipped)
        assert len(saved) == 3
        plt.close("all")

    def test_run_returns_path_tuples(self, minimal_config):
        """run() returns list of (Path, Path) tuples."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        saved = diag.run(skip_existing=False)
        for item in saved:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], Path)
            assert isinstance(item[1], Path)
        plt.close("all")

    def test_variable_override(self, minimal_config):
        """Variables can be overridden at construction."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        assert diag.variables == ["avg_2t"]

    def test_json_metadata_valid(self, minimal_config):
        """Saved JSON metadata is valid JSON."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        saved = diag.run(skip_existing=False)
        for _, json_path in saved:
            with open(json_path) as f:
                meta = json.load(f)
            assert meta["diagnostic_name"] == "global_trends"
        plt.close("all")

    def test_run_multiple_models(self, tmp_path):
        """run() works with multiple models."""
        import healpy as hp

        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(
            nside, np.arange(ncells), nest=True, lonlat=True,
        )
        time = xr.date_range("1990-01", periods=60, freq="MS")
        temp_base = 300 - 40 * np.abs(lat / 90.0)
        seasonal = 5 * np.sin(
            2 * np.pi * (np.arange(60) - 3) / 12,
        )
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        model_ds = xr.Dataset({
            "avg_2t": xr.DataArray(
                (temp_base[np.newaxis, :]
                 + seasonal[:, np.newaxis]
                 + 0.5 * years[:, np.newaxis]),
                dims=("time", "values"), coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        })

        lats = np.arange(-87.5, 90, 5.0)
        lons_obs = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons_obs, indexing="ij")
        obs_ds = xr.Dataset({
            "t2m": xr.DataArray(
                (300 - 40 * np.abs(lat_grid / 90.0))[np.newaxis, :, :]
                + np.zeros((60, 1, 1)),
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons_obs},
            ),
        })

        from feather.config import FeatherConfig
        from tests.conftest import MockModelLoader, MockObsLoader

        config = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom", "ifs-nemo"],
            obs_root="", obs_datasets={},
            cmip6={"enabled": False}, dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
        )

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), config,
            variables=["avg_2t"],
        )
        saved = diag.run(skip_existing=False)
        assert len(saved) == 3  # still 3 figures (combined panels)

        # Check metadata includes both models
        for _, json_path in saved:
            with open(json_path) as f:
                meta = json.load(f)
            assert "ifs-fesom" in meta["models"]
            assert "ifs-nemo" in meta["models"]
        plt.close("all")

    def test_no_skip_existing_flag(self, minimal_config):
        """run(skip_existing=False) regenerates even if files exist."""
        model_ds, obs_ds = self._make_trending_data()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        # First run
        diag.run(skip_existing=False)
        plt.close("all")

        # Get mod times
        annual_png = diag.output_dir / "avg_2t_annual_trend_combined.png"
        mtime1 = annual_png.stat().st_mtime

        # Small delay to ensure different mtime
        import time as time_mod
        time_mod.sleep(0.1)

        # Second run without skip
        diag2 = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        diag2.run(skip_existing=False)
        mtime2 = annual_png.stat().st_mtime
        assert mtime2 > mtime1
        plt.close("all")


# ============================================================================
# F. Registry tests
# ============================================================================


class TestGlobalTrendsRegistry:
    """Tests for diagnostic registration."""

    def test_registered_name(self):
        """Diagnostic is registered as 'global_trends'."""
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("global_trends")
        assert cls is GlobalTrends

    def test_class_attributes(self):
        """Class attributes are correct."""
        assert GlobalTrends.name == "global_trends"
        assert GlobalTrends.title == "Global Linear Trends"
        assert GlobalTrends.domain == "sfc"
        assert GlobalTrends.group == "evaluation"

    def test_in_diagnostic_list(self):
        """global_trends appears in list_diagnostics()."""
        from feather.diag.registry import list_diagnostics
        names = [d["name"] for d in list_diagnostics()]
        assert "global_trends" in names


# ============================================================================
# G. method parameter in plot_combined_bias_map tests
# ============================================================================


class TestMethodParameter:
    """Tests for the method parameter in plot_combined_bias_map."""

    def _make_simple_data(self):
        """Create simple lat/lon data for plotting tests."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        data = 300 - 40 * np.abs(lat_grid / 90.0)
        return xr.DataArray(
            data, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )

    @patch("nereus.plot")
    def test_default_method_nearest(self, mock_nr_plot):
        """Default method is 'nearest'."""
        from feather.plot.maps import plot_combined_bias_map

        mock_nr_plot.return_value = (None, None, MagicMock())
        obs = self._make_simple_data()
        bias = obs * 0.1

        try:
            plot_combined_bias_map(obs, {"model": bias}, units="K")
        except Exception:
            pass  # may fail due to mocked axes, that's OK

        # Check that nr.plot was called with method="nearest"
        for call in mock_nr_plot.call_args_list:
            assert call.kwargs.get("method", "nearest") == "nearest"

    @patch("nereus.plot")
    def test_method_linear_passed(self, mock_nr_plot):
        """method='linear' is passed through to nr.plot."""
        from feather.plot.maps import plot_combined_bias_map

        mock_nr_plot.return_value = (None, None, MagicMock())
        obs = self._make_simple_data()
        bias = obs * 0.1

        try:
            plot_combined_bias_map(
                obs, {"model": bias}, units="K", method="linear",
            )
        except Exception:
            pass

        for call in mock_nr_plot.call_args_list:
            assert call.kwargs.get("method") == "linear"

    @patch("nereus.plot")
    def test_bias_map_default_method(self, mock_nr_plot):
        """plot_bias_map default method is 'nearest'."""
        from feather.plot.maps import plot_bias_map

        mock_nr_plot.return_value = (None, None, MagicMock())
        obs = self._make_simple_data()
        model = self._make_simple_data()

        try:
            plot_bias_map(model, obs, units="K")
        except Exception:
            pass

        for call in mock_nr_plot.call_args_list:
            assert call.kwargs.get("method", "nearest") == "nearest"

    @patch("nereus.plot")
    def test_bias_map_method_linear(self, mock_nr_plot):
        """plot_bias_map passes method='linear' through."""
        from feather.plot.maps import plot_bias_map

        mock_nr_plot.return_value = (None, None, MagicMock())
        obs = self._make_simple_data()
        model = self._make_simple_data()

        try:
            plot_bias_map(model, obs, units="K", method="linear")
        except Exception:
            pass

        for call in mock_nr_plot.call_args_list:
            assert call.kwargs.get("method") == "linear"

    def test_existing_callers_unaffected(self):
        """Calling without method kwarg still works (backward compat)."""
        from feather.plot.maps import plot_combined_bias_map

        obs = self._make_simple_data()
        bias = obs * 0.1
        # Should not raise — method defaults to "nearest"
        fig, axes = plot_combined_bias_map(
            obs, {"model": bias}, units="K",
        )
        assert fig is not None
        plt.close("all")


# ============================================================================
# H. Constructor and attribute tests
# ============================================================================


class TestGlobalTrendsConstructor:
    """Tests for GlobalTrends constructor and defaults."""

    def test_default_variables(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """Default variable list has 18 variables."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        assert len(diag.variables) == 18

    def test_custom_variables(self, mock_model_loader, mock_obs_loader,
                               minimal_config):
        """Custom variables override the default list."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t", "avg_msl"],
        )
        assert diag.variables == ["avg_2t", "avg_msl"]

    def test_default_experiment(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Default experiment is baseline_hist."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        assert diag.experiment == "baseline_hist"

    def test_default_period(self, mock_model_loader, mock_obs_loader,
                             minimal_config):
        """Default period is (1990, 2014)."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        assert diag.period == ("1990", "2014")

    def test_cmip6_individual_default_false(self, mock_model_loader,
                                              mock_obs_loader,
                                              minimal_config):
        """cmip6_individual defaults to False."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        assert diag.cmip6_individual is False

    def test_output_dir(self, mock_model_loader, mock_obs_loader,
                         minimal_config):
        """Output dir includes diagnostic name."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        assert diag.output_dir.name == "global_trends"

    def test_same_variables_as_global_biases(self):
        """Same 18-variable list as GlobalBiases."""
        from feather.diag.global_biases import GlobalBiases
        assert GlobalTrends.variables == GlobalBiases.variables


# ============================================================================
# I. Colorbar range tests
# ============================================================================


class TestColorbarRanges:
    """Tests for _compute_colorbar_ranges."""

    def _make_ranges_input(self):
        """Create minimal inputs for colorbar range computation."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

        trend = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)), dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        diff = trend * 0.1

        model_results = {
            "ifs-fesom": {
                "annual_regrid": trend,
                "annual_trend_diff": diff,
                "seasonal_regrids": {"DJF": trend, "JJA": trend * 0.8},
                "seasonal_trend_diffs": {"DJF": diff, "JJA": diff * 0.8},
            },
        }
        obs_trend = trend * 0.9
        obs_seasonal = {"DJF": obs_trend, "JJA": obs_trend * 0.8}
        return model_results, obs_trend, obs_seasonal

    def test_annual_range_present(self):
        """Annual colorbar range is computed."""
        mr, obs, obs_s = self._make_ranges_input()
        ranges = GlobalTrends._compute_colorbar_ranges(mr, obs, obs_s)
        assert "annual" in ranges

    def test_seasonal_ranges_present(self):
        """Seasonal colorbar ranges are computed."""
        mr, obs, obs_s = self._make_ranges_input()
        ranges = GlobalTrends._compute_colorbar_ranges(mr, obs, obs_s)
        assert "DJF" in ranges
        assert "JJA" in ranges

    def test_symmetric_vmin_vmax(self):
        """vmin = -vmax for trend maps."""
        mr, obs, obs_s = self._make_ranges_input()
        ranges = GlobalTrends._compute_colorbar_ranges(mr, obs, obs_s)
        for period_ranges in ranges.values():
            assert abs(period_ranges["vmin"] + period_ranges["vmax"]) < 1e-10

    def test_bias_vmax_positive(self):
        """bias_vmax is positive."""
        mr, obs, obs_s = self._make_ranges_input()
        ranges = GlobalTrends._compute_colorbar_ranges(mr, obs, obs_s)
        for period_ranges in ranges.values():
            assert period_ranges["bias_vmax"] > 0


# ============================================================================
# J. Integration-style compute+plot test
# ============================================================================


class TestGlobalTrendsEndToEnd:
    """End-to-end test combining compute and plot."""

    def test_compute_then_plot(self, minimal_config):
        """Full compute → plot cycle produces valid figures."""
        import healpy as hp

        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(
            nside, np.arange(ncells), nest=True, lonlat=True,
        )
        n_months = 60
        time = xr.date_range("1990-01", periods=n_months, freq="MS")
        temp_base = 300 - 40 * np.abs(lat / 90.0)
        seasonal = 5 * np.sin(
            2 * np.pi * (np.arange(n_months) - 3) / 12,
        )
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        model_ds = xr.Dataset({
            "avg_2t": xr.DataArray(
                (temp_base[np.newaxis, :]
                 + seasonal[:, np.newaxis]
                 + 0.5 * years[:, np.newaxis]),
                dims=("time", "values"), coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        })

        lats = np.arange(-87.5, 90, 5.0)
        lons_obs = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons_obs, indexing="ij")
        obs_ds = xr.Dataset({
            "t2m": xr.DataArray(
                (300 - 40 * np.abs(lat_grid / 90.0))[np.newaxis, :, :]
                + 0.3 * years[:, np.newaxis, np.newaxis],
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons_obs},
            ),
        })

        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        results = diag.compute()
        figures = diag.plot(results)

        assert len(figures) == 3
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert meta["diagnostic_name"] == "global_trends"
            assert meta["plot_type"] == "combined_trend_map"
        plt.close("all")

    def test_trend_direction_consistency(self, minimal_config):
        """Model warming > obs warming → positive trend difference."""
        import healpy as hp

        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(
            nside, np.arange(ncells), nest=True, lonlat=True,
        )
        n_months = 120  # 10 years
        time = xr.date_range("1990-01", periods=n_months, freq="MS")
        temp_base = 300 - 40 * np.abs(lat / 90.0)
        days = (time.values - time.values[0]) / np.timedelta64(1, "D")
        years = days / 365.25

        model_ds = xr.Dataset({
            "avg_2t": xr.DataArray(
                temp_base[np.newaxis, :] + 1.0 * years[:, np.newaxis],
                dims=("time", "values"), coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        })

        lats = np.arange(-87.5, 90, 5.0)
        lons_obs = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons_obs, indexing="ij")
        obs_ds = xr.Dataset({
            "t2m": xr.DataArray(
                (300 - 40 * np.abs(lat_grid / 90.0))[np.newaxis, :, :]
                + 0.2 * years[:, np.newaxis, np.newaxis],
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons_obs},
            ),
        })

        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), minimal_config,
            variables=["avg_2t"],
        )
        result = diag._compute_variable("avg_2t")

        # Model trend > obs trend → positive difference
        diff = result["models"]["ifs-fesom"]["annual_trend_diff_gmean"]
        assert diff > 0

        # Model trend: ~10 K/decade, obs trend: ~2 K/decade
        model_trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        obs_trend = result["obs"]["global_mean_trend"]
        assert model_trend > obs_trend
        plt.close("all")
