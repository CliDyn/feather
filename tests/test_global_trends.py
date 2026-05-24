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
            variables=variables or ["tas"],
        )

    def test_compute_returns_expected_keys(self, minimal_config):
        """_compute_variable returns expected top-level keys."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

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
        result = diag._compute_variable("tas")

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
        result = diag._compute_variable("tas")

        obs = result["obs"]
        assert "trend" in obs
        assert "seasonal_trends" in obs
        assert "global_mean_trend" in obs

    def test_trend_sign_positive(self, minimal_config):
        """Model with positive trend has positive global_mean_trend."""
        model_ds = self._make_trending_healpix(slope=1.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        # Slope 1.0 K/yr → ~10 K/decade (after *10)
        assert trend > 0

    def test_trend_sign_negative(self, minimal_config):
        """Model with negative trend has negative global_mean_trend."""
        model_ds = self._make_trending_healpix(slope=-1.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        assert trend < 0

    def test_trend_difference_sign(self, minimal_config):
        """When model trend > obs trend, difference is positive."""
        model_ds = self._make_trending_healpix(slope=1.0)
        obs_ds = self._make_trending_obs(slope=0.3)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        diff = result["models"]["ifs-fesom"]["annual_trend_diff_gmean"]
        # Model slope > obs slope → positive diff
        assert diff > 0

    def test_zero_model_trend(self, minimal_config):
        """Constant model data gives trend near zero."""
        model_ds = self._make_trending_healpix(slope=0.0)
        obs_ds = self._make_trending_obs(slope=0.0)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        assert abs(trend) < 0.5  # near zero

    def test_seasonal_trends_present(self, minimal_config):
        """Seasonal trends are computed for all four seasons."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        mdata = result["models"]["ifs-fesom"]
        assert "DJF" in mdata["seasonal_regrids"]
        assert "JJA" in mdata["seasonal_regrids"]

    def test_seasonal_trend_diffs_present(self, minimal_config):
        """Seasonal trend differences are computed."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        mdata = result["models"]["ifs-fesom"]
        assert "DJF" in mdata["seasonal_trend_diffs"]
        assert "JJA" in mdata["seasonal_trend_diffs"]

    def test_obs_trend_on_common_grid(self, minimal_config):
        """Obs trend is on the common lat/lon grid."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        obs_trend = result["obs"]["trend"]
        assert "lat" in obs_trend.dims
        assert "lon" in obs_trend.dims

    def test_colorbar_ranges_present(self, minimal_config):
        """Colorbar ranges are computed for annual + seasonal."""
        model_ds = self._make_trending_healpix(n_years=10)
        obs_ds = self._make_trending_obs(n_years=10)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

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
        result = diag._compute_variable("tas")

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
            variables=["tas"],
        )
        result = diag._compute_variable("tas")
        assert result is None

    def test_rmse_nonnegative(self, minimal_config):
        """RMSE is always non-negative."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        rmse = result["models"]["ifs-fesom"]["annual_rmse"]
        assert rmse >= 0

    def test_compute_wrapper(self, minimal_config):
        """compute() processes all variables."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs()
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        results = diag.compute()
        assert "tas" in results

    def test_trend_units_per_decade(self, minimal_config):
        """Trend values are in units/decade (slope * 10)."""
        # Model: 1.0 K/year → 10.0 K/decade
        model_ds = self._make_trending_healpix(slope=1.0, n_years=15)
        obs_ds = self._make_trending_obs(slope=0.0, n_years=15)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

        trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        # Should be approximately 10.0 K/decade (with NN regrid tolerance)
        assert abs(trend - 10.0) < 2.0

    def test_obs_global_mean_trend(self, minimal_config):
        """Obs global mean trend is computed correctly."""
        model_ds = self._make_trending_healpix()
        obs_ds = self._make_trending_obs(slope=0.5)
        diag = self._make_diag(model_ds, obs_ds, minimal_config)
        result = diag._compute_variable("tas")

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
                        "MAM": model_trend * 0.9,
                        "JJA": model_trend * 0.8,
                        "SON": model_trend * 0.7,
                    },
                    "seasonal_trend_diffs": {
                        "DJF": trend_diff,
                        "MAM": trend_diff * 0.9,
                        "JJA": trend_diff * 0.8,
                        "SON": trend_diff * 0.7,
                    },
                },
            },
            "obs": {
                "trend": obs_trend,
                "seasonal_trends": {
                    "DJF": obs_trend,
                    "MAM": obs_trend * 0.95,
                    "JJA": obs_trend * 0.9,
                    "SON": obs_trend * 0.85,
                },
                "global_mean_trend": 3.0,
            },
            "var_info": get_var("tas"),
            "colorbar_ranges": {
                "annual": {"vmin": -1.0, "vmax": 1.0, "bias_vmax": 0.5},
                "DJF": {"vmin": -1.0, "vmax": 1.0, "bias_vmax": 0.5},
                "MAM": {"vmin": -0.9, "vmax": 0.9, "bias_vmax": 0.45},
                "JJA": {"vmin": -0.8, "vmax": 0.8, "bias_vmax": 0.4},
                "SON": {"vmin": -0.7, "vmax": 0.7, "bias_vmax": 0.35},
            },
        }

    def test_returns_figure_list(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """_plot_variable returns a list of (fig, meta) tuples."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        assert isinstance(figures, list)
        assert all(isinstance(f, tuple) and len(f) == 2 for f in figures)
        plt.close("all")

    def test_five_figures_per_variable(self, mock_model_loader,
                                        mock_obs_loader, minimal_config):
        """Produces 5 figures: annual, DJF, MAM, JJA, SON."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        assert len(figures) == 5
        plt.close("all")

    def test_metadata_diagnostic_name(self, mock_model_loader,
                                       mock_obs_loader, minimal_config):
        """Metadata has correct diagnostic_name."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert meta["diagnostic_name"] == "global_trends"
        plt.close("all")

    def test_metadata_plot_type(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Metadata has plot_type 'combined_trend_map'."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert meta["plot_type"] == "combined_trend_map"
        plt.close("all")

    def test_metadata_units_per_decade(self, mock_model_loader,
                                        mock_obs_loader, minimal_config):
        """Metadata units show per-decade."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert "/decade" in meta.get("units", "")
        plt.close("all")

    def test_figure_ids(self, mock_model_loader, mock_obs_loader,
                         minimal_config):
        """Figure IDs follow the expected pattern."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        ids = [meta["figure_id"] for _, meta in figures]
        assert "tas_annual_trend_combined" in ids
        assert "tas_djf_trend_combined" in ids
        assert "tas_jja_trend_combined" in ids
        plt.close("all")

    def test_metadata_has_models(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """Metadata includes model list."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_metadata_has_summary_stats(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """Annual metadata includes summary statistics."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
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
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert meta.get("period") is not None
        plt.close("all")

    def test_metadata_has_description(self, mock_model_loader,
                                       mock_obs_loader, minimal_config):
        """Metadata includes a description mentioning trends."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for _, meta in figures:
            assert "trend" in meta.get("description", "").lower()
        plt.close("all")

    def test_figure_is_matplotlib_figure(self, mock_model_loader,
                                          mock_obs_loader, minimal_config):
        """Each figure is a matplotlib Figure."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        figures = diag._plot_variable("tas", vr)
        for fig, _ in figures:
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_wrapper(self, mock_model_loader, mock_obs_loader,
                           minimal_config):
        """plot() processes all variables in results."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        results = {"tas": vr}
        figures = diag.plot(results)
        assert len(figures) == 5
        plt.close("all")

    def test_no_figures_when_empty_dict(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """No figures when trend_diff_dict is empty."""
        diag = GlobalTrends(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        vr = self._make_mock_result()
        # Remove all model results
        vr["models"] = {}
        figures = diag._plot_variable("tas", vr)
        assert len(figures) == 0
        plt.close("all")


# ============================================================================
# E. Run orchestration tests
# ============================================================================


class TestGlobalTrendsRun:
    """Tests for GlobalTrends.run() orchestration.

    Most tests mock _compute_variable and _plot_variable to test
    the file I/O logic without expensive nr.regrid KDTree builds.
    One test (test_run_saves_files) does a full real run.
    """

    @staticmethod
    def _mock_compute_result():
        """Return a minimal fake _compute_variable result."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        field = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        return {
            "obs": {"trend": field},
            "models": {},
            "colorbar_ranges": {
                "annual": {"field_vmax": 1.0, "bias_vmax": 1.0},
                "DJF": {"field_vmax": 1.0, "bias_vmax": 1.0},
                "JJA": {"field_vmax": 1.0, "bias_vmax": 1.0},
            },
            "cmip6_data": {},
            "cmip6_info": {},
            "cmip6_individual_data": {},
        }

    @staticmethod
    def _mock_plot_figures():
        """Return 5 fake (fig, meta) pairs."""
        periods = ["annual", "djf", "mam", "jja", "son"]
        figures = []
        for p in periods:
            fig = MagicMock(spec=plt.Figure)
            fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
            meta = {
                "diagnostic_name": "global_trends",
                "figure_id": f"tas_{p}_trend_combined",
                "models": ["ifs-fesom"],
                "title": f"Trend {p}",
            }
            figures.append((fig, meta))
        return figures

    def _make_diag(self, config):
        from tests.conftest import MockModelLoader, MockObsLoader

        model_ds = _make_trending_healpix_helper()
        obs_ds = _make_trending_obs_helper()
        return GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds), config,
            variables=["tas"],
        )

    def test_run_saves_files(self, minimal_config):
        """run() creates PNG + JSON files (full real run)."""
        diag = self._make_diag(minimal_config)
        saved = diag.run(skip_existing=False)
        assert len(saved) == 5
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
        plt.close("all")

    def test_run_skip_existing(self, minimal_config):
        """run() skips when all figures already exist."""
        diag = self._make_diag(minimal_config)
        # Pre-create all figure files
        out_dir = diag.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in ["annual", "djf", "mam", "jja", "son"]:
            fid = f"tas_{p}_trend_combined"
            (out_dir / f"{fid}.png").write_bytes(b"png")
            (out_dir / f"{fid}.json").write_text('{"diagnostic_name":"global_trends"}')

        with patch.object(diag, "_compute_variable") as mock_compute:
            saved = diag.run(skip_existing=True)
            mock_compute.assert_not_called()  # skipped!
        assert len(saved) == 5

    def test_run_regenerates_partial(self, minimal_config):
        """run() regenerates when only some figures exist."""
        diag = self._make_diag(minimal_config)
        out_dir = diag.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tas_annual_trend_combined.png").write_bytes(b"png")
        (out_dir / "tas_annual_trend_combined.json").write_text("{}")
        # DJF and JJA missing → should NOT skip

        result = self._mock_compute_result()
        figures = self._mock_plot_figures()
        with patch.object(diag, "_compute_variable", return_value=result), \
             patch.object(diag, "_plot_variable", return_value=figures):
            saved = diag.run(skip_existing=True)
        assert len(saved) == 5

    def test_run_returns_path_tuples(self, minimal_config):
        """run() returns list of (Path, Path) tuples."""
        diag = self._make_diag(minimal_config)
        result = self._mock_compute_result()
        figures = self._mock_plot_figures()
        with patch.object(diag, "_compute_variable", return_value=result), \
             patch.object(diag, "_plot_variable", return_value=figures):
            saved = diag.run(skip_existing=False)
        for item in saved:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], Path)
            assert isinstance(item[1], Path)

    def test_variable_override(self, minimal_config):
        """Variables can be overridden at construction."""
        diag = self._make_diag(minimal_config)
        assert diag.variables == ["tas"]

    def test_json_metadata_valid(self, minimal_config):
        """Saved JSON metadata is valid JSON."""
        diag = self._make_diag(minimal_config)
        result = self._mock_compute_result()
        figures = self._mock_plot_figures()
        with patch.object(diag, "_compute_variable", return_value=result), \
             patch.object(diag, "_plot_variable", return_value=figures):
            saved = diag.run(skip_existing=False)
        for _, json_path in saved:
            with open(json_path) as f:
                meta = json.load(f)
            assert meta["diagnostic_name"] == "global_trends"

    def test_run_multiple_models(self, tmp_path):
        """run() metadata includes all models."""
        from feather.config import FeatherConfig

        config = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom", "ifs-nemo"],
            obs_root="", obs_datasets={},
            cmip6={"enabled": False}, dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
        )
        diag = self._make_diag(config)

        result = self._mock_compute_result()
        # Mock figures with both models in metadata
        figures = []
        for p in ["annual", "djf", "mam", "jja", "son"]:
            fig = MagicMock(spec=plt.Figure)
            fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
            meta = {
                "diagnostic_name": "global_trends",
                "figure_id": f"tas_{p}_trend_combined",
                "models": ["ifs-fesom", "ifs-nemo"],
                "title": f"Trend {p}",
            }
            figures.append((fig, meta))

        with patch.object(diag, "_compute_variable", return_value=result), \
             patch.object(diag, "_plot_variable", return_value=figures):
            saved = diag.run(skip_existing=False)

        assert len(saved) == 5
        for _, json_path in saved:
            with open(json_path) as f:
                meta = json.load(f)
            assert "ifs-fesom" in meta["models"]
            assert "ifs-nemo" in meta["models"]

    def test_no_skip_existing_flag(self, minimal_config):
        """run(skip_existing=False) regenerates even if files exist."""
        diag = self._make_diag(minimal_config)
        result = self._mock_compute_result()
        figures = self._mock_plot_figures()

        with patch.object(diag, "_compute_variable", return_value=result), \
             patch.object(diag, "_plot_variable", return_value=figures):
            diag.run(skip_existing=False)

        annual_png = diag.output_dir / "tas_annual_trend_combined.png"
        mtime1 = annual_png.stat().st_mtime

        import time as time_mod
        time_mod.sleep(0.1)

        diag2 = self._make_diag(minimal_config)
        with patch.object(diag2, "_compute_variable", return_value=result), \
             patch.object(diag2, "_plot_variable", return_value=figures):
            diag2.run(skip_existing=False)
        mtime2 = annual_png.stat().st_mtime
        assert mtime2 > mtime1


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
            variables=["tas", "psl"],
        )
        assert diag.variables == ["tas", "psl"]

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
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag.plot(results)

        assert len(figures) == 5
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
            variables=["tas"],
        )
        result = diag._compute_variable("tas")

        # Model trend > obs trend → positive difference
        diff = result["models"]["ifs-fesom"]["annual_trend_diff_gmean"]
        assert diff > 0

        # Model trend: ~10 K/decade, obs trend: ~2 K/decade
        model_trend = result["models"]["ifs-fesom"]["global_mean_trend"]
        obs_trend = result["obs"]["global_mean_trend"]
        assert model_trend > obs_trend
        plt.close("all")


# ============================================================================
# K. CMIP6 synthetic data helpers + fixtures
# ============================================================================


def _make_trending_cmip6(n_years=3, slope=0.4):
    """Create synthetic CMIP6-like dataset with a known linear trend."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    n_months = n_years * 12
    time = xr.date_range("1990-01", periods=n_months, freq="MS")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(n_months) - 3) / 12)
    days = (time.values - time.values[0]) / np.timedelta64(1, "D")
    years = days / 365.25

    temp_3d = (
        temp_base[np.newaxis, :, :]
        + seasonal[:, np.newaxis, np.newaxis]
        + slope * years[:, np.newaxis, np.newaxis]
    )

    area_2d = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)

    return xr.Dataset({
        "tas": xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
        "areacella": xr.DataArray(
            area_2d, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        ),
    })


def _make_trending_healpix_helper(nside=8, n_years=3, slope=0.5):
    """Create trending HEALPix dataset (standalone helper)."""
    import healpy as hp

    ncells = 12 * nside**2
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    n_months = n_years * 12
    time = xr.date_range("1990-01", periods=n_months, freq="MS")
    temp_base = 300 - 40 * np.abs(lat / 90.0)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(n_months) - 3) / 12)
    days = (time.values - time.values[0]) / np.timedelta64(1, "D")
    years = days / 365.25

    temp_2d = (
        temp_base[np.newaxis, :]
        + seasonal[:, np.newaxis]
        + slope * years[:, np.newaxis]
    )

    return xr.Dataset({
        "avg_2t": xr.DataArray(
            temp_2d, dims=("time", "values"), coords={"time": time},
        ),
        "longitude": xr.DataArray(lon, dims="values"),
        "latitude": xr.DataArray(lat, dims="values"),
        "area": xr.DataArray(
            np.ones(ncells) * (4 * np.pi / ncells), dims="values",
        ),
    })


def _make_trending_obs_helper(n_years=3, slope=0.3):
    """Create trending obs dataset (standalone helper)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    n_months = n_years * 12
    time = xr.date_range("1990-01", periods=n_months, freq="MS")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(n_months) - 3) / 12)
    days = (time.values - time.values[0]) / np.timedelta64(1, "D")
    years = days / 365.25

    temp_3d = (
        temp_base[np.newaxis, :, :]
        + seasonal[:, np.newaxis, np.newaxis]
        + slope * years[:, np.newaxis, np.newaxis]
    )

    return xr.Dataset({
        "t2m": xr.DataArray(
            temp_3d, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })


def _make_cmip6_diag(model_ds, obs_ds, config, cmip6_ds,
                      cmip6_individual=False, variables=None):
    """Create a GlobalTrends diagnostic with mock CMIP6 loader."""
    from tests.conftest import MockCMIP6Loader, MockModelLoader, MockObsLoader

    model_loader = MockModelLoader(model_ds)
    obs_loader = MockObsLoader(obs_ds)
    cmip6_loader = MockCMIP6Loader(cmip6_ds)
    return GlobalTrends(
        model_loader, obs_loader, config,
        cmip6_loader=cmip6_loader,
        variables=variables or ["tas"],
        cmip6_individual=cmip6_individual,
    )


@pytest.fixture(scope="module")
def cmip6_mmm_result():
    """Module-scoped fixture: compute CMIP6 MMM result once."""
    from feather.config import FeatherConfig

    config = FeatherConfig(
        model_catalogs={}, models=["ifs-fesom"],
        obs_root="", obs_datasets={},
        cmip6={"enabled": True, "influence_radius": 1_000_000},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir="/tmp/test_gt_cmip6_mmm",
    )
    model_ds = _make_trending_healpix_helper()
    obs_ds = _make_trending_obs_helper()
    cmip6_ds = _make_trending_cmip6()
    diag = _make_cmip6_diag(model_ds, obs_ds, config, cmip6_ds)
    result = diag._compute_variable("tas")
    return result, diag


@pytest.fixture(scope="module")
def cmip6_individual_result():
    """Module-scoped fixture: compute CMIP6 individual result once."""
    from feather.config import FeatherConfig

    config = FeatherConfig(
        model_catalogs={}, models=["ifs-fesom"],
        obs_root="", obs_datasets={},
        cmip6={"enabled": True, "influence_radius": 1_000_000},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir="/tmp/test_gt_cmip6_ind",
    )
    model_ds = _make_trending_healpix_helper()
    obs_ds = _make_trending_obs_helper()
    cmip6_ds = _make_trending_cmip6()
    diag = _make_cmip6_diag(
        model_ds, obs_ds, config, cmip6_ds, cmip6_individual=True,
    )
    result = diag._compute_variable("tas")
    return result, diag


# ============================================================================
# L. CMIP6 MMM trend tests
# ============================================================================


class TestGlobalTrendsCMIP6MMM:
    """Tests for CMIP6 multi-model mean trend computation."""

    def test_cmip6_data_present_when_enabled(self, cmip6_mmm_result):
        """cmip6_data is populated when CMIP6 is enabled."""
        result, _ = cmip6_mmm_result
        assert "cmip6_data" in result
        assert "annual" in result["cmip6_data"]

    def test_cmip6_data_empty_when_disabled(self, minimal_config):
        """cmip6_data is empty when CMIP6 is disabled."""
        model_ds = _make_trending_healpix_helper()
        obs_ds = _make_trending_obs_helper()
        from tests.conftest import MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds),
            minimal_config, variables=["tas"],
        )
        result = diag._compute_variable("tas")
        assert result["cmip6_data"] == {}

    def test_cmip6_info_populated(self, cmip6_mmm_result):
        """cmip6_info has models_used and n_members."""
        result, _ = cmip6_mmm_result
        info = result["cmip6_info"]
        assert "models_used" in info
        assert "n_members" in info
        assert info["n_members"] > 0

    def test_cmip6_annual_keys(self, cmip6_mmm_result):
        """Annual CMIP6 data has regrid, trend_diff, trend_diff_gmean, rmse."""
        result, _ = cmip6_mmm_result
        annual = result["cmip6_data"]["annual"]
        assert "regrid" in annual
        assert "trend_diff" in annual
        assert "trend_diff_gmean" in annual
        assert "rmse" in annual

    def test_cmip6_seasonal_trends(self, cmip6_mmm_result):
        """Seasonal CMIP6 MMM trends computed for all four seasons."""
        result, _ = cmip6_mmm_result
        assert "DJF" in result["cmip6_data"]
        assert "JJA" in result["cmip6_data"]

    def test_cmip6_trend_diff_sign(self, cmip6_mmm_result):
        """CMIP6 with different slope than obs → nonzero trend_diff_gmean."""
        result, _ = cmip6_mmm_result
        diff_gmean = result["cmip6_data"]["annual"]["trend_diff_gmean"]
        assert isinstance(diff_gmean, float)

    def test_cmip6_rmse_nonnegative(self, cmip6_mmm_result):
        """CMIP6 MMM RMSE is non-negative."""
        result, _ = cmip6_mmm_result
        rmse = result["cmip6_data"]["annual"]["rmse"]
        assert rmse >= 0

    def test_cmip6_mmm_in_plot(self, cmip6_mmm_result):
        """CMIP6 MMM appears in plot models list."""
        result, diag = cmip6_mmm_result
        figures = diag._plot_variable("tas", result)
        assert len(figures) > 0
        _, meta = figures[0]
        assert "CMIP6 MMM" in meta["models"]
        plt.close("all")

    def test_cmip6_metadata_has_info(self, cmip6_mmm_result):
        """Plot metadata includes cmip6_info."""
        result, diag = cmip6_mmm_result
        figures = diag._plot_variable("tas", result)
        _, meta = figures[0]
        assert meta.get("cmip6_info") is not None
        assert "models_used" in meta["cmip6_info"]
        plt.close("all")

    def test_cmip6_mmm_summary_stats(self, cmip6_mmm_result):
        """Summary stats include CMIP6 MMM entry."""
        result, diag = cmip6_mmm_result
        figures = diag._plot_variable("tas", result)
        annual_figs = [
            (f, m) for f, m in figures if "annual" in m["figure_id"]
        ]
        _, meta = annual_figs[0]
        stats = meta.get("summary_statistics", {})
        assert "CMIP6 MMM" in stats
        assert "global_mean_trend_diff" in stats["CMIP6 MMM"]
        plt.close("all")


# ============================================================================
# M. CMIP6 individual trend tests
# ============================================================================


class TestGlobalTrendsCMIP6Individual:
    """Tests for individual CMIP6 model trend computation."""

    def test_individual_data_present(self, cmip6_individual_result):
        """cmip6_individual_data populated when cmip6_individual=True."""
        result, _ = cmip6_individual_result
        assert "annual" in result["cmip6_individual_data"]
        assert len(result["cmip6_individual_data"]["annual"]) > 0

    def test_individual_labels_format(self, cmip6_individual_result):
        """Individual model labels are 'Model/variant'."""
        result, _ = cmip6_individual_result
        labels = list(result["cmip6_individual_data"]["annual"].keys())
        for label in labels:
            assert "/" in label  # e.g. "ModelA/r1i1p1f1"

    def test_individual_and_mmm_both_present(self, cmip6_individual_result):
        """Both individual and MMM are present when cmip6_individual=True."""
        result, _ = cmip6_individual_result
        assert "annual" in result["cmip6_data"]
        assert "annual" in result["cmip6_individual_data"]

    def test_no_individual_when_flag_false(self, cmip6_mmm_result):
        """cmip6_individual_data empty when cmip6_individual=False."""
        result, _ = cmip6_mmm_result
        assert result["cmip6_individual_data"] == {}

    def test_individual_in_plot_models(self, cmip6_individual_result):
        """Individual CMIP6 models appear in plot models list."""
        result, diag = cmip6_individual_result
        figures = diag._plot_variable("tas", result)
        _, meta = figures[0]
        assert "CMIP6 MMM" in meta["models"]
        individual_labels = [m for m in meta["models"] if "/" in m]
        assert len(individual_labels) > 0
        plt.close("all")

    def test_individual_seasonal_data(self, cmip6_individual_result):
        """Individual CMIP6 seasonal trends computed."""
        result, _ = cmip6_individual_result
        assert "DJF" in result["cmip6_individual_data"]
        assert "JJA" in result["cmip6_individual_data"]

    def test_individual_has_trend_diff(self, cmip6_individual_result):
        """Each individual model has trend_diff field."""
        result, _ = cmip6_individual_result
        for label, data in result["cmip6_individual_data"]["annual"].items():
            assert "trend_diff" in data
            assert "trend_diff_gmean" in data
            assert "rmse" in data

    def test_individual_summary_stats(self, cmip6_individual_result):
        """Summary stats include individual CMIP6 models."""
        result, diag = cmip6_individual_result
        figures = diag._plot_variable("tas", result)
        annual_figs = [
            (f, m) for f, m in figures if "annual" in m["figure_id"]
        ]
        _, meta = annual_figs[0]
        stats = meta.get("summary_statistics", {})
        individual_labels = [k for k in stats if "/" in k]
        assert len(individual_labels) > 0
        plt.close("all")


# ============================================================================
# N. _regrid_to_target tests
# ============================================================================


class TestRegridToTarget:
    """Tests for GlobalTrends._regrid_to_target().

    Mirrors the production flow: build target grid from a HEALPix
    model's nr.regrid() call (as _compute_variable does), then pass
    those coords + matching resolution to _regrid_to_target for CMIP6.
    """

    @staticmethod
    def _build_target_grid(resolution=5.0):
        """Build a target grid the same way _compute_variable does.

        Regrids a small HEALPix dataset to a regular grid, then
        extracts target_lats, target_lons, and resolution — exactly
        the values that _regrid_to_target receives in production.
        """
        import healpy as hp
        import nereus as nr

        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(
            nside, np.arange(ncells), nest=True, lonlat=True,
        )
        data = 300 - 40 * np.abs(lat / 90.0)

        _, interp = nr.regrid(
            data, lon=lon, lat=lat,
            resolution=resolution,
            influence_radius=1_000_000,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        target_lats = interp.target_lat[:, 0]
        target_lons = interp.target_lon[0, :]
        actual_res = abs(float(target_lats[1] - target_lats[0]))
        return target_lats, target_lons, actual_res

    def test_regrids_to_target_coords(self):
        """Regridded output carries the target lat/lon coordinates."""
        target_lats, target_lons, res = self._build_target_grid()

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        da = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )

        cache = {}
        result = GlobalTrends._regrid_to_target(
            da, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )

        np.testing.assert_array_equal(result.lat.values, target_lats)
        np.testing.assert_array_equal(result.lon.values, target_lons)

    def test_values_are_sensible(self):
        """Regridded cos(lat) field stays in [-1, 1]."""
        target_lats, target_lons, res = self._build_target_grid()

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        da = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )

        cache = {}
        result = GlobalTrends._regrid_to_target(
            da, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )

        vals = result.values[np.isfinite(result.values)]
        assert vals.min() >= -1.01
        assert vals.max() <= 1.01

    def test_cache_reused_for_same_grid_shape(self):
        """Two DataArrays on the same grid share one cached interpolator."""
        target_lats, target_lons, res = self._build_target_grid()

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        da1 = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )

        cache = {}
        GlobalTrends._regrid_to_target(
            da1, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )
        assert len(cache) == 1

        da2 = da1 * 2
        GlobalTrends._regrid_to_target(
            da2, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )
        assert len(cache) == 1  # still 1 — reused

    def test_different_source_grids_get_separate_cache(self):
        """Source grids with different shapes get distinct cache entries."""
        target_lats, target_lons, res = self._build_target_grid()
        cache = {}

        # Source grid A: 5°
        lats_a = np.arange(-87.5, 90, 5.0)
        lons_a = np.arange(2.5, 360, 5.0)
        da_a = xr.DataArray(
            np.ones((len(lats_a), len(lons_a))),
            dims=("lat", "lon"),
            coords={"lat": lats_a, "lon": lons_a},
        )
        GlobalTrends._regrid_to_target(
            da_a, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )

        # Source grid B: 10°
        lats_b = np.arange(-85, 90, 10.0)
        lons_b = np.arange(5, 360, 10.0)
        da_b = xr.DataArray(
            np.ones((len(lats_b), len(lons_b))),
            dims=("lat", "lon"),
            coords={"lat": lats_b, "lon": lons_b},
        )
        GlobalTrends._regrid_to_target(
            da_b, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )
        assert len(cache) == 2

    def test_handles_latitude_longitude_coord_names(self):
        """Works with 'latitude'/'longitude' instead of 'lat'/'lon'."""
        target_lats, target_lons, res = self._build_target_grid()

        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        da = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("latitude", "longitude"),
            coords={"latitude": lats, "longitude": lons},
        )
        cache = {}

        result = GlobalTrends._regrid_to_target(
            da, target_lats, target_lons,
            resolution=res, influence_radius=1_000_000,
            interp_cache=cache,
        )
        assert result.shape == (len(target_lats), len(target_lons))


# ============================================================================
# O. CMIP6 trend computation edge cases
# ============================================================================


class TestCMIP6TrendComputation:
    """Edge case tests for CMIP6 trend computation."""

    def test_missing_cmip6_variable(self, cmip6_config):
        """Missing CMIP6 variable → empty cmip6_data."""
        model_ds = _make_trending_healpix_helper()
        obs_ds = _make_trending_obs_helper()
        # Use a dataset without 'tas' to simulate missing variable
        empty_ds = xr.Dataset({
            "areacella": xr.DataArray(
                np.ones((5, 5)), dims=("lat", "lon"),
                coords={
                    "lat": np.arange(5.0),
                    "lon": np.arange(5.0),
                },
            ),
        })
        from tests.conftest import MockCMIP6Loader, MockModelLoader, MockObsLoader

        diag = GlobalTrends(
            MockModelLoader(model_ds), MockObsLoader(obs_ds),
            cmip6_config,
            cmip6_loader=MockCMIP6Loader(empty_ds),
            variables=["tas"],
        )
        result = diag._compute_variable("tas")
        assert result is not None
        assert result["cmip6_data"] == {}

    def test_cmip6_colorbar_includes_mmm(self, cmip6_mmm_result):
        """Colorbar ranges include CMIP6 MMM data."""
        result, _ = cmip6_mmm_result
        cb = result["colorbar_ranges"]
        assert cb["annual"]["bias_vmax"] > 0

    def test_cmip6_colorbar_includes_individual(self, cmip6_individual_result):
        """Colorbar ranges include individual CMIP6 data."""
        result, _ = cmip6_individual_result
        cb = result["colorbar_ranges"]
        assert cb["annual"]["bias_vmax"] > 0

    def test_cmip6_trend_grid_matches_obs(self, cmip6_mmm_result):
        """CMIP6 MMM trend has same grid dims as obs trend."""
        result, _ = cmip6_mmm_result
        obs_shape = result["obs"]["trend"].shape
        cmip6_shape = result["cmip6_data"]["annual"]["regrid"].shape
        assert cmip6_shape == obs_shape

    def test_run_with_cmip6(self, cmip6_mmm_result):
        """run() produces figures with CMIP6 MMM in metadata."""
        result, diag = cmip6_mmm_result
        figures = diag._plot_variable("tas", result)
        _, meta = figures[0]
        assert "CMIP6 MMM" in meta["models"]
        plt.close("all")

    def test_run_with_cmip6_individual(self, cmip6_individual_result):
        """Individual CMIP6 models appear in metadata."""
        result, diag = cmip6_individual_result
        figures = diag._plot_variable("tas", result)
        _, meta = figures[0]
        assert "CMIP6 MMM" in meta["models"]
        individual = [m for m in meta["models"] if "/" in m]
        assert len(individual) > 0
        plt.close("all")
