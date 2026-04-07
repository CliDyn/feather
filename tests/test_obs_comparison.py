"""Tests for the obs_comparison diagnostic (ERA5 vs Berkeley Earth)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.obs_comparison import ObsComparisonDiag, _finite_concat
from feather.util.temporal import linear_trend

# ── Synthetic data helpers ────────────────────────────────────────────


def _make_tas_latlon(lats, lons, ntimes=36, base_temp=288.0,
                     trend_kperdecade=0.2, seed=0):
    """Monthly tas on a regular lat/lon grid with a known linear trend.

    T(t, lat, lon) = base_temp - 40*|lat/90| + seasonal + trend*t
    """
    rng = np.random.default_rng(seed)
    time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                         calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = base_temp - 40 * np.abs(lat_grid / 90.0)

    # Seasonal cycle
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(ntimes) - 3) / 12)

    # Linear trend in K/month (from K/decade)
    trend_per_month = trend_kperdecade / (10 * 12)
    trend = trend_per_month * np.arange(ntimes)

    data = (
        temp_base[np.newaxis, :, :]
        + seasonal[:, np.newaxis, np.newaxis]
        + trend[:, np.newaxis, np.newaxis]
        + rng.normal(0, 0.05, (ntimes, len(lats), len(lons)))
    )
    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


def _make_berkeley_degc(lats, lons, ntimes=36, base_temp=14.85,
                        trend_kperdecade=0.18, seed=1):
    """Monthly tas in degC on Berkeley Earth grid (latitude/longitude dims,
    lons in -180..180 convention)."""
    rng = np.random.default_rng(seed)
    time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                         calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = base_temp - 40 * np.abs(lat_grid / 90.0)

    seasonal = 4 * np.sin(2 * np.pi * (np.arange(ntimes) - 3) / 12)
    trend_per_month = trend_kperdecade / (10 * 12)
    trend = trend_per_month * np.arange(ntimes)

    data = (
        temp_base[np.newaxis, :, :]
        + seasonal[:, np.newaxis, np.newaxis]
        + trend[:, np.newaxis, np.newaxis]
        + rng.normal(0, 0.05, (ntimes, len(lats), len(lons)))
    )
    # Berkeley Earth uses latitude/longitude dim names and -180..180 lons
    return xr.DataArray(
        data, dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lats, "longitude": lons},
    )


# Small test grids (5° resolution for speed)
_LATS = np.arange(-87.5, 90.0, 5.0)
_LONS_360 = np.arange(2.5, 360.0, 5.0)       # ERA5 convention (0..360)
_LONS_180 = np.arange(-177.5, 180.0, 5.0)    # Berkeley Earth (-180..180)

_N_SHORT = 35 * 12   # 35 years → 420 months (1980-2014)
_N_LONG = 45 * 12    # 45 years → 540 months (1980-2024)


@pytest.fixture
def era5_short():
    return _make_tas_latlon(_LATS, _LONS_360, ntimes=_N_SHORT,
                            trend_kperdecade=0.20, seed=0)


@pytest.fixture
def era5_long():
    return _make_tas_latlon(_LATS, _LONS_360, ntimes=_N_LONG,
                            trend_kperdecade=0.22, seed=2)


@pytest.fixture
def be_short_degc():
    return _make_berkeley_degc(_LATS, _LONS_180, ntimes=_N_SHORT,
                               trend_kperdecade=0.18, seed=3)


@pytest.fixture
def be_long_degc():
    return _make_berkeley_degc(_LATS, _LONS_180, ntimes=_N_LONG,
                               trend_kperdecade=0.19, seed=4)


@pytest.fixture
def obs_config(tmp_path):
    """Minimal FeatherConfig pointing to tmp_path for obs."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root=str(tmp_path),
        obs_datasets={
            "ERA5": {"path": str(tmp_path / "ERA5"), "variables": {"t2m": "t2m.nc"}},
            "BERKELEY_EARTH": {
                "path": str(tmp_path / "BE"),
                "variables": {"2t": "be.nc"},
            },
        },
        cmip6={"enabled": False},
        dask={},
        output_dir=str(tmp_path / "output"),
        nereus={"method": "nearest"},
    )


@pytest.fixture
def mock_obs_loader(era5_short, era5_long, be_short_degc, be_long_degc):
    """Mock ObsLoader that returns synthetic data for both periods."""
    loader = MagicMock()

    def _load_era5(dataset, variable, period=None):
        if period is None or period[1] == "2014":
            return era5_short
        return era5_long

    def _load_be(dataset, variable, period=None):
        if period is None or period[1] == "2014":
            return be_short_degc
        return be_long_degc

    def _load(dataset, variable, period=None):
        if dataset == "ERA5":
            return _load_era5(dataset, variable, period)
        if dataset == "BERKELEY_EARTH":
            return _load_be(dataset, variable, period)
        raise KeyError(f"Unknown dataset: {dataset}")

    loader.load.side_effect = _load
    loader.load_for_model_var.side_effect = lambda var, period=None: _load_era5(
        "ERA5", "t2m", period,
    )
    return loader


@pytest.fixture
def diag(obs_config, mock_obs_loader):
    """ObsComparisonDiag instance with mock loaders."""
    return ObsComparisonDiag(
        model_loader=MagicMock(),
        obs_loader=mock_obs_loader,
        config=obs_config,
    )


# ══════════════════════════════════════════════════════════════════════════════
# A. _finite_concat utility
# ══════════════════════════════════════════════════════════════════════════════


class TestFiniteConcat:
    def test_basic(self):
        a = np.array([1.0, 2.0, np.nan])
        b = np.array([3.0, np.inf])
        result = _finite_concat([a, b])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_empty_input(self):
        result = _finite_concat([])
        assert result[0] == 0.0

    def test_all_nan(self):
        result = _finite_concat([np.array([np.nan, np.nan])])
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════════════
# B. Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadERA5:
    def test_returns_data_array(self, diag, era5_short):
        result = diag._load_era5(("1980", "2014"))
        assert isinstance(result, xr.DataArray)

    def test_period_short(self, diag, era5_short):
        result = diag._load_era5(("1980", "2014"))
        assert result.shape == era5_short.shape

    def test_period_long(self, diag, era5_long):
        result = diag._load_era5(("1980", "2024"))
        assert result.shape == era5_long.shape


class TestLoadBerkeleyEarth:
    def test_renames_dims(self, diag, be_short_degc):
        result = diag._load_berkeley_earth(("1980", "2014"))
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims

    def test_lons_shifted_to_0_360(self, diag):
        result = diag._load_berkeley_earth(("1980", "2014"))
        assert float(result.lon.min()) >= 0.0
        assert float(result.lon.max()) <= 360.0

    def test_converts_degc_to_kelvin(self, diag):
        result = diag._load_berkeley_earth(("1980", "2014"))
        # At mid-latitudes, K values should be > 200 K
        assert float(result.mean()) > 200.0

    def test_short_vs_long_period(self, diag, be_short_degc, be_long_degc):
        short = diag._load_berkeley_earth(("1980", "2014"))
        long_ = diag._load_berkeley_earth(("1980", "2024"))
        assert long_.shape[0] > short.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
# C. Grid helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestCommonGrid:
    def test_shape(self):
        lats, lons = ObsComparisonDiag._common_grid()
        assert len(lats) == 180
        assert len(lons) == 360

    def test_lat_range(self):
        lats, _ = ObsComparisonDiag._common_grid()
        assert abs(float(lats[0]) - (-89.5)) < 0.01
        assert abs(float(lats[-1]) - 89.5) < 0.01

    def test_lon_range(self):
        _, lons = ObsComparisonDiag._common_grid()
        assert abs(float(lons[0]) - 0.5) < 0.01
        assert abs(float(lons[-1]) - 359.5) < 0.01


class TestInterpToCommon:
    def test_output_shape(self, era5_short):
        lats = np.arange(-87.5, 90.0, 5.0)
        lons = np.arange(2.5, 360.0, 5.0)
        result = ObsComparisonDiag._interp_to_common(era5_short, lats, lons)
        assert result.sizes["lat"] == len(lats)
        assert result.sizes["lon"] == len(lons)

    def test_output_dim_names(self, era5_short):
        lats = np.arange(-87.5, 90.0, 5.0)
        lons = np.arange(2.5, 360.0, 5.0)
        result = ObsComparisonDiag._interp_to_common(era5_short, lats, lons)
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_latitude_longitude_dims(self, be_short_degc):
        """Fields with latitude/longitude dim names are handled and renamed."""
        lats = np.arange(-87.5, 90.0, 5.0)
        lons_360 = np.arange(2.5, 360.0, 5.0)
        # Shift BE lons to 0..360 first (as the loader does)
        be_360 = be_short_degc.assign_coords(
            longitude=((be_short_degc.longitude + 360) % 360),
        ).sortby("longitude")
        result = ObsComparisonDiag._interp_to_common(be_360, lats, lons_360)
        assert "lat" in result.dims
        assert "lon" in result.dims


# ══════════════════════════════════════════════════════════════════════════════
# D. _compute_trend static method
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeTrend:
    def _make_da(self, ntimes, slope_kperdecade=0.2):
        """Simple 2D lat/lon field with prescribed linear trend."""
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([90.0, 180.0])
        time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                             calendar="standard")
        trend_per_month = slope_kperdecade / (10 * 12)
        trend = trend_per_month * np.arange(ntimes)
        data = (
            280.0
            + trend[:, np.newaxis, np.newaxis]
            * np.ones((ntimes, len(lats), len(lons)))
        )
        return xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )

    def test_annual_trend_sign(self):
        da = self._make_da(ntimes=24, slope_kperdecade=0.5)
        trend = ObsComparisonDiag._compute_trend(da, "annual")
        assert trend is not None
        assert float(trend.mean()) > 0

    def test_annual_trend_magnitude(self):
        """Annual trend should recover ~prescribed K/decade."""
        da = self._make_da(ntimes=120, slope_kperdecade=0.3)
        trend = ObsComparisonDiag._compute_trend(da, "annual")
        assert trend is not None
        np.testing.assert_allclose(float(trend.mean()), 0.3, atol=0.05)

    def test_djf_trend_returns_array(self):
        da = self._make_da(ntimes=120, slope_kperdecade=0.2)
        trend = ObsComparisonDiag._compute_trend(da, "DJF")
        assert trend is not None
        assert trend.dims == ("lat", "lon")

    def test_jja_trend_returns_array(self):
        da = self._make_da(ntimes=120, slope_kperdecade=0.2)
        trend = ObsComparisonDiag._compute_trend(da, "JJA")
        assert trend is not None

    def test_insufficient_data_returns_none(self):
        da = self._make_da(ntimes=1, slope_kperdecade=0.2)
        trend = ObsComparisonDiag._compute_trend(da, "annual")
        assert trend is None

    def test_trend_units_per_decade(self):
        """Trend is K/decade, so values should be ~10x the K/month rate."""
        slope_kperdecade = 0.4
        da = self._make_da(ntimes=120, slope_kperdecade=slope_kperdecade)
        trend = ObsComparisonDiag._compute_trend(da, "annual")
        np.testing.assert_allclose(
            float(trend.mean()), slope_kperdecade, atol=0.05,
        )


# ══════════════════════════════════════════════════════════════════════════════
# E. _seasonal_mean static method
# ══════════════════════════════════════════════════════════════════════════════


class TestSeasonalMean:
    def _make_seasonal_da(self, ntimes=120):
        lats = np.array([-30.0, 0.0, 30.0])
        lons = np.array([0.0, 90.0])
        time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                             calendar="standard")
        # Constant field for easy checking
        data = np.full((ntimes, len(lats), len(lons)), 280.0)
        return xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )

    def test_annual_mean_returns_2d(self):
        da = self._make_seasonal_da()
        result = ObsComparisonDiag._seasonal_mean(da, "annual")
        assert "time" not in result.dims
        assert result.dims == ("lat", "lon")

    def test_annual_mean_value(self):
        da = self._make_seasonal_da()
        result = ObsComparisonDiag._seasonal_mean(da, "annual")
        np.testing.assert_allclose(float(result.mean()), 280.0, atol=0.01)

    def test_djf_mean_returns_2d(self):
        da = self._make_seasonal_da()
        result = ObsComparisonDiag._seasonal_mean(da, "DJF")
        assert "time" not in result.dims

    def test_jja_mean_returns_2d(self):
        da = self._make_seasonal_da()
        result = ObsComparisonDiag._seasonal_mean(da, "JJA")
        assert "time" not in result.dims

    def test_seasonal_mean_value_constant_field(self):
        da = self._make_seasonal_da()
        djf = ObsComparisonDiag._seasonal_mean(da, "DJF")
        np.testing.assert_allclose(float(djf.mean()), 280.0, atol=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# F. _compute_all_trends
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeAllTrends:
    @pytest.fixture
    def small_das(self):
        """Small 3-lat × 2-lon lat/lon fields, short enough for fast tests."""
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([90.0, 270.0])
        ntimes = 48  # 4 years

        def _make(slope, seed):
            rng = np.random.default_rng(seed)
            time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                                 calendar="standard")
            trend = (slope / (10 * 12)) * np.arange(ntimes)
            data = (
                280.0
                + trend[:, np.newaxis, np.newaxis]
                + rng.normal(0, 0.01, (ntimes, len(lats), len(lons)))
            )
            return xr.DataArray(
                data, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            )

        return {
            "e_s": _make(0.2, 0), "e_l": _make(0.25, 1),
            "b_s": _make(0.18, 2), "b_l": _make(0.22, 3),
        }

    def test_returns_all_period_keys(self, diag, small_das):
        d = small_das
        result = diag._compute_all_trends(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        assert "annual" in result
        # DJF/JJA may be skipped if ntimes=48 gives <2 seasonal years

    def test_annual_has_required_keys(self, diag, small_das):
        d = small_das
        result = diag._compute_all_trends(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        annual = result["annual"]
        for key in ["era5_short", "era5_long", "be_short", "be_long",
                    "era5_period_diff", "be_period_diff",
                    "dataset_diff_short", "dataset_diff_long"]:
            assert key in annual, f"Missing key: {key}"

    def test_period_diff_is_long_minus_short(self, diag, small_das):
        d = small_das
        result = diag._compute_all_trends(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        annual = result["annual"]
        expected = annual["era5_long"] - annual["era5_short"]
        np.testing.assert_allclose(
            annual["era5_period_diff"].values, expected.values, atol=1e-10,
        )

    def test_dataset_diff_is_era5_minus_be(self, diag, small_das):
        d = small_das
        result = diag._compute_all_trends(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        annual = result["annual"]
        expected = annual["era5_short"] - annual["be_short"]
        np.testing.assert_allclose(
            annual["dataset_diff_short"].values, expected.values, atol=1e-10,
        )


# ══════════════════════════════════════════════════════════════════════════════
# G. _compute_clim_bias
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeClimBias:
    @pytest.fixture
    def small_das(self):
        lats = np.array([-30.0, 0.0, 30.0])
        lons = np.array([0.0, 90.0])
        ntimes = 36

        def _make(base, seed):
            rng = np.random.default_rng(seed)
            time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                                 calendar="standard")
            data = base + rng.normal(0, 0.01, (ntimes, len(lats), len(lons)))
            return xr.DataArray(
                data, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            )

        return {
            "e_s": _make(280.0, 0), "e_l": _make(280.5, 1),
            "b_s": _make(279.5, 2), "b_l": _make(280.0, 3),
        }

    def test_returns_all_period_keys(self, diag, small_das):
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        for k in ["annual", "DJF", "JJA"]:
            assert k in result

    def test_annual_keys(self, diag, small_das):
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        for key in ["era5_short", "era5_long", "be_short", "be_long",
                    "diff_short", "diff_long"]:
            assert key in result["annual"]

    def test_diff_short_is_era5_minus_be(self, diag, small_das):
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        expected = result["annual"]["era5_short"] - result["annual"]["be_short"]
        np.testing.assert_allclose(
            result["annual"]["diff_short"].values, expected.values, atol=1e-10,
        )

    def test_diff_long_is_era5_minus_be_long(self, diag, small_das):
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        expected = result["annual"]["era5_long"] - result["annual"]["be_long"]
        np.testing.assert_allclose(
            result["annual"]["diff_long"].values, expected.values, atol=1e-10,
        )

    def test_climatology_removes_time_dim(self, diag, small_das):
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        clim = result["annual"]["era5_short"]
        assert "time" not in clim.dims

    def test_era5_warmer_than_be_in_diff(self, diag, small_das):
        """ERA5 base=280 > BE base=279.5, so diff_short should be positive."""
        d = small_das
        result = diag._compute_clim_bias(d["e_s"], d["e_l"], d["b_s"], d["b_l"])
        assert float(result["annual"]["diff_short"].mean()) > 0


# ══════════════════════════════════════════════════════════════════════════════
# H. _compute_timeseries
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeTimeseries:
    @pytest.fixture
    def small_das_and_area(self):
        from feather.util.spatial import compute_latlon_areas
        lats = np.array([-30.0, 0.0, 30.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0])
        ntimes = 24
        time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                             calendar="standard")
        e = xr.DataArray(
            np.full((ntimes, len(lats), len(lons)), 285.0),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )
        b = xr.DataArray(
            np.full((ntimes, len(lats), len(lons)), 284.0),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )
        area = compute_latlon_areas(lats, lons)
        return e, b, area

    def test_returns_dict_with_era5_and_be(self, diag, small_das_and_area):
        e, b, area = small_das_and_area
        result = diag._compute_timeseries(e, b, area)
        assert "era5" in result
        assert "be" in result

    def test_era5_ts_has_time_or_year_dim(self, diag, small_das_and_area):
        e, b, area = small_das_and_area
        result = diag._compute_timeseries(e, b, area)
        # annual_mean may return 'time' (year-end timestamps) or 'year'
        assert "year" in result["era5"].dims or "time" in result["era5"].dims

    def test_era5_global_mean_value(self, diag, small_das_and_area):
        e, b, area = small_das_and_area
        result = diag._compute_timeseries(e, b, area)
        np.testing.assert_allclose(float(result["era5"].mean()), 285.0, atol=0.01)

    def test_be_global_mean_value(self, diag, small_das_and_area):
        e, b, area = small_das_and_area
        result = diag._compute_timeseries(e, b, area)
        np.testing.assert_allclose(float(result["be"].mean()), 284.0, atol=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# I. plot() methods (figure structure tests)
# ══════════════════════════════════════════════════════════════════════════════


def _make_small_trend_map(seed=0):
    rng = np.random.default_rng(seed)
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    data = rng.normal(0.2, 0.05, (len(lats), len(lons)))
    return xr.DataArray(data, dims=("lat", "lon"),
                        coords={"lat": lats, "lon": lons})


def _make_small_clim_map(base, seed=0):
    rng = np.random.default_rng(seed)
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    data = base + rng.normal(0, 0.5, (len(lats), len(lons)))
    return xr.DataArray(data, dims=("lat", "lon"),
                        coords={"lat": lats, "lon": lons})


@pytest.fixture
def synthetic_results():
    """Pre-built compute() output for fast plot tests."""
    from feather.util.spatial import compute_latlon_areas
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    area = compute_latlon_areas(lats, lons)

    def _t(seed):
        return _make_small_trend_map(seed)

    def _c(base, seed):
        return _make_small_clim_map(base, seed)

    trends = {}
    clim = {}
    for period_key in ["annual", "DJF", "JJA"]:
        s = hash(period_key) % 100
        trends[period_key] = {
            "era5_short": _t(s), "era5_long": _t(s + 1),
            "be_short": _t(s + 2), "be_long": _t(s + 3),
            "era5_period_diff": _t(s + 4), "be_period_diff": _t(s + 5),
            "dataset_diff_short": _t(s + 6), "dataset_diff_long": _t(s + 7),
        }
        clim[period_key] = {
            "era5_short": _c(285.0, s), "era5_long": _c(285.5, s + 1),
            "be_short": _c(284.5, s + 2), "be_long": _c(285.0, s + 3),
            "diff_short": _c(0.5, s + 4), "diff_long": _c(0.5, s + 5),
        }

    # Global mean timeseries
    years = np.arange(1980, 2025)
    era5_ts = xr.DataArray(
        280.0 + 0.02 * np.arange(len(years)),
        dims=("year",), coords={"year": years},
    )
    be_ts = xr.DataArray(
        279.8 + 0.019 * np.arange(len(years)),
        dims=("year",), coords={"year": years},
    )

    return {
        "trends": trends,
        "clim": clim,
        "timeseries": {"era5": era5_ts, "be": be_ts},
        "common_lats": lats,
        "common_lons": lons,
        "area": area,
    }


class TestPlotTrendMaps:
    def test_returns_3_figures(self, diag, synthetic_results):
        figs = diag._plot_trend_maps(synthetic_results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids_annual_djf_jja(self, diag, synthetic_results):
        figs = diag._plot_trend_maps(synthetic_results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert "tas_annual_obs_trends" in ids
        assert "tas_djf_obs_trends" in ids
        assert "tas_jja_obs_trends" in ids
        plt.close("all")

    def test_metadata_has_required_fields(self, diag, synthetic_results):
        figs = diag._plot_trend_maps(synthetic_results)
        _, meta = figs[0]
        for field in ["figure_id", "title", "description", "plot_type"]:
            assert field in meta
        plt.close("all")

    def test_plot_type_is_combined_trend_map(self, diag, synthetic_results):
        figs = diag._plot_trend_maps(synthetic_results)
        _, meta = figs[0]
        assert meta["plot_type"] == "combined_trend_map"
        plt.close("all")


class TestPlotTrendDiffs:
    def test_returns_3_figures(self, diag, synthetic_results):
        figs = diag._plot_trend_diffs(synthetic_results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, synthetic_results):
        figs = diag._plot_trend_diffs(synthetic_results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert "tas_annual_obs_trend_diffs" in ids
        assert "tas_djf_obs_trend_diffs" in ids
        assert "tas_jja_obs_trend_diffs" in ids
        plt.close("all")


class TestPlotClimBias:
    def test_returns_3_figures(self, diag, synthetic_results):
        figs = diag._plot_clim_bias(synthetic_results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, synthetic_results):
        figs = diag._plot_clim_bias(synthetic_results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert "tas_annual_obs_clim_bias" in ids
        assert "tas_djf_obs_clim_bias" in ids
        assert "tas_jja_obs_clim_bias" in ids
        plt.close("all")

    def test_plot_type(self, diag, synthetic_results):
        figs = diag._plot_clim_bias(synthetic_results)
        _, meta = figs[0]
        assert meta["plot_type"] == "combined_bias_map"
        plt.close("all")


class TestPlotTimeseries:
    def test_returns_1_figure(self, diag, synthetic_results):
        figs = diag._plot_timeseries(synthetic_results)
        assert len(figs) == 1
        plt.close("all")

    def test_figure_id(self, diag, synthetic_results):
        figs = diag._plot_timeseries(synthetic_results)
        _, meta = figs[0]
        assert meta["figure_id"] == "tas_obs_timeseries"
        plt.close("all")

    def test_plot_type(self, diag, synthetic_results):
        figs = diag._plot_timeseries(synthetic_results)
        _, meta = figs[0]
        assert meta["plot_type"] == "timeseries"
        plt.close("all")


class TestPlotAll:
    def test_total_10_figures(self, diag, synthetic_results):
        figs = diag.plot(synthetic_results)
        assert len(figs) == 10
        plt.close("all")

    def test_all_figure_ids_unique(self, diag, synthetic_results):
        figs = diag.plot(synthetic_results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert len(ids) == len(set(ids))
        plt.close("all")

    def test_all_figs_are_figures(self, diag, synthetic_results):
        figs = diag.plot(synthetic_results)
        for fig, _ in figs:
            assert isinstance(fig, plt.Figure)
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# J. run() method — skip_existing logic
# ══════════════════════════════════════════════════════════════════════════════


class TestRunSkipExisting:
    def test_skip_when_all_figures_exist(self, diag, tmp_path):
        """run() with skip_existing=True skips if all PNGs+JSONs are present."""
        out_dir = diag.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        all_ids = ["tas_obs_timeseries"]
        for pk in ["annual", "djf", "jja"]:
            all_ids += [
                f"tas_{pk}_obs_trends",
                f"tas_{pk}_obs_trend_diffs",
                f"tas_{pk}_obs_clim_bias",
            ]
        # Create dummy PNG+JSON pairs
        for fid in all_ids:
            (out_dir / f"{fid}.png").touch()
            (out_dir / f"{fid}.json").write_text("{}")

        with patch.object(diag, "compute") as mock_compute:
            diag.run(skip_existing=True)
            mock_compute.assert_not_called()

    def test_no_skip_when_figures_missing(self, diag, synthetic_results):
        """run() calls compute() if any figure is absent."""
        with patch.object(diag, "compute", return_value=synthetic_results), \
             patch.object(diag, "plot", return_value=[]), \
             patch.object(diag, "_save", return_value=("a.png", "a.json")):
            diag.run(skip_existing=True)
            diag.compute.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# K. Registration
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_diagnostic_is_registered(self):
        from feather.diag.registry import get_diagnostic
        import feather.diag  # noqa: F401 — triggers @register
        diag_cls = get_diagnostic("obs_comparison")
        assert diag_cls is ObsComparisonDiag

    def test_name_attribute(self):
        assert ObsComparisonDiag.name == "obs_comparison"

    def test_variables_attribute(self):
        assert "tas" in ObsComparisonDiag.variables

    def test_period_constants(self):
        assert ObsComparisonDiag.PERIOD_SHORT == ("1980", "2014")
        assert ObsComparisonDiag.PERIOD_LONG == ("1980", "2024")
