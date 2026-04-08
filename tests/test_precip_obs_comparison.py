"""Tests for the precip_obs_comparison diagnostic (ERA5 vs MSWEP)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.precip_obs_comparison import (
    PrecipObsComparisonDiag,
    _PR_MMDAY,
    _REL_BIAS_THRESHOLD,
    _finite_concat,
)

# ── Synthetic data helpers ─────────────────────────────────────────────────


def _make_pr_latlon(lats, lons, ntimes=36, base_pr_mmday=2.5,
                    trend=0.0, seed=0):
    """Monthly precipitation (kg/m²/s) on a regular lat/lon grid.

    Precipitation ~ base * exp(-|lat|/30) (tropical max) + small trend.
    """
    rng = np.random.default_rng(seed)
    time = xr.date_range("1980-01", periods=ntimes, freq="MS",
                         calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

    # Tropical maximum, near-zero at poles
    pr_base = (base_pr_mmday / _PR_MMDAY) * np.exp(-np.abs(lat_grid) / 30.0)

    # Weak linear trend in kg/m²/s/month
    trend_per_month = trend / (10 * 12 * _PR_MMDAY)
    trend_arr = trend_per_month * np.arange(ntimes)

    data = (
        pr_base[np.newaxis, :, :]
        + trend_arr[:, np.newaxis, np.newaxis]
        + rng.normal(0, pr_base.max() * 0.02,
                     (ntimes, len(lats), len(lons)))
    ).clip(0)  # precipitation is non-negative

    return xr.DataArray(
        data.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


def _make_pr_neglon(lats, lons_180, ntimes=36, base_pr_mmday=2.3, seed=10):
    """ERA5-style precipitation with -180..180 lons and latitude/longitude dims."""
    da = _make_pr_latlon(lats, lons_180, ntimes=ntimes,
                         base_pr_mmday=base_pr_mmday, seed=seed)
    return da.rename({"lat": "latitude", "lon": "longitude"})


# Small test grids (5° resolution for speed)
_LATS = np.arange(-87.5, 90.0, 5.0)
_LONS_360 = np.arange(2.5, 360.0, 5.0)       # MSWEP / normalised ERA5 (0..360)
_LONS_180 = np.arange(-177.5, 180.0, 5.0)    # ERA5 raw (-180..180)

_N_SHORT = 35 * 12   # 35 years → 420 months (1980–2014)
_N_LONG = 44 * 12    # 44 years → 528 months (1980–2023)


@pytest.fixture
def era5_short_360():
    """ERA5 precip already on 0..360 grid (lat/lon dims)."""
    return _make_pr_latlon(_LATS, _LONS_360, ntimes=_N_SHORT, seed=0)


@pytest.fixture
def era5_long_360():
    return _make_pr_latlon(_LATS, _LONS_360, ntimes=_N_LONG, seed=1)


@pytest.fixture
def era5_short_180():
    """ERA5 precip on -180..180 grid (latitude/longitude dims)."""
    return _make_pr_neglon(_LATS, _LONS_180, ntimes=_N_SHORT, seed=2)


@pytest.fixture
def era5_long_180():
    return _make_pr_neglon(_LATS, _LONS_180, ntimes=_N_LONG, seed=3)


@pytest.fixture
def mswep_short():
    """MSWEP precip on 0..360 grid (lat/lon dims)."""
    return _make_pr_latlon(_LATS, _LONS_360, ntimes=_N_SHORT,
                           base_pr_mmday=2.4, seed=4)


@pytest.fixture
def mswep_long():
    return _make_pr_latlon(_LATS, _LONS_360, ntimes=_N_LONG,
                           base_pr_mmday=2.4, seed=5)


@pytest.fixture
def obs_config(tmp_path):
    """Minimal FeatherConfig with ERA5 and MSWEP configured."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root=str(tmp_path),
        obs_datasets={
            "ERA5": {
                "path": str(tmp_path / "ERA5"),
                "variables": {"tp": "tp.nc"},
            },
            "MSWEP": {
                "path": str(tmp_path / "MSWEP"),
                "variables": {"pr": "zarr/mswep.zarr"},
            },
        },
        cmip6={"enabled": False},
        dask={},
        output_dir=str(tmp_path / "output"),
        nereus={"method": "nearest"},
    )


@pytest.fixture
def mock_obs_loader(era5_short_360, era5_long_360, mswep_short, mswep_long):
    """Mock ObsLoader returning synthetic data."""
    loader = MagicMock()

    def _load_for_model_var(var, period=None):
        if period is None or period[1] <= "2014":
            return era5_short_360
        return era5_long_360

    def _load_mswep(period=None):
        if period is None or period[1] <= "2014":
            return mswep_short
        return mswep_long

    loader.load_for_model_var.side_effect = _load_for_model_var
    loader.load_mswep.side_effect = _load_mswep
    return loader


@pytest.fixture
def diag(obs_config, mock_obs_loader):
    """PrecipObsComparisonDiag with mock loaders."""
    return PrecipObsComparisonDiag(
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

    def test_xarray_input(self):
        da = xr.DataArray([1.0, np.nan, 3.0])
        result = _finite_concat([da])
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# B. _normalise_lons
# ══════════════════════════════════════════════════════════════════════════════


class TestNormaliseLons:
    def test_already_360_unchanged(self, era5_short_360):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_360)
        assert float(result.lon.min()) >= 0.0
        np.testing.assert_array_equal(result.lon.values, era5_short_360.lon.values)

    def test_negative_lons_shifted(self, era5_short_180):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_180)
        assert float(result.lon.min()) >= 0.0
        assert float(result.lon.max()) <= 360.0

    def test_renames_latitude_longitude(self, era5_short_180):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_180)
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims

    def test_lons_sorted_after_shift(self, era5_short_180):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_180)
        lons = result.lon.values
        assert np.all(lons[1:] >= lons[:-1]), "lons must be sorted ascending"

    def test_preserves_data_values(self, era5_short_360):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_360)
        np.testing.assert_allclose(
            np.sort(result.values.ravel()),
            np.sort(era5_short_360.values.ravel()),
            rtol=1e-5,
        )

    def test_preserves_time_dim(self, era5_short_360):
        result = PrecipObsComparisonDiag._normalise_lons(era5_short_360)
        assert "time" in result.dims
        assert result.sizes["time"] == era5_short_360.sizes["time"]


# ══════════════════════════════════════════════════════════════════════════════
# C. _interp_to_era5
# ══════════════════════════════════════════════════════════════════════════════


class TestInterpToEra5:
    def test_output_on_target_grid(self, mswep_short, era5_short_360):
        target_lats = era5_short_360.lat.values
        target_lons = era5_short_360.lon.values
        result = PrecipObsComparisonDiag._interp_to_era5(
            mswep_short, target_lats, target_lons
        )
        np.testing.assert_array_equal(result.lat.values, target_lats)
        np.testing.assert_array_equal(result.lon.values, target_lons)

    def test_output_dims_lat_lon(self, mswep_short, era5_short_360):
        result = PrecipObsComparisonDiag._interp_to_era5(
            mswep_short,
            era5_short_360.lat.values,
            era5_short_360.lon.values,
        )
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_no_all_nan_output(self, mswep_short, era5_short_360):
        result = PrecipObsComparisonDiag._interp_to_era5(
            mswep_short,
            era5_short_360.lat.values,
            era5_short_360.lon.values,
        )
        # With wraparound padding there should be very few NaNs
        nan_frac = float(np.isnan(result.values).mean())
        assert nan_frac < 0.05, f"Too many NaNs: {nan_frac:.1%}"

    def test_wraparound_padding_applied(self, era5_short_360):
        """Wrap-around test: MSWEP starting at lon=2.5 still covers lon~0."""
        target_lats = era5_short_360.lat.values
        target_lons = np.arange(0.0, 360.0, 5.0)  # includes 0.0
        result = PrecipObsComparisonDiag._interp_to_era5(
            era5_short_360, target_lats, target_lons
        )
        # lon=0 must not be all-NaN
        assert not np.all(np.isnan(result.sel(lon=0.0, method="nearest").values))

    def test_preserves_time_dimension(self, mswep_short, era5_short_360):
        result = PrecipObsComparisonDiag._interp_to_era5(
            mswep_short,
            era5_short_360.lat.values,
            era5_short_360.lon.values,
        )
        assert result.sizes["time"] == mswep_short.sizes["time"]


# ══════════════════════════════════════════════════════════════════════════════
# D. Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadEra5Pr:
    def test_returns_data_array(self, diag):
        result = diag._load_era5_pr(("1980", "2014"))
        assert isinstance(result, xr.DataArray)

    def test_short_period(self, diag, era5_short_360):
        result = diag._load_era5_pr(("1980", "2014"))
        assert result.shape == era5_short_360.shape

    def test_long_period(self, diag, era5_long_360):
        result = diag._load_era5_pr(("1980", "2023"))
        assert result.shape == era5_long_360.shape


class TestLoadMswepPr:
    def test_returns_data_array(self, diag):
        result = diag._load_mswep_pr(("1980", "2014"))
        assert isinstance(result, xr.DataArray)

    def test_short_period(self, diag, mswep_short):
        result = diag._load_mswep_pr(("1980", "2014"))
        assert result.shape == mswep_short.shape


# ══════════════════════════════════════════════════════════════════════════════
# E. _compute_trend (static)
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeTrend:
    @pytest.fixture
    def da_short(self):
        lats = np.arange(-85.0, 90.0, 10.0)
        lons = np.arange(5.0, 360.0, 10.0)
        return _make_pr_latlon(lats, lons, ntimes=_N_SHORT, trend=0.3, seed=20)

    def test_annual_returns_data_array(self, da_short):
        result = PrecipObsComparisonDiag._compute_trend(da_short, "annual")
        assert isinstance(result, xr.DataArray)

    def test_annual_shape(self, da_short):
        result = PrecipObsComparisonDiag._compute_trend(da_short, "annual")
        assert result.shape == (da_short.sizes["lat"], da_short.sizes["lon"])

    def test_djf_trend(self, da_short):
        result = PrecipObsComparisonDiag._compute_trend(da_short, "DJF")
        assert result is not None
        assert result.shape == (da_short.sizes["lat"], da_short.sizes["lon"])

    def test_jja_trend(self, da_short):
        result = PrecipObsComparisonDiag._compute_trend(da_short, "JJA")
        assert result is not None

    def test_insufficient_data_returns_none(self):
        lats = np.array([-5.0, 5.0])
        lons = np.array([10.0, 20.0])
        da = _make_pr_latlon(lats, lons, ntimes=1, seed=99)
        result = PrecipObsComparisonDiag._compute_trend(da, "annual")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# F. _seasonal_mean (static)
# ══════════════════════════════════════════════════════════════════════════════


class TestSeasonalMean:
    @pytest.fixture
    def da(self):
        lats = np.arange(-85.0, 90.0, 10.0)
        lons = np.arange(5.0, 360.0, 10.0)
        return _make_pr_latlon(lats, lons, ntimes=36, seed=30)

    def test_annual_mean_shape(self, da):
        result = PrecipObsComparisonDiag._seasonal_mean(da, "annual")
        assert result.shape == (da.sizes["lat"], da.sizes["lon"])

    def test_djf_mean_shape(self, da):
        result = PrecipObsComparisonDiag._seasonal_mean(da, "DJF")
        assert result.shape == (da.sizes["lat"], da.sizes["lon"])

    def test_jja_mean_shape(self, da):
        result = PrecipObsComparisonDiag._seasonal_mean(da, "JJA")
        assert result.shape == (da.sizes["lat"], da.sizes["lon"])

    def test_values_non_negative(self, da):
        result = PrecipObsComparisonDiag._seasonal_mean(da, "annual")
        assert float(result.min()) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# G. _compute_all_trends
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeAllTrends:
    @pytest.fixture
    def trend_inputs(self):
        lats = np.arange(-85.0, 90.0, 10.0)
        lons = np.arange(5.0, 360.0, 10.0)
        return (
            _make_pr_latlon(lats, lons, ntimes=_N_SHORT, seed=40),
            _make_pr_latlon(lats, lons, ntimes=_N_LONG, seed=41),
            _make_pr_latlon(lats, lons, ntimes=_N_SHORT, seed=42),
            _make_pr_latlon(lats, lons, ntimes=_N_LONG, seed=43),
        )

    def test_returns_all_period_keys(self, diag, trend_inputs):
        era5_s, era5_l, mswep_s, mswep_l = trend_inputs
        result = diag._compute_all_trends(era5_s, era5_l, mswep_s, mswep_l)
        assert set(result.keys()) == {"annual", "DJF", "JJA"}

    def test_period_diff_keys_present(self, diag, trend_inputs):
        era5_s, era5_l, mswep_s, mswep_l = trend_inputs
        result = diag._compute_all_trends(era5_s, era5_l, mswep_s, mswep_l)
        annual = result["annual"]
        for key in ["era5_short", "era5_long", "mswep_short", "mswep_long",
                    "era5_period_diff", "mswep_period_diff",
                    "dataset_diff_short", "dataset_diff_long"]:
            assert key in annual, f"Missing key: {key}"

    def test_period_diff_is_difference(self, diag, trend_inputs):
        era5_s, era5_l, mswep_s, mswep_l = trend_inputs
        result = diag._compute_all_trends(era5_s, era5_l, mswep_s, mswep_l)
        annual = result["annual"]
        expected = annual["era5_long"] - annual["era5_short"]
        np.testing.assert_allclose(
            annual["era5_period_diff"].values, expected.values, rtol=1e-5,
        )


# ══════════════════════════════════════════════════════════════════════════════
# H. _compute_clim
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeClim:
    @pytest.fixture
    def clim_inputs(self):
        lats = np.arange(-85.0, 90.0, 10.0)
        lons = np.arange(5.0, 360.0, 10.0)
        return (
            _make_pr_latlon(lats, lons, ntimes=_N_SHORT, seed=50),
            _make_pr_latlon(lats, lons, ntimes=_N_LONG, seed=51),
            _make_pr_latlon(lats, lons, ntimes=_N_SHORT, seed=52),
            _make_pr_latlon(lats, lons, ntimes=_N_LONG, seed=53),
        )

    def test_returns_all_keys(self, diag, clim_inputs):
        era5_s, era5_l, ms_s, ms_l = clim_inputs
        result = diag._compute_clim(era5_s, era5_l, ms_s, ms_l)
        assert set(result.keys()) == {"annual", "DJF", "JJA"}

    def test_clim_dict_keys(self, diag, clim_inputs):
        era5_s, era5_l, ms_s, ms_l = clim_inputs
        result = diag._compute_clim(era5_s, era5_l, ms_s, ms_l)
        for key in ["era5_short", "era5_long", "mswep_short", "mswep_long",
                    "diff_short", "diff_long",
                    "rel_bias_short", "rel_bias_long"]:
            assert key in result["annual"], f"Missing key: {key}"

    def test_diff_short_is_difference(self, diag, clim_inputs):
        era5_s, era5_l, ms_s, ms_l = clim_inputs
        result = diag._compute_clim(era5_s, era5_l, ms_s, ms_l)
        annual = result["annual"]
        expected = annual["era5_short"] - annual["mswep_short"]
        np.testing.assert_allclose(
            annual["diff_short"].values, expected.values, rtol=1e-5,
        )

    def test_rel_bias_masked_in_dry_regions(self, diag):
        """Points below threshold should be NaN in relative bias."""
        lats = np.array([-5.0, 5.0])
        lons = np.array([10.0, 20.0, 30.0])
        # Build MSWEP with some sub-threshold values
        era5_c = xr.DataArray(
            np.ones((len(lats), len(lons))) * 1e-6,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        mswep_c = xr.DataArray(
            np.array([[1e-8, 1e-5, 1e-4],   # below, below, above
                      [1e-3, 1e-6, 1e-4]]),  # above, below, above
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        rel = xr.where(
            mswep_c > _REL_BIAS_THRESHOLD,
            (era5_c - mswep_c) / mswep_c * 100.0,
            np.nan,
        )
        below_mask = mswep_c <= _REL_BIAS_THRESHOLD
        assert np.all(np.isnan(rel.values[below_mask.values]))
        assert not np.any(np.isnan(rel.values[~below_mask.values]))

    def test_rel_bias_units_are_percent(self, diag, clim_inputs):
        """Relative bias values should be on a % scale (not fractional)."""
        era5_s, era5_l, ms_s, ms_l = clim_inputs
        result = diag._compute_clim(era5_s, era5_l, ms_s, ms_l)
        rel = result["annual"]["rel_bias_short"]
        finite_vals = rel.values[np.isfinite(rel.values)]
        if len(finite_vals) > 0:
            # Typical percentage biases are in the range ±100%, not ±1
            assert np.abs(finite_vals).max() > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# I. _compute_timeseries
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeTimeseries:
    def test_returns_era5_and_mswep_keys(self, diag, era5_long_360,
                                          mswep_long):
        era5_n = PrecipObsComparisonDiag._normalise_lons(era5_long_360)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(era5_n.lat.values, era5_n.lon.values)
        mswep_r = PrecipObsComparisonDiag._interp_to_era5(
            mswep_long, era5_n.lat.values, era5_n.lon.values
        )
        ts = diag._compute_timeseries(era5_n, mswep_r, area)
        assert "era5" in ts
        assert "mswep" in ts

    def test_timeseries_has_correct_length(self, diag, era5_long_360,
                                            mswep_long):
        era5_n = PrecipObsComparisonDiag._normalise_lons(era5_long_360)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(era5_n.lat.values, era5_n.lon.values)
        mswep_r = PrecipObsComparisonDiag._interp_to_era5(
            mswep_long, era5_n.lat.values, era5_n.lon.values
        )
        ts = diag._compute_timeseries(era5_n, mswep_r, area)
        n_years = _N_LONG // 12
        assert len(ts["era5"]) == n_years
        assert len(ts["mswep"]) == n_years

    def test_global_mean_positive(self, diag, era5_long_360, mswep_long):
        era5_n = PrecipObsComparisonDiag._normalise_lons(era5_long_360)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(era5_n.lat.values, era5_n.lon.values)
        mswep_r = PrecipObsComparisonDiag._interp_to_era5(
            mswep_long, era5_n.lat.values, era5_n.lon.values
        )
        ts = diag._compute_timeseries(era5_n, mswep_r, area)
        assert float(ts["era5"].mean()) > 0.0
        assert float(ts["mswep"].mean()) > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# J. compute() integration
# ══════════════════════════════════════════════════════════════════════════════


class TestCompute:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_top_level_keys(self, results):
        for key in ["trends", "clim", "timeseries",
                    "target_lats", "target_lons", "area"]:
            assert key in results, f"Missing key: {key}"

    def test_trends_period_keys(self, results):
        assert set(results["trends"].keys()) == {"annual", "DJF", "JJA"}

    def test_clim_period_keys(self, results):
        assert set(results["clim"].keys()) == {"annual", "DJF", "JJA"}

    def test_target_grid_is_era5(self, diag, results, era5_short_360):
        np.testing.assert_array_equal(
            results["target_lons"], era5_short_360.lon.values
        )
        np.testing.assert_array_equal(
            results["target_lats"], era5_short_360.lat.values
        )

    def test_area_shape_matches_grid(self, results):
        nlats = len(results["target_lats"])
        nlons = len(results["target_lons"])
        assert results["area"].shape == (nlats, nlons)

    def test_timeseries_both_datasets(self, results):
        assert "era5" in results["timeseries"]
        assert "mswep" in results["timeseries"]


# ══════════════════════════════════════════════════════════════════════════════
# K. plot() — figure shape and type
# ══════════════════════════════════════════════════════════════════════════════


class TestPlot:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_13_figures(self, diag, results):
        figures = diag.plot(results)
        assert len(figures) == 13

    def test_all_matplotlib_figures(self, diag, results):
        figures = diag.plot(results)
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_all_have_metadata(self, diag, results):
        figures = diag.plot(results)
        for fig, meta in figures:
            assert isinstance(meta, dict)
            assert "figure_id" in meta
        plt.close("all")

    def test_figure_ids_unique(self, diag, results):
        figures = diag.plot(results)
        ids = [meta["figure_id"] for _, meta in figures]
        assert len(ids) == len(set(ids)), "Duplicate figure IDs"
        plt.close("all")


class TestPlotTrendMaps:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_3_figures(self, diag, results):
        figs = diag._plot_trend_maps(results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, results):
        figs = diag._plot_trend_maps(results)
        ids = {meta["figure_id"] for _, meta in figs}
        assert ids == {
            "pr_obs_annual_trends",
            "pr_obs_djf_trends",
            "pr_obs_jja_trends",
        }
        plt.close("all")


class TestPlotTrendDiffs:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_3_figures(self, diag, results):
        figs = diag._plot_trend_diffs(results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, results):
        figs = diag._plot_trend_diffs(results)
        ids = {meta["figure_id"] for _, meta in figs}
        assert ids == {
            "pr_obs_annual_trend_diffs",
            "pr_obs_djf_trend_diffs",
            "pr_obs_jja_trend_diffs",
        }
        plt.close("all")


class TestPlotClim:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_3_figures(self, diag, results):
        figs = diag._plot_clim(results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, results):
        figs = diag._plot_clim(results)
        ids = {meta["figure_id"] for _, meta in figs}
        assert ids == {
            "pr_obs_annual_clim",
            "pr_obs_djf_clim",
            "pr_obs_jja_clim",
        }
        plt.close("all")

    def test_5_axes_per_figure(self, diag, results):
        """Each climatology figure has 5 map panels (6 axes, 1 hidden)."""
        figs = diag._plot_clim(results)
        for fig, _ in figs:
            n_visible = sum(1 for ax in fig.axes if ax.get_visible())
            # 5 map axes + 5 colorbar axes
            assert n_visible >= 5
        plt.close("all")


class TestPlotRelativeBias:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_3_figures(self, diag, results):
        figs = diag._plot_relative_bias(results)
        assert len(figs) == 3
        plt.close("all")

    def test_figure_ids(self, diag, results):
        figs = diag._plot_relative_bias(results)
        ids = {meta["figure_id"] for _, meta in figs}
        assert ids == {
            "pr_obs_annual_relative_bias",
            "pr_obs_djf_relative_bias",
            "pr_obs_jja_relative_bias",
        }
        plt.close("all")

    def test_2_panels_per_figure(self, diag, results):
        figs = diag._plot_relative_bias(results)
        for fig, _ in figs:
            # 2 map axes + 2 colorbar axes visible
            n_visible = sum(1 for ax in fig.axes if ax.get_visible())
            assert n_visible >= 2
        plt.close("all")

    def test_units_in_metadata(self, diag, results):
        figs = diag._plot_relative_bias(results)
        for _, meta in figs:
            assert "%" in meta["description"]
        plt.close("all")


class TestPlotTimeseries:
    @pytest.fixture
    def results(self, diag):
        return diag.compute()

    def test_returns_1_figure(self, diag, results):
        figs = diag._plot_timeseries(results)
        assert len(figs) == 1
        plt.close("all")

    def test_figure_id(self, diag, results):
        figs = diag._plot_timeseries(results)
        assert figs[0][1]["figure_id"] == "pr_obs_timeseries"
        plt.close("all")

    def test_ylabel_mmday(self, diag, results):
        figs = diag._plot_timeseries(results)
        fig, _ = figs[0]
        ylabel = fig.axes[0].get_ylabel()
        assert "mm/day" in ylabel
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# L. run() — end-to-end and skip_existing
# ══════════════════════════════════════════════════════════════════════════════


class TestRun:
    def test_run_returns_list(self, diag):
        saved = diag.run(skip_existing=False)
        assert isinstance(saved, list)
        plt.close("all")

    def test_run_saves_13_figures(self, diag):
        saved = diag.run(skip_existing=False)
        assert len(saved) == 13
        plt.close("all")

    def test_run_creates_png_and_json(self, diag):
        saved = diag.run(skip_existing=False)
        for png_path, json_path in saved:
            assert png_path.exists(), f"PNG missing: {png_path}"
            assert json_path.exists(), f"JSON missing: {json_path}"
        plt.close("all")

    def test_run_json_has_figure_id(self, diag):
        saved = diag.run(skip_existing=False)
        for _, json_path in saved:
            data = json.loads(json_path.read_text())
            assert "figure_id" in data
        plt.close("all")

    def test_skip_existing_skips_all(self, diag):
        """Second run with skip_existing=True should return paths without recomputing."""
        diag.run(skip_existing=False)
        call_count_before = diag.obs_loader.load_for_model_var.call_count
        diag.run(skip_existing=True)
        # No new calls to loader — data was skipped
        assert (
            diag.obs_loader.load_for_model_var.call_count == call_count_before
        )
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# M. Registration
# ══════════════════════════════════════════════════════════════════════════════


def test_diagnostic_registered():
    from feather.diag.registry import get_diagnostic
    diag_cls = get_diagnostic("precip_obs_comparison")
    assert diag_cls is PrecipObsComparisonDiag


def test_diagnostic_name():
    assert PrecipObsComparisonDiag.name == "precip_obs_comparison"


def test_diagnostic_group():
    assert PrecipObsComparisonDiag.group == "precipitation"


def test_diagnostic_variables():
    assert "pr" in PrecipObsComparisonDiag.variables
