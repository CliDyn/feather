"""Tests for spatial utilities."""

import numpy as np
import pytest

import xarray as xr

from feather.util.spatial import (
    global_mean,
    spatial_anova,
    spatial_ttest,
    spatial_variance_ratio,
    zonal_mean,
    zonal_profile_to_axis,
)


def test_zonal_mean_basic(synth_healpix):
    """Zonal mean of synthetic temperature field."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    lat = ds["latitude"]

    # Use 10° bins for nside=8 (768 cells) to ensure bins are populated
    bins = np.arange(-90, 91, 10.0)
    zm = zonal_mean(t2m, lat, lat_bins=bins)

    assert "lat" in zm.dims
    # Equatorial bins should be warmer than polar bins
    valid = ~np.isnan(zm.values)
    assert valid.any()
    equator_idx = np.argmin(np.abs(zm.lat.values))
    south_idx = np.where(valid)[0][0]
    north_idx = np.where(valid)[0][-1]
    assert zm.values[equator_idx] > zm.values[south_idx]
    assert zm.values[equator_idx] > zm.values[north_idx]


def test_zonal_mean_with_time(synth_healpix):
    """Zonal mean preserves time dimension."""
    ds = synth_healpix
    t2m = ds["avg_2t"]
    lat = ds["latitude"]

    zm = zonal_mean(t2m, lat)

    assert "time" in zm.dims
    assert "lat" in zm.dims
    assert zm.sizes["time"] == 12


def test_zonal_mean_weighted(synth_healpix):
    """Zonal mean with area weights."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    lat = ds["latitude"]
    area = ds["area"]

    zm = zonal_mean(t2m, lat, weights=area)
    assert not np.all(np.isnan(zm.values))


def test_zonal_mean_nan_handling():
    """Zonal mean excludes NaN pixels from denominator.

    Regression test: previously NaN (land) pixels were included in the
    denominator, diluting the zonal mean for ocean-only variables.
    """
    import xarray as xr

    # 10 pixels: 5 in band 0 (0-10°), 5 in band 1 (10-20°)
    lat = np.array([5.0, 5.0, 5.0, 5.0, 5.0,
                     15.0, 15.0, 15.0, 15.0, 15.0])
    # Band 0: 3 ocean (value 20) + 2 land (NaN)
    # Band 1: all ocean (value 10)
    data = np.array([20.0, 20.0, 20.0, np.nan, np.nan,
                      10.0, 10.0, 10.0, 10.0, 10.0])
    da = xr.DataArray(data, dims=("values",))

    bins = np.array([0.0, 10.0, 20.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    # Band 0 should be 20.0 (mean of the 3 valid pixels), NOT 12.0 (60/5)
    assert zm.values[0] == pytest.approx(20.0)
    # Band 1 should be 10.0
    assert zm.values[1] == pytest.approx(10.0)


def test_zonal_mean_nan_with_time():
    """Zonal mean NaN handling works with extra time dimension."""
    import xarray as xr

    lat = np.array([5.0, 5.0, 5.0, 5.0])
    # 2 timesteps × 4 pixels; pixel 3 is always NaN (land)
    data = np.array([
        [10.0, 20.0, 30.0, np.nan],
        [40.0, 50.0, 60.0, np.nan],
    ])
    da = xr.DataArray(data, dims=("time", "values"))
    bins = np.array([0.0, 10.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    # t=0: mean of [10, 20, 30] = 20.0 (not 60/4 = 15.0)
    assert zm.values[0, 0] == pytest.approx(20.0)
    # t=1: mean of [40, 50, 60] = 50.0 (not 150/4 = 37.5)
    assert zm.values[1, 0] == pytest.approx(50.0)


def test_zonal_mean_all_nan_band():
    """Zonal mean returns NaN for latitude band where all pixels are NaN."""
    import xarray as xr

    lat = np.array([5.0, 5.0, 15.0, 15.0])
    data = np.array([np.nan, np.nan, 10.0, 20.0])
    da = xr.DataArray(data, dims=("values",))
    bins = np.array([0.0, 10.0, 20.0])
    zm = zonal_mean(da, lat, lat_bins=bins)

    assert np.isnan(zm.values[0])    # all-NaN band → NaN
    assert zm.values[1] == pytest.approx(15.0)


def test_global_mean(synth_healpix):
    """Global mean of synthetic temperature field."""
    ds = synth_healpix
    t2m = ds["avg_2t"].isel(time=0)
    area = ds["area"]

    gm = global_mean(t2m, area)

    # Should be roughly between 260 and 300 K
    assert 250 < float(gm) < 310


def test_global_mean_with_time(synth_healpix):
    """Global mean preserves time dimension."""
    ds = synth_healpix
    t2m = ds["avg_2t"]
    area = ds["area"]

    gm = global_mean(t2m, area)
    assert "time" in gm.dims
    assert gm.sizes["time"] == 12


# -- spatial_ttest tests ---------------------------------------------------


def test_spatial_ttest_identical_fields():
    """Identical fields -> high p-value (no significant difference)."""
    rng = np.random.default_rng(42)
    field = rng.normal(280, 5, (10, 20))
    t, p = spatial_ttest(field, field)
    assert t == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_spatial_ttest_shifted_fields():
    """Shifted fields -> significant difference."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 5, (50, 100))
    model = obs + 2.0  # systematic 2K warm bias
    t, p = spatial_ttest(model, obs)
    assert t > 0  # positive bias
    assert p < 0.01  # highly significant


def test_spatial_ttest_nan_handling():
    """NaN grid points are excluded from test."""
    rng = np.random.default_rng(42)
    field_a = rng.normal(280, 5, (10, 20))
    field_b = rng.normal(280, 5, (10, 20))
    # Sprinkle NaNs in different locations
    field_a[0, 0] = np.nan
    field_b[1, 1] = np.nan
    t, p = spatial_ttest(field_a, field_b)
    assert np.isfinite(t) and np.isfinite(p)


def test_spatial_ttest_too_few_points():
    """Returns NaN when too few valid points."""
    t, p = spatial_ttest(np.array([1.0, np.nan]), np.array([np.nan, 2.0]))
    assert np.isnan(t) and np.isnan(p)


def test_spatial_ttest_xarray_input():
    """Works with xarray DataArrays as input."""
    import xarray as xr

    rng = np.random.default_rng(42)
    a = xr.DataArray(rng.normal(280, 5, (10, 20)), dims=("lat", "lon"))
    b = xr.DataArray(rng.normal(281, 5, (10, 20)), dims=("lat", "lon"))
    t, p = spatial_ttest(a, b)
    assert np.isfinite(t) and np.isfinite(p)


def test_spatial_ttest_weighted():
    """Weighted t-test uses area weights and Kish's effective sample size."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 5, (50, 100))
    model = obs + 2.0 + rng.normal(0, 0.5, (50, 100))
    weights = np.ones((50, 100))

    # Uniform weights should match unweighted result closely
    t_w, p_w = spatial_ttest(model, obs, weights=weights)
    t_u, p_u = spatial_ttest(model, obs)
    assert abs(t_w - t_u) / abs(t_u) < 0.05  # within 5%

    # With non-uniform weights, result should differ
    weights_nonunif = rng.uniform(0.5, 2.0, (50, 100))
    t_nw, p_nw = spatial_ttest(model, obs, weights=weights_nonunif)
    assert np.isfinite(t_nw) and np.isfinite(p_nw)
    assert t_nw != t_u  # different from unweighted


def test_spatial_ttest_weighted_sign_consistency():
    """Weighted t-test sign matches sign of weighted mean difference."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 5, (50, 100))
    model = obs + 2.0 + rng.normal(0, 0.5, (50, 100))
    weights = rng.uniform(0.5, 2.0, (50, 100))

    t, p = spatial_ttest(model, obs, weights=weights)
    # Model is warmer -> positive weighted mean difference -> positive t
    assert t > 0


def test_spatial_ttest_weighted_neff_smaller():
    """Kish's n_eff with non-uniform weights gives larger p-value than raw n."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 5, (50, 100))
    model = obs + 0.1 + rng.normal(0, 0.5, (50, 100))  # small bias

    # Very non-uniform weights (simulates cos-lat)
    lats = np.linspace(-90, 90, 50)
    cos_lat = np.cos(np.deg2rad(lats))
    weights = np.repeat(cos_lat[:, np.newaxis], 100, axis=1)

    _, p_w = spatial_ttest(model, obs, weights=weights)
    _, p_u = spatial_ttest(model, obs)
    # Weighted p-value should be >= unweighted (fewer effective DOF)
    assert p_w >= p_u or (p_w < 1e-10 and p_u < 1e-10)


# -- spatial_anova tests ---------------------------------------------------


def test_spatial_anova_distinct_groups():
    """ANOVA detects significant difference between distinct groups."""
    rng = np.random.default_rng(42)
    a = rng.normal(280, 2, (50, 100))
    b = rng.normal(285, 2, (50, 100))
    c = rng.normal(282, 2, (50, 100))
    f, p = spatial_anova(a, b, c)
    assert f > 1.0
    assert p < 0.01


def test_spatial_anova_identical_groups():
    """ANOVA with identical groups -> non-significant."""
    rng = np.random.default_rng(42)
    field = rng.normal(280, 5, (10, 20))
    f, p = spatial_anova(field, field.copy(), field.copy())
    assert p == pytest.approx(1.0, abs=0.01)


def test_spatial_anova_nan_handling():
    """NaN grid points excluded from ANOVA."""
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.5, np.nan, 3.5, 4.5])
    c = np.array([np.nan, 2.5, 3.0, 4.0])
    f, p = spatial_anova(a, b, c)
    # Only index 3 is valid for all -> only 1 sample per group -> too few
    assert np.isnan(f) and np.isnan(p)


def test_spatial_anova_single_group():
    """Returns NaN with only one group."""
    f, p = spatial_anova(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(f) and np.isnan(p)


def test_spatial_anova_two_groups():
    """ANOVA with two distinct groups works (reduces to t-test)."""
    rng = np.random.default_rng(42)
    a = rng.normal(280, 2, (50, 100))
    b = rng.normal(285, 2, (50, 100))
    f, p = spatial_anova(a, b)
    assert f > 1.0
    assert p < 0.01


# -- spatial_variance_ratio tests ------------------------------------------


def test_variance_ratio_equal_variance():
    """Fields with same variance -> F close to 1."""
    rng = np.random.default_rng(42)
    a = rng.normal(280, 5, (50, 100))
    b = rng.normal(282, 5, (50, 100))
    f, p = spatial_variance_ratio(a, b)
    assert 0.9 < f < 1.1  # close to 1
    assert p > 0.05  # not significant


def test_variance_ratio_model_more_variable():
    """Model with more variance -> F > 1."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 3, (50, 100))
    model = rng.normal(280, 10, (50, 100))  # much more variable
    f, p = spatial_variance_ratio(model, obs)
    assert f > 5.0  # ~(10/3)^2 ≈ 11
    assert p < 0.01


def test_variance_ratio_model_less_variable():
    """Model with less variance -> F < 1."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 10, (50, 100))
    model = rng.normal(280, 3, (50, 100))  # much less variable
    f, p = spatial_variance_ratio(model, obs)
    assert f < 0.2  # ~(3/10)^2 ≈ 0.09
    assert p < 0.01


def test_variance_ratio_identical_fields():
    """Identical fields -> F = 1, p = 1."""
    rng = np.random.default_rng(42)
    field = rng.normal(280, 5, (10, 20))
    f, p = spatial_variance_ratio(field, field)
    assert f == pytest.approx(1.0)
    assert p == pytest.approx(1.0)


def test_variance_ratio_nan_handling():
    """NaN grid points excluded from variance ratio."""
    rng = np.random.default_rng(42)
    a = rng.normal(280, 5, (10, 20))
    b = rng.normal(280, 5, (10, 20))
    a[0, 0] = np.nan
    b[1, 1] = np.nan
    f, p = spatial_variance_ratio(a, b)
    assert np.isfinite(f) and np.isfinite(p)


def test_variance_ratio_too_few_points():
    """Returns NaN when too few valid points."""
    f, p = spatial_variance_ratio(
        np.array([1.0, np.nan]), np.array([np.nan, 2.0]),
    )
    assert np.isnan(f) and np.isnan(p)


def test_variance_ratio_zero_obs_variance():
    """When obs variance is 0 but model is not, F = inf."""
    model = np.array([1.0, 2.0, 3.0, 4.0])
    obs = np.array([5.0, 5.0, 5.0, 5.0])
    f, p = spatial_variance_ratio(model, obs)
    assert f == float("inf")
    assert p == 0.0


def test_variance_ratio_both_zero_variance():
    """Both fields constant -> F = 1, p = 1."""
    model = np.array([5.0, 5.0, 5.0, 5.0])
    obs = np.array([3.0, 3.0, 3.0, 3.0])
    f, p = spatial_variance_ratio(model, obs)
    assert f == pytest.approx(1.0)
    assert p == pytest.approx(1.0)


def test_variance_ratio_weighted():
    """Weighted variance ratio uses area weights."""
    rng = np.random.default_rng(42)
    obs = rng.normal(280, 5, (50, 100))
    model = rng.normal(280, 5, (50, 100))
    weights = np.ones((50, 100))

    # Uniform weights should match unweighted result closely
    f_w, p_w = spatial_variance_ratio(model, obs, weights=weights)
    f_u, p_u = spatial_variance_ratio(model, obs)
    assert abs(f_w - f_u) / f_u < 0.05

    # Non-uniform weights should differ
    weights_nonunif = rng.uniform(0.5, 2.0, (50, 100))
    f_nw, p_nw = spatial_variance_ratio(model, obs, weights=weights_nonunif)
    assert np.isfinite(f_nw) and np.isfinite(p_nw)


def test_variance_ratio_uses_population_variance():
    """Variance ratio uses ddof=0 (population variance)."""
    # 4 points: var with ddof=0 is 1.25, var with ddof=1 is 1.6667
    model = np.array([1.0, 2.0, 3.0, 4.0])
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    f, _ = spatial_variance_ratio(model, obs)
    assert f == pytest.approx(1.0)  # same data -> F=1 regardless of ddof

    # Different data: check actual variance value
    model2 = np.array([0.0, 0.0, 2.0, 2.0])  # var(ddof=0) = 1.0
    obs2 = np.array([0.0, 0.0, 4.0, 4.0])    # var(ddof=0) = 4.0
    f2, _ = spatial_variance_ratio(model2, obs2)
    assert f2 == pytest.approx(0.25)  # 1.0/4.0 = 0.25


class TestZonalProfileToAxis:
    """zonal_profile_to_axis: rectilinear + unstructured onto a fixed axis."""

    TARGET = np.arange(-89.5, 90.0, 1.0)

    def test_rectilinear_lat_lon(self):
        lats = np.arange(-88.0, 90, 4.0)
        lons = np.arange(0, 360, 5.0)
        data = np.broadcast_to(
            (280 - 30 * np.abs(lats / 90.0))[:, None], (len(lats), len(lons)),
        ).astype(float)
        da = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        zm = zonal_profile_to_axis(da, self.TARGET)
        assert zm.dims == ("lat",)
        assert zm.sizes["lat"] == len(self.TARGET)
        assert bool(np.isfinite(zm.sel(lat=slice(-80, 80))).all())

    def test_rectilinear_latitude_longitude_names(self):
        lats = np.arange(-85.0, 90, 5.0)
        lons = np.arange(0, 360, 10.0)
        da = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("latitude", "longitude"),
            coords={"latitude": lats, "longitude": lons},
        )
        zm = zonal_profile_to_axis(da, self.TARGET)
        assert zm.dims == ("lat",)
        assert float(zm.sel(lat=0.5)) == pytest.approx(1.0)

    def test_unstructured_per_cell_coords(self):
        """ICON-like: 1-D lat/lon coords on a single non-lat dim."""
        rng = np.random.default_rng(0)
        n = 4000
        ulat = rng.uniform(-89, 89, n)
        ulon = rng.uniform(0, 360, n)
        da = xr.DataArray(
            280 - 30 * np.abs(ulat / 90.0),
            dims=("ncells",),
            coords={"latitude": ("ncells", ulat),
                    "longitude": ("ncells", ulon)},
        )
        zm = zonal_profile_to_axis(da, self.TARGET)
        assert zm.dims == ("lat",)
        assert zm.sizes["lat"] == len(self.TARGET)
        # Interior bands populated; profile decreases away from equator.
        interior = zm.sel(lat=slice(-80, 80))
        assert float(np.isfinite(interior).mean()) > 0.9
        assert float(zm.sel(lat=0.5)) > float(zm.sel(lat=70.5))

    def test_returns_none_without_lat(self):
        da = xr.DataArray(np.ones(5), dims=("x",))
        assert zonal_profile_to_axis(da, self.TARGET) is None

    def test_drops_scalar_height_coord_for_concat(self):
        """Scalar coords (e.g. height=2m on tas) are dropped so members concat.

        Regression: members with a scalar ``height`` coord and members without
        could not be xr.concat'd ("'height' not present in all datasets").
        """
        lats = np.arange(-85.0, 90, 5.0)
        lons = np.arange(0, 360, 10.0)
        with_h = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons, "height": 2.0},
        )
        without_h = xr.DataArray(
            np.ones((len(lats), len(lons))) * 2,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        za = zonal_profile_to_axis(with_h, self.TARGET)
        zb = zonal_profile_to_axis(without_h, self.TARGET)
        assert "height" not in za.coords
        # Both clean → concat across members works.
        stacked = xr.concat([za, zb], dim="_member")
        assert stacked.sizes["_member"] == 2
