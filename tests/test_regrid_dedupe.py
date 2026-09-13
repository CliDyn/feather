"""Tests for the pole-safe conservative regridding wrapper.

A regular lat/lon grid that includes ±90° gives SphericalVoronoi duplicate
generators (every longitude collapses to one point at the pole), which is how
conservative remapping broke on the EERIE/ICON/ERA5 grids.
"""

import numpy as np
import pytest

from feather.util.regrid import (
    DedupedInterpolator,
    collapse_values,
    dedupe_points,
    regrid,
)

R_EARTH = 6.371e6


def _area_integral(field, lat2d, dlon, dlat):
    w = (np.deg2rad(dlon) * R_EARTH) * (np.deg2rad(dlat) * R_EARTH) * np.cos(
        np.deg2rad(lat2d))
    m = np.isfinite(field)
    return float(np.nansum(field[m] * w[m]))


@pytest.fixture(scope="module")
def pole_grid():
    """Pole-INCLUSIVE grid, as EERIE/ICON/ERA5 actually publish."""
    lat = np.linspace(-90.0, 90.0, 73)
    lon = np.linspace(0.0, 357.5, 144)
    lo, la = np.meshgrid(lon, lat)
    rng = np.random.default_rng(0)
    return lo, la, rng.gamma(0.4, 8.0, size=la.shape)


@pytest.fixture(scope="module")
def polefree_grid():
    lat = np.linspace(-89.75, 89.75, 72)
    lon = np.linspace(0.25, 359.75, 144)
    lo, la = np.meshgrid(lon, lat)
    return lo, la, np.abs(np.cos(np.deg2rad(la)))


# ── Point deduplication ──────────────────────────────────────────────────


class TestDedupePoints:
    def test_pole_grid_has_duplicates(self, pole_grid):
        lo, la, _ = pole_grid
        _, _, inv = dedupe_points(lo.ravel(), la.ravel())
        assert inv is not None

    def test_collapses_each_pole_row_to_one_point(self, pole_grid):
        lo, la, _ = pole_grid
        lon_u, _, _ = dedupe_points(lo.ravel(), la.ravel())
        n_lon = lo.shape[1]
        # two pole rows of n_lon points each become one point each
        assert lon_u.size == lo.size - 2 * (n_lon - 1)

    def test_polefree_grid_untouched(self, polefree_grid):
        """No duplicates → None, so callers skip the collapse entirely."""
        lo, la, _ = polefree_grid
        lon_u, lat_u, inv = dedupe_points(lo.ravel(), la.ravel())
        assert inv is None
        assert lon_u.size == lo.size

    def test_longitude_wraparound_is_deduped(self):
        """0° and 360° are the same meridian."""
        lon = np.array([0.0, 90.0, 360.0])
        lat = np.array([10.0, 10.0, 10.0])
        lon_u, _, inv = dedupe_points(lon, lat)
        assert inv is not None and lon_u.size == 2

    def test_distinct_nearby_points_kept(self):
        """A 0.25° spacing must never be merged."""
        lon = np.array([0.0, 0.25, 0.5])
        lat = np.array([45.0, 45.0, 45.0])
        _, _, inv = dedupe_points(lon, lat)
        assert inv is None


# ── Value collapsing ─────────────────────────────────────────────────────


class TestCollapseValues:
    def test_averages_duplicate_group(self):
        inv = np.array([0, 0, 0, 1])
        out = collapse_values(np.array([1.0, 2.0, 3.0, 7.0]), inv)
        np.testing.assert_allclose(out, [2.0, 7.0])

    def test_nan_aware_partial_group(self):
        """A group averages only its finite members."""
        inv = np.array([0, 0, 1])
        out = collapse_values(np.array([2.0, np.nan, 5.0]), inv)
        np.testing.assert_allclose(out, [2.0, 5.0])

    def test_all_nan_group_stays_nan(self):
        """A fully masked pole row must not become zero."""
        inv = np.array([0, 0, 1])
        out = collapse_values(np.array([np.nan, np.nan, 5.0]), inv)
        assert np.isnan(out[0]) and out[1] == 5.0

    def test_preserves_leading_dimensions(self):
        inv = np.array([0, 0, 1])
        out = collapse_values(np.ones((4, 3)), inv)
        assert out.shape == (4, 2)


# ── The wrapper ──────────────────────────────────────────────────────────


class TestRegridWrapper:
    def test_conservative_succeeds_on_pole_grid(self, pole_grid):
        """This is the case that failed outright before the wrapper."""
        lo, la, data = pole_grid
        out, _ = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                        resolution=10.0, method="conservative",
                        lon_bounds=(0.0, 360.0))
        assert np.isfinite(out).all()

    def test_conservative_conserves_on_pole_grid(self, pole_grid):
        lo, la, data = pole_grid
        out, it = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                         resolution=10.0, method="conservative",
                         lon_bounds=(0.0, 360.0))
        src = _area_integral(data, la, 2.5, 2.5)
        got = _area_integral(out, it.target_lat, 10.0, 10.0)
        assert abs(got - src) / src < 0.01

    def test_returns_wrapper_only_when_deduped(self, pole_grid, polefree_grid):
        lo, la, data = pole_grid
        _, it = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                       resolution=10.0, method="conservative",
                       lon_bounds=(0.0, 360.0))
        assert isinstance(it, DedupedInterpolator)

        lo2, la2, d2 = polefree_grid
        _, it2 = regrid(d2.ravel(), lon=lo2.ravel(), lat=la2.ravel(),
                        resolution=10.0, method="conservative",
                        lon_bounds=(0.0, 360.0))
        assert not isinstance(it2, DedupedInterpolator)

    def test_cached_interpolator_reusable_on_new_field(self, pole_grid):
        """feather caches one interpolator per source grid and reuses it."""
        lo, la, data = pole_grid
        _, it = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                       resolution=10.0, method="conservative",
                       lon_bounds=(0.0, 360.0))
        other = np.random.default_rng(1).gamma(0.4, 8.0, size=la.shape)
        out = it(other.ravel())
        src = _area_integral(other, la, 2.5, 2.5)
        got = _area_integral(out, it.target_lat, 10.0, 10.0)
        assert abs(got - src) / src < 0.01

    def test_target_coords_proxied(self, pole_grid):
        """Callers do interp.target_lat[:, 0] — must pass through."""
        lo, la, data = pole_grid
        _, it = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                       resolution=10.0, method="conservative",
                       lon_bounds=(0.0, 360.0))
        assert it.target_lat.ndim == 2
        assert it.target_lat[:, 0].size == it.target_lat.shape[0]
        assert it.method == "conservative"

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_non_conservative_delegates_unchanged(self, pole_grid, method):
        import nereus as nr

        lo, la, data = pole_grid
        a, _ = regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                      resolution=10.0, method=method, lon_bounds=(0.0, 360.0))
        b, _ = nr.regrid(data.ravel(), lon=lo.ravel(), lat=la.ravel(),
                         resolution=10.0, method=method,
                         lon_bounds=(0.0, 360.0))
        np.testing.assert_array_equal(a, b)

    def test_masked_pole_row_does_not_become_zero(self, pole_grid):
        lo, la, data = pole_grid
        masked = data.copy()
        masked[0, :] = np.nan
        out, it = regrid(masked.ravel(), lon=lo.ravel(), lat=la.ravel(),
                         resolution=10.0, method="conservative",
                         lon_bounds=(0.0, 360.0))
        assert np.nanmin(out) > 0.0
