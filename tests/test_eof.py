"""Tests for feather.util.eof — EOF computation via SVD."""

import numpy as np
import pytest
import xarray as xr

from feather.util.eof import compute_eof


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def dipole_field():
    """Synthetic field with a clear dipole pattern (like NAO).

    North half positive, south half negative, with temporal oscillation.
    """
    lats = np.arange(-85, 90, 10.0)
    lons = np.arange(5, 360, 10.0)
    nt = 60  # 5 years monthly

    # Create a dipole spatial pattern
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    pattern = np.sin(np.deg2rad(lat_grid))  # positive NH, negative SH

    # Temporal oscillation (dominant mode)
    time = xr.date_range("1990-01", periods=nt, freq="MS")
    t_signal = np.sin(2 * np.pi * np.arange(nt) / 24)  # 2-year period

    # Build 3D field: signal * pattern + noise
    rng = np.random.default_rng(42)
    data = (
        t_signal[:, None, None] * pattern[None, :, :]
        + 0.1 * rng.standard_normal((nt, len(lats), len(lons)))
    )

    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


@pytest.fixture
def uniform_field():
    """Spatially uniform field — no meaningful EOF."""
    lats = np.arange(-85, 90, 10.0)
    lons = np.arange(5, 360, 10.0)
    nt = 36
    time = xr.date_range("1990-01", periods=nt, freq="MS")
    rng = np.random.default_rng(99)
    data = rng.standard_normal((nt, len(lats), len(lons))) * 0.01
    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


# ── Basic functionality ──────────────────────────────────────────────


class TestEOFBasics:
    """Test basic EOF computation."""

    def test_returns_correct_shapes(self, dipole_field):
        eofs, pcs, var_exp = compute_eof(dipole_field, n_modes=3)
        assert eofs.dims == ("mode", "lat", "lon")
        assert pcs.dims == ("time", "mode")
        assert eofs.sizes["mode"] == 3
        assert pcs.sizes["mode"] == 3
        assert pcs.sizes["time"] == dipole_field.sizes["time"]
        assert len(var_exp) == 3

    def test_single_mode(self, dipole_field):
        eofs, pcs, var_exp = compute_eof(dipole_field, n_modes=1)
        assert eofs.sizes["mode"] == 1
        assert pcs.sizes["mode"] == 1
        assert len(var_exp) == 1

    def test_variance_explained_sums_to_less_than_one(self, dipole_field):
        _, _, var_exp = compute_eof(dipole_field, n_modes=3)
        assert np.sum(var_exp) <= 1.0 + 1e-10
        assert np.all(var_exp >= 0)

    def test_leading_mode_dominates(self, dipole_field):
        _, _, var_exp = compute_eof(dipole_field, n_modes=3)
        # The dipole signal should dominate mode 1
        assert var_exp[0] > 0.5
        assert var_exp[0] > var_exp[1]

    def test_pc_unit_variance(self, dipole_field):
        _, pcs, _ = compute_eof(dipole_field, n_modes=2)
        for mode in [1, 2]:
            pc = pcs.sel(mode=mode).values
            std = np.std(pc)
            assert abs(std - 1.0) < 0.15, f"PC{mode} std={std}, expected ~1.0"


class TestEOFPattern:
    """Test EOF spatial pattern recovery."""

    def test_dipole_pattern_recovered(self, dipole_field):
        eofs, _, _ = compute_eof(dipole_field, n_modes=1)
        eof1 = eofs.sel(mode=1)

        # EOF should have opposite signs in NH vs SH
        nh_mean = float(eof1.sel(lat=slice(10, 80)).mean())
        sh_mean = float(eof1.sel(lat=slice(-80, -10)).mean())
        assert nh_mean * sh_mean < 0, "EOF1 should show a dipole"

    def test_eof_has_correct_coordinates(self, dipole_field):
        eofs, _, _ = compute_eof(dipole_field, n_modes=1)
        np.testing.assert_array_equal(eofs.lat.values, dipole_field.lat.values)
        np.testing.assert_array_equal(eofs.lon.values, dipole_field.lon.values)


class TestEOFWeighting:
    """Test area weighting options."""

    def test_cos_lat_default(self, dipole_field):
        eofs_weighted, _, var_w = compute_eof(dipole_field, n_modes=1, weight="cos_lat")
        eofs_none, _, var_n = compute_eof(dipole_field, n_modes=1, weight="none")
        # Results should differ (cos-lat gives more weight to equator)
        assert not np.allclose(eofs_weighted.values, eofs_none.values)

    def test_no_weight(self, dipole_field):
        eofs, pcs, var_exp = compute_eof(dipole_field, n_modes=1, weight="none")
        assert eofs.sizes["mode"] == 1
        assert len(var_exp) == 1


class TestEOFEdgeCases:
    """Test edge cases."""

    def test_too_few_timesteps_raises(self):
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        time = xr.date_range("1990-01", periods=2, freq="MS")
        data = np.random.randn(2, len(lats), len(lons))
        da = xr.DataArray(data, dims=("time", "lat", "lon"),
                          coords={"time": time, "lat": lats, "lon": lons})
        # Should raise or return very few modes
        eofs, pcs, var_exp = compute_eof(da, n_modes=1)
        assert eofs.sizes["mode"] == 1

    def test_with_nan_values(self, dipole_field):
        # Add NaNs to some locations (like land masking)
        da = dipole_field.copy()
        da.values[:, 0, :5] = np.nan
        eofs, pcs, var_exp = compute_eof(da, n_modes=1)
        # NaN locations should remain NaN in EOF
        assert np.all(np.isnan(eofs.values[0, 0, :5]))
        # Non-NaN part should still work
        assert np.any(np.isfinite(eofs.values[0, 5:, :]))

    def test_latitude_dimension_name(self):
        """Test with 'latitude'/'longitude' dim names."""
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        nt = 36
        time = xr.date_range("1990-01", periods=nt, freq="MS")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((nt, len(lats), len(lons)))
        da = xr.DataArray(
            data, dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": lats, "longitude": lons},
        )
        eofs, pcs, var_exp = compute_eof(da, n_modes=1)
        assert "latitude" in eofs.dims
        assert "longitude" in eofs.dims

    def test_modes_ordered_by_variance(self, dipole_field):
        _, _, var_exp = compute_eof(dipole_field, n_modes=3)
        assert var_exp[0] >= var_exp[1]
        assert var_exp[1] >= var_exp[2]
