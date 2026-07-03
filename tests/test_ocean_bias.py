"""Tests for the shared ocean benchmark-bias module (feather.diag.ocean_bias).

Pure-function coverage only (no real data): grid construction, unit/grid
helpers, scattered regridding, and the export guard.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from feather.diag import ocean_bias as ob


def _latlon(value, lats, lons):
    return xr.DataArray(
        np.full((len(lats), len(lons)), value, dtype=float),
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
    )


class TestGrid:
    def test_grid_resolution_from_config(self):
        cfg = SimpleNamespace(nereus={"resolution": 0.25})
        assert ob.grid_resolution(cfg) == 0.25

    def test_grid_resolution_fallback(self):
        cfg = SimpleNamespace(nereus={})
        assert ob.grid_resolution(cfg) == ob.OCEAN_RES == 0.25

    def test_common_grid_quarter_degree(self):
        cfg = SimpleNamespace(nereus={"resolution": 0.25})
        lat, lon = ob.common_grid(cfg)
        assert len(lat) == 720 and len(lon) == 1440
        assert lat[0] == pytest.approx(-89.875)
        assert lon[0] == pytest.approx(0.125)

    def test_common_grid_one_degree(self):
        lat, lon = ob.common_grid(res=1.0)
        assert len(lat) == 180 and len(lon) == 360


class TestFieldHelpers:
    def test_latlon_names_default(self):
        da = _latlon(1.0, [0, 1], [0, 1])
        assert ob.latlon_names(da) == ("lat", "lon")

    def test_latlon_names_curvilinear(self):
        da = xr.DataArray(
            np.zeros((2, 2)), dims=("y", "x"),
            coords={"nav_lat": (("y", "x"), np.zeros((2, 2))),
                    "nav_lon": (("y", "x"), np.zeros((2, 2)))},
        )
        assert ob.latlon_names(da) == ("nav_lat", "nav_lon")

    def test_surface_slice(self):
        da = xr.DataArray(
            np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(float),
            dims=("lev", "lat", "lon"),
            coords={"lev": [0.0, 50.0], "lat": [0, 1], "lon": [0, 1]},
        )
        assert "lev" not in ob.surface_slice(da).dims
        assert float(ob.surface_slice(da).isel(lat=0, lon=0)) == 0.0

    def test_to_celsius_kelvin(self):
        da = _latlon(300.0, [0, 1], [0, 1])
        da.attrs["units"] = "K"
        assert float(ob.to_celsius_if_needed(da).max()) == pytest.approx(26.85, abs=1e-2)

    def test_to_celsius_heuristic(self):
        da = _latlon(290.0, [0, 1], [0, 1])  # no units, >150 => Kelvin
        assert float(ob.to_celsius_if_needed(da).max()) < 100.0

    def test_siconc_percent(self):
        da = _latlon(75.0, [0, 1], [0, 1])
        assert float(ob.siconc_to_fraction(da).max()) == pytest.approx(0.75)

    def test_sa_to_sp_noop_without_config_flag(self):
        cfg = SimpleNamespace(model_configs={})
        da = _latlon(35.0, [0, 1], [0, 1])
        assert float(ob.surface_sa_to_sp(cfg, da, "X").max()) == 35.0


class TestRegridScatter:
    def test_rectilinear_constant_preserved(self):
        src = _latlon(5.0, np.arange(-80, 81, 20.0), np.arange(10, 360, 20.0))
        tlat = np.arange(-89.5, 90, 1.0)
        tlon = np.arange(0.5, 360, 1.0)
        out = ob.regrid_scatter(src, tlat, tlon, 1.0, 1_000_000, {})
        assert out.shape == (len(tlat), len(tlon))
        assert float(np.nanmax(out.values)) == pytest.approx(5.0, abs=1e-6)

    def test_curvilinear(self):
        ny, nx = 8, 10
        lat2d = np.tile(np.linspace(-70, 70, ny)[:, None], (1, nx))
        lon2d = np.tile(np.linspace(0, 330, nx)[None, :], (ny, 1))
        src = xr.DataArray(
            np.full((ny, nx), 2.0), dims=("y", "x"),
            coords={"nav_lat": (("y", "x"), lat2d),
                    "nav_lon": (("y", "x"), lon2d)},
        )
        tlat = np.arange(-89, 90, 2.0)
        tlon = np.arange(1, 360, 2.0)
        out = ob.regrid_scatter(src, tlat, tlon, 2.0, 2_000_000, {})
        assert out.shape == (len(tlat), len(tlon))


class TestExportGuard:
    def test_noop_without_save_netcdf(self):
        diag = SimpleNamespace(save_netcdf=False, benchmarks=[object()])
        assert ob.maybe_export_ocean_bias(diag, ["tos"]) == []

    def test_noop_without_benchmarks(self):
        diag = SimpleNamespace(save_netcdf=True, benchmarks=[])
        assert ob.maybe_export_ocean_bias(diag, ["tos"]) == []


class TestObsDiagMap:
    def test_matches_added_value_source(self):
        from feather.diag.added_value import AddedValueDiag
        for obs, diag_name in ob.OCEAN_OBS_DIAG.items():
            assert AddedValueDiag._OBS_NETCDF_SOURCE.get(obs) == diag_name
