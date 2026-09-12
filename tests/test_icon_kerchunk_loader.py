"""Tests for ICONKerchunkLoader — ICON-ESM-ER r2/r3 gr025 reference stores."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.icon_kerchunk_loader import (
    ICONKerchunkLoader,
    _ATMOS2D,
    _FILL_THRESHOLD,
    _OCEAN2D,
    _STORE_FILES,
)

MODEL = "ICON-ESM-ER-r2"

# Tiny stand-in for the real 721×1440 grid, same orientation:
# lat ascending -90→90, lon 0→360 exclusive.
LATS = np.array([-60.0, 0.0, 60.0])
LONS = np.array([0.0, 90.0, 180.0, 270.0])
N_LAT, N_LON = LATS.size, LONS.size


# ── Synthetic store builders ─────────────────────────────────────────────


def _make_atmos_store(n_months=6):
    """Atmos store: every mapped variable, singleton height dims where real."""
    time = xr.date_range("1980-01", periods=n_months, freq="MS")
    base = np.arange(n_months * N_LAT * N_LON, dtype=np.float32)
    base = base.reshape(n_months, N_LAT, N_LON)

    # These carry a singleton vertical dim in the real store.
    levelled = {
        "tas": "height", "tasmin": "height", "tasmax": "height",
        "hus2m": "height", "uas": "height_2", "vas": "height_2",
        "sfcwind": "height_2", "hur": "height_3",
    }

    ds_vars = {}
    for store_name in sorted({v[0] for v in _ATMOS2D.values()}):
        if store_name in levelled:
            dim = levelled[store_name]
            ds_vars[store_name] = xr.DataArray(
                base[:, None, :, :].copy(),
                dims=["time", dim, "lat", "lon"],
                attrs={"units": "K"},
            )
        else:
            ds_vars[store_name] = xr.DataArray(
                base.copy(), dims=["time", "lat", "lon"], attrs={"units": "K"},
            )

    coords = {"time": time, "lat": LATS, "lon": LONS,
              "height": [2.0], "height_2": [10.0], "height_3": [2.0]}
    return xr.Dataset(ds_vars, coords=coords)


def _make_ocean_store(n_months=6):
    """Ocean store: land marked with the -9e33 sentinel, real zeros kept."""
    time = xr.date_range("1980-01", periods=n_months, freq="MS")
    base = np.ones((n_months, N_LAT, N_LON), dtype=np.float32) * 0.5
    # One cell is land (sentinel), one is a genuine zero.
    base[:, 0, 0] = -9e33
    base[:, 0, 1] = 0.0

    levelled = {"to": "depth", "so": "depth",
                "conc": "lev", "hi": "lev", "hs": "lev"}

    ds_vars = {}
    for store_name in sorted({v[0] for v in _OCEAN2D.values()}):
        if store_name in levelled:
            dim = levelled[store_name]
            ds_vars[store_name] = xr.DataArray(
                base[:, None, :, :].copy(),
                dims=["time", dim, "lat", "lon"],
                attrs={"units": "C"},
            )
        else:
            ds_vars[store_name] = xr.DataArray(
                base.copy(), dims=["time", "lat", "lon"], attrs={"units": "m"},
            )

    coords = {"time": time, "lat": LATS, "lon": LONS,
              "depth": [1.0], "lev": [0.0]}
    return xr.Dataset(ds_vars, coords=coords)


def _make_config(tmp_path, *, member=2, scale_factors=None):
    return FeatherConfig(
        model_catalogs={},
        models=[MODEL],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "icon_kerchunk", "root": str(tmp_path / "stores")},
        model_configs={
            MODEL: ModelConfig(
                name=MODEL,
                institution="MPI-M",
                experiment="hist-1950",
                variant="r2i1p1f1",
                data_source_type="icon_kerchunk",
                member=member,
                grids={"sfc": "latlon", "o2d": "latlon"},
                color="#66c266",
                scale_factors=scale_factors or {},
            )
        },
    )


def _make_loader(tmp_path, monkeypatch, **cfg_kwargs):
    cfg = _make_config(tmp_path, **cfg_kwargs)
    loader = ICONKerchunkLoader(cfg)
    stores = {"atmos2d": _make_atmos_store(), "ocean2d": _make_ocean_store()}
    monkeypatch.setattr(loader, "_open_store", lambda m, s: stores[s])
    return loader, stores


# ── Variable maps ────────────────────────────────────────────────────────


class TestVariableMaps:
    def test_atmos_and_ocean_names_disjoint(self):
        assert not set(_ATMOS2D) & set(_OCEAN2D)

    def test_clt_scaled_fraction_to_percent(self):
        assert _ATMOS2D["clt"] == ("clt", 100.0, 0.0)

    def test_siconc_scaled_fraction_to_percent(self):
        assert _OCEAN2D["siconc"] == ("conc", 100.0, 0.0)

    @pytest.mark.parametrize("var", ["hfss", "hfls"])
    def test_turbulent_fluxes_sign_flipped_to_cmor(self, var):
        """ICON is down-positive; CMOR (and ICON r1) is up-positive."""
        assert _ATMOS2D[var][1] == -1.0

    @pytest.mark.parametrize("var", ["rlut", "rsut", "rsds", "rlds", "rsus", "rlus"])
    def test_radiation_components_pass_through(self, var):
        """Components are already CMOR-signed — no flip, no rescale."""
        assert _ATMOS2D[var] == (var, 1.0, 0.0)

    def test_sfcwind_case_is_remapped(self):
        assert _ATMOS2D["sfcWind"][0] == "sfcwind"

    @pytest.mark.parametrize("var", ["rss", "rls", "rst", "rlt",
                                     "rsscs", "rlscs", "rstcs", "rltcs"])
    def test_net_radiation_deliberately_absent(self, var):
        """Nets are not derived, so r2/r3 stay comparable with CMOR r1."""
        assert var not in _ATMOS2D and var not in _OCEAN2D

    @pytest.mark.parametrize("var", ["thetao", "so", "uo", "vo", "ta", "ua", "zg"])
    def test_three_d_fields_absent(self, var):
        """No usable 3-D archive for these members."""
        assert var not in _ATMOS2D and var not in _OCEAN2D


# ── Shape / dimension handling ───────────────────────────────────────────


class TestDimensions:
    def test_atmos_returns_time_lat_lon(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        assert loader.load_var(MODEL, "tas").dims == ("time", "lat", "lon")

    def test_ocean_returns_time_lat_lon(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        assert loader.load_var(MODEL, "tos").dims == ("time", "lat", "lon")

    @pytest.mark.parametrize("var", ["tas", "uas", "sfcWind", "hurs", "huss"])
    def test_singleton_height_dims_squeezed(self, tmp_path, monkeypatch, var):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, var)
        assert not {"height", "height_2", "height_3"} & set(da.dims)

    @pytest.mark.parametrize("var", ["tos", "sos", "siconc", "sithick"])
    def test_singleton_vertical_dims_squeezed(self, tmp_path, monkeypatch, var):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, var)
        assert not {"lev", "depth"} & set(da.dims)

    def test_variable_is_renamed_to_cmor(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        assert loader.load_var(MODEL, "sfcWind").name == "sfcWind"

    def test_grid_orientation_preserved(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "tas")
        np.testing.assert_array_equal(da.lat.values, LATS)
        np.testing.assert_array_equal(da.lon.values, LONS)


# ── Value transforms ─────────────────────────────────────────────────────


class TestTransforms:
    def test_clt_multiplied_by_100(self, tmp_path, monkeypatch):
        loader, stores = _make_loader(tmp_path, monkeypatch)
        got = loader.load_var(MODEL, "clt").values
        np.testing.assert_allclose(got, stores["atmos2d"]["clt"].values * 100.0)

    def test_hfss_sign_flipped(self, tmp_path, monkeypatch):
        loader, stores = _make_loader(tmp_path, monkeypatch)
        got = loader.load_var(MODEL, "hfss").values
        np.testing.assert_allclose(got, -stores["atmos2d"]["hfss"].values)

    def test_unscaled_variable_untouched(self, tmp_path, monkeypatch):
        loader, stores = _make_loader(tmp_path, monkeypatch)
        got = loader.load_var(MODEL, "psl").values
        np.testing.assert_allclose(got, stores["atmos2d"]["psl"].values)

    def test_land_sentinel_becomes_nan(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "tos")
        assert np.isnan(da.isel(time=0, lat=0, lon=0).item())

    def test_genuine_zero_survives(self, tmp_path, monkeypatch):
        """The stores declare fill_value 0.0; real zeros must not be masked."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "tos")
        assert da.isel(time=0, lat=0, lon=1).item() == 0.0

    def test_sentinel_masked_before_scaling(self, tmp_path, monkeypatch):
        """siconc is ×100 — a surviving sentinel would become -9e35."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "siconc")
        assert np.nanmin(da.values) >= 0.0

    def test_config_scale_factor_applied(self, tmp_path, monkeypatch):
        loader, stores = _make_loader(
            tmp_path, monkeypatch, scale_factors={"psl": 2.0},
        )
        got = loader.load_var(MODEL, "psl").values
        np.testing.assert_allclose(got, stores["atmos2d"]["psl"].values * 2.0)


# ── Selection / API ──────────────────────────────────────────────────────


class TestSelection:
    def test_period_slices_time(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "tas", period=("1980-01", "1980-03"))
        assert da.sizes["time"] == 3

    def test_time_mean_collapses_time(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var(MODEL, "tas", time_mean=True)
        assert "time" not in da.dims

    def test_table_kwarg_accepted_and_ignored(self, tmp_path, monkeypatch):
        """API parity with CMORLoader — CompositeModelLoader may pass it."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        a = loader.load_var(MODEL, "tas", table="Amon")
        b = loader.load_var(MODEL, "tas")
        np.testing.assert_array_equal(a.values, b.values)

    def test_load_coords_returns_lon_lat(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon, lat = loader.load_coords(MODEL, "tas")
        np.testing.assert_array_equal(lon, LONS)
        np.testing.assert_array_equal(lat, LATS)

    def test_unknown_variable_raises_keyerror(self, tmp_path, monkeypatch):
        """Diagnostics catch KeyError to skip a model lacking a variable."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        with pytest.raises(KeyError):
            loader.load_var(MODEL, "thetao")

    def test_variable_missing_from_store_raises_keyerror(self, tmp_path, monkeypatch):
        loader, stores = _make_loader(tmp_path, monkeypatch)
        stores["atmos2d"] = stores["atmos2d"].drop_vars("clt")
        with pytest.raises(KeyError):
            loader.load_var(MODEL, "clt")


# ── Path resolution ──────────────────────────────────────────────────────


class TestStorePath:
    def test_member_selects_subdirectory(self, tmp_path):
        cfg = _make_config(tmp_path, member=3)
        loader = ICONKerchunkLoader(cfg)
        root = tmp_path / "stores" / "3"
        root.mkdir(parents=True)
        (root / _STORE_FILES["atmos2d"]).touch()
        assert loader._store_path(MODEL, "atmos2d").parent.name == "3"

    def test_data_root_overrides_global_root(self, tmp_path):
        cfg = _make_config(tmp_path, member=2)
        override = tmp_path / "elsewhere"
        cfg.model_configs[MODEL].data_root = str(override)
        loader = ICONKerchunkLoader(cfg)
        (override / "2").mkdir(parents=True)
        (override / "2" / _STORE_FILES["ocean2d"]).touch()
        assert override in loader._store_path(MODEL, "ocean2d").parents

    def test_missing_store_raises_filenotfound(self, tmp_path):
        loader = ICONKerchunkLoader(_make_config(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader._store_path(MODEL, "atmos2d")

    def test_fill_threshold_below_any_physical_value(self):
        assert _FILL_THRESHOLD < -1e20
