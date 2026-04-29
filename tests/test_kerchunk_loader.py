"""Tests for KerchunkParquetLoader — parquet kerchunk reference stores."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.kerchunk_loader import (
    KerchunkParquetLoader,
    _ATMOS2D,
    _ATMOS3D,
    _DERIVED,
    _OCEAN2D,
)


# ── Grid constants ───────────────────────────────────────────────────────

# Full-resolution grid is 721×1440.  For tests, use a tiny 3×4 grid so the
# reshape logic is exercised without allocating large arrays.
N_LAT = 3
N_LON = 4
N_CELLS = N_LAT * N_LON  # 12


# ── Synthetic dataset builders ───────────────────────────────────────────


def _make_atmos_lat_lon():
    """Return flat lat/lon arrays matching the IFS kerchunk atmos layout.

    Real data: lat 90→-90 (north-first), lon 0..179.75 then -180..-0.25.
    Here we use a tiny 3×4 grid with the same ordering convention.
    """
    # lat descending (north-first): 60, 0, -60
    lats = np.array([60.0, 0.0, -60.0])
    # lon: positive half then negative (like IFS 0..179.75,-180..-0.25)
    lons = np.array([0.0, 90.0, -90.0, -0.1])
    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")   # (3,4)
    return lat_2d.ravel(), lon_2d.ravel()


def _make_atmos2d_store(n_months=6):
    """Synthetic atmos-2D Dataset mimicking a kerchunk zarr store."""
    time = xr.date_range("1980-01", periods=n_months, freq="MS")
    lat_flat, lon_flat = _make_atmos_lat_lon()
    data = np.ones((n_months, N_CELLS), dtype=np.float32) * 280.0

    ds_vars = {}
    for kname in set(v[0] for v in _ATMOS2D.values()):
        ds_vars[kname] = xr.DataArray(
            data.copy(),
            dims=["time", "value"],
            attrs={"units": "K"},
        )

    return xr.Dataset(
        ds_vars,
        coords={
            "time": time,
            "lat": ("value", lat_flat),
            "lon": ("value", lon_flat),
        },
    )


def _make_atmos3d_store(n_months=6, n_levels=3):
    """Synthetic atmos-3D Dataset (time, level, value)."""
    time = xr.date_range("1980-01", periods=n_months, freq="MS")
    lat_flat, lon_flat = _make_atmos_lat_lon()
    levels = np.array([500.0, 700.0, 850.0])[:n_levels]
    data = np.ones((n_months, n_levels, N_CELLS), dtype=np.float32) * 250.0

    ds_vars = {}
    for kname in set(v[0] for v in _ATMOS3D.values()):
        ds_vars[kname] = xr.DataArray(
            data.copy(),
            dims=["time", "level", "value"],
            attrs={"units": "K"},
        )

    return xr.Dataset(
        ds_vars,
        coords={
            "time": time,
            "level": levels,
            "lat": ("value", lat_flat),
            "lon": ("value", lon_flat),
        },
    )


def _make_ocean2d_store(n_days=30):
    """Synthetic ocean-2D Dataset (time=daily, lat, lon)."""
    time = xr.date_range("1980-01-01", periods=n_days, freq="D")
    lat_1d = np.linspace(-60.0, 60.0, N_LAT)
    lon_1d = np.linspace(0.0, 270.0, N_LON)

    ds_vars = {}
    for kname, (_, _scale, _offset) in _OCEAN2D.items():
        raw_kname = _OCEAN2D[kname][0]
        ds_vars[raw_kname] = xr.DataArray(
            np.full((n_days, 1, N_LAT, N_LON), 285.0, dtype=np.float32),
            dims=["time", "depth", "lat", "lon"],
            attrs={"units": "K"},
        )

    return xr.Dataset(
        ds_vars,
        coords={
            "time": time,
            "lat": lat_1d,
            "lon": lon_1d,
        },
    )


# ── Config helpers ───────────────────────────────────────────────────────


def _make_config(tmp_path, *, variant="r2i1p1f1", model="IFS-FESOM2-SR"):
    """Create a minimal FeatherConfig for kerchunk tests."""
    return FeatherConfig(
        model_catalogs={},
        models=[model],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "kerchunk_parquet", "root": str(tmp_path / "stores")},
        model_configs={
            model: ModelConfig(
                name=model,
                institution="AWI",
                variant=variant,
                grids={"sfc": "latlon", "o2d": "latlon"},
                color="#1f77b4",
            )
        },
    )


def _make_loader(tmp_path, monkeypatch, *, atmos2d=None, atmos3d=None, ocean2d=None,
                 variant="r2i1p1f1", model="IFS-FESOM2-SR"):
    """Create a KerchunkParquetLoader with _open_store monkeypatched."""
    config = _make_config(tmp_path, variant=variant, model=model)
    loader = KerchunkParquetLoader(config)

    stores = {
        "atmos2d": atmos2d if atmos2d is not None else _make_atmos2d_store(),
        "atmos3d": atmos3d if atmos3d is not None else _make_atmos3d_store(),
        "ocean2d": ocean2d if ocean2d is not None else _make_ocean2d_store(),
    }

    def fake_open_store(m, store_type):
        return stores[store_type]

    monkeypatch.setattr(loader, "_open_store", fake_open_store)
    return loader, config


# ── Grid reshape tests ───────────────────────────────────────────────────


class TestGridReshaping:
    def test_atmos_lat_ascending_after_sort(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon_1d, lat_1d = loader._get_atmos_grid("IFS-FESOM2-SR")
        assert lat_1d[0] < lat_1d[-1], "lat should be ascending after reshape"

    def test_atmos_lon_zero_to_360(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon_1d, lat_1d = loader._get_atmos_grid("IFS-FESOM2-SR")
        assert lon_1d.min() >= 0.0, "lon should be ≥0 after normalisation"
        assert lon_1d.max() < 360.0, "lon should be <360 after normalisation"

    def test_reshape_atmos_flat_2d(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        ds = _make_atmos2d_store(n_months=3)
        flat_data = np.ones((3, N_CELLS), dtype=np.float32)
        lat_1d, lon_1d, out = loader._reshape_atmos_flat(flat_data, ds)
        assert out.shape == (3, N_LAT, N_LON)
        assert lat_1d.shape == (N_LAT,)
        assert lon_1d.shape == (N_LON,)

    def test_reshape_atmos_flat_3d(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        ds = _make_atmos2d_store(n_months=3)
        flat_data = np.ones((3, 2, N_CELLS), dtype=np.float32)
        lat_1d, lon_1d, out = loader._reshape_atmos_flat(flat_data, ds, has_level=True)
        assert out.shape == (3, 2, N_LAT, N_LON)

    def test_get_atmos_grid_cached(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon1, lat1 = loader._get_atmos_grid("IFS-FESOM2-SR")
        lon2, lat2 = loader._get_atmos_grid("IFS-FESOM2-SR")
        assert lon1 is lon2, "should return cached arrays"


# ── Atmos 2-D loading ────────────────────────────────────────────────────


class TestAtmos2D:
    def test_load_tas_shape(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tas")
        assert da.dims == ("time", "lat", "lon")
        assert da.shape[1] == N_LAT
        assert da.shape[2] == N_LON

    def test_load_tas_name(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tas")
        assert da.name == "tas"

    def test_scale_factor_applied(self, tmp_path, monkeypatch):
        """clt has scale=100 (fraction→%); raw value 0.5 → 50."""
        store = _make_atmos2d_store()
        store["meantcc"].values[:] = 0.5
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "clt")
        assert float(da.mean()) == pytest.approx(50.0, abs=1e-3)

    def test_negative_scale_factor(self, tmp_path, monkeypatch):
        """hfss has scale=-1; raw 100 → -100 (sign flip)."""
        store = _make_atmos2d_store()
        store["msshf"].values[:] = 100.0
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "hfss")
        assert float(da.mean()) == pytest.approx(-100.0, abs=1e-3)

    def test_fill_value_masked(self, tmp_path, monkeypatch):
        """Values ≥9999 should become NaN."""
        store = _make_atmos2d_store()
        store["mean2t"].values[0, 0] = 9999.0
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "tas")
        assert np.isnan(da.values).any()

    def test_period_filter(self, tmp_path, monkeypatch):
        store = _make_atmos2d_store(n_months=24)
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "tas", period=("1980", "1980"))
        assert da.time.dt.year.max() <= 1980

    def test_time_mean(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tas", time_mean=True)
        assert "time" not in da.dims

    def test_coords_match_data(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tas")
        assert set(da.coords) >= {"lat", "lon", "time"}

    def test_all_atmos2d_variables_loadable(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        for var in _ATMOS2D:
            da = loader.load_var("IFS-FESOM2-SR", var)
            assert da.dims[0] == "time", f"{var} missing time dim"


# ── Atmos 3-D loading ────────────────────────────────────────────────────


class TestAtmos3D:
    def test_load_ta_shape(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "ta")
        assert da.dims == ("time", "level", "lat", "lon")
        assert da.shape[2] == N_LAT
        assert da.shape[3] == N_LON

    def test_level_coordinate_present(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "ua")
        assert "level" in da.coords

    def test_all_atmos3d_variables_loadable(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        for var in _ATMOS3D:
            da = loader.load_var("IFS-FESOM2-SR", var)
            assert "level" in da.dims, f"{var} missing level dim"


# ── Derived radiation variables ──────────────────────────────────────────


class TestDerivedVars:
    def _store_with_values(self, **kv):
        """Return atmos2d store with specific raw-field values set."""
        store = _make_atmos2d_store()
        for name, val in kv.items():
            store[name].values[:] = float(val)
        return store

    def test_rsut_derived_from_rsdt_and_rst(self, tmp_path, monkeypatch):
        """rsut = rsdt - rst = 400 - 300 = 100."""
        store = self._store_with_values(mtdwswrf=400.0, mtnswrf=300.0)
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "rsut")
        assert float(da.mean()) == pytest.approx(100.0, abs=1e-3)

    def test_rlut_is_negative_rlt(self, tmp_path, monkeypatch):
        """rlut = -rlt = -mtnlwrf; raw=50 → rlut=-50."""
        store = self._store_with_values(mtnlwrf=50.0)
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "rlut")
        assert float(da.mean()) == pytest.approx(-50.0, abs=1e-3)

    def test_rsus_derived(self, tmp_path, monkeypatch):
        """rsus = rsds - rss = msdwswrf - msnswrf = 300-200=100."""
        store = self._store_with_values(msdwswrf=300.0, msnswrf=200.0)
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "rsus")
        assert float(da.mean()) == pytest.approx(100.0, abs=1e-3)

    def test_rlus_derived(self, tmp_path, monkeypatch):
        """rlus = rlds - rls = msdwlwrf - msnlwrf = 250-200=50."""
        store = self._store_with_values(msdwlwrf=250.0, msnlwrf=200.0)
        loader, _ = _make_loader(tmp_path, monkeypatch, atmos2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "rlus")
        assert float(da.mean()) == pytest.approx(50.0, abs=1e-3)

    def test_derived_name_set(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "rsut")
        assert da.name == "rsut"

    def test_all_derived_variables_loadable(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        for var in _DERIVED:
            da = loader.load_var("IFS-FESOM2-SR", var)
            assert da.name == var


# ── Ocean 2-D loading ────────────────────────────────────────────────────


class TestOcean2D:
    def test_tos_resampled_to_monthly(self, tmp_path, monkeypatch):
        """Daily ocean store should be resampled to monthly means."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tos")
        assert da.dims[0] == "time"
        # 30 daily → 1 monthly
        assert da.sizes["time"] == 1

    def test_tos_offset_applied(self, tmp_path, monkeypatch):
        """tos: raw K - 273.15 → °C; raw=285 → 11.85."""
        store = _make_ocean2d_store()
        store["avg_tos"].values[:] = 285.0
        loader, _ = _make_loader(tmp_path, monkeypatch, ocean2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "tos")
        assert float(da.mean()) == pytest.approx(285.0 - 273.15, abs=1e-2)

    def test_singleton_depth_squeezed(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tos")
        assert "depth" not in da.dims

    def test_fill_values_masked(self, tmp_path, monkeypatch):
        """Values below _OCEAN_FILL_THRESHOLD should be masked."""
        from feather.data.kerchunk_loader import _OCEAN_FILL_THRESHOLD
        store = _make_ocean2d_store()
        # Mask all time steps at (lat=0, lon=0) so the monthly mean is NaN
        store["avg_tos"].values[:, 0, 0, 0] = _OCEAN_FILL_THRESHOLD - 1.0
        loader, _ = _make_loader(tmp_path, monkeypatch, ocean2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "tos")
        assert np.isnan(da.values).any()

    def test_siconc_no_offset(self, tmp_path, monkeypatch):
        """siconc has scale=1, offset=0; raw 0.5 stays 0.5."""
        store = _make_ocean2d_store()
        store["avg_siconc"].values[:] = 0.5
        loader, _ = _make_loader(tmp_path, monkeypatch, ocean2d=store)
        da = loader.load_var("IFS-FESOM2-SR", "siconc")
        assert float(da.mean()) == pytest.approx(0.5, abs=1e-3)

    def test_tos_name_set(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        da = loader.load_var("IFS-FESOM2-SR", "tos")
        assert da.name == "tos"

    def test_all_ocean2d_variables_loadable(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        for var in _OCEAN2D:
            da = loader.load_var("IFS-FESOM2-SR", var)
            assert da.dims[0] == "time", f"{var} missing time dim"


# ── load_coords ──────────────────────────────────────────────────────────


class TestLoadCoords:
    def test_sfc_coords_are_1d(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon, lat = loader.load_coords("IFS-FESOM2-SR", "tas")
        assert lon.ndim == 1
        assert lat.ndim == 1

    def test_sfc_lon_range(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon, _ = loader.load_coords("IFS-FESOM2-SR", "tas")
        assert lon.min() >= 0.0
        assert lon.max() < 360.0

    def test_ocean_coords_from_ocean_store(self, tmp_path, monkeypatch):
        """Ocean variables should return lat/lon from the ocean store."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon, lat = loader.load_coords("IFS-FESOM2-SR", "tos")
        assert len(lon) == N_LON
        assert len(lat) == N_LAT

    def test_unknown_var_falls_back_to_atmos(self, tmp_path, monkeypatch):
        """Unknown variable domain → falls back to atmos grid."""
        loader, _ = _make_loader(tmp_path, monkeypatch)
        lon, lat = loader.load_coords("IFS-FESOM2-SR", "nonexistent_var")
        assert lon.ndim == 1


# ── Error handling ───────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_variable_raises_key_error(self, tmp_path, monkeypatch):
        loader, _ = _make_loader(tmp_path, monkeypatch)
        with pytest.raises(KeyError, match="not supported"):
            loader.load_var("IFS-FESOM2-SR", "completely_unknown_var")

    def test_store_cache_populated(self, tmp_path, monkeypatch):
        """After the first load the store is cached; second call reuses it."""
        config = _make_config(tmp_path)
        loader = KerchunkParquetLoader(config)

        call_count = {"n": 0}
        fake_store = _make_atmos2d_store()

        def caching_open_store(model, store_type):
            call_count["n"] += 1
            key = (model, store_type)
            loader._store_cache[key] = fake_store
            return fake_store

        monkeypatch.setattr(loader, "_open_store", caching_open_store)
        loader.load_var("IFS-FESOM2-SR", "tas")
        assert ("IFS-FESOM2-SR", "atmos2d") in loader._store_cache

    def test_store_path_variant_fallback(self, tmp_path):
        """ModelConfig with no variant falls back to r1i1p1f1 in the path."""
        config = FeatherConfig(
            model_catalogs={},
            models=["M"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "out"),
            data_source={"type": "kerchunk_parquet", "root": str(tmp_path)},
            model_configs={
                "M": ModelConfig(
                    name="M",
                    variant="",          # empty → falls back to r1i1p1f1
                    grids={},
                    color="#000000",
                )
            },
        )
        loader = KerchunkParquetLoader(config)
        # _store_path raises FileNotFoundError but the path should use r1i1p1f1
        with pytest.raises(FileNotFoundError, match="r1i1p1f1"):
            loader._store_path("M", "atmos2d")

    def test_missing_store_raises_file_not_found(self, tmp_path):
        """_store_path should raise FileNotFoundError for non-existent stores."""
        config = _make_config(tmp_path)
        loader = KerchunkParquetLoader(config)
        with pytest.raises(FileNotFoundError):
            loader._store_path("IFS-FESOM2-SR", "atmos2d")

    def test_per_model_data_root_overrides_global(self, tmp_path):
        """ModelConfig.data_root should take precedence over global data_source.root."""
        custom_root = tmp_path / "custom_kerchunk_root"
        config = FeatherConfig(
            model_catalogs={},
            models=["M"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": False},
            dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "out"),
            data_source={"type": "cmor", "root": "/cmor/root"},
            model_configs={
                "M": ModelConfig(
                    name="M",
                    variant="r2i1p1f1",
                    data_root=str(custom_root),
                    grids={},
                    color="#000000",
                )
            },
        )
        loader = KerchunkParquetLoader(config)
        with pytest.raises(FileNotFoundError, match=str(custom_root)):
            loader._store_path("M", "atmos2d")


# ── Variable registry coverage ───────────────────────────────────────────


class TestVariableRegistry:
    def test_atmos2d_has_radiation_vars(self):
        assert "rsds" in _ATMOS2D
        assert "rlds" in _ATMOS2D
        assert "rsdt" in _ATMOS2D

    def test_atmos2d_has_heat_flux_vars(self):
        assert "hfss" in _ATMOS2D
        assert "hfls" in _ATMOS2D

    def test_atmos2d_heat_flux_sign_negative(self):
        assert _ATMOS2D["hfss"][1] < 0
        assert _ATMOS2D["hfls"][1] < 0

    def test_derived_keys_not_in_atmos2d(self):
        """Derived vars should not be in _ATMOS2D — they are computed."""
        for var in _DERIVED:
            assert var not in _ATMOS2D, f"{var} should not be in _ATMOS2D"

    def test_ocean2d_tos_has_kelvin_offset(self):
        _, scale, offset = _OCEAN2D["tos"]
        assert offset == pytest.approx(-273.15, abs=1e-6)
