"""Tests for the CMIP6Loader (feather.data.cmip6)."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.cmip6 import CMIP6Loader
from feather.data.variables import VARIABLE_REGISTRY


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _make_config(tmp_path, models=None, ensemble_mode="one_per_model",
                 regrid_resolution=1.0):
    """Build a FeatherConfig with a cmip6 section pointing at tmp_path."""
    if models is None:
        models = {
            "ModelA": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
            "ModelB": {"variants": ["r1i1p1f1"]},
        }
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={
            "enabled": True,
            "catalog_path": str(tmp_path / "fake_catalog.yaml"),
            "regrid_resolution": regrid_resolution,
            "influence_radius": 1_000_000,
            "ensemble_mode": ensemble_mode,
            "models": models,
        },
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


def _write_zarr(tmp_path, model, variant, table, var, ds):
    """Write a synthetic dataset to the expected zarr path."""
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir(exist_ok=True)
    zarr_path = zarr_dir / f"{model}_historical_{variant}_{table}_{var}.zarr"
    ds.to_zarr(str(zarr_path), mode="w")
    return str(zarr_path)


def _synth_cmip6_ds(var_name="tas", lats=None, lons=None, n_months=12,
                     base_temp=290.0):
    """Create a synthetic CMIP6-like dataset."""
    if lats is None:
        lats = np.arange(-87.5, 90, 5.0)
    if lons is None:
        lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=n_months, freq="MS")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = base_temp - 30 * np.abs(lat_grid / 90.0)
    seasonal = 3 * np.sin(2 * np.pi * (np.arange(n_months) - 3) / 12)
    data = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    return xr.Dataset({
        var_name: xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        ),
    })


# ═════════════════════════════════════════════════════════════════════
# TestCMIP6LoaderInit
# ═════════════════════════════════════════════════════════════════════


class TestCMIP6LoaderInit:
    """Constructor, models property, zarr_dir derivation, empty config."""

    def test_init_basic(self, cmip6_config):
        loader = CMIP6Loader(cmip6_config)
        assert loader._cfg is cmip6_config.cmip6
        assert isinstance(loader.models, dict)

    def test_models_property(self, cmip6_config):
        loader = CMIP6Loader(cmip6_config)
        assert "ModelA" in loader.models
        assert "ModelB" in loader.models

    def test_zarr_dir_fallback(self, tmp_path):
        """When catalog can't be parsed, zarr_dir falls back to sibling dir."""
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        expected = str(tmp_path / "zarr")
        assert loader.zarr_dir == expected

    def test_empty_models(self, tmp_path):
        config = _make_config(tmp_path, models={})
        loader = CMIP6Loader(config)
        assert loader.models == {}


# ═════════════════════════════════════════════════════════════════════
# TestVariantHandling
# ═════════════════════════════════════════════════════════════════════


class TestVariantHandling:
    """_get_variants() with list, legacy str, and empty."""

    def test_variants_list(self):
        cfg = {"variants": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]}
        assert CMIP6Loader._get_variants(cfg) == ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]

    def test_variant_legacy_str(self):
        cfg = {"variant": "r10i1p1f1"}
        assert CMIP6Loader._get_variants(cfg) == ["r10i1p1f1"]

    def test_empty_config(self):
        assert CMIP6Loader._get_variants({}) == []


# ═════════════════════════════════════════════════════════════════════
# TestMemberPairs
# ═════════════════════════════════════════════════════════════════════


class TestMemberPairs:
    """_get_member_pairs() in one_per_model vs all_members mode."""

    def test_one_per_model(self, cmip6_config):
        loader = CMIP6Loader(cmip6_config)
        pairs = loader._get_member_pairs("one_per_model")
        # Two models, one variant each
        assert len(pairs) == 2
        models = [m for m, v in pairs]
        assert "ModelA" in models
        assert "ModelB" in models
        # Only first variant for ModelA
        model_a_pair = [(m, v) for m, v in pairs if m == "ModelA"][0]
        assert model_a_pair[1] == "r1i1p1f1"

    def test_all_members(self, cmip6_config):
        loader = CMIP6Loader(cmip6_config)
        pairs = loader._get_member_pairs("all_members")
        # ModelA has 2 variants, ModelB has 2 → 4 total
        assert len(pairs) == 4

    def test_config_default(self, tmp_path):
        """Uses config ensemble_mode when no override given."""
        config = _make_config(tmp_path, ensemble_mode="all_members")
        loader = CMIP6Loader(config)
        pairs = loader._get_member_pairs()  # No override
        # all_members: ModelA(2) + ModelB(1) = 3
        assert len(pairs) == 3


# ═════════════════════════════════════════════════════════════════════
# TestZarrPaths
# ═════════════════════════════════════════════════════════════════════


class TestZarrPaths:
    """Path construction for variables, atmos areas, ocean areas."""

    def test_variable_path(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        path = loader._zarr_path("MIROC6", "r1i1p1f1", "Amon", "tas")
        assert path.endswith("MIROC6_historical_r1i1p1f1_Amon_tas.zarr")

    def test_area_path_atmos(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        path = loader._area_zarr_path("MIROC6", "r1i1p1f1", "Amon")
        assert "fx_areacella" in path

    def test_area_path_ocean(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        path = loader._area_zarr_path("MIROC6", "r1i1p1f1", "Omon")
        assert "Ofx_areacello" in path


# ═════════════════════════════════════════════════════════════════════
# TestTimeNormalization
# ═════════════════════════════════════════════════════════════════════


class TestTimeNormalization:
    """Standard calendar, cftime (noleap), no-time-dim, empty time."""

    def test_standard_calendar(self):
        time = xr.date_range("2000-01-15", periods=3, freq="MS")
        da = xr.DataArray([1, 2, 3], dims="time", coords={"time": time})
        result = CMIP6Loader._normalize_time(da)
        assert result is not None
        # All should be first-of-month
        for t in result.time.values:
            assert str(t).endswith("-01T00:00:00.000000000") or str(t)[8:10] == "01"

    def test_cftime_noleap(self):
        import cftime
        times = [cftime.DatetimeNoLeap(2000, m, 15) for m in range(1, 4)]
        da = xr.DataArray([1, 2, 3], dims="time", coords={"time": times})
        result = CMIP6Loader._normalize_time(da)
        assert result is not None
        assert len(result.time) == 3

    def test_no_time_dim(self):
        da = xr.DataArray([1, 2, 3], dims="x")
        result = CMIP6Loader._normalize_time(da)
        # Should return da unchanged
        assert result is not None
        assert "time" not in result.dims

    def test_empty_time(self):
        da = xr.DataArray(
            np.array([]).reshape(0, 3),
            dims=("time", "x"),
            coords={"time": []},
        )
        result = CMIP6Loader._normalize_time(da)
        assert result is None


# ═════════════════════════════════════════════════════════════════════
# TestFindLatLon
# ═════════════════════════════════════════════════════════════════════


class TestFindLatLon:
    """lat/lon names, latitude/longitude names, missing coords."""

    def test_lat_lon(self):
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=("lat", "lon"),
            coords={"lat": [10, 20, 30], "lon": [0, 1, 2, 3]},
        )
        lat, lon = CMIP6Loader._find_lat_lon(da)
        np.testing.assert_array_equal(lat, [10, 20, 30])
        np.testing.assert_array_equal(lon, [0, 1, 2, 3])

    def test_latitude_longitude(self):
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=("latitude", "longitude"),
            coords={"latitude": [10, 20, 30], "longitude": [0, 1, 2, 3]},
        )
        lat, lon = CMIP6Loader._find_lat_lon(da)
        assert len(lat) == 3

    def test_missing_coords(self):
        da = xr.DataArray(np.zeros((3, 4)), dims=("x", "y"))
        with pytest.raises(ValueError, match="Cannot find lat/lon"):
            CMIP6Loader._find_lat_lon(da)


# ═════════════════════════════════════════════════════════════════════
# TestSiconcNormalization
# ═════════════════════════════════════════════════════════════════════


class TestSiconcNormalization:
    """Percentage → fraction, fraction unchanged."""

    def test_percentage_to_fraction(self):
        da = xr.DataArray([0, 50, 100])
        result = CMIP6Loader._normalise_siconc(da)
        np.testing.assert_allclose(result.values, [0, 0.5, 1.0])

    def test_fraction_unchanged(self):
        da = xr.DataArray([0, 0.5, 1.0])
        result = CMIP6Loader._normalise_siconc(da)
        np.testing.assert_allclose(result.values, [0, 0.5, 1.0])


# ═════════════════════════════════════════════════════════════════════
# TestTableInference
# ═════════════════════════════════════════════════════════════════════


class TestTableInference:
    """tas→Amon, tos→Omon, siconc→SImon, unknown→Amon."""

    def test_tas(self):
        assert CMIP6Loader._infer_table("tas") == "Amon"

    def test_tos(self):
        assert CMIP6Loader._infer_table("tos") == "Omon"

    def test_siconc(self):
        assert CMIP6Loader._infer_table("siconc") == "SImon"

    def test_unknown(self):
        assert CMIP6Loader._infer_table("xyzzy_nonexistent") == "Amon"


# ═════════════════════════════════════════════════════════════════════
# TestLoadVar
# ═════════════════════════════════════════════════════════════════════


class TestLoadVar:
    """Missing zarr, synthetic zarr read, period filtering, siconc norm."""

    def test_missing_zarr_returns_none(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        # Manually set zarr_dir to tmp
        loader._zarr_dir = str(tmp_path / "zarr")
        result = loader.load_var("tas", "ModelA", variant="r1i1p1f1")
        assert result is None

    def test_synthetic_zarr_read(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)

        result = loader.load_var("tas", "ModelA", variant="r1i1p1f1", table="Amon")
        assert result is not None
        assert "time" not in result.dims  # Should be time-averaged
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_period_filtering(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas", n_months=24)
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)

        # Load only first 6 months
        result = loader.load_var(
            "tas", "ModelA",
            variant="r1i1p1f1", table="Amon",
            period=("1990-01", "1990-06"),
        )
        assert result is not None

    def test_siconc_normalization(self, tmp_path):
        """Sea ice in percentage (0-100) gets divided by 100."""
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1990-01", periods=12, freq="MS")
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        # Sea ice: 80% in polar regions
        siconc_data = np.where(
            np.abs(lat_grid) > 60,
            80.0,
            0.0,
        )[np.newaxis, :, :] * np.ones((12, 1, 1))

        ds = xr.Dataset({
            "siconc": xr.DataArray(
                siconc_data, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        })
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "SImon", "siconc", ds)

        result = loader.load_var(
            "siconc", "ModelA",
            variant="r1i1p1f1", table="SImon",
        )
        assert result is not None
        # Max should be ~0.8 (divided by 100)
        assert float(result.max()) < 1.1


# ═════════════════════════════════════════════════════════════════════
# TestLoadVarForModelVar
# ═════════════════════════════════════════════════════════════════════


class TestLoadVarForModelVar:
    """avg_2t→tas mapping, unmapped var returns None."""

    def test_avg_2t_maps_to_tas(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)

        result = loader.load_var_for_model_var(
            "avg_2t", "ModelA", variant="r1i1p1f1",
        )
        assert result is not None

    def test_unmapped_var_returns_none(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        # avg_10ws has no cmip6_variable
        result = loader.load_var_for_model_var(
            "avg_10ws", "ModelA", variant="r1i1p1f1",
        )
        assert result is None


# ═════════════════════════════════════════════════════════════════════
# TestMultiModelMean
# ═════════════════════════════════════════════════════════════════════


class TestMultiModelMean:
    """Two models one_per_model, all_members, skip missing, per-call override, no data."""

    def _setup_two_models(self, tmp_path, base_temps=(290, 300)):
        """Write zarr for two models, return config and loader."""
        config = _make_config(
            tmp_path,
            models={
                "ModelA": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
                "ModelB": {"variants": ["r1i1p1f1"]},
            },
        )
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds_a = _synth_cmip6_ds("tas", base_temp=base_temps[0])
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds_a)
        _write_zarr(tmp_path, "ModelA", "r2i1p1f1", "Amon", "tas", ds_a)

        ds_b = _synth_cmip6_ds("tas", base_temp=base_temps[1])
        _write_zarr(tmp_path, "ModelB", "r1i1p1f1", "Amon", "tas", ds_b)

        return config, loader

    def test_one_per_model(self, tmp_path):
        _, loader = self._setup_two_models(tmp_path, base_temps=(280, 300))
        mmm, info = loader.load_multi_model_mean(
            "tas", table="Amon", ensemble_mode="one_per_model",
        )
        assert mmm is not None
        assert info["n_members"] == 2
        assert len(info["models_used"]) == 2
        # MMM should be roughly average of 280 and 300 at equator
        equator_val = float(mmm.sel(lat=0.5, method="nearest").mean())
        assert 270 < equator_val < 310

    def test_all_members(self, tmp_path):
        _, loader = self._setup_two_models(tmp_path, base_temps=(280, 300))
        mmm, info = loader.load_multi_model_mean(
            "tas", table="Amon", ensemble_mode="all_members",
        )
        assert mmm is not None
        # all_members: ModelA has 2 variants, ModelB has 1 → 3 total
        assert info["n_members"] == 3

    def test_skip_missing(self, tmp_path):
        """Models without zarr are skipped, not errors."""
        config = _make_config(tmp_path, models={
            "ModelA": {"variants": ["r1i1p1f1"]},
            "ModelMissing": {"variants": ["r1i1p1f1"]},
        })
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)
        # Don't write zarr for ModelMissing

        mmm, info = loader.load_multi_model_mean(
            "tas", table="Amon", ensemble_mode="one_per_model",
        )
        assert mmm is not None
        assert info["n_members"] == 1
        assert "ModelMissing/r1i1p1f1" in info["models_skipped"]

    def test_per_call_override(self, tmp_path):
        """ensemble_mode kwarg overrides config default."""
        config = _make_config(tmp_path, ensemble_mode="one_per_model")
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)
        _write_zarr(tmp_path, "ModelA", "r2i1p1f1", "Amon", "tas", ds)
        _write_zarr(tmp_path, "ModelB", "r1i1p1f1", "Amon", "tas", ds)

        # Config default is one_per_model, but override to all_members
        mmm, info = loader.load_multi_model_mean(
            "tas", table="Amon", ensemble_mode="all_members",
        )
        assert info["n_members"] == 3  # 2 from ModelA + 1 from ModelB

    def test_no_data_returns_none(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        mmm, info = loader.load_multi_model_mean("tas", table="Amon")
        assert mmm is None
        assert info["n_members"] == 0


# ═════════════════════════════════════════════════════════════════════
# TestAreaWeights
# ═════════════════════════════════════════════════════════════════════


class TestAreaWeights:
    """Load areacella, missing → None, caching."""

    def test_load_areacella(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        area = np.cos(np.deg2rad(lat_grid)) * 1e10

        ds = xr.Dataset({
            "areacella": xr.DataArray(
                area, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        })
        # Write to fx path
        zarr_dir = tmp_path / "zarr"
        zarr_dir.mkdir(exist_ok=True)
        ds.to_zarr(str(zarr_dir / "ModelA_historical_r1i1p1f1_fx_areacella.zarr"))

        result = loader.load_area("ModelA", variant="r1i1p1f1", table="Amon")
        assert result is not None
        assert "lat" in result.dims

    def test_missing_returns_none(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        result = loader.load_area("ModelA", variant="r1i1p1f1")
        assert result is None

    def test_caching(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        # First call caches None
        result1 = loader.load_area("ModelA", variant="r1i1p1f1")
        assert result1 is None
        assert "ModelA_r1i1p1f1_Amon" in loader._area_cache

        # Second call returns cached value
        result2 = loader.load_area("ModelA", variant="r1i1p1f1")
        assert result2 is None


# ═════════════════════════════════════════════════════════════════════
# TestAvailableModels
# ═════════════════════════════════════════════════════════════════════


class TestAvailableModels:
    """Available models, available members, model_var convenience."""

    def test_available_models(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)
        # ModelB has no zarr

        models = loader.available_models("tas", table="Amon")
        assert "ModelA" in models
        assert "ModelB" not in models

    def test_available_members(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)
        _write_zarr(tmp_path, "ModelA", "r2i1p1f1", "Amon", "tas", ds)

        members = loader.available_members("tas", table="Amon")
        assert ("ModelA", "r1i1p1f1") in members
        assert ("ModelA", "r2i1p1f1") in members

    def test_model_var_convenience(self, tmp_path):
        config = _make_config(tmp_path)
        loader = CMIP6Loader(config)
        loader._zarr_dir = str(tmp_path / "zarr")

        ds = _synth_cmip6_ds("tas")
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas", ds)

        models = loader.available_models_for_model_var("avg_2t")
        assert "ModelA" in models


# ═════════════════════════════════════════════════════════════════════
# TestMockCMIP6Loader
# ═════════════════════════════════════════════════════════════════════


class TestMockCMIP6Loader:
    """Verify the mock follows the real API."""

    def test_load_var(self, mock_cmip6_loader):
        da = mock_cmip6_loader.load_var("tas", "ModelA")
        assert da is not None
        assert "time" not in da.dims

    def test_load_var_missing(self, mock_cmip6_loader):
        result = mock_cmip6_loader.load_var("nonexistent", "ModelA")
        assert result is None

    def test_load_mmm(self, mock_cmip6_loader):
        mmm, info = mock_cmip6_loader.load_multi_model_mean("tas")
        assert mmm is not None
        assert info["n_members"] > 0

    def test_available_models(self, mock_cmip6_loader):
        models = mock_cmip6_loader.available_models("tas")
        assert len(models) == 2


# ═════════════════════════════════════════════════════════════════════
# TestCMIP6Integration — requires real Levante data
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestCMIP6Integration:
    """Integration tests using real CMIP6 zarr data on Levante.

    These only open metadata / small slices and are safe on the login node.
    """

    CATALOG_PATH = "/home/a/a270088/PYTHON/DestinE/cmip6/cmip6_zarr_catalog.yaml"

    @pytest.fixture
    def real_config(self, tmp_path):
        return FeatherConfig(
            model_catalogs={},
            models=["ifs-fesom"],
            obs_root="",
            obs_datasets={},
            cmip6={
                "enabled": True,
                "catalog_path": self.CATALOG_PATH,
                "regrid_resolution": 1.0,
                "influence_radius": 80_000,
                "ensemble_mode": "one_per_model",
                "models": {
                    "MIROC6": {"variants": ["r1i1p1f1"]},
                },
            },
            dask={},
            nereus={"influence_radius": 80_000},
            output_dir=str(tmp_path / "output"),
        )

    def test_real_miroc6_tas(self, real_config):
        """Load MIROC6 tas for 1990-2000 period."""
        if not os.path.exists(self.CATALOG_PATH):
            pytest.skip("CMIP6 catalog not available")

        loader = CMIP6Loader(real_config)
        da = loader.load_var(
            "tas", "MIROC6",
            variant="r1i1p1f1", table="Amon",
            period=("1990-01", "2000-12"),
        )
        if da is None:
            pytest.skip("MIROC6 tas zarr not found")

        assert da is not None
        # Global mean temperature should be ~280-300 K
        gmean = float(da.mean())
        assert 250 < gmean < 310, f"Suspicious global mean: {gmean}"

    def test_available_models(self, real_config):
        """Check which models have tas data."""
        if not os.path.exists(self.CATALOG_PATH):
            pytest.skip("CMIP6 catalog not available")

        loader = CMIP6Loader(real_config)
        models = loader.available_models("tas", table="Amon")
        # At minimum MIROC6 should be available
        assert isinstance(models, list)
