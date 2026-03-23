"""Tests for CompositeModelLoader — multi-data-source routing."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.composite_loader import CompositeModelLoader, _DestinECatalogAdapter


# ── Helpers ────────────────────────────────────────────────────────────

SMALL_CELLS = 768  # nside=8


def _make_multi_source_config(
    tmp_path, *, grib_models=None, catalog_models=None,
):
    """Build a multi-source FeatherConfig for testing."""
    models = []
    model_configs = {}

    for name in (catalog_models or []):
        models.append(name)
        model_configs[name] = ModelConfig(
            name=name,
            experiment="baseline_hist",
            catalog_key="ifs-fesom",
            grids={"sfc": "healpix", "o2d": "healpix"},
            color="#2ca02c",
        )

    for name in (grib_models or []):
        models.append(name)
        model_configs[name] = ModelConfig(
            name=name,
            experiment="hist",
            data_source_type="grib_healpix",
            data_root=str(tmp_path / "grib"),
            grids={"sfc": "healpix", "o2d": "healpix"},
            color="#1f77b4",
            variable_aliases={"tas": "sfc_mean2t"},
        )

    return FeatherConfig(
        model_catalogs={"2d": str(tmp_path / "cat.yaml")},
        models=models,
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "destine_catalog"},
        model_configs=model_configs,
    )


def _make_synth_da(ncells=SMALL_CELLS, n_months=12):
    """Create a synthetic HEALPix DataArray."""
    lon = np.linspace(0, 360, ncells, endpoint=False)
    lat = np.linspace(-90, 90, ncells)
    time = xr.date_range("1990-01", periods=n_months, freq="MS")
    data = np.random.default_rng(42).normal(280, 10, (n_months, ncells))
    return xr.DataArray(
        data,
        dims=["time", "values"],
        coords={
            "time": time,
            "latitude": ("values", lat),
            "longitude": ("values", lon),
        },
    )


# ── Fake backends for testing ─────────────────────────────────────────


class _FakeGRIBLoader:
    """Minimal mock of GRIBLoader."""

    def __init__(self, config):
        self._da = _make_synth_da()

    def load_var(self, model, variable, *, period=None, time_mean=False):
        da = self._da
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da


class _FakeCatalogAdapter:
    """Minimal mock of _DestinECatalogAdapter."""

    def __init__(self, config, catalog_loader):
        self._da = _make_synth_da()

    def load_var(self, model, variable, *, period=None, time_mean=False):
        da = self._da
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(self, model, variable):
        return (
            np.asarray(self._da["longitude"]),
            np.asarray(self._da["latitude"]),
        )


# ── Tests: config detection ───────────────────────────────────────────


class TestMultiSourceDetection:
    """Test that is_multi_source() correctly identifies mixed configs."""

    def test_mixed_config_detected(self, tmp_path):
        cfg = _make_multi_source_config(
            tmp_path,
            catalog_models=["TCO2999"],
            grib_models=["TCO399"],
        )
        assert cfg.is_multi_source() is True

    def test_single_source_not_detected(self, tmp_path):
        cfg = _make_multi_source_config(
            tmp_path,
            grib_models=["TCO399", "TCO319"],
        )
        assert cfg.is_multi_source() is False

    def test_model_data_source_types(self, tmp_path):
        cfg = _make_multi_source_config(
            tmp_path,
            catalog_models=["TCO2999"],
            grib_models=["TCO399"],
        )
        assert cfg.get_model_data_source_type("TCO2999") == "destine_catalog"
        assert cfg.get_model_data_source_type("TCO399") == "grib_healpix"


# ── Tests: CompositeModelLoader routing ───────────────────────────────


class TestCompositeRouting:
    """Test that CompositeModelLoader routes to the correct backend."""

    @pytest.fixture
    def loader(self, tmp_path, monkeypatch):
        """Create a CompositeModelLoader with fake backends."""
        cfg = _make_multi_source_config(
            tmp_path,
            catalog_models=["TCO2999"],
            grib_models=["TCO399", "TCO319"],
        )

        # Patch _create_backend to return our fakes
        def fake_create_backend(self_loader, src_type):
            if src_type == "grib_healpix":
                return _FakeGRIBLoader(cfg)
            if src_type == "destine_catalog":
                return _FakeCatalogAdapter(cfg, None)
            raise ValueError(f"Unknown: {src_type}")

        monkeypatch.setattr(
            CompositeModelLoader, "_create_backend", fake_create_backend,
        )
        return CompositeModelLoader(cfg)

    def test_has_all_models(self, loader):
        assert "TCO2999" in loader._backends
        assert "TCO399" in loader._backends
        assert "TCO319" in loader._backends

    def test_load_var_grib(self, loader):
        da = loader.load_var("TCO399", "tas")
        assert "time" in da.dims
        assert "values" in da.dims

    def test_load_var_catalog(self, loader):
        da = loader.load_var("TCO2999", "tas")
        assert "time" in da.dims
        assert "values" in da.dims

    def test_load_var_period(self, loader):
        da = loader.load_var("TCO399", "tas", period=("1990", "1990"))
        assert da.sizes["time"] == 12

    def test_load_var_time_mean(self, loader):
        da = loader.load_var("TCO399", "tas", time_mean=True)
        assert "time" not in da.dims

    def test_unknown_model_raises(self, loader):
        with pytest.raises(KeyError, match="not configured"):
            loader.load_var("UNKNOWN", "tas")

    def test_load_coords_catalog(self, loader):
        lon, lat = loader.load_coords("TCO2999", "tas")
        assert len(lon) == SMALL_CELLS
        assert len(lat) == SMALL_CELLS

    def test_load_coords_grib(self, loader):
        lon, lat = loader.load_coords("TCO399", "tas")
        assert len(lon) == SMALL_CELLS
        assert len(lat) == SMALL_CELLS


# ── Tests: run.py dispatch ────────────────────────────────────────────


class TestRunDispatch:
    """Test that _create_model_loader returns CompositeModelLoader."""

    def test_multi_source_dispatch(self, tmp_path, monkeypatch):
        """is_multi_source() → CompositeModelLoader."""
        from feather.run import _create_model_loader

        cfg = _make_multi_source_config(
            tmp_path,
            catalog_models=["TCO2999"],
            grib_models=["TCO399"],
        )

        # Patch to avoid real intake/GRIB loading
        def fake_create_backend(self_loader, src_type):
            if src_type == "grib_healpix":
                return _FakeGRIBLoader(cfg)
            if src_type == "destine_catalog":
                return _FakeCatalogAdapter(cfg, None)
            raise ValueError(f"Unknown: {src_type}")

        monkeypatch.setattr(
            CompositeModelLoader, "_create_backend", fake_create_backend,
        )

        loader = _create_model_loader(cfg)
        assert isinstance(loader, CompositeModelLoader)

    def test_single_source_no_composite(self, tmp_path):
        """Single-source config does NOT create CompositeModelLoader."""
        from feather.data.grib_loader import GRIBLoader
        from feather.run import _create_model_loader

        cfg = _make_multi_source_config(
            tmp_path,
            grib_models=["TCO399", "TCO319"],
        )
        # Override global type to grib
        cfg.data_source = {"type": "grib_healpix"}

        loader = _create_model_loader(cfg)
        assert isinstance(loader, GRIBLoader)


# ── Tests: base.py integration ────────────────────────────────────────


class TestBaseClassIntegration:
    """Test that DiagnosticBase dispatches through CompositeModelLoader."""

    @pytest.fixture
    def diag(self, tmp_path, monkeypatch):
        """Create a minimal diagnostic with CompositeModelLoader."""
        from feather.data.obs import ObsLoader
        from feather.diag.base import DiagnosticBase

        cfg = _make_multi_source_config(
            tmp_path,
            catalog_models=["TCO2999"],
            grib_models=["TCO399"],
        )

        def fake_create_backend(self_loader, src_type):
            if src_type == "grib_healpix":
                return _FakeGRIBLoader(cfg)
            if src_type == "destine_catalog":
                return _FakeCatalogAdapter(cfg, None)
            raise ValueError(f"Unknown: {src_type}")

        monkeypatch.setattr(
            CompositeModelLoader, "_create_backend", fake_create_backend,
        )

        loader = CompositeModelLoader(cfg)

        class _TestDiag(DiagnosticBase):
            name = "test"
            title = "Test"
            domain = "sfc"
            variables = ["tas"]
            group = "test"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        return _TestDiag(loader, ObsLoader(cfg), cfg)

    def test_load_model_var_grib(self, diag):
        da = diag._load_model_var("TCO399", "tas")
        assert "time" in da.dims

    def test_load_model_var_catalog(self, diag):
        da = diag._load_model_var("TCO2999", "tas")
        assert "time" in da.dims

    def test_load_model_coords_grib(self, diag):
        lon, lat = diag._load_model_coords("TCO399", "tas")
        assert len(lon) == SMALL_CELLS

    def test_load_model_coords_catalog(self, diag):
        lon, lat = diag._load_model_coords("TCO2999", "tas")
        assert len(lon) == SMALL_CELLS


# ── Tests: combined config YAML ───────────────────────────────────────


class TestCombinedConfig:
    """Test loading the combined YAML config."""

    def test_load_combined_config(self):
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "ifs_fesom_combined.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Combined config not found")

        cfg = FeatherConfig.from_yaml(config_path)

        # Models
        assert len(cfg.models) == 3
        assert "IFS-FESOM-TCO2999" in cfg.models
        assert "IFS-FESOM-TCO399" in cfg.models
        assert "IFS-FESOM-TCO319" in cfg.models

        # Multi-source detection
        assert cfg.is_multi_source() is True
        assert cfg.get_model_data_source_type("IFS-FESOM-TCO2999") == "destine_catalog"
        assert cfg.get_model_data_source_type("IFS-FESOM-TCO399") == "grib_healpix"
        assert cfg.get_model_data_source_type("IFS-FESOM-TCO319") == "grib_healpix"

        # Per-model fields
        mc2999 = cfg.model_configs["IFS-FESOM-TCO2999"]
        assert mc2999.catalog_key == "ifs-fesom"
        assert mc2999.data_source_type == ""  # inherits global

        mc399 = cfg.model_configs["IFS-FESOM-TCO399"]
        assert mc399.data_source_type == "grib_healpix"
        assert mc399.data_root != ""
        assert mc399.variable_aliases["tas"] == "sfc_mean2t"

        # Per-model depth levels
        depth_levels = cfg.ocean_3d["depth_levels"]
        assert isinstance(depth_levels, dict)
        assert len(depth_levels["IFS-FESOM-TCO2999"]) == 69
        assert len(depth_levels["IFS-FESOM-TCO399"]) == 28
        assert len(depth_levels["IFS-FESOM-TCO319"]) == 28

        # Global settings
        assert cfg.get_period() == ("1990", "2014")
        assert cfg.get_experiment() == "baseline_hist"
