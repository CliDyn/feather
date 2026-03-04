"""Tests for the diagnostic base class and registry."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.config import FeatherConfig
from feather.diag.base import DiagnosticBase
from feather.diag.registry import (
    _REGISTRY,
    get_diagnostic,
    list_diagnostics,
    register,
    registered_names,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal FeatherConfig for testing."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before and after each test."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


# ── Concrete diagnostic for testing ──────────────────────────────────


class _MockDiagnostic(DiagnosticBase):
    """Concrete diagnostic for testing."""

    name = "mock_test"
    title = "Mock Test Diagnostic"
    domain = "sfc"
    variables = ["avg_2t"]
    group = "temperature"

    def compute(self) -> dict[str, Any]:
        return {"values": np.array([1.0, 2.0, 3.0])}

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        fig, ax = plt.subplots()
        ax.plot(results["values"])

        meta = self._build_metadata(
            title="Mock Plot",
            figure_id="mock_test_fig",
            models=["ifs-fesom"],
            description="A mock figure for testing.",
            plot_type="timeseries",
        )
        return [(fig, meta)]


# ── DiagnosticBase tests ─────────────────────────────────────────────


class TestDiagnosticBase:
    """Tests for DiagnosticBase."""

    def test_output_dir(self, minimal_config, tmp_path):
        """Output directory is derived from config + diagnostic name."""
        diag = _MockDiagnostic(None, None, minimal_config)
        expected = Path(str(tmp_path / "output")) / "figures" / "mock_test"
        assert diag.output_dir == expected

    def test_cmip6_disabled_by_default(self, minimal_config):
        """CMIP6 is disabled when no loader is provided."""
        diag = _MockDiagnostic(None, None, minimal_config)
        assert not diag.cmip6_enabled

    def test_cmip6_enabled_with_loader(self, minimal_config):
        """CMIP6 is enabled when loader is provided and config says enabled."""
        minimal_config.cmip6 = {"enabled": True}
        diag = _MockDiagnostic(
            None, None, minimal_config, cmip6_loader="fake"
        )
        assert diag.cmip6_enabled

    def test_cmip6_disabled_without_config(self, minimal_config):
        """CMIP6 is disabled even with loader if config says disabled."""
        minimal_config.cmip6 = {"enabled": False}
        diag = _MockDiagnostic(
            None, None, minimal_config, cmip6_loader="fake"
        )
        assert not diag.cmip6_enabled

    def test_compute_returns_dict(self, minimal_config):
        """compute() returns a dict of results."""
        diag = _MockDiagnostic(None, None, minimal_config)
        results = diag.compute()
        assert isinstance(results, dict)
        assert "values" in results

    def test_plot_returns_fig_meta_pairs(self, minimal_config):
        """plot() returns a list of (Figure, dict) pairs."""
        diag = _MockDiagnostic(None, None, minimal_config)
        results = diag.compute()
        pairs = diag.plot(results)

        assert len(pairs) == 1
        fig, meta = pairs[0]
        assert isinstance(fig, plt.Figure)
        assert isinstance(meta, dict)
        assert meta["figure_id"] == "mock_test_fig"
        plt.close(fig)

    def test_run_saves_files(self, minimal_config, tmp_path):
        """run() saves PNG + JSON and returns paths."""
        diag = _MockDiagnostic(None, None, minimal_config)
        saved = diag.run()

        assert len(saved) == 1
        png_path, json_path = saved[0]
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.suffix == ".png"
        assert json_path.suffix == ".json"

    def test_run_metadata_content(self, minimal_config, tmp_path):
        """Saved metadata JSON has correct content."""
        diag = _MockDiagnostic(None, None, minimal_config)
        saved = diag.run()
        _, json_path = saved[0]

        with open(json_path) as f:
            meta = json.load(f)

        assert meta["diagnostic_name"] == "mock_test"
        assert meta["title"] == "Mock Plot"
        assert meta["models"] == ["ifs-fesom"]
        assert meta["plot_type"] == "timeseries"
        assert "generated_at" in meta

    def test_build_metadata_helper(self, minimal_config):
        """_build_metadata() uses diagnostic's own variables by default."""
        diag = _MockDiagnostic(None, None, minimal_config)
        meta = diag._build_metadata(
            title="Test",
            figure_id="test_fig",
            models=["ifs-fesom"],
        )

        assert meta["diagnostic_name"] == "mock_test"
        assert meta["variables_used"] == ["avg_2t"]
        assert meta["units"] == "K"

    def test_build_metadata_override_variables(self, minimal_config):
        """_build_metadata() allows overriding variables."""
        diag = _MockDiagnostic(None, None, minimal_config)
        meta = diag._build_metadata(
            title="Test",
            figure_id="test_fig",
            models=["ifs-fesom"],
            variables=["avg_tprate"],
        )

        assert meta["variables_used"] == ["avg_tprate"]

    def test_figure_exists_both_files(self, minimal_config, tmp_path):
        """_figure_exists() returns True when both PNG and JSON exist."""
        diag = _MockDiagnostic(None, None, minimal_config)
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "mock_test_fig.png").write_bytes(b"fake")
        (diag.output_dir / "mock_test_fig.json").write_text("{}")

        assert diag._figure_exists("mock_test_fig") is True

    def test_figure_exists_missing_png(self, minimal_config, tmp_path):
        """_figure_exists() returns False when PNG is missing."""
        diag = _MockDiagnostic(None, None, minimal_config)
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "mock_test_fig.json").write_text("{}")

        assert diag._figure_exists("mock_test_fig") is False

    def test_figure_exists_missing_json(self, minimal_config, tmp_path):
        """_figure_exists() returns False when JSON is missing."""
        diag = _MockDiagnostic(None, None, minimal_config)
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "mock_test_fig.png").write_bytes(b"fake")

        assert diag._figure_exists("mock_test_fig") is False

    def test_figure_exists_no_dir(self, minimal_config, tmp_path):
        """_figure_exists() returns False when output dir doesn't exist."""
        diag = _MockDiagnostic(None, None, minimal_config)
        assert diag._figure_exists("mock_test_fig") is False

    def test_run_accepts_skip_existing(self, minimal_config, tmp_path):
        """run() accepts skip_existing kwarg without error."""
        diag = _MockDiagnostic(None, None, minimal_config)
        saved = diag.run(skip_existing=False)
        assert len(saved) == 1


# ── Registry tests ───────────────────────────────────────────────────


class TestRegistry:
    """Tests for the diagnostic registry."""

    def test_register_decorator(self):
        """@register adds the class to the registry."""

        @register
        class TestDiag(DiagnosticBase):
            name = "test_diag"
            title = "Test"
            domain = "sfc"
            variables = []
            group = "test"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        assert "test_diag" in _REGISTRY
        assert _REGISTRY["test_diag"] is TestDiag

    def test_register_without_name_raises(self):
        """@register raises ValueError if name is empty."""
        with pytest.raises(ValueError, match="non-empty 'name'"):

            @register
            class BadDiag(DiagnosticBase):
                name = ""

                def compute(self):
                    return {}

                def plot(self, results):
                    return []

    def test_get_diagnostic(self):
        """get_diagnostic() retrieves a registered class."""

        @register
        class MyDiag(DiagnosticBase):
            name = "my_diag"
            title = "My"
            domain = "sfc"
            variables = []
            group = "test"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        assert get_diagnostic("my_diag") is MyDiag

    def test_get_diagnostic_missing(self):
        """get_diagnostic() raises KeyError for unknown names."""
        with pytest.raises(KeyError, match="unknown_diag"):
            get_diagnostic("unknown_diag")

    def test_list_diagnostics_all(self):
        """list_diagnostics() returns all registered diagnostics."""

        @register
        class Diag1(DiagnosticBase):
            name = "diag1"
            title = "Diag 1"
            domain = "sfc"
            variables = ["avg_2t"]
            group = "temperature"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        @register
        class Diag2(DiagnosticBase):
            name = "diag2"
            title = "Diag 2"
            domain = "o2d"
            variables = ["avg_tos"]
            group = "ocean_surface"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        all_diags = list_diagnostics()
        assert len(all_diags) == 2
        names = {d["name"] for d in all_diags}
        assert names == {"diag1", "diag2"}

    def test_list_diagnostics_filtered_by_group(self):
        """list_diagnostics(group=...) filters by group."""

        @register
        class TempDiag(DiagnosticBase):
            name = "temp_diag"
            title = "Temp"
            domain = "sfc"
            variables = []
            group = "temperature"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        @register
        class OceanDiag(DiagnosticBase):
            name = "ocean_diag"
            title = "Ocean"
            domain = "o2d"
            variables = []
            group = "ocean_surface"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        temp_diags = list_diagnostics(group="temperature")
        assert len(temp_diags) == 1
        assert temp_diags[0]["name"] == "temp_diag"

    def test_registered_names(self):
        """registered_names() returns sorted list."""

        @register
        class BDiag(DiagnosticBase):
            name = "b_diag"
            title = "B"
            variables = []
            group = ""

            def compute(self):
                return {}

            def plot(self, results):
                return []

        @register
        class ADiag(DiagnosticBase):
            name = "a_diag"
            title = "A"
            variables = []
            group = ""

            def compute(self):
                return {}

            def plot(self, results):
                return []

        assert registered_names() == ["a_diag", "b_diag"]

    def test_list_diagnostics_info_fields(self):
        """list_diagnostics() returns dicts with expected fields."""

        @register
        class InfoDiag(DiagnosticBase):
            name = "info_diag"
            title = "Info Diagnostic"
            domain = "pl"
            variables = ["avg_t", "avg_u"]
            group = "circulation"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        diags = list_diagnostics()
        assert len(diags) == 1
        info = diags[0]
        assert info["name"] == "info_diag"
        assert info["title"] == "Info Diagnostic"
        assert info["domain"] == "pl"
        assert info["group"] == "circulation"
        assert info["variables"] == ["avg_t", "avg_u"]


# ── Base class helpers (Phase 3) ─────────────────────────────────────


class TestModelGlobalMean:
    """Tests for _model_global_mean() dispatch."""

    def test_healpix_dispatch(self, tmp_path, synth_healpix):
        """HEALPix data dispatches to global_mean()."""
        from feather.config import ModelConfig

        cfg = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            model_configs={
                "ifs-fesom": ModelConfig(
                    name="ifs-fesom",
                    grids={"sfc": "healpix"},
                ),
            },
        )
        diag = _MockDiagnostic(None, None, cfg)
        da = synth_healpix["avg_2t"]
        result = diag._model_global_mean(da, "ifs-fesom")
        assert "values" not in result.dims
        assert 260 < float(result.mean()) < 300

    def test_latlon_dispatch(self, tmp_path, synth_obs):
        """Lat/lon data dispatches to latlon_global_mean()."""
        from feather.config import ModelConfig

        cfg = FeatherConfig(
            model_catalogs={}, models=["MyModel"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            model_configs={
                "MyModel": ModelConfig(
                    name="MyModel",
                    grids={"sfc": "latlon"},
                ),
            },
        )
        diag = _MockDiagnostic(None, None, cfg)
        da = synth_obs["t2m"]
        result = diag._model_global_mean(da, "MyModel")
        assert "lat" not in result.dims
        assert "lon" not in result.dims
        assert 260 < float(result.mean()) < 300


class TestLoadModelVar:
    """Tests for _load_model_var() dispatch."""

    def test_cmor_dispatch(self, tmp_path):
        """CMOR data source dispatches to model_loader.load_var(model, var)."""
        from unittest.mock import MagicMock

        from feather.config import ModelConfig

        cfg = FeatherConfig(
            model_catalogs={}, models=["TestModel"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor"},
            model_configs={
                "TestModel": ModelConfig(
                    name="TestModel", grids={"sfc": "latlon"},
                ),
            },
        )

        mock_loader = MagicMock()
        mock_loader.load_var.return_value = "fake_data"
        diag = _MockDiagnostic(mock_loader, None, cfg)

        result = diag._load_model_var("TestModel", "tas")
        mock_loader.load_var.assert_called_once_with(
            "TestModel", "tas", period=None, time_mean=False,
        )
        assert result == "fake_data"

    def test_destine_dispatch(self, tmp_path):
        """DestinE data source dispatches to load_var(key, destine_var)."""
        from unittest.mock import MagicMock

        cfg = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            project={"experiment": "baseline_hist"},
        )

        mock_loader = MagicMock()
        mock_da = MagicMock()
        mock_da.dims = ()  # no time dim
        mock_loader.load_var.return_value = mock_da
        diag = _MockDiagnostic(mock_loader, None, cfg)

        diag._load_model_var("ifs-fesom", "tas", experiment="baseline_hist")

        # Should call with DestinE catalog key and DestinE variable name
        call_args = mock_loader.load_var.call_args
        key = call_args[0][0]
        var = call_args[0][1]
        assert "ifs-fesom" in key
        assert var == "avg_2t"  # DestinE name for tas


class TestLoadModelCoords:
    """Tests for _load_model_coords() dispatch."""

    def test_cmor_coords(self, tmp_path):
        """CMOR loader returns lat/lon from DataArray coords."""
        from unittest.mock import MagicMock

        from feather.config import ModelConfig

        cfg = FeatherConfig(
            model_catalogs={}, models=["TestModel"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor"},
            model_configs={
                "TestModel": ModelConfig(
                    name="TestModel", grids={"sfc": "latlon"},
                ),
            },
        )

        import xarray as xr

        lats = np.arange(-90, 91, 5.0)
        lons = np.arange(0, 360, 5.0)
        da = xr.DataArray(
            np.zeros((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        mock_loader = MagicMock()
        mock_loader.load_var.return_value = da
        diag = _MockDiagnostic(mock_loader, None, cfg)

        lon, lat = diag._load_model_coords("TestModel", "tas")
        assert len(lon) == 72
        assert len(lat) == 37

    def test_destine_coords(self, tmp_path, synth_healpix):
        """DestinE loader returns lon/lat from Dataset variables."""
        from unittest.mock import MagicMock

        cfg = FeatherConfig(
            model_catalogs={}, models=["ifs-fesom"],
            obs_root="", obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path / "output"),
            project={"experiment": "baseline_hist"},
        )

        mock_loader = MagicMock()
        mock_loader.load.return_value = synth_healpix
        diag = _MockDiagnostic(mock_loader, None, cfg)

        lon, lat = diag._load_model_coords(
            "ifs-fesom", "tas", experiment="baseline_hist",
        )
        assert len(lon) == 768  # nside=8
        assert len(lat) == 768
