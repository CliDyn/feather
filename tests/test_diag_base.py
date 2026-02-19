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
