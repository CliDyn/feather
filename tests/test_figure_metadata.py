"""Tests for the figure metadata sidecar system."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.diag.figure_meta import build_metadata, save_figure_with_metadata


def _make_fig():
    """Create a simple matplotlib figure for testing."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test")
    return fig


class TestSaveFigureWithMetadata:
    """Tests for save_figure_with_metadata()."""

    def test_creates_png_and_json(self, tmp_path):
        """Both PNG and JSON files are created."""
        fig = _make_fig()
        meta = {"diagnostic_name": "test", "title": "Test Figure"}

        png, json_path = save_figure_with_metadata(
            fig, meta, tmp_path, "test_fig"
        )

        assert png.exists()
        assert json_path.exists()
        assert png.suffix == ".png"
        assert json_path.suffix == ".json"

    def test_json_contains_metadata(self, tmp_path):
        """JSON sidecar contains the provided metadata."""
        fig = _make_fig()
        meta = {
            "diagnostic_name": "global_biases",
            "title": "2m Temperature Bias",
            "variables_used": ["avg_2t"],
            "models": ["ifs-fesom"],
        }

        _, json_path = save_figure_with_metadata(fig, meta, tmp_path, "bias")

        with open(json_path) as f:
            saved = json.load(f)

        assert saved["diagnostic_name"] == "global_biases"
        assert saved["title"] == "2m Temperature Bias"
        assert saved["variables_used"] == ["avg_2t"]
        assert saved["models"] == ["ifs-fesom"]

    def test_adds_timestamp(self, tmp_path):
        """JSON sidecar includes a generated_at timestamp."""
        fig = _make_fig()
        meta = {"diagnostic_name": "test"}

        _, json_path = save_figure_with_metadata(fig, meta, tmp_path, "ts")

        with open(json_path) as f:
            saved = json.load(f)

        assert "generated_at" in saved
        assert "T" in saved["generated_at"]  # ISO format

    def test_creates_subdirectories(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        fig = _make_fig()
        meta = {"diagnostic_name": "test"}
        nested = tmp_path / "a" / "b" / "c"

        png, json_path = save_figure_with_metadata(fig, meta, nested, "deep")

        assert png.exists()
        assert json_path.exists()

    def test_does_not_mutate_input_metadata(self, tmp_path):
        """The input metadata dict is not modified in place."""
        fig = _make_fig()
        meta = {"diagnostic_name": "test"}
        original = dict(meta)

        save_figure_with_metadata(fig, meta, tmp_path, "immutable")

        assert meta == original
        assert "generated_at" not in meta

    def test_closes_figure_by_default(self, tmp_path):
        """Figure is closed after saving by default."""
        fig = _make_fig()
        fig_num = fig.number
        meta = {"diagnostic_name": "test"}

        save_figure_with_metadata(fig, meta, tmp_path, "closed")

        assert fig_num not in plt.get_fignums()

    def test_keeps_figure_open_when_requested(self, tmp_path):
        """Figure stays open when close=False."""
        fig = _make_fig()
        fig_num = fig.number
        meta = {"diagnostic_name": "test"}

        save_figure_with_metadata(fig, meta, tmp_path, "open", close=False)

        assert fig_num in plt.get_fignums()
        plt.close(fig)


class TestBuildMetadata:
    """Tests for build_metadata()."""

    def test_basic_fields(self):
        """Basic fields are populated correctly."""
        meta = build_metadata(
            "global_biases",
            "2m Temperature Bias",
            figure_id="t2m_bias_ifs-fesom",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
        )

        assert meta["diagnostic_name"] == "global_biases"
        assert meta["title"] == "2m Temperature Bias"
        assert meta["figure_id"] == "t2m_bias_ifs-fesom"
        assert meta["variables_used"] == ["avg_2t"]
        assert meta["models"] == ["ifs-fesom"]

    def test_pulls_units_from_registry(self):
        """Units are pulled from VARIABLE_REGISTRY."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
        )

        assert meta["units"] == "K"
        assert meta["domain"] == "sfc"
        assert meta["group"] == "temperature"

    def test_auto_fills_obs_info(self):
        """Obs dataset and variable are auto-filled from registry."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
        )

        assert meta["obs_dataset"] == "ERA5"
        assert meta["obs_variable"] == "t2m"

    def test_explicit_obs_overrides_registry(self):
        """Explicit obs_dataset/obs_variable overrides registry values."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
            obs_dataset="BERKELEY_EARTH",
            obs_variable="temperature",
        )

        assert meta["obs_dataset"] == "BERKELEY_EARTH"
        assert meta["obs_variable"] == "temperature"

    def test_period(self):
        """Period is stored as a list."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
            period=("1990", "2014"),
        )

        assert meta["period"] == ["1990", "2014"]

    def test_no_period(self):
        """Period is None when not provided."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
        )

        assert meta["period"] is None

    def test_summary_statistics(self):
        """Summary statistics are included."""
        stats = {"global_mean_bias": -0.42, "rmse": 2.1}
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
            summary_statistics=stats,
        )

        assert meta["summary_statistics"]["global_mean_bias"] == -0.42
        assert meta["summary_statistics"]["rmse"] == 2.1

    def test_unknown_variable_graceful(self):
        """Unknown variables don't crash — just leave fields empty."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["unknown_var"],
            models=["ifs-fesom"],
        )

        assert meta["units"] == ""
        assert meta["domain"] == ""

    def test_empty_variables(self):
        """Empty variables list is handled gracefully."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=[],
            models=[],
        )

        assert meta["units"] == ""

    def test_extra_fields_merged(self):
        """Extra fields are merged into metadata."""
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
            extra={"custom_key": "custom_value", "number": 42},
        )

        assert meta["custom_key"] == "custom_value"
        assert meta["number"] == 42

    def test_cmip6_info(self):
        """CMIP6 info is included when provided."""
        cmip6 = {"models_used": ["MIROC6", "CESM2"], "n_models": 2}
        meta = build_metadata(
            "test",
            "Test",
            figure_id="test",
            variables_used=["avg_2t"],
            models=["ifs-fesom"],
            cmip6_info=cmip6,
        )

        assert meta["cmip6_info"]["models_used"] == ["MIROC6", "CESM2"]
