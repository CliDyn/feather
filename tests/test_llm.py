"""Tests for feather.llm — schemas, prompts, analyzer, JSON parsing.

All tests mock google.generativeai — no real API calls.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feather.config import FeatherConfig
from feather.llm.schemas import DiagnosticSynthesis, FigureAnalysis
from feather.llm.prompts import (
    build_figure_analysis_system,
    build_figure_prompt,
    build_synthesis_prompt,
    build_synthesis_system,
)
from feather.llm.analyzer import FigureAnalyzer


# ── Fixtures ─────────────────────────────────────────────────────────

def _valid_analysis_dict():
    """Return a valid FigureAnalysis dict."""
    return {
        "summary": "The bias map shows warm biases in all three models.",
        "key_findings": [
            "IFS-FESOM has the smallest global mean bias (+0.3K).",
            "ICON shows a persistent cold bias over Antarctica.",
            "All models overestimate SST in the Southern Ocean.",
        ],
        "spatial_patterns": "Warm biases are concentrated in the tropics.",
        "model_agreement": "Models agree on the sign of the bias over oceans.",
        "physical_interpretation": "Tropical warm biases likely relate to cloud parameterization.",
        "caveats": ["ERA5 has known biases in polar regions."],
        "confidence": "high",
    }


def _valid_synthesis_dict():
    """Return a valid DiagnosticSynthesis dict."""
    return {
        "narrative": "The models show consistent warm biases. Para 2. Para 3.",
        "headline_finding": "All three DestinE models exhibit systematic warm biases in the tropics.",
        "connections": ["seasonal_cycle", "timeseries"],
    }


def _sample_metadata():
    """Return sample figure metadata (JSON sidecar contents)."""
    return {
        "diagnostic_name": "global_biases",
        "title": "2m Temperature Annual Mean Bias",
        "figure_id": "t2m_annual_bias",
        "variables_used": ["avg_2t"],
        "models": ["ifs-fesom", "ifs-nemo", "icon"],
        "obs_dataset": "ERA5",
        "obs_variable": "t2m",
        "units": "K",
        "period": ["1990", "2014"],
        "description": "Annual mean bias (model - obs) for 2m temperature.",
        "computation_notes": "Climatology computed over 1990-2014.",
        "domain": "sfc",
        "group": "temperature",
        "spatial_extent": "global",
        "plot_type": "bias_map",
        "colormap": "RdBu_r",
        "summary_statistics": {
            "ifs-fesom": {"global_mean_bias": 0.3},
            "ifs-nemo": {"global_mean_bias": 0.5},
            "icon": {"global_mean_bias": -0.2},
        },
        "cmip6_info": {
            "n_models": {"tas": 5},
            "models_used": {"tas": ["CanESM5", "MPI-ESM1-2-LR", "GISS-E2-1-G", "IPSL-CM6A-LR", "ACCESS-ESM1-5"]},
        },
    }


def _minimal_config(tmp_path):
    """Return a minimal FeatherConfig for testing."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom", "ifs-nemo", "icon"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path),
        llm={
            "figure_analysis": {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "max_retries": 2,
                "retry_delay": 0,
                "skip_existing": True,
            }
        },
    )


def _setup_figures(tmp_path, diagnostic_name="global_biases", n_figures=2):
    """Create dummy PNG+JSON pairs in the figures directory."""
    figures_dir = tmp_path / "figures" / diagnostic_name
    figures_dir.mkdir(parents=True)
    meta = _sample_metadata()
    paths = []
    for i in range(n_figures):
        stem = f"figure_{i}"
        png = figures_dir / f"{stem}.png"
        png.write_bytes(b"\x89PNG fake data")
        json_path = figures_dir / f"{stem}.json"
        json_path.write_text(json.dumps(meta))
        paths.append((png, json_path))
    return paths


# ── Schema tests ─────────────────────────────────────────────────────

class TestFigureAnalysis:
    def test_valid_data(self):
        analysis = FigureAnalysis(**_valid_analysis_dict())
        assert analysis.confidence == "high"
        assert len(analysis.key_findings) == 3

    def test_invalid_confidence(self):
        data = _valid_analysis_dict()
        data["confidence"] = "very_high"
        with pytest.raises(Exception):
            FigureAnalysis(**data)

    def test_min_findings(self):
        data = _valid_analysis_dict()
        data["key_findings"] = ["only one"]
        with pytest.raises(Exception):
            FigureAnalysis(**data)

    def test_max_findings(self):
        data = _valid_analysis_dict()
        data["key_findings"] = [f"finding {i}" for i in range(8)]
        with pytest.raises(Exception):
            FigureAnalysis(**data)

    def test_empty_caveats(self):
        data = _valid_analysis_dict()
        data["caveats"] = []
        analysis = FigureAnalysis(**data)
        assert analysis.caveats == []

    def test_default_caveats(self):
        data = _valid_analysis_dict()
        del data["caveats"]
        analysis = FigureAnalysis(**data)
        assert analysis.caveats == []


class TestDiagnosticSynthesis:
    def test_valid_data(self):
        synthesis = DiagnosticSynthesis(**_valid_synthesis_dict())
        assert "warm biases" in synthesis.headline_finding
        assert len(synthesis.connections) == 2

    def test_empty_connections(self):
        data = _valid_synthesis_dict()
        data["connections"] = []
        synthesis = DiagnosticSynthesis(**data)
        assert synthesis.connections == []

    def test_default_connections(self):
        data = _valid_synthesis_dict()
        del data["connections"]
        synthesis = DiagnosticSynthesis(**data)
        assert synthesis.connections == []


# ── Prompt tests ─────────────────────────────────────────────────────

class TestFigurePrompt:
    def test_contains_diagnostic_name(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "global_biases" in prompt

    def test_contains_title(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "2m Temperature Annual Mean Bias" in prompt

    def test_contains_variables(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "avg_2t" in prompt

    def test_contains_models(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "ifs-fesom" in prompt
        assert "ifs-nemo" in prompt
        assert "icon" in prompt

    def test_contains_units(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "K" in prompt

    def test_contains_period(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "1990" in prompt
        assert "2014" in prompt

    def test_contains_obs_info(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "ERA5" in prompt
        assert "t2m" in prompt

    def test_contains_summary_statistics(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "global_mean_bias" in prompt

    def test_contains_cmip6_context(self):
        prompt = build_figure_prompt(_sample_metadata())
        assert "CMIP6" in prompt
        assert "5" in prompt
        assert "CanESM5" in prompt

    def test_missing_fields_graceful(self):
        """Minimal metadata should not raise."""
        prompt = build_figure_prompt({"diagnostic_name": "test"})
        assert "test" in prompt

    def test_no_cmip6(self):
        meta = _sample_metadata()
        del meta["cmip6_info"]
        prompt = build_figure_prompt(meta)
        assert "CMIP6" not in prompt

    def test_no_summary_stats(self):
        meta = _sample_metadata()
        meta["summary_statistics"] = {}
        prompt = build_figure_prompt(meta)
        assert "global_mean_bias" not in prompt


class TestSystemPrompts:
    def test_figure_system_mentions_evaluation(self):
        system = build_figure_analysis_system()
        assert "evaluat" in system.lower()

    def test_figure_system_mentions_healpix(self):
        system = build_figure_analysis_system()
        assert "HEALPix" in system

    def test_figure_system_mentions_all_models(self):
        system = build_figure_analysis_system()
        assert "IFS-FESOM" in system
        assert "IFS-NEMO" in system
        assert "ICON" in system

    def test_figure_system_mentions_json_schema(self):
        system = build_figure_analysis_system()
        assert '"summary"' in system
        assert '"key_findings"' in system

    def test_synthesis_system_mentions_synthesis(self):
        system = build_synthesis_system()
        assert "synthesis" in system.lower()


class TestSynthesisPrompt:
    def test_contains_diagnostic_name(self):
        prompt = build_synthesis_prompt("global_biases", [_valid_analysis_dict()])
        assert "global_biases" in prompt

    def test_contains_analyses(self):
        prompt = build_synthesis_prompt("test", [_valid_analysis_dict()])
        assert "warm biases" in prompt


# ── JSON parsing tests ───────────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        data = {"key": "value"}
        result = FigureAnalyzer._parse_json_response(json.dumps(data))
        assert result == data

    def test_markdown_fencing(self):
        data = {"key": "value"}
        text = f"```json\n{json.dumps(data)}\n```"
        result = FigureAnalyzer._parse_json_response(text)
        assert result == data

    def test_latex_escapes(self):
        text = '{"summary": "Temperature change \\Delta T is 2K"}'
        result = FigureAnalyzer._parse_json_response(text)
        assert "\\Delta" in result["summary"]

    def test_latex_degree(self):
        """\\degree is not a valid JSON escape and triggers the fixer."""
        text = '{"summary": "Warming of 2\\degree C"}'
        result = FigureAnalyzer._parse_json_response(text)
        assert "\\degree" in result["summary"]

    def test_latex_u_escape(self):
        """\\units should not be treated as \\uXXXX unicode escape."""
        text = '{"summary": "Flux in \\units of W/m2"}'
        result = FigureAnalyzer._parse_json_response(text)
        assert "\\units" in result["summary"]


# ── Analyzer tests ───────────────────────────────────────────────────

@patch("feather.llm.analyzer.genai")
class TestFigureAnalyzerInit:
    def test_init_with_api_key(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        analyzer = FigureAnalyzer(cfg, api_key="test-key")
        mock_genai.configure.assert_called_once_with(api_key="test-key")
        assert analyzer.model_name == "gemini-2.5-flash"

    def test_init_from_env(self, mock_genai, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        cfg = _minimal_config(tmp_path)
        analyzer = FigureAnalyzer(cfg)
        mock_genai.configure.assert_called_once_with(api_key="env-key")

    def test_init_no_key_raises(self, mock_genai, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = _minimal_config(tmp_path)
        with pytest.raises(RuntimeError, match="API key not found"):
            FigureAnalyzer(cfg)

    def test_init_default_config(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        cfg.llm = {}
        # No api_key_env in config, falls back to GEMINI_API_KEY
        with pytest.raises(RuntimeError):
            FigureAnalyzer(cfg)


@patch("feather.llm.analyzer.genai")
class TestFigureAnalyzerDiscovery:
    def test_discover_empty(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer._discover_figures()
        assert result == {}

    def test_discover_no_figures_dir(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer._discover_figures()
        assert result == {}

    def test_discover_finds_pairs(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=3)
        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer._discover_figures()
        assert "global_biases" in result
        assert len(result["global_biases"]) == 3

    def test_discover_skips_orphan_png(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=1)
        # Add an orphan PNG (no matching JSON)
        orphan = tmp_path / "figures" / "global_biases" / "orphan.png"
        orphan.write_bytes(b"\x89PNG")
        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer._discover_figures()
        assert len(result["global_biases"]) == 1


@patch("feather.llm.analyzer.genai")
class TestFigureAnalyzerAnalyze:
    def _mock_gemini_response(self, mock_genai, response_dict):
        """Set up mock_genai to return a specific response dict."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_dict)
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = MagicMock()

    def test_analyze_figure(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        paths = _setup_figures(tmp_path, n_figures=1)
        self._mock_gemini_response(mock_genai, _valid_analysis_dict())

        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer.analyze_figure(paths[0][0], paths[0][1])

        assert isinstance(result, FigureAnalysis)
        assert result.confidence == "high"
        mock_genai.upload_file.assert_called_once()

    def test_synthesize_diagnostic(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        self._mock_gemini_response(mock_genai, _valid_synthesis_dict())

        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer.synthesize_diagnostic(
            "global_biases", [_valid_analysis_dict()]
        )

        assert isinstance(result, DiagnosticSynthesis)
        assert "warm biases" in result.headline_finding

    def test_run_end_to_end(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=2)

        # First call returns figure analysis, last call returns synthesis
        analysis_response = MagicMock()
        analysis_response.text = json.dumps(_valid_analysis_dict())
        synthesis_response = MagicMock()
        synthesis_response.text = json.dumps(_valid_synthesis_dict())

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            analysis_response,
            analysis_response,
            synthesis_response,
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = MagicMock()

        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer.run(skip_existing=False)

        assert result["figure_analyses"] == 2
        assert result["syntheses"] == 1

        # Check analysis files were saved
        analysis_dir = tmp_path / "analysis" / "global_biases"
        assert (analysis_dir / "figure_0_analysis.json").exists()
        assert (analysis_dir / "figure_1_analysis.json").exists()
        assert (analysis_dir / "synthesis.json").exists()

    def test_skip_existing(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=1)

        # Pre-create analysis file
        analysis_dir = tmp_path / "analysis" / "global_biases"
        analysis_dir.mkdir(parents=True)
        existing = analysis_dir / "figure_0_analysis.json"
        existing.write_text(json.dumps(_valid_analysis_dict()))

        # Mock for synthesis only
        synthesis_response = MagicMock()
        synthesis_response.text = json.dumps(_valid_synthesis_dict())
        mock_model = MagicMock()
        mock_model.generate_content.return_value = synthesis_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer.run(skip_existing=True)

        # Figure was skipped, only synthesis was done
        assert result["figure_analyses"] == 0
        assert result["syntheses"] == 1
        # upload_file should NOT have been called (figure was skipped)
        mock_genai.upload_file.assert_not_called()

    def test_diagnostics_filter(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, diagnostic_name="global_biases", n_figures=1)
        _setup_figures(tmp_path, diagnostic_name="timeseries", n_figures=1)

        self._mock_gemini_response(mock_genai, _valid_analysis_dict())

        # We need separate responses for analysis + synthesis
        analysis_resp = MagicMock()
        analysis_resp.text = json.dumps(_valid_analysis_dict())
        synthesis_resp = MagicMock()
        synthesis_resp.text = json.dumps(_valid_synthesis_dict())
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [analysis_resp, synthesis_resp]
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = MagicMock()

        analyzer = FigureAnalyzer(cfg, api_key="k")
        result = analyzer.run(
            skip_existing=False,
            diagnostics=["timeseries"],
        )

        assert result["figure_analyses"] == 1
        assert result["syntheses"] == 1
        # Only timeseries analysis should exist
        assert (tmp_path / "analysis" / "timeseries" / "figure_0_analysis.json").exists()
        assert not (tmp_path / "analysis" / "global_biases").exists()

    def test_run_handles_failure(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=1)

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = RuntimeError("API error")
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = MagicMock()

        analyzer = FigureAnalyzer(cfg, api_key="k")
        # max_retries=2, retry_delay=0 from config
        result = analyzer.run(skip_existing=False)

        # Should not crash, just log the error
        assert result["figure_analyses"] == 0
        assert result["syntheses"] == 0
