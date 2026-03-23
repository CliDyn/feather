"""Tests for the report generation pipeline (feather.export)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feather.config import FeatherConfig
from feather.export.latex_builder import (
    build_document,
    copy_figures,
    escape_latex,
)
from feather.export.openai_client import OpenAIClient
from feather.export.prompts import (
    build_curation_prompt,
    build_curation_system,
    build_section_prompt,
    build_section_system,
)
from feather.export.report import ReportGenerator
from feather.export.schemas import (
    ReportSection,
    ReportStructure,
    SelectedFigure,
    WrittenSection,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _report_config(tmp_path):
    """Create a minimal FeatherConfig with report settings."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom", "ifs-nemo", "icon"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
        report={
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.3,
            "api_key_env": "OPENAI_API_KEY",
            "n_highlights": 6,
        },
    )


def _sample_structure():
    """Return a minimal valid ReportStructure dict."""
    return {
        "title": "Test Report",
        "abstract": "Abstract text.",
        "introduction": "Intro text.",
        "sections": [
            {
                "section_id": "01_temp",
                "title": "Temperature",
                "narrative_hook": "Temperature biases.",
                "figure_ids": ["fig_t2m_bias"],
                "diagnostics": ["global_biases"],
            },
            {
                "section_id": "02_precip",
                "title": "Precipitation",
                "narrative_hook": "Precip patterns.",
                "figure_ids": ["fig_precip_bias"],
                "diagnostics": ["global_biases"],
            },
        ],
        "selected_figures": [
            {
                "diagnostic": "global_biases",
                "figure_id": "fig_t2m_bias",
                "caption": "T2M bias map.",
                "label": "fig:t2m_bias",
            },
            {
                "diagnostic": "global_biases",
                "figure_id": "fig_precip_bias",
                "caption": "Precipitation bias.",
                "label": "fig:precip_bias",
            },
            {
                "diagnostic": "timeseries",
                "figure_id": "fig_ts_1",
                "caption": "Time series 1.",
                "label": "fig:ts1",
            },
            {
                "diagnostic": "timeseries",
                "figure_id": "fig_ts_2",
                "caption": "Time series 2.",
                "label": "fig:ts2",
            },
        ],
        "conclusion": "Conclusion text.",
    }


def _sample_written_sections():
    """Return written sections matching the sample structure."""
    return [
        {
            "section_id": "01_temp",
            "title": "Temperature",
            "body": "Temperature bias analysis.",
        },
        {
            "section_id": "02_precip",
            "title": "Precipitation",
            "body": "Precipitation patterns.",
        },
    ]


# ── Schema tests ────────────────────────────────────────────────────


class TestSelectedFigure:
    def test_valid(self):
        fig = SelectedFigure(
            diagnostic="global_biases",
            figure_id="t2m_bias",
            caption="T2M bias map",
            label="fig:t2m",
        )
        assert fig.diagnostic == "global_biases"
        assert fig.label == "fig:t2m"

    def test_missing_required(self):
        with pytest.raises(Exception):
            SelectedFigure(diagnostic="x")


class TestReportSection:
    def test_valid(self):
        sec = ReportSection(
            section_id="01_temp",
            title="Temperature",
            narrative_hook="Hook.",
            figure_ids=["a", "b"],
            diagnostics=["global_biases"],
        )
        assert sec.section_id == "01_temp"
        assert len(sec.figure_ids) == 2


class TestReportStructure:
    def test_valid(self):
        data = _sample_structure()
        s = ReportStructure(**data)
        assert len(s.sections) == 2
        assert len(s.selected_figures) == 4

    def test_too_few_sections(self):
        data = _sample_structure()
        data["sections"] = [data["sections"][0]]
        with pytest.raises(Exception):
            ReportStructure(**data)

    def test_too_few_figures(self):
        data = _sample_structure()
        data["selected_figures"] = data["selected_figures"][:3]
        with pytest.raises(Exception):
            ReportStructure(**data)


class TestWrittenSection:
    def test_valid(self):
        ws = WrittenSection(
            section_id="01_temp",
            title="Temperature",
            body="Analysis text.",
        )
        assert ws.section_id == "01_temp"


# ── escape_latex tests ──────────────────────────────────────────────


class TestEscapeLatex:
    def test_special_chars(self):
        result = escape_latex("100% & $5 cost_variable")
        assert "\\%" in result
        assert "\\&" in result
        assert "\\$" in result
        assert "\\_" in result

    def test_unicode_degree(self):
        result = escape_latex("Temperature is 20\u00b0C")
        assert "$^{\\circ}$" in result

    def test_unicode_dashes(self):
        result = escape_latex("1990\u20132014")
        assert "--" in result

    def test_preserves_ref(self):
        text = r"See Figure~\ref{fig:t2m} for details."
        result = escape_latex(text)
        assert r"\ref{fig:t2m}" in result

    def test_preserves_multiple_refs(self):
        text = r"Figure~\ref{fig:a} and Figure~\ref{fig:b}."
        result = escape_latex(text)
        assert r"\ref{fig:a}" in result
        assert r"\ref{fig:b}" in result

    def test_empty_string(self):
        assert escape_latex("") == ""

    def test_plain_text(self):
        text = "This is plain text."
        assert escape_latex(text) == text


# ── copy_figures tests ──────────────────────────────────────────────


class TestCopyFigures:
    def test_copies_existing(self, tmp_path):
        src = tmp_path / "figures" / "global_biases"
        src.mkdir(parents=True)
        (src / "t2m_bias.png").write_bytes(b"fake png")

        dest = tmp_path / "pub" / "figures"
        figs = [{"diagnostic": "global_biases", "figure_id": "t2m_bias"}]
        paths = copy_figures(figs, tmp_path / "figures", dest)

        assert "t2m_bias" in paths
        assert (dest / "global_biases" / "t2m_bias.png").exists()

    def test_missing_source(self, tmp_path):
        src = tmp_path / "figures"
        src.mkdir(parents=True)
        dest = tmp_path / "pub" / "figures"
        figs = [{"diagnostic": "missing", "figure_id": "no_file"}]
        paths = copy_figures(figs, src, dest)

        assert len(paths) == 0


# ── build_document tests ────────────────────────────────────────────


class TestBuildDocument:
    def test_renders_template(self, tmp_path):
        structure = _sample_structure()
        written = _sample_written_sections()

        # Create fake figure files
        src = tmp_path / "figures" / "global_biases"
        src.mkdir(parents=True)
        (src / "fig_t2m_bias.png").write_bytes(b"png")
        (src / "fig_precip_bias.png").write_bytes(b"png")

        dest = tmp_path / "pub" / "figures"
        paths = copy_figures(structure["selected_figures"], tmp_path / "figures", dest)

        doc = build_document(structure, written, paths)

        assert "\\title{" in doc
        assert "\\begin{document}" in doc
        assert "\\end{document}" in doc
        assert "Temperature" in doc
        assert "\\begin{abstract}" in doc

    def test_empty_sections(self):
        structure = _sample_structure()
        doc = build_document(structure, [], {})
        assert "\\begin{document}" in doc


# ── Prompt tests ────────────────────────────────────────────────────


class TestPrompts:
    def test_curation_system(self):
        sys_prompt = build_curation_system()
        assert "DestinE" in sys_prompt
        assert "JSON" in sys_prompt
        assert "model evaluation" in sys_prompt.lower() or "evaluating" in sys_prompt.lower()

    def test_curation_prompt(self):
        syntheses = {"global_biases": {"headline_finding": "Warm bias"}}
        metadata = {"global_biases": [
            {"figure_id": "t2m", "title": "T2M Bias", "variables_used": ["avg_2t"],
             "models": ["ifs-fesom"], "description": "Test"}
        ]}
        analyses = {"global_biases": [
            {"figure_id": "t2m", "summary": "Warm bias", "confidence": "high",
             "key_findings": ["Finding 1"]}
        ]}
        prompt = build_curation_prompt(syntheses, metadata, analyses, n_highlights=5)
        assert "5 figures" in prompt
        assert "global_biases" in prompt
        assert "Finding 1" in prompt

    def test_section_system(self):
        sys_prompt = build_section_system()
        assert "IPCC" in sys_prompt
        assert "\\ref{" in sys_prompt

    def test_section_prompt(self):
        section = {
            "section_id": "01_temp",
            "title": "Temperature",
            "narrative_hook": "Hook",
            "figure_ids": ["t2m"],
            "diagnostics": ["global_biases"],
        }
        figs = [{"figure_id": "t2m", "diagnostic": "global_biases",
                 "caption": "T2M", "label": "fig:t2m"}]
        prompt = build_section_prompt(
            section, figs, {}, {"global_biases": {"headline_finding": "Warm"}},
        )
        assert "01_temp" in prompt
        assert "Temperature" in prompt
        assert "fig:t2m" in prompt

    # ── comparison_type tests ──

    def test_curation_system_resolution_sensitivity(self):
        sys_prompt = build_curation_system(
            comparison_type="resolution_sensitivity",
        )
        assert "resolution sensitivity" in sys_prompt.lower()
        assert "high-resolution added value" in sys_prompt.lower()

    def test_curation_system_single_model(self):
        sys_prompt = build_curation_system(
            comparison_type="single_model",
        )
        assert "outperforms or underperforms" in sys_prompt

    def test_curation_system_default_multi_model(self):
        sys_prompt = build_curation_system()
        assert "agree or disagree" in sys_prompt

    def test_curation_system_comparison_description(self):
        sys_prompt = build_curation_system(
            comparison_description="Custom context for this project.",
        )
        assert "Custom context for this project." in sys_prompt

    def test_section_system_resolution_sensitivity(self):
        sys_prompt = build_section_system(
            comparison_type="resolution_sensitivity",
        )
        assert "resolution scaling" in sys_prompt.lower()
        assert "cost-benefit" in sys_prompt.lower()

    def test_section_system_single_model(self):
        sys_prompt = build_section_system(
            comparison_type="single_model",
        )
        assert "outperforms or underperforms" in sys_prompt

    def test_section_system_comparison_description(self):
        sys_prompt = build_section_system(
            comparison_description="Same model at 3 resolutions.",
        )
        assert "Same model at 3 resolutions." in sys_prompt

    # ── baseline_evaluation tests ──

    def test_curation_system_baseline_evaluation(self):
        sys_prompt = build_curation_system(
            comparison_type="baseline_evaluation",
        )
        assert "baseline performance" in sys_prompt.lower()
        assert "cryosphere" in sys_prompt.lower()

    def test_section_system_baseline_evaluation(self):
        sys_prompt = build_section_system(
            comparison_type="baseline_evaluation",
        )
        assert "baseline assessment" in sys_prompt.lower()
        assert "terradt" in sys_prompt.lower()
        assert "cryosphere" in sys_prompt.lower()


# ── OpenAIClient tests ──────────────────────────────────────────────


class TestOpenAIClient:
    def test_parse_json_clean(self):
        text = '{"key": "value"}'
        result = OpenAIClient._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_with_fencing(self):
        text = '```json\n{"key": "value"}\n```'
        result = OpenAIClient._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_invalid_escapes(self):
        text = '{"text": "Temperature is \\Delta T = 5K"}'
        result = OpenAIClient._parse_json(text)
        assert "Delta" in result["text"]

    def test_init_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="API key"):
                OpenAIClient({"api_key_env": "NONEXISTENT_KEY_VAR"})


# ── ReportGenerator tests ──────────────────────────────────────────


class TestReportGenerator:
    def test_load_syntheses(self, tmp_path):
        cfg = _report_config(tmp_path)
        analysis_dir = tmp_path / "output" / "analysis" / "global_biases"
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "synthesis.json").write_text(
            json.dumps({"headline_finding": "Warm bias"})
        )

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._load_syntheses()

        assert "global_biases" in result
        assert result["global_biases"]["headline_finding"] == "Warm bias"

    def test_load_figure_metadata(self, tmp_path):
        cfg = _report_config(tmp_path)
        fig_dir = tmp_path / "output" / "figures" / "global_biases"
        fig_dir.mkdir(parents=True)
        (fig_dir / "t2m_bias.json").write_text(
            json.dumps({"title": "T2M Bias", "models": ["ifs-fesom"]})
        )

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._load_figure_metadata()

        assert "global_biases" in result
        assert result["global_biases"][0]["figure_id"] == "t2m_bias"

    def test_load_figure_analyses(self, tmp_path):
        cfg = _report_config(tmp_path)
        analysis_dir = tmp_path / "output" / "analysis" / "global_biases"
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "t2m_bias_analysis.json").write_text(
            json.dumps({"summary": "Test analysis"})
        )

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._load_figure_analyses()

        assert "global_biases" in result
        assert result["global_biases"][0]["figure_id"] == "t2m_bias"

    def test_load_empty_dirs(self, tmp_path):
        cfg = _report_config(tmp_path)
        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")

        assert gen._load_syntheses() == {}
        assert gen._load_figure_metadata() == {}
        assert gen._load_figure_analyses() == {}

    def test_stage1_caching(self, tmp_path):
        cfg = _report_config(tmp_path)
        pub_dir = tmp_path / "output" / "publication"
        pub_dir.mkdir(parents=True)
        structure = _sample_structure()
        (pub_dir / "structure.json").write_text(json.dumps(structure))

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._stage1_curation({}, {}, {}, skip_existing=True)

        assert result["title"] == "Test Report"

    def test_stage2_caching(self, tmp_path):
        cfg = _report_config(tmp_path)
        sections_dir = tmp_path / "output" / "publication" / "sections"
        sections_dir.mkdir(parents=True)

        structure = _sample_structure()
        for ws in _sample_written_sections():
            (sections_dir / f"{ws['section_id']}.json").write_text(
                json.dumps(ws)
            )

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._stage2_sections(structure, {}, {}, skip_existing=True)

        assert len(result) == 2
        assert result[0]["section_id"] == "01_temp"

    def test_full_run_mocked(self, tmp_path):
        """Test full run with mocked OpenAI client."""
        cfg = _report_config(tmp_path)

        # Create minimal figure + analysis data
        fig_dir = tmp_path / "output" / "figures" / "global_biases"
        fig_dir.mkdir(parents=True)
        (fig_dir / "fig_t2m_bias.png").write_bytes(b"fake png")
        (fig_dir / "fig_t2m_bias.json").write_text(
            json.dumps({"title": "T2M Bias", "models": ["ifs-fesom"]})
        )
        (fig_dir / "fig_precip_bias.png").write_bytes(b"fake png")
        (fig_dir / "fig_precip_bias.json").write_text(
            json.dumps({"title": "Precip Bias", "models": ["ifs-fesom"]})
        )

        ts_dir = tmp_path / "output" / "figures" / "timeseries"
        ts_dir.mkdir(parents=True)
        for fid in ["fig_ts_1", "fig_ts_2"]:
            (ts_dir / f"{fid}.png").write_bytes(b"fake png")
            (ts_dir / f"{fid}.json").write_text(
                json.dumps({"title": fid})
            )

        analysis_dir = tmp_path / "output" / "analysis" / "global_biases"
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "synthesis.json").write_text(
            json.dumps({"headline_finding": "Warm"})
        )

        # Mock OpenAI responses
        structure = _sample_structure()
        mock_client = MagicMock()
        mock_client.chat_json.side_effect = [
            structure,  # Stage 1
            _sample_written_sections()[0],  # Stage 2, section 1
            _sample_written_sections()[1],  # Stage 2, section 2
        ]

        with patch("feather.export.report.OpenAIClient", return_value=mock_client):
            gen = ReportGenerator(cfg, api_key="fake-key")
            gen.client = mock_client
            tex_path = gen.run(skip_existing=False)

        assert tex_path.exists()
        content = tex_path.read_text()
        assert "\\begin{document}" in content
        assert "Test Report" in content

    def test_fix_diagnostic_names(self, tmp_path):
        """Test that wrong diagnostic names from LLM are corrected."""
        cfg = _report_config(tmp_path)
        pub_dir = tmp_path / "output" / "publication"
        pub_dir.mkdir(parents=True)

        # Structure with WRONG diagnostic names (as the LLM might return)
        structure = {
            "title": "Report",
            "abstract": "Abstract.",
            "introduction": "Intro.",
            "sections": [
                {
                    "section_id": "01_temp",
                    "title": "Temperature",
                    "narrative_hook": "Hook.",
                    "figure_ids": ["avg_2t_annual_bias_combined"],
                    "diagnostics": ["2 m temperature annual mean bias (ERA5)"],
                },
                {
                    "section_id": "02_rad",
                    "title": "Radiation",
                    "narrative_hook": "Hook.",
                    "figure_ids": ["radiation_budget_bars"],
                    "diagnostics": ["Global mean radiation budget (CERES)"],
                },
            ],
            "selected_figures": [
                {
                    "diagnostic": "2 m temperature annual mean bias (ERA5)",
                    "figure_id": "avg_2t_annual_bias_combined",
                    "caption": "T2M bias.",
                    "label": "fig:t2m",
                },
                {
                    "diagnostic": "Global mean radiation budget (CERES)",
                    "figure_id": "radiation_budget_bars",
                    "caption": "Budget bars.",
                    "label": "fig:budget",
                },
                {
                    "diagnostic": "global_biases",
                    "figure_id": "avg_msl_annual_bias_combined",
                    "caption": "MSLP bias.",
                    "label": "fig:mslp",
                },
                {
                    "diagnostic": "timeseries",
                    "figure_id": "avg_2t_timeseries",
                    "caption": "T2M timeseries.",
                    "label": "fig:ts",
                },
            ],
            "conclusion": "Conclusion.",
        }

        # Figure metadata keyed by actual directory names
        figure_metadata = {
            "global_biases": [
                {"figure_id": "avg_2t_annual_bias_combined"},
                {"figure_id": "avg_msl_annual_bias_combined"},
            ],
            "radiation_budget": [
                {"figure_id": "radiation_budget_bars"},
            ],
            "timeseries": [
                {"figure_id": "avg_2t_timeseries"},
            ],
        }

        n_fixed = ReportGenerator._fix_diagnostic_names(structure, figure_metadata)

        # Two figures had wrong diagnostic names
        assert n_fixed == 2

        # Check all diagnostics are now correct directory names
        for fig in structure["selected_figures"]:
            assert fig["diagnostic"] in ("global_biases", "radiation_budget", "timeseries"), (
                f"diagnostic '{fig['diagnostic']}' not fixed"
            )

        # Check sections[].diagnostics are also fixed
        assert structure["sections"][0]["diagnostics"] == ["global_biases"]
        assert structure["sections"][1]["diagnostics"] == ["radiation_budget"]

    def test_fix_diagnostic_names_noop(self, tmp_path):
        """Test that already-correct names are not changed."""
        structure = _sample_structure()
        figure_metadata = {
            "global_biases": [
                {"figure_id": "fig_t2m_bias"},
                {"figure_id": "fig_precip_bias"},
            ],
            "timeseries": [
                {"figure_id": "fig_ts_1"},
                {"figure_id": "fig_ts_2"},
            ],
        }

        n_fixed = ReportGenerator._fix_diagnostic_names(structure, figure_metadata)
        assert n_fixed == 0

    def test_stage1_caching_fixes_diagnostics(self, tmp_path):
        """Test that cached structure.json also gets diagnostic names fixed."""
        cfg = _report_config(tmp_path)
        pub_dir = tmp_path / "output" / "publication"
        pub_dir.mkdir(parents=True)

        # Write cached structure with WRONG diagnostic name
        structure = _sample_structure()
        structure["selected_figures"][0]["diagnostic"] = "Wrong Title Name"
        (pub_dir / "structure.json").write_text(json.dumps(structure))

        figure_metadata = {
            "global_biases": [
                {"figure_id": "fig_t2m_bias"},
                {"figure_id": "fig_precip_bias"},
            ],
            "timeseries": [
                {"figure_id": "fig_ts_1"},
                {"figure_id": "fig_ts_2"},
            ],
        }

        with patch("feather.export.report.OpenAIClient"):
            gen = ReportGenerator(cfg, api_key="fake-key")
        result = gen._stage1_curation({}, figure_metadata, {}, skip_existing=True)

        # The wrong name should be fixed
        assert result["selected_figures"][0]["diagnostic"] == "global_biases"

    def test_stage1_no_cache(self, tmp_path):
        """Test that stage1 calls OpenAI when no cache exists."""
        cfg = _report_config(tmp_path)
        pub_dir = tmp_path / "output" / "publication"
        pub_dir.mkdir(parents=True)

        structure = _sample_structure()
        mock_client = MagicMock()
        mock_client.chat_json.return_value = structure

        with patch("feather.export.report.OpenAIClient", return_value=mock_client):
            gen = ReportGenerator(cfg, api_key="fake-key")
            gen.client = mock_client
            result = gen._stage1_curation({}, {}, {}, skip_existing=True)

        assert result["title"] == "Test Report"
        mock_client.chat_json.assert_called_once()
        # Verify cache was written
        assert (pub_dir / "structure.json").exists()
