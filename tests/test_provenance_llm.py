"""Tests for interpretation provenance in the LLM analysis step."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from feather import provenance
from feather.llm.analyzer import FigureAnalyzer, _parse_json_with_repairs
from tests.test_llm import (
    _minimal_config,
    _setup_figures,
    _valid_analysis_dict,
    _valid_synthesis_dict,
)


def _response(payload, *, finish="STOP"):
    return SimpleNamespace(
        text=json.dumps(payload),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=4871, candidates_token_count=1123,
            thoughts_token_count=0, total_token_count=5994,
        ),
    )


def _client(mock_genai, *responses):
    client = MagicMock()
    client.models.generate_content.side_effect = list(responses)
    mock_genai.Client.return_value = client
    return client


@patch("feather.llm.analyzer.genai")
class TestInterpretationProvenance:
    def test_analysis_record(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        (png, side), = _setup_figures(tmp_path, n_figures=1)
        _client(mock_genai, _response(_valid_analysis_dict()))
        analyzer = FigureAnalyzer(cfg, api_key="k")

        analysis, prov = analyzer.analyze_figure_with_provenance(png, side)

        assert analysis.confidence == "high"
        assert prov["schema"] == "FigureAnalysis"
        assert prov["model"] == analyzer.model_name
        assert prov["tokens"] == {"input": 4871, "output": 1123,
                                  "thinking": 0, "total": 5994}
        assert prov["finish_reason"] == "STOP"
        assert prov["attempts"] == 1 and prov["failed_attempts"] == []
        assert prov["repairs"] == []
        assert prov["sidecar_sha256"] == provenance.sidecar_digest(
            json.loads(side.read_text()))
        assert prov["figure_sha256"] == provenance.bytes_digest(png.read_bytes())
        assert set(prov["prompt_sha256"]) == {"system", "user"}

    def test_retry_recorded(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        (png, side), = _setup_figures(tmp_path, n_figures=1)
        _client(mock_genai, RuntimeError("503"), _response(_valid_analysis_dict()))
        analyzer = FigureAnalyzer(cfg, api_key="k")
        analyzer.retry_delay = 0

        _, prov = analyzer.analyze_figure_with_provenance(png, side)
        assert prov["attempts"] == 2
        assert prov["failed_attempts"] == ["RuntimeError: 503"]

    def test_run_writes_provenance_and_keeps_it_out_of_synthesis(
        self, mock_genai, tmp_path,
    ):
        cfg = _minimal_config(tmp_path)
        _setup_figures(tmp_path, n_figures=2)
        client = _client(
            mock_genai,
            _response(_valid_analysis_dict()), _response(_valid_analysis_dict()),
            _response(_valid_synthesis_dict()),
        )
        analyzer = FigureAnalyzer(cfg, api_key="k")
        analyzer.run(skip_existing=False)

        saved = json.loads(
            (tmp_path / "analysis" / "global_biases" / "figure_0_analysis.json").read_text())
        assert saved["provenance"]["sidecar_sha256"]
        assert saved["summary"] == _valid_analysis_dict()["summary"]

        synth = json.loads(
            (tmp_path / "analysis" / "global_biases" / "synthesis.json").read_text())
        assert synth["provenance"]["schema"] == "DiagnosticSynthesis"
        assert synth["provenance"]["n_analyses"] == 2

        synth_prompt = client.models.generate_content.call_args_list[-1].kwargs["contents"][0]
        assert "sidecar_sha256" not in synth_prompt
        assert "provenance" not in synth_prompt

    def test_stale_analysis_warned(self, mock_genai, tmp_path, caplog):
        cfg = _minimal_config(tmp_path)
        (png, side), = _setup_figures(tmp_path, n_figures=1)
        _client(mock_genai, _response(_valid_analysis_dict()),
                _response(_valid_synthesis_dict()),
                _response(_valid_synthesis_dict()))
        analyzer = FigureAnalyzer(cfg, api_key="k")
        analyzer.run(skip_existing=False)

        # Re-saving with a new timestamp/run id is not a change ...
        meta = json.loads(side.read_text())
        meta["generated_at"] = "later"
        meta["run_id"] = "another-run"
        side.write_text(json.dumps(meta))
        with caplog.at_level(logging.WARNING):
            analyzer.run(skip_existing=True)
        assert "Stale analysis" not in caplog.text

        # ... a changed number is.
        meta["summary_statistics"] = {"global_mean_bias": 9.9}
        side.write_text(json.dumps(meta))
        with caplog.at_level(logging.WARNING):
            analyzer.run(skip_existing=True)
        assert "Stale analysis for figure_0" in caplog.text

    def test_run_id_recorded_inside_pipeline_run(self, mock_genai, tmp_path):
        cfg = _minimal_config(tmp_path)
        (png, side), = _setup_figures(tmp_path, n_figures=1)
        _client(mock_genai, _response(_valid_analysis_dict()))
        analyzer = FigureAnalyzer(cfg, api_key="k")
        with provenance.run_scope(cfg) as rec:
            _, prov = analyzer.analyze_figure_with_provenance(png, side)
        assert prov["run_id"] == rec.run_id


class TestParseRepairs:
    def test_clean(self):
        assert _parse_json_with_repairs('{"a": 1}') == ({"a": 1}, [])

    def test_fence(self):
        data, repairs = _parse_json_with_repairs('```json\n{"a": 1}\n```')
        assert data == {"a": 1} and repairs == ["strip_markdown_fence"]

    def test_latex_escape(self):
        data, repairs = _parse_json_with_repairs(r'{"a": "\Delta T"}')
        assert data == {"a": r"\Delta T"}
        assert repairs == ["fix_invalid_escapes"]

    def test_prose_wrapped(self):
        data, repairs = _parse_json_with_repairs('Here you go: {"a": 1} hope it helps')
        assert data == {"a": 1}
        assert repairs[-1] == "extract_outer_object"

    def test_unparseable_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_with_repairs("not json")


def test_sidecar_digest_ignores_volatile_fields():
    base = {"figure_id": "x", "summary_statistics": {"bias": 1.0}}
    d = provenance.sidecar_digest(base)
    assert provenance.sidecar_digest(
        {**base, "generated_at": "t", "run_id": "r", "provenance": {"a": 1}}) == d
    assert provenance.sidecar_digest({**base, "summary_statistics": {"bias": 1.1}}) != d
