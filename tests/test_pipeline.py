"""Tests for the pipeline runner (feather.run)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feather.config import FeatherConfig
from feather.run import run_pipeline


@pytest.fixture
def pipeline_config(tmp_path):
    """FeatherConfig for pipeline tests."""
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


class TestRunPipeline:
    def test_returns_summary_dict(self, pipeline_config):
        """Pipeline returns a dict with expected keys."""
        result = run_pipeline(pipeline_config, steps=[])
        assert "figures" in result
        assert "analyses" in result
        assert "syntheses" in result
        assert "report" in result
        assert "site_dir" in result

    def test_no_steps(self, pipeline_config):
        """Running with empty steps list does nothing."""
        result = run_pipeline(pipeline_config, steps=[])
        assert result["figures"] == 0
        assert result["analyses"] == 0
        assert result["report"] is None
        assert result["site_dir"] is None

    def test_steps_string(self, pipeline_config):
        """Steps can be a single string."""
        # "website" step should work since it just scans (possibly empty) dirs
        result = run_pipeline(pipeline_config, steps="website")
        assert result["site_dir"] is not None

    def test_all_expands(self, pipeline_config):
        """'all' should expand to all four steps."""
        with patch("feather.run._run_diagnostics", return_value=0) as mock_diag, \
             patch("feather.llm.analyzer.FigureAnalyzer") as mock_analyzer, \
             patch("feather.export.report.ReportGenerator") as mock_report, \
             patch("feather.website.generator.SiteGenerator") as mock_site:

            mock_analyzer_inst = MagicMock()
            mock_analyzer_inst.run.return_value = {
                "figure_analyses": 5, "syntheses": 2,
            }
            mock_analyzer.return_value = mock_analyzer_inst

            mock_report_inst = MagicMock()
            mock_report_inst.run.return_value = Path("/fake/report.tex")
            mock_report.return_value = mock_report_inst

            mock_site_inst = MagicMock()
            mock_site_inst.build.return_value = Path("/fake/site")
            mock_site.return_value = mock_site_inst

            result = run_pipeline(pipeline_config, steps="all")

            mock_diag.assert_called_once()
            mock_analyzer_inst.run.assert_called_once()
            mock_report_inst.run.assert_called_once()
            mock_site_inst.build.assert_called_once()

            assert result["analyses"] == 5
            assert result["syntheses"] == 2

    def test_website_only(self, pipeline_config):
        """Website step runs independently."""
        result = run_pipeline(pipeline_config, steps=["website"])
        assert result["site_dir"] is not None
        assert result["figures"] == 0

    def test_analyze_step(self, pipeline_config):
        """Analyze step creates FigureAnalyzer and calls run."""
        with patch("feather.llm.analyzer.FigureAnalyzer") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.run.return_value = {
                "figure_analyses": 3, "syntheses": 1,
            }
            mock_cls.return_value = mock_inst

            result = run_pipeline(
                pipeline_config, steps=["analyze"], api_key="fake",
            )

            mock_cls.assert_called_once_with(pipeline_config, api_key="fake")
            assert result["analyses"] == 3
            assert result["syntheses"] == 1

    def test_report_step(self, pipeline_config):
        """Report step creates ReportGenerator and calls run."""
        with patch("feather.export.report.ReportGenerator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.run.return_value = Path("/fake/report.tex")
            mock_cls.return_value = mock_inst

            result = run_pipeline(
                pipeline_config, steps=["report"],
                openai_api_key="fake-openai",
            )

            mock_cls.assert_called_once_with(
                pipeline_config, compile_pdf=False, api_key="fake-openai",
            )
            assert result["report"] == Path("/fake/report.tex")

    def test_diagnostics_filter_passed(self, pipeline_config):
        """Diagnostic names filter is passed through."""
        with patch("feather.run._run_diagnostics", return_value=2) as mock_diag:
            result = run_pipeline(
                pipeline_config,
                steps=["diagnostics"],
                diagnostics=["global_biases"],
            )
            mock_diag.assert_called_once()
            call_kwargs = mock_diag.call_args
            assert call_kwargs.kwargs["diagnostics"] == ["global_biases"]

    def test_variables_filter_passed(self, pipeline_config):
        """Variables filter is passed through."""
        with patch("feather.run._run_diagnostics", return_value=1) as mock_diag:
            run_pipeline(
                pipeline_config,
                steps=["diagnostics"],
                variables=["tas"],
            )
            call_kwargs = mock_diag.call_args
            assert call_kwargs.kwargs["variables"] == ["tas"]

    def test_skip_existing_passed(self, pipeline_config):
        """skip_existing flag is forwarded to sub-steps."""
        with patch("feather.llm.analyzer.FigureAnalyzer") as mock_analyzer:
            mock_inst = MagicMock()
            mock_inst.run.return_value = {
                "figure_analyses": 0, "syntheses": 0,
            }
            mock_analyzer.return_value = mock_inst

            run_pipeline(
                pipeline_config, steps=["analyze"],
                api_key="fake", skip_existing=False,
            )

            mock_inst.run.assert_called_once_with(
                skip_existing=False, diagnostics=None,
            )

    def test_skip_existing_passed_to_diagnostics(self, pipeline_config):
        """skip_existing flag is forwarded to _run_diagnostics."""
        with patch("feather.run._run_diagnostics", return_value=0) as mock_diag:
            run_pipeline(
                pipeline_config, steps=["diagnostics"],
                skip_existing=False,
            )
            call_kwargs = mock_diag.call_args
            assert call_kwargs.kwargs["skip_existing"] is False

    def test_compile_pdf_passed(self, pipeline_config):
        """compile_pdf flag is forwarded to report step."""
        with patch("feather.export.report.ReportGenerator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.run.return_value = Path("/fake/report.tex")
            mock_cls.return_value = mock_inst

            run_pipeline(
                pipeline_config, steps=["report"],
                openai_api_key="fake", compile_pdf=True,
            )

            mock_cls.assert_called_once_with(
                pipeline_config, compile_pdf=True, api_key="fake",
            )

    def test_cmip6_loader_creation(self, tmp_path):
        """When CMIP6 is enabled, a CMIP6Loader is created."""
        cfg = FeatherConfig(
            model_catalogs={},
            models=["ifs-fesom"],
            obs_root="",
            obs_datasets={},
            cmip6={
                "enabled": True,
                "catalog_path": str(tmp_path / "fake.yaml"),
                "models": {},
            },
            dask={},
            nereus={},
            output_dir=str(tmp_path / "output"),
        )

        with patch("feather.data.loader.DataLoader"), \
             patch("feather.data.obs.ObsLoader"), \
             patch("feather.data.cmip6.CMIP6Loader") as mock_cmip6, \
             patch("feather.diag.registry.registered_names", return_value=["global_biases"]), \
             patch("feather.diag.registry.get_diagnostic") as mock_get:

            mock_cls = MagicMock()
            mock_inst = MagicMock()
            mock_inst.run.return_value = []
            mock_cls.return_value = mock_inst
            mock_get.return_value = mock_cls

            from feather.run import _run_diagnostics
            _run_diagnostics(cfg)

            mock_cmip6.assert_called_once_with(cfg)
