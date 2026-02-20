"""Tests for feather.website — static site generator."""

import json
from pathlib import Path

import pytest

from feather.config import FeatherConfig
from feather.website.generator import SiteGenerator, _humanize, _load_json


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(tmp_path, **overrides):
    """Create a minimal FeatherConfig for website tests."""
    defaults = dict(
        model_catalogs={},
        models=["ifs-fesom", "ifs-nemo", "icon"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
        website={
            "title": "Test Dashboard",
            "subtitle": "Unit tests",
            "group_order": ["temperature", "radiation"],
            "group_labels": {
                "temperature": "Temperature",
                "radiation": "Radiation Budget",
            },
        },
    )
    defaults.update(overrides)
    return FeatherConfig(**defaults)


def _create_figure(
    figures_dir, diag_name, stem, metadata=None, create_png=True
):
    """Create a dummy PNG and/or JSON sidecar in figures/{diag_name}/."""
    diag_dir = figures_dir / diag_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    if create_png:
        # 1x1 pixel PNG (minimal valid PNG)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
            b"\x00\x00\x00IEND\xaeB`\x82"
        )
        (diag_dir / f"{stem}.png").write_bytes(png_bytes)

    if metadata is not None:
        (diag_dir / f"{stem}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    return diag_dir


def _create_analysis(analysis_dir, diag_name, stem, analysis_data):
    """Create an analysis JSON file."""
    out_dir = analysis_dir / diag_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_analysis.json"
    path.write_text(json.dumps(analysis_data), encoding="utf-8")
    return path


def _create_synthesis(analysis_dir, diag_name, synthesis_data):
    """Create a synthesis JSON file."""
    out_dir = analysis_dir / diag_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthesis.json"
    path.write_text(json.dumps(synthesis_data), encoding="utf-8")
    return path


# ── Init Tests ───────────────────────────────────────────────────────


class TestInit:
    def test_paths(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        assert gen.figures_dir == Path(cfg.output_dir) / "figures"
        assert gen.analysis_dir == Path(cfg.output_dir) / "analysis"
        assert gen.site_dir == Path(cfg.output_dir) / "site"

    def test_jinja_env(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        assert gen.env is not None
        # Templates should be loadable
        tpl = gen.env.get_template("base.html")
        assert tpl is not None

    def test_default_website_config(self, tmp_path):
        cfg = _make_config(tmp_path, website={})
        gen = SiteGenerator(cfg)
        # Should work with empty website config
        assert gen.config.website == {}


# ── collect_diagnostics Tests ────────────────────────────────────────


class TestCollectDiagnostics:
    def test_empty_figures_dir(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        # figures dir doesn't exist
        result = gen.collect_diagnostics()
        assert result == []

    def test_empty_dir_exists(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        gen.figures_dir.mkdir(parents=True)
        result = gen.collect_diagnostics()
        assert result == []

    def test_single_diagnostic(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        figures_dir = gen.figures_dir

        meta = {
            "title": "T2M Annual Bias",
            "group": "temperature",
            "variables_used": ["avg_2t"],
            "models": ["ifs-fesom"],
            "units": "K",
        }
        _create_figure(figures_dir, "global_biases", "t2m_bias", meta)

        result = gen.collect_diagnostics()
        assert len(result) == 1
        diag = result[0]
        assert diag["name"] == "global_biases"
        assert diag["group"] == "temperature"
        assert len(diag["figures"]) == 1
        assert diag["figures"][0]["stem"] == "t2m_bias"
        assert diag["thumbnail"] == "t2m_bias.png"
        assert diag["has_analysis"] is False
        assert diag["has_cmip6"] is False

    def test_multiple_diagnostics(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        figures_dir = gen.figures_dir

        _create_figure(
            figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )
        _create_figure(
            figures_dir, "timeseries", "fig2",
            {"title": "Fig 2", "group": "temperature"},
        )

        result = gen.collect_diagnostics()
        assert len(result) == 2
        names = [d["name"] for d in result]
        assert "global_biases" in names
        assert "timeseries" in names

    def test_with_analysis(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "t2m_bias",
            {"title": "T2M Bias", "group": "temperature"},
        )
        _create_analysis(
            gen.analysis_dir, "global_biases", "t2m_bias",
            {"summary": "A warm bias.", "confidence": "high",
             "key_findings": ["f1", "f2"]},
        )
        _create_synthesis(
            gen.analysis_dir, "global_biases",
            {"headline_finding": "Overall warm bias",
             "narrative": "Details here."},
        )

        result = gen.collect_diagnostics()
        assert len(result) == 1
        diag = result[0]
        assert diag["has_analysis"] is True
        assert diag["synthesis"]["headline_finding"] == "Overall warm bias"
        assert diag["figures"][0]["analysis"]["summary"] == "A warm bias."

    def test_with_cmip6_info(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        cmip6_info = {
            "n_members": 10,
            "models_used": {
                "tas": ["CanESM5/r1i1p1f1", "MPI-ESM1-2-LR/r1i1p1f1"]
            },
        }
        _create_figure(
            gen.figures_dir, "global_biases", "t2m_bias",
            {"title": "T2M Bias", "group": "temperature",
             "cmip6_info": cmip6_info},
        )

        result = gen.collect_diagnostics()
        diag = result[0]
        assert diag["has_cmip6"] is True
        assert diag["cmip6_info"]["n_members"] == 10
        assert "CanESM5" in diag["cmip6_model_names"]
        assert "MPI-ESM1-2-LR" in diag["cmip6_model_names"]

    def test_cmip6_models_used_as_list(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        cmip6_info = {
            "n_members": 4,
            "models_used": ["CanESM5/r1i1p1f1", "GISS-E2-1-G/r1i1p1f2"],
        }
        _create_figure(
            gen.figures_dir, "timeseries", "ts_fig",
            {"title": "TS", "group": "temperature",
             "cmip6_info": cmip6_info},
        )

        result = gen.collect_diagnostics()
        diag = result[0]
        assert diag["has_cmip6"] is True
        assert "CanESM5" in diag["cmip6_model_names"]

    def test_png_without_json(self, tmp_path):
        """PNG without sidecar JSON should still be collected."""
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "orphan",
            metadata=None, create_png=True,
        )

        result = gen.collect_diagnostics()
        assert len(result) == 1
        fig = result[0]["figures"][0]
        assert fig["metadata"] == {}
        assert fig["title"] == "Orphan"  # humanized from stem


# ── _group_diagnostics Tests ─────────────────────────────────────────


class TestGroupDiagnostics:
    def test_grouping(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        diags = [
            {"name": "d1", "group": "temperature"},
            {"name": "d2", "group": "radiation"},
            {"name": "d3", "group": "temperature"},
        ]

        groups = gen._group_diagnostics(diags)
        assert len(groups) == 2
        assert groups[0][0] == "temperature"
        assert groups[0][1] == "Temperature"
        assert len(groups[0][2]) == 2
        assert groups[1][0] == "radiation"
        assert groups[1][1] == "Radiation Budget"

    def test_ordering_follows_config(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        diags = [
            {"name": "d1", "group": "radiation"},
            {"name": "d2", "group": "temperature"},
        ]

        groups = gen._group_diagnostics(diags)
        # temperature comes before radiation in group_order
        assert groups[0][0] == "temperature"
        assert groups[1][0] == "radiation"

    def test_unknown_groups_appended(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        diags = [
            {"name": "d1", "group": "temperature"},
            {"name": "d2", "group": "exotic"},
        ]

        groups = gen._group_diagnostics(diags)
        assert len(groups) == 2
        assert groups[0][0] == "temperature"
        assert groups[1][0] == "exotic"
        assert groups[1][1] == "Exotic"  # humanized

    def test_empty_input(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        assert gen._group_diagnostics([]) == []


# ── build Tests ──────────────────────────────────────────────────────


class TestBuild:
    def test_creates_site_dir(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        site_dir = gen.build()
        assert site_dir.is_dir()
        assert (site_dir / "index.html").exists()

    def test_copies_static_assets(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        site_dir = gen.build()
        assert (site_dir / "static" / "style.css").exists()

    def test_copies_figures(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "t2m_bias",
            {"title": "T2M Bias", "group": "temperature"},
        )

        site_dir = gen.build()
        assert (
            site_dir / "figures" / "global_biases" / "t2m_bias.png"
        ).exists()

    def test_renders_diagnostic_pages(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )
        _create_figure(
            gen.figures_dir, "timeseries", "fig2",
            {"title": "Fig 2", "group": "radiation"},
        )

        site_dir = gen.build()
        assert (site_dir / "global_biases.html").exists()
        assert (site_dir / "timeseries.html").exists()

    def test_returns_site_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        result = gen.build()
        assert result == gen.site_dir

    def test_index_contains_diagnostic_title(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Global Temperature Bias", "group": "temperature"},
        )

        site_dir = gen.build()
        index_html = (site_dir / "index.html").read_text()
        assert "Global Biases" in index_html or "global_biases" in index_html

    def test_rebuild_overwrites(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        site1 = gen.build()
        # Create a marker file
        (site1 / "marker.txt").write_text("old")

        # Rebuild should remove old site
        site2 = gen.build()
        assert not (site2 / "marker.txt").exists()
        assert (site2 / "index.html").exists()

    def test_empty_output(self, tmp_path):
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)
        # No figures dir → site with just index
        site_dir = gen.build()
        assert (site_dir / "index.html").exists()


# ── Graceful Degradation Tests ───────────────────────────────────────


class TestGracefulDegradation:
    def test_no_analysis(self, tmp_path):
        """Works when analysis dir doesn't exist."""
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        site_dir = gen.build()
        diag_html = (site_dir / "global_biases.html").read_text()
        assert "not yet available" in diag_html

    def test_partial_analysis(self, tmp_path):
        """Works when only some figures have analysis."""
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )
        _create_figure(
            gen.figures_dir, "global_biases", "fig2",
            {"title": "Fig 2", "group": "temperature"},
        )
        _create_analysis(
            gen.analysis_dir, "global_biases", "fig1",
            {"summary": "Found bias.", "confidence": "high",
             "key_findings": ["warm", "cold"]},
        )

        result = gen.collect_diagnostics()
        assert len(result) == 1
        assert result[0]["figures"][0]["analysis"] != {}
        assert result[0]["figures"][1]["analysis"] == {}

    def test_empty_synthesis(self, tmp_path):
        """Works when synthesis.json is empty/invalid."""
        cfg = _make_config(tmp_path)
        gen = SiteGenerator(cfg)

        _create_figure(
            gen.figures_dir, "global_biases", "fig1",
            {"title": "Fig 1", "group": "temperature"},
        )

        synth_dir = gen.analysis_dir / "global_biases"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synthesis.json").write_text("{}")

        result = gen.collect_diagnostics()
        assert result[0]["synthesis"] == {}
        assert result[0]["has_analysis"] is False


# ── Helper Tests ─────────────────────────────────────────────────────


class TestHelpers:
    def test_humanize(self):
        assert _humanize("global_biases") == "Global Biases"
        assert _humanize("sst_patterns") == "Sst Patterns"
        assert _humanize("simple") == "Simple"

    def test_load_json_success(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}')
        assert _load_json(path) == {"key": "value"}

    def test_load_json_invalid(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert _load_json(path) == {}

    def test_load_json_missing(self, tmp_path):
        path = tmp_path / "missing.json"
        assert _load_json(path) == {}
