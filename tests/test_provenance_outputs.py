"""Tests for provenance in NetCDF exports, the dashboard and PROV-JSON."""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from feather import provenance
from feather.diag.netcdf_export import write_netcdf
from feather.prov_export import figure_to_prov
from feather.provenance_cli import figure_provenance
from feather.website.generator import SiteGenerator
from tests.test_provenance import _cfg
from tests.test_website import _create_analysis, _create_figure, _make_config


def _ds():
    return xr.Dataset({"tas_bias": ("x", np.arange(3.0))}, attrs={"history": "created"})


# ── NetCDF ───────────────────────────────────────────────────────────


class TestNetCDF:
    def test_outside_run_unchanged(self, tmp_path):
        p = write_netcdf(_ds(), tmp_path / "tas_annual_1980-2014.nc")
        with xr.open_dataset(p) as ds:
            assert "feather_provenance" not in ds.attrs
            assert ds.attrs["history"] == "created"

    def test_stamped_inside_run(self, tmp_path):
        cfg = _cfg(tmp_path)
        src = _ds()
        with provenance.run_scope(cfg) as rec:
            with provenance.diagnostic_scope("global_biases"):
                provenance.emit("read", role="model", variable="tas", name="M",
                                paths=["/a", "/b"])
                provenance.emit("read", role="model", variable="pr", name="M")
                p = write_netcdf(src, tmp_path / "tas_annual_1980-2014.nc")
        with xr.open_dataset(p) as ds:
            assert ds.attrs["feather_run_id"] == rec.run_id
            payload = json.loads(ds.attrs["feather_provenance"])
            hist = ds.attrs["history"].splitlines()
            np.testing.assert_array_equal(ds["tas_bias"].values, [0.0, 1.0, 2.0])
        assert payload["run_id"] == rec.run_id
        assert payload["diagnostic"] == "global_biases"
        assert [e["variable"] for e in payload["events"]] == ["tas"]
        assert "paths" not in payload["events"][0]
        assert hist[0] == "created" and rec.run_id in hist[1]
        # The caller's dataset is not mutated.
        assert "feather_provenance" not in src.attrs


# ── Dashboard ────────────────────────────────────────────────────────


def _meta(**prov_events):
    return {
        "title": "T", "group": "temperature", "figure_id": "tas_bias",
        "variables_used": ["tas"],
        "run_id": "RUN1",
        "provenance": {
            "run_id": "RUN1", "n_events": 3, "events_file": "provenance/RUN1/d.json",
            "events": [
                {"step": "read", "role": "model", "name": "ICON-r2", "variable": "tas",
                 "n_files": 35, "time_coverage": ["1975-02-15", "2014-12-15"],
                 "declared_experiment": "hist-1950", "n_missing_months": 5,
                 "root": "/work/icon"},
                {"step": "regrid_method", "variable": "pr", "requested": "conservative",
                 "used": "linear", "fallback": True, "reason": "cost_guard: big"},
                {"step": "benchmark", "benchmark": "CMIP6", "variable": "tas", "n_used": 2,
                 "used": ["A/r1", "B/r1"],
                 "excluded": [{"member": "C/r1", "reason": "partial_coverage"}]},
            ],
        },
    }


class TestDashboard:
    def test_panel_rendered(self, tmp_path):
        gen = SiteGenerator(_make_config(tmp_path))
        _create_figure(gen.figures_dir, "global_biases", "tas_bias", _meta())
        html = (gen.build() / "global_biases.html").read_text()
        assert 'class="provenance-panel"' in html
        assert "run RUN1" in html
        assert "regrid fallback" in html
        assert "1975-02-15–2014-12-15" in html
        assert "declared hist-1950" in html
        assert "5 month(s) missing" in html
        assert "C/r1 (partial coverage)" in html

    def test_no_panel_without_provenance(self, tmp_path):
        gen = SiteGenerator(_make_config(tmp_path))
        _create_figure(gen.figures_dir, "global_biases", "f",
                       {"title": "T", "group": "temperature"})
        html = (gen.build() / "global_biases.html").read_text()
        assert 'class="provenance-panel"' not in html

    def test_stale_analysis_flagged(self, tmp_path):
        gen = SiteGenerator(_make_config(tmp_path))
        meta = _meta()
        _create_figure(gen.figures_dir, "global_biases", "tas_bias", meta)
        fresh = {"summary": "S", "provenance": {"sidecar_sha256": provenance.sidecar_digest(meta)}}
        _create_analysis(gen.analysis_dir, "global_biases", "tas_bias", fresh)
        html = (gen.build() / "global_biases.html").read_text()
        assert "earlier version of the figure" not in html

        stale = {"summary": "S", "provenance": {"sidecar_sha256": "0" * 64}}
        _create_analysis(gen.analysis_dir, "global_biases", "tas_bias", stale)
        html = (gen.build() / "global_biases.html").read_text()
        assert "earlier version of the figure" in html


# ── PROV-JSON ────────────────────────────────────────────────────────


class TestProvJson:
    def test_figure_document(self, tmp_path):
        out = tmp_path
        diag = out / "figures" / "global_biases"
        diag.mkdir(parents=True)
        (diag / "tas_bias.json").write_text(json.dumps(_meta()))
        (out / "provenance").mkdir()
        (out / "provenance" / "run_RUN1.json").write_text(json.dumps(
            {"run_id": "RUN1", "git_commit": "abc123", "feather_version": "0.1.0",
             "started": "2026-10-02T00:00:00+00:00", "argv": ["feather"]}))
        adir = out / "analysis" / "global_biases"
        adir.mkdir(parents=True)
        (adir / "tas_bias_analysis.json").write_text(json.dumps(
            {"provenance": {"provider": "vertex", "model": "gemini-2.5-flash",
                            "schema": "FigureAnalysis", "sidecar_sha256": "x"}}))
        (out / "provenance" / "RUN1").mkdir()
        (out / "provenance" / "RUN1" / "d.json").write_text("{}")
        (diag / "tas_bias.json").write_text(json.dumps({**_meta(), "provenance": {
            **_meta()["provenance"], "run_record": "provenance/run_RUN1.json"}}))

        doc = figure_to_prov(figure_provenance(out, "global_biases/tas_bias"))

        types = {v["prov:type"] for v in doc["entity"].values()}
        assert {"feather:Figure", "feather:ModelInput", "feather:BenchmarkEnsemble",
                "feather:Interpretation"} <= types
        agents = {v["prov:label"] for v in doc["agent"].values()}
        assert agents == {"feather", "gemini-2.5-flash"}
        assert doc["agent"][next(k for k, v in doc["agent"].items()
                                 if v["prov:label"] == "feather")]["feather:git_commit"] == "abc123"
        fallback = [a for a in doc["activity"].values()
                    if a.get("prov:type") == "feather:regrid_method"]
        assert fallback[0]["feather:fallback"] is True
        assert len(doc["used"]) == 2  # input + benchmark
        assert len(doc["wasAttributedTo"]) == 1
        json.dumps(doc)  # serialisable
