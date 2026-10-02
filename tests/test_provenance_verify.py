"""Tests for benchmark/coverage provenance and ``feather verify``."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather import provenance, verify
from feather.data.cmip6 import CMIP6Loader
from feather.data.cmor_loader import CMORLoader
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata
from tests.test_cmip6 import _make_config, _synth_cmip6_ds, _write_zarr
from tests.test_provenance_inputs import cmor_cfg  # noqa: F401  (fixture)


# ── Benchmark membership ─────────────────────────────────────────────


class TestBenchmarkMembers:
    def _loader(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_zarr(tmp_path, "ModelA", "r1i1p1f1", "Amon", "tas",
                    _synth_cmip6_ds(n_months=24))
        # ModelB only has 12 months -> partial coverage of 1990–1991
        _write_zarr(tmp_path, "ModelB", "r1i1p1f1", "Amon", "tas",
                    _synth_cmip6_ds(n_months=12))
        loader = CMIP6Loader(cfg)
        loader._zarr_dir = str(tmp_path / "zarr")
        return cfg, loader

    def test_verdicts_recorded_and_summarised(self, tmp_path):
        cfg, loader = self._loader(tmp_path)
        with provenance.run_scope(cfg) as rec:
            for model in ("ModelA", "ModelB", "ModelC"):
                loader.load_var("tas", model, variant="r1i1p1f1",
                                period=("1990", "1991"),
                                require_full_coverage=True)
        members = [e for e in rec.events("_run") if e["step"] == "benchmark_member"]
        status = {e["model"]: (e["status"], e.get("reason")) for e in members}
        assert status["ModelA"] == ("used", None)
        assert status["ModelB"] == ("excluded", "partial_coverage")
        assert status["ModelC"] == ("excluded", "no_store")
        used = next(e for e in members if e["model"] == "ModelA")
        assert used["stores"][0].endswith("ModelA_historical_r1i1p1f1_Amon_tas.zarr")
        assert used["time_coverage"] == ["1990-01-01", "1991-12-01"]

        (summary,) = provenance.summarise_benchmarks(members)
        assert summary["step"] == "benchmark"
        assert summary["n_used"] == 1
        assert summary["used"] == ["ModelA/r1i1p1f1"]
        assert {x["reason"] for x in summary["excluded"]} == {"partial_coverage", "no_store"}

    def test_sidecar_carries_summary_not_members(self, tmp_path):
        cfg, loader = self._loader(tmp_path)
        with provenance.run_scope(cfg):
            with provenance.diagnostic_scope("probe"):
                loader.load_var("tas", "ModelA", variant="r1i1p1f1")
                loader.load_var("tas", "ModelB", variant="r1i1p1f1")
                fig, _ = plt.subplots()
                meta = build_metadata("probe", "T", figure_id="tas_x",
                                      variables_used=["tas"], models=["M"])
                _, jp = save_figure_with_metadata(fig, meta, tmp_path / "f", "tas_x")
        steps = [e["step"] for e in json.loads(Path(jp).read_text())["provenance"]["events"]]
        assert "benchmark" in steps and "benchmark_member" not in steps

    def test_used_in_one_season_counts_as_used(self):
        evs = [
            {"step": "benchmark_member", "benchmark": "CMIP6", "variable": "tas",
             "model": "A", "variant": "r1", "status": "excluded",
             "reason": "no_timesteps_in_period"},
            {"step": "benchmark_member", "benchmark": "CMIP6", "variable": "tas",
             "model": "A", "variant": "r1", "status": "used"},
        ]
        (s,) = provenance.summarise_benchmarks(evs)
        assert s["used"] == ["A/r1"] and s["excluded"] == []

    def test_no_lat_lon_overrides_used(self):
        evs = [
            {"step": "benchmark_member", "benchmark": "B", "variable": "tas",
             "model": "A", "variant": "r1", "status": "used"},
            {"step": "benchmark_member", "benchmark": "B", "variable": "tas",
             "model": "A", "variant": "r1", "status": "excluded", "reason": "no_lat_lon"},
        ]
        (s,) = provenance.summarise_benchmarks(evs)
        assert s["n_used"] == 0 and s["excluded"][0]["reason"] == "no_lat_lon"

    def test_load_unchanged_by_recording(self, tmp_path):
        cfg, loader = self._loader(tmp_path)
        out = loader.load_var("tas", "ModelA", variant="r1i1p1f1")
        with provenance.run_scope(cfg):
            inside = loader.load_var("tas", "ModelA", variant="r1i1p1f1")
        np.testing.assert_array_equal(out.values, inside.values)


# ── Period completeness ──────────────────────────────────────────────


class TestCompleteness:
    def test_gap_reported_on_read(self, cmor_cfg):  # noqa: F811
        loader = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            loader.load_var("M", "tas", period=("1975", "1976"))
        ev = rec.events("_run")[0]
        # Data starts 1975-02: January 1975 is the only gap.
        assert ev["n_per_month"] == [1] + [2] * 11
        assert ev["expected_per_month"] == 2
        assert ev["missing_months"] == ["1975-01"]

    def test_daily_data_skipped(self):
        d = xr.DataArray(np.zeros(60), dims="time",
                         coords={"time": xr.date_range("1990-01-01", periods=60, freq="D")})
        assert provenance.period_completeness(d, ("1990", "1990")) == {}

    def test_cftime(self):
        import cftime

        t = [cftime.Datetime360Day(1990, m, 16) for m in range(1, 13) if m != 6]
        d = xr.DataArray(np.zeros(len(t)), dims="time", coords={"time": t})
        out = provenance.period_completeness(d, ("1990", "1990"))
        assert out["missing_months"] == ["1990-06"]


# ── feather verify ───────────────────────────────────────────────────


@pytest.fixture
def produced(cmor_cfg, tmp_path):  # noqa: F811
    """An output tree produced by one recorded run over the CMOR fixture."""
    loader = CMORLoader(cmor_cfg)
    out = Path(cmor_cfg.output_dir)
    with provenance.run_scope(cmor_cfg) as rec:
        with provenance.diagnostic_scope("probe"):
            loader.load_var("M", "tas", period=("1975", "1976"))
            provenance.emit("regrid_method", variable="tas", requested="conservative",
                            used="linear", fallback=True, reason="cost_guard: x")
            fig, _ = plt.subplots()
            meta = build_metadata("probe", "T", figure_id="tas_bias",
                                  variables_used=["tas"], models=["M"],
                                  summary_statistics={"bias": 1.0})
            png, side = save_figure_with_metadata(fig, meta, out / "figures" / "probe", "tas_bias")
    return out, rec, png, side


def _issues(report, check):
    return report.issues.get(check, [])


class TestVerify:
    def test_clean_inputs(self, produced):
        out, *_ = produced
        report = verify.verify(out, ("inputs",))
        assert report.info["inputs_checked"] == 1
        assert _issues(report, "inputs") == []

    def test_input_drift_detected(self, produced):
        out, rec, *_ = produced
        read = next(e for e in rec.events("probe") if e["step"] == "read")
        f = Path(read["paths"][0])
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        (issue,) = _issues(verify.verify(out, ("inputs",)), "inputs")
        assert issue["source"] == "M" and issue["variable"] == "tas"

    def test_removed_input_detected(self, produced):
        out, rec, *_ = produced
        read = next(e for e in rec.events("probe") if e["step"] == "read")
        Path(read["paths"][-1]).unlink()
        (issue,) = _issues(verify.verify(out, ("inputs",)), "inputs")
        assert issue["missing_paths"] == 1

    def test_regrid_fallback_flagged(self, produced):
        out, *_ = produced
        issues = _issues(verify.verify(out, ("regrid",)), "regrid")
        assert any(i.get("figure") == "probe/tas_bias" and i["used"] == "linear"
                   for i in issues)

    def test_declared_experiment_contradiction(self, produced):
        out, *_ = produced
        issues = _issues(verify.verify(out, ("coverage",)), "coverage")
        start = [i for i in issues if i["problem"] == "starts_after_declared_experiment"]
        assert start and start[0]["declared_experiment"] == "hist-1950"
        assert start[0]["data_start"] == 1975
        gaps = [i for i in issues if i["problem"] == "missing_months_in_period"]
        assert gaps and gaps[0]["missing"] == ["1975-01"]

    def test_analyses(self, produced):
        out, _, png, side = produced
        adir = out / "analysis" / "probe"
        adir.mkdir(parents=True)
        bound = {"summary": "s", "provenance": {
            "sidecar_sha256": provenance.sidecar_digest(json.loads(Path(side).read_text())),
            "figure_sha256": provenance.file_digest(png)}}
        (adir / "tas_bias_analysis.json").write_text(json.dumps(bound))
        (adir / "gone_analysis.json").write_text(json.dumps({"summary": "s"}))
        issues = _issues(verify.verify(out, ("analyses",)), "analyses")
        assert issues == [{"figure": "probe/gone", "problem": "figure_missing"}]

        meta = json.loads(Path(side).read_text())
        meta["summary_statistics"] = {"bias": 2.0}
        Path(side).write_text(json.dumps(meta))
        issues = _issues(verify.verify(out, ("analyses",)), "analyses")
        assert {"figure": "probe/tas_bias", "problem": "metadata_changed"} in issues

    def test_code_check_flags_figures_without_provenance(self, produced):
        out, *_ = produced
        fig, _ = plt.subplots()
        save_figure_with_metadata(
            fig, build_metadata("probe", "T", figure_id="old", variables_used=["tas"],
                                models=["M"]), out / "figures" / "probe", "old")
        issues = _issues(verify.verify(out, ("code",)), "code")
        assert any(i["problem"] == "figures_without_provenance" and i["n_figures"] == 1
                   for i in issues)

    def test_cli_strict_exit_code(self, produced, capsys):
        from feather.cli import main

        out, *_ = produced
        assert main(["verify", "--output", str(out), "--checks", "regrid"]) == 0
        assert main(["verify", "--output", str(out), "--checks", "regrid",
                     "--strict"]) == 1
        assert "regrid" in capsys.readouterr().out

    def test_cli_json(self, produced, capsys):
        from feather.verify import main

        out, *_ = produced
        main(["--output", str(out), "--json", "--checks", "inputs", "coverage"])
        data = json.loads(capsys.readouterr().out)
        assert data["info"]["inputs_checked"] == 1
        assert data["issues"]["coverage"]
