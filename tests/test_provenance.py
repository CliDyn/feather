"""Tests for run-scoped provenance capture (feather/provenance.py)."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather import provenance
from feather.config import FeatherConfig
from feather.diag.base import DiagnosticBase
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata

#: Sidecar fields that legitimately differ between two runs.
_VOLATILE = {"generated_at", "run_id", "provenance"}


def _cfg(tmp_path, **nereus):
    base = {"method": "linear"}
    base.update(nereus)
    return FeatherConfig(
        model_catalogs={}, models=["M"], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus=base,
        output_dir=str(tmp_path / "out"), data_source={"type": "cmor"},
        model_configs={},
    )


class _Probe(DiagnosticBase):
    name = "probe"
    title = "Probe"
    domain = "sfc"
    variables = ["tas"]
    group = "g"

    def compute(self):
        return {}

    def plot(self, results):
        return []


def _save(out, figure_id="tas_annual_bias", variables=("tas",)):
    fig, _ = plt.subplots()
    meta = build_metadata(
        "probe", "T", figure_id=figure_id, variables_used=list(variables),
        models=["M"],
    )
    _, json_path = save_figure_with_metadata(fig, meta, out, figure_id)
    return json.loads(Path(json_path).read_text())


def _run_record(cfg):
    files = sorted((Path(cfg.output_dir) / "provenance").glob("run_*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


# ── No-op outside a run ──────────────────────────────────────────────


class TestOutsideRun:
    def test_emit_is_noop(self):
        assert provenance.current_run() is None
        provenance.emit("anything", x=1)  # must not raise

    def test_sidecar_unchanged(self, tmp_path):
        side = _save(tmp_path)
        assert "run_id" not in side
        assert "provenance" not in side

    def test_figure_block_is_none(self):
        assert provenance.figure_block({"figure_id": "x"}) is None


# ── Run record ───────────────────────────────────────────────────────


class TestRunScope:
    def test_writes_run_record(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg, argv=["feather", "-v"],
                                  options={"steps": ["diagnostics"]}) as rec:
            assert provenance.current_run_id() == rec.run_id
        record = _run_record(cfg)
        assert record["run_id"] == rec.run_id
        assert record["status"] == "finished"
        assert record["finished"] is not None
        assert record["argv"] == ["feather", "-v"]
        assert record["options"] == {"steps": ["diagnostics"]}
        assert record["config_resolved"]["nereus"] == {"method": "linear"}
        assert "numpy" in record["packages"]
        assert record["schema_version"] == provenance.SCHEMA_VERSION
        assert provenance.current_run() is None

    def test_records_git_commit_of_checkout(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            pass
        record = _run_record(cfg)
        # Tests run from the feather checkout.
        assert record["git_commit"] and len(record["git_commit"]) == 40
        assert rec.run_id.endswith(record["git_commit"][:7])
        assert isinstance(record["git_dirty"], bool)

    def test_failed_run_is_marked(self, tmp_path):
        cfg = _cfg(tmp_path)
        with pytest.raises(RuntimeError):
            with provenance.run_scope(cfg):
                raise RuntimeError("boom")
        assert _run_record(cfg)["status"] == "failed"

    def test_secrets_redacted(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.llm = {"figure_analysis": {"api_key": "SECRET", "model": "m"}}
        with provenance.run_scope(cfg, options={"openai_api_key": "SK"}):
            pass
        text = (Path(cfg.output_dir) / "provenance").glob("run_*.json")
        text = next(text).read_text()
        assert "SECRET" not in text and "SK" not in text
        record = json.loads(text)
        assert record["options"]["openai_api_key"] == "<redacted>"
        assert record["config_resolved"]["llm"]["figure_analysis"]["model"] == "m"

    def test_empty_secret_not_marked(self):
        assert provenance.redact({"api_key": None}) == {"api_key": None}

    def test_nested_scope_reuses_outer(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as outer:
            with provenance.run_scope(cfg) as inner:
                assert inner is outer

    def test_setup_failure_degrades_to_no_provenance(self, tmp_path):
        cfg = _cfg(tmp_path)
        with patch.object(provenance, "build_run_record",
                          side_effect=RuntimeError("x")):
            with provenance.run_scope(cfg) as rec:
                assert rec is None
                provenance.emit("noop")


# ── Events and figure blocks ─────────────────────────────────────────


class TestEvents:
    def test_events_attributed_and_written(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            with provenance.diagnostic_scope("probe"):
                provenance.emit("read", variable="tas", path="/a")
                provenance.emit("read", variable="tas", path="/a")  # dup
                provenance.emit("read", variable="pr", path="/b")
        data = json.loads(rec.events_path("probe").read_text())
        assert [e["path"] for e in data["events"]] == ["/a", "/b"]
        assert data["run_id"] == rec.run_id

    def test_figure_block_filters_by_variable(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            with provenance.diagnostic_scope("probe"):
                provenance.emit("read", variable="tas")
                provenance.emit("read", variable="pr")
                provenance.emit("regrid", method="conservative")  # untagged
                side = _save(tmp_path / "figs")
        assert side["run_id"] == rec.run_id
        steps = [(e["step"], e.get("variable")) for e in side["provenance"]["events"]]
        assert ("read", "tas") in steps
        assert ("read", "pr") not in steps
        assert ("regrid", None) in steps
        assert side["provenance"]["events_file"] == f"provenance/{rec.run_id}/probe.json"

    def test_figure_id_prefix_matches_derived_quantity(self, tmp_path):
        """radiation_budget keys regrid decisions by derived names (dq_key)."""
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg):
            with provenance.diagnostic_scope("radiation_budget"):
                provenance.emit("regrid_method", variable="net_toa")
                side = _save(tmp_path / "f", figure_id="net_toa_annual_bias",
                             variables=("rsdt", "rsut"))
        assert side["provenance"]["events"][0]["variable"] == "net_toa"

    def test_truncation(self, tmp_path):
        cfg = _cfg(tmp_path)
        n = provenance.MAX_FIGURE_EVENTS + 5
        with provenance.run_scope(cfg):
            with provenance.diagnostic_scope("probe"):
                for i in range(n):
                    provenance.emit("read", variable="tas", i=i)
                side = _save(tmp_path / "f")
        block = side["provenance"]
        assert block["n_events"] == n
        assert len(block["events"]) == provenance.MAX_FIGURE_EVENTS
        assert block["truncated"] is True

    def test_unserialisable_value_stringified(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            provenance.emit("x", path=Path("/p"), arr=np.float32(1.5))
        assert rec.events("_run")[0]["path"] == "/p"

    def test_scopes_are_isolated(self, tmp_path):
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            with provenance.diagnostic_scope("a"):
                provenance.emit("read", variable="tas")
            with provenance.diagnostic_scope("b"):
                side = _save(tmp_path / "f")
        assert side["provenance"]["n_events"] == 0
        assert rec.events("a")


# ── Regrid decisions ─────────────────────────────────────────────────


class TestRegridMethodRecords:
    def _decide(self, tmp_path, var, monkeypatch=None, **kw):
        cfg = _cfg(tmp_path, **kw.pop("nereus", {}))
        probe = _Probe(None, None, cfg)
        with provenance.run_scope(cfg) as rec:
            with provenance.diagnostic_scope("probe"):
                used = probe._regrid_method_for(var, **kw)
        return used, rec.events("probe")[-1]

    def test_non_flux(self, tmp_path):
        used, ev = self._decide(tmp_path, "tas", default="nearest")
        assert used == "nearest"
        assert ev == {
            "step": "regrid_method", "variable": "tas",
            "requested": "nearest", "used": "nearest", "fallback": False,
            "reason": "non_flux", "n_src": None, "n_tgt": None,
            "target_resolution": None,
        }

    def test_conservative(self, tmp_path):
        used, ev = self._decide(tmp_path, "pr", n_source=1000, resolution=1.0)
        assert used == "conservative"
        assert ev["used"] == "conservative" and not ev["fallback"]
        assert ev["n_tgt"] == 360 * 180

    def test_cost_guard_fallback_recorded(self, tmp_path):
        used, ev = self._decide(
            tmp_path, "pr", n_source=6_480_000, resolution=0.25,
        )
        assert used == "linear"
        assert ev["requested"] == "conservative"
        assert ev["fallback"] is True
        assert ev["reason"].startswith("cost_guard")
        assert "6480000" in ev["reason"]

    def test_unavailable_fallback_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(DiagnosticBase, "_CONSERVATIVE_SUPPORTED", False)
        used, ev = self._decide(tmp_path, "pr")
        assert used == "linear"
        assert ev["fallback"] and ev["reason"] == "conservative_unavailable"

    def test_disabled_recorded(self, tmp_path):
        used, ev = self._decide(
            tmp_path, "pr", nereus={"conservative_fluxes": False},
        )
        assert used == "linear"
        assert ev["reason"] == "conservative_fluxes_disabled"

    def test_decision_unchanged_by_recording(self, tmp_path):
        """Same answers inside and outside a run scope."""
        cfg = _cfg(tmp_path)
        probe = _Probe(None, None, cfg)
        cases = [("tas", {}), ("pr", {}), ("pr", {"n_source": 9_000_000}),
                 ("rst", {"is_flux": False}), ("tas", {"default": "nearest"})]
        outside = [probe._regrid_method_for(v, **k) for v, k in cases]
        with provenance.run_scope(cfg):
            inside = [probe._regrid_method_for(v, **k) for v, k in cases]
        assert inside == outside


class TestConservativeRegridRecord:
    def test_pole_merge_recorded(self, tmp_path, monkeypatch):
        from feather.util import regrid as rg

        calls = {}

        def fake_regrid(data, lon, lat, method, **kw):
            calls["n"] = np.asarray(lon).size
            return np.zeros(3), object()

        import nereus as nr
        monkeypatch.setattr(nr, "regrid", fake_regrid)
        lon = np.array([0.0, 10.0, 20.0, 0.0, 90.0])
        lat = np.array([90.0, 90.0, 90.0, 0.0, 0.0])
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            rg.regrid(np.arange(5.0), lon=lon, lat=lat,
                      method="conservative", resolution=1.0)
        ev = rec.events("_run")[0]
        assert ev["step"] == "regrid"
        assert ev["n_src"] == 5
        assert ev["n_src_after_merge"] == 3 == calls["n"]
        assert ev["points_merged"] == 2
        assert ev["target_resolution"] == 1.0

    def test_non_conservative_not_recorded(self, tmp_path, monkeypatch):
        from feather.util import regrid as rg
        import nereus as nr

        monkeypatch.setattr(nr, "regrid", lambda *a, **k: (None, None))
        cfg = _cfg(tmp_path)
        with provenance.run_scope(cfg) as rec:
            rg.regrid(np.zeros(2), lon=np.zeros(2), lat=np.zeros(2),
                      method="nearest")
        assert rec.events("_run") == []


# ── Pipeline integration ─────────────────────────────────────────────


class TestPipeline:
    def test_run_pipeline_records_run(self, tmp_path):
        from feather.run import run_pipeline

        cfg = _cfg(tmp_path)
        with patch("feather.run._run_diagnostics", return_value=0):
            summary = run_pipeline(cfg, steps=["diagnostics"], api_key="K")
        record = _run_record(cfg)
        assert summary["run_id"] == record["run_id"]
        assert record["options"]["steps"] == ["diagnostics"]
        assert record["options"]["api_key"] == "<redacted>"
        assert "nereus_conservative_available" in record["regrid"]

    def test_failed_diagnostic_recorded(self, tmp_path):
        from feather.diag import registry
        from feather.run import run_pipeline

        class Boom(_Probe):
            name = "boom_prov"

            def __init__(self, *args, **kwargs):
                pass

            def run(self, skip_existing=True):
                raise ValueError("bad")

        cfg = _cfg(tmp_path)
        with patch.dict(registry._REGISTRY, {"boom_prov": Boom}):
            with patch("feather.data.obs.ObsLoader"), \
                    patch("feather.run._create_model_loader"):
                run_pipeline(cfg, steps=["diagnostics"],
                             diagnostics=["boom_prov"])
        record = _run_record(cfg)
        events = json.loads(
            (Path(cfg.output_dir) / "provenance" / record["run_id"]
             / "boom_prov.json").read_text())["events"]
        assert events[-1]["status"] == "failed"
        assert "ValueError: bad" in events[-1]["error"]


# ── Diagnostics produce identical results with provenance on ────────


def _sidecars(directory):
    out = {}
    for p in sorted(Path(directory).rglob("*.json")):
        if "provenance" in p.parts:
            continue
        d = json.loads(p.read_text())
        out[p.name] = {k: v for k, v in d.items() if k not in _VOLATILE}
    return out


@pytest.mark.parametrize("diag_path", [
    "feather.diag.global_biases.GlobalBiases",
    "feather.diag.timeseries.TimeseriesDiag",
    "feather.diag.seasonal_cycle.SeasonalCycleDiag",
])
def test_diagnostic_outputs_identical_with_provenance(
    diag_path, mock_model_loader, mock_obs_loader, minimal_config, tmp_path,
):
    import importlib

    mod, cls_name = diag_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod), cls_name)

    cfg_off = dataclasses.replace(minimal_config, output_dir=str(tmp_path / "off"))
    cfg_on = dataclasses.replace(minimal_config, output_dir=str(tmp_path / "on"))

    cls(mock_model_loader, mock_obs_loader, cfg_off, variables=["tas"]).run(
        skip_existing=False)
    with provenance.run_scope(cfg_on) as rec:
        with provenance.diagnostic_scope(cls.name):
            cls(mock_model_loader, mock_obs_loader, cfg_on,
                variables=["tas"]).run(skip_existing=False)

    off, on = _sidecars(tmp_path / "off"), _sidecars(tmp_path / "on")
    assert off and off == on

    # And the provenance actually landed in the "on" sidecars.
    for p in (tmp_path / "on").rglob("*.json"):
        if "provenance" in p.parts:
            continue
        side = json.loads(p.read_text())
        assert side["run_id"] == rec.run_id
        assert side["provenance"]["run_id"] == rec.run_id


# ── `feather provenance` CLI ─────────────────────────────────────────


class TestProvenanceCLI:
    def _make(self, tmp_path):
        cfg = _cfg(tmp_path)
        out = Path(cfg.output_dir)
        with provenance.run_scope(cfg, argv=["feather", "-v"]) as rec:
            with provenance.diagnostic_scope("probe"):
                provenance.emit("read", role="model", name="M", variable="tas",
                                n_files=2, first="/a", time_coverage=["1975-02-01", "2014-12-01"],
                                declared_experiment="hist-1950", identity="stat:abc")
                provenance.emit("regrid_method", variable="tas", requested="conservative",
                                used="linear", fallback=True, reason="cost_guard: x")
                _save(out / "figures" / "probe")
        return out, rec

    def test_figure_text(self, tmp_path, capsys):
        from feather.cli import main

        out, rec = self._make(tmp_path)
        assert main(["provenance", "--output", str(out),
                     "--figure", "probe/tas_annual_bias"]) == 0
        text = capsys.readouterr().out
        assert rec.run_id in text
        assert "1975-02-01→2014-12-01" in text
        assert "declared hist-1950" in text
        assert "FALLBACK" in text

    def test_figure_json_full(self, tmp_path, capsys):
        from feather.provenance_cli import main

        out, rec = self._make(tmp_path)
        main(["--output", str(out), "--figure", "probe/tas_annual_bias",
              "--full", "--json"])
        info = json.loads(capsys.readouterr().out)
        assert info["run"]["argv"] == ["feather", "-v"]
        assert {e["step"] for e in info["events"]} == {"read", "regrid_method"}

    def test_runs(self, tmp_path, capsys):
        from feather.provenance_cli import main

        out, rec = self._make(tmp_path)
        main(["--output", str(out), "--runs"])
        assert rec.run_id in capsys.readouterr().out

    def test_figure_without_provenance(self, tmp_path, capsys):
        from feather.provenance_cli import main

        _save(tmp_path / "figures" / "probe")
        main(["--output", str(tmp_path), "--figure", "probe/tas_annual_bias"])
        assert "no provenance recorded" in capsys.readouterr().out
