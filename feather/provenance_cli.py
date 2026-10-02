"""``feather provenance`` — inspect the provenance of figures and runs.

Examples::

    feather provenance --config configs/eerie.yaml --runs
    feather provenance --config configs/eerie.yaml \\
        --figure global_biases/tas_annual_bias
    feather provenance --output /path/to/output --figure ... --full --json
    feather provenance --config ... --figure ... --export prov-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    from feather.config import FeatherConfig

    return Path(FeatherConfig.from_yaml(args.config).output_dir)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def figure_provenance(output_dir: Path, figure: str, *, full: bool = False) -> dict[str, Any]:
    """Collect everything recorded about *figure* (``diagnostic/figure_id``)."""
    diag, _, fid = figure.partition("/")
    if not fid:
        raise SystemExit(f"--figure must be DIAGNOSTIC/FIGURE_ID, got {figure!r}")
    sidecar_path = output_dir / "figures" / diag / f"{fid}.json"
    sidecar = _load_json(sidecar_path)
    if sidecar is None:
        raise SystemExit(f"No sidecar at {sidecar_path}")

    block = sidecar.get("provenance") or {}
    out: dict[str, Any] = {
        "figure": figure,
        "sidecar": str(sidecar_path),
        "generated_at": sidecar.get("generated_at"),
        "run_id": sidecar.get("run_id"),
    }
    if not block:
        out["note"] = ("no provenance recorded — figure predates provenance "
                       "capture or was produced outside run_pipeline")
        return out

    events = block.get("events", [])
    if full and block.get("events_file"):
        from feather.provenance import _event_matches, summarise_benchmarks

        data = _load_json(output_dir / block["events_file"])
        if data:
            variables = set(sidecar.get("variables_used") or ())
            events = summarise_benchmarks([
                e for e in data.get("events", [])
                if _event_matches(e, variables, fid)])
    out["events"] = events
    out["truncated"] = bool(block.get("truncated")) and not full

    record = _load_json(output_dir / block["run_record"]) if block.get("run_record") else None
    if record:
        out["run"] = {k: record.get(k) for k in (
            "run_id", "status", "started", "finished", "feather_version",
            "git_commit", "git_branch", "git_dirty", "argv", "hostname",
            "slurm_job_id", "hash_policy", "regrid",
        )}

    analysis = _load_json(output_dir / "analysis" / diag / f"{fid}_analysis.json")
    if analysis and analysis.get("provenance"):
        out["interpretation"] = analysis["provenance"]
    return out


def _fmt_event(e: dict) -> str:
    step = e.get("step")
    if e.get("error"):
        return f"[{step}] ERROR {e['error']}"
    if step == "read":
        who = e.get("name") or e.get("dataset") or e.get("catalog_key") or "?"
        bits = [f"{e.get('role', '?'):5s} {who}", e.get("variable") or e.get("obs_variable") or ""]
        if e.get("n_files"):
            bits.append(f"{e['n_files']} file(s) under {e.get('root') or e.get('first')}")
        if e.get("time_coverage"):
            bits.append("data " + "→".join(e["time_coverage"]))
        if e.get("declared_experiment"):
            bits.append(f"declared {e['declared_experiment']}")
        if e.get("identity"):
            bits.append(e["identity"][:20] + "…")
        return "[read] " + " | ".join(b for b in bits if b)
    if step == "regrid_method":
        flag = "  ⚠ FALLBACK" if e.get("fallback") else ""
        return (f"[regrid] {e.get('variable')}: requested {e.get('requested')} → "
                f"used {e.get('used')} ({e.get('reason')}){flag}")
    if step == "regrid":
        return (f"[regrid] conservative weights: {e.get('n_src')} source pts, "
                f"{e.get('points_merged')} coincident merged")
    if step == "benchmark":
        excl = ", ".join(f"{x['member']} ({x['reason']})" for x in e.get("excluded", []))
        return (f"[benchmark] {e.get('benchmark')} {e.get('variable')}: "
                f"{e.get('n_used')} member(s) used"
                + (f"; excluded: {excl}" if excl else ""))
    if step == "convert":
        return (f"[convert] {e.get('role')} {e.get('variable')}: "
                + ", ".join(f"{k}={e[k]}" for k in ("op", "factor", "offset", "to_units") if k in e))
    rest = {k: v for k, v in e.items() if k != "step"}
    return f"[{step}] {json.dumps(rest, default=str)}"


def _print_figure(info: dict) -> None:
    print(f"Figure:   {info['figure']}")
    print(f"Sidecar:  {info['sidecar']}")
    print(f"Saved:    {info.get('generated_at')}")
    print(f"Run:      {info.get('run_id')}")
    if "note" in info:
        print(f"\n{info['note']}")
        return
    run = info.get("run")
    if run:
        dirty = " (dirty tree)" if run.get("git_dirty") else ""
        print(f"Code:     feather {run.get('feather_version')} @ "
              f"{(run.get('git_commit') or '?')[:10]}{dirty} [{run.get('git_branch')}]")
        print(f"Command:  {' '.join(run.get('argv') or [])}")
        regrid = run.get("regrid") or {}
        if regrid:
            print(f"Regrid:   nereus conservative available="
                  f"{regrid.get('nereus_conservative_available')}, "
                  f"max_points={regrid.get('conservative_max_points')}")
    print()
    events = info.get("events", [])
    for kind, title in (("read", "Inputs"),
                        (("regrid_method", "regrid", "convert"), "Transformations"),
                        ("benchmark", "Benchmarks")):
        kinds = (kind,) if isinstance(kind, str) else kind
        rows = [e for e in events if e.get("step") in kinds]
        if rows:
            print(f"{title}:")
            for e in rows:
                print("  " + _fmt_event(e))
    others = [e for e in events
              if e.get("step") not in ("read", "regrid_method", "regrid", "convert",
                                       "benchmark")]
    if others:
        print("Other:")
        for e in others:
            print("  " + _fmt_event(e))
    if info.get("truncated"):
        print("\n(events truncated in the sidecar — use --full)")
    if info.get("interpretation"):
        print("\nInterpretation:")
        for k, v in info["interpretation"].items():
            print(f"  {k}: {v}")


def list_runs(output_dir: Path) -> list[dict]:
    runs = []
    for p in sorted((output_dir / "provenance").glob("run_*.json")):
        r = _load_json(p)
        if r:
            runs.append({k: r.get(k) for k in (
                "run_id", "status", "started", "finished", "git_commit",
                "git_dirty", "argv")})
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="feather provenance",
        description="Show how a figure was produced (inputs, regridding, "
                    "code version) from its recorded provenance.",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML (used to find output_dir)")
    parser.add_argument("--output", default=None,
                        help="Output directory (overrides config)")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--figure", help="DIAGNOSTIC/FIGURE_ID")
    what.add_argument("--runs", action="store_true",
                      help="List recorded pipeline runs")
    parser.add_argument("--full", action="store_true",
                        help="Read all events from the per-diagnostic file "
                             "instead of the (capped) sidecar copy")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--export", choices=["prov-json"], default=None,
                        help="Export the figure's provenance as W3C PROV-JSON")
    args = parser.parse_args(argv)

    out_dir = _output_dir(args)
    if args.runs:
        runs = list_runs(out_dir)
        if args.json:
            json.dump(runs, sys.stdout, indent=2, default=str)
            print()
        else:
            for r in runs:
                dirty = "*" if r.get("git_dirty") else ""
                print(f"{r['run_id']}  {r.get('status'):8s}  "
                      f"{(r.get('git_commit') or '?')[:10]}{dirty}  "
                      f"{' '.join(r.get('argv') or [])}")
        return 0

    info = figure_provenance(out_dir, args.figure, full=args.full)
    if args.export == "prov-json":
        from feather.prov_export import figure_to_prov

        json.dump(figure_to_prov(info), sys.stdout, indent=2, default=str)
        print()
    elif args.json:
        json.dump(info, sys.stdout, indent=2, default=str)
        print()
    else:
        _print_figure(info)
    return 0
