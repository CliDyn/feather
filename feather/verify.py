"""``feather verify`` — check an output tree against its recorded provenance.

Turns recorded provenance into signals:

- **inputs**: re-fingerprint every recorded input and report drift (an
  archive republished, an obs file swapped in place, a file gone).
- **analyses**: flag LLM analyses written against a different version of
  their figure (``sidecar_sha256`` / ``figure_sha256`` mismatch) and
  analyses whose figure no longer exists.
- **regrid**: figures produced under a conservative→point-interpolation
  fallback, and runs whose nereus lacked conservative support while this
  environment has it (or vice versa).
- **coverage**: inputs whose time axis contradicts their declared
  experiment (e.g. ``hist-1950`` data that starts in 1975) or that miss
  months inside the requested period.
- **code**: figures produced from a dirty tree or a commit other than the
  current one, and figures with no provenance at all.

Examples::

    feather verify --config configs/eerie.yaml
    feather verify --output /path/out --checks analyses regrid --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from feather import provenance

CHECKS = ("inputs", "analyses", "regrid", "coverage", "code")

#: First simulated year implied by an experiment label.  Only labels listed
#: here are checked; anything else (piControl, amip variants, …) is skipped.
EXPERIMENT_START_YEAR = {
    "hist-1950": 1950,
    "hist-1975": 1975,
    "historical": 1850,
    "baseline_hist": 1990,
    "ssp119": 2015, "ssp126": 2015, "ssp245": 2015, "ssp370": 2015,
    "ssp585": 2015, "projections_ssp3-7.0": 2015,
    "highres-future": 2015,
}

#: Years of slack before a late start counts as a contradiction (archives
#: often begin with the first full year, or a spin-up month).
START_TOLERANCE_YEARS = 1


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def iter_sidecars(output_dir: Path) -> Iterator[tuple[str, Path, dict]]:
    """Yield ``(diagnostic, path, sidecar)`` for every figure sidecar."""
    figures = output_dir / "figures"
    if not figures.is_dir():
        return
    for diag_dir in sorted(p for p in figures.iterdir() if p.is_dir()):
        for path in sorted(diag_dir.glob("*.json")):
            data = _load(path)
            if isinstance(data, dict) and "figure_id" in data:
                yield diag_dir.name, path, data


def _events_files(output_dir: Path, sidecars: list[dict]) -> dict[str, list[dict]]:
    """Events of every (run, diagnostic) referenced by a current figure."""
    out: dict[str, list[dict]] = {}
    for side in sidecars:
        rel = (side.get("provenance") or {}).get("events_file")
        if rel and rel not in out:
            data = _load(output_dir / rel)
            out[rel] = (data or {}).get("events", [])
    return out


class Report:
    def __init__(self) -> None:
        self.issues: dict[str, list[dict]] = defaultdict(list)
        self.info: dict[str, Any] = {}

    def add(self, check: str, **fields: Any) -> None:
        self.issues[check].append(fields)

    @property
    def n_issues(self) -> int:
        return sum(len(v) for v in self.issues.values())


# ── Checks ───────────────────────────────────────────────────────────


def check_inputs(report: Report, events_by_file: dict[str, list[dict]],
                 *, allow_content: bool = True) -> None:
    seen: set[tuple] = set()
    n_checked = 0
    for events in events_by_file.values():
        for e in events:
            if e.get("step") != "read" or not e.get("identity") or not e.get("paths"):
                continue
            method = e.get("identity_method", "stat")
            if method == "content" and not allow_content:
                method = "skip"
            key = (tuple(e["paths"]), method, e["identity"])
            if key in seen or method == "skip":
                continue
            seen.add(key)
            n_checked += 1
            now = provenance.input_identity(e["paths"], method)
            if now.get("identity") != e["identity"]:
                report.add(
                    "inputs", role=e.get("role"),
                    source=e.get("name") or e.get("dataset") or e.get("catalog_key"),
                    variable=e.get("variable") or e.get("obs_variable"),
                    root=e.get("root") or e.get("first"),
                    recorded=e["identity"], current=now.get("identity"),
                    missing_paths=now.get("missing_paths"),
                )
    report.info["inputs_checked"] = n_checked


def check_analyses(report: Report, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    n_checked = n_unbound = 0
    if not analysis_dir.is_dir():
        report.info["analyses_checked"] = 0
        return
    for path in sorted(analysis_dir.glob("*/*_analysis.json")):
        diag = path.parent.name
        stem = path.name[: -len("_analysis.json")]
        side_path = output_dir / "figures" / diag / f"{stem}.json"
        png_path = output_dir / "figures" / diag / f"{stem}.png"
        if not side_path.exists():
            report.add("analyses", figure=f"{diag}/{stem}", problem="figure_missing")
            continue
        data = _load(path) or {}
        prov = data.get("provenance") or {}
        if not prov.get("sidecar_sha256"):
            n_unbound += 1
            continue
        n_checked += 1
        side = _load(side_path) or {}
        if provenance.sidecar_digest(side) != prov["sidecar_sha256"]:
            report.add("analyses", figure=f"{diag}/{stem}", problem="metadata_changed")
        elif prov.get("figure_sha256") and provenance.file_digest(png_path) != prov["figure_sha256"]:
            report.add("analyses", figure=f"{diag}/{stem}", problem="image_changed")
    report.info["analyses_checked"] = n_checked
    report.info["analyses_without_binding"] = n_unbound


def check_regrid(report: Report, sidecars: list[tuple[str, dict]],
                 runs: dict[str, dict]) -> None:
    for fig, side in sidecars:
        for e in (side.get("provenance") or {}).get("events", []):
            if e.get("step") == "regrid_method" and e.get("fallback"):
                report.add("regrid", figure=fig, variable=e.get("variable"),
                           requested=e.get("requested"), used=e.get("used"),
                           reason=e.get("reason"))
    try:
        from feather.diag.base import DiagnosticBase

        available_now = DiagnosticBase._conservative_available()
    except Exception:
        available_now = None
    report.info["nereus_conservative_available_now"] = available_now
    for run_id, run in runs.items():
        then = (run.get("regrid") or {}).get("nereus_conservative_available")
        if available_now is not None and then is not None and then != available_now:
            report.add("regrid", run_id=run_id,
                       problem="conservative_capability_differs",
                       recorded=then, now=available_now)


def _start_year(coverage: list | None) -> int | None:
    try:
        return int(str(coverage[0])[:4])
    except (TypeError, ValueError, IndexError):
        return None


def check_coverage(report: Report, sidecars: list[tuple[str, dict]],
                   events_by_file: dict[str, list[dict]]) -> None:
    seen: set[tuple] = set()
    all_events = [e for evs in events_by_file.values() for e in evs]
    all_events += [e for _, s in sidecars for e in (s.get("provenance") or {}).get("events", [])]
    for e in all_events:
        if e.get("step") != "read":
            continue
        source = e.get("name") or e.get("dataset") or e.get("catalog_key")
        declared = e.get("declared_experiment")
        start = _start_year(e.get("time_coverage"))
        expected = EXPERIMENT_START_YEAR.get(declared or "")
        if expected is not None and start is not None \
                and start > expected + START_TOLERANCE_YEARS:
            key = ("start", source, e.get("variable"))
            if key not in seen:
                seen.add(key)
                report.add("coverage", source=source, variable=e.get("variable"),
                           problem="starts_after_declared_experiment",
                           declared_experiment=declared,
                           implied_start=expected, data_start=start)
        if e.get("n_missing_months"):
            key = ("gap", source, e.get("variable"), tuple(e.get("period_requested") or ()))
            if key not in seen:
                seen.add(key)
                report.add("coverage", source=source, variable=e.get("variable"),
                           problem="missing_months_in_period",
                           period=e.get("period_requested"),
                           n_missing=e["n_missing_months"],
                           missing=e.get("missing_months"))


def check_code(report: Report, sidecars: list[tuple[str, dict]],
               runs: dict[str, dict]) -> None:
    head = provenance.git_info().get("git_commit")
    report.info["git_head"] = head
    no_prov = [fig for fig, s in sidecars if not s.get("provenance")]
    if no_prov:
        report.add("code", problem="figures_without_provenance",
                   n_figures=len(no_prov), examples=no_prov[:5])
    by_commit: Counter = Counter()
    dirty: Counter = Counter()
    for fig, s in sidecars:
        run = runs.get(s.get("run_id") or "")
        if not run:
            continue
        by_commit[run.get("git_commit")] += 1
        if run.get("git_dirty"):
            dirty[s["run_id"]] += 1
    report.info["figures_by_commit"] = {str(k)[:10]: v for k, v in by_commit.items()}
    for commit, n in by_commit.items():
        if head and commit and commit != head:
            report.add("code", problem="produced_by_other_commit",
                       commit=commit[:10], n_figures=n)
    for run_id, n in dirty.items():
        report.add("code", problem="produced_from_dirty_tree",
                   run_id=run_id, n_figures=n)


# ── Driver ───────────────────────────────────────────────────────────


def verify(output_dir: str | Path, checks: tuple[str, ...] = CHECKS,
           *, allow_content: bool = True) -> Report:
    output_dir = Path(output_dir)
    report = Report()
    sidecar_rows = [(f"{d}/{p.stem}", s) for d, p, s in iter_sidecars(output_dir)]
    report.info["figures"] = len(sidecar_rows)
    runs: dict[str, dict] = {}
    for p in (output_dir / "provenance").glob("run_*.json"):
        r = _load(p)
        if r and r.get("run_id"):
            runs[r["run_id"]] = r
    events_by_file = _events_files(output_dir, [s for _, s in sidecar_rows])

    if "inputs" in checks:
        check_inputs(report, events_by_file, allow_content=allow_content)
    if "analyses" in checks:
        check_analyses(report, output_dir)
    if "regrid" in checks:
        check_regrid(report, sidecar_rows, runs)
    if "coverage" in checks:
        check_coverage(report, sidecar_rows, events_by_file)
    if "code" in checks:
        check_code(report, sidecar_rows, runs)
    return report


def _print(report: Report, checks: tuple[str, ...]) -> None:
    info = report.info
    print(f"Figures: {info.get('figures', 0)}  |  inputs re-fingerprinted: "
          f"{info.get('inputs_checked', '-')}  |  analyses checked: "
          f"{info.get('analyses_checked', '-')}")
    for check in checks:
        issues = report.issues.get(check, [])
        mark = "OK " if not issues else "!! "
        print(f"\n{mark}{check}: {len(issues)} issue(s)")
        for issue in issues[:50]:
            print("   - " + ", ".join(f"{k}={v}" for k, v in issue.items() if v is not None))
        if len(issues) > 50:
            print(f"   … {len(issues) - 50} more (use --json)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="feather verify",
        description="Check figures, analyses and inputs against their "
                    "recorded provenance.",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML (used to find output_dir)")
    parser.add_argument("--output", default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS),
                        help="Checks to run (default: all)")
    parser.add_argument("--no-content", action="store_true",
                        help="Skip re-hashing inputs recorded with the "
                             "'content' policy (they are re-read in full)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with status 1 if any issue is found")
    args = parser.parse_args(argv)

    if args.output:
        out = Path(args.output)
    else:
        from feather.config import FeatherConfig

        out = Path(FeatherConfig.from_yaml(args.config).output_dir)

    checks = tuple(args.checks)
    report = verify(out, checks, allow_content=not args.no_content)
    if args.json:
        json.dump({"info": report.info, "issues": report.issues}, sys.stdout,
                  indent=2, default=str)
        print()
    else:
        _print(report, checks)
    return 1 if (args.strict and report.n_issues) else 0
