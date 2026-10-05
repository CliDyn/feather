"""Run-scoped provenance capture.

Feather already works out at a handful of choke points (loaders, the
regrid-method selection, the pole-safe regrid) everything needed to say
*how* a figure was produced, and then discards it.  This module keeps it.

Design
------
Provenance is captured as **events emitted into an active run scope**, not
attached to the data.  Diagnostics never see it:

- :func:`run_pipeline` opens a run scope (:func:`run_scope`), which writes one
  run record per invocation to ``{output_dir}/provenance/run_{run_id}.json``.
- Each diagnostic runs inside :func:`diagnostic_scope`; events emitted while
  it runs (:func:`emit`) are collected for it and written to
  ``{output_dir}/provenance/{run_id}/{diagnostic}.json``.
- :func:`~feather.diag.figure_meta.save_figure_with_metadata` asks
  :func:`figure_block` for the ``run_id`` and the events relevant to the
  figure being saved, and adds them to the JSON sidecar.

Data arrays are deliberately left alone: carrying the chain in
``DataArray.attrs`` would need ``keep_attrs=True`` globally, which makes
derived fields inherit stale ``units`` attributes that some diagnostics read
to decide on a K→°C conversion.  It would also not survive the nereus
interpolators, which return numpy arrays.

Two invariants:

- **Provenance can never fail a run.**  Every public function swallows its
  own errors (logged at DEBUG/WARNING) instead of raising.
- **Outside a run scope everything is a no-op**, so diagnostics called
  directly (e.g. from unit tests or notebooks) behave exactly as before and
  their sidecars are unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Bump when the layout of the run record or of a figure block changes.
SCHEMA_VERSION = 1

#: Packages whose versions can change feather's numbers.
_TRACKED_PACKAGES = (
    "numpy", "scipy", "xarray", "dask", "pandas", "nereus", "healpy",
    "cartopy", "matplotlib", "netCDF4", "zarr", "cftime", "gsw",
    "regionmask", "cmocean", "kerchunk", "fsspec", "google-genai",
)

#: Config keys whose values are redacted from the run record.
_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password)", re.I)

#: Per-figure cap on events written into the sidecar.  The complete list
#: is always in the per-diagnostic events file referenced by the block.
MAX_FIGURE_EVENTS = 50

_RUN: contextvars.ContextVar["RunRecorder | None"] = contextvars.ContextVar(
    "feather_provenance_run", default=None,
)
_DIAG: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "feather_provenance_diag", default=None,
)

#: Events emitted outside any diagnostic scope are filed under this name.
_RUN_LEVEL = "_run"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_key(event: dict) -> str:
    return json.dumps(event, sort_keys=True, default=str)


# ── Environment capture ──────────────────────────────────────────────


def git_info(path: str | Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty flag of the feather checkout (best effort)."""
    root = Path(path) if path else Path(__file__).resolve().parent.parent

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": None, "git_branch": None, "git_dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of *names* (``None`` when not installed)."""
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def redact(obj: Any) -> Any:
    """Return *obj* with values under secret-looking keys replaced."""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if isinstance(k, str) and _SECRET_KEY.search(k)
                and v not in (None, "") else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def resolved_config(config: Any) -> Any:
    """JSON-safe, redacted dict of the fully parsed config.

    Stores the configuration *as it was used*, not a path to a file that
    may be edited in place later.
    """
    try:
        raw = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(vars(config))
    except Exception:
        raw = {"repr": repr(config)}
    return json.loads(json.dumps(redact(raw), default=str))


def new_run_id(commit: str | None = None) -> str:
    """``YYYYmmddTHHMMSSmmmZ-<short commit>`` (``-nogit`` outside a checkout).

    Milliseconds keep two runs started in the same second distinct.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
    return f"{stamp}-{commit[:7] if commit else 'nogit'}"


# ── Recorder ─────────────────────────────────────────────────────────


class RunRecorder:
    """Collects the run record and per-diagnostic events for one invocation."""

    def __init__(
        self,
        output_dir: str | Path,
        record: dict[str, Any],
        *,
        hash_policy: str = "stat",
        hash_policy_obs: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.record = record
        self.run_id: str = record["run_id"]
        self.hash_policy = _check_policy(hash_policy)
        self.hash_policy_obs = _check_policy(hash_policy_obs or hash_policy)
        self._events: dict[str, list[dict]] = {}
        self._seen: dict[str, set[str]] = {}
        self._identities: dict[tuple, dict] = {}

    def policy_for(self, role: str) -> str:
        """Hash policy for inputs of *role* (``"obs"`` may differ)."""
        return self.hash_policy_obs if role == "obs" else self.hash_policy

    def identity(self, paths: tuple[str, ...], policy: str) -> dict:
        """Memoised :func:`input_identity` (each input set hashed once per run)."""
        key = (paths, policy)
        if key not in self._identities:
            self._identities[key] = input_identity(paths, policy)
        return self._identities[key]

    # Paths ---------------------------------------------------------------

    @property
    def provenance_dir(self) -> Path:
        return self.output_dir / "provenance"

    @property
    def record_path(self) -> Path:
        return self.provenance_dir / f"run_{self.run_id}.json"

    def events_path(self, diagnostic: str) -> Path:
        return self.provenance_dir / self.run_id / f"{diagnostic}.json"

    # Events --------------------------------------------------------------

    def add(self, scope: str, event: dict) -> None:
        """Append *event* under *scope*, dropping exact duplicates."""
        key = _event_key(event)
        seen = self._seen.setdefault(scope, set())
        if key in seen:
            return
        seen.add(key)
        self._events.setdefault(scope, []).append(event)

    def events(self, scope: str) -> list[dict]:
        return list(self._events.get(scope, ()))

    # Persistence ---------------------------------------------------------

    def write_record(self) -> None:
        _write_json(self.record_path, self.record)

    def write_events(self, scope: str) -> None:
        events = self._events.get(scope)
        if not events:
            return
        _write_json(self.events_path(scope), {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "diagnostic": scope,
            "events": events,
        })


# ── Sidecar digest ───────────────────────────────────────────────────

#: Sidecar fields that change on every save without changing the figure.
SIDECAR_VOLATILE = ("generated_at", "run_id", "provenance")


def sidecar_digest(metadata: dict[str, Any]) -> str:
    """SHA-256 of a sidecar's content, ignoring per-save fields.

    Binds an LLM interpretation to the exact figure metadata it was written
    against: re-saving an unchanged figure keeps the digest, a changed
    number changes it.
    """
    import hashlib

    stable = {k: v for k, v in metadata.items() if k not in SIDECAR_VOLATILE}
    blob = json.dumps(stable, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def file_digest(path: str | Path) -> str | None:
    """SHA-256 of a file's bytes (``None`` if unreadable)."""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def text_digest(text: str) -> str:
    return bytes_digest(text.encode())


def bytes_digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ── Input identity ───────────────────────────────────────────────────

#: Hash policies, cheapest first.
HASH_POLICIES = ("none", "stat", "content")

#: Stop walking a directory store (zarr, kerchunk) after this many files.
_MAX_WALK_FILES = 200_000

_CHUNK = 8 * 1024 * 1024


def _check_policy(policy: str) -> str:
    if policy not in HASH_POLICIES:
        raise ValueError(
            f"provenance hash policy must be one of {HASH_POLICIES}, got {policy!r}")
    return policy


def _walk(path: Path) -> Iterator[Path]:
    """*path* itself if a file, else every file below it, in sorted order."""
    if path.is_file():
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            yield Path(root) / name


def input_identity(paths: tuple[str, ...] | list[str], policy: str = "stat") -> dict:
    """Fingerprint a set of input files or directory stores.

    ``stat`` hashes the sorted ``(path, size, mtime_ns)`` of every file — a
    cheap, deterministic check that detects a republished archive or a
    replaced file (not a bitwise change preserving size and mtime).
    ``content`` hashes the bytes; only affordable for small inputs such as
    observation files.  ``none`` records nothing.

    Directory stores (zarr, kerchunk reference dirs) are walked; for
    kerchunk stores this fingerprints the *references*, not the archive
    they point at.
    """
    import hashlib

    if policy == "none" or not paths:
        return {"identity": None, "identity_method": policy}
    h = hashlib.sha256()
    n_files = 0
    n_bytes = 0
    missing = 0
    for p in sorted(str(x) for x in paths):
        path = Path(p)
        if not path.exists():
            missing += 1
            h.update(f"{p}\0missing\n".encode())
            continue
        for f in _walk(path):
            n_files += 1
            if n_files > _MAX_WALK_FILES:
                return {"identity": None, "identity_method": policy,
                        "identity_error": f"more than {_MAX_WALK_FILES} files"}
            st = f.stat()
            n_bytes += st.st_size
            if policy == "stat":
                h.update(f"{f}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
            else:
                h.update(f"{f}\0".encode())
                with open(f, "rb") as fh:
                    while chunk := fh.read(_CHUNK):
                        h.update(chunk)
    out = {
        "identity": f"{policy}:{h.hexdigest()}",
        "identity_method": policy,
        "n_bytes": n_bytes,
    }
    if missing:
        out["missing_paths"] = missing
    return out


def time_coverage(data: Any) -> dict[str, Any]:
    """First/last time stamp and length of *data*'s ``time`` axis.

    Records what the data says, independently of what the configuration
    claims — e.g. a store declared ``hist-1950`` that actually begins in
    1975.  Empty dict when there is no time axis.
    """
    try:
        if data is None or "time" not in getattr(data, "coords", {}):
            return {}
        t = np.asarray(data["time"].values).ravel()
        if t.size == 0:
            return {"n_times": 0}
        valid = t[~pd.isnull(t)]
        if valid.size == 0:
            return {"n_times": int(t.size), "time_coverage": None}
        return {
            "time_coverage": [_fmt_time(valid.min()), _fmt_time(valid.max())],
            "n_times": int(t.size),
        }
    except Exception as exc:
        return {"time_coverage_error": f"{type(exc).__name__}: {exc}"}


#: Cap on missing months listed individually in a ``read`` record.
_MAX_MISSING_LISTED = 36


def period_completeness(data: Any, period: Any) -> dict[str, Any]:
    """Monthly completeness of *data* within the requested *period*.

    For monthly data, returns the number of samples per calendar month
    (``n_per_month``), the number expected (one per year) and the missing
    year-months — the archive gaps that otherwise live only in prose.
    Empty for non-monthly or time-less data.
    """
    try:
        if not period or data is None or "time" not in getattr(data, "coords", {}):
            return {}
        y0, y1 = int(str(period[0])[:4]), int(str(period[1])[:4])
        t = np.asarray(data["time"].values).ravel()
        if t.dtype.kind == "M":
            idx = pd.DatetimeIndex(t)
            years, months = idx.year.to_numpy(), idx.month.to_numpy()
        else:
            years = np.array([v.year for v in t])
            months = np.array([v.month for v in t])
        sel = (years >= y0) & (years <= y1)
        ym = set(zip(years[sel].tolist(), months[sel].tolist()))
        if len(ym) != int(sel.sum()):
            return {}  # sub-monthly data: counts per month are not samples
        first = min(years.min(), y1) if years.size else y0
        last = max(years.max(), y0) if years.size else y1
        lo, hi = max(y0, int(first)), min(y1, int(last))
        n_per_month = [sum(1 for (_, m) in ym if m == mm) for mm in range(1, 13)]
        out: dict[str, Any] = {
            "n_per_month": n_per_month,
            "expected_per_month": y1 - y0 + 1,
        }
        # Gaps inside the span the data actually covers (an archive that
        # starts late is already visible in ``time_coverage``).
        missing = [f"{y}-{m:02d}" for y in range(lo, hi + 1) for m in range(1, 13)
                   if (y, m) not in ym]
        if missing:
            out["n_missing_months"] = len(missing)
            out["missing_months"] = missing[:_MAX_MISSING_LISTED]
        return out
    except Exception as exc:
        return {"completeness_error": f"{type(exc).__name__}: {exc}"}


def _fmt_time(v: Any) -> str:
    """ISO date of a datetime64 or cftime value."""
    if isinstance(v, np.datetime64):
        return str(np.datetime_as_string(v, unit="D"))
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)


def read_event(
    *,
    role: str,
    backend: str,
    variable: str | None,
    paths: Any = (),
    data: Any = None,
    **fields: Any,
) -> dict[str, Any] | None:
    """Assemble an input (``read``) record; ``None`` outside a run.

    *paths* may be a sequence or a zero-argument callable returning one
    (so a loader can defer listing files until a run is known to be
    recording).  *data* is the opened, **unsliced** array, from which the
    actual time coverage is read.
    """
    rec = _RUN.get()
    if rec is None:
        return None
    if callable(paths):
        paths = paths()
    paths = tuple(sorted(str(p) for p in (paths or ())))
    event: dict[str, Any] = {"role": role, "backend": backend}
    if variable:
        event["variable"] = variable
    event.update({k: v for k, v in fields.items() if v is not None})
    if paths:
        # The full list stays in the per-diagnostic events file (so
        # ``feather verify`` can re-fingerprint it); sidecars drop it.
        event["paths"] = list(paths)
        event["n_files"] = len(paths)
        event["first"] = paths[0]
        if len(paths) > 1:
            event["last"] = paths[-1]
            try:
                event["root"] = os.path.commonpath(paths)
            except ValueError:
                pass
        event.update(rec.identity(paths, rec.policy_for(role)))
    event.update(time_coverage(data))
    return event


def model_fields(config: Any, model: str) -> dict[str, Any]:
    """Declared identity of *model* from its config, for ``read`` records.

    Kept beside the loaded time coverage so the two can be compared: the
    declared experiment is an assertion, the time axis is evidence.
    """
    mc = getattr(config, "model_configs", {}).get(model)
    out: dict[str, Any] = {"name": model}
    if mc is None:
        return out
    try:
        declared = mc.experiment or config.get_experiment()
    except Exception:
        declared = getattr(mc, "experiment", None)
    out.update({
        "institution": getattr(mc, "institution", None) or None,
        "declared_experiment": declared or None,
        "variant": getattr(mc, "variant", None) or None,
        "member": getattr(mc, "member", None),
    })
    return {k: v for k, v in out.items() if v is not None}


def record_read(memo: dict, key: Any, *, period: Any = None, **kwargs: Any) -> None:
    """Emit a ``read`` record for a loader's cached source *key*.

    Loaders cache opened stores and are shared across diagnostics, so the
    record is built once per run (memoised in *memo*, which the loader owns)
    and re-emitted on every load — including cache hits — so each
    diagnostic sees the inputs it used.  *kwargs* go to :func:`read_event`.
    Never raises; a no-op outside a run.
    """
    rec = _RUN.get()
    if rec is None:
        return
    try:
        cached = memo.get(key)
        if cached is None or cached[0] != rec.run_id:
            event = read_event(**kwargs)
            if event is None:
                return
            cached = (rec.run_id, event)
            memo[key] = cached
        event = dict(cached[1])
        if period:
            event["period_requested"] = [str(p) for p in period]
            event.update(period_completeness(kwargs.get("data"), period))
        emit("read", **event)
    except Exception as exc:
        emit("read", variable=kwargs.get("variable"),
             backend=kwargs.get("backend"),
             error=f"{type(exc).__name__}: {exc}")


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        # Lustre (/work on Levante) occasionally loses track of a file it
        # has just created, so the rename fails with ENOENT.  Fall back to a
        # plain write: a non-atomic record beats no record.
        logger.debug("Atomic rename failed for %s; writing directly", path)
        tmp.unlink(missing_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


def current_run() -> RunRecorder | None:
    """The active :class:`RunRecorder`, or ``None`` outside a run scope."""
    return _RUN.get()


def current_run_id() -> str | None:
    rec = _RUN.get()
    return rec.run_id if rec is not None else None


def build_run_record(
    config: Any,
    *,
    run_id: str | None = None,
    argv: list[str] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the once-per-invocation run record."""
    from feather import __version__

    git = git_info()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or new_run_id(git.get("git_commit")),
        "status": "running",
        "started": _utcnow(),
        "finished": None,
        "feather_version": __version__,
        **git,
        "argv": list(sys.argv if argv is None else argv),
        "options": json.loads(json.dumps(redact(options or {}), default=str)),
        "config_resolved": resolved_config(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": package_versions(),
    }
    return record


# ── Scopes ───────────────────────────────────────────────────────────


@contextlib.contextmanager
def run_scope(
    config: Any,
    *,
    output_dir: str | Path | None = None,
    argv: list[str] | None = None,
    options: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    hash_policy: str = "stat",
    hash_policy_obs: str | None = None,
) -> Iterator[RunRecorder | None]:
    """Activate provenance capture for one pipeline invocation.

    Yields the :class:`RunRecorder` (``None`` if it could not be set up —
    the run then proceeds without provenance).  Nested calls reuse the
    outer recorder.
    """
    if _RUN.get() is not None:
        yield _RUN.get()
        return

    recorder = None
    token = None
    try:
        out = output_dir if output_dir is not None else getattr(config, "output_dir", "./output")
        record = build_run_record(config, argv=argv, options=options)
        if extra:
            record.update(json.loads(json.dumps(extra, default=str)))
        recorder = RunRecorder(out, record, hash_policy=hash_policy,
                               hash_policy_obs=hash_policy_obs)
        record["hash_policy"] = {"default": recorder.hash_policy,
                                 "obs": recorder.hash_policy_obs}
        recorder.write_record()
        token = _RUN.set(recorder)
        logger.info("Provenance run id: %s (%s)", recorder.run_id, recorder.record_path)
    except Exception:
        logger.warning("Provenance capture disabled for this run", exc_info=True)
        recorder = None

    status = "failed"
    try:
        yield recorder
        status = "finished"
    finally:
        if token is not None:
            _RUN.reset(token)
        if recorder is not None:
            try:
                recorder.write_events(_RUN_LEVEL)
                recorder.record["status"] = status
                recorder.record["finished"] = _utcnow()
                recorder.write_record()
            except Exception:
                logger.warning("Could not finalise provenance record", exc_info=True)


@contextlib.contextmanager
def diagnostic_scope(name: str) -> Iterator[None]:
    """Attribute events emitted inside the block to diagnostic *name*."""
    token = _DIAG.set(name)
    try:
        yield
    finally:
        _DIAG.reset(token)
        rec = _RUN.get()
        if rec is not None:
            try:
                rec.write_events(name)
            except Exception:
                logger.warning("Could not write provenance events for %s", name,
                               exc_info=True)


def update_record(**fields: Any) -> None:
    """Merge *fields* into the active run record (no-op outside a run)."""
    rec = _RUN.get()
    if rec is None:
        return
    try:
        rec.record.update(json.loads(json.dumps(fields, default=str)))
        rec.write_record()
    except Exception:
        logger.debug("provenance.update_record failed", exc_info=True)


# ── Emission ─────────────────────────────────────────────────────────


def emit(step: str, **fields: Any) -> None:
    """Record one provenance event in the active scope.

    A no-op outside a run scope.  Never raises: a value that cannot be
    serialised is stored as its ``str()``, and any other failure degrades
    to an ``{"step": ..., "error": ...}`` entry.
    """
    rec = _RUN.get()
    if rec is None:
        return
    scope = _DIAG.get() or _RUN_LEVEL
    try:
        event = json.loads(json.dumps({"step": step, **fields}, default=str))
    except Exception as exc:  # pragma: no cover - default=str makes this rare
        event = {"step": step, "error": f"unserialisable event: {exc}"}
    try:
        rec.add(scope, event)
    except Exception:
        logger.debug("provenance.emit failed", exc_info=True)


def _event_matches(event: dict, variables: set[str], figure_id: str) -> bool:
    """Whether *event* is relevant to a figure.

    Events without a ``variable`` apply to every figure of the diagnostic.
    Variable-tagged events match the figure's ``variables_used`` or — for
    derived quantities keyed by their own name, such as the radiation-budget
    ``dq_key`` fields — a ``{variable}_`` prefix of the figure id.
    """
    var = event.get("variable")
    if not var:
        return True
    return var in variables or figure_id.startswith(f"{var}_")


#: Exclusions that happen after a successful load (so override "used").
_DROPS_AFTER_LOAD = frozenset({"no_lat_lon"})


#: Event fields kept out of sidecars (bulky; full copy in the events file).
_SIDECAR_DROP = ("paths", "stores")


def _compact(event: dict) -> dict:
    return {k: v for k, v in event.items() if k not in _SIDECAR_DROP}


def summarise_benchmarks(events: list[dict]) -> list[dict]:
    """Collapse per-member ``benchmark_member`` events into summaries.

    A benchmark ensemble contributes one event per member and load; a
    figure only needs, per benchmark and variable, which members were used
    and which were excluded and why.  A member counts as used if any of its
    loads succeeded (exclusions can be season- or period-specific).
    """
    out: list[dict] = []
    groups: dict[tuple, dict] = {}
    for e in events:
        if e.get("step") != "benchmark_member":
            out.append(e)
            continue
        key = (e.get("benchmark"), e.get("variable"))
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"step": "benchmark", "benchmark": key[0],
                               "variable": key[1], "_used": {}, "_excl": {}}
            out.append(g)
        member = f"{e.get('model')}/{e.get('variant')}"
        if e.get("status") == "used":
            g["_used"][member] = e.get("time_coverage")
        else:
            g["_excl"].setdefault(member, e.get("reason"))
            if e.get("reason") in _DROPS_AFTER_LOAD:
                g["_used"].pop(member, None)
    for g in groups.values():
        used = g.pop("_used")
        excl = {m: r for m, r in g.pop("_excl").items() if m not in used}
        g["n_used"] = len(used)
        g["used"] = sorted(used)
        g["excluded"] = [{"member": m, "reason": r} for m, r in sorted(excl.items())]
    return out


def figure_block(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Provenance block for the figure described by *metadata*.

    Returns ``None`` outside a run scope (the sidecar is then written
    exactly as before).
    """
    rec = _RUN.get()
    if rec is None:
        return None
    try:
        diag = _DIAG.get() or metadata.get("diagnostic_name") or _RUN_LEVEL
        variables = set(metadata.get("variables_used") or ())
        figure_id = str(metadata.get("figure_id", ""))
        events = [e for e in rec.events(diag)
                  if _event_matches(e, variables, figure_id)]
        events += [e for e in rec.events(_RUN_LEVEL)
                   if _event_matches(e, variables, figure_id)]
        events = [_compact(e) for e in summarise_benchmarks(events)]
        block: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": rec.run_id,
            "run_record": os.path.relpath(rec.record_path, rec.output_dir),
            "events_file": os.path.relpath(rec.events_path(diag), rec.output_dir),
            "n_events": len(events),
            "events": events[:MAX_FIGURE_EVENTS],
        }
        if len(events) > MAX_FIGURE_EVENTS:
            block["truncated"] = True
        return block
    except Exception:
        logger.debug("provenance.figure_block failed", exc_info=True)
        return None


def netcdf_attrs(stem: str) -> dict[str, str] | None:
    """Global attributes recording provenance for a NetCDF export.

    *stem* is the file name without extension; exports are named
    ``{variable}_...``, so variable-tagged events are matched against it the
    same way a figure id is.  Returns ``None`` outside a run.
    """
    rec = _RUN.get()
    if rec is None:
        return None
    try:
        diag = _DIAG.get() or _RUN_LEVEL
        events = [e for e in rec.events(diag) + rec.events(_RUN_LEVEL)
                  if _event_matches(e, set(), stem)]
        events = [_compact(e) for e in summarise_benchmarks(events)]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": rec.run_id,
            "diagnostic": diag,
            "feather_version": rec.record.get("feather_version"),
            "git_commit": rec.record.get("git_commit"),
            "git_dirty": rec.record.get("git_dirty"),
            "run_record": os.path.relpath(rec.record_path, rec.output_dir),
            "events": events[:MAX_FIGURE_EVENTS],
        }
        if len(events) > MAX_FIGURE_EVENTS:
            payload["truncated"] = True
        commit = (rec.record.get("git_commit") or "nogit")[:10]
        return {
            "feather_run_id": rec.run_id,
            "feather_provenance": json.dumps(payload, default=str),
            "history": f"{_utcnow()}: written by feather "
                       f"{rec.record.get('feather_version')} ({commit}) "
                       f"diagnostic {diag}, run {rec.run_id}",
        }
    except Exception:
        logger.debug("provenance.netcdf_attrs failed", exc_info=True)
        return None
