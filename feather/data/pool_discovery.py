"""Discovery helpers for the DKRZ CMIP6 on-disk DRS tree.

The DKRZ CMIP6 pool (``/pool/data/CMIP6/data``) stores data as a CMIP6
Data Reference Syntax (DRS) directory tree::

    {activity}/{institution}/{source}/{experiment}/{member}/
        {table}/{variable}/{grid}/{version}/*.nc

These pure functions walk that tree to enumerate models, pick a single
ensemble member, resolve the preferred grid label and latest version, and
list the NetCDF files for a given variable.  They take no heavy action
(only ``Path.glob``/``iterdir``) so they are safe on a login node and easy
to unit-test against a synthetic tree.

Used by :mod:`scripts.convert_pool_cmip6` to build the per-variable zarr
cache that :class:`feather.data.cmip6.CMIP6Loader` then reads.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Member labels look like r1i1p1f1; this orders them numerically.
_MEMBER_RE = re.compile(r"r(\d+)i(\d+)p(\d+)f(\d+)$")

# Default preference order for grid labels.  Regular lat/lon (``gr``,
# ``gr1`` …) is preferred over native (``gn``) because it avoids
# curvilinear-grid regridding surprises downstream.
_DEFAULT_GRID_PREFERENCE = ("gr", "gr1", "gr2", "gn", "gm")


def iter_model_dirs(activity_root: Path, experiment: str):
    """Yield ``(model, experiment_dir)`` for every model with *experiment*.

    Parameters
    ----------
    activity_root : Path
        e.g. ``/pool/data/CMIP6/data/CMIP`` or ``.../HighResMIP``.
    experiment : str
        e.g. ``"historical"`` or ``"hist-1950"``.

    Yields
    ------
    (str, Path)
        Model (``source_id``) name and the path to its
        ``{institution}/{model}/{experiment}`` directory.
    """
    activity_root = Path(activity_root)
    if not activity_root.is_dir():
        return
    for inst_dir in sorted(activity_root.iterdir()):
        if not inst_dir.is_dir():
            continue
        for model_dir in sorted(inst_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            exp_dir = model_dir / experiment
            if exp_dir.is_dir():
                yield model_dir.name, exp_dir


def discover_models(activity_root: Path, experiment: str) -> list[str]:
    """Return a sorted, de-duplicated list of models that have *experiment*."""
    return sorted({m for m, _ in iter_model_dirs(activity_root, experiment)})


def _member_sort_key(member: str) -> tuple:
    """Sort key that orders members numerically (r1 < r2 < r10)."""
    m = _MEMBER_RE.match(member)
    if m:
        return (0,) + tuple(int(g) for g in m.groups())
    # Unparseable labels sort last, alphabetically.
    return (1, member)


def select_member(exp_dir: Path, prefer: str = "r1i1p1f1") -> str | None:
    """Pick a single ensemble member directory name.

    Prefers *prefer* (default ``r1i1p1f1``) when present, otherwise the
    numerically lowest member label.

    Returns
    -------
    str or None
        Member label, or ``None`` if the directory has no members.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return None
    members = [d.name for d in exp_dir.iterdir() if d.is_dir()]
    if not members:
        return None
    if prefer in members:
        return prefer
    return sorted(members, key=_member_sort_key)[0]


def select_grid(
    var_dir: Path, prefer: tuple[str, ...] = _DEFAULT_GRID_PREFERENCE,
) -> str | None:
    """Pick a grid-label subdirectory for a variable.

    Honours the *prefer* order; falls back to the first available grid
    label (sorted) when none of the preferred labels exist.
    """
    var_dir = Path(var_dir)
    if not var_dir.is_dir():
        return None
    grids = [d.name for d in var_dir.iterdir() if d.is_dir()]
    if not grids:
        return None
    for g in prefer:
        if g in grids:
            return g
    return sorted(grids)[0]


def latest_version(grid_dir: Path) -> Path | None:
    """Return the latest ``v*`` version directory under *grid_dir*.

    Falls back to *grid_dir* itself if it directly contains NetCDF files
    (some trees omit the version level).
    """
    grid_dir = Path(grid_dir)
    if not grid_dir.is_dir():
        return None
    versions = sorted(d for d in grid_dir.glob("v*") if d.is_dir())
    if versions:
        return versions[-1]
    if list(grid_dir.glob("*.nc")):
        return grid_dir
    return None


#: Filename time-span suffix, e.g. ``..._gn_195001-200012.nc``.  CMIP6
#: filenames end with the period covered; fixed fields (``areacella``) have
#: no such suffix.
_SPAN_RE = re.compile(r"_(\d{4,8})-(\d{4,8})\.nc$")


def _file_span(path: Path) -> tuple[str, str] | None:
    """``(start, end)`` from a CMIP6 filename, or ``None`` if it has no span."""
    m = _SPAN_RE.search(path.name)
    return (m.group(1), m.group(2)) if m else None


def union_version_files(grid_dir: Path) -> list[Path]:
    """Files from *every* version directory, newest winning on equal spans.

    The usual rule -- read the latest version only -- assumes a new version
    republishes the whole series.  A few centres instead publish *different
    time segments* under different version dates, so the newest directory
    holds one chunk and the rest are stranded.  BCC-CSM2-HR's ``hist-1950``
    ``Amon`` record is the known case: ``v20200822`` carries 1950-2000 and
    ``v20200921`` carries 2001-2014.

    Selection is strictly additive: a file from an older version is taken
    only when **no** newer version publishes the same time span, so a
    genuine supersede still wins and this can never downgrade a corrected
    file to its earlier edition.  Files with no parseable span fall back to
    latest-version-only, since without a span there is no way to tell a
    supersede from a distinct segment.
    """
    grid_dir = Path(grid_dir)
    if not grid_dir.is_dir():
        return []
    versions = sorted(d for d in grid_dir.glob("v*") if d.is_dir())
    if len(versions) < 2:
        latest = latest_version(grid_dir)
        return sorted(latest.glob("*.nc")) if latest else []

    # Later versions are visited last, so they overwrite earlier entries
    # for the same span.
    by_span: dict[tuple[str, str], Path] = {}
    spanless: list[Path] = []
    for version in versions:
        for nc in sorted(version.glob("*.nc")):
            span = _file_span(nc)
            if span is None:
                spanless.append(nc)
            else:
                by_span[span] = nc

    if not by_span:
        latest = latest_version(grid_dir)
        return sorted(latest.glob("*.nc")) if latest else []

    if spanless:
        # Mixed span/spanless in one variable directory is not a layout we
        # can reason about; stay with the conservative default.
        logger.warning(
            "union_version_files: %s mixes files with and without a time "
            "span; falling back to the latest version only", grid_dir,
        )
        latest = latest_version(grid_dir)
        return sorted(latest.glob("*.nc")) if latest else []

    chosen = sorted(by_span.items(), key=lambda kv: kv[0][0])
    extra = [p for span, p in chosen if p.parent != versions[-1]]
    if extra:
        logger.info(
            "union_version_files: %s -- recovered %d file(s) from older "
            "version(s): %s", grid_dir, len(extra),
            ", ".join(sorted({p.parent.name for p in extra})),
        )
    return [p for _, p in chosen]


def variable_files(
    exp_dir: Path,
    member: str,
    table: str,
    variable: str,
    *,
    grid_prefer: tuple[str, ...] = _DEFAULT_GRID_PREFERENCE,
    union_versions: bool = False,
) -> list[Path]:
    """Resolve the sorted NetCDF files for one variable.

    Walks ``{exp_dir}/{member}/{table}/{variable}/{grid}/{version}/`` using
    the grid and version preferences.

    Parameters
    ----------
    union_versions : bool, optional
        When True, combine non-overlapping time segments across version
        directories instead of reading the latest one alone -- see
        :func:`union_version_files`.  Off by default: "latest supersedes"
        is the correct CMIP6 reading and applies to all but a handful of
        publications.

    Returns
    -------
    list[Path]
        Sorted NetCDF paths, or an empty list if the variable is absent.
    """
    var_dir = Path(exp_dir) / member / table / variable
    grid = select_grid(var_dir, prefer=grid_prefer)
    if grid is None:
        return []
    if union_versions:
        return union_version_files(var_dir / grid)
    version_dir = latest_version(var_dir / grid)
    if version_dir is None:
        return []
    return sorted(version_dir.glob("*.nc"))
