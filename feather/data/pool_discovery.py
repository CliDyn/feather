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


def variable_files(
    exp_dir: Path,
    member: str,
    table: str,
    variable: str,
    *,
    grid_prefer: tuple[str, ...] = _DEFAULT_GRID_PREFERENCE,
) -> list[Path]:
    """Resolve the sorted NetCDF files for one variable.

    Walks ``{exp_dir}/{member}/{table}/{variable}/{grid}/{version}/`` using
    the grid and version preferences.

    Returns
    -------
    list[Path]
        Sorted NetCDF paths, or an empty list if the variable is absent.
    """
    var_dir = Path(exp_dir) / member / table / variable
    grid = select_grid(var_dir, prefer=grid_prefer)
    if grid is None:
        return []
    version_dir = latest_version(var_dir / grid)
    if version_dir is None:
        return []
    return sorted(version_dir.glob("*.nc"))
