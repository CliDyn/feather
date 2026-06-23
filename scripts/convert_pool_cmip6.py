#!/usr/bin/env python
"""Convert DKRZ CMIP6 pool (DRS NetCDF) into feather's per-variable zarr cache.

Reads the on-disk CMIP6 DRS tree at ``/pool/data/CMIP6/data/{activity}`` and
writes one zarr store per (model, variable, table) using the naming
convention that :class:`feather.data.cmip6.CMIP6Loader` already understands::

    {out}/{model}_{experiment}_{member}_{table}_{var}.zarr
    {out}/{model}_{experiment}_{member}_fx_areacella.zarr     (atmosphere area)
    {out}/{model}_{experiment}_{member}_Ofx_areacello.zarr    (ocean area)

One ensemble member is selected per model (``r1i1p1f1`` preferred).  Models
are auto-discovered from the tree unless an explicit ``--models`` list is
given.  Missing variables are skipped silently — the loader tolerates gaps.

This is a heavy I/O job (opening the full archive); run it on a compute node,
not the login node.  It only needs to run once per cache.

Examples
--------
    python scripts/convert_pool_cmip6.py --activity CMIP \
        --experiment historical --period 1980 2014 \
        --out /work/bm1344/AWI/EERIE/cmip6_pool_zarr/cmip6_historical -v

    python scripts/convert_pool_cmip6.py --activity HighResMIP \
        --experiment hist-1950 --period 1980 2014 \
        --out /work/bm1344/AWI/EERIE/cmip6_pool_zarr/highresmip_hist-1950 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from feather.data import pool_discovery as disc
from feather.data.variables import VARIABLE_REGISTRY

logger = logging.getLogger("convert_pool_cmip6")

DEFAULT_POOL_ROOT = "/pool/data/CMIP6/data"

# Extra (table, variable) pairs needed by diagnostics but not all present as
# standalone entries in VARIABLE_REGISTRY (e.g. radiation flux components used
# by the radiation-budget diagnostic, and 3-D ocean fields).
_EXTRA_VAR_TABLES: dict[str, list[str]] = {
    "Amon": [
        "rsdt", "rsut", "rlut", "rsutcs", "rlutcs",
        "rsus", "rlus", "rsdscs", "rsuscs", "rldscs", "rluscs",
        "hfss", "hfls",
    ],
    "Omon": ["thetao", "so"],
}

# Area weights to convert (table key on disk → area variable).
_AREA_TABLES = {"fx": "areacella", "Ofx": "areacello"}

# Tables excluded from the benchmark cache by default.  ``day`` (daily
# tasmin/tasmax) is huge and only consumed by the extremes diagnostics,
# which use ERA5/Berkeley references rather than the CMIP6/HighResMIP
# benchmark MMM.
_SKIP_TABLES = {"day"}


def build_var_tables() -> dict[str, list[str]]:
    """Build the {table: [variables]} mapping to convert.

    Combines variables declared in VARIABLE_REGISTRY (those with a CMIP6
    name + table) with the extra components diagnostics derive.  Tables in
    ``_SKIP_TABLES`` are omitted.
    """
    var_tables: dict[str, set[str]] = {}
    for vinfo in VARIABLE_REGISTRY.values():
        cvar = getattr(vinfo, "cmip6_variable", "")
        ctab = getattr(vinfo, "cmip6_table", "")
        if cvar and ctab and ctab not in _SKIP_TABLES:
            var_tables.setdefault(ctab, set()).add(cvar)
    for table, extras in _EXTRA_VAR_TABLES.items():
        if table in _SKIP_TABLES:
            continue
        var_tables.setdefault(table, set()).update(extras)
    return {t: sorted(v) for t, v in var_tables.items()}


def _open_and_slice(files, variable, period):
    """Open NetCDF files for a variable and slice to *period*.

    Returns the sliced ``xr.Dataset`` (single data var) or ``None`` on
    failure / empty selection.  Imported lazily so the module stays import
    -light for unit tests.
    """
    import numpy as np
    import xarray as xr

    # Use cftime so non-standard calendars (360_day, noleap) decode; prefer
    # the new CFDatetimeCoder API, falling back to the legacy kwarg.
    try:
        open_kwargs = {
            "decode_times": xr.coders.CFDatetimeCoder(use_cftime=True),
        }
    except AttributeError:  # older xarray
        open_kwargs = {"decode_times": True, "use_cftime": True}

    try:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            chunks="auto",
            **open_kwargs,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("    open failed (%s): %s", variable, e)
        return None

    if variable not in ds.data_vars:
        logger.warning("    %s not a data var in files", variable)
        return None

    if "time" in ds[variable].dims:
        # Some models concatenate to a non-monotonic / duplicated time axis
        # (overlapping file ranges); sort and de-duplicate before slicing.
        ds = ds.sortby("time")
        times = ds["time"].values
        _, keep_idx = np.unique(times, return_index=True)
        if keep_idx.size != times.size:
            ds = ds.isel(time=np.sort(keep_idx))

        if period is not None:
            ds = ds.sel(time=slice(str(period[0]), str(period[1])))
            if ds.sizes.get("time", 0) == 0:
                logger.warning("    %s empty after period slice", variable)
                return None

    # Keep only the target variable plus its coordinates/bounds.
    keep = [variable]
    return ds[keep]


def convert_model(
    exp_dir: Path,
    model: str,
    experiment: str,
    member: str,
    var_tables: dict[str, list[str]],
    out_dir: Path,
    period,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> int:
    """Convert all variables for one model member. Returns count written."""
    written = 0
    for table, variables in var_tables.items():
        for variable in variables:
            store = out_dir / f"{model}_{experiment}_{member}_{table}_{variable}.zarr"
            if skip_existing and store.exists():
                logger.debug("    skip existing %s", store.name)
                continue
            files = disc.variable_files(exp_dir, member, table, variable)
            if not files:
                continue
            logger.info("    %s/%s (%d files)", table, variable, len(files))
            if dry_run:
                written += 1
                continue
            ds = _open_and_slice(files, variable, period)
            if ds is None:
                continue
            try:
                ds.to_zarr(store, mode="w", consolidated=True)
                written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("    write failed for %s: %s", store.name, e)

    # Area weights (no period slice).
    for table, area_var in _AREA_TABLES.items():
        store = out_dir / f"{model}_{experiment}_{member}_{table}_{area_var}.zarr"
        if skip_existing and store.exists():
            continue
        files = disc.variable_files(exp_dir, member, table, area_var)
        if not files:
            continue
        logger.info("    %s/%s (area)", table, area_var)
        if dry_run:
            written += 1
            continue
        ds = _open_and_slice(files, area_var, None)
        if ds is None:
            continue
        try:
            ds.to_zarr(store, mode="w", consolidated=True)
            written += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("    write failed for %s: %s", store.name, e)

    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activity", required=True,
                        help="CMIP6 activity_id (e.g. CMIP, HighResMIP)")
    parser.add_argument("--experiment", required=True,
                        help="experiment_id (e.g. historical, hist-1950)")
    parser.add_argument("--out", required=True, help="output zarr cache dir")
    parser.add_argument("--pool-root", default=DEFAULT_POOL_ROOT,
                        help=f"CMIP6 pool data root (default {DEFAULT_POOL_ROOT})")
    parser.add_argument("--period", nargs=2, metavar=("START", "END"),
                        default=None, help="time slice, e.g. 1980 2014")
    parser.add_argument("--models", nargs="+", default=None,
                        help="explicit model list (default: auto-discover)")
    parser.add_argument("--member", default="r1i1p1f1",
                        help="preferred member label (default r1i1p1f1)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="overwrite existing zarr stores")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be converted without writing")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else
        logging.INFO if args.verbose == 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    activity_root = Path(args.pool_root) / args.activity
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or disc.discover_models(activity_root, args.experiment)
    if not models:
        logger.error("No models found under %s for experiment %s",
                     activity_root, args.experiment)
        return 1

    var_tables = build_var_tables()
    logger.info("Converting %d models (%s/%s) → %s",
                len(models), args.activity, args.experiment, out_dir)
    logger.info("Variables: %s",
                {t: len(v) for t, v in var_tables.items()})

    exp_dirs = dict(disc.iter_model_dirs(activity_root, args.experiment))
    total_written = 0
    for model in models:
        exp_dir = exp_dirs.get(model)
        if exp_dir is None:
            exp_dir = _resolve_exp_dir(activity_root, model, args.experiment)
        if exp_dir is None:
            logger.warning("  %s: experiment dir not found — skipping", model)
            continue
        member = disc.select_member(exp_dir, prefer=args.member)
        if member is None:
            logger.warning("  %s: no members — skipping", model)
            continue
        logger.info("  %s [%s]", model, member)
        total_written += convert_model(
            exp_dir, model, args.experiment, member, var_tables,
            out_dir, args.period,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run,
        )

    logger.info("Done — %d zarr store(s) %s",
                total_written, "planned" if args.dry_run else "written")
    return 0


def _resolve_exp_dir(activity_root: Path, model: str, experiment: str):
    """Find the experiment dir for an explicitly-named model."""
    for m, exp_dir in disc.iter_model_dirs(activity_root, experiment):
        if m == model:
            return exp_dir
    return None


if __name__ == "__main__":
    sys.exit(main())
