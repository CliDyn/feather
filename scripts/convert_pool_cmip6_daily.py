#!/usr/bin/env python
"""Convert DKRZ CMIP6 pool *daily* tas/tasmax/tasmin/pr into per-variable zarr.

Companion to :mod:`scripts.convert_pool_cmip6`, restricted to the daily
(``day`` table) temperature and precipitation variables consumed by the daily
extremes / temperature-change diagnostics.  The monthly converter deliberately
skips the ``day`` table (it is huge); this script fills that gap into a
dedicated cache directory (default ``/work/bk1580/cmip6_zarr``).

Reads the on-disk CMIP6 DRS tree at ``/pool/data/CMIP6/data/{activity}`` and
writes one zarr store per (model, variable) using the naming convention that
:class:`feather.data.cmip6.CMIP6Loader` already understands::

    {out}/{model}_{experiment}_{member}_day_{var}.zarr        (tas/tasmax/tasmin/pr)
    {out}/{model}_{experiment}_{member}_fx_areacella.zarr     (atmosphere area)

One ensemble member is selected per model (``r1i1p1f1`` preferred).  Models are
auto-discovered from the tree unless an explicit ``--models`` list is given.
Missing variables are skipped silently — the loader tolerates gaps.

The experiment is encoded in each store filename, so several experiments
(e.g. ``historical`` and ``ssp245``) can share one output directory.

This is a heavy I/O job (daily archive is large); run it on a compute node, not
the login node.  Existing stores are skipped, so it is resumable.

Examples
--------
    python scripts/convert_pool_cmip6_daily.py --activity CMIP \
        --experiment historical --period 1980 2014 \
        --out /work/bk1580/cmip6_zarr -v

    python scripts/convert_pool_cmip6_daily.py --activity ScenarioMIP \
        --experiment ssp245 --period 2015 2050 \
        --out /work/bk1580/cmip6_zarr -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from feather.data import pool_discovery as disc

# Reuse the (tested) heavy-I/O helpers from the monthly converter rather than
# duplicating them.  ``scripts/`` is not a package, so add this file's dir to
# the path before importing its sibling module.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import convert_pool_cmip6 as base  # noqa: E402

logger = logging.getLogger("convert_pool_cmip6_daily")

DEFAULT_POOL_ROOT = base.DEFAULT_POOL_ROOT
DEFAULT_OUT = "/work/bk1580/cmip6_zarr"

# The daily variables to convert, keyed by CMIP6 table.  ``day`` holds the
# daily-mean 2 m temperature (``tas``), daily min/max (``tasmin``/``tasmax``)
# and precipitation flux (``pr``).
_DAILY_VAR_TABLES: dict[str, list[str]] = {
    "day": ["pr", "tas", "tasmax", "tasmin"],
}

# Atmosphere cell-area weights (fixed field, no period slice) — useful for
# area-weighted global means of the daily fields.  Ocean area is not needed
# here (all daily variables are atmospheric).
_AREA_TABLES = {"fx": "areacella"}


def convert_model(
    exp_dir: Path,
    model: str,
    experiment: str,
    member: str,
    out_dir: Path,
    period,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> int:
    """Convert the daily variables (+ areacella) for one model member.

    Returns the number of zarr stores written.
    """
    written = 0
    for table, variables in _DAILY_VAR_TABLES.items():
        for variable in variables:
            store = out_dir / f"{model}_{experiment}_{member}_{table}_{variable}.zarr"
            if skip_existing and base._store_is_valid(store, variable):
                logger.debug("    skip existing %s", store.name)
                continue
            if store.exists():
                logger.info("    reconverting incomplete store %s", store.name)
            files = disc.variable_files(exp_dir, member, table, variable)
            if not files:
                continue
            logger.info("    %s/%s (%d files)", table, variable, len(files))
            if dry_run:
                written += 1
                continue
            ds = base._open_and_slice(files, variable, period)
            if ds is None:
                continue
            try:
                base._write_zarr_atomic(ds, store)
                written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("    write failed for %s: %s", store.name, e)

    # Atmosphere area weights (no period slice).
    for table, area_var in _AREA_TABLES.items():
        store = out_dir / f"{model}_{experiment}_{member}_{table}_{area_var}.zarr"
        if skip_existing and base._store_is_valid(store, area_var):
            continue
        files = disc.variable_files(exp_dir, member, table, area_var)
        if not files:
            continue
        logger.info("    %s/%s (area)", table, area_var)
        if dry_run:
            written += 1
            continue
        ds = base._open_and_slice(files, area_var, None)
        if ds is None:
            continue
        try:
            base._write_zarr_atomic(ds, store)
            written += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("    write failed for %s: %s", store.name, e)

    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--activity", required=True,
                        help="CMIP6 activity_id (e.g. CMIP, ScenarioMIP)")
    parser.add_argument("--experiment", required=True,
                        help="experiment_id (e.g. historical, ssp245)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output zarr cache dir (default {DEFAULT_OUT})")
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

    logger.info("Converting %d models (%s/%s) daily → %s",
                len(models), args.activity, args.experiment, out_dir)
    logger.info("Variables: %s", _DAILY_VAR_TABLES)

    exp_dirs = dict(disc.iter_model_dirs(activity_root, args.experiment))
    total_written = 0
    for model in models:
        exp_dir = exp_dirs.get(model)
        if exp_dir is None:
            exp_dir = base._resolve_exp_dir(activity_root, model, args.experiment)
        if exp_dir is None:
            logger.warning("  %s: experiment dir not found — skipping", model)
            continue
        member = disc.select_member(exp_dir, prefer=args.member)
        if member is None:
            logger.warning("  %s: no members — skipping", model)
            continue
        logger.info("  %s [%s]", model, member)
        total_written += convert_model(
            exp_dir, model, args.experiment, member,
            out_dir, args.period,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run,
        )

    logger.info("Done — %d zarr store(s) %s",
                total_written, "planned" if args.dry_run else "written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
