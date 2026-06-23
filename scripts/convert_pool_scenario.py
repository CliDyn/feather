#!/usr/bin/env python
"""Extend feather's benchmark zarr cache with the 2015-2025 scenario period.

The historical benchmark caches built by ``convert_pool_cmip6.py`` end in
2014 (CMIP6 ``historical``) / 2014 (HighResMIP ``hist-1950``).  This script
appends the matching future-scenario segment so the benchmark MMM can be
stitched across ``historical -> ssp370`` (CMIP6) or
``hist-1950 -> highres-future`` (HighResMIP) — see ``CMIP6Loader._open_stitched``.

Only the models AND the member that are already present in the existing cache
are converted (no new models are added).  For each ``{model}_{base_experiment}_
{variant}_...zarr`` already on disk, the same ``variant`` is requested for the
scenario; if that exact member is missing from the pool for the scenario, the
model is skipped with a warning (pass ``--member-fallback`` to instead pick any
available member, preferring the cached one).

Scenario stores are written into the SAME cache dir with the scenario
experiment id in the filename::

    {cache}/{model}_ssp370_{variant}_{table}_{var}.zarr           (CMIP6)
    {cache}/{model}_highres-future_{variant}_{table}_{var}.zarr   (HighResMIP)

so the loader finds them alongside the historical stores when configured with
``experiments: [historical, ssp370]`` / ``[hist-1950, highres-future]``.

Heavy I/O (opens the pool archive) — run on a compute node, not the login node.

Examples
--------
    # CMIP6 ssp370 extension for the models already in cmip6_historical
    python scripts/convert_pool_scenario.py \
        --activity ScenarioMIP --experiment ssp370 \
        --base-experiment historical \
        --cache /work/bm1344/AWI/EERIE/cmip6_pool_zarr/cmip6_historical \
        --period 2015 2025 -v

    # HighResMIP highres-future extension for the models already in hist-1950
    python scripts/convert_pool_scenario.py \
        --activity HighResMIP --experiment highres-future \
        --base-experiment hist-1950 \
        --cache /work/bm1344/AWI/EERIE/cmip6_pool_zarr/highresmip_hist-1950 \
        --period 2015 2025 -v
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from feather.data import pool_discovery as disc

# Reuse the historical converter's machinery (variable tables, per-model
# conversion, area weights, period slicing) — the only thing that changes for
# the scenario is which (model, member) we feed it and the experiment label.
# Run as a path script (``python scripts/convert_pool_scenario.py``), so
# ``scripts/`` is on sys.path and the sibling module imports directly; fall
# back to the package path when imported as ``scripts.convert_pool_scenario``.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from convert_pool_cmip6 import (  # type: ignore[import-not-found]
        DEFAULT_POOL_ROOT,
        build_var_tables,
        convert_model,
    )
except ImportError:  # pragma: no cover - fallback for package-style import
    from scripts.convert_pool_cmip6 import (
        DEFAULT_POOL_ROOT,
        build_var_tables,
        convert_model,
    )

logger = logging.getLogger("convert_pool_scenario")


def discover_cache_members(cache_dir: Path, base_experiment: str) -> dict[str, str]:
    """Return ``{model: variant}`` already present in the cache.

    Scans ``{cache}/{model}_{base_experiment}_{variant}_{table}_{var}.zarr``,
    taking one variant per model (the lowest, matching the converter's member
    preference).  CMIP6 ``source_id`` / ``experiment_id`` never contain
    underscores, so the filename splits unambiguously on ``_``.
    """
    if not cache_dir.is_dir():
        logger.error("Cache dir not found: %s", cache_dir)
        return {}

    pattern = re.compile(
        rf"^(?P<model>.+?)_{re.escape(base_experiment)}_"
        r"(?P<variant>[^_]+)_(?P<table>[^_]+)_(?P<var>.+)\.zarr$"
    )
    found: dict[str, set[str]] = {}
    for entry in sorted(cache_dir.glob(f"*_{base_experiment}_*.zarr")):
        m = pattern.match(entry.name)
        if not m:
            continue
        found.setdefault(m.group("model"), set()).add(m.group("variant"))

    return {model: sorted(variants)[0] for model, variants in sorted(found.items())}


def _resolve_exp_dir(activity_root: Path, model: str, experiment: str):
    """Find the scenario experiment dir for a named model (or None)."""
    for m, exp_dir in disc.iter_model_dirs(activity_root, experiment):
        if m == model:
            return exp_dir
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activity", required=True,
                        help="scenario activity_id (ScenarioMIP | HighResMIP)")
    parser.add_argument("--experiment", required=True,
                        help="scenario experiment_id (ssp370 | highres-future)")
    parser.add_argument("--base-experiment", required=True,
                        help="experiment whose cached stores define the "
                             "(model, member) set (historical | hist-1950)")
    parser.add_argument("--cache", required=True,
                        help="existing zarr cache dir (read base stores, "
                             "write scenario stores alongside)")
    parser.add_argument("--pool-root", default=DEFAULT_POOL_ROOT,
                        help=f"CMIP6 pool data root (default {DEFAULT_POOL_ROOT})")
    parser.add_argument("--period", nargs=2, metavar=("START", "END"),
                        default=["2015", "2025"],
                        help="time slice (default 2015 2025)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="restrict to these models (default: all in cache)")
    parser.add_argument("--member-fallback", action="store_true",
                        help="if the cached member is absent for the scenario, "
                             "pick any available member (preferring the cached "
                             "one) instead of skipping the model")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="overwrite existing scenario zarr stores")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be converted without writing")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else
        logging.INFO if args.verbose == 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir = Path(args.cache)
    activity_root = Path(args.pool_root) / args.activity

    cache_members = discover_cache_members(cache_dir, args.base_experiment)
    if not cache_members:
        logger.error("No %s stores found in cache %s — nothing to extend",
                     args.base_experiment, cache_dir)
        return 1

    if args.models:
        wanted = set(args.models)
        cache_members = {m: v for m, v in cache_members.items() if m in wanted}
        missing = wanted - set(cache_members)
        for m in sorted(missing):
            logger.warning("  %s not in cache — skipping", m)

    var_tables = build_var_tables()
    logger.info("Extending %d cached models with %s/%s (%s) → %s",
                len(cache_members), args.activity, args.experiment,
                "-".join(args.period), cache_dir)

    total_written = 0
    n_models = 0
    for model, cached_variant in cache_members.items():
        exp_dir = _resolve_exp_dir(activity_root, model, args.experiment)
        if exp_dir is None:
            logger.warning("  %s: no %s in pool — skipping",
                           model, args.experiment)
            continue

        # Prefer the exact member already in the cache so the stitched series
        # stays self-consistent.  Variant labels can differ between historical
        # and scenario (e.g. CanESM5 ssp370 is r1i1p2f1) — only diverge from the
        # cached member when --member-fallback is given.
        member = disc.select_member(exp_dir, prefer=cached_variant)
        if member is None:
            logger.warning("  %s: no members for %s — skipping",
                           model, args.experiment)
            continue
        if member != cached_variant and not args.member_fallback:
            logger.warning(
                "  %s: cached member %s absent for %s (found %s) — skipping "
                "(use --member-fallback to convert anyway)",
                model, cached_variant, args.experiment, member,
            )
            continue
        if member != cached_variant:
            logger.warning("  %s: using %s for %s (cached %s)",
                           model, member, args.experiment, cached_variant)

        logger.info("  %s [%s]", model, member)
        n_models += 1
        total_written += convert_model(
            exp_dir, model, args.experiment, member, var_tables,
            cache_dir, args.period,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run,
        )

    logger.info("Done — %d model(s), %d zarr store(s) %s",
                n_models, total_written,
                "planned" if args.dry_run else "written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
