#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a CMIP6 *daily* ssp370 zarr cache (tas, tasmax, tasmin, pr) from the
# DKRZ pool, restricted to 2015-2030.
#
# Wraps scripts/convert_pool_cmip6_daily.py (the generic daily converter),
# pinned to ScenarioMIP / ssp370 over the near-term 2015-2030 window.  Writes
# into the same daily cache dir ($CACHE) as the historical/ssp245 wrapper — the
# store filename encodes the experiment, so ssp370 coexists with the others:
#   {model}_ssp370_{member}_day_{var}.zarr
#
# Heavy I/O (opens the daily archive) — run on a compute node, never the login
# node.  Existing zarr stores are skipped (resumable) unless --no-skip is given.
#
# Usage:
#   sbatch scripts/convert_pool_cmip6_daily_ssp370.sh              # convert
#   sbatch scripts/convert_pool_cmip6_daily_ssp370.sh --dry-run    # list only
#   sbatch scripts/convert_pool_cmip6_daily_ssp370.sh --no-skip    # overwrite
#   # or interactively on an already-allocated compute node:
#   bash scripts/convert_pool_cmip6_daily_ssp370.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cmip6_daily_ssp370
#SBATCH --account=bk1580
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bk1580/cmip6_zarr/logs/convert_daily_ssp370_%j.out

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO=/home/a/a270292/feather
CACHE=/work/bk1580/cmip6_zarr
PERIOD=(2015 2030)

# ── Argument parsing ─────────────────────────────────────────────────────────
EXTRA_ARGS=()           # passthrough to convert_pool_cmip6_daily.py
for arg in "$@"; do
    case "$arg" in
        --dry-run) EXTRA_ARGS+=(--dry-run) ;;
        --no-skip) EXTRA_ARGS+=(--no-skip-existing) ;;
        *) echo "usage: $0 [--dry-run] [--no-skip]" >&2; exit 2 ;;
    esac
done

mkdir -p "$CACHE/logs"

# ── Conda env ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate feather

cd "$REPO"

echo "[$(date)] Converting CMIP6 daily ssp370 (${PERIOD[0]}-${PERIOD[1]}) → $CACHE"
python scripts/convert_pool_cmip6_daily.py \
    --activity ScenarioMIP \
    --experiment ssp370 \
    --period "${PERIOD[@]}" \
    --out "$CACHE" \
    "${EXTRA_ARGS[@]}" \
    -v

echo "[$(date)] DONE (ssp370)"
