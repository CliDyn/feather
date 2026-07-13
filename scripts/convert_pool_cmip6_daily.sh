#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a CMIP6 *daily* zarr cache (tas, tasmax, tasmin, pr) from the DKRZ pool.
#
# Wraps scripts/convert_pool_cmip6_daily.py, which walks the on-disk CMIP6 DRS
# tree (/pool/data/CMIP6/data/{activity}) and writes one per-variable zarr store
# per (model, variable) under the cache dir.  Heavy I/O (opens the daily
# archive) — run on a compute node, never the login node.  Only needs to run
# once per cache.
#
# Two experiments are produced into the *same* cache dir ($CACHE); the store
# filename encodes the experiment, so historical and ssp245 coexist:
#   CMIP / historical     (reference,  period 1980-2014)
#   ScenarioMIP / ssp245  (SSP2-4.5 future, period 2015-2050)
#
# Existing zarr stores are skipped (resumable) unless --no-skip is given.
#
# Usage:
#   sbatch scripts/convert_pool_cmip6_daily.sh                # both experiments
#   sbatch scripts/convert_pool_cmip6_daily.sh historical     # only historical
#   sbatch scripts/convert_pool_cmip6_daily.sh ssp245         # only ssp245
#   sbatch scripts/convert_pool_cmip6_daily.sh --dry-run      # list, write nothing
#   sbatch scripts/convert_pool_cmip6_daily.sh historical --no-skip
#   # or interactively on an already-allocated compute node:
#   bash scripts/convert_pool_cmip6_daily.sh historical --dry-run
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cmip6_daily_zarr
#SBATCH --account=bk1580
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bk1580/cmip6_zarr/logs/convert_daily_%j.out

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO=/home/a/a270292/feather
CACHE=/work/bk1580/cmip6_zarr
HIST_PERIOD=(1980 2014)
SSP_PERIOD=(2015 2050)

# ── Argument parsing ─────────────────────────────────────────────────────────
WHICH=both              # historical | ssp245 | both
EXTRA_ARGS=()           # passthrough to convert_pool_cmip6_daily.py
for arg in "$@"; do
    case "$arg" in
        historical|ssp245|both) WHICH="$arg" ;;
        --dry-run)              EXTRA_ARGS+=(--dry-run) ;;
        --no-skip)              EXTRA_ARGS+=(--no-skip-existing) ;;
        *) echo "usage: $0 [historical|ssp245|both] [--dry-run] [--no-skip]" >&2; exit 2 ;;
    esac
done

mkdir -p "$CACHE/logs"

# ── Conda env ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate feather

cd "$REPO"

run_historical() {
    echo "[$(date)] Converting CMIP6 daily historical → $CACHE"
    python scripts/convert_pool_cmip6_daily.py \
        --activity CMIP \
        --experiment historical \
        --period "${HIST_PERIOD[@]}" \
        --out "$CACHE" \
        "${EXTRA_ARGS[@]}" \
        -v
}

run_ssp245() {
    echo "[$(date)] Converting CMIP6 daily ssp245 → $CACHE"
    python scripts/convert_pool_cmip6_daily.py \
        --activity ScenarioMIP \
        --experiment ssp245 \
        --period "${SSP_PERIOD[@]}" \
        --out "$CACHE" \
        "${EXTRA_ARGS[@]}" \
        -v
}

case "$WHICH" in
    historical) run_historical ;;
    ssp245)     run_ssp245 ;;
    both)       run_historical; run_ssp245 ;;
esac

echo "[$(date)] DONE ($WHICH)"
