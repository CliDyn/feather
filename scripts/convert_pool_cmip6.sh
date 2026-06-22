#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build feather's CMIP6 / HighResMIP benchmark zarr cache from the DKRZ pool.
#
# Wraps scripts/convert_pool_cmip6.py, which walks the on-disk CMIP6 DRS tree
# (/pool/data/CMIP6/data/{activity}) and writes one per-variable zarr store per
# (model, variable, table) under the cache dir.  Heavy I/O (opens the full
# archive) — run on a compute node, never the login node.  Only needs to run
# once per cache.
#
# Two benchmarks are produced into separate sub-dirs of $CACHE:
#   cmip6_historical        CMIP / historical    (~67 models, 1 member each)
#   highresmip_hist-1950    HighResMIP / hist-1950 (~22 models)
#
# Existing zarr stores are skipped (resumable) unless --no-skip is given.
#
# Usage:
#   sbatch scripts/convert_pool_cmip6.sh                 # both benchmarks
#   sbatch scripts/convert_pool_cmip6.sh cmip6           # only CMIP6 historical
#   sbatch scripts/convert_pool_cmip6.sh highresmip      # only HighResMIP
#   sbatch scripts/convert_pool_cmip6.sh --dry-run       # list, write nothing
#   sbatch scripts/convert_pool_cmip6.sh cmip6 --no-skip # overwrite existing
#   # or interactively on an already-allocated compute node:
#   bash scripts/convert_pool_cmip6.sh cmip6 --dry-run
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cmip6_pool_zarr
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bm1344/AWI/EERIE/cmip6_pool_zarr/logs/convert_pool_%j.out

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO=/home/a/a270292/feather
# Permanent storage — scratch is purged periodically.
CACHE=/work/bm1344/AWI/EERIE/cmip6_pool_zarr
PERIOD=(1980 2014)

# ── Argument parsing ─────────────────────────────────────────────────────────
WHICH=both              # cmip6 | highresmip | both
EXTRA_ARGS=()           # passthrough to convert_pool_cmip6.py
for arg in "$@"; do
    case "$arg" in
        cmip6|highresmip|both) WHICH="$arg" ;;
        --dry-run)             EXTRA_ARGS+=(--dry-run) ;;
        --no-skip)             EXTRA_ARGS+=(--no-skip-existing) ;;
        *) echo "usage: $0 [cmip6|highresmip|both] [--dry-run] [--no-skip]" >&2; exit 2 ;;
    esac
done

mkdir -p "$CACHE/logs"

# ── Conda env ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate feather

cd "$REPO"

run_cmip6() {
    echo "[$(date)] Converting CMIP6 historical → $CACHE/cmip6_historical"
    python scripts/convert_pool_cmip6.py \
        --activity CMIP \
        --experiment historical \
        --period "${PERIOD[@]}" \
        --out "$CACHE/cmip6_historical" \
        "${EXTRA_ARGS[@]}" \
        -v
}

run_highresmip() {
    echo "[$(date)] Converting HighResMIP hist-1950 → $CACHE/highresmip_hist-1950"
    python scripts/convert_pool_cmip6.py \
        --activity HighResMIP \
        --experiment hist-1950 \
        --period "${PERIOD[@]}" \
        --out "$CACHE/highresmip_hist-1950" \
        "${EXTRA_ARGS[@]}" \
        -v
}

case "$WHICH" in
    cmip6)      run_cmip6 ;;
    highresmip) run_highresmip ;;
    both)       run_cmip6; run_highresmip ;;
esac

echo "[$(date)] DONE ($WHICH)"
