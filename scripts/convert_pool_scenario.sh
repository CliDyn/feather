#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Extend feather's benchmark zarr cache with the 2015-2025 scenario period.
#
# Wraps scripts/convert_pool_scenario.py, which appends the future-scenario
# segment to the existing historical caches so the benchmark MMM stitches
# across historical -> ssp370 (CMIP6) and hist-1950 -> highres-future
# (HighResMIP).  Only the models AND member already in each cache are
# converted (no new models).  Heavy I/O — run on a compute node.
#
# After this runs, point the benchmark config at the stitched experiments:
#   benchmarks:
#     - name: CMIP6
#       experiments: [historical, ssp370]
#     - name: HighResMIP
#       experiments: [hist-1950, highres-future]
#
# Usage:
#   sbatch scripts/convert_pool_scenario.sh                 # both extensions
#   sbatch scripts/convert_pool_scenario.sh cmip6           # only ssp370
#   sbatch scripts/convert_pool_scenario.sh highresmip      # only highres-future
#   sbatch scripts/convert_pool_scenario.sh --dry-run       # list, write nothing
#   sbatch scripts/convert_pool_scenario.sh cmip6 --no-skip # overwrite existing
#   # or interactively on an already-allocated compute node:
#   bash scripts/convert_pool_scenario.sh cmip6 --dry-run
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cmip6_scenario_zarr
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bm1344/AWI/EERIE/cmip6_pool_zarr/logs/convert_scenario_%j.out

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO=/home/a/a270292/feather
CACHE=/work/bm1344/AWI/EERIE/cmip6_pool_zarr
PERIOD=(2015 2025)

# ── Argument parsing ─────────────────────────────────────────────────────────
WHICH=both              # cmip6 | highresmip | both
EXTRA_ARGS=()           # passthrough to convert_pool_scenario.py
for arg in "$@"; do
    case "$arg" in
        cmip6|highresmip|both) WHICH="$arg" ;;
        --dry-run)             EXTRA_ARGS+=(--dry-run) ;;
        --no-skip)             EXTRA_ARGS+=(--no-skip-existing) ;;
        --member-fallback)     EXTRA_ARGS+=(--member-fallback) ;;
        *) echo "usage: $0 [cmip6|highresmip|both] [--dry-run] [--no-skip] [--member-fallback]" >&2; exit 2 ;;
    esac
done

mkdir -p "$CACHE/logs"

# ── Conda env ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate feather

cd "$REPO"

run_cmip6() {
    echo "[$(date)] Extending CMIP6 historical with ssp370 (2015-2025)"
    python scripts/convert_pool_scenario.py \
        --activity ScenarioMIP \
        --experiment ssp370 \
        --base-experiment historical \
        --cache "$CACHE/cmip6_historical" \
        --period "${PERIOD[@]}" \
        "${EXTRA_ARGS[@]}" \
        -v
}

run_highresmip() {
    echo "[$(date)] Extending HighResMIP hist-1950 with highres-future (2015-2025)"
    python scripts/convert_pool_scenario.py \
        --activity HighResMIP \
        --experiment highres-future \
        --base-experiment hist-1950 \
        --cache "$CACHE/highresmip_hist-1950" \
        --period "${PERIOD[@]}" \
        "${EXTRA_ARGS[@]}" \
        -v
}

case "$WHICH" in
    cmip6)      run_cmip6 ;;
    highresmip) run_highresmip ;;
    both)       run_cmip6; run_highresmip ;;
esac

echo "[$(date)] DONE ($WHICH)"
