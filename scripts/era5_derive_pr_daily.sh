#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily ERA5 precipitation on the regular 0.25° grid (CMOR tree).
#
# Source : /pool/data/ERA5/E5/sf/fc/1D/228/E5sf12_1D_YYYY-MM_228.grb
#          (daily total precipitation, m/day, reduced-Gaussian N320, one file
#           per month, time stamp 12:30)
#
# Output (regular 0.25° lat/lon, 1440×721 incl. poles, kg m-2 s-1):
#   $OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day/pr/gr/v1/
#       pr_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
#
# Remapping is area-conservative (first-order, cdo remapcon): the reduced
# Gaussian grid has no cell corners, so it is first expanded to the full
# N320 Gaussian grid (setgridtype,regular), then remapped with weights built
# once per job.  m/day → kg m-2 s-1 is ×1000/86400; time stamps are set to
# 12:00 like the CMOR model output.
#
# Usage:
#   sbatch --array=1981-2014 scripts/era5_derive_pr_daily.sh
#   bash scripts/era5_derive_pr_daily.sh 1995          # one year, compute node
#   OUT=/some/test/dir bash scripts/era5_derive_pr_daily.sh 1995
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=era5_pr_day
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/era5_derived/logs/era5_pr_day_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/pool/data/ERA5/E5/sf/fc/1D/228
OUT="${OUT:-/work/bm1344/AWI/OBS/era5_derived}"
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)

DEST=$OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day/pr/gr/v1
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/era5_pr_day_${YEAR}"
mkdir -p "$DEST" "$WDIR"

echo "[$(date)] year=$YEAR  src=$SRC  out=$DEST  work=$WDIR"

shopt -s nullglob
files=("$SRC"/E5sf12_1D_${YEAR}-??_228.grb)
if [ "${#files[@]}" -ne 12 ]; then
    echo "ERROR: expected 12 monthly files for $YEAR, found ${#files[@]}" >&2
    exit 1
fi

# 1. Conservative weights (full N320 Gaussian → 0.25°), computed once.
WTS=$WDIR/weights_con.nc
cdo -s gencon,"$GRID" -setgridtype,regular -seltimestep,1 "${files[0]}" "$WTS"

# 2. Remap each month and convert m/day → kg m-2 s-1.
for f in "${files[@]}"; do
    m=$(basename "$f" .grb)              # E5sf12_1D_YYYY-MM_228
    cdo -s -f nc4 -b F32 -settime,12:00:00 -setname,pr \
        -mulc,0.0115740740740741 -remap,"$GRID","$WTS" -setgridtype,regular \
        "$f" "$WDIR/$m.nc"
done

# 3. Concatenate the year (mergetime is variadic, so it runs on its own) and
#    set CMOR attributes in a separate pass.
TARGET=$DEST/pr_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/E5sf12_1D_*.nc "$WDIR/pr_${YEAR}.nc"
ndays=$(cdo -s ntime "$WDIR/pr_${YEAR}.nc")
cdo -s -O -z zip_1 \
    -setattribute,pr@least_significant_digit=,pr@units="kg m-2 s-1",pr@cell_methods="time: mean",pr@standard_name=precipitation_flux,pr@long_name="Precipitation",source="ERA5 daily total precipitation (code 228); conservatively remapped from N320 to 0.25 deg" \
    "$WDIR/pr_${YEAR}.nc" "$TARGET"

mean=$(cdo -s outputf,%.3f,1 -mulc,86400 -fldmean -timmean "$TARGET")
rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ($ndays days, global mean $mean mm/day)  ->  $TARGET"
