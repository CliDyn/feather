#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily + monthly ERA5 tasmin/tasmax from hourly 2 m temperature (167).
#
# Source : /pool/data/ERA5/E5/sf/an/1H/167/E5sf00_1H_YYYY-MM-DD_167.grb
#          (24 hourly steps/day, native reduced-Gaussian grid, ~542k points)
#
# Output (regular 0.25° lat/lon, 1440×721, Kelvin):
#   Daily  CMOR tree (table=day), one file per year:
#     $OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day/tasmin/gr/v1/
#         tasmin_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
#     $OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day/tasmax/gr/v1/
#         tasmax_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
#   Monthly (per-year, merged later by the finalize step):
#     $OUT/mon/_tmp/ERA5_tasmin_daymin_mon_${YEAR}.nc  (and tasmax)
#
# Daily tasmin/tasmax = min/max of the 24 hourly values over the UTC calendar
# day.  Interpolation weights are computed once and reused (remapbil is slow if
# weights are recomputed for every one of the ~365×2 daily calls).
#
# Usage:
#   sbatch --array=1980-2023 scripts/era5_derive_tasminmax.sh
#   # or single year, interactively on a compute node:
#   bash scripts/era5_derive_tasminmax.sh 1995
#
# After all year-jobs finish, run scripts/era5_derive_finalize.sh to merge the
# monthly files into ERA5_tasmin_daymin_mon_1981-2023.nc / tasmax.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=era5_tnmx
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/era5_derived/logs/era5_derive_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/pool/data/ERA5/E5/sf/an/1H/167
OUT=/work/bm1344/AWI/OBS/era5_derived
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)

CMOR_DAY=$OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day
MON_TMP=$OUT/mon/_tmp
# Levante scratch layout: /scratch/<first-letter-of-user>/<user>/
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/era5_derive_${YEAR}"

mkdir -p "$CMOR_DAY/tasmin/gr/v1" "$CMOR_DAY/tasmax/gr/v1" "$MON_TMP" "$WDIR"

echo "[$(date)] year=$YEAR  src=$SRC  out=$OUT  work=$WDIR"

# 1. Bilinear interpolation weights (reduced-Gaussian → 0.25°), computed once.
WTS=$WDIR/weights_bil.nc
SAMPLE=$(ls "$SRC"/E5sf00_1H_${YEAR}-01-*_167.grb | head -1)
cdo -s genbil,"$GRID" -setgridtype,regular "$SAMPLE" "$WTS"

# 2. Per-day daily min / max, then remap with the cached weights.
shopt -s nullglob
for f in "$SRC"/E5sf00_1H_${YEAR}-*_167.grb; do
    d=$(basename "$f" .grb)              # E5sf00_1H_YYYY-MM-DD_167
    cdo -s -f nc4 -setname,tasmin -remap,"$GRID","$WTS" -daymin -setgridtype,regular "$f" "$WDIR/min_$d.nc"
    cdo -s -f nc4 -setname,tasmax -remap,"$GRID","$WTS" -daymax -setgridtype,regular "$f" "$WDIR/max_$d.nc"
done

# 3. Concatenate the year into daily CMOR files.
#    (mergetime is variadic, so it must run on its own — not chained inside
#     another operator — and the units attribute is set in a separate pass.)
TASMIN_DAY=$CMOR_DAY/tasmin/gr/v1/tasmin_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
TASMAX_DAY=$CMOR_DAY/tasmax/gr/v1/tasmax_day_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/min_*.nc "$WDIR/tasmin_${YEAR}.nc"
cdo -s -O mergetime "$WDIR"/max_*.nc "$WDIR/tasmax_${YEAR}.nc"
cdo -s -O setattribute,tasmin@units=K "$WDIR/tasmin_${YEAR}.nc" "$TASMIN_DAY"
cdo -s -O setattribute,tasmax@units=K "$WDIR/tasmax_${YEAR}.nc" "$TASMAX_DAY"

# 4. Monthly means of the daily extremes (merged across years by finalize step).
cdo -s -O monmean "$TASMIN_DAY" "$MON_TMP/ERA5_tasmin_daymin_mon_${YEAR}.nc"
cdo -s -O monmean "$TASMAX_DAY" "$MON_TMP/ERA5_tasmax_daymax_mon_${YEAR}.nc"

rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ->  $TASMIN_DAY"
