#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily MSWEP v2.8 precipitation on the regular 0.25° grid (CMOR tree).
#
# Source : /pool/data/ICDC/atmosphere/mswep_precipitation/DATA/YYYY/YYYYDDD.HH.nc
#          (ICDC copy of MSWEP v2.8: 0.1° lat/lon 3600×1800, 3-hourly
#           accumulations in mm 3h-1, one file per time step, HH = 00..21)
#
# Output (regular 0.25° lat/lon, 1440×721 incl. poles, kg m-2 s-1):
#   $OUT/CMOR/GloH2O/MSWEP/mswep/r1i1p1f1/day/pr/gr/v1/
#       pr_day_MSWEP_mswep_r1i1p1f1_gr_${YEAR}.nc
#
# Daily total = sum of the eight 3-hourly steps 00..21 of the day (MSWEP time
# stamps mark the start of each 3-h interval, so this is 00–24 UTC).  A day
# with fewer than eight steps is skipped and logged, never summed short — the
# index code then treats it as missing.  The 0.1° → 0.25° remap is
# area-conservative (cdo remapcon) with weights built once per job.
# mm/day → kg m-2 s-1 is ÷86400; time stamps are set to 12:00.
#
# Cost: ~2 s per day (≈15 min per year) and ~3 GB RSS.
#
# Usage:
#   sbatch --array=1981-2014 scripts/mswep_derive_pr_daily.sh
#   bash scripts/mswep_derive_pr_daily.sh 1995          # one year, compute node
#   OUT=/tmp/x MAX_DAYS=3 bash scripts/mswep_derive_pr_daily.sh 1995   # smoke test
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=mswep_pr_day
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/mswep_derived/logs/mswep_pr_day_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/pool/data/ICDC/atmosphere/mswep_precipitation/DATA/$YEAR
OUT="${OUT:-/work/bm1344/AWI/OBS/mswep_derived}"
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)
MAX_DAYS="${MAX_DAYS:-366}"          # smoke-test knob

DEST=$OUT/CMOR/GloH2O/MSWEP/mswep/r1i1p1f1/day/pr/gr/v1
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/mswep_pr_day_${YEAR}"
mkdir -p "$DEST" "$WDIR/day"

echo "[$(date)] year=$YEAR  src=$SRC  out=$DEST  work=$WDIR"

# 1. Conservative weights (0.1° → 0.25°), computed once.
WTS=$WDIR/weights_con.nc
cdo -s -P "${SLURM_CPUS_PER_TASK:-4}" gencon,"$GRID" "$SRC/${YEAR}001.00.nc" "$WTS"

# 2. Daily sums of the eight 3-hourly steps, remapped with the cached weights.
ndays_year=$(( ( $(date -ud "$((YEAR + 1))-01-01" +%s) - $(date -ud "$YEAR-01-01" +%s) ) / 86400 ))
skipped=()
for doy in $(seq 1 "$ndays_year"); do
    [ "$doy" -gt "$MAX_DAYS" ] && break
    ddd=$(printf "%03d" "$doy")
    steps=()
    for hh in 00 03 06 09 12 15 18 21; do
        f="$SRC/${YEAR}${ddd}.${hh}.nc"
        [ -f "$f" ] && steps+=("$f")
    done
    if [ "${#steps[@]}" -ne 8 ]; then
        skipped+=("$ddd(${#steps[@]}/8)")
        continue
    fi
    date_str=$(date -ud "$YEAR-01-01 +$((doy - 1)) days" +%Y-%m-%d)
    cdo -s -O -timsum -mergetime "${steps[@]}" "$WDIR/sum.nc"
    cdo -s -O -f nc4 -b F32 -settaxis,"$date_str",12:00:00,1day -setname,pr \
        -divc,86400 -remap,"$GRID","$WTS" "$WDIR/sum.nc" "$WDIR/day/${ddd}.nc"
done
if [ "${#skipped[@]}" -gt 0 ]; then
    echo "WARNING: $YEAR: ${#skipped[@]} incomplete day(s) skipped: ${skipped[*]}"
fi

# 3. Concatenate the year and set CMOR attributes in a separate pass.
TARGET=$DEST/pr_day_MSWEP_mswep_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/day/*.nc "$WDIR/pr_${YEAR}.nc"
ndays=$(cdo -s ntime "$WDIR/pr_${YEAR}.nc")
cdo -s -O -z zip_1 \
    -setattribute,pr@least_significant_digit=,pr@units="kg m-2 s-1",pr@cell_methods="time: mean",pr@standard_name=precipitation_flux,pr@long_name="Precipitation",source="MSWEP v2.8 (ICDC copy); daily sum of 3-hourly steps 00-21 UTC; conservatively remapped from 0.1 to 0.25 deg" \
    "$WDIR/pr_${YEAR}.nc" "$TARGET"

mean=$(cdo -s outputf,%.3f,1 -mulc,86400 -fldmean -timmean "$TARGET")
rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ($ndays days, global mean $mean mm/day)  ->  $TARGET"
