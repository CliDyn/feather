#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily NOAA OISST v2.1 sea-surface temperature on the regular 0.25°
# grid of the EERIE models (CMOR tree).
#
# Source : /pool/data/ICDC/ocean/reynolds_sst/DATA/DAILY/YYYY/
#              oisst-avhrr-v02r01.YYYYMMDD.nc
#          (AVHRR-only OISST v2.1, daily, 0.25° cell-centred 720×1440, degC,
#           one file per day; September 1981 onwards)
#
# Output (regular 0.25° lat/lon, 1440×721 incl. poles, degC):
#   $OUT/CMOR/NOAA/OISST/oisst/r1i1p1f1/Oday/tos/gr/v1/
#       tos_Oday_OISST_oisst_r1i1p1f1_gr_${YEAR}.nc
#
# OISST's cells are centred on ±89.875…; the models' 0.25° grid has nodes at
# the poles and every quarter degree, so the field is bilinearly remapped
# (cdo remapbil, as de Toma et al. 2026 do) onto r1440x721.  Time stamps are
# set to 12:00 like the CMOR model output.  The year is skipped with an error
# if a day is missing (daily files are complete on ICDC).
#
# Usage:
#   sbatch --array=1982-2014 scripts/oisst_derive_tos_daily.sh
#   bash scripts/oisst_derive_tos_daily.sh 1995          # one year, compute node
#   OUT=/some/test/dir bash scripts/oisst_derive_tos_daily.sh 1995
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=oisst_tos_day
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/oisst_derived/logs/oisst_tos_day_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/pool/data/ICDC/ocean/reynolds_sst/DATA/DAILY/$YEAR
OUT="${OUT:-/work/bm1344/AWI/OBS/oisst_derived}"
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)

DEST=$OUT/CMOR/NOAA/OISST/oisst/r1i1p1f1/Oday/tos/gr/v1
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/oisst_tos_day_${YEAR}"
mkdir -p "$DEST" "$WDIR"

echo "[$(date)] year=$YEAR  src=$SRC  out=$DEST  work=$WDIR"

shopt -s nullglob
files=("$SRC"/oisst-avhrr-v02r01.${YEAR}????.nc)
expected=$(( $(date -d "${YEAR}-12-31" +%j) ))
if [ "$YEAR" -eq 1981 ]; then
    expected=$(( expected - $(date -d "1981-08-31" +%j) ))   # starts 1 Sept
fi
if [ "${#files[@]}" -ne "$expected" ]; then
    echo "ERROR: expected $expected daily files for $YEAR, found ${#files[@]}" >&2
    exit 1
fi

# 1. Bilinear weights (OISST 0.25° cell centres → 0.25° nodes), computed once.
WTS=$WDIR/weights_bil.nc
cdo -s genbil,"$GRID" -selname,sst "${files[0]}" "$WTS"

# 2. One month at a time: merge the daily files, keep sst, remap.
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    mfiles=("$SRC"/oisst-avhrr-v02r01.${YEAR}${m}??.nc)
    [ "${#mfiles[@]}" -eq 0 ] && continue
    cdo -s -O mergetime -apply,-selname,sst [ "${mfiles[@]}" ] "$WDIR/raw_${m}.nc"
    cdo -s -f nc4 -b F32 -settime,12:00:00 -setname,tos \
        -remap,"$GRID","$WTS" "$WDIR/raw_${m}.nc" "$WDIR/tos_${m}.nc"
    rm -f "$WDIR/raw_${m}.nc"
done

# 3. Concatenate the year and set CMOR attributes in a separate pass.
TARGET=$DEST/tos_Oday_OISST_oisst_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/tos_??.nc "$WDIR/tos_${YEAR}.nc"
ndays=$(cdo -s ntime "$WDIR/tos_${YEAR}.nc")
cdo -s -O -z zip_1 \
    -setattribute,tos@units="degC",tos@cell_methods="area: mean where sea time: mean",tos@standard_name=sea_surface_temperature,tos@long_name="Sea Surface Temperature",source="NOAA OISST v2.1 (AVHRR-only) daily SST from ICDC; bilinearly remapped to the 0.25 deg r1440x721 grid" \
    "$WDIR/tos_${YEAR}.nc" "$TARGET"

mean=$(cdo -s outputf,%.3f,1 -fldmean -timmean "$TARGET")
rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ($ndays days, ocean mean $mean degC)  ->  $TARGET"
