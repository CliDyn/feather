#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily and monthly AVISO sea-surface height (absolute dynamic
# topography) on the regular 0.25° grid of the EERIE models (CMOR tree).
#
# Source : /work/uc0928/FUTURE_DATA/aviso_allsat/DATA/YYYY/MM/
#              dt_global_allsat_phy_l4_YYYYMMDD_<processing date>.nc
#          (CMEMS SSALTO/DUACS DT2024 "allsat" L4, daily, 0.125° cell-centred
#           1440×2880, adt in m; 1993-01-01 onwards).  These are the files
#          behind the EERIE intake entry
#          dkrz.disk.observations.AVISO.ssh-L4.daily_multi
#          (kerchunk /work/bm1344/k202193/Kerchunk/Obs/aviso_allsat.parq).
#
# Output (regular 0.25° lat/lon, 1440×721 incl. poles, m):
#   $OUT/CMOR/CMEMS/AVISO/aviso/r1i1p1f1/Oday/zos/gr/v1/
#       zos_Oday_AVISO_aviso_r1i1p1f1_gr_${YEAR}.nc
#   $OUT/CMOR/CMEMS/AVISO/aviso/r1i1p1f1/Omon/zos/gr/v1/
#       zos_Omon_AVISO_aviso_r1i1p1f1_gr_${YEAR}.nc        (monthly means)
#
# adt is remapped area-conservatively (cdo remapcon; four 0.125° cells per
# 0.25° cell, partly-ocean target cells average their ocean part).  Set
# METHOD=bil for bilinear instead.  Time stamps are set to 12:00 like the
# CMOR model output.  The year is skipped with an error if a day is missing.
#
# Usage:
#   sbatch --array=1993-2014 scripts/aviso_derive_zos_daily.sh
#   bash scripts/aviso_derive_zos_daily.sh 1995          # one year, compute node
#   OUT=/some/test/dir bash scripts/aviso_derive_zos_daily.sh 1995
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=aviso_zos_day
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/aviso_derived/logs/aviso_zos_day_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/work/uc0928/FUTURE_DATA/aviso_allsat/DATA/$YEAR
OUT="${OUT:-/work/bm1344/AWI/OBS/aviso_derived}"
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)
METHOD="${METHOD:-con}"              # con (area-conservative) | bil

BASE=$OUT/CMOR/CMEMS/AVISO/aviso/r1i1p1f1
DEST=$BASE/Oday/zos/gr/v1
DEST_MON=$BASE/Omon/zos/gr/v1
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/aviso_zos_day_${YEAR}"
mkdir -p "$DEST" "$DEST_MON" "$WDIR"

echo "[$(date)] year=$YEAR  src=$SRC  out=$BASE  method=$METHOD  work=$WDIR"

shopt -s nullglob
files=("$SRC"/??/dt_global_allsat_phy_l4_${YEAR}????_*.nc)
expected=$(( $(date -d "${YEAR}-12-31" +%j) ))
ndates=$(printf '%s\n' "${files[@]##*/}" | cut -c25-32 | sort -u | wc -l)
if [ "${#files[@]}" -ne "$expected" ] || [ "$ndates" -ne "$expected" ]; then
    echo "ERROR: expected $expected daily files for $YEAR, found ${#files[@]}" \
         "(${ndates} distinct dates)" >&2
    exit 1
fi

# 1. Remap weights (AVISO 0.125° cells → 0.25° grid), computed once.
WTS=$WDIR/weights_${METHOD}.nc
cdo -s gen${METHOD},"$GRID" -selname,adt "${files[0]}" "$WTS"

# 2. One month at a time: merge the daily files, keep adt, remap.
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    mfiles=("$SRC"/$m/dt_global_allsat_phy_l4_${YEAR}${m}??_*.nc)
    [ "${#mfiles[@]}" -eq 0 ] && continue
    cdo -s -O mergetime -apply,-selname,adt [ "${mfiles[@]}" ] "$WDIR/raw_${m}.nc"
    cdo -s -f nc4 -b F32 -settime,12:00:00 -setname,zos \
        -remap,"$GRID","$WTS" "$WDIR/raw_${m}.nc" "$WDIR/zos_${m}.nc"
    rm -f "$WDIR/raw_${m}.nc"
done

# 3. Concatenate the year; set CMOR attributes in a separate pass.
ATTRS="zos@units=m,zos@standard_name=sea_surface_height_above_geoid,zos@long_name=AVISO absolute dynamic topography,zos@cell_methods=area: mean where sea time: mean"
SRCTXT="CMEMS SSALTO-DUACS DT2024 allsat L4 adt daily from /work/uc0928/FUTURE_DATA/aviso_allsat remapped with cdo remap${METHOD} to the 0.25 deg r1440x721 grid"
TARGET=$DEST/zos_Oday_AVISO_aviso_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/zos_??.nc "$WDIR/zos_${YEAR}.nc"
ndays=$(cdo -s ntime "$WDIR/zos_${YEAR}.nc")
cdo -s -O -z zip_1 \
    "-setattribute,${ATTRS},source=${SRCTXT}" \
    "$WDIR/zos_${YEAR}.nc" "$TARGET"

# 4. Monthly means of the remapped daily fields (stamped mid-month).
TARGET_MON=$DEST_MON/zos_Omon_AVISO_aviso_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O monmean "$TARGET" "$WDIR/zos_mon_${YEAR}.nc"
cdo -s -O -z zip_1 \
    "-setattribute,${ATTRS},source=${SRCTXT} - monthly means of the daily fields" \
    -settime,12:00:00 -setday,15 "$WDIR/zos_mon_${YEAR}.nc" "$TARGET_MON"

std=$(cdo -s outputf,%.4f,1 -fldmean -timstd "$TARGET")
rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ($ndays days, ocean-mean daily std $std m)"
echo "    -> $TARGET"
echo "    -> $TARGET_MON"
