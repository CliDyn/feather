#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive daily ERA5 sea-surface temperature on the regular 0.25° grid (CMOR tree).
#
# Source : /pool/data/ERA5/E5/sf/an/1D/034/E5sf00_1D_YYYY-MM_034.grb
#          (daily-mean SST, K, reduced-Gaussian N320, land missing, one file
#           per month, time stamp 11:30)
#
# Output (regular 0.25° lat/lon, 1440×721 incl. poles, degC):
#   $OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/Oday/tos/gr/v1/
#       tos_Oday_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
#
# Remapping is bilinear (cdo remapbil, as de Toma et al. 2026 do for their
# marine-heatwave inputs) after expanding the reduced Gaussian grid
# (setgridtype,regular).  K → degC; time
# stamps are set to 12:00 like the CMOR model output.  ERA5 SST is prescribed
# from HadISST2 before September 2007 and from OSTIA afterwards.
#
# Usage:
#   sbatch --array=1982-2014 scripts/era5_derive_tos_daily.sh
#   bash scripts/era5_derive_tos_daily.sh 1995          # one year, compute node
#   OUT=/some/test/dir bash scripts/era5_derive_tos_daily.sh 1995
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=era5_tos_day
#SBATCH --account=bm1344
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/era5_derived/logs/era5_tos_day_%A_%a.out

set -euo pipefail

module load cdo 2>/dev/null || true

YEAR="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch --array=YYYY-YYYY $0  OR  bash $0 YEAR}}"

SRC=/pool/data/ERA5/E5/sf/an/1D/034
OUT="${OUT:-/work/bm1344/AWI/OBS/era5_derived}"
GRID=r1440x721                       # global regular 0.25° lat/lon (poles incl.)

DEST=$OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/Oday/tos/gr/v1
WDIR="${SCRATCH:-/scratch/${USER:0:1}/$USER}/era5_tos_day_${YEAR}"
mkdir -p "$DEST" "$WDIR"

echo "[$(date)] year=$YEAR  src=$SRC  out=$DEST  work=$WDIR"

shopt -s nullglob
files=("$SRC"/E5sf00_1D_${YEAR}-??_034.grb)
if [ "${#files[@]}" -ne 12 ]; then
    echo "ERROR: expected 12 monthly files for $YEAR, found ${#files[@]}" >&2
    exit 1
fi

# 1. Remap each month and convert K → degC.  Precomputed weights would be
#    ignored: the land-masked SST grid does not match the unmasked grid the
#    weights were built on, so cdo rebuilds them per call anyway.
for f in "${files[@]}"; do
    m=$(basename "$f" .grb)              # E5sf00_1D_YYYY-MM_034
    cdo -s -f nc4 -b F32 -settime,12:00:00 -setname,tos \
        -subc,273.15 -remapbil,"$GRID" -setgridtype,regular \
        "$f" "$WDIR/$m.nc"
done

# 2. Concatenate the year (mergetime is variadic, so it runs on its own) and
#    set CMOR attributes in a separate pass.
TARGET=$DEST/tos_Oday_ERA5_era5_r1i1p1f1_gr_${YEAR}.nc
cdo -s -O mergetime "$WDIR"/E5sf00_1D_*.nc "$WDIR/tos_${YEAR}.nc"
ndays=$(cdo -s ntime "$WDIR/tos_${YEAR}.nc")
cdo -s -O -z zip_1 \
    -setattribute,tos@units="degC",tos@cell_methods="area: mean where sea time: mean",tos@standard_name=sea_surface_temperature,tos@long_name="Sea Surface Temperature",source="ERA5 daily-mean SST (code 34); bilinearly remapped from N320 to 0.25 deg" \
    "$WDIR/tos_${YEAR}.nc" "$TARGET"

mean=$(cdo -s outputf,%.3f,1 -fldmean -timmean "$TARGET")
rm -rf "$WDIR"
echo "[$(date)] year=$YEAR DONE  ($ndays days, ocean mean $mean degC)  ->  $TARGET"
