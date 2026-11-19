#!/bin/bash
#SBATCH --job-name=osisaf_mon
#SBATCH --partition=shared
#SBATCH --account=bm1344
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/aqua/OSI-SAF/logs/osisaf_mon_%j.out
#
# Monthly OSI-SAF sea-ice concentration (OSI-450-a CDR v3.0 to 2020 + OSI-430-a
# ICDR v3.0 from 2021) from the ICDC daily files, for feather's sea_ice
# diagnostic (obs_datasets.OSI_SAF).
#
#   in : /pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA/YYYY/MM/
#        ice_conc_{nh,sh}_ease2-250_{cdr,icdr}-v3p0_YYYYMMDD1200.nc
#   out: $OUT/ice_conc_{nh,sh}_ease2-250_cdr-icdr-v3p0_monthly_${Y0}-${Y1}.nc
#
# Each month = plain mean of the daily ice_conc fields (%, EASE2 25 km,
# 432x432, 2-D lat/lon kept), stamped on the 15th at 12:00.  A month with
# fewer than MIN_DAYS daily fields is left out (not written as missing):
# 1979-mid-1987 is SMMR every-other-day (~15 fields/month), and 1986-04/05
# (0), 1986-06 (4), 1987-01 (9) and 1987-12 (2) are too thin.
# Skipped months are listed in $OUT/skipped_months.txt.
#
# Usage:  sbatch scripts/osisaf_derive_siconc_monthly.sh
#         Y0=1979 Y1=2024 MIN_DAYS=10 (environment overrides)

set -euo pipefail

SRC=/pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA
OUT=${OUT:-/work/bm1344/AWI/OBS/aqua/OSI-SAF/v3p0/monthly}
Y0=${Y0:-1979}
Y1=${Y1:-2024}
MIN_DAYS=${MIN_DAYS:-10}
NPROC=${SLURM_CPUS_PER_TASK:-4}

module load cdo 2>/dev/null || true
command -v cdo >/dev/null

TMP=$OUT/tmp_months
mkdir -p "$OUT" "$TMP"
: > "$OUT/skipped_months.txt"

one_month() {  # one_month <hemi> <YYYY> <MM>
    local h=$1 y=$2 m=$3
    local out=$TMP/${h}_${y}${m}.nc
    [[ -s $out ]] && return 0
    shopt -s nullglob
    local files=("$SRC/$y/$m"/ice_conc_${h}_ease2-250_*-v3p0_${y}${m}??1200.nc)
    if (( ${#files[@]} < MIN_DAYS )); then
        echo "$h $y-$m ${#files[@]}" >> "$OUT/skipped_months.txt"
        return 0
    fi
    cdo -s -O --no_history -f nc4 -z zip_4 \
        -settime,12:00:00 -setday,15 -timmean \
        -mergetime -apply,-selname,ice_conc [ "${files[@]}" ] \
        "$out.part" 2> >(grep -v "Time bounds unsupported" >&2)
    mv "$out.part" "$out"
}
export -f one_month
export SRC TMP OUT MIN_DAYS

echo "=== $(date '+%F %T') monthly means $Y0-$Y1 with $NPROC processes"
for h in nh sh; do
    for y in $(seq "$Y0" "$Y1"); do
        for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
            echo "$h $y $m"
        done
    done
done | xargs -P "$NPROC" -n 3 bash -c 'one_month "$@"' _

for h in nh sh; do
    final=$OUT/ice_conc_${h}_ease2-250_cdr-icdr-v3p0_monthly_${Y0}-${Y1}.nc
    echo "=== $(date '+%F %T') merging $h -> $final"
    cdo -s -O --no_history -f nc4 -z zip_4 mergetime "$TMP/${h}_"*.nc "$final.part"
    cdo -s -O --no_history \
        -setattribute,ice_conc@long_name="monthly mean sea ice area fraction" \
        -setattribute,history_feather="monthly means of ICDC daily OSI-SAF v3.0 ice_conc - scripts/osisaf_derive_siconc_monthly.sh - months with fewer than ${MIN_DAYS} daily fields omitted" \
        "$final.part" "$final"
    rm -f "$final.part"
    echo "    $(cdo -s ntime "$final") months"
done

sort -o "$OUT/skipped_months.txt" "$OUT/skipped_months.txt"
echo "=== skipped months (hemi month n_daily_files):"
cat "$OUT/skipped_months.txt"
rm -rf "$TMP"
ls -la "$OUT"
echo "=== DONE"
