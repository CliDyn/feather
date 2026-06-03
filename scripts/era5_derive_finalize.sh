#!/bin/bash
# Merge the per-year monthly ERA5 tasmin/tasmax files (produced by
# era5_derive_tasminmax.sh) into single 1981–2023 monthly files used by the
# observation-comparison diagnostics.
#
# Usage (login node is fine — this is a light mergetime):
#   bash scripts/era5_derive_finalize.sh
set -euo pipefail
module load cdo 2>/dev/null || true

OUT=/work/bm1344/AWI/OBS/era5_derived
MON=$OUT/mon
TMP=$MON/_tmp

cdo -s -O mergetime "$TMP"/ERA5_tasmin_daymin_mon_*.nc "$MON/ERA5_tasmin_daymin_mon_1981-2023.nc"
cdo -s -O mergetime "$TMP"/ERA5_tasmax_daymax_mon_*.nc "$MON/ERA5_tasmax_daymax_mon_1981-2023.nc"

echo "Monthly files written:"
ls -la "$MON"/ERA5_tas*_mon_1981-2023.nc
