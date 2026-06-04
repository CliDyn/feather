#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Derive the ERA5 land-sea mask (CMOR sftlf, land area fraction in %) from the
# invariant ERA5 land-sea mask field (code 172) and write it into the ERA5
# CMOR fx tree so CMORLoader can read it via table="fx".
#
# Source : /pool/data/ERA5/E5/sf/an/IV/172/E5sf00_IV_INVARIANT_172.grb
#          (lsm, 0..1 land fraction, native reduced-Gaussian grid)
# Output : $OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/fx/sftlf/gr/v1/
#              sftlf_fx_ERA5_era5_r1i1p1f1_gr.nc   (regular 0.25°, %)
#
# Single invariant field → light enough to run on a login node.
# Conservative remap (remapcon) preserves the land fraction at coastlines.
#
# Usage:  bash scripts/era5_derive_landmask.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
module load cdo 2>/dev/null || true

SRC=/pool/data/ERA5/E5/sf/an/IV/172/E5sf00_IV_INVARIANT_172.grb
OUT=/work/bm1344/AWI/OBS/era5_derived
GRID=r1440x721
FXDIR=$OUT/CMOR/ECMWF/ERA5/era5/r1i1p1f1/fx/sftlf/gr/v1

mkdir -p "$FXDIR"

cdo -s -f nc4 -setattribute,sftlf@units=% -setname,sftlf \
    -mulc,100 -remapcon,"$GRID" -setgridtype,regular \
    "$SRC" "$FXDIR/sftlf_fx_ERA5_era5_r1i1p1f1_gr.nc"

echo "Land-sea mask written:"
ls -la "$FXDIR/sftlf_fx_ERA5_era5_r1i1p1f1_gr.nc"
