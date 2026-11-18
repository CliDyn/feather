#!/bin/bash
#SBATCH --job-name=copy_obs
#SBATCH --partition=shared
#SBATCH --account=bm1344
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bm1344/AWI/OBS/aqua/logs/copy_obs_%j.out
#
# Copy the observations feather reads from the (unsupported, pre-DVC) AQUA
# tree of b382289, plus the ESA-CCI SST files of a270088, into
# /work/bm1344/AWI/OBS/aqua.  COPY only — sources are never modified.
#
# The old obs_root (/work/bb1153/b382289/data/aqua-dvc/datasets) was deleted
# around 2026-10-14; data_old/AQUA/datasets is what remains.  Several datasets
# there are older versions than the configs name — the target directories are
# named after the version actually copied, so configs must be updated
# afterwards (see README.txt written into the destination).
#
# Usage:
#   sbatch scripts/copy_obs_to_bm1344.sh          # copy
#   DRY_RUN=1 bash scripts/copy_obs_to_bm1344.sh  # list what would be copied
#
# Estimated volume: ~175 GB (ERA5 48.5, ESA-CCI 82, EN4 36, MSWEP 5.5,
# CERES 2.6, PSC 0.6, Berkeley 0.1).  Re-running resumes / skips done files.

set -euo pipefail

SRC=/work/bb1153/b382289/data_old/AQUA/datasets
MSWEP_SRC=/work/bb1153/b382289/MSWEP/mswep280.past-nrt.monthly.zarr
ESACCI_SRC=/work/ab0995/a270088/DestinE/OBS/ESA-CCI_temp
DEST=${DEST:-/work/bm1344/AWI/OBS/aqua}

# -a archive, -L follow symlinks (ERA5/EN4/PSC point into other users' dirs),
# no owner/group (copies belong to us, group from the setgid bm1344 dir).
RSYNC_OPTS=(-aL --no-owner --no-group --chmod=D2755,F644 --partial --info=progress2,stats1)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    RSYNC_OPTS+=(--dry-run)
fi

ERA5_VARS=(
    10m_u_component_of_wind 10m_v_component_of_wind 2m_temperature
    evaporation mean_sea_level_pressure sea_ice_cover sea_surface_temperature
    surface_latent_heat_flux surface_sensible_heat_flux
    surface_net_solar_radiation surface_net_solar_radiation_clear_sky
    surface_net_thermal_radiation surface_net_thermal_radiation_clear_sky
    surface_solar_radiation_downwards surface_thermal_radiation_downwards
    top_net_solar_radiation top_net_solar_radiation_clear_sky
    top_net_thermal_radiation top_net_thermal_radiation_clear_sky
    total_cloud_cover total_column_cloud_ice_water
    total_column_cloud_liquid_water total_precipitation
)

copy() {  # copy <src> <dest-dir>   (src: file, or dir with trailing /)
    echo "=== $(date '+%F %T')  $1  ->  $2"
    [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$2"
    rsync "${RSYNC_OPTS[@]}" "$1" "$2"
}

[[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$DEST/logs"

# --- ERA5 monthly surface (23 files used by the configs, 1940-2025) --------
for v in "${ERA5_VARS[@]}"; do
    copy "$SRC/ERA5/mon/ERA5_${v}_mon_full_sfc_1940-2025.nc" "$DEST/ERA5/mon/"
done

# --- MSWEP v2.8 past+NRT monthly zarr (same store the configs name) --------
copy "$MSWEP_SRC" "$DEST/MSWEP/v2.8/zarr/"

# --- Berkeley Earth 1 deg aqua-filled (same file the configs name) ---------
copy "$SRC/BERKELEY-EARTH/aqua-filled/" "$DEST/BERKELEY-EARTH/aqua-filled/"

# --- CERES EBAF Ed4.2 (configs want Ed4.2.1, not available) ----------------
copy "$SRC/CERES/EBAF_v4.2/" "$DEST/CERES/EBAF_v4.2/"

# --- EN4 thetao/so 1950-2022 (configs want v4.2.2 1950-2024) ---------------
copy "$SRC/EN4/so-1950_2022.nc"     "$DEST/EN4/1950-2022/"
copy "$SRC/EN4/thetao-1950_2022.nc" "$DEST/EN4/1950-2022/"

# --- PIOMAS / GIOMAS sivol 1979-2022 (configs want sithick 1979-2024) -----
copy "$SRC/PSC/GIOMAS/" "$DEST/PSC/GIOMAS/"
copy "$SRC/PSC/PIOMAS/" "$DEST/PSC/PIOMAS/"

# --- ESA-CCI L4 SST v3.0.1 monthly (a270088) -------------------------------
copy "$ESACCI_SRC/" "$DEST/ESA-CCI-L4/v3.0.1/"

[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

# --- Verify: every source file present at the destination with equal size -
echo "=== $(date '+%F %T')  verifying sizes"
fail=0
check() {  # check <src> <dest-dir>
    local s=$1 d=$2
    if [[ -d "$s" ]]; then
        while IFS= read -r -d '' f; do
            local rel=${f#"$s"/}
            [[ $(stat -Lc %s "$f") == $(stat -c %s "$d/$rel" 2>/dev/null) ]] \
                || { echo "MISMATCH $f"; fail=1; }
        done < <(find -L "$s" -type f -print0)
    else
        [[ $(stat -Lc %s "$s") == $(stat -c %s "$d/$(basename "$s")" 2>/dev/null) ]] \
            || { echo "MISMATCH $s"; fail=1; }
    fi
}
for v in "${ERA5_VARS[@]}"; do
    check "$SRC/ERA5/mon/ERA5_${v}_mon_full_sfc_1940-2025.nc" "$DEST/ERA5/mon"
done
check "$MSWEP_SRC" "$DEST/MSWEP/v2.8/zarr/$(basename "$MSWEP_SRC")"
check "$SRC/BERKELEY-EARTH/aqua-filled" "$DEST/BERKELEY-EARTH/aqua-filled"
check "$SRC/CERES/EBAF_v4.2" "$DEST/CERES/EBAF_v4.2"
check "$SRC/EN4/so-1950_2022.nc" "$DEST/EN4/1950-2022"
check "$SRC/EN4/thetao-1950_2022.nc" "$DEST/EN4/1950-2022"
check "$SRC/PSC/GIOMAS" "$DEST/PSC/GIOMAS"
check "$SRC/PSC/PIOMAS" "$DEST/PSC/PIOMAS"
check "$ESACCI_SRC" "$DEST/ESA-CCI-L4/v3.0.1"

cat > "$DEST/README.txt" <<EOF
Observations copied for feather on $(date '+%F') by $(whoami)
(scripts/copy_obs_to_bm1344.sh).  Sources:
  $SRC   (b382289 pre-DVC AQUA tree, "not supported anymore";
          ERA5 -> b382076, EN4 -> b382119, PSC -> b382075/b382385)
  $MSWEP_SRC
  $ESACCI_SRC

Versions differ from what the configs used to read from aqua-dvc:
  ERA5/mon            file names end _1940-2025 (configs: _1940-2024)
  CERES/EBAF_v4.2     Ed4.2, TOA to 2023-01, surface to 2022-03 (configs: Ed4.2.1 to 2025-06)
  EN4/1950-2022       so/thetao 1950-2022 (configs: v4.2.2 1950-2024)
  PSC/{GIOMAS,PIOMAS} sivol 1979-2022 (configs: sithick 1979-2024)
  OSI-SAF             NOT copied (monthly v3 1979-2024 files not in the old tree)
Unchanged: MSWEP/v2.8/zarr/mswep280.past-nrt.monthly.zarr,
           BERKELEY-EARTH/aqua-filled, ESA-CCI-L4/v3.0.1
EOF

du -sh "$DEST"/*
if [[ $fail == 0 ]]; then echo "=== DONE, all sizes match"; else echo "=== DONE WITH MISMATCHES"; exit 1; fi
