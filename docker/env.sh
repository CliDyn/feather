# Sourced by entrypoint.sh and smoke_test.sh.
#
# Singularity passes the host environment into the container, and host values
# win over the image's ENV defaults.  On Levante a loaded conda/module setup
# exports GDAL_DATA, PROJ_LIB, LD_LIBRARY_PATH, ... pointing at host installs
# of other versions (pyogrio then refuses the host's GDAL data and regionmask
# fails to import).  Drop those and pin the image's own paths.  Docker does not
# pass the host environment, so this is a no-op there.

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME PYTHONSTARTUP \
      PROJ_LIB GDAL_DRIVER_PATH HDF5_PLUGIN_PATH UDUNITS2_XML_PATH \
      CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE

export PATH="/opt/conda/bin:${PATH}"
export PROJ_DATA=/opt/conda/share/proj
export GDAL_DATA=/opt/conda/share/gdal
export CARTOPY_DATA_DIR=/opt/feather-cache/cartopy
export XDG_CACHE_HOME=/opt/feather-cache
# $HOME is bind-mounted: keep packages from ~/.local out of the image's Python.
export PYTHONNOUSERSITE=1

# The image is read-only under Singularity: matplotlib needs a writable
# per-user config/cache directory.
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/feather-mpl-$(id -u)"
fi
mkdir -p "$MPLCONFIGDIR"
