#!/usr/bin/env bash
# Entry point of the feather image.
#
#   <no args> / --flags ...   -> feather --flags ...
#   any other command         -> run it (bash, pytest, python, ...)
#
# env.sh drops host variables Singularity passes in (GDAL_DATA, PROJ_LIB,
# LD_LIBRARY_PATH, ...) and sets a writable per-user matplotlib directory.
set -euo pipefail
source /opt/feather/docker/env.sh

if [[ $# -eq 0 || "$1" == -* ]]; then
    exec feather "$@"
fi
exec "$@"
