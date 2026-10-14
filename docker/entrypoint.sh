#!/usr/bin/env bash
# Entry point of the feather image.
#
#   <no args> / --flags ...   -> feather --flags ...
#   any other command         -> run it (bash, pytest, python, ...)
#
# Singularity runs the image read-only as the calling user, so matplotlib
# gets a writable per-user config/cache directory under $TMPDIR.
set -euo pipefail

if [[ -z "${MPLCONFIGDIR:-}" ]]; then
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/feather-mpl-$(id -u)"
fi
mkdir -p "$MPLCONFIGDIR"

if [[ $# -eq 0 || "$1" == -* ]]; then
    exec feather "$@"
fi
exec "$@"
