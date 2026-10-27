#!/usr/bin/env bash
# Quick checks that a built image is usable.  Run inside the container:
#
#   docker run --rm feather:latest bash docker/smoke_test.sh
#   singularity exec feather_latest.sif bash /opt/feather/docker/smoke_test.sh
#
# Covers the CLI, the nereus build (conservative remapping is not in the 0.4.1
# PyPI release), offline data, LaTeX, templates, and a short unit-test subset.
# The full unit suite takes ~1.5 h; run it with `pytest tests/ -m "not integration"`.
set -euo pipefail
source /opt/feather/docker/env.sh
cd /opt/feather

echo "== host environment does not leak in"
python - <<'PY2'
import os
for var, want in [("GDAL_DATA", "/opt/conda/share/gdal"),
                  ("PROJ_DATA", "/opt/conda/share/proj")]:
    assert os.environ.get(var) == want, f"{var}={os.environ.get(var)!r}"
assert "LD_LIBRARY_PATH" not in os.environ, os.environ["LD_LIBRARY_PATH"]
print("ok")
PY2

echo "== feather CLI"
feather --help > /dev/null

echo "== provenance commit: ${FEATHER_GIT_COMMIT:-<unset>}"

echo "== conda packages all from conda-forge (no Anaconda channels)"
python - <<'PY'
import json, subprocess
pkgs = json.loads(subprocess.run(["micromamba", "list", "-n", "base", "--json"],
                                 capture_output=True, text=True, check=True).stdout)
if isinstance(pkgs, dict):  # newer micromamba: {"packages": [...], ...}
    pkgs = pkgs["packages"]
# pip-installed packages (nereus, gsw, ...) may be listed with channel "pypi"
bad = ["%s (%s)" % (p["name"], p["channel"]) for p in pkgs
       if p.get("channel") and p["channel"] != "pypi"
       and "conda-forge" not in p["channel"]]
assert not bad, "packages from non-conda-forge channels: " + ", ".join(bad)
print("ok, %d packages" % len(pkgs))
PY

echo "== nereus conservative remapping"
python - <<'PY'
from feather.diag.base import DiagnosticBase
assert DiagnosticBase._conservative_available(), \
    "nereus lacks method='conservative' -- installed from PyPI instead of the pinned commit?"
print("ok")
PY

echo "== offline data (no network needed)"
python - <<'PY'
import cartopy.io.shapereader as shp, regionmask
shp.natural_earth(resolution="110m", category="physical", name="coastline")
assert len(regionmask.defined_regions.ar6.all) == 58
print("ok")
PY

echo "== package data"
python - <<'PY'
from importlib.resources import files
root = files("feather")
for rel in ("website/templates/base.html", "website/static/style.css",
            "export/templates/report.tex.jinja2"):
    assert (root / rel).is_file(), rel
print("ok")
PY

echo "== pdflatex with the report template's packages"
work=$(mktemp -d)
cat > "$work/t.tex" <<'TEX'
\documentclass{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}\usepackage{graphicx}\usepackage{booktabs}
\usepackage{caption}\usepackage{fancyhdr}\usepackage{float}\usepackage{parskip}
\usepackage{xcolor}\usepackage{hyperref}
\begin{document}feather\end{document}
TEX
(cd "$work" && pdflatex -interaction=nonstopmode -halt-on-error t.tex > /dev/null)
test -s "$work/t.pdf" && echo "ok"
rm -rf "$work"

echo "== unit-test subset"
pytest -q -p no:cacheprovider -m "not integration" \
    tests/test_provenance.py tests/test_conservative_regrid.py tests/test_regrid_dedupe.py \
    tests/test_mhw.py tests/test_ar6_regions.py
echo "== all smoke checks passed"
