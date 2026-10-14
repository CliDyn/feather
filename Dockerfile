# syntax=docker/dockerfile:1
#
# Feather container image: diagnostics, LLM analysis, LaTeX report, website
# and the test suite.  Runs under Docker and Singularity/Apptainer.
#
#   docker build -t feather:latest .
#   docker run --rm feather:latest --help
#
# See docker/README.md for bind mounts (Levante /work, /pool) and API keys.

FROM mambaorg/micromamba:2.0.5

# ── System packages (root) ──────────────────────────────────────────
# git: pip installs nereus from a pinned GitHub commit (environment.yml).
# TeX Live: pdflatex for `--steps report`; latex-extra carries parskip,
# lmodern the fonts the report template selects.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended lmodern \
 && rm -rf /var/lib/apt/lists/*

# ── Conda environment (from the same file developers use) ───────────
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# `singularity exec` bypasses the micromamba entrypoint that activates the
# environment, so put it on PATH and set what its activate.d scripts would.
ENV PATH=/opt/conda/bin:$PATH \
    PROJ_DATA=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal

# ── Offline data caches ─────────────────────────────────────────────
# Compute nodes have no internet: pre-fetch the Natural Earth shapes cartopy
# draws (coastlines/land, 110m and 50m) and the AR6 region polygons that
# regionmask downloads via pooch on first use.  Both live under /opt/feather-
# cache, read-only at run time (Singularity mounts the image read-only).
ENV CARTOPY_DATA_DIR=/opt/feather-cache/cartopy \
    XDG_CACHE_HOME=/opt/feather-cache
USER root
RUN mkdir -p /opt/feather-cache && chown $MAMBA_USER:$MAMBA_USER /opt/feather-cache
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/fetch_offline_data.py /tmp/fetch_offline_data.py
RUN python /tmp/fetch_offline_data.py

# ── Feather itself ──────────────────────────────────────────────────
USER root
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/feather
USER $MAMBA_USER
RUN pip install --no-deps --no-cache-dir /opt/feather \
 && feather --help > /dev/null

# Provenance records the commit; the image has no .git, so stamp it here.
ARG FEATHER_GIT_COMMIT=""
ARG FEATHER_GIT_BRANCH=""
ARG FEATHER_GIT_DIRTY=""
ENV FEATHER_GIT_COMMIT=$FEATHER_GIT_COMMIT \
    FEATHER_GIT_BRANCH=$FEATHER_GIT_BRANCH \
    FEATHER_GIT_DIRTY=$FEATHER_GIT_DIRTY
LABEL org.opencontainers.image.title="feather" \
      org.opencontainers.image.description="Lightweight climate model evaluation framework" \
      org.opencontainers.image.source="https://github.com/CliDyn/feather" \
      org.opencontainers.image.revision=$FEATHER_GIT_COMMIT

# Headless plotting; per-user writable matplotlib cache (see entrypoint).
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

WORKDIR /opt/feather
COPY --chown=$MAMBA_USER:$MAMBA_USER --chmod=755 docker/entrypoint.sh /usr/local/bin/feather-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/usr/local/bin/feather-entrypoint.sh"]
CMD ["--help"]
