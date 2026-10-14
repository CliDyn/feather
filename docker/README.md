# Feather container image

One image for every pipeline step — diagnostics, LLM analysis, LaTeX report,
website — plus the test suite. It runs under Docker (laptop, cloud, CI) and
under Singularity/Apptainer (Levante, which has no Docker daemon).

## Contents

| Layer | What |
|---|---|
| Base | `mambaorg/micromamba` (Debian), environment from `environment.yml`, **conda-forge only** — no Anaconda channels (the smoke test fails if any package comes from elsewhere) |
| nereus | pinned GitHub commit from `environment.yml` (PyPI 0.4.1 lacks conservative remapping) |
| TeX Live | `pdflatex` + the packages the report template uses |
| Offline data | Natural Earth coastline/land/ocean (110m, 50m) and regionmask AR6 polygons, under `/opt/feather-cache` |
| feather | installed from the checkout at `/opt/feather` (tests and configs included) |

The image tracks the latest `main`: `ghcr.io/clidyn/feather:latest`. Every
build is also tagged `:sha-<short commit>` so a run can be pinned to an exact
version. The commit is stamped into the image (`FEATHER_GIT_COMMIT`) so
provenance run records and `feather verify --checks code` still see it,
although the image has no `.git`.

## Build

**GitHub** — `.github/workflows/docker.yml` builds on every push to `main`
(and `v*` tags), runs `docker/smoke_test.sh` in the image, and pushes to
`ghcr.io/<owner>/feather`. Pull requests that touch the image are built and
smoke-tested but not pushed. It can also be started by hand (Actions → docker →
Run workflow).

**Local** (any machine with Docker; not the Levante login node):

```bash
docker build -t feather:latest \
  --build-arg FEATHER_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg FEATHER_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
  --build-arg FEATHER_GIT_DIRTY=$([ -n "$(git status --porcelain --untracked-files=no)" ] && echo true || echo false) \
  .
docker run --rm feather:latest bash docker/smoke_test.sh
```

## Run with Docker

Arguments starting with `-` go to `feather`; anything else runs as a command.

```bash
docker run --rm feather:latest --help

docker run --rm \
  -v /path/to/data:/path/to/data:ro \
  -v "$PWD/output:/output" \
  -v "$PWD/configs:/configs:ro" \
  -e VERTEX_API_KEY -e OPENAI_API_KEY \
  feather:latest --config /configs/my_project.yaml --output /output -v

docker run --rm -it feather:latest bash                  # shell
docker run --rm feather:latest pytest tests/ -m "not integration"   # unit tests
```

Data paths in the configs are absolute, so mount the data at the **same path**
inside the container as on the host.

## Run on Levante (Singularity)

Pull once; the conversion is light enough for a login node. Keep the cache and
the `.sif` off `$HOME` (quota):

```bash
module load singularity
export SINGULARITY_CACHEDIR=/scratch/${USER:0:1}/$USER/singularity-cache
cd /work/<project>/$USER/containers
singularity pull feather_latest.sif docker://ghcr.io/clidyn/feather:latest
# a specific build: docker://ghcr.io/clidyn/feather:sha-<short>
```

If the package is private, log in first:
`singularity remote login -u <github-user> docker://ghcr.io` (token with
`read:packages`).

Without a registry, convert a locally built image instead:
`docker save feather:latest -o feather.tar` on the build machine, copy it over,
then `singularity build feather_latest.sif docker-archive://feather.tar`.

Run diagnostics on a **compute node** (SLURM job or interactive allocation),
never on the login node:

```bash
singularity exec \
  -B /work,/pool,/scratch,/home \
  --env VERTEX_API_KEY="$VERTEX_API_KEY",OPENAI_API_KEY="$OPENAI_API_KEY" \
  feather_latest.sif \
  /usr/local/bin/feather-entrypoint.sh \
  --config configs/eerie.yaml --diagnostics timeseries --variables tas -v
```

- `-B /work,/pool,/scratch,/home`: configs and some loaders use absolute
  Levante paths (`/work/bm1344`, `/work/ab0995`, `/work/bb1153`,
  `/work/ik1017`, `/pool/data`, the CMIP6 catalogue under `/home/a`).
  Singularity mounts your `$HOME` and the current directory by default; the
  rest must be bound.
- Run from your feather checkout (or pass absolute `--config`/`--output`)
  so relative config paths and `output_*` resolve on the host. The image itself
  is read-only.
- `exec … feather-entrypoint.sh` sets a writable matplotlib cache under
  `$TMPDIR`. `singularity run feather_latest.sif --help` works too.
- Smoke test: `singularity exec feather_latest.sif bash /opt/feather/docker/smoke_test.sh`.

Example SLURM job:

```bash
#!/bin/bash
#SBATCH --partition=compute --nodes=1 --time=08:00:00 --account=<project>
module load singularity
singularity exec -B /work,/pool,/scratch,/home feather_latest.sif \
  /usr/local/bin/feather-entrypoint.sh \
  --config configs/eerie_10_mems_cmip6.yaml --steps diagnostics -v
```

## API keys

Never baked into the image. Pass them at run time: `-e VERTEX_API_KEY`
(Docker) or `--env VERTEX_API_KEY=…` (Singularity; host variables are also
inherited by default). The `analyze` step needs `VERTEX_API_KEY`, `report`
needs `OPENAI_API_KEY`.

## Offline use

The cartopy shapes and AR6 polygons feather draws are in the image, so map
diagnostics need no network. A map that asks cartopy for another scale or
feature (e.g. 10m) would try to download it and fail on an offline node.
LLM steps (`analyze`, `report`) need network access to the APIs.
