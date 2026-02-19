# Feather — Development Guide

## What is this project?

Feather is a lightweight climate model evaluation framework for DestinE high-resolution (~5 km) simulations. It compares three climate models (IFS-FESOM, IFS-NEMO, ICON) against observations (ERA5, CERES, EN4, etc.) on native HEALPix grids and generates a web dashboard with LLM-analyzed diagnostic figures.

## Development environment

- **Machine:** DKRZ Levante HPC (`levante.dkrz.de`)
- **Conda env:** `nereus` — activate with `conda activate nereus`
- **Python:** 3.11+
- **Git root:** `/home/a/a270088/PYTHON/feather/feather/`
- **Install:** `pip install -e .` (editable install from `pyproject.toml`)

**IMPORTANT — Login node resource limits:**
We develop on a Levante login node. Do NOT run heavy computations (loading full-resolution model data, dask clusters, end-to-end diagnostic runs on real data). These require significant memory and CPU. If a task needs real data processing (e.g., end-to-end tests, running diagnostics on nside=1024 data, generating actual figures from catalogs), ask the user to run it — they will execute it in a dedicated compute environment (SLURM job or Jupyter on a compute node) with sufficient resources. Keep login-node work to: writing code, running unit tests with synthetic data, small integration tests that only open metadata/small slices.

## Running tests

```bash
conda activate nereus

# Unit tests (no real data needed)
pytest tests/ -v -m "not integration"

# Integration tests (requires Levante data access)
pytest tests/ -v -m "integration"

# All tests
pytest tests/ -v
```

Current test count: 95 unit tests + 2 integration tests.

**Note:** Unit tests use small synthetic data (nside=8, 768 cells) and are safe to run on the login node. Integration tests (`-m integration`) access real data files but only open metadata/small slices — they are also safe on the login node. For any end-to-end test that runs full diagnostics on real data (nside=1024, 12.6M cells), ask the user to execute it in a compute environment.

## Project structure

```
feather/                     # Package root
├── config.py                # FeatherConfig dataclass + YAML loading
├── data/
│   ├── loader.py            # DataLoader (intake catalogs + file paths)
│   ├── obs.py               # ObsLoader (observations from config)
│   └── variables.py         # VarInfo dataclass + VARIABLE_REGISTRY (27 vars)
├── util/
│   ├── spatial.py           # zonal_mean, global_mean, regional_mean, latlon_global_mean, regrid_to_latlon
│   ├── temporal.py          # climatology, anomaly, seasonal/monthly grouping
│   └── units.py             # Unit conversion functions
├── plot/
│   ├── maps.py              # plot_bias_map (3-panel), plot_single_map (nereus-based)
│   ├── lines.py             # plot_timeseries, plot_seasonal_cycle, plot_zonal_profile
│   └── styles.py            # MODEL_COLORS, OBS_COLOR, apply_style
├── diag/
│   ├── base.py              # DiagnosticBase ABC (compute → plot → run)
│   ├── figure_meta.py       # save_figure_with_metadata, build_metadata
│   ├── registry.py          # @register decorator, get/list diagnostics
│   ├── global_biases.py     # GlobalBiases: climatology bias maps
│   ├── timeseries.py        # TimeseriesDiag: global-mean time series
│   └── seasonal_cycle.py    # SeasonalCycleDiag: monthly climatological cycle
└── export/                  # Placeholder for future notebook export
```

## Key files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Default configuration with all Levante data paths |
| `feather/data/variables.py` | Central variable registry — add new variables here |
| `feather/diag/base.py` | Base class for all diagnostics — subclass this |
| `feather/diag/registry.py` | `@register` decorator for diagnostic auto-discovery |
| `tests/conftest.py` | Synthetic HEALPix/obs fixtures, mock loaders, minimal_config |
| `PLAN.md` | Full implementation plan with phase status |

## Configuration

`FeatherConfig` is loaded from YAML via `FeatherConfig.from_yaml("configs/default.yaml")`.

Key fields:
- `model_catalogs` — paths to intake catalog YAML files (2D, 3D)
- `models` — list of model names: `["ifs-fesom", "ifs-nemo", "icon"]`
- `obs_root` — root path for observation data
- `obs_datasets` — nested dict mapping dataset → variables → filenames
- `cmip6` — CMIP6 comparison config (disabled by default)
- `output_dir` — where figures/analysis/site are written

The `{obs_root}` placeholder in obs dataset paths is resolved at load time.

## Data paths on Levante

### Model data (intake catalogs)
- 2D: `/work/ab0995/a270088/DestinE/GENERATION2_joint/2D/catalog.yaml`
- 3D: `/work/ab0995/a270088/DestinE/GENERATION2_joint/3D/catalog.yaml`
- Format: zarr, HEALPix grid, dim `values` (1D), coords `latitude`/`longitude`/`time`
- Catalog key: `{experiment}_2_{model}_{member}_0001_clmn_{high|standard}_{domain}`
  - 2D domains (sfc, o2d) use `clmn_high` (nside=1024, 12.6M cells)
  - 3D domains (pl, o3d) use `clmn_standard` (nside=128, 196K cells)

### Observations
- Root: `/work/bb1153/b382289/data/aqua-dvc/datasets/`
- ERA5: `ERA5/mon/ERA5_*.nc` (30 vars, 1940-2024)
- CERES: `CERES/EBAF_v4.2.1/` (TOA/surface radiation, 2000-2025)
- EN4: `EN4/v4.2.2/` (ocean T/S profiles, 1950-2024)

### CMIP6
- Catalog: `/home/a/a270088/PYTHON/DestinE/cmip6/cmip6_zarr_catalog.yaml` (593 entries)
- Data: `/work/ab0995/a270088/DestinE/cmip6/zarr/`
- Variable mapping: `feather/data/variables.py` has `cmip6_variable` and `cmip6_table` fields

## How to add a new diagnostic

1. Create `feather/diag/my_diagnostic.py`
2. Subclass `DiagnosticBase` and implement `compute()` and `plot()`
3. Decorate with `@register`

```python
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register

@register
class MyDiagnostic(DiagnosticBase):
    name = "my_diagnostic"           # unique ID, used in filenames
    title = "My Diagnostic"          # human-readable
    domain = "sfc"                   # sfc, o2d, pl, o3d
    variables = ["avg_2t"]           # model variables used
    group = "temperature"            # dashboard nav group

    def compute(self):
        # Load data, compute results
        return {"model_clim": ..., "obs_clim": ...}

    def plot(self, results):
        # Create figures, return (fig, metadata) pairs
        fig, ax = plt.subplots()
        # ... plot ...
        meta = self._build_metadata(
            title="My Plot",
            figure_id="my_diagnostic_fig1",
            models=self.config.models,
            description="...",
            plot_type="timeseries",
        )
        return [(fig, meta)]
```

4. Import the module somewhere so `@register` fires (e.g., in `feather/diag/__init__.py`)
5. `_build_metadata()` auto-fills units, domain, group, obs info from `VARIABLE_REGISTRY`

## How to add a new variable

Add an entry to `VARIABLE_REGISTRY` in `feather/data/variables.py`:

```python
"avg_newvar": VarInfo(
    name="avg_newvar",
    long_name="My New Variable",
    units="K",
    domain="sfc",
    cmap="RdBu_r",
    obs_dataset="ERA5",
    obs_variable="newvar",
    cmip6_variable="newvar_cmip6",
    cmip6_table="Amon",
    group="temperature",
),
```

## Important patterns and gotchas

### HEALPix data
- Model data is 1D on `values` dimension, not 2D lat/lon
- Use `nr.plot(data, lon, lat)` for map plotting — nereus handles NN interpolation to a regular grid
- Zonal means: use `feather.util.spatial.zonal_mean()` with `np.digitize` lat-band binning (no regridding)
- For synthetic test data, use nside=8 (768 cells) with ≥10° lat bins — 1° bins leave polar bins empty at low nside

### ERA5 variable naming
- File keys in config (e.g., `t2m`) differ from NetCDF internal names (e.g., `T2M`)
- `ObsLoader._find_variable()` handles this with case-insensitive fallback

### Metadata sidecar pattern
- Every saved figure gets a companion `.json` with full metadata
- `build_metadata()` auto-fills from `VARIABLE_REGISTRY`
- `save_figure_with_metadata()` appends `generated_at` timestamp without mutating input dict
- The metadata JSON is the contract between diagnostics → LLM analyzer → dashboard

### Test fixtures
- `synth_healpix` in `conftest.py`: nside=8, 768 cells, 12 timesteps, temperature gradient pole→equator
- `synth_obs` in `conftest.py`: 5° regular lat/lon grid, matching temperature field
- Registry tests use `autouse` fixture to save/restore `_REGISTRY` global state

## Dependencies

Core: xarray, dask, distributed, numpy, scipy, matplotlib, cartopy, intake, intake-xarray, healpy, nereus, pyyaml, netcdf4, zarr, cmocean, nbformat

Dev: pytest, pytest-cov

## Reference implementation

The existing climate diagnostics pipeline (more complex, climate change focus) is at:
`/home/a/a270088/PYTHON/DestinE/phase2/climate_change_analysis/climate_diagnostics/`

Useful modules to reference:
- `climate_diag/llm_analysis/` — LLM analysis with Gemini (schemas, prompts, analyzer)
- `climate_diag/website/` — static HTML dashboard (Jinja2 templates, dark-theme CSS)
- `climate_diag/plot_helpers.py` — multi-model map panels, envelope plots
- `climate_diag/cmip6_loader.py` — CMIP6 data loading with calendar normalization
- `climate_diag/cmip6_variables.py` — 25-variable CMIP6↔DestinE mapping
