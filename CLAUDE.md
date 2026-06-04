# Feather — Development Guide

## What is this project?

Feather is a lightweight climate model evaluation framework supporting multiple model sets. It compares climate models against observations (ERA5, CERES, EN4, etc.) and generates a web dashboard with LLM-analyzed diagnostic figures.

Currently supported model sets:
- **DestinE**: IFS-FESOM, IFS-NEMO, ICON (~5 km, HEALPix grids, intake catalogs)
- **EERIE HighResMIP**: IFS-FESOM2-SR, IFS-NEMO-ER, ICON-ESM-ER, HadGEM3-GC5 (~10 km atm, ~5-10 km ocean, 0.25° lat/lon output, CMOR directory tree)
- **Custom NetCDF/HEALPix**: Per-year NetCDF files on HEALPix grid (e.g. IFS-FESOM T319)
- **GRIB**: IFS-FESOM TCO399/TCO319 (~25-35 km, regular lat/lon, GRIB files)
- **TerraDT**: IFS-FESOM, IFS-NEMO, ICON (~5 km, HEALPix, per-model ensemble members)
- **Combined multi-source**: Mix models from different backends (e.g. DestinE catalog + GRIB) via CompositeModelLoader

The framework is grid-agnostic: diagnostics automatically dispatch between HEALPix and regular lat/lon grids based on per-model config.

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

Current test count: ~1372 tests (1367 unit + 5 integration).

**Note:** Unit tests use small synthetic data (nside=8, 768 cells) and are safe to run on the login node. Integration tests (`-m integration`) access real data files but only open metadata/small slices — they are also safe on the login node. For any end-to-end test that runs full diagnostics on real data (nside=1024, 12.6M cells), ask the user to execute it in a compute environment.

## Project structure

```
feather/                     # Package root
├── cli.py                   # CLI entry point (feather command)
├── run.py                   # Pipeline orchestration (run_pipeline)
├── __main__.py              # python -m feather support
├── config.py                # FeatherConfig dataclass + YAML loading
├── data/
│   ├── loader.py            # MultiCatalogLoader (DestinE intake catalogs)
│   ├── cmor_loader.py       # CMORLoader (CMOR directory tree, e.g. EERIE)
│   ├── netcdf_loader.py     # NetCDFLoader (per-year NetCDF on HEALPix grid)
│   ├── grib_loader.py       # GRIBLoader (GRIB files on regular lat/lon grid)
│   ├── composite_loader.py  # CompositeModelLoader (multi-source per-model routing)
│   ├── obs.py               # ObsLoader (observations from config) + load_ceres()
│   ├── cmip6.py             # CMIP6Loader (multi-model mean from zarr)
│   └── variables.py         # VarInfo dataclass + VARIABLE_REGISTRY (33 vars, CMOR canonical names)
├── util/
│   ├── spatial.py           # zonal_mean, global_mean, regional_mean, latlon_global_mean, compute_latlon_areas, spatial_ttest, spatial_variance_ratio
│   ├── temporal.py          # climatology, anomaly, seasonal/monthly grouping
│   ├── eof.py               # EOF computation via SVD (teleconnections)
│   ├── spectrum.py           # Power spectrum via Welch (teleconnections)
│   └── units.py             # Unit conversion functions
├── plot/
│   ├── maps.py              # plot_combined_bias_map (multi-panel), plot_bias_map (3-panel), plot_single_map
│   ├── lines.py             # plot_timeseries, plot_seasonal_cycle, plot_zonal_profile, plot_budget_bars, plot_gregory
│   └── styles.py            # MODEL_COLORS, OBS_COLOR, apply_style
├── diag/
│   ├── base.py              # DiagnosticBase ABC (compute → plot → run)
│   ├── figure_meta.py       # save_figure_with_metadata, build_metadata
│   ├── registry.py          # @register decorator, get/list diagnostics
│   ├── global_biases.py     # GlobalBiases: climatology bias maps
│   ├── timeseries.py        # TimeseriesDiag: global-mean time series
│   ├── seasonal_cycle.py    # SeasonalCycleDiag: monthly climatological cycle
│   ├── radiation_budget.py  # RadiationBudget: TOA/surface radiation vs CERES
│   ├── sea_ice.py           # SeaIceDiag: sea ice area/extent/volume/spatial
│   ├── ocean_sst.py         # OceanSST: SST evaluation vs ESA-CCI
│   ├── ocean_en4.py         # OceanEN4: 3D ocean T/S evaluation vs EN4
│   ├── global_trends.py     # GlobalTrends: per-grid-point linear trends
│   ├── climate_variability.py # ClimateVariability: STD of deseasonalised, detrended fields
│   ├── precipitation_mswep.py # PrecipitationMSWEP: precip eval vs MSWEP v2.8
│   ├── temperature_berkeley.py # TemperatureBerkeley: T2m eval vs Berkeley Earth
│   ├── teleconnections.py    # TeleconnectionDiag: variability modes (ENSO, NAO, etc.)
│   ├── obs_comparison.py     # ObsComparisonDiag: ERA5 vs Berkeley Earth trends and biases
│   └── precip_obs_comparison.py # PrecipObsComparisonDiag: ERA5 vs MSWEP precip trends and biases
├── llm/
│   ├── schemas.py           # FigureAnalysis, DiagnosticSynthesis (Pydantic)
│   ├── prompts.py           # System + user prompts for Gemini analysis
│   └── analyzer.py          # FigureAnalyzer: discover, analyze, synthesize
├── website/
│   ├── generator.py         # SiteGenerator: collect, group, build static site
│   ├── templates/           # Jinja2 templates (base, index, diagnostic)
│   └── static/style.css     # Dark theme CSS with group badges
└── export/
    ├── report.py            # ReportGenerator (OpenAI, 3-stage)
    ├── schemas.py           # ReportStructure, WrittenSection, SelectedFigure
    ├── prompts.py           # OpenAI prompts (curation + writing)
    ├── openai_client.py     # Thin OpenAI wrapper with retry
    ├── latex_builder.py     # LaTeX escaping, template rendering, PDF
    └── templates/           # Jinja2 LaTeX template
```

## Key files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | DestinE configuration (legacy list format) |
| `configs/eerie.yaml` | EERIE HighResMIP configuration (structured dict format) |
| `configs/eerie_psl.yaml` | EERIE HighResMIP + psl for HadGEM3 (symlinked from HadGEM3-GC5E-HH/historical) |
| `configs/terradt.yaml` | TerraDT baseline evaluation (per-model members) |
| `configs/ifs_fesom_combined.yaml` | IFS-FESOM multi-resolution (mixed data sources) |
| `configs/obs_only.yaml` | Observation-only intercomparison (ERA5/Berkeley/MSWEP/CHIRPS/CRU; no models) |
| `feather/cli.py` | CLI entry point — `feather` command (argparse) |
| `feather/run.py` | Pipeline orchestration — `run_pipeline()` |
| `feather/config.py` | FeatherConfig + ModelConfig dataclasses, dual-format YAML loading |
| `feather/data/variables.py` | Central variable registry (CMOR canonical names) |
| `feather/data/cmor_loader.py` | CMORLoader — load from CMOR directory tree (EERIE etc.) |
| `feather/data/netcdf_loader.py` | NetCDFLoader — load per-year NetCDF on HEALPix grid |
| `feather/data/grib_loader.py` | GRIBLoader — load GRIB files on regular lat/lon grid |
| `feather/data/composite_loader.py` | CompositeModelLoader — multi-source per-model routing |
| `feather/diag/base.py` | Base class — grid-agnostic helpers (`_load_model_var`, `_model_global_mean`) |
| `feather/diag/registry.py` | `@register` decorator for diagnostic auto-discovery |
| `feather/data/cmip6.py` | CMIP6Loader — load zarr, compute multi-model mean |
| `feather/llm/analyzer.py` | FigureAnalyzer — Gemini-based, comparison-type aware |
| `feather/llm/schemas.py` | Pydantic models for structured LLM output |
| `feather/export/report.py` | ReportGenerator — LaTeX report via OpenAI (3-stage) |
| `feather/website/generator.py` | SiteGenerator — static HTML dashboard from figures + analysis |
| `tests/conftest.py` | Synthetic HEALPix/obs/CMIP6 fixtures, mock loaders, minimal_config |

## Configuration

`FeatherConfig` is loaded from YAML via `FeatherConfig.from_yaml("configs/eerie.yaml")`.

### Dual-format config loading

The `models` key determines the config format:
- **List** → legacy DestinE format: `models: [ifs-fesom, ifs-nemo, icon]`. Auto-populates `model_configs` with HEALPix grid defaults.
- **Dict** → structured format: `models: {ModelA: {institution: ..., grids: ...}}`. Builds `ModelConfig` instances.

### Structured config format (EERIE-style)

```yaml
project:
  name: "EERIE"
  description: "EERIE HighResMIP evaluation"
  experiment: "hist-1950"
  period: ["1980", "2014"]
  resolution: "high-resolution (~10 km atm, ~5-10 km ocean)"  # used in LLM prompts
  comparison_type: "multi_model"      # multi_model | resolution_sensitivity | single_model | baseline_evaluation
  comparison_description: ""          # optional free-text for LLM prompt framing

data_source:
  type: "cmor"                        # "cmor" | "destine_catalog" | "netcdf_healpix" | "grib"
  root: "/path/to/CMOR/tree"

models:
  IFS-FESOM2-SR:
    institution: AWI
    experiment: hist-1950
    variant: r1i1p1f1
    grids:
      sfc: latlon                     # "healpix" or "latlon" per domain
      o2d: latlon
      o3d: latlon
    color: "#1f77b4"                  # hex color for plots
  IFS-NEMO-ER:
    institution: BSC
    # ...
```

### FeatherConfig helpers

| Method | Returns | Default |
|--------|---------|---------|
| `get_grid_type(model, domain)` | `"healpix"` or `"latlon"` | `"healpix"` |
| `get_model_color(model)` | hex color string | palette cycle |
| `get_period()` | `(start, end)` tuple | `("1990", "2014")` |
| `get_experiment()` | experiment string | `"baseline_hist"` |
| `get_data_source_type()` | `"cmor"`, `"netcdf_healpix"`, `"grib"`, or `"destine_catalog"` | `"destine_catalog"` |
| `get_model_data_source_type(model)` | per-model backend (falls back to global) | global type |
| `is_multi_source()` | `True` when models use different backends | `False` |
| `get_comparison_type()` | LLM prompt framing type | `"multi_model"` |
| `get_comparison_description()` | free-text comparison context | `""` |

### ModelConfig dataclass

```python
@dataclass
class ModelConfig:
    name: str           # Display name
    institution: str    # Modelling centre
    experiment: str     # e.g. "hist-1950"
    variant: str        # e.g. "r1i1p1f1"
    grids: dict         # domain → "healpix"|"latlon"
    color: str          # Hex color for plots
    data_root: str      # Per-model data path override (HadGEM3)
    grid_label: str     # CMOR grid label override (default "gr")
    variable_aliases: dict  # CMOR var name → on-disk name (e.g. thetao→thetao-con)
    scale_factors: dict     # var → multiplier (e.g. clt: 100 for fraction→%)
    absolute_salinity: bool # True if model outputs SA (TEOS-10) instead of SP (EOS-80)
    data_source_type: str   # Per-model backend override (e.g. "grib" for mixed-source)
    catalog_key: str        # Catalog entry name when different from model name
    member: int             # Ensemble member index (default 1)
```

### Other config fields
- `model_catalogs` — paths to intake catalog YAML files (DestinE only)
- `obs_root` — root path for observation data
- `obs_datasets` — nested dict mapping dataset → variables → filenames
- `cmip6` — CMIP6 comparison config (enabled by default)
- `output_dir` — where figures/analysis/site are written
- `llm` — LLM provider config (per-purpose: `figure_analysis` uses Gemini)
- `report` — report generation config (model, max_tokens, temperature, api_key_env, n_highlights)
- `website` — site title, subtitle, group ordering
- `nereus` — nereus interpolation settings: `projection`, `resolution`, `influence_radius`, `ocean_influence_radius`, `method` (interpolation method: `"nearest"`, `"idw"`, `"linear"`, `"cubic"` — used for CMIP6 regridding only)

The `{obs_root}` placeholder in obs dataset paths is resolved at load time.

### Variable naming: CMOR canonical

Variables use CMOR names as canonical IDs (e.g., `"tas"` not `"avg_2t"`). The `VARIABLE_REGISTRY` in `feather/data/variables.py` maps each variable:
- `name` — CMOR canonical name (registry key)
- `destine_variable` — DestinE name for backward compat (e.g., `"avg_2t"`)
- `cmip6_variable` / `cmip6_table` — CMIP6 lookup
- `obs_variable` — observation file key
- `cmor_obs_sign` — sign flip for CMOR vs ERA5 conventions (default 1.0, set to -1.0 for `hfss`/`hfls`)

`get_var()` accepts both CMOR and DestinE names via `_DESTINE_TO_CANONICAL` fallback.

## Data paths on Levante

### DestinE model data (intake catalogs)
- 2D: `/work/ab0995/a270088/DestinE/GENERATION2_joint/2D/catalog.yaml`
- 3D: `/work/ab0995/a270088/DestinE/GENERATION2_joint/3D/catalog.yaml`
- Format: zarr, HEALPix grid, dim `values` (1D), coords `latitude`/`longitude`/`time`
- Catalog key: `{experiment}_2_{model}_{member}_0001_clmn_{high|standard}_{domain}`
  - 2D domains (sfc, o2d) use `clmn_high` (nside=1024, 12.6M cells)
  - 3D domains (pl, o3d) use `clmn_standard` (nside=128, 196K cells)

### EERIE model data (CMOR directory tree)
- Root: `/work/bm1344/DKRZ/CMOR/EERIE/HighResMIP/`
- Path pattern: `{root}/{institution}/{model}/hist-1950/r1i1p1f1/{table}/{variable}/gr/v*/`
- Format: NetCDF, 0.25° regular lat/lon (721×1440), standard dims `(time, lat, lon)`
- Tables: Amon (atmosphere), Omon (ocean), SImon (sea ice)
- 3 models: IFS-FESOM2-SR (AWI), IFS-NEMO-ER (BSC), ICON-ESM-ER (MPI-M)
- Period: 1950–2014 (config uses 1980–2014)
- CMORLoader in `feather/data/cmor_loader.py` handles path construction and file discovery

### Custom NetCDF/HEALPix model data (NetCDFLoader)
- Data source type: `"netcdf_healpix"` in config
- Root: configurable per-project (e.g. `/work/ab0995/a270135/MN5/projt319/netcdf`)
- Path pattern: `{root}/{dir_name}/{dir_name}_{year}.nc` (per-year files)
- Grid: HEALPix 1D, dim named `gsize` (renamed to `values` on load)
- Coordinates: derived from `healpy.pix2ang(nside, ...)` and attached as `longitude`/`latitude`
- nside auto-detected from `gsize` dim size, or overridden via `data_source.nside` config
- Singleton dims (`height`, `depth`) squeezed automatically
- Variable mapping: `variable_aliases` in ModelConfig maps CMOR names → on-disk dir names
- Fallback: `destine_variable` from `VARIABLE_REGISTRY`, then canonical name
- Scale factors and caching work identically to CMORLoader
- IFS conventions: positive-downward fluxes (same as ERA5/DestinE, NOT CMOR)
- NetCDFLoader in `feather/data/netcdf_loader.py`
- Config: `configs/himansu_319.yaml` (IFS-FESOM T319 example)

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
4. Use grid-agnostic base class helpers (not hardcoded loader calls)

```python
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register

@register
class MyDiagnostic(DiagnosticBase):
    name = "my_diagnostic"           # unique ID, used in filenames
    title = "My Diagnostic"          # human-readable
    domain = "sfc"                   # sfc, o2d, pl, o3d
    variables = ["tas"]              # CMOR canonical variable names
    group = "temperature"            # dashboard nav group

    def compute(self):
        for model in self.config.models:
            mdata = self._load_model_var(model, "tas")
            ts = self._model_global_mean(mdata.data, model)
            # ...
        obs = self._load_obs_var("tas")
        return {"model_ts": ..., "obs_ts": ...}

    def plot(self, results):
        fig, ax = plt.subplots()
        meta = self._build_metadata(
            title="My Plot",
            figure_id="my_diagnostic_fig1",
            models=self.config.models,
            description="...",
            plot_type="timeseries",
        )
        return [(fig, meta)]
```

5. Import the module somewhere so `@register` fires (e.g., in `feather/diag/__init__.py`)
6. `_build_metadata()` auto-fills units, domain, group, obs info from `VARIABLE_REGISTRY`

## How to add a new variable

Add an entry to `VARIABLE_REGISTRY` in `feather/data/variables.py` (CMOR name as key):

```python
"tas": VarInfo(
    name="tas",                      # CMOR canonical name
    long_name="2m Temperature",
    units="K",
    domain="sfc",
    cmap="RdBu_r",
    obs_dataset="ERA5",
    obs_variable="t2m",
    destine_variable="avg_2t",       # DestinE catalog name (empty if N/A)
    cmip6_variable="tas",
    cmip6_table="Amon",
    group="temperature",
),
```

## How to add a new model set

1. Create a config YAML using the structured format (see `configs/eerie.yaml` as template)
2. Set `data_source.type` to `"cmor"`, `"destine_catalog"`, `"netcdf_healpix"`, or `"grib"`
3. Define `models` as a dict with per-model `institution`, `experiment`, `variant`, `grids`, `color`
4. Set `project.name`, `project.period`, `project.experiment`
5. Optionally set `project.comparison_type` for LLM prompt framing
6. For mixed data sources: set per-model `data_source_type` to route via `CompositeModelLoader`
7. For `"netcdf_healpix"`: add `variable_aliases` mapping CMOR names → on-disk directory names (see `configs/himansu_319.yaml`)
8. Run: `feather --config configs/my_project.yaml --diagnostics timeseries --variables tas -v`

If your data format is not supported, create a new loader class (see `GRIBLoader`, `NetCDFLoader`, or `CMORLoader` as templates) and add a dispatch case in `feather/run.py:_create_model_loader()`.

## Important patterns and gotchas

### HEALPix data
- Model data is 1D on `values` dimension, not 2D lat/lon
- Use `nr.plot(data, lon, lat)` for map plotting — nereus handles NN interpolation to a regular grid
- Zonal means: use `feather.util.spatial.zonal_mean()` with `np.digitize` lat-band binning (no regridding)
- For synthetic test data, use nside=8 (768 cells) with ≥10° lat bins — 1° bins leave polar bins empty at low nside

### ERA5 variable naming and units
- File keys in config (e.g., `t2m`) differ from NetCDF internal names (e.g., `T2M`)
- `ObsLoader._find_variable()` handles this with case-insensitive fallback
- ERA5 radiation/flux variables are stored as **daily accumulations** (J/m²/day), not W/m². Divide by 86400 to get W/m².
- ERA5 precipitation (`TP`) is in m/day. Multiply by `1000/86400` to get kg/m²/s.
- ERA5 cloud cover (`TCC`) is 0-1 fraction; DestinE `avg_tcc` is 0-100%. Multiply by 100.
- Unit conversions are defined via `obs_unit_factor` in `VARIABLE_REGISTRY` (`feather/data/variables.py`)

### Sign conventions (ERA5 vs CMOR)
- ERA5 & DestinE (IFS): surface heat fluxes **positive downward** (into surface)
- CMOR/CMIP6/EERIE: surface heat fluxes **positive upward** (away from surface)
- `VarInfo.cmor_obs_sign` field: default 1.0, set to -1.0 for `hfss` and `hfls`
- `DiagnosticBase._load_obs_var()` applies sign flip when `data_source == "cmor"`
- Diagnostics use `_load_obs_var()` instead of direct `obs_loader.load_for_model_var()`

### Grid-agnostic base class helpers
- `_load_model_var(model, var)` → `ModelData` — dispatches to correct loader (intake vs CMOR)
- `_load_model_coords(model)` → `(lon, lat)` — coordinate arrays for any grid type
- `_model_global_mean(da, model)` → area-weighted mean (HEALPix: simple `.mean()`, latlon: cos-weighted)
- `_load_obs_var(var)` → obs DataArray with sign convention handling
- Error handling: `except (KeyError, FileNotFoundError)` for all model loads (CMOR raises FileNotFoundError, DestinE raises KeyError)

### Metadata sidecar pattern
- Every saved figure gets a companion `.json` with full metadata
- `build_metadata()` auto-fills from `VARIABLE_REGISTRY`
- `save_figure_with_metadata()` appends `generated_at` timestamp without mutating input dict
- The metadata JSON is the contract between diagnostics → LLM analyzer → dashboard

### CMIP6 data
- `CMIP6Loader` loads per-variable zarr files directly (no intake at load time) — avoids staggered-grid conflicts
- Two ensemble modes: `"one_per_model"` (first variant per model) or `"all_members"` (all variants)
- Config supports both `variants: [list]` (new) and `variant: str` (legacy) format
- `load_var()` returns `None` for missing data — diagnostics should handle gracefully
- Calendar normalization (360_day, noleap, standard) → first-of-month pandas timestamps
- Sea ice (`siconc`): auto-normalized from percentage (0-100) to fraction (0-1) if needed
- All 7 diagnostics (timeseries, seasonal_cycle, global_biases, radiation_budget, sea_ice, ocean_sst, ocean_en4) integrated — CMIP6 MMM lines/bias maps added when `cmip6.enabled: true`
- `get_member_pairs()` public API for listing (model, variant) tuples

### GlobalBiases diagnostic
- Produces **combined multi-panel figures**: 1 obs panel + N bias panels per variable per period
- `plot_combined_bias_map()` in `plot/maps.py` handles layout with `max_cols` overflow to extra rows
- 18 validated DestinE↔ERA5 surface variables (temperature, pressure, wind, clouds, precipitation, radiation, heat fluxes)
- Gracefully skips models missing a variable (e.g., ICON lacks `avg_tcc`) with a warning
- `--cmip6-individual` CLI flag enables individual CMIP6 model bias panels **plus** MMM (both computed together)
- Without the flag, only CMIP6 MMM is shown (default behavior)
- `--variables` CLI flag intersects with diagnostic's supported list; warns about unsupported variables
- **CMIP6 interpolation method**: configurable via `nereus.method` (default `"nearest"`, recommended `"linear"` for smoother CMIP6 maps). Only applies to CMIP6 regridding; model/obs stay nearest neighbor.
- **CMIP6 interpolation method**: configurable via `nereus.method` (default `"nearest"`, recommended `"linear"` for smoother CMIP6 maps). Only applies to CMIP6 regridding; model/obs stay nearest neighbor.
- CMIP6 source lons converted to -180..180 before interpolation to avoid NaN stripe at prime meridian (Delaunay triangulation gap)
- MMM computed as "regrid each model individually then average" — when `cmip6_individual=True`, MMM reuses already-regridded individual fields (`_mmm_from_individual()`) to avoid double interpolation
- **Statistical significance tests**: per-model paired t-test (area-weighted, Kish's effective sample size) tests whether mean spatial bias differs significantly from zero (`t_test_statistic`, `t_test_p_value` in JSON metadata). Variance ratio F-test (population variance, ddof=0, area-weighted) computes F = Var(model) / Var(obs) — F=1 means equal spatial variability, F>1 model more variable, F<1 model smoother (`variance_ratio`, `variance_ratio_p_value`). Both computed for evaluated models, CMIP6 MMM, and individual CMIP6 models. Utility functions: `spatial_ttest()` and `spatial_variance_ratio()` in `feather/util/spatial.py`. Website uses `_format_stat` Jinja2 filter for scientific notation on small p-values.

### Timeseries & SeasonalCycle diagnostics
- Both share the same 18-variable list as GlobalBiases (temperature, pressure, wind, clouds, precipitation, radiation, heat fluxes)
- Both support `cmip6_individual=True` for individual CMIP6 model lines + MMM
- 4-layer plotting: individual CMIP6 (background, semi-transparent) → MMM (dashed) → evaluated models (foreground) → Obs (top)
- Gracefully skip models missing a variable via `try/except (KeyError, FileNotFoundError)`
- Per-variable `try/except Exception` in `run()` — one missing variable doesn't kill the diagnostic
- Group is `"evaluation"` (spans multiple physical domains)
- CMIP6 individual series come from `_cmip6_global_mean_timeseries(return_individual=True)` on `DiagnosticBase`

### RadiationBudget diagnostic
- 4th diagnostic: computes derived radiation quantities and produces budget bars (two-panel: full budget + TOA Net zoom), Gregory plot, imbalance time series, and bias maps
- Uses CERES EBAF as primary obs (not ERA5) — satellite gold standard for radiation, loaded via `ObsLoader.load_ceres()`
- CERES files contain many variables per file — bypasses `VARIABLE_REGISTRY`, uses `config.obs_datasets["CERES_EBAF"]` with `toa`/`surface` file keys
- Sign conventions: DestinE uses positive-downward (net into system); CERES OLR (`toa_lw_all_mon`) is positive upward — corrected via `ceres_sign=-1.0` in `_BUDGET_COMPONENTS`
- CMIP6 net TOA is derived: `rsdt - rsut - rlut` (3 vars combined, no single CMIP6 variable)
- Gregory plot uses two obs datasets: CERES for TOA radiation + ERA5 for T2m
- Per-figure-group incremental saving (not per-variable like other diagnostics)
- Imbalance time series: monthly as semi-transparent background, annual means as thick foreground; x-axis set to model range, each series starts from its own data start
- Error handling: catches `AttributeError` alongside `KeyError`/`FileNotFoundError` for obs loaders lacking `load_ceres()`
- `cmip6_individual` support: individual CMIP6 scatter on Gregory plot (gray dots) + MMM regression (dashed)
- 50 dedicated tests in `tests/test_radiation_budget.py`

### SeaIceDiag diagnostic
- 5th diagnostic: sea ice area, extent, volume evaluation against OSI-SAF and PIOMAS/GIOMAS
- 13 figures across 4 groups: time series (A), seasonal cycles (B), March/September trends (C), spatial maps (D)
- Uses nereus ice functions directly: `nr.ice_area_nh/sh()`, `nr.ice_extent_nh/sh()`, `nr.ice_volume_nh/sh()`
- Spatial maps use `nr.plot(ax=ax, projection="np"/"sp")` for polar stereographic
- Obs: OSI-SAF (EASE2 grid, concentration), PIOMAS/GIOMAS (curvilinear, thickness/volume)
- CMIP6 integration: custom `_compute_cmip6_timeseries()` loads `siconc`/`sithick` from SImon table, computes hemisphere-specific metrics (not global-mean like base class helper)
- CMIP6 data sanitization: fill values (1e20, 9.97e36) in siconc/sithick/areacello must be ZEROED not clipped — clipping converts land fill values to fake 100% ice. Uses `np.where(vals > threshold, 0.0, vals)` pattern.
- Auto-detects siconc still in percentage (0-100) and re-normalizes to fraction (0-1)
- `_CMIP6_SEA_ICE_EXCLUDE` blocklist: models with known unrealistic sea ice (currently FGOALS-g3) are skipped with WARNING-level log
- Post-computation validation: models with any metric > 1e14 are excluded from MMM
- Per-model diagnostic logging: prints mean/max of each metric in plot units for outlier detection
- CMIP6 only on Groups A-C (time series/seasonal/trends); Group D (spatial maps) excluded — CMIP6 ~100km resolution too coarse for polar stereo alongside 5km DestinE
- `cmip6_individual` support: individual CMIP6 model lines + MMM with 4-layer plotting convention
- 106 dedicated tests in `tests/test_sea_ice.py`

### OceanSST diagnostic
- 6th diagnostic: SST evaluation against ESA-CCI L4 v3.0.1 satellite observations
- 6 figures in 4 groups (A: 3 bias maps, B: timeseries, C: seasonal cycle, D: zonal mean)
- K→°C conversion: CMOR model `tos` is already in °C (skip conversion), DestinE `tos` is in K (convert). ESA-CCI obs always in K (always convert).
- Obs global mean uses cos(lat) weights (avoids 25.9M-cell mesh for 0.05° grid)
- `ocean_influence_radius` (20km default) avoids coast contamination for ESA-CCI's 0.05° grid
- `land=True` passed to `plot_combined_bias_map()` for ocean-only display
- 104 dedicated tests in `tests/test_ocean_sst.py`

### OceanEN4 diagnostic
- 7th diagnostic: 3D ocean T/S evaluation against EN4 v4.2.2 (temperature + salinity)
- 12 figures in 8 groups: A-B surface bias maps (3+3), C-F Hovmoller diagrams (1 combined each), G-H depth-layer time series (1 each)
- Hovmoller figures are combined: EN4 + all models as subpanels with shared symmetric colorbar
- Two anomaly types: anom1 (each minus own first timestep) and anomref (all minus EN4 first timestep)
- Uses `nr.hovmoller()` with integer time indices (datetime64 buffer workaround), `nr.mesh_from_arrays()` for EN4 areas, `nr.volume_mean()` for depth-layer averaging
- Depth levels: auto-detected from `lev`/`depth`/`deptht` coordinate when not in config. Config levels take priority (needed for DestinE integer-indexed depth dim). NEMO-based models (HadGEM3) use `deptht` as depth dimension name instead of `lev`.
- `nr.hovmoller()` and `nr.volume_mean()` are dask-friendly — pass DataArrays directly, never `.values`
- For latlon 4D data: `.stack(space=(lat_dim, lon_dim))` before `nr.volume_mean()` (needs 3D input)
- K→°C conversion: `_get_convert(var)` for model (skips for CMOR), `_get_convert(var, for_obs=True)` for EN4 (always converts). EN4 thetao on Levante is stored in Kelvin, CMOR model thetao is in °C.
- **SA→SP salinity conversion**: NEMO-based models (IFS-NEMO-ER, HadGEM3-GC5) output absolute salinity (TEOS-10, g/kg) instead of practical salinity (EOS-80, PSU). `_apply_sa_to_sp()` converts using `gsw.SP_from_SA(SA, p, lon, lat)` for models with `absolute_salinity: true` in config. Uses `xr.apply_ufunc` with `dask='parallelized'` — stays lazy.
- EN4 observations use practical salinity — no conversion needed for obs
- EN4 influence radius: 200km default (`en4_influence_radius`) for 1° grid; model uses standard 80km
- 118 dedicated tests in `tests/test_ocean_en4.py`

### PrecipitationMSWEP diagnostic
- 11th diagnostic: dedicated precipitation evaluation against MSWEP v2.8
- 8 figures in 6 groups: A (3 absolute bias maps), B (1 relative bias %), C (1 timeseries), D (1 seasonal cycle), E (1 zonal mean), F (1 intensity PDF)
- MSWEP v2.8: merged gauge+satellite+reanalysis precipitation (0.1°, zarr, mm/month)
- `ObsLoader.load_mswep()`: converts mm/month → kg/m²/s (time-varying days_in_month), shifts lons from -180..180 → 0..360
- Relative bias: masked where obs < 0.1 mm/day threshold to avoid division artifacts in deserts
- Intensity PDF: area-weighted histogram with log-scale bins, shows drizzle bias and heavy precip representation
- Zonal mean: shows ITCZ position, subtropical dry zones, extratropical storm tracks
- Enhanced statistics: pattern correlation, STD ratio, RMSE, tropical (30S-30N) and extratropical mean bias
- CMIP6 support: MMM + individual model biases via `_regrid_to_target()`, `_mmm_from_individual()`
- Per-group incremental saving (groups A-F independently saveable)
- 102 dedicated tests in `tests/test_precipitation_mswep.py`

### TemperatureBerkeley diagnostic
- 12th diagnostic: 2m temperature evaluation against Berkeley Earth Land+Ocean (independent station-based dataset)
- 10 figures in 6 groups: A (3 bias maps: annual/DJF/JJA), B (1 timeseries), C (1 seasonal cycle), D (1 zonal mean), E (3 warming trend maps: global + Arctic/Antarctic polar stereo), F (1 Taylor diagram)
- Berkeley Earth data: 1° lat/lon, degC, dims `latitude`/`longitude`, lons -180..180, var name `2t`
- Path: `/work/bb1153/b382289/data/aqua-dvc/datasets/BERKELEY-EARTH/aqua-filled/Berkeley-Earth_aqua-filled_1x1_1979-2024.nc`
- `_load_berkeley_earth()`: renames dims (latitude→lat, longitude→lon), shifts lons (-180..180 → 0..360), converts degC→K (+273.15)
- All spatial bias/trend computations done on common nereus 0.25° grid (both model and obs regridded)
- Trend maps: `linear_trend()` × 10 for K/decade, symmetric colorbar (centered on zero)
- Polar trend maps: cartopy `NorthPolarStereo()`/`SouthPolarStereo()`, `nr.plot(projection="np"/"sp")`
- Taylor diagram: `plot_taylor_diagram()` in `plot/lines.py`, polar axes with `theta=arccos(corr)`, `r=std_ratio`, CRMS contour circles
- Statistics: area-weighted pattern correlation, normalised STD ratio, RMSE, regional mean bias
- CMIP6 trends: regrid each model individually to common grid, then average (never `xr.align()` on native grids)
- 81 dedicated tests in `tests/test_temperature_berkeley.py`

### TeleconnectionDiag diagnostic
- 13th diagnostic: large-scale climate variability modes (ENSO, NAO, SAM, AO, IOD, PDO, QBO)
- 7 modes × 4 figure types = up to 28 figures: index time series, spatial pattern map, power spectrum, seasonal variance profile
- Mode registry: `ModeDefinition` dataclass with method (box_mean, box_diff, eof, zonal_mean), region, sign convention
- ENSO: Nino 3.4 box mean (190-240E, 5S-5N) of deseasonalised SST anomalies
- NAO/SAM/AO: EOF1 of regional SLP anomalies (NAO: N. Atlantic, SAM: SH, AO: NH)
- IOD: western IO box mean minus eastern IO box mean (SST)
- PDO: EOF1 of N. Pacific SST with global-mean SST removed
- QBO: equatorial zonal-mean zonal wind at 50 hPa (gracefully skips if no pressure levels)
- Spatial patterns: EOF loading (EOF modes) or regression map (box-index modes) — shows teleconnection footprint
- Power spectrum: `feather/util/spectrum.py` wraps `scipy.signal.welch()`, returns periods in years
- EOF utility: `feather/util/eof.py` uses `np.linalg.svd` with cos-lat weighting, sign convention fixing
- No regridding for index computation (box means on native grid, EOF on native regional subset)
- CMIP6: per-model indices → aligned + averaged for MMM; `cmip6_individual` for individual model overlay
- 4-layer z-order for time series: CMIP6 individual → models → obs (no MMM for variability modes — averaging smears signal)
- `_field_global_mean()` uses cos-lat weighting directly (grid-config-independent), used for PDO global SST removal
- **Curvilinear CMIP6 ocean grids** (ORCA, tripolar): `_find_latlon()` detects `nav_lat`/`nav_lon` etc.; `_get_latlon_arrays()` wraps `nereus.extract_coordinates()` with validation (falls back to named coords when nereus returns integer dim indices). Box means use `nr.subset_by_bbox()` on flattened coord arrays. EOF on curvilinear grids uses `_compute_eof_flat()` (returns PC index only, no spatial pattern). Regression patterns work on any grid shape.
- **Pattern display**: all patterns (obs, model, CMIP6) regridded to common 1° grid via `_regrid_patterns_to_common()`. Rectilinear uses `xr.DataArray.interp()`, curvilinear uses `nr.regrid()` with explicit lon/lat arrays. All panels rendered with `method="linear"` in `nr.plot()` for smooth display.
- 87 dedicated tests across `tests/test_eof.py` (13), `tests/test_spectrum.py` (9), `tests/test_teleconnections.py` (65)

### ObsComparisonDiag diagnostic
- 15th diagnostic: ERA5 vs Berkeley Earth comparison for global trends and climatological biases
- Obs-only: no model data loaded — purely observational dataset intercomparison
- Compares two periods: **1980–2014** (model-comparable) and **1980–2024** (full record)
- Variable: `tas` (2m temperature)
- 10 figures across 4 groups:
  - **A** (×3 annual/DJF/JJA): 4-panel trend maps — ERA5 1980–2014, ERA5 1980–2024, BE 1980–2014, BE 1980–2024
  - **B** (×3): 4-panel trend differences — ERA5 period diff, BE period diff, ERA5−BE for each period
  - **C** (×1): global mean annual time series for both datasets (1980–2024)
  - **D** (×3): 5-panel maps — ERA5 ref, Berkeley Earth ref, ERA5 period diff, ERA5−BE (short), ERA5−BE (long)
- Trend: `linear_trend(annual_mean(da)) * 10` for annual; `seasonal_annual_mean` for DJF/JJA
- Common grid: ERA5 0.25°, bilinear via `xr.DataArray.interp` with lon wraparound padding (avoids white line at 0°/360° seam in Robinson maps)
- All display units: **°C** (subtract 273.15 at plot time; stored as K internally); trends are °C/decade
- Berkeley Earth loading: same as `TemperatureBerkeley._load_berkeley_earth()` — renames dims, shifts lons to 0..360, converts degC→K
- Requires `BERKELEY_EARTH` in `obs_datasets` config (path same as used by `temperature_berkeley`)
- Climatological bias range fixed at **±3°C** for difference panels
- 70 dedicated tests in `tests/test_obs_comparison.py`

### PrecipObsComparisonDiag diagnostic
- 16th diagnostic: ERA5 vs MSWEP v2.8 precipitation comparison for trends and biases
- Obs-only: no model data loaded — purely observational dataset intercomparison
- Compares two periods: **1980–2014** (model-comparable) and **1980–2023** (MSWEP near-real-time coverage)
- Variable: `pr` (total precipitation)
- 13 figures across 5 groups:
  - **A** (×3 annual/DJF/JJA): 4-panel trend maps — ERA5 1980–2014, ERA5 1980–2023, MSWEP 1980–2014, MSWEP 1980–2023
  - **B** (×3): 4-panel trend differences — ERA5 period diff, MSWEP period diff, ERA5−MSWEP (short), ERA5−MSWEP (long)
  - **C** (×1): global mean annual time series for both datasets (mm/day)
  - **D** (×3): 5-panel maps — ERA5 ref, MSWEP ref, ERA5 period diff, ERA5−MSWEP (short), ERA5−MSWEP (long)
  - **E** (×3): relative bias (ERA5 − MSWEP) / MSWEP × 100 %, masked where MSWEP < 0.1 mm/day
- Common grid: **ERA5 native 0.25°** — MSWEP (0.1°) regridded to ERA5 coordinates
- MSWEP loaded via `obs_loader.load_mswep(period)` → kg/m²/s; ERA5 via `_load_obs_var("pr")` → kg/m²/s
- All display units: **mm/day** (× 86400 at plot time; kg/m²/s internally); trends are mm/day/decade
- Colormaps: `cmo.tarn` for absolute precipitation and trends; `BrBG` for biases and differences
- Lon wraparound padding in `_interp_to_era5()` avoids white line at 0°/360° seam
- Requires `MSWEP` in `obs_datasets` config (already present in `eerie.yaml`)
- 73 dedicated tests in `tests/test_precip_obs_comparison.py`
- **Extra secondary datasets (CHIRPS, CRU)**: when configured, `_run_extra_secondaries()` layers additional ERA5-vs-CHIRPS and ERA5-vs-CRU figure sets (figure ids `pr_chirps_*`, `pr_cru_*`) via the shared engine on a **0.5° land-masked** grid over **1981–2014 / 1981–2023**. Skipped silently if the dataset key is absent from `obs_datasets`, so existing MSWEP-only configs/tests are unaffected. CHIRPS comparisons clipped to 60°N–60°S. Primary ERA5/MSWEP clim+diff also exported to NetCDF.

### Observational-comparison engine (`_obs_compare_common.py`)
- Shared pairwise obs-vs-obs comparison engine used by `obs_comparison` (CRU `tmp` secondary), `precip_obs_comparison` (CHIRPS + CRU `pre` secondaries), `cloud_obs_comparison`, and `temp_extremes_obs_comparison`
- `CompareSpec` declares display units/factor/offset, colormaps, `land_only`, `relative_bias`, `extent`; `PairwiseObsComparison` does compute → figures → `export_netcdf`
- Common grid **0.5° land-masked** (`common_grid_05()`); reference (often global ERA5/Berkeley) masked to the land-only secondary's valid cells when `land_only=True`
- `compute_trend()`: annual uses `annual_mean`→`dim="time"`; seasonal uses `seasonal_annual_mean`→`dim="year"` (mismatching these raises `KeyError: 'year'`)
- `area_weighted_annual_series()`: cos-lat weighted, NaN-aware (land-only safe); re-indexes year-end time → integer `year`
- NetCDF written to `{output_dir}/netcdf/{diag}/{var}_{token}_{period}_clim_diff.nc` in canonical units

### CloudObsComparisonDiag + TempExtremesObsComparisonDiag
- `cloud_obs_comparison` (group `clouds`): ERA5 `tcc`→`clt` (%) vs CRU `cld`; trend/trend-diff/clim figures + land-mean timeseries
- `temp_extremes_obs_comparison` (group `temperature`): Berkeley Earth **Land** TMAX/TMIN vs CRU `tmx`/`tmn` for `tasmax`/`tasmin` (registry entries added); reference = Berkeley, both land-only 0.25°
- Berkeley Land TMAX/TMIN share the Global TAVG HR format (decimal-year time, anomaly + 12-month climatology) → loaded by the generalized `ObsLoader.load_berkeley_hr(dataset_key, period)`

### ERA5 derived daily/monthly tasmin/tasmax
- Source: ERA5 hourly 2 m temperature (code 167) at `/pool/data/ERA5/E5/sf/an/1H/167/E5sf00_1H_YYYY-MM-DD_167.grb` (24 steps/day, native reduced-Gaussian 1D grid)
- `scripts/era5_derive_tasminmax.sh` (SLURM array over years): per-day `cdo daymin`/`daymax` of the 24 hourly values → regridded to **regular 0.25° lat/lon** (`r1440x721`, K) using cached bilinear weights (reduced-Gaussian needs `-setgridtype,regular` before `remapbil`). Default period **1980–2023**. `mergetime` must run standalone (not chained inside `-setattribute`); units set in a separate `setattribute` pass.
- **ERA5 land-sea mask**: `scripts/era5_derive_landmask.sh` derives `sftlf` (CMOR `fx`, land area %) from the invariant ERA5 LSM (code 172, `/pool/data/ERA5/E5/sf/an/IV/172/E5sf00_IV_INVARIANT_172.grb`) via conservative remap → `…/era5_derived/CMOR/ECMWF/ERA5/era5/r1i1p1f1/fx/sftlf/gr/v1/sftlf_fx_ERA5_era5_r1i1p1f1_gr.nc`. Light single field (login-node OK). Registry gained `sftlf` (group `fixed`).
- **Land masking in `tropical_nights`/`heatwave`**: `_load_land_mask(model, lat, lon)` now prefers the model's own `sftlf` (fx, land>50%) via `_load_model_land_mask()`, falling back to the Berkeley Earth `land_mask` when the model has no `sftlf` (most EERIE models). So the ERA5 run is masked by ERA5's own LSM.
- Daily output is a **CMOR tree** so `CMORLoader` reads it via `table="day"`: `/work/bm1344/AWI/OBS/era5_derived/CMOR/ECMWF/ERA5/era5/r1i1p1f1/day/{tasmin,tasmax}/gr/v1/{var}_day_ERA5_era5_r1i1p1f1_gr_{YYYY}.nc`
- Monthly means of the daily extremes → `/work/bm1344/AWI/OBS/era5_derived/mon/ERA5_{tasmin_daymin,tasmax_daymax}_mon_1981-2023.nc` (merged by `scripts/era5_derive_finalize.sh`)
- `configs/era5_indices.yaml`: ERA5 as a single CMOR "model" → runs `tropical_nights`/`heatwave` on the derived daily data (Berkeley Land TMIN/TMAX for bias maps)
- `temp_extremes_obs_comparison` adds **ERA5 as a second secondary** (alongside CRU) when `ERA5_TMINMAX` is in `obs_datasets` (monthly files); figure ids `tasmax_era5_*`/`tasmin_era5_*`. Each secondary is loaded independently so a missing ERA5 file doesn't drop the CRU comparison.

### Configurable extremes obs reference (ERA5 vs Berkeley)
- `feather/diag/_extremes_obs.py`: when `project.extremes_obs_reference: "ERA5"` and `ERA5_TMINMAX` is in `obs_datasets`, the tropical-nights/heatwave diagnostics (+ their `_change` variants) use the derived ERA5 monthly tasmin/tasmax as the obs reference instead of Berkeley Earth Land TMIN/TMAX
- `use_era5_obs()`, `obs_ref_label()`, `era5_mean_available()`, `load_era5_mean()` (period-mean K on model grid), `era5_approx_exceedance_series()` (approx obs-TN series from monthly means, land-masked via `ERA5_SFTLF`)
- ERA5 monthly tasmin/tasmax are absolute K on regular 0.25° — period mean is a plain time average (no anomaly+climatology reconstruction); Berkeley path unchanged (default)
- `configs/eerie_all_members_indices.yaml`: single combined config for all four diagnostics — flat `models` (all EERIE members + ERA5 as a CMOR "model" reading its derived daily tree) for base TN/heatwave, plus a `climate_change` block for the `_change` diagnostics. ERA5 participates in the base diagnostics (daily indices, masked by its own `sftlf`) AND is the obs reference everywhere. ERA5 is not in `climate_change.models` (no future scenario).
- 9 dedicated tests in `tests/test_extremes_obs.py`

### CRU TS / CHIRPS loaders (`ObsLoader`)
- `load_cru(variable, period)`: globs per-decade `cru_ts4.09.*.{var}.dat.nc`, `open_mfdataset`, drops aux vars (`stn`/`mae`/`maea`); units → canonical (`pre` mm/month→kg/m²/s, `tmp`/`tmn`/`tmx` °C→K, `cld` % unchanged); land-only (ocean NaN); lons → 0..360
- `load_chirps(period)`: single 0.05° file (60°N–60°S land), `precip` mm/month→kg/m²/s, `-9999`→NaN, dims renamed lat/lon, lons → 0..360
- 30 dedicated tests in `tests/test_obs_cru_chirps.py` (loaders + engine + both new diagnostics)
### AddedValueDiag diagnostic
- 14th diagnostic: Dosio et al. (2015) Added Value (AV) of EERIE ensemble vs CMIP6 MMM
- AV = (sq_err_CMIP6 - sq_err_EERIE) / max(sq_err_CMIP6, sq_err_EERIE), bounded [-1, 1]
- AV > 0: EERIE reduces squared error vs obs compared to CMIP6 MMM (EERIE adds value)
- 18 surface variables (same list as GlobalBiases); 3 periods per variable: annual, DJF, JJA
- **Figure 1** (`{var}_{period}_added_value`): 2-panel — AV of EERIE ensemble mean + ensemble median
- **Figure 2** (`{var}_{period}_added_value_models`): one panel per EERIE model (vs CMIP6 MMM) + one per CMIP6 model (vs EERIE mean)
- **Colormap**: `cmo.tarn` cropped 50% both ends via `cmocean.tools.crop_by_percent` (module-level `_AV_CMAP`; AV > 0 = teal/green = EERIE adds value; AV < 0 = brown = CMIP6 better)
- **Alternative obs datasets**: `tas` uses Berkeley Earth 0.25° HR (`BERKELEY_EARTH_HR` in config) instead of ERA5; `pr` uses MSWEP v2.8 (`MSWEP` in config); all other variables use ERA5. Falls back to ERA5 if the preferred dataset is not in config.
- **`pr` units**: mm/day — MSWEP (kg/m²/s) and all model `pr` fields multiplied by 86400 before AV computation. AV is dimensionless so values are unaffected; units are consistent.
- `_OBS_ALT_DATASETS` class dict maps variable → preferred obs dataset name
- `_UNIT_FACTORS` class dict maps variable → unit conversion factor (pr: 86400.0)
- `_load_obs_for_var()`: dispatches to `_load_berkeley_earth_for_av()`, `_load_mswep_for_av()`, or `_load_obs_var()`
- `_load_berkeley_earth_for_av()`: decimal-year time → datetime, anomaly + climatology reconstruction, degC→K (same logic as ObsComparisonDiag)
- NC checkpoint: 2 etypes × 3 periods = 6 files per variable (`{var}_{period}_{etype}_av.nc`)
- NC path only reconstructs Figure 1; Figure 2 requires full recomputation (per-model data not persisted)
- EERIE models regridded with per-grid interpolator cache (same pattern as GlobalBiases)
- CMIP6 models regridded via `_regrid_to_target()` with 250 km floor on influence_radius
- `_compute_av(m1, m2, ref)`: static, handles denom=0 via `np.errstate` + `np.where`
- 24 dedicated tests in `tests/test_added_value.py`

### LLM analysis
- `FigureAnalyzer` scans `{output_dir}/figures/` for PNG+JSON pairs, sends to Gemini, saves to `{output_dir}/analysis/`
- No dependency on xarray/dask/healpy — works entirely on already-generated figures
- **Config-driven prompts**: system prompts are parameterized with model names, project name, and resolution from config. Functions accept `models=`, `project_name=`, `resolution=` kwargs with DestinE defaults.
- Config uses per-purpose providers: `llm.figure_analysis` (Gemini), future `llm.report_generation` (OpenAI, etc.)
- API key from env var (`VERTEX_API_KEY` by default) or passed directly
- `_parse_json_response()` handles markdown fencing and LaTeX escape sequences (`\Delta`, `\degree`)
- `skip_existing` (default true) enables incremental re-runs
- Run via: `feather --steps analyze --api-key $VERTEX_API_KEY -v`

### Report generation (OpenAI)
- `ReportGenerator` produces a LaTeX report from LLM analysis output via OpenAI
- 3-stage pipeline: curation (select/group figures) → section writing (per-section prose) → LaTeX assembly
- **Config-driven prompts**: curation and section system prompts use config-derived model names, project name, resolution, and period
- Pydantic schemas: `ReportStructure`, `WrittenSection`, `SelectedFigure`, `ReportSection`
- `escape_latex()` preserves `\ref{}` via placeholder pattern, escapes LaTeX specials + Unicode
- LaTeX template uses custom Jinja2 delimiters (`<< >>`, `<% %>`) to avoid LaTeX `{}` conflicts
- Caching: `publication/structure.json` and `publication/sections/{id}.json` for resumable runs
- `OpenAIClient` has retry logic (3 attempts) and handles markdown-fenced JSON + LaTeX escapes in responses
- Run via: `feather --steps report --openai-api-key $OPENAI_API_KEY -v`

### GRIBLoader
- `feather/data/grib_loader.py`: loads GRIB files on regular lat/lon grid
- Uses `xr.open_mfdataset()` with `engine="cfgrib"` and `preprocess=_normalise_ocean_layers()`
- Ocean depth levels: configurable per model via dict format `depth_levels: {model_name: [depths]}`
- `_normalise_ocean_layers()`: fixes inconsistent `oceanModelLayer` indices across GRIB files (even vs odd numbering) by normalizing to sequential 0..N-1
- Variable mapping: CMOR → GRIB short name via `_GRIB_VAR_MAP` (e.g., `tas → 2t`, `pr → tp`)
- Grid: regular lat/lon, standard `(time, latitude, longitude)` dims
- Config: `configs/tco_grib.yaml`, also used as secondary backend in `configs/ifs_fesom_combined.yaml`

### CompositeModelLoader
- `feather/data/composite_loader.py`: routes `load_var()` and `load_coords()` to the correct backend per model
- Triggered when `config.is_multi_source()` returns `True` (models have different `data_source_type`)
- Groups models by backend type, instantiates one loader per type
- `_DestinECatalogAdapter`: wraps `MultiCatalogLoader` with the unified `load_var(model, variable)` API
- Uses `catalog_key` from ModelConfig to map model names to catalog entries (when they differ)
- Uses `member` from ModelConfig for ensemble member selection in catalog keys
- `DiagnosticBase._load_model_var()` detects `CompositeModelLoader` and delegates directly (bypasses old dispatch logic)

### Comparison-type LLM prompts
- LLM prompts adapt to four comparison types: `multi_model`, `resolution_sensitivity`, `single_model`, `baseline_evaluation`
- Set via `project.comparison_type` in config YAML
- `project.comparison_description` adds optional free-text context
- Affects figure analysis (Gemini), synthesis, report curation, and section writing (OpenAI)
- Prompts are composed from reusable blocks: shared figure-type descriptions + type-specific focus sections
- `build_figure_analysis_system()`, `build_synthesis_system()` in `llm/prompts.py`
- `build_curation_system()`, `build_section_system()` in `export/prompts.py`

### Per-grid interpolator cache
- When models have different grid sizes (e.g., nside=1024 vs nside=128, or different lat/lon resolutions), each grid needs its own nereus interpolator
- `_interp_cache: dict[int, Any]` caches interpolators keyed by `n_src` (number of source points)
- Applied across 7 bias-map diagnostics: global_biases, global_trends, climate_variability, ocean_en4, precipitation_mswep, radiation_budget, temperature_berkeley
- Essential for multi-resolution evaluations via `CompositeModelLoader`

### Pipeline runner
- `feather` CLI command registered via `[project.scripts]` in `pyproject.toml`
- Also available as `python -m feather`
- 4-stage pipeline: `diagnostics → analyze → report → website`
- Each step independently runnable via `--steps`; default is `all`
- `--cmip6-individual` flag plots individual CMIP6 model lines/biases + MMM (passed only to diagnostics that accept it via `inspect.signature()`; supported by `GlobalBiases`, `SeasonalCycleDiag`, `TimeseriesDiag`, `RadiationBudget`, `SeaIceDiag`, `OceanSST`, `OceanEN4`)
- `--no-llm` flag generates a figures-only website without LLM content: skips the `analyze` step entirely and suppresses synthesis boxes, analysis panels, headline findings, and AI Analysis badges in the HTML output. Metadata tables remain visible. Useful for quick dashboards without running LLM analysis.
- `run_pipeline()` returns summary dict: `{"figures": N, "analyses": N, ...}`
- All `scripts/` files are legacy thin wrappers delegating to `feather.cli:main()`

### Test fixtures
- `synth_healpix` in `conftest.py`: nside=8, 768 cells, 12 timesteps, temperature gradient pole→equator
- `synth_obs` in `conftest.py`: 5° regular lat/lon grid, matching temperature field
- `synth_cmip6` in `conftest.py`: 5° lat/lon grid, 12 timesteps, `tas` + `areacella`
- `MockCMIP6Loader` in `conftest.py`: follows real CMIP6Loader API with synthetic data
- Registry tests use `autouse` fixture to save/restore `_REGISTRY` global state

## Dependencies

Core: xarray, dask, distributed, numpy, scipy, matplotlib, cartopy, intake, intake-xarray, healpy, nereus, pyyaml, netcdf4, zarr, cmocean, nbformat, pydantic, google-generativeai, openai, jinja2

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
