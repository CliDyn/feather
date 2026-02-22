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

Current test count: 487 unit tests + 4 integration tests.

**Note:** Unit tests use small synthetic data (nside=8, 768 cells) and are safe to run on the login node. Integration tests (`-m integration`) access real data files but only open metadata/small slices — they are also safe on the login node. For any end-to-end test that runs full diagnostics on real data (nside=1024, 12.6M cells), ask the user to execute it in a compute environment.

## Project structure

```
feather/                     # Package root
├── cli.py                   # CLI entry point (feather command)
├── run.py                   # Pipeline orchestration (run_pipeline)
├── __main__.py              # python -m feather support
├── config.py                # FeatherConfig dataclass + YAML loading
├── data/
│   ├── loader.py            # DataLoader (intake catalogs + file paths)
│   ├── obs.py               # ObsLoader (observations from config) + load_ceres()
│   ├── cmip6.py             # CMIP6Loader (multi-model mean from zarr)
│   └── variables.py         # VarInfo dataclass + VARIABLE_REGISTRY (33 vars)
├── util/
│   ├── spatial.py           # zonal_mean, global_mean, regional_mean, latlon_global_mean, compute_latlon_areas
│   ├── temporal.py          # climatology, anomaly, seasonal/monthly grouping
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
│   └── sea_ice.py           # SeaIceDiag: sea ice area/extent/volume/spatial
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
| `configs/default.yaml` | Default configuration with all Levante data paths |
| `feather/cli.py` | CLI entry point — `feather` command (argparse) |
| `feather/run.py` | Pipeline orchestration — `run_pipeline()` |
| `feather/data/variables.py` | Central variable registry — add new variables here |
| `feather/diag/base.py` | Base class for all diagnostics — subclass this |
| `feather/diag/registry.py` | `@register` decorator for diagnostic auto-discovery |
| `feather/data/cmip6.py` | CMIP6Loader — load zarr, compute multi-model mean |
| `feather/llm/analyzer.py` | FigureAnalyzer — Gemini-based figure analysis |
| `feather/llm/schemas.py` | Pydantic models for structured LLM output |
| `feather/export/report.py` | ReportGenerator — LaTeX report via OpenAI (3-stage) |
| `feather/website/generator.py` | SiteGenerator — static HTML dashboard from figures + analysis |
| `tests/conftest.py` | Synthetic HEALPix/obs/CMIP6 fixtures, mock loaders, minimal_config |
| `PLAN.md` | Full implementation plan with phase status |

## Configuration

`FeatherConfig` is loaded from YAML via `FeatherConfig.from_yaml("configs/default.yaml")`.

Key fields:
- `model_catalogs` — paths to intake catalog YAML files (2D, 3D)
- `models` — list of model names: `["ifs-fesom", "ifs-nemo", "icon"]`
- `obs_root` — root path for observation data
- `obs_datasets` — nested dict mapping dataset → variables → filenames
- `cmip6` — CMIP6 comparison config (enabled by default)
- `output_dir` — where figures/analysis/site are written
- `llm` — LLM provider config (per-purpose: `figure_analysis` uses Gemini)
- `report` — report generation config (model, max_tokens, temperature, api_key_env, n_highlights)

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

### ERA5 variable naming and units
- File keys in config (e.g., `t2m`) differ from NetCDF internal names (e.g., `T2M`)
- `ObsLoader._find_variable()` handles this with case-insensitive fallback
- ERA5 radiation/flux variables are stored as **daily accumulations** (J/m²/day), not W/m². Divide by 86400 to get W/m².
- ERA5 precipitation (`TP`) is in m/day. Multiply by `1000/86400` to get kg/m²/s.
- ERA5 cloud cover (`TCC`) is 0-1 fraction; DestinE `avg_tcc` is 0-100%. Multiply by 100.
- ERA5 and DestinE share IFS sign convention: surface heat fluxes **positive downward** (into surface)
- CMIP6 `hfss`/`hfls` are positive **upward** (opposite sign). CMIP6 mapping omitted for these variables.
- Unit conversions are defined via `obs_unit_factor` in `VARIABLE_REGISTRY` (`feather/data/variables.py`)

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
- All 5 diagnostics (timeseries, seasonal_cycle, global_biases, radiation_budget, sea_ice) integrated — CMIP6 MMM lines/bias maps added when `cmip6.enabled: true`
- `get_member_pairs()` public API for listing (model, variant) tuples

### GlobalBiases diagnostic
- Produces **combined multi-panel figures**: 1 obs panel + N bias panels per variable per period
- `plot_combined_bias_map()` in `plot/maps.py` handles layout with `max_cols` overflow to extra rows
- 18 validated DestinE↔ERA5 surface variables (temperature, pressure, wind, clouds, precipitation, radiation, heat fluxes)
- Gracefully skips models missing a variable (e.g., ICON lacks `avg_tcc`) with a warning
- `--cmip6-individual` CLI flag enables individual CMIP6 model bias panels **plus** MMM (both computed together)
- Without the flag, only CMIP6 MMM is shown (default behavior)
- `--variables` CLI flag intersects with diagnostic's supported list; warns about unsupported variables

### Timeseries & SeasonalCycle diagnostics
- Both share the same 18-variable list as GlobalBiases (temperature, pressure, wind, clouds, precipitation, radiation, heat fluxes)
- Both support `cmip6_individual=True` for individual CMIP6 model lines + MMM
- 4-layer plotting: individual CMIP6 (background, semi-transparent) → MMM (dashed) → DestinE models (foreground) → Obs (top)
- Gracefully skip models missing a variable via `try/except KeyError`
- Group is `"evaluation"` (spans multiple physical domains)
- CMIP6 individual series come from `_cmip6_global_mean_timeseries(return_individual=True)` on `DiagnosticBase`

### RadiationBudget diagnostic
- 4th diagnostic: computes derived radiation quantities and produces budget bars, Gregory plot, imbalance time series, and bias maps
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

### LLM analysis
- `FigureAnalyzer` scans `{output_dir}/figures/` for PNG+JSON pairs, sends to Gemini, saves to `{output_dir}/analysis/`
- No dependency on xarray/dask/healpy — works entirely on already-generated figures
- Config uses per-purpose providers: `llm.figure_analysis` (Gemini), future `llm.report_generation` (OpenAI, etc.)
- API key from env var (`GEMINI_API_KEY` by default) or passed directly
- `_parse_json_response()` handles markdown fencing and LaTeX escape sequences (`\Delta`, `\degree`)
- `skip_existing` (default true) enables incremental re-runs
- Run via: `feather --steps analyze --api-key $GEMINI_API_KEY -v`

### Report generation (OpenAI)
- `ReportGenerator` produces a LaTeX report from LLM analysis output via OpenAI
- 3-stage pipeline: curation (select/group figures) → section writing (per-section prose) → LaTeX assembly
- Pydantic schemas: `ReportStructure`, `WrittenSection`, `SelectedFigure`, `ReportSection`
- `escape_latex()` preserves `\ref{}` via placeholder pattern, escapes LaTeX specials + Unicode
- LaTeX template uses custom Jinja2 delimiters (`<< >>`, `<% %>`) to avoid LaTeX `{}` conflicts
- Caching: `publication/structure.json` and `publication/sections/{id}.json` for resumable runs
- `OpenAIClient` has retry logic (3 attempts) and handles markdown-fenced JSON + LaTeX escapes in responses
- Run via: `feather --steps report --openai-api-key $OPENAI_API_KEY -v`

### Pipeline runner
- `feather` CLI command registered via `[project.scripts]` in `pyproject.toml`
- Also available as `python -m feather`
- 4-stage pipeline: `diagnostics → analyze → report → website`
- Each step independently runnable via `--steps`; default is `all`
- `--cmip6-individual` flag plots individual CMIP6 model lines/biases + MMM (passed only to diagnostics that accept it via `inspect.signature()`; supported by `GlobalBiases`, `SeasonalCycleDiag`, `TimeseriesDiag`, `RadiationBudget`, `SeaIceDiag`)
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
