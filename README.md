# Feather

Lightweight climate model evaluation framework supporting multiple high-resolution model sets.

Feather compares high-resolution climate models against observations (ERA5, CERES, EN4, ESA-CCI, OSI-SAF, PIOMAS/GIOMAS, MSWEP, Berkeley Earth) and a CMIP6 multi-model mean. It generates diagnostic figures, LLM-analyzed results, a LaTeX report, and a static web dashboard.

## Supported model sets

| Model set | Models | Resolution | Grid | Data backend |
|-----------|--------|-----------|------|-------------|
| **DestinE** | IFS-FESOM, IFS-NEMO, ICON | ~5 km | HEALPix | intake catalogs |
| **EERIE HighResMIP** | IFS-FESOM2-SR, IFS-NEMO-ER, ICON-ESM-ER, HadGEM3-GC5 | ~10 km atm, ~5-10 km ocean | 0.25° lat/lon | CMOR directory tree |
| **IFS-FESOM T319** | IFS-FESOM (T319) | ~60 km | HEALPix | per-year NetCDF |
| **DestinE GRIB** | IFS-FESOM TCO399/TCO319 | ~25-35 km | lat/lon | GRIB files |
| **TerraDT** | IFS-FESOM, IFS-NEMO, ICON | ~5 km | HEALPix | intake catalogs |
| **Combined** | TCO2999 + TCO399 + TCO319 | mixed | mixed | multi-source |

The framework is **grid-agnostic**: diagnostics automatically dispatch between HEALPix and regular lat/lon grids based on per-model configuration. The **CompositeModelLoader** enables mixing models from different data backends in the same evaluation run.

## Installation

### From scratch (conda/mamba)

```bash
# Create the environment with all dependencies
conda env create -f environment.yml

# Or using mamba/micromamba (faster)
mamba env create -f environment.yml

# Activate and install feather in editable mode
conda activate feather
pip install -e .
```

### Into an existing environment

```bash
pip install -e .
```

See `environment.yml` for the full list of dependencies. Key packages that are best installed via conda-forge: `cartopy`, `healpy`, `netcdf4`, `eccodes`, `cfgrib`.

After installation, the `feather` command is available.

## Pipeline

Feather has four pipeline stages that can be run individually or together:

```
diagnostics  →  analyze  →  report  →  website
(compute+plot)  (Gemini)   (OpenAI)   (HTML)
```

### Run the full pipeline

```bash
# DestinE (default config)
feather --config configs/default.yaml -v

# EERIE HighResMIP
feather --config configs/eerie.yaml -v

# TerraDT baseline evaluation
feather --config configs/terradt.yaml -v

# Multi-resolution IFS-FESOM comparison (mixed data sources)
feather --config configs/ifs_fesom_combined.yaml -v
```

### Run individual steps

```bash
# Generate diagnostic figures (requires compute node for real data)
feather --steps diagnostics -v

# LLM analysis of figures (Gemini)
feather --steps analyze --api-key $GEMINI_API_KEY -v

# Generate LaTeX report (OpenAI)
feather --steps report --openai-api-key $OPENAI_API_KEY -v

# Build static website
feather --steps website -v

# Combine steps
feather --steps analyze report website -v
```

### CLI options

```bash
# Filter diagnostics by name
feather --diagnostics global_biases timeseries -v

# Filter by variable (CMOR canonical names)
feather --variables tas pr -v

# Include individual CMIP6 model lines/biases alongside MMM
feather --cmip6-individual -v

# Figures-only website (no LLM analysis)
feather --no-llm -v

# Custom output directory
feather --output ./my_output -v

# Re-run everything (ignore cached results)
feather --no-skip-existing -v

# Custom time period
feather --period 1990 2014 -v

# Compile LaTeX report to PDF
feather --steps report --compile-pdf -v

# Also works as a module
python -m feather --config configs/eerie.yaml -v
```

## Python API

```python
from feather import FeatherConfig, run_pipeline

# Load any config
cfg = FeatherConfig.from_yaml("configs/eerie.yaml")

# Full pipeline
result = run_pipeline(cfg, steps="all", api_key="...", openai_api_key="...")

# Individual steps
result = run_pipeline(cfg, steps=["diagnostics"], diagnostics=["global_biases"])
result = run_pipeline(cfg, steps=["report"], openai_api_key="...")

# Result dict
print(result)
# {"figures": 42, "analyses": 42, "syntheses": 12, "report": Path(...), "site_dir": Path(...)}
```

## Available diagnostics

Feather provides 12 registered diagnostics across atmosphere, ocean, cryosphere, and cross-domain evaluation:

### Atmosphere

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `global_biases` | `GlobalBiases` | ERA5 | Multi-panel bias maps (obs + all model biases), annual/DJF/JJA |
| `timeseries` | `TimeseriesDiag` | ERA5 | Global-mean time series, all models + obs overlaid |
| `seasonal_cycle` | `SeasonalCycleDiag` | ERA5 | 12-month climatological cycle, all models + obs overlaid |
| `global_trends` | `GlobalTrends` | ERA5 | Per-grid-point linear trends over the evaluation period |
| `climate_variability` | `ClimateVariability` | ERA5 | STD maps of deseasonalised, detrended monthly fields |
| `radiation_budget` | `RadiationBudget` | CERES EBAF | TOA/surface radiation budget bars, Gregory plot, imbalance time series, bias maps |

### Ocean

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `ocean_sst` | `OceanSST` | ESA-CCI | SST bias maps, time series, seasonal cycle, zonal mean |
| `ocean_en4` | `OceanEN4` | EN4 v4.2.2 | 3D ocean T/S: surface bias maps, Hovmoller diagrams, depth-layer time series |

### Cryosphere

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `sea_ice` | `SeaIceDiag` | OSI-SAF, PIOMAS/GIOMAS | Sea ice area/extent/volume time series, seasonal cycles, trends, polar spatial maps |

### Cross-domain

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `precipitation_mswep` | `PrecipitationMSWEP` | MSWEP v2.8 | Precipitation bias maps (absolute + relative), time series, seasonal cycle, zonal mean, intensity PDF |
| `temperature_berkeley` | `TemperatureBerkeley` | Berkeley Earth | T2m bias maps, warming trend maps (global + polar), Taylor diagram |
| `teleconnections` | `TeleconnectionDiag` | ERA5 | Climate variability modes (ENSO, NAO, SAM, AO, IOD, PDO, QBO): index time series, spatial patterns, power spectra, seasonal variance |

All diagnostics support:
- `variables=["tas", ...]` — filter which variables to evaluate (CMOR canonical names)
- `experiment="hist-1950"` — experiment identifier (from config by default)
- `period=("1980", "2014")` — time period for climatologies (from config by default)
- `cmip6_individual=True` — individual CMIP6 model lines/biases alongside MMM

## Configuration

Feather uses YAML configuration files. Nine configs are provided:

| Config | Model set | Data source | Comparison type |
|--------|----------|------------|-----------------|
| `configs/default.yaml` | DestinE (3 models) | intake catalogs | `multi_model` |
| `configs/eerie.yaml` | EERIE HighResMIP (4 models) | CMOR directory tree | `multi_model` |
| `configs/himansu_319.yaml` | IFS-FESOM T319 | per-year NetCDF | `single_model` |
| `configs/tco_grib.yaml` | IFS-FESOM TCO399/TCO319 | GRIB files | `resolution_sensitivity` |
| `configs/destine_ifs_fesom.yaml` | IFS-FESOM only | intake catalogs | `single_model` |
| `configs/destine_ifs_nemo.yaml` | IFS-NEMO only | intake catalogs | `single_model` |
| `configs/destine_icon.yaml` | ICON only | intake catalogs | `single_model` |
| `configs/terradt.yaml` | TerraDT (3 models) | intake catalogs | `baseline_evaluation` |
| `configs/ifs_fesom_combined.yaml` | IFS-FESOM multi-res | mixed (catalog + GRIB) | `resolution_sensitivity` |

### Config format

The `models` key determines the config format:
- **List** (`models: [ifs-fesom, ifs-nemo, icon]`) — legacy DestinE format, auto-populates HEALPix grid defaults
- **Dict** (`models: {ModelA: {institution: ..., grids: ...}}`) — structured format with per-model configuration

### Structured config example (EERIE-style)

```yaml
project:
  name: "EERIE"
  description: "EERIE HighResMIP evaluation"
  experiment: "hist-1950"
  period: ["1980", "2014"]
  resolution: "high-resolution (~10 km atm, ~5-10 km ocean)"
  comparison_type: "multi_model"         # multi_model | resolution_sensitivity | single_model | baseline_evaluation
  comparison_description: ""             # optional free-text for LLM prompt framing

data_source:
  type: "cmor"                           # cmor | destine_catalog | netcdf_healpix | grib
  root: "/path/to/CMOR/tree"

models:
  IFS-FESOM2-SR:
    institution: AWI
    experiment: hist-1950
    variant: r1i1p1f1
    grids:
      sfc: latlon                        # "healpix" or "latlon" per domain
      o2d: latlon
      o3d: latlon
    color: "#1f77b4"
```

### Multi-source config (CompositeModelLoader)

For evaluations mixing data backends, set per-model `data_source_type`:

```yaml
data_source:
  type: "destine_catalog"                # default backend

models:
  TCO2999:
    data_source_type: "destine_catalog"  # per-model override (optional, uses default if omitted)
    catalog_key: "ifs-fesom"             # catalog entry name (if different from model name)
    member: 1                            # ensemble member index
    grids: {sfc: healpix, o2d: healpix, o3d: healpix}
  TCO399:
    data_source_type: "grib"             # different backend
    grids: {sfc: latlon, o2d: latlon, o3d: latlon}
```

### Per-model overrides

`ModelConfig` supports:
- `data_root` — custom data path (e.g., HadGEM3 outside shared CMOR root)
- `grid_label` — CMOR grid label override (default `"gr"`)
- `variable_aliases` — CMOR var → on-disk name mapping (e.g., `thetao: thetao-con`)
- `scale_factors` — per-variable multiplier (e.g., `clt: 100` for fraction → %)
- `absolute_salinity: true` — enable SA→SP salinity conversion (NEMO-based models)

### Comparison types

The `comparison_type` field controls how LLM prompts frame the analysis:

| Type | Focus |
|------|-------|
| `multi_model` | Inter-model differences and common biases across a multi-model ensemble |
| `resolution_sensitivity` | How model skill scales with resolution (same model at different grid spacings) |
| `single_model` | Detailed evaluation of one model against observations and CMIP6 |
| `baseline_evaluation` | Whether a model is adequate as a starting point for a downstream project |

### Common config sections

All configs share:
- **Observations:** ERA5, CERES, EN4, ESA-CCI, OSI-SAF, MSWEP, Berkeley Earth
- **CMIP6:** 12 models from zarr archives (configurable ensemble mode)
- **LLM analysis:** Gemini (`GEMINI_API_KEY` / `VERTEX_API_KEY`)
- **Report generation:** OpenAI (`OPENAI_API_KEY`)
- **nereus:** Interpolation settings (`method`, `resolution`, `influence_radius`)

## Data backends

Feather supports five data loading backends:

| Backend | Class | Config type | Grid |
|---------|-------|------------|------|
| DestinE intake catalogs | `MultiCatalogLoader` | `destine_catalog` | HEALPix (zarr) |
| CMOR directory tree | `CMORLoader` | `cmor` | regular lat/lon (NetCDF) |
| Per-year NetCDF | `NetCDFLoader` | `netcdf_healpix` | HEALPix (NetCDF) |
| GRIB files | `GRIBLoader` | `grib` | regular lat/lon (GRIB) |
| Multi-source | `CompositeModelLoader` | mixed | per-model |

The `CompositeModelLoader` automatically routes `load_var()` calls to the correct backend per model based on `data_source_type` in the model config.

## Adding a new model set

1. Create a config YAML using the structured format (see `configs/eerie.yaml` as template)
2. Set `data_source.type` to one of: `"cmor"`, `"destine_catalog"`, `"netcdf_healpix"`, `"grib"`
3. Define `models` as a dict with per-model `institution`, `experiment`, `variant`, `grids`, `color`
4. Set `project.name`, `project.period`, `project.experiment`, `project.resolution`
5. Optionally set `project.comparison_type` for LLM prompt framing
6. For mixed data sources, set per-model `data_source_type` and use `CompositeModelLoader`
7. Run: `feather --config configs/my_project.yaml -v`

If your data format is not supported, implement a new loader class (see `CMORLoader` or `GRIBLoader` as templates) and add a dispatch case in `feather/run.py:_create_model_loader()`.

## Output structure

```
output/
  figures/                          # Diagnostic figures + JSON metadata
    global_biases/
      tas_annual_bias_combined.png
      tas_annual_bias_combined.json
      ...
    timeseries/
    seasonal_cycle/
    radiation_budget/
    sea_ice/
    ocean_sst/
    ocean_en4/
    global_trends/
    climate_variability/
    precipitation_mswep/
    temperature_berkeley/
    teleconnections/
  analysis/                         # LLM analysis (Gemini)
    global_biases/
      tas_annual_bias_combined_analysis.json
      synthesis.json
    ...
  publication/                      # LaTeX report (OpenAI)
    structure.json                  # Stage 1: editorial curation
    sections/                       # Stage 2: per-section prose
    figures/                        # Copied figures for LaTeX
    report.tex                      # Final LaTeX document
    report.pdf                      # (if --compile-pdf)
  site/                             # Static HTML dashboard
    index.html
    global_biases.html
    ...
```

## Project structure

```
feather/
  cli.py                 # CLI entry point (feather command)
  config.py              # FeatherConfig + ModelConfig dataclasses
  run.py                 # Pipeline orchestration (run_pipeline)
  __main__.py            # python -m feather support
  data/
    loader.py            # MultiCatalogLoader (DestinE intake catalogs)
    cmor_loader.py       # CMORLoader (CMOR directory tree, e.g. EERIE)
    netcdf_loader.py     # NetCDFLoader (per-year NetCDF on HEALPix grid)
    grib_loader.py       # GRIBLoader (GRIB files on regular lat/lon grid)
    composite_loader.py  # CompositeModelLoader (multi-source routing)
    obs.py               # ObsLoader (observations from config)
    cmip6.py             # CMIP6Loader (multi-model mean from zarr)
    variables.py         # VARIABLE_REGISTRY (33 vars, CMOR canonical names)
  util/
    spatial.py           # Zonal/global means, regridding, latlon areas
    temporal.py          # Climatology, anomaly, deseason, detrend
    eof.py               # EOF computation via SVD (teleconnections)
    spectrum.py          # Power spectrum via Welch (teleconnections)
    units.py             # Unit conversions
  plot/
    maps.py              # Bias maps (nereus), combined multi-panel maps
    lines.py             # Time series, seasonal cycle, zonal profiles, Taylor diagram
    styles.py            # Model colors, plot defaults
  diag/
    base.py              # DiagnosticBase ABC + grid-agnostic helpers
    registry.py          # @register decorator, diagnostic discovery
    figure_meta.py       # Figure metadata sidecar system
    global_biases.py     # Climatology bias maps (18 variables)
    timeseries.py        # Global-mean time series
    seasonal_cycle.py    # Monthly climatological cycle
    radiation_budget.py  # Radiation budget analysis (CERES EBAF)
    sea_ice.py           # Sea ice evaluation (OSI-SAF, PIOMAS/GIOMAS)
    ocean_sst.py         # SST evaluation (ESA-CCI)
    ocean_en4.py         # 3D ocean T/S (EN4 v4.2.2)
    global_trends.py     # Per-grid-point linear trends
    climate_variability.py # STD of deseasonalised, detrended fields
    precipitation_mswep.py # Precipitation evaluation (MSWEP v2.8)
    temperature_berkeley.py # T2m evaluation (Berkeley Earth)
    teleconnections.py    # Variability modes (ENSO, NAO, SAM, AO, IOD, PDO, QBO)
  llm/
    analyzer.py          # FigureAnalyzer (Gemini, comparison-type aware)
    schemas.py           # FigureAnalysis, DiagnosticSynthesis (Pydantic)
    prompts.py           # Gemini prompts (composable, comparison-type aware)
  export/
    report.py            # ReportGenerator (OpenAI, 3-stage)
    schemas.py           # ReportStructure, WrittenSection
    prompts.py           # OpenAI prompts (composable, comparison-type aware)
    openai_client.py     # Thin OpenAI wrapper with retry
    latex_builder.py     # LaTeX escaping, template rendering, PDF
    templates/           # Jinja2 LaTeX template
  website/
    generator.py         # SiteGenerator (static HTML dashboard)
    templates/           # Jinja2 HTML templates
    static/              # Dark-theme CSS
```

## Adding a new diagnostic

Use CMOR canonical variable names and grid-agnostic base class helpers:

```python
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register

@register
class MyDiagnostic(DiagnosticBase):
    name = "my_diagnostic"
    title = "My Diagnostic"
    domain = "sfc"
    variables = ["tas"]         # CMOR canonical names
    group = "temperature"

    def compute(self):
        for model in self.config.models:
            mdata = self._load_model_var(model, "tas")      # Grid-agnostic
            ts = self._model_global_mean(mdata.data, model)  # HEALPix or latlon
            # ...
        obs = self._load_obs_var("tas")                      # Sign-convention aware
        return {"model_ts": ..., "obs_ts": ...}

    def plot(self, results):
        fig, ax = plt.subplots()
        meta = self._build_metadata(title="...", figure_id="...", models=[...])
        return [(fig, meta)]
```

Then import the module in `feather/diag/__init__.py` so `@register` fires. See `NEW_DIAGNOSTIC_SPEC.md` for the full implementation guide.

## Testing

```bash
conda activate feather

# Unit tests (synthetic data, no real data needed)
pytest tests/ -v -m "not integration"

# Integration tests (requires data access on DKRZ Levante)
pytest tests/ -v -m "integration"

# All tests
pytest tests/ -v
```

1372 tests (1367 unit + 5 integration) across 30 test files.

## Requirements

Python 3.10+. See `environment.yml` for the full dependency list, or install via `conda env create -f environment.yml`.

Key dependencies: xarray, dask, numpy, scipy, matplotlib, cartopy, healpy, nereus, gsw, cfgrib, pydantic, google-genai, openai.
