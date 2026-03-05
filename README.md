# Feather

Lightweight climate model evaluation framework supporting multiple high-resolution model sets.

Feather compares high-resolution climate models against observations (ERA5, CERES, EN4, ESA-CCI, OSI-SAF, PIOMAS/GIOMAS) and a CMIP6 multi-model mean. It generates diagnostic figures, LLM-analyzed results, a LaTeX report, and a static web dashboard.

Currently supported model sets:
- **DestinE**: IFS-FESOM, IFS-NEMO, ICON (~5 km, HEALPix grids, intake catalogs)
- **EERIE HighResMIP**: IFS-FESOM2-SR, IFS-NEMO-ER, ICON-ESM-ER (~10 km atm / ~5-10 km ocean, 0.25° regular lat/lon, CMOR directory tree)

The framework is **grid-agnostic**: diagnostics automatically dispatch between HEALPix and regular lat/lon grids based on per-model configuration.

## Quick start

```bash
# On DKRZ Levante
conda activate nereus
cd /home/a/a270088/PYTHON/feather/feather
pip install -e .
```

After `pip install -e .`, the `feather` command is available.

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

# EERIE
feather --config configs/eerie.yaml -v
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

### Common options

```bash
# Filter diagnostics by name
feather --diagnostics global_biases timeseries -v

# Filter by variable (CMOR canonical names)
feather --variables tas pr -v

# Include individual CMIP6 model lines/biases alongside MMM
feather --cmip6-individual -v

# Custom output directory
feather --output ./my_output -v

# Re-run everything (ignore cached results)
feather --no-skip-existing -v

# Compile LaTeX report to PDF
feather --steps report --compile-pdf -v

# Also works as a module
python -m feather --steps report -v
```

## Python API

```python
from feather import FeatherConfig, run_pipeline

# Load any config (DestinE or EERIE)
cfg = FeatherConfig.from_yaml("configs/eerie.yaml")

# Full pipeline
result = run_pipeline(cfg, steps="all", api_key="...", openai_api_key="...")

# Individual steps
result = run_pipeline(cfg, steps=["diagnostics"], diagnostics=["global_biases"])
result = run_pipeline(cfg, steps=["report"], openai_api_key="...")

# Result dict
print(result)
# {"figures": 12, "analyses": 12, "syntheses": 3, "report": Path(...), "site_dir": Path(...)}
```

## Available diagnostics

| Diagnostic | Class | What it produces |
|---|---|---|
| `global_biases` | `GlobalBiases` | Combined multi-panel bias maps (Obs + all model biases), annual + DJF/JJA |
| `timeseries` | `TimeseriesDiag` | Global-mean time series, all models + obs overlaid |
| `seasonal_cycle` | `SeasonalCycleDiag` | 12-month climatological cycle, all models + obs overlaid |
| `radiation_budget` | `RadiationBudget` | TOA/surface radiation budget bars, Gregory plot, imbalance time series, bias maps vs CERES EBAF |
| `sea_ice` | `SeaIceDiag` | Sea ice area/extent/volume time series, seasonal cycles, trends, polar spatial maps vs OSI-SAF + PIOMAS/GIOMAS |
| `ocean_sst` | `OceanSST` | SST evaluation vs ESA-CCI: bias maps, time series, seasonal cycle, zonal mean |
| `ocean_en4` | `OceanEN4` | 3D ocean T/S evaluation vs EN4: surface bias maps, Hovmoller diagrams, depth-layer time series |
| `global_trends` | `GlobalTrends` | Per-grid-point linear trends over the evaluation period |

All diagnostics accept these constructor arguments:

- `variables=["tas", ...]` — which variables to evaluate (CMOR canonical names)
- `experiment="hist-1950"` — experiment identifier
- `period=("1980", "2014")` — time period for climatologies (read from config by default)

## Configuration

Feather uses YAML configuration files. Two configs are provided:

### DestinE (`configs/default.yaml`)

```yaml
project:
  name: "DestinE"
  resolution: "high-resolution (~5 km)"
  experiment: "baseline_hist"
  period: ["1990", "2014"]

data_source:
  type: "destine_catalog"

models: [ifs-fesom, ifs-nemo, icon]   # List format → HEALPix grid defaults
```

- **Model data:** intake catalogs (HEALPix grids, zarr format)
- **Grid:** nside=1024 (12.6M cells), 1D `values` dimension

### EERIE (`configs/eerie.yaml`)

```yaml
project:
  name: "EERIE"
  resolution: "high-resolution (~10 km atm, ~5-10 km ocean)"
  experiment: "hist-1950"
  period: ["1980", "2014"]

data_source:
  type: "cmor"
  root: "/work/bm1344/DKRZ/CMOR/EERIE/HighResMIP"

models:                              # Dict format → per-model config
  IFS-FESOM2-SR:
    institution: AWI
    experiment: hist-1950
    variant: r1i1p1f1
    grids:
      sfc: latlon
      o2d: latlon
      o3d: latlon
    color: "#1f77b4"
  IFS-NEMO-ER:
    institution: BSC
    # ...
```

- **Model data:** CMOR directory tree (NetCDF, standard dims)
- **Grid:** 0.25° regular lat/lon (721x1440), standard `(time, lat, lon)` dims

### Common config sections

Both configs share:
- **Observations:** ERA5, CERES, EN4, ESA-CCI, OSI-SAF at `/work/bb1153/b382289/data/aqua-dvc/datasets/`
- **CMIP6:** 12 models from `/work/ab0995/a270088/DestinE/cmip6/zarr/`
- **LLM analysis:** Gemini (`GEMINI_API_KEY` / `VERTEX_API_KEY`)
- **Report generation:** OpenAI (`OPENAI_API_KEY`)
- **Output:** configurable via `output_dir`

## Adding a new model set

1. Create a config YAML using the structured format (see `configs/eerie.yaml` as template)
2. Set `data_source.type` to `"cmor"` (or `"destine_catalog"` for intake catalogs)
3. Define `models` as a dict with per-model `institution`, `experiment`, `variant`, `grids`, `color`
4. Set `project.name`, `project.period`, `project.experiment`, `project.resolution`
5. Run: `feather --config configs/my_project.yaml -v`

If your data format is not CMOR or intake catalogs, implement a new loader class (see `CMORLoader` in `feather/data/cmor_loader.py` as template).

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
    loader.py            # DataLoader (intake catalogs, DestinE)
    cmor_loader.py       # CMORLoader (CMOR directory tree, EERIE)
    obs.py               # ObsLoader (observations)
    cmip6.py             # CMIP6Loader (multi-model mean from zarr)
    variables.py         # VARIABLE_REGISTRY (33 vars, CMOR canonical names)
  util/
    spatial.py           # Zonal/global means, NN regridding, latlon areas
    temporal.py          # Climatology, seasonal/monthly grouping
    units.py             # Unit conversions
  plot/
    maps.py              # Bias maps (nereus), combined multi-panel maps
    lines.py             # Time series, seasonal cycle, zonal profiles
    styles.py            # Model colors, plot defaults
  diag/
    base.py              # DiagnosticBase ABC + grid-agnostic helpers
    registry.py          # @register decorator
    figure_meta.py       # Figure metadata sidecar system
    global_biases.py     # Climatology bias maps
    timeseries.py        # Global-mean time series
    seasonal_cycle.py    # Monthly climatological cycle
    radiation_budget.py  # Radiation budget (CERES EBAF)
    sea_ice.py           # Sea ice (OSI-SAF, PIOMAS/GIOMAS)
    ocean_sst.py         # SST evaluation (ESA-CCI)
    ocean_en4.py         # 3D ocean T/S (EN4 v4.2.2)
    global_trends.py     # Per-grid-point linear trends
  llm/
    analyzer.py          # FigureAnalyzer (Gemini)
    schemas.py           # FigureAnalysis, DiagnosticSynthesis
    prompts.py           # Gemini prompts (config-parameterized)
  export/
    report.py            # ReportGenerator (OpenAI, 3-stage)
    schemas.py           # ReportStructure, WrittenSection
    prompts.py           # OpenAI prompts (config-parameterized)
    openai_client.py     # Thin OpenAI wrapper with retry
    latex_builder.py     # LaTeX escaping, template rendering, PDF
    templates/           # Jinja2 LaTeX template
  website/
    generator.py         # SiteGenerator (static HTML)
    templates/           # Jinja2 HTML templates
    static/              # CSS
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
conda activate nereus

# Unit tests (synthetic data, safe on login node)
pytest tests/ -v -m "not integration"

# Integration tests (real data, also safe on login node)
pytest tests/ -v -m "integration"

# All tests
pytest tests/ -v
```

847 unit tests + 4 integration tests.

## Requirements

Python 3.10+. Core dependencies: xarray, dask, numpy, scipy, matplotlib, cartopy, intake, healpy, nereus, pyyaml, zarr, pydantic, google-generativeai, openai, jinja2.

Development: `conda activate nereus` on DKRZ Levante provides everything.
