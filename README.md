# Feather

Lightweight climate model evaluation framework for [DestinE](https://destination-earth.eu/) high-resolution simulations.

Feather compares three climate models (IFS-FESOM, IFS-NEMO, ICON) running on HEALPix grids at ~5 km resolution against observations (ERA5, CERES, EN4, etc.) and a CMIP6 multi-model mean. It generates diagnostic figures, LLM-analyzed results, a LaTeX report, and a static web dashboard.

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
feather -v
feather --config configs/default.yaml -v
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

# Filter by variable
feather --variables avg_2t -v

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

cfg = FeatherConfig.from_yaml("configs/default.yaml")

# Full pipeline
result = run_pipeline(cfg, steps="all", api_key="...", openai_api_key="...")

# Individual steps
result = run_pipeline(cfg, steps=["diagnostics"], diagnostics=["global_biases"])
result = run_pipeline(cfg, steps=["report"], openai_api_key="...")

# Result dict
print(result)
# {"figures": 12, "analyses": 12, "syntheses": 3, "report": Path(...), "site_dir": Path(...)}
```

Or use classes directly:

```python
from feather import FeatherConfig, DataLoader, ObsLoader
from feather.diag.global_biases import GlobalBiases

cfg = FeatherConfig.from_yaml("configs/default.yaml")
model_loader = DataLoader.from_catalog(cfg.model_catalogs["2d"])
obs_loader = ObsLoader(cfg)

diag = GlobalBiases(model_loader, obs_loader, cfg, variables=["avg_2t"])
saved_files = diag.run()  # returns [(png_path, json_path), ...]
```

## Available diagnostics

| Diagnostic | Class | What it produces |
|---|---|---|
| `global_biases` | `GlobalBiases` | 3-panel bias maps (Model \| Obs \| Bias) per model, annual + DJF/JJA |
| `timeseries` | `TimeseriesDiag` | Global-mean time series, all models + obs overlaid |
| `seasonal_cycle` | `SeasonalCycleDiag` | 12-month climatological cycle, all models + obs overlaid |

All diagnostics accept these constructor arguments:

- `variables=["avg_2t", ...]` — which model variables to evaluate (default: `["avg_2t"]`)
- `experiment="baseline_hist"` — catalog experiment key
- `period=("1990", "2014")` — time period for climatologies

## Output structure

```
output/
  figures/                          # Diagnostic figures + JSON metadata
    global_biases/
      avg_2t_annual_bias_ifs-fesom.png
      avg_2t_annual_bias_ifs-fesom.json
      ...
    timeseries/
    seasonal_cycle/
  analysis/                         # LLM analysis (Gemini)
    global_biases/
      avg_2t_annual_bias_ifs-fesom_analysis.json
      synthesis.json
    ...
  publication/                      # LaTeX report (OpenAI)
    structure.json                  # Stage 1: editorial curation
    sections/                       # Stage 2: per-section prose
      01_temperature.json
      ...
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
  config.py              # FeatherConfig (YAML loading)
  run.py                 # Pipeline orchestration (run_pipeline)
  __main__.py            # python -m feather support
  data/
    loader.py            # DataLoader (intake catalogs + file paths)
    obs.py               # ObsLoader (observations)
    cmip6.py             # CMIP6Loader (multi-model mean from zarr)
    variables.py         # VARIABLE_REGISTRY (27 variables)
  util/
    spatial.py           # Zonal/global means, NN regridding
    temporal.py          # Climatology, seasonal/monthly grouping
    units.py             # Unit conversions
  plot/
    maps.py              # Bias maps (nereus), single maps
    lines.py             # Time series, seasonal cycle, zonal profiles
    styles.py            # Model colors, plot defaults
  diag/
    base.py              # DiagnosticBase ABC
    registry.py          # @register decorator
    figure_meta.py       # Figure metadata sidecar system
    global_biases.py     # Bias map diagnostic
    timeseries.py        # Time series diagnostic
    seasonal_cycle.py    # Seasonal cycle diagnostic
  llm/
    analyzer.py          # FigureAnalyzer (Gemini)
    schemas.py           # FigureAnalysis, DiagnosticSynthesis
    prompts.py           # Gemini prompts
  export/
    report.py            # ReportGenerator (OpenAI, 3-stage)
    schemas.py           # ReportStructure, WrittenSection
    prompts.py           # OpenAI prompts (curation + writing)
    openai_client.py     # Thin OpenAI wrapper with retry
    latex_builder.py     # LaTeX escaping, template rendering, PDF
    templates/           # Jinja2 LaTeX template
  website/
    generator.py         # SiteGenerator (static HTML)
    templates/           # Jinja2 HTML templates
    static/              # CSS
```

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

284 unit tests + 4 integration tests.

## Configuration

Default config at `configs/default.yaml` points to Levante data paths:

- **Model data:** intake catalogs at `/work/ab0995/a270088/DestinE/GENERATION2_joint/{2D,3D}/`
- **Observations:** ERA5, CERES, EN4, ESA-CCI, MSWEP, OSI-SAF at `/work/bb1153/b382289/data/aqua-dvc/datasets/`
- **CMIP6:** 12 models from `/work/ab0995/a270088/DestinE/cmip6/zarr/`
- **LLM analysis:** Gemini 2.5 Flash (`GEMINI_API_KEY`)
- **Report generation:** GPT-4o (`OPENAI_API_KEY`)
- **Output:** `./output/` (configurable)

## Adding a new diagnostic

```python
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register

@register
class MyDiagnostic(DiagnosticBase):
    name = "my_diagnostic"
    title = "My Diagnostic"
    domain = "sfc"
    variables = ["avg_2t"]
    group = "temperature"

    def compute(self):
        # Load data, return results dict
        return {"model_clim": ..., "obs_clim": ...}

    def plot(self, results):
        # Create figures, return [(fig, metadata_dict), ...]
        fig, ax = plt.subplots()
        meta = self._build_metadata(title="...", figure_id="...", models=[...])
        return [(fig, meta)]
```

Then import the module in `feather/diag/__init__.py` so `@register` fires.

## Requirements

Python 3.10+. Core dependencies: xarray, dask, numpy, scipy, matplotlib, cartopy, intake, healpy, nereus, pyyaml, zarr, pydantic, google-generativeai, openai, jinja2.

Development: `conda activate nereus` on DKRZ Levante provides everything.
