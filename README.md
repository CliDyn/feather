# Feather

Lightweight climate model evaluation framework for [DestinE](https://destination-earth.eu/) high-resolution simulations.

Feather compares three climate models (IFS-FESOM, IFS-NEMO, ICON) running on HEALPix grids at ~5 km resolution against observations (ERA5, CERES, EN4, etc.). It generates diagnostic figures with JSON metadata sidecars, designed to feed into an LLM analysis pipeline and a static web dashboard.

## Quick start

```bash
# On DKRZ Levante
conda activate nereus
cd /home/a/a270088/PYTHON/feather/feather
pip install -e .
```

## Running diagnostics

Each diagnostic follows the same pattern: **compute** (load data, calculate statistics) then **plot** (generate figures with metadata).

```python
from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.obs import ObsLoader
from feather.diag.global_biases import GlobalBiases
from feather.diag.timeseries import TimeseriesDiag
from feather.diag.seasonal_cycle import SeasonalCycleDiag

# Load config
cfg = FeatherConfig.from_yaml("configs/default.yaml")

# Set up data loaders
model_loader = DataLoader.from_catalog(cfg.model_catalogs["2d"])
obs_loader = ObsLoader(cfg)

# Run a diagnostic (compute + plot + save)
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

## Output

Figures are saved to `{output_dir}/figures/{diagnostic_name}/` with paired JSON metadata:

```
output/figures/global_biases/
  avg_2t_annual_bias_ifs-fesom.png
  avg_2t_annual_bias_ifs-fesom.json   # metadata sidecar
  avg_2t_djf_bias_ifs-fesom.png
  avg_2t_djf_bias_ifs-fesom.json
  ...
```

The JSON sidecar contains variable info, units, obs dataset, period, summary statistics (global mean bias, RMSE), and is the contract for downstream LLM analysis and the dashboard.

## Project structure

```
feather/
  config.py              # FeatherConfig (YAML loading)
  data/
    loader.py            # DataLoader (intake catalogs + file paths)
    obs.py               # ObsLoader (observations)
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

97 tests (95 unit + 2 integration).

## Configuration

Default config at `configs/default.yaml` points to Levante data paths:

- **Model data:** intake catalogs at `/work/ab0995/a270088/DestinE/GENERATION2_joint/{2D,3D}/`
- **Observations:** ERA5, CERES, EN4, ESA-CCI, MSWEP, OSI-SAF at `/work/bb1153/b382289/data/aqua-dvc/datasets/`
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

Python 3.10+. Core dependencies: xarray, dask, numpy, scipy, matplotlib, cartopy, intake, healpy, nereus, pyyaml, zarr.

Development: `conda activate nereus` on DKRZ Levante provides everything.
