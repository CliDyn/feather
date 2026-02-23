# Feather — New Diagnostic Implementation Specification

This document provides everything needed to implement a new diagnostic for the Feather climate model evaluation framework. It serves as a complete reference for AI-assisted implementation, covering architecture, data handling, plotting, CMIP6 integration, LLM analysis, website/report integration, testing, and known pitfalls.

---

## 1. Architecture Overview

### 1.1 What Feather Does

Feather evaluates three high-resolution (~5 km) DestinE coupled climate models — **IFS-FESOM**, **IFS-NEMO**, **ICON** — against observations (ERA5, CERES, EN4, etc.) on native HEALPix grids. Optionally, it includes CMIP6 multi-model mean (MMM) as a conventional-resolution (~100 km) baseline for context.

The pipeline has 4 stages:
1. **Diagnostics** — compute and plot figures with JSON metadata sidecars
2. **Analyze** — Gemini LLM analyzes each figure → structured JSON analysis
3. **Report** — OpenAI curates figures + writes LaTeX report
4. **Website** — static HTML dashboard with figures + LLM analyses

### 1.2 Diagnostic Lifecycle

```
@register decorator → registered in _REGISTRY at import time
                 ↓
Pipeline discovers via registered_names() / get_diagnostic(name)
                 ↓
Constructor: cls(model_loader, obs_loader, config, cmip6_loader=..., **kwargs)
                 ↓
diag.run(skip_existing=True)  ← the main entry point
    ├── _compute_variable(var) / _compute_single(var) / _compute_budget()
    ├── _plot_variable(var, result) / _plot_single(var, result)
    └── _save(fig, meta, figure_id)  → (png_path, json_path)
                 ↓
Returns list[tuple[Path, Path]]  (png + json pairs)
```

### 1.3 File Layout

```
feather/diag/my_diagnostic.py     ← New diagnostic module
feather/diag/__init__.py          ← Add import for @register
tests/test_my_diagnostic.py       ← Unit tests
```

---

## 2. Step-by-Step Implementation

### 2.1 Create the Diagnostic Class

**File:** `feather/diag/my_diagnostic.py`

```python
"""My new diagnostic — brief description."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from feather.data.loader import DataLoader
from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import CMIP6_COLOR, MODEL_COLORS, OBS_COLOR
from feather.util.spatial import global_mean, latlon_global_mean
from feather.util.temporal import climatology

logger = logging.getLogger(__name__)


@register
class MyDiagnostic(DiagnosticBase):
    """Brief description of this diagnostic."""

    name = "my_diagnostic"       # Unique ID — used in filenames, CLI, registry
    title = "My Diagnostic"      # Human-readable — used in plot titles
    domain = "sfc"               # "sfc", "o2d", "pl", "o3d"
    variables = ["avg_2t"]       # Model variable names from VARIABLE_REGISTRY
    group = "evaluation"         # Dashboard nav group

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
```

**Required class attributes:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Unique machine ID. Used in filenames, CLI `--diagnostics` flag, registry lookup. Must be non-empty. |
| `title` | `str` | Human-readable title for plot suptitles. |
| `domain` | `str` | Data domain: `"sfc"` (surface), `"o2d"` (ocean 2D), `"pl"` (pressure levels), `"o3d"` (ocean 3D). Determines which intake catalog key suffix to use. |
| `variables` | `list[str]` | Model variable names this diagnostic uses. Must exist in `VARIABLE_REGISTRY`. |
| `group` | `str` | Thematic group for the web dashboard sidebar. Existing groups: `"evaluation"`, `"radiation"`, `"temperature"`, `"ocean_surface"`, `"sea_ice"`, `"circulation"`, `"wind"`, `"clouds"`, `"precipitation"`, `"moisture"`, `"surface_fluxes"`. |

**Required constructor parameters:**

| Parameter | Purpose |
|-----------|---------|
| `cmip6_loader` | Keyword-only. Passed by pipeline. `None` when CMIP6 disabled. |
| `variables` | Optional override to restrict variables. Pipeline passes this when user uses `--variables`. |
| `cmip6_individual` | When `True`, plot individual CMIP6 models + MMM. Pipeline passes this via `inspect.signature()` check. |

### 2.2 Implement the `run()` Method (Per-Variable Incremental)

Override `run()` for per-variable incremental saving. This is the recommended pattern — it saves progress after each variable so a crash on variable N doesn't lose variables 1..N-1.

```python
def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
    logger.info("Running diagnostic: %s", self.name)
    saved: list[tuple[Path, Path]] = []

    for var in self.variables:
        figure_id = f"{var}_my_plot"

        if skip_existing and self._figure_exists(figure_id):
            logger.info("Skipping %s — figure exists", var)
            saved.append((
                self.output_dir / f"{figure_id}.png",
                self.output_dir / f"{figure_id}.json",
            ))
            continue

        result = self._compute_single(var)
        figures = self._plot_single(var, result)
        for fig, meta in figures:
            paths = self._save(fig, meta, meta["figure_id"])
            saved.append(paths)

    logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
    return saved
```

**Alternative: per-figure-group incremental** (used by `RadiationBudget`):
```python
def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
    saved = []

    # Group A
    fid = "my_bars"
    if skip_existing and self._figure_exists(fid):
        saved.append((self.output_dir / f"{fid}.png", self.output_dir / f"{fid}.json"))
    else:
        results = self._compute_bars()
        for fig, meta in self._plot_bars(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))

    # Group B
    # ... etc.

    return saved
```

### 2.3 Implement `compute()` and `plot()` (Backward Compatibility)

These are required by the ABC but in practice only used by the base `run()` (which you override). Keep them as thin wrappers:

```python
def compute(self) -> dict[str, Any]:
    results = {}
    for var in self.variables:
        results[var] = self._compute_single(var)
    return results

def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
    figures = []
    for var, vr in results.items():
        figures.extend(self._plot_single(var, vr))
    return figures
```

### 2.4 Implement `_compute_single()` — The Computation Core

```python
def _compute_single(self, var: str) -> dict[str, Any]:
    var_info = get_var(var)
    logger.info("Processing variable: %s (%s)", var, var_info.long_name)
    model_data = {}

    # ── Load DestinE models ──
    for model in self.config.models:
        key = DataLoader.make_key(self.experiment, model, var_info.domain)
        try:
            da = self.model_loader.load_var(key, var)
        except KeyError:
            logger.warning("Variable %s not available for %s — skipping", var, model)
            continue

        # Time slicing
        if self.period and "time" in da.dims:
            da = da.sel(time=slice(self.period[0], self.period[1]))

        # Compute your quantity (calls .compute() to materialize dask arrays)
        result = global_mean(da).compute()
        model_data[model] = result

    # ── Load observations ──
    obs = self.obs_loader.load_for_model_var(var, self.period)
    obs_result = latlon_global_mean(obs)

    # ── Load CMIP6 (optional) ──
    cmip6_result = None
    cmip6_info = {}
    cmip6_result, cmip6_info = self._cmip6_global_mean_timeseries(
        var, period=self.period,
        return_individual=self.cmip6_individual,
    )

    return {
        "models": model_data,
        "obs": obs_result,
        "var_info": var_info,
        "cmip6_result": cmip6_result,
        "cmip6_info": cmip6_info,
    }
```

### 2.5 Implement `_plot_single()` — The Plotting Core

```python
def _plot_single(self, var: str, vr: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
    var_info = vr["var_info"]
    all_models = list(self.config.models)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot DestinE models
    for model, data in vr["models"].items():
        color = MODEL_COLORS.get(model)
        ax.plot(data.time.values, data.values, label=model, color=color, linewidth=2.0)

    # Plot CMIP6 MMM (dashed gray)
    if vr.get("cmip6_result") is not None:
        ax.plot(..., label="CMIP6 MMM", color=CMIP6_COLOR, linestyle="--", linewidth=2.0)

    # Plot observations (black, on top)
    obs = vr["obs"]
    ax.plot(obs.time.values, obs.values, label="Obs", color=OBS_COLOR, linewidth=2.5)

    ax.set_title(f"{var_info.long_name} — My Title")
    ax.set_ylabel(f"{var_info.long_name} ({var_info.units})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Build metadata sidecar
    meta = self._build_metadata(
        title=f"{var_info.long_name} My Title",
        figure_id=f"{var}_my_plot",
        models=all_models,
        variables=[var],
        description=f"Description of what this figure shows for {var_info.long_name}.",
        plot_type="timeseries",        # or "bias_map", "seasonal_cycle", "budget_bars", "gregory", "combined_bias_map"
        period=self.period,
        cmip6_info=vr.get("cmip6_info") or None,
        summary_statistics={},         # Optional computed stats dict
    )
    return [(fig, meta)]
```

### 2.6 Register the Diagnostic

**File:** `feather/diag/__init__.py` — add an import so `@register` fires:

```python
import feather.diag.my_diagnostic  # noqa: F401
```

That's it. The pipeline will auto-discover it.

---

## 3. Data Access Patterns

### 3.1 Model Data (DestinE on HEALPix)

```python
from feather.data.loader import DataLoader

# Build catalog key: "{experiment}_2_{model}_{member}_0001_clmn_{high|standard}_{domain}"
key = DataLoader.make_key(experiment, model, domain)
# domain="sfc"|"o2d" → "clmn_high" (nside=1024, 12.6M cells)
# domain="pl"|"o3d"  → "clmn_standard" (nside=128, 196K cells)

ds = self.model_loader.load(key)        # Full dataset
da = self.model_loader.load_var(key, var)  # Single variable

# Dataset structure:
# dims: (time, values) — values is the 1D HEALPix dimension
# coords: longitude(values), latitude(values), time
# No area variable — HEALPix cells are equal area
```

**Critical: dask arrays.** Model data is dask-backed. After reductions (climatology, mean), call `.compute()` before passing to numpy ops or storing:
```python
clim = climatology(da, period).compute()  # ← .compute() materializes
```

### 3.2 Observation Data

```python
# Via VARIABLE_REGISTRY mapping (recommended):
obs_da = self.obs_loader.load_for_model_var("avg_2t", period)
# Automatically applies obs_unit_factor and obs_unit_offset

# Direct access (for datasets not in VARIABLE_REGISTRY):
da = self.obs_loader.load("ERA5", "t2m", period=("1990", "2014"))

# CERES (special — multi-variable files):
da = self.obs_loader.load_ceres("toa_net_all_mon", period=period, file_key="toa")
# file_key: "toa" or "surface" — references CERES_EBAF config
```

**ERA5 gotchas:**
- File keys (config) differ from NetCDF internal names: `"t2m"` in config → `"T2M"` inside file
- `ObsLoader._find_variable()` handles this via case-insensitive fallback
- Radiation/flux vars are **daily accumulations** (J/m\u00b2/day), not W/m\u00b2. Conversion handled by `obs_unit_factor` in `VARIABLE_REGISTRY`
- Precipitation is m/day → multiply by 1000/86400 for kg/m\u00b2/s (also in registry)
- Cloud cover is 0-1 fraction; DestinE is 0-100% (factor=100 in registry)
- Sign convention: positive downward (same as DestinE/IFS)

### 3.3 CMIP6 Data (Optional)

Always guard CMIP6 code with `self.cmip6_enabled`:

```python
if self.cmip6_enabled:
    # Multi-model mean (time-averaged 2D field):
    mmm, info = self.cmip6_loader.load_mmm_for_model_var(
        var, period=self.period, season="DJF",  # season optional
    )
    # mmm: DataArray on regular lat/lon grid, or None
    # info: {"n_members": N, "models_used": [...], "models_skipped": [...]}

    # Individual model (time-averaged):
    da = self.cmip6_loader.load_var_for_model_var(
        var, model, variant="r1i1p1f1", period=self.period,
    )

    # Raw monthly time series (for timeseries/seasonal cycle):
    da = self.cmip6_loader.load_var(
        "tas", model, table="Amon", period=self.period, time_mean=False,
    )

    # Area weights:
    area = self.cmip6_loader.load_area(model, table="Amon")
    area = self._align_area(da, area)  # Inherited from DiagnosticBase
```

**CMIP6 gotchas:**
- `load_var()` returns `None` for missing data — always check!
- Calendar normalization (360_day, noleap, standard) happens inside the loader
- Sea ice `siconc`: auto-normalized from % to fraction if needed
- Sign conventions differ: `hfss`/`hfls` are positive **upward** (opposite to DestinE)
- `rsut`/`rlut` are outgoing components only, not net fluxes
- Net TOA must be derived: `rsdt - rsut - rlut`
- Area weighting: use `areacella`/`areacello`, NOT cos(lat)
- The `_align_area()` helper on `DiagnosticBase` converts areacella to numpy to avoid dim-name misalignment

**Built-in CMIP6 helper on DiagnosticBase:**
```python
# Computes CMIP6 ensemble-mean global-mean monthly time series
cmip6_ts, info = self._cmip6_global_mean_timeseries(
    var, period=self.period, return_individual=True,
)
# cmip6_ts: DataArray with time dim (MMM), or None
# info["individual_series"]: dict[model_name → DataArray]
```

---

## 4. Spatial Computation Patterns

### 4.1 Global Means

```python
from feather.util.spatial import global_mean, latlon_global_mean

# HEALPix model data (equal-area cells → simple .mean()):
gm = global_mean(da)  # Averages over 'values' dim

# Regular lat/lon grid (NEEDS area weighting):
gm = latlon_global_mean(obs_da)  # Uses nr.mesh_from_arrays for cell areas
gm = latlon_global_mean(obs_da, area=precomputed_area)  # Faster with precomputed
```

**CRITICAL: unweighted mean on lat/lon grid is ~8K too cold for temperature.** Always use `latlon_global_mean()`.

**CRITICAL: cos(lat) on HEALPix is ~3K too warm.** Always use simple `.mean()` for equal-area data.

### 4.2 Regridding (HEALPix → Regular Grid for Bias Maps)

```python
import nereus as nr

influence_radius = self.config.nereus.get("influence_radius", 80_000.0)

# Build interpolator ONCE (expensive: ~30s at nside=1024, ~1s reuse):
regridded, interpolator = nr.regrid(
    model_clim.values.ravel(),
    lon=np.asarray(lon), lat=np.asarray(lat),
    resolution=obs_res,  # match obs grid resolution
    influence_radius=influence_radius,
    lon_bounds=(0.0, 360.0),
    as_xarray=True,
)
target_lats = interpolator.target_lat[:, 0]
target_lons = interpolator.target_lon[0, :]

# Reuse for subsequent models/seasons:
regridded_np = interpolator(model_clim_2.values.ravel())
regridded_da = xr.DataArray(
    regridded_np, dims=("lat", "lon"),
    coords={"lat": target_lats, "lon": target_lons},
)

# Regrid obs to same common grid (for consistent bias maps):
obs_common = obs_clim.interp({lat_name: target_lats, lon_name: target_lons})
```

**Influence radius must match source data density:**
- Production (nside=1024, ~5 km): 80,000 m (80 km)
- Tests (nside=8, ~815 km): 1,000,000 m (1000 km)
- CMIP6 (~1-2 deg): 1,000,000 m
- Config: `config.nereus["influence_radius"]` for diagnostics, `config.cmip6["influence_radius"]` for CMIP6

### 4.3 Zonal Means

```python
from feather.util.spatial import zonal_mean

# On native HEALPix (no regridding):
zm = zonal_mean(da.values, lat.values, bin_width=10.0)
# bin_width: 10° for nside=8 tests, 1° for nside=1024 production
```

### 4.4 Temporal Utilities

```python
from feather.util.temporal import (
    climatology,
    seasonal_climatology,
    monthly_climatology,
    annual_mean,
    anomaly,
)

clim = climatology(da, period=("1990", "2014"))       # Time-mean
seasonal = seasonal_climatology(da, period)             # DJF, MAM, JJA, SON
monthly = monthly_climatology(ts, period)               # 12-value Jan-Dec cycle
ann = annual_mean(ts)                                   # Annual means from monthly
anom = anomaly(ts, clim)                               # Deviations from climatology
```

---

## 5. Plotting Conventions

### 5.1 Color Scheme

```python
from feather.plot.styles import MODEL_COLORS, OBS_COLOR, CMIP6_COLOR

# MODEL_COLORS = {"ifs-fesom": "#1f77b4", "ifs-nemo": "#ff7f0e", "icon": "#2ca02c"}
# OBS_COLOR = "black"
# CMIP6_COLOR = "#888888"
```

### 5.2 Layering Order (z-order for line/scatter plots)

The established 4-layer convention:
1. **Background:** Individual CMIP6 models (semi-transparent, thin lines, `alpha=0.2-0.35`, `linewidth=0.5-0.8`)
2. **Middle:** CMIP6 MMM (dashed gray, `linestyle="--"`, `linewidth=1.5-2.0`)
3. **Foreground:** DestinE models (solid colored, `linewidth=2.0`)
4. **Top:** Observations (black, `linewidth=2.5`)

### 5.3 Two-Pass Plotting for Time Series (Monthly + Annual)

```python
# Pass 1: Monthly data as semi-transparent background
for model, ts in model_data.items():
    ax.plot(time, ts.values, color=color, alpha=0.3, linewidth=0.7)

# Pass 2: Annual means as thick foreground with labels
for model, ts in model_data.items():
    ts_annual = annual_mean(ts)
    ax.plot(time, ts_annual.values, label=model, color=color, linewidth=2.0)
```

### 5.4 Map Plotting

**Combined multi-panel bias map (recommended for bias maps):**
```python
from feather.plot.maps import plot_combined_bias_map

fig, axes = plot_combined_bias_map(
    obs_clim,       # 2D xr.DataArray with lat/lon coords
    bias_dict,      # OrderedDict of {label: bias_DataArray}
    title="My Title",
    cmap="RdBu_r",
    bias_cmap="RdBu_r",
    vmin=shared_vmin,        # MUST be shared across models for comparability
    vmax=shared_vmax,
    bias_vmax=shared_bias_max,
    units="K",
)
```

**Shared colorbar ranges are essential.** Compute across all models:
```python
field_arrays = [model_regrid for model_regrid in model_regrids.values()]
field_arrays.append(obs_clim)
vmin = float(np.percentile(all_finite_values, 2))
vmax = float(np.percentile(all_finite_values, 98))
bias_vmax = float(np.percentile(np.abs(all_bias_values), 98))
```

### 5.5 Other Plot Functions

```python
from feather.plot.lines import (
    plot_timeseries,       # Simple time series
    plot_seasonal_cycle,   # 12-month cycle
    plot_zonal_profile,    # Latitude vs variable
    plot_budget_bars,      # Grouped bar chart
    plot_gregory,          # Scatter: T2m vs TOA radiation
)
```

### 5.6 cftime → matplotlib

Matplotlib cannot plot cftime objects directly. Use:
```python
import pandas as pd

def _to_plot_time(time_values):
    if len(time_values) > 0 and hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
```

---

## 6. Metadata Sidecar System

Every figure must have a companion `.json` file. The metadata is the contract between diagnostics → LLM analyzer → website → report.

### 6.1 Building Metadata

```python
meta = self._build_metadata(
    title="Human-readable figure title",
    figure_id="unique_stem_name",           # Used as filename (no extension)
    models=["ifs-fesom", "ifs-nemo", "CMIP6 MMM"],
    variables=["avg_2t"],                   # Variables shown in figure
    description=(
        "Detailed description for the LLM. Be specific about what"
        " the figure shows, how it was computed, and what to look for."
    ),
    computation_notes="How the plotted quantity was derived.",
    plot_type="timeseries",                 # Type identifier
    period=self.period,                     # ("1990", "2014")
    obs_dataset="ERA5",                     # Auto-filled from VARIABLE_REGISTRY
    obs_variable="t2m",                     # Auto-filled from VARIABLE_REGISTRY
    spatial_extent="global",                # or "NH", "SH", "tropics", etc.
    summary_statistics={                    # Computed values for LLM context
        "ifs-fesom": {"global_mean_bias": -0.5, "rmse": 1.2},
    },
    cmip6_info=cmip6_info or None,         # From CMIP6 loader
    extra={"custom_field": "value"},        # Any additional fields
)
```

**Auto-filled from VARIABLE_REGISTRY** (when `variables` is provided): `units`, `domain`, `group`, `obs_dataset`, `obs_variable`, `colormap`.

### 6.2 Saving

```python
# Via _save() helper (inherited):
png_path, json_path = self._save(fig, meta, meta["figure_id"])

# The base saves to: {output_dir}/figures/{diagnostic_name}/{figure_id}.{png|json}
```

### 6.3 Skip-Existing Check

```python
if self._figure_exists(figure_id):
    # Both .png AND .json must exist — partial outputs are regenerated
    ...
```

### 6.4 Figure ID Conventions

Existing patterns (follow these for consistency):
- `{var}_{period}_bias_combined` (global_biases)
- `{var}_timeseries` (timeseries)
- `{var}_seasonal_cycle` (seasonal_cycle)
- `radiation_budget_bars`, `gregory_plot`, `radiation_imbalance_timeseries` (radiation_budget)
- `{derived_key}_annual_bias` (radiation_budget bias maps)

### 6.5 Plot Types Used in Metadata

- `"bias_map"` — 3-panel model/obs/bias
- `"combined_bias_map"` — multi-panel obs + N bias panels
- `"timeseries"` — time series line plot
- `"seasonal_cycle"` — 12-month cycle
- `"budget_bars"` — two-panel: grouped bar chart (left) + zoomed TOA Net (right)
- `"gregory"` — scatter plot with regression

---

## 7. CMIP6 Integration Patterns

### 7.1 Guiding Principle

CMIP6 is **always secondary context**. It must never break a diagnostic when disabled or when data is missing. Every CMIP6 addition is guarded by `self.cmip6_enabled` and `None` checks.

### 7.2 MMM Only vs Individual Models

- **Default (`cmip6_individual=False`):** Show only CMIP6 MMM (dashed gray line / single bias panel)
- **Individual (`cmip6_individual=True`):** Show individual CMIP6 models (semi-transparent background) **plus** MMM (dashed)
- The `--cmip6-individual` CLI flag controls this. Pipeline passes it via `inspect.signature()` check.

### 7.3 CMIP6 for Time Series Diagnostics

```python
# Use the built-in helper on DiagnosticBase:
cmip6_ts, info = self._cmip6_global_mean_timeseries(
    var, period=self.period,
    return_individual=self.cmip6_individual,
)
# Returns (mmm_ts_DataArray, {"n_members": N, "models_used": [...], "individual_series": {...}})
```

### 7.4 CMIP6 for Bias Map Diagnostics

```python
if self.cmip6_enabled:
    mmm, info = self.cmip6_loader.load_mmm_for_model_var(var, period=self.period)
    if mmm is not None:
        cmip6_common = mmm.interp(lat=target_lats, lon=target_lons)
        cmip6_bias = cmip6_common - obs_common
        # Add "CMIP6 MMM" to bias_dict for plot_combined_bias_map
```

### 7.5 CMIP6 for Derived Quantities

When CMIP6 doesn't have a direct variable, derive from components:
```python
# Net TOA = rsdt - rsut - rlut
rsdt, _ = self.cmip6_loader.load_multi_model_mean("rsdt", period=self.period)
rsut, _ = self.cmip6_loader.load_multi_model_mean("rsut", period=self.period)
rlut, _ = self.cmip6_loader.load_multi_model_mean("rlut", period=self.period)
if all(v is not None for v in [rsdt, rsut, rlut]):
    net_toa = rsdt - rsut - rlut
```

### 7.6 CMIP6 Sign Conventions

| Variable | DestinE/ERA5 Convention | CMIP6 Convention |
|----------|------------------------|------------------|
| Surface heat fluxes (`hfss`, `hfls`) | Positive downward | Positive **upward** |
| Outgoing radiation (`rsut`, `rlut`) | Net (positive down) | Outgoing only (positive **upward**) |

Variables with sign mismatches have `cmip6_variable=""` in `VARIABLE_REGISTRY` to prevent incorrect comparison.

---

## 8. Variable Registry Reference

### 8.1 Adding New Variables

If your diagnostic uses variables not yet in `VARIABLE_REGISTRY`, add entries in `feather/data/variables.py`:

```python
"avg_newvar": VarInfo(
    name="avg_newvar",
    long_name="My New Variable",
    units="K",
    domain="sfc",
    cmap="RdBu_r",
    obs_dataset="ERA5",
    obs_variable="newvar",
    cmip6_variable="newvar_cmip6",  # "" if no CMIP6 mapping
    cmip6_table="Amon",             # "" if no CMIP6 mapping
    obs_unit_factor=1.0,            # Multiply obs by this to match model units
    obs_unit_offset=0.0,            # Add after multiplying
    group="temperature",
),
```

### 8.2 Existing Variables (34 total)

**Surface atmospheric (23):** `avg_2t`, `avg_skt`*, `avg_msl`, `avg_10u`, `avg_10v`, `avg_10ws`*, `avg_tcc`, `avg_tcwv`*, `avg_tclw`, `avg_tciw`, `avg_tprate`, `avg_ishf`, `avg_slhtf`, `avg_sdswrf`, `avg_sdlwrf`, `avg_snswrf`, `avg_snlwrf`, `avg_snswrfcs`, `avg_snlwrfcs`, `avg_tnswrf`, `avg_tnlwrf`, `avg_tnswrfcs`, `avg_tnlwrfcs`
(*Note: 18 of these are fully validated for general use. `avg_skt` needs land-correction, `avg_10ws` needs derived-var support, `avg_tcwv` is currently missing an ERA5 file).*

**Ocean 2D (5):** `avg_tos`, `avg_siconc`, `avg_sithick`, `avg_zos`, `avg_sos`

**Ocean 3D (2):** `avg_thetao`, `avg_so`

**Pressure levels (4):** `avg_t`, `avg_u`, `avg_v`, `avg_q`

---

## 9. Pipeline Integration

### 9.1 How Diagnostics Are Discovered and Run

In `feather/run.py` → `_run_diagnostics()`:
1. `registered_names()` gets all diagnostic names
2. Optionally filtered by `--diagnostics` and `--variables` CLI flags
3. For each diagnostic:
   - `get_diagnostic(name)` retrieves the class
   - `inspect.signature(cls.__init__)` checks if it accepts `cmip6_individual`
   - Constructor called with `model_loader, obs_loader, config, **kwargs`
   - `diag.run(skip_existing=...)` executed

### 9.2 CLI Flags Your Diagnostic Inherits Automatically

| Flag | Effect |
|------|--------|
| `--diagnostics my_diagnostic` | Only run your diagnostic |
| `--variables avg_2t avg_msl` | Intersect with your `variables` list |
| `--cmip6-individual` | Passed as kwarg if your `__init__` accepts it |
| `--no-skip-existing` | Forces regeneration of all figures |
| `--experiment baseline_hist` | Passed to constructor |
| `--period 1990 2014` | Passed to constructor |

### 9.3 What You Don't Need to Touch

- `feather/cli.py` — no changes needed (auto-discovers registered diagnostics)
- `feather/run.py` — no changes needed (iterates all registered diagnostics)
- `feather/llm/analyzer.py` — auto-discovers PNG+JSON pairs in `figures/{name}/`
- `feather/website/generator.py` — auto-groups figures by diagnostic name
- `feather/export/report.py` — auto-includes figures with analysis in report curation

---

## 10. LLM Analysis Integration

### 10.1 How It Works (Automatic)

The `FigureAnalyzer` scans `{output_dir}/figures/` for all PNG+JSON pairs, sends each figure image + metadata to Gemini, and saves structured analysis to `{output_dir}/analysis/{diagnostic_name}/{figure_id}.json`.

**You don't need to do anything special** — as long as your figures are saved with proper metadata sidecars via `_save()`, they'll be auto-discovered and analyzed.

### 10.2 Making Your Figures LLM-Friendly

The quality of LLM analysis depends on your metadata:

1. **`description`** — Be specific about what the figure shows. "Global mean time series of 2m temperature" is better than "temperature plot".
2. **`computation_notes`** — Explain derived quantities: "Net TOA = TOA net SW + TOA net LW. Positive = energy into system."
3. **`summary_statistics`** — Include computed values: `{"ifs-fesom": {"global_mean_bias": -0.5, "rmse": 1.2}}`
4. **`plot_type`** — Use a recognized type string so the LLM system prompt can tailor its analysis.
5. **`cmip6_info`** — Pass the info dict so the LLM knows about CMIP6 context.

### 10.3 Recognized Plot Types in LLM Prompts

The system prompt (`feather/llm/prompts.py`) describes these figure types:
1. Bias maps (from `global_biases` and `radiation_budget`)
2. Time series
3. Seasonal cycles
4. Radiation budget bars
5. Gregory plot
6. Radiation imbalance time series

**If your diagnostic produces a new plot type**, update `feather/llm/prompts.py` → `_FIGURE_ANALYSIS_SYSTEM` to add a description:
```python
7. **My new plot type** — description of what this plot shows and
what the LLM should focus on when analyzing it.
```

Also update `feather/export/prompts.py` → `CURATION_SYSTEM` to add your diagnostic to the "Available diagnostics" list.

---

## 11. Website Integration

### 11.1 How It Works (Automatic)

`SiteGenerator` collects all figures from `{output_dir}/figures/`, groups them by `diagnostic_name` from the JSON sidecar, and generates HTML pages. Your diagnostic's figures appear automatically.

### 11.2 Navigation Group

Your `group` class attribute controls which section of the sidebar your diagnostic appears in. Use an existing group or create a new one.

---

## 12. Testing

### 12.1 Test File Structure

Create `tests/test_my_diagnostic.py`. Use existing test files as templates — especially `tests/test_timeseries.py` (simplest) or `tests/test_radiation_budget.py` (most complex).

### 12.2 Available Test Fixtures (from `conftest.py`)

| Fixture | Description |
|---------|-------------|
| `synth_healpix` | nside=8, 768 cells, 12 months, temperature gradient pole→equator |
| `synth_obs` | 5\u00b0 regular lat/lon grid, 12 months, matching temperature |
| `synth_cmip6` | 5\u00b0 lat/lon grid, 12 months, `tas` + `areacella` |
| `minimal_config` | `FeatherConfig` with CMIP6 disabled, influence_radius=1M |
| `cmip6_config` | `FeatherConfig` with CMIP6 enabled, 2 fake models |
| `mock_model_loader` | `MockModelLoader` backed by `synth_healpix` |
| `mock_obs_loader` | `MockObsLoader` backed by `synth_obs` |
| `mock_cmip6_loader` | `MockCMIP6Loader` backed by `synth_cmip6` |

### 12.3 Essential Test Categories

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from feather.diag.my_diagnostic import MyDiagnostic


class TestMyDiagnosticCompute:
    """Tests for computation logic."""

    def test_compute_single_returns_expected_keys(
        self, mock_model_loader, mock_obs_loader, minimal_config
    ):
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        result = diag._compute_single("avg_2t")
        assert "models" in result
        assert "obs" in result
        assert "var_info" in result

    def test_compute_skips_missing_variable(
        self, mock_model_loader, mock_obs_loader, minimal_config
    ):
        """Models missing a variable should be skipped with warning."""
        # MockModelLoader returns whatever is in synth_healpix
        # Testing a var not in the dataset would need a custom mock
        pass

    def test_compute_with_cmip6(
        self, mock_model_loader, mock_obs_loader,
        mock_cmip6_loader, cmip6_config
    ):
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["avg_2t"],
        )
        result = diag._compute_single("avg_2t")
        assert result["cmip6_result"] is not None


class TestMyDiagnosticPlot:
    """Tests for plotting."""

    def test_plot_returns_figure_and_metadata(
        self, mock_model_loader, mock_obs_loader, minimal_config
    ):
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        result = diag._compute_single("avg_2t")
        figures = diag._plot_single("avg_2t", result)
        assert len(figures) >= 1
        fig, meta = figures[0]
        assert isinstance(fig, plt.Figure)
        assert "figure_id" in meta
        assert "diagnostic_name" in meta
        plt.close(fig)

    def test_plot_metadata_has_correct_diagnostic_name(
        self, mock_model_loader, mock_obs_loader, minimal_config
    ):
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        result = diag._compute_single("avg_2t")
        figures = diag._plot_single("avg_2t", result)
        _, meta = figures[0]
        assert meta["diagnostic_name"] == "my_diagnostic"


class TestMyDiagnosticRun:
    """Tests for full run orchestration."""

    def test_run_saves_files(
        self, mock_model_loader, mock_obs_loader, minimal_config, tmp_path
    ):
        minimal_config.output_dir = str(tmp_path / "output")
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 1
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()

    def test_run_skip_existing(
        self, mock_model_loader, mock_obs_loader, minimal_config, tmp_path
    ):
        minimal_config.output_dir = str(tmp_path / "output")
        diag = MyDiagnostic(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        # First run creates files
        saved1 = diag.run(skip_existing=False)
        # Second run skips
        saved2 = diag.run(skip_existing=True)
        assert len(saved2) == len(saved1)


class TestMyDiagnosticRegistry:
    """Test that the diagnostic is properly registered."""

    def test_registered(self):
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("my_diagnostic")
        assert cls is MyDiagnostic
```

### 12.4 Testing with Mocked Figures

When testing figure saving with mocked figures:
```python
# MagicMock(spec=plt.Figure).savefig doesn't create files!
# Use this pattern instead:
mock_fig = MagicMock(spec=plt.Figure)
mock_fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
```

### 12.5 Test Tolerances

- NN regridding at nside=8 introduces ~1-3K error vs true field values
- Use generous tolerances: `pytest.approx(expected, abs=5.0)` for temperature
- For relative checks: `assert abs(result - expected) / expected < 0.1`

### 12.6 Registry Isolation

Tests that import diagnostic modules trigger `@register`. Use an `autouse` fixture to save/restore registry state:
```python
@pytest.fixture(autouse=True)
def _clean_registry():
    from feather.diag.registry import _REGISTRY
    backup = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(backup)
```

### 12.7 Running Tests

```bash
conda activate nereus
pytest tests/test_my_diagnostic.py -v          # Just your tests
pytest tests/ -v -m "not integration"           # All unit tests
```

---

## 13. Common Pitfalls and Solutions

### 13.1 Area Weighting Mistakes

| Data Type | Correct Method | Wrong Method | Error |
|-----------|---------------|--------------|-------|
| HEALPix model data | `global_mean(da)` (simple .mean) | cos(lat) weights | ~+3K warm bias |
| Regular lat/lon obs | `latlon_global_mean(da)` | Unweighted .mean | ~-8K cold bias |
| CMIP6 data | `latlon_global_mean(da, area=areacella)` | cos(lat) | Grid-dependent |

### 13.2 Grid Misalignment in Multi-Panel Plots

**Problem:** When using `nr.plot()` with a shared interpolator across panels, the interpolator maps array indices (not coordinates). If Panel 1 data is south→north and Panel 2 is north→south, the shared interpolator flips Panel 2.

**Solution:** Always regrid obs to the common nereus target grid in `compute()`:
```python
obs_common = obs_clim.interp({lat_name: target_lats, lon_name: target_lons})
```

### 13.3 Dask Arrays Not Materialized

**Problem:** Passing dask-backed arrays to numpy operations or storing in results dicts causes memory issues or errors.

**Solution:** Call `.compute()` after reductions:
```python
clim = climatology(da, period).compute()
```

### 13.4 Missing Variables

**Problem:** Not all models have all variables. ICON lacks `avg_tcc`, for example.

**Solution:** Always wrap model loading in try/except:
```python
try:
    da = self.model_loader.load_var(key, var)
except KeyError:
    logger.warning("Variable %s not available for %s — skipping", var, model)
    continue
```

### 13.5 CMIP6 `None` Returns

**Problem:** `load_var()` and `load_mmm_for_model_var()` return `None` for missing data.

**Solution:** Always check before using:
```python
mmm, info = self.cmip6_loader.load_mmm_for_model_var(var, period=self.period)
if mmm is not None:
    # safe to use
```

### 13.6 cftime Objects in Plots

**Problem:** `matplotlib.pyplot.plot()` cannot handle cftime datetime objects.

**Solution:** Convert via string parsing:
```python
pd.to_datetime([str(t) for t in time_values])
```

### 13.7 Interpolator Rebuild Cost

**Problem:** `nr.regrid()` builds a KDTree (~30s at nside=1024). Rebuilding per model/season wastes minutes.

**Solution:** Build once, reuse:
```python
interpolator = None
for model in models:
    if interpolator is None:
        _, interpolator = nr.regrid(...)
    else:
        result = interpolator(data.values.ravel())
```

### 13.8 CERES Loader Error Handling

**Problem:** `obs_loader.load_ceres()` raises `AttributeError` if the mock loader doesn't have it.

**Solution:** Catch `AttributeError` alongside other exceptions:
```python
try:
    da = self.obs_loader.load_ceres(...)
except (KeyError, FileNotFoundError, AttributeError) as e:
    logger.warning("CERES data not available: %s", e)
```

### 13.9 `_figure_exists()` Partial Outputs

The check requires **both** `.png` and `.json`. If only one exists (e.g., crash during save), the variable is reprocessed. This is intentional.

### 13.10 inspect.signature() for Optional kwargs

The pipeline uses `inspect.signature()` to check if a diagnostic's `__init__` accepts `cmip6_individual` before passing it:
```python
import inspect
sig = inspect.signature(cls.__init__)
if "cmip6_individual" in sig.parameters:
    kwargs["cmip6_individual"] = True
```
Always accept `cmip6_individual` as a keyword argument if your diagnostic should support it.

---

## 14. Complete Checklist

- [ ] **Create** `feather/diag/my_diagnostic.py` with `@register`-decorated class
- [ ] **Set** class attributes: `name`, `title`, `domain`, `variables`, `group`
- [ ] **Accept** `cmip6_loader`, `variables`, `cmip6_individual` in `__init__`
- [ ] **Override** `run()` with per-variable incremental saving
- [ ] **Implement** `compute()` and `plot()` as thin wrappers
- [ ] **Implement** `_compute_single()` / `_compute_variable()` with:
  - DestinE model loading with `try/except KeyError`
  - Obs loading via `obs_loader.load_for_model_var()` or `load_ceres()`
  - CMIP6 loading guarded by `self.cmip6_enabled`
  - `.compute()` calls on dask arrays after reductions
- [ ] **Implement** `_plot_single()` / `_plot_variable()` with:
  - 4-layer z-ordering (CMIP6 individual → MMM → DestinE → Obs)
  - Consistent color scheme from `plot/styles.py`
  - Shared colorbar ranges for cross-model comparison (bias maps)
  - Proper `_build_metadata()` call with descriptive fields
- [ ] **Add** import to `feather/diag/__init__.py`
- [ ] **Add** new variables to `VARIABLE_REGISTRY` if needed
- [ ] **Update** `feather/llm/prompts.py` if introducing a new plot type
- [ ] **Update** `feather/export/prompts.py` to mention new diagnostic
- [ ] **Create** `tests/test_my_diagnostic.py` with:
  - Compute tests (expected keys, missing variable handling, CMIP6)
  - Plot tests (figure type, metadata fields)
  - Run tests (file creation, skip-existing)
  - Registry test
- [ ] **Run** tests: `pytest tests/test_my_diagnostic.py -v`
- [ ] **Run** full suite: `pytest tests/ -v -m "not integration"`
- [ ] **Verify** test count increase in CLAUDE.md

---

## 15. Reference: Existing Diagnostics

| Diagnostic | File | Lines | Figure Types | Saving Pattern |
|-----------|------|-------|--------------|----------------|
| `global_biases` | `diag/global_biases.py` | ~672 | Combined bias maps (per var per period) | Per-variable (3 figures per var) |
| `timeseries` | `diag/timeseries.py` | ~305 | Global-mean time series | Per-variable (1 figure per var) |
| `seasonal_cycle` | `diag/seasonal_cycle.py` | ~261 | 12-month cycle | Per-variable (1 figure per var) |
| `radiation_budget` | `diag/radiation_budget.py` | ~1107 | Budget bars, Gregory, imbalance TS, bias maps | Per-figure-group |

Use `timeseries.py` as the simplest template for line-plot diagnostics.
Use `global_biases.py` as the template for bias-map diagnostics.
Use `radiation_budget.py` as the template for complex multi-figure-type diagnostics.
