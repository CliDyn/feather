# Feather — New Diagnostic Implementation Specification

This document provides everything needed to implement a new diagnostic for the Feather climate model evaluation framework. It serves as a complete reference for AI-assisted implementation, covering architecture, data handling, plotting, CMIP6 integration, LLM analysis, website/report integration, testing, and known pitfalls.

---

## 1. Architecture Overview

### 1.1 What Feather Does

Feather evaluates high-resolution coupled climate models against observations (ERA5, CERES, EN4, etc.). It supports multiple model sets:

- **DestinE**: IFS-FESOM, IFS-NEMO, ICON (~5 km, HEALPix grids, intake catalogs)
- **EERIE Ensemble**: IFS-FESOM2-SR, IFS-NEMO-ER, ICON-ESM-ER (~10 km atm / ~5-10 km ocean, 0.25° lat/lon, CMOR directory tree)

The framework is **grid-agnostic**: diagnostics automatically dispatch between HEALPix and regular lat/lon grids based on per-model configuration. Optionally, it includes CMIP6 multi-model mean (MMM) as a conventional-resolution (~100 km) baseline for context.

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

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.temporal import climatology

logger = logging.getLogger(__name__)


@register
class MyDiagnostic(DiagnosticBase):
    """Brief description of this diagnostic."""

    name = "my_diagnostic"       # Unique ID — used in filenames, CLI, registry
    title = "My Diagnostic"      # Human-readable — used in plot titles
    domain = "sfc"               # "sfc", "o2d", "pl", "o3d"
    variables = ["tas"]          # CMOR canonical names from VARIABLE_REGISTRY
    group = "evaluation"         # Dashboard nav group

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment=None, period=None,
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment or config.get_experiment()
        self.period = period or config.get_period()
        self.cmip6_individual = cmip6_individual
```

**Required class attributes:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Unique machine ID. Used in filenames, CLI `--diagnostics` flag, registry lookup. Must be non-empty. |
| `title` | `str` | Human-readable title for plot suptitles. |
| `domain` | `str` | Data domain: `"sfc"` (surface), `"o2d"` (ocean 2D), `"pl"` (pressure levels), `"o3d"` (ocean 3D). Used for grid-type lookup and catalog key construction. |
| `variables` | `list[str]` | CMOR canonical variable names this diagnostic uses (e.g., `"tas"` not `"avg_2t"`). Must exist in `VARIABLE_REGISTRY`. |
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

Use the grid-agnostic base class helpers instead of direct loader calls. These automatically dispatch between HEALPix and lat/lon grids:

```python
def _compute_single(self, var: str) -> dict[str, Any]:
    var_info = get_var(var)
    logger.info("Processing variable: %s (%s)", var, var_info.long_name)
    model_data = {}

    # ── Load models (grid-agnostic) ──
    for model in self.config.models:
        try:
            mdata = self._load_model_var(model, var)  # Returns ModelData
        except (KeyError, FileNotFoundError):
            logger.warning("Variable %s not available for %s — skipping", var, model)
            continue

        da = mdata.data
        # Time slicing
        if self.period and "time" in da.dims:
            da = da.sel(time=slice(self.period[0], self.period[1]))

        # Grid-agnostic global mean (HEALPix: simple .mean(), latlon: area-weighted)
        result = self._model_global_mean(da, model).compute()
        model_data[model] = result

    # ── Load observations (sign-convention aware) ──
    obs = self._load_obs_var(var)
    if self.period and "time" in obs.dims:
        obs = obs.sel(time=slice(self.period[0], self.period[1]))

    # ── Load CMIP6 (optional) ──
    cmip6_result, cmip6_info = self._cmip6_global_mean_timeseries(
        var, period=self.period,
        return_individual=self.cmip6_individual,
    )

    return {
        "models": model_data,
        "obs": obs,
        "var_info": var_info,
        "cmip6_result": cmip6_result,
        "cmip6_info": cmip6_info,
    }
```

**Key base class helpers:**
- `_load_model_var(model, var)` → `ModelData(data, lon, lat)` — dispatches to intake (DestinE) or CMOR loader (EERIE) based on config
- `_model_global_mean(da, model)` → area-weighted mean — HEALPix uses simple `.mean()`, latlon uses cos(lat) weighting
- `_load_obs_var(var)` → obs DataArray — applies sign convention flip for CMOR data sources (e.g., `hfss`/`hfls`)
- `_load_model_coords(model)` → `(lon, lat)` — coordinate arrays for any grid type

### 2.5 Implement `_plot_single()` — The Plotting Core

```python
def _plot_single(self, var: str, vr: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
    var_info = vr["var_info"]
    all_models = list(self.config.models)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot models (config-driven colors work for any model set)
    for model, data in vr["models"].items():
        color = self.config.get_model_color(model)
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

### 3.1 Model Data (Grid-Agnostic)

**Recommended: use base class helpers** — they dispatch to the correct loader automatically:

```python
# Grid-agnostic loading (works for DestinE HEALPix and EERIE lat/lon):
mdata = self._load_model_var(model, "tas")  # Returns ModelData(data, lon, lat)
da = mdata.data    # xr.DataArray
lon = mdata.lon    # 1D array
lat = mdata.lat    # 1D array

# Grid-agnostic global mean:
gm = self._model_global_mean(da, model)  # HEALPix: .mean(), latlon: cos-weighted

# Grid-agnostic coordinates:
lon, lat = self._load_model_coords(model)
```

**Error handling:** CMOR raises `FileNotFoundError`, DestinE raises `KeyError` — always catch both:
```python
try:
    mdata = self._load_model_var(model, var)
except (KeyError, FileNotFoundError):
    logger.warning("Variable %s not available for %s — skipping", var, model)
    continue
```

**Under the hood**, the base class dispatches based on `config.get_data_source_type()`:
- `"destine_catalog"` → `DataLoader` (intake catalogs, HEALPix, `values` dim)
- `"cmor"` → `CMORLoader` (CMOR directory tree, regular lat/lon, `(time, lat, lon)` dims)

#### Direct loader access (advanced)

For DestinE-specific use cases, you can still access the catalog directly:

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

However, **new diagnostics should use the base class helpers** above — they work across all model sets.

**Critical: dask arrays.** Model data is dask-backed. After reductions (climatology, mean), call `.compute()` before passing to numpy ops or storing:
```python
clim = climatology(da, period).compute()  # ← .compute() materializes
```

### 3.2 Observation Data

**Recommended: use base class helper** for sign-convention awareness:

```python
# Via base class helper (recommended — handles CMOR sign conventions):
obs_da = self._load_obs_var("tas")
# Applies obs_unit_factor, obs_unit_offset, AND cmor_obs_sign when data_source is "cmor"

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

**Sign convention gotcha (CMOR vs ERA5):**
- ERA5 & DestinE (IFS): surface heat fluxes **positive downward** (into surface)
- CMOR/CMIP6/EERIE: surface heat fluxes **positive upward** (away from surface)
- `_load_obs_var()` applies sign flip automatically for `hfss`/`hfls` when `data_source == "cmor"`
- Always use `_load_obs_var()` instead of direct `obs_loader.load_for_model_var()` to get correct signs

**Salinity convention gotcha (TEOS-10 vs EOS-80):**
- NEMO-based models (IFS-NEMO-ER, HadGEM3-GC5) output **absolute salinity** (SA, TEOS-10, g/kg)
- Other models (FESOM, ICON) and observations (EN4) use **practical salinity** (SP, EOS-80, PSU)
- Difference is ~0.17 g/kg globally, varies with location (up to ~0.5 near river mouths)
- Flagged per-model via `absolute_salinity: true` in `ModelConfig`
- Conversion uses `gsw.SP_from_SA(SA, p, lon, lat)` — see `ocean_en4._apply_sa_to_sp()`
- See pitfall 13.14 for implementation details

**NEMO depth dimension gotcha:**
- NEMO ocean output uses `deptht` as the depth dimension name, not `lev` or `depth`
- Always search for depth using `("lev", "depth", "deptht")` — see pitfall 13.15

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

**Recommended: use the base class helper** — it dispatches automatically:
```python
gm = self._model_global_mean(da, model)  # HEALPix: .mean(), latlon: cos-weighted
```

For direct use outside a diagnostic class:
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

### 4.2 Regridding to the Common Grid

**Key principle: all spatial bias computations happen on the common nereus 0.25° grid.** Both model data and observation data are regridded to this common grid before computing biases. This ensures consistency regardless of the native resolution of model or obs data.

For HEALPix data, regridding uses `nr.regrid()` with nearest-neighbor interpolation. For lat/lon data (EERIE), use `np.meshgrid(lon, lat)` before `nr.regrid()` — HEALPix per-pixel coordinate arrays pass directly.

**Interpolation method:** `nr.regrid()` and `nr.plot()` support a `method` parameter: `"nearest"`, `"idw"`, `"linear"`, `"cubic"`. The method is configured in YAML under `nereus.method` (default `"nearest"`). Read it in `__init__` as `self._regrid_method = self.config.nereus.get("method", "nearest")`.

**Design principle — method applies only to CMIP6 regridding.** High-resolution model and obs data (DestinE ~5 km, EERIE ~25 km) always use nearest neighbor. Only CMIP6 (~100 km) benefits from smoother interpolation (linear/cubic) to reduce blocky artifacts in bias maps.

**The common grid pattern (used by ALL bias-map diagnostics):**

```python
import nereus as nr

influence_radius = self.config.nereus.get("influence_radius", 80_000.0)
resolution = self.config.nereus.get("resolution", 0.25)

interpolator = None
target_lats = None
target_lons = None

for model in self.config.models:
    lon, lat = model_coords[model]
    grid_type = self.config.get_grid_type(model, self.domain)
    if grid_type != "healpix":
        lon, lat = np.meshgrid(lon, lat)

    if interpolator is None:
        # Build interpolator ONCE from first model (defines common grid):
        regridded, interpolator = nr.regrid(
            model_clim.values.ravel(),
            lon=np.asarray(lon).ravel(),
            lat=np.asarray(lat).ravel(),
            resolution=resolution,
            influence_radius=influence_radius,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        target_lats = interpolator.target_lat[:, 0]
        target_lons = interpolator.target_lon[0, :]

        # Regrid obs to the SAME common grid (not obs native grid!):
        obs_lons_2d, obs_lats_2d = np.meshgrid(obs_lons, obs_lats)
        _, obs_interp = nr.regrid(
            obs_clim.values.ravel(),
            lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
            resolution=resolution,
            influence_radius=influence_radius,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        obs_common = xr.DataArray(
            obs_interp(obs_clim.values.ravel()),
            dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )
    else:
        # Reuse interpolator for subsequent models:
        regridded = interpolator(model_clim.values.ravel())

    model_common = xr.DataArray(
        regridded, dims=("lat", "lon"),
        coords={"lat": target_lats, "lon": target_lons},
    )
    bias = model_common - obs_common
```

**CRITICAL: always regrid BOTH model AND obs data to the common nereus grid.** Even if they happen to share a resolution (e.g., EERIE 0.25° model and 0.25° nereus grid), always regrid for consistency. Do NOT assume grids match — different lat/lon grids with the same resolution can have different point counts (e.g., Berkeley Earth 1° has 181×360, EERIE 0.25° has 721×1440). Using `plot_combined_bias_map()` with a shared `interpolator` requires ALL panels to have the same grid shape, so everything must be on the common grid.

**CRITICAL: `plot_combined_bias_map()` shares its interpolator across panels.** If the obs panel is on a 1° grid (65K points) and a bias panel is on a 0.25° grid (1M points), the shared interpolator will fail with `ValueError: different number of values and points`. The fix is always computing biases on the common grid in `_compute_*()`, not in `_plot_*()`.

**CMIP6 → regular grid (uses configurable method):**

CMIP6 regridding uses `_regrid_to_target()`, a shared static method that handles the longitude wrapping issue for linear/cubic interpolation.

```python
@staticmethod
def _regrid_to_target(da, target_lats, target_lons,
                      resolution, influence_radius,
                      interp_cache, method="nearest"):
    """Regrid a CMIP6 lat/lon DataArray to the common target grid.

    CRITICAL: Source longitudes are converted from 0..360 to -180..180
    before regridding. This prevents a NaN stripe at 0° (prime meridian)
    caused by Delaunay triangulation not wrapping at 0°/360° boundary.
    After regridding, the output is rolled back to 0..360 to match the
    target grid used by model/obs data.
    """
    ir = max(influence_radius, 250_000.0)
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat_arr = da[lat_name].values
    lon_arr = da[lon_name].values

    # Convert to -180..180 to avoid gap at 0° in triangulation
    lon_arr = np.where(lon_arr > 180, lon_arr - 360, lon_arr)
    sort_idx = np.argsort(lon_arr)
    lon_arr = lon_arr[sort_idx]

    grid_key = (len(lat_arr), len(lon_arr))
    if grid_key not in interp_cache:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
        _, interp_cache[grid_key] = nr.regrid(
            da.values[:, sort_idx].ravel(),
            lon=lon_2d.ravel(), lat=lat_2d.ravel(),
            resolution=resolution, method=method,
            influence_radius=ir, lon_bounds=(-180.0, 180.0),
            as_xarray=True,
        )
    regridded = interp_cache[grid_key](da.values[:, sort_idx].ravel())

    # Roll back to 0..360 to match target_lons
    n_roll = regridded.shape[1] // 2
    regridded = np.roll(regridded, -n_roll, axis=1)
    return xr.DataArray(
        regridded, dims=("lat", "lon"),
        coords={"lat": target_lats, "lon": target_lons},
    )
```

**Key points for `_regrid_to_target()`:**
1. **Longitude wrapping:** Linear/cubic interpolation uses Delaunay triangulation which does NOT wrap at 0°/360°. Converting to -180..180 moves any gap to ±180° (antimeridian, at map edges in Robinson projection) instead of 0° (prime meridian, map center).
2. **Sort columns:** After converting lons, sort data columns to match the new lon order.
3. **Roll output:** After regridding with `lon_bounds=(-180, 180)`, roll the result by `n_cols//2` to match the 0..360 target grid used by model/obs data.
4. **Interpolator caching:** Cache per `(n_lat, n_lon)` key — all CMIP6 models with the same grid shape reuse the interpolator.
5. **Full signature:** `_regrid_to_target(da, target_lats, target_lons, resolution, influence_radius, interp_cache, method="nearest")` — all 6 positional args are required.

**CRITICAL: CMIP6 MMM must be "regrid first, then average" — never "average on native grids, then regrid".** CMIP6 models have different native grids (e.g., 1° vs 1.5°). Using `xr.align(*fields, join="inner")` on incompatible grids produces empty intersections (different lat/lon float values). Always regrid each CMIP6 model to the common target grid individually using `_regrid_to_target()`, then average the regridded fields:

**Influence radius must match source data density:**
- DestinE production (nside=1024, ~5 km): 80,000 m (80 km)
- EERIE (0.25 deg, ~25 km): 80,000 m (80 km)
- Tests (nside=8, ~815 km): 1,000,000 m (1000 km)
- CMIP6 (~1-2 deg): ≥250,000 m (forced minimum in `_regrid_to_target`)
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
deseas = deseason(da, period)                           # Remove monthly seasonal cycle
detrended = detrend(deseas)                             # Remove per-grid-point linear trend
```

**`deseason(da, period=None)`** — thin wrapper combining `monthly_climatology()` + `anomaly()`. Returns deseasonalised anomalies.

**`detrend(da, dim="time")`** — removes per-grid-point linear trend using `linear_trend()`. Data must be materialised (call `.compute()` first). Works on any dimensionality (1D, 2D+time).

---

## 5. Plotting Conventions

### 5.1 Color Scheme

```python
from feather.plot.styles import OBS_COLOR, CMIP6_COLOR

# Model colors are config-driven (work for any model set):
color = self.config.get_model_color(model)  # Returns hex color from config

# OBS_COLOR = "black"
# CMIP6_COLOR = "#888888"
```

**Do not hardcode model colors.** Use `self.config.get_model_color(model)` — this reads from the config YAML and falls back to a color cycle for models without explicit colors.

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
    method=self._regrid_method,  # Pass interpolation method to nr.plot()
)
```

**Combined multi-panel map with shared colorbar (for always-positive fields like STD):**
```python
from feather.plot.maps import plot_combined_map

fig, axes = plot_combined_map(
    data_dict,       # OrderedDict of {label: DataArray} — all panels same cmap
    title="My Title",
    cmap="YlOrRd",
    vmin=shared_vmin, vmax=shared_vmax,
    units="K",
    method=self._regrid_method,
)
```

Unlike `plot_combined_bias_map`, all panels use the **same colormap and range** — no separate obs/bias split. Useful for fields that are always positive (STD, variance, etc.).

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
    plot_taylor_diagram,   # Polar plot: pattern corr vs normalised STD
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
- `{var}_std_combined`, `{var}_std_diff_combined` (climate_variability)

### 6.5 Plot Types Used in Metadata

- `"bias_map"` — 3-panel model/obs/bias
- `"combined_bias_map"` — multi-panel obs + N bias panels
- `"timeseries"` — time series line plot
- `"seasonal_cycle"` — 12-month cycle
- `"combined_map"` — multi-panel with shared colormap (all panels identical rendering)
- `"budget_bars"` — two-panel: grouped bar chart (left) + zoomed TOA Net (right)
- `"gregory"` — scatter plot with regression
- `"zonal_profile"` — latitude vs variable profile
- `"polar_map"` — polar stereographic projection map
- `"taylor_diagram"` — polar plot of pattern correlation vs normalised STD

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

CMIP6 bias maps require regridding each CMIP6 model to the common target grid. The key design principle is **"regrid then average"** — regrid each model individually using `_regrid_to_target()` with the configurable method, then average the regridded fields to form the MMM. This ensures per-model interpolation quality and avoids artifacts from averaging on disparate native grids.

**MMM computation (regrid per-model, then average):**
```python
def _compute_cmip6_mmm(self, var, obs_common, target_lats, target_lons,
                        obs_res, cmip6_ir):
    interp_cache = {}
    fields, info = [], {}
    for model, variant in self.cmip6_loader.get_member_pairs():
        da = self.cmip6_loader.load_var_for_model_var(
            var, model, variant=variant, period=self.period,
        )
        if da is None:
            continue
        regridded = self._regrid_to_target(
            da, target_lats, target_lons,
            obs_res, cmip6_ir, interp_cache,
            method=self._regrid_method,      # configurable from nereus.method
        )
        fields.append(regridded)
    if fields:
        mmm = sum(fields) / len(fields)
        return mmm, obs_common, info
    return None, None, {}
```

**When `cmip6_individual=True`, reuse already-regridded fields:**
```python
@staticmethod
def _mmm_from_individual(cmip6_individual_data, obs_common):
    """Derive MMM from already-regridded individual fields (no double interpolation)."""
    fields = [entry["regrid"] for entry in cmip6_individual_data.values()
              if entry.get("regrid") is not None]
    if not fields:
        return None, None, {}
    mmm = sum(fields) / len(fields)
    return mmm, obs_common, {"n_members": len(fields)}
```

In `_compute_variable()`, call `_mmm_from_individual()` when individual CMIP6 data is already computed:
```python
if self.cmip6_individual and cmip6_individual_data:
    cmip6_mmm_result = self._mmm_from_individual(cmip6_individual_data, obs_common)
else:
    cmip6_mmm_result = self._compute_cmip6_mmm(var, obs_common, ...)
```

**Plotting:** pass `method=self._regrid_method` to `plot_combined_bias_map()` so `nr.plot()` also uses the configured interpolation method for rendering:
```python
fig, axes = plot_combined_bias_map(..., method=self._regrid_method)
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

Variables use **CMOR canonical names** as keys (e.g., `"tas"` not `"avg_2t"`). The `destine_variable` field maps back to DestinE names for backward compatibility.

If your diagnostic uses variables not yet in `VARIABLE_REGISTRY`, add entries in `feather/data/variables.py`:

```python
"tas": VarInfo(
    name="tas",                      # CMOR canonical name (registry key)
    long_name="2m Temperature",
    units="K",
    domain="sfc",
    cmap="RdBu_r",
    obs_dataset="ERA5",
    obs_variable="t2m",
    destine_variable="avg_2t",       # DestinE catalog name (empty if N/A)
    cmip6_variable="tas",            # "" if no CMIP6 mapping
    cmip6_table="Amon",              # "" if no CMIP6 mapping
    obs_unit_factor=1.0,             # Multiply obs by this to match model units
    obs_unit_offset=0.0,             # Add after multiplying
    cmor_obs_sign=1.0,               # -1.0 for hfss/hfls (ERA5→CMOR sign flip)
    group="temperature",
),
```

`get_var()` accepts both CMOR and DestinE names via `_DESTINE_TO_CANONICAL` fallback.

### 8.2 Existing Variables (33 total)

**Surface atmospheric (23):** `tas`, `ts`, `psl`, `uas`, `vas`, `sfcWind`, `clt`, `prw`, `clwvi`, `clivi`, `pr`, `hfss`, `hfls`, `rsds`, `rlds`, `rss`, `rls`, `rsscs`, `rlscs`, `rst`, `rlt`, `rstcs`, `rltcs`

**Ocean 2D (5):** `tos`, `siconc`, `sithick`, `zos`, `sos`

**Ocean 3D (2):** `thetao`, `so`

**Pressure levels (4):** `ta`, `ua`, `va`, `hus`

Variables confirmed available in EERIE: `tas`, `clt`, `hfls`, `hfss`, `pr`, `psl`, `rlds`, `rsds`, `tos`, `thetao`, `so`, `siconc`, `sithick`. Net radiation variables (`rss`, `rls`, etc.) are auto-derived from CMOR component fluxes.

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
| `--variables tas psl` | Intersect with your `variables` list (CMOR names) |
| `--cmip6-individual` | Passed as kwarg if your `__init__` accepts it |
| `--no-skip-existing` | Forces regeneration of all figures |
| `--config configs/eerie.yaml` | Use a different model set config |

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
7. Climate variability STD maps (shared sequential colormap)
8. Climate variability STD difference maps (obs STD + diverging diff panels)
9. Precipitation bias maps (absolute, from `precipitation_mswep`)
10. Relative precipitation bias maps (%, masked in arid regions)
11. Precipitation intensity distribution (area-weighted PDF, log-scale)
12. Precipitation zonal mean (ITCZ, storm tracks)
13. Temperature bias maps vs Berkeley Earth (independent station-based reference)
14. Temperature warming trend maps (K/decade, global Robinson + polar stereographic)
15. Taylor diagram (pattern correlation vs normalised standard deviation, multi-season)

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
| HEALPix model data | `_model_global_mean(da, model)` or `global_mean(da)` | cos(lat) weights | ~+3K warm bias |
| Regular lat/lon model data (EERIE) | `_model_global_mean(da, model)` or `latlon_global_mean(da)` | Unweighted .mean | ~-8K cold bias |
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

**Solution:** Always wrap model loading in try/except (catch both exception types):
```python
try:
    mdata = self._load_model_var(model, var)
except (KeyError, FileNotFoundError):
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

### 13.8 NaN Stripe at Prime Meridian with Linear/Cubic Interpolation

**Problem:** When using `method="linear"` or `method="cubic"` with `nr.regrid()`, Delaunay triangulation does not wrap around the 0°/360° longitude boundary. This creates a NaN stripe at the prime meridian (center of Robinson projection maps).

**Solution:** Convert source longitudes from 0..360 to -180..180 before interpolation, sort data columns accordingly, use `lon_bounds=(-180.0, 180.0)`, and roll the output back to 0..360 after regridding. The gap moves to ±180° (antimeridian, at map edges where it's invisible). See `_regrid_to_target()` in Section 4.2 for the full implementation.

### 13.9 CERES Loader Error Handling

**Problem:** `obs_loader.load_ceres()` raises `AttributeError` if the mock loader doesn't have it.

**Solution:** Catch `AttributeError` alongside other exceptions:
```python
try:
    da = self.obs_loader.load_ceres(...)
except (KeyError, FileNotFoundError, AttributeError) as e:
    logger.warning("CERES data not available: %s", e)
```

### 13.10 `_figure_exists()` Partial Outputs

The check requires **both** `.png` and `.json`. If only one exists (e.g., crash during save), the variable is reprocessed. This is intentional.

### 13.11 inspect.signature() for Optional kwargs

The pipeline uses `inspect.signature()` to check if a diagnostic's `__init__` accepts `cmip6_individual` before passing it:
```python
import inspect
sig = inspect.signature(cls.__init__)
if "cmip6_individual" in sig.parameters:
    kwargs["cmip6_individual"] = True
```
Always accept `cmip6_individual` as a keyword argument if your diagnostic should support it.

### 13.12 Symmetric Colorbars for Trend Maps

**Problem:** For trend maps (K/decade), the auto-computed `vmin`/`vmax` (2nd/98th percentile) are asymmetric — e.g., vmin=-0.1, vmax=0.5. With `RdBu_r` colormap, the blue dominates visually and gives a false impression of mostly negative trends.

**Solution:** Use a symmetric colorbar centered on zero for the obs panel:
```python
obs_vals = np.asarray(obs_trend).ravel()
obs_vals = obs_vals[np.isfinite(obs_vals)]
obs_vmax = float(np.percentile(np.abs(obs_vals), 98)) or 0.5

fig, axes = plot_combined_bias_map(
    obs_trend, bias_dict,
    vmin=-obs_vmax, vmax=obs_vmax,  # symmetric around zero
    cmap="RdBu_r",
    ...
)
```
The bias panels (`bias_vmax`) are automatically symmetric. Only the obs panel needs explicit symmetric `vmin`/`vmax`.

### 13.13 CMIP6 `xr.align()` Fails on Incompatible Grids

**Problem:** Different CMIP6 models have different native grids (e.g., 1° vs 1.5°). Using `xr.align(*fields, join="inner")` takes the coordinate intersection, which can be **empty** when lat/lon float values don't match exactly. This produces `ValueError: No points given` in Delaunay triangulation.

**Solution:** Always regrid each CMIP6 model to the common target grid individually using `_regrid_to_target()` before averaging. Never use `xr.align()` to combine fields from different native grids:
```python
# WRONG: align on native grids → empty intersection
aligned = xr.align(*trend_fields, join="inner")
mmm = sum(aligned) / len(aligned)

# RIGHT: regrid each to common grid first
for model, variant in member_pairs:
    trend = linear_trend(da.compute()) * 10
    regridded = GlobalBiases._regrid_to_target(
        trend, target_lats, target_lons,
        resolution, influence_radius, interp_cache,
        method=self._regrid_method,
    )
    trend_fields.append(regridded)
mmm = sum(trend_fields) / len(trend_fields)
```

### 13.14 NEMO Models Output Absolute Salinity (TEOS-10)

**Problem:** NEMO-based ocean models (IFS-NEMO-ER, HadGEM3-GC5 in EERIE; ifs-nemo in DestinE) output **absolute salinity** (SA, TEOS-10, g/kg) instead of **practical salinity** (SP, EOS-80, PSU). EN4 observations and most other models (FESOM, ICON) use practical salinity. Comparing SA directly against SP introduces a systematic bias of ~0.17 g/kg (varying with location).

**Solution:** The `ocean_en4` diagnostic has `_apply_sa_to_sp()` which converts SA→SP using `gsw.SP_from_SA(SA, p, lon, lat)` for models with `absolute_salinity: true` in their `ModelConfig`. The conversion is:
- **Opt-in per model** — set `absolute_salinity: true` in the YAML config (not all NEMO models necessarily need this)
- **Dask-compatible** — uses `xr.apply_ufunc(gsw.SP_from_SA, ..., dask='parallelized')`, stays lazy
- **Coordinate-aware** — uses depth as pressure (dbar ≈ metres), lat/lon from DataArray dims (latlon grids) or external model_coords (HEALPix)
- **Applied per-model** — only flagged models get converted; others pass through unchanged

```yaml
# In config YAML:
models:
  IFS-NEMO-ER:
    absolute_salinity: true   # NEMO outputs TEOS-10 absolute salinity
  IFS-FESOM2-SR:
    # no flag → practical salinity assumed (default)
```

```python
# In diagnostic code — applied inside model loop after base convert:
if variable == "so":
    da_conv = self._apply_sa_to_sp(
        da_conv, model, model_coords=mc, depth=depth)
```

**Requires:** `gsw` package (`conda install -c conda-forge gsw` or `pip install gsw`).

### 13.15 NEMO Depth Dimension Named `deptht`

**Problem:** NEMO ocean model output uses `deptht` as the depth dimension name (e.g., HadGEM3-GC5 ocean files), not the standard `lev` or `depth` used by other models. Code that only searches for `lev`/`depth` will fail to find depth levels, causing "No depth levels configured" errors.

**Solution:** Always search for depth coordinates using all known names: `("lev", "depth", "deptht")`. The `ocean_en4` diagnostic does this in `_get_depth_levels()`, `_get_layer_thickness()`, and `_apply_sa_to_sp()`. Similarly, HadGEM3's depth bounds coordinate is `deptht_bnds` (not `lev_bnds`).

For the surface level extraction, the fallback `da.dims[1]` already handles unknown depth dim names:
```python
level_dim = "level" if "level" in da.dims else da.dims[1]  # catches "deptht" etc.
```

### 13.16 Standalone Observation Loading (Non-VARIABLE_REGISTRY Datasets)

**Problem:** Some diagnostics use observation datasets not mapped through `VARIABLE_REGISTRY` (e.g., Berkeley Earth for `temperature_berkeley`, MSWEP for `precipitation_mswep`). The base class `_load_obs_var()` only works with variables registered in the variable registry.

**Solution:** Load directly via `self.obs_loader.load(dataset_name, var_name)` and handle unit conversion, dimension renaming, and longitude shifting manually:
```python
def _load_my_obs(self, period=None):
    da = self.obs_loader.load("MY_DATASET", "var_name", period=period)
    # Rename dims if needed (e.g., latitude→lat, longitude→lon)
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    # Shift lons from -180..180 → 0..360 if needed
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    # Unit conversion (e.g., degC → K)
    if float(da.mean()) < 200:  # heuristic: degC range
        da = da + 273.15
    return da
```

The dataset must be configured in the YAML config under `obs_datasets`:
```yaml
obs_datasets:
  MY_DATASET:
    path: "{obs_root}/MY-DATASET/subdir"
    variables:
      var_name: "filename.nc"
```

---

## 14. Complete Checklist

- [ ] **Create** `feather/diag/my_diagnostic.py` with `@register`-decorated class
- [ ] **Set** class attributes: `name`, `title`, `domain`, `variables`, `group`
- [ ] **Accept** `cmip6_loader`, `variables`, `cmip6_individual` in `__init__`
- [ ] **Override** `run()` with per-variable incremental saving
- [ ] **Implement** `compute()` and `plot()` as thin wrappers
- [ ] **Implement** `_compute_single()` / `_compute_variable()` with:
  - Model loading via `_load_model_var()` with `try/except (KeyError, FileNotFoundError)`
  - Obs loading via `_load_obs_var()` (sign-convention aware) or `load_ceres()`
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

| Diagnostic | File | Figure Types | Saving Pattern |
|-----------|------|--------------|----------------|
| `global_biases` | `diag/global_biases.py` | Combined bias maps (per var per period) | Per-variable (3 figures per var) |
| `timeseries` | `diag/timeseries.py` | Global-mean time series | Per-variable (1 figure per var) |
| `seasonal_cycle` | `diag/seasonal_cycle.py` | 12-month cycle | Per-variable (1 figure per var) |
| `radiation_budget` | `diag/radiation_budget.py` | Budget bars, Gregory, imbalance TS, bias maps | Per-figure-group |
| `sea_ice` | `diag/sea_ice.py` | Area/extent/volume TS, seasonal cycles, trends, polar maps | Per-figure-group |
| `ocean_sst` | `diag/ocean_sst.py` | SST bias maps, TS, seasonal cycle, zonal mean | Per-figure-group |
| `ocean_en4` | `diag/ocean_en4.py` | Surface bias maps, Hovmoller, depth-layer TS | Per-figure-group |
| `global_trends` | `diag/global_trends.py` | Per-grid-point linear trend maps | Per-variable |
| `climate_variability` | `diag/climate_variability.py` | STD maps + STD diff maps (deseasonalised, detrended) | Per-variable (2 figures per var) |
| `precipitation_mswep` | `diag/precipitation_mswep.py` | Abs/rel bias maps, TS, seasonal cycle, zonal mean, intensity PDF | Per-figure-group (6 groups, 8 figures) |
| `temperature_berkeley` | `diag/temperature_berkeley.py` | Bias maps, TS, seasonal cycle, zonal mean, warming trends (global + polar stereo), Taylor diagram | Per-figure-group (6 groups, 10 figures) |

All diagnostics are grid-agnostic and work with both DestinE (HEALPix) and EERIE (lat/lon) model sets.

Use `timeseries.py` as the simplest template for line-plot diagnostics.
Use `global_biases.py` as the template for bias-map diagnostics.
Use `radiation_budget.py` or `precipitation_mswep.py` as the template for complex multi-figure-type diagnostics.
Use `temperature_berkeley.py` as the template for diagnostics with standalone obs loading, Taylor diagrams, polar stereographic maps, and trend analysis.
