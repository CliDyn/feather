# Feather — Lightweight Climate Model Evaluation Framework

## Context

We need a lightweight, human-readable framework for evaluating DestinE high-resolution climate simulations against observations. The simulations (IFS-FESOM, IFS-NEMO, ICON) run on HEALPix grids at ~5km resolution and are available via intake catalogs in zarr format. The framework is inspired by AQUA/AQUA-diagnostics but deliberately much simpler — each diagnostic should be understandable as a standalone script and exportable as a Jupyter notebook.

**Key constraints:**
- Work on native HEALPix data; only regrid (NN via nereus) for map plotting
- Zonal means via latitude-band binning on native HEALPix grid (no regridding)
- Use nereus for all map/transect plots; matplotlib for line/scatter
- Dask-ready throughout
- Multiple data sources: intake catalogs (default), explicit file paths, FDB (later)
- Model evaluation focus (model vs observations, primary); optional CMIP6 comparison (secondary)
- Observation paths configurable in YAML, with Levante defaults
- **Every figure saved with a JSON metadata sidecar** — rich context for LLM analysis and dashboard
- **Web dashboard generation** — static HTML site with LLM-generated scientific analysis per figure

---

## Phase 1: Core Infrastructure — COMPLETED

**Status:** All 10 steps implemented and verified.

### What was built

| Module | File | Status |
|--------|------|--------|
| Packaging | `pyproject.toml`, `feather/__init__.py` | Done |
| Configuration | `feather/config.py`, `configs/default.yaml` | Done |
| Variable Registry | `feather/data/variables.py` (27 variables, VarInfo dataclass) | Done |
| Model Loader | `feather/data/loader.py` (DataLoader: catalog + paths) | Done |
| Obs Loader | `feather/data/obs.py` (ObsLoader: config-driven) | Done |
| Spatial Utils | `feather/util/spatial.py` (zonal_mean, global_mean, regional_mean) | Done |
| Temporal Utils | `feather/util/temporal.py` (climatology, seasonal, monthly, anomaly, annual_mean) | Done |
| Unit Conversions | `feather/util/units.py` (K↔°C, precip flux↔mm/day, Pa↔hPa) | Done |
| Map Plots | `feather/plot/maps.py` (plot_bias_map, plot_single_map via nereus) | Done |
| Line Plots | `feather/plot/lines.py` (timeseries, seasonal_cycle, zonal_profile) | Done |
| Plot Styles | `feather/plot/styles.py` (MODEL_COLORS, apply_style) | Done |
| Tests | `tests/` (26 unit tests + 2 integration tests, all passing) | Done |

### Key findings from Phase 1

1. **Data shapes:** Model 2D data is `(time=300, values=12582912)` with 33 vars. Lazy dask loading via intake works well. Datasets contain `longitude`, `latitude`, `time`, and variable data — but NO `area` variable. HEALPix cells are equal area, so simple `.mean()` suffices for spatial averages.
2. **Zonal mean on HEALPix:** 1° lat-band binning works on full-resolution data. At nside=8 (tests), need ≥10° bins to avoid empty polar bins. At nside=1024 (production), 1° is fine.
3. **ERA5 variable naming:** Variable names inside the NetCDF differ from the file-key names (e.g., file key `t2m` → variable inside is `T2M`). The ObsLoader `_find_variable()` handles this via case-insensitive fallback.
4. **cftime deprecation:** `xr.cftime_range()` is deprecated — all test code migrated to `xr.date_range()` in Phase 3.
5. **Catalog key convention:** 2D uses `clmn_high`, 3D uses `clmn_standard`. The `DataLoader.make_key()` method encodes this.

### Verification results

```
pytest tests/ -v -m "not integration"  → 26/26 passed
pytest tests/ -v -m "integration"      → 2/2 passed (real catalog + ERA5)
```

Integration tests confirmed:
- Model: `baseline_hist_2_ifs-fesom_1_0001_clmn_high_sfc` → 300 months × 12.6M cells, 33 vars
- ERA5: 300 months (1990-2014), 0.25° grid, variable `T2M` inside file
- Zonal mean on real data: equator ~300K, south pole ~243K

---

## Phase 2: Figure Metadata System & Diagnostic Base Class — COMPLETED

**Status:** All 3 modules implemented and verified.

### What was built

| Module | File | Status |
|--------|------|--------|
| Figure metadata sidecar | `feather/diag/figure_meta.py` | Done |
| Diagnostic base class | `feather/diag/base.py` | Done |
| Diagnostic registry | `feather/diag/registry.py` | Done |
| Diag `__init__` exports | `feather/diag/__init__.py` | Done |
| Metadata tests | `tests/test_figure_metadata.py` (18 tests) | Done |
| Base class + registry tests | `tests/test_diag_base.py` (18 tests) | Done |

### Key design decisions from Phase 2

1. **`build_metadata()` auto-fills from variable registry.** When given `variables_used=["avg_2t"]`, it automatically pulls `units="K"`, `domain="sfc"`, `group="temperature"`, `obs_dataset="ERA5"`, `obs_variable="t2m"` from `VARIABLE_REGISTRY`. Explicit values override auto-fill.

2. **`save_figure_with_metadata()` is non-mutating.** It copies the input metadata dict before adding `generated_at`, so the caller's dict is never modified. This matters when the same metadata dict is reused across multiple saves.

3. **`DiagnosticBase.run()` returns `list[tuple[Path, Path]]`.** Each tuple is `(png_path, json_path)`. This differs slightly from the plan sketch (which showed `list[Path]`) — returning both paths is more useful for the downstream LLM analyzer which needs both the image and the metadata.

4. **`DiagnosticBase.plot()` returns `list[tuple[Figure, dict]]`.** The base class `run()` method handles calling `save_figure_with_metadata()` for each pair. This keeps diagnostics focused on computation and plotting without worrying about file I/O.

5. **Registry uses `autouse` fixture in tests to isolate state.** The `_REGISTRY` dict is module-level global state, so tests save/restore it to avoid cross-test contamination.

6. **`register` validates non-empty `name`.** A class without `name` raises `ValueError` at decoration time, catching mistakes early.

### Verification results

```
pytest tests/ -v -m "not integration"  → 62/62 passed
```

All 36 new tests pass alongside the original 26 from Phase 1.

---

## Existing Data Infrastructure

### Model Data (via intake catalogs)

**2D catalog:** `/work/ab0995/a270088/DestinE/GENERATION2_joint/2D/catalog.yaml`
- Format: zarr, multiple files per entry combined `by_coords`
- Coords: `latitude`, `longitude`, `time`; dim: `values` (1D HEALPix)
- SFC: nside=1024 (12,582,912 cells), 300 timesteps (1990-2014 monthly)
- O2D: same resolution, ocean 2D variables
- Key pattern: `{experiment}_2_{model}_{member}_0001_clmn_high_{sfc|o2d}`
- Variables (SFC): `avg_2t`, `avg_10ws`, `avg_tprate`, `avg_msl`, `avg_tcc`, `avg_tnswrf`, `avg_tnlwrf`, etc. (33 total)
- Variables (O2D): `avg_tos`, `avg_siconc`, `avg_sithick`, `avg_zos`, `avg_sos`, `avg_hc300m`, etc. (12 total)

**3D catalog:** `/work/ab0995/a270088/DestinE/GENERATION2_joint/3D/catalog.yaml`
- O3D: nside=128 (196,608 cells), 72 depth levels, vars: `avg_so`, `avg_thetao`, `avg_von`, `avg_uoe`
- PL: nside=128, 19 pressure levels (1-1000 hPa), vars: `avg_t`, `avg_u`, `avg_v`, `avg_q`, `avg_z`, `avg_w`, `avg_r`, `avg_clwc`, `avg_pv`
- Key pattern: `{experiment}_2_{model}_{member}_0001_clmn_standard_{o3d|pl}`

**Experiments:** `baseline_hist`, `baseline_cont`, `projections_ssp3-7.0`, `story-nudging_hist/cont`
**Models:** `ifs-fesom`, `ifs-nemo`, `icon`

### Observation Data (at `/work/bb1153/b382289/data/aqua-dvc/datasets/`)

| Dataset | Path | Format | Variables | Period |
|---------|------|--------|-----------|--------|
| ERA5 | `ERA5/mon/ERA5_*.nc` | NetCDF | 30 monthly vars (T2m, SST, MSLP, radiation, fluxes, clouds, wind) | 1940-2024 |
| CERES EBAF | `CERES/EBAF_v4.2.1/` | NetCDF | TOA & surface radiation (all-sky + clear-sky) | 2000-2025 |
| EN4 | `EN4/v4.2.2/` | NetCDF | Ocean temperature (`thetao`) & salinity (`so`) profiles | 1950-2024 |
| ESA-CCI-L4 | `ESA-CCI-L4/v3.0.1/` | NetCDF | SST | multi-year |
| MSWEP | `MSWEP/v2.8/` | NetCDF+Zarr | Precipitation | multi-year |
| OSI-SAF | `OSI-SAF/OSI-AQUA/` | NetCDF | Sea ice concentration | multi-year |
| PSC/PIOMAS | `PSC/PIOMAS/`, `PSC/GIOMAS/` | NetCDF | Sea ice thickness (Arctic + Antarctic) | multi-year |
| AVISO | `AVISO/vDT2024/` | Zarr | Sea surface height, geostrophic currents | multi-year |
| BERKELEY-EARTH | `BERKELEY-EARTH/aqua-filled/` | NetCDF | Surface temperature (filled) | multi-year |

### CMIP6 Reference Data (optional comparison)

**Catalog:** `/home/a/a270088/PYTHON/DestinE/cmip6/cmip6_zarr_catalog.yaml` (593 entries)
**Data format:** Zarr, stored at `/work/ab0995/a270088/DestinE/cmip6/zarr/`
**Entry format:** `{Model}_{Experiment}_{Variant}_{Table}` (e.g., `MIROC6_historical_r1i1p1f1_Amon`)

**Available models (8 core):** AWI-CM-1-1-MR, MPI-ESM1-2-LR, EC-Earth3, CNRM-CM6-1, MIROC6, CESM2, UKESM1-0-LL, GFDL-CM4
**Tables:** Amon (atmosphere monthly), Omon (ocean monthly), SImon (sea ice monthly)

**25 mapped variables** (CMIP6 → DestinE naming) in existing `cmip6_variables.py`:
```python
# /home/a/a270088/PYTHON/DestinE/phase2/.../cmip6_variables.py
CMIP6_VAR_MAP = {
    "tas":    {"destine": "avg_2t",     "table": "Amon"},
    "pr":     {"destine": "avg_tprate", "table": "Amon"},
    "tos":    {"destine": "avg_tos",    "table": "Omon"},
    "siconc": {"destine": "avg_siconc", "table": "SImon"},
    "rsut":   {"destine": "avg_tnswrf", "table": "Amon"},
    "rlut":   {"destine": "avg_tnlwrf", "table": "Amon"},
    # ... 25 total
}
```

### Nereus Library (installed in conda `nereus` env)

Reuse from nereus:
- `nr.plot(data, lon, lat, *, ax, projection, resolution, interpolator, cmap, vmin, vmax, colorbar, colorbar_label, title, ...)` → `(fig, ax, interpolator)` — **3 return values**, map plotting with NN interpolation
- `nereus.plotting.get_projection(name)` — returns cartopy projection object (NOT `nr.projection()` which doesn't exist)
- `nr.mesh_from_arrays(lon, lat)` → mesh with `.area` attribute — compute cell areas for regular grids
- `nr.transect(data, lon, lat, depth, start, end, ...)` — vertical cross-sections
- `nr.regrid()` / `RegridInterpolator` — NN regridding with caching
- `nr.surface_mean(data, area)` — area-weighted surface mean
- `nr.volume_mean(data, area, thickness)` — volume-weighted mean
- `nr.heat_content(temp, area, thickness)` — ocean heat content
- `nr.ice_area()`, `nr.ice_extent()`, `nr.ice_volume()` — sea ice diagnostics (NH/SH variants)
- `nr.hovmoller()`, `nr.plot_hovmoller()` — Hovmoller diagrams
- `nr.get_region_mask()`, `nr.subset_by_bbox()` — regional masking
- `nr.find_nearest()`, `nr.haversine_distance()` — spatial queries
- `nr.healpix.load_mesh(ncells)` — HEALPix mesh with lon/lat/area

### Reference implementation

The existing climate diagnostics pipeline at `/home/a/a270088/PYTHON/DestinE/phase2/climate_change_analysis/climate_diagnostics/` has a working implementation of the dashboard + LLM analysis pattern. Key modules to reference:

| Module | Path | Reuse |
|--------|------|-------|
| LLM schemas | `climate_diag/llm_analysis/schemas.py` | Adapt Pydantic models |
| LLM prompts | `climate_diag/llm_analysis/prompts.py` | Adapt system/user prompts for model evaluation |
| LLM analyzer | `climate_diag/llm_analysis/analyzer.py` | Adapt Gemini figure analysis pipeline |
| Website generator | `climate_diag/website/generator.py` | Adapt SiteGenerator for feather |
| HTML templates | `climate_diag/website/templates/{base,index,diagnostic}.html` | Adapt Jinja2 templates |
| CSS styles | `climate_diag/website/static/style.css` | Reuse dark-theme design |
| Plot helpers | `climate_diag/plot_helpers.py` | `save_figure_with_metadata()` pattern |
| Diagnostic base | `climate_diag/diagnostics/base.py` | Adapt `DiagnosticBase` + registry |

---

## Current Package Structure

```
feather/                              # Git root: /home/a/a270088/PYTHON/feather/feather/
├── pyproject.toml
├── PLAN.md
├── CLAUDE.md
├── README.md
├── configs/
│   └── default.yaml                  # Default config with Levante paths
├── scripts/
│   ├── run_global_biases.py          # CLI: argparse-based global biases runner
│   ├── run_timeseries.py             # CLI: time series runner
│   ├── run_seasonal_cycle.py         # CLI: seasonal cycle runner
│   ├── run_all.py                    # CLI: run all diagnostics with --diagnostics filter
│   └── test_cmip6_e2e.py            # E2E CMIP6 test (run on compute node)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Synthetic fixtures, mock loaders, minimal_config
│   ├── test_config.py                # 3 tests
│   ├── test_loader.py                # 6 tests (+ 1 integration)
│   ├── test_obs.py                   # 5 tests (+ 1 integration)
│   ├── test_spatial.py               # 5 tests
│   ├── test_temporal.py              # 7 tests
│   ├── test_figure_metadata.py       # 18 tests
│   ├── test_diag_base.py             # 18 tests
│   ├── test_global_biases.py         # 16 tests
│   ├── test_timeseries.py            # 9 tests
│   ├── test_seasonal_cycle.py        # 8 tests
│   └── test_cmip6.py                # 47 unit + 2 integration tests
└── feather/
    ├── __init__.py                   # v0.1.0, exports FeatherConfig, DataLoader, ObsLoader, CMIP6Loader
    ├── config.py                     # FeatherConfig dataclass + YAML loader
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                 # DataLoader: catalog + paths
    │   ├── obs.py                    # ObsLoader: config-driven obs access
    │   ├── cmip6.py                  # CMIP6Loader: zarr loading + multi-model mean
    │   └── variables.py              # VarInfo + VARIABLE_REGISTRY (27 vars)
    ├── util/
    │   ├── __init__.py
    │   ├── spatial.py                # zonal_mean, global_mean, latlon_global_mean, RegridIndex, regrid_to_latlon, compute_latlon_areas
    │   ├── temporal.py               # climatology, seasonal, monthly, anomaly, annual_mean
    │   └── units.py                  # K↔°C, precip flux↔mm/day, Pa↔hPa
    ├── plot/
    │   ├── __init__.py
    │   ├── maps.py                   # plot_bias_map (3-panel), plot_single_map (nereus)
    │   ├── lines.py                  # plot_timeseries, plot_seasonal_cycle, plot_zonal_profile
    │   └── styles.py                 # MODEL_COLORS, OBS_COLOR, apply_style
    ├── diag/
    │   ├── __init__.py               # Exports + auto-imports diagnostics for @register
    │   ├── base.py                   # DiagnosticBase ABC (compute → plot → run)
    │   ├── figure_meta.py            # save_figure_with_metadata, build_metadata
    │   ├── registry.py               # @register, get_diagnostic, list_diagnostics
    │   ├── global_biases.py          # GlobalBiases diagnostic
    │   ├── timeseries.py             # TimeseriesDiag diagnostic
    │   └── seasonal_cycle.py         # SeasonalCycleDiag diagnostic
    └── export/
        └── __init__.py               # Placeholder
```

---

## Phase 3: First Diagnostics — COMPLETED

**Status:** All 3 diagnostics implemented and verified.

### What was built

| Module | File | Status |
|--------|------|--------|
| Spatial utils additions | `feather/util/spatial.py` (+`latlon_global_mean`, `regrid_to_latlon`, `RegridIndex`, `compute_latlon_areas`) | Done |
| Bias map plotting | `feather/plot/maps.py` (`plot_bias_map` using `nr.plot()` for all panels) | Done |
| CLI scripts | `scripts/run_global_biases.py`, `run_timeseries.py`, `run_seasonal_cycle.py`, `run_all.py` | Done |
| Global biases diagnostic | `feather/diag/global_biases.py` | Done |
| Time series diagnostic | `feather/diag/timeseries.py` | Done |
| Seasonal cycle diagnostic | `feather/diag/seasonal_cycle.py` | Done |
| Diag `__init__` imports | `feather/diag/__init__.py` (auto-registers diagnostics) | Done |
| Shared test fixtures | `tests/conftest.py` (+MockModelLoader, MockObsLoader, minimal_config) | Done |
| Global biases tests | `tests/test_global_biases.py` (16 tests) | Done |
| Timeseries tests | `tests/test_timeseries.py` (9 tests) | Done |
| Seasonal cycle tests | `tests/test_seasonal_cycle.py` (8 tests) | Done |

### Key design decisions from Phase 3

1. **NN regridding via `RegridIndex` (scipy KDTree).** `RegridIndex.build()` converts source lon/lat to 3D Cartesian coords and builds a `cKDTree` once. `RegridIndex.apply()` does cheap index lookup for subsequent fields. The convenience function `regrid_to_latlon()` wraps this. At nside=1024 (12.6M source points), building the KDTree takes ~30s but each `.apply()` is ~1s — critical since global_biases calls it 9+ times (3 models × annual + DJF + JJA).

2. **Pre-computed bias for plotting.** `plot_bias_map()` now accepts an optional `bias_data` parameter (xr.DataArray on regular grid). The diagnostic's `compute()` does the regridding + bias calculation, and `plot()` just visualizes. This keeps the plotting code simple and testable.

3. **Three-panel bias map uses nereus for all panels.** All three panels (model, obs, bias) use `nr.plot()` for consistent sizing and colorbar placement. Model data is regridded to the obs grid in `compute()`, so all panels share the same lat/lon grid and can reuse the nereus interpolator. Bias colorbar is symmetric (98th percentile of |bias|). Shared vmin/vmax for model+obs panels computed from 2nd/98th percentile.

4. **cftime → matplotlib compatibility.** Time series plot uses a `_to_plot_time()` helper that converts cftime datetime objects to pandas Timestamps via string parsing, since matplotlib cannot directly plot cftime dates.

5. **Diagnostics are configurable at construction.** All three accept `variables=`, `experiment=`, and `period=` overrides. The class-level defaults (`variables=["avg_2t"]`, `experiment="baseline_hist"`, `period=("1990","2014")`) cover the most common use case.

6. **Obs global mean uses nereus cell areas.** `latlon_global_mean()` computes proper cell areas via `nereus.mesh_from_arrays()` with LRU caching. Accepts optional pre-computed areas for CMIP6 `areacella`/`areacello`. Robust dim name detection supports `lat`/`latitude`/`nav_lat`/`y`/`rlat` and equivalents.

7. **Mock loaders in conftest.py.** `MockModelLoader` wraps `synth_healpix` dataset and returns it for any key. `MockObsLoader` wraps `synth_obs` dataset. This lets diagnostic tests run without real data or intake catalogs.

### Figures produced per diagnostic

- **global_biases:** Per model × variable: `{var}_annual_bias_{model}.png`, `{var}_djf_bias_{model}.png`, `{var}_jja_bias_{model}.png` (3 figures × N models × N variables)
- **timeseries:** Per variable: `{var}_timeseries.png` (all models + obs on same axes)
- **seasonal_cycle:** Per variable: `{var}_seasonal_cycle.png` (all models + obs on same axes)

### Verification results

```
pytest tests/ -v -m "not integration"  → 95/95 passed
pytest tests/ -v -m "integration"      → 2/2 passed
Total: 97 tests (33 new + 64 existing)
```

### Practical notes for future diagnostics

- **HEALPix grids are equal area — no area weighting needed.** Use `global_mean(data)` with `area=None` (simple `.mean()`) for model data on HEALPix. Using `cos(lat)` weights on equal-area cells over-weights the tropics and introduces a ~+3.3K warm bias for temperature. This was the single biggest bug found during initial testing.
- **Regular lat/lon grids need proper area weighting.** For observations (ERA5 etc.) use `latlon_global_mean()` which computes cell areas via `nereus.mesh_from_arrays()`. Unweighted means on lat/lon grids over-weight polar regions and are ~8K too cold for temperature.
- **CMIP6 area weighting must use `areacella`/`areacello`.** The CMIP6 catalog has 234 area-weight files (`{Model}_{Experiment}_{Variant}_{fx|Ofx}_areacell{a|o}.zarr`). Do NOT use `cos(lat)` for CMIP6 — models may have irregular grids.
- **Dask arrays need explicit `.compute()` after reductions.** When loading model data via intake catalogs, data is dask-backed. After computing climatologies or global means, call `.compute()` to materialise before passing to numpy operations or storing in results dicts.
- **`RegridIndex` builds KDTree once, applies cheaply.** At nside=1024, the model grid has 12.6M points. Building the KDTree takes ~30s but lookup is fast (~1s). All models share the same HEALPix grid, so one index serves all 3 models × all seasons. Never rebuild per field.
- **NN regridding at nside=8 introduces ~1-3K error in global mean.** Test tolerances must account for this. At nside=1024 (production), NN error is negligible (<0.1K).
- **cftime dates cannot be plotted by matplotlib directly.** The `_to_plot_time()` helper in `timeseries.py` converts via `pd.to_datetime([str(t) for t in time_values])`.
- **`plot_bias_map` requires nereus.** Tests that call it must either mock it (as in `test_global_biases.py`) or be marked as integration tests. The `compute()` methods do NOT need nereus — only `plot()` for map-based diagnostics.

### nereus API reference (verified)

- `nr.plot(data, lon, lat, *, ax, projection, resolution, interpolator, cmap, vmin, vmax, colorbar, colorbar_label, title, ...)` → `(fig, ax, interpolator)` — **3 return values**
- `nereus.plotting.get_projection(name)` — returns cartopy projection (NOT `nr.projection()`, which doesn't exist)
- `nr.mesh_from_arrays(lon, lat)` → mesh with `.area` attribute — for computing cell areas of regular grids
- `nr.surface_mean(data, area)` — area-weighted surface mean

### Lessons learned

1. **Always verify area weighting.** The most common climate data trap: unweighted mean on lat/lon is ~8K cold; cos(lat) on equal-area HEALPix is ~3K warm. Both fail silently.
2. **Don't mix plotting APIs.** Using `nr.plot()` for one panel and `xr.DataArray.plot()` for another creates inconsistent colorbars and panel sizes. Use the same API for all panels.
3. **Build expensive indices once.** KDTree from 12.6M points takes ~30s. Building it 9 times (3 models × 3 time periods) made the diagnostic take 5+ minutes instead of ~1 minute.
4. **Model datasets may not contain `area`.** The intake catalog datasets have `longitude`, `latitude`, `time`, and variable data — but no `area` variable. HEALPix being equal-area makes this a non-issue for global means.

---

## Phase 4: CMIP6 Loader — COMPLETED

**Status:** CMIP6Loader implemented and verified (47 new tests, 142 total passing).

### What was built

| Module | File | Status |
|--------|------|--------|
| CMIP6 Loader | `feather/data/cmip6.py` (~320 lines) | Done |
| Config update | `configs/default.yaml` (ensemble_mode, multi-variant models) | Done |
| Exports | `feather/data/__init__.py`, `feather/__init__.py` | Done |
| Test fixtures | `tests/conftest.py` (synth_cmip6, MockCMIP6Loader, cmip6_config) | Done |
| Tests | `tests/test_cmip6.py` (47 unit + 2 integration tests) | Done |

### CMIP6Loader API

```python
class CMIP6Loader:
    """Load CMIP6 data and compute multi-model mean on a common grid."""

    def load_var(cmip6_var, model, *, variant, table, period, season) -> DataArray | None
    def load_var_for_model_var(model_var, model, **kw) -> DataArray | None
    def load_multi_model_mean(cmip6_var, *, table, period, season, ensemble_mode) -> (DataArray | None, info)
    def load_mmm_for_model_var(model_var, **kw) -> (DataArray | None, info)
    def load_area(model, variant, table) -> DataArray | None
    def available_models(cmip6_var, table) -> list[str]
    def available_members(cmip6_var, table) -> list[tuple[str, str]]
    def available_models_for_model_var(model_var) -> list[str]
```

### Key design decisions from Phase 4

1. **Per-variable zarr loading only (no intake at load time).** Avoids staggered-grid conflicts that occur when loading merged multi-variable datasets. Path format: `{zarr_dir}/{Model}_historical_{variant}_{table}_{var}.zarr`.

2. **Two ensemble modes.** Config default `ensemble_mode: "one_per_model"` uses first variant per model (7 members for MMM). `"all_members"` uses all listed variants (~35 members). Per-call override via `ensemble_mode` kwarg on `load_multi_model_mean()`.

3. **Backward-compatible variant config.** `_get_variants(model_cfg)` supports both new `variants: [list]` and legacy `variant: str` format.

4. **Calendar normalization inside `load_var()`.** Different CMIP6 models use 360_day, noleap, standard calendars with different mid-month conventions. `_normalize_time()` converts all to first-of-month pandas timestamps before time slicing.

5. **Silent skip for missing data.** `load_var()` returns `None` when zarr not found or variable missing. `load_multi_model_mean()` collects models_skipped in info dict.

6. **Regular grid meshgrid for RegridIndex.** CMIP6 data is on regular lat/lon grids (1D lat + 1D lon arrays of different length). Must meshgrid before passing to `RegridIndex.build()` which expects scattered points of equal length.

7. **Area weights cached per `{model}_{variant}_{table}` key.** `load_area()` looks for `areacella` (fx table, atmosphere) or `areacello` (Ofx table, ocean).

### Config changes

- `regrid_resolution`: 0.25 → 1.0 (sufficient for CMIP6 comparison)
- `ensemble_mode: "one_per_model"` added
- `influence_radius` removed (not used by feather's RegridIndex)
- Models expanded from single `variant: str` to `variants: [list]` (7 models, up to 6 variants each)

### Verification results

```
pytest tests/test_cmip6.py -v -m "not integration"  → 47/47 passed
pytest tests/ -v -m "not integration"                → 142/142 passed (47 new + 95 existing)
```

### Important TODO: Replace RegridIndex with nereus RegridInterpolator

Feather's `RegridIndex` (in `util/spatial.py`) reimplements the same Cartesian-KDTree NN algorithm that nereus's `nr.RegridInterpolator` provides. Key differences:

| Feature | Feather `RegridIndex` | Nereus `RegridInterpolator` |
|---------|----------------------|----------------------------|
| Target grid | Manual target arrays | Auto-generated from `resolution` |
| Influence radius | None — every cell gets a value | 80km default — distant points become NaN |
| Multi-dim input | 1D only | 1D, 2D, ND natively |

The influence radius matters: without it, ocean values bleed onto land (and vice versa), creating artifacts. This refactor should replace `RegridIndex` with `nr.RegridInterpolator` everywhere: `util/spatial.py`, `data/cmip6.py`, and any diagnostic that calls `RegridIndex` or `regrid_to_latlon`.

CMIP6 comparison is always optional (`cmip6.enabled` in config). When enabled, diagnostics add:
- CMIP6 MMM line/panel to existing figures
- Additional metadata in `cmip6_info` field of the JSON sidecar

---

## Phase 5: LLM Analysis Pipeline

**Goal:** Automatically analyze each diagnostic figure using an LLM (Gemini), producing structured scientific interpretations that feed the dashboard.

### Step 5.1 — Analysis schemas

**File:** `feather/llm/schemas.py`

Pydantic models for validated LLM output:

```python
class FigureAnalysis(BaseModel):
    """LLM analysis of a single diagnostic figure."""
    summary: str              # 1-2 sentence overview
    key_findings: list[str]   # 3-5 bullet points
    spatial_patterns: str     # Notable geographic patterns
    model_agreement: str      # Inter-model consistency / obs agreement
    physical_interpretation: str  # Physical mechanisms
    caveats: list[str]        # Limitations, data quality notes
    confidence: Literal["high", "medium", "low"]

class DiagnosticSynthesis(BaseModel):
    """Cross-figure synthesis for one diagnostic."""
    narrative: str            # 2-3 paragraph synthesis
    headline_finding: str     # One-sentence executive summary
    connections: list[str]    # Related diagnostics / implications
```

### Step 5.2 — Prompt templates

**File:** `feather/llm/prompts.py`

System prompt adapted for **model evaluation** context (not climate change projections):

```python
_FIGURE_ANALYSIS_SYSTEM = """
You are a climate scientist analysing diagnostic figures from the
DestinE high-resolution climate model evaluation framework (Feather).

The framework compares three climate models against observations:
- IFS-FESOM, IFS-NEMO, ICON — all run on HEALPix grids at ~5 km
- Observation datasets: ERA5, CERES EBAF, EN4, ESA-CCI, MSWEP, OSI-SAF, etc.
- Period: 1990-2014 historical simulations

Your task is to interpret each figure scientifically. Focus on:
- Bias patterns (warm/cold, wet/dry, regional structure)
- Model-observation agreement (where models succeed vs fail)
- Physical mechanisms that explain the biases
- Inter-model differences (which model performs best where)
- Resolution-dependent features visible at ~5 km

{model_context}

Return your analysis as JSON matching this schema:
{schema}
"""
```

User prompt built dynamically from the metadata JSON sidecar:

```python
def build_figure_prompt(metadata: dict) -> str:
    """Build user prompt from figure metadata.

    Includes: diagnostic_name, title, variables, models, obs_dataset,
    units, period, description, computation_notes, summary_statistics.
    """
```

### Step 5.3 — Figure analyzer

**File:** `feather/llm/analyzer.py`

```python
class FigureAnalyzer:
    """Analyze diagnostic figures using an LLM."""

    def __init__(self, config: FeatherConfig):
        self.figures_dir = Path(config.output_dir) / "figures"
        self.analysis_dir = Path(config.output_dir) / "analysis"

    def analyze_figure(self, png_path: Path, metadata_path: Path) -> FigureAnalysis:
        """Send PNG + metadata to LLM, return validated analysis."""

    def analyze_diagnostic(self, diagnostic_name: str) -> DiagnosticSynthesis:
        """Synthesize all figure analyses for one diagnostic."""

    def analyze_all(self, skip_existing: bool = True):
        """Run analysis on all diagnostics. Skip already-analyzed figures."""
```

**Output structure:**
```
output/
├── figures/
│   └── global_biases/
│       ├── t2m_annual_bias_ifs-fesom.png
│       ├── t2m_annual_bias_ifs-fesom.json     # metadata sidecar
│       └── ...
└── analysis/
    └── global_biases/
        ├── t2m_annual_bias_ifs-fesom_analysis.json   # LLM figure analysis
        └── synthesis.json                             # cross-figure synthesis
```

### Step 5.4 — LLM configuration

Add to `configs/default.yaml`:

```yaml
llm:
  provider: "gemini"          # or "anthropic", "openai"
  model: "gemini-2.5-flash"
  max_retries: 3
  retry_delay: 10
  skip_existing: true         # don't re-analyze already-analyzed figures
```

Add to `FeatherConfig`:

```python
@dataclass
class FeatherConfig:
    # ... existing fields ...
    llm: dict                 # LLM provider config
```

---

## Phase 6: Web Dashboard

**Goal:** Generate a static HTML site with all diagnostic figures, their metadata, and LLM analyses — viewable in any browser, no server needed.

### Step 6.1 — Site generator

**File:** `feather/website/generator.py`

```python
class SiteGenerator:
    """Build static HTML dashboard from figures + analyses."""

    def __init__(self, config: FeatherConfig):
        self.figures_dir = Path(config.output_dir) / "figures"
        self.analysis_dir = Path(config.output_dir) / "analysis"
        self.site_dir = Path(config.output_dir) / "site"
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TEMPLATES_DIR)
        )

    def collect_diagnostics(self) -> list[dict]:
        """Scan figures/ and analysis/ directories.

        For each diagnostic:
        - Discover all PNG + JSON pairs
        - Load metadata sidecars
        - Load LLM analyses (if available)
        - Load synthesis (if available)
        - Group by diagnostic group
        """

    def build(self):
        """Build the complete static site.

        1. Collect all diagnostic data
        2. Copy static assets (CSS)
        3. Copy figure PNGs
        4. Group diagnostics by group
        5. Render index.html (gallery overview)
        6. Render one HTML page per diagnostic
        """
```

### Step 6.2 — Jinja2 templates

Adapted from the reference implementation's dark-theme design.

**`templates/base.html`:**
- Sidebar navigation grouped by diagnostic group (temperature, radiation, ocean, sea ice, etc.)
- Main content area
- Lightbox modal for figure zoom (click to enlarge)
- Google Fonts (Inter)

**`templates/index.html`:**
- Overview header (models, period, diagnostic count)
- Card grid: one card per diagnostic with:
  - Thumbnail (first figure)
  - Title + group badge
  - Headline finding (from synthesis, if available)
  - Figure count + analysis status

**`templates/diagnostic.html`:**
- Page header with group badge + diagnostic title
- Synthesis box (headline finding + narrative + connections)
- Per-figure sections:
  - Figure image (clickable for lightbox zoom)
  - Metadata table (variables, models, obs dataset, units, period, method)
  - Summary statistics (global mean bias, RMSE, etc.)
  - LLM analysis panel (if available):
    - Summary
    - Key findings (bulleted)
    - Spatial patterns
    - Model agreement
    - Physical interpretation
    - Caveats
    - Confidence badge (high/medium/low)

### Step 6.3 — CSS styles

**File:** `feather/website/static/style.css`

Reuse the dark-theme design from the reference implementation:
- Dark background (`#0f1117`, `#1a1d27`, `#21242f`)
- Inter font, clean typography
- Card grid layout for gallery
- Side-by-side figure + analysis layout
- Confidence badges (green/yellow/red)
- Group badges (color-coded by domain)
- Responsive breakpoints (1024px, 768px)
- Lightbox overlay for figure zoom

### Step 6.4 — Dashboard configuration

Add to `configs/default.yaml`:

```yaml
website:
  title: "Feather — Climate Model Evaluation"
  subtitle: "DestinE High-Resolution Simulations vs Observations"
  group_labels:
    temperature: "Temperature"
    radiation: "Radiation Budget"
    precipitation: "Precipitation"
    circulation: "Atmospheric Circulation"
    ocean_surface: "Ocean Surface"
    sea_ice: "Sea Ice"
    ocean_3d: "Ocean 3D"
```

---

## Phase 7: Pipeline Runner

**File:** `feather/run.py` (or CLI entry point)

```python
def run_pipeline(config_path, *, steps=None, diagnostics=None):
    """Run the feather pipeline.

    Steps:
    1. "diagnostics" — run compute + plot for selected diagnostics
    2. "analyze"     — run LLM analysis on generated figures
    3. "website"     — build static HTML dashboard
    4. "all"         — run all steps in sequence
    """
```

```bash
# Usage examples:
python -m feather.run --config configs/default.yaml --step diagnostics
python -m feather.run --config configs/default.yaml --step analyze
python -m feather.run --config configs/default.yaml --step website
python -m feather.run --config configs/default.yaml --step all
```

---

## Phase 8: Atmosphere Diagnostics

- `radiation_budget` — TOA & surface radiation vs CERES (+optional CMIP6 MMM)
- `precipitation` — Precip evaluation vs MSWEP/ERA5 (+optional CMIP6)
- `lat_profiles` — Zonal mean profiles (+optional CMIP6 spread)

---

## Phase 9: Ocean & Sea Ice Diagnostics

- `seaice` — Extent, area, concentration vs OSI-SAF/PIOMAS (+optional CMIP6)
- `ocean_surface` — SST, SSS, SSH vs ESA-CCI/EN4/AVISO (+optional CMIP6)
- `ocean_drift` — Hovmoller diagrams (needs 3D data)

---

## Phase 10: Export & Reproducibility

- Jupyter notebook generation per diagnostic (via nbformat)
- Static HTML gallery archive (self-contained zip)
- FDB data source integration

---

## Key Design Principles

### Every figure is self-describing

The JSON metadata sidecar is the single source of truth for what a figure shows. It contains:
- **What:** diagnostic name, variable, units, domain
- **How:** computation method, spatial extent, plot type, colormap
- **Context:** models, obs dataset, period, summary statistics
- **Provenance:** timestamp, CMIP6 info (if applicable)

This means:
1. The LLM analyzer never needs to see the code — the metadata provides full context
2. The dashboard can render rich detail pages without running diagnostics
3. Figures can be regenerated or audited from metadata alone
4. New LLM providers can be swapped in without changing the diagnostic code

### Diagnostic isolation

Each diagnostic is a self-contained class that:
1. Declares its inputs (variables, obs datasets)
2. Runs computation → returns results dict
3. Generates figures → returns (fig, metadata) pairs
4. The base class handles saving, metadata assembly, and output paths

This means diagnostics can be:
- Run independently (`python -m feather.diag.global_biases`)
- Exported as standalone Jupyter notebooks
- Tested with synthetic data
- Added without touching any other code (just `@register`)

### Pipeline stages are independent

```
diagnostics → figures/{diag}/*.png + *.json
analyze     → analysis/{diag}/*_analysis.json + synthesis.json
website     → site/index.html + {diag}.html + figures/ + static/
```

Each stage reads from the output of the previous stage via the filesystem. This means:
- You can re-run LLM analysis without re-running diagnostics
- You can rebuild the website without re-running analysis
- You can manually edit metadata JSONs and re-run downstream stages
- Multiple people can work on different stages

---

## Implementation Order (Next Steps)

After Phase 3 (done), the recommended order is:

1. **Phase 5** — LLM analysis pipeline (can test with Phase 3 figures)
2. **Phase 6** — Web dashboard (can test with Phase 3 figures + Phase 5 analyses)
3. **Phase 4** — CMIP6 loader (adds optional comparison to existing diagnostics)
4. **Phase 7** — Pipeline runner (ties everything together)
5. **Phase 8-9** — Additional diagnostics
6. **Phase 10** — Export / notebook generation
