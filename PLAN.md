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
│   ├── test_global_biases.py         # 23 tests (16 + 7 CMIP6)
│   ├── test_timeseries.py            # 15 tests (9 + 6 CMIP6)
│   ├── test_seasonal_cycle.py        # 14 tests (8 + 6 CMIP6)
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
    │   ├── spatial.py                # zonal_mean, global_mean, latlon_global_mean, compute_latlon_areas
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
| Spatial utils additions | `feather/util/spatial.py` (+`latlon_global_mean`, `compute_latlon_areas`) | Done |
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

1. **NN regridding via nereus `RegridInterpolator`.** Uses `nr.regrid()` on first call (builds KDTree + returns reusable interpolator), then `interpolator(data)` for subsequent fields. Includes influence radius masking (80km default) to prevent ocean→land value bleeding. At nside=1024 (12.6M source points), building the interpolator takes ~30s but each reuse is ~1s — critical since global_biases calls it 9+ times (3 models × annual + DJF + JJA).

2. **Pre-computed bias for plotting.** `plot_bias_map()` now accepts an optional `bias_data` parameter (xr.DataArray on regular grid). The diagnostic's `compute()` does the regridding + bias calculation, and `plot()` just visualizes. This keeps the plotting code simple and testable.

3. **Three-panel bias map uses nereus for all panels.** All three panels (model, obs, bias) use `nr.plot()` for consistent sizing and colorbar placement. Model and obs data are both on the nereus common grid in `compute()`, so all panels share the same lat/lon grid and can reuse the nereus interpolator.

4. **Shared colorbar ranges across models for cross-model comparison.** `compute()` collects all model fields, obs, and biases per period (annual, DJF, JJA) and computes shared `vmin`/`vmax` (2nd/98th percentile across all models + obs) and `bias_vmax` (98th percentile of |bias| across all models). These are stored in `results[var]["colorbar_ranges"]` and passed to `plot_bias_map()` via its `vmin`, `vmax`, and `bias_vmax` parameters. This ensures that when comparing model A vs model B for the same variable and period, the color scales are identical. The same principle applies to any diagnostic that produces per-model figures — shared ranges should be computed across models so figures are directly comparable.

5. **cftime → matplotlib compatibility.** Time series plot uses a `_to_plot_time()` helper that converts cftime datetime objects to pandas Timestamps via string parsing, since matplotlib cannot directly plot cftime dates.

6. **Diagnostics are configurable at construction.** All three accept `variables=`, `experiment=`, and `period=` overrides. The class-level defaults (`variables=["avg_2t"]`, `experiment="baseline_hist"`, `period=("1990","2014")`) cover the most common use case.

7. **Obs global mean uses nereus cell areas.** `latlon_global_mean()` computes proper cell areas via `nereus.mesh_from_arrays()` with LRU caching. Accepts optional pre-computed areas for CMIP6 `areacella`/`areacello`. Robust dim name detection supports `lat`/`latitude`/`nav_lat`/`y`/`rlat` and equivalents.

8. **Mock loaders in conftest.py.** `MockModelLoader` wraps `synth_healpix` dataset and returns it for any key. `MockObsLoader` wraps `synth_obs` dataset. This lets diagnostic tests run without real data or intake catalogs.

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
- **nereus `RegridInterpolator` builds KDTree once, reuses cheaply.** At nside=1024, the model grid has 12.6M points. Building the interpolator via `nr.regrid()` takes ~30s but each `interpolator(data)` call is ~1s. All models share the same HEALPix grid, so one interpolator serves all 3 models × all seasons. Never rebuild per field.
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
5. **All data passed to multi-panel plots must be on the same grid.** When `nr.plot()` reuses a shared interpolator across panels, it maps source array indices (not coordinates) to target pixels. If Panel 1 data is on a south→north grid (nereus) and Panel 2 data is on a north→south grid (ERA5), the shared interpolator flips Panel 2 upside down. Fix: regrid obs onto the common nereus target grid in `compute()` and store the common-grid obs in the results dict — never mix grids across panels.
6. **Shared colorbar ranges are essential for cross-model comparison.** Per-model auto-computed colorbars make it impossible to compare figures visually. Always compute shared `vmin`/`vmax`/`bias_vmax` across all models after the model loop, then pass explicitly to plotting functions. This applies to any diagnostic that produces separate figures per model.
7. **Influence radius must match source data density.** `nr.regrid()` masks target cells beyond the influence radius as NaN. Production HEALPix nside=1024 (~5 km spacing) works with the 80 km default. Test data (nside=8, ~815 km spacing) and coarse CMIP6 grids (5° in tests) need ~1000 km. Always make influence_radius configurable via config, not hardcoded.

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

6. **Regular grid meshgrid for nereus regrid.** CMIP6 data is on regular lat/lon grids (1D lat + 1D lon arrays of different length). Must meshgrid before passing to `nr.regrid()` which expects scattered points of equal length.

7. **Area weights cached per `{model}_{variant}_{table}` key.** `load_area()` looks for `areacella` (fx table, atmosphere) or `areacello` (Ofx table, ocean).

### Config changes

- `regrid_resolution`: 0.25 → 1.0 (sufficient for CMIP6 comparison)
- `ensemble_mode: "one_per_model"` added
- `influence_radius: 80000` in both cmip6 and nereus config sections (meters; prevents ocean→land bleeding)
- Models expanded from single `variant: str` to `variants: [list]` (7 models, up to 6 variants each)

### Verification results

```
pytest tests/test_cmip6.py -v -m "not integration"  → 47/47 passed
pytest tests/ -v -m "not integration"                → 142/142 passed (47 new + 95 existing)
```

CMIP6 comparison is always optional (`cmip6.enabled` in config). When enabled, diagnostics add:
- CMIP6 MMM line/panel to existing figures
- Additional metadata in `cmip6_info` field of the JSON sidecar

---

## Phase 4b: CMIP6 Integration into Diagnostics — COMPLETED

**Status:** All 3 diagnostics wired to CMIP6Loader (19 new tests, 161 total passing).

**Goal:** Wire the existing CMIP6Loader into the three diagnostics (global_biases, timeseries, seasonal_cycle) so each figure gains an optional CMIP6 multi-model mean (MMM) reference — providing historical context for how DestinE models compare to the broader CMIP6 ensemble.

**Guiding principle:** CMIP6 is always secondary context. It must never break a diagnostic when disabled or when data is missing. Every CMIP6 addition is guarded by `self.cmip6_enabled` and `None` checks.

### What already exists

- `DiagnosticBase` has `cmip6_loader` attribute and `cmip6_enabled` property
- `CMIP6Loader` has `load_mmm_for_model_var(model_var, period=..., season=...)` → `(mmm_da | None, info)`
- `CMIP6Loader` has `load_var_for_model_var(model_var, model, period=..., season=...)` for individual models
- `CMIP6_COLOR = "#888888"` in `plot/styles.py`
- `_build_metadata()` already accepts `cmip6_info` kwarg

### Step 4b.1 — `timeseries.py`: add CMIP6 MMM line

**Simplest diagnostic to start with — one line on existing axes.**

In `compute()`, after obs:
```python
# CMIP6 multi-model mean time series (optional)
cmip6_ts = None
cmip6_info = {}
if self.cmip6_enabled:
    mmm, info = self.cmip6_loader.load_mmm_for_model_var(
        var, period=self.period,
    )
    if mmm is not None:
        # MMM is a 2D field (lat, lon) — need monthly time series
        # Load per-month instead: iterate available models, compute
        # global-mean time series, average across models
        cmip6_ts = self._compute_cmip6_timeseries(var, var_info)
        cmip6_info = info
```

**Problem:** `load_mmm_for_model_var` returns a time-averaged 2D field, not a time series. For timeseries we need monthly global means per CMIP6 model, then ensemble-average.

**New helper `_compute_cmip6_timeseries()`:**
```python
def _compute_cmip6_timeseries(self, var, var_info):
    """Compute CMIP6 ensemble-mean global-mean time series."""
    member_series = []
    for model in self.cmip6_loader.models:
        da = self.cmip6_loader.load_var_for_model_var(
            var, model, period=self.period,
        )
        if da is None:
            continue
        # da is time-averaged (load_var computes .mean("time"))
        # → Need raw monthly data: call load_var directly with
        #   period but WITHOUT time-averaging
        ...
```

**Issue:** `load_var()` always time-averages. We need a lower-level method that returns the time series. Options:
1. Add `time_mean=False` kwarg to `CMIP6Loader.load_var()` — cleanest
2. Call `xr.open_zarr()` directly in the diagnostic — too coupled

**→ Add `time_mean` parameter to `CMIP6Loader.load_var()`:**
```python
def load_var(self, ..., time_mean: bool = True) -> xr.DataArray | None:
    ...
    if "time" in da.dims:
        ...
        if time_mean:
            da = da.mean("time")
    return da.compute()
```

Then in `_compute_cmip6_timeseries`:
```python
def _compute_cmip6_timeseries(self, var, var_info):
    """Compute CMIP6 MMM global-mean time series."""
    from feather.util.spatial import latlon_global_mean

    member_series = []
    models_used = []
    for model in self.cmip6_loader.models:
        vinfo = get_var(var)
        da = self.cmip6_loader.load_var(
            vinfo.cmip6_variable, model,
            table=vinfo.cmip6_table,
            period=self.period,
            time_mean=False,  # keep time dim
        )
        if da is None:
            continue
        # Area-weighted global mean per timestep
        area = self.cmip6_loader.load_area(model)
        ts = latlon_global_mean(da, area=area)
        member_series.append(ts)
        models_used.append(model)

    if not member_series:
        return None, {}

    # Align to common time axis, then ensemble mean
    aligned = xr.align(*member_series, join="inner")
    mmm_ts = sum(aligned) / len(aligned)
    info = {"n_members": len(models_used), "models_used": models_used}
    return mmm_ts, info
```

In `plot()`, add CMIP6 line:
```python
if "cmip6_ts" in vr and vr["cmip6_ts"] is not None:
    cmip6_ts = vr["cmip6_ts"]
    cmip6_time = _to_plot_time(cmip6_ts.time.values)
    ax.plot(
        cmip6_time, cmip6_ts.values,
        label="CMIP6 MMM", color=CMIP6_COLOR,
        linewidth=1.5, linestyle="--",
    )
```

In metadata: pass `cmip6_info=vr.get("cmip6_info")`.

### Step 4b.2 — `seasonal_cycle.py`: add CMIP6 MMM line

Same pattern as timeseries but with monthly climatology.

In `compute()`:
```python
cmip6_monthly = None
cmip6_info = {}
if self.cmip6_enabled:
    cmip6_ts, info = self._compute_cmip6_timeseries(var, var_info)
    if cmip6_ts is not None:
        cmip6_monthly = monthly_climatology(cmip6_ts)
        cmip6_info = info
```

In `plot()`:
```python
if "cmip6_monthly" in vr and vr["cmip6_monthly"] is not None:
    ax.plot(
        months, vr["cmip6_monthly"].values,
        marker="d", label="CMIP6 MMM", color=CMIP6_COLOR,
        linewidth=1.5, linestyle="--",
    )
```

**Note:** `_compute_cmip6_timeseries` can be shared — extract to base class or a helper module. Both timeseries and seasonal_cycle need the same CMIP6 time series; seasonal_cycle just applies `monthly_climatology()` on top.

### Step 4b.3 — `global_biases.py`: add CMIP6 MMM bias panel

**Most complex — adds a 4th panel to the existing 3-panel map.**

In `compute()`, after the model loop:
```python
cmip6_data = {}
cmip6_info = {}
if self.cmip6_enabled:
    mmm, info = self.cmip6_loader.load_mmm_for_model_var(
        var, period=self.period,
    )
    if mmm is not None:
        # mmm is on CMIP6 regridded grid (lat/lon)
        # Interpolate to common nereus target grid
        cmip6_regrid = mmm.interp(lat=target_lats, lon=target_lons)
        cmip6_bias = cmip6_regrid - obs_clim_common
        cmip6_data = {
            "regrid": cmip6_regrid,
            "bias": cmip6_bias,
            "bias_gmean": float(latlon_global_mean(cmip6_bias).values),
        }
        cmip6_info = info

        # Include CMIP6 in shared colorbar ranges
        # (recompute with CMIP6 fields included)
```

In `plot()`, extend the figure:
- If CMIP6 data exists, create a **4-panel** layout: Model | Obs | Bias | CMIP6 Bias
- Or: overlay CMIP6 bias contours on the bias panel
- **Recommended:** 4th panel. Keeps it clean and comparable.

**Approach for 4-panel:** Add `ncols` parameter to `plot_bias_map`, or create separate CMIP6 bias figure, or extend figure manually.

**Simplest approach:** Generate a separate CMIP6 bias map figure per variable (not per model) since CMIP6 MMM is model-independent:
```python
if cmip6_data:
    fig_c, _ = plot_bias_map(
        cmip6_data["regrid"], obs_clim,
        bias_data=cmip6_data["bias"],
        title=f"{var_info.long_name} Annual Mean — CMIP6 MMM",
        model_title="CMIP6 MMM",
        cmap=var_info.cmap, units=var_info.units,
        vmin=ann_cb["vmin"], vmax=ann_cb["vmax"],
        bias_vmax=ann_cb["bias_vmax"],
    )
    meta_c = self._build_metadata(
        title=f"{var_info.long_name} Annual Bias — CMIP6 MMM",
        figure_id=f"{var}_annual_bias_cmip6_mmm",
        models=["CMIP6 MMM"],
        cmip6_info=cmip6_info,
        ...
    )
    figures.append((fig_c, meta_c))
```

Uses the **same shared colorbar ranges** as the DestinE model figures → directly comparable.

### Step 4b.4 — `CMIP6Loader.load_var()`: add `time_mean` parameter

```python
def load_var(self, ..., time_mean: bool = True) -> xr.DataArray | None:
```

Default `True` preserves backward compatibility. When `False`, returns the full time series after period/season filtering and calendar normalization.

### Step 4b.5 — Shared CMIP6 time series helper

Extract `_compute_cmip6_timeseries()` as a method on `DiagnosticBase` (or a standalone utility) since both timeseries and seasonal_cycle need it:

```python
# In DiagnosticBase or a mixin:
def _cmip6_global_mean_timeseries(self, var):
    """CMIP6 ensemble-mean global-mean monthly time series."""
```

### Step 4b.6 — Config and test updates

- `configs/default.yaml`: change `cmip6.enabled: true` to test
- `tests/conftest.py`: `MockCMIP6Loader` already has `load_var`, `load_mmm`, `load_area` — add `time_mean` support
- New tests per diagnostic: verify CMIP6 data appears in results when enabled, is absent when disabled
- Verify shared colorbar ranges include CMIP6 fields when present

### Step 4b.7 — Metadata enrichment

When CMIP6 is included, add to metadata sidecar:
```json
{
  "cmip6_info": {
    "n_members": 7,
    "models_used": ["MIROC6/r1i1p1f1", "CESM2/r10i1p1f1", ...],
    "ensemble_mode": "one_per_model"
  }
}
```

This is already supported by `_build_metadata(cmip6_info=...)`.

### Implementation order

1. `CMIP6Loader.load_var()` — add `time_mean=False` support
2. `MockCMIP6Loader` — update for `time_mean`
3. `timeseries.py` — add CMIP6 MMM line (simplest, proves the pattern)
4. `seasonal_cycle.py` — add CMIP6 MMM line (reuses same helper)
5. `global_biases.py` — add CMIP6 MMM bias map (separate figure, shared colorbar)
6. Tests for all three diagnostics with CMIP6 enabled/disabled
7. Verify with real data on compute node

### Design decisions confirmed

- **CMIP6 bias map as separate figure (not 4th panel).** Keeps `plot_bias_map` simple. The CMIP6 MMM is the same for all DestinE models, so one CMIP6 figure per variable (not per model) is natural. Uses shared colorbar ranges for comparability.
- **CMIP6 MMM line style:** dashed gray (`--`, `CMIP6_COLOR`) to visually distinguish from solid model lines and thick obs line.
- **Missing CMIP6 data:** silently skip — no error, no empty panel. `cmip6_info` in metadata is `None` or absent.
- **Shared helper `_cmip6_global_mean_timeseries()` on `DiagnosticBase`.** Both timeseries and seasonal_cycle use it. Loads raw monthly data (`time_mean=False`), computes per-model area-weighted global mean, aligns on common time axis, averages across models.

### What was built

| Module | File | Change |
|--------|------|--------|
| CMIP6Loader | `feather/data/cmip6.py` | Added `time_mean: bool = True` kwarg to `load_var()` |
| DiagnosticBase | `feather/diag/base.py` | Added `_cmip6_global_mean_timeseries()` shared helper |
| TimeseriesDiag | `feather/diag/timeseries.py` | CMIP6 MMM line in compute() + plot() + metadata |
| SeasonalCycleDiag | `feather/diag/seasonal_cycle.py` | CMIP6 MMM line in compute() + plot() + metadata |
| GlobalBiases | `feather/diag/global_biases.py` | CMIP6 MMM bias maps (annual + DJF + JJA), included in shared colorbar ranges |
| MockCMIP6Loader | `tests/conftest.py` | Updated `load_var()` for `time_mean` |
| Tests | `tests/test_timeseries.py` | +6 CMIP6 tests |
| Tests | `tests/test_seasonal_cycle.py` | +6 CMIP6 tests |
| Tests | `tests/test_global_biases.py` | +7 CMIP6 tests |

### Verification results

```
pytest tests/ -v -m "not integration"  → 161/161 passed (19 new + 142 existing)
```

### Key implementation notes

1. **`CMIP6Loader.load_var(time_mean=False)` returns full time series.** Default `True` preserves backward compatibility. When `False`, returns the DataArray after period/season filtering and calendar normalization but without `.mean("time")`.

2. **`_cmip6_global_mean_timeseries()` on DiagnosticBase.** Iterates configured CMIP6 models (one variant per model by default), loads raw monthly data, computes `latlon_global_mean()` per timestep (area-weighted via `load_area()`), aligns on common time axis, averages across models. Returns `(mmm_ts, info)`.

3. **GlobalBiases CMIP6 integration.** After the model loop, loads CMIP6 MMM for annual and seasonal periods via `load_mmm_for_model_var()`. Interpolates to the common nereus target grid via `.interp()`. Computes bias against obs. CMIP6 fields are included in `_compute_colorbar_ranges()` so all figures (DestinE + CMIP6) share identical color scales.

4. **CMIP6 bias maps are separate figures (one per variable per period).** Not 4th panels. The CMIP6 MMM is model-independent, so one figure per variable is natural. Figure IDs: `{var}_{period}_bias_cmip6_mmm`.

5. **All CMIP6 additions are guarded.** `self.cmip6_enabled` checks + `None` checks ensure diagnostics work identically when CMIP6 is disabled. Existing tests (minimal_config with `cmip6.enabled: False`) verify this.

### Post-implementation fixes and improvements

1. **areacella alignment bug (critical).** When `latlon_global_mean(da, area=area)` receives an xr.DataArray from `load_area()`, coordinate misalignment between areacella and data variables causes `da.weighted(area)` to silently produce wrong results (cold-biased CMIP6 MMM). **Fix:** Added `_align_area()` static method to `DiagnosticBase` that converts areacella to numpy array, letting `latlon_global_mean` re-wrap with the data's own coordinates. This ensures positional (not coordinate-based) alignment.

2. **Interpolator cache locality bug.** `interp_cache` in `load_multi_model_mean()` was a local dict, rebuilt from scratch on each of the 3 calls (annual, DJF, JJA). Building a KDTree for a CMIP6 model grid (~65K points at 1°) takes a few seconds each, so rebuilding 3× per model was wasteful. **Fix:** Promoted to `self._interp_cache` instance attribute on `CMIP6Loader`, persisting across calls. Cache key: `f"{model}_{table}_{resolution}"`.

3. **CMIP6 model list updated to validated 12-model ensemble.** Original config had 7 models with guessed variants. Updated to user's validated 12-model list: CanESM5, MPI-ESM1-2-LR, GISS-E2-1-G, IPSL-CM6A-LR, ACCESS-ESM1-5, EC-Earth3, CNRM-CM6-1, AWI-CM-1-1-MR, CNRM-ESM2-1, FGOALS-g3, INM-CM5-0, MRI-ESM2-0 (5 variants each).

4. **Verbose logging (`-v` / `-vv` flag).** All 4 runner scripts (`run_global_biases.py`, `run_timeseries.py`, `run_seasonal_cycle.py`, `run_all.py`) now accept `-v` for INFO-level and `-vv` for DEBUG-level logging. Informative `logger.info()` calls added throughout: diagnostic compute steps, model/obs/CMIP6 loading, interpolator building, regridding (cached vs new), global mean values, and MMM statistics. Uses Python `logging` module with timestamped format.

---

## Phase 5: LLM Analysis Pipeline — COMPLETED

**Goal:** Automatically analyze each diagnostic figure using an LLM (Gemini), producing structured scientific interpretations that feed the dashboard.

**Status:** All steps implemented and verified. 47 unit tests passing.

### What was built

| Module | File | Status |
|--------|------|--------|
| Config | `feather/config.py` — added `llm: dict` field | Done |
| Config | `configs/default.yaml` — added `llm:` section | Done |
| Schemas | `feather/llm/schemas.py` (FigureAnalysis, DiagnosticSynthesis) | Done |
| Prompts | `feather/llm/prompts.py` (system + user prompts for analysis/synthesis) | Done |
| Analyzer | `feather/llm/analyzer.py` (FigureAnalyzer: discover, analyze, synthesize) | Done |
| CLI | `scripts/run_analysis.py` (argparse runner) | Done |
| Package | `feather/llm/__init__.py`, updated `feather/__init__.py` | Done |
| Deps | `pyproject.toml` — added pydantic, google-generativeai | Done |
| Tests | `tests/test_llm.py` (47 tests, all mocked) | Done |

### Key design decisions

1. **Per-purpose LLM config** — `llm.figure_analysis` uses Gemini; future `llm.report_generation` can use OpenAI
2. **`google.generativeai` with `genai.upload_file()`** for figure vision analysis
3. **Prompts focus on model evaluation** (bias patterns, model-obs agreement) not climate projections
4. **`_parse_json_response` handles LaTeX escapes** — Gemini uses `\Delta` etc. in scientific text
5. **Incremental runs** via `skip_existing` — don't re-analyze already-analyzed figures
6. **No data processing dependency** — works entirely on already-generated PNG+JSON figures

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

### Cross-model comparability

All diagnostics that produce per-model figures must use **shared colorbar ranges** across models for the same variable and time period. This is a core design principle — the primary purpose of feather is comparing models against each other and against observations.

**Implementation pattern:** `compute()` first loops over all models to collect regridded fields and biases, then computes shared ranges (2nd/98th percentile for field panels, 98th percentile of |bias| for bias panels) across all models. These ranges are stored in the results dict and passed to plotting functions. This ensures identical color scales when placing figures side by side.

Applies to:
- **global_biases**: `colorbar_ranges` per period (annual, DJF, JJA) with `vmin`, `vmax`, `bias_vmax`
- **Future map diagnostics**: same pattern — compute shared ranges after the model loop
- **Line plots** (timeseries, seasonal_cycle): already overlay all models on shared axes — inherently comparable

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
