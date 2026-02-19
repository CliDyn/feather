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

1. **Data shapes:** Model 2D data is `(time=300, values=12582912)` with 33 vars. Lazy dask loading via intake works well.
2. **Zonal mean on HEALPix:** 1° lat-band binning works on full-resolution data. At nside=8 (tests), need ≥10° bins to avoid empty polar bins. At nside=1024 (production), 1° is fine.
3. **ERA5 variable naming:** Variable names inside the NetCDF differ from the file-key names (e.g., file key `t2m` → variable inside is `T2M`). The ObsLoader `_find_variable()` handles this via case-insensitive fallback.
4. **cftime deprecation:** `xr.cftime_range()` is deprecated; use `xr.date_range(..., use_cftime=True)` in future test code.
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
- `nr.plot(data, lon, lat, ...)` — map plotting with NN interpolation, returns reusable `interpolator`
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
├── configs/
│   └── default.yaml                  # Default config with Levante paths
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Synthetic HEALPix fixtures (nside=8)
│   ├── test_config.py                # 3 tests
│   ├── test_loader.py                # 6 tests (+ 1 integration)
│   ├── test_obs.py                   # 5 tests (+ 1 integration)
│   ├── test_spatial.py               # 5 tests
│   ├── test_temporal.py              # 7 tests
│   ├── test_figure_metadata.py       # 18 tests
│   └── test_diag_base.py             # 18 tests
└── feather/
    ├── __init__.py                   # v0.1.0, exports FeatherConfig, DataLoader, ObsLoader
    ├── config.py                     # FeatherConfig dataclass + YAML loader
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                 # DataLoader: catalog + paths
    │   ├── obs.py                    # ObsLoader: config-driven obs access
    │   └── variables.py              # VarInfo + VARIABLE_REGISTRY (27 vars)
    ├── util/
    │   ├── __init__.py
    │   ├── spatial.py                # zonal_mean, global_mean, regional_mean
    │   ├── temporal.py               # climatology, seasonal, monthly, anomaly, annual_mean
    │   └── units.py                  # K↔°C, precip flux↔mm/day, Pa↔hPa
    ├── plot/
    │   ├── __init__.py
    │   ├── maps.py                   # plot_bias_map, plot_single_map (nereus)
    │   ├── lines.py                  # plot_timeseries, plot_seasonal_cycle, plot_zonal_profile
    │   └── styles.py                 # MODEL_COLORS, OBS_COLOR, apply_style
    ├── diag/
    │   ├── __init__.py               # Exports DiagnosticBase, register, build_metadata, etc.
    │   ├── base.py                   # DiagnosticBase ABC (compute → plot → run)
    │   ├── figure_meta.py            # save_figure_with_metadata, build_metadata
    │   └── registry.py               # @register, get_diagnostic, list_diagnostics
    └── export/
        └── __init__.py               # Placeholder
```

---

## Phase 3: First Diagnostics

### Step 3.1 — Global biases diagnostic

**File:** `feather/diag/global_biases.py`

Climatology bias maps (model − obs), 3-panel maps via `plot_bias_map()`.

For each model × variable:
1. Compute model climatology (time-mean over period)
2. Load matching obs climatology
3. Compute bias = model − obs
4. Generate 3-panel figure: Model | Observation | Bias
5. Compute summary statistics (global_mean_bias, RMSE, bias_range)
6. Save with full metadata JSON

**Figures produced per variable:**
- `{var}_annual_bias_{model}.png` + `.json` — annual mean bias map
- `{var}_seasonal_bias_{model}_{season}.png` + `.json` — seasonal bias maps (DJF, JJA)

**Important notes for implementation:**
- Obs data is on regular lat/lon; model data is on HEALPix. For bias computation, either regrid obs to HEALPix or compute global scalars separately. For map plotting, nereus handles the HEALPix → regular grid interpolation.
- The `plot_bias_map()` in `feather/plot/maps.py` currently has placeholder logic for the obs and bias panels — needs to be completed with actual regridding/plotting.

### Step 3.2 — Time series diagnostic

**File:** `feather/diag/timeseries.py`

Global-mean time series (model vs obs).

For each model × variable:
1. Compute area-weighted global mean at each timestep
2. Same for obs (regrid or already global)
3. Plot time series overlay
4. Save with metadata including correlation, trend

### Step 3.3 — Seasonal cycle diagnostic

**File:** `feather/diag/seasonal_cycle.py`

Monthly climatological cycle (Jan-Dec) comparison.

For each variable:
1. Compute monthly climatology (model + obs)
2. Plot 12-month cycle (all models + obs on same axes)
3. Save with metadata

---

## Phase 4: CMIP6 Loader

**File:** `feather/data/cmip6.py`

Simplified CMIP6 loader adapted from existing implementation at
`/home/a/a270088/PYTHON/DestinE/phase2/.../cmip6_loader.py`.

```python
class CMIP6Loader:
    """Load CMIP6 historical climatologies for comparison."""

    def __init__(self, config: FeatherConfig): ...

    def load_var(self, cmip6_var: str, model: str, experiment: str = "historical",
                 period: tuple[str, str] = None) -> xr.DataArray: ...

    def load_multi_model_mean(self, cmip6_var: str,
                               period: tuple[str, str] = None) -> xr.DataArray: ...

    def available_models(self, cmip6_var: str) -> list[str]: ...
```

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

After Phase 2 (done), the recommended order is:

1. **Phase 3** — First diagnostics (global_biases, timeseries, seasonal_cycle)
2. **Phase 5** — LLM analysis pipeline (can test with Phase 3 figures)
3. **Phase 6** — Web dashboard (can test with Phase 3 figures + Phase 5 analyses)
4. **Phase 4** — CMIP6 loader (adds optional comparison to existing diagnostics)
5. **Phase 7** — Pipeline runner (ties everything together)
6. **Phase 8-9** — Additional diagnostics
7. **Phase 10** — Export / notebook generation

### Important considerations for Phase 3

- **`plot_bias_map()` needs work.** The current implementation in `feather/plot/maps.py` has placeholder text for the obs and bias panels. To produce real bias maps, either:
  (a) Regrid obs to HEALPix and compute bias on native grid, then plot all three via `nr.plot()`, or
  (b) Regrid model to regular grid, compute bias on regular grid, and plot with `pcolormesh`.
  Option (a) is preferred as it keeps everything on HEALPix until the plotting step.
- **Global mean for obs vs model.** The model global mean uses `nr.surface_mean(data, area)` on HEALPix. For obs on regular lat/lon, use latitude-weighted mean (`cos(lat)` weighting).
- **Period alignment.** Model data is 1990-2014 (300 months). Obs may have different periods — always slice obs to match model period before comparison.
