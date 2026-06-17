# Feather

Lightweight climate model evaluation framework supporting multiple high-resolution model sets.

Feather compares high-resolution climate models against observations (ERA5, CERES, EN4, ESA-CCI, OSI-SAF, PIOMAS/GIOMAS, MSWEP, Berkeley Earth) and a CMIP6 multi-model mean. It generates diagnostic figures, LLM-analyzed results, a LaTeX report, and a static web dashboard.

## Supported model sets

| Model set | Models | Resolution | Grid | Data backend |
|-----------|--------|-----------|------|-------------|
| **DestinE** | IFS-FESOM, IFS-NEMO, ICON | ~5 km | HEALPix | intake catalogs |
| **EERIE Ensemble** | IFS-FESOM2-SR, IFS-NEMO-ER, ICON-ESM-ER, HadGEM3-GC5 | ~10-20 km atm, ~5-10 km ocean | 0.25° lat/lon | CMOR directory tree |
| **EERIE multi-member** | IFS-FESOM2-SR (r1–r3), IFS-NEMO-ER (r1–r3), ICON-ESM-ER, HadGEM3-GC5 | ~10-20 km atm, ~5-10 km ocean | 0.25° lat/lon | CMOR + kerchunk parquet |
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

**nereus** must be installed directly from GitHub (not available on PyPI/conda-forge):

```bash
pip install git+https://github.com/koldunovn/nereus.git@0.4.1
```

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

# EERIE Ensemble (4 models)
feather --config configs/eerie.yaml -v

# EERIE — 3 IFS-NEMO-ER members + IFS-FESOM2-SR + ICON-ESM-ER + HadGEM3-GC5
feather --config configs/eerie_ifsnemo_members.yaml -v

# EERIE — all 8 members (3×IFS-FESOM2-SR + 3×IFS-NEMO-ER + ICON-ESM-ER + HadGEM3-GC5)
feather --config configs/eerie_all_members.yaml -v

# EERIE Tropical Nights climate change signal (SSP2-4.5)
feather --config configs/eerie_climchange_tn.yaml \
        --steps diagnostics \
        --diagnostics tropical_nights_change -v

# EERIE Heatwave indices climate change signal (SSP2-4.5)
feather --config configs/eerie_climchange_hw.yaml \
        --steps diagnostics \
        --diagnostics heatwave_change -v

# TerraDT baseline evaluation
feather --config configs/terradt.yaml -v

# DestinE Added Value 1990–2025 (hist + ssp3-7.0 stitched; timeseries to 2049/2044)
feather --config configs/destine_added_value.yaml -v

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

# Also write per-source diagnostic fields to NetCDF (obs, each model,
# CMIP6/HighResMIP MMM). Filenames include the analysis period and
# already-present files are skipped, so it is safe to re-run.
feather --save-netcdf --diagnostics global_biases temperature_berkeley -v

# Custom time period
feather --period 1990 2014 -v

# Compile LaTeX report to PDF
feather --steps report --compile-pdf -v

# Also works as a module
python -m feather --config configs/eerie.yaml -v
```

### NetCDF export (`--save-netcdf`)

With `--save-netcdf`, diagnostics also write the per-source fields underlying
their figures — the observations, each evaluated model, and each benchmark MMM
(CMIP6, HighResMIP) — to NetCDF under `{output_dir}/netcdf/{diagnostic}/`. Each
filename carries the analysis period (e.g. `tas_annual_1980-2014.nc`), and files
already present are skipped, so the flag is incremental and safe to re-run.

Supported by `global_biases`, `temperature_berkeley`, `precipitation_mswep`,
`global_trends`, `climate_variability`, `ocean_sst`, `ocean_en4`,
`radiation_budget`, `sea_ice`, `timeseries`, `seasonal_cycle`, and
`teleconnections`. The obs-comparison and extremes/classification diagnostics
write NetCDF as part of their normal operation. A new diagnostic can opt in by
following the template in `NEW_DIAGNOSTIC_SPEC.md`.

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

Feather provides 19 registered diagnostics across atmosphere, ocean, cryosphere, extremes, cross-domain evaluation, and model intercomparison:

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

### Extremes

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `tropical_nights` | `TropicalNightsDiag` | Berkeley Earth Land TMIN | Annual tropical nights count (TN > 20 °C) per grid point; land-only bias map vs Berkeley Earth |
| `tropical_nights_change` | `TropicalNightsChangeDiag` | Berkeley Earth Land TMIN | Climate change signal in TN under SSP2-4.5: [Reference \| Future \| Change] maps per model, land-mean time series hist+SSP stitched, mean Tmin bias map |
| `heatwave` | `HeatwaveDiag` | Berkeley Earth Land TMAX | Five TX90 heatwave indices (HWN, HWF, HWD, HWM, HWA): climatological maps + annual time series; TMAX bias map |
| `heatwave_change` | `HeatwaveChangeDiag` | Berkeley Earth Land TMAX | Climate change signal in five TX90 heatwave indices under SSP2-4.5: [Reference \| Future \| Change] maps per model per index, land-mean time series hist+SSP stitched, mean Tmax bias map |

The **Tropical Nights Index** (TN20) counts nights per year where daily minimum temperature exceeds 20 °C. Requires daily `tasmin` (CMOR) or kerchunk-parquet mn2t24 store. Per-model NC checkpoints are written to `{output_dir}/tropical_nights/`.

The **Tropical Nights Climate Change Signal** (`tropical_nights_change`) compares mean annual TN between a historical reference period (default 1981–2000) and a future period (default 2031–2050) under SSP2-4.5. It is purpose-built for experiments that span multiple data backends (e.g. CMOR r1 + kerchunk-native r2/r3):
- **Group A** — Combined map: one row per model × three columns [Reference climatology | Future climatology | Change (Future − Reference)]. Sequential YlOrRd colormap for absolute counts; diverging RdYlBu_r for the change column. Models whose SSP run ends before the future period show a placeholder in columns 2–3.
- **Group B** — Stitched land-mean time series: historical segment (hist-1950) joined to SSP2-4.5 at 2015 with a dashed vertical line; reference and future windows highlighted; dashed black line shows Berkeley Earth approximate observed TN (months with mean TMIN > 20 °C weighted by days-in-month).
- **Group C** — Mean daily Tmin bias map vs Berkeley Earth Land TMIN over the reference period (skipped when BE data is unavailable).

Configuration lives under `project.climate_change` (see `configs/eerie_climchange_tn.yaml`). Each model declares its hist/future data source (`cmor` or `kerchunk_native`) and optionally a `future_only_to` year. Land masking is applied per-model using the Berkeley Earth land mask (NaN over ocean). Per-model NC checkpoints written to `{output_dir}/tropical_nights_change/`.

The **Heatwave diagnostic** computes five interconnected indices using the TX90 method (tasmax > DOY-specific 90th-percentile threshold, runs ≥ 3 consecutive days):
- **HWN** — heatwave number (events per year)
- **HWF** — heatwave frequency (heatwave days per year)
- **HWD** — heatwave duration (length of longest event, days)
- **HWM** — heatwave magnitude (mean tasmax on heatwave days, °C)
- **HWA** — heatwave amplitude (peak tasmax on heatwave days, °C)

Summer windows are hemisphere-aware (NH: May–Sep; SH: Nov–Mar). Requires daily `tasmax` (CMOR `day/tasmax` table or kerchunk-parquet mx2t24 store). Per-model NC checkpoints stored in `{output_dir}/heatwave/`.

The **Heatwave Climate Change Signal** (`heatwave_change`) compares all five TX90 indices between a historical reference period (default 1981–2000) and a future period (default 2031–2050) under SSP2-4.5. The T90 threshold is computed once from the reference period and applied to both periods. One figure set is produced per index:
- **Group A** — Combined change map: one row per model × three columns [Reference climatology | Future climatology | Change (Future − Reference)]. Sequential colormap for absolute values; diverging RdYlBu_r for the change column. Models whose SSP run ends before the future period show a placeholder in columns 2–3.
- **Group B** — Stitched land-mean time series: historical segment joined to SSP2-4.5 at 2015 with a dashed vertical line; reference and future windows highlighted.
- **Group C** — Mean daily Tmax bias map vs Berkeley Earth Land TMAX over the reference period (skipped when BE data is unavailable).

Configuration lives under `project.climate_change` (see `configs/eerie_climchange_hw.yaml`). Each model declares its hist/future data source (`cmor` or `kerchunk_native`). Land masking is applied per-model using the Berkeley Earth land mask. Per-model NC checkpoints written to `{output_dir}/heatwave_change/`.

### Climate classification

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `climate_classification` | `KTClimateClassification` | ERA5, Berkeley Earth HR + MSWEP | Köppen–Trewartha (KT14) land climate type per grid cell; discrete classification maps, % land-area bar chart, and summary table per source |

The **Köppen–Trewartha Climate Classification** (`climate_classification`) classifies the monthly climatology of `tas` and `pr` into the 14 KT types (Ar, Aw, As, BW, BS, Cs, Cw, Cf, Do, Dc, Eo, Ec, Ft, Fi) using the Trewartha dryness threshold `R = 2.3·Tann − 0.64·Pw + 41` (cm) and the F→E→B→D→C→A priority order. The classification is restricted to land via the Berkeley Earth land mask and computed on a common 0.25° grid so per-type area fractions are directly comparable. Each model is classified from its own `tas`+`pr`; observations come from ERA5 (`t2m`+`tp`) and a Berkeley Earth HR + MSWEP combination; the CMIP6 multi-model mean is included when enabled. Four figures are produced:

- **Classification maps** — one discrete KT panel per source, ordered ERA5, Berkeley Earth HR + MSWEP, the EERIE models, then the CMIP6 multi-model mean.
- **Ensemble comparison** — the observational datasets alongside the **mean** and **median** of each model family. Families are defined by a per-model `ensemble` label in the config (e.g. `EERIE`, or `CORDEX`/`CMIP5`/`CMIP6` for the Africa config), and each ensemble is classified from the per-cell, per-month mean/median of its members' `tas` and `pr` climatologies (not by averaging KT codes).
- **Bar chart** and **summary table** — % land area per KT type per source, including every family's ensemble mean/median and the CMIP6 MMM (each source sums to 100 %).

**Regional mode (Africa).** Setting `project.region` (e.g. `africa`) restricts the classification to a regional bounding box, builds the common grid on that box (−180..180 longitude convention so western Africa is not dropped), handles rotated-pole (2-D lat/lon) sources, and draws zoomed PlateCarree maps. `configs/africa_climate.yaml` classifies, on a common 0.25° Africa grid for the 1981–2010 reference: ERA5/obs, **CORDEX-CORE AFR-22** (10 GCM-driven members + 4 ERAINT evaluation runs; `CORDEXLoader` stitches historical 1981–2005 + rcp85 2006–2010 and falls back from `mon` to `day`-resampled-to-monthly), **CMIP5** (5 driving GCMs, 1st member, historical+rcp85 stitch via `CMIP5Loader`), and **CMIP6** (5 driving-GCM successors, historical, via the NetCDF-tree `CMIP6NCLoader`). Coarse GCMs use a 250 km regrid influence-radius floor.

**Climate shifts at a warming level (`climate_shifts`, Phase 2).** Compares the reference classification with the classification at a target global warming level (default **+2 °C above the 1850–1900 pre-industrial baseline**, configured under `project.gwl`). The 30-year window is found **per GCM** from its own global-mean temperature (`feather/util/gwl.py`; CMIP5→rcp85, CMIP6→ssp585), and CORDEX regional models inherit the window of their **driving GCM** (`driving_gcm` in the config). Per model family it produces: **shift maps** [reference | future | change], a **reference-vs-future area bar**, a **net gain/loss** summary (Δ % land area per type), and a **transition matrix** (reference type → warming-level type, area-weighted). Reference and future KT-code NetCDFs (per source and per family ensemble mean) and an area-shift CSV are written to `{output_dir}/climate_shifts/`. Run with `--diagnostics climate_shifts` on the same `africa_climate.yaml`.

Outputs written to `{output_dir}/climate_classification/` for later regional analysis: a `{source}_clim_{period}.nc` per source with the monthly `tas` (°C) and `pr` (mm/month) climatology — both a full-global field and a land-only version, on the common grid — for ERA5, Berkeley Earth HR, MSWEP, every EERIE model, every CMIP6 model, the EERIE ensemble mean/median, and the CMIP6 MMM; a `{source}_kt_{period}.nc` with the integer `kt_code` field per classified source; and a `kt_area_percent` CSV. See `feather/util/koeppen_trewartha.py` for the pure, vectorised classifier.

### Cross-domain

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `precipitation_mswep` | `PrecipitationMSWEP` | MSWEP v2.8 | Precipitation bias maps (absolute + relative), time series, seasonal cycle, zonal mean, intensity PDF |
| `temperature_berkeley` | `TemperatureBerkeley` | Berkeley Earth | T2m bias maps, warming trend maps (global + polar), Taylor diagram |
| `teleconnections` | `TeleconnectionDiag` | ERA5 | Climate variability modes (ENSO, NAO, SAM, AO, IOD, PDO, QBO): index time series, spatial patterns, power spectra, seasonal variance |
| `obs_comparison` | `ObsComparisonDiag` | ERA5 + Berkeley Earth | ERA5 vs Berkeley Earth T2m trend and bias comparison across two periods (1980–2014, 1980–2024) |
| `precip_obs_comparison` | `PrecipObsComparisonDiag` | ERA5 + MSWEP v2.8 | ERA5 vs MSWEP precipitation trend and bias comparison across two periods (1980–2014, 1980–2023) |

### Model intercomparison

| Diagnostic | Class | Observation | What it produces |
|---|---|---|---|
| `added_value` | `AddedValueDiag` | ERA5 / Berkeley Earth / MSWEP | Dosio et al. (2015) Added Value: EERIE ensemble vs CMIP6 MMM — ensemble summary maps + per-model panels |

**Added Value** (AV) quantifies where the EERIE ensemble outperforms the CMIP6 multi-model mean relative to observations. AV ∈ [-1, 1]: AV > 0 means EERIE reduces squared error vs CMIP6 MMM at that grid point. Two figures per period (annual, DJF, JJA): ensemble mean/median summary and one panel per individual EERIE and CMIP6 model.

All diagnostics support:
- `variables=["tas", ...]` — filter which variables to evaluate (CMOR canonical names)
- `experiment="hist-1950"` — experiment identifier (from config by default)
- `period=("1980", "2014")` — time period for climatologies (from config by default)
- `cmip6_individual=True` — individual CMIP6 model lines/biases alongside MMM

## Configuration

Feather uses YAML configuration files. Fourteen configs are provided:

| Config | Model set | Data source | Comparison type |
|--------|----------|------------|-----------------|
| `configs/default.yaml` | DestinE (3 models) | intake catalogs | `multi_model` |
| `configs/eerie.yaml` | EERIE Ensemble (4 models) | CMOR directory tree | `multi_model` |
| `configs/eerie_ifsnemo_members.yaml` | EERIE — 3 IFS-NEMO-ER + 3 other models | CMOR (hist-1950 + hist-1975) | `multi_model` |
| `configs/eerie_all_members.yaml` | EERIE — 8 models (3×FESOM2 + 3×NEMO + 2) | CMOR + kerchunk parquet | `multi_model` |
| `configs/eerie_climchange_tn.yaml` | EERIE — IFS-FESOM2-SR (r1–r3) + ICON-ESM-ER, SSP2-4.5 TN signal | CMOR + kerchunk native | `multi_model` |
| `configs/eerie_climchange_hw.yaml` | EERIE — IFS-FESOM2-SR (r1–r3) + ICON-ESM-ER, SSP2-4.5 heatwave signal | CMOR + kerchunk native | `multi_model` |
| `configs/himansu_319.yaml` | IFS-FESOM T319 | per-year NetCDF | `single_model` |
| `configs/tco_grib.yaml` | IFS-FESOM TCO399/TCO319 | GRIB files | `resolution_sensitivity` |
| `configs/destine_ifs_fesom.yaml` | IFS-FESOM only | intake catalogs | `single_model` |
| `configs/destine_ifs_nemo.yaml` | IFS-NEMO only | intake catalogs | `single_model` |
| `configs/destine_icon.yaml` | ICON only | intake catalogs | `single_model` |
| `configs/terradt.yaml` | TerraDT (3 models) | intake catalogs | `baseline_evaluation` |
| `configs/destine_added_value.yaml` | DestinE (3 models), 1990–2025 | intake catalogs (hist + ssp3-7.0 stitched) | `multi_model` |
| `configs/ifs_fesom_combined.yaml` | IFS-FESOM multi-res | mixed (catalog + GRIB) | `resolution_sensitivity` |

### Config format

The `models` key determines the config format:
- **List** (`models: [ifs-fesom, ifs-nemo, icon]`) — legacy DestinE format, auto-populates HEALPix grid defaults
- **Dict** (`models: {ModelA: {institution: ..., grids: ...}}`) — structured format with per-model configuration

`data_source.type` selects the loading backend: `"cmor"`, `"destine_catalog"`, `"netcdf_healpix"`, `"grib"`, or `"kerchunk_parquet"`.  Individual models can override the global backend with a per-model `data_source_type` key (used by `eerie_all_members.yaml` to mix CMOR and kerchunk parquet in the same run).

### Structured config example (EERIE-style)

```yaml
project:
  name: "EERIE"
  description: "EERIE Ensemble evaluation"
  experiment: "hist-1950"
  period: ["1980", "2014"]
  resolution: "high-resolution (~10-20 km atm, ~5-10 km ocean)"
  comparison_type: "multi_model"         # multi_model | resolution_sensitivity | single_model | baseline_evaluation
  comparison_description: ""             # optional free-text for LLM prompt framing

data_source:
  type: "cmor"                           # cmor | destine_catalog | netcdf_healpix | grib | kerchunk_parquet
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

The same pattern is used in `eerie_all_members.yaml` to mix CMOR (r1 members) with kerchunk parquet (r2/r3 members):

```yaml
data_source:
  type: "cmor"
  root: "/path/to/CMOR/EERIE/HighResMIP"

models:
  IFS-FESOM2-SR:                         # r1: standard CMOR path
    institution: AWI
    variant: r1i1p1f1
    ...
  IFS-FESOM2-SR-r2:                      # r2: kerchunk parquet reference store
    institution: AWI
    variant: r2i1p1f1
    data_source_type: "kerchunk_parquet"
    data_root: "/path/to/kerchunks/IFS-FESOM2/hist-1950"
    ...
```

### Per-model overrides

`ModelConfig` supports:
- `data_root` — custom data path (e.g., HadGEM3 outside shared CMOR root)
- `grid_label` — CMOR grid label override (default `"gr"`)
- `variable_aliases` — CMOR var → on-disk name mapping (e.g., `thetao: thetao-con`)
- `scale_factors` — per-variable multiplier (e.g., `clt: 100` for fraction → %)
- `absolute_salinity: true` — enable SA→SP salinity conversion (NEMO-based models)

### Experiment stitching (extended time periods)

Some evaluation windows are not covered by a single experiment. For DestinE, the historical `baseline_hist` run ends in 2014 while the `projections_ssp3-7.0` run continues to 2049 (IFS-FESOM / IFS-NEMO) or 2044 (ICON). To analyse e.g. **1990–2025**, list the experiments to concatenate along the time axis:

```yaml
project:
  period: ["1990", "2025"]
  # Stitched along time: historical baseline + ssp3-7.0 projection.
  experiments: ["baseline_hist", "projections_ssp3-7.0"]
  # Optional: let the timeseries diagnostic extend to each model's native end
  # (IFS-FESOM/IFS-NEMO → 2049, ICON → 2044) while every other diagnostic
  # uses `period` above. Defaults to `period` when omitted.
  timeseries_period: ["1990", "2050"]

cmip6:
  # Stitch CMIP6 symmetrically so added-value / bias diagnostics cover the
  # same window. Models lacking ssp370 for their variant fall back to
  # historical-only (end at 2014).
  experiments: ["historical", "ssp370"]
```

Segments are concatenated, sorted, and de-duplicated on overlapping months (the first experiment in the list wins). Models that did not run a listed experiment are skipped gracefully. Period slicing happens after concatenation, so a single stitched series serves both the analysis window and the extended timeseries. Omitting `experiments` preserves the legacy single-experiment behaviour, so existing configs are unaffected. See `configs/destine_added_value.yaml` for a complete example.

> **Note:** an ssp370 variant label may differ from its historical counterpart (e.g. CanESM5 ssp370 is `r1i1p2f1`, not `r1i1p1f1`); the configured variant must exist for both experiments, or the model reverts to historical-only.

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

Feather supports six data loading backends:

| Backend | Class | Config type | Grid |
|---------|-------|------------|------|
| DestinE intake catalogs | `MultiCatalogLoader` | `destine_catalog` | HEALPix (zarr) |
| CMOR directory tree | `CMORLoader` | `cmor` | regular lat/lon (NetCDF) |
| Per-year NetCDF | `NetCDFLoader` | `netcdf_healpix` | HEALPix (NetCDF) |
| GRIB files | `GRIBLoader` | `grib` | regular lat/lon (GRIB) |
| Kerchunk parquet reference stores | `KerchunkParquetLoader` | `kerchunk_parquet` | regular lat/lon (via fsspec) |
| Multi-source | `CompositeModelLoader` | mixed | per-model |

`KerchunkParquetLoader` reads IFS-FESOM2 ensemble members stored as kerchunk parquet reference files pointing to raw GRIB/FESOM output. Two store layouts are supported:
- **Regridded 0.25°** (`gr025`): `{root}/{variant}/atmos/gr025/*.parq` — atmosphere 2D/3D monthly; `ocean/gr025/2D_daily_avg_*.parq` ocean 2D daily. Used by `eerie_all_members.yaml`.
- **Native grid** (`native`): `{root}/{variant}/atmos/native/2D_daily_native_atmos_min.parq` / `…_max.parq` — daily min/max on the native HEALPix grid. Selected via `data_source_type: kerchunk_native` in the model config; used by `eerie_climchange_tn.yaml` for the `tropical_nights_change` diagnostic.

Requires `kerchunk`, `fastparquet`, and `fsspec` in the Python environment.

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
    obs_comparison/
    precip_obs_comparison/
    added_value/
    tropical_nights/
    tropical_nights_change/
    heatwave/
    heatwave_change/
  tropical_nights/                    # NC checkpoints (outside figures tree)
    {model}_tropical_nights_tn20_{start}_{end}.nc
  tropical_nights_change/             # NC checkpoints (outside figures tree)
    {model}_tn_hist_{start}_{end}.nc
    {model}_tn_ssp_{start}_{end}.nc
  heatwave/                           # NC checkpoints (outside figures tree)
    {model}_heatwave_tx90_{start}_{end}.nc
  heatwave_change/                    # NC checkpoints (outside figures tree)
    {model}_hw_hist_{start}_{end}.nc
    {model}_hw_ssp_{start}_{end}.nc
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
    kerchunk_loader.py   # KerchunkParquetLoader (kerchunk parquet reference stores)
    composite_loader.py  # CompositeModelLoader (multi-source routing)
    obs.py               # ObsLoader (observations from config)
    cmip6.py             # CMIP6Loader (multi-model mean from zarr)
    variables.py         # VARIABLE_REGISTRY (36 vars, CMOR canonical names)
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
    obs_comparison.py     # ERA5 vs Berkeley Earth T2m trends and biases
    precip_obs_comparison.py # ERA5 vs MSWEP precipitation trends and biases
    added_value.py        # Added Value: EERIE ensemble vs CMIP6 MMM (Dosio et al. 2015)
    tropical_nights.py    # Tropical Nights Index (TN > 20 °C, daily tasmin)
    tropical_nights_change.py # Tropical Nights climate change signal (SSP2-4.5)
    heatwave.py           # Heatwave Indices (TX90: HWN, HWF, HWD, HWM, HWA)
    heatwave_change.py    # Heatwave climate change signal (SSP2-4.5)
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

1880 tests across 41 test files.

## Requirements

Python 3.10+. See `environment.yml` for the full dependency list, or install via `conda env create -f environment.yml`.

Key dependencies: xarray, dask, numpy, scipy, matplotlib, cartopy, healpy, nereus, gsw, cfgrib, pydantic, google-genai, openai.
