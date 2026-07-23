"""Climate variability modes (teleconnections) diagnostic.

Evaluates large-scale teleconnection patterns (ENSO, NAO, SAM, AO, IOD,
PDO, QBO) by comparing model-simulated indices against observations and
optionally CMIP6. For each mode, produces four figures:

1. Index time series (monthly + annual smoothed)
2. Spatial pattern map (EOF loading or regression)
3. Power spectrum (Welch periodogram)
4. Seasonal variance profile (monthly STD of index)
"""

import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import xarray as xr
from nereus.plotting import get_projection

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.eof import compute_eof
from feather.util.spectrum import power_spectrum
from feather.util.temporal import deseason

logger = logging.getLogger(__name__)


# ── Mode definitions ────────────────────────────────────────────────


@dataclass
class ModeDefinition:
    """Definition of a climate variability mode."""

    name: str
    long_name: str
    variable: str          # CMOR variable name
    method: str            # "box_mean", "box_diff", "eof", "zonal_mean"
    domain: str            # "sfc", "o2d", "pl"
    # Optional obs override: load the reference field directly from this
    # obs dataset/variable instead of the registry default for ``variable``.
    # Used for SST modes (ENSO/IOD/PDO) to pick HadISST (full 1870-present
    # record) rather than ESA-CCI, whose file only spans 1990-2014.
    obs_dataset: str | None = None
    obs_variable: str | None = None
    # Label for the observation panel/legend (matches obs_dataset).
    obs_label: str = "ERA5"
    # Box regions: {name: (lon_min, lon_max, lat_min, lat_max)}
    boxes: dict = field(default_factory=dict)
    # EOF region
    eof_region: tuple | None = None   # (lon_min, lon_max, lat_min, lat_max)
    eof_all_lons: bool = False
    # Zonal mean settings
    pressure_level: float | None = None
    lat_band: tuple | None = None
    # Metadata
    typical_period: str = ""
    seasonal_peak: str = "all"
    detrend_global: bool = False
    # Sign convention: spatial point that should be positive for canonical +
    sign_point: tuple | None = None  # (lat, lon) for sign check
    # Plot settings for spatial pattern maps
    plot_projection: str = "rob"  # "rob", "np", "sp"
    plot_extent: tuple | None = None  # (lon_min, lon_max, lat_min, lat_max)


_MODE_REGISTRY: dict[str, ModeDefinition] = {}



def _register_mode(mode: ModeDefinition) -> ModeDefinition:
    _MODE_REGISTRY[mode.name] = mode
    return mode


# ENSO: Nino 3.4 box mean of SST anomalies
_register_mode(ModeDefinition(
    name="enso",
    long_name="ENSO (Nino 3.4)",
    variable="tos",
    method="box_mean",
    domain="o2d",
    boxes={"nino34": (190, 240, -5, 5)},  # 170W-120W = 190-240 in 0-360
    typical_period="3-7 years",
    seasonal_peak="DJF",
    obs_dataset="HADISST", obs_variable="sst",  # full 1870-present record
    obs_label="HadISST",
))

# NAO: EOF1 of SLP over North Atlantic
_register_mode(ModeDefinition(
    name="nao",
    long_name="North Atlantic Oscillation",
    variable="psl",
    method="eof",
    domain="sfc",
    eof_region=(270, 40, 20, 80),  # 90W-40E = 270-360,0-40 in 0-360
    typical_period="interannual",
    seasonal_peak="DJF",
    sign_point=(65, 340),  # Iceland — should be negative for NAO+
    plot_projection="np",
    plot_extent=(-180, 180, 15, 90),
))

# SAM: EOF1 of SLP over Southern Hemisphere
_register_mode(ModeDefinition(
    name="sam",
    long_name="Southern Annular Mode",
    variable="psl",
    method="eof",
    domain="sfc",
    eof_region=None,
    eof_all_lons=True,
    lat_band=(-90, -20),
    typical_period="interannual",
    seasonal_peak="all",
    sign_point=(-65, 0),  # Antarctic — should be negative for SAM+
    plot_projection="sp",
    plot_extent=(-180, 180, -90, -15),
))

# AO: EOF1 of SLP over Northern Hemisphere
_register_mode(ModeDefinition(
    name="ao",
    long_name="Arctic Oscillation",
    variable="psl",
    method="eof",
    domain="sfc",
    eof_region=None,
    eof_all_lons=True,
    lat_band=(20, 90),
    typical_period="interannual",
    seasonal_peak="DJF",
    sign_point=(90, 0),  # Arctic — should be negative for AO+
    plot_projection="np",
    plot_extent=(-180, 180, 15, 90),
))

# IOD: Dipole Mode Index = western box - eastern box SST anomaly
_register_mode(ModeDefinition(
    name="iod",
    long_name="Indian Ocean Dipole",
    variable="tos",
    method="box_diff",
    domain="o2d",
    boxes={
        "west": (50, 70, -10, 10),
        "east": (90, 110, -10, 0),
    },
    typical_period="2-4 years",
    seasonal_peak="SON",
    obs_dataset="HADISST", obs_variable="sst",  # full 1870-present record
    obs_label="HadISST",
))

# PDO: EOF1 of N. Pacific SST with global-mean SST removed
_register_mode(ModeDefinition(
    name="pdo",
    long_name="Pacific Decadal Oscillation",
    variable="tos",
    method="eof",
    domain="o2d",
    eof_region=(120, 260, 20, 70),  # 120E-100W = 120-260 in 0-360
    typical_period="20-30 years",
    seasonal_peak="all",
    detrend_global=True,
    sign_point=(45, 200),  # Central N. Pacific — negative for PDO+
    plot_projection="np",
    plot_extent=(-180, 180, 15, 90),
    obs_dataset="HADISST", obs_variable="sst",  # full 1870-present record
    obs_label="HadISST",
))

# QBO: Equatorial zonal-mean zonal wind at 50 hPa
_register_mode(ModeDefinition(
    name="qbo",
    long_name="Quasi-Biennial Oscillation",
    variable="ua",
    method="zonal_mean",
    domain="pl",
    pressure_level=50.0,
    lat_band=(-5, 5),
    typical_period="~28 months",
    seasonal_peak="all",
))


# ── Diagnostic class ───────────────────────────────────────────────


@register
class TeleconnectionDiag(DiagnosticBase):
    """Climate variability modes diagnostic.

    Computes teleconnection indices (ENSO, NAO, SAM, AO, IOD, PDO, QBO)
    and produces index time series, spatial patterns, power spectra, and
    seasonal variance profiles for each mode.
    """

    name = "teleconnections"
    title = "Climate Variability Modes"
    domain = "sfc"
    variables = ["tos", "psl"]
    group = "evaluation"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment=None, period=None,
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, save_netcdf=save_netcdf)
        self.experiment = experiment or config.get_experiment()
        self.period = period or config.get_period()
        self.cmip6_individual = cmip6_individual
        self._modes = list(_MODE_REGISTRY.keys())
        # Data cache: avoid reloading same variable for multiple modes
        # Key: (source, model, variable) → DataArray (dask-backed)
        self._data_cache: dict[tuple, xr.DataArray] = {}

    # ── Orchestration ───────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-mode: compute index → plot 4 figures → save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        for mode_name in self._modes:
            mode_def = _MODE_REGISTRY[mode_name]
            figure_ids = [
                f"{mode_name}_timeseries",
                f"{mode_name}_spectrum",
                f"{mode_name}_seasonal_variance",
            ]
            # Zonal-mean modes (QBO) emit no spatial-pattern figure.
            if mode_def.method != "zonal_mean":
                figure_ids.insert(1, f"{mode_name}_pattern")

            if skip_existing and all(
                self._figure_exists(fid) for fid in figure_ids
            ):
                logger.info("Skipping %s — all figures exist", mode_name)
                saved.extend([
                    (self.output_dir / f"{fid}.png",
                     self.output_dir / f"{fid}.json")
                    for fid in figure_ids
                ])
                continue

            try:
                t0 = _time.time()
                mode_result = self._compute_mode(mode_def)
                dt = _time.time() - t0
                if mode_result is None:
                    logger.info(
                        "  Mode %s: no data — skipped (%.1fs)", mode_name, dt,
                    )
                    continue
                self._maybe_export_netcdf(mode_result, mode_name)
                logger.info(
                    "  Mode %s computed in %.1fs — plotting...",
                    mode_name, dt,
                )

                figures = self._plot_mode(mode_def, mode_result)
                for fig, meta in figures:
                    paths = self._save(fig, meta, meta["figure_id"])
                    saved.append(paths)
                    plt.close(fig)
                logger.info(
                    "  Mode %s: %d figures saved", mode_name, len(figures),
                )
            except Exception:
                logger.warning(
                    "Mode %s failed — skipping", mode_name, exc_info=True,
                )

        # Clear data cache after all modes are done
        self._data_cache.clear()

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── ABC compat ──────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        results = {}
        for mode_name in self._modes:
            mode_def = _MODE_REGISTRY[mode_name]
            try:
                r = self._compute_mode(mode_def)
                if r is not None:
                    results[mode_name] = r
            except Exception:
                logger.warning("Mode %s failed", mode_name, exc_info=True)
        self._data_cache.clear()
        return results

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figures = []
        for mode_name, mode_result in results.items():
            mode_def = _MODE_REGISTRY[mode_name]
            figures.extend(self._plot_mode(mode_def, mode_result))
        return figures

    # ── Core computation ────────────────────────────────────────────

    def _compute_mode(self, mode_def: ModeDefinition) -> dict[str, Any] | None:
        """Compute index and pattern for a single mode."""
        logger.info("Computing mode: %s (%s)", mode_def.name, mode_def.long_name)

        var = mode_def.variable
        model_indices: dict[str, xr.DataArray] = {}
        model_patterns: dict[str, xr.DataArray] = {}
        model_var_explained: dict[str, float] = {}

        # -- Models --
        for model in self.config.models:
            try:
                t0 = _time.time()
                logger.info("  %s / %s: loading %s...", mode_def.name, model, var)
                idx, pattern, var_exp = self._compute_index(
                    mode_def, model, source="model",
                )
                dt = _time.time() - t0
                if idx is not None:
                    model_indices[model] = idx
                    logger.info(
                        "  %s / %s: index computed (std=%.4f) in %.1fs",
                        mode_def.name, model, float(idx.std()), dt,
                    )
                if pattern is not None:
                    model_patterns[model] = pattern
                if var_exp is not None:
                    model_var_explained[model] = var_exp
                    logger.info(
                        "  %s / %s: EOF1 explains %.1f%% variance",
                        mode_def.name, model, var_exp * 100,
                    )
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "  Variable %s not available for %s — skipping",
                    var, model,
                )
            except Exception:
                logger.warning(
                    "  Mode %s failed for %s", mode_def.name, model,
                    exc_info=True,
                )

        if not model_indices:
            logger.warning(
                "  No models have data for mode %s — skipping", mode_def.name,
            )
            return None

        # -- Observations --
        obs_index = None
        obs_pattern = None
        obs_var_explained = None
        try:
            t0 = _time.time()
            logger.info("  %s / obs: loading %s...", mode_def.name, var)
            obs_index, obs_pattern, obs_var_explained = self._compute_index(
                mode_def, None, source="obs",
            )
            dt = _time.time() - t0
            if obs_index is not None:
                logger.info(
                    "  %s / obs: index computed (std=%.4f) in %.1fs",
                    mode_def.name, float(obs_index.std()), dt,
                )
        except Exception:
            logger.warning(
                "  Obs computation failed for %s", mode_def.name,
                exc_info=True,
            )

        # -- CMIP6 --
        cmip6_mmm_index = None
        cmip6_mmm_pattern = None
        cmip6_individual: dict[str, xr.DataArray] = {}
        cmip6_individual_patterns: dict[str, xr.DataArray] = {}
        cmip6_info: dict[str, Any] = {}
        if self.cmip6_enabled:
            t0 = _time.time()
            logger.info("  %s: computing CMIP6 indices...", mode_def.name)
            (cmip6_mmm_index, cmip6_mmm_pattern,
             cmip6_individual, cmip6_individual_patterns,
             cmip6_info) = self._compute_cmip6_index(mode_def)
            dt = _time.time() - t0
            n_cmip6 = cmip6_info.get("n_members", 0)
            logger.info(
                "  %s: CMIP6 done (%d models) in %.1fs",
                mode_def.name, n_cmip6, dt,
            )

        return {
            "mode_def": mode_def,
            "model_indices": model_indices,
            "model_patterns": model_patterns,
            "model_var_explained": model_var_explained,
            "obs_index": obs_index,
            "obs_pattern": obs_pattern,
            "obs_var_explained": obs_var_explained,
            "cmip6_mmm_index": cmip6_mmm_index,
            "cmip6_mmm_pattern": cmip6_mmm_pattern,
            "cmip6_individual": cmip6_individual,
            "cmip6_individual_patterns": cmip6_individual_patterns,
            "cmip6_info": cmip6_info,
        }

    def _compute_index(
        self,
        mode_def: ModeDefinition,
        model: str | None,
        source: str = "model",
    ) -> tuple[xr.DataArray | None, xr.DataArray | None, float | None]:
        """Compute index + spatial pattern for a single source.

        Returns (index, pattern, variance_explained).
        """
        if mode_def.method == "box_mean":
            return self._compute_box_mean(mode_def, model, source)
        elif mode_def.method == "box_diff":
            return self._compute_box_diff(mode_def, model, source)
        elif mode_def.method == "eof":
            return self._compute_eof_mode(mode_def, model, source)
        elif mode_def.method == "zonal_mean":
            return self._compute_zonal_mean(mode_def, model, source)
        else:
            raise ValueError(f"Unknown method: {mode_def.method}")

    def _load_field(
        self,
        mode_def: ModeDefinition,
        model: str | None,
        source: str,
    ) -> xr.DataArray:
        """Load the underlying field for a mode, with caching.

        Caches loaded DataArrays by (source, model, variable) so the
        same field is not loaded from disk multiple times across modes
        (e.g. tos is used by ENSO, IOD, and PDO).
        """
        var = mode_def.variable
        cache_key = (source, model, var)

        if cache_key in self._data_cache:
            logger.debug("    Cache hit for (%s, %s, %s)", source, model, var)
            return self._data_cache[cache_key]

        if source == "obs":
            if mode_def.obs_dataset is not None:
                # Explicit obs override (e.g. HadISST for the SST modes,
                # which covers the full analysis window unlike ESA-CCI).
                da = self.obs_loader.load(
                    mode_def.obs_dataset, mode_def.obs_variable,
                    period=self.period,
                )
                # HadISST flags land/sea-ice cells with sentinel fill values
                # (e.g. -1000.0, -1e30). Mask any non-physical SST to NaN so
                # they cannot corrupt box means or the PDO North-Pacific EOF
                # (EOF drops non-finite columns; box means skip NaN). The
                # window is generous enough to cover either °C (~[-2, 40]) or
                # K (~[270, 313]) storage while excluding the fill sentinels.
                da = da.where((da > -100.0) & (da < 1000.0))
            else:
                da = self._load_obs_var(var, self.period)
        else:
            da = self._load_model_var(model, var, period=self.period)
            if "time" in da.dims and self.period:
                da = da.sel(time=slice(self.period[0], self.period[1]))

        # Normalise coordinates: lons to 0..360, lats ascending
        # (mode definitions use 0..360; slice() needs ascending lats)
        lon_name = "lon" if "lon" in da.dims else "longitude"
        lat_name = "lat" if "lat" in da.dims else "latitude"
        if lon_name in da.dims:
            lons = da[lon_name].values
            if np.any(lons < 0):
                da = da.assign_coords({lon_name: lons % 360})
                da = da.sortby(lon_name)
        if lat_name in da.dims:
            lats = da[lat_name].values
            if len(lats) > 1 and lats[0] > lats[-1]:
                da = da.sortby(lat_name)

        self._data_cache[cache_key] = da
        return da

    # ── Box-mean index (ENSO) ───────────────────────────────────────

    def _compute_box_mean(self, mode_def, model, source):
        """Box area-mean index (e.g. Nino 3.4)."""
        da = self._load_field(mode_def, model, source)
        box = list(mode_def.boxes.values())[0]

        logger.info("    Extracting box mean %s...", list(mode_def.boxes.keys())[0])
        box_ts = self._extract_box_mean(da, box)

        # Deseasonalise (fast — 1D time series)
        logger.info("    Deseasonalising index...")
        idx = deseason(box_ts)
        if hasattr(idx, "compute"):
            idx = idx.compute()

        # Regression pattern: fully lazy until final .compute()
        logger.info("    Computing regression pattern (lazy)...")
        pattern = self._regression_pattern(da, idx)

        return idx, pattern, None

    def _compute_box_diff(self, mode_def, model, source):
        """Box difference index (e.g. IOD = west - east)."""
        da = self._load_field(mode_def, model, source)
        box_names = list(mode_def.boxes.keys())

        logger.info("    Extracting box means (%s - %s)...", box_names[0], box_names[1])
        ts_a = self._extract_box_mean(da, mode_def.boxes[box_names[0]])
        ts_b = self._extract_box_mean(da, mode_def.boxes[box_names[1]])
        diff = ts_a - ts_b

        logger.info("    Deseasonalising index...")
        idx = deseason(diff)
        if hasattr(idx, "compute"):
            idx = idx.compute()

        logger.info("    Computing regression pattern (lazy)...")
        pattern = self._regression_pattern(da, idx)

        return idx, pattern, None

    @staticmethod
    def _find_latlon(da):
        """Find lat/lon coordinate names and whether grid is rectilinear.

        Returns (lat_name, lon_name, is_rectilinear).
        - is_rectilinear=True: lat/lon are 1D dimensions → use ``.sel()``
        - is_rectilinear=False: lat/lon are 2D coords (curvilinear ocean
          grids like ORCA) → use ``nr.subset_by_bbox()`` masking
        """
        lat_name = lon_name = None
        for name in ("lat", "latitude", "nav_lat", "glat",
                      "yt_ocean", "yh", "nod2d_lat"):
            if name in da.dims or name in da.coords:
                lat_name = name
                break
        for name in ("lon", "longitude", "nav_lon", "glon",
                      "xt_ocean", "xh", "nod2d_lon"):
            if name in da.dims or name in da.coords:
                lon_name = name
                break

        if lat_name is None or lon_name is None:
            raise ValueError(
                f"No lat/lon found: dims={list(da.dims)}, "
                f"coords={list(da.coords)}"
            )

        is_rect = lat_name in da.dims and lon_name in da.dims
        return lat_name, lon_name, is_rect

    @staticmethod
    def _get_latlon_arrays(da):
        """Return lat/lon numpy arrays from a DataArray.

        Uses ``nereus.extract_coordinates`` but validates the result:
        if it returns integer dim indices (e.g. IPSL's ``x``/``y``),
        falls back to reading the named coords found by
        ``_find_latlon`` (e.g. ``nav_lat``/``nav_lon``).
        """
        from nereus.core.grids import extract_coordinates

        lat_name, lon_name, is_rect = TeleconnectionDiag._find_latlon(da)
        lon_arr, lat_arr = extract_coordinates(da)

        if lon_arr is not None and lat_arr is not None:
            lat_np = np.asarray(lat_arr)
            lon_np = np.asarray(lon_arr)
            # Validate: if values look like integer dim indices
            # (0, 1, 2, ..., N-1), nereus picked up the wrong coords.
            lat_is_idx = (lat_np.ndim == 1
                          and np.array_equal(lat_np, np.arange(len(lat_np))))
            lon_is_idx = (lon_np.ndim == 1
                          and np.array_equal(lon_np, np.arange(len(lon_np))))
            if not lat_is_idx and not lon_is_idx:
                return lat_np, lon_np, is_rect

        # Fallback: read from named coords directly
        lat_np = np.asarray(da[lat_name].values)
        lon_np = np.asarray(da[lon_name].values)
        return lat_np, lon_np, is_rect

    @staticmethod
    def _extract_box_mean(da, box):
        """Extract area-weighted mean over a lat/lon box.

        Handles both rectilinear (1D lat/lon dims) and curvilinear
        (2D lat/lon coords) grids.  For curvilinear grids, uses
        ``nereus.subset_by_bbox()`` for robust region masking.

        Parameters
        ----------
        da : xr.DataArray
            Field with lat/lon dims or coords.
        box : tuple
            (lon_min, lon_max, lat_min, lat_max).
        """
        lon_min, lon_max, lat_min, lat_max = box

        lat_name, lon_name, is_rect = TeleconnectionDiag._find_latlon(da)
        lat_arr, lon_arr = (np.asarray(da[lat_name].values),
                            np.asarray(da[lon_name].values))

        if not is_rect:
            # Non-rectilinear grid: curvilinear (2D coords like nav_lat)
            # or rectilinear with renamed dims (1D coords on i/j).
            spatial_dims = [d for d in da.dims if d != "time"]
            spatial_shape = tuple(da.sizes[d] for d in spatial_dims)

            # If both are 1D but different lengths → meshgrid
            if (lon_arr.ndim == 1 and lat_arr.ndim == 1
                    and lon_arr.shape != lat_arr.shape):
                lat_2d, lon_2d = np.meshgrid(
                    lat_arr, lon_arr, indexing="ij",
                )
            else:
                lon_2d = lon_arr
                lat_2d = lat_arr

            # Flatten + normalise lons to 0..360
            lon_flat = lon_2d.ravel() % 360
            lat_flat = lat_2d.ravel()

            if lon_min < lon_max:
                mask_flat = nr.subset_by_bbox(
                    lon_flat, lat_flat,
                    lon_min, lon_max, lat_min, lat_max,
                )
            else:
                m1 = nr.subset_by_bbox(
                    lon_flat, lat_flat,
                    lon_min, 360.0, lat_min, lat_max,
                )
                m2 = nr.subset_by_bbox(
                    lon_flat, lat_flat,
                    0.0, lon_max, lat_min, lat_max,
                )
                mask_flat = m1 | m2

            mask = mask_flat.reshape(spatial_shape)
            mask_da = xr.DataArray(mask, dims=spatial_dims)

            da_masked = da.where(mask_da)

            # Build cos-lat weights as numpy (no NaN issues)
            w = np.cos(np.deg2rad(lat_2d.reshape(spatial_shape)))
            w[~mask] = 0.0
            weights = xr.DataArray(w, dims=spatial_dims)
            return da_masked.weighted(weights).mean(dim=spatial_dims)

        # Rectilinear grid: use .sel() slicing
        lons = da[lon_name].values
        if lon_min < lon_max:
            da_sel = da.sel(
                {lat_name: slice(lat_min, lat_max),
                 lon_name: slice(lon_min, lon_max)},
            )
        else:
            # Wrapping case: select lon_min..360 and 0..lon_max
            mask = (lons >= lon_min) | (lons <= lon_max)
            da_sel = da.sel({lat_name: slice(lat_min, lat_max)})
            da_sel = da_sel.isel({lon_name: mask})

        # Cos-lat weighting
        lat_vals = da_sel[lat_name]
        weights = np.cos(np.deg2rad(lat_vals))
        return da_sel.weighted(weights).mean(dim=[lat_name, lon_name])

    @staticmethod
    def _regression_pattern(da, idx):
        """Compute regression of field onto normalised index.

        Uses temporal mean subtraction (NOT full deseasonalisation) on
        the field, keeping everything lazy until the final materialise.
        This is mathematically equivalent because the seasonal cycle is
        orthogonal to the deseasonalised index by construction.

        Returns 2D map of regression coefficients (original units per
        unit index STD).
        """
        if idx is None or len(idx) < 3:
            return None

        # Align times (cheap — coordinate matching). Skip the .sel() copy when
        # the field and index already share the same time axis (the common case
        # for an EOF PC derived from this field), avoiding a full-field copy.
        if (da.sizes.get("time") == idx.sizes.get("time")
                and np.array_equal(da.time.values, idx.time.values)):
            da_aligned, idx_aligned = da, idx
        else:
            common_times = np.intersect1d(da.time.values, idx.time.values)
            if len(common_times) < 3:
                return None
            da_aligned = da.sel(time=common_times)
            idx_aligned = idx.sel(time=common_times)

        # Normalise index
        idx_std = float(idx_aligned.std())
        if idx_std == 0:
            return None
        idx_norm = idx_aligned / idx_std

        # Dask-backed (lazy) fields: let dask stream the reduction.
        if da_aligned.chunks is not None:
            da_centred = da_aligned - da_aligned.mean("time")
            pattern = (da_centred * idx_norm).mean("time")
            return pattern.compute()

        # In-memory fields (e.g. a materialised global ocean grid): compute the
        # regression via a streaming matmul on the flattened array rather than
        # broadcasting a full-size (field × index) product, which needs ~2–3×
        # the field in RAM and OOMs on large curvilinear grids. Uses the
        # covariance identity  mean((X−X̄)·i) = mean(X·i) − X̄·mean(i)  so no
        # centred copy of the field is materialised.
        spatial_dims = [d for d in da_aligned.dims if d != "time"]
        nt = da_aligned.sizes["time"]
        arr = da_aligned.transpose("time", *spatial_dims).values
        flat = arr.reshape(nt, -1)
        idxv = np.asarray(idx_norm.values, dtype=flat.dtype)
        coef = (idxv @ flat) / nt - flat.mean(axis=0) * float(idxv.mean())

        # Reuse a single-time slice as a template to preserve spatial dims and
        # coords (incl. 2-D curvilinear lat/lon) without a large allocation.
        template = da_aligned.isel(time=0).drop_vars("time", errors="ignore")
        return template.copy(data=coef.reshape(template.shape))

    # ── EOF modes (NAO, SAM, AO, PDO) ──────────────────────────────

    def _compute_eof_mode(self, mode_def, model, source):
        """EOF-based index (NAO, SAM, AO, PDO)."""
        da = self._load_field(mode_def, model, source)

        # Check grid type: HEALPix and curvilinear grids use flat EOF
        _, _, is_rect = self._find_latlon(da)
        if not is_rect:
            return self._compute_eof_mode_flat(da, mode_def)

        # Extract regional subset FIRST (before deseasonalising)
        # This is the key optimisation: operate on a small region, not global
        logger.info("    Extracting EOF region for %s...", mode_def.name)
        da_region = self._extract_eof_region(da, mode_def)

        # For PDO: remove global-mean SST (lazy, computed on full field)
        if mode_def.detrend_global:
            logger.info("    Removing global-mean SST (lazy)...")
            gm = self._field_global_mean(da)  # lazy on full field
            da_region = da_region - gm  # broadcasts, still lazy

        # Deseasonalise the REGIONAL subset (much smaller than global)
        logger.info("    Deseasonalising regional subset...")
        da_anom = deseason(da_region)

        # Materialise regional subset only
        if hasattr(da_anom, "compute"):
            logger.info("    Materialising regional subset...")
            t0 = _time.time()
            da_anom = da_anom.compute()
            dt = _time.time() - t0
            shape = da_anom.shape
            mb = da_anom.nbytes / 1e6
            logger.info(
                "    Materialised %s (%.1f MB) in %.1fs", shape, mb, dt,
            )

        if da_anom.sizes["time"] < 12:
            logger.warning(
                "  Too few timesteps (%d) for EOF of %s",
                da_anom.sizes["time"], mode_def.name,
            )
            return None, None, None

        # Compute EOF
        logger.info("    Computing EOF (SVD)...")
        t0 = _time.time()
        eofs, pcs, var_exp = compute_eof(da_anom, n_modes=1)
        dt = _time.time() - t0
        logger.info(
            "    EOF done in %.1fs — var explained: %.1f%%",
            dt, var_exp[0] * 100,
        )

        # Extract leading mode
        eof1 = eofs.sel(mode=1)
        pc1 = pcs.sel(mode=1)
        var_explained = float(var_exp[0])

        # Fix sign convention
        if mode_def.sign_point is not None:
            pc1, eof1 = self._fix_eof_sign(
                pc1, eof1, mode_def.sign_point,
            )

        return pc1, eof1, var_explained

    def _compute_eof_mode_flat(self, da, mode_def):
        """EOF-based index for non-rectilinear grids (HEALPix, curvilinear).

        Uses _compute_eof_flat for the PC index, then computes a regression
        pattern from the full field so spatial maps still work.

        Returns (pc1, pattern, var_explained) or (None, None, None).
        """
        logger.info("    Using flat EOF for non-rectilinear grid...")

        # For PDO: remove global-mean SST
        if mode_def.detrend_global:
            logger.info("    Removing global-mean SST (lazy)...")
            gm = self._field_global_mean(da)
            da = da - gm

        # Deseasonalise the full field (flat EOF handles regional masking)
        logger.info("    Deseasonalising field...")
        da_anom = deseason(da)
        if hasattr(da_anom, "compute"):
            logger.info("    Materialising field...")
            t0 = _time.time()
            da_anom = da_anom.compute()
            dt = _time.time() - t0
            logger.info(
                "    Materialised %s (%.1f MB) in %.1fs",
                da_anom.shape, da_anom.nbytes / 1e6, dt,
            )

        if da_anom.sizes["time"] < 12:
            logger.warning(
                "  Too few timesteps (%d) for EOF of %s",
                da_anom.sizes["time"], mode_def.name,
            )
            return None, None, None

        logger.info("    Computing flat EOF (SVD)...")
        t0 = _time.time()
        result = self._compute_eof_flat(da_anom, mode_def, n_modes=1)
        dt = _time.time() - t0

        if result[0] is None:
            logger.warning("  Flat EOF returned None for %s", mode_def.name)
            return None, None, None

        pc1, var_explained = result
        logger.info(
            "    Flat EOF done in %.1fs — var explained: %.1f%%",
            dt, var_explained * 100,
        )

        # Fix sign convention using the PC index
        if mode_def.sign_point is not None:
            # Regression at sign point: if positive → flip
            lat_check, lon_check = mode_def.sign_point
            try:
                lat_arr, lon_arr, _ = self._get_latlon_arrays(da_anom)
                lat_flat = np.asarray(lat_arr).ravel()
                lon_flat = np.asarray(lon_arr).ravel() % 360
                lon_check_360 = lon_check % 360
                dist = np.sqrt(
                    (lat_flat - lat_check) ** 2
                    + (lon_flat - lon_check_360) ** 2
                )
                nearest_idx = int(np.argmin(dist))
                spatial_dims = [d for d in da_anom.dims if d != "time"]
                data_flat = da_anom.values.reshape(
                    da_anom.sizes["time"], -1,
                )
                sign_val = np.corrcoef(
                    pc1.values, data_flat[:, nearest_idx],
                )[0, 1]
                if sign_val > 0:
                    pc1 = -pc1
            except Exception:
                pass  # sign convention best-effort

        # Compute regression pattern from full field
        logger.info("    Computing regression pattern (lazy)...")
        pattern = self._regression_pattern(da_anom, pc1)

        return pc1, pattern, var_explained

    @staticmethod
    def _extract_eof_region(da, mode_def):
        """Extract the regional subset for EOF computation.

        For rectilinear grids, uses ``.sel()`` slicing.
        For non-rectilinear grids, raises ValueError — caller should
        use ``_compute_eof_flat()`` instead.
        """
        lat_name, lon_name, is_rect = TeleconnectionDiag._find_latlon(da)

        if not is_rect:
            raise ValueError(
                "EOF region extraction on non-rectilinear grid not "
                f"supported via .sel() (dims={list(da.dims)}). "
                "Use _compute_eof_flat() instead."
            )

        if mode_def.eof_all_lons and mode_def.lat_band:
            lat_min, lat_max = mode_def.lat_band
            return da.sel({lat_name: slice(lat_min, lat_max)})

        if mode_def.eof_region is not None:
            lon_min, lon_max, lat_min, lat_max = mode_def.eof_region
            da_lat = da.sel({lat_name: slice(lat_min, lat_max)})

            lons = da_lat[lon_name].values
            if lon_min < lon_max:
                return da_lat.sel(
                    {lon_name: slice(lon_min, lon_max)},
                )
            else:
                # Wrapping case
                mask = (lons >= lon_min) | (lons <= lon_max)
                return da_lat.isel({lon_name: mask})

        return da

    @staticmethod
    def _compute_eof_flat(da, mode_def, n_modes=1):
        """Compute EOF on non-rectilinear grid via flat point array.

        Extracts the regional mask, flattens valid spatial points to a
        (time, n_points) matrix, runs SVD, and returns only the PC time
        series (no spatial pattern — can't display on common grid).

        Returns (pc1, variance_explained) or (None, None).
        """
        try:
            lat_arr, lon_arr, _ = TeleconnectionDiag._get_latlon_arrays(da)
        except ValueError:
            return None, None

        spatial_dims = [d for d in da.dims if d != "time"]
        spatial_shape = tuple(da.sizes[d] for d in spatial_dims)

        # Make 2D arrays if needed
        if (lon_arr.ndim == 1 and lat_arr.ndim == 1
                and lon_arr.shape != lat_arr.shape):
            lat_2d, lon_2d = np.meshgrid(
                lat_arr, lon_arr, indexing="ij",
            )
        else:
            lon_2d = lon_arr
            lat_2d = lat_arr

        # Build regional mask
        lon_flat = lon_2d.ravel() % 360
        lat_flat = lat_2d.ravel()

        if mode_def.eof_all_lons and mode_def.lat_band:
            lat_min, lat_max = mode_def.lat_band
            mask_flat = (lat_flat >= lat_min) & (lat_flat <= lat_max)
        elif mode_def.eof_region is not None:
            lon_min, lon_max, lat_min, lat_max = mode_def.eof_region
            lat_mask = (lat_flat >= lat_min) & (lat_flat <= lat_max)
            if lon_min < lon_max:
                lon_mask = (lon_flat >= lon_min) & (lon_flat <= lon_max)
            else:
                lon_mask = (lon_flat >= lon_min) | (lon_flat <= lon_max)
            mask_flat = lat_mask & lon_mask
        else:
            mask_flat = np.ones(len(lat_flat), dtype=bool)

        n_valid = int(mask_flat.sum())
        if n_valid < n_modes + 1:
            logger.warning(
                "  Too few valid points (%d) for EOF", n_valid,
            )
            return None, None

        # Flatten spatial dims, select valid points
        nt = da.sizes["time"]
        data_flat = np.asarray(da.values).reshape(nt, -1)[:, mask_flat]
        lat_valid = lat_flat[mask_flat]

        # Cos-lat weights
        cos_w = np.sqrt(np.maximum(
            np.cos(np.deg2rad(lat_valid)), 0.0,
        ))

        # Remove NaN columns
        valid_cols = np.all(np.isfinite(data_flat), axis=0) & (cos_w > 0)
        if valid_cols.sum() < n_modes + 1:
            return None, None
        data_valid = data_flat[:, valid_cols]
        w = cos_w[valid_cols]

        # Centre + weight + SVD
        data_centred = data_valid - data_valid.mean(axis=0)
        data_weighted = data_centred * w[np.newaxis, :]

        n_svd = min(n_modes, min(data_weighted.shape) - 1)
        if n_svd < 1:
            return None, None

        U, S, Vt = np.linalg.svd(data_weighted, full_matrices=False)
        U = U[:, :n_svd]
        S = S[:n_svd]

        total_var = np.sum(data_weighted ** 2)
        var_explained = (S ** 2) / total_var

        pcs_raw = U * S[np.newaxis, :]
        pc_std = pcs_raw.std(axis=0)
        pc_std[pc_std == 0] = 1.0
        pcs_norm = pcs_raw / pc_std

        # Build xarray PC
        mode_coord = np.arange(1, n_svd + 1)
        pcs_da = xr.DataArray(
            pcs_norm,
            dims=("time", "mode"),
            coords={"time": da.time, "mode": mode_coord},
        )

        return pcs_da.sel(mode=1), float(var_explained[0])

    @staticmethod
    def _fix_eof_sign(pc, eof, sign_point):
        """Ensure EOF has canonical sign orientation.

        For NAO+, Iceland should have *negative* SLP anomaly.
        For SAM+, Antarctic should have *negative* SLP anomaly.
        """
        lat_check, lon_check = sign_point
        try:
            lat_name, lon_name, _ = TeleconnectionDiag._find_latlon(eof)
        except ValueError:
            return pc, eof

        try:
            val = float(eof.sel(
                {lat_name: lat_check, lon_name: lon_check},
                method="nearest",
            ))
        except Exception:
            return pc, eof

        # Convention: sign_point should be NEGATIVE for positive phase
        if val > 0:
            pc = -pc
            eof = -eof

        return pc, eof

    @staticmethod
    def _field_global_mean(da):
        """Compute cos-lat-weighted global mean for any grid type."""
        try:
            lat_arr, lon_arr, is_rect = TeleconnectionDiag._get_latlon_arrays(
                da,
            )
        except ValueError:
            spatial = [d for d in da.dims if d != "time"]
            return da.mean(dim=spatial)

        lat_name, lon_name, _ = TeleconnectionDiag._find_latlon(da)

        if is_rect:
            weights = np.cos(np.deg2rad(da[lat_name]))
            return da.weighted(weights).mean(dim=[lat_name, lon_name])

        # Non-rectilinear: build proper weight array
        spatial_dims = [d for d in da.dims if d != "time"]
        spatial_shape = tuple(da.sizes[d] for d in spatial_dims)

        if (lat_arr.ndim == 1 and lon_arr.ndim == 1
                and lon_arr.shape != lat_arr.shape):
            lat_2d, _ = np.meshgrid(lat_arr, lon_arr, indexing="ij")
        else:
            lat_2d = lat_arr

        w = np.cos(np.deg2rad(lat_2d.reshape(spatial_shape)))
        weights = xr.DataArray(w, dims=spatial_dims)
        return da.weighted(weights).mean(dim=spatial_dims)

    # ── Zonal-mean index (QBO) ──────────────────────────────────────

    def _compute_zonal_mean(self, mode_def, model, source):
        """Equatorial zonal-mean at a pressure level (QBO)."""
        da = self._load_field(mode_def, model, source)

        # Select pressure level
        if mode_def.pressure_level is not None:
            level_name = None
            for name in ("plev", "level", "lev"):
                if name in da.dims:
                    level_name = name
                    break
            if level_name is None:
                logger.warning("  No pressure level dimension for QBO")
                return None, None, None

            # Try to select level (may be in Pa or hPa)
            level_val = mode_def.pressure_level
            levels = da[level_name].values
            # If levels are in Pa, convert target to Pa
            if np.max(levels) > 1100:
                level_val = level_val * 100
            da = da.sel({level_name: level_val}, method="nearest")

        # Equatorial band zonal mean
        lat_name = "lat" if "lat" in da.dims else "latitude"
        lon_name = "lon" if "lon" in da.dims else "longitude"
        if mode_def.lat_band:
            lat_min, lat_max = mode_def.lat_band
            da = da.sel({lat_name: slice(lat_min, lat_max)})

        # Cos-lat weighted zonal mean
        weights = np.cos(np.deg2rad(da[lat_name]))
        idx = da.weighted(weights).mean(dim=[lat_name, lon_name])

        # Deseasonalise
        logger.info("    Deseasonalising QBO index...")
        idx = deseason(idx)
        if hasattr(idx, "compute"):
            idx = idx.compute()

        return idx, None, None

    # ── CMIP6 ───────────────────────────────────────────────────────

    def _compute_cmip6_index(self, mode_def):
        """Compute CMIP6 indices and patterns for a mode."""
        if not self.cmip6_enabled:
            return None, None, {}, {}, {}

        var = mode_def.variable
        vinfo = get_var(var)
        if not vinfo.cmip6_variable:
            return None, None, {}, {}, {}

        # The zonal-mean modes (QBO) have no CMIP6 implementation —
        # ``_compute_cmip6_single`` returns ``(None, None)`` for them.  Skip
        # the loop entirely so we never load the full 4-D pressure-level field
        # (e.g. ``ua``) per model just to discard it — that eager load is both
        # wasted I/O and an OOM risk for fine grids.
        if mode_def.method == "zonal_mean":
            logger.info(
                "  %s: no CMIP6 zonal-mean implementation — skipping CMIP6",
                mode_def.name,
            )
            return None, None, {}, {}, {}

        individual_idx: dict[str, xr.DataArray] = {}
        individual_pat: dict[str, xr.DataArray] = {}
        models_used = []

        member_pairs = self.cmip6_loader.get_member_pairs()
        n_total = len(member_pairs)
        for i, (cmip6_model, variant) in enumerate(member_pairs):
            label = f"{cmip6_model}/{variant}"
            try:
                logger.debug(
                    "    CMIP6 %s (%d/%d)...", label, i + 1, n_total,
                )
                da = self.cmip6_loader.load_var(
                    vinfo.cmip6_variable, cmip6_model,
                    variant=variant,
                    table=vinfo.cmip6_table or None,
                    period=self.period,
                    time_mean=False,
                )
                if da is None:
                    continue
                if "time" not in da.dims or da.sizes["time"] < 12:
                    continue

                # Normalise lons to 0..360 and lats ascending
                # (only for rectilinear grids; curvilinear handled
                # inside _extract_box_mean via masking)
                try:
                    lat_name, lon_name, is_rect = self._find_latlon(da)
                except ValueError:
                    continue
                if is_rect:
                    lons = da[lon_name].values
                    if np.any(lons < 0):
                        da = da.assign_coords({lon_name: lons % 360})
                        da = da.sortby(lon_name)
                    lats = da[lat_name].values
                    if len(lats) > 1 and lats[0] > lats[-1]:
                        da = da.sortby(lat_name)

                idx, pat = self._compute_cmip6_single(mode_def, da)
                if idx is not None:
                    individual_idx[label] = idx
                    models_used.append(label)
                    logger.debug(
                        "    CMIP6 %s: index OK (std=%.4f)",
                        label, float(idx.std()),
                    )
                if pat is not None:
                    individual_pat[label] = pat
                    logger.debug("    CMIP6 %s: pattern OK", label)
                elif idx is not None:
                    logger.info(
                        "    CMIP6 %s: index OK but NO pattern", label,
                    )
            except Exception:
                logger.warning(
                    "  CMIP6 %s failed for %s", label, mode_def.name,
                    exc_info=True,
                )

        if not individual_idx:
            return None, None, {}, {}, {}

        # MMM of indices: align + average
        aligned = xr.align(*individual_idx.values(), join="inner")
        if len(aligned[0].time) < 3:
            return (None, None, individual_idx, individual_pat,
                    {"n_members": len(models_used),
                     "models_used": models_used})
        mmm_idx = sum(aligned) / len(aligned)

        # MMM of patterns: align + average
        mmm_pat = None
        if individual_pat:
            try:
                pat_aligned = xr.align(*individual_pat.values(), join="inner")
                mmm_pat = sum(pat_aligned) / len(pat_aligned)
            except Exception:
                logger.debug("  CMIP6 MMM pattern alignment failed",
                             exc_info=True)

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        if not self.cmip6_individual:
            individual_idx = {}
            individual_pat = {}

        return mmm_idx, mmm_pat, individual_idx, individual_pat, cmip6_info

    def _compute_cmip6_single(self, mode_def, da):
        """Compute index + pattern from a single CMIP6 DataArray.

        Returns (index, pattern).  Pattern is None for curvilinear
        grids (can't be displayed on common lat/lon grid).
        """
        try:
            _, _, is_rect = self._find_latlon(da)
        except ValueError:
            return None, None

        if mode_def.method == "box_mean":
            box = list(mode_def.boxes.values())[0]
            ts = self._extract_box_mean(da, box)
            idx = deseason(ts)
            if hasattr(idx, "compute"):
                idx = idx.compute()
            pat = self._regression_pattern(da, idx)
            return idx, pat
        elif mode_def.method == "box_diff":
            box_names = list(mode_def.boxes.keys())
            ts_a = self._extract_box_mean(da, mode_def.boxes[box_names[0]])
            ts_b = self._extract_box_mean(da, mode_def.boxes[box_names[1]])
            diff = ts_a - ts_b
            idx = deseason(diff)
            if hasattr(idx, "compute"):
                idx = idx.compute()
            pat = self._regression_pattern(da, idx)
            return idx, pat
        elif mode_def.method == "eof":
            if not is_rect:
                # Non-rectilinear: use flat EOF (index only, no pattern)
                if mode_def.detrend_global:
                    gm = self._field_global_mean(da)
                    da = da - gm
                da_anom = deseason(da)
                if hasattr(da_anom, "compute"):
                    da_anom = da_anom.compute()
                if da_anom.sizes["time"] < 12:
                    return None, None
                pc1, var_exp = self._compute_eof_flat(
                    da_anom, mode_def, n_modes=1,
                )
                return pc1, None  # no spatial pattern
            da_region = self._extract_eof_region(da, mode_def)
            if mode_def.detrend_global:
                gm = self._field_global_mean(da)
                da_region = da_region - gm
            da_anom = deseason(da_region)
            if hasattr(da_anom, "compute"):
                da_anom = da_anom.compute()
            if da_anom.sizes["time"] < 12:
                return None, None
            eofs, pcs, _ = compute_eof(da_anom, n_modes=1)
            pc1 = pcs.sel(mode=1)
            eof1 = eofs.sel(mode=1)
            if mode_def.sign_point is not None:
                pc1, eof1 = self._fix_eof_sign(
                    pc1, eof1, mode_def.sign_point,
                )
            return pc1, eof1
        elif mode_def.method == "zonal_mean":
            return None, None
        return None, None

    # ── Plotting ────────────────────────────────────────────────────

    def _plot_mode(
        self,
        mode_def: ModeDefinition,
        result: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Generate 4 figures for a single mode."""
        figures: list[tuple[plt.Figure, dict]] = []

        all_models = list(result["model_indices"].keys())
        cmip6_info = result.get("cmip6_info", {})

        # 1. Time series
        fig_ts, meta_ts = self._plot_timeseries(mode_def, result)
        if fig_ts is not None:
            figures.append((fig_ts, meta_ts))

        # 2. Spatial pattern — zonal-mean modes (QBO) have no horizontal
        # pattern by definition, so skip the (empty) placeholder figure.
        if mode_def.method != "zonal_mean":
            fig_pat, meta_pat = self._plot_pattern(mode_def, result)
            if fig_pat is not None:
                figures.append((fig_pat, meta_pat))

        # 3. Power spectrum
        fig_sp, meta_sp = self._plot_spectrum(mode_def, result)
        if fig_sp is not None:
            figures.append((fig_sp, meta_sp))

        # 4. Seasonal variance
        fig_sv, meta_sv = self._plot_seasonal_variance(mode_def, result)
        if fig_sv is not None:
            figures.append((fig_sv, meta_sv))

        return figures

    def _plot_timeseries(self, mode_def, result):
        """Index time series: one subplot per source with red/blue fill."""
        all_models = list(result["model_indices"].keys())

        # Build ordered list of (label, index, color) for subplots
        panels: list[tuple[str, xr.DataArray, str]] = []

        # Observations first
        obs_idx = result.get("obs_index")
        if obs_idx is not None:
            panels.append((mode_def.obs_label, obs_idx, OBS_COLOR))

        # Models
        for model, idx in result["model_indices"].items():
            color = self.config.get_model_color(model)
            panels.append((model, idx, color))

        # No CMIP6 MMM: averaging mode indices across models smears out
        # the variability signal (modes are not in phase).

        if not panels:
            return None, None

        n_panels = len(panels)
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(14, 2.5 * n_panels + 0.5),
            sharex=True, sharey=True,
            squeeze=False,
        )

        # Determine shared y-axis limits
        all_vals = []
        for _, idx, _ in panels:
            v = idx.values
            v = v[np.isfinite(v)]
            all_vals.extend(v)
        if all_vals:
            ymax = float(np.max(np.abs(all_vals))) * 1.1
        else:
            ymax = 3.0

        stats = {}
        for i, (label, idx, color) in enumerate(panels):
            ax = axes[i, 0]
            times = idx.time.values
            vals = idx.values

            # Monthly line
            ax.plot(times, vals, color="k", linewidth=0.5, alpha=0.6)

            # Red/blue fill
            ax.fill_between(
                times, vals, 0,
                where=(vals >= 0), color="red", alpha=0.5,
                interpolate=True,
            )
            ax.fill_between(
                times, vals, 0,
                where=(vals < 0), color="blue", alpha=0.5,
                interpolate=True,
            )

            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_ylim(-ymax, ymax)
            ax.set_ylabel(label, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if label not in (mode_def.obs_label, "CMIP6 MMM"):
                stats[label] = {
                    "std": float(idx.std()),
                    "mean": float(idx.mean()),
                }

        axes[-1, 0].set_xlabel("Time")
        fig.suptitle(
            f"{mode_def.long_name} — Index Time Series",
            fontsize=14, y=1.0,
        )
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{mode_def.long_name} — Index Time Series",
            figure_id=f"{mode_def.name}_timeseries",
            models=all_models,
            variables=[mode_def.variable],
            description=(
                f"Monthly {mode_def.long_name} index for each source. "
                f"Positive values in red, negative in blue. "
                f"Typical period: {mode_def.typical_period}."
            ),
            plot_type="teleconnection_timeseries",
            period=self.period,
            cmip6_info=result.get("cmip6_info") or None,
            summary_statistics=stats,
        )
        return fig, meta

    def _plot_pattern(self, mode_def, result):
        """Spatial pattern: EOF loading or regression map.

        All patterns are regridded to a common 1-degree grid and plotted
        with nereus for coastlines and consistent projection.
        """
        patterns = {}
        cmip6_labels: set[str] = set()
        if result.get("obs_pattern") is not None:
            patterns[mode_def.obs_label] = result["obs_pattern"]
        for model, pat in result.get("model_patterns", {}).items():
            if pat is not None:
                patterns[model] = pat
        # CMIP6 individual patterns only (when --cmip6-individual)
        # No MMM pattern: averaging EOF/regression patterns across models
        # smears out the mode structure and is not physically meaningful.
        for label, pat in result.get(
            "cmip6_individual_patterns", {},
        ).items():
            if pat is not None:
                patterns[label] = pat
                cmip6_labels.add(label)

        if not patterns:
            # No patterns to plot — create placeholder
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"No spatial pattern for {mode_def.long_name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{mode_def.long_name} — Spatial Pattern")

            meta = self._build_metadata(
                title=f"{mode_def.long_name} — Spatial Pattern",
                figure_id=f"{mode_def.name}_pattern",
                models=list(result["model_indices"].keys()),
                variables=[mode_def.variable],
                description=f"No spatial pattern available for {mode_def.long_name}.",
                plot_type="teleconnection_pattern",
                period=self.period,
            )
            return fig, meta

        # Regrid all patterns to a common grid
        patterns = self._regrid_patterns_to_common(patterns)

        # Per-mode projection
        map_proj = mode_def.plot_projection  # "rob", "np", "sp"
        map_extent = mode_def.plot_extent
        is_polar = map_proj in ("np", "sp")

        # Multi-panel figure with cartopy projection
        n_panels = len(patterns)
        ncols = min(n_panels, 3)
        nrows = (n_panels + ncols - 1) // ncols
        panel_w = 6 if is_polar else 7
        panel_h = 6 if is_polar else 5
        proj = get_projection(map_proj)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(panel_w * ncols, panel_h * nrows),
            subplot_kw={"projection": proj},
            squeeze=False,
        )

        # Shared colorbar range (symmetric)
        all_vals = []
        for pat in patterns.values():
            v = np.asarray(pat).ravel()
            v = v[np.isfinite(v)]
            all_vals.extend(v)
        if all_vals:
            vmax = float(np.percentile(np.abs(all_vals), 98)) or 1.0
        else:
            vmax = 1.0

        if mode_def.method == "eof":
            cb_label = f"EOF loading ({get_var(mode_def.variable).units})"
        else:
            cb_label = f"Regression ({get_var(mode_def.variable).units}/STD)"

        plot_kwargs: dict[str, Any] = {
            "projection": map_proj,
            "resolution": 0.5,
            "cmap": "RdBu_r",
            "vmin": -vmax,
            "vmax": vmax,
            "colorbar": False,
            "land": True,
        }
        if map_extent is not None:
            plot_kwargs["extent"] = map_extent

        # All patterns are on the common 1° grid after regridding,
        # so use linear interpolation for smooth rendering.
        plot_kwargs["method"] = "linear"
        interpolator = None
        for i, (label, pat) in enumerate(patterns.items()):
            row, col = divmod(i, ncols)
            ax = axes[row, col]

            lats = pat["lat"].values
            lons = pat["lon"].values
            lon2d, lat2d = np.meshgrid(lons, lats)

            _, _, interpolator = nr.plot(
                pat.values.ravel(), lon2d.ravel(), lat2d.ravel(),
                ax=ax, interpolator=interpolator,
                title=label, **plot_kwargs,
            )

        # Remove unused axes
        for i in range(n_panels, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"{mode_def.long_name} — Spatial Pattern", fontsize=14, y=0.98,
        )
        fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.1)

        # Shared horizontal colorbar
        cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax),
        )
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=cb_label)

        all_models = list(result["model_indices"].keys())
        var_exp_stats = {}
        for model, ve in result.get("model_var_explained", {}).items():
            var_exp_stats[model] = {"variance_explained": ve}
        if result.get("obs_var_explained") is not None:
            var_exp_stats[mode_def.obs_label] = {
                "variance_explained": result["obs_var_explained"],
            }

        meta = self._build_metadata(
            title=f"{mode_def.long_name} — Spatial Pattern",
            figure_id=f"{mode_def.name}_pattern",
            models=all_models,
            variables=[mode_def.variable],
            description=(
                f"Spatial pattern of {mode_def.long_name}. "
                f"{'EOF loading pattern' if mode_def.method == 'eof' else 'Regression map (field onto index)'}. "
                f"Diverging colorbar centred on zero."
            ),
            plot_type="teleconnection_pattern",
            period=self.period,
            cmip6_info=result.get("cmip6_info") or None,
            summary_statistics=var_exp_stats if var_exp_stats else None,
        )
        return fig, meta

    @staticmethod
    def _regrid_patterns_to_common(patterns):
        """Regrid all patterns to a common 1-degree lat/lon grid.

        Handles both rectilinear patterns (``lat``/``lon`` dims) and
        curvilinear patterns (2D coordinate arrays) by using
        ``nr.regrid`` for the latter.
        """
        # Separate rectilinear and curvilinear patterns
        rect_patterns: dict[str, xr.DataArray] = {}
        curv_patterns: dict[str, xr.DataArray] = {}

        for key, pat in patterns.items():
            is_rect = ("lat" in pat.dims or "latitude" in pat.dims)
            if is_rect:
                # Normalise dim names
                lat_name = "lat" if "lat" in pat.dims else "latitude"
                lon_name = "lon" if "lon" in pat.dims else "longitude"
                if lat_name != "lat" or lon_name != "lon":
                    pat = pat.rename({lat_name: "lat", lon_name: "lon"})
                lons = pat["lon"].values
                if np.any(lons > 180):
                    new_lons = np.where(lons > 180, lons - 360, lons)
                    pat = pat.assign_coords(lon=new_lons).sortby("lon")
                rect_patterns[key] = pat
            else:
                curv_patterns[key] = pat

        # Collect all lat/lon extents for common bounding box
        lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
        for pat in rect_patterns.values():
            lat_mins.append(float(pat["lat"].min()))
            lat_maxs.append(float(pat["lat"].max()))
            lon_mins.append(float(pat["lon"].min()))
            lon_maxs.append(float(pat["lon"].max()))
        for pat in curv_patterns.values():
            lat_np, lon_np, _ = TeleconnectionDiag._get_latlon_arrays(pat)
            # min/max of each coord array independently (works for
            # 1D, 2D, and mismatched-length 1D arrays)
            lon_np = lon_np % 360
            lon_np = np.where(lon_np > 180, lon_np - 360, lon_np)
            lat_mins.append(float(np.nanmin(lat_np)))
            lat_maxs.append(float(np.nanmax(lat_np)))
            lon_mins.append(float(np.nanmin(lon_np)))
            lon_maxs.append(float(np.nanmax(lon_np)))

        if not lat_mins:
            return {}

        lat_lo = np.floor(min(lat_mins))
        lat_hi = np.ceil(max(lat_maxs))
        lon_lo = np.floor(min(lon_mins))
        lon_hi = np.ceil(max(lon_maxs))

        target_lats = np.arange(lat_lo, lat_hi + 0.5, 1.0)
        target_lons = np.arange(lon_lo, lon_hi + 0.5, 1.0)

        result = {}

        # Rectilinear: xarray interp
        for key, pat in rect_patterns.items():
            result[key] = pat.interp(
                lat=target_lats, lon=target_lons, method="nearest",
            )

        # Curvilinear / renamed-rectilinear: nereus regrid
        # Pass validated lon/lat arrays explicitly (extract_coordinates
        # returns integer indices for some grids like IPSL ORCA).
        for key, pat in curv_patterns.items():
            try:
                lat_np, lon_np, _ = TeleconnectionDiag._get_latlon_arrays(
                    pat,
                )
                regridded, _ = nr.regrid(
                    pat.values, lon_np, lat_np,
                    resolution=1.0,
                    influence_radius=200_000,
                    lon_bounds=(float(lon_lo), float(lon_hi)),
                    lat_bounds=(float(lat_lo), float(lat_hi)),
                    as_xarray=True,
                )
                # Align to same target grid as rectilinear patterns
                if "lat" in regridded.dims and "lon" in regridded.dims:
                    result[key] = regridded.interp(
                        lat=target_lats, lon=target_lons,
                        method="nearest",
                    )
                else:
                    result[key] = regridded
            except Exception:
                logger.warning(
                    "  Failed to regrid curvilinear pattern for %s", key,
                    exc_info=True,
                )

        return result

    def _plot_spectrum(self, mode_def, result):
        """Power spectrum: Welch periodogram."""
        fig, ax = plt.subplots(figsize=(10, 5))

        all_models = list(result["model_indices"].keys())

        # CMIP6 individual (background)
        cmip6_individual = result.get("cmip6_individual", {})
        for i, (label, idx) in enumerate(cmip6_individual.items()):
            periods, psd = power_spectrum(idx.values)
            if len(periods) > 0:
                ax.plot(
                    periods, psd,
                    color=CMIP6_COLOR, alpha=0.2, linewidth=0.5,
                    label="CMIP6 members" if i == 0 else "_nolegend_",
                )

        # No CMIP6 MMM spectrum: averaging mode indices smears out
        # the variability signal, producing artificially flat spectra.

        # Models
        for model, idx in result["model_indices"].items():
            color = self.config.get_model_color(model)
            periods, psd = power_spectrum(idx.values)
            if len(periods) > 0:
                ax.plot(periods, psd, color=color, linewidth=1.5,
                        label=model)

        # Observations
        obs_idx = result.get("obs_index")
        if obs_idx is not None:
            periods, psd = power_spectrum(obs_idx.values)
            if len(periods) > 0:
                ax.plot(periods, psd, color=OBS_COLOR, linewidth=2.5,
                        label=mode_def.obs_label)

        # Mark typical period range
        if mode_def.typical_period and "-" in mode_def.typical_period:
            try:
                parts = mode_def.typical_period.replace("~", "").split("-")
                p_min = float(parts[0].strip().split()[0])
                p_max = float(parts[1].strip().split()[0])
                ax.axvspan(p_min, p_max, alpha=0.1, color="gold",
                           label=f"Typical ({mode_def.typical_period})")
            except (ValueError, IndexError):
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(f"{mode_def.long_name} — Power Spectrum")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.invert_xaxis()  # Longest periods on left
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{mode_def.long_name} — Power Spectrum",
            figure_id=f"{mode_def.name}_spectrum",
            models=all_models,
            variables=[mode_def.variable],
            description=(
                f"Welch periodogram of {mode_def.long_name} index. "
                f"Typical period: {mode_def.typical_period}. "
                f"Log-log axes with longest periods on left."
            ),
            plot_type="teleconnection_spectrum",
            period=self.period,
            cmip6_info=result.get("cmip6_info") or None,
        )
        return fig, meta

    def _plot_seasonal_variance(self, mode_def, result):
        """Monthly STD of index (seasonal variance profile)."""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = np.arange(1, 13)
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]
        all_models = list(result["model_indices"].keys())
        has_obs = result.get("obs_index") is not None
        # No CMIP6 MMM: averaging mode indices smears out variability.
        n_sources = len(all_models)
        if has_obs:
            n_sources += 1
        bar_width = 0.8 / max(n_sources, 1)
        stats: dict[str, dict] = {}

        source_idx = 0

        # Model bars
        for model, idx in result["model_indices"].items():
            color = self.config.get_model_color(model)
            monthly_std = idx.groupby("time.month").std()
            offset = (source_idx - n_sources / 2 + 0.5) * bar_width
            ax.bar(
                months + offset, monthly_std.values,
                bar_width, color=color, label=model,
            )
            stats[model] = {
                "peak_month": int(monthly_std.argmax() + 1),
                "peak_std": float(monthly_std.max()),
                "annual_std": float(idx.std()),
            }
            source_idx += 1

        # Obs bars
        if has_obs:
            obs_idx = result["obs_index"]
            monthly_std = obs_idx.groupby("time.month").std()
            offset = (source_idx - n_sources / 2 + 0.5) * bar_width
            ax.bar(
                months + offset, monthly_std.values,
                bar_width, color=OBS_COLOR, label=mode_def.obs_label,
            )

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("Month")
        ax.set_ylabel("Standard Deviation")
        ax.set_title(
            f"{mode_def.long_name} — Seasonal Variance Profile"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{mode_def.long_name} — Seasonal Variance Profile",
            figure_id=f"{mode_def.name}_seasonal_variance",
            models=all_models,
            variables=[mode_def.variable],
            description=(
                f"Monthly standard deviation of {mode_def.long_name} "
                f"index, showing seasonal dependence of variability. "
                f"Expected peak: {mode_def.seasonal_peak}."
            ),
            plot_type="teleconnection_seasonal_variance",
            period=self.period,
            cmip6_info=result.get("cmip6_info") or None,
            summary_statistics=stats if stats else None,
        )
        return fig, meta
