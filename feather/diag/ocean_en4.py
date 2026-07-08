"""Ocean 3D evaluation diagnostic against EN4 v4.2.2.

Compares DestinE high-resolution models against EN4 v4.2.2 observations
for temperature (thetao) and salinity (so).  Produces surface bias maps,
Hovmoller (time-depth) diagrams, and depth-layer mean time series.

All temperature data is converted from Kelvin to degrees Celsius for display.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import OBS_COLOR
from feather.util.spatial import (
    compute_latlon_areas,
    latlon_global_mean,
)
from feather.util.temporal import (
    annual_mean,
    climatology,
    seasonal_climatology,
)

logger = logging.getLogger(__name__)

_K_TO_C = 273.15

# Depth ranges for volume-mean time series (meters)
_DEPTH_RANGES = [
    (0, 700, "0\u2013700 m"),
    (700, 2000, "700\u20132000 m"),
    (2000, None, "2000 m\u2013bottom"),
]

# Variable display configuration
_VAR_CFG = {
    "thetao": {
        "long_name": "Temperature",
        "short": "T",
        "units": "\u00b0C",
        "en4_var": "thetao",
        "destine_var": "avg_thetao",
        "convert": lambda da: da - _K_TO_C,
        "cmap": "cmo.thermal",
        "bias_cmap": "RdBu_r",
    },
    "so": {
        "long_name": "Salinity",
        "short": "S",
        "units": "PSU",
        "en4_var": "so",
        "destine_var": "avg_so",
        "convert": lambda da: da,  # no conversion needed
        "cmap": "YlGnBu",
        "bias_cmap": "RdBu_r",
    },
}


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values


@register
class OceanEN4(DiagnosticBase):
    """Ocean 3D evaluation against EN4 v4.2.2.

    Produces 12 figures across 8 groups:
    A) SST bias maps (3): annual, DJF, JJA — surface thetao
    B) SSS bias maps (3): annual, DJF, JJA — surface so
    C) T Hovmoller anomaly-from-first-timestep (1): combined EN4 + models
    D) S Hovmoller anomaly-from-first-timestep (1): combined EN4 + models
    E) T Hovmoller anomaly-from-EN4-reference (1): combined EN4 + models
    F) S Hovmoller anomaly-from-EN4-reference (1): combined EN4 + models
    G) T depth-layer time series (1): 3 panels (0-700, 700-2000, 2000-btm)
    H) S depth-layer time series (1): 3 panels (0-700, 700-2000, 2000-btm)
    """

    name = "ocean_en4"
    title = "Ocean Evaluation (EN4)"
    domain = "o3d"
    variables = ["thetao", "so"]
    group = "ocean_3d"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks,
                         save_netcdf=save_netcdf)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # ── Unit conversion ────────────────────────────────────────────────

    def _get_convert(self, variable: str, *, for_obs: bool = False):
        """Return the appropriate unit conversion function.

        For model data: CMOR (EERIE) thetao is already in °C — skip
        conversion.  DestinE thetao is in Kelvin — convert to °C.

        For obs data (EN4): thetao is always stored in Kelvin on
        Levante, so always convert regardless of model data source.
        """
        if variable == "thetao" and not for_obs:
            if self.config.get_data_source_type() == "cmor":
                return lambda da: da  # CMOR model data already in °C
        return _VAR_CFG[variable]["convert"]

    # ── Salinity conversion (SA → SP) ─────────────────────────────────

    def _apply_sa_to_sp(
        self,
        da: xr.DataArray,
        model: str,
        *,
        model_coords: tuple | None = None,
        depth: np.ndarray | None = None,
    ) -> xr.DataArray:
        """Convert absolute salinity to practical salinity for NEMO models.

        NEMO-based models (IFS-NEMO, HadGEM3) output absolute salinity
        (TEOS-10, g/kg) instead of practical salinity (EOS-80, PSU).
        This applies ``gsw.SP_from_SA(SA, p, lon, lat)`` where pressure
        is approximated from ocean depth (p ≈ depth in dbar).

        No-op for models without ``absolute_salinity: true`` in config.
        """
        mcfg = self.config.model_configs.get(model)
        if not (mcfg and mcfg.absolute_salinity):
            return da

        import gsw

        # --- Pressure (dbar ≈ depth in metres) ---
        depth_dim = next(
            (d for d in da.dims if d in ("lev", "depth", "level", "deptht")), None)

        if depth_dim is not None:
            if depth is not None:
                # External depth array (handles DestinE integer-indexed levels)
                coords = ({depth_dim: da[depth_dim].values}
                          if depth_dim in da.coords else None)
                p = xr.DataArray(
                    depth.astype(np.float64), dims=depth_dim, coords=coords)
            else:
                # CMOR: lev coordinate IS real depth in metres
                p = da[depth_dim].astype(np.float64)
        else:
            p = 0.0  # surface data

        # --- Latitude & longitude ---
        lat_da: xr.DataArray | float = 0.0
        lon_da: xr.DataArray | float = 0.0

        for lat_name in ("lat", "latitude"):
            if lat_name in da.dims:
                lat_da = da[lat_name].astype(np.float64)
                break

        for lon_name in ("lon", "longitude"):
            if lon_name in da.dims:
                lon_da = da[lon_name].astype(np.float64)
                break

        # HEALPix: lat/lon not in dims, use external coordinates
        if isinstance(lat_da, float) and model_coords is not None:
            lon_ext, lat_ext = model_coords
            spatial_dim = next(
                (d for d in da.dims if d == "values"), None)
            if spatial_dim:
                lat_da = xr.DataArray(
                    np.asarray(lat_ext, dtype=np.float64), dims=spatial_dim)
                lon_da = xr.DataArray(
                    np.asarray(lon_ext, dtype=np.float64), dims=spatial_dim)

        logger.info("Converting SA → SP for %s using gsw.SP_from_SA", model)

        return xr.apply_ufunc(
            gsw.SP_from_SA,
            da, p, lon_da, lat_da,
            dask='parallelized',
            output_dtypes=[float],
        )

    # ── Depth level helpers ────────────────────────────────────────────

    def _get_depth_levels(
        self, model: str, da: xr.DataArray | None = None,
    ) -> np.ndarray:
        """Look up full depth levels (cell centres) for a model.

        Priority:
        1. Config ``ocean_3d.depth_levels`` (required for DestinE models
           whose depth dimension uses integer indices, not real depths).
        2. ``lev`` or ``depth`` coordinate from the loaded DataArray
           (works for CMOR data that stores real depth values in metres).
        """
        o3d_cfg = self.config.ocean_3d
        depth_levels = o3d_cfg.get("depth_levels", {})
        if model in depth_levels:
            return np.array(depth_levels[model], dtype=np.float64)

        # Fallback: extract from data coordinate (CMOR files)
        if da is not None:
            for dim_name in ("lev", "depth", "deptht"):
                if dim_name in da.coords:
                    values = da[dim_name].values.astype(np.float64)
                    if len(values) > 1:
                        logger.info(
                            "Using '%s' coordinate from data for %s "
                            "(%d levels, %.1f–%.1f m)",
                            dim_name, model, len(values),
                            values[0], values[-1],
                        )
                        return values

        raise KeyError(
            f"No depth levels configured for model {model!r}. "
            f"Available: {list(depth_levels.keys())}"
        )

    def _get_layer_thickness(
        self, model: str, da: xr.DataArray | None = None,
    ) -> np.ndarray:
        """Compute layer thickness from half-levels or approximate from centres.

        Returns array of shape (n_levels,) with thickness in metres.
        """
        o3d_cfg = self.config.ocean_3d
        half_levels = o3d_cfg.get("half_levels", {})
        if model in half_levels:
            hl = np.array(half_levels[model], dtype=np.float64)
            return np.diff(hl)

        # Try lev_bnds from the DataArray coordinates (CMOR files)
        if da is not None:
            for dim_name in ("lev", "depth", "deptht"):
                bnds_name = f"{dim_name}_bnds"
                if bnds_name in da.coords:
                    bnds = da[bnds_name].values
                    thickness = bnds[:, 1] - bnds[:, 0]
                    logger.info(
                        "Using '%s' from data for %s layer thickness",
                        bnds_name, model,
                    )
                    return thickness.astype(np.float64)

        # Fallback: approximate from full level midpoints
        depth = self._get_depth_levels(model, da=da)
        # Approximate half-levels as midpoints between full levels
        hl = np.zeros(len(depth) + 1)
        hl[0] = 0.0
        for i in range(len(depth) - 1):
            hl[i + 1] = 0.5 * (depth[i] + depth[i + 1])
        hl[-1] = depth[-1] + (depth[-1] - hl[-2])
        return np.diff(hl)

    @staticmethod
    def _get_en4_thickness(ds: xr.Dataset) -> np.ndarray:
        """Extract layer thickness from EN4 lev_bnds.

        Falls back to approximation from level coordinates.
        """
        if "lev_bnds" in ds:
            bnds = ds["lev_bnds"].values
            return bnds[:, 1] - bnds[:, 0]
        if "lev" in ds.coords:
            lev = ds["lev"].values
            # Approximate
            thickness = np.zeros(len(lev))
            thickness[0] = lev[1] - lev[0] if len(lev) > 1 else lev[0] * 2
            for i in range(1, len(lev) - 1):
                thickness[i] = 0.5 * (lev[i + 1] - lev[i - 1])
            if len(lev) > 1:
                thickness[-1] = lev[-1] - lev[-2]
            return thickness
        raise KeyError("EN4 dataset has neither 'lev_bnds' nor 'lev'")

    # ── Orchestration (per-figure-group incremental) ──────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-figure-group: compute -> plot -> save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        out = self.output_dir

        # Benchmark (CMIP6/HighResMIP) surface bias NetCDFs for Added Value.
        from feather.diag import ocean_bias
        ocean_bias.maybe_export_ocean_bias(
            self, [v for v in ("thetao", "so") if v in self.variables],
            want_individual=self.cmip6_individual, skip_existing=skip_existing,
        )

        # Determine which groups need work
        bias_a_ids = [f"en4_sst_{p}_bias_combined"
                      for p in ("annual", "djf", "jja")]
        bias_b_ids = [f"en4_sss_{p}_bias_combined"
                      for p in ("annual", "djf", "jja")]

        need_a = "thetao" in self.variables and (
            not skip_existing or not all(
                self._figure_exists(f) for f in bias_a_ids))
        need_b = "so" in self.variables and (
            not skip_existing or not all(
                self._figure_exists(f) for f in bias_b_ids))

        # Hovmoller figures (C-F) — one combined figure per type
        need_hov = {}
        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            vcfg = _VAR_CFG[var]
            prefix = f"en4_{vcfg['en4_var']}_hovmoller"
            anom1_id = f"{prefix}_anom1_combined"
            anomref_id = f"{prefix}_anomref_combined"
            need_anom1 = not skip_existing or not self._figure_exists(anom1_id)
            need_anomref = (not skip_existing
                            or not self._figure_exists(anomref_id))
            need_hov[var] = (need_anom1, need_anomref)

        # Depth time series (G-H)
        need_depth = {}
        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            vcfg = _VAR_CFG[var]
            fid = f"en4_{vcfg['en4_var']}_depth_timeseries"
            need_depth[var] = not skip_existing or not self._figure_exists(fid)

        # Collect already-existing paths
        if not need_a:
            if "thetao" in self.variables:
                logger.info("Skipping SST bias maps -- figures exist")
                saved.extend([
                    (out / f"{f}.png", out / f"{f}.json") for f in bias_a_ids
                ])
        if not need_b:
            if "so" in self.variables:
                logger.info("Skipping SSS bias maps -- figures exist")
                saved.extend([
                    (out / f"{f}.png", out / f"{f}.json") for f in bias_b_ids
                ])

        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            vcfg = _VAR_CFG[var]
            prefix = f"en4_{vcfg['en4_var']}_hovmoller"
            a1, ar = need_hov.get(var, (False, False))
            if not a1:
                fid = f"{prefix}_anom1_combined"
                saved.append((out / f"{fid}.png", out / f"{fid}.json"))
            if not ar:
                fid = f"{prefix}_anomref_combined"
                saved.append((out / f"{fid}.png", out / f"{fid}.json"))

            fid = f"en4_{vcfg['en4_var']}_depth_timeseries"
            if not need_depth.get(var, False):
                if var in self.variables:
                    logger.info("Skipping %s depth time series -- exists", var)
                    saved.append((out / f"{fid}.png", out / f"{fid}.json"))

        anything_needed = (
            need_a or need_b
            or any(a or b for a, b in need_hov.values())
            or any(need_depth.values())
        )
        if not anything_needed:
            logger.info(
                "Diagnostic %s complete -- all figures exist", self.name,
            )
            return saved

        # ── Load shared data ─────────────────────────────────────────
        model_3d, model_coords, model_depths, model_thickness = (
            self._load_model_data()
        )
        en4_data, en4_ds_cache = self._load_en4_data()

        # ── Group A: SST bias maps ───────────────────────────────────
        if need_a:
            results = self._compute_bias_maps(
                "thetao", model_3d, model_coords, en4_data,
            )
            self._maybe_export_netcdf(results, "en4_thetao_bias")
            for fig, meta in self._plot_bias_maps("thetao", results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # ── Group B: SSS bias maps ───────────────────────────────────
        if need_b:
            results = self._compute_bias_maps(
                "so", model_3d, model_coords, en4_data,
            )
            self._maybe_export_netcdf(results, "en4_so_bias")
            for fig, meta in self._plot_bias_maps("so", results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # ── Groups C-F: Hovmoller diagrams ───────────────────────────
        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            a1_need, ar_need = need_hov.get(var, (False, False))
            if not a1_need and not ar_need:
                continue

            hov_data = self._compute_hovmoller(
                var, model_3d, model_depths, model_thickness,
                en4_data, en4_ds_cache, model_coords=model_coords,
            )
            self._maybe_export_netcdf(hov_data, f"en4_{var}_hovmoller")

            if a1_need:
                for fig, meta in self._plot_hovmoller_anom1(var, hov_data):
                    saved.append(self._save(fig, meta, meta["figure_id"]))

            if ar_need:
                for fig, meta in self._plot_hovmoller_anomref(var, hov_data):
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # ── Groups G-H: Depth-layer time series ──────────────────────
        for var in self.variables:
            if var not in _VAR_CFG or not need_depth.get(var, False):
                continue
            ts_data = self._compute_depth_timeseries(
                var, model_3d, model_depths, model_thickness,
                en4_data, en4_ds_cache, model_coords=model_coords,
            )
            self._maybe_export_netcdf(ts_data, f"en4_{var}_depth_ts")
            for fig, meta in self._plot_depth_timeseries(var, ts_data):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        logger.info(
            "Diagnostic %s complete -- %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Abstract interface (backward compat) ──────────────────────────

    def compute(self) -> dict[str, Any]:
        """Compute all results (backward compat wrapper)."""
        model_3d, model_coords, model_depths, model_thickness = (
            self._load_model_data()
        )
        en4_data, en4_ds_cache = self._load_en4_data()
        results: dict[str, Any] = {}

        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            results[f"{var}_bias"] = self._compute_bias_maps(
                var, model_3d, model_coords, en4_data,
            )
            results[f"{var}_hov"] = self._compute_hovmoller(
                var, model_3d, model_depths, model_thickness,
                en4_data, en4_ds_cache, model_coords=model_coords,
            )
            results[f"{var}_depth_ts"] = self._compute_depth_timeseries(
                var, model_3d, model_depths, model_thickness,
                en4_data, en4_ds_cache, model_coords=model_coords,
            )
        return results

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figures (backward compat wrapper)."""
        figures = []
        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            if f"{var}_bias" in results:
                figures.extend(
                    self._plot_bias_maps(var, results[f"{var}_bias"]))
            if f"{var}_hov" in results:
                figures.extend(
                    self._plot_hovmoller_anom1(var, results[f"{var}_hov"]))
                figures.extend(
                    self._plot_hovmoller_anomref(var, results[f"{var}_hov"]))
            if f"{var}_depth_ts" in results:
                figures.extend(
                    self._plot_depth_timeseries(
                        var, results[f"{var}_depth_ts"]))
        return figures

    # ── Shared data loading ───────────────────────────────────────────

    def _load_model_data(self):
        """Load 3D model data for all available models.

        Returns
        -------
        model_3d : dict[str, dict[str, xr.DataArray]]
            Model name -> variable name -> DataArray (time, level, values).
        model_coords : dict[str, tuple]
            Model name -> (lon, lat) coordinate arrays.
        model_depths : dict[str, np.ndarray]
            Model name -> depth level array.
        model_thickness : dict[str, np.ndarray]
            Model name -> layer thickness array.
        """
        model_3d: dict[str, dict[str, xr.DataArray]] = {}
        model_coords: dict[str, tuple] = {}
        model_depths: dict[str, np.ndarray] = {}
        model_thickness: dict[str, np.ndarray] = {}

        for model in self.config.models:
            var_data = {}
            coords_set = False
            for var in self.variables:
                if var not in _VAR_CFG:
                    continue
                try:
                    da = self._load_model_var(
                        model, var, period=self.period,
                    )
                    var_data[var] = da
                    if not coords_set:
                        lon, lat = self._load_model_coords(model, var)
                        model_coords[model] = (lon, lat)
                        coords_set = True
                except (KeyError, FileNotFoundError):
                    logger.warning("%s not available for %s", var, model)

            if not var_data:
                logger.warning("o3d data not available for %s", model)
                continue

            model_3d[model] = var_data

            # Use first available DataArray for depth coordinate fallback
            sample_da = next(iter(var_data.values()))
            try:
                model_depths[model] = self._get_depth_levels(
                    model, da=sample_da)
                model_thickness[model] = self._get_layer_thickness(
                    model, da=sample_da)
            except KeyError as e:
                logger.warning("Depth levels not configured for %s: %s",
                               model, e)

        return model_3d, model_coords, model_depths, model_thickness

    def _load_en4_data(self):
        """Load EN4 observation data for all requested variables.

        Returns
        -------
        en4_data : dict[str, xr.DataArray]
            EN4 variable name -> DataArray (time, lev, lat, lon).
        en4_ds_cache : dict[str, xr.Dataset]
            EN4 variable name -> full Dataset (for lev_bnds access).
        """
        en4_data: dict[str, xr.DataArray] = {}
        en4_ds_cache: dict[str, xr.Dataset] = {}

        for var in self.variables:
            if var not in _VAR_CFG:
                continue
            en4_var = _VAR_CFG[var]["en4_var"]
            try:
                ds = self.obs_loader.load_en4_dataset(
                    en4_var, period=self.period)
                da = self.obs_loader.load_en4(en4_var, period=self.period)
                en4_data[var] = da
                en4_ds_cache[var] = ds
            except (KeyError, FileNotFoundError, AttributeError) as e:
                logger.warning("EN4 %s loading failed: %s", en4_var, e)

        return en4_data, en4_ds_cache

    # ── Group A/B: Bias maps ──────────────────────────────────────────

    def _compute_bias_maps(self, variable, model_3d, model_coords, en4_data):
        """Compute surface bias maps for a variable."""
        import nereus as nr

        vcfg = _VAR_CFG[variable]
        convert = self._get_convert(variable)
        obs_convert = self._get_convert(variable, for_obs=True)
        en4_var = vcfg["en4_var"]
        logger.info("Computing %s surface bias maps...", vcfg["long_name"])

        # Extract EN4 surface (first depth level)
        en4_da = en4_data.get(variable)
        if en4_da is None:
            logger.warning("No EN4 data for %s, skipping bias maps", variable)
            return {"models": {}, "periods": {}}

        # EN4 surface layer: take first lev index
        lev_dim = "lev" if "lev" in en4_da.dims else "depth"
        en4_sfc = obs_convert(en4_da.isel({lev_dim: 0}))

        # Compute EN4 climatologies
        obs_annual = climatology(en4_sfc, self.period).compute()
        obs_seasonal = seasonal_climatology(en4_sfc, self.period)
        obs_djf = (obs_seasonal["DJF"].compute()
                   if "DJF" in obs_seasonal else obs_annual)
        obs_jja = (obs_seasonal["JJA"].compute()
                   if "JJA" in obs_seasonal else obs_annual)

        resolution = self.config.nereus.get("resolution", 0.25)
        # Model (HEALPix nside=128, ~27 km spacing) uses standard radius
        model_influence = self.config.nereus.get("influence_radius", 80_000.0)
        # EN4 is 1° (~111 km spacing) — needs ≥1.5× grid spacing
        en4_influence = self.config.nereus.get(
            "en4_influence_radius", 200_000.0)

        periods_data = {
            "annual": {"obs": obs_annual},
            "djf": {"obs": obs_djf},
            "jja": {"obs": obs_jja},
        }
        model_results = {}
        _model_interp_cache: dict[int, Any] = {}
        obs_interpolator = None
        target_lats = None
        target_lons = None
        common_area = None

        for model in model_3d:
            if variable not in model_3d[model]:
                continue

            da = model_3d[model][variable]
            lon, lat = model_coords[model]

            # For latlon grids, meshgrid 1D coord arrays to per-pixel arrays
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                regrid_lon, regrid_lat = np.meshgrid(lon, lat)
            else:
                regrid_lon, regrid_lat = np.asarray(lon), np.asarray(lat)

            # Extract surface level and convert
            level_dim = "level" if "level" in da.dims else da.dims[1]
            model_sfc = convert(da.isel({level_dim: 0}))
            # SA → SP conversion for NEMO-based models
            if variable == "so":
                model_sfc = self._apply_sa_to_sp(model_sfc, model)

            # Compute model climatologies
            model_annual = climatology(model_sfc, self.period).compute()
            model_seasonal = seasonal_climatology(model_sfc, self.period)
            model_djf = (model_seasonal["DJF"].compute()
                         if "DJF" in model_seasonal else model_annual)
            model_jja = (model_seasonal["JJA"].compute()
                         if "JJA" in model_seasonal else model_annual)

            # Build/reuse interpolator keyed by source grid size
            n_src = regrid_lon.ravel().shape[0]
            if n_src not in _model_interp_cache:
                _, interp = nr.regrid(
                    model_annual.values.ravel(),
                    lon=regrid_lon.ravel(), lat=regrid_lat.ravel(),
                    resolution=resolution,
                    influence_radius=model_influence,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _model_interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )

                    # Regrid EN4 obs to common grid
                    lat_name = ("lat" if "lat" in obs_annual.coords
                                else "latitude")
                    lon_name = ("lon" if "lon" in obs_annual.coords
                                else "longitude")
                    obs_lats = obs_annual[lat_name].values
                    obs_lons = obs_annual[lon_name].values
                    obs_lons_2d, obs_lats_2d = np.meshgrid(
                        obs_lons, obs_lats,
                    )

                    _, obs_interpolator = nr.regrid(
                        obs_annual.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=resolution,
                        influence_radius=en4_influence,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )

                    for pkey in periods_data:
                        obs_field = periods_data[pkey]["obs"]
                        obs_common = xr.DataArray(
                            obs_interpolator(obs_field.values.ravel()),
                            dims=("lat", "lon"),
                            coords={
                                "lat": target_lats, "lon": target_lons,
                            },
                        )
                        periods_data[pkey]["obs_common"] = obs_common

            model_interpolator = _model_interp_cache[n_src]

            # Regrid model to common grid
            regrid_fields = {
                "annual": model_annual,
                "djf": model_djf,
                "jja": model_jja,
            }
            model_results[model] = {}
            for pkey, field in regrid_fields.items():
                regrid = xr.DataArray(
                    model_interpolator(field.values.ravel()),
                    dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )
                obs_common = periods_data[pkey]["obs_common"]
                bias = regrid - obs_common
                bias_gmean = float(
                    latlon_global_mean(bias, area=common_area).values)
                rmse = float(np.sqrt(
                    latlon_global_mean(bias ** 2, area=common_area).values))
                model_results[model][pkey] = {
                    "regrid": regrid,
                    "bias": bias,
                    "bias_gmean": bias_gmean,
                    "rmse": rmse,
                }

        return {
            "models": model_results,
            "periods": periods_data,
            "common_area": common_area,
        }

    def _plot_bias_maps(self, variable, results):
        """Plot combined bias maps for a variable's surface field."""
        from feather.plot.maps import plot_combined_bias_map

        vcfg = _VAR_CFG[variable]
        prefix = "sst" if variable == "thetao" else "sss"
        figures = []
        periods_data = results["periods"]

        for pkey, plabel in [
            ("annual", "Annual Mean"),
            ("djf", "DJF"),
            ("jja", "JJA"),
        ]:
            obs_common = periods_data[pkey].get("obs_common")
            if obs_common is None:
                continue

            bias_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in results["models"].items():
                if pkey in mdata:
                    bias_dict[model] = mdata[pkey]["bias"]
                    summary_stats[model] = {
                        "global_mean_bias": mdata[pkey]["bias_gmean"],
                        "rmse": mdata[pkey]["rmse"],
                    }
                    all_models.append(model)

            if not bias_dict:
                continue

            fig, axes = plot_combined_bias_map(
                obs_common, bias_dict,
                title=f"{vcfg['long_name']} Surface {plabel}",
                obs_title="EN4",
                cmap=vcfg["cmap"],
                bias_cmap=vcfg["bias_cmap"],
                units=vcfg["units"],
                land=True,
                method=self._regrid_method,
            )

            meta = self._build_metadata(
                title=f"{vcfg['long_name']} Surface {plabel} Bias",
                figure_id=f"en4_{prefix}_{pkey}_bias_combined",
                models=all_models,
                description=(
                    f"{plabel} surface {vcfg['long_name'].lower()} "
                    f"climatology and model biases relative to EN4 v4.2.2."
                ),
                plot_type="combined_bias_map",
                period=self.period,
                obs_dataset="EN4 v4.2.2",
                obs_variable=vcfg["en4_var"],
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures

    # ── Groups C-F: Hovmoller diagrams ────────────────────────────────

    def _compute_hovmoller(self, variable, model_3d, model_depths,
                           model_thickness, en4_data, en4_ds_cache,
                           model_coords=None):
        """Compute raw Hovmoller (time-depth) data for models and EN4.

        Returns dict with model Hovmollers, EN4 Hovmoller, and EN4
        reference profile (first-year mean).
        """
        import nereus as nr

        vcfg = _VAR_CFG[variable]
        convert = self._get_convert(variable)
        obs_convert = self._get_convert(variable, for_obs=True)
        logger.info("Computing %s Hovmoller diagrams...", vcfg["long_name"])

        model_hovs: dict[str, dict] = {}

        for model in model_3d:
            if variable not in model_3d[model]:
                continue
            if model not in model_depths:
                logger.warning(
                    "No depth levels for %s, skipping Hovmoller", model)
                continue

            da = model_3d[model][variable]
            depth = model_depths[model]

            # Load cell areas (grid-type dependent)
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                ncells = da.sizes.get("values", da.shape[-1])
                try:
                    mesh = nr.healpix.load_mesh(ncells)
                    area = mesh.area.values
                except Exception:
                    area = np.ones(ncells) * (4 * np.pi / ncells)
            else:
                # Latlon: compute areas from coordinates
                lon, lat = model_coords[model]
                area = compute_latlon_areas(
                    np.asarray(lat), np.asarray(lon),
                ).ravel()

            # Convert (lazy — stays dask-backed)
            da_conv = convert(da)
            # SA → SP conversion for NEMO-based models
            if variable == "so":
                mc = model_coords.get(model) if model_coords else None
                da_conv = self._apply_sa_to_sp(
                    da_conv, model, model_coords=mc, depth=depth)
            time_vals = da_conv.time.values

            # nr.hovmoller is dask-friendly: pass DataArray directly,
            # nereus computes area-weighted means chunk by chunk
            try:
                logger.info("Computing %s Hovmoller (dask)...", model)
                hov = nr.hovmoller(
                    da_conv, area, depth=depth,
                    mode="depth", as_xarray=True,
                ).compute()
                # Ensure original time coordinates are assigned
                hov = hov.assign_coords(time=time_vals)
            except Exception as e:
                logger.warning("Hovmoller failed for %s/%s: %s",
                               model, variable, e)
                continue

            model_hovs[model] = {
                "hovmoller": hov,
                "time": time_vals,
                "depth": depth,
            }

        # EN4 Hovmoller
        en4_hov = None

        en4_da = en4_data.get(variable)
        if en4_da is not None:
            en4_conv = obs_convert(en4_da)

            # EN4 depth coordinate
            lev_dim = "lev" if "lev" in en4_conv.dims else "depth"
            en4_depth = en4_conv[lev_dim].values

            # Build area from EN4 lat/lon via nereus mesh
            lat_name = "lat" if "lat" in en4_conv.dims else "latitude"
            lon_name = "lon" if "lon" in en4_conv.dims else "longitude"
            en4_mesh = nr.mesh_from_arrays(
                en4_conv[lon_name].values, en4_conv[lat_name].values)
            en4_area = en4_mesh.area.values  # 1D (npoints,)

            en4_time = en4_conv.time.values
            logger.info("Computing EN4 Hovmoller...")

            # nr.hovmoller auto-flattens 4D (time, lev, lat, lon)
            try:
                en4_hov = nr.hovmoller(
                    en4_conv, en4_area, depth=en4_depth,
                    mode="depth", as_xarray=True,
                )
                if hasattr(en4_hov, "compute"):
                    en4_hov = en4_hov.compute()
                en4_hov = en4_hov.assign_coords(time=en4_time)
            except Exception as e:
                logger.warning("EN4 Hovmoller failed: %s", e)

        return {
            "models": model_hovs,
            "en4_hov": en4_hov,
            "en4_depth": en4_depth if en4_da is not None else None,
        }

    def _plot_hovmoller_anom1(self, variable, hov_data):
        """Plot combined Hovmoller — anomaly relative to own first timestep.

        One figure with EN4 + all models as subpanels, shared color range.
        """
        import nereus as nr

        vcfg = _VAR_CFG[variable]
        en4_var = vcfg["en4_var"]

        # Collect panels: (label, time, depth, raw_data) — anomaly=True
        # lets nr.plot_hovmoller subtract the first timestep internally
        panels = []
        en4_hov = hov_data.get("en4_hov")
        if en4_hov is not None:
            panels.append(("EN4", en4_hov.time.values,
                           en4_hov.depth.values, en4_hov.values))
        for model, mhov in hov_data["models"].items():
            panels.append((model, mhov["hovmoller"].time.values,
                           mhov["depth"], mhov["hovmoller"].values))

        if not panels:
            return []

        # Pre-compute anomalies to determine shared color range
        anoms = [d - d[0:1, :] for _, _, _, d in panels]
        vmax = max(np.nanpercentile(np.abs(a), 98) for a in anoms)
        if vmax == 0:
            vmax = 1.0

        npanels = len(panels)
        fig, axes = plt.subplots(
            npanels, 1, figsize=(12, 4 * npanels), squeeze=False)

        for i, (label, time, depth, data) in enumerate(panels):
            ax = axes[i, 0]
            nr.plot_hovmoller(
                time, depth, data,
                anomaly=True, y_scale="sqrt",
                cmap="RdBu_r", ax=ax,
                vmin=-vmax, vmax=vmax,
                colorbar=True,
                colorbar_label=vcfg["units"],
            )
            ax.set_title(label)
            ax.set_ylabel("Depth (m)")
            if i < npanels - 1:
                ax.set_xlabel("")

        fig.suptitle(
            f"{vcfg['long_name']} Hovmoller \u2014 anomaly from first timestep",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()

        fid = f"en4_{en4_var}_hovmoller_anom1_combined"
        meta = self._build_metadata(
            title=f"{vcfg['long_name']} Hovmoller (first-timestep anomaly)",
            figure_id=fid,
            models=list(hov_data["models"].keys()),
            description=(
                f"Combined time-depth Hovmoller diagram of "
                f"{vcfg['long_name'].lower()} anomaly relative to first "
                f"timestep for EN4 and all models."
            ),
            plot_type="hovmoller",
            period=self.period,
            obs_dataset="EN4 v4.2.2",
        )
        return [(fig, meta)]

    def _plot_hovmoller_anomref(self, variable, hov_data):
        """Plot combined Hovmoller — anomaly relative to EN4 first timestep.

        One figure with EN4 + all models as subpanels, shared color range.
        Each panel shows data minus EN4's January first-timestep profile.
        """
        import nereus as nr

        vcfg = _VAR_CFG[variable]
        en4_var = vcfg["en4_var"]
        en4_hov = hov_data.get("en4_hov")

        if en4_hov is None:
            logger.warning("No EN4 Hovmoller for %s anomref", variable)
            return []

        # EN4 reference: first timestep (January) profile
        en4_ref = en4_hov.values[0, :]  # (ndepth,)
        en4_depth = hov_data["en4_depth"]

        # Collect panels: (label, time, depth, anomaly_2d)
        panels = []
        # EN4 panel
        panels.append(("EN4", en4_hov.time.values, en4_depth,
                        en4_hov.values - en4_ref[np.newaxis, :]))

        for model, mhov in hov_data["models"].items():
            depth = mhov["depth"]
            # Interpolate EN4 ref profile to model depth levels
            ref_interp = np.interp(depth, en4_depth, en4_ref)
            anom = mhov["hovmoller"].values - ref_interp[np.newaxis, :]
            panels.append((model, mhov["hovmoller"].time.values, depth, anom))

        if not panels:
            return []

        # Shared symmetric color range
        vmax = max(np.nanpercentile(np.abs(p[3]), 98) for p in panels)
        if vmax == 0:
            vmax = 1.0

        npanels = len(panels)
        fig, axes = plt.subplots(
            npanels, 1, figsize=(12, 4 * npanels), squeeze=False)

        for i, (label, time, depth, anom) in enumerate(panels):
            ax = axes[i, 0]
            nr.plot_hovmoller(
                time, depth, anom,
                anomaly=False, y_scale="sqrt",
                cmap="RdBu_r", ax=ax,
                vmin=-vmax, vmax=vmax,
                colorbar=True,
                colorbar_label=vcfg["units"],
            )
            ax.set_title(label)
            ax.set_ylabel("Depth (m)")
            if i < npanels - 1:
                ax.set_xlabel("")

        fig.suptitle(
            f"{vcfg['long_name']} Hovmoller \u2014 anomaly from EN4 reference",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()

        fid = f"en4_{en4_var}_hovmoller_anomref_combined"
        meta = self._build_metadata(
            title=f"{vcfg['long_name']} Hovmoller (EN4-ref anomaly)",
            figure_id=fid,
            models=list(hov_data["models"].keys()),
            description=(
                f"Combined time-depth Hovmoller diagram of "
                f"{vcfg['long_name'].lower()} anomaly relative to EN4 "
                f"first-timestep reference profile."
            ),
            plot_type="hovmoller",
            period=self.period,
            obs_dataset="EN4 v4.2.2",
        )
        return [(fig, meta)]

    # ── Groups G-H: Depth-layer time series ──────────────────────────

    @staticmethod
    def _volume_mean_layers(da_flat, area, thickness, depth, time_vals,
                            *, log_ctx=""):
        """Volume-weighted means for every ``_DEPTH_RANGES`` layer in one pass.

        ``nr.volume_mean`` masks out-of-range levels but still references the
        full 3D array (``data_arr * volumes`` spans all levels), so computing
        each depth range with its own ``.compute()`` re-reads the entire field
        once per range.  Building the lazy results and computing them together
        (single dask graph) reads the source once and shares the conversion
        subgraph across ranges — numerically identical, ~N× less I/O.
        """
        import dask
        import nereus as nr

        lazy: dict[str, Any] = {}
        for dmin, dmax, label in _DEPTH_RANGES:
            dmax_eff = dmax if dmax is not None else float(depth[-1] + 1)
            try:
                lazy[label] = nr.volume_mean(
                    da_flat, area, thickness, depth,
                    depth_min=dmin, depth_max=dmax_eff, as_xarray=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "volume_mean setup failed for %s/%s: %s",
                    log_ctx, label, e)
        if not lazy:
            return {}

        labels = list(lazy)
        try:
            computed = list(dask.compute(*(lazy[l] for l in labels)))
        except Exception as e:  # noqa: BLE001
            # Fall back to per-layer compute so one bad range does not drop
            # the others (preserves the original robustness).
            logger.warning(
                "Batched volume_mean failed for %s (%s); retrying per layer",
                log_ctx, e)
            computed = []
            for l in labels:
                try:
                    ts = lazy[l]
                    computed.append(ts.compute() if hasattr(ts, "compute")
                                    else ts)
                except Exception as e2:  # noqa: BLE001
                    logger.warning("volume_mean failed for %s/%s: %s",
                                   log_ctx, l, e2)
                    computed.append(None)

        out: dict[str, xr.DataArray] = {}
        for label, ts in zip(labels, computed):
            if ts is None:
                continue
            if isinstance(ts, xr.DataArray) and "time" not in ts.dims:
                ts = xr.DataArray(
                    ts.values, dims="time", coords={"time": time_vals})
            out[label] = ts
        return out

    def _compute_depth_timeseries(self, variable, model_3d, model_depths,
                                  model_thickness, en4_data, en4_ds_cache,
                                  model_coords=None):
        """Compute volume-weighted depth-layer mean time series."""
        import nereus as nr

        vcfg = _VAR_CFG[variable]
        convert = self._get_convert(variable)
        obs_convert = self._get_convert(variable, for_obs=True)
        logger.info("Computing %s depth-layer time series...",
                     vcfg["long_name"])

        model_ts: dict[str, dict[str, xr.DataArray]] = {}

        for model in model_3d:
            if variable not in model_3d[model]:
                continue
            if model not in model_depths:
                continue

            da = model_3d[model][variable]
            depth = model_depths[model]
            thickness = model_thickness[model]

            # Load cell areas (grid-type dependent)
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                ncells = da.sizes.get("values", da.shape[-1])
                try:
                    mesh = nr.healpix.load_mesh(ncells)
                    area = mesh.area.values
                except Exception:
                    area = np.ones(ncells) * (4 * np.pi / ncells)
            else:
                # Latlon: compute areas from coordinates
                lon, lat = model_coords[model]
                area = compute_latlon_areas(
                    np.asarray(lat), np.asarray(lon),
                ).ravel()

            # Convert (lazy — stays dask-backed)
            da_conv = convert(da)
            # SA → SP conversion for NEMO-based models
            if variable == "so":
                mc = model_coords.get(model) if model_coords else None
                da_conv = self._apply_sa_to_sp(
                    da_conv, model, model_coords=mc, depth=depth)
            time_vals = da_conv.time.values

            # Flatten spatial dims for nr.volume_mean which expects
            # (ntime, nlevels, npoints) — 4D latlon needs stacking
            lat_dim = next(
                (d for d in da_conv.dims if d in ("lat", "latitude")), None)
            lon_dim = next(
                (d for d in da_conv.dims if d in ("lon", "longitude")), None)
            if lat_dim and lon_dim:
                da_flat = da_conv.stack(space=(lat_dim, lon_dim))
            else:
                da_flat = da_conv  # HEALPix: already 3D

            logger.info("Computing %s depth timeseries (dask)...", model)
            # All depth ranges computed in one dask pass (reads the 3D field
            # once instead of once per range — see _volume_mean_layers).
            layer_ts = self._volume_mean_layers(
                da_flat, area, thickness, depth, time_vals,
                log_ctx=f"{model}/{variable}",
            )

            if layer_ts:
                model_ts[model] = layer_ts

        # EN4 depth-layer time series
        # EN4 depth-layer time series — use nr.volume_mean()
        en4_ts: dict[str, xr.DataArray] = {}
        en4_da = en4_data.get(variable)
        if en4_da is not None:
            en4_conv = obs_convert(en4_da)
            en4_ds = en4_ds_cache.get(variable)

            lev_dim = "lev" if "lev" in en4_conv.dims else "depth"
            en4_depth = en4_conv[lev_dim].values

            try:
                en4_thick = self._get_en4_thickness(en4_ds)
            except KeyError:
                # Approximate from depth centers
                en4_thick = np.ones(len(en4_depth))
                if len(en4_depth) > 1:
                    en4_thick[0] = en4_depth[1] - en4_depth[0]
                    for i in range(1, len(en4_depth) - 1):
                        en4_thick[i] = 0.5 * (en4_depth[i+1] - en4_depth[i-1])
                    en4_thick[-1] = en4_depth[-1] - en4_depth[-2]

            # Build area from EN4 lat/lon via nereus mesh
            lat_name = "lat" if "lat" in en4_conv.dims else "latitude"
            lon_name = "lon" if "lon" in en4_conv.dims else "longitude"
            en4_mesh = nr.mesh_from_arrays(
                en4_conv[lon_name].values, en4_conv[lat_name].values)
            en4_area = en4_mesh.area.values  # 1D (npoints,)

            # Flatten spatial dims for volume_mean
            logger.info("Computing EN4 depth-layer time series...")
            en4_lat_dim = next(
                (d for d in en4_conv.dims if d in ("lat", "latitude")), None)
            en4_lon_dim = next(
                (d for d in en4_conv.dims if d in ("lon", "longitude")), None)
            if en4_lat_dim and en4_lon_dim:
                en4_flat = en4_conv.stack(
                    space=(en4_lat_dim, en4_lon_dim))
            else:
                en4_flat = en4_conv

            # All depth ranges in one dask pass (see _volume_mean_layers).
            en4_ts = self._volume_mean_layers(
                en4_flat, en4_area, en4_thick, en4_depth,
                en4_conv.time.values, log_ctx=f"EN4/{variable}",
            )

        return {"models": model_ts, "en4": en4_ts}

    def _plot_depth_timeseries(self, variable, ts_data):
        """Plot depth-layer time series with 3 subplots."""
        vcfg = _VAR_CFG[variable]
        en4_var = vcfg["en4_var"]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        all_models = []

        for i, (dmin, dmax, label) in enumerate(_DEPTH_RANGES):
            ax = axes[i]

            # Monthly semi-transparent background
            for model, layer_ts in ts_data["models"].items():
                if label not in layer_ts:
                    continue
                ts = layer_ts[label]
                color = self.config.get_model_color(model)
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values, color=color, alpha=0.3,
                        linewidth=0.7)

            en4_ts = ts_data.get("en4", {})
            if label in en4_ts:
                ts = en4_ts[label]
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values, color=OBS_COLOR, alpha=0.3,
                        linewidth=0.7)

            # Annual thick foreground
            for model, layer_ts in ts_data["models"].items():
                if label not in layer_ts:
                    continue
                ts = layer_ts[label]
                ts_annual = annual_mean(ts)
                color = self.config.get_model_color(model)
                time_vals = _to_plot_time(ts_annual.time.values)
                lbl = model if i == 0 else None
                ax.plot(time_vals, ts_annual.values, label=lbl,
                        color=color, linewidth=2.0)
                if model not in all_models:
                    all_models.append(model)

            if label in en4_ts:
                ts_annual = annual_mean(en4_ts[label])
                time_vals = _to_plot_time(ts_annual.time.values)
                lbl = "EN4" if i == 0 else None
                ax.plot(time_vals, ts_annual.values, label=lbl,
                        color=OBS_COLOR, linewidth=2.5)

            ax.set_title(label)
            ax.set_ylabel(f"{vcfg['long_name']} ({vcfg['units']})")
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc="upper left")
        fig.suptitle(
            f"{vcfg['long_name']} Depth-Layer Mean Time Series",
            fontsize=14, y=1.0,
        )
        plt.tight_layout()

        fid = f"en4_{en4_var}_depth_timeseries"
        meta = self._build_metadata(
            title=f"{vcfg['long_name']} Depth-Layer Time Series",
            figure_id=fid,
            models=all_models + ["EN4"],
            description=(
                f"Volume-weighted mean {vcfg['long_name'].lower()} time "
                f"series in three depth ranges (0-700m, 700-2000m, "
                f"2000m-bottom). Monthly as semi-transparent lines, "
                f"annual means as thick lines."
            ),
            plot_type="depth_timeseries",
            period=self.period,
            obs_dataset="EN4 v4.2.2",
        )
        return [(fig, meta)]


