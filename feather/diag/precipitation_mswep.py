"""Precipitation evaluation diagnostic against MSWEP v2.8.

Produces 8 figures across 6 groups:
A (x3): Absolute bias maps (annual, DJF, JJA)
B (x1): Relative bias map (annual, %)
C (x1): Global-mean time series
D (x1): Seasonal cycle
E (x1): Zonal mean profile
F (x1): Precipitation intensity distribution (PDF)
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import xarray as xr

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.spatial import (
    compute_latlon_areas,
    latlon_global_mean,
    zonal_mean,
)
from feather.util.temporal import (
    annual_mean,
    climatology,
    monthly_climatology,
    seasonal_climatology,
)

logger = logging.getLogger(__name__)

# Minimum obs precipitation threshold for relative bias (kg/m²/s).
# Below this, relative bias is masked to avoid division artifacts.
_REL_BIAS_THRESHOLD = 0.1 / 86400  # ~0.1 mm/day

# Conversion factor: kg/m²/s → mm/day (used for display only).
_PR_TO_MMDAY = 86400.0


@register
class PrecipitationMSWEP(DiagnosticBase):
    """Precipitation evaluation against MSWEP v2.8.

    Uses MSWEP (Multi-Source Weighted-Ensemble Precipitation) as the
    primary reference dataset instead of ERA5.  Provides richer
    statistics than the generic ``pr`` evaluation in GlobalBiases.
    """

    name = "precipitation_mswep"
    title = "Precipitation Evaluation (MSWEP)"
    domain = "sfc"
    variables = ["pr"]
    group = "precipitation"

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
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # ── Orchestration (per-group incremental) ─────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple["Path", "Path"]]:
        """Execute per-group: compute -> plot -> save.

        Groups are processed independently so partial progress is
        preserved if a later group fails.
        """
        from pathlib import Path

        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        out = self.output_dir

        # Determine which groups need computing
        bias_ids = [
            f"pr_{p}_bias_combined"
            for p in ["annual", "djf", "mam", "jja", "son"]
        ]
        need_a = not skip_existing or not all(
            self._figure_exists(fid) for fid in bias_ids
        )
        rel_bias_ids = [
            f"pr_{p}_relative_bias"
            for p in ["annual", "djf", "mam", "jja", "son"]
        ]
        need_b = not skip_existing or not all(
            self._figure_exists(fid) for fid in rel_bias_ids
        )
        need_c = not skip_existing or not self._figure_exists(
            "pr_timeseries"
        )
        need_d = not skip_existing or not self._figure_exists(
            "pr_seasonal_cycle"
        )
        need_e = not skip_existing or not self._figure_exists(
            "pr_zonal_mean"
        )
        need_f = not skip_existing or not self._figure_exists(
            "pr_intensity_distribution"
        )

        # Collect existing paths
        if not need_a:
            logger.info("Skipping bias maps -- figures exist")
            saved.extend([
                (out / f"{fid}.png", out / f"{fid}.json")
                for fid in bias_ids
            ])
        if not need_b:
            logger.info("Skipping relative bias -- figures exist")
            saved.extend([
                (out / f"{fid}.png", out / f"{fid}.json")
                for fid in rel_bias_ids
            ])
        if not need_c:
            logger.info("Skipping timeseries -- figure exists")
            saved.append((
                out / "pr_timeseries.png", out / "pr_timeseries.json",
            ))
        if not need_d:
            logger.info("Skipping seasonal cycle -- figure exists")
            saved.append((
                out / "pr_seasonal_cycle.png",
                out / "pr_seasonal_cycle.json",
            ))
        if not need_e:
            logger.info("Skipping zonal mean -- figure exists")
            saved.append((
                out / "pr_zonal_mean.png", out / "pr_zonal_mean.json",
            ))
        if not need_f:
            logger.info("Skipping intensity PDF -- figure exists")
            saved.append((
                out / "pr_intensity_distribution.png",
                out / "pr_intensity_distribution.json",
            ))

        if not any([need_a, need_b, need_c, need_d, need_e, need_f]):
            logger.info(
                "Diagnostic %s complete -- all figures exist", self.name,
            )
            return saved

        # Load shared data (model + obs)
        try:
            shared = self._load_shared_data()
        except Exception:
            logger.warning(
                "Failed to load data for precipitation_mswep",
                exc_info=True,
            )
            return saved

        # Group A: Absolute bias maps
        if need_a:
            try:
                results = self._compute_bias_maps(shared)
                for fig, meta in self._plot_bias_maps(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group A (bias maps) failed", exc_info=True,
                )

        # Group B: Relative bias map
        if need_b:
            try:
                results = self._compute_relative_bias(shared)
                for fig, meta in self._plot_relative_bias(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group B (relative bias) failed", exc_info=True,
                )

        # Group C: Timeseries
        if need_c:
            try:
                results = self._compute_timeseries(shared)
                for fig, meta in self._plot_timeseries(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group C (timeseries) failed", exc_info=True,
                )

        # Group D: Seasonal cycle
        if need_d:
            try:
                results = self._compute_seasonal_cycle(shared)
                for fig, meta in self._plot_seasonal_cycle(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group D (seasonal cycle) failed", exc_info=True,
                )

        # Group E: Zonal mean
        if need_e:
            try:
                results = self._compute_zonal_mean(shared)
                for fig, meta in self._plot_zonal_mean(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group E (zonal mean) failed", exc_info=True,
                )

        # Group F: Intensity PDF
        if need_f:
            try:
                results = self._compute_intensity_pdf(shared)
                for fig, meta in self._plot_intensity_pdf(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning(
                    "Group F (intensity PDF) failed", exc_info=True,
                )

        logger.info(
            "Diagnostic %s complete -- %d figure(s)", self.name, len(saved),
        )
        return saved

    def compute(self) -> dict[str, Any]:
        """Compute all results (backward compat wrapper)."""
        shared = self._load_shared_data()
        return {
            "bias_maps": self._compute_bias_maps(shared),
            "relative_bias": self._compute_relative_bias(shared),
            "timeseries": self._compute_timeseries(shared),
            "seasonal_cycle": self._compute_seasonal_cycle(shared),
            "zonal_mean": self._compute_zonal_mean(shared),
            "intensity_pdf": self._compute_intensity_pdf(shared),
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figures (backward compat wrapper)."""
        figures = []
        figures.extend(self._plot_bias_maps(results["bias_maps"]))
        figures.extend(self._plot_relative_bias(results["relative_bias"]))
        figures.extend(self._plot_timeseries(results["timeseries"]))
        figures.extend(self._plot_seasonal_cycle(results["seasonal_cycle"]))
        figures.extend(self._plot_zonal_mean(results["zonal_mean"]))
        figures.extend(self._plot_intensity_pdf(results["intensity_pdf"]))
        return figures

    # ── Shared data loading ──────────────────────────────────────────

    def _load_shared_data(self) -> dict[str, Any]:
        """Load model and MSWEP data used across multiple groups."""
        logger.info("Loading shared data for precipitation_mswep...")

        model_monthly: dict[str, xr.DataArray] = {}
        model_coords: dict[str, tuple] = {}

        for model in self.config.models:
            try:
                da = self._load_model_var(model, "pr", period=self.period)
                model_monthly[model] = da
                lon, lat = self._load_model_coords(model, "pr")
                model_coords[model] = (lon, lat)
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "Variable pr not available for %s -- skipping", model,
                )

        if not model_monthly:
            raise RuntimeError("No models have pr data")

        # Load MSWEP
        logger.info("  Loading MSWEP v2.8 observations...")
        mswep = self.obs_loader.load_mswep(period=self.period)

        return {
            "model_monthly": model_monthly,
            "model_coords": model_coords,
            "mswep": mswep,
        }

    # ── Group A: Absolute bias maps ──────────────────────────────────

    def _compute_bias_maps(self, shared: dict) -> dict[str, Any]:
        """Compute annual and seasonal precipitation bias maps."""
        logger.info("Computing precipitation bias maps...")
        var_info = get_var("pr")
        model_monthly = shared["model_monthly"]
        model_coords = shared["model_coords"]
        mswep = shared["mswep"]

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )

        # Obs climatologies
        obs_clim = climatology(mswep, self.period)
        obs_seasonal = seasonal_climatology(mswep, self.period)
        obs_gmean = float(latlon_global_mean(obs_clim).values)

        obs_lats = obs_clim.lat.values
        obs_lons = obs_clim.lon.values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Cache nereus interpolator per source grid size
        _interp_cache: dict[int, Any] = {}
        obs_clim_common = None
        obs_seasonal_common = {}
        common_area = None
        target_lats = None
        target_lons = None

        model_results: dict[str, dict] = {}

        for model in self.config.models:
            if model not in model_monthly:
                continue
            logger.info("  Computing biases for %s...", model)

            model_data = model_monthly[model]
            lon, lat = model_coords[model]

            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            model_clim = climatology(model_data, self.period).compute()
            model_gmean = float(model_clim.mean().values)

            model_seas = seasonal_climatology(model_data, self.period)
            model_seas = {
                s: model_seas[s].compute() for s in model_seas.data_vars
            }

            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _interp_cache:
                logger.info("  Building nereus interpolator (grid size %d)...",
                            n_src)
                annual_regrid, interp = nr.regrid(
                    model_clim.values.ravel(),
                    lon=np.asarray(lon).ravel(),
                    lat=np.asarray(lat).ravel(),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]

                    # Regrid obs to common grid
                    obs_lons_2d, obs_lats_2d = np.meshgrid(
                        obs_lons, obs_lats,
                    )
                    _, obs_interp = nr.regrid(
                        obs_clim.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=obs_res,
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    obs_clim_common = xr.DataArray(
                        obs_interp(obs_clim.values.ravel()),
                        dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )

                    for season in obs_seasonal:
                        s_np = obs_interp(
                            obs_seasonal[season].values.ravel(),
                        )
                        obs_seasonal_common[season] = xr.DataArray(
                            s_np, dims=("lat", "lon"),
                            coords={
                                "lat": target_lats, "lon": target_lons,
                            },
                        )
            else:
                interp = _interp_cache[n_src]
                regridded_np = interp(model_clim.values.ravel())
                annual_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            # Annual bias + statistics
            annual_bias = annual_regrid - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(annual_bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(annual_bias ** 2, area=common_area).values
            ))

            # Pattern correlation and STD ratio
            pattern_corr = self._pattern_correlation(
                annual_regrid, obs_clim_common, common_area,
            )
            std_ratio = self._std_ratio(
                annual_regrid, obs_clim_common, common_area,
            )

            # Tropical/extratropical bias
            trop_bias = self._regional_mean_bias(
                annual_bias, target_lats, common_area, lat_min=-30, lat_max=30,
            )
            extratrop_bias = self._extratropical_mean_bias(
                annual_bias, target_lats, common_area,
            )

            # Seasonal biases
            seasonal_biases: dict[str, Any] = {}
            seasonal_regrids: dict[str, Any] = {}
            for season in ["DJF", "MAM", "JJA", "SON"]:
                if season in model_seas:
                    s_np = _interp_cache[n_src](model_seas[season].values.ravel())
                    s_regrid = xr.DataArray(
                        s_np, dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
                    seasonal_regrids[season] = s_regrid
                    if season in obs_seasonal_common:
                        seasonal_biases[season] = s_regrid - obs_seasonal_common[season]

            model_results[model] = {
                "annual_regrid": annual_regrid,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": model_gmean,
                "annual_bias": annual_bias,
                "annual_bias_gmean": bias_gmean,
                "annual_rmse": rmse,
                "pattern_correlation": pattern_corr,
                "std_ratio": std_ratio,
                "tropical_mean_bias": trop_bias,
                "extratropical_mean_bias": extratrop_bias,
                "seasonal_biases": seasonal_biases,
            }

        # CMIP6 bias (optional)
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        if self.cmip6_enabled and target_lats is not None:
            if self.cmip6_individual:
                cmip6_individual_data = self._compute_cmip6_individual(
                    target_lats, target_lons,
                    obs_clim_common, obs_seasonal_common, common_area,
                )
                cmip6_data, cmip6_info = self._mmm_from_individual(
                    cmip6_individual_data,
                    obs_clim_common, obs_seasonal_common, common_area,
                )
            else:
                cmip6_data, cmip6_info = self._compute_cmip6_mmm(
                    target_lats, target_lons,
                    obs_clim_common, obs_seasonal_common, common_area,
                )

        # Shared colorbar ranges
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
            cmip6_data=cmip6_data,
            cmip6_individual_data=cmip6_individual_data,
        )

        return {
            "models": model_results,
            "obs": {
                "clim": obs_clim_common,
                "seasonal_clim": obs_seasonal_common,
                "global_mean": obs_gmean,
            },
            "var_info": var_info,
            "colorbar_ranges": colorbar_ranges,
            "cmip6_data": cmip6_data,
            "cmip6_info": cmip6_info,
            "cmip6_individual_data": cmip6_individual_data,
        }

    def _plot_bias_maps(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot absolute bias maps for annual, DJF, JJA."""
        figures = []
        var_info = results["var_info"]
        obs_clim = results["obs"]["clim"]
        cb = results["colorbar_ranges"]
        cmip6_data = results.get("cmip6_data", {})
        cmip6_info = results.get("cmip6_info", {})
        cmip6_individual_data = results.get("cmip6_individual_data", {})

        periods = [("annual", "Annual Mean")]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in cb:
                periods.append((season, season))

        for period_key, period_label in periods:
            bias_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in results["models"].items():
                if period_key == "annual":
                    bias_field = mdata["annual_bias"]
                    summary_stats[model] = {
                        "global_mean_bias": mdata["annual_bias_gmean"] * _PR_TO_MMDAY,
                        "rmse": mdata["annual_rmse"] * _PR_TO_MMDAY,
                        "pattern_correlation": mdata["pattern_correlation"],
                        "std_ratio": mdata["std_ratio"],
                        "tropical_mean_bias": mdata["tropical_mean_bias"] * _PR_TO_MMDAY,
                        "extratropical_mean_bias": mdata["extratropical_mean_bias"] * _PR_TO_MMDAY,
                    }
                else:
                    bias_field = mdata["seasonal_biases"].get(period_key)
                    if bias_field is None:
                        continue
                bias_dict[model] = bias_field
                all_models.append(model)

            # CMIP6 MMM
            if period_key in cmip6_data:
                c_data = cmip6_data[period_key]
                bias_dict["CMIP6 MMM"] = c_data["bias"]
                all_models.append("CMIP6 MMM")
                summary_stats["CMIP6 MMM"] = {
                    "global_mean_bias": c_data["bias_gmean"] * _PR_TO_MMDAY,
                    "rmse": (c_data["rmse"] * _PR_TO_MMDAY
                             if c_data.get("rmse") is not None else None),
                }

            # CMIP6 individual
            if period_key in cmip6_individual_data:
                for label, c_data in cmip6_individual_data[period_key].items():
                    bias_dict[label] = c_data["bias"]
                    all_models.append(label)

            if not bias_dict:
                continue

            # Get obs data for period
            if period_key == "annual":
                obs_period = obs_clim
            else:
                obs_period = results["obs"]["seasonal_clim"].get(period_key)
                if obs_period is None:
                    continue

            p_cb = cb.get(period_key, cb.get("annual", {}))

            # Convert to mm/day for display (internal data stays in kg/m²/s)
            obs_plot = obs_period * _PR_TO_MMDAY
            bias_plot = {k: v * _PR_TO_MMDAY for k, v in bias_dict.items()}
            vmin_plot = (p_cb["vmin"] * _PR_TO_MMDAY
                         if p_cb.get("vmin") is not None else None)
            vmax_plot = (p_cb["vmax"] * _PR_TO_MMDAY
                         if p_cb.get("vmax") is not None else None)
            bvmax_plot = (p_cb["bias_vmax"] * _PR_TO_MMDAY
                          if p_cb.get("bias_vmax") is not None else None)

            fig, axes = plot_combined_bias_map(
                obs_plot, bias_plot,
                title=f"Precipitation {period_label}",
                obs_title="MSWEP v2.8",
                cmap="YlGnBu",
                bias_cmap="BrBG",
                vmin=vmin_plot,
                vmax=vmax_plot,
                bias_vmax=bvmax_plot,
                units="mm/day",
                method=self._regrid_method,
            )

            meta = self._build_metadata(
                title=f"Precipitation {period_label} Bias",
                figure_id=f"pr_{period_key.lower()}_bias_combined",
                models=all_models,
                variables=["pr"],
                description=(
                    f"{period_label} precipitation bias maps "
                    f"(model - MSWEP v2.8)."
                ),
                obs_dataset="MSWEP",
                obs_variable="precipitation",
                plot_type="combined_bias_map",
                period=self.period,
                cmip6_info=cmip6_info or None,
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures

    # ── Group B: Relative bias ───────────────────────────────────────

    def _compute_relative_bias(self, shared: dict) -> dict[str, Any]:
        """Compute relative precipitation bias (% of obs) for annual + seasonal."""
        logger.info("Computing relative precipitation bias...")
        bias_results = self._compute_bias_maps(shared)

        obs_clim = bias_results["obs"]["clim"]
        obs_seasonal_clim = bias_results["obs"]["seasonal_clim"]
        cmip6_data = bias_results.get("cmip6_data", {})
        cmip6_individual_data = bias_results.get("cmip6_individual_data", {})

        # ── Annual ──────────────────────────────────────────────────────
        obs_masked = obs_clim.where(obs_clim > _REL_BIAS_THRESHOLD)
        rel_bias_annual: dict[str, Any] = {}
        for model, mdata in bias_results["models"].items():
            rel_bias_annual[model] = (mdata["annual_bias"] / obs_masked) * 100
        if "annual" in cmip6_data:
            rel_bias_annual["CMIP6 MMM"] = (
                cmip6_data["annual"]["bias"] / obs_masked
            ) * 100
        if "annual" in cmip6_individual_data:
            for label, c_data in cmip6_individual_data["annual"].items():
                rel_bias_annual[label] = (c_data["bias"] / obs_masked) * 100

        # ── Seasonal (DJF, JJA) ─────────────────────────────────────────
        rel_bias_seasonal: dict[str, dict] = {}
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season not in obs_seasonal_clim:
                continue
            obs_s = obs_seasonal_clim[season]
            obs_s_masked = obs_s.where(obs_s > _REL_BIAS_THRESHOLD)
            rel_s: dict[str, Any] = {}
            for model, mdata in bias_results["models"].items():
                if season in mdata.get("seasonal_biases", {}):
                    rel_s[model] = (
                        mdata["seasonal_biases"][season] / obs_s_masked
                    ) * 100
            if season in cmip6_data:
                rel_s["CMIP6 MMM"] = (
                    cmip6_data[season]["bias"] / obs_s_masked
                ) * 100
            if season in cmip6_individual_data:
                for label, c_data in cmip6_individual_data[season].items():
                    rel_s[label] = (c_data["bias"] / obs_s_masked) * 100
            if rel_s:
                rel_bias_seasonal[season] = rel_s

        return {
            "rel_bias_annual": rel_bias_annual,
            "rel_bias_seasonal": rel_bias_seasonal,
            "obs_clim": obs_clim,
        }

    def _plot_relative_bias(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot relative bias maps (%) for annual, DJF, and JJA."""
        figures = []

        periods = [("annual", "Annual")]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in results.get("rel_bias_seasonal", {}):
                periods.append((season, season))

        for period_key, period_label in periods:
            if period_key == "annual":
                bias_dict = results["rel_bias_annual"]
            else:
                bias_dict = results["rel_bias_seasonal"][period_key]
            if not bias_dict:
                continue

            fig, axes = plot_combined_map(
                bias_dict,
                title=f"Precipitation Relative Bias ({period_label})",
                cmap="BrBG",
                vmin=-100, vmax=100,
                units="%",
                method=self._regrid_method,
            )
            meta = self._build_metadata(
                title=f"Precipitation Relative Bias ({period_label})",
                figure_id=f"pr_{period_key.lower()}_relative_bias",
                models=list(bias_dict.keys()),
                variables=["pr"],
                description=(
                    f"{period_label} precipitation relative bias (%) vs "
                    "MSWEP v2.8. Masked where obs < 0.1 mm/day to avoid "
                    "division artifacts in arid regions. "
                    "Brown = dry bias, green = wet bias."
                ),
                obs_dataset="MSWEP",
                obs_variable="precipitation",
                plot_type="combined_map",
                period=self.period,
            )
            figures.append((fig, meta))

        return figures

    # ── Group C: Timeseries ──────────────────────────────────────────

    def _compute_timeseries(self, shared: dict) -> dict[str, Any]:
        """Compute global-mean precipitation time series."""
        logger.info("Computing precipitation time series...")
        model_ts: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            ts = self._model_global_mean(da, model).compute()
            model_ts[model] = ts

        mswep = shared["mswep"]
        obs_ts = latlon_global_mean(mswep)

        # CMIP6
        cmip6_ts = None
        cmip6_info = {}
        cmip6_individual_ts: dict[str, Any] = {}
        cmip6_ts, info = self._cmip6_global_mean_timeseries(
            "pr", period=self.period,
            return_individual=self.cmip6_individual,
        )
        if cmip6_ts is not None:
            cmip6_info = info
            if self.cmip6_individual and "individual_series" in info:
                cmip6_individual_ts = dict(info["individual_series"])

        return {
            "models": model_ts,
            "obs": obs_ts,
            "cmip6_ts": cmip6_ts,
            "cmip6_info": cmip6_info,
            "cmip6_individual_ts": cmip6_individual_ts,
        }

    def _plot_timeseries(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot global-mean precipitation time series."""
        import pandas as pd

        fig, ax = plt.subplots(figsize=(12, 5))
        all_models = list(self.config.models)

        cmip6_indiv = results.get("cmip6_individual_ts", {})
        if cmip6_indiv:
            all_models.extend(cmip6_indiv.keys())

        # Monthly pass (background)
        for _mname, ts in cmip6_indiv.items():
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values * _PR_TO_MMDAY,
                    color=CMIP6_COLOR, alpha=0.2, linewidth=0.5)

        if results.get("cmip6_ts") is not None:
            cmip6_ts = results["cmip6_ts"]
            time_vals = _to_plot_time(cmip6_ts.time.values)
            ax.plot(time_vals, cmip6_ts.values * _PR_TO_MMDAY,
                    color=CMIP6_COLOR, alpha=0.3, linewidth=0.7,
                    linestyle="--")

        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values * _PR_TO_MMDAY,
                    color=color, alpha=0.3, linewidth=0.7)

        obs_ts = results["obs"]
        obs_time = _to_plot_time(obs_ts.time.values)
        ax.plot(obs_time, obs_ts.values * _PR_TO_MMDAY,
                color=OBS_COLOR, alpha=0.3, linewidth=0.7)

        # Annual pass (foreground)
        for i, (mname, ts) in enumerate(cmip6_indiv.items()):
            label = "CMIP6 members" if i == 0 else "_nolegend_"
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values * _PR_TO_MMDAY,
                    color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                    label=label)

        if results.get("cmip6_ts") is not None:
            cmip6_annual = annual_mean(results["cmip6_ts"])
            time_vals = _to_plot_time(cmip6_annual.time.values)
            ax.plot(time_vals, cmip6_annual.values * _PR_TO_MMDAY,
                    label="CMIP6 MMM", color=CMIP6_COLOR,
                    linewidth=2.0, linestyle="--")

        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values * _PR_TO_MMDAY,
                    label=model, color=color, linewidth=2.0)

        obs_annual = annual_mean(obs_ts)
        obs_annual_time = _to_plot_time(obs_annual.time.values)
        ax.plot(obs_annual_time, obs_annual.values * _PR_TO_MMDAY,
                label="MSWEP", color=OBS_COLOR, linewidth=2.5)

        ax.set_title("Precipitation \u2014 Global Mean")
        ax.set_ylabel("Precipitation (mm/day)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="Precipitation Global Mean Time Series",
            figure_id="pr_timeseries",
            models=all_models,
            variables=["pr"],
            description=(
                "Area-weighted global mean monthly precipitation "
                "time series for all models vs MSWEP v2.8."
            ),
            obs_dataset="MSWEP",
            obs_variable="precipitation",
            plot_type="timeseries",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group D: Seasonal cycle ──────────────────────────────────────

    def _compute_seasonal_cycle(self, shared: dict) -> dict[str, Any]:
        """Compute monthly climatological cycle of global-mean precip."""
        logger.info("Computing precipitation seasonal cycle...")
        model_monthly_clim: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            ts = self._model_global_mean(da, model).compute()
            model_monthly_clim[model] = monthly_climatology(ts, self.period)

        mswep = shared["mswep"]
        obs_ts = latlon_global_mean(mswep)
        obs_monthly = monthly_climatology(obs_ts, self.period)

        # CMIP6
        cmip6_monthly = None
        cmip6_info = {}
        cmip6_individual_monthly: dict[str, Any] = {}
        cmip6_ts, info = self._cmip6_global_mean_timeseries(
            "pr", period=self.period,
            return_individual=self.cmip6_individual,
        )
        if cmip6_ts is not None:
            cmip6_monthly = monthly_climatology(cmip6_ts)
            cmip6_info = info
            if self.cmip6_individual and "individual_series" in info:
                for mname, mts in info["individual_series"].items():
                    cmip6_individual_monthly[mname] = monthly_climatology(mts)

        return {
            "models": model_monthly_clim,
            "obs": obs_monthly,
            "cmip6_monthly": cmip6_monthly,
            "cmip6_info": cmip6_info,
            "cmip6_individual_monthly": cmip6_individual_monthly,
        }

    def _plot_seasonal_cycle(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot 12-month seasonal cycle."""
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]

        fig, ax = plt.subplots(figsize=(8, 5))
        months = np.arange(1, 13)
        all_models = list(self.config.models)

        # Layer 1: CMIP6 individual
        cmip6_indiv = results.get("cmip6_individual_monthly", {})
        for i, (mname, monthly) in enumerate(cmip6_indiv.items()):
            label = "CMIP6 members" if i == 0 else "_nolegend_"
            ax.plot(months, monthly.values * _PR_TO_MMDAY,
                    color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                    label=label)
        if cmip6_indiv:
            all_models.extend(cmip6_indiv.keys())

        # Layer 2: CMIP6 MMM
        if results.get("cmip6_monthly") is not None:
            ax.plot(months, results["cmip6_monthly"].values * _PR_TO_MMDAY,
                    marker="d", label="CMIP6 MMM", color=CMIP6_COLOR,
                    linewidth=1.5, linestyle="--")

        # Layer 3: Model lines
        for model, monthly in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(months, monthly.values * _PR_TO_MMDAY,
                    marker="o", label=model, color=color)

        # Layer 4: Observations
        ax.plot(months, results["obs"].values * _PR_TO_MMDAY,
                marker="s", label="MSWEP", color=OBS_COLOR, linewidth=2)

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title("Precipitation \u2014 Seasonal Cycle")
        ax.set_ylabel("Precipitation (mm/day)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="Precipitation Seasonal Cycle",
            figure_id="pr_seasonal_cycle",
            models=all_models,
            variables=["pr"],
            description=(
                "Monthly climatological cycle (Jan-Dec) of global mean "
                "precipitation for all models vs MSWEP v2.8."
            ),
            obs_dataset="MSWEP",
            obs_variable="precipitation",
            plot_type="seasonal_cycle",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group E: Zonal mean ──────────────────────────────────────────

    def _compute_zonal_mean(self, shared: dict) -> dict[str, Any]:
        """Compute zonal mean precipitation profiles."""
        logger.info("Computing precipitation zonal mean profiles...")
        model_zonal: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            lon, lat = shared["model_coords"][model]
            clim = climatology(da, self.period).compute()
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                zm = zonal_mean(clim, lat)
            else:
                lon_dim = "lon" if "lon" in clim.dims else "longitude"
                zm = clim.mean(lon_dim)
                if "latitude" in zm.dims:
                    zm = zm.rename({"latitude": "lat"})
            model_zonal[model] = zm

        # Obs zonal mean
        mswep = shared["mswep"]
        obs_clim = climatology(mswep, self.period)
        obs_zonal = obs_clim.mean("lon")

        # CMIP6
        cmip6_zonal = None
        cmip6_info = {}
        if self.cmip6_enabled:
            cmip6_ts, info = self._cmip6_global_mean_timeseries(
                "pr", period=self.period,
            )
            if cmip6_ts is not None:
                cmip6_info = info
            # For zonal mean, we need spatial data not global mean
            # Compute CMIP6 MMM zonal mean from climatologies
            cmip6_zonal = self._compute_cmip6_zonal_mean()

        return {
            "models": model_zonal,
            "obs": obs_zonal,
            "cmip6_zonal": cmip6_zonal,
            "cmip6_info": cmip6_info,
        }

    def _compute_cmip6_zonal_mean(self):
        """Compute CMIP6 MMM zonal mean precipitation."""
        if not self.cmip6_enabled:
            return None

        var_info = get_var("pr")
        member_pairs = self.cmip6_loader.get_member_pairs()
        zonal_fields = []

        for model, variant in member_pairs:
            da = self.cmip6_loader.load_var_for_model_var(
                "pr", model, variant=variant, period=self.period,
            )
            if da is None:
                continue
            # Compute zonal mean
            lon_dim = "lon" if "lon" in da.dims else "longitude"
            zm = da.mean(lon_dim)
            if "latitude" in zm.dims:
                zm = zm.rename({"latitude": "lat"})
            zonal_fields.append(zm)

        if not zonal_fields:
            return None

        aligned = xr.align(*zonal_fields, join="inner")
        return sum(aligned) / len(aligned)

    def _plot_zonal_mean(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot zonal mean precipitation profile."""
        fig, ax = plt.subplots(figsize=(6, 8))
        all_models = []

        # CMIP6 MMM
        if results.get("cmip6_zonal") is not None:
            zm = results["cmip6_zonal"]
            ax.plot(zm.values * _PR_TO_MMDAY, zm.lat.values,
                    label="CMIP6 MMM", color=CMIP6_COLOR,
                    linewidth=1.5, linestyle="--")
            all_models.append("CMIP6 MMM")

        # Models
        for model, zm in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(zm.values * _PR_TO_MMDAY, zm.lat.values,
                    label=model, color=color, linewidth=1.5)
            all_models.append(model)

        # Observations
        obs_zm = results["obs"]
        ax.plot(obs_zm.values * _PR_TO_MMDAY, obs_zm.lat.values,
                label="MSWEP", color=OBS_COLOR, linewidth=2.5)
        all_models.append("MSWEP")

        ax.set_ylabel("Latitude")
        ax.set_xlabel("Precipitation (mm/day)")
        ax.set_title("Precipitation \u2014 Zonal Mean")
        ax.set_ylim(-90, 90)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="Precipitation Zonal Mean Profile",
            figure_id="pr_zonal_mean",
            models=all_models,
            variables=["pr"],
            description=(
                "Zonal mean precipitation profile for all models, "
                "MSWEP v2.8, and CMIP6 MMM. Shows ITCZ position, "
                "extratropical storm tracks, and subtropical dry zones."
            ),
            obs_dataset="MSWEP",
            obs_variable="precipitation",
            plot_type="zonal_profile",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group F: Intensity PDF ───────────────────────────────────────

    def _compute_intensity_pdf(self, shared: dict) -> dict[str, Any]:
        """Compute area-weighted precipitation intensity PDF."""
        logger.info("Computing precipitation intensity PDF...")

        # Log-spaced bins in kg/m²/s
        bins = np.logspace(-7, -3, 50)
        bin_centres = np.sqrt(bins[:-1] * bins[1:])  # geometric mean

        pdfs: dict[str, np.ndarray] = {}

        # Obs PDF
        mswep = shared["mswep"]
        obs_clim = climatology(mswep, self.period).compute()
        obs_area = compute_latlon_areas(
            obs_clim.lat.values, obs_clim.lon.values,
        )
        pdfs["MSWEP"] = self._histogram_pdf(
            obs_clim.values.ravel(), obs_area.ravel(), bins,
        )

        # Model PDFs
        for model, da in shared["model_monthly"].items():
            clim = climatology(da, self.period).compute()
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                # Equal-area cells -> uniform weights
                area = np.ones(clim.values.ravel().shape)
            else:
                lon, lat = shared["model_coords"][model]
                area = compute_latlon_areas(lat, lon).ravel()
            pdfs[model] = self._histogram_pdf(
                clim.values.ravel(), area, bins,
            )

        # CMIP6 MMM PDF (if available)
        cmip6_info = {}
        if self.cmip6_enabled:
            cmip6_pdf = self._compute_cmip6_intensity_pdf(bins)
            if cmip6_pdf is not None:
                pdfs["CMIP6 MMM"] = cmip6_pdf

        return {
            "pdfs": pdfs,
            "bins": bins,
            "bin_centres": bin_centres,
            "cmip6_info": cmip6_info,
        }

    def _compute_cmip6_intensity_pdf(self, bins: np.ndarray):
        """Compute CMIP6 MMM intensity PDF."""
        if not self.cmip6_enabled:
            return None

        member_pairs = self.cmip6_loader.get_member_pairs()
        all_vals = []
        all_areas = []

        for model, variant in member_pairs:
            da = self.cmip6_loader.load_var_for_model_var(
                "pr", model, variant=variant, period=self.period,
            )
            if da is None:
                continue
            area = self.cmip6_loader.load_area(model)
            if area is None:
                continue
            all_vals.append(da.values.ravel())
            all_areas.append(np.broadcast_to(
                np.asarray(area), da.shape,
            ).ravel())

        if not all_vals:
            return None

        combined_vals = np.concatenate(all_vals)
        combined_area = np.concatenate(all_areas)
        return self._histogram_pdf(combined_vals, combined_area, bins)

    @staticmethod
    def _histogram_pdf(values, weights, bins):
        """Compute area-weighted histogram PDF."""
        # Remove NaN and negative values
        valid = np.isfinite(values) & (values > 0)
        hist, _ = np.histogram(
            values[valid], bins=bins,
            weights=weights[valid], density=True,
        )
        return hist

    def _plot_intensity_pdf(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot precipitation intensity PDF."""
        fig, ax = plt.subplots(figsize=(10, 6))
        # Convert bin centres to mm/day for display
        bin_centres_mmday = results["bin_centres"] * _PR_TO_MMDAY
        all_models = []

        # CMIP6 MMM
        if "CMIP6 MMM" in results["pdfs"]:
            ax.plot(bin_centres_mmday, results["pdfs"]["CMIP6 MMM"],
                    label="CMIP6 MMM", color=CMIP6_COLOR,
                    linewidth=1.5, linestyle="--")
            all_models.append("CMIP6 MMM")

        # Models
        for model in self.config.models:
            if model in results["pdfs"]:
                color = self.config.get_model_color(model)
                ax.plot(bin_centres_mmday, results["pdfs"][model],
                        label=model, color=color, linewidth=1.5)
                all_models.append(model)

        # Observations
        if "MSWEP" in results["pdfs"]:
            ax.plot(bin_centres_mmday, results["pdfs"]["MSWEP"],
                    label="MSWEP", color=OBS_COLOR, linewidth=2.5)
            all_models.append("MSWEP")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Precipitation rate (mm/day)")
        ax.set_ylabel("Probability density")
        ax.set_title("Precipitation Intensity Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        meta = self._build_metadata(
            title="Precipitation Intensity Distribution",
            figure_id="pr_intensity_distribution",
            models=all_models,
            variables=["pr"],
            description=(
                "Area-weighted PDF of grid-cell annual-mean precipitation "
                "rates. Log-scale axes. Shows whether models produce too "
                "much drizzle (excess low-intensity cells) or underestimate "
                "heavy precipitation regions."
            ),
            obs_dataset="MSWEP",
            obs_variable="precipitation",
            plot_type="intensity_pdf",
            period=self.period,
        )
        return [(fig, meta)]

    # ── CMIP6 helpers ────────────────────────────────────────────────

    def _compute_cmip6_mmm(self, target_lats, target_lons,
                            obs_clim_common, obs_seasonal_common,
                            common_area):
        """Compute CMIP6 MMM precipitation biases."""
        from feather.diag.global_biases import GlobalBiases

        cmip6_data = {}
        cmip6_info = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Computing CMIP6 MMM for pr...")
        member_pairs = self.cmip6_loader.get_member_pairs()

        annual_fields = []
        seasonal_fields: dict[str, list] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        models_used = []

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                "pr", model, variant=variant, period=self.period,
            )
            if da is None:
                continue

            regridded = GlobalBiases._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            annual_fields.append(regridded)
            models_used.append(label)

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    "pr", model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is not None:
                    s_regridded = GlobalBiases._regrid_to_target(
                        da_s, target_lats, target_lons,
                        resolution, influence_radius, cmip6_interp_cache,
                        method=self._regrid_method,
                    )
                    seasonal_fields[season].append(s_regridded)

        if not annual_fields:
            return cmip6_data, cmip6_info

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            ),
            "rmse": float(np.sqrt(
                latlon_global_mean(
                    cmip6_bias ** 2, area=common_area,
                ).values
            )),
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if not seasonal_fields[season]:
                continue
            if season not in obs_seasonal_common:
                continue
            s_mmm = xr.concat(
                seasonal_fields[season], dim="member",
            ).mean("member")
            s_bias = s_mmm - obs_seasonal_common[season]
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(
                        s_bias, area=common_area,
                    ).values
                ),
            }

        return cmip6_data, cmip6_info

    def _compute_cmip6_individual(self, target_lats, target_lons,
                                   obs_clim_common, obs_seasonal_common,
                                   common_area):
        """Compute individual CMIP6 model precipitation biases."""
        from feather.diag.global_biases import GlobalBiases

        cmip6_individual_data: dict[str, dict] = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Loading individual CMIP6 models for pr...")
        member_pairs = self.cmip6_loader.get_member_pairs()

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                "pr", model, variant=variant, period=self.period,
            )
            if da is None:
                continue

            cmip6_common = GlobalBiases._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            cmip6_bias = cmip6_common - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            )

            cmip6_individual_data.setdefault("annual", {})[label] = {
                "regrid": cmip6_common,
                "bias": cmip6_bias,
                "bias_gmean": bias_gmean,
            }

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    "pr", model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is None or season not in obs_seasonal_common:
                    continue
                cmip6_s = GlobalBiases._regrid_to_target(
                    da_s, target_lats, target_lons,
                    resolution, influence_radius, cmip6_interp_cache,
                    method=self._regrid_method,
                )
                cmip6_s_bias = cmip6_s - obs_seasonal_common[season]
                cmip6_individual_data.setdefault(season, {})[label] = {
                    "regrid": cmip6_s,
                    "bias": cmip6_s_bias,
                    "bias_gmean": float(
                        latlon_global_mean(
                            cmip6_s_bias, area=common_area,
                        ).values
                    ),
                }

        return cmip6_individual_data

    @staticmethod
    def _mmm_from_individual(cmip6_individual_data, obs_clim_common,
                              obs_seasonal_common, common_area):
        """Derive MMM from already-regridded individual CMIP6 fields."""
        cmip6_data = {}

        if "annual" not in cmip6_individual_data:
            return cmip6_data, {}

        annual_members = cmip6_individual_data["annual"]
        models_used = list(annual_members.keys())
        annual_fields = [m["regrid"] for m in annual_members.values()]

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            ),
            "rmse": float(np.sqrt(
                latlon_global_mean(
                    cmip6_bias ** 2, area=common_area,
                ).values
            )),
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season not in cmip6_individual_data:
                continue
            if season not in obs_seasonal_common:
                continue
            s_fields = [
                m["regrid"]
                for m in cmip6_individual_data[season].values()
            ]
            s_mmm = xr.concat(s_fields, dim="member").mean("member")
            s_bias = s_mmm - obs_seasonal_common[season]
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(
                        s_bias, area=common_area,
                    ).values
                ),
            }

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }
        return cmip6_data, cmip6_info

    # ── Colorbar range computation ───────────────────────────────────

    @staticmethod
    def _compute_colorbar_ranges(
        model_results, obs_clim_common, obs_seasonal_common,
        cmip6_data=None, cmip6_individual_data=None,
    ):
        """Compute shared colorbar ranges across all models per period."""
        ranges: dict[str, dict] = {}
        cmip6_data = cmip6_data or {}
        cmip6_individual_data = cmip6_individual_data or {}

        def _finite_vals(arrays):
            parts = []
            for a in arrays:
                v = np.asarray(a).ravel()
                parts.append(v[np.isfinite(v)])
            return np.concatenate(parts)

        def _percentile_range(arrays):
            vals = _finite_vals(arrays)
            return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

        def _bias_max(arrays):
            vals = _finite_vals(arrays)
            return float(np.percentile(np.abs(vals), 98)) or 1.0

        # Annual
        field_arrays = [mr["annual_regrid"] for mr in model_results.values()]
        field_arrays.append(obs_clim_common)
        bias_arrays = [mr["annual_bias"] for mr in model_results.values()]
        if "annual" in cmip6_data:
            field_arrays.append(cmip6_data["annual"]["regrid"])
            bias_arrays.append(cmip6_data["annual"]["bias"])
        if "annual" in cmip6_individual_data:
            for member_data in cmip6_individual_data["annual"].values():
                bias_arrays.append(member_data["bias"])
        vmin, vmax = _percentile_range(field_arrays)
        ranges["annual"] = {
            "vmin": vmin, "vmax": vmax,
            "bias_vmax": _bias_max(bias_arrays),
        }

        # Seasonal
        for season in ["DJF", "MAM", "JJA", "SON"]:
            s_fields = [
                mr["seasonal_regrids"][season]
                for mr in model_results.values()
                if season in mr["seasonal_regrids"]
            ]
            s_biases = [
                mr["seasonal_biases"][season]
                for mr in model_results.values()
                if season in mr["seasonal_biases"]
            ]
            if not s_fields:
                continue
            if season in obs_seasonal_common:
                s_fields.append(obs_seasonal_common[season])
            if season in cmip6_data:
                s_fields.append(cmip6_data[season]["regrid"])
                s_biases.append(cmip6_data[season]["bias"])
            if season in cmip6_individual_data:
                for member_data in cmip6_individual_data[season].values():
                    s_biases.append(member_data["bias"])
            vmin, vmax = _percentile_range(s_fields)
            ranges[season] = {
                "vmin": vmin, "vmax": vmax,
                "bias_vmax": _bias_max(s_biases) if s_biases else 1.0,
            }

        return ranges

    # ── Statistics helpers ───────────────────────────────────────────

    @staticmethod
    def _pattern_correlation(model_field, obs_field, area):
        """Compute area-weighted spatial pattern correlation."""
        area_da = xr.DataArray(
            area, dims=("lat", "lon"),
            coords={"lat": model_field.lat, "lon": model_field.lon},
        )
        m = model_field.values.ravel()
        o = obs_field.values.ravel()
        w = np.asarray(area_da).ravel()

        valid = np.isfinite(m) & np.isfinite(o)
        m, o, w = m[valid], o[valid], w[valid]

        m_mean = np.average(m, weights=w)
        o_mean = np.average(o, weights=w)
        m_anom = m - m_mean
        o_anom = o - o_mean

        cov = np.average(m_anom * o_anom, weights=w)
        m_std = np.sqrt(np.average(m_anom ** 2, weights=w))
        o_std = np.sqrt(np.average(o_anom ** 2, weights=w))

        if m_std == 0 or o_std == 0:
            return 0.0
        return float(cov / (m_std * o_std))

    @staticmethod
    def _std_ratio(model_field, obs_field, area):
        """Compute ratio of model to obs spatial standard deviation."""
        area_flat = np.asarray(area).ravel()
        m = model_field.values.ravel()
        o = obs_field.values.ravel()

        valid = np.isfinite(m) & np.isfinite(o)
        m, o, w = m[valid], o[valid], area_flat[valid]

        m_std = np.sqrt(np.average((m - np.average(m, weights=w)) ** 2, weights=w))
        o_std = np.sqrt(np.average((o - np.average(o, weights=w)) ** 2, weights=w))

        if o_std == 0:
            return float("inf")
        return float(m_std / o_std)

    @staticmethod
    def _regional_mean_bias(bias_field, lats, area, lat_min, lat_max):
        """Compute area-weighted mean bias in a latitude band."""
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if not lat_mask.any():
            return 0.0
        bias_sub = bias_field.isel(lat=lat_mask)
        area_sub = area[lat_mask, :]
        area_da = xr.DataArray(
            area_sub, dims=("lat", "lon"),
            coords={"lat": bias_sub.lat, "lon": bias_sub.lon},
        )
        return float(bias_sub.weighted(area_da).mean().values)

    @staticmethod
    def _extratropical_mean_bias(bias_field, lats, area):
        """Compute area-weighted mean bias poleward of 30 degrees."""
        lat_mask = (lats < -30) | (lats > 30)
        if not lat_mask.any():
            return 0.0
        bias_sub = bias_field.isel(lat=lat_mask)
        area_sub = area[lat_mask, :]
        area_da = xr.DataArray(
            area_sub, dims=("lat", "lon"),
            coords={"lat": bias_sub.lat, "lon": bias_sub.lon},
        )
        return float(bias_sub.weighted(area_da).mean().values)


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    import pandas as pd

    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
