"""Global climatology bias maps diagnostic.

For each variable x period, produces a combined multi-panel figure with
obs climatology + bias maps for all models (DestinE + CMIP6).
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import xarray as xr

from feather.data.loader import DataLoader
from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_bias_map
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)


@register
class GlobalBiases(DiagnosticBase):
    """Climatology bias maps (model - obs).

    Produces combined multi-panel figures per variable per period
    (annual, DJF, JJA) showing obs climatology and bias maps for
    all configured models (DestinE + optionally CMIP6).
    """

    name = "global_biases"
    title = "Global Climatology Biases"
    domain = "sfc"
    variables = [
        # Temperature & pressure
        "avg_2t", "avg_msl",
        # Wind
        "avg_10u", "avg_10v",
        # Cloud cover
        "avg_tcc",
        # Precipitation
        "avg_tprate",
        # Surface heat fluxes
        "avg_ishf", "avg_slhtf",
        # Surface downwelling radiation
        "avg_sdswrf", "avg_sdlwrf",
        # Surface net radiation (all-sky + clear-sky)
        "avg_snswrf", "avg_snlwrf",
        "avg_snswrfcs", "avg_snlwrfcs",
        # TOA net radiation (all-sky + clear-sky)
        "avg_tnswrf", "avg_tnlwrf",
        "avg_tnswrfcs", "avg_tnlwrfcs",
    ]
    group = "evaluation"

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

    # -- Orchestration (per-variable incremental) ----------------------------

    def run(self, skip_existing: bool = True) -> list[tuple["Path", "Path"]]:
        """Execute per-variable: compute → plot → save immediately.

        Saves figures after each variable so that partial progress is
        preserved if a later variable crashes.

        Parameters
        ----------
        skip_existing : bool
            When True, skip variables whose output figures already
            exist on disk (all 3 period figures must be present).
        """
        from pathlib import Path

        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        for var in self.variables:
            # Check if all period figures already exist
            figure_ids = [
                f"{var}_{p}_bias_combined" for p in ["annual", "djf", "jja"]
            ]
            if skip_existing and all(
                self._figure_exists(fid) for fid in figure_ids
            ):
                logger.info(
                    "Skipping %s — all figures exist", var,
                )
                saved.extend([
                    (self.output_dir / f"{fid}.png",
                     self.output_dir / f"{fid}.json")
                    for fid in figure_ids
                ])
                continue

            var_result = self._compute_variable(var)
            if var_result is None:
                continue

            figures = self._plot_variable(var, var_result)
            for fig, meta in figures:
                paths = self._save(fig, meta, meta["figure_id"])
                saved.append(paths)

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # -- Computation --------------------------------------------------------

    def compute(self) -> dict[str, Any]:
        """Compute annual and seasonal climatologies + bias statistics.

        Returns
        -------
        dict
            Keyed by variable name.  Each entry contains model
            climatologies (on HEALPix), obs climatology (regular grid),
            regridded bias fields, and summary statistics.
        """
        results: dict[str, Any] = {}
        for var in self.variables:
            var_result = self._compute_variable(var)
            if var_result is not None:
                results[var] = var_result
        return results

    def _compute_variable(self, var: str) -> dict[str, Any] | None:
        """Compute climatologies + biases for a single variable.

        Returns None if no models have the variable.
        """
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_results: dict[str, dict] = {}

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0
        )

        # Load observation (usually small regular grid)
        logger.info("  Loading observations for %s", var)
        obs_data = self.obs_loader.load_for_model_var(var, self.period)
        obs_clim = climatology(obs_data)
        obs_gmean = float(latlon_global_mean(obs_clim).values)
        obs_seasonal = seasonal_climatology(obs_data)

        lat_name = "lat" if "lat" in obs_clim.coords else "latitude"
        lon_name = "lon" if "lon" in obs_clim.coords else "longitude"
        obs_lats = obs_clim[lat_name].values
        obs_lons = obs_clim[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Build nereus interpolator once (all models share the same
        # HEALPix grid, so the KDTree is built only once).
        interpolator = None
        obs_clim_common = None
        obs_seasonal_common = {}
        # Pre-computed area weights for the common grid (set once)
        common_area = None
        target_lats = None
        target_lons = None

        for model in self.config.models:
            logger.info("Computing biases for %s / %s ...", var, model)

            key = DataLoader.make_key(
                self.experiment, model, var_info.domain,
            )
            try:
                model_data = self.model_loader.load_var(key, var)
            except KeyError:
                logger.warning(
                    "  Variable %s not available for %s — skipping",
                    var, model,
                )
                continue
            ds = self.model_loader.load(key)
            lon = ds["longitude"]
            lat = ds["latitude"]

            # Compute climatologies with dask, then materialise
            model_clim = climatology(model_data, self.period).compute()
            model_gmean = float(model_clim.mean().values)

            model_seasonal = seasonal_climatology(model_data, self.period)
            model_seasonal = {
                s: model_seasonal[s].compute()
                for s in model_seasonal.data_vars
            }

            # Build interpolator once (reused across models + seasons)
            if interpolator is None:
                logger.info("  Building nereus interpolator (first model)...")
                annual_regrid, interpolator = nr.regrid(
                    model_clim.values.ravel(),
                    lon=np.asarray(lon), lat=np.asarray(lat),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                target_lats = interpolator.target_lat[:, 0]
                target_lons = interpolator.target_lon[0, :]

                # Regrid obs to common nereus grid (once)
                obs_clim_common = obs_clim.interp(
                    {lat_name: target_lats, lon_name: target_lons}
                )
                if lat_name != "lat":
                    obs_clim_common = obs_clim_common.rename(
                        {lat_name: "lat", lon_name: "lon"}
                    )

                # Pre-compute area weights for the common grid (once)
                from feather.util.spatial import compute_latlon_areas
                common_area = compute_latlon_areas(
                    target_lats, target_lons,
                )

                # Also regrid seasonal obs
                obs_seasonal_common = {}
                for season in obs_seasonal:
                    obs_s = obs_seasonal[season].interp(
                        {lat_name: target_lats, lon_name: target_lons}
                    )
                    if lat_name != "lat":
                        obs_s = obs_s.rename(
                            {lat_name: "lat", lon_name: "lon"}
                        )
                    obs_seasonal_common[season] = obs_s
            else:
                regridded_np = interpolator(model_clim.values.ravel())
                annual_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            # --- Annual bias ---
            annual_bias = annual_regrid - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(annual_bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(annual_bias ** 2, area=common_area).values
            ))

            # --- Seasonal biases (DJF, JJA) ---
            seasonal_biases: dict[str, Any] = {}
            seasonal_regrids: dict[str, Any] = {}
            for season in ["DJF", "JJA"]:
                if season in model_seasonal:
                    s_np = interpolator(
                        model_seasonal[season].values.ravel()
                    )
                    s_regrid = xr.DataArray(
                        s_np, dims=("lat", "lon"),
                        coords={
                            "lat": target_lats, "lon": target_lons,
                        },
                    )
                    seasonal_regrids[season] = s_regrid
                    if season in obs_seasonal_common:
                        s_bias = s_regrid - obs_seasonal_common[season]
                        seasonal_biases[season] = s_bias

            model_results[model] = {
                "annual_regrid": annual_regrid,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": model_gmean,
                "annual_bias": annual_bias,
                "annual_bias_gmean": bias_gmean,
                "annual_rmse": rmse,
                "seasonal_biases": seasonal_biases,
            }

        if not model_results:
            logger.warning(
                "  No models have variable %s — skipping", var,
            )
            return None

        # CMIP6 bias (optional)
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        if self.cmip6_enabled and interpolator is not None:
            if self.cmip6_individual:
                # Individual CMIP6 models + MMM
                cmip6_individual_data = self._compute_cmip6_individual(
                    var, target_lats, target_lons,
                    obs_clim_common, obs_seasonal_common,
                    common_area,
                )
                cmip6_data, cmip6_info = self._compute_cmip6_mmm(
                    var, target_lats, target_lons,
                    obs_clim_common, obs_seasonal_common,
                    common_area,
                )
            else:
                # MMM mode (default)
                cmip6_data, cmip6_info = self._compute_cmip6_mmm(
                    var, target_lats, target_lons,
                    obs_clim_common, obs_seasonal_common,
                    common_area,
                )

        # Compute shared colorbar ranges across all models per period
        logger.info("  Computing shared colorbar ranges")
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

    # -- CMIP6 computation helpers ------------------------------------------

    def _compute_cmip6_mmm(self, var, target_lats, target_lons,
                           obs_clim_common, obs_seasonal_common,
                           common_area):
        """Compute CMIP6 multi-model mean biases."""
        cmip6_data = {}
        cmip6_info = {}

        logger.info("  Loading CMIP6 multi-model mean for %s...", var)
        mmm, info = self.cmip6_loader.load_mmm_for_model_var(
            var, period=self.period,
        )
        if mmm is not None:
            cmip6_common = mmm.interp(
                lat=target_lats, lon=target_lons,
            )
            cmip6_bias = cmip6_common - obs_clim_common
            cmip6_bias_gmean = float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            )
            cmip6_rmse = float(np.sqrt(
                latlon_global_mean(
                    cmip6_bias ** 2, area=common_area,
                ).values
            ))

            cmip6_data["annual"] = {
                "regrid": cmip6_common,
                "bias": cmip6_bias,
                "bias_gmean": cmip6_bias_gmean,
                "rmse": cmip6_rmse,
            }
            cmip6_info = info

            # Seasonal CMIP6 MMM biases
            logger.info("  Loading CMIP6 seasonal MMM (DJF, JJA)...")
            for season in ["DJF", "JJA"]:
                mmm_s, _ = self.cmip6_loader.load_mmm_for_model_var(
                    var, period=self.period, season=season,
                )
                if mmm_s is not None and season in obs_seasonal_common:
                    cmip6_s = mmm_s.interp(
                        lat=target_lats, lon=target_lons,
                    )
                    cmip6_s_bias = cmip6_s - obs_seasonal_common[season]
                    cmip6_data[season] = {
                        "regrid": cmip6_s,
                        "bias": cmip6_s_bias,
                        "bias_gmean": float(
                            latlon_global_mean(
                                cmip6_s_bias, area=common_area,
                            ).values
                        ),
                    }

        return cmip6_data, cmip6_info

    def _compute_cmip6_individual(self, var, target_lats, target_lons,
                                  obs_clim_common, obs_seasonal_common,
                                  common_area):
        """Compute individual CMIP6 model biases."""
        cmip6_individual_data: dict[str, dict] = {}

        logger.info("  Loading individual CMIP6 models for %s...", var)
        member_pairs = self.cmip6_loader.get_member_pairs()

        for model, variant in member_pairs:
            label = f"{model}/{variant}"

            # Annual
            da = self.cmip6_loader.load_var_for_model_var(
                var, model, variant=variant, period=self.period,
            )
            if da is None:
                logger.debug("  Skipping %s — no data", label)
                continue

            cmip6_common = da.interp(lat=target_lats, lon=target_lons)
            cmip6_bias = cmip6_common - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(
                    cmip6_bias ** 2, area=common_area,
                ).values
            ))

            cmip6_individual_data.setdefault("annual", {})[label] = {
                "regrid": cmip6_common,
                "bias": cmip6_bias,
                "bias_gmean": bias_gmean,
                "rmse": rmse,
            }

            # Seasonal
            for season in ["DJF", "JJA"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    var, model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is None or season not in obs_seasonal_common:
                    continue
                cmip6_s = da_s.interp(lat=target_lats, lon=target_lons)
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

    # -- Colorbar range computation -----------------------------------------

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_clim_common: xr.DataArray,
        obs_seasonal_common: dict[str, xr.DataArray],
        cmip6_data: dict[str, dict] | None = None,
        cmip6_individual_data: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """Compute shared colorbar ranges across all models per period.

        Parameters
        ----------
        cmip6_data : dict, optional
            CMIP6 MMM data (when using MMM mode).
        cmip6_individual_data : dict, optional
            Individual CMIP6 model data (when using individual mode).

        Returns a dict keyed by period name ("annual", "DJF", "JJA")
        with ``vmin``, ``vmax`` (field panels) and ``bias_vmax``
        (symmetric bias panel) values.
        """
        ranges: dict[str, dict] = {}
        cmip6_data = cmip6_data or {}
        cmip6_individual_data = cmip6_individual_data or {}

        def _finite_vals(arrays):
            """Extract all finite values from a list of arrays."""
            parts = []
            for a in arrays:
                v = np.asarray(a).ravel()
                parts.append(v[np.isfinite(v)])
            return np.concatenate(parts)

        def _percentile_range(arrays):
            """Compute vmin/vmax from 2nd/98th percentile of arrays."""
            vals = _finite_vals(arrays)
            return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

        def _bias_max(arrays):
            """Compute symmetric bias range from 98th percentile of |bias|."""
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
        for season in ["DJF", "JJA"]:
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

    # -- Plotting -----------------------------------------------------------

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate combined multi-panel bias maps per variable per period.

        For each variable and period (annual, DJF, JJA), produces ONE
        figure with obs climatology + bias panels for all models.

        Returns
        -------
        list of (Figure, metadata-dict)
        """
        figures: list[tuple[plt.Figure, dict]] = []
        for var, vr in results.items():
            figures.extend(self._plot_variable(var, vr))
        return figures

    def _plot_variable(
        self, var: str, vr: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Generate combined multi-panel bias maps for a single variable."""
        figures: list[tuple[plt.Figure, dict]] = []

        var_info = vr["var_info"]
        obs_clim = vr["obs"]["clim"]
        cb = vr["colorbar_ranges"]
        cmip6_data = vr.get("cmip6_data", {})
        cmip6_info = vr.get("cmip6_info", {})
        cmip6_individual_data = vr.get("cmip6_individual_data", {})

        periods = [("annual", "Annual Mean")]
        for season in ["DJF", "JJA"]:
            if season in cb:
                periods.append((season, season))

        for period_key, period_label in periods:
            # Build ordered bias dict: DestinE models first, then CMIP6
            bias_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in vr["models"].items():
                if period_key == "annual":
                    bias_field = mdata["annual_bias"]
                    summary_stats[model] = {
                        "global_mean_bias": mdata["annual_bias_gmean"],
                        "rmse": mdata["annual_rmse"],
                    }
                else:
                    bias_field = mdata["seasonal_biases"].get(period_key)
                    if bias_field is None:
                        continue
                bias_dict[model] = bias_field
                all_models.append(model)

            # Add CMIP6 MMM if available
            if period_key in cmip6_data:
                c_data = cmip6_data[period_key]
                bias_dict["CMIP6 MMM"] = c_data["bias"]
                all_models.append("CMIP6 MMM")
                summary_stats["CMIP6 MMM"] = {
                    "global_mean_bias": c_data["bias_gmean"],
                    "rmse": c_data.get("rmse"),
                }

            # Add individual CMIP6 models if available
            if period_key in cmip6_individual_data:
                for label, c_data in cmip6_individual_data[period_key].items():
                    bias_dict[label] = c_data["bias"]
                    all_models.append(label)
                    summary_stats[label] = {
                        "global_mean_bias": c_data["bias_gmean"],
                        "rmse": c_data.get("rmse"),
                    }

            if not bias_dict:
                continue

            # Get obs data for this period
            if period_key == "annual":
                obs_period = obs_clim
            else:
                obs_period = vr["obs"]["seasonal_clim"].get(period_key)
                if obs_period is None:
                    continue

            # Get colorbar ranges
            p_cb = cb.get(period_key, cb.get("annual", {}))

            fig, axes = plot_combined_bias_map(
                obs_period, bias_dict,
                title=f"{var_info.long_name} {period_label}",
                cmap=var_info.cmap,
                bias_cmap="RdBu_r",
                vmin=p_cb.get("vmin"),
                vmax=p_cb.get("vmax"),
                bias_vmax=p_cb.get("bias_vmax"),
                units=var_info.units,
            )

            meta = self._build_metadata(
                title=(
                    f"{var_info.long_name} {period_label} Bias"
                ),
                figure_id=f"{var}_{period_key.lower()}_bias_combined",
                models=all_models,
                variables=[var],
                description=(
                    f"{period_label} climatology bias maps for "
                    f"{var_info.long_name} — all models combined."
                ),
                plot_type="combined_bias_map",
                period=self.period,
                cmip6_info=cmip6_info or None,
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures
