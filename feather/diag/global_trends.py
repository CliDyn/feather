"""Global linear trends diagnostic.

For each variable x period, produces a combined multi-panel figure with
obs trend map + trend difference panels (model trend - obs trend) for
all models. Units are per decade (e.g., K/decade).
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
from feather.util.temporal import annual_mean, linear_trend, seasonal_annual_mean

logger = logging.getLogger(__name__)


@register
class GlobalTrends(DiagnosticBase):
    """Linear trend maps (model trend - obs trend).

    Produces combined multi-panel figures per variable per period
    (annual, DJF, JJA) showing obs trend and trend difference maps
    for all configured models (DestinE).
    """

    name = "global_trends"
    title = "Global Linear Trends"
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
        """Execute per-variable: compute -> plot -> save immediately.

        Saves figures after each variable so that partial progress is
        preserved if a later variable crashes.
        """
        from pathlib import Path

        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        for var in self.variables:
            figure_ids = [
                f"{var}_{p}_trend_combined" for p in ["annual", "djf", "jja"]
            ]
            if skip_existing and all(
                self._figure_exists(fid) for fid in figure_ids
            ):
                logger.info("Skipping %s — all figures exist", var)
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
        """Compute annual and seasonal trends + trend difference statistics.

        Returns
        -------
        dict
            Keyed by variable name.
        """
        results: dict[str, Any] = {}
        for var in self.variables:
            var_result = self._compute_variable(var)
            if var_result is not None:
                results[var] = var_result
        return results

    def _compute_variable(self, var: str) -> dict[str, Any] | None:
        """Compute trends + trend differences for a single variable.

        Returns None if no models have the variable.
        """
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_results: dict[str, dict] = {}

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )

        # Load observation (usually small regular grid)
        logger.info("  Loading observations for %s", var)
        obs_data = self.obs_loader.load_for_model_var(var, self.period)

        # Compute obs annual means and trend
        obs_annual = annual_mean(obs_data)
        obs_annual_trend = linear_trend(obs_annual) * 10  # per decade

        # Compute obs seasonal annual means and trends
        obs_seasonal_trends = {}
        for season in ["DJF", "JJA"]:
            obs_season_annual = seasonal_annual_mean(obs_data, season)
            if len(obs_season_annual.year) >= 2:
                obs_seasonal_trends[season] = (
                    linear_trend(obs_season_annual, dim="year") * 10
                )

        lat_name = "lat" if "lat" in obs_annual_trend.coords else "latitude"
        lon_name = "lon" if "lon" in obs_annual_trend.coords else "longitude"
        obs_lats = obs_annual_trend[lat_name].values
        obs_lons = obs_annual_trend[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Build nereus interpolator once
        interpolator = None
        obs_trend_common = None
        obs_seasonal_trends_common = {}
        common_area = None
        target_lats = None
        target_lons = None

        for model in self.config.models:
            logger.info("Computing trends for %s / %s ...", var, model)

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

            # Time-slice to period
            model_data = model_data.sel(
                time=slice(self.period[0], self.period[1]),
            )

            # Compute annual means, materialise dask, then trend
            model_annual = annual_mean(model_data).compute()
            model_annual_trend = linear_trend(model_annual) * 10  # per decade

            # Build interpolator once (reused across models + seasons)
            if interpolator is None:
                logger.info("  Building nereus interpolator (first model)...")
                annual_regrid, interpolator = nr.regrid(
                    model_annual_trend.values.ravel(),
                    lon=np.asarray(lon), lat=np.asarray(lat),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                target_lats = interpolator.target_lat[:, 0]
                target_lons = interpolator.target_lon[0, :]

                # Regrid obs trend to common nereus grid (once)
                obs_lons_2d, obs_lats_2d = np.meshgrid(
                    obs_lons, obs_lats,
                )
                _, obs_interpolator = nr.regrid(
                    obs_annual_trend.values.ravel(),
                    lon=obs_lons_2d.ravel(),
                    lat=obs_lats_2d.ravel(),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                obs_trend_common = xr.DataArray(
                    obs_interpolator(obs_annual_trend.values.ravel()),
                    dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

                # Pre-compute area weights for the common grid (once)
                from feather.util.spatial import compute_latlon_areas
                common_area = compute_latlon_areas(
                    target_lats, target_lons,
                )

                # Also regrid seasonal obs trends (reuse obs interpolator)
                for season, obs_s_trend in obs_seasonal_trends.items():
                    s_np = obs_interpolator(obs_s_trend.values.ravel())
                    obs_seasonal_trends_common[season] = xr.DataArray(
                        s_np, dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
            else:
                regridded_np = interpolator(
                    model_annual_trend.values.ravel(),
                )
                annual_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            # Annual trend difference
            annual_trend_diff = annual_regrid - obs_trend_common
            trend_gmean = float(
                latlon_global_mean(annual_regrid, area=common_area).values,
            )
            trend_diff_gmean = float(
                latlon_global_mean(
                    annual_trend_diff, area=common_area,
                ).values,
            )
            rmse = float(np.sqrt(
                latlon_global_mean(
                    annual_trend_diff ** 2, area=common_area,
                ).values,
            ))

            # Seasonal trends
            seasonal_regrids: dict[str, Any] = {}
            seasonal_trend_diffs: dict[str, Any] = {}
            for season in ["DJF", "JJA"]:
                model_season_annual = seasonal_annual_mean(
                    model_data, season,
                )
                if hasattr(model_season_annual, "compute"):
                    model_season_annual = model_season_annual.compute()
                if len(model_season_annual.year) < 2:
                    continue
                model_s_trend = linear_trend(
                    model_season_annual, dim="year",
                ) * 10
                s_np = interpolator(model_s_trend.values.ravel())
                s_regrid = xr.DataArray(
                    s_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )
                seasonal_regrids[season] = s_regrid
                if season in obs_seasonal_trends_common:
                    seasonal_trend_diffs[season] = (
                        s_regrid - obs_seasonal_trends_common[season]
                    )

            model_results[model] = {
                "annual_regrid": annual_regrid,
                "seasonal_regrids": seasonal_regrids,
                "global_mean_trend": trend_gmean,
                "annual_trend_diff": annual_trend_diff,
                "annual_trend_diff_gmean": trend_diff_gmean,
                "annual_rmse": rmse,
                "seasonal_trend_diffs": seasonal_trend_diffs,
            }

        if not model_results:
            logger.warning(
                "  No models have variable %s — skipping", var,
            )
            return None

        # Compute shared colorbar ranges across all models per period
        logger.info("  Computing shared colorbar ranges")
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_trend_common, obs_seasonal_trends_common,
        )

        return {
            "models": model_results,
            "obs": {
                "trend": obs_trend_common,
                "seasonal_trends": obs_seasonal_trends_common,
                "global_mean_trend": float(
                    latlon_global_mean(
                        obs_trend_common, area=common_area,
                    ).values,
                ),
            },
            "var_info": var_info,
            "colorbar_ranges": colorbar_ranges,
        }

    # -- Colorbar range computation -----------------------------------------

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_trend_common: xr.DataArray,
        obs_seasonal_trends_common: dict[str, xr.DataArray],
    ) -> dict[str, dict]:
        """Compute shared colorbar ranges across all models per period.

        Returns a dict keyed by period name ("annual", "DJF", "JJA")
        with ``vmin``, ``vmax`` (field panels) and ``bias_vmax``
        (symmetric trend difference panel) values.
        """
        ranges: dict[str, dict] = {}

        def _finite_vals(arrays):
            parts = []
            for a in arrays:
                v = np.asarray(a).ravel()
                parts.append(v[np.isfinite(v)])
            return np.concatenate(parts)

        def _symmetric_range(arrays):
            vals = _finite_vals(arrays)
            if len(vals) == 0:
                return -1.0, 1.0
            absmax = float(np.percentile(np.abs(vals), 98)) or 1.0
            return -absmax, absmax

        def _bias_max(arrays):
            vals = _finite_vals(arrays)
            if len(vals) == 0:
                return 1.0
            return float(np.percentile(np.abs(vals), 98)) or 1.0

        # Annual
        field_arrays = [mr["annual_regrid"] for mr in model_results.values()]
        field_arrays.append(obs_trend_common)
        diff_arrays = [mr["annual_trend_diff"] for mr in model_results.values()]
        vmin, vmax = _symmetric_range(field_arrays)
        ranges["annual"] = {
            "vmin": vmin, "vmax": vmax,
            "bias_vmax": _bias_max(diff_arrays),
        }

        # Seasonal
        for season in ["DJF", "JJA"]:
            s_fields = [
                mr["seasonal_regrids"][season]
                for mr in model_results.values()
                if season in mr["seasonal_regrids"]
            ]
            s_diffs = [
                mr["seasonal_trend_diffs"][season]
                for mr in model_results.values()
                if season in mr["seasonal_trend_diffs"]
            ]
            if not s_fields:
                continue
            if season in obs_seasonal_trends_common:
                s_fields.append(obs_seasonal_trends_common[season])
            vmin, vmax = _symmetric_range(s_fields)
            ranges[season] = {
                "vmin": vmin, "vmax": vmax,
                "bias_vmax": _bias_max(s_diffs) if s_diffs else 1.0,
            }

        return ranges

    # -- Plotting -----------------------------------------------------------

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate combined multi-panel trend maps per variable per period.

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
        """Generate combined multi-panel trend maps for a single variable."""
        figures: list[tuple[plt.Figure, dict]] = []

        var_info = vr["var_info"]
        obs_trend = vr["obs"]["trend"]
        cb = vr["colorbar_ranges"]

        periods = [("annual", "Annual")]
        for season in ["DJF", "JJA"]:
            if season in cb:
                periods.append((season, season))

        trend_units = f"{var_info.units}/decade"

        for period_key, period_label in periods:
            # Build ordered trend difference dict
            trend_diff_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in vr["models"].items():
                if period_key == "annual":
                    diff_field = mdata["annual_trend_diff"]
                    summary_stats[model] = {
                        "global_mean_trend": mdata["global_mean_trend"],
                        "global_mean_trend_diff": mdata[
                            "annual_trend_diff_gmean"
                        ],
                        "trend_rmse": mdata["annual_rmse"],
                    }
                else:
                    diff_field = mdata["seasonal_trend_diffs"].get(period_key)
                    if diff_field is None:
                        continue
                trend_diff_dict[model] = diff_field
                all_models.append(model)

            if not trend_diff_dict:
                continue

            # Get obs trend for this period
            if period_key == "annual":
                obs_period = obs_trend
            else:
                obs_period = vr["obs"]["seasonal_trends"].get(period_key)
                if obs_period is None:
                    continue

            # Get colorbar ranges
            p_cb = cb.get(period_key, cb.get("annual", {}))

            fig, axes = plot_combined_bias_map(
                obs_period, trend_diff_dict,
                title=f"{var_info.long_name} {period_label} Trend",
                obs_title="ERA5 Trend",
                cmap="RdBu_r",
                bias_cmap="RdBu_r",
                vmin=p_cb.get("vmin"),
                vmax=p_cb.get("vmax"),
                bias_vmax=p_cb.get("bias_vmax"),
                units=trend_units,
            )

            meta = self._build_metadata(
                title=(
                    f"{var_info.long_name} {period_label} Linear Trend"
                ),
                figure_id=f"{var}_{period_key.lower()}_trend_combined",
                models=all_models,
                variables=[var],
                description=(
                    f"{period_label} linear trend maps ({trend_units}) for "
                    f"{var_info.long_name} over {self.period[0]}-"
                    f"{self.period[1]} — obs trend + model-obs trend "
                    f"difference panels."
                ),
                plot_type="combined_trend_map",
                period=self.period,
                summary_statistics=summary_stats,
                extra={"units": trend_units},
            )
            figures.append((fig, meta))

        return figures
