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
        "tas", "psl",
        # Wind
        "uas", "vas",
        # Cloud cover
        "clt",
        # Precipitation
        "pr",
        # Surface heat fluxes
        "hfss", "hfls",
        # Surface downwelling radiation
        "rsds", "rlds",
        # Surface net radiation (all-sky + clear-sky)
        "rss", "rls",
        "rsscs", "rlscs",
        # TOA net radiation (all-sky + clear-sky)
        "rst", "rlt",
        "rstcs", "rltcs",
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

            try:
                var_result = self._compute_variable(var)
                if var_result is None:
                    continue

                figures = self._plot_variable(var, var_result)
                for fig, meta in figures:
                    paths = self._save(fig, meta, meta["figure_id"])
                    saved.append(paths)
            except Exception:
                logger.warning(
                    "Variable %s failed — skipping", var, exc_info=True,
                )

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
        obs_data = self._load_obs_var(var, self.period)

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

            try:
                model_data = self._load_model_var(
                    model, var, period=self.period,
                )
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "  Variable %s not available for %s — skipping",
                    var, model,
                )
                continue
            lon, lat = self._load_model_coords(model, var)

            # For latlon grids, meshgrid 1D coord arrays to per-pixel arrays
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

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

        # CMIP6 trends (optional)
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        if self.cmip6_enabled and interpolator is not None:
            if self.cmip6_individual:
                cmip6_individual_data = self._compute_cmip6_individual_trends(
                    var, target_lats, target_lons,
                    obs_trend_common, obs_seasonal_trends_common,
                    common_area,
                )
                cmip6_data, cmip6_info = self._compute_cmip6_mmm_trends(
                    var, target_lats, target_lons,
                    obs_trend_common, obs_seasonal_trends_common,
                    common_area,
                )
            else:
                cmip6_data, cmip6_info = self._compute_cmip6_mmm_trends(
                    var, target_lats, target_lons,
                    obs_trend_common, obs_seasonal_trends_common,
                    common_area,
                )

        # Compute shared colorbar ranges across all models per period
        logger.info("  Computing shared colorbar ranges")
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_trend_common, obs_seasonal_trends_common,
            cmip6_data=cmip6_data,
            cmip6_individual_data=cmip6_individual_data,
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
            "cmip6_data": cmip6_data,
            "cmip6_info": cmip6_info,
            "cmip6_individual_data": cmip6_individual_data,
        }

    # -- CMIP6 computation helpers ------------------------------------------

    def _compute_cmip6_mmm_trends(
        self, var, target_lats, target_lons,
        obs_trend_common, obs_seasonal_trends_common,
        common_area,
    ):
        """Compute CMIP6 multi-model mean trends.

        Loads per-model time series (time_mean=False), computes
        annual_mean → linear_trend → regrid for each, then averages
        across models to get the MMM trend.

        Returns (cmip6_data, cmip6_info).
        """
        cmip6_data = {}
        cmip6_info = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Computing CMIP6 MMM trends for %s...", var)
        member_pairs = self.cmip6_loader.get_member_pairs()

        # -- Annual trends per model --
        annual_trends = []
        seasonal_trends: dict[str, list] = {"DJF": [], "JJA": []}
        models_used = []

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                var, model, variant=variant,
                period=self.period, time_mean=False,
            )
            if da is None:
                logger.debug("  Skipping %s — no data", label)
                continue
            if "time" not in da.dims or da.sizes["time"] < 2:
                logger.debug("  Skipping %s — insufficient time steps", label)
                continue

            # Annual mean → materialise → linear trend
            da_annual = annual_mean(da)
            if hasattr(da_annual, "compute"):
                da_annual = da_annual.compute()
            model_trend = linear_trend(da_annual) * 10  # per decade

            regridded = self._regrid_to_target(
                model_trend, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
            )
            annual_trends.append(regridded)
            models_used.append(label)

            # Seasonal trends
            for season in ["DJF", "JJA"]:
                da_s = seasonal_annual_mean(da, season)
                if hasattr(da_s, "compute"):
                    da_s = da_s.compute()
                if len(da_s.year) < 2:
                    continue
                s_trend = linear_trend(da_s, dim="year") * 10
                s_regridded = self._regrid_to_target(
                    s_trend, target_lats, target_lons,
                    resolution, influence_radius, cmip6_interp_cache,
                )
                seasonal_trends[season].append(s_regridded)

        if not annual_trends:
            logger.info("  No CMIP6 models available for %s", var)
            return cmip6_data, cmip6_info

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        # MMM annual trend
        stacked = xr.concat(annual_trends, dim="member")
        mmm_trend = stacked.mean("member")
        trend_diff = mmm_trend - obs_trend_common
        trend_diff_gmean = float(
            latlon_global_mean(trend_diff, area=common_area).values,
        )
        rmse = float(np.sqrt(
            latlon_global_mean(trend_diff ** 2, area=common_area).values,
        ))
        cmip6_data["annual"] = {
            "regrid": mmm_trend,
            "trend_diff": trend_diff,
            "trend_diff_gmean": trend_diff_gmean,
            "rmse": rmse,
        }

        # MMM seasonal trends
        for season in ["DJF", "JJA"]:
            if not seasonal_trends[season]:
                continue
            s_stacked = xr.concat(seasonal_trends[season], dim="member")
            s_mmm = s_stacked.mean("member")
            if season in obs_seasonal_trends_common:
                s_diff = s_mmm - obs_seasonal_trends_common[season]
                cmip6_data[season] = {
                    "regrid": s_mmm,
                    "trend_diff": s_diff,
                    "trend_diff_gmean": float(
                        latlon_global_mean(
                            s_diff, area=common_area,
                        ).values,
                    ),
                }

        return cmip6_data, cmip6_info

    def _compute_cmip6_individual_trends(
        self, var, target_lats, target_lons,
        obs_trend_common, obs_seasonal_trends_common,
        common_area,
    ):
        """Compute per-model CMIP6 trend differences.

        Same as MMM but keeps per-model trends separate instead of
        averaging.

        Returns cmip6_individual_data dict structured as
        {period: {label: {regrid, trend_diff, trend_diff_gmean, rmse}}}.
        """
        cmip6_individual_data: dict[str, dict] = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info(
            "  Computing individual CMIP6 trends for %s...", var,
        )
        member_pairs = self.cmip6_loader.get_member_pairs()

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                var, model, variant=variant,
                period=self.period, time_mean=False,
            )
            if da is None:
                logger.debug("  Skipping %s — no data", label)
                continue
            if "time" not in da.dims or da.sizes["time"] < 2:
                logger.debug("  Skipping %s — insufficient time steps", label)
                continue

            # Annual
            da_annual = annual_mean(da)
            if hasattr(da_annual, "compute"):
                da_annual = da_annual.compute()
            model_trend = linear_trend(da_annual) * 10
            regridded = self._regrid_to_target(
                model_trend, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
            )
            trend_diff = regridded - obs_trend_common
            rmse = float(np.sqrt(
                latlon_global_mean(
                    trend_diff ** 2, area=common_area,
                ).values,
            ))
            cmip6_individual_data.setdefault("annual", {})[label] = {
                "regrid": regridded,
                "trend_diff": trend_diff,
                "trend_diff_gmean": float(
                    latlon_global_mean(
                        trend_diff, area=common_area,
                    ).values,
                ),
                "rmse": rmse,
            }

            # Seasonal
            for season in ["DJF", "JJA"]:
                da_s = seasonal_annual_mean(da, season)
                if hasattr(da_s, "compute"):
                    da_s = da_s.compute()
                if len(da_s.year) < 2:
                    continue
                s_trend = linear_trend(da_s, dim="year") * 10
                s_regridded = self._regrid_to_target(
                    s_trend, target_lats, target_lons,
                    resolution, influence_radius, cmip6_interp_cache,
                )
                if season not in obs_seasonal_trends_common:
                    continue
                s_diff = s_regridded - obs_seasonal_trends_common[season]
                cmip6_individual_data.setdefault(season, {})[label] = {
                    "regrid": s_regridded,
                    "trend_diff": s_diff,
                    "trend_diff_gmean": float(
                        latlon_global_mean(
                            s_diff, area=common_area,
                        ).values,
                    ),
                }

        return cmip6_individual_data

    # -- Regridding helper --------------------------------------------------

    @staticmethod
    def _regrid_to_target(da, target_lats, target_lons,
                          resolution, influence_radius,
                          interp_cache):
        """Regrid a regular lat/lon DataArray to the target grid via nereus NN.

        Uses *interp_cache* (keyed by grid shape) to avoid rebuilding
        the KDTree for models that share the same native grid.

        For coarse-resolution source grids (e.g. CMIP6 at 1-2°) the
        configured *influence_radius* (tuned for 5 km HEALPix) is too
        small.  We use 250 km as the floor.
        """
        # 250 km floor — covers CMIP6 grids up to ~2° at the equator
        ir = max(influence_radius, 250_000.0)

        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lat_arr = da[lat_name].values
        lon_arr = da[lon_name].values

        grid_key = (len(lat_arr), len(lon_arr))

        if grid_key not in interp_cache:
            lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
            _, interp_cache[grid_key] = nr.regrid(
                da.values.ravel(),
                lon=lon_2d.ravel(), lat=lat_2d.ravel(),
                resolution=resolution,
                influence_radius=ir,
                lon_bounds=(0.0, 360.0),
                as_xarray=True,
            )

        regridded = interp_cache[grid_key](da.values.ravel())
        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    # -- Colorbar range computation -----------------------------------------

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_trend_common: xr.DataArray,
        obs_seasonal_trends_common: dict[str, xr.DataArray],
        cmip6_data: dict[str, dict] | None = None,
        cmip6_individual_data: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """Compute shared colorbar ranges across all models per period.

        Parameters
        ----------
        cmip6_data : dict, optional
            CMIP6 MMM trend data.
        cmip6_individual_data : dict, optional
            Individual CMIP6 model trend data.

        Returns a dict keyed by period name ("annual", "DJF", "JJA")
        with ``vmin``, ``vmax`` (field panels) and ``bias_vmax``
        (symmetric trend difference panel) values.
        """
        ranges: dict[str, dict] = {}
        cmip6_data = cmip6_data or {}
        cmip6_individual_data = cmip6_individual_data or {}

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
        if "annual" in cmip6_data:
            field_arrays.append(cmip6_data["annual"]["regrid"])
            diff_arrays.append(cmip6_data["annual"]["trend_diff"])
        if "annual" in cmip6_individual_data:
            for member_data in cmip6_individual_data["annual"].values():
                diff_arrays.append(member_data["trend_diff"])
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
            if season in cmip6_data:
                s_fields.append(cmip6_data[season]["regrid"])
                s_diffs.append(cmip6_data[season]["trend_diff"])
            if season in cmip6_individual_data:
                for member_data in cmip6_individual_data[season].values():
                    s_diffs.append(member_data["trend_diff"])
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
        cmip6_data = vr.get("cmip6_data", {})
        cmip6_info = vr.get("cmip6_info", {})
        cmip6_individual_data = vr.get("cmip6_individual_data", {})

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

            # Add CMIP6 MMM if available
            if period_key in cmip6_data:
                c_data = cmip6_data[period_key]
                trend_diff_dict["CMIP6 MMM"] = c_data["trend_diff"]
                all_models.append("CMIP6 MMM")
                summary_stats["CMIP6 MMM"] = {
                    "global_mean_trend_diff": c_data["trend_diff_gmean"],
                    "trend_rmse": c_data.get("rmse"),
                }

            # Add individual CMIP6 models if available
            if period_key in cmip6_individual_data:
                for label, c_data in (
                    cmip6_individual_data[period_key].items()
                ):
                    trend_diff_dict[label] = c_data["trend_diff"]
                    all_models.append(label)
                    summary_stats[label] = {
                        "global_mean_trend_diff": c_data[
                            "trend_diff_gmean"
                        ],
                        "trend_rmse": c_data.get("rmse"),
                    }

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
                cmip6_info=cmip6_info or None,
                summary_statistics=summary_stats,
                extra={"units": trend_units},
            )
            figures.append((fig, meta))

        return figures
