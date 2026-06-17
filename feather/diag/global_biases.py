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

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.util.spatial import (
    latlon_global_mean,
    spatial_ttest,
    spatial_variance_ratio,
)
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)

# Precipitation display constants (pr only)
_PR_TO_MMDAY = 86400.0          # kg/m²/s → mm/day (display only)
_REL_BIAS_THRESHOLD = 0.1 / 86400  # mask relative bias where obs < 0.1 mm/day


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
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks)
        self.save_netcdf = save_netcdf
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")
        self._project_name = self.config.project.get("name", "Ensemble")

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
                f"{var}_{p}_bias_combined"
                for p in ["annual", "djf", "mam", "jja", "son"]
            ]
            # For precipitation, also require relative-bias figures
            if var == "pr":
                figure_ids += [
                    f"pr_{p}_relative_bias_combined"
                    for p in ["annual", "djf", "mam", "jja", "son"]
                ]
            # Ensemble summary figures (only when ≥2 models configured)
            if len(self.config.models) >= 2:
                figure_ids += [
                    f"{var}_{p}_ens_bias_combined"
                    for p in ["annual", "djf", "mam", "jja", "son"]
                ]
            figs_exist = all(self._figure_exists(fid) for fid in figure_ids)
            nc_needed = self.save_netcdf and not self._netcdf_exists(var)

            if skip_existing and figs_exist and not nc_needed:
                logger.info(
                    "Skipping %s — all figures exist", var,
                )
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

                # Save figures only when they were not already present.
                if not (skip_existing and figs_exist):
                    figures = self._plot_variable(var, var_result)
                    for fig, meta in figures:
                        paths = self._save(fig, meta, meta["figure_id"])
                        saved.append(paths)

                if self.save_netcdf:
                    self._export_netcdf(var, var_result)
            except Exception:
                logger.warning(
                    "Variable %s failed — skipping", var, exc_info=True,
                )

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # -- NetCDF export -------------------------------------------------------

    @property
    def _netcdf_dir(self):
        from pathlib import Path
        return Path(self.config.output_dir) / "netcdf" / self.name

    def _netcdf_exists(self, var: str) -> bool:
        from feather.diag import netcdf_export
        paths = netcdf_export.biasmap_netcdf_paths(
            self._netcdf_dir, var, self.period,
        )
        return all(p.exists() for p in paths)

    def _export_netcdf(self, var: str, var_result: dict) -> None:
        from feather.diag import netcdf_export
        var_info = var_result.get("var_info")
        units = getattr(var_info, "units", "") if var_info else ""
        netcdf_export.export_biasmap_netcdf(
            self._netcdf_dir, var, var_result, self.period,
            units=units, skip_existing=True,
        )

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
        obs_data = self._load_obs_var(var, self.period)
        obs_clim = climatology(obs_data)
        obs_gmean = float(latlon_global_mean(obs_clim).values)
        obs_seasonal = seasonal_climatology(obs_data)

        lat_name = "lat" if "lat" in obs_clim.coords else "latitude"
        lon_name = "lon" if "lon" in obs_clim.coords else "longitude"
        obs_lats = obs_clim[lat_name].values
        obs_lons = obs_clim[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Cache nereus interpolator per source grid size.
        # Different-resolution models (e.g. nside=1024 vs nside=128) need
        # separate interpolators, but models sharing a grid reuse the same one.
        _interp_cache: dict[int, Any] = {}
        obs_clim_common = None
        obs_seasonal_common = {}
        # Pre-computed area weights for the common grid (set once)
        common_area = None
        target_lats = None
        target_lons = None

        for model in self.config.models:
            logger.info("Computing biases for %s / %s ...", var, model)

            try:
                model_data = self._load_model_var(model, var)
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

            # Compute climatologies with dask, then materialise
            model_clim = climatology(model_data, self.period).compute()
            model_gmean = float(model_clim.mean().values)

            model_seasonal = seasonal_climatology(model_data, self.period)
            model_seasonal = {
                s: model_seasonal[s].compute()
                for s in model_seasonal.data_vars
            }

            # Build/reuse interpolator keyed by source grid size
            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _interp_cache:
                logger.info("  Building nereus interpolator (grid size %d)...",
                            n_src)
                annual_regrid, interp = nr.regrid(
                    model_clim.values.ravel(),
                    lon=np.asarray(lon), lat=np.asarray(lat),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]

                    # Regrid obs to common nereus grid via NN (once).
                    obs_lons_2d, obs_lats_2d = np.meshgrid(
                        obs_lons, obs_lats,
                    )
                    _, obs_interpolator = nr.regrid(
                        obs_clim.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=obs_res,
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    obs_clim_common = xr.DataArray(
                        obs_interpolator(obs_clim.values.ravel()),
                        dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )

                    # Pre-compute area weights for the common grid (once)
                    from feather.util.spatial import compute_latlon_areas
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )

                    # Also regrid seasonal obs (reuse obs interpolator)
                    obs_seasonal_common = {}
                    for season in obs_seasonal:
                        s_np = obs_interpolator(
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

            # --- Annual bias ---
            annual_bias = annual_regrid - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(annual_bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(annual_bias ** 2, area=common_area).values
            ))

            # --- Seasonal biases (DJF, MAM, JJA, SON) ---
            seasonal_biases: dict[str, Any] = {}
            seasonal_regrids: dict[str, Any] = {}
            for season in ["DJF", "MAM", "JJA", "SON"]:
                if season in model_seasonal:
                    s_np = _interp_cache[n_src](
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

            # --- t-test: model vs obs across grid points ---
            t_stat, t_pval = spatial_ttest(
                annual_regrid, obs_clim_common, weights=common_area,
            )
            # --- Variance ratio F-test ---
            f_stat, f_pval = spatial_variance_ratio(
                annual_regrid, obs_clim_common, weights=common_area,
            )

            seasonal_ttest_stats: dict[str, dict] = {}
            for season in ["DJF", "MAM", "JJA", "SON"]:
                if season in seasonal_regrids and season in obs_seasonal_common:
                    s_t, s_p = spatial_ttest(
                        seasonal_regrids[season], obs_seasonal_common[season],
                        weights=common_area,
                    )
                    s_f, s_fp = spatial_variance_ratio(
                        seasonal_regrids[season], obs_seasonal_common[season],
                        weights=common_area,
                    )
                    seasonal_ttest_stats[season] = {
                        "ttest_statistic": s_t, "ttest_pvalue": s_p,
                        "ftest_statistic": s_f, "ftest_pvalue": s_fp,
                    }

            model_results[model] = {
                "annual_regrid": annual_regrid,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": model_gmean,
                "annual_bias": annual_bias,
                "annual_bias_gmean": bias_gmean,
                "annual_rmse": rmse,
                "seasonal_biases": seasonal_biases,
                "ttest_statistic": t_stat,
                "ttest_pvalue": t_pval,
                "ftest_statistic": f_stat,
                "ftest_pvalue": f_pval,
                "seasonal_ttest": seasonal_ttest_stats,
            }

        if not model_results:
            logger.warning(
                "  No models have variable %s — skipping", var,
            )
            return None

        # Benchmark biases (CMIP6, HighResMIP, …) — one MMM per benchmark.
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        benchmark_data: dict[str, dict] = {}    # {label: per-period MMM data}
        benchmark_info: dict[str, dict] = {}
        if self.cmip6_enabled and target_lats is not None:
            for i, bench in enumerate(self.benchmarks):
                label = getattr(bench, "label", "CMIP6 MMM")
                if i == 0 and self.cmip6_individual:
                    # Primary benchmark: individual models + MMM from them.
                    cmip6_individual_data = self._compute_cmip6_individual(
                        var, target_lats, target_lons,
                        obs_clim_common, obs_seasonal_common,
                        common_area,
                    )
                    b_data, b_info = self._mmm_from_individual(
                        cmip6_individual_data,
                        obs_clim_common, obs_seasonal_common,
                        common_area,
                    )
                else:
                    b_data, b_info = self._compute_cmip6_mmm(
                        var, target_lats, target_lons,
                        obs_clim_common, obs_seasonal_common,
                        common_area, loader=bench,
                    )
                if b_data:
                    benchmark_data[label] = b_data
                    benchmark_info[label] = b_info

            # Back-compat: expose the primary benchmark under cmip6_* keys.
            if benchmark_data:
                primary_label = next(iter(benchmark_data))
                cmip6_data = benchmark_data[primary_label]
                cmip6_info = benchmark_info[primary_label]

        # EERIE ensemble mean/median bias maps (when ≥2 models available)
        ens_data: dict[str, dict] = {}
        if len(model_results) >= 2 and obs_clim_common is not None:
            ens_data = self._compute_ens_stats(
                model_results, obs_clim_common, obs_seasonal_common,
                common_area,
            )

        # Compute shared colorbar ranges across all models per period
        logger.info("  Computing shared colorbar ranges")
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
            cmip6_data=cmip6_data,
            cmip6_individual_data=cmip6_individual_data,
            ens_data=ens_data,
            benchmark_data=benchmark_data,
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
            "benchmark_data": benchmark_data,
            "benchmark_info": benchmark_info,
            "ens_data": ens_data,
        }

    # -- Ensemble stats helpers ---------------------------------------------

    @staticmethod
    def _compute_ens_stats(model_results, obs_clim_common,
                           obs_seasonal_common, common_area):
        """Compute EERIE ensemble mean and median bias maps.

        Stacks regridded model fields on a new ``member`` dimension and
        computes the element-wise mean and median.  Requires at least two
        models in *model_results*.

        Parameters
        ----------
        model_results : dict
            Per-model computation output from ``_compute_variable``.
        obs_clim_common, obs_seasonal_common, common_area
            Common-grid observation and area weights.

        Returns
        -------
        dict
            Keyed by period (``"annual"``, ``"DJF"``, ``"JJA"``).  Each entry
            holds ``mean``, ``median``, ``mean_bias``, ``median_bias``,
            ``mean_bias_gmean``, ``median_bias_gmean``, ``mean_rmse``, and
            ``median_rmse``.
        """
        ens_data: dict[str, dict] = {}

        # Annual
        annual_fields = [mr["annual_regrid"] for mr in model_results.values()]
        if len(annual_fields) < 2:
            return ens_data

        stacked = xr.concat(annual_fields, dim="member")
        ens_mean = stacked.mean("member")
        ens_median = stacked.median("member")
        mean_bias = ens_mean - obs_clim_common
        median_bias = ens_median - obs_clim_common

        mean_bias_gmean = float(
            latlon_global_mean(mean_bias, area=common_area).values
        )
        mean_rmse = float(np.sqrt(
            latlon_global_mean(mean_bias ** 2, area=common_area).values
        ))
        median_bias_gmean = float(
            latlon_global_mean(median_bias, area=common_area).values
        )
        median_rmse = float(np.sqrt(
            latlon_global_mean(median_bias ** 2, area=common_area).values
        ))

        ens_data["annual"] = {
            "mean": ens_mean,
            "median": ens_median,
            "mean_bias": mean_bias,
            "median_bias": median_bias,
            "mean_bias_gmean": mean_bias_gmean,
            "median_bias_gmean": median_bias_gmean,
            "mean_rmse": mean_rmse,
            "median_rmse": median_rmse,
            "n_members": len(annual_fields),
        }

        # Seasonal
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season not in obs_seasonal_common:
                continue
            s_fields = [
                mr["seasonal_regrids"].get(season)
                for mr in model_results.values()
            ]
            s_fields = [f for f in s_fields if f is not None]
            if len(s_fields) < 2:
                continue
            s_stacked = xr.concat(s_fields, dim="member")
            s_mean = s_stacked.mean("member")
            s_median = s_stacked.median("member")
            s_mean_bias = s_mean - obs_seasonal_common[season]
            s_median_bias = s_median - obs_seasonal_common[season]
            ens_data[season] = {
                "mean": s_mean,
                "median": s_median,
                "mean_bias": s_mean_bias,
                "median_bias": s_median_bias,
                "mean_bias_gmean": float(
                    latlon_global_mean(s_mean_bias, area=common_area).values
                ),
                "median_bias_gmean": float(
                    latlon_global_mean(s_median_bias, area=common_area).values
                ),
                "mean_rmse": float(np.sqrt(
                    latlon_global_mean(
                        s_mean_bias ** 2, area=common_area,
                    ).values
                )),
                "median_rmse": float(np.sqrt(
                    latlon_global_mean(
                        s_median_bias ** 2, area=common_area,
                    ).values
                )),
                "n_members": len(s_fields),
            }

        return ens_data

    # -- CMIP6 computation helpers ------------------------------------------

    def _compute_cmip6_mmm(self, var, target_lats, target_lons,
                           obs_clim_common, obs_seasonal_common,
                           common_area, loader=None):
        """Compute benchmark multi-model mean biases.

        Loads per-model climatologies, regrids each individually (so
        that the configured interpolation method is applied per model),
        then averages the regridded fields to form the MMM.  *loader*
        defaults to the primary benchmark; pass another benchmark loader
        (e.g. HighResMIP) to compute its MMM.
        """
        loader = loader or self.cmip6_loader
        cmip6_data = {}
        cmip6_info = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Computing %s MMM for %s...",
                    getattr(loader, "label", "CMIP6"), var)
        member_pairs = loader.get_member_pairs()

        annual_fields = []
        seasonal_fields: dict[str, list] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        models_used = []

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = loader.load_var_for_model_var(
                var, model, variant=variant, period=self.period,
            )
            if da is None:
                logger.debug("  Skipping %s — no data", label)
                continue

            regridded = self._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            annual_fields.append(regridded)
            models_used.append(label)

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = loader.load_var_for_model_var(
                    var, model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is not None:
                    s_regridded = self._regrid_to_target(
                        da_s, target_lats, target_lons,
                        resolution, influence_radius, cmip6_interp_cache,
                        method=self._regrid_method,
                    )
                    seasonal_fields[season].append(s_regridded)

        if not annual_fields:
            logger.info("  No CMIP6 models available for %s", var)
            return cmip6_data, cmip6_info

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        # MMM annual
        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_bias_gmean = float(
            latlon_global_mean(cmip6_bias, area=common_area).values
        )
        cmip6_rmse = float(np.sqrt(
            latlon_global_mean(
                cmip6_bias ** 2, area=common_area,
            ).values
        ))
        cmip6_t, cmip6_p = spatial_ttest(
            mmm, obs_clim_common, weights=common_area,
        )
        cmip6_f, cmip6_fp = spatial_variance_ratio(
            mmm, obs_clim_common, weights=common_area,
        )
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": cmip6_bias_gmean,
            "rmse": cmip6_rmse,
            "ttest_statistic": cmip6_t,
            "ttest_pvalue": cmip6_p,
            "ftest_statistic": cmip6_f,
            "ftest_pvalue": cmip6_fp,
        }

        # MMM seasonal
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if not seasonal_fields[season]:
                continue
            if season not in obs_seasonal_common:
                continue
            s_mmm = xr.concat(
                seasonal_fields[season], dim="member",
            ).mean("member")
            s_bias = s_mmm - obs_seasonal_common[season]
            s_t, s_p = spatial_ttest(
                s_mmm, obs_seasonal_common[season], weights=common_area,
            )
            s_f, s_fp = spatial_variance_ratio(
                s_mmm, obs_seasonal_common[season], weights=common_area,
            )
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(
                        s_bias, area=common_area,
                    ).values
                ),
                "ttest_statistic": s_t,
                "ttest_pvalue": s_p,
                "ftest_statistic": s_f,
                "ftest_pvalue": s_fp,
            }

        return cmip6_data, cmip6_info

    @staticmethod
    def _mmm_from_individual(cmip6_individual_data,
                             obs_clim_common, obs_seasonal_common,
                             common_area):
        """Derive MMM from already-regridded individual CMIP6 fields.

        Avoids regridding each model a second time when both individual
        and MMM results are needed (``cmip6_individual=True``).
        """
        cmip6_data = {}

        if "annual" not in cmip6_individual_data:
            return cmip6_data, {}

        annual_entries = cmip6_individual_data["annual"]
        models_used = list(annual_entries.keys())
        annual_fields = [e["regrid"] for e in annual_entries.values()]

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_bias_gmean = float(
            latlon_global_mean(cmip6_bias, area=common_area).values
        )
        cmip6_rmse = float(np.sqrt(
            latlon_global_mean(
                cmip6_bias ** 2, area=common_area,
            ).values
        ))
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": cmip6_bias_gmean,
            "rmse": cmip6_rmse,
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season not in cmip6_individual_data:
                continue
            if season not in obs_seasonal_common:
                continue
            s_fields = [
                e["regrid"]
                for e in cmip6_individual_data[season].values()
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

    def _compute_cmip6_individual(self, var, target_lats, target_lons,
                                  obs_clim_common, obs_seasonal_common,
                                  common_area):
        """Compute individual CMIP6 model biases."""
        cmip6_individual_data: dict[str, dict] = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))

        # Cache nereus interpolators per grid shape so models on the
        # same native grid share a single KDTree build.
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

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

            cmip6_common = self._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            cmip6_bias = cmip6_common - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(
                    cmip6_bias ** 2, area=common_area,
                ).values
            ))

            ind_t, ind_p = spatial_ttest(
                cmip6_common, obs_clim_common, weights=common_area,
            )
            ind_f, ind_fp = spatial_variance_ratio(
                cmip6_common, obs_clim_common, weights=common_area,
            )
            cmip6_individual_data.setdefault("annual", {})[label] = {
                "regrid": cmip6_common,
                "bias": cmip6_bias,
                "bias_gmean": bias_gmean,
                "rmse": rmse,
                "ttest_statistic": ind_t,
                "ttest_pvalue": ind_p,
                "ftest_statistic": ind_f,
                "ftest_pvalue": ind_fp,
            }

            # Seasonal
            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    var, model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is None or season not in obs_seasonal_common:
                    continue
                cmip6_s = self._regrid_to_target(
                    da_s, target_lats, target_lons,
                    resolution, influence_radius, cmip6_interp_cache,
                    method=self._regrid_method,
                )
                cmip6_s_bias = cmip6_s - obs_seasonal_common[season]
                s_t, s_p = spatial_ttest(
                    cmip6_s, obs_seasonal_common[season],
                    weights=common_area,
                )
                s_f, s_fp = spatial_variance_ratio(
                    cmip6_s, obs_seasonal_common[season],
                    weights=common_area,
                )
                cmip6_individual_data.setdefault(season, {})[label] = {
                    "regrid": cmip6_s,
                    "bias": cmip6_s_bias,
                    "bias_gmean": float(
                        latlon_global_mean(
                            cmip6_s_bias, area=common_area,
                        ).values
                    ),
                    "ttest_statistic": s_t,
                    "ttest_pvalue": s_p,
                    "ftest_statistic": s_f,
                    "ftest_pvalue": s_fp,
                }

        return cmip6_individual_data

    @staticmethod
    def _mmm_from_individual(cmip6_individual_data, obs_clim_common,
                             obs_seasonal_common, common_area):
        """Derive MMM from already-regridded individual CMIP6 fields.

        Avoids regridding a second time — reuses the ``"regrid"`` arrays
        stored in *cmip6_individual_data*.

        Returns (cmip6_data, cmip6_info) in the same format as
        ``_compute_cmip6_mmm``.
        """
        cmip6_data = {}

        if "annual" not in cmip6_individual_data:
            return cmip6_data, {}

        annual_members = cmip6_individual_data["annual"]
        models_used = list(annual_members.keys())
        annual_fields = [m["regrid"] for m in annual_members.values()]

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        mmm_t, mmm_p = spatial_ttest(
            mmm, obs_clim_common, weights=common_area,
        )
        mmm_f, mmm_fp = spatial_variance_ratio(
            mmm, obs_clim_common, weights=common_area,
        )
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
            "ttest_statistic": mmm_t,
            "ttest_pvalue": mmm_p,
            "ftest_statistic": mmm_f,
            "ftest_pvalue": mmm_fp,
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
            s_t, s_p = spatial_ttest(
                s_mmm, obs_seasonal_common[season], weights=common_area,
            )
            s_f, s_fp = spatial_variance_ratio(
                s_mmm, obs_seasonal_common[season], weights=common_area,
            )
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(
                        s_bias, area=common_area,
                    ).values
                ),
                "ttest_statistic": s_t,
                "ttest_pvalue": s_p,
                "ftest_statistic": s_f,
                "ftest_pvalue": s_fp,
            }

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }
        return cmip6_data, cmip6_info

    # -- Regridding helper --------------------------------------------------

    @staticmethod
    def _regrid_to_target(da, target_lats, target_lons,
                          resolution, influence_radius,
                          interp_cache, method="nearest"):
        """Regrid a regular lat/lon DataArray to the target grid via nereus.

        Uses *interp_cache* (keyed by grid shape) to avoid rebuilding
        the KDTree for models that share the same native grid.

        For coarse-resolution source grids (e.g. CMIP6 at 1-2°) the
        configured *influence_radius* (tuned for 5 km HEALPix) is too
        small.  We use 250 km as the floor, which is safe for the
        atmospheric variables handled by GlobalBiases.  Ocean diagnostics
        would need a more careful choice to avoid smearing across coasts.

        Source longitudes are converted to -180..180 and the target grid
        uses ``lon_bounds=(-180, 180)`` so that Delaunay triangulation
        (used by ``method="linear"`` / ``"cubic"``) does not produce a
        NaN stripe at the prime meridian.  The output columns are rolled
        back to 0..360 to match *target_lons*.
        """
        # 250 km floor — covers CMIP6 grids up to ~2° at the equator
        ir = max(influence_radius, 250_000.0)

        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lat_arr = da[lat_name].values
        lon_arr = da[lon_name].values

        # Convert to -180..180 to avoid gap at 0° in triangulation
        lon_arr = np.where(lon_arr > 180, lon_arr - 360, lon_arr)
        sort_idx = np.argsort(lon_arr)
        lon_arr = lon_arr[sort_idx]

        grid_key = (len(lat_arr), len(lon_arr))

        if grid_key not in interp_cache:
            lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
            _, interp_cache[grid_key] = nr.regrid(
                da.values[:, sort_idx].ravel(),
                lon=lon_2d.ravel(), lat=lat_2d.ravel(),
                resolution=resolution,
                method=method,
                influence_radius=ir,
                lon_bounds=(-180.0, 180.0),
                as_xarray=True,
            )

        regridded = interp_cache[grid_key](da.values[:, sort_idx].ravel())

        # Roll from -180..180 to 0..360 order to match target_lons
        n_roll = regridded.shape[1] // 2
        regridded = np.roll(regridded, -n_roll, axis=1)

        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    # -- Colorbar range computation -----------------------------------------

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_clim_common: xr.DataArray,
        obs_seasonal_common: dict[str, xr.DataArray],
        cmip6_data: dict[str, dict] | None = None,
        cmip6_individual_data: dict[str, dict] | None = None,
        ens_data: dict[str, dict] | None = None,
        benchmark_data: dict[str, dict] | None = None,
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
        ens_data = ens_data or {}
        benchmark_data = benchmark_data or {}

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
        for b_data in benchmark_data.values():
            if "annual" in b_data:
                field_arrays.append(b_data["annual"]["regrid"])
                bias_arrays.append(b_data["annual"]["bias"])
        if "annual" in ens_data:
            bias_arrays.append(ens_data["annual"]["mean_bias"])
            bias_arrays.append(ens_data["annual"]["median_bias"])
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
            for b_data in benchmark_data.values():
                if season in b_data:
                    s_fields.append(b_data[season]["regrid"])
                    s_biases.append(b_data[season]["bias"])
            if season in ens_data:
                s_biases.append(ens_data[season]["mean_bias"])
                s_biases.append(ens_data[season]["median_bias"])
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
        """Generate combined multi-panel bias maps for a single variable.

        For ``pr``, additionally produces relative-bias (%) figures using the
        IPCC BrBG colormap (brown = dry bias, green = wet bias) and converts
        display units to mm/day.
        """
        figures: list[tuple[plt.Figure, dict]] = []

        var_info = vr["var_info"]
        obs_clim = vr["obs"]["clim"]
        cb = vr["colorbar_ranges"]
        cmip6_info = vr.get("cmip6_info", {})
        cmip6_individual_data = vr.get("cmip6_individual_data", {})
        benchmark_data = vr.get("benchmark_data", {})
        benchmark_info = vr.get("benchmark_info", {})
        ens_data = vr.get("ens_data", {})

        is_pr = (var == "pr")

        periods = [("annual", "Annual Mean")]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in cb:
                periods.append((season, season))

        for period_key, period_label in periods:
            # Build ordered bias dict: DestinE models first, then CMIP6
            bias_dict = {}
            summary_stats = {}
            all_models = []

            # Unit multiplier for display: pr → mm/day, others → 1
            unit_scale = _PR_TO_MMDAY if is_pr else 1.0

            for model, mdata in vr["models"].items():
                if period_key == "annual":
                    bias_field = mdata["annual_bias"]
                    summary_stats[model] = {
                        "global_mean_bias": mdata["annual_bias_gmean"] * unit_scale,
                        "rmse": mdata["annual_rmse"] * unit_scale,
                        "t_test_statistic": mdata["ttest_statistic"],
                        "t_test_p_value": mdata["ttest_pvalue"],
                        "variance_ratio": mdata["ftest_statistic"],
                        "variance_ratio_p_value": mdata["ftest_pvalue"],
                    }
                else:
                    bias_field = mdata["seasonal_biases"].get(period_key)
                    if bias_field is None:
                        continue
                    s_tt = mdata.get("seasonal_ttest", {}).get(period_key, {})
                    summary_stats[model] = {
                        "global_mean_bias": float(
                            latlon_global_mean(bias_field).values
                        ) * unit_scale,
                        "t_test_statistic": s_tt.get("ttest_statistic"),
                        "t_test_p_value": s_tt.get("ttest_pvalue"),
                        "variance_ratio": s_tt.get("ftest_statistic"),
                        "variance_ratio_p_value": s_tt.get("ftest_pvalue"),
                    }
                bias_dict[model] = bias_field
                all_models.append(model)

            # Add each benchmark MMM (CMIP6, HighResMIP, …) if available
            for b_label, b_data in benchmark_data.items():
                if period_key not in b_data:
                    continue
                c_data = b_data[period_key]
                bias_dict[b_label] = c_data["bias"]
                all_models.append(b_label)
                summary_stats[b_label] = {
                    "global_mean_bias": c_data["bias_gmean"] * unit_scale,
                    "rmse": (c_data["rmse"] * unit_scale
                             if c_data.get("rmse") is not None else None),
                    "t_test_statistic": c_data.get("ttest_statistic"),
                    "t_test_p_value": c_data.get("ttest_pvalue"),
                    "variance_ratio": c_data.get("ftest_statistic"),
                    "variance_ratio_p_value": c_data.get("ftest_pvalue"),
                }

            # Add individual CMIP6 models if available
            if period_key in cmip6_individual_data:
                for label, c_data in cmip6_individual_data[period_key].items():
                    bias_dict[label] = c_data["bias"]
                    all_models.append(label)
                    summary_stats[label] = {
                        "global_mean_bias": c_data["bias_gmean"] * unit_scale,
                        "rmse": (c_data["rmse"] * unit_scale
                                 if c_data.get("rmse") is not None else None),
                        "t_test_statistic": c_data.get("ttest_statistic"),
                        "t_test_p_value": c_data.get("ttest_pvalue"),
                        "variance_ratio": c_data.get("ftest_statistic"),
                        "variance_ratio_p_value": c_data.get("ftest_pvalue"),
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

            # ── Precipitation-specific: relative bias + IPCC colormaps ──────
            if is_pr:
                # Relative bias figure (%) — brown=dry, green=wet
                obs_masked = obs_period.where(obs_period > _REL_BIAS_THRESHOLD)
                rel_bias_dict = {
                    k: (v / obs_masked) * 100 for k, v in bias_dict.items()
                }
                # Compute relative bias stats in % for metadata
                rel_stats = {}
                for label, rel_field in rel_bias_dict.items():
                    finite = rel_field.values[np.isfinite(rel_field.values)]
                    if finite.size > 0:
                        rel_stats[label] = {
                            "mean_relative_bias_pct": float(np.nanmean(finite)),
                        }
                fig_rel, _ = plot_combined_map(
                    rel_bias_dict,
                    title=f"Precipitation Relative Bias ({period_label})",
                    cmap="BrBG",
                    vmin=-100, vmax=100,
                    units="%",
                    method=self._regrid_method,
                )
                meta_rel = self._build_metadata(
                    title=f"Precipitation Relative Bias ({period_label})",
                    figure_id=f"pr_{period_key.lower()}_relative_bias_combined",
                    models=all_models,
                    variables=["pr"],
                    description=(
                        f"{period_label} precipitation relative bias (%) vs "
                        f"{var_info.obs_dataset}. Masked where obs < 0.1 mm/day. "
                        "Brown = dry bias, green = wet bias."
                    ),
                    plot_type="combined_map",
                    period=self.period,
                    cmip6_info=cmip6_info or None,
                    benchmark_info=self._benchmark_meta_from_info(
                        benchmark_info) or None,
                    summary_statistics=rel_stats,
                )
                figures.append((fig_rel, meta_rel))

                # Scale absolute fields to mm/day for display
                obs_plot = obs_period * _PR_TO_MMDAY
                bias_plot = {k: v * _PR_TO_MMDAY for k, v in bias_dict.items()}
                vmin_p = (p_cb["vmin"] * _PR_TO_MMDAY
                          if p_cb.get("vmin") is not None else None)
                vmax_p = (p_cb["vmax"] * _PR_TO_MMDAY
                          if p_cb.get("vmax") is not None else None)
                bvmax_p = (p_cb["bias_vmax"] * _PR_TO_MMDAY
                           if p_cb.get("bias_vmax") is not None else None)
                disp_cmap = "YlGnBu"
                disp_bias_cmap = "BrBG"
                disp_units = "mm/day"
            elif var_info.display_offset != 0.0:
                # Temperature (K→°C): apply constant offset to the obs absolute
                # panel and its colorbar limits; bias panels are invariant.
                _off = var_info.display_offset
                obs_plot = obs_period + _off
                bias_plot = bias_dict
                vmin_p = ((p_cb["vmin"] + _off)
                          if p_cb.get("vmin") is not None else None)
                vmax_p = ((p_cb["vmax"] + _off)
                          if p_cb.get("vmax") is not None else None)
                bvmax_p = p_cb.get("bias_vmax")
                disp_cmap = var_info.cmap
                disp_bias_cmap = "RdBu_r"
                disp_units = var_info.display_units or var_info.units
            else:
                obs_plot = obs_period
                bias_plot = bias_dict
                vmin_p = p_cb.get("vmin")
                vmax_p = p_cb.get("vmax")
                bvmax_p = p_cb.get("bias_vmax")
                disp_cmap = var_info.cmap
                disp_bias_cmap = "RdBu_r"
                disp_units = var_info.display_units or var_info.units

            # Cloud cover: reverse the bias colormap so that blue = more
            # cloud (positive bias) and red = less cloud (negative bias).
            if var == "clt":
                disp_bias_cmap = "RdBu"

            # ── Absolute bias figure ─────────────────────────────────────────
            fig, axes = plot_combined_bias_map(
                obs_plot, bias_plot,
                title=f"{var_info.long_name} {period_label}",
                obs_title=var_info.obs_dataset,
                cmap=disp_cmap,
                bias_cmap=disp_bias_cmap,
                vmin=vmin_p,
                vmax=vmax_p,
                bias_vmax=bvmax_p,
                units=disp_units,
                method=self._regrid_method,
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
                    benchmark_info=self._benchmark_meta_from_info(
                        benchmark_info) or None,
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

            # ── Ensemble summary figure (obs + ens. median + ens. mean + CMIP6 MMM) ──
            if period_key in ens_data:
                edata = ens_data[period_key]
                n = edata["n_members"]

                # Panel labels include member count in mathtext bold
                lbl_median = rf"{self._project_name} ens. median $\mathbf{{({n})}}$"
                lbl_mean   = rf"{self._project_name} ens. mean $\mathbf{{({n})}}$"

                # Build the bias panel dict in display units
                ens_bias_dict = {
                    lbl_median: edata["median_bias"] * unit_scale,
                    lbl_mean:   edata["mean_bias"] * unit_scale,
                }
                bench_panel_labels = {}
                for b_label, b_data in benchmark_data.items():
                    if period_key not in b_data:
                        continue
                    m = benchmark_info.get(b_label, {}).get("n_members", 0)
                    panel_lbl = rf"{b_label} $\mathbf{{({m})}}$"
                    bench_panel_labels[b_label] = panel_lbl
                    ens_bias_dict[panel_lbl] = (
                        b_data[period_key]["bias"] * unit_scale
                    )

                ens_summary_stats = {
                    lbl_median: {
                        "global_mean_bias": edata["median_bias_gmean"] * unit_scale,
                        "rmse": edata.get("median_rmse", 0.0) * unit_scale,
                        "n_members": n,
                    },
                    lbl_mean: {
                        "global_mean_bias": edata["mean_bias_gmean"] * unit_scale,
                        "rmse": edata.get("mean_rmse", 0.0) * unit_scale,
                        "n_members": n,
                    },
                }
                for b_label, panel_lbl in bench_panel_labels.items():
                    c_data = benchmark_data[b_label][period_key]
                    ens_summary_stats[panel_lbl] = {
                        "global_mean_bias": c_data["bias_gmean"] * unit_scale,
                        "rmse": (c_data["rmse"] * unit_scale
                                 if c_data.get("rmse") is not None else None),
                        "n_members": benchmark_info.get(
                            b_label, {}).get("n_members", 0),
                    }

                fig_ens, _ = plot_combined_bias_map(
                    obs_plot,
                    ens_bias_dict,
                    title=f"{var_info.long_name} {period_label} — Ensemble",
                    obs_title=var_info.obs_dataset,
                    cmap=disp_cmap,
                    bias_cmap=disp_bias_cmap,
                    vmin=vmin_p,
                    vmax=vmax_p,
                    bias_vmax=bvmax_p,
                    units=disp_units,
                    method=self._regrid_method,
                )
                meta_ens = self._build_metadata(
                    title=(
                        f"{var_info.long_name} {period_label} Bias "
                        f"— Ensemble Summary"
                    ),
                    figure_id=f"{var}_{period_key.lower()}_ens_bias_combined",
                    models=list(self.config.models),
                    variables=[var],
                    description=(
                        f"{period_label} climatology bias maps for "
                        f"{var_info.long_name} — EERIE ensemble median, "
                        f"ensemble mean, and CMIP6 MMM."
                    ),
                    plot_type="combined_bias_map",
                    period=self.period,
                    cmip6_info=cmip6_info or None,
                    benchmark_info=self._benchmark_meta_from_info(
                        benchmark_info) or None,
                    summary_statistics=ens_summary_stats,
                )
                figures.append((fig_ens, meta_ens))

        return figures
