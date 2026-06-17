"""Climate variability diagnostic.

For each variable, produces two multi-panel figures:
1. STD maps — ERA5 STD + model STD panels (sequential colormap, always positive)
2. STD difference maps — ERA5 STD + (model STD - ERA5 STD) bias panels (diverging)

Pre-processing: monthly data → deseason → detrend → std("time").
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
from feather.util.spatial import compute_latlon_areas, latlon_global_mean
from feather.util.temporal import deseason, detrend

logger = logging.getLogger(__name__)


@register
class ClimateVariability(DiagnosticBase):
    """Climate variability maps (standard deviation of deseasonalised,
    detrended monthly fields).

    Produces two combined multi-panel figures per variable:
    - STD maps (obs + models, shared sequential colorbar)
    - STD difference maps (obs STD + model-obs STD bias, diverging)
    """

    name = "climate_variability"
    title = "Climate Variability (ERA5)"
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
                 experiment=None, period=None,
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment or config.get_experiment()
        self.period = period or config.get_period()
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # -- Orchestration (per-variable incremental) ----------------------------

    def run(self, skip_existing: bool = True) -> list[tuple["Path", "Path"]]:
        """Execute per-variable: compute → plot → save immediately."""
        from pathlib import Path

        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        for var in self.variables:
            figure_ids = [
                f"{var}_std_combined",
                f"{var}_std_diff_combined",
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

    # -- ABC compat ---------------------------------------------------------

    def compute(self) -> dict[str, Any]:
        """Compute STD fields for all variables."""
        results: dict[str, Any] = {}
        for var in self.variables:
            var_result = self._compute_variable(var)
            if var_result is not None:
                results[var] = var_result
        return results

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate STD + STD diff maps for all computed variables."""
        figures: list[tuple[plt.Figure, dict]] = []
        for var, vr in results.items():
            figures.extend(self._plot_variable(var, vr))
        return figures

    # -- Core computation ---------------------------------------------------

    def _compute_variable(self, var: str) -> dict[str, Any] | None:
        """Compute STD of deseasonalised, detrended data for one variable.

        Returns None if no models have the variable.
        """
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_results: dict[str, dict] = {}

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )

        # Load observation
        logger.info("  Loading observations for %s", var)
        obs_data = self._load_obs_var(var, self.period)

        # Pre-process obs: deseason → compute → detrend → std
        obs_deseas = deseason(obs_data)
        if hasattr(obs_deseas, "compute"):
            obs_deseas = obs_deseas.compute()
        obs_detrended = detrend(obs_deseas)
        obs_std = obs_detrended.std("time")

        lat_name = "lat" if "lat" in obs_std.coords else "latitude"
        lon_name = "lon" if "lon" in obs_std.coords else "longitude"
        obs_lats = obs_std[lat_name].values
        obs_lons = obs_std[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Cache nereus interpolator per source grid size
        _interp_cache: dict[int, Any] = {}
        obs_std_common = None
        common_area = None
        target_lats = None
        target_lons = None

        for model in self.config.models:
            logger.info("Computing variability for %s / %s ...", var, model)

            try:
                model_data = self._load_model_var(model, var)
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "  Variable %s not available for %s — skipping",
                    var, model,
                )
                continue
            lon, lat = self._load_model_coords(model, var)

            # For latlon grids, meshgrid 1D coord arrays
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            # Slice to period, deseason, compute, detrend, std
            if "time" in model_data.dims and self.period:
                model_data = model_data.sel(
                    time=slice(self.period[0], self.period[1]),
                )
            model_deseas = deseason(model_data)
            if hasattr(model_deseas, "compute"):
                model_deseas = model_deseas.compute()
            model_detrended = detrend(model_deseas)
            model_std = model_detrended.std("time")

            # Build/reuse interpolator keyed by source grid size
            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _interp_cache:
                logger.info("  Building nereus interpolator (grid size %d)...",
                            n_src)
                std_regrid, interp = nr.regrid(
                    model_std.values.ravel(),
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

                    # Regrid obs to common nereus grid (once)
                    obs_lons_2d, obs_lats_2d = np.meshgrid(obs_lons, obs_lats)
                    _, obs_interpolator = nr.regrid(
                        obs_std.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=obs_res,
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    obs_std_common = xr.DataArray(
                        obs_interpolator(obs_std.values.ravel()),
                        dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )

                    # Pre-compute area weights for the common grid
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )
            else:
                interp = _interp_cache[n_src]
                regridded_np = interp(model_std.values.ravel())
                std_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            # STD difference
            std_diff = std_regrid - obs_std_common

            # Summary stats
            std_gmean = float(
                latlon_global_mean(std_regrid, area=common_area).values,
            )
            diff_gmean = float(
                latlon_global_mean(std_diff, area=common_area).values,
            )
            rmse = float(np.sqrt(
                latlon_global_mean(std_diff ** 2, area=common_area).values,
            ))

            model_results[model] = {
                "std_regrid": std_regrid,
                "std_diff": std_diff,
                "std_gmean": std_gmean,
                "diff_gmean": diff_gmean,
                "rmse": rmse,
            }

        if not model_results:
            logger.warning(
                "  No models have variable %s — skipping", var,
            )
            return None

        # Benchmark STDs (CMIP6, HighResMIP, …) — one MMM per benchmark.
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        benchmark_data: dict[str, dict] = {}
        benchmark_info: dict[str, dict] = {}
        if self.cmip6_enabled and target_lats is not None:
            for i, bench in enumerate(self.benchmarks):
                label = getattr(bench, "label", "CMIP6 MMM")
                if i == 0 and self.cmip6_individual:
                    cmip6_individual_data = self._compute_cmip6_individual_std(
                        var, target_lats, target_lons,
                        obs_std_common, common_area,
                    )
                    b_data, b_info = self._mmm_from_individual_std(
                        cmip6_individual_data,
                        obs_std_common, common_area,
                    )
                else:
                    b_data, b_info = self._compute_cmip6_mmm_std(
                        var, target_lats, target_lons,
                        obs_std_common, common_area, loader=bench,
                    )
                if b_data:
                    benchmark_data[label] = b_data
                    benchmark_info[label] = b_info

            if benchmark_data:
                primary_label = next(iter(benchmark_data))
                cmip6_data = benchmark_data[primary_label]
                cmip6_info = benchmark_info[primary_label]

        # Compute colorbar ranges
        obs_gmean = float(
            latlon_global_mean(obs_std_common, area=common_area).values,
        )
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_std_common,
            cmip6_data=cmip6_data,
            cmip6_individual_data=cmip6_individual_data,
            benchmark_data=benchmark_data,
        )

        return {
            "obs_std": obs_std_common,
            "obs_gmean": obs_gmean,
            "models": model_results,
            "var_info": var_info,
            "target_lats": target_lats,
            "target_lons": target_lons,
            "colorbar_ranges": colorbar_ranges,
            "cmip6_data": cmip6_data,
            "cmip6_info": cmip6_info,
            "cmip6_individual_data": cmip6_individual_data,
            "benchmark_data": benchmark_data,
            "benchmark_info": benchmark_info,
        }

    # -- CMIP6 helpers ------------------------------------------------------

    def _compute_cmip6_mmm_std(self, var, target_lats, target_lons,
                                obs_std_common, common_area, loader=None):
        """Compute benchmark multi-model mean STD (per-benchmark loader)."""
        loader = loader or self.cmip6_loader
        cmip6_data = {}
        cmip6_info = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Computing %s MMM STD for %s...",
                    getattr(loader, "label", "CMIP6"), var)
        member_pairs = loader.get_member_pairs()

        std_fields = []
        models_used = []

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = loader.load_var_for_model_var(
                var, model, variant=variant,
                period=self.period, time_mean=False,
            )
            if da is None:
                logger.debug("  Skipping %s — no data", label)
                continue
            if "time" not in da.dims or da.sizes["time"] < 3:
                logger.debug("  Skipping %s — insufficient timesteps", label)
                continue

            # deseason → compute → detrend → std
            da_deseas = deseason(da)
            if hasattr(da_deseas, "compute"):
                da_deseas = da_deseas.compute()
            da_detrended = detrend(da_deseas)
            da_std = da_detrended.std("time")

            regridded = self._regrid_to_target(
                da_std, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            std_fields.append(regridded)
            models_used.append(label)

        if not std_fields:
            logger.info("  No CMIP6 models available for %s", var)
            return cmip6_data, cmip6_info

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        mmm_std = xr.concat(std_fields, dim="member").mean("member")
        std_diff = mmm_std - obs_std_common
        diff_gmean = float(
            latlon_global_mean(std_diff, area=common_area).values,
        )
        rmse = float(np.sqrt(
            latlon_global_mean(std_diff ** 2, area=common_area).values,
        ))
        cmip6_data = {
            "std_regrid": mmm_std,
            "std_diff": std_diff,
            "std_gmean": float(
                latlon_global_mean(mmm_std, area=common_area).values,
            ),
            "diff_gmean": diff_gmean,
            "rmse": rmse,
        }

        return cmip6_data, cmip6_info

    def _compute_cmip6_individual_std(self, var, target_lats, target_lons,
                                       obs_std_common, common_area):
        """Compute per-CMIP6-model STD fields."""
        cmip6_individual_data: dict[str, dict] = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, nr.RegridInterpolator] = {}

        logger.info("  Loading individual CMIP6 STD for %s...", var)
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
            if "time" not in da.dims or da.sizes["time"] < 3:
                logger.debug("  Skipping %s — insufficient timesteps", label)
                continue

            da_deseas = deseason(da)
            if hasattr(da_deseas, "compute"):
                da_deseas = da_deseas.compute()
            da_detrended = detrend(da_deseas)
            da_std = da_detrended.std("time")

            regridded = self._regrid_to_target(
                da_std, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            std_diff = regridded - obs_std_common
            diff_gmean = float(
                latlon_global_mean(std_diff, area=common_area).values,
            )
            rmse = float(np.sqrt(
                latlon_global_mean(std_diff ** 2, area=common_area).values,
            ))

            cmip6_individual_data[label] = {
                "std_regrid": regridded,
                "std_diff": std_diff,
                "std_gmean": float(
                    latlon_global_mean(regridded, area=common_area).values,
                ),
                "diff_gmean": diff_gmean,
                "rmse": rmse,
            }

        return cmip6_individual_data

    @staticmethod
    def _mmm_from_individual_std(cmip6_individual_data,
                                  obs_std_common, common_area):
        """Derive MMM from already-regridded individual CMIP6 STD fields."""
        if not cmip6_individual_data:
            return {}, {}

        models_used = list(cmip6_individual_data.keys())
        std_fields = [d["std_regrid"] for d in cmip6_individual_data.values()]

        mmm_std = xr.concat(std_fields, dim="member").mean("member")
        std_diff = mmm_std - obs_std_common
        diff_gmean = float(
            latlon_global_mean(std_diff, area=common_area).values,
        )
        rmse = float(np.sqrt(
            latlon_global_mean(std_diff ** 2, area=common_area).values,
        ))

        cmip6_data = {
            "std_regrid": mmm_std,
            "std_diff": std_diff,
            "std_gmean": float(
                latlon_global_mean(mmm_std, area=common_area).values,
            ),
            "diff_gmean": diff_gmean,
            "rmse": rmse,
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

        Identical to GlobalBiases._regrid_to_target: uses interp_cache,
        converts source lons to -180..180, rolls output back to 0..360.
        """
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

        # Roll from -180..180 to 0..360 to match target_lons
        n_roll = regridded.shape[1] // 2
        regridded = np.roll(regridded, -n_roll, axis=1)

        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    # -- Colorbar ranges ----------------------------------------------------

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_std_common: xr.DataArray,
        cmip6_data: dict | None = None,
        cmip6_individual_data: dict | None = None,
        benchmark_data: dict | None = None,
    ) -> dict[str, dict]:
        """Compute shared colorbar ranges for STD and diff panels."""
        cmip6_data = cmip6_data or {}
        cmip6_individual_data = cmip6_individual_data or {}
        benchmark_data = benchmark_data or {}

        def _finite_vals(arrays):
            parts = []
            for a in arrays:
                v = np.asarray(a).ravel()
                parts.append(v[np.isfinite(v)])
            return np.concatenate(parts)

        # STD range (always positive)
        std_arrays = [mr["std_regrid"] for mr in model_results.values()]
        std_arrays.append(obs_std_common)
        if cmip6_data:
            std_arrays.append(cmip6_data["std_regrid"])
        for member_data in cmip6_individual_data.values():
            std_arrays.append(member_data["std_regrid"])
        for b_data in benchmark_data.values():
            std_arrays.append(b_data["std_regrid"])

        std_vals = _finite_vals(std_arrays)
        std_vmin = float(np.percentile(std_vals, 2))
        std_vmax = float(np.percentile(std_vals, 98))

        # Diff range (symmetric)
        diff_arrays = [mr["std_diff"] for mr in model_results.values()]
        if cmip6_data:
            diff_arrays.append(cmip6_data["std_diff"])
        for member_data in cmip6_individual_data.values():
            diff_arrays.append(member_data["std_diff"])
        for b_data in benchmark_data.values():
            diff_arrays.append(b_data["std_diff"])

        diff_vals = _finite_vals(diff_arrays)
        bias_vmax = float(np.percentile(np.abs(diff_vals), 98)) or 1.0

        return {
            "std": {"vmin": std_vmin, "vmax": std_vmax},
            "diff": {"bias_vmax": bias_vmax},
        }

    # -- Plotting -----------------------------------------------------------

    def _plot_variable(
        self, var: str, vr: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Generate STD map + STD diff map for a single variable."""
        figures: list[tuple[plt.Figure, dict]] = []

        var_info = vr["var_info"]
        obs_std = vr["obs_std"]
        cb = vr["colorbar_ranges"]
        cmip6_info = vr.get("cmip6_info", {})
        cmip6_individual_data = vr.get("cmip6_individual_data", {})
        benchmark_data = vr.get("benchmark_data", {})

        all_models = list(vr["models"].keys())

        # --- Figure 1: STD maps (all panels same colormap) ---
        std_data_dict = {var_info.obs_dataset: obs_std}
        for model, mdata in vr["models"].items():
            std_data_dict[model] = mdata["std_regrid"]
        for b_label, b_data in benchmark_data.items():
            std_data_dict[b_label] = b_data["std_regrid"]
            all_models.append(b_label)
        for label, cdata in cmip6_individual_data.items():
            std_data_dict[label] = cdata["std_regrid"]
            all_models.append(label)

        fig1, _ = plot_combined_map(
            std_data_dict,
            title=f"{var_info.long_name} — Variability (STD)",
            cmap="YlOrRd",
            vmin=cb["std"]["vmin"],
            vmax=cb["std"]["vmax"],
            units=var_info.units,
            method=self._regrid_method,
        )

        summary_stats = {}
        for model, mdata in vr["models"].items():
            summary_stats[model] = {
                "std_gmean": mdata["std_gmean"],
                "diff_gmean": mdata["diff_gmean"],
                "rmse": mdata["rmse"],
            }
        for b_label, b_data in benchmark_data.items():
            summary_stats[b_label] = {
                "std_gmean": b_data["std_gmean"],
                "diff_gmean": b_data["diff_gmean"],
                "rmse": b_data["rmse"],
            }

        meta1 = self._build_metadata(
            title=f"{var_info.long_name} — Variability (STD)",
            figure_id=f"{var}_std_combined",
            models=all_models,
            variables=[var],
            description=(
                f"Standard deviation of deseasonalised, detrended monthly "
                f"{var_info.long_name} — all models and ERA5."
            ),
            plot_type="combined_map",
            period=self.period,
            cmip6_info=cmip6_info or None,
            summary_statistics=summary_stats,
        )
        figures.append((fig1, meta1))

        # --- Figure 2: STD difference maps (obs + bias panels) ---
        bias_dict = {}
        for model, mdata in vr["models"].items():
            bias_dict[model] = mdata["std_diff"]
        for b_label, b_data in benchmark_data.items():
            bias_dict[b_label] = b_data["std_diff"]
        for label, cdata in cmip6_individual_data.items():
            bias_dict[label] = cdata["std_diff"]

        fig2, _ = plot_combined_bias_map(
            obs_std, bias_dict,
            title=f"{var_info.long_name} — Variability Bias (STD diff)",
            obs_title=f"{var_info.obs_dataset} STD",
            cmap="YlOrRd",
            bias_cmap="RdBu_r",
            vmin=cb["std"]["vmin"],
            vmax=cb["std"]["vmax"],
            bias_vmax=cb["diff"]["bias_vmax"],
            units=var_info.units,
            method=self._regrid_method,
        )

        meta2 = self._build_metadata(
            title=f"{var_info.long_name} — Variability Bias (STD diff)",
            figure_id=f"{var}_std_diff_combined",
            models=all_models,
            variables=[var],
            description=(
                f"ERA5 STD and model-ERA5 STD differences for "
                f"{var_info.long_name}. Positive = model more variable."
            ),
            plot_type="combined_bias_map",
            period=self.period,
            cmip6_info=cmip6_info or None,
            summary_statistics=summary_stats,
        )
        figures.append((fig2, meta2))

        return figures
