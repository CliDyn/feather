"""Global climatology bias maps diagnostic.

For each model x variable, produces 3-panel maps (Model | Obs | Bias)
for annual-mean and selected seasons (DJF, JJA).
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
from feather.plot.maps import plot_bias_map
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)


@register
class GlobalBiases(DiagnosticBase):
    """Climatology bias maps (model - obs).

    Produces annual-mean and seasonal (DJF, JJA) bias maps for each
    configured model and variable.
    """

    name = "global_biases"
    title = "Global Climatology Biases"
    domain = "sfc"
    variables = ["avg_2t"]
    group = "temperature"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014")):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period

    # ── Computation ────────────────────────────────────────────────────

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

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0
        )

        for var in self.variables:
            var_info = get_var(var)
            logger.info("Processing variable: %s (%s)", var, var_info.long_name)
            model_results: dict[str, dict] = {}

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

            for model in self.config.models:
                logger.info("Computing biases for %s / %s ...", var, model)

                key = DataLoader.make_key(
                    self.experiment, model, var_info.domain,
                )
                model_data = self.model_loader.load_var(key, var)
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
                bias_gmean = float(latlon_global_mean(annual_bias).values)
                rmse = float(np.sqrt(
                    latlon_global_mean(annual_bias ** 2).values
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

            # CMIP6 MMM bias (optional)
            cmip6_data = {}
            cmip6_info = {}
            if self.cmip6_enabled and interpolator is not None:
                logger.info("  Loading CMIP6 multi-model mean for %s...", var)
                mmm, info = self.cmip6_loader.load_mmm_for_model_var(
                    var, period=self.period,
                )
                if mmm is not None:
                    # Interpolate CMIP6 MMM to the common nereus target grid
                    cmip6_common = mmm.interp(
                        lat=target_lats, lon=target_lons,
                    )
                    cmip6_bias = cmip6_common - obs_clim_common
                    cmip6_bias_gmean = float(
                        latlon_global_mean(cmip6_bias).values
                    )
                    cmip6_rmse = float(np.sqrt(
                        latlon_global_mean(cmip6_bias ** 2).values
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
                                    latlon_global_mean(cmip6_s_bias).values
                                ),
                            }

            # Compute shared colorbar ranges across all models per period
            # (include CMIP6 if available)
            logger.info("  Computing shared colorbar ranges")
            colorbar_ranges = self._compute_colorbar_ranges(
                model_results, obs_clim_common, obs_seasonal_common,
                cmip6_data=cmip6_data,
            )

            results[var] = {
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
            }

        return results

    # ── Colorbar range computation ────────────────────────────────────

    @staticmethod
    def _compute_colorbar_ranges(
        model_results: dict[str, dict],
        obs_clim_common: xr.DataArray,
        obs_seasonal_common: dict[str, xr.DataArray],
        cmip6_data: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """Compute shared colorbar ranges across all models per period.

        Parameters
        ----------
        cmip6_data : dict, optional
            If provided, CMIP6 fields and biases are included in the
            range computation so all figures share the same color scale.

        Returns a dict keyed by period name ("annual", "DJF", "JJA")
        with ``vmin``, ``vmax`` (field panels) and ``bias_vmax``
        (symmetric bias panel) values.
        """
        ranges: dict[str, dict] = {}
        cmip6_data = cmip6_data or {}

        def _percentile_range(arrays):
            """Compute vmin/vmax from 2nd/98th percentile of arrays."""
            vals = np.concatenate([
                np.asarray(a).ravel()[np.isfinite(np.asarray(a).ravel())]
                for a in arrays
            ])
            return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

        def _bias_max(arrays):
            """Compute symmetric bias range from 98th percentile of |bias|."""
            vals = np.concatenate([
                np.asarray(a).ravel()[np.isfinite(np.asarray(a).ravel())]
                for a in arrays
            ])
            return float(np.percentile(np.abs(vals), 98)) or 1.0

        # Annual
        field_arrays = [mr["annual_regrid"] for mr in model_results.values()]
        field_arrays.append(obs_clim_common)
        bias_arrays = [mr["annual_bias"] for mr in model_results.values()]
        if "annual" in cmip6_data:
            field_arrays.append(cmip6_data["annual"]["regrid"])
            bias_arrays.append(cmip6_data["annual"]["bias"])
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
            vmin, vmax = _percentile_range(s_fields)
            ranges[season] = {
                "vmin": vmin, "vmax": vmax,
                "bias_vmax": _bias_max(s_biases) if s_biases else 1.0,
            }

        return ranges

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate 3-panel bias maps for each model x variable.

        Uses shared colorbar ranges across all models per period so
        figures can be compared side by side.

        Returns
        -------
        list of (Figure, metadata-dict)
        """
        figures: list[tuple[plt.Figure, dict]] = []

        for var, vr in results.items():
            var_info = vr["var_info"]
            obs_clim = vr["obs"]["clim"]
            obs_gmean = vr["obs"]["global_mean"]
            cb = vr["colorbar_ranges"]

            for model, mdata in vr["models"].items():
                # --- Annual bias map ---
                ann_cb = cb["annual"]
                fig, axes = plot_bias_map(
                    mdata["annual_regrid"], obs_clim,
                    bias_data=mdata["annual_bias"],
                    title=f"{var_info.long_name} Annual Mean \u2014 {model}",
                    cmap=var_info.cmap,
                    units=var_info.units,
                    vmin=ann_cb["vmin"], vmax=ann_cb["vmax"],
                    bias_vmax=ann_cb["bias_vmax"],
                )
                meta = self._build_metadata(
                    title=f"{var_info.long_name} Annual Bias \u2014 {model}",
                    figure_id=f"{var}_annual_bias_{model}",
                    models=[model],
                    variables=[var],
                    description=(
                        f"Annual mean climatology bias map for "
                        f"{var_info.long_name}."
                    ),
                    plot_type="bias_map",
                    period=self.period,
                    summary_statistics={
                        "global_mean_bias": mdata["annual_bias_gmean"],
                        "rmse": mdata["annual_rmse"],
                        "model_global_mean": mdata["global_mean"],
                        "obs_global_mean": obs_gmean,
                    },
                )
                figures.append((fig, meta))

                # --- Seasonal bias maps ---
                for season, bias in mdata["seasonal_biases"].items():
                    obs_s = vr["obs"]["seasonal_clim"][season]
                    model_s = mdata["seasonal_regrids"][season]
                    s_cb = cb.get(season, {})

                    fig_s, _ = plot_bias_map(
                        model_s, obs_s,
                        bias_data=bias,
                        title=(
                            f"{var_info.long_name} {season} \u2014 {model}"
                        ),
                        cmap=var_info.cmap,
                        units=var_info.units,
                        vmin=s_cb.get("vmin"),
                        vmax=s_cb.get("vmax"),
                        bias_vmax=s_cb.get("bias_vmax"),
                    )
                    meta_s = self._build_metadata(
                        title=(
                            f"{var_info.long_name} {season} Bias \u2014 "
                            f"{model}"
                        ),
                        figure_id=f"{var}_{season.lower()}_bias_{model}",
                        models=[model],
                        variables=[var],
                        description=(
                            f"{season} climatology bias for "
                            f"{var_info.long_name}."
                        ),
                        plot_type="bias_map",
                        period=self.period,
                        computation_notes=(
                            f"Seasonal climatology ({season}) bias map."
                        ),
                    )
                    figures.append((fig_s, meta_s))

            # --- CMIP6 MMM bias maps (one per variable, not per model) ---
            cmip6_data = vr.get("cmip6_data", {})
            cmip6_info = vr.get("cmip6_info", {})
            if cmip6_data:
                # Annual CMIP6 bias
                if "annual" in cmip6_data:
                    ann_cb = cb["annual"]
                    c_data = cmip6_data["annual"]
                    fig_c, _ = plot_bias_map(
                        c_data["regrid"], obs_clim,
                        bias_data=c_data["bias"],
                        title=(
                            f"{var_info.long_name} Annual Mean "
                            f"\u2014 CMIP6 MMM"
                        ),
                        model_title="CMIP6 MMM",
                        cmap=var_info.cmap,
                        units=var_info.units,
                        vmin=ann_cb["vmin"], vmax=ann_cb["vmax"],
                        bias_vmax=ann_cb["bias_vmax"],
                    )
                    meta_c = self._build_metadata(
                        title=(
                            f"{var_info.long_name} Annual Bias "
                            f"\u2014 CMIP6 MMM"
                        ),
                        figure_id=f"{var}_annual_bias_cmip6_mmm",
                        models=["CMIP6 MMM"],
                        variables=[var],
                        description=(
                            f"Annual mean climatology bias map for "
                            f"{var_info.long_name} — CMIP6 multi-model mean."
                        ),
                        plot_type="bias_map",
                        period=self.period,
                        cmip6_info=cmip6_info,
                        summary_statistics={
                            "global_mean_bias": c_data["bias_gmean"],
                            "rmse": c_data.get("rmse"),
                            "obs_global_mean": obs_gmean,
                        },
                    )
                    figures.append((fig_c, meta_c))

                # Seasonal CMIP6 biases
                for season in ["DJF", "JJA"]:
                    if season not in cmip6_data:
                        continue
                    s_cb = cb.get(season, {})
                    c_s = cmip6_data[season]
                    obs_s = vr["obs"]["seasonal_clim"].get(season)
                    if obs_s is None:
                        continue

                    fig_cs, _ = plot_bias_map(
                        c_s["regrid"], obs_s,
                        bias_data=c_s["bias"],
                        title=(
                            f"{var_info.long_name} {season} "
                            f"\u2014 CMIP6 MMM"
                        ),
                        model_title="CMIP6 MMM",
                        cmap=var_info.cmap,
                        units=var_info.units,
                        vmin=s_cb.get("vmin"),
                        vmax=s_cb.get("vmax"),
                        bias_vmax=s_cb.get("bias_vmax"),
                    )
                    meta_cs = self._build_metadata(
                        title=(
                            f"{var_info.long_name} {season} Bias "
                            f"\u2014 CMIP6 MMM"
                        ),
                        figure_id=f"{var}_{season.lower()}_bias_cmip6_mmm",
                        models=["CMIP6 MMM"],
                        variables=[var],
                        description=(
                            f"{season} climatology bias for "
                            f"{var_info.long_name} — CMIP6 multi-model mean."
                        ),
                        plot_type="bias_map",
                        period=self.period,
                        cmip6_info=cmip6_info,
                        computation_notes=(
                            f"Seasonal climatology ({season}) CMIP6 MMM "
                            f"bias map."
                        ),
                    )
                    figures.append((fig_cs, meta_cs))

        return figures
