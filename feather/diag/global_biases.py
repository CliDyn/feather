"""Global climatology bias maps diagnostic.

For each model x variable, produces 3-panel maps (Model | Obs | Bias)
for annual-mean and selected seasons (DJF, JJA).
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from feather.data.loader import DataLoader
from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_bias_map
from feather.util.spatial import (
    RegridIndex,
    global_mean,
    latlon_global_mean,
    regrid_to_latlon,
)
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

        for var in self.variables:
            var_info = get_var(var)
            model_results: dict[str, dict] = {}

            # Load observation (usually small regular grid)
            obs_data = self.obs_loader.load_for_model_var(var, self.period)
            obs_clim = climatology(obs_data)
            obs_gmean = float(latlon_global_mean(obs_clim).values)
            obs_seasonal = seasonal_climatology(obs_data)

            obs_lats = obs_clim.lat.values if "lat" in obs_clim.coords else obs_clim.latitude.values
            obs_lons = obs_clim.lon.values if "lon" in obs_clim.coords else obs_clim.longitude.values

            # Pre-build regrid index once (all models share the same
            # HEALPix grid, so the KDTree is built only once).
            regrid_idx: RegridIndex | None = None

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
                # Materialise seasonal climatologies
                model_seasonal = {
                    s: model_seasonal[s].compute()
                    for s in model_seasonal.data_vars
                }

                # Build KDTree index once (reused across models + seasons)
                if regrid_idx is None:
                    lon_np = np.asarray(lon)
                    lat_np = np.asarray(lat)
                    regrid_idx = RegridIndex.build(
                        lon_np, lat_np, obs_lats, obs_lons,
                    )

                # --- Annual bias (fast: just index lookup) ---
                annual_regrid = regrid_idx.apply(model_clim)
                annual_bias = annual_regrid - obs_clim.values
                annual_bias = annual_bias.assign_coords(
                    lat=obs_lats, lon=obs_lons,
                )
                bias_gmean = float(latlon_global_mean(annual_bias).values)
                rmse = float(np.sqrt(
                    latlon_global_mean(annual_bias ** 2).values
                ))

                # --- Seasonal biases (DJF, JJA) ---
                seasonal_biases: dict[str, Any] = {}
                for season in ["DJF", "JJA"]:
                    if season in model_seasonal:
                        s_regrid = regrid_idx.apply(model_seasonal[season])
                        if season in obs_seasonal:
                            obs_s = obs_seasonal[season]
                            s_bias = s_regrid - obs_s.values
                            s_bias = s_bias.assign_coords(
                                lat=obs_lats, lon=obs_lons,
                            )
                            seasonal_biases[season] = s_bias

                model_results[model] = {
                    "annual_clim": model_clim,
                    "seasonal_clim": model_seasonal,
                    "lon": lon,
                    "lat": lat,
                    "global_mean": model_gmean,
                    "annual_bias": annual_bias,
                    "annual_bias_gmean": bias_gmean,
                    "annual_rmse": rmse,
                    "seasonal_biases": seasonal_biases,
                }

            results[var] = {
                "models": model_results,
                "obs": {
                    "clim": obs_clim,
                    "seasonal_clim": obs_seasonal,
                    "global_mean": obs_gmean,
                },
                "var_info": var_info,
            }

        return results

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate 3-panel bias maps for each model x variable.

        Returns
        -------
        list of (Figure, metadata-dict)
        """
        figures: list[tuple[plt.Figure, dict]] = []

        for var, vr in results.items():
            var_info = vr["var_info"]
            obs_clim = vr["obs"]["clim"]
            obs_gmean = vr["obs"]["global_mean"]

            for model, mdata in vr["models"].items():
                # --- Annual bias map ---
                fig, axes, _ = plot_bias_map(
                    mdata["annual_clim"], obs_clim,
                    mdata["lon"], mdata["lat"],
                    bias_data=mdata["annual_bias"],
                    title=f"{var_info.long_name} Annual Mean \u2014 {model}",
                    cmap=var_info.cmap,
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
                    model_s = mdata["seasonal_clim"][season]

                    fig_s, _, _ = plot_bias_map(
                        model_s, obs_s,
                        mdata["lon"], mdata["lat"],
                        bias_data=bias,
                        title=(
                            f"{var_info.long_name} {season} \u2014 {model}"
                        ),
                        cmap=var_info.cmap,
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

        return figures
