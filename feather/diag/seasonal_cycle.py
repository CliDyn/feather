"""Seasonal cycle diagnostic.

Plots the 12-month climatological cycle (Jan-Dec) of global-mean values
for each variable, with all configured models and observations overlaid.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from feather.data.loader import DataLoader
from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import MODEL_COLORS, OBS_COLOR
from feather.util.spatial import global_mean, latlon_global_mean
from feather.util.temporal import monthly_climatology

logger = logging.getLogger(__name__)


@register
class SeasonalCycleDiag(DiagnosticBase):
    """Monthly climatological cycle (Jan-Dec).

    For each variable, produces one figure with all models and obs
    overlaid on the same 12-month axes.
    """

    name = "seasonal_cycle"
    title = "Seasonal Cycle"
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
        """Compute monthly climatological cycle of global means.

        Returns
        -------
        dict
            Keyed by variable name.  Each entry contains 12-value
            monthly climatologies for every model and for observations.
        """
        results: dict[str, Any] = {}

        for var in self.variables:
            var_info = get_var(var)
            model_monthly: dict[str, Any] = {}

            for model in self.config.models:
                key = DataLoader.make_key(
                    self.experiment, model, var_info.domain,
                )
                model_data = self.model_loader.load_var(key, var)
                ds = self.model_loader.load(key)
                area = ds["area"] if "area" in ds else np.cos(
                    np.deg2rad(ds["latitude"])
                )

                # Global mean at each timestep → monthly climatology
                ts = global_mean(model_data, area)
                monthly = monthly_climatology(ts, self.period)
                model_monthly[model] = monthly

            # Observation
            obs_data = self.obs_loader.load_for_model_var(var, self.period)
            obs_ts = latlon_global_mean(obs_data)
            obs_monthly = monthly_climatology(obs_ts, self.period)

            results[var] = {
                "models": model_monthly,
                "obs": obs_monthly,
                "var_info": var_info,
            }

        return results

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot 12-month seasonal cycle per variable.

        Returns
        -------
        list of (Figure, metadata-dict)
        """
        figures: list[tuple[plt.Figure, dict]] = []
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]

        for var, vr in results.items():
            var_info = vr["var_info"]

            fig, ax = plt.subplots(figsize=(8, 5))
            months = np.arange(1, 13)

            for model, monthly in vr["models"].items():
                color = MODEL_COLORS.get(model)
                ax.plot(
                    months, monthly.values,
                    marker="o", label=model, color=color,
                )

            obs_monthly = vr["obs"]
            ax.plot(
                months, obs_monthly.values,
                marker="s", label="Obs", color=OBS_COLOR, linewidth=2,
            )

            ax.set_xticks(months)
            ax.set_xticklabels(month_labels)
            ax.set_title(f"{var_info.long_name} \u2014 Seasonal Cycle")
            ax.set_ylabel(f"{var_info.long_name} ({var_info.units})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            meta = self._build_metadata(
                title=f"{var_info.long_name} Seasonal Cycle",
                figure_id=f"{var}_seasonal_cycle",
                models=self.config.models,
                variables=[var],
                description=(
                    f"Monthly climatological cycle (Jan-Dec) of global mean "
                    f"{var_info.long_name} for all models vs observations."
                ),
                plot_type="seasonal_cycle",
                period=self.period,
            )
            figures.append((fig, meta))

        return figures
