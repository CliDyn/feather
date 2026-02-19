"""Global-mean time series diagnostic.

Plots area-weighted global mean time series for each variable, with all
configured models and observations overlaid on the same axes.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feather.data.loader import DataLoader
from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import MODEL_COLORS, OBS_COLOR
from feather.util.spatial import global_mean, latlon_global_mean

logger = logging.getLogger(__name__)


@register
class TimeseriesDiag(DiagnosticBase):
    """Global-mean time series (model vs observations).

    For each variable, produces one figure with all models and obs
    overlaid.
    """

    name = "timeseries"
    title = "Global Mean Time Series"
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
        """Compute global-mean time series for each variable.

        Returns
        -------
        dict
            Keyed by variable name.  Each entry contains model time
            series (DataArrays with ``time`` dim) and obs time series.
        """
        results: dict[str, Any] = {}

        for var in self.variables:
            var_info = get_var(var)
            model_ts: dict[str, Any] = {}

            for model in self.config.models:
                key = DataLoader.make_key(
                    self.experiment, model, var_info.domain,
                )
                model_data = self.model_loader.load_var(key, var)

                # Slice to period
                if self.period is not None and "time" in model_data.dims:
                    model_data = model_data.sel(
                        time=slice(self.period[0], self.period[1]),
                    )

                # HEALPix cells are equal area — simple mean is correct
                ts = global_mean(model_data).compute()
                model_ts[model] = ts

            # Observation time series
            obs_data = self.obs_loader.load_for_model_var(var, self.period)
            obs_ts = latlon_global_mean(obs_data)

            results[var] = {
                "models": model_ts,
                "obs": obs_ts,
                "var_info": var_info,
            }

        return results

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot global-mean time series per variable.

        Returns
        -------
        list of (Figure, metadata-dict)
        """
        figures: list[tuple[plt.Figure, dict]] = []

        for var, vr in results.items():
            var_info = vr["var_info"]

            fig, ax = plt.subplots(figsize=(12, 5))

            for model, ts in vr["models"].items():
                color = MODEL_COLORS.get(model)
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values, label=model, color=color)

            obs_ts = vr["obs"]
            obs_time = _to_plot_time(obs_ts.time.values)
            ax.plot(
                obs_time, obs_ts.values,
                label="Obs", color=OBS_COLOR, linewidth=2,
            )

            ax.set_title(f"{var_info.long_name} \u2014 Global Mean")
            ax.set_ylabel(f"{var_info.long_name} ({var_info.units})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            meta = self._build_metadata(
                title=f"{var_info.long_name} Global Mean Time Series",
                figure_id=f"{var}_timeseries",
                models=self.config.models,
                variables=[var],
                description=(
                    f"Area-weighted global mean time series of "
                    f"{var_info.long_name} for all models vs observations."
                ),
                plot_type="timeseries",
                period=self.period,
            )
            figures.append((fig, meta))

        return figures


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format.

    Handles cftime objects (which matplotlib cannot plot directly) by
    converting them to pandas Timestamps.
    """
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        # cftime objects → parse to pandas Timestamp via ISO string
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
