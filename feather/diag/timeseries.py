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
from feather.plot.styles import CMIP6_COLOR, MODEL_COLORS, OBS_COLOR
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

    # ── Orchestration (per-variable incremental) ─────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple["Path", "Path"]]:
        """Execute per-variable: compute → plot → save immediately.

        Parameters
        ----------
        skip_existing : bool
            When True, skip variables whose output figure already
            exists on disk.
        """
        from pathlib import Path

        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        for var in self.variables:
            figure_id = f"{var}_timeseries"
            if skip_existing and self._figure_exists(figure_id):
                logger.info("Skipping %s — figure exists", var)
                saved.append((
                    self.output_dir / f"{figure_id}.png",
                    self.output_dir / f"{figure_id}.json",
                ))
                continue

            results = self._compute_single(var)
            figures = self._plot_single(var, results)
            for fig, meta in figures:
                paths = self._save(fig, meta, meta["figure_id"])
                saved.append(paths)

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Computation ────────────────────────────────────────────────────

    def _compute_single(self, var: str) -> dict[str, Any]:
        """Compute global-mean time series for a single variable."""
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_ts: dict[str, Any] = {}

        for model in self.config.models:
            logger.info("  Loading model data: %s", model)
            key = DataLoader.make_key(
                self.experiment, model, var_info.domain,
            )
            model_data = self.model_loader.load_var(key, var)

            if self.period is not None and "time" in model_data.dims:
                model_data = model_data.sel(
                    time=slice(self.period[0], self.period[1]),
                )

            ts = global_mean(model_data).compute()
            model_ts[model] = ts
            logger.info("    Global mean: %.2f %s", float(ts.mean()), var_info.units)

        logger.info("  Loading observations for %s", var)
        obs_data = self.obs_loader.load_for_model_var(var, self.period)
        obs_ts = latlon_global_mean(obs_data)
        logger.info("    Obs global mean: %.2f %s", float(obs_ts.mean()), var_info.units)

        cmip6_ts, cmip6_info = self._cmip6_global_mean_timeseries(
            var, period=self.period,
        )

        return {
            "models": model_ts,
            "obs": obs_ts,
            "var_info": var_info,
            "cmip6_ts": cmip6_ts,
            "cmip6_info": cmip6_info,
        }

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
            results[var] = self._compute_single(var)
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
            figures.extend(self._plot_single(var, vr))
        return figures

    def _plot_single(
        self, var: str, vr: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot global-mean time series for a single variable."""
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

        # CMIP6 MMM line (optional)
        if vr.get("cmip6_ts") is not None:
            cmip6_ts = vr["cmip6_ts"]
            cmip6_time = _to_plot_time(cmip6_ts.time.values)
            ax.plot(
                cmip6_time, cmip6_ts.values,
                label="CMIP6 MMM", color=CMIP6_COLOR,
                linewidth=1.5, linestyle="--",
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
            cmip6_info=vr.get("cmip6_info") or None,
        )
        return [(fig, meta)]


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
