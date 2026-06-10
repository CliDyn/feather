"""Global-mean time series diagnostic.

Plots area-weighted global mean time series for each variable, with all
configured models and observations overlaid on the same axes.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import CMIP6_COLOR, ENS_COLOR, OBS_COLOR
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import annual_mean

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
        self._project_name = self.config.project.get("name", "Ensemble")

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

            try:
                results = self._compute_single(var)
                if results is None:
                    continue
                figures = self._plot_single(var, results)
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

    # ── Computation ────────────────────────────────────────────────────

    def _compute_single(self, var: str) -> dict[str, Any]:
        """Compute global-mean time series for a single variable."""
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_ts: dict[str, Any] = {}

        for model in self.config.models:
            logger.info("  Loading model data: %s", model)
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

            ts = self._model_global_mean(model_data, model).compute()
            model_ts[model] = ts
            logger.info("    Global mean: %.2f %s", float(ts.mean()), var_info.units)

        if not model_ts:
            logger.warning("  No model data for %s — skipping variable", var)
            return None

        logger.info("  Loading observations for %s", var)
        obs_data = self._load_obs_var(var, self.period)
        obs_ts = latlon_global_mean(obs_data)
        logger.info("    Obs global mean: %.2f %s", float(obs_ts.mean()), var_info.units)

        cmip6_ts = None
        cmip6_info = {}
        cmip6_individual_ts: dict[str, Any] = {}
        cmip6_ts, info = self._cmip6_global_mean_timeseries(
            var, period=self.period,
            return_individual=self.cmip6_individual,
        )
        if cmip6_ts is not None:
            cmip6_info = info
            if self.cmip6_individual and "individual_series" in info:
                cmip6_individual_ts = dict(info["individual_series"])

        ens_mean, ens_median = self._compute_ensemble_stats(model_ts)

        return {
            "models": model_ts,
            "obs": obs_ts,
            "var_info": var_info,
            "cmip6_ts": cmip6_ts,
            "cmip6_info": cmip6_info,
            "cmip6_individual_ts": cmip6_individual_ts,
            "ens_mean": ens_mean,
            "ens_median": ens_median,
        }

    @staticmethod
    def _compute_ensemble_stats(
        model_ts: dict,
    ) -> tuple["xr.DataArray | None", "xr.DataArray | None"]:
        """Compute ensemble mean and median across available model time series.

        Uses the inner time union so models with different lengths are
        aligned to their common period before averaging.

        Parameters
        ----------
        model_ts : dict
            Mapping of model name → DataArray with a ``time`` dimension.

        Returns
        -------
        ens_mean, ens_median : DataArray or None
            Returns ``None`` for both when fewer than 2 models are present.
        """
        import xarray as xr

        series = list(model_ts.values())
        if len(series) < 2:
            return None, None

        aligned = xr.align(*series, join="inner")
        stacked = xr.concat(list(aligned), dim="member")
        return stacked.mean("member"), stacked.median("member")

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
            vr = self._compute_single(var)
            if vr is not None:
                results[var] = vr
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
        """Plot global-mean time series for a single variable.

        Monthly data is plotted as semi-transparent background lines;
        annual means as thicker foreground lines.
        """
        var_info = vr["var_info"]
        _off = var_info.display_offset          # e.g. -273.15 for K\u2192\u00b0C, 0 otherwise
        _disp_units = var_info.display_units or var_info.units
        all_models = list(self.config.models)

        fig, ax = plt.subplots(figsize=(12, 5))

        cmip6_indiv = vr.get("cmip6_individual_ts", {})
        if cmip6_indiv:
            all_models.extend(cmip6_indiv.keys())

        # --- Monthly pass (background, washed-out) ---

        # CMIP6 individual monthly
        for _mname, ts in cmip6_indiv.items():
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values + _off,
                    color=CMIP6_COLOR, alpha=0.2, linewidth=0.5)

        # CMIP6 MMM monthly
        if vr.get("cmip6_ts") is not None:
            cmip6_ts = vr["cmip6_ts"]
            time_vals = _to_plot_time(cmip6_ts.time.values)
            ax.plot(time_vals, cmip6_ts.values + _off,
                    color=CMIP6_COLOR, alpha=0.3, linewidth=0.7,
                    linestyle="--")

        # DestinE model monthly
        for model, ts in vr["models"].items():
            color = self.config.get_model_color(model)
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values + _off,
                    color=color, alpha=0.3, linewidth=0.7)

        # Obs monthly
        obs_ts = vr["obs"]
        obs_time = _to_plot_time(obs_ts.time.values)
        ax.plot(obs_time, obs_ts.values + _off,
                color=OBS_COLOR, alpha=0.3, linewidth=0.7)

        # --- Annual pass (foreground, thick with labels) ---

        # Pre-compute member counts for legend labels
        n_eerie = len(vr["models"])
        n_cmip6_mmm = vr.get("cmip6_info", {}).get("n_members", 0)
        n_cmip6_indiv = len(cmip6_indiv)

        # CMIP6 individual annual
        for i, (mname, ts) in enumerate(cmip6_indiv.items()):
            label = (f"CMIP6 members ({n_cmip6_indiv})"
                     if i == 0 else "_nolegend_")
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values + _off,
                    color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                    label=label)

        # CMIP6 MMM annual
        if vr.get("cmip6_ts") is not None:
            cmip6_annual = annual_mean(vr["cmip6_ts"])
            time_vals = _to_plot_time(cmip6_annual.time.values)
            ax.plot(time_vals, cmip6_annual.values + _off,
                    label=f"CMIP6 MMM ({n_cmip6_mmm})", color=CMIP6_COLOR,
                    linewidth=2.0, linestyle="--")

        # DestinE model annual
        for model, ts in vr["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values + _off,
                    label=model, color=color, linewidth=2.0)

        # Ensemble median annual (dashed)
        if vr.get("ens_median") is not None:
            ens_med_annual = annual_mean(vr["ens_median"])
            time_vals = _to_plot_time(ens_med_annual.time.values)
            ax.plot(time_vals, ens_med_annual.values + _off,
                    label=f"{self._project_name} ensemble median ({n_eerie})",
                    color=ENS_COLOR, linewidth=2.5, linestyle="--")

        # Ensemble mean annual (solid)
        if vr.get("ens_mean") is not None:
            ens_mean_annual = annual_mean(vr["ens_mean"])
            time_vals = _to_plot_time(ens_mean_annual.time.values)
            ax.plot(time_vals, ens_mean_annual.values + _off,
                    label=f"{self._project_name} ensemble mean ({n_eerie})",
                    color=ENS_COLOR, linewidth=2.5)

        # Obs annual
        obs_annual = annual_mean(obs_ts)
        obs_annual_time = _to_plot_time(obs_annual.time.values)
        ax.plot(obs_annual_time, obs_annual.values + _off,
                label=var_info.obs_dataset, color=OBS_COLOR, linewidth=2.5)

        ax.set_title(f"{var_info.long_name} \u2014 Global Mean")
        ax.set_ylabel(f"{var_info.long_name} ({_disp_units})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{var_info.long_name} Global Mean Time Series",
            figure_id=f"{var}_timeseries",
            models=all_models,
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
