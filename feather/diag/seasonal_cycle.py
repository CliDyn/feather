"""Seasonal cycle diagnostic.

Plots the 12-month climatological cycle (Jan-Dec) of global-mean values
for each variable, with all configured models and observations overlaid.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.spatial import latlon_global_mean
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
            figure_id = f"{var}_seasonal_cycle"
            if skip_existing and self._figure_exists(figure_id):
                logger.info("Skipping %s — figure exists", var)
                saved.append((
                    self.output_dir / f"{figure_id}.png",
                    self.output_dir / f"{figure_id}.json",
                ))
                continue

            try:
                vr = self._compute_single(var)
                if vr is None:
                    continue
                figures = self._plot_single(var, vr)
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
        """Compute monthly climatological cycle for a single variable."""
        var_info = get_var(var)
        logger.info("Processing variable: %s (%s)", var, var_info.long_name)
        model_monthly: dict[str, Any] = {}

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
            monthly = monthly_climatology(ts, self.period)
            model_monthly[model] = monthly

        if not model_monthly:
            logger.warning("  No model data for %s — skipping variable", var)
            return None

        logger.info("  Loading observations for %s", var)
        obs_data = self._load_obs_var(var, self.period)
        obs_ts = latlon_global_mean(obs_data)
        obs_monthly = monthly_climatology(obs_ts, self.period)

        cmip6_monthly = None
        cmip6_info = {}
        cmip6_individual_monthly: dict[str, Any] = {}
        cmip6_ts, info = self._cmip6_global_mean_timeseries(
            var, period=self.period,
            return_individual=self.cmip6_individual,
        )
        if cmip6_ts is not None:
            cmip6_monthly = monthly_climatology(cmip6_ts)
            cmip6_info = info

            if self.cmip6_individual and "individual_series" in info:
                for mname, mts in info["individual_series"].items():
                    cmip6_individual_monthly[mname] = (
                        monthly_climatology(mts)
                    )

        return {
            "models": model_monthly,
            "obs": obs_monthly,
            "var_info": var_info,
            "cmip6_monthly": cmip6_monthly,
            "cmip6_info": cmip6_info,
            "cmip6_individual_monthly": cmip6_individual_monthly,
        }

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
            vr = self._compute_single(var)
            if vr is not None:
                results[var] = vr
        return results

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot 12-month seasonal cycle per variable.

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
        """Plot 12-month seasonal cycle for a single variable."""
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]

        var_info = vr["var_info"]

        fig, ax = plt.subplots(figsize=(8, 5))
        months = np.arange(1, 13)
        all_models = list(self.config.models)

        # Layer 1: Individual CMIP6 model lines (background)
        cmip6_indiv = vr.get("cmip6_individual_monthly", {})
        for i, (mname, monthly) in enumerate(cmip6_indiv.items()):
            label = "CMIP6 members" if i == 0 else "_nolegend_"
            ax.plot(
                months, monthly.values,
                color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                label=label,
            )
        if cmip6_indiv:
            all_models.extend(cmip6_indiv.keys())

        # Layer 2: CMIP6 MMM line (middle)
        if vr.get("cmip6_monthly") is not None:
            ax.plot(
                months, vr["cmip6_monthly"].values,
                marker="d", label="CMIP6 MMM", color=CMIP6_COLOR,
                linewidth=1.5, linestyle="--",
            )

        # Layer 3: DestinE model lines (foreground)
        for model, monthly in vr["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(
                months, monthly.values,
                marker="o", label=model, color=color,
            )

        # Layer 4: Observations (top)
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
            models=all_models,
            variables=[var],
            description=(
                f"Monthly climatological cycle (Jan-Dec) of global mean "
                f"{var_info.long_name} for all models vs observations."
            ),
            plot_type="seasonal_cycle",
            period=self.period,
            cmip6_info=vr.get("cmip6_info") or None,
        )
        return [(fig, meta)]
