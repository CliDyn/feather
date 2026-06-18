"""Radiation budget diagnostic.

Computes derived radiation quantities (net TOA/surface radiation, CRE,
atmospheric absorption) and produces budget bar charts, Gregory plots,
radiation imbalance time series, and bias maps using CERES EBAF as the
primary observational reference.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import pandas as pd
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.lines import plot_budget_bars, plot_gregory
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import annual_mean, climatology

logger = logging.getLogger(__name__)

# Derived radiation quantity definitions
_DERIVED_QUANTITIES = {
    "toa_net": {
        "long_name": "TOA Net Radiation",
        "components": ("rst", "rlt"),
        "ceres_var": "toa_net_all_mon",
        "ceres_file": "toa",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
    "sfc_net": {
        "long_name": "Surface Net Radiation",
        "components": ("rss", "rls"),
        "ceres_var": "sfc_net_tot_all_mon",
        "ceres_file": "surface",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
    "toa_cre_sw": {
        "long_name": "TOA CRE Shortwave",
        "components": ("rst", "rstcs"),
        "operation": "subtract",
        "ceres_var": "toa_cre_sw_mon",
        "ceres_file": "toa",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
    "toa_cre_lw": {
        "long_name": "TOA CRE Longwave",
        "components": ("rlt", "rltcs"),
        "operation": "subtract",
        "ceres_var": "toa_cre_lw_mon",
        "ceres_file": "toa",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
    "sfc_net_sw": {
        "long_name": "Surface Net Shortwave",
        "components": ("rss",),
        "ceres_var": "sfc_net_sw_all_mon",
        "ceres_file": "surface",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
    "sfc_net_lw": {
        "long_name": "Surface Net Longwave",
        "components": ("rls",),
        "ceres_var": "sfc_net_lw_all_mon",
        "ceres_file": "surface",
        "units": "W/m\u00b2",
        "cmap": "coolwarm",
    },
}

# Budget bar chart component ordering
# (label, model_var, ceres_var, ceres_file, ceres_sign)
# ceres_sign: multiply CERES value by this to match DestinE convention
# (positive downward = into system). CERES OLR is positive upward → negate.
_BUDGET_COMPONENTS = [
    ("TOA SW", "rst", "toa_sw_all_mon", "toa", 1.0),
    ("TOA LW", "rlt", "toa_lw_all_mon", "toa", -1.0),  # CERES OLR positive up
    ("TOA Net", None, "toa_net_all_mon", "toa", 1.0),  # derived
    ("Sfc SW", "rss", "sfc_net_sw_all_mon", "surface", 1.0),
    ("Sfc LW", "rls", "sfc_net_lw_all_mon", "surface", 1.0),
    ("Sfc Net", None, "sfc_net_tot_all_mon", "surface", 1.0),  # derived
    ("Atm Abs", None, None, None, 1.0),  # TOA Net - Sfc Net
]

# Derivation rules for net radiation variables from CMOR component fluxes.
# CMOR stores component fluxes (rsdt, rsut, rlut, rsds, rsus, rlds, rlus)
# rather than net quantities (rst, rlt, rss, rls).  All derivations produce
# positive-downward values (DestinE convention).
_CMOR_NET_DERIVATIONS = {
    "rst":   (("rsdt", "rsut"),     lambda a, b: a - b),      # rsdt - rsut
    "rlt":   (("rlut",),            lambda a: -a),             # -rlut
    "rstcs": (("rsdt", "rsutcs"),   lambda a, b: a - b),
    "rltcs": (("rlutcs",),          lambda a: -a),
    "rss":   (("rsds", "rsus"),     lambda a, b: a - b),      # rsds - rsus
    "rls":   (("rlds", "rlus"),     lambda a, b: a - b),      # rlds - rlus
    "rsscs": (("rsdscs", "rsuscs"), lambda a, b: a - b),
    "rlscs": (("rldscs", "rluscs"), lambda a, b: a - b),
}

# Net radiation quantities the budget depends on, probed per model at the
# start of a run so that missing-component skips are reported explicitly.
_RADIATION_PROBE = ["rst", "rlt", "rss", "rls", "rstcs", "rltcs"]

# CMIP6 component formulas for derived radiation quantities.
# Each entry maps a derived-quantity key (matching _DERIVED_QUANTITIES)
# to the CMIP6 component variables and a formula to combine them.
_CMIP6_DERIVED = {
    "toa_net": {
        "vars": ["rsdt", "rsut", "rlut"],
        "formula": lambda r: r[0] - r[1] - r[2],
    },
    "sfc_net": {
        "vars": ["rsds", "rsus", "rlds", "rlus"],
        "formula": lambda r: (r[0] - r[1]) + (r[2] - r[3]),
    },
    "toa_cre_sw": {
        "vars": ["rsutcs", "rsut"],
        "formula": lambda r: r[0] - r[1],
    },
    "toa_cre_lw": {
        "vars": ["rlutcs", "rlut"],
        "formula": lambda r: r[0] - r[1],
    },
    "sfc_net_sw": {
        "vars": ["rsds", "rsus"],
        "formula": lambda r: r[0] - r[1],
    },
    "sfc_net_lw": {
        "vars": ["rlds", "rlus"],
        "formula": lambda r: r[0] - r[1],
    },
}


@register
class RadiationBudget(DiagnosticBase):
    """Radiation budget diagnostic.

    Produces 4 types of figures:
    A) Budget bar chart — radiation components for all models + obs
    B) Gregory plot — T2m vs net TOA radiation
    C) Radiation imbalance time series — net TOA over time
    D) Bias maps — derived radiation quantities vs CERES
    """

    name = "radiation_budget"
    title = "Radiation Budget"
    domain = "sfc"
    variables = [
        "rst", "rlt",
        "rstcs", "rltcs",
        "rss", "rls",
        "rsscs", "rlscs",
        "rsds", "rlds",
        "tas",
    ]
    group = "radiation"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, benchmarks=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks, save_netcdf=save_netcdf)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # ── CMOR net-radiation derivation ─────────────────────────────────

    def _load_model_radiation_var(
        self,
        model: str,
        var: str,
        *,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> "xr.DataArray":
        """Load a radiation variable, deriving from CMOR components if needed.

        Tries a direct load first (works for DestinE net variables).  On
        failure, if the data source is CMOR and the variable has a known
        derivation in ``_CMOR_NET_DERIVATIONS``, loads the component
        fluxes and computes the net quantity.

        Parameters
        ----------
        model, var, period, time_mean
            Same as ``_load_model_var``.

        Returns
        -------
        xr.DataArray

        Raises
        ------
        KeyError, FileNotFoundError
            When neither direct load nor derivation succeeds.
        """
        # 1) Try direct load (DestinE path, or CMOR var that exists directly)
        try:
            return self._load_model_var(
                model, var, period=period, time_mean=time_mean,
            )
        except (KeyError, FileNotFoundError):
            pass

        # 2) Attempt derivation from CMOR components
        if self.config.get_data_source_type() != "cmor":
            raise KeyError(f"{var} not available for {model}")

        if var not in _CMOR_NET_DERIVATIONS:
            raise FileNotFoundError(
                f"{var} not in CMOR derivation table for {model}"
            )

        component_names, formula = _CMOR_NET_DERIVATIONS[var]
        components = []
        for comp_var in component_names:
            # Let FileNotFoundError propagate if a component is missing
            da = self._load_model_var(
                model, comp_var, period=period, time_mean=time_mean,
            )
            components.append(da)

        result = formula(*components)
        result.name = var
        logger.debug(
            "Derived %s for %s from components %s",
            var, model, component_names,
        )
        return result

    # ── Orchestration (per-figure-group incremental) ──────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-figure-group: compute → plot → save."""
        logger.info("Running diagnostic: %s", self.name)
        self._log_radiation_availability()
        saved: list[tuple[Path, Path]] = []

        # Group A: Budget bar chart
        fid_bars = "radiation_budget_bars"
        if skip_existing and self._figure_exists(fid_bars):
            logger.info("Skipping budget bars — figure exists")
            saved.append((
                self.output_dir / f"{fid_bars}.png",
                self.output_dir / f"{fid_bars}.json",
            ))
        else:
            results_a = self._compute_budget()
            self._maybe_export_netcdf(results_a, "budget")
            figs_a = self._plot_budget(results_a)
            for fig, meta in figs_a:
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group B: Gregory plot
        fid_greg = "gregory_plot"
        if skip_existing and self._figure_exists(fid_greg):
            logger.info("Skipping Gregory plot — figure exists")
            saved.append((
                self.output_dir / f"{fid_greg}.png",
                self.output_dir / f"{fid_greg}.json",
            ))
        else:
            results_b = self._compute_gregory()
            self._maybe_export_netcdf(results_b, "gregory")
            figs_b = self._plot_gregory(results_b)
            for fig, meta in figs_b:
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group C: Radiation imbalance time series
        fid_imb = "radiation_imbalance_timeseries"
        if skip_existing and self._figure_exists(fid_imb):
            logger.info("Skipping imbalance time series — figure exists")
            saved.append((
                self.output_dir / f"{fid_imb}.png",
                self.output_dir / f"{fid_imb}.json",
            ))
        else:
            results_c = self._compute_imbalance_timeseries()
            self._maybe_export_netcdf(results_c, "imbalance")
            figs_c = self._plot_imbalance_timeseries(results_c)
            for fig, meta in figs_c:
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group D: Bias maps for derived quantities
        for dq_key, dq_info in _DERIVED_QUANTITIES.items():
            fid = f"{dq_key}_annual_bias"
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s bias map — figure exists", dq_key)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
                continue
            results_d = self._compute_bias_map(dq_key, dq_info)
            if results_d is not None:
                self._maybe_export_netcdf(results_d, f"{dq_key}_bias")
                figs_d = self._plot_bias_map(dq_key, dq_info, results_d)
                for fig, meta in figs_d:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    def _log_radiation_availability(self) -> None:
        """Log a per-model summary of available/missing radiation quantities.

        Probes each model for the net radiation quantities the budget needs
        (``rst``, ``rlt``, ``rss``, ``rls`` and the clear-sky TOA pair used
        for cloud radiative effect).  Each quantity is resolved exactly as
        the compute passes do — a direct load (DestinE net fields) falling
        back to CMOR component derivation — so the summary reflects what the
        diagnostic can actually produce.  Loads are lazy (metadata only) and
        cached by the underlying loader, so this adds no extra heavy IO.

        Models with no available quantities are excluded from the budget
        entirely; partial models appear with reduced panels.  This makes the
        otherwise-silent skips obvious in the logs.
        """
        logger.info("Radiation quantity availability per model:")
        for model in self.config.models:
            available: list[str] = []
            missing: list[str] = []
            for var in _RADIATION_PROBE:
                try:
                    self._load_model_radiation_var(model, var)
                    available.append(var)
                except (KeyError, FileNotFoundError):
                    missing.append(var)

            if not available:
                logger.warning(
                    "  %s: NO radiation quantities available — excluded "
                    "from radiation budget (missing: %s)",
                    model, ", ".join(missing),
                )
            elif missing:
                logger.warning(
                    "  %s: partial — available: %s; MISSING: %s",
                    model, ", ".join(available), ", ".join(missing),
                )
            else:
                logger.info(
                    "  %s: all radiation quantities available", model,
                )

    # ── Abstract interface (thin wrappers for backward compat) ────────

    def compute(self) -> dict[str, Any]:
        """Compute all radiation budget results."""
        return {
            "budget": self._compute_budget(),
            "gregory": self._compute_gregory(),
            "imbalance": self._compute_imbalance_timeseries(),
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figure types from precomputed results."""
        figures = []
        figures.extend(self._plot_budget(results["budget"]))
        figures.extend(self._plot_gregory(results["gregory"]))
        figures.extend(self._plot_imbalance_timeseries(results["imbalance"]))
        return figures

    # ── Group A: Budget bar chart ─────────────────────────────────────

    def _compute_budget(self) -> dict[str, Any]:
        """Compute global-mean climatology for each budget component."""
        logger.info("Computing radiation budget components...")
        model_budgets: dict[str, dict[str, float]] = {}

        for model in self.config.models:
            logger.info("  Loading model: %s", model)
            budget: dict[str, float] = {}

            for comp_name, model_var, _, _, _ in _BUDGET_COMPONENTS:
                if model_var is not None:
                    val = self._model_global_mean_clim(model, model_var)
                    if val is not None:
                        budget[comp_name] = val
                elif comp_name == "TOA Net":
                    sw = budget.get("TOA SW")
                    lw = budget.get("TOA LW")
                    if sw is not None and lw is not None:
                        budget["TOA Net"] = sw + lw
                elif comp_name == "Sfc Net":
                    sw = budget.get("Sfc SW")
                    lw = budget.get("Sfc LW")
                    if sw is not None and lw is not None:
                        budget["Sfc Net"] = sw + lw
                elif comp_name == "Atm Abs":
                    toa = budget.get("TOA Net")
                    sfc = budget.get("Sfc Net")
                    if toa is not None and sfc is not None:
                        budget["Atm Abs"] = toa - sfc

            if budget:
                model_budgets[model] = budget

        # Observations (CERES)
        obs_budget = self._compute_ceres_budget()

        # Benchmark MMMs (CMIP6, HighResMIP, …)
        benchmark_budgets: dict[str, dict[str, float]] = {}
        for bench in self.benchmarks:
            b = self._compute_cmip6_budget(loader=bench)
            if b:
                benchmark_budgets[getattr(bench, "label", "CMIP6 MMM")] = b
        primary = (
            next(iter(benchmark_budgets.values())) if benchmark_budgets else {}
        )

        return {
            "models": model_budgets,
            "obs": obs_budget,
            "cmip6": primary,
            "benchmarks": benchmark_budgets,
        }

    def _model_global_mean_clim(self, model: str, var: str) -> float | None:
        """Load model var, compute climatology, then global mean."""
        try:
            da = self._load_model_radiation_var(model, var)
        except (KeyError, FileNotFoundError):
            logger.warning("  %s not available for %s", var, model)
            return None
        clim = climatology(da, self.period).compute()
        return float(self._model_global_mean(clim, model).values)

    def _compute_ceres_budget(self) -> dict[str, float]:
        """Compute obs budget from CERES.

        Applies sign corrections where CERES convention differs from
        DestinE (positive downward).  CERES ``toa_lw_all_mon`` is OLR
        (positive upward) — negated to match DestinE net-down convention.
        CERES ``toa_sw_all_mon`` is *reflected* (upward) SW — net TOA SW
        is computed as ``solar_mon - toa_sw_all_mon``.
        """
        budget: dict[str, float] = {}
        try:
            for comp_name, _, ceres_var, ceres_file, ceres_sign in _BUDGET_COMPONENTS:
                if ceres_var is not None:
                    if comp_name == "TOA SW":
                        # CERES toa_sw_all_mon is reflected (upward) SW,
                        # not net.  Net TOA SW = solar - reflected.
                        solar = self.obs_loader.load_ceres(
                            "solar_mon", period=self.period, file_key="toa",
                        )
                        reflected = self.obs_loader.load_ceres(
                            "toa_sw_all_mon", period=self.period,
                            file_key="toa",
                        )
                        solar_clim = climatology(solar, self.period)
                        reflected_clim = climatology(reflected, self.period)
                        budget[comp_name] = float(
                            latlon_global_mean(
                                solar_clim - reflected_clim
                            ).values
                        )
                    else:
                        da = self.obs_loader.load_ceres(
                            ceres_var, period=self.period, file_key=ceres_file,
                        )
                        clim = climatology(da, self.period)
                        budget[comp_name] = float(
                            latlon_global_mean(clim).values
                        ) * ceres_sign
                elif comp_name == "Atm Abs":
                    toa = budget.get("TOA Net")
                    sfc = budget.get("Sfc Net")
                    if toa is not None and sfc is not None:
                        budget["Atm Abs"] = toa - sfc
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("CERES budget computation failed: %s", e)
        return budget

    def _compute_cmip6_budget(self, loader=None) -> dict[str, float]:
        """Compute benchmark MMM budget from individual flux components."""
        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return {}

        budget: dict[str, float] = {}
        # CMIP6 uses separate up/down components:
        # Net TOA = rsdt - rsut - rlut
        # Net Sfc SW = rsds - rsus, Net Sfc LW = rlds - rlus
        cmip6_map = {
            "rsdt": None, "rsut": None, "rlut": None,
            "rsds": None, "rsus": None, "rlds": None, "rlus": None,
        }

        for cmip6_var in cmip6_map:
            mmm, _ = loader.load_multi_model_mean(
                cmip6_var, period=self.period,
            )
            if mmm is not None:
                cmip6_map[cmip6_var] = float(
                    latlon_global_mean(mmm).values
                )

        if cmip6_map["rsdt"] is not None and cmip6_map["rsut"] is not None:
            budget["TOA SW"] = cmip6_map["rsdt"] - cmip6_map["rsut"]
        if cmip6_map["rlut"] is not None:
            budget["TOA LW"] = -cmip6_map["rlut"]  # Outgoing → net down
        if "TOA SW" in budget and "TOA LW" in budget:
            budget["TOA Net"] = budget["TOA SW"] + budget["TOA LW"]
        if cmip6_map["rsds"] is not None and cmip6_map["rsus"] is not None:
            budget["Sfc SW"] = cmip6_map["rsds"] - cmip6_map["rsus"]
        if cmip6_map["rlds"] is not None and cmip6_map["rlus"] is not None:
            budget["Sfc LW"] = cmip6_map["rlds"] - cmip6_map["rlus"]
        if "Sfc SW" in budget and "Sfc LW" in budget:
            budget["Sfc Net"] = budget["Sfc SW"] + budget["Sfc LW"]
        if "TOA Net" in budget and "Sfc Net" in budget:
            budget["Atm Abs"] = budget["TOA Net"] - budget["Sfc Net"]

        return budget

    def _plot_budget(
        self, results: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot budget bar chart with zoomed TOA Net inset panel."""
        budget_data: dict[str, dict[str, float]] = {}
        all_models = []

        comp_names = [c[0] for c in _BUDGET_COMPONENTS]
        for comp in comp_names:
            comp_vals: dict[str, float] = {}
            for model, mbud in results["models"].items():
                if comp in mbud:
                    comp_vals[model] = mbud[comp]
            if comp in results.get("obs", {}):
                comp_vals["CERES"] = results["obs"][comp]
            for b_label, b_budget in results.get("benchmarks", {}).items():
                if comp in b_budget:
                    comp_vals[b_label] = b_budget[comp]
            if comp_vals:
                budget_data[comp] = comp_vals

        all_models = list(results["models"].keys())
        if results.get("obs"):
            all_models.append("CERES")
        all_models.extend(results.get("benchmarks", {}).keys())

        # Two-panel figure: full budget (left) + TOA Net zoom (right)
        fig, (ax_main, ax_zoom) = plt.subplots(
            1, 2, figsize=(16, 6),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        # Left panel: full budget bars
        plot_budget_bars(budget_data, title="Global Mean Radiation Budget",
                         ax=ax_main)

        # Right panel: zoomed TOA Net
        toa_net_data = budget_data.get("TOA Net", {})
        if toa_net_data:
            from feather.plot.lines import _budget_bar_color
            sources = list(toa_net_data.keys())
            values = [toa_net_data[s] for s in sources]
            colors = [_budget_bar_color(s) for s in sources]
            x = np.arange(len(sources))
            ax_zoom.bar(x, values, color=colors, width=0.6)
            ax_zoom.set_xticks(x)
            ax_zoom.set_xticklabels(sources, rotation=30, ha="right",
                                    fontsize=9)
            ax_zoom.set_ylabel("W/m\u00b2")
            ax_zoom.set_title("TOA Net (zoom)")
            ax_zoom.grid(True, alpha=0.3, axis="y")
            ax_zoom.axhline(0, color="k", linewidth=0.5)
            # Add value annotations on bars
            for i, v in enumerate(values):
                ax_zoom.text(i, v + 0.05 * max(abs(min(values)), abs(max(values))),
                             f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        else:
            ax_zoom.set_title("TOA Net (zoom)")
            ax_zoom.text(0.5, 0.5, "No data", transform=ax_zoom.transAxes,
                         ha="center", va="center")

        plt.tight_layout()

        meta = self._build_metadata(
            title="Global Mean Radiation Budget",
            figure_id="radiation_budget_bars",
            models=all_models,
            description=(
                "Grouped bar chart of global-mean radiation budget components "
                "(TOA SW/LW/Net, Surface SW/LW/Net, Atmospheric Absorption) "
                "for DestinE models, CERES observations, and CMIP6 MMM. "
                "Right panel zooms in on TOA Net radiation (~1 W/m\u00b2) "
                "which is too small to distinguish in the full budget view."
            ),
            plot_type="budget_bars",
            period=self.period,
            obs_dataset="CERES_EBAF",
        )
        return [(fig, meta)]

    # ── Group B: Gregory plot ─────────────────────────────────────────

    def _compute_gregory(self) -> dict[str, Any]:
        """Compute T2m and net TOA time series for Gregory plot."""
        logger.info("Computing Gregory plot data...")
        model_data: dict[str, dict] = {}

        for model in self.config.models:
            t2m_ts = self._model_global_mean_ts(model, "tas")
            sw_ts = self._model_global_mean_ts(model, "rst")
            lw_ts = self._model_global_mean_ts(model, "rlt")

            if t2m_ts is not None and sw_ts is not None and lw_ts is not None:
                # Align times before adding
                t2m_ts, sw_ts, lw_ts = xr.align(t2m_ts, sw_ts, lw_ts, join="inner")
                toa_net_ts = sw_ts + lw_ts
                model_data[model] = {
                    "t2m_monthly": t2m_ts,
                    "toa_monthly": toa_net_ts,
                    "t2m_annual": annual_mean(t2m_ts),
                    "toa_annual": annual_mean(toa_net_ts),
                }

        # Observations: CERES for TOA, ERA5 for T2m
        obs_data = self._compute_gregory_obs()

        # CMIP6
        cmip6_data = self._compute_gregory_cmip6()

        return {
            "models": model_data,
            "obs": obs_data,
            "cmip6": cmip6_data,
        }

    def _model_global_mean_ts(self, model: str, var: str) -> xr.DataArray | None:
        """Load model variable and return global-mean monthly time series."""
        try:
            da = self._load_model_radiation_var(model, var, period=self.period)
        except (KeyError, FileNotFoundError):
            return None
        return self._model_global_mean(da, model).compute()

    def _compute_gregory_obs(self) -> dict[str, Any] | None:
        """Compute obs data for Gregory plot (CERES TOA + ERA5 T2m)."""
        try:
            # CERES net TOA
            toa_da = self.obs_loader.load_ceres(
                "toa_net_all_mon", period=self.period, file_key="toa",
            )
            toa_ts = latlon_global_mean(toa_da)

            # ERA5 T2m
            t2m_da = self.obs_loader.load_for_model_var("tas", self.period)
            t2m_ts = latlon_global_mean(t2m_da)

            # Align times
            t2m_ts, toa_ts = xr.align(t2m_ts, toa_ts, join="inner")
            if len(t2m_ts) == 0:
                return None

            return {
                "t2m_monthly": t2m_ts,
                "toa_monthly": toa_ts,
                "t2m_annual": annual_mean(t2m_ts),
                "toa_annual": annual_mean(toa_ts),
            }
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("Gregory obs computation failed: %s", e)
            return None

    def _compute_gregory_cmip6(self) -> dict[str, Any]:
        """Compute CMIP6 data for Gregory plot."""
        if not self.cmip6_enabled:
            return {}

        result: dict[str, Any] = {}

        # Individual models
        if self.cmip6_individual:
            individual: dict[str, dict] = {}
            for model in self.cmip6_loader.models:
                t2m = self.cmip6_loader.load_var(
                    "tas", model, table="Amon",
                    period=self.period, time_mean=False,
                )
                rsdt = self.cmip6_loader.load_var(
                    "rsdt", model, table="Amon",
                    period=self.period, time_mean=False,
                )
                rsut = self.cmip6_loader.load_var(
                    "rsut", model, table="Amon",
                    period=self.period, time_mean=False,
                )
                rlut = self.cmip6_loader.load_var(
                    "rlut", model, table="Amon",
                    period=self.period, time_mean=False,
                )
                if all(v is not None for v in [t2m, rsdt, rsut, rlut]):
                    area = self.cmip6_loader.load_area(model)
                    area = self._align_area(t2m, area)
                    t2m_ts = latlon_global_mean(t2m, area=area)
                    rsdt, rsut, rlut = xr.align(rsdt, rsut, rlut, join="inner")
                    toa_net = rsdt - rsut - rlut
                    toa_ts = latlon_global_mean(toa_net, area=area)
                    t2m_ts, toa_ts = xr.align(t2m_ts, toa_ts, join="inner")
                    if len(t2m_ts) > 0:
                        individual[model] = {
                            "t2m_monthly": t2m_ts,
                            "toa_monthly": toa_ts,
                        }
            result["individual"] = individual

        # MMM
        mmm_data = self._compute_gregory_cmip6_mmm()
        if mmm_data is not None:
            result["mmm"] = mmm_data

        # Per-benchmark MMM (CMIP6, HighResMIP, …) for extra regression lines
        from feather.plot.styles import benchmark_color
        benchmarks = []
        for i, bench in enumerate(self.benchmarks):
            b_mmm = self._compute_gregory_cmip6_mmm(loader=bench)
            if b_mmm is None:
                continue
            benchmarks.append({
                "label": getattr(bench, "label", "CMIP6 MMM"),
                "color": getattr(bench, "color", None) or benchmark_color(i),
                "mmm": b_mmm,
            })
        result["benchmarks"] = benchmarks

        return result

    def _compute_gregory_cmip6_mmm(self, loader=None) -> dict[str, Any] | None:
        """Compute benchmark MMM for Gregory plot (per-benchmark loader)."""
        loader = loader or self.cmip6_loader
        t2m_mmm, _ = self._cmip6_global_mean_timeseries(
            "tas", period=self.period, loader=loader,
        )
        if t2m_mmm is None:
            return None

        # Compute net TOA MMM from individual components
        member_toa = []
        for model in loader.models:
            rsdt = loader.load_var(
                "rsdt", model, table="Amon",
                period=self.period, time_mean=False,
            )
            rsut = loader.load_var(
                "rsut", model, table="Amon",
                period=self.period, time_mean=False,
            )
            rlut = loader.load_var(
                "rlut", model, table="Amon",
                period=self.period, time_mean=False,
            )
            if all(v is not None for v in [rsdt, rsut, rlut]):
                area = loader.load_area(model)
                area = self._align_area(rsdt, area)
                rsdt, rsut, rlut = xr.align(rsdt, rsut, rlut, join="inner")
                toa_net = rsdt - rsut - rlut
                toa_ts = latlon_global_mean(toa_net, area=area)
                member_toa.append(toa_ts)

        if not member_toa:
            return None

        aligned = xr.align(*member_toa, join="inner")
        toa_mmm = sum(aligned) / len(aligned)

        t2m_mmm, toa_mmm = xr.align(t2m_mmm, toa_mmm, join="inner")
        if len(t2m_mmm) == 0:
            return None

        return {
            "t2m_monthly": t2m_mmm,
            "toa_monthly": toa_mmm,
            "t2m_annual": annual_mean(t2m_mmm),
            "toa_annual": annual_mean(toa_mmm),
        }

    def _plot_gregory(
        self, results: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot Gregory scatter plot."""
        scatter_data = []
        all_models = []

        # CMIP6 individual (background layer)
        cmip6_data = results.get("cmip6", {})
        if cmip6_data.get("individual"):
            for i, (mname, mdata) in enumerate(cmip6_data["individual"].items()):
                scatter_data.append({
                    "label": "CMIP6 members" if i == 0 else "_nolegend_",
                    "t2m_monthly": mdata["t2m_monthly"].values,
                    "toa_monthly": mdata["toa_monthly"].values,
                    "color": CMIP6_COLOR,
                    "alpha": 0.2,
                    "show_regression": False,
                })
            all_models.extend(cmip6_data["individual"].keys())

        # Benchmark MMMs (middle layer) — CMIP6, HighResMIP, …
        for bench in cmip6_data.get("benchmarks", []):
            mmm = bench["mmm"]
            scatter_data.append({
                "label": bench["label"],
                "t2m_monthly": mmm["t2m_monthly"].values,
                "toa_monthly": mmm["toa_monthly"].values,
                "t2m_annual": mmm.get("t2m_annual", mmm["t2m_monthly"]).values,
                "toa_annual": mmm.get("toa_annual", mmm["toa_monthly"]).values,
                "color": bench["color"],
                "alpha": 0.3,
                "linestyle": "--",
                "show_regression": True,
            })
            all_models.append(bench["label"])

        # DestinE models (foreground)
        for model, mdata in results["models"].items():
            color = self.config.get_model_color(model)
            scatter_data.append({
                "label": model,
                "t2m_monthly": mdata["t2m_monthly"].values,
                "toa_monthly": mdata["toa_monthly"].values,
                "t2m_annual": mdata["t2m_annual"].values,
                "toa_annual": mdata["toa_annual"].values,
                "color": color,
                "alpha": 0.4,
                "show_regression": True,
            })
            all_models.append(model)

        # Observations (top layer)
        if results.get("obs") is not None:
            obs = results["obs"]
            scatter_data.append({
                "label": "CERES + ERA5",
                "t2m_monthly": obs["t2m_monthly"].values,
                "toa_monthly": obs["toa_monthly"].values,
                "t2m_annual": obs["t2m_annual"].values,
                "toa_annual": obs["toa_annual"].values,
                "color": OBS_COLOR,
                "alpha": 0.5,
                "show_regression": True,
            })
            all_models.append("CERES + ERA5")

        fig, ax = plot_gregory(
            scatter_data,
            title="Gregory Plot \u2014 Global Mean T2m vs Net TOA Radiation",
        )

        meta = self._build_metadata(
            title="Gregory Plot",
            figure_id="gregory_plot",
            models=all_models,
            variables=["tas", "rst", "rlt"],
            description=(
                "Scatter plot of global-mean 2m temperature vs net TOA "
                "radiation. Monthly values as small dots, annual means as "
                "diamonds. Regression slopes indicate climate feedback "
                "parameter (W/m\u00b2/K)."
            ),
            computation_notes=(
                "Net TOA = TOA net SW + TOA net LW. "
                "Obs uses CERES EBAF for TOA + ERA5 for T2m."
            ),
            plot_type="gregory",
            period=self.period,
            obs_dataset="CERES_EBAF + ERA5",
        )
        return [(fig, meta)]

    # ── Group C: Radiation imbalance time series ──────────────────────

    def _compute_imbalance_timeseries(self) -> dict[str, Any]:
        """Compute net TOA radiation time series for all sources."""
        logger.info("Computing radiation imbalance time series...")
        model_ts: dict[str, xr.DataArray] = {}

        for model in self.config.models:
            sw_ts = self._model_global_mean_ts(model, "rst")
            lw_ts = self._model_global_mean_ts(model, "rlt")
            if sw_ts is not None and lw_ts is not None:
                sw_ts, lw_ts = xr.align(sw_ts, lw_ts, join="inner")
                model_ts[model] = sw_ts + lw_ts

        # CERES obs
        obs_ts = None
        try:
            toa_da = self.obs_loader.load_ceres(
                "toa_net_all_mon", period=self.period, file_key="toa",
            )
            obs_ts = latlon_global_mean(toa_da)
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("CERES imbalance time series failed: %s", e)

        # Benchmark MMMs (CMIP6, HighResMIP, …)
        from feather.plot.styles import benchmark_color
        benchmarks = []
        if self.cmip6_enabled:
            for i, bench in enumerate(self.benchmarks):
                b_ts, b_info = self._compute_cmip6_net_toa_timeseries(
                    loader=bench,
                )
                if b_ts is None:
                    continue
                benchmarks.append({
                    "label": getattr(bench, "label", "CMIP6 MMM"),
                    "color": getattr(bench, "color", None) or benchmark_color(i),
                    "ts": b_ts,
                    "info": b_info,
                })
        primary = benchmarks[0] if benchmarks else None

        return {
            "models": model_ts,
            "obs": obs_ts,
            "benchmarks": benchmarks,
            "cmip6_ts": primary["ts"] if primary else None,
            "cmip6_info": primary["info"] if primary else {},
        }

    def _compute_cmip6_net_toa_timeseries(self, loader=None):
        """Compute benchmark MMM net TOA time series (per-benchmark loader)."""
        loader = loader or self.cmip6_loader
        member_toa = []
        models_used = []

        for model in loader.models:
            rsdt = loader.load_var(
                "rsdt", model, table="Amon",
                period=self.period, time_mean=False,
            )
            rsut = loader.load_var(
                "rsut", model, table="Amon",
                period=self.period, time_mean=False,
            )
            rlut = loader.load_var(
                "rlut", model, table="Amon",
                period=self.period, time_mean=False,
            )
            if all(v is not None for v in [rsdt, rsut, rlut]):
                area = loader.load_area(model)
                area = self._align_area(rsdt, area)
                rsdt, rsut, rlut = xr.align(rsdt, rsut, rlut, join="inner")
                toa_net = rsdt - rsut - rlut
                toa_ts = latlon_global_mean(toa_net, area=area)
                member_toa.append(toa_ts)
                models_used.append(model)

        if not member_toa:
            return None, {}

        aligned = xr.align(*member_toa, join="inner")
        mmm = sum(aligned) / len(aligned)
        return mmm, {"n_members": len(models_used), "models_used": models_used}

    def _plot_imbalance_timeseries(
        self, results: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot radiation imbalance time series.

        Monthly data is plotted as semi-transparent background lines;
        annual means as thicker foreground lines.  The x-axis is set
        to the model time range; each series (obs, CMIP6) starts
        from whenever its data begins within that range.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        all_models = []

        # Determine model time range for x-axis limits
        model_start = model_end = None
        for ts in results["models"].values():
            t0, t1 = ts.time.values[0], ts.time.values[-1]
            if model_start is None or t0 < model_start:
                model_start = t0
            if model_end is None or t1 > model_end:
                model_end = t1

        # --- Layer 1: monthly (background, washed-out) ---

        # Benchmark MMM monthly (CMIP6, HighResMIP, …)
        for bench in results.get("benchmarks", []):
            time_vals = _to_plot_time(bench["ts"].time.values)
            ax.plot(time_vals, bench["ts"].values,
                    color=bench["color"], alpha=0.3, linewidth=0.7,
                    linestyle="--")

        # DestinE model monthly
        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values,
                    color=color, alpha=0.3, linewidth=0.7)

        # Obs monthly
        if results.get("obs") is not None:
            obs_ts = results["obs"]
            time_vals = _to_plot_time(obs_ts.time.values)
            ax.plot(time_vals, obs_ts.values,
                    color=OBS_COLOR, alpha=0.3, linewidth=0.7)

        # --- Layer 2: annual means (foreground, thick) ---

        # Benchmark MMM annual (CMIP6, HighResMIP, …)
        for bench in results.get("benchmarks", []):
            b_annual = annual_mean(bench["ts"])
            time_vals = _to_plot_time(b_annual.time.values)
            ax.plot(time_vals, b_annual.values,
                    label=bench["label"], color=bench["color"],
                    linewidth=2.0, linestyle="--")

        # DestinE model annual
        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values,
                    label=model, color=color, linewidth=2.0)
            all_models.append(model)

        # Obs annual
        if results.get("obs") is not None:
            obs_annual = annual_mean(results["obs"])
            time_vals = _to_plot_time(obs_annual.time.values)
            ax.plot(time_vals, obs_annual.values,
                    label="CERES", color=OBS_COLOR, linewidth=2.5)

        # Set x-axis to model time range
        if model_start is not None:
            ax.set_xlim(
                _to_plot_time(np.array([model_start]))[0],
                _to_plot_time(np.array([model_end]))[0],
            )

        ax.set_title("Net TOA Radiation (Earth's Energy Imbalance)")
        ax.set_ylabel("Net TOA Radiation (W/m\u00b2)")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append("CERES")
        all_models.extend(b["label"] for b in results.get("benchmarks", []))

        meta = self._build_metadata(
            title="Radiation Imbalance Time Series",
            figure_id="radiation_imbalance_timeseries",
            models=all_models,
            variables=["rst", "rlt"],
            description=(
                "Time series of global-mean net TOA radiation for DestinE "
                "models, CERES observations, and CMIP6 MMM. Monthly values "
                "shown as semi-transparent lines, annual means as thick "
                "lines. Positive values indicate net energy gain (warming)."
            ),
            plot_type="timeseries",
            period=self.period,
            obs_dataset="CERES_EBAF",
        )
        return [(fig, meta)]

    # ── Group D: Bias maps ────────────────────────────────────────────

    def _compute_bias_map(
        self, dq_key: str, dq_info: dict,
    ) -> dict[str, Any] | None:
        """Compute bias map data for a derived radiation quantity."""
        from feather.util.spatial import compute_latlon_areas

        logger.info("Computing bias map for %s...", dq_info["long_name"])
        influence_radius = self.config.nereus.get("influence_radius", 80_000.0)

        # Load CERES observation
        try:
            ceres_da = self.obs_loader.load_ceres(
                dq_info["ceres_var"],
                period=self.period,
                file_key=dq_info["ceres_file"],
            )
            obs_clim = climatology(ceres_da, self.period)
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("CERES data for %s not available: %s", dq_key, e)
            return None

        lat_name = "lat" if "lat" in obs_clim.coords else "latitude"
        lon_name = "lon" if "lon" in obs_clim.coords else "longitude"
        obs_lats = obs_clim[lat_name].values
        obs_lons = obs_clim[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Cache nereus interpolator per source grid size
        _interp_cache: dict[int, Any] = {}
        obs_clim_common = None
        common_area = None
        target_lats = None
        target_lons = None
        model_results: dict[str, dict] = {}

        for model in self.config.models:
            model_clim = self._compute_model_derived(model, dq_key, dq_info)
            if model_clim is None:
                continue

            # Get coords for regridding — use first component variable.
            # For CMOR sources, net vars (rst, rlt, ...) don't exist as
            # files, so resolve to the first CMOR component for coords.
            first_var = dq_info["components"][0]
            coord_var = first_var
            if (self.config.get_data_source_type() == "cmor"
                    and first_var in _CMOR_NET_DERIVATIONS):
                coord_var = _CMOR_NET_DERIVATIONS[first_var][0][0]
            lon, lat = self._load_model_coords(model, coord_var)

            # For latlon grids, meshgrid 1D coord arrays to per-pixel arrays
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _interp_cache:
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

                    obs_clim_common = obs_clim.interp(
                        {lat_name: target_lats, lon_name: target_lons}
                    )
                    if lat_name != "lat":
                        obs_clim_common = obs_clim_common.rename(
                            {lat_name: "lat", lon_name: "lon"}
                        )
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )
            else:
                interp = _interp_cache[n_src]
                regridded_np = interp(model_clim.values.ravel())
                annual_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            bias = annual_regrid - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(bias, area=common_area).values
            )
            rmse = float(np.sqrt(
                latlon_global_mean(bias ** 2, area=common_area).values
            ))
            model_results[model] = {
                "bias": bias,
                "bias_gmean": bias_gmean,
                "rmse": rmse,
            }

        if not model_results:
            return None

        # Benchmark MMM biases (CMIP6, HighResMIP, …)
        cmip6_bias_data = {}
        if self.cmip6_enabled and target_lats is not None:
            cmip6_ir = self.config.nereus.get("influence_radius", 80_000.0)
            for bench in self.benchmarks:
                b_mmm = self._compute_cmip6_mmm_regridded(
                    dq_key, target_lats, target_lons, obs_res, cmip6_ir,
                    loader=bench,
                )
                if b_mmm is None:
                    continue
                b_bias = b_mmm - obs_clim_common
                cmip6_bias_data[getattr(bench, "label", "CMIP6 MMM")] = {
                    "bias": b_bias,
                    "bias_gmean": float(
                        latlon_global_mean(b_bias, area=common_area).values
                    ),
                }

        return {
            "models": model_results,
            "obs_clim": obs_clim_common,
            "cmip6_bias": cmip6_bias_data,
        }

    def _compute_model_derived(
        self, model: str, dq_key: str, dq_info: dict,
    ) -> xr.DataArray | None:
        """Compute a derived quantity climatology for a model on its native grid."""
        components = dq_info["components"]
        operation = dq_info.get("operation", "add")

        arrays = []
        for var in components:
            try:
                da = self._load_model_radiation_var(model, var)
            except (KeyError, FileNotFoundError):
                return None
            arrays.append(climatology(da, self.period).compute())

        if len(arrays) == 1:
            return arrays[0]
        elif operation == "subtract":
            return arrays[0] - arrays[1]
        else:
            return sum(arrays)

    @staticmethod
    def _regrid_to_target(da, target_lats, target_lons,
                          resolution, influence_radius,
                          interp_cache, method="nearest"):
        """Regrid a regular lat/lon DataArray to the target grid via nereus.

        Uses *interp_cache* (keyed by grid shape) to avoid rebuilding
        the KDTree for models that share the same native grid.

        Source longitudes are converted to -180..180 and the target grid
        uses ``lon_bounds=(-180, 180)`` so that Delaunay triangulation
        (used by ``method="linear"`` / ``"cubic"``) does not produce a
        NaN stripe at the prime meridian.  The output columns are rolled
        back to 0..360 to match *target_lons*.
        """
        ir = max(influence_radius, 250_000.0)

        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lat_arr = da[lat_name].values
        lon_arr = da[lon_name].values

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

        n_roll = regridded.shape[1] // 2
        regridded = np.roll(regridded, -n_roll, axis=1)

        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    def _compute_cmip6_mmm_regridded(
        self, dq_key: str, target_lats, target_lons, obs_res,
        cmip6_influence_radius, loader=None,
    ) -> xr.DataArray | None:
        """Compute CMIP6 MMM of a derived quantity, regridded to target grid.

        For each CMIP6 (model, variant) pair:
        1. Load all component variables
        2. Derive the quantity on the CMIP6 native grid
        3. Regrid to target grid using ``_regrid_to_target``
        Then average the regridded fields to form the MMM.

        Returns the regridded MMM DataArray, or None if no data.
        """
        if dq_key not in _CMIP6_DERIVED:
            return None

        loader = loader or self.cmip6_loader
        derived_info = _CMIP6_DERIVED[dq_key]
        component_vars = derived_info["vars"]
        formula = derived_info["formula"]

        member_pairs = loader.get_member_pairs()
        cmip6_interp_cache: dict[tuple, "nr.RegridInterpolator"] = {}
        regridded_fields = []

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            components = []
            all_ok = True

            for cvar in component_vars:
                da = loader.load_var(
                    cvar, model, variant=variant,
                    period=self.period, time_mean=True,
                )
                if da is None:
                    all_ok = False
                    break
                components.append(da)

            if not all_ok:
                logger.debug(
                    "  Skipping CMIP6 %s for %s — missing components", label, dq_key,
                )
                continue

            derived = formula(components)

            regridded = self._regrid_to_target(
                derived, target_lats, target_lons,
                obs_res, cmip6_influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            regridded_fields.append(regridded)

        if not regridded_fields:
            return None

        mmm = xr.concat(regridded_fields, dim="member").mean("member")
        return mmm

    def _plot_bias_map(
        self, dq_key: str, dq_info: dict, results: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot combined bias map for a derived quantity."""
        from feather.plot.maps import plot_combined_bias_map

        obs_clim = results["obs_clim"]
        bias_dict = {}
        summary_stats = {}
        all_models = []

        for model, mdata in results["models"].items():
            bias_dict[model] = mdata["bias"]
            summary_stats[model] = {
                "global_mean_bias": mdata["bias_gmean"],
                "rmse": mdata.get("rmse"),
            }
            all_models.append(model)

        for label, cdata in results.get("cmip6_bias", {}).items():
            bias_dict[label] = cdata["bias"]
            summary_stats[label] = {
                "global_mean_bias": cdata["bias_gmean"],
            }
            all_models.append(label)

        fig, axes = plot_combined_bias_map(
            obs_clim, bias_dict,
            title=f"{dq_info['long_name']} Annual Mean",
            obs_title="CERES",
            cmap=dq_info.get("cmap", "coolwarm"),
            bias_cmap="RdBu_r",
            units=dq_info.get("units", "W/m\u00b2"),
            method=self._regrid_method,
        )

        meta = self._build_metadata(
            title=f"{dq_info['long_name']} Annual Bias",
            figure_id=f"{dq_key}_annual_bias",
            models=all_models,
            description=(
                f"Annual mean {dq_info['long_name']} climatology and model "
                f"biases relative to CERES EBAF observations."
            ),
            plot_type="combined_bias_map",
            period=self.period,
            obs_dataset="CERES_EBAF",
            summary_statistics=summary_stats,
        )
        return [(fig, meta)]


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
