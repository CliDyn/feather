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
from feather.diag._ts_panel import build_envelope_timeseries
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import (
    ENS_COLOR,
    OBS_COLOR,
    benchmark_color as _benchmark_color,
)
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import annual_mean, normalize_monthly_time

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
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks, save_netcdf=save_netcdf)
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
            figure_ids = [
                f"{var}_timeseries",
                f"{var}_timeseries_envelope",
                f"{var}_timeseries_anomaly",
            ]
            if skip_existing and all(
                self._figure_exists(fid) for fid in figure_ids
            ):
                logger.info("Skipping %s — figures exist", var)
                saved.extend(
                    (self.output_dir / f"{fid}.png",
                     self.output_dir / f"{fid}.json")
                    for fid in figure_ids
                )
                continue

            try:
                results = self._compute_single(var)
                if results is None:
                    continue
                self._maybe_export_netcdf(results, var)
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

    # ── NetCDF export / replot ─────────────────────────────────────────

    def _maybe_export_netcdf(self, results, token: str) -> None:
        """Export per-source series (incl. the benchmark envelope band).

        Delegates to :func:`export_timeseries_netcdf`, which tags the
        ``ts_benchmark_*`` metadata attrs needed to rebuild the figures from
        the NetCDF alone via :meth:`replot_from_netcdf`.
        """
        if not getattr(self, "save_netcdf", False):
            return
        from feather.diag._ts_panel import export_timeseries_netcdf

        period = getattr(self, "period", None) or self.config.get_period()
        try:
            export_timeseries_netcdf(self._netcdf_dir, token, results, period)
        except Exception:  # noqa: BLE001
            logger.warning(
                "NetCDF export failed for %s/%s", self.name, token,
                exc_info=True,
            )

    def replot_from_netcdf(
        self, skip_existing: bool = True,
    ) -> list[tuple["Path", "Path"]]:
        """Regenerate the time-series figures from previously saved NetCDF.

        Reads ``{output}/netcdf/timeseries/{var}_{start}-{end}.nc`` (written by
        an earlier run with ``--save-netcdf``) and re-renders the main,
        envelope, and anomaly figures *without* touching the source
        model/obs/CMIP6 data — cheap enough for a login node.  Variables whose
        NetCDF file is missing are skipped.
        """
        from pathlib import Path

        from feather.diag.netcdf_export import sanitize_name

        logger.info("Replotting %s from NetCDF", self.name)
        saved: list[tuple[Path, Path]] = []
        # sanitised config-model name → real name (recovers display colors)
        name_map = {sanitize_name(m): m for m in self.config.models}

        for var in self.variables:
            nc = self._netcdf_dir / (
                f"{var}_{self.period[0]}-{self.period[1]}.nc"
            )
            if not nc.exists():
                logger.info("  %s: no NetCDF (%s) — skipping", var, nc.name)
                continue

            figure_ids = [
                f"{var}_timeseries",
                f"{var}_timeseries_envelope",
                f"{var}_timeseries_anomaly",
            ]
            if skip_existing and all(
                self._figure_exists(fid) for fid in figure_ids
            ):
                logger.info("  %s: figures exist — skipping", var)
                saved.extend(
                    (self.output_dir / f"{fid}.png",
                     self.output_dir / f"{fid}.json")
                    for fid in figure_ids
                )
                continue

            try:
                vr = self._results_from_netcdf(nc, var, name_map)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "  %s: could not rebuild from NetCDF — skipping",
                    var, exc_info=True,
                )
                continue
            if vr is None:
                continue

            for fig, meta in self._plot_single(var, vr):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        logger.info(
            "Replot %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    def _results_from_netcdf(
        self, nc_path: "Path", var: str, name_map: dict[str, str],
    ) -> dict[str, Any] | None:
        """Reconstruct a ``_compute_single``-shaped result dict from NetCDF."""
        from feather.diag._ts_panel import load_timeseries_netcdf

        vr = load_timeseries_netcdf(
            nc_path, name_map, self.benchmarks, _benchmark_color,
        )
        if vr is None:
            return None
        vr["var_info"] = get_var(var)
        return vr

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
            logger.info("    Global mean: %.4g %s", float(ts.mean()), var_info.units)

        if not model_ts:
            logger.warning("  No model data for %s — skipping variable", var)
            return None

        logger.info("  Loading observations for %s", var)
        obs_data = self._load_obs_var(var, self.period)
        obs_ts = latlon_global_mean(obs_data)
        logger.info("    Obs global mean: %.4g %s", float(obs_ts.mean()), var_info.units)

        # Per-benchmark global-mean MMM time series (CMIP6, HighResMIP, …).
        # ``_benchmark_timeseries`` always derives the min/max envelope band
        # internally; the individual *spaghetti* lines stay gated on
        # ``cmip6_individual``.
        benchmarks_ts = self._benchmark_timeseries(
            var, period=self.period, return_individual=self.cmip6_individual,
        )

        # Back-compat: expose the primary benchmark under the cmip6_* keys.
        primary = benchmarks_ts[0] if benchmarks_ts else None
        cmip6_ts = primary["ts"] if primary else None
        cmip6_info = primary["info"] if primary else {}
        cmip6_individual_ts = primary["individual"] if primary else {}

        ens_mean, ens_median = self._compute_ensemble_stats(model_ts)

        return {
            "models": model_ts,
            "obs": obs_ts,
            "var_info": var_info,
            "benchmarks_ts": benchmarks_ts,
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

        Time coordinates are first normalised to first-of-month timestamps so
        members on different calendars (e.g. HadGEM3's 360-day cftime vs other
        models' ``datetime64``) and differing mid-month day conventions still
        overlap.  Series are then aligned on the inner time union so models
        with different lengths are reduced to their common period before
        averaging.

        Parameters
        ----------
        model_ts : dict
            Mapping of model name → DataArray with a ``time`` dimension.

        Returns
        -------
        ens_mean, ens_median : DataArray or None
            Returns ``None`` for both when fewer than 2 members remain or the
            members share no common month.
        """
        import xarray as xr

        series = list(model_ts.values())
        if len(series) < 2:
            return None, None

        # Drop non-dimension scalar coords (e.g. ``height`` on tas, ``depth``
        # on ocean vars) that some models carry and others don't — otherwise
        # xr.concat with the default coords="different" raises when the coord
        # is not present in every member.  Normalise calendars to first-of-
        # month so mixed-calendar members (360-day vs datetime64) still align.
        series = [
            normalize_monthly_time(s.reset_coords(drop=True)) for s in series
        ]
        series = [s for s in series if s is not None]
        if len(series) < 2:
            return None, None

        aligned = xr.align(*series, join="inner")
        if aligned[0].sizes.get("time", 0) == 0:
            logger.warning(
                "Ensemble members share no common month — "
                "skipping ensemble mean/median",
            )
            return None, None
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

        benchmarks = vr.get("benchmarks_ts", [])
        for bench in benchmarks:
            all_models.append(bench["label"])
            all_models.extend(bench["individual"].keys())

        # --- Monthly pass (background, washed-out) ---

        # Benchmark individual + MMM monthly (CMIP6, HighResMIP, \u2026)
        for bench in benchmarks:
            b_color = bench["color"]
            for ts in bench["individual"].values():
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values + _off,
                        color=b_color, alpha=0.2, linewidth=0.5)
            b_ts = bench["ts"]
            time_vals = _to_plot_time(b_ts.time.values)
            ax.plot(time_vals, b_ts.values + _off,
                    color=b_color, alpha=0.3, linewidth=0.7,
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

        # Benchmark individual + MMM annual (CMIP6, HighResMIP, …)
        for bench in benchmarks:
            b_color = bench["color"]
            b_label = bench["label"]
            # strip a trailing " MMM" for the members legend label
            members_name = b_label[:-4] if b_label.endswith(" MMM") else b_label
            n_indiv = len(bench["individual"])
            for i, ts in enumerate(bench["individual"].values()):
                label = (f"{members_name} members ({n_indiv})"
                         if i == 0 else "_nolegend_")
                ts_annual = annual_mean(ts)
                time_vals = _to_plot_time(ts_annual.time.values)
                ax.plot(time_vals, ts_annual.values + _off,
                        color=b_color, alpha=0.35, linewidth=0.8,
                        label=label)
            n_mmm = bench["info"].get("n_members", 0)
            b_annual = annual_mean(bench["ts"])
            time_vals = _to_plot_time(b_annual.time.values)
            ax.plot(time_vals, b_annual.values + _off,
                    label=f"{b_label} ({n_mmm})", color=b_color,
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
            benchmark_info=self._benchmark_meta_from_list(
                vr.get("benchmarks_ts")) or None,
        )
        figures = [(fig, meta)]

        # Two extra figures: an absolute time series with a gray CMIP6
        # min/max envelope band, and the same as anomalies relative to the
        # full-period mean (also with the envelope band).
        figures.append(self._plot_envelope(var, vr, anomaly=False))
        figures.append(self._plot_envelope(var, vr, anomaly=True))
        return figures

    def _plot_envelope(
        self, var: str, vr: dict[str, Any], *, anomaly: bool,
    ) -> tuple[plt.Figure, dict]:
        """Time-series figure with a gray benchmark min/max envelope band.

        Replaces the per-member benchmark spaghetti with a shaded min-to-max
        band (computed across the benchmark's individual members).  When
        *anomaly* is True every series is shown relative to its own full-period
        mean, so offsets cancel and the band highlights spread about each
        series' baseline.
        """
        var_info = vr["var_info"]
        _disp_units = var_info.display_units or var_info.units
        benchmarks = vr.get("benchmarks_ts", [])
        all_models = list(self.config.models)
        for bench in benchmarks:
            all_models.append(bench["label"])

        fig = build_envelope_timeseries(
            anomaly=anomaly,
            models=vr["models"],
            model_color=self.config.get_model_color,
            obs=vr["obs"],
            obs_label=var_info.obs_dataset,
            long_name=var_info.long_name,
            units=_disp_units,
            benchmarks=benchmarks,
            ens_mean=vr.get("ens_mean"),
            ens_median=vr.get("ens_median"),
            ens_prefix=self._project_name,
            offset=var_info.display_offset,
        )

        if anomaly:
            suffix, title_kind = "anomaly", " Anomaly"
            descr = (
                f"Area-weighted global mean {var_info.long_name} anomaly "
                f"(relative to the {self.period[0]}–{self.period[1]} mean) "
                f"with a gray benchmark min–max envelope."
            )
        else:
            suffix, title_kind = "envelope", ""
            descr = (
                f"Area-weighted global mean {var_info.long_name} with a gray "
                f"benchmark min–max envelope across members."
            )

        meta = self._build_metadata(
            title=f"{var_info.long_name} Global Mean{title_kind} Time Series",
            figure_id=f"{var}_timeseries_{suffix}",
            models=all_models,
            variables=[var],
            description=descr,
            plot_type="timeseries",
            period=self.period,
            cmip6_info=vr.get("cmip6_info") or None,
            benchmark_info=self._benchmark_meta_from_list(
                vr.get("benchmarks_ts")) or None,
        )
        return (fig, meta)


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
