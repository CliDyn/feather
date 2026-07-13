"""Daily Tmin/Tmax climate change signal diagnostic.

Computes the climate change signal of the **mean daily minimum (tasmin)** and
**mean daily maximum (tasmax)** 2 m temperature for the EERIE ensemble: the
difference between a future period (SSP2-4.5, e.g. 2031–2050) and a historical
reference period (e.g. 1981–2000), for the annual mean and each of the four
seasons (DJF, MAM, JJA, SON).

Unlike :mod:`feather.diag.tropical_nights_change` (a threshold-exceedance
*count*) this diagnostic works directly on the daily field's period means, so
the signal is a plain temperature change (°C).  It is **global** (land + ocean)
by design — daily Tmin/Tmax are physically meaningful over the ocean too.

Figures (per variable in ``["tasmin", "tasmax"]``):

A (×5): Combined map — N_models rows × 3 columns
         [Reference period | Future period | Change (Future − Reference)]
         for the annual mean and each of DJF, MAM, JJA, SON.
         Models without SSP2-4.5 data show a placeholder in columns 2–3.
B (×1): Stitched hist+ssp245 global-mean annual time series (°C) with a
         vertical line at 2015 and shaded reference / future windows.

Per-model NetCDF checkpoints (outside figures tree), holding the annual- and
seasonal-mean series derived from the daily data (``year, lat, lon``):

  ``{output_dir}/temp_extremes_change/{model}_{var}_hist_{start}_{end}.nc``
  ``{output_dir}/temp_extremes_change/{model}_{var}_ssp_{start}_{end}.nc``

Each file stores ``{var}_annual`` and ``{var}_{DJF,MAM,JJA,SON}``.  Re-runs
load these instead of re-reading the daily data (login-node-safe replot).

Configuration lives in ``config.project["climate_change"]`` (shared with the
other ``*_change`` diagnostics — see
:mod:`feather.diag.tropical_nights_change`).
"""

import copy
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.config import ModelConfig
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.util.spatial import latlon_global_mean
from feather.util.temporal import seasonal_annual_mean

logger = logging.getLogger(__name__)

_K_TO_C: float = 273.15
_HIST_BOUNDARY_YEAR: int = 2015          # dashed line in timeseries plot
_SEASONS: list[str] = ["DJF", "MAM", "JJA", "SON"]
_PERIODS: list[str] = ["annual", *_SEASONS]

_VAR_LABEL: dict[str, str] = {
    "tasmin": "Mean Daily Minimum Temperature (Tmin)",
    "tasmax": "Mean Daily Maximum Temperature (Tmax)",
}
_VAR_SHORT: dict[str, str] = {"tasmin": "Tmin", "tasmax": "Tmax"}


@register
class TempExtremesChangeDiag(DiagnosticBase):
    """Climate change signal of mean daily Tmin/Tmax under SSP2-4.5.

    Loads daily ``tasmin`` and ``tasmax`` from the historical and
    highres-future-ssp245 experiments, computes annual- and seasonal-mean
    fields for both, and produces reference/future/change maps plus a stitched
    global-mean time series per variable.
    """

    name = "temp_extremes_change"
    title = "Daily Tmin/Tmax — Climate Change Signal (SSP2-4.5)"
    domain = "sfc"
    variables = ["tasmin", "tasmax"]
    group = "extremes"

    def __init__(
        self,
        model_loader,
        obs_loader,
        config,
        *,
        cmip6_loader=None,
        experiment: str = "hist-1950",
        period: tuple[str, str] = ("1981", "2000"),
        variables=None,
    ):
        super().__init__(model_loader, obs_loader, config, cmip6_loader=cmip6_loader)
        # Honour the pipeline's --variables selection, restricted to the pair
        # this diagnostic supports; empty overlap → keep both.
        if variables:
            self.variables = [v for v in variables if v in type(self).variables] \
                or list(type(self).variables)
        cc = config.project.get("climate_change", {})
        self.ref_period: tuple[str, str] = tuple(
            cc.get("reference_period", list(period))
        )
        self.fut_period: tuple[str, str] = tuple(
            cc.get("future_period", ["2031", "2050"])
        )
        self.hist_load_period: tuple[str, str] = tuple(
            cc.get("hist_load_period", [period[0], "2014"])
        )
        self.ssp_load_period: tuple[str, str] = tuple(
            cc.get("ssp_load_period", ["2015", "2050"])
        )
        self._cc_models: dict = cc.get("models", {})

    # ── Paths ──────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for NC checkpoints (outside figures tree)."""
        return Path(self.config.output_dir) / "temp_extremes_change"

    def _nc_path(self, model: str, var: str, seg: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        period = self.hist_load_period if seg == "hist" else self.ssp_load_period
        return self.nc_dir / f"{safe}_{var}_{seg}_{period[0]}_{period[1]}.nc"

    # ── Loader factories (mirror TropicalNightsChangeDiag) ─────────────

    def _make_cmor_loader(self, model: str, experiment: str, data_root: str = ""):
        """Build a CMORLoader with the given experiment/root for *model*."""
        from feather.data.cmor_loader import CMORLoader

        mc_orig = self.config.model_configs.get(model) or ModelConfig(name=model)
        mc = copy.copy(mc_orig)
        mc.experiment = experiment
        if data_root:
            mc.data_root = data_root
        cfg = copy.copy(self.config)
        cfg.model_configs = {**self.config.model_configs, model: mc}
        return CMORLoader(cfg)

    def _make_kerchunk_loader(self, data_root: str, model: str, variant: str):
        """Build a KerchunkParquetLoader using the native daily store."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        mc_orig = self.config.model_configs.get(model) or ModelConfig(name=model)
        mc = copy.copy(mc_orig)
        mc.variant = variant
        mc.data_root = ""       # use global root from data_source, not per-model
        cfg = copy.copy(self.config)
        cfg.model_configs = {**self.config.model_configs, model: mc}
        cfg.data_source = {"type": "kerchunk_parquet", "root": data_root}
        return KerchunkParquetLoader(cfg)

    def _make_hist_loader(self, model: str):
        """Return loader for the historical experiment of *model*."""
        cc_cfg = self._cc_models.get(model, {})
        src = cc_cfg.get("hist_data_source", "cmor")
        if src == "kerchunk_native":
            mc = self.config.model_configs.get(model)
            variant = mc.variant if mc else "r1i1p1f1"
            return self._make_kerchunk_loader(
                cc_cfg["hist_data_root"], model, variant
            )
        experiment = cc_cfg.get("hist_experiment", "hist-1950")
        return self._make_cmor_loader(model, experiment, cc_cfg.get("hist_data_root", ""))

    def _make_fut_loader(self, model: str):
        """Return loader for the SSP2-4.5 experiment, or None if unavailable."""
        cc_cfg = self._cc_models.get(model, {})

        fut_only_to = cc_cfg.get("future_only_to")
        if fut_only_to and int(fut_only_to) < int(self.fut_period[0]):
            logger.info(
                "  %s: future run ends %s < %s — no change map, "
                "but SSP time series will be loaded",
                model, fut_only_to, self.fut_period[0],
            )

        src = cc_cfg.get("future_data_source", "cmor")
        if src == "kerchunk_native":
            fut_root = cc_cfg.get("future_data_root")
            if not fut_root:
                return None
            mc = self.config.model_configs.get(model)
            variant = mc.variant if mc else "r1i1p1f1"
            return self._make_kerchunk_loader(fut_root, model, variant)

        fut_exp = cc_cfg.get("future_experiment")
        if not fut_exp:
            return None
        return self._make_cmor_loader(
            model, fut_exp, cc_cfg.get("future_data_root", "")
        )

    # ── Loading / derivation ───────────────────────────────────────────

    def _load_from_loader(
        self, loader, model: str, var: str, period: tuple[str, str]
    ) -> xr.DataArray:
        """Call load_var, handling CMORLoader (needs table=) vs Kerchunk."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        if isinstance(loader, KerchunkParquetLoader):
            return loader.load_var(model, var, period=period)
        return loader.load_var(model, var, table="day", period=period)

    @staticmethod
    def _annual_series(da: xr.DataArray) -> xr.DataArray:
        """Per-year mean of a daily field, dims (year, lat, lon)."""
        return da.groupby("time.year").mean("time")

    def _load_and_save_means(
        self,
        model: str,
        var: str,
        loader,
        period: tuple[str, str],
        nc_path: Path,
    ) -> dict[str, xr.DataArray] | None:
        """Load/compute annual + seasonal mean series from daily *var*.

        Returns a dict ``{"annual": DA(year,lat,lon), "DJF": ..., ...}`` or
        ``None`` if the daily data cannot be loaded.  Persists the series to
        ``nc_path`` so re-runs skip the daily read.
        """
        if nc_path.exists():
            logger.info("  %s/%s: loading means from %s", model, var, nc_path.name)
            ds = xr.open_dataset(nc_path)
            return {
                s: ds[f"{var}_{s}"] for s in _PERIODS if f"{var}_{s}" in ds
            }

        logger.info("  %s/%s: computing means %s–%s", model, var, *period)
        try:
            da = self._load_from_loader(loader, model, var, period)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.warning("  %s/%s: cannot load (%s) — skipping", model, var, exc)
            return None

        raw: dict[str, xr.DataArray] = {"annual": self._annual_series(da)}
        for s in _SEASONS:
            raw[s] = seasonal_annual_mean(da, s)

        # Trigger the dask graph once for all season/annual series.
        keys = list(raw)
        try:
            import dask
            computed = dask.compute(*(raw[k] for k in keys))
            series = dict(zip(keys, computed))
        except (ImportError, AttributeError):
            series = {
                k: (v.compute() if hasattr(v, "compute") else v)
                for k, v in raw.items()
            }

        nc_path.parent.mkdir(parents=True, exist_ok=True)
        ds_vars = {
            f"{var}_{k}": v.assign_attrs(
                {"long_name": f"{_VAR_LABEL[var]} ({k}) annual-mean series",
                 "units": "K"}
            )
            for k, v in series.items()
        }
        xr.Dataset(
            ds_vars,
            attrs={
                "model": model,
                "variable": var,
                "period_start": period[0],
                "period_end": period[1],
                "note": "Annual- and seasonal-mean series derived from daily data.",
            },
        ).to_netcdf(nc_path)
        logger.info("  Saved means NC: %s", nc_path)
        return series

    @staticmethod
    def _period_mean(
        series: dict[str, xr.DataArray], period: tuple[str, str]
    ) -> dict[str, xr.DataArray]:
        """Mean over ``year`` in *period* for each season/annual series."""
        lo, hi = int(period[0]), int(period[1])
        out: dict[str, xr.DataArray] = {}
        for s, da in series.items():
            sl = da.sel(year=slice(lo, hi))
            if len(sl.year) > 0:
                out[s] = sl.mean("year")
        return out

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    def compute(self) -> dict[str, Any]:
        """Load/compute mean Tmin/Tmax for ref and future periods.

        Returns a nested dict keyed by variable then model:
          ``ref[var][model][season]``   — reference-period mean field (lat, lon)
          ``fut[var][model][season]``   — future-period mean field
          ``change[var][model][season]``— future − reference
          ``hist_series[var][model]``   — global-mean annual series (K)
          ``ssp_series[var][model]``    — global-mean annual series (K)
          ``models[var]``               — models with hist data for *var*
        """
        cc_models = [m for m in self.config.models if m in self._cc_models]
        if not cc_models:
            cc_models = list(self.config.models)

        ref: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
        fut: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
        change: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
        hist_series: dict[str, dict[str, xr.DataArray]] = {}
        ssp_series: dict[str, dict[str, xr.DataArray]] = {}
        models: dict[str, list[str]] = {}
        for var in self.variables:
            ref[var], fut[var], change[var] = {}, {}, {}
            hist_series[var], ssp_series[var] = {}, {}
            models[var] = []

        for model in cc_models:
            logger.info("  Processing: %s", model)
            hist_loader = self._make_hist_loader(model)
            fut_loader = self._make_fut_loader(model)

            for var in self.variables:
                hist = self._load_and_save_means(
                    model, var, hist_loader,
                    self.hist_load_period, self._nc_path(model, var, "hist"),
                )
                if hist is None:
                    continue

                models[var].append(model)
                ref[var][model] = self._period_mean(hist, self.ref_period)
                hist_series[var][model] = latlon_global_mean(
                    hist["annual"]).compute()

                if fut_loader is None:
                    logger.info(
                        "  %s/%s: no future loader — reference-only", model, var)
                    continue

                ssp = self._load_and_save_means(
                    model, var, fut_loader,
                    self.ssp_load_period, self._nc_path(model, var, "ssp"),
                )
                if ssp is None:
                    logger.warning(
                        "  %s/%s: SSP load failed — reference only", model, var)
                    continue

                ssp_series[var][model] = latlon_global_mean(
                    ssp["annual"]).compute()
                fut_means = self._period_mean(ssp, self.fut_period)
                if not fut_means:
                    logger.info(
                        "  %s/%s: SSP run ends before future period (%s) — "
                        "excluded from change map", model, var, self.fut_period[0])
                    continue
                fut[var][model] = fut_means
                change[var][model] = {
                    s: fut_means[s] - ref[var][model][s]
                    for s in fut_means if s in ref[var][model]
                }

        return {
            "variables": self.variables,
            "models": models,
            "ref": ref,
            "fut": fut,
            "change": change,
            "hist_series": hist_series,
            "ssp_series": ssp_series,
        }

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        for var in results["variables"]:
            if not results["models"].get(var):
                logger.warning("%s: no model data for %s — skipping", self.name, var)
                continue
            for season in _PERIODS:
                fig_meta = self._plot_change_panels(results, var, season)
                if fig_meta is not None:
                    figs.append(fig_meta)
            figs.append(self._plot_timeseries(results, var))
        return figs

    @staticmethod
    def _collect_finite(arrays: list) -> np.ndarray:
        parts = [np.asarray(a).ravel() for a in arrays if a is not None]
        if not parts:
            return np.array([], dtype=np.float64)
        merged = np.concatenate(parts)
        return merged[np.isfinite(merged)]

    def _plot_change_panels(
        self, results: dict, var: str, season: str
    ) -> tuple[plt.Figure, dict] | None:
        """Combined figure: N_models rows × 3 cols [Reference | Future | Change]."""
        import nereus as nr
        from nereus.plotting import get_projection
        from feather.plot.maps import _flatten_latlon

        models = results["models"][var]
        ref = {m: results["ref"][var][m].get(season) for m in models}
        fut = {m: results["fut"][var].get(m, {}).get(season) for m in models}
        chg = {m: results["change"][var].get(m, {}).get(season) for m in models}
        if all(v is None for v in ref.values()):
            return None

        n_models = len(models)
        season_lbl = "Annual" if season == "annual" else season

        # Reference/future panels in °C; change panels are a difference (ΔK=Δ°C).
        rf_finite = self._collect_finite(
            [ref[m] for m in models] + [fut[m] for m in models]
        ) - _K_TO_C
        if len(rf_finite) > 0:
            vmin_rf = float(np.percentile(rf_finite, 2))
            vmax_rf = float(np.percentile(rf_finite, 98))
        else:
            vmin_rf, vmax_rf = -30.0, 40.0
        ch_finite = self._collect_finite(list(chg.values()))
        vlim = max(float(np.percentile(np.abs(ch_finite), 98)), 0.5) \
            if len(ch_finite) > 0 else 5.0

        proj = get_projection("rob")
        fig, axes = plt.subplots(
            n_models, 3,
            figsize=(21, 5 * n_models),
            subplot_kw={"projection": proj},
        )
        if n_models == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            f"Reference {self.ref_period[0]}–{self.ref_period[1]}",
            f"Future {self.fut_period[0]}–{self.fut_period[1]}",
            f"Change (Future − Reference)\n"
            f"{self.fut_period[0]}–{self.fut_period[1]} minus "
            f"{self.ref_period[0]}–{self.ref_period[1]}",
        ]
        cmaps = ["cmo.thermal", "cmo.thermal", "RdBu_r"]
        vmins = [vmin_rf, vmin_rf, -vlim]
        vmaxs = [vmax_rf, vmax_rf, vlim]
        offsets = [_K_TO_C, _K_TO_C, 0.0]

        for row, model in enumerate(models):
            row_interp = None
            panels = [ref[model], fut[model], chg[model]]
            for col, (data, ctitle, cmap, vmin, vmax, off) in enumerate(
                zip(panels, col_titles, cmaps, vmins, vmaxs, offsets)
            ):
                ax = axes[row, col]
                panel_title = f"{model}\n{ctitle}" if col == 0 else ctitle
                if data is not None:
                    vals, lons, lats = _flatten_latlon(data)
                    _, _, row_interp = nr.plot(
                        np.asarray(vals) - off, lons, lats,
                        ax=ax,
                        projection="rob",
                        resolution=0.25,
                        interpolator=row_interp,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        colorbar=True,
                        colorbar_label="°C",
                        title=panel_title,
                    )
                else:
                    ax.set_title(panel_title, fontsize=9)
                    ax.text(
                        0.5, 0.5, "No SSP2-4.5 data available",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="gray", style="italic",
                    )

        fig.suptitle(
            f"{_VAR_LABEL[var]} — {season_lbl}\n"
            f"Reference: {self.ref_period[0]}–{self.ref_period[1]}  ·  "
            f"Future: {self.fut_period[0]}–{self.fut_period[1]} (SSP2-4.5)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        fut_models = [m for m in models if chg[m] is not None]
        stats = {}
        for model in models:
            s = {}
            if ref[model] is not None:
                s["ref_mean_degC"] = self._latlon_field_mean(ref[model]) - _K_TO_C
            if fut[model] is not None:
                s["future_mean_degC"] = self._latlon_field_mean(fut[model]) - _K_TO_C
            if chg[model] is not None:
                s["change_degC"] = self._latlon_field_mean(chg[model])
            if s:
                stats[model] = s

        meta = self._build_metadata(
            title=f"{_VAR_SHORT[var]} Change — {season_lbl} (Reference / Future / Change)",
            figure_id=f"{var}_{season}_change_panels",
            models=models,
            description=(
                f"Per-model {season_lbl.lower()} {_VAR_LABEL[var].lower()} (rows). "
                f"Left: mean over the reference period "
                f"{self.ref_period[0]}–{self.ref_period[1]} (°C). "
                f"Centre: mean over the future period "
                f"{self.fut_period[0]}–{self.fut_period[1]} under SSP2-4.5 (°C). "
                f"Right: change (future − reference, °C) with diverging colormap. "
                f"Global (land + ocean). Models with SSP2-4.5 data: "
                f"{', '.join(fut_models) or 'none'}."
            ),
            period=(self.ref_period[0], self.fut_period[1]),
            obs_dataset="",
            obs_variable="",
            plot_type="map",
            summary_statistics=stats,
        )
        return fig, meta

    def _plot_timeseries(self, results: dict, var: str) -> tuple[plt.Figure, dict]:
        """Stitched hist+ssp245 global-mean annual time series (°C)."""
        fig, ax = plt.subplots(figsize=(13, 5))

        ax.axvspan(int(self.ref_period[0]), int(self.ref_period[1]) + 1,
                   alpha=0.10, color="#1f77b4", zorder=0)
        ax.axvspan(int(self.fut_period[0]), int(self.fut_period[1]) + 1,
                   alpha=0.10, color="#d62728", zorder=0)
        ax.axvline(_HIST_BOUNDARY_YEAR, color="k", lw=1.2, ls="--", alpha=0.55,
                   label=f"hist | ssp245 ({_HIST_BOUNDARY_YEAR})")

        models = results["models"][var]
        stats = {}
        for model in models:
            color = self.config.get_model_color(model)
            h = results["hist_series"][var].get(model)
            s = results["ssp_series"][var].get(model)
            if h is not None and s is not None:
                combined = xr.concat([h, s], dim="year")
            else:
                combined = h if h is not None else s
            if combined is None:
                continue
            ax.plot(
                np.asarray(combined["year"]), np.asarray(combined) - _K_TO_C,
                color=color, lw=1.5, label=model,
            )
            stats[model] = self._series_stats(combined - _K_TO_C)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{_VAR_SHORT[var]} (°C)")
        ax.set_title(
            f"{_VAR_LABEL[var]} — Global Mean\n"
            f"Blue shading: ref {self.ref_period[0]}–{self.ref_period[1]}; "
            f"red shading: future {self.fut_period[0]}–{self.fut_period[1]}"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{_VAR_SHORT[var]} — Global Mean Time Series",
            figure_id=f"{var}_change_timeseries",
            models=models,
            description=(
                f"Area-weighted global-mean annual {_VAR_LABEL[var].lower()} (°C). "
                f"Continuous line: hist-1950 ({self.hist_load_period[0]}–"
                f"{self.hist_load_period[1]}) joining SSP2-4.5 from "
                f"{_HIST_BOUNDARY_YEAR}. Blue shading: reference period "
                f"({self.ref_period[0]}–{self.ref_period[1]}); red shading: future "
                f"period ({self.fut_period[0]}–{self.fut_period[1]}). Models without "
                "future data show only the historical segment."
            ),
            period=(self.hist_load_period[0], self.ssp_load_period[1]),
            obs_dataset="",
            obs_variable="",
            plot_type="timeseries",
            summary_statistics=stats,
        )
        return fig, meta
