"""Köppen–Trewartha climate shifts at a global warming level (Phase 2).

For each model family (CORDEX / CMIP5 / CMIP6) this diagnostic compares the
reference KT classification with the classification at a target global warming
level (default +2 °C above the 1850–1900 pre-industrial baseline):

- the warming-level 30-year window is found **per GCM** from its own
  global-mean temperature (CMIP5 → rcp85, CMIP6 → ssp585);
- regional models (CORDEX) inherit the window of their **driving GCM**
  (``driving_gcm`` in the config);
- the future climatology is classified the same way as the reference, and the
  family ensemble mean is used for the shift products.

Outputs (in ``{output_dir}/climate_shifts/``):

- ``{source}_kt_ref_*.nc`` / ``{source}_kt_gwl*_*.nc`` — reference & future KT
  codes per source and per family ensemble mean,
- a CSV of % land area per KT type (reference, future, change) per source,
- per-family **shift maps** [reference | future | change], an **area-shift**
  bar chart, **transition-matrix** heatmaps, and a **net gain/loss** summary.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from feather.diag.climate_classification import KTClimateClassification, _CMIP6_MMM
from feather.diag.registry import register
from feather.util.gwl import global_mean_annual_tas, warming_level_window
from feather.util.koeppen_trewartha import (
    KT_DESCRIPTIONS,
    KT_LABELS,
    area_percent_by_type,
    transition_matrix,
)
from feather.util.spatial import compute_latlon_areas

logger = logging.getLogger(__name__)


@register
class ClimateShiftsDiag(KTClimateClassification):
    """KT climate shifts between the reference period and a 2 °C GWL window."""

    name = "climate_shifts"
    title = "Köppen–Trewartha Climate Shifts"
    domain = "sfc"
    variables = ["tas", "pr"]
    group = "climate_classification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gwl = self.config.project.get("gwl", {})
        self._level = float(gwl.get("level", 2.0))
        self._baseline = tuple(gwl.get("baseline", [1850, 1900]))
        self._window = int(gwl.get("window", 30))
        self._gwl_max_year = int(gwl.get("max_year", 2100))

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / "climate_shifts"

    @property
    def _gwl_tag(self) -> str:
        return f"gwl{self._level:g}".replace(".", "p")

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = self.compute()
        if not results.get("families"):
            logger.warning("  No family with a %g°C window — no figures", self._level)
            return saved
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)
        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        # 1. GWL windows per GCM (from CMIP5/CMIP6 global-mean tas)
        gcm_windows = self._compute_gwl_windows()
        logger.info("  GWL %g°C windows: %s", self._level, gcm_windows)

        # 2. Per-model reference + future climatologies → KT codes
        ref_t, ref_p, fut_t, fut_p = {}, {}, {}, {}
        windows: dict[str, tuple[int, int]] = {}
        for model in self.config.models:
            win = self._model_window(model, gcm_windows)
            if win is None:
                continue
            try:
                rt, rp = self._model_clim(model, self.period)
                ft, fp = self._model_clim(model, (str(win[0]), str(win[1])))
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue
            ref_t[model], ref_p[model] = rt, rp
            fut_t[model], fut_p[model] = ft, fp
            windows[model] = win

        if not windows:
            return {"families": {}}

        # 3. Group members into families
        families: dict[str, dict] = {}
        for model, win in windows.items():
            mc = self.config.model_configs.get(model)
            label = mc.ensemble if mc and mc.ensemble else "Models"
            families.setdefault(label, {"members": [], "windows": []})
            families[label]["members"].append(model)
            families[label]["windows"].append(win)

        area_da = xr.DataArray(
            compute_latlon_areas(self._tlat, self._tlon),
            dims=("lat", "lon"), coords={"lat": self._tlat, "lon": self._tlon},
        )

        self.nc_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []          # per-source area-shift table rows
        sources: dict[str, dict] = {}  # per-source ref/future codes + pct

        # Per-source classification (kept for the per-source shift maps)
        for model, win in windows.items():
            rc = self._classify_from(ref_t[model], ref_p[model])
            fc = self._classify_from(fut_t[model], fut_p[model])
            self._save_code(model, "ref", rc, win)
            self._save_code(model, self._gwl_tag, fc, win)
            rp = area_percent_by_type(rc, area_da)
            fp = area_percent_by_type(fc, area_da)
            sources[model] = {"ref_code": rc, "fut_code": fc,
                              "ref_pct": rp, "fut_pct": fp, "window": win}
            for lbl in KT_LABELS:
                rows.append({
                    "family": self.config.model_configs[model].ensemble,
                    "source": model, "kt_type": lbl,
                    "ref_pct": rp[lbl], "future_pct": fp[lbl],
                    "change_pct": fp[lbl] - rp[lbl],
                    "window": f"{win[0]}-{win[1]}",
                })

        # Per-family ensemble mean and median (of the member climatologies)
        for label, fam in families.items():
            members = fam["members"]
            fam["window"] = _modal_window(fam["windows"])
            ref_t_stack = xr.concat([ref_t[m] for m in members], "member")
            ref_p_stack = xr.concat([ref_p[m] for m in members], "member")
            fut_t_stack = xr.concat([fut_t[m] for m in members], "member")
            fut_p_stack = xr.concat([fut_p[m] for m in members], "member")
            for stat in ("mean", "median"):
                rt = getattr(ref_t_stack, stat)("member")
                rp_ = getattr(ref_p_stack, stat)("member")
                ft = getattr(fut_t_stack, stat)("member")
                fp_ = getattr(fut_p_stack, stat)("member")
                ref_code = self._classify_from(rt, rp_)
                fut_code = self._classify_from(ft, fp_)
                fam[stat] = {
                    "ref_code": ref_code, "fut_code": fut_code,
                    "ref_pct": area_percent_by_type(ref_code, area_da),
                    "fut_pct": area_percent_by_type(fut_code, area_da),
                    "transition": transition_matrix(ref_code, fut_code, area_da),
                }
                self._save_code(f"{label}_ens_{stat}", "ref", ref_code, fam["window"])
                self._save_code(f"{label}_ens_{stat}", self._gwl_tag, fut_code, fam["window"])

        pd.DataFrame(rows).to_csv(
            self.nc_dir / f"kt_area_shift_{self._gwl_tag}.csv", index=False,
        )

        return {
            "families": families,
            "sources": sources,
            "windows": windows,
            "lat": self._tlat,
            "lon": self._tlon,
        }

    # ── GWL helpers ────────────────────────────────────────────────────

    def _compute_gwl_windows(self) -> dict[str, tuple[int, int]]:
        """Find the warming-level window for every CMIP5/CMIP6 GCM (by gcm name)."""
        windows: dict[str, tuple[int, int]] = {}
        for model in self.config.models:
            mc = self.config.model_configs.get(model)
            src = self.config.get_model_data_source_type(model)
            if src not in ("cmip5", "cmip6_nc"):
                continue
            gcm = mc.gcm or model
            if gcm in windows:
                continue
            try:
                tas = self._load_model_var(model, "tas")   # full stitched series
                annual = global_mean_annual_tas(tas)
                annual = annual.sel(year=slice(None, self._gwl_max_year))
                win = warming_level_window(
                    annual, self._level, self._baseline, self._window,
                )
            except Exception as exc:
                logger.warning("  GWL for %s failed: %s", gcm, exc)
                win = None
            if win is not None:
                windows[gcm] = win
        return windows

    def _model_window(
        self, model: str, gcm_windows: dict[str, tuple[int, int]],
    ) -> tuple[int, int] | None:
        """The GWL window for a model: own (GCM) or driving GCM's (RCM)."""
        mc = self.config.model_configs.get(model)
        if mc is None:
            return None
        src = self.config.get_model_data_source_type(model)
        if src in ("cmip5", "cmip6_nc"):
            return gcm_windows.get(mc.gcm or model)
        if src == "cordex" and mc.driving_gcm:
            return gcm_windows.get(mc.driving_gcm)
        return None  # ERAINT-evaluation members / obs have no scenario

    # ── Persistence ────────────────────────────────────────────────────

    def _save_code(self, source, kind, code, window) -> None:
        out = code.astype("float32").rename("kt_code")
        ds = xr.Dataset({"kt_code": out}, attrs={
            "Conventions": "CF-1.8",
            "title": f"KT classification — {source} ({kind})",
            "source_dataset": source, "kind": kind,
            "window": f"{window[0]}-{window[1]}" if window else "",
            "warming_level": self._level,
            "baseline": f"{self._baseline[0]}-{self._baseline[1]}",
            "land_only": "True — ocean pixels are NaN",
        })
        path = self.nc_dir / f"{self._safe(source)}_kt_{kind}_{self._safe(str(self.period[0]))}.nc"
        ds.to_netcdf(path)

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        for stat in ("mean", "median"):
            figs.append(self._plot_shift_maps(results, stat))
            figs.append(self._plot_area_shift_bar(results, stat))
            figs.append(self._plot_transition_matrices(results, stat))
        figs.append(self._plot_net_gain_loss(results))
        # Per-source [ref | future | change] maps, one figure per family.
        for label in results["families"]:
            figs.append(self._plot_source_shift_maps(results, label))
        return figs

    def _render_shift_rows(self, panels, lon, lat, suptitle):
        """Render rows of [reference | future | change] KT maps.

        *panels* is a list of dicts with keys: ``label``, ``ref``, ``fut``,
        ``window``.
        """
        import cartopy.crs as ccrs
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap, norm = self._kt_cmap_norm()
        lon = np.asarray(lon)
        lon_plot = np.where(lon > 180.0, lon - 360.0, lon)
        order = np.argsort(lon_plot)
        lon_plot = lon_plot[order]
        extent = [self._lon_bounds[0], self._lon_bounds[1],
                  self._lat_bounds[0], self._lat_bounds[1]]
        chg_cmap = ListedColormap(["#dddddd", "#d62728"])
        chg_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

        n = len(panels)
        fig, axes = plt.subplots(
            n, 3, figsize=(15, 3.9 * n),
            subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
        )
        for r, p in enumerate(panels):
            w = p["window"]
            changed = (p["fut"] != p["ref"]).where(p["ref"].notnull())
            cols = [
                (f"{p['label']} — Reference {self.period[0]}–{self.period[1]}", p["ref"], cmap, norm),
                (f"{p['label']} — {self._level:g}°C ({w[0]}–{w[1]})", p["fut"], cmap, norm),
                (f"{p['label']} — Change (shifted=red)", changed, chg_cmap, chg_norm),
            ]
            for c, (title, field, cm, nm) in enumerate(cols):
                ax = axes[r, c]
                ax.pcolormesh(
                    lon_plot, lat, np.asarray(field.values)[:, order],
                    transform=ccrs.PlateCarree(), cmap=cm, norm=nm, shading="auto",
                )
                ax.coastlines(linewidth=0.4)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.set_title(title, fontsize=9)
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        return fig

    def _plot_shift_maps(self, results, stat) -> tuple[plt.Figure, dict]:
        """Ensemble (mean or median) [ref | future | change] maps, per family."""
        families = results["families"]
        panels = [
            {"label": f"{label} ({stat})", "ref": fam[stat]["ref_code"],
             "fut": fam[stat]["fut_code"], "window": fam["window"]}
            for label, fam in families.items()
        ]
        fig = self._render_shift_rows(
            panels, results["lon"], results["lat"],
            f"{self.title} — {self._level:g}°C GWL vs Reference (ensemble {stat})",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Shift Maps ({stat})",
            figure_id=f"kt_shift_maps_{stat}",
            models=[m for f in families.values() for m in f["members"]],
            description=(
                f"Per family: ensemble-{stat} KT classification for the reference "
                "period and at the warming level, plus a change map (shifted cells "
                "in red). Land only."
            ),
            period=self.period, plot_type="map",
        )
        return fig, meta

    def _plot_source_shift_maps(self, results, label) -> tuple[plt.Figure, dict]:
        """Per-source [ref | future | change] maps for one family's members."""
        fam = results["families"][label]
        sources = results["sources"]
        panels = [
            {"label": m, "ref": sources[m]["ref_code"], "fut": sources[m]["fut_code"],
             "window": sources[m]["window"]}
            for m in fam["members"]
        ]
        fig = self._render_shift_rows(
            panels, results["lon"], results["lat"],
            f"{self.title} — {label} members ({self._level:g}°C vs reference)",
        )
        meta = self._build_metadata(
            title=f"{self.title} — {label} Member Shifts",
            figure_id=f"kt_shift_maps_source_{self._safe(label)}",
            models=fam["members"],
            description=(
                f"Per-member KT classification for the {label} family: reference, "
                "warming-level, and change maps for each source. Land only."
            ),
            period=self.period, plot_type="map",
        )
        return fig, meta

    def _plot_area_shift_bar(self, results, stat) -> tuple[plt.Figure, dict]:
        """Reference vs future % land area per KT type, per family (mean/median)."""
        families = results["families"]
        nfam = len(families)
        fig, axes = plt.subplots(nfam, 1, figsize=(13, 3.2 * nfam), squeeze=False)
        x = np.arange(len(KT_LABELS))
        for r, (label, fam) in enumerate(families.items()):
            ax = axes[r, 0]
            ref = [fam[stat]["ref_pct"][l] for l in KT_LABELS]
            fut = [fam[stat]["fut_pct"][l] for l in KT_LABELS]
            ax.bar(x - 0.2, ref, 0.4, label="Reference", color="#4477aa")
            ax.bar(x + 0.2, fut, 0.4, label=f"{self._level:g}°C", color="#cc6677")
            ax.set_xticks(x)
            ax.set_xticklabels(KT_LABELS, fontsize=8)
            ax.set_ylabel("% land area")
            ax.set_title(f"{label} ensemble ({stat})", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{self.title} — Land-area Fraction by Type (ensemble {stat})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        meta = self._build_metadata(
            title=f"{self.title} — Area Shift ({stat})",
            figure_id=f"kt_area_shift_bar_{stat}",
            models=[m for f in families.values() for m in f["members"]],
            description=f"Reference vs warming-level % land area per KT type, per family ensemble {stat}.",
            period=self.period, plot_type="bar",
        )
        return fig, meta

    def _plot_net_gain_loss(self, results) -> tuple[plt.Figure, dict]:
        """Net change in % land area per type (future − reference), mean & median."""
        families = results["families"]
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(KT_LABELS))
        width = 0.8 / max(len(families), 1)
        for i, (label, fam) in enumerate(families.items()):
            mean_d = [fam["mean"]["fut_pct"][l] - fam["mean"]["ref_pct"][l] for l in KT_LABELS]
            med_d = [fam["median"]["fut_pct"][l] - fam["median"]["ref_pct"][l] for l in KT_LABELS]
            xb = x + i * width
            ax.bar(xb, mean_d, width, label=f"{label} (mean)")
            # median shown as black markers on top of the mean bars
            ax.plot(xb, med_d, "k_", markersize=6, markeredgewidth=1.4,
                    label="median" if i == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(KT_LABELS, fontsize=8)
        ax.set_ylabel("Δ % land area (future − reference)")
        ax.set_title(f"{self.title} — Net Gain/Loss per Type at {self._level:g}°C")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{self.title} — Net Gain/Loss", figure_id="kt_net_gain_loss",
            models=[m for f in families.values() for m in f["members"]],
            description=(
                "Net change in % land area per KT type (warming-level − reference) "
                "per family; bars = ensemble mean, black markers = ensemble median."
            ),
            period=self.period, plot_type="bar",
        )
        return fig, meta

    def _plot_transition_matrices(self, results, stat) -> tuple[plt.Figure, dict]:
        """KT transition matrix (reference → future) per family (mean/median)."""
        families = results["families"]
        nfam = len(families)
        fig, axes = plt.subplots(1, nfam, figsize=(6.2 * nfam, 5.6), squeeze=False)
        for c, (label, fam) in enumerate(families.items()):
            ax = axes[0, c]
            M = np.array(fam[stat]["transition"])
            im = ax.imshow(np.where(M > 0, M, np.nan), cmap="viridis", vmin=0)
            ax.set_xticks(np.arange(len(KT_LABELS)))
            ax.set_xticklabels(KT_LABELS, fontsize=6, rotation=90)
            ax.set_yticks(np.arange(len(KT_LABELS)))
            ax.set_yticklabels(KT_LABELS, fontsize=6)
            ax.set_xlabel(f"{self._level:g}°C type")
            ax.set_ylabel("Reference type")
            ax.set_title(f"{label} ({stat})", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% land area")
        fig.suptitle(
            f"{self.title} — Transition Matrices (ref → {self._level:g}°C, ensemble {stat})",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        meta = self._build_metadata(
            title=f"{self.title} — Transition Matrix ({stat})",
            figure_id=f"kt_transition_matrix_{stat}",
            models=[m for f in families.values() for m in f["members"]],
            description=(
                f"Area-weighted KT transition matrix per family ensemble {stat}: "
                "rows = reference type, columns = warming-level type, values = % "
                "of classified land area."
            ),
            period=self.period, plot_type="heatmap",
        )
        return fig, meta


def _modal_window(windows: list[tuple[int, int]]) -> tuple[int, int]:
    """Most common window among members (windows are identical within a GCM)."""
    from collections import Counter
    return Counter(windows).most_common(1)[0][0]
