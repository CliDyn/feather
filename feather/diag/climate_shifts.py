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

        # 3. Family ensembles (mean of member climatologies, ref & future)
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
        for label, fam in families.items():
            members = fam["members"]
            ref_mean_t = xr.concat([ref_t[m] for m in members], "member").mean("member")
            ref_mean_p = xr.concat([ref_p[m] for m in members], "member").mean("member")
            fut_mean_t = xr.concat([fut_t[m] for m in members], "member").mean("member")
            fut_mean_p = xr.concat([fut_p[m] for m in members], "member").mean("member")
            ref_code = self._classify_from(ref_mean_t, ref_mean_p)
            fut_code = self._classify_from(fut_mean_t, fut_mean_p)

            fam["ref_code"] = ref_code
            fam["fut_code"] = fut_code
            fam["ref_pct"] = area_percent_by_type(ref_code, area_da)
            fam["fut_pct"] = area_percent_by_type(fut_code, area_da)
            fam["transition"] = transition_matrix(ref_code, fut_code, area_da)
            fam["window"] = _modal_window(fam["windows"])

            self._save_code(f"{label}_ens_mean", "ref", ref_code, fam["window"])
            self._save_code(f"{label}_ens_mean", self._gwl_tag, fut_code, fam["window"])

            for m in members:
                rc = self._classify_from(ref_t[m], ref_p[m])
                fc = self._classify_from(fut_t[m], fut_p[m])
                self._save_code(m, "ref", rc, windows[m])
                self._save_code(m, self._gwl_tag, fc, windows[m])
                rp = area_percent_by_type(rc, area_da)
                fp = area_percent_by_type(fc, area_da)
                for lbl in KT_LABELS:
                    rows.append({
                        "family": label, "source": m, "kt_type": lbl,
                        "ref_pct": rp[lbl], "future_pct": fp[lbl],
                        "change_pct": fp[lbl] - rp[lbl],
                        "window": f"{windows[m][0]}-{windows[m][1]}",
                    })

        pd.DataFrame(rows).to_csv(
            self.nc_dir / f"kt_area_shift_{self._gwl_tag}.csv", index=False,
        )

        return {
            "families": families,
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
        figs.append(self._plot_shift_maps(results))
        figs.append(self._plot_area_shift_bar(results))
        figs.append(self._plot_net_gain_loss(results))
        figs.append(self._plot_transition_matrices(results))
        return figs

    def _plot_shift_maps(self, results: dict) -> tuple[plt.Figure, dict]:
        """Per family: [reference | future (GWL) | change] ensemble-mean maps."""
        import cartopy.crs as ccrs
        from matplotlib.colors import BoundaryNorm, ListedColormap

        families = results["families"]
        lon, lat = results["lon"], results["lat"]
        cmap, norm = self._kt_cmap_norm()
        lon_plot = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
        order = np.argsort(lon_plot)
        lon_plot = lon_plot[order]
        extent = [self._lon_bounds[0], self._lon_bounds[1],
                  self._lat_bounds[0], self._lat_bounds[1]]

        nfam = len(families)
        fig, axes = plt.subplots(
            nfam, 3, figsize=(15, 4.4 * nfam),
            subplot_kw={"projection": ccrs.PlateCarree()},
            squeeze=False,
        )
        # binary change colormap
        chg_cmap = ListedColormap(["#dddddd", "#d62728"])
        chg_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

        for r, (label, fam) in enumerate(families.items()):
            w = fam["window"]
            cols = [
                (f"{label} — Reference {self.period[0]}–{self.period[1]}", fam["ref_code"], cmap, norm),
                (f"{label} — {self._level:g}°C ({w[0]}–{w[1]})", fam["fut_code"], cmap, norm),
            ]
            changed = (fam["fut_code"] != fam["ref_code"]).where(fam["ref_code"].notnull())
            cols.append((f"{label} — Change (shifted=red)", changed, chg_cmap, chg_norm))
            for c, (title, field, cm, nm) in enumerate(cols):
                ax = axes[r, c]
                ax.pcolormesh(
                    lon_plot, lat, np.asarray(field.values)[:, order],
                    transform=ccrs.PlateCarree(), cmap=cm, norm=nm, shading="auto",
                )
                ax.coastlines(linewidth=0.4)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.set_title(title, fontsize=9)

        fig.suptitle(
            f"{self.title} — {self._level:g}°C GWL vs Reference", fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        meta = self._build_metadata(
            title=f"{self.title} — Shift Maps",
            figure_id="kt_shift_maps",
            models=[m for f in families.values() for m in f["members"]],
            description=(
                "Per model family: ensemble-mean KT classification for the "
                "reference period and at the warming level, plus a change map "
                "(cells where the KT type shifts in red). Land only."
            ),
            period=self.period, plot_type="map",
        )
        return fig, meta

    def _plot_area_shift_bar(self, results: dict) -> tuple[plt.Figure, dict]:
        """Reference vs future % land area per KT type, per family."""
        families = results["families"]
        nfam = len(families)
        fig, axes = plt.subplots(nfam, 1, figsize=(13, 3.2 * nfam), squeeze=False)
        x = np.arange(len(KT_LABELS))
        for r, (label, fam) in enumerate(families.items()):
            ax = axes[r, 0]
            ref = [fam["ref_pct"][l] for l in KT_LABELS]
            fut = [fam["fut_pct"][l] for l in KT_LABELS]
            ax.bar(x - 0.2, ref, 0.4, label="Reference", color="#4477aa")
            ax.bar(x + 0.2, fut, 0.4, label=f"{self._level:g}°C", color="#cc6677")
            ax.set_xticks(x)
            ax.set_xticklabels(KT_LABELS, fontsize=8)
            ax.set_ylabel("% land area")
            ax.set_title(f"{label} ensemble", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{self.title} — Land-area Fraction by Type", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        meta = self._build_metadata(
            title=f"{self.title} — Area Shift", figure_id="kt_area_shift_bar",
            models=[m for f in families.values() for m in f["members"]],
            description="Reference vs warming-level % land area per KT type, per family ensemble mean.",
            period=self.period, plot_type="bar",
        )
        return fig, meta

    def _plot_net_gain_loss(self, results: dict) -> tuple[plt.Figure, dict]:
        """Net change in % land area per KT type (future − reference), per family."""
        families = results["families"]
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(KT_LABELS))
        width = 0.8 / max(len(families), 1)
        for i, (label, fam) in enumerate(families.items()):
            delta = [fam["fut_pct"][l] - fam["ref_pct"][l] for l in KT_LABELS]
            ax.bar(x + i * width, delta, width, label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(KT_LABELS, fontsize=8)
        ax.set_ylabel("Δ % land area (future − reference)")
        ax.set_title(f"{self.title} — Net Gain/Loss per Type at {self._level:g}°C")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{self.title} — Net Gain/Loss", figure_id="kt_net_gain_loss",
            models=[m for f in families.values() for m in f["members"]],
            description="Net change in % land area per KT type (warming-level − reference), per family.",
            period=self.period, plot_type="bar",
        )
        return fig, meta

    def _plot_transition_matrices(self, results: dict) -> tuple[plt.Figure, dict]:
        """KT transition matrix (reference → future) per family ensemble mean."""
        families = results["families"]
        nfam = len(families)
        fig, axes = plt.subplots(1, nfam, figsize=(6.2 * nfam, 5.6), squeeze=False)
        for c, (label, fam) in enumerate(families.items()):
            ax = axes[0, c]
            M = np.array(fam["transition"])
            Mshow = np.where(M > 0, M, np.nan)
            im = ax.imshow(Mshow, cmap="viridis", vmin=0)
            ax.set_xticks(np.arange(len(KT_LABELS)))
            ax.set_xticklabels(KT_LABELS, fontsize=6, rotation=90)
            ax.set_yticks(np.arange(len(KT_LABELS)))
            ax.set_yticklabels(KT_LABELS, fontsize=6)
            ax.set_xlabel(f"{self._level:g}°C type")
            ax.set_ylabel("Reference type")
            ax.set_title(f"{label}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% land area")
        fig.suptitle(f"{self.title} — Transition Matrices (ref → {self._level:g}°C)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        meta = self._build_metadata(
            title=f"{self.title} — Transition Matrix", figure_id="kt_transition_matrix",
            models=[m for f in families.values() for m in f["members"]],
            description=(
                "Area-weighted KT transition matrix per family ensemble mean: "
                "rows = reference type, columns = warming-level type, values = "
                "% of classified land area."
            ),
            period=self.period, plot_type="heatmap",
        )
        return fig, meta


def _modal_window(windows: list[tuple[int, int]]) -> tuple[int, int]:
    """Most common window among members (windows are identical within a GCM)."""
    from collections import Counter
    return Counter(windows).most_common(1)[0][0]
