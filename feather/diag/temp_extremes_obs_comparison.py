"""Temperature-extremes observational comparison: Berkeley Earth vs CRU TS.

Compares daily maximum (``tasmax``) and minimum (``tasmin``) near-surface
temperature between the Berkeley Earth high-resolution **Land** TMAX/TMIN
products and the CRU TS v4.09 ``tmx``/``tmn`` fields.  Both datasets are
land-only; the comparison is performed on a common 0.5° land-masked grid with
Berkeley Earth as the reference.

For each variable and each seasonal window (annual/DJF/JJA): trend maps, trend
differences, climatology + bias maps; plus one land-mean time series per
variable.  Climatology and difference fields are exported as NetCDF.

Periods: short 1981–2014 (model-comparable), long 1981–2023.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from feather.diag._obs_compare_common import (
    CompareSpec,
    PairwiseObsComparison,
    area_weighted_annual_series,
)
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import OBS_COLOR

logger = logging.getLogger(__name__)

_CRU_COLOR = "#9467bd"
_ERA5_COLOR = "#1f77b4"

# Map each canonical variable to (Berkeley obs_datasets key, CRU file var).
_VAR_SOURCES = {
    "tasmax": ("BERKELEY_EARTH_LAND_TMAX", "tmx"),
    "tasmin": ("BERKELEY_EARTH_LAND_TMIN", "tmn"),
}

#: obs_datasets key holding the derived ERA5 monthly tasmin/tasmax files.
_ERA5_DATASET = "ERA5_TMINMAX"


@register
class TempExtremesObsComparisonDiag(DiagnosticBase):
    """Berkeley Earth Land TMAX/TMIN vs CRU TS tmx/tmn (land, 0.5°)."""

    name = "temp_extremes_obs_comparison"
    title = "Temperature Extremes Comparison (Berkeley Earth Land vs CRU TS)"
    domain = "sfc"
    variables = ["tasmax", "tasmin"]
    group = "temperature"

    PERIOD_SHORT = ("1981", "2014")
    PERIOD_LONG = ("1981", "2023")

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1981", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        self.experiment = experiment
        self._vars = variables or list(self.variables)

    @property
    def _netcdf_dir(self) -> Path:
        return Path(self.config.output_dir) / "netcdf" / self.name

    def _spec(self, var: str, sec_name: str, sec_token: str,
              sec_color: str) -> CompareSpec:
        return CompareSpec(
            var=var, ref_name="Berkeley Earth",
            sec_name=sec_name, sec_token=sec_token,
            units_label="°C", display_offset=-273.15,
            abs_cmap="cmo.thermal", diff_cmap="RdBu_r", trend_cmap="coolwarm",
            land_only=True, sec_color=sec_color,
        )

    def _secondaries(self, var: str):
        """Return (sec_name, token, color, loader) for each configured secondary.

        CRU TS is always present; the derived ERA5 monthly tasmin/tasmax is
        added when ``ERA5_TMINMAX`` is configured in ``obs_datasets``.
        """
        cru_var = _VAR_SOURCES[var][1]
        secs = [("CRU", "cru", _CRU_COLOR,
                 lambda p, v=cru_var: self.obs_loader.load_cru(v, p))]
        if _ERA5_DATASET in self.config.obs_datasets:
            secs.append(("ERA5", "era5", _ERA5_COLOR,
                         lambda p, v=var: self.obs_loader.load(_ERA5_DATASET, v, p)))
        return secs

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        all_ids = []
        for var in self._vars:
            all_ids.append(f"{var}_obs_timeseries")
            for _, token, _, _ in self._secondaries(var):
                for pk in ["annual", "djf", "jja"]:
                    all_ids += [
                        f"{var}_{token}_{pk}_trends",
                        f"{var}_{token}_{pk}_trend_diffs",
                        f"{var}_{token}_{pk}_clim",
                    ]
        if skip_existing and all(self._figure_exists(f) for f in all_ids):
            logger.info("Skipping %s — all figures exist", self.name)
            return [(self.output_dir / f"{f}.png", self.output_dir / f"{f}.json")
                    for f in all_ids]

        for var in self._vars:
            try:
                results = self._compute_var(var)
                for fig, meta in self._plot_var(var, results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
                    plt.close(fig)
            except Exception:
                logger.error("%s failed for %s", self.name, var, exc_info=True)
        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # base-class compute()/plot() satisfied by per-variable orchestration
    def compute(self) -> dict[str, Any]:  # pragma: no cover - run() drives work
        return {}

    def plot(self, results):  # pragma: no cover - run() drives work
        return []

    # ── Per-variable computation ───────────────────────────────────────

    def _compute_var(self, var: str) -> dict[str, Any]:
        be_key = _VAR_SOURCES[var][0]
        be_s = self.obs_loader.load_berkeley_hr(be_key, self.PERIOD_SHORT)
        be_l = self.obs_loader.load_berkeley_hr(be_key, self.PERIOD_LONG)

        per_sec = []  # (sec_name, color, engine, results)
        for sec_name, token, color, loader in self._secondaries(var):
            try:
                spec = self._spec(var, sec_name, token, color)
                engine = PairwiseObsComparison(
                    self, spec, self.PERIOD_SHORT, self.PERIOD_LONG)
                results = engine.compute(
                    be_s, be_l,
                    loader(self.PERIOD_SHORT), loader(self.PERIOD_LONG))
                engine.export_netcdf(results, self._netcdf_dir)
                per_sec.append((sec_name, color, engine, results))
            except (KeyError, FileNotFoundError, OSError):
                logger.warning("%s: secondary %s unavailable for %s — skipping",
                               self.name, sec_name, var)
        return {"secondaries": per_sec}

    def _plot_var(self, var, data):
        if not data["secondaries"]:
            return []
        figures = []
        for _, _, engine, results in data["secondaries"]:
            figures += engine.figures(results)
        figures.append(self._plot_timeseries(var, data["secondaries"]))
        return figures

    def _plot_timeseries(self, var, per_sec):
        fig, ax = plt.subplots(figsize=(12, 5))
        # Reference (Berkeley) — same across secondaries; take the first.
        be = area_weighted_annual_series(
            per_sec[0][3]["fields"]["ref_long"]) - 273.15
        ax.plot(be.year.values, be.values, color=OBS_COLOR, lw=2.0,
                label=f"Berkeley Earth ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
        names = ["Berkeley Earth"]
        for sec_name, color, _, results in per_sec:
            s = area_weighted_annual_series(results["fields"]["sec_long"]) - 273.15
            ax.plot(s.year.values, s.values, color=color, lw=2.0, ls="--",
                    label=f"{sec_name} ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
            names.append(sec_name)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Land-mean {var} (°C)")
        ax.set_title(
            f"{var}: Berkeley Earth Land vs {', '.join(names[1:])} — "
            f"Land-Mean Annual Series")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{var} Land-Mean Annual Series — {', '.join(names)}",
            figure_id=f"{var}_obs_timeseries", models=[], variables=[var],
            description=(
                f"Land-mean annual {var} (°C) from {', '.join(names)} "
                f"({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})."
            ),
            plot_type="timeseries", period=self.PERIOD_LONG,
            obs_dataset=", ".join(names),
        )
        return fig, meta
