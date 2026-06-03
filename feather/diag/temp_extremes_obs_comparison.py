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

# Map each canonical variable to (Berkeley obs_datasets key, CRU file var).
_VAR_SOURCES = {
    "tasmax": ("BERKELEY_EARTH_LAND_TMAX", "tmx"),
    "tasmin": ("BERKELEY_EARTH_LAND_TMIN", "tmn"),
}


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

    def _spec(self, var: str) -> CompareSpec:
        return CompareSpec(
            var=var, ref_name="Berkeley Earth", sec_name="CRU", sec_token="cru",
            units_label="°C", display_offset=-273.15,
            abs_cmap="cmo.thermal", diff_cmap="RdBu_r", trend_cmap="coolwarm",
            land_only=True, sec_color=_CRU_COLOR,
        )

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        all_ids = []
        for var in self._vars:
            all_ids.append(f"{var}_obs_timeseries")
            for pk in ["annual", "djf", "jja"]:
                all_ids += [
                    f"{var}_cru_{pk}_trends",
                    f"{var}_cru_{pk}_trend_diffs",
                    f"{var}_cru_{pk}_clim",
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
        be_key, cru_var = _VAR_SOURCES[var]
        spec = self._spec(var)
        engine = PairwiseObsComparison(self, spec, self.PERIOD_SHORT, self.PERIOD_LONG)

        be_s = self.obs_loader.load_berkeley_hr(be_key, self.PERIOD_SHORT)
        be_l = self.obs_loader.load_berkeley_hr(be_key, self.PERIOD_LONG)
        cru_s = self.obs_loader.load_cru(cru_var, self.PERIOD_SHORT)
        cru_l = self.obs_loader.load_cru(cru_var, self.PERIOD_LONG)

        results = engine.compute(be_s, be_l, cru_s, cru_l)
        engine.export_netcdf(results, self._netcdf_dir)
        return {"engine": engine, "results": results}

    def _plot_var(self, var, results):
        engine = results["engine"]
        res = results["results"]
        figures = engine.figures(res)
        figures.append(self._plot_timeseries(var, res))
        return figures

    def _plot_timeseries(self, var, res):
        be = area_weighted_annual_series(res["fields"]["ref_long"]) - 273.15
        cru = area_weighted_annual_series(res["fields"]["sec_long"]) - 273.15
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(be.year.values, be.values, color=OBS_COLOR, lw=2.0,
                label=f"Berkeley Earth ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
        ax.plot(cru.year.values, cru.values, color=_CRU_COLOR, lw=2.0, ls="--",
                label=f"CRU TS ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Land-mean {var} (°C)")
        ax.set_title(
            f"{var}: Berkeley Earth Land vs CRU TS — Land-Mean Annual Series")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{var} Land-Mean Annual Series — Berkeley Earth vs CRU TS",
            figure_id=f"{var}_obs_timeseries", models=[], variables=[var],
            description=(
                f"Land-mean annual {var} (°C) from Berkeley Earth Land and "
                f"CRU TS ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})."
            ),
            plot_type="timeseries", period=self.PERIOD_LONG,
            obs_dataset="Berkeley Earth, CRU",
        )
        return fig, meta
