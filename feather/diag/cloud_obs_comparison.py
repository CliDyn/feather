"""Cloud-cover observational comparison: ERA5 vs CRU TS.

Compares total cloud cover (``clt``, %) from ERA5 against the CRU TS v4.09
land cloud-cover field on a common 0.5° land-masked grid.  CRU is land-only,
so the ERA5 reference is masked to the CRU coverage for a fair comparison.

Figures (per seasonal window annual/DJF/JJA): trend maps, trend differences,
climatology + bias maps — plus one global land-mean time series.  Climatology
and difference fields are also exported as NetCDF.

Periods: short 1981–2014 (model-comparable), long 1981–2023.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


@register
class CloudObsComparisonDiag(DiagnosticBase):
    """ERA5 vs CRU TS total cloud cover comparison (land, 0.5°)."""

    name = "cloud_obs_comparison"
    title = "Cloud Cover Dataset Comparison (ERA5 vs CRU TS, land 0.5°)"
    domain = "sfc"
    variables = ["clt"]
    group = "clouds"

    PERIOD_SHORT = ("1981", "2014")
    PERIOD_LONG = ("1981", "2023")

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1981", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        self.experiment = experiment

    @property
    def _netcdf_dir(self) -> Path:
        return Path(self.config.output_dir) / "netcdf" / self.name

    def _spec(self) -> CompareSpec:
        return CompareSpec(
            var="clt", ref_name="ERA5", sec_name="CRU", sec_token="cru",
            units_label="%", abs_cmap="Greys_r", diff_cmap="RdBu_r",
            trend_cmap="RdBu_r", land_only=True, sec_color=_CRU_COLOR,
        )

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        all_ids = ["clt_obs_timeseries"]
        for pk in ["annual", "djf", "jja"]:
            all_ids += [
                f"clt_cru_{pk}_trends",
                f"clt_cru_{pk}_trend_diffs",
                f"clt_cru_{pk}_clim",
            ]
        if skip_existing and all(self._figure_exists(f) for f in all_ids):
            logger.info("Skipping %s — all figures exist", self.name)
            return [(self.output_dir / f"{f}.png", self.output_dir / f"{f}.json")
                    for f in all_ids]
        try:
            results = self.compute()
            for fig, meta in self.plot(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))
                plt.close(fig)
        except Exception:
            logger.error("Diagnostic %s failed", self.name, exc_info=True)
        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Computation ────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        spec = self._spec()
        engine = PairwiseObsComparison(self, spec, self.PERIOD_SHORT, self.PERIOD_LONG)

        era5_s = self._load_obs_var("clt", self.PERIOD_SHORT)
        era5_l = self._load_obs_var("clt", self.PERIOD_LONG)
        era5_s = self._to_latlon360(era5_s)
        era5_l = self._to_latlon360(era5_l)
        cru_s = self.obs_loader.load_cru("cld", self.PERIOD_SHORT)
        cru_l = self.obs_loader.load_cru("cld", self.PERIOD_LONG)

        results = engine.compute(era5_s, era5_l, cru_s, cru_l)
        engine.export_netcdf(results, self._netcdf_dir)
        return {"engine": engine, "results": results}

    @staticmethod
    def _to_latlon360(da: xr.DataArray) -> xr.DataArray:
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)
        if float(da.lon.min()) < 0:
            da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")
        return da

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        engine = results["engine"]
        res = results["results"]
        figures = engine.figures(res)
        figures.append(self._plot_timeseries(res))
        return figures

    def _plot_timeseries(self, res) -> tuple[plt.Figure, dict]:
        era5 = area_weighted_annual_series(res["fields"]["ref_long"])
        cru = area_weighted_annual_series(res["fields"]["sec_long"])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(era5.year.values, era5.values, color=OBS_COLOR, lw=2.0,
                label=f"ERA5 ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
        ax.plot(cru.year.values, cru.values, color=_CRU_COLOR, lw=2.0, ls="--",
                label=f"CRU TS ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Land-mean cloud cover (%)")
        ax.set_title("Total Cloud Cover: ERA5 vs CRU TS — Land-Mean Annual Series")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title="Cloud Cover Land-Mean Annual Series — ERA5 vs CRU TS",
            figure_id="clt_obs_timeseries", models=[], variables=["clt"],
            description=(
                f"Land-mean annual total cloud cover (%) from ERA5 and CRU TS "
                f"({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})."
            ),
            plot_type="timeseries", period=self.PERIOD_LONG,
            obs_dataset="ERA5, CRU",
        )
        return fig, meta
