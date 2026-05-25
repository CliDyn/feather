"""Tropical Nights Index diagnostic (climdex TN20p).

Annual count of days where daily minimum 2m temperature (tasmin) exceeds
20 °C (293.15 K) — the climdex Tropical Nights index.

Produces figures across 3 groups:
A (x1): Multi-panel climatology map — one panel per model showing the
         mean annual TN count (days/year).
C (x1): Global-mean time series — area-weighted annual TN count per year.
D (x1): Zonal mean profile — latitude profile of mean TN count.

Group B (bias maps vs obs) is generated only when a daily tasmin
observation dataset is configured under ``ERA5_DAILY_TASMIN`` in
``obs_datasets``; this dataset is not yet available at the standard
Levante paths, so Group B is silently skipped by default.

Annual TN-count fields per model are saved to NetCDF for downstream
climate change analysis:
  ``{output_dir}/figures/tropical_nights/{model}_tn_annual_{start}_{end}.nc``
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_map
from feather.util.spatial import compute_latlon_areas, latlon_global_mean

logger = logging.getLogger(__name__)

# Tropical Nights threshold (20 °C in Kelvin)
_TN_THRESHOLD_K: float = 293.15

# Conversion: K → °C offset for display labels only
_K_TO_C: float = 273.15


@register
class TropicalNightsDiag(DiagnosticBase):
    """Tropical Nights Index — annual count of days with TN > 20 °C.

    Uses daily minimum 2m temperature (CMOR ``tasmin``, ``day`` table).
    Models that do not publish daily data are gracefully skipped.
    Results per model are written to NetCDF for climate change analysis.
    """

    name = "tropical_nights"
    title = "Tropical Nights Index (TN > 20 °C)"
    domain = "sfc"
    variables = ["tasmin"]
    group = "extremes"

    def __init__(
        self,
        model_loader,
        obs_loader,
        config,
        *,
        cmip6_loader=None,
        experiment: str = "baseline_hist",
        period: tuple[str, str] = ("1990", "2014"),
    ):
        super().__init__(model_loader, obs_loader, config, cmip6_loader=cmip6_loader)
        self.experiment = experiment
        self.period = period

    # ── NC checkpoint paths ────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-model TN-count NetCDF files."""
        return self.output_dir

    def _nc_path(self, model: str) -> Path:
        start, end = self.period
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_tn_annual_{start}_{end}.nc"

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute per model → save NC → plot → save figures."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Figure IDs we produce
        fig_ids = [
            "tropical_nights_climatology",
            "tropical_nights_timeseries",
            "tropical_nights_zonal_mean",
        ]
        if skip_existing and all(self._figure_exists(fid) for fid in fig_ids):
            logger.info("  All TN figures exist — skipping")
            return saved

        results = self.compute()
        pairs = self.plot(results)
        for fig, meta in pairs:
            pair = self._save(fig, meta, meta["figure_id"])
            saved.append(pair)
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Load daily tasmin per model, count TN days, save NC checkpoints.

        Returns a dict with keys:
        - ``tn_clim``   : dict[model, DataArray(lat, lon)] — mean TN count
        - ``tn_series`` : dict[model, DataArray(year)]     — global-mean TN
        - ``tn_zonal``  : dict[model, DataArray(lat)]      — zonal-mean TN
        - ``lat``       : DataArray — shared latitude coordinate
        - ``lon``       : DataArray — shared longitude coordinate
        - ``models``    : list[str] — models with data
        """
        tn_clim: dict[str, xr.DataArray] = {}
        tn_series: dict[str, xr.DataArray] = {}
        tn_zonal: dict[str, xr.DataArray] = {}
        lat_coord: xr.DataArray | None = None
        lon_coord: xr.DataArray | None = None

        for model in self.config.models:
            try:
                tn_annual = self._load_or_compute(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue

            # Mean climatology (lat, lon)
            clim = tn_annual.mean("year")
            tn_clim[model] = clim

            if lat_coord is None:
                lat_coord = tn_annual["lat"]
                lon_coord = tn_annual["lon"]

            # Area-weighted global-mean time series (year,)
            areas = compute_latlon_areas(
                np.asarray(tn_annual["lat"]),
                np.asarray(tn_annual["lon"]),
            )
            areas_da = xr.DataArray(areas, dims=("lat", "lon"))
            series = latlon_global_mean(tn_annual, areas_da)
            tn_series[model] = series

            # Zonal mean
            tn_zonal[model] = clim.mean("lon")

        return {
            "tn_clim": tn_clim,
            "tn_series": tn_series,
            "tn_zonal": tn_zonal,
            "lat": lat_coord,
            "lon": lon_coord,
            "models": list(tn_clim.keys()),
        }

    def _load_or_compute(self, model: str) -> xr.DataArray:
        """Return per-year TN count for *model*, loading NC if available."""
        nc_path = self._nc_path(model)
        if nc_path.exists():
            logger.info("  %s: loading TN count from %s", model, nc_path.name)
            ds = xr.open_dataset(nc_path)
            return ds["tn_count"]

        logger.info("  %s: computing TN count from daily tasmin", model)
        da = self.model_loader.load_var(
            model, "tasmin", table="day", period=self.period,
        )
        tn_annual = self._count_tn_days(da)
        self._save_nc(model, tn_annual)
        return tn_annual

    @staticmethod
    def _count_tn_days(da: xr.DataArray) -> xr.DataArray:
        """Count days with tasmin > 20 °C, grouped by year.

        Parameters
        ----------
        da : DataArray
            Daily tasmin in Kelvin, dims ``(time, lat, lon)``.

        Returns
        -------
        DataArray
            Annual count, dims ``(year, lat, lon)``, units ``days/year``.
        """
        exceed = (da > _TN_THRESHOLD_K).astype(np.int16)
        annual = exceed.groupby("time.year").sum("time")
        annual.name = "tn_count"
        annual.attrs = {
            "long_name": "Tropical Nights count (TN > 20 °C)",
            "units": "days/year",
            "threshold_K": _TN_THRESHOLD_K,
        }
        return annual

    def _save_nc(self, model: str, tn_annual: xr.DataArray) -> None:
        """Write per-model TN annual count to NetCDF checkpoint."""
        nc_path = self._nc_path(model)
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        start, end = self.period
        ds = xr.Dataset(
            {"tn_count": tn_annual},
            attrs={
                "Conventions": "CF-1.8",
                "title": (
                    f"Tropical Nights Index — {model} "
                    f"({start}–{end})"
                ),
                "institution": "Feather climate evaluation framework",
                "source": "feather/diag/tropical_nights.py",
                "model": model,
                "period_start": start,
                "period_end": end,
                "threshold": f"tasmin > {_TN_THRESHOLD_K} K (20 °C)",
                "reference": "climdex — Alexander et al. (2006)",
            },
        )
        ds.to_netcdf(nc_path)
        logger.info("  Saved TN NetCDF: %s", nc_path)

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate climatology map, time series, and zonal mean figures."""
        figs: list[tuple[plt.Figure, dict]] = []
        models = results["models"]
        if not models:
            logger.warning("TropicalNightsDiag: no model data — no figures produced")
            return figs

        figs.append(self._plot_climatology(results))
        figs.append(self._plot_timeseries(results))
        figs.append(self._plot_zonal_mean(results))
        return figs

    def _plot_climatology(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group A: multi-panel mean TN count map."""
        models = results["models"]
        tn_clim = results["tn_clim"]
        colors = {m: self.config.get_model_color(m) for m in models}

        data_dict = {m: tn_clim[m] for m in models}
        fig, _ = plot_combined_map(
            data_dict,
            title=f"{self.title} — Mean Annual Count",
            cmap="YlOrRd",
            vmin=0,
            units="days/year",
        )

        meta = self._build_metadata(
            title=f"{self.title} — Mean Annual Count",
            figure_id="tropical_nights_climatology",
            models=models,
            description=(
                "Mean annual count of Tropical Nights (days where daily minimum "
                "temperature exceeds 20 °C). Computed from CMOR daily tasmin."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_timeseries(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group C: global-mean annual TN count time series."""
        models = results["models"]
        tn_series = results["tn_series"]

        fig, ax = plt.subplots(figsize=(10, 4))
        for model in models:
            color = self.config.get_model_color(model)
            series = tn_series[model]
            years = np.asarray(series["year"])
            ax.plot(years, np.asarray(series), color=color, lw=1.5, label=model)

        ax.set_xlabel("Year")
        ax.set_ylabel("Tropical Nights (days/year)")
        ax.set_title(f"{self.title} — Global Mean")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — Global Mean Time Series",
            figure_id="tropical_nights_timeseries",
            models=models,
            description=(
                "Area-weighted global-mean annual count of Tropical Nights "
                "(TN > 20 °C) for each model."
            ),
            period=self.period,
            plot_type="timeseries",
        )
        return fig, meta

    def _plot_zonal_mean(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group D: zonal mean TN count profile."""
        models = results["models"]
        tn_zonal = results["tn_zonal"]
        lat = results["lat"]

        fig, ax = plt.subplots(figsize=(5, 7))
        for model in models:
            color = self.config.get_model_color(model)
            lats = np.asarray(tn_zonal[model]["lat"] if "lat" in tn_zonal[model].dims
                              else lat)
            vals = np.asarray(tn_zonal[model])
            ax.plot(vals, lats, color=color, lw=1.5, label=model)

        ax.set_xlabel("Tropical Nights (days/year)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{self.title} — Zonal Mean")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — Zonal Mean",
            figure_id="tropical_nights_zonal_mean",
            models=models,
            description=(
                "Zonal (longitude) mean of the annual Tropical Nights count "
                "as a function of latitude."
            ),
            period=self.period,
            plot_type="zonal_profile",
        )
        return fig, meta
