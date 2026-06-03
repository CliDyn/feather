"""Tropical Nights Index diagnostic (climdex TN20p).

Annual count of days where daily minimum 2m temperature (tasmin) exceeds
20 °C (293.15 K) — the climdex Tropical Nights index.  Computed over
land only using Berkeley Earth's land mask.

Produces figures across 4 groups:
A (x1): Multi-panel climatology map — one panel per model (days/year).
B (x1): Mean Tmin bias maps (model − Berkeley Earth Land TMIN, °C).
         Skipped when the BE obs file is not accessible.
C (x1): Global-mean time series — area-weighted annual TN count per year.
D (x1): Zonal mean profile — latitude profile of mean TN count.

Per-model annual TN-count fields are saved to NetCDF for downstream
climate change analysis.  The checkpoint directory is:
  ``{output_dir}/tropical_nights/``
  (outside the figures tree so it persists across re-runs).

Berkeley Earth Land TMIN obs:
  Configured via ``obs_datasets.BERKELEY_EARTH_TMIN.path`` in the config
  YAML, or auto-detected at the well-known Levante path:
  ``/work/bm1344/AWI/OBS/berkeleyearth/Land_TMIN_Gridded_0p25deg.nc``
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.diag._extremes_obs import (
    era5_mean_available,
    load_era5_mean,
    obs_ref_label,
    use_era5_obs,
)
from feather.util.spatial import compute_latlon_areas, latlon_global_mean

logger = logging.getLogger(__name__)

# Tropical Nights threshold (20 °C in Kelvin)
_TN_THRESHOLD_K: float = 293.15

# Conversion: add this to °C to get K (= subtract to get °C from K)
_K_TO_C: float = 273.15

# Well-known Levante path for the beta Berkeley Earth Land TMIN dataset
_BE_TMIN_LAND_PATH = Path(
    "/work/bm1344/AWI/OBS/berkeleyearth/Land_TMIN_Gridded_0p25deg.nc"
)


@register
class TropicalNightsDiag(DiagnosticBase):
    """Tropical Nights Index — annual count of days with TN > 20 °C.

    Uses daily minimum 2m temperature (CMOR ``tasmin``, ``day`` table).
    Ocean pixels are masked using Berkeley Earth's land mask.
    Models that do not publish daily tasmin are gracefully skipped.
    Per-model results are written to NetCDF for climate change analysis.
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

    # ── Paths ─────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-model TN NetCDF files (outside figures tree)."""
        return Path(self.config.output_dir) / "tropical_nights"

    def _nc_path(self, model: str) -> Path:
        start, end = self.period
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_tn_annual_{start}_{end}.nc"

    def _be_tmin_path(self) -> Path | None:
        """Return BE Land TMIN file path if accessible, otherwise None."""
        ds_cfg = self.config.obs_datasets.get("BERKELEY_EARTH_TMIN", {})
        if ds_cfg:
            p = Path(ds_cfg.get("path", ""))
            if p.exists():
                return p
        if _BE_TMIN_LAND_PATH.exists():
            return _BE_TMIN_LAND_PATH
        return None

    # ── Orchestration ─────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute per model → save NC → plot → save figures."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        has_obs = (
            era5_mean_available(self.config, "tasmin")
            or self._be_tmin_path() is not None
        )
        fig_ids = [
            "tropical_nights_climatology",
            "tropical_nights_timeseries",
            "tropical_nights_zonal_mean",
        ]
        if has_obs:
            fig_ids.append("tropical_nights_tmin_bias")

        if skip_existing and all(self._figure_exists(fid) for fid in fig_ids):
            logger.info("  All TN figures exist — skipping")
            return saved

        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Load daily tasmin per model, count TN days, save NC checkpoints.

        Returns
        -------
        dict with keys:

        - ``tn_clim``        : dict[model → DataArray(lat, lon)]
        - ``tn_series``      : dict[model → DataArray(year)]
        - ``tn_zonal``       : dict[model → DataArray(lat)]
        - ``model_mean_tmin``: dict[model → DataArray(lat, lon)] — K, land only
        - ``obs_mean_tmin``  : DataArray(lat, lon) | None — K, BE land obs
        - ``lat``            : DataArray — shared lat coordinate
        - ``lon``            : DataArray — shared lon coordinate
        - ``models``         : list[str] — models with successful data load
        """
        tn_clim: dict[str, xr.DataArray] = {}
        tn_series: dict[str, xr.DataArray] = {}
        tn_zonal: dict[str, xr.DataArray] = {}
        model_mean_tmin: dict[str, xr.DataArray] = {}
        lat_coord: xr.DataArray | None = None
        lon_coord: xr.DataArray | None = None

        for model in self.config.models:
            try:
                tn_annual, mean_tmin = self._load_or_compute(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue

            clim = tn_annual.mean("year")
            tn_clim[model] = clim
            model_mean_tmin[model] = mean_tmin

            if lat_coord is None:
                lat_coord = tn_annual["lat"]
                lon_coord = tn_annual["lon"]

            # Area-weighted global land mean per year
            areas = compute_latlon_areas(
                np.asarray(tn_annual["lat"]),
                np.asarray(tn_annual["lon"]),
            )
            areas_da = xr.DataArray(areas, dims=("lat", "lon"))
            tn_series[model] = latlon_global_mean(tn_annual, areas_da)

            tn_zonal[model] = clim.mean("lon")

        # Load obs mean Tmin for bias Group B (ERA5 if configured, else BE)
        obs_mean_tmin = None
        if lat_coord is not None:
            if use_era5_obs(self.config):
                obs_mean_tmin = load_era5_mean(
                    self.config, "tasmin", self.period,
                    lat_coord.values, lon_coord.values,
                )
            else:
                obs_mean_tmin = self._load_be_mean_tmin(
                    lat_coord.values, lon_coord.values,
                )

        return {
            "tn_clim": tn_clim,
            "tn_series": tn_series,
            "tn_zonal": tn_zonal,
            "model_mean_tmin": model_mean_tmin,
            "obs_mean_tmin": obs_mean_tmin,
            "lat": lat_coord,
            "lon": lon_coord,
            "models": list(tn_clim.keys()),
        }

    def _load_or_compute(
        self, model: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return ``(tn_annual, mean_tmin)``, loading from NC or computing fresh.

        When a checkpoint NC exists and contains both ``tn_count`` and
        ``tmin_mean``, data is loaded directly.  Otherwise, daily tasmin
        is loaded, both fields computed (land-masked), and saved to NC.
        """
        nc_path = self._nc_path(model)
        if nc_path.exists():
            logger.info("  %s: loading TN count from %s", model, nc_path.name)
            ds = xr.open_dataset(nc_path)
            if "tmin_mean" in ds:
                return ds["tn_count"], ds["tmin_mean"]
            logger.info("  %s: NC missing tmin_mean — recomputing", model)

        logger.info("  %s: computing TN count from daily tasmin", model)
        da = self.model_loader.load_var(
            model, "tasmin", table="day", period=self.period,
        )

        tn_raw = self._count_tn_days(da)
        mean_tmin_raw = da.mean("time")

        # Land-only masking
        land_mask = self._load_land_mask(
            model, np.asarray(da["lat"]), np.asarray(da["lon"]),
        )
        if land_mask is not None:
            tn_raw = tn_raw.where(land_mask)
            mean_tmin_raw = mean_tmin_raw.where(land_mask)

        # Compute both in one dask pass when possible
        try:
            import dask
            tn_annual, mean_tmin = dask.compute(tn_raw, mean_tmin_raw)
        except (ImportError, AttributeError):
            tn_annual = tn_raw.compute() if hasattr(tn_raw, "compute") else tn_raw
            mean_tmin = (
                mean_tmin_raw.compute()
                if hasattr(mean_tmin_raw, "compute")
                else mean_tmin_raw
            )

        self._save_nc(model, tn_annual, mean_tmin)
        return tn_annual, mean_tmin

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

    def _load_model_land_mask(
        self,
        model: str,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
    ) -> xr.DataArray | None:
        """Boolean land mask from the model's own ``sftlf`` (fx, %), or None.

        Land where land area fraction > 50 %.  Returns None (so the caller can
        fall back to the Berkeley mask) if the model has no ``sftlf`` field.
        """
        try:
            sftlf = self.model_loader.load_var(model, "sftlf", table="fx")
        except (KeyError, FileNotFoundError, OSError, AttributeError,
                ValueError, TypeError):
            return None
        try:
            sftlf = sftlf.squeeze(drop=True)
            for d in ("time", "height", "depth"):
                if d in sftlf.dims:
                    sftlf = sftlf.isel({d: 0})
            mask = sftlf.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="nearest", kwargs={"fill_value": 0.0},
            )
            return mask > 50.0
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("  %s: could not use sftlf land mask: %s", model, exc)
            return None

    def _load_land_mask(
        self,
        model: str,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
    ) -> xr.DataArray | None:
        """Boolean land mask (True = land) on the model lat/lon grid.

        Prefers the model's own land-sea mask (CMOR ``sftlf``, ``fx`` table,
        land area fraction in %), e.g. the derived ERA5 mask.  Falls back to
        the Berkeley Earth ``land_mask`` when ``sftlf`` is unavailable, and to
        ``None`` if neither can be loaded.
        """
        model_mask = self._load_model_land_mask(model, model_lat, model_lon)
        if model_mask is not None:
            return model_mask

        path = self._be_tmin_path()
        if path is None:
            return None
        try:
            ds = xr.open_dataset(path)
            mask = ds["land_mask"].rename({"latitude": "lat", "longitude": "lon"})
            if float(mask.lon.min()) < 0:
                mask = mask.assign_coords(
                    lon=((mask.lon + 360) % 360),
                ).sortby("lon")
            mask_interp = mask.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="nearest",
                kwargs={"fill_value": 0.0},
            )
            return mask_interp > 0.5
        except Exception as exc:
            logger.warning("  Could not load BE land mask: %s", exc)
            return None

    def _load_be_mean_tmin(
        self,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
    ) -> xr.DataArray | None:
        """Load BE Land TMIN, reconstruct absolute temp, return period mean (K).

        Reconstructs absolute monthly Tmin as ``anomaly + climatology[month]``,
        computes the mean over the analysis period, and interpolates to the
        model grid.  Returns None if the file is not accessible.
        """
        import pandas as pd

        path = self._be_tmin_path()
        if path is None:
            return None
        try:
            ds = xr.open_dataset(path, chunks="auto")

            # Decimal-year time → DatetimeIndex
            dec_years = ds["time"].values
            years_int = dec_years.astype(int)
            months_int = np.floor((dec_years - years_int) * 12).astype(int) + 1
            months_int = np.clip(months_int, 1, 12)
            datetimes = pd.to_datetime(
                [f"{y:04d}-{m:02d}-01" for y, m in zip(years_int, months_int)]
            )
            ds = ds.assign_coords(time=datetimes)

            # Slice to analysis period
            start, end = self.period
            anom = ds["temperature"].sel(time=slice(start, end))
            clim = ds["climatology"]  # (month_number, latitude, longitude)

            # Reconstruct absolute temperature: anomaly + climatology[month]
            month_idx = anom.time.dt.month.values - 1   # 0-based
            clim_np = clim.values                        # (12, nlat, nlon)
            clim_matched = clim_np[month_idx]            # (ntime, nlat, nlon)
            abs_temp = anom + xr.DataArray(
                clim_matched, dims=anom.dims, coords=anom.coords,
            )

            # Rename dims and shift lon to 0..360
            rename = {}
            if "latitude" in abs_temp.dims:
                rename["latitude"] = "lat"
            if "longitude" in abs_temp.dims:
                rename["longitude"] = "lon"
            if rename:
                abs_temp = abs_temp.rename(rename)
            if float(abs_temp.lon.min()) < 0:
                abs_temp = abs_temp.assign_coords(
                    lon=((abs_temp.lon + 360) % 360),
                ).sortby("lon")

            # Period mean in °C → K; interpolate to model grid
            mean_k = (abs_temp.mean("time").compute() + _K_TO_C)
            mean_k_interp = mean_k.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
            return mean_k_interp

        except Exception as exc:
            logger.warning("  Could not load BE Land TMIN obs: %s", exc)
            return None

    def _save_nc(
        self,
        model: str,
        tn_annual: xr.DataArray,
        mean_tmin: xr.DataArray,
    ) -> None:
        """Write per-model TN annual count + mean Tmin to NetCDF checkpoint."""
        nc_path = self._nc_path(model)
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        start, end = self.period
        mean_tmin_out = mean_tmin.assign_attrs({
            "long_name": "Period-mean Daily Minimum Temperature",
            "units": "K",
            "note": "Ocean pixels are NaN (land only)",
        })
        ds = xr.Dataset(
            {"tn_count": tn_annual, "tmin_mean": mean_tmin_out},
            attrs={
                "Conventions": "CF-1.8",
                "title": (
                    f"Tropical Nights Index — {model} ({start}–{end})"
                ),
                "institution": "Feather climate evaluation framework",
                "source": "feather/diag/tropical_nights.py",
                "model": model,
                "period_start": start,
                "period_end": end,
                "threshold": f"tasmin > {_TN_THRESHOLD_K} K (20 °C)",
                "reference": "climdex — Alexander et al. (2006)",
                "land_only": "True — ocean pixels are NaN",
            },
        )
        ds.to_netcdf(nc_path)
        logger.info("  Saved TN NetCDF: %s", nc_path)

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate climatology map, time series, zonal mean, and bias figures."""
        figs: list[tuple[plt.Figure, dict]] = []
        models = results["models"]
        if not models:
            logger.warning("TropicalNightsDiag: no model data — no figures produced")
            return figs

        figs.append(self._plot_climatology(results))
        figs.append(self._plot_timeseries(results))
        figs.append(self._plot_zonal_mean(results))

        if results.get("obs_mean_tmin") is not None:
            figs.append(self._plot_tmin_bias(results))

        return figs

    def _plot_climatology(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group A: multi-panel mean TN count map."""
        models = results["models"]
        data_dict = {m: results["tn_clim"][m] for m in models}
        fig, _ = plot_combined_map(
            data_dict,
            title=f"{self.title} — Mean Annual Count (land only)",
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
                "temperature exceeds 20 °C). Land-only; computed from CMOR daily tasmin."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_timeseries(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group C: global land-mean annual TN count time series."""
        models = results["models"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for model in models:
            color = self.config.get_model_color(model)
            series = results["tn_series"][model]
            ax.plot(
                np.asarray(series["year"]), np.asarray(series),
                color=color, lw=1.5, label=model,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Tropical Nights (days/year)")
        ax.set_title(f"{self.title} — Global Land Mean")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{self.title} — Global Land Mean Time Series",
            figure_id="tropical_nights_timeseries",
            models=models,
            description=(
                "Area-weighted global land-mean annual count of Tropical Nights "
                "(TN > 20 °C) for each model."
            ),
            period=self.period,
            plot_type="timeseries",
        )
        return fig, meta

    def _plot_zonal_mean(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group D: zonal mean TN count profile."""
        models = results["models"]
        fig, ax = plt.subplots(figsize=(5, 7))
        for model in models:
            color = self.config.get_model_color(model)
            zonal = results["tn_zonal"][model]
            lats = np.asarray(
                zonal["lat"] if "lat" in zonal.dims else results["lat"]
            )
            ax.plot(np.asarray(zonal), lats, color=color, lw=1.5, label=model)
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
            description="Zonal mean of the mean annual Tropical Nights count by latitude.",
            period=self.period,
            plot_type="zonal_profile",
        )
        return fig, meta

    def _plot_tmin_bias(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group B: mean daily Tmin bias maps (model − Berkeley Earth Land TMIN).

        Compares the climatological mean daily minimum temperature from
        each model against Berkeley Earth Land TMIN monthly observations.
        Both are in °C for display; differences are in °C.
        """
        models = results["models"]
        obs_k = results["obs_mean_tmin"]         # (lat, lon) in K
        obs_c = obs_k - _K_TO_C                  # display in °C
        obs_label = obs_ref_label(self.config, "tasmin")

        bias_dict = {
            m: results["model_mean_tmin"][m] - obs_k   # K difference = °C difference
            for m in models
        }

        fig, _ = plot_combined_bias_map(
            obs_c,
            bias_dict,
            title=f"{self.title} — Mean Tmin Bias vs {obs_label}",
            obs_title=obs_label,
            cmap="cmo.thermal",
            bias_cmap="RdBu_r",
            units="°C",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Mean Tmin Bias",
            figure_id="tropical_nights_tmin_bias",
            models=models,
            description=(
                "Bias in climatological mean daily minimum temperature "
                f"(model − {obs_label}, °C). "
                "Both model and obs are land-only. Model from CMOR daily tasmin; "
                f"obs mean from {obs_label}."
            ),
            obs_dataset=("ERA5_TMINMAX" if use_era5_obs(self.config)
                         else "BERKELEY_EARTH_TMIN"),
            obs_variable=("tasmin" if use_era5_obs(self.config)
                          else "temperature"),
            period=self.period,
            plot_type="bias_map",
        )
        return fig, meta
