"""Heatwave Indices diagnostic (TX90 method).

Five interconnected heatwave indices based on the 90th percentile of daily
maximum 2 m temperature (CMOR ``tasmax``, ``day`` table):

- **HWN** — Heatwave Number: count of distinct events per summer season.
- **HWF** — Heatwave Frequency: total heatwave days per summer season.
- **HWD** — Heatwave Duration: length of the longest event.
- **HWM** — Heatwave Magnitude: mean tasmax over all heatwave days.
- **HWA** — Heatwave Amplitude: peak tasmax across all heatwave days.

A heatwave event is a run of ≥ 3 consecutive days where tasmax exceeds
the local day-of-year 90th percentile.  The percentile threshold is
computed over the full analysis period (base period = model run period).

Summer season is hemisphere-aware:
  NH (lat ≥ 0): May–Sep
  SH (lat < 0): Nov–Mar (spanning calendar years; attributed to the year
                          ending in Jan–Mar)

Produces figures across 3 groups:
  A (×5): Climatological maps — one per index (HWN, HWF, HWD, HWM, HWA).
  B (×5): Annual time series of global-land-mean for each index.
  C (×1): Mean TMAX bias map — model vs Berkeley Earth Land TMAX (°C).
           Note: observed heatwave indices are not computable from monthly
           Berkeley Earth data; only mean TMAX bias is shown here.

Per-model annual index fields are saved to NetCDF for downstream climate
change analysis.  Checkpoint directory: ``{output_dir}/heatwave/``
(outside the figures tree).

Berkeley Earth Land TMAX obs path (configurable via
``obs_datasets.BERKELEY_EARTH_TMAX.path`` or auto-detected at the
well-known Levante path):
  ``/work/bm1344/AWI/OBS/berkeleyearth/Land_TMAX_Gridded_0p25deg.nc``
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

_K_TO_C: float = 273.15

_BE_TMAX_LAND_PATH = Path(
    "/work/bm1344/AWI/OBS/berkeleyearth/Land_TMAX_Gridded_0p25deg.nc"
)

# Summer month sets per hemisphere
_NH_SUMMER_MONTHS = frozenset([5, 6, 7, 8, 9])    # May–Sep
_SH_SUMMER_MONTHS = frozenset([11, 12, 1, 2, 3])  # Nov–Mar


@register
class HeatwaveDiag(DiagnosticBase):
    """Heatwave Indices — TX90 method (5 indices, annual, land only).

    Uses daily maximum 2 m temperature (CMOR ``tasmax``, ``day`` table).
    Ocean pixels are masked using Berkeley Earth's land mask.
    Models that do not publish daily tasmax are gracefully skipped.
    Per-model results are written to NetCDF for climate change analysis.
    """

    name = "heatwave"
    title = "Heatwave Indices (TX90)"
    domain = "sfc"
    variables = ["tasmax"]
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
        """Directory for per-model heatwave NetCDF files (outside figures tree)."""
        return Path(self.config.output_dir) / "heatwave"

    def _nc_path(self, model: str) -> Path:
        start, end = self.period
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_heatwave_tx90_{start}_{end}.nc"

    def _be_tmax_path(self) -> Path | None:
        """Return BE Land TMAX file path if accessible, otherwise None."""
        ds_cfg = self.config.obs_datasets.get("BERKELEY_EARTH_TMAX", {})
        if ds_cfg:
            p = Path(ds_cfg.get("path", ""))
            if p.exists():
                return p
        if _BE_TMAX_LAND_PATH.exists():
            return _BE_TMAX_LAND_PATH
        return None

    # ── Orchestration ─────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute per model → save NC → plot → save figures."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        has_obs = (
            era5_mean_available(self.config, "tasmax")
            or self._be_tmax_path() is not None
        )
        fig_ids = [
            f"heatwave_{idx}_{kind}"
            for idx in ("hwn", "hwf", "hwd", "hwm", "hwa")
            for kind in ("map", "timeseries")
        ]
        if has_obs:
            fig_ids.append("heatwave_tmax_bias")

        if skip_existing and all(self._figure_exists(fid) for fid in fig_ids):
            logger.info("  All heatwave figures exist — skipping")
            return saved

        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Load daily tasmax per model, compute HW indices, save NC checkpoints.

        Returns
        -------
        dict with keys:

        - ``hw_clim``        : dict[model → dict[index → DataArray(lat, lon)]]
        - ``hw_series``      : dict[model → dict[index → DataArray(year)]]
        - ``model_mean_tmax``: dict[model → DataArray(lat, lon)] — K, land only
        - ``obs_mean_tmax``  : DataArray(lat, lon) | None — K, BE land obs
        - ``lat``            : DataArray — shared lat coordinate
        - ``lon``            : DataArray — shared lon coordinate
        - ``models``         : list[str] — models with successful data load
        """
        hw_clim: dict[str, dict] = {}
        hw_series: dict[str, dict] = {}
        model_mean_tmax: dict[str, xr.DataArray] = {}
        lat_coord: xr.DataArray | None = None
        lon_coord: xr.DataArray | None = None

        for model in self.config.models:
            try:
                indices_ds, mean_tmax = self._load_or_compute(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue

            # Per-index climatology (mean over years) and time series
            model_clim = {}
            model_srs = {}
            for idx in ("hwn", "hwf", "hwd", "hwm", "hwa"):
                da = indices_ds[idx]
                model_clim[idx] = da.mean("year")

                if lat_coord is None:
                    lat_coord = da["lat"]
                    lon_coord = da["lon"]

                areas = compute_latlon_areas(
                    np.asarray(da["lat"]), np.asarray(da["lon"])
                )
                areas_da = xr.DataArray(areas, dims=("lat", "lon"))
                model_srs[idx] = latlon_global_mean(da, areas_da)

            hw_clim[model] = model_clim
            hw_series[model] = model_srs
            model_mean_tmax[model] = mean_tmax

        obs_mean_tmax = None
        if lat_coord is not None:
            if use_era5_obs(self.config):
                obs_mean_tmax = load_era5_mean(
                    self.config, "tasmax", self.period,
                    lat_coord.values, lon_coord.values,
                )
            else:
                obs_mean_tmax = self._load_be_mean_tmax(
                    lat_coord.values, lon_coord.values
                )

        return {
            "hw_clim": hw_clim,
            "hw_series": hw_series,
            "model_mean_tmax": model_mean_tmax,
            "obs_mean_tmax": obs_mean_tmax,
            "lat": lat_coord,
            "lon": lon_coord,
            "models": list(hw_clim.keys()),
        }

    def _load_or_compute(
        self, model: str
    ) -> tuple[xr.Dataset, xr.DataArray]:
        """Return ``(indices_ds, mean_tmax)``, loading from NC or computing fresh.

        ``indices_ds`` is an xr.Dataset with variables ``hwn, hwf, hwd, hwm, hwa``
        (dims year, lat, lon).  ``mean_tmax`` is shape (lat, lon), units K.
        """
        nc_path = self._nc_path(model)
        if nc_path.exists():
            logger.info("  %s: loading HW indices from %s", model, nc_path.name)
            # decode_timedelta=False prevents xarray from treating hwd (units="days")
            # as a timedelta64 coordinate.
            ds = xr.open_dataset(nc_path, decode_timedelta=False)
            required = {"hwn", "hwf", "hwd", "hwm", "hwa", "tmax_mean"}
            if required.issubset(ds.data_vars):
                return ds[list(required - {"tmax_mean"})], ds["tmax_mean"]
            logger.info("  %s: NC incomplete — recomputing", model)

        logger.info("  %s: computing HW indices from daily tasmax", model)
        da = self.model_loader.load_var(
            model, "tasmax", table="day", period=self.period
        )

        # Land mask
        land_mask = self._load_land_mask(
            model, np.asarray(da["lat"]), np.asarray(da["lon"])
        )

        # Compute 90th-percentile threshold per DOY (lazy → compute once)
        logger.info("  %s: computing TX90 threshold ...", model)
        threshold = self._compute_threshold(da)

        # Year-by-year index computation
        logger.info("  %s: computing annual HW indices ...", model)
        indices_ds, mean_tmax = self._compute_indices(da, threshold)

        # Apply land mask
        if land_mask is not None:
            for var in list(indices_ds.data_vars):
                indices_ds[var] = indices_ds[var].where(land_mask)
            mean_tmax = mean_tmax.where(land_mask)

        self._save_nc(model, indices_ds, mean_tmax)
        return indices_ds, mean_tmax

    # ── Core algorithm ─────────────────────────────────────────────────

    @staticmethod
    def _compute_threshold(da: xr.DataArray) -> xr.DataArray:
        """90th percentile of tasmax per day-of-year over the full period.

        Returns DataArray with dim ``dayofyear`` (1–366) and spatial dims.
        DOY 366 is filled with DOY 365 values to handle leap/non-leap mixing.
        """
        thresh = da.groupby("time.dayofyear").quantile(0.9).compute()
        # Ensure DOY 366 exists (needed when model has leap-year days)
        if 366 not in thresh["dayofyear"].values:
            doy365 = thresh.sel(dayofyear=365)
            doy366 = doy365.assign_coords(dayofyear=366).expand_dims("dayofyear")
            thresh = xr.concat([thresh, doy366], dim="dayofyear")
        return thresh

    def _compute_indices(
        self,
        da: xr.DataArray,
        threshold: xr.DataArray,
    ) -> tuple[xr.Dataset, xr.DataArray]:
        """Compute 5 HW indices year-by-year to bound memory usage.

        Processes NH summer (May–Sep) and SH summer (Nov–Mar) separately,
        then combines using a latitude mask.

        Returns
        -------
        indices_ds : xr.Dataset
            Variables hwn, hwf, hwd, hwm, hwa, each (year, lat, lon).
        mean_tmax : xr.DataArray
            Period-mean tasmax (lat, lon), K, computed over all days.
        """
        start_year = int(self.period[0])
        end_year = int(self.period[1])

        lat_vals = np.asarray(da["lat"])
        lon_vals = np.asarray(da["lon"])
        is_nh = (lat_vals >= 0)[:, np.newaxis]  # (nlat, 1) for broadcasting

        thresh_np = threshold.values  # (n_doy, nlat, nlon)

        # Pre-build DOY → threshold-row-index mapping (avoids dict lookup per day)
        max_doy = int(threshold["dayofyear"].values.max())
        doy2idx = np.zeros(max_doy + 1, dtype=np.intp)
        for i, d in enumerate(threshold["dayofyear"].values):
            doy2idx[int(d)] = i

        # Pre-compute time metadata from full coordinate — avoids .dt accessor
        # issues on sub-selected DataArrays in some xarray/cftime combinations.
        all_years = da.time.dt.year.values      # (n_time,) int
        all_months = da.time.dt.month.values    # (n_time,) int
        all_doys = da.time.dt.dayofyear.values  # (n_time,) int

        nh_months = np.array(list(_NH_SUMMER_MONTHS))
        sh_extra_months = np.array([11, 12])
        sh_early_months = np.array([1, 2, 3])

        years = []
        hwn_list: list[np.ndarray] = []
        hwf_list: list[np.ndarray] = []
        hwd_list: list[np.ndarray] = []
        hwm_list: list[np.ndarray] = []
        hwa_list: list[np.ndarray] = []

        tmax_sum = None
        tmax_n = 0

        for year in range(start_year, end_year + 1):
            # Integer index arrays for NH and SH summer time steps
            nh_tidx = np.where(
                (all_years == year) & np.isin(all_months, nh_months)
            )[0]
            sh_tidx = np.where(
                ((all_years == year - 1) & np.isin(all_months, sh_extra_months)) |
                ((all_years == year) & np.isin(all_months, sh_early_months))
            )[0]

            if len(nh_tidx) == 0 and len(sh_tidx) == 0:
                continue

            # Load data for the two windows (triggers dask compute for those slices)
            nh_np = da.isel(time=nh_tidx).values.astype(np.float32)
            sh_np = da.isel(time=sh_tidx).values.astype(np.float32)

            # Threshold rows for each day via pre-computed DOY mapping
            nh_thresh = thresh_np[doy2idx[all_doys[nh_tidx]]]   # (n_nh, nlat, nlon)
            sh_thresh = thresh_np[doy2idx[all_doys[sh_tidx]]]   # (n_sh, nlat, nlon)

            nh_idx = self._compute_hw_indices_numpy(nh_np, nh_np > nh_thresh)
            sh_idx = self._compute_hw_indices_numpy(sh_np, sh_np > sh_thresh)

            year_idx = {
                k: np.where(is_nh, nh_idx[k], sh_idx[k])
                for k in ("hwn", "hwf", "hwd", "hwm", "hwa")
            }

            years.append(year)
            hwn_list.append(year_idx["hwn"])
            hwf_list.append(year_idx["hwf"])
            hwd_list.append(year_idx["hwd"])
            hwm_list.append(year_idx["hwm"])
            hwa_list.append(year_idx["hwa"])

            # Full-year mean tmax accumulation
            yr_tidx = np.where(all_years == year)[0]
            if len(yr_tidx) > 0:
                yr_np = da.isel(time=yr_tidx).values.astype(np.float64)
                tmax_sum = yr_np.sum(axis=0) if tmax_sum is None else tmax_sum + yr_np.sum(axis=0)
                tmax_n += len(yr_tidx)

        def _to_da(arrays, name, long_name, units):
            data = np.stack(arrays, axis=0).astype(np.float32)  # (year, nlat, nlon)
            return xr.DataArray(
                data,
                dims=["year", "lat", "lon"],
                coords={"year": years, "lat": lat_vals, "lon": lon_vals},
                name=name,
                attrs={"long_name": long_name, "units": units},
            )

        indices_ds = xr.Dataset({
            "hwn": _to_da(hwn_list, "hwn", "Heatwave Number",    "events/year"),
            "hwf": _to_da(hwf_list, "hwf", "Heatwave Frequency", "days/year"),
            "hwd": _to_da(hwd_list, "hwd", "Heatwave Duration",  "days"),
            "hwm": _to_da(hwm_list, "hwm", "Heatwave Magnitude", "K"),
            "hwa": _to_da(hwa_list, "hwa", "Heatwave Amplitude", "K"),
        })

        mean_tmax_np = (tmax_sum / max(tmax_n, 1)).astype(np.float32)
        mean_tmax = xr.DataArray(
            mean_tmax_np,
            dims=["lat", "lon"],
            coords={"lat": lat_vals, "lon": lon_vals},
            name="tmax_mean",
            attrs={"long_name": "Period-mean Daily Maximum Temperature", "units": "K"},
        )

        return indices_ds, mean_tmax

    @staticmethod
    def _compute_hw_indices_numpy(
        tasmax: np.ndarray,
        hot: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute 5 heatwave indices from one summer window.

        Parameters
        ----------
        tasmax : (n_days, nlat, nlon) float32
            Daily maximum temperature.
        hot : (n_days, nlat, nlon) bool
            True where tasmax > DOY-specific 90th percentile.

        Returns
        -------
        dict with keys hwn, hwf, hwd, hwm, hwa, all shape (nlat, nlon).
        All-NaN cells (no data) return zero for counts, NaN for temperatures.
        """
        n_days = hot.shape[0]
        shape2d = hot.shape[1:]

        if n_days == 0:
            zeros = np.zeros(shape2d, dtype=np.float32)
            nans = np.full(shape2d, np.nan, dtype=np.float32)
            return {"hwn": zeros, "hwf": zeros, "hwd": zeros, "hwm": nans, "hwa": nans}

        # --- forward run lengths: fwd[t] = length of consecutive hot run ending at t ---
        fwd = np.zeros_like(hot, dtype=np.int16)
        fwd[0] = hot[0].astype(np.int16)
        for t in range(1, n_days):
            fwd[t] = np.where(hot[t], fwd[t - 1] + 1, 0)

        # --- heatwave days: part of a run ≥ 3 ---
        # Any day at the END of a run ≥ 3 marks that position.
        # Back-propagate: if hw[t+1] is True and hot[t] is True → hw[t] is True.
        hw = (fwd >= 3).copy()
        for t in range(n_days - 2, -1, -1):
            hw[t] |= hw[t + 1] & hot[t]

        # --- HWF ---
        hwf = hw.sum(axis=0).astype(np.float32)

        # --- HWN: count event onsets ---
        # An onset is: hw[t] is True AND hw[t-1] is False (or t=0)
        padded = np.concatenate([np.zeros((1,) + shape2d, dtype=bool), hw], axis=0)
        starts = hw & ~padded[:-1]
        hwn = starts.sum(axis=0).astype(np.float32)

        # --- HWD: max run length (= max of fwd on hw days) ---
        hwd = (fwd.astype(np.int32) * hw).max(axis=0).astype(np.float32)

        # --- HWM: mean tasmax over heatwave days ---
        no_hw = hwf == 0
        hw_count = np.where(no_hw, 1.0, hwf)  # avoid div-by-zero
        hw_sum = np.where(hw, tasmax, 0.0).sum(axis=0)
        hwm = np.where(no_hw, np.nan, hw_sum / hw_count).astype(np.float32)

        # --- HWA: peak tasmax over heatwave days ---
        # Approximation: max over all heatwave days (exact for ≤ 1 event/year;
        # may differ from the strict "hottest heatwave" definition when multiple
        # events occur).
        hw_masked = np.where(hw, tasmax, np.nan)
        with np.errstate(all="ignore"):
            hwa = np.nanmax(hw_masked, axis=0).astype(np.float32)
        hwa[no_hw] = np.nan

        return {"hwn": hwn, "hwf": hwf, "hwd": hwd, "hwm": hwm, "hwa": hwa}

    # ── Obs helpers ────────────────────────────────────────────────────

    def _load_model_land_mask(
        self, model: str, model_lat: np.ndarray, model_lon: np.ndarray
    ) -> xr.DataArray | None:
        """Boolean land mask from the model's own ``sftlf`` (fx, %), or None.

        Land where land area fraction > 50 %.  Returns None so the caller can
        fall back to the Berkeley mask when the model has no ``sftlf``.
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
        self, model: str, model_lat: np.ndarray, model_lon: np.ndarray
    ) -> xr.DataArray | None:
        """Boolean land mask (True = land), preferring the model's own sftlf."""
        model_mask = self._load_model_land_mask(model, model_lat, model_lon)
        if model_mask is not None:
            return model_mask
        path = self._be_tmax_path()
        if path is None:
            return None
        try:
            ds = xr.open_dataset(path)
            mask = ds["land_mask"].rename({"latitude": "lat", "longitude": "lon"})
            if float(mask.lon.min()) < 0:
                mask = mask.assign_coords(
                    lon=((mask.lon + 360) % 360)
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

    def _load_be_mean_tmax(
        self, model_lat: np.ndarray, model_lon: np.ndarray
    ) -> xr.DataArray | None:
        """Load BE Land TMAX, reconstruct absolute temp, return period mean (K)."""
        import pandas as pd

        path = self._be_tmax_path()
        if path is None:
            return None
        try:
            ds = xr.open_dataset(path, chunks="auto")

            dec_years = ds["time"].values
            years_int = dec_years.astype(int)
            months_int = np.floor((dec_years - years_int) * 12).astype(int) + 1
            months_int = np.clip(months_int, 1, 12)
            datetimes = pd.to_datetime(
                [f"{y:04d}-{m:02d}-01" for y, m in zip(years_int, months_int)]
            )
            ds = ds.assign_coords(time=datetimes)

            start, end = self.period
            anom = ds["temperature"].sel(time=slice(start, end))
            clim = ds["climatology"]  # (month_number, lat, lon)

            month_idx = anom.time.dt.month.values - 1
            clim_np = clim.values
            clim_matched = clim_np[month_idx]
            abs_temp = anom + xr.DataArray(
                clim_matched, dims=anom.dims, coords=anom.coords
            )

            rename = {}
            if "latitude" in abs_temp.dims:
                rename["latitude"] = "lat"
            if "longitude" in abs_temp.dims:
                rename["longitude"] = "lon"
            if rename:
                abs_temp = abs_temp.rename(rename)
            if float(abs_temp.lon.min()) < 0:
                abs_temp = abs_temp.assign_coords(
                    lon=((abs_temp.lon + 360) % 360)
                ).sortby("lon")

            # Period mean in °C → K
            mean_k = abs_temp.mean("time").compute() + _K_TO_C
            mean_k_interp = mean_k.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
            return mean_k_interp
        except Exception as exc:
            logger.warning("  Could not load BE Land TMAX obs: %s", exc)
            return None

    # ── NC I/O ─────────────────────────────────────────────────────────

    def _save_nc(
        self,
        model: str,
        indices_ds: xr.Dataset,
        mean_tmax: xr.DataArray,
    ) -> None:
        """Write per-model HW indices + mean TMAX to NetCDF checkpoint."""
        nc_path = self._nc_path(model)
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        start, end = self.period
        ds = xr.Dataset(
            {**{v: indices_ds[v] for v in indices_ds.data_vars},
             "tmax_mean": mean_tmax},
            attrs={
                "Conventions": "CF-1.8",
                "title": f"Heatwave Indices (TX90) — {model} ({start}–{end})",
                "institution": "Feather climate evaluation framework",
                "source": "feather/diag/heatwave.py",
                "model": model,
                "period_start": start,
                "period_end": end,
                "method": "tx90",
                "base_period": f"{start}-{end}",
                "definition": (
                    "Heatwave = run of >=3 days with tasmax > "
                    "DOY-specific 90th percentile"
                ),
                "land_only": "True — ocean pixels are NaN",
            },
        )
        ds.to_netcdf(nc_path)
        logger.info("  Saved HW NetCDF: %s", nc_path)

    # ── Plot ───────────────────────────────────────────────────────────

    _INDEX_META: dict[str, dict] = {
        "hwn": dict(long_name="Heatwave Number",    units="events/year", cmap="YlOrRd", vmin=0),
        "hwf": dict(long_name="Heatwave Frequency", units="days/year",   cmap="YlOrRd", vmin=0),
        "hwd": dict(long_name="Heatwave Duration",  units="days",        cmap="YlOrRd", vmin=0),
        "hwm": dict(long_name="Heatwave Magnitude", units="°C",          cmap="cmo.thermal", vmin=None),
        "hwa": dict(long_name="Heatwave Amplitude", units="°C",          cmap="cmo.thermal", vmin=None),
    }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate 5 climatology maps, 5 time series, and optional TMAX bias map."""
        figs: list[tuple[plt.Figure, dict]] = []
        models = results["models"]
        if not models:
            logger.warning("HeatwaveDiag: no model data — no figures produced")
            return figs

        for idx in ("hwn", "hwf", "hwd", "hwm", "hwa"):
            figs.append(self._plot_index_map(results, idx))
            figs.append(self._plot_index_timeseries(results, idx))

        if results.get("obs_mean_tmax") is not None:
            figs.append(self._plot_tmax_bias(results))

        return figs

    def _plot_index_map(self, results: dict, idx: str) -> tuple[plt.Figure, dict]:
        """Group A: multi-panel climatological map for one HW index."""
        meta_info = self._INDEX_META[idx]
        models = results["models"]
        # Convert K → °C for temperature indices at plot time
        k2c = _K_TO_C if meta_info["units"] == "°C" else 0.0
        data_dict = {
            m: results["hw_clim"][m][idx] - k2c for m in models
        }

        # Pre-compute vmin/vmax with NaN-safe fallback so plot_combined_map
        # never calls np.percentile on an empty array (happens when no
        # heatwaves occur, e.g. in synthetic constant-temperature tests).
        vmin = meta_info["vmin"]
        all_finite = np.concatenate([
            np.asarray(d).ravel()[np.isfinite(np.asarray(d).ravel())]
            for d in data_dict.values()
        ])
        if len(all_finite) == 0:
            vmin = vmin if vmin is not None else 0.0
            vmax: float | None = 1.0
        else:
            vmax = None  # let plot_combined_map compute from data

        fig, _ = plot_combined_map(
            data_dict,
            title=f"{self.title} — {meta_info['long_name']} (land only)",
            cmap=meta_info["cmap"],
            vmin=vmin,
            vmax=vmax,
            units=meta_info["units"],
        )
        meta = self._build_metadata(
            title=f"{self.title} — {meta_info['long_name']}",
            figure_id=f"heatwave_{idx}_map",
            models=models,
            description=(
                f"Mean annual {meta_info['long_name']} ({meta_info['units']}) "
                f"from the TX90 heatwave method.  Land-only; computed from "
                f"CMOR daily tasmax."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_index_timeseries(self, results: dict, idx: str) -> tuple[plt.Figure, dict]:
        """Group B: global-land-mean annual time series for one HW index."""
        meta_info = self._INDEX_META[idx]
        models = results["models"]
        fig, ax = plt.subplots(figsize=(10, 4))
        # HWM/HWA are absolute tasmax stored in K; convert to °C at plot time
        # (matching the maps), other indices have non-temperature units (k2c=0).
        k2c = _K_TO_C if meta_info["units"] == "°C" else 0.0
        for model in models:
            color = self.config.get_model_color(model)
            series = results["hw_series"][model][idx] - k2c
            ax.plot(
                np.asarray(series["year"]),
                np.asarray(series),
                color=color, lw=1.5, label=model,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{meta_info['long_name']} ({meta_info['units']})")
        ax.set_title(f"{self.title} — {meta_info['long_name']} Global Land Mean")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{self.title} — {meta_info['long_name']} Time Series",
            figure_id=f"heatwave_{idx}_timeseries",
            models=models,
            description=(
                f"Area-weighted global land-mean annual {meta_info['long_name']} "
                f"({meta_info['units']}) for each model."
            ),
            period=self.period,
            plot_type="timeseries",
        )
        return fig, meta

    def _plot_tmax_bias(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group C: mean daily TMAX bias map (model − Berkeley Earth Land TMAX, °C)."""
        models = results["models"]
        obs_k = results["obs_mean_tmax"]
        obs_c = obs_k - _K_TO_C
        obs_label = obs_ref_label(self.config, "tasmax")

        bias_dict = {
            m: results["model_mean_tmax"][m] - obs_k
            for m in models
        }

        fig, _ = plot_combined_bias_map(
            obs_c,
            bias_dict,
            title=f"{self.title} — Mean TMAX Bias vs {obs_label}",
            obs_title=obs_label,
            cmap="cmo.thermal",
            bias_cmap="RdBu_r",
            units="°C",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Mean TMAX Bias",
            figure_id="heatwave_tmax_bias",
            models=models,
            description=(
                "Bias in climatological mean daily maximum temperature "
                f"(model − {obs_label}, °C).  Land-only. "
                "Note: observed heatwave indices are not computable from "
                "monthly obs; only mean TMAX bias is shown."
            ),
            obs_dataset=("ERA5_TMINMAX" if use_era5_obs(self.config)
                         else "BERKELEY_EARTH_TMAX"),
            obs_variable=("tasmax" if use_era5_obs(self.config)
                          else "temperature"),
            period=self.period,
            plot_type="bias_map",
        )
        return fig, meta
