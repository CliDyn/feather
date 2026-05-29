"""Heatwave climate change signal diagnostic (Perkins & Alexander 2013).

Compares five heatwave indices between a historical reference period (hist-1950)
and a future period (ssp245) under SSP2-4.5, quantifying the climate change
signal per model.

The five indices are all derived from daily Tmax relative to **T90** — the 90th
percentile of the reference-period daily Tmax distribution at each grid point:

  HWF — Heatwave Frequency  : total heatwave days per year
  HWD — Heatwave Duration   : length of the longest event
  HWN — Heatwave Number     : number of distinct events per year
  HWA — Heatwave Amplitude  : peak (Tmax − T90) anomaly during hottest event
  HWM — Heatwave Magnitude  : mean (Tmax − T90) anomaly over all HW days

An **event** = ≥ 3 consecutive days where Tmax > T90.

Produces 11 figure groups:

A×5 (one per index): Combined map — N_models rows × 3 columns:
     [Reference period | Future period | Change (Δ = Future − Reference)]
B×5 (one per index): Stitched annual time series hist+ssp245 with period shading.
C×1: Mean daily Tmax bias vs Berkeley Earth Land TMAX (reference period).
     Skipped when BE data is unavailable.

Per-model NetCDF checkpoints (outside figures tree):
  ``{output_dir}/heatwave_change/{model}_hw_hist_{start}_{end}.nc``
  ``{output_dir}/heatwave_change/{model}_hw_ssp_{start}_{end}.nc``

The hist checkpoint stores HWF/HWD/HWN/HWA/HWM (year, lat, lon), T90 (lat, lon),
and tmax_mean (lat, lon).  The SSP checkpoint stores the five index arrays only.

Configuration lives in ``config.project["climate_change"]``:

.. code-block:: yaml

    project:
      climate_change:
        reference_period: ["1981", "2000"]
        future_period:    ["2031", "2050"]
        hist_load_period: ["1981", "2014"]
        ssp_load_period:  ["2015", "2050"]
        obs_end_year:     "2025"          # cap BE obs (optional)
        models:
          IFS-FESOM2-SR:
            hist_experiment:    hist-1950
            hist_data_source:   cmor
            future_experiment:  highres-future-ssp245
            future_data_source: cmor
          IFS-FESOM2-SR-r2:
            hist_data_source:   kerchunk_native
            hist_data_root:     /work/.../hist-1950
            future_data_source: kerchunk_native
            future_data_root:   /work/.../ssp245
            future_only_to:     "2030"
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
from feather.plot.maps import plot_combined_bias_map
from feather.util.spatial import latlon_global_mean

logger = logging.getLogger(__name__)

_HW_MIN_RUN: int = 3            # consecutive days threshold
_HW_PERCENTILE: float = 0.90   # reference-period Tmax quantile for T90
_K_TO_C: float = 273.15
_HIST_BOUNDARY_YEAR: int = 2015

_BE_TMAX_LAND_PATH = Path(
    "/work/bm1344/AWI/OBS/berkeleyearth/Land_TMAX_Gridded_0p25deg.nc"
)

_INDICES: dict[str, dict] = {
    "HWF": {
        "title": "Heatwave Frequency",
        "long_name": "Annual heatwave days (Tmax > T90, ≥3 consecutive days)",
        "units": "days/year",
        "cmap_abs": "YlOrRd",
        "cmap_change": "RdYlBu_r",
        "ylabel": "days/year",
    },
    "HWD": {
        "title": "Heatwave Duration",
        "long_name": "Longest heatwave event duration per year",
        "units": "days",
        "cmap_abs": "YlOrRd",
        "cmap_change": "RdYlBu_r",
        "ylabel": "days",
    },
    "HWN": {
        "title": "Heatwave Number",
        "long_name": "Number of heatwave events per year",
        "units": "events/year",
        "cmap_abs": "YlOrRd",
        "cmap_change": "RdYlBu_r",
        "ylabel": "events/year",
    },
    "HWA": {
        "title": "Heatwave Amplitude",
        "long_name": "Peak Tmax − T90 anomaly during hottest heatwave event",
        "units": "°C",
        "cmap_abs": "plasma",
        "cmap_change": "RdYlBu_r",
        "ylabel": "°C above T90",
    },
    "HWM": {
        "title": "Heatwave Magnitude",
        "long_name": "Mean Tmax − T90 anomaly during all heatwave days",
        "units": "°C",
        "cmap_abs": "plasma",
        "cmap_change": "RdYlBu_r",
        "ylabel": "°C above T90",
    },
}


def compute_hw_indices_year(
    tmax_year: np.ndarray,
    t90: np.ndarray,
    min_run: int = _HW_MIN_RUN,
) -> dict[str, np.ndarray]:
    """Compute the five Perkins & Alexander heatwave indices for one calendar year.

    Parameters
    ----------
    tmax_year:
        Daily Tmax array of shape (n_days, n_lat, n_lon), float.
    t90:
        Reference-period 90th-percentile array (n_lat, n_lon), float.
    min_run:
        Minimum consecutive heatwave-day count to constitute an event (default 3).

    Returns
    -------
    dict with keys ``"HWF"``, ``"HWD"``, ``"HWN"``, ``"HWA"``, ``"HWM"``,
    each a (n_lat, n_lon) float32 array.  Grid points with no heatwave events
    get 0 for count-based indices and NaN for anomaly-based ones (HWA, HWM).
    """
    n_days, n_lat, n_lon = tmax_year.shape

    # Boolean mask: heatwave candidate day
    hw = tmax_year > t90[np.newaxis]  # (T, lat, lon)

    # Forward pass: consecutive hw-day count ending at each day.
    consec = np.zeros((n_days, n_lat, n_lon), dtype=np.int16)
    consec[0] = hw[0].astype(np.int16)
    for t in range(1, n_days):
        consec[t] = np.where(hw[t], consec[t - 1] + 1, 0)

    # Backward pass: consecutive hw-day count starting from each day.
    remain = np.zeros((n_days, n_lat, n_lon), dtype=np.int16)
    remain[-1] = hw[-1].astype(np.int16)
    for t in range(n_days - 2, -1, -1):
        remain[t] = np.where(hw[t], remain[t + 1] + 1, 0)

    # A day is part of a qualifying event if its total run length >= min_run.
    # total_run = consec[t] + remain[t] - 1  (day t counted once in both passes)
    in_event = hw & ((consec + remain - 1) >= min_run)

    # ── HWF: total heatwave days ──────────────────────────────────────
    hwf = in_event.sum(axis=0).astype(np.float32)

    # ── HWN: number of distinct events ───────────────────────────────
    # An event starts when in_event turns True from False.
    prev_pad = np.concatenate(
        [np.zeros((1, n_lat, n_lon), dtype=bool), in_event[:-1]], axis=0
    )
    event_start = in_event & ~prev_pad
    hwn = event_start.sum(axis=0).astype(np.float32)

    # ── HWD: duration of longest event ───────────────────────────────
    # At each event end, consec[t] equals the event length.
    next_pad = np.concatenate(
        [in_event[1:], np.zeros((1, n_lat, n_lon), dtype=bool)], axis=0
    )
    event_end = in_event & ~next_pad
    consec_at_ends = np.where(event_end, consec, 0)
    hwd = consec_at_ends.max(axis=0).astype(np.float32)

    # ── HWA: peak anomaly during any event ───────────────────────────
    anom = (tmax_year - t90[np.newaxis]).astype(np.float32)
    anom_ev = np.where(in_event, anom, -np.inf)
    hwa = anom_ev.max(axis=0)
    hwa = np.where(np.isneginf(hwa), np.nan, hwa)

    # ── HWM: mean anomaly over all heatwave days ──────────────────────
    anom_sum = np.where(in_event, anom, 0.0).sum(axis=0)
    # np.where evaluates both branches; suppress the divide-by-zero warning
    # at grid points with no heatwave days (hwf == 0) — those become NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        hwm = np.where(hwf > 0, anom_sum / hwf, np.nan).astype(np.float32)

    return {"HWF": hwf, "HWD": hwd, "HWN": hwn, "HWA": hwa, "HWM": hwm}


@register
class HeatwaveChangeDiag(DiagnosticBase):
    """Heatwave climate change signal (Perkins & Alexander 2013) under SSP2-4.5.

    Loads daily tasmax from hist-1950 and highres-future-ssp245, computes the
    five heatwave indices (HWF, HWD, HWN, HWA, HWM) for both periods, and
    produces reference/future climatology maps, change maps, and time series.
    """

    name = "heatwave_change"
    title = "Heatwave Indices — Climate Change Signal (SSP2-4.5)"
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
        experiment: str = "hist-1950",
        period: tuple[str, str] = ("1981", "2000"),
    ):
        super().__init__(model_loader, obs_loader, config, cmip6_loader=cmip6_loader)
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
        return Path(self.config.output_dir) / "heatwave_change"

    def _nc_hist_path(self, model: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return (
            self.nc_dir
            / f"{safe}_hw_hist_{self.hist_load_period[0]}_{self.hist_load_period[1]}.nc"
        )

    def _nc_ssp_path(self, model: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return (
            self.nc_dir
            / f"{safe}_hw_ssp_{self.ssp_load_period[0]}_{self.ssp_load_period[1]}.nc"
        )

    # ── Loader factories ───────────────────────────────────────────────

    def _make_cmor_loader(self, model: str, experiment: str, data_root: str = ""):
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
        """Build a KerchunkParquetLoader using the 0.25° daily-max store."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        mc_orig = self.config.model_configs.get(model) or ModelConfig(name=model)
        mc = copy.copy(mc_orig)
        mc.variant = variant
        mc.data_root = ""
        cfg = copy.copy(self.config)
        cfg.model_configs = {**self.config.model_configs, model: mc}
        cfg.data_source = {"type": "kerchunk_parquet", "root": data_root}
        return KerchunkParquetLoader(cfg)

    def _make_hist_loader(self, model: str):
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
        return self._make_cmor_loader(model, fut_exp, cc_cfg.get("future_data_root", ""))

    # ── Static / shared helpers ────────────────────────────────────────

    @staticmethod
    def _collect_finite(arrays: list) -> np.ndarray:
        parts = [np.asarray(a).ravel() for a in arrays]
        if not parts:
            return np.array([], dtype=np.float64)
        merged = np.concatenate(parts)
        return merged[np.isfinite(merged)]

    def _be_tmax_path(self) -> Path | None:
        ds_cfg = self.config.obs_datasets.get("BERKELEY_EARTH_TMAX", {})
        if ds_cfg:
            base = Path(ds_cfg.get("path", ""))
            for fname in ds_cfg.get("variables", {}).values():
                p = base / fname
                if p.exists():
                    return p
        if _BE_TMAX_LAND_PATH.exists():
            return _BE_TMAX_LAND_PATH
        return None

    def _load_land_mask(
        self, model_lat: np.ndarray, model_lon: np.ndarray
    ) -> xr.DataArray | None:
        """BE land mask (True = land) nearest-interpolated to model lat/lon."""
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
            return mask.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="nearest",
                kwargs={"fill_value": 0.0},
            ) > 0.5
        except Exception as exc:
            logger.warning("  Could not load BE land mask: %s", exc)
            return None

    def _load_be_mean_tmax(
        self,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
        period: tuple[str, str],
    ) -> xr.DataArray | None:
        """BE Land TMAX period mean interpolated to model grid (K)."""
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
            start, end = period
            anom = ds["temperature"].sel(time=slice(start, end))
            clim = ds["climatology"]
            month_idx = anom.time.dt.month.values - 1
            clim_matched = clim.values[month_idx]
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
            mean_k = abs_temp.mean("time").compute() + _K_TO_C
            return mean_k.interp(
                lat=xr.DataArray(model_lat, dims="lat"),
                lon=xr.DataArray(model_lon, dims="lon"),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
        except Exception as exc:
            logger.warning("  Could not load BE Land TMAX: %s", exc)
            return None

    def _land_mean_series(self, hw_annual: xr.DataArray) -> xr.DataArray:
        """Area-weighted land-mean of an annual index DataArray (year,)."""
        return latlon_global_mean(hw_annual)

    # ── NC I/O ─────────────────────────────────────────────────────────

    def _load_from_loader(
        self, loader, model: str, period: tuple[str, str]
    ) -> xr.DataArray:
        """Call load_var for tasmax, handling CMORLoader (table="day") and Kerchunk."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        if isinstance(loader, KerchunkParquetLoader):
            return loader.load_var(model, "tasmax", period=period)
        return loader.load_var(model, "tasmax", table="day", period=period)

    def _load_and_save_hw(
        self,
        model: str,
        loader,
        period: tuple[str, str],
        nc_path: Path,
        *,
        ref_period: tuple[str, str] | None = None,
        t90_external: xr.DataArray | None = None,
    ) -> tuple[xr.Dataset | None, xr.DataArray | None, xr.DataArray | None]:
        """Load (or compute and checkpoint) annual heatwave indices.

        Parameters
        ----------
        ref_period:
            If given, compute T90 from this sub-period of *da* and store it
            in the NC checkpoint.  Pass for hist; omit for SSP.
        t90_external:
            Pre-computed T90 DataArray (lat, lon) to use when *ref_period* is
            not given.  Required for SSP to apply the historical baseline.

        Returns
        -------
        (hw_ds, t90, tmax_mean)
            ``hw_ds``    : Dataset with HWF/HWD/HWN/HWA/HWM (year, lat, lon)
            ``t90``      : DataArray(lat, lon) 90th-pctl threshold, or None
            ``tmax_mean``: DataArray(lat, lon) reference-period mean Tmax (K), or None
        """
        if nc_path.exists():
            logger.info("  %s: loading HW indices from %s", model, nc_path.name)
            # decode_timedelta=False prevents xarray from misinterpreting the
            # "days" units on HWD as a timedelta64 dtype (CF timedelta encoding).
            ds = xr.open_dataset(nc_path, decode_timedelta=False)
            idx_vars = [v for v in _INDICES if v in ds]
            hw_ds = ds[idx_vars] if idx_vars else None
            t90 = ds["t90"] if "t90" in ds else None
            tmax_mean = ds["tmax_mean"] if "tmax_mean" in ds else None
            return hw_ds, t90, tmax_mean

        logger.info("  %s: computing HW indices %s–%s", model, *period)
        try:
            da = self._load_from_loader(loader, model, period)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.warning("  %s: cannot load tasmax (%s) — skipping", model, exc)
            return None, None, None

        lat_vals = np.asarray(da["lat"])
        lon_vals = np.asarray(da["lon"])

        # Compute T90 from reference sub-period (hist case)
        t90_np: np.ndarray | None = None
        tmax_mean_da: xr.DataArray | None = None
        if ref_period is not None:
            ref_slice = da.sel(time=slice(ref_period[0], ref_period[1]))
            logger.info("  %s: computing T90 from %s–%s", model, *ref_period)
            t90_xr = ref_slice.quantile(_HW_PERCENTILE, dim="time")
            if "quantile" in t90_xr.dims:
                t90_xr = t90_xr.squeeze("quantile", drop=True)
            t90_xr = t90_xr.compute()
            t90_np = np.asarray(t90_xr)
            tmax_mean_da = ref_slice.mean("time").compute()
        elif t90_external is not None:
            t90_np = np.asarray(t90_external)
        else:
            logger.warning("  %s: no T90 source — skipping", model)
            return None, None, None

        # Year-by-year index computation
        all_years = sorted({int(y) for y in da.time.dt.year.values})
        n_lat, n_lon = len(lat_vals), len(lon_vals)
        annual: dict[str, list] = {idx: [] for idx in _INDICES}
        year_list: list[int] = []

        for yr in all_years:
            yr_mask = da.time.dt.year == yr
            yr_da = da.isel(time=yr_mask)
            try:
                yr_np = yr_da.values  # (n_days, lat, lon)
            except Exception as exc:
                logger.warning("  %s year %d: load failed (%s) — skipping", model, yr, exc)
                continue
            if yr_np.ndim != 3 or yr_np.shape[0] == 0:
                continue
            idx_dict = compute_hw_indices_year(yr_np, t90_np)
            for key in _INDICES:
                annual[key].append(idx_dict[key])
            year_list.append(yr)

        if not year_list:
            logger.warning("  %s: no yearly data after processing — skipping", model)
            return None, None, None

        year_arr = np.array(year_list)
        ds_vars: dict[str, xr.DataArray] = {}
        for key, meta in _INDICES.items():
            stacked = np.stack(annual[key], axis=0)  # (years, lat, lon)
            ds_vars[key] = xr.DataArray(
                stacked,
                dims=("year", "lat", "lon"),
                coords={"year": year_arr, "lat": lat_vals, "lon": lon_vals},
                attrs={"long_name": meta["long_name"], "units": meta["units"]},
            )

        # Apply land mask before saving (use .values to avoid coordinate alignment)
        land_mask = self._load_land_mask(lat_vals, lon_vals)
        land_mask_np = land_mask.values if land_mask is not None else None
        if land_mask_np is not None:
            for key in _INDICES:
                ds_vars[key] = ds_vars[key].where(land_mask_np)
            if tmax_mean_da is not None:
                tmax_mean_da = tmax_mean_da.where(land_mask_np)

        # Build t90 DataArray for storage
        t90_da: xr.DataArray | None = None
        if t90_np is not None and ref_period is not None:
            t90_da = xr.DataArray(
                t90_np,
                dims=("lat", "lon"),
                coords={"lat": lat_vals, "lon": lon_vals},
                attrs={
                    "long_name": (
                        f"Reference-period ({ref_period[0]}–{ref_period[1]}) "
                        "90th percentile of daily Tmax"
                    ),
                    "units": "K",
                },
            )
            ds_vars["t90"] = t90_da

        if tmax_mean_da is not None:
            ds_vars["tmax_mean"] = tmax_mean_da.assign_attrs({
                "long_name": (
                    f"Period-mean daily maximum temperature "
                    f"({ref_period[0]}–{ref_period[1]})"
                    if ref_period else "Period-mean daily maximum temperature"
                ),
                "units": "K",
            })

        hw_ds = xr.Dataset(
            ds_vars,
            attrs={
                "model": model,
                "period_start": period[0],
                "period_end": period[1],
                "t90_percentile": str(_HW_PERCENTILE),
                "min_run": str(_HW_MIN_RUN),
            },
        )
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        hw_ds.to_netcdf(nc_path)
        logger.info("  Saved HW NC: %s", nc_path)

        idx_vars = [v for v in _INDICES if v in hw_ds]
        return hw_ds[idx_vars], t90_da, (ds_vars.get("tmax_mean") if tmax_mean_da is not None else None)

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
        """Load/compute heatwave indices for ref and future periods for all models.

        Returns
        -------
        dict with keys:
          models         : list[str]
          ref_clim       : {idx: {model: DataArray(lat, lon)}}
          fut_clim       : {idx: {model: DataArray(lat, lon)}}
          change         : {idx: {model: DataArray(lat, lon)}}
          hist_series    : {idx: {model: DataArray(year)}}
          ssp_series     : {idx: {model: DataArray(year)}}
          model_mean_tmax: {model: DataArray(lat, lon)}
          obs_mean_tmax  : {model: DataArray(lat, lon)}
          lat / lon      : shared coordinates
        """
        models: list[str] = []
        ref_clim: dict[str, dict] = {idx: {} for idx in _INDICES}
        fut_clim: dict[str, dict] = {idx: {} for idx in _INDICES}
        change: dict[str, dict] = {idx: {} for idx in _INDICES}
        hist_series: dict[str, dict] = {idx: {} for idx in _INDICES}
        ssp_series: dict[str, dict] = {idx: {} for idx in _INDICES}
        model_mean_tmax: dict[str, xr.DataArray] = {}
        lat_coord: xr.DataArray | None = None
        lon_coord: xr.DataArray | None = None

        for model in self.config.models:
            logger.info("  Processing: %s", model)

            # ── Historical ────────────────────────────────────────────
            hist_loader = self._make_hist_loader(model)
            hist_ds, hist_t90, hist_tmax_mean = self._load_and_save_hw(
                model,
                hist_loader,
                self.hist_load_period,
                self._nc_hist_path(model),
                ref_period=self.ref_period,
            )
            if hist_ds is None:
                logger.warning("  %s: no hist HW data — skipping", model)
                continue

            models.append(model)
            if lat_coord is None:
                lat_coord = hist_ds["HWF"]["lat"]
                lon_coord = hist_ds["HWF"]["lon"]

            # Re-apply land mask to handle NCs saved before masking was added.
            land_mask = self._load_land_mask(
                np.asarray(hist_ds["HWF"]["lat"]),
                np.asarray(hist_ds["HWF"]["lon"]),
            )
            land_mask_np = land_mask.values if land_mask is not None else None

            for idx in _INDICES:
                idx_da = hist_ds[idx]
                if land_mask_np is not None:
                    idx_da = idx_da.where(land_mask_np)
                ref_slice = idx_da.sel(year=slice(*self.ref_period))
                ref_clim[idx][model] = ref_slice.mean("year")
                hist_series[idx][model] = self._land_mean_series(idx_da)

            if hist_tmax_mean is not None:
                model_mean_tmax[model] = (
                    hist_tmax_mean.where(land_mask_np)
                    if land_mask_np is not None else hist_tmax_mean
                )

            # ── Future (SSP) ──────────────────────────────────────────
            fut_loader = self._make_fut_loader(model)
            if fut_loader is None:
                logger.info("  %s: no future loader — reference-only", model)
                continue

            ssp_ds, _, _ = self._load_and_save_hw(
                model,
                fut_loader,
                self.ssp_load_period,
                self._nc_ssp_path(model),
                t90_external=hist_t90,
            )
            if ssp_ds is None:
                logger.warning("  %s: SSP HW load failed — reference only", model)
                continue

            for idx in _INDICES:
                idx_da = ssp_ds[idx]
                if land_mask_np is not None:
                    idx_da = idx_da.where(land_mask_np)
                ssp_series[idx][model] = self._land_mean_series(idx_da)
                fut_slice = idx_da.sel(year=slice(*self.fut_period))
                if len(fut_slice.year) > 0:
                    fut_clim[idx][model] = fut_slice.mean("year")
                    change[idx][model] = fut_clim[idx][model] - ref_clim[idx][model]
                else:
                    logger.info(
                        "  %s: SSP run ends before future period (%s) — "
                        "excluded from change map",
                        model, self.fut_period[0],
                    )

        # Berkeley Earth TMAX for Group C: interpolate to each model's own grid.
        obs_mean_tmax: dict[str, xr.DataArray] = {}
        for _m, _m_tmax in model_mean_tmax.items():
            _obs = self._load_be_mean_tmax(
                np.asarray(_m_tmax["lat"]),
                np.asarray(_m_tmax["lon"]),
                self.ref_period,
            )
            if _obs is not None:
                obs_mean_tmax[_m] = _obs

        return {
            "models": models,
            "ref_clim": ref_clim,
            "fut_clim": fut_clim,
            "change": change,
            "hist_series": hist_series,
            "ssp_series": ssp_series,
            "model_mean_tmax": model_mean_tmax,
            "obs_mean_tmax": obs_mean_tmax,
            "lat": lat_coord,
            "lon": lon_coord,
        }

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        if not results["models"]:
            logger.warning("%s: no model data — no figures", self.name)
            return figs

        # Group A: one [ref | future | change] map figure per index
        for idx in _INDICES:
            figs.append(self._plot_change_panels(results, idx))

        # Group B: one timeseries figure per index
        for idx in _INDICES:
            figs.append(self._plot_timeseries(results, idx))

        # Group C: Tmax bias vs Berkeley Earth (when obs available)
        if results.get("obs_mean_tmax") and results["model_mean_tmax"]:
            figs.append(self._plot_tmax_bias(results))

        return figs

    def _plot_change_panels(
        self, results: dict, idx: str
    ) -> tuple[plt.Figure, dict]:
        """Group A: N_models rows × 3 cols [Reference | Future | Change] for *idx*."""
        import nereus as nr
        from nereus.plotting import get_projection
        from feather.plot.maps import _flatten_latlon

        meta_idx = _INDICES[idx]
        models = results["models"]
        n_models = len(models)

        rf_finite = self._collect_finite(
            [results["ref_clim"][idx][m] for m in models if m in results["ref_clim"][idx]]
            + [results["fut_clim"][idx][m] for m in models if m in results["fut_clim"][idx]]
        )
        vmax_rf = max(float(np.percentile(rf_finite, 98)), 1e-3) if len(rf_finite) > 0 else 1.0

        ch_finite = self._collect_finite(list(results["change"][idx].values()))
        vlim = (
            max(float(np.percentile(np.abs(ch_finite), 98)), 1e-3)
            if len(ch_finite) > 0 else 1.0
        )

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
            f"Change (Future − Reference)",
        ]
        cmaps = [meta_idx["cmap_abs"], meta_idx["cmap_abs"], meta_idx["cmap_change"]]
        vmins = [0, 0, -vlim]
        vmaxs = [vmax_rf, vmax_rf, vlim]
        cb_labels = [meta_idx["units"]] * 2 + [meta_idx["units"]]

        row_interp_cache: dict[str, Any] = {}

        for row, model in enumerate(models):
            panels = [
                results["ref_clim"][idx].get(model),
                results["fut_clim"][idx].get(model),
                results["change"][idx].get(model),
            ]
            for col, (data, ctitle, cmap, vmin, vmax, cblabel) in enumerate(
                zip(panels, col_titles, cmaps, vmins, vmaxs, cb_labels)
            ):
                ax = axes[row, col]
                panel_title = f"{model}\n{ctitle}" if col == 0 else ctitle
                if data is not None:
                    vals, lons, lats = _flatten_latlon(data)
                    _, _, interp = nr.plot(
                        vals, lons, lats,
                        ax=ax,
                        projection="rob",
                        resolution=0.25,
                        interpolator=row_interp_cache.get(model),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        colorbar=True,
                        colorbar_label=cblabel,
                        title=panel_title,
                    )
                    row_interp_cache[model] = interp
                else:
                    ax.set_title(panel_title, fontsize=9)
                    ax.text(
                        0.5, 0.5, "No SSP2-4.5 data available",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=10, color="gray", style="italic",
                    )

        fig.suptitle(
            f"{self.title} — {meta_idx['title']} ({meta_idx['units']})\n"
            f"Reference: {self.ref_period[0]}–{self.ref_period[1]}  ·  "
            f"Future: {self.fut_period[0]}–{self.fut_period[1]}  ·  "
            f"T90 baseline: {self.ref_period[0]}–{self.ref_period[1]}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        fig_id = f"heatwave_change_{idx.lower()}_maps"
        all_fut_models = list(results["fut_clim"][idx].keys())
        meta = self._build_metadata(
            title=f"{self.title} — {meta_idx['title']} Maps",
            figure_id=fig_id,
            models=models,
            description=(
                f"Three-panel climate change maps for {meta_idx['title']} ({idx}). "
                f"{meta_idx['long_name']}. "
                f"Left: reference period {self.ref_period[0]}–{self.ref_period[1]}. "
                f"Centre: future period {self.fut_period[0]}–{self.fut_period[1]} (SSP2-4.5). "
                f"Right: change (future − reference). "
                f"T90 threshold from reference period. "
                f"Models with SSP2-4.5 data: {', '.join(all_fut_models) or 'none'}."
            ),
            period=(self.ref_period[0], self.fut_period[1]),
            obs_dataset="",
            obs_variable="",
            plot_type="map",
        )
        return fig, meta

    def _plot_timeseries(
        self, results: dict, idx: str
    ) -> tuple[plt.Figure, dict]:
        """Group B: stitched hist+ssp245 annual land-mean time series for *idx*."""
        meta_idx = _INDICES[idx]
        fig, ax = plt.subplots(figsize=(13, 5))

        ax.axvspan(
            int(self.ref_period[0]), int(self.ref_period[1]) + 1,
            alpha=0.10, color="#1f77b4", zorder=0,
        )
        ax.axvspan(
            int(self.fut_period[0]), int(self.fut_period[1]) + 1,
            alpha=0.10, color="#d62728", zorder=0,
        )
        ax.axvline(
            _HIST_BOUNDARY_YEAR, color="k", lw=1.2, ls="--", alpha=0.55,
            label=f"hist | ssp245 ({_HIST_BOUNDARY_YEAR})",
        )

        for model in results["models"]:
            color = self.config.get_model_color(model)
            h = results["hist_series"][idx].get(model)
            s = results["ssp_series"][idx].get(model)
            if h is not None and s is not None:
                combined = xr.concat([h, s], dim="year")
                ax.plot(
                    np.asarray(combined["year"]), np.asarray(combined),
                    color=color, lw=1.5, label=model,
                )
            elif h is not None:
                ax.plot(
                    np.asarray(h["year"]), np.asarray(h),
                    color=color, lw=1.5, label=model,
                )
            elif s is not None:
                ax.plot(
                    np.asarray(s["year"]), np.asarray(s),
                    color=color, lw=1.5, label=model,
                )

        ax.set_xlabel("Year")
        ax.set_ylabel(meta_idx["ylabel"])
        ax.set_title(
            f"{self.title} — {meta_idx['title']} Global Land Mean\n"
            f"Blue: ref {self.ref_period[0]}–{self.ref_period[1]}; "
            f"red: future {self.fut_period[0]}–{self.fut_period[1]}"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig_id = f"heatwave_change_{idx.lower()}_timeseries"
        meta = self._build_metadata(
            title=f"{self.title} — {meta_idx['title']} Time Series",
            figure_id=fig_id,
            models=results["models"],
            description=(
                f"Area-weighted global land-mean annual {meta_idx['title']} ({idx}). "
                f"{meta_idx['long_name']}. "
                f"Continuous line: hist-1950 ({self.hist_load_period[0]}–"
                f"{self.hist_load_period[1]}) joined to SSP2-4.5 from {_HIST_BOUNDARY_YEAR}. "
                f"Blue shading: reference period; red shading: future period. "
                f"T90 computed from {self.ref_period[0]}–{self.ref_period[1]} and "
                f"applied to both periods."
            ),
            period=(self.hist_load_period[0], self.ssp_load_period[1]),
            obs_dataset="",
            obs_variable="",
            plot_type="timeseries",
        )
        return fig, meta

    def _plot_tmax_bias(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group C: model mean Tmax bias vs Berkeley Earth (reference period)."""
        obs_mean_tmax = results["obs_mean_tmax"]
        models = [
            m for m in results["models"]
            if m in results["model_mean_tmax"] and m in obs_mean_tmax
        ]
        obs_k = next(iter(obs_mean_tmax.values()))
        obs_c = obs_k - _K_TO_C
        bias_dict = {
            m: results["model_mean_tmax"][m] - obs_mean_tmax[m]
            for m in models
        }
        fig, _ = plot_combined_bias_map(
            obs_c,
            bias_dict,
            title=(
                f"{self.title}\n"
                f"Mean Tmax Bias vs Berkeley Earth "
                f"({self.ref_period[0]}–{self.ref_period[1]})"
            ),
            obs_title="Berkeley Earth Land TMAX",
            cmap="cmo.thermal",
            bias_cmap="RdBu_r",
            units="°C",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Mean Tmax Bias (reference period)",
            figure_id="heatwave_change_tmax_bias",
            models=models,
            description=(
                f"Bias in climatological mean daily maximum temperature "
                f"(model − Berkeley Earth Land TMAX, °C) for the reference period "
                f"{self.ref_period[0]}–{self.ref_period[1]}. "
                "Land-only. Model: CMOR daily tasmax mean; "
                "obs: Berkeley Earth monthly Land TMAX (anomaly + climatology, °C→K)."
            ),
            obs_dataset="BERKELEY_EARTH_TMAX",
            obs_variable="temperature",
            period=self.ref_period,
            plot_type="bias_map",
        )
        return fig, meta
