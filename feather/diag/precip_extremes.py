"""ETCCDI daily-precipitation extremes: present day and SSP2-4.5 change.

Computes nine ETCCDI indices from **daily** precipitation for every EERIE
member and for an auto-discovered daily CMIP6 ensemble (one member per
model), for a historical reference period and an SSP2-4.5 future period.
Index definitions and units are in :mod:`feather.util.precip_indices`
(R1mm, R10mm, R20mm, PRCPTOT, SDII, R95p, R99p, Rx1day, Rx5day).

Units
-----
Every loader hands back CMOR ``pr`` in kg m⁻² s⁻¹ (the IFS-FESOM kerchunk
stores' ``tprate`` m s⁻¹ is scaled ×1000 by the loader).  Daily values are
converted ×86400 to mm/day, so amounts are mm (per year), intensities mm/day
and counts days per year.  Each model's first year is sanity-checked: a
global-mean daily precipitation outside 0.3–20 mm/day means the data is not
in kg m⁻² s⁻¹ and the model is skipped with an error instead of silently
producing indices off by 10³ or 86400.

Periods
-------
Shared with the other ``*_change`` diagnostics through
``project.climate_change`` (reference, future, hist/ssp load windows).  The
R95p/R99p thresholds are each model's own wet-day percentiles over the
**reference period** (ETCCDI uses 1961–1990, which most EERIE members do not
cover) and are reused unchanged for the future period, so a change in R95p
is a change in rainfall above the present-day 95th percentile.

Figures (per index)
-------------------
A ``{idx}_reference``    Reference-period climatology: one panel per EERIE
                         model family (mean of its members) + CMIP6 MMM.
B ``{idx}_change``       Future − reference, same panels.  Counts and the
                         R95p/R99p totals as absolute change; PRCPTOT, SDII,
                         Rx1day and Rx5day as percent change.
C ``{idx}_diff_cmip6``   Reference period, EERIE family − CMIP6 MMM.
D ``{idx}_timeseries``   Global-mean annual index, hist + SSP2-4.5, members
                         thin, family means thick, CMIP6 MMM + min–max range,
                         observations in black/grey.
E ``{idx}_bias_{obs}``   Reference period, one figure per observational
                         dataset: EERIE family − obs, CMIP6 MMM − obs and the
                         other observational datasets − obs.

Seasons
-------
``project.precip_extremes.seasons`` (any of DJF, MAM, JJA, SON) adds the same
figure set per season, ids ``{idx}_{season}_reference`` etc. (lower-case
season).  Seasonal indices use the definitions of
:mod:`feather.util.precip_indices` over the season's days, with the annual
R95p/R99p thresholds; DJF of year *Y* is December *Y−1* + January–February *Y*.
They are checkpointed next to the annual files as ``…_seasons.nc``, so adding
seasons to a finished run re-reads the daily data once and keeps the annual
checkpoints.

Observations
------------
``project.precip_extremes.obs`` names daily datasets laid out as CMOR trees
(``{data_root}/{experiment}/r1i1p1f1/day/pr/gr/v*/``), read with
:class:`~feather.data.cmor_loader.CMORLoader` and treated like a historical
member: same indices, own base-period percentiles.  The EERIE configs use
ERA5 and MSWEP v2.8, derived to daily 0.25° (area-conservative) by
``scripts/era5_derive_pr_daily.sh`` and ``scripts/mswep_derive_pr_daily.sh``.
The reference climatology figure (A) gains one panel per dataset.

In pooled ``project.ensemble_mode`` the families collapse into a single
"EERIE" group.

Checkpoints
-----------
Annual index fields on each model's native grid, one file per model and
segment, so re-runs (and re-plots on a login node) skip the daily read:

  ``{output_dir}/precip_extremes/{model}_{hist|ssp}_{start}_{end}.nc``
  ``{output_dir}/precip_extremes/cmip6/{model}_{hist|ssp}_{start}_{end}.nc``
  ``{output_dir}/precip_extremes/obs/{dataset}_hist_{start}_{end}.nc``

The ``hist`` file also stores the base-period thresholds ``rr95``/``rr99``
(mm/day).  Delete a file to force its recomputation.  Open them with
``decode_timedelta=False``: the count indices carry ``units = "days"``, which
xarray otherwise turns into timedeltas.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.netcdf_export import write_netcdf
from feather.diag.registry import register
from feather.diag.temp_extremes_change import TempExtremesChangeDiag
from feather.util.precip_indices import (
    CARRY_DAYS,
    MIN_VALID_DAYS,
    INDEX_INFO,
    INDICES,
    KG_M2_S_TO_MM_DAY,
    SEASONS,
    WetDayPercentiles,
    annual_indices,
    season_blocks,
    season_min_valid_days,
)
from feather.util.spatial import latlon_global_mean

logger = logging.getLogger(__name__)

_HIST_BOUNDARY_YEAR: int = 2015

#: Indices whose change is shown in percent of the reference value.
_PERCENT_CHANGE: frozenset[str] = frozenset({"prcptot", "sdii", "rx1day", "rx5day"})

#: Reference values below which a percent change is masked (index units).
_PERCENT_FLOOR: dict[str, float] = {
    "prcptot": 10.0, "sdii": 1.0, "rx1day": 1.0, "rx5day": 2.0,
}

#: Plausible global-mean daily precipitation (mm/day) for the unit check.
_PLAUSIBLE_MM_DAY: tuple[float, float] = (0.3, 20.0)

#: Fraction of the future window a run must cover to enter the change maps.
_MIN_WINDOW_FRACTION: float = 0.75

#: Cells per call into :func:`annual_indices` (bounds temporary memory).
_CELL_CHUNK: int = 262_144

_EERIE_RES: float = 0.25      # common grid for EERIE family means
_CMIP6_RES: float = 1.0       # common grid for the CMIP6 MMM

_MM_DAY_UNITS = {"mm/day", "mm day-1", "mm d-1", "mm/d", "mm day**-1"}


def _regular_grid(res: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-edge-aligned (``res`` = 0.25 → −90…90) or centred lat/lon axes."""
    if res == _EERIE_RES:
        lat = np.linspace(-90.0, 90.0, int(round(180 / res)) + 1)
        lon = np.arange(0.0, 360.0, res)
    else:
        lat = np.arange(-90.0 + res / 2, 90.0, res)
        lon = np.arange(res / 2, 360.0, res)
    return lat, lon


def _to_grid(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Bilinear interpolation of a (lat, lon) field onto a regular grid.

    Longitudes are wrapped to 0–360 and padded across the seam so there is
    no NaN column at 0°/360°.  Points poleward of the source grid take the
    nearest source row.
    """
    da = da.rename({d: n for d, n in (("latitude", "lat"), ("longitude", "lon"))
                    if d in da.dims})
    if (da.sizes["lat"] == len(lat) and da.sizes["lon"] == len(lon)
            and np.allclose(np.sort(da["lat"].values), lat)
            and np.allclose(np.sort(da["lon"].values % 360.0), lon)):
        return da.assign_coords(lon=da["lon"] % 360.0).sortby(["lat", "lon"])
    da = da.assign_coords(lon=da["lon"].values % 360.0).sortby(["lat", "lon"])
    left = da.isel(lon=slice(-2, None)).assign_coords(
        lon=da["lon"].values[-2:] - 360.0)
    right = da.isel(lon=slice(0, 2)).assign_coords(
        lon=da["lon"].values[:2] + 360.0)
    padded = xr.concat([left, da, right], dim="lon")
    lat_c = np.clip(lat, float(da["lat"].min()), float(da["lat"].max()))
    out = padded.interp(lat=lat_c, lon=lon, method="linear")
    return out.assign_coords(lat=lat)


@register
class PrecipExtremesDiag(DiagnosticBase):
    """ETCCDI precipitation extremes for EERIE and daily CMIP6, hist + SSP2-4.5."""

    name = "precip_extremes"
    title = "Daily Precipitation Extremes (ETCCDI) — Present Day and SSP2-4.5 Change"
    domain = "sfc"
    variables = ["pr"]
    group = "precipitation_extremes"

    #: Rasterisation resolution (°) for the map panels.
    plot_resolution: float = 0.25

    # Future-run loaders are built exactly as for the other *_change
    # diagnostics, from the same ``project.climate_change.models`` block.
    _make_cmor_loader = TempExtremesChangeDiag._make_cmor_loader
    _make_kerchunk_loader = TempExtremesChangeDiag._make_kerchunk_loader
    _make_fut_loader = TempExtremesChangeDiag._make_fut_loader

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
        save_netcdf: bool = False,
    ):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, save_netcdf=save_netcdf)
        cc = config.project.get("climate_change", {})
        self.ref_period: tuple[str, str] = tuple(cc.get("reference_period", list(period)))
        self.fut_period: tuple[str, str] = tuple(cc.get("future_period", ["2031", "2050"]))
        self.hist_load_period: tuple[str, str] = tuple(
            cc.get("hist_load_period", [self.ref_period[0], "2014"]))
        self.ssp_load_period: tuple[str, str] = tuple(
            cc.get("ssp_load_period", ["2015", self.fut_period[1]]))
        self._cc_models: dict = cc.get("models", {})

        pe = config.project.get("precip_extremes", {}) or {}
        wanted = [i.lower() for i in pe.get("indices", INDICES)]
        unknown = sorted(set(wanted) - set(INDICES))
        if unknown:
            logger.warning("precip_extremes: unknown indices %s ignored", unknown)
        self.indices: list[str] = [i for i in INDICES if i in wanted]
        self.base_period: tuple[str, str] = tuple(pe.get("base_period", self.ref_period))
        seasons = [str(x).upper() for x in pe.get("seasons", []) or []]
        bad = sorted(set(seasons) - set(SEASONS))
        if bad:
            logger.warning("precip_extremes: unknown seasons %s ignored", bad)
        self.seasons: list[str] = [x for x in SEASONS if x in seasons]
        self._obs_cfg: dict[str, dict] = dict(pe.get("obs", {}) or {})
        self.obs_load_period: tuple[str, str] = tuple(
            pe.get("obs_load_period", self.hist_load_period))
        self._cmip6_cfg: dict = getattr(config, "cmip6_daily", {}) or {}
        self._cmip6_label: str = self._cmip6_cfg.get("label", "CMIP6")

    # ── Paths ──────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / "precip_extremes"

    def _nc_path(self, model: str, seg: str, cmip6: bool = False,
                 obs: bool = False) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        if obs:
            p, base = self.obs_load_period, self.nc_dir / "obs"
        else:
            p = self.hist_load_period if seg == "hist" else self.ssp_load_period
            base = self.nc_dir / "cmip6" if cmip6 else self.nc_dir
        return base / f"{safe}_{seg}_{p[0]}_{p[1]}.nc"

    @staticmethod
    def _season_path(path: Path) -> Path:
        return path.with_name(f"{path.stem}_seasons.nc")

    def _season_vars(self) -> list[str]:
        return [f"{i}_{x.lower()}" for x in self.seasons for i in self.indices]

    def _keys(self) -> list[str]:
        """Field keys: ``{idx}`` (annual) then ``{idx}_{season}``."""
        return list(self.indices) + self._season_vars()

    @staticmethod
    def _split(key: str) -> tuple[str, str | None]:
        idx, _, season = key.partition("_")
        return idx, (season.upper() or None)

    def _info(self, key: str) -> tuple[str, str, str, str | None]:
        """``(label, long name, units, season)``; label carries the season."""
        idx, season = self._split(key)
        label, long_name, units = INDEX_INFO[idx]
        return (f"{label} {season}" if season else label), long_name, units, season

    @staticmethod
    def _per(season: str | None) -> str:
        if season is None:
            return "annual"
        if season == "DJF":
            return "DJF (December of the previous year to February)"
        return season

    # ── Daily data → annual indices ────────────────────────────────────

    @staticmethod
    def _std_dims(da: xr.DataArray) -> xr.DataArray:
        ren = {d: n for d, n in (("latitude", "lat"), ("longitude", "lon"))
               if d in da.dims}
        da = da.rename(ren) if ren else da
        return da.transpose("time", "lat", "lon")

    @staticmethod
    def _to_mm_day_factor(da: xr.DataArray) -> float:
        units = str(da.attrs.get("units", "")).strip()
        return 1.0 if units in _MM_DAY_UNITS else KG_M2_S_TO_MM_DAY

    def _year_values(self, da: xr.DataArray, year: int, factor: float) -> np.ndarray:
        """One year of daily precipitation as float32 mm/day, (time, lat, lon)."""
        import dask

        sel = da.isel(time=np.flatnonzero(da["time"].dt.year.values == year))
        return self._values(sel, factor)

    @staticmethod
    def _values(sel: xr.DataArray, factor: float) -> np.ndarray:
        import dask

        # Threads, not the distributed cluster: the year is consumed right
        # here, so shipping ~1.5 GB per year through the scheduler is waste.
        with dask.config.set(scheduler="threads"):
            vals = np.asarray(sel.values, dtype=np.float32)
        return vals * np.float32(factor)

    @staticmethod
    def _check_units(model: str, arr: np.ndarray, year: int) -> None:
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        lo, hi = _PLAUSIBLE_MM_DAY
        if not lo <= mean <= hi:
            raise ValueError(
                f"{model}: mean daily precipitation in {year} is {mean:.4g} "
                f"mm/day after ×{KG_M2_S_TO_MM_DAY:g} — outside {lo}–{hi}, so "
                f"the input is not in kg m-2 s-1")
        logger.info("  %s: unit check OK (%d mean %.2f mm/day)", model, year, mean)

    def _base_percentiles(self, model: str, da: xr.DataArray, factor: float):
        """``(rr95, rr99)`` (lat, lon) over the base period, or ``(None, None)``."""
        years = sorted(set(int(y) for y in da["time"].dt.year.values))
        b0, b1 = int(self.base_period[0]), int(self.base_period[1])
        base_years = [y for y in years if b0 <= y <= b1]
        if not base_years:
            logger.warning("  %s: no data in base period %s–%s — R95p/R99p "
                           "will be missing", model, b0, b1)
            return None, None
        ny, nx = da.sizes["lat"], da.sizes["lon"]
        acc = WetDayPercentiles(ny * nx, max_days=366 * len(base_years))
        for y in base_years:
            arr = self._year_values(da, y, factor)
            acc.update(arr.reshape(arr.shape[0], -1))
        res = acc.result()
        logger.info("  %s: wet-day percentiles from %d base years (%d–%d)",
                    model, len(base_years), base_years[0], base_years[-1])
        return res[0.95].reshape(ny, nx), res[0.99].reshape(ny, nx)

    def _stream_indices(
        self,
        model: str,
        da: xr.DataArray,
        factor: float,
        rr95: np.ndarray | None,
        rr99: np.ndarray | None,
        prev: tuple[np.ndarray, np.ndarray] | None = None,
        check_units: bool = True,
        seasons: bool = False,
    ) -> tuple[xr.Dataset, xr.Dataset | None]:
        """Annual (and seasonal) indices for every year of *da*.

        *prev* is ``(values, months)`` of the last :data:`CARRY_DAYS` days
        before *da* starts (Rx5day tail and the December of the first DJF).
        Returns ``(annual, seasonal)`` (year, lat, lon) Datasets; *seasonal*
        is ``None`` unless *seasons* is set.
        """
        years = sorted(set(int(y) for y in da["time"].dt.year.values))
        all_years = da["time"].dt.year.values
        all_months = da["time"].dt.month.values
        ny, nx = da.sizes["lat"], da.sizes["lon"]
        n = ny * nx
        p95 = None if rr95 is None else rr95.reshape(-1)
        p99 = None if rr99 is None else rr99.reshape(-1)
        out = {k: np.full((len(years), n), np.nan, dtype=np.float32)
               for k in INDICES}
        sea_keys = self._season_vars() if seasons else []
        sea_out = {k: np.full((len(years), n), np.nan, dtype=np.float32)
                   for k in sea_keys}
        wanted = set(self.indices)

        def run(block, tail, min_valid, store, iy, suffix=""):
            for s in range(0, n, _CELL_CHUNK):
                e = min(s + _CELL_CHUNK, n)
                res = annual_indices(
                    block[:, s:e],
                    p95=None if p95 is None else p95[s:e],
                    p99=None if p99 is None else p99[s:e],
                    prev_tail=None if tail is None else tail[:, s:e],
                    min_valid_days=min_valid,
                )
                for k in INDICES:
                    if not suffix:
                        store[k][iy, s:e] = res[k]
                    elif k in wanted:
                        store[f"{k}{suffix}"][iy, s:e] = res[k]

        pv, pm = (None, None) if prev is None else prev
        if pv is not None:
            pv = pv.reshape(pv.shape[0], -1)
        for iy, y in enumerate(years):
            sel = np.flatnonzero(all_years == y)
            arr = self._values(da.isel(time=sel), factor)
            months = all_months[sel]
            if check_units and iy == 0:
                self._check_units(model, arr, y)
            flat = arr.reshape(arr.shape[0], -1)
            run(flat, None if pv is None else pv[-4:], MIN_VALID_DAYS, out, iy)
            if sea_keys:
                for season, block, tail in season_blocks(
                        flat, months, self.seasons, pv, pm):
                    if block.shape[0]:
                        run(block, tail, season_min_valid_days(season), sea_out,
                            iy, f"_{season.lower()}")
            pv, pm = flat[-CARRY_DAYS:], months[-CARRY_DAYS:]
            logger.debug("  %s: %d done (%d days)", model, y, arr.shape[0])
        logger.info("  %s: indices for %d years (%d–%d)%s", model, len(years),
                    years[0], years[-1],
                    f" + seasons {', '.join(self.seasons)}" if sea_keys else "")

        coords = {"year": np.asarray(years), "lat": da["lat"].values,
                  "lon": da["lon"].values}

        def as_ds(store):
            data_vars = {}
            for k, vals in store.items():
                label, long_name, units, season = self._info(k)
                data_vars[k] = xr.DataArray(
                    vals.reshape(len(years), ny, nx), dims=("year", "lat", "lon"),
                    coords=coords,
                    attrs={"long_name": f"{label}: {long_name}", "units": units},
                )
            ds = xr.Dataset(data_vars)
            ds.attrs["daily_tail_note"] = "Rx5day windows assigned to their last day"
            return ds

        sea_ds = None
        if sea_keys:
            sea_ds = as_ds(sea_out)
            sea_ds.attrs["djf_note"] = ("DJF of year Y = December Y-1 + January-"
                                        "February Y; the first year's DJF is "
                                        "missing unless the previous December "
                                        "was carried in")
        return as_ds(out), sea_ds

    def _segment(
        self,
        model: str,
        seg: str,
        path: Path,
        get_da: Callable[[], xr.DataArray] | None,
        thresholds: tuple | None = None,
        prev_fn: Callable[[], tuple | None] | None = None,
    ) -> xr.Dataset | None:
        """Load the checkpoints for one segment, computing what is missing.

        The annual indices live in *path*, the seasonal ones next to it in
        ``…_seasons.nc``.  When only the seasonal file is missing, the daily
        data are streamed again for it (the annual file is kept as is and its
        base-period thresholds are reused).
        """
        # Counts carry units "days", which xarray would decode to timedelta.
        ann = xr.open_dataset(path, decode_timedelta=False) if path.exists() else None
        sea_vars = self._season_vars()
        sea_path = self._season_path(path)
        sea = None
        if sea_vars and sea_path.exists():
            sea = xr.open_dataset(sea_path, decode_timedelta=False)
            missing = [v for v in sea_vars if v not in sea]
            if missing:
                logger.info("  %s/%s: %s lacks %s — recomputing seasons", model,
                            seg, sea_path.name, ", ".join(missing[:4]))
                sea.close()
                sea = None
        if ann is not None and (not sea_vars or sea is not None):
            logger.info("  %s/%s: loading indices from %s%s", model, seg, path.name,
                        f" + {sea_path.name}" if sea is not None else "")
            return self._merge(ann, sea)
        if get_da is None:
            return ann
        try:
            da = self._std_dims(get_da())
        except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("  %s/%s: no daily pr (%s) — skipping", model, seg, exc)
            return ann
        if da.sizes.get("time", 0) == 0:
            logger.warning("  %s/%s: no daily pr in the load window", model, seg)
            return ann
        da = da.sortby("lat")
        factor = self._to_mm_day_factor(da)

        if thresholds is None and ann is not None and "rr95" in ann:
            thresholds = (ann["rr95"].values,
                          ann["rr99"].values if "rr99" in ann else None)
        if thresholds is None:
            rr95, rr99 = self._base_percentiles(model, da, factor)
        else:
            rr95, rr99 = thresholds
        if ann is not None:
            logger.info("  %s/%s: annual indices from %s; computing seasons",
                        model, seg, path.name)
        prev = prev_fn() if prev_fn else None
        if prev is not None:
            # Only a December directly before the segment start continues it.
            y0, m0 = int(da["time"].dt.year.values[0]), int(da["time"].dt.month.values[0])
            if len(prev) > 2 and (prev[2] != (y0 - 1, 12) or m0 != 1):
                logger.info("  %s/%s: previous segment ends %d-%02d, not right "
                            "before %d-%02d — not carried over", model, seg,
                            *prev[2], y0, m0)
                prev = None
            else:
                prev = prev[:2]
        try:
            ds, sea_ds = self._stream_indices(model, da, factor, rr95, rr99, prev,
                                              seasons=bool(sea_vars))
        except ValueError as exc:
            logger.error("  %s/%s: %s — skipping", model, seg, exc)
            return ann

        meta = {
            "model": model, "segment": seg,
            "base_period": f"{self.base_period[0]}-{self.base_period[1]}",
            "wet_day_threshold": "1 mm/day",
            "source_units": "kg m-2 s-1 (x86400 -> mm/day)",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if sea_ds is not None:
            sea_ds.attrs.update(meta)
            write_netcdf(sea_ds, sea_path,
                         encoding={v: {"zlib": True, "complevel": 1}
                                   for v in sea_ds.data_vars})
            logger.info("  Saved %s", sea_path)
        if ann is not None:
            return self._merge(ann, sea_ds)

        if seg == "hist":
            for q, arr in (("rr95", rr95), ("rr99", rr99)):
                if arr is not None:
                    ds[q] = xr.DataArray(
                        arr, dims=("lat", "lon"),
                        coords={"lat": ds["lat"], "lon": ds["lon"]},
                        attrs={"long_name": f"{q[2:]}th percentile of wet-day "
                               f"precipitation, {self.base_period[0]}–"
                               f"{self.base_period[1]}",
                               "units": "mm/day"})
        ds.attrs.update(meta)
        enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
        write_netcdf(ds, path, encoding=enc)
        logger.info("  Saved %s", path)
        return self._merge(ds, sea_ds)

    @staticmethod
    def _merge(ann: xr.Dataset, sea: xr.Dataset | None) -> xr.Dataset:
        if sea is None:
            return ann
        return xr.merge([ann, sea], join="override", combine_attrs="override")

    def _model_indices(
        self,
        model: str,
        get_hist: Callable[[], xr.DataArray],
        get_fut: Callable[[], xr.DataArray] | None,
        cmip6: bool = False,
    ) -> dict[str, xr.Dataset | None] | None:
        hist = self._segment(model, "hist", self._nc_path(model, "hist", cmip6),
                             get_hist)
        if hist is None:
            return None
        thresholds = (
            hist["rr95"].values if "rr95" in hist else None,
            hist["rr99"].values if "rr99" in hist else None,
        )

        def prev():
            # The last days of the historical run: whole first-2015 Rx5day
            # windows and the December of DJF 2015.
            try:
                da = self._std_dims(get_hist()).sortby("lat")
                da = da.isel(time=slice(-CARRY_DAYS, None))
                t = da["time"]
                return (self._values(da, self._to_mm_day_factor(da)),
                        t.dt.month.values,
                        (int(t.dt.year.values[-1]), int(t.dt.month.values[-1])))
            except Exception:   # noqa: BLE001 — only shortens 4 windows / DJF 2015
                return None

        ssp = self._segment(model, "ssp", self._nc_path(model, "ssp", cmip6),
                            get_fut, thresholds=thresholds,
                            prev_fn=prev) if get_fut is not None else None
        return {"hist": hist, "ssp": ssp}

    # ── Data sources ───────────────────────────────────────────────────

    def _eerie_sources(self) -> dict[str, tuple[Callable, Callable | None]]:
        out = {}
        for model in self.config.models:
            def get_hist(m=model):
                return self.model_loader.load_var(
                    m, "pr", table="day", period=self.hist_load_period)

            get_fut = None
            if model in self._cc_models:
                fut_loader = self._make_fut_loader(model)
                if fut_loader is not None:
                    def get_fut(m=model, ldr=fut_loader):
                        return ldr.load_var(
                            m, "pr", table="day", period=self.ssp_load_period)
            out[model] = (get_hist, get_fut)
        return out

    def _make_obs_loader(self, name: str, spec: dict):
        """CMORLoader for one observational dataset laid out as a CMOR tree."""
        from feather.config import ModelConfig
        from feather.data.cmor_loader import CMORLoader

        mc = ModelConfig(
            name=name, institution=spec.get("institution", ""),
            experiment=spec.get("experiment", name.lower()),
            variant=spec.get("variant", "r1i1p1f1"),
            data_root=spec["data_root"], grid_label=spec.get("grid_label", "gr"),
            grids={"sfc": "latlon"},
        )
        cfg = copy.copy(self.config)
        cfg.model_configs = {**self.config.model_configs, name: mc}
        return CMORLoader(cfg)

    def _obs_sources(self) -> dict[str, Callable[[], xr.DataArray]]:
        out = {}
        for name, spec in self._obs_cfg.items():
            spec = spec or {}
            if not spec.get("data_root"):
                logger.warning("precip_extremes: obs %s has no data_root — skipped",
                               name)
                continue

            def get_hist(n=name, sp=spec):
                return self._make_obs_loader(n, sp).load_var(
                    n, "pr", table="day", period=self.obs_load_period)

            out[name] = get_hist
        return out

    def _obs_color(self, name: str, i: int) -> str:
        default = ("black", "0.45", "0.7")[min(i, 2)]
        return (self._obs_cfg.get(name) or {}).get("color", default)

    def _discover_cmip6(self):
        """Daily-pr CMIP6 models with the same member in hist and future."""
        cfg = self._cmip6_cfg
        if not cfg.get("enabled", False):
            return None, []
        from feather.data import pool_discovery as _pd
        from feather.data.cmip6_nc_loader import (
            CMIP6NCLoader, discover_daily_models, experiment_dirs,
        )

        root = cfg.get("root", "/work/ik1017/CMIP6/data/CMIP6")
        hist_exp = cfg.get("experiment", "historical")
        fut_exp = cfg.get("future_experiment", "ssp245")
        try:
            found = discover_daily_models(
                root, experiment=hist_exp, table="day", variable="pr",
                member=cfg.get("member", "r1i1p1f1"),
                exclude=tuple(cfg.get("exclude", [])),
                # Pick the member that has day/pr in both experiments, not
                # r1i1p1f1-or-lowest: that rule dropped e.g. CESM2 (r1 has no
                # ssp245 day/pr, r4/r11 have both) and CAMS-CSM1-0 (only r2).
                # The future run may sit under another institution.
                require_also=(fut_exp,),
            )
        except Exception as exc:  # noqa: BLE001 — discovery must not kill the run
            logger.warning("CMIP6 daily discovery failed (%s) — EERIE only", exc)
            return None, []
        wanted = cfg.get("models", "auto")
        if isinstance(wanted, (list, tuple, set)):
            found = [m for m in found if m.name in set(wanted)]

        keep, hist_only = [], []
        for mc in found:
            if any(_pd.variable_files(d, mc.variant, "day", "pr")
                   for d in experiment_dirs(root, mc.name, fut_exp, mc.institution)):
                mc.experiments = [hist_exp, fut_exp]
                keep.append(mc)
            else:
                hist_only.append(mc.name)
        if hist_only:
            logger.info("CMIP6 daily: no %s %s/day/pr for the hist member — "
                        "skipped: %s", fut_exp, "same-member", ", ".join(hist_only))
        if cfg.get("max_models"):
            keep = keep[: int(cfg["max_models"])]
        if not keep:
            logger.warning("CMIP6 daily: no models with daily pr in %s + %s",
                           hist_exp, fut_exp)
            return None, []
        logger.info("CMIP6 daily: %d models — %s", len(keep),
                    ", ".join(f"{m.name}({m.variant})" for m in keep))
        return CMIP6NCLoader.for_models(self.config, keep, root=root), keep

    # ── Reductions ─────────────────────────────────────────────────────

    def _window_mean(self, ds: xr.Dataset | None, idx: str,
                     window: tuple[str, str]) -> xr.DataArray | None:
        if ds is None or idx not in ds:
            return None
        lo, hi = int(window[0]), int(window[1])
        sl = ds[idx].sel(year=slice(lo, hi))
        if sl.sizes["year"] < _MIN_WINDOW_FRACTION * (hi - lo + 1):
            return None
        return sl.mean("year").load()

    @staticmethod
    def _global_series(ds: xr.Dataset | None, idx: str) -> xr.DataArray | None:
        if ds is None or idx not in ds:
            return None
        return latlon_global_mean(ds[idx]).load()

    def _summarise(self, raw: dict[str, dict], grid: tuple) -> dict[str, dict]:
        """Per model and index: reference/future means on *grid*, global series."""
        out: dict[str, dict] = {}
        for model, segs in raw.items():
            m = {"ref": {}, "fut": {}, "ts": {}}
            for idx in self._keys():
                ref = self._window_mean(segs["hist"], idx, self.ref_period)
                fut = self._window_mean(segs.get("ssp"), idx, self.fut_period)
                if ref is not None:
                    m["ref"][idx] = _to_grid(ref, *grid)
                if fut is not None and ref is not None:
                    m["fut"][idx] = _to_grid(fut, *grid)
                parts = [s for s in (self._global_series(segs["hist"], idx),
                                     self._global_series(segs.get("ssp"), idx))
                         if s is not None]
                if parts:
                    ts = xr.concat(parts, dim="year") if len(parts) > 1 else parts[0]
                    m["ts"][idx] = ts.sortby("year")
            out[model] = m
        return out

    @classmethod
    def _change(cls, key: str, ref: xr.DataArray, fut: xr.DataArray) -> xr.DataArray:
        idx, _ = cls._split(key)
        if idx in _PERCENT_CHANGE:
            pct = 100.0 * (fut - ref) / ref
            return pct.where(ref >= cls._percent_floor(key))
        return fut - ref

    @classmethod
    def _percent_floor(cls, key: str) -> float:
        idx, season = cls._split(key)
        # A season holds about a quarter of the annual total.
        scale = 0.25 if (season and idx == "prcptot") else 1.0
        return _PERCENT_FLOOR[idx] * scale

    def _group(self, summary: dict, members: list[str], idx: str) -> dict | None:
        """Mean reference field and change over *members* (future-capable ones)."""
        refs = [summary[m]["ref"][idx] for m in members if idx in summary[m]["ref"]]
        if not refs:
            return None
        fut_m = [m for m in members if idx in summary[m]["fut"]]
        g = {"ref": xr.concat(refs, dim="member").mean("member"),
             "n": len(refs), "n_fut": len(fut_m), "change": None}
        if fut_m:
            r = xr.concat([summary[m]["ref"][idx] for m in fut_m], "member").mean("member")
            f = xr.concat([summary[m]["fut"][idx] for m in fut_m], "member").mean("member")
            g["change"] = self._change(idx, r, f)
        return g

    def _families(self, models: list[str]) -> dict[str, list[str]]:
        if self._per_family:
            ordered = [m for m in self.config.models if m in models]
            return self.config.get_model_families(ordered)
        return {"EERIE": list(models)} if models else {}

    # ── Orchestration ──────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        logger.info("precip_extremes: reference %s–%s, future %s–%s, base %s–%s",
                    *self.ref_period, *self.fut_period, *self.base_period)
        eerie_raw: dict[str, dict] = {}
        for model, (get_hist, get_fut) in self._eerie_sources().items():
            logger.info("  EERIE member: %s", model)
            r = self._model_indices(model, get_hist, get_fut)
            if r is not None:
                eerie_raw[model] = r

        cmip6_raw: dict[str, dict] = {}
        loader, cmip6_models = self._discover_cmip6()
        for mc in cmip6_models:
            logger.info("  CMIP6 model: %s", mc.name)

            def get_hist(m=mc.name):
                return loader.load_var(m, "pr", table="day",
                                       period=self.hist_load_period)

            def get_fut(m=mc.name):
                return loader.load_var(m, "pr", table="day",
                                       period=self.ssp_load_period)

            r = self._model_indices(mc.name, get_hist, get_fut, cmip6=True)
            if r is not None:
                cmip6_raw[mc.name] = r

        obs_raw: dict[str, dict] = {}
        for name, get_hist in self._obs_sources().items():
            logger.info("  Observations: %s", name)
            ds = self._segment(name, "hist", self._nc_path(name, "hist", obs=True),
                               get_hist)
            if ds is not None:
                obs_raw[name] = {"hist": ds, "ssp": None}

        eerie = self._summarise(eerie_raw, _regular_grid(_EERIE_RES))
        obs = self._summarise(obs_raw, _regular_grid(_EERIE_RES))
        cmip6 = self._summarise(cmip6_raw, _regular_grid(_CMIP6_RES))
        families = self._families(list(eerie))

        fam: dict[str, dict] = {idx: {} for idx in self._keys()}
        mmm: dict[str, dict | None] = {}
        for idx in self._keys():
            for f, members in families.items():
                g = self._group(eerie, members, idx)
                if g is not None:
                    g["members"] = [m for m in members if idx in eerie[m]["ref"]]
                    fam[idx][f] = g
            mmm[idx] = self._group(cmip6, list(cmip6), idx)

        return {"eerie": eerie, "cmip6": cmip6, "obs": obs, "families": families,
                "family": fam, "mmm": mmm}

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = self.compute()
        saved = []
        # One figure at a time: with seasons there are ~270, each with several
        # 0.25° map panels — building them all before saving would hold them
        # all in memory.
        for fig, meta in self._iter_figures(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)
        if self.save_netcdf:
            self._export_summary(results)
        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        return list(self._iter_figures(results))

    def _iter_figures(self, results: dict[str, Any]):
        for idx in self._keys():
            if not results["family"][idx] and not results["mmm"].get(idx):
                logger.warning("%s: no data for %s — skipping", self.name, idx)
                continue
            for fn in (self._plot_reference, self._plot_change,
                       self._plot_diff_cmip6, self._plot_timeseries):
                fm = fn(results, idx)
                if fm is not None:
                    yield fm
            for name in self._obs_with(results, idx):
                fm = self._plot_bias(results, idx, name)
                if fm is not None:
                    yield fm

    @staticmethod
    def _obs_with(results, idx) -> list[str]:
        return [o for o, d in results.get("obs", {}).items() if idx in d["ref"]]

    def _label(self, f: str, g: dict, n_key: str = "n") -> str:
        n = g[n_key]
        return f if n == 1 else f"{f} mean ({n} members)"

    def _mmm_label(self, g: dict, n_key: str = "n") -> str:
        return f"{self._cmip6_label} MMM ({g[n_key]} models)"

    @staticmethod
    def _robust_range(fields: list, symmetric: bool, floor: float = 0.0):
        vals = [np.asarray(f).ravel() for f in fields if f is not None]
        v = np.concatenate(vals) if vals else np.array([])
        v = v[np.isfinite(v)]
        if not len(v):
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
        if symmetric:
            lim = max(float(np.percentile(np.abs(v), 98)), floor or 1e-6)
            return -lim, lim
        return float(np.percentile(v, 2)), float(np.percentile(v, 98))

    def _map_figure(self, panels: list[tuple[str, xr.DataArray]], *, cmap: str,
                    vmin: float, vmax: float, cbar_label: str, suptitle: str):
        import nereus as nr
        from nereus.plotting import get_projection
        from feather.plot.maps import _flatten_latlon

        n = len(panels)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows),
                                 subplot_kw={"projection": get_projection("rob")},
                                 squeeze=False)
        for ax, (title, field) in zip(axes.flat, panels):
            vals, lons, lats = _flatten_latlon(field)
            nr.plot(np.asarray(vals), lons, lats, ax=ax, projection="rob",
                    resolution=self.plot_resolution, cmap=cmap, vmin=vmin, vmax=vmax,
                    colorbar=True, colorbar_label=cbar_label, title=title)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    def _panel_stats(self, panels) -> dict:
        return {t.replace("\n", " "): {"global_mean": self._latlon_field_mean(f)}
                for t, f in panels}

    def _ref_txt(self) -> str:
        return f"{self.ref_period[0]}–{self.ref_period[1]}"

    def _fut_txt(self) -> str:
        return f"{self.fut_period[0]}–{self.fut_period[1]}"

    def _plot_reference(self, results, idx):
        label, long_name, units, season = self._info(idx)
        obs = self._obs_with(results, idx)
        panels = [(o, results["obs"][o]["ref"][idx]) for o in obs]
        panels += [(self._label(f, g), g["ref"])
                   for f, g in results["family"][idx].items()]
        mmm = results["mmm"].get(idx)
        if mmm:
            panels.append((self._mmm_label(mmm), mmm["ref"]))
        if not panels:
            return None
        vmin, vmax = self._robust_range([p[1] for p in panels], symmetric=False)
        vmin = max(0.0, vmin)
        fig = self._map_figure(
            panels, cmap="cmo.rain", vmin=vmin, vmax=vmax,
            cbar_label=f"{label} ({units})",
            suptitle=f"{label} — {long_name}\nReference period {self._ref_txt()}")
        meta = self._build_metadata(
            title=f"{label} — Reference Climatology ({self._ref_txt()})",
            figure_id=f"{idx}_reference",
            models=self._meta_models(results, idx),
            description=(
                f"Mean {self._per(season)} {label} ({long_name.lower()}, {units}) over "
                f"{self._ref_txt()} from daily precipitation (wet day: RR ≥ 1 mm). "
                + self._panel_note(results, idx)
                + (f" Observational panels: {', '.join(obs)}." if obs else "")),
            computation_notes=self._computation_notes(idx),
            period=self.ref_period, units=units, plot_type="map",
            summary_statistics=self._panel_stats(panels),
            benchmark_info=self._bench_meta(results, idx),
            extra={"index": label, "season": season or "annual", **self._obs_meta(obs)},
        )
        return fig, meta

    def _plot_change(self, results, idx):
        label, long_name, units, season = self._info(idx)
        panels = [(self._label(f, g, "n_fut"), g["change"])
                  for f, g in results["family"][idx].items()
                  if g["change"] is not None]
        mmm = results["mmm"].get(idx)
        if mmm and mmm["change"] is not None:
            panels.append((self._mmm_label(mmm, "n_fut"), mmm["change"]))
        if not panels:
            logger.info("%s: no future data for %s — no change map", self.name, idx)
            return None
        pct = self._split(idx)[0] in _PERCENT_CHANGE
        cunits = "%" if pct else units
        vmin, vmax = self._robust_range([p[1] for p in panels], symmetric=True,
                                        floor=5.0 if pct else 0.0)
        fig = self._map_figure(
            panels, cmap="BrBG", vmin=vmin, vmax=vmax,
            cbar_label=f"Δ{label} ({cunits})",
            suptitle=(f"{label} change, SSP2-4.5 {self._fut_txt()} minus "
                      f"{self._ref_txt()}" + (" (relative)" if pct else "")))
        no_fut = [f for f, g in results["family"][idx].items() if g["change"] is None]
        meta = self._build_metadata(
            title=f"{label} — Change {self._fut_txt()} vs {self._ref_txt()} (SSP2-4.5)",
            figure_id=f"{idx}_change",
            models=self._meta_models(results, idx, future=True),
            description=(
                f"Change in mean {self._per(season)} {label} ({long_name.lower()}) between the "
                f"SSP2-4.5 future period {self._fut_txt()} and the reference period "
                f"{self._ref_txt()}, "
                + (f"in percent of the reference value (masked where the reference "
                   f"is below {self._percent_floor(idx):g} {units}). " if pct else
                   f"in {units}. ")
                + "Only members whose SSP2-4.5 run covers the future period enter "
                  "the family means"
                + (f"; families without one: {', '.join(no_fut)}." if no_fut else ".")
                + (" R95p/R99p use each model's reference-period wet-day "
                   "percentile as a fixed threshold." if self._split(idx)[0] in
                   ("r95p", "r99p") else "")),
            computation_notes=self._computation_notes(idx),
            period=(self.ref_period[0], self.fut_period[1]), units=cunits,
            plot_type="map", summary_statistics=self._panel_stats(panels),
            benchmark_info=self._bench_meta(results, idx, "n_fut"),
            extra={"index": label, "season": season or "annual",
                   "change_type": "relative" if pct else "absolute"},
        )
        return fig, meta

    def _plot_diff_cmip6(self, results, idx):
        label, long_name, units, season = self._info(idx)
        mmm = results["mmm"].get(idx)
        fam = results["family"][idx]
        if not mmm or not fam:
            return None
        target = _regular_grid(_EERIE_RES)
        mmm_hr = _to_grid(mmm["ref"], *target)
        panels = [(f"{self._label(f, g)}\nminus {self._cmip6_label} MMM",
                   g["ref"] - mmm_hr) for f, g in fam.items()]
        vmin, vmax = self._robust_range([p[1] for p in panels], symmetric=True)
        fig = self._map_figure(
            panels, cmap="BrBG", vmin=vmin, vmax=vmax,
            cbar_label=f"Δ{label} ({units})",
            suptitle=(f"{label}: EERIE minus {self._cmip6_label} MMM "
                      f"({mmm['n']} models), {self._ref_txt()}"))
        meta = self._build_metadata(
            title=f"{label} — EERIE minus {self._cmip6_label} MMM ({self._ref_txt()})",
            figure_id=f"{idx}_diff_cmip6",
            models=self._meta_models(results, idx),
            description=(
                f"Difference in mean {self._per(season)} {label} ({units}) over {self._ref_txt()} "
                f"between each EERIE model family and the {self._cmip6_label} "
                f"multi-model mean of {mmm['n']} models (one member each), the MMM "
                "interpolated bilinearly from 1° to 0.25°. Positive: EERIE has more "
                f"{'days' if units == 'days' else 'precipitation'} than CMIP6. "
                "No observational reference is involved."),
            computation_notes=self._computation_notes(idx),
            period=self.ref_period, units=units, plot_type="map",
            summary_statistics=self._panel_stats(panels),
            benchmark_info=self._bench_meta(results, idx),
            extra={"index": label, "season": season or "annual"},
        )
        return fig, meta

    def _plot_bias(self, results, idx, obs_name):
        label, long_name, units, season = self._info(idx)
        ref = results["obs"][obs_name]["ref"][idx]
        target = _regular_grid(_EERIE_RES)
        fields = [(self._label(f, g), g["ref"])
                  for f, g in results["family"][idx].items()]
        mmm = results["mmm"].get(idx)
        if mmm:
            fields.append((self._mmm_label(mmm), _to_grid(mmm["ref"], *target)))
        others = [o for o in self._obs_with(results, idx) if o != obs_name]
        fields += [(o, results["obs"][o]["ref"][idx]) for o in others]
        if not fields:
            return None
        panels = [(f"{t}\nminus {obs_name}", f - ref) for t, f in fields]
        vmin, vmax = self._robust_range([p[1] for p in panels], symmetric=True)
        fig = self._map_figure(
            panels, cmap="BrBG", vmin=vmin, vmax=vmax,
            cbar_label=f"Δ{label} ({units})",
            suptitle=f"{label} bias vs {obs_name}, {self._ref_txt()}")
        stats = {f"{t} minus {obs_name}": v for t, v in
                 ((t, self._latlon_bias_stats(f, ref)) for t, f in fields)}
        stats[f"{obs_name} (reference)"] = {"global_mean": self._latlon_field_mean(ref)}
        meta = self._build_metadata(
            title=f"{label} — Bias vs {obs_name} ({self._ref_txt()})",
            figure_id=f"{idx}_bias_{obs_name.lower()}",
            models=self._meta_models(results, idx),
            description=(
                f"Bias of mean {self._per(season)} {label} ({long_name.lower()}, {units}) over "
                f"{self._ref_txt()} against {obs_name}: each EERIE model family"
                + (f", the {self._cmip6_label} multi-model mean (interpolated "
                   "bilinearly from 1° to 0.25°)" if mmm else "")
                + (f" and, as a measure of observational uncertainty, "
                   f"{', '.join(others)}" if others else "")
                + f" minus {obs_name}. Positive: more "
                + ("days" if units == "days" else "precipitation")
                + f" than {obs_name}. The observations are daily data "
                  "conservatively remapped to 0.25° and processed exactly like "
                  "the models (indices from daily values; R95p/R99p against the "
                  "dataset's own base-period percentiles). "
                + self._panel_note(results, idx)),
            computation_notes=self._computation_notes(idx),
            period=self.ref_period, units=units, plot_type="map",
            summary_statistics=stats,
            benchmark_info=self._bench_meta(results, idx),
            extra={"index": label, "season": season or "annual", **self._obs_meta([obs_name])},
        )
        return fig, meta

    def _plot_timeseries(self, results, idx):
        label, long_name, units, season = self._info(idx)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.axvspan(int(self.ref_period[0]), int(self.ref_period[1]) + 1,
                   alpha=0.10, color="#1f77b4", zorder=0)
        ax.axvspan(int(self.fut_period[0]), int(self.fut_period[1]) + 1,
                   alpha=0.10, color="#d62728", zorder=0)
        ax.axvline(_HIST_BOUNDARY_YEAR, color="k", lw=1.0, ls="--", alpha=0.5)

        stats: dict[str, dict] = {}
        cm = [results["cmip6"][m]["ts"][idx] for m in results["cmip6"]
              if idx in results["cmip6"][m]["ts"]]
        if cm:
            stack = xr.concat(cm, dim="model", join="outer")
            yrs = stack["year"].values
            ax.fill_between(yrs, stack.min("model"), stack.max("model"),
                            color="0.6", alpha=0.3, lw=0,
                            label=f"{self._cmip6_label} range ({len(cm)} models)")
            mean = stack.mean("model")
            ax.plot(yrs, mean, color="0.25", lw=2, ls="--",
                    label=f"{self._cmip6_label} MMM")
            stats[f"{self._cmip6_label} MMM"] = self._series_stats(mean)

        eerie = results["eerie"]
        for f, members in results["families"].items():
            series = {m: eerie[m]["ts"][idx] for m in members
                      if idx in eerie.get(m, {}).get("ts", {})}
            if not series:
                continue
            thin = len(series) > 1
            for m, ts in series.items():
                ax.plot(ts["year"], ts, color=self.config.get_model_color(m),
                        lw=0.9 if thin else 2.0, alpha=0.6 if thin else 1.0,
                        label=None if thin else m)
                stats[m] = self._series_stats(ts)
            if thin:
                fm = xr.concat(list(series.values()), dim="member",
                               join="outer").mean("member")
                ax.plot(fm["year"], fm, lw=2.6,
                        color=self.config.get_model_color(next(iter(series))),
                        label=f"{f} mean ({len(series)})")
                stats[f"{f} mean"] = self._series_stats(fm)

        obs = self._obs_with(results, idx)
        for i, o in enumerate(obs):
            ts = results["obs"][o]["ts"].get(idx)
            if ts is None:
                continue
            ax.plot(ts["year"], ts, color=self._obs_color(o, i), lw=2.2,
                    ls=("-", "-.", ":")[min(i, 2)], label=o, zorder=10)
            stats[o] = self._series_stats(ts)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{label} ({units})")
        ax.set_title(f"{label} — {long_name}, global area-weighted mean\n"
                     f"Blue: reference {self._ref_txt()}; red: future "
                     f"{self._fut_txt()} (SSP2-4.5)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc="best")
        fig.tight_layout()
        meta = self._build_metadata(
            title=f"{label} — Global Mean Time Series",
            figure_id=f"{idx}_timeseries",
            models=self._meta_models(results, idx),
            description=(
                f"Area-weighted global (land + ocean) mean of {self._per(season)} {label} "
                f"({units}), {self.hist_load_period[0]}–{self.ssp_load_period[1]}: "
                f"historical to {_HIST_BOUNDARY_YEAR - 1}, SSP2-4.5 after. Thin lines "
                "are individual EERIE members, thick lines family means; dashed grey "
                f"is the {self._cmip6_label} MMM with its min–max range shaded"
                + (f"; observations ({', '.join(obs)}, "
                   f"{self.obs_load_period[0]}–{self.obs_load_period[1]}) in "
                   "black/grey" if obs else "")
                + ". Each model is averaged on its own grid."),
            computation_notes=self._computation_notes(idx),
            period=(self.hist_load_period[0], self.ssp_load_period[1]),
            units=units, plot_type="timeseries", summary_statistics=stats,
            benchmark_info=self._bench_meta(results, idx),
            extra={"index": label, "season": season or "annual", **self._obs_meta(obs)},
        )
        return fig, meta

    # ── Metadata helpers ───────────────────────────────────────────────

    def _build_metadata(self, *args, extra: dict | None = None, **kwargs):
        """Registry defaults for ``pr`` name monthly ERA5 and the monthly group —
        neither applies here (own dashboard page; figures that use the daily
        observations name them through *extra*)."""
        extra = {"group": self.group, "obs_dataset": "", "obs_variable": "",
                 **(extra or {})}
        return super()._build_metadata(*args, extra=extra, **kwargs)

    @staticmethod
    def _obs_meta(obs: list[str]) -> dict:
        return {"obs_dataset": ", ".join(obs), "obs_variable": "pr"} if obs else {}

    def _meta_models(self, results, idx, future: bool = False) -> list[str]:
        key = "fut" if future else "ref"
        return [m for m, d in results["eerie"].items() if idx in d[key]]

    def _bench_meta(self, results, idx, n_key: str = "n") -> dict | None:
        mmm = results["mmm"].get(idx)
        if not mmm:
            return None
        key = "fut" if n_key == "n_fut" else "ref"
        used = [m for m, d in results["cmip6"].items() if idx in d[key]]
        return {self._cmip6_label: {"n_members": len(used), "models_used": used}}

    def _panel_note(self, results, idx) -> str:
        fams = results["family"][idx]
        parts = [f"{f} ({g['n']} member{'s' if g['n'] > 1 else ''})"
                 for f, g in fams.items()]
        mmm = results["mmm"].get(idx)
        txt = "Panels: " + ", ".join(parts) if parts else ""
        if mmm:
            txt += f"; {self._cmip6_label} multi-model mean of {mmm['n']} models."
        return txt

    def _computation_notes(self, idx: str) -> str:
        season = self._split(idx)[1]
        return (
            "Daily pr (kg m-2 s-1) ×86400 → mm/day. Indices computed per year on "
            "each model's native grid (ETCCDI definitions; wet day RR ≥ 1 mm; a "
            "cell-year with fewer than 350 valid days is missing). R95p/R99p "
            "thresholds: per-cell wet-day percentiles over "
            f"{self.base_period[0]}–{self.base_period[1]}, kept fixed for the "
            "future. Rx5day windows belong to the year of their last day. Period "
            "means bilinearly interpolated to 0.25° (EERIE) or 1° (CMIP6) before "
            "averaging across members."
            + (f" Observations ({', '.join(self._obs_cfg)}): daily totals "
               "conservatively remapped to 0.25° before the indices are computed, "
               "each with its own base-period percentiles." if self._obs_cfg else "")
            + (f" Seasonal index: the same definitions over the days of {season} "
               f"(cell-season missing below {season_min_valid_days(season)} valid "
               "days; R95p/R99p with the annual base-period thresholds; Rx5day "
               "windows ending in the season)."
               + (" DJF of year Y = December Y-1 + January-February Y, so the "
                  "first year of a run without a preceding December is missing."
                  if season == "DJF" else "") if season else "")
        )

    def _export_summary(self, results) -> None:
        out_dir = self._netcdf_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in self.indices:
            label, _, units, _season = self._info(idx)
            fields = {}
            for f, g in results["family"][idx].items():
                key = f"family_{f}".replace("-", "_")
                fields[f"{key}_ref"] = g["ref"]
                if g["change"] is not None:
                    fields[f"{key}_change"] = g["change"]
            for o in self._obs_with(results, idx):
                fields[f"obs_{o}_ref".replace("-", "_")] = results["obs"][o]["ref"][idx]
            mmm = results["mmm"].get(idx)
            if mmm:
                fields["cmip6_mmm_ref"] = _to_grid(mmm["ref"], *_regular_grid(_EERIE_RES))
                if mmm["change"] is not None:
                    fields["cmip6_mmm_change"] = _to_grid(
                        mmm["change"], *_regular_grid(_EERIE_RES))
            if not fields:
                continue
            ds = xr.Dataset(fields)
            ds.attrs.update({
                "index": label, "units": units,
                "change_units": "%" if self._split(idx)[0] in _PERCENT_CHANGE else units,
                "reference_period": self._ref_txt(), "future_period": self._fut_txt(),
            })
            write_netcdf(ds, out_dir / f"{idx}_{self.ref_period[0]}-"
                                      f"{self.fut_period[1]}_summary.nc")
