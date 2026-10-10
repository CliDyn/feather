"""ETCCDI daily-temperature extremes: present day and SSP2-4.5 change.

Computes fourteen ETCCDI indices from **daily** maximum and minimum
near-surface air temperature (``tasmax``/``tasmin``) for every EERIE member,
for an auto-discovered daily CMIP6 ensemble (one member per model) and for
daily observations, over **land only**.  Definitions and units are in
:mod:`feather.util.temp_indices` (TXx, TNx, TXn, TNn, DTR, FD, ID, TX10p,
TX90p, TN10p, TN90p, WSDI, CSDI, GSL).

This is the temperature counterpart of
:class:`~feather.diag.precip_extremes.PrecipExtremesDiag` and reuses its
periods, family means, CMIP6 multi-model mean, figure set and summary export;
only the daily-data side differs.

Land only
---------
Every index is computed on land cells alone, so the time series are land
means and ocean is blank on the maps (ETCCDI/HadEX convention; FD, ID and GSL
mean little over the sea).  A cell is land where the land-area fraction is
above 50 %: the model's own ``sftlf`` (fx) when published, else the ERA5
land-sea mask (``obs_datasets.ERA5_SFTLF``) sampled onto the model grid
(nearest neighbour).  No EERIE member publishes ``sftlf``, so they all use
the ERA5 mask; CMIP6 models mostly use their own.  Skipping the ocean also
keeps the base-period percentile computation within memory.

Units
-----
Daily values arrive in K from every loader; anything whose first-year mean
exceeds 100 is taken as Kelvin and converted to °C.  The first year's land
mean is then checked to lie within −60…50 °C, else the model is skipped.

Percentile indices
------------------
``TX10``/``TX90``/``TN10``/``TN90`` are each dataset's own calendar-day
percentiles over ``project.temp_extremes_indices.base_period`` (default: the
climate-change reference period, 1981–2000), 5-day window, no bootstrap,
held fixed for the future.  Over the base period the four exceedance indices
are therefore ≈ 10 % for every dataset by construction, so they get change
maps and time series only — no reference, bias or CMIP6-difference maps.

Seasons
-------
``project.temp_extremes_indices.seasons`` adds seasonal versions of every
index except WSDI, CSDI and GSL; DJF of year *Y* is December *Y−1* +
January–February *Y*, carried across years and from the historical run into
the SSP run when it ends the December before.

GSL
---
Northern-hemisphere cells use January–December; southern-hemisphere cells
the growing year July *Y−1* – June *Y*, labelled *Y* (so the first year of a
run is missing there).  ``TG = (TX + TN) / 2``.

Checkpoints
-----------
Annual (and seasonal, ``…_seasons.nc``) index fields on each dataset's
native grid, ocean NaN::

  ``{output_dir}/temp_extremes_indices/{model}_{hist|ssp}_{start}_{end}.nc``
  ``{output_dir}/temp_extremes_indices/cmip6/{model}_{hist|ssp}_{start}_{end}.nc``
  ``{output_dir}/temp_extremes_indices/obs/{dataset}_hist_{start}_{end}.nc``

The base-period thresholds are not stored (≈ 1 GB per 0.25° model): when
only the SSP file is missing they are recomputed from the historical run.
Open checkpoints with ``decode_timedelta=False`` (``units = "days"``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from feather.diag.netcdf_export import write_netcdf
from feather.diag.precip_extremes import PrecipExtremesDiag
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.util.precip_indices import (
    MIN_VALID_DAYS,
    SEASONS,
    season_blocks,
    season_min_valid_days,
)
from feather.util.temp_indices import (
    INDEX_INFO,
    INDICES,
    NEEDS,
    PERCENTILE_INDICES,
    SEASONAL_INDICES,
    THRESHOLDS,
    annual_indices,
    calendar_day_percentiles,
    calendar_doy,
    growing_season_length,
)

logger = logging.getLogger(__name__)

#: Daily variables read, in the order the indices use them.
VARIABLES: tuple[str, ...] = ("tasmax", "tasmin")

#: Plausible land-mean daily TX/TN (°C) for the unit check.
_PLAUSIBLE_C: tuple[float, float] = (-60.0, 50.0)

#: Land: land-area fraction above this (%).
_LAND_PERCENT: float = 50.0

#: Cells per call into :func:`annual_indices` (bounds temporary memory).
_CELL_CHUNK: int = 131_072

#: Default memory budget (GB) for the base-period block behind the
#: calendar-day percentiles; above it the cells are processed in groups,
#: re-reading the base period once per group.
_PERCENTILE_GB: float = 12.0

#: Index → colour map of the reference climatology.
_REF_CMAPS: dict[str, str] = {
    "txx": "RdYlBu_r", "tnx": "RdYlBu_r", "txn": "RdYlBu_r", "tnn": "RdYlBu_r",
    "dtr": "viridis", "fd": "cmo.ice_r", "id": "cmo.ice_r",
    "wsdi": "OrRd", "csdi": "PuBu", "gsl": "YlGn",
}


def _nearest(src: np.ndarray, tgt: np.ndarray, period: float | None = None) -> np.ndarray:
    """Index into sorted *src* of the nearest point to each *tgt* value.

    With *period* (360 for longitude) distances wrap round.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if period is not None:
        ext = np.concatenate([src - period, src, src + period])
        pos = np.clip(np.searchsorted(ext, tgt), 1, len(ext) - 1)
        left = np.where(tgt - ext[pos - 1] <= ext[pos] - tgt, pos - 1, pos)
        return left % len(src)
    pos = np.clip(np.searchsorted(src, tgt), 1, len(src) - 1)
    return np.where(tgt - src[pos - 1] <= src[pos] - tgt, pos - 1, pos)


def _carry(arr: np.ndarray | None, months: np.ndarray) -> np.ndarray | None:
    """July–December of a year: the previous half-year for DJF and SH GSL."""
    if arr is None:
        return None
    return arr[np.asarray(months) >= 7]


@register
class TempExtremesIndicesDiag(PrecipExtremesDiag):
    """ETCCDI temperature extremes for EERIE, daily CMIP6 and ERA5, land only."""

    name = "temp_extremes_indices"
    title = "Daily Temperature Extremes (ETCCDI) — Present Day and SSP2-4.5 Change"
    domain = "sfc"
    variables = ["tasmax", "tasmin"]
    group = "temperature_extremes"

    _ref_cmap = "RdYlBu_r"
    _diff_cmap = "RdBu_r"
    _domain = "land-only"
    _obs_variable = "tasmax, tasmin"

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
        DiagnosticBase.__init__(self, model_loader, obs_loader, config,
                                cmip6_loader=cmip6_loader, save_netcdf=save_netcdf)
        cc = config.project.get("climate_change", {})
        self.ref_period: tuple[str, str] = tuple(cc.get("reference_period", list(period)))
        self.fut_period: tuple[str, str] = tuple(cc.get("future_period", ["2031", "2050"]))
        self.hist_load_period: tuple[str, str] = tuple(
            cc.get("hist_load_period", [self.ref_period[0], "2014"]))
        self.ssp_load_period: tuple[str, str] = tuple(
            cc.get("ssp_load_period", ["2015", self.fut_period[1]]))
        self._cc_models: dict = cc.get("models", {})

        te = config.project.get("temp_extremes_indices", {}) or {}
        wanted = [i.lower() for i in te.get("indices", INDICES)]
        unknown = sorted(set(wanted) - set(INDICES))
        if unknown:
            logger.warning("temp_extremes_indices: unknown indices %s ignored", unknown)
        self.indices: list[str] = [i for i in INDICES if i in wanted]
        self.base_period: tuple[str, str] = tuple(te.get("base_period", self.ref_period))
        seasons = [str(x).upper() for x in te.get("seasons", []) or []]
        bad = sorted(set(seasons) - set(SEASONS))
        if bad:
            logger.warning("temp_extremes_indices: unknown seasons %s ignored", bad)
        self.seasons: list[str] = [x for x in SEASONS if x in seasons]
        self._obs_cfg: dict[str, dict] = dict(te.get("obs", {}) or {})
        self.obs_load_period: tuple[str, str] = tuple(
            te.get("obs_load_period", self.hist_load_period))
        self.percentile_gb: float = float(te.get("percentile_memory_gb", _PERCENTILE_GB))
        self._cmip6_cfg: dict = getattr(config, "cmip6_daily", {}) or {}
        self._cmip6_label: str = self._cmip6_cfg.get("label", "CMIP6")
        self._cmip6_models: dict = {}
        self._era5_frac: xr.DataArray | None | bool = False   # False = not loaded

    # ── Paths and keys ─────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / "temp_extremes_indices"

    def _season_vars(self) -> list[str]:
        return [f"{i}_{x.lower()}" for x in self.seasons for i in self.indices
                if i in SEASONAL_INDICES]

    def _info(self, key: str) -> tuple[str, str, str, str | None]:
        idx, season = self._split(key)
        label, long_name, units = INDEX_INFO[idx]
        return (f"{label} {season}" if season else label), long_name, units, season

    # ── Land mask ──────────────────────────────────────────────────────

    @staticmethod
    def _fraction_on(frac: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Land fraction (%) sampled onto *lat*/*lon* by nearest neighbour."""
        frac = frac.squeeze(drop=True)
        ren = {d: n for d, n in (("latitude", "lat"), ("longitude", "lon"))
               if d in frac.dims}
        frac = frac.rename(ren) if ren else frac
        for d in [d for d in frac.dims if d not in ("lat", "lon")]:
            frac = frac.isel({d: 0}, drop=True)
        frac = frac.assign_coords(lon=frac["lon"].values % 360.0).sortby(["lat", "lon"])
        frac = frac.drop_duplicates("lon").transpose("lat", "lon")
        iy = _nearest(frac["lat"].values, lat)
        ix = _nearest(frac["lon"].values, np.asarray(lon) % 360.0, period=360.0)
        out = np.asarray(frac.values, dtype=np.float32)[np.ix_(iy, ix)]
        if np.nanmax(out) <= 1.0 + 1e-6:      # fraction, not %
            out = out * 100.0
        return out

    def _era5_fraction(self) -> xr.DataArray | None:
        if self._era5_frac is False:
            self._era5_frac = None
            spec = (getattr(self.config, "obs_datasets", {}) or {}).get("ERA5_SFTLF")
            if spec:
                path = Path(spec["path"]) / spec["variables"]["sftlf"]
                try:
                    with xr.open_dataset(path) as ds:
                        self._era5_frac = ds["sftlf"].load()
                except (OSError, KeyError, ValueError) as exc:
                    logger.warning("temp_extremes_indices: ERA5 land mask %s "
                                   "unreadable (%s)", path, exc)
        return self._era5_frac

    def _own_fraction(self, model: str, kind: str) -> xr.DataArray | None:
        """The dataset's own ``sftlf``, or ``None``."""
        try:
            if kind == "obs":
                spec = self._obs_cfg[model]
                return self._make_obs_loader(model, spec).load_var(model, "sftlf", table="fx")
            if kind == "cmip6":
                return self._cmip6_sftlf(model)
            return self.model_loader.load_var(model, "sftlf", table="fx")
        except Exception:  # noqa: BLE001 — any failure means "no own mask"
            return None

    def _cmip6_sftlf(self, model: str) -> xr.DataArray | None:
        from feather.data.cmip6_nc_loader import experiment_dirs

        mc = self._cmip6_models.get(model)
        if mc is None:
            return None
        root = self._cmip6_cfg.get("root", "/work/ik1017/CMIP6/data/CMIP6")
        for d in experiment_dirs(root, model, self._cmip6_cfg.get("experiment", "historical"),
                                 mc.institution):
            files = (sorted(d.glob(f"{mc.variant}/fx/sftlf/*/*/*.nc"))
                     or sorted(d.glob("*/fx/sftlf/*/*/*.nc")))
            same_grid = [f for f in files if f"/{mc.grid_dir}/" in str(f)] if mc.grid_dir else []
            for f in same_grid or files:
                with xr.open_dataset(f) as ds:
                    return ds["sftlf"].load()
        return None

    def _land_cells(self, model: str, kind: str, lat: np.ndarray,
                    lon: np.ndarray) -> np.ndarray:
        """Flat indices of the land cells of a (lat, lon) grid."""
        own = self._own_fraction(model, kind)
        source = "own sftlf"
        if own is None:
            own, source = self._era5_fraction(), "ERA5 land-sea mask"
        if own is None:
            logger.warning("  %s: no land mask available — using every cell", model)
            return np.arange(len(lat) * len(lon))
        frac = self._fraction_on(own, lat, lon)
        cells = np.flatnonzero(frac.reshape(-1) > _LAND_PERCENT)
        logger.info("  %s: %d land cells of %d (%s)", model, len(cells),
                    frac.size, source)
        return cells

    # ── Daily data → indices ───────────────────────────────────────────

    @staticmethod
    def _calendar(da: xr.DataArray) -> str:
        try:
            return str(da["time"].dt.calendar)
        except (AttributeError, TypeError):
            return "standard"

    def _open(self, model: str, seg: str,
              getters: dict[str, Callable]) -> dict[str, xr.DataArray]:
        das = {}
        for var in VARIABLES:
            get = getters.get(var)
            if get is None:
                continue
            try:
                da = self._std_dims(get())
            except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
                logger.warning("  %s/%s: no daily %s (%s)", model, seg, var, exc)
                continue
            if da.sizes.get("time", 0) == 0:
                logger.warning("  %s/%s: no daily %s in the load window", model, seg, var)
                continue
            das[var] = da.sortby("lat")
        if len(das) == 2:
            a, b = das["tasmax"], das["tasmin"]
            if a.sizes["lat"] != b.sizes["lat"] or a.sizes["lon"] != b.sizes["lon"]:
                logger.warning("  %s/%s: tasmax and tasmin grids differ — using "
                               "tasmax only", model, seg)
                das.pop("tasmin")
        return das

    @staticmethod
    def _days(da: xr.DataArray) -> np.ndarray:
        """Integer day stamps (y*10000+m*100+d) for aligning TX and TN."""
        t = da["time"].dt
        return (t.year.values.astype(np.int64) * 10000 + t.month.values * 100
                + t.day.values)

    def _year(self, das: dict, year: int, cells: np.ndarray, conv: dict):
        """One year of TX/TN (°C) on the land cells, aligned on common days.

        Returns ``(arrays, months, days)`` with ``arrays = {var: (n, cells)}``.
        """
        stamps = {}
        for var, da in das.items():
            st = self._days(da)
            stamps[var] = (st, np.flatnonzero(st // 10000 == year))
        common = None
        for st, sel in stamps.values():
            common = st[sel] if common is None else np.intersect1d(common, st[sel])
        out = {}
        for var, da in das.items():
            st, sel = stamps[var]
            keep = sel[np.isin(st[sel], common)]
            vals = self._values(da.isel(time=keep), 1.0)
            flat = vals.reshape(vals.shape[0], -1)[:, cells]
            if var not in conv:
                mean = float(np.nanmean(flat)) if np.isfinite(flat).any() else float("nan")
                conv[var] = -273.15 if mean > 100.0 else 0.0
            out[var] = flat + np.float32(conv[var])
        months = (common // 100) % 100
        days = common % 100
        return out, months, days

    def _check_units(self, model: str, arrays: dict, year: int) -> None:
        for var, arr in arrays.items():
            mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
            lo, hi = _PLAUSIBLE_C
            if not lo <= mean <= hi:
                raise ValueError(
                    f"{model}: land-mean daily {var} in {year} is {mean:.4g} °C "
                    f"— outside {lo}…{hi}, so the input is neither K nor °C")
            logger.info("  %s: unit check OK (%d land-mean %s %.1f °C)",
                        model, year, var, mean)

    def _thresholds(self, model: str, das: dict, cells: np.ndarray,
                    calendar: str, conv: dict) -> dict[str, np.ndarray]:
        """Calendar-day base-period percentiles on the land cells."""
        b0, b1 = int(self.base_period[0]), int(self.base_period[1])
        out: dict[str, np.ndarray] = {}
        for var in das:
            years = sorted({int(y) for y in das[var]["time"].dt.year.values})
            base = [y for y in years if b0 <= y <= b1]
            if not base:
                logger.warning("  %s: no %s in base period %d–%d — percentile "
                               "indices will be missing", model, var, b0, b1)
                continue
            need = 366 * len(base) * len(cells) * 4 / 1e9
            n_groups = max(1, int(np.ceil(need / self.percentile_gb)))
            groups = np.array_split(np.arange(len(cells)), n_groups)
            keys = [k for k, (v, _) in THRESHOLDS.items() if v == var]
            res = {k: np.full((360 if calendar.startswith("360") else 365, len(cells)),
                              np.nan, dtype=np.float32) for k in keys}
            for g in groups:
                blocks, doys = [], []
                for y in base:
                    arrays, months, days = self._year({var: das[var]}, y, cells[g], conv)
                    blocks.append(arrays[var])
                    doys.append(calendar_doy(months, days, calendar)[0])
                n_doy = calendar_doy(np.array([1]), np.array([1]), calendar)[1]
                q = calendar_day_percentiles(np.concatenate(blocks), np.concatenate(doys),
                                             n_doy, quantiles=(0.1, 0.9))
                for k in keys:
                    res[k][:, g] = q[THRESHOLDS[k][1]]
            out.update(res)
            logger.info("  %s: %s calendar-day percentiles from %d base years "
                        "(%d–%d)%s", model, var, len(base), base[0], base[-1],
                        f", {n_groups} cell groups" if n_groups > 1 else "")
        return out

    def _stream(
        self, model: str, das: dict, cells: np.ndarray, ny: int, nx: int,
        lat: np.ndarray, thr: dict, conv: dict, calendar: str,
        prev: dict | None = None, check_units: bool = True,
    ) -> tuple[dict, dict, dict | None, list[int]]:
        """Annual and seasonal indices for every year of *das*.

        *prev* holds July–December of the year before (``arrays``,
        ``months``, ``doy``) for DJF and the southern-hemisphere GSL.
        Returns ``(annual, seasonal, carry, years)``: ``{key: (n_years,
        ny*nx)}`` arrays, the last year's July–December for the next segment
        and the years.
        """
        first = next(iter(das.values()))
        years = sorted({int(y) for y in first["time"].dt.year.values})
        n = ny * nx
        ann_keys = [i for i in self.indices if NEEDS[i] <= set(das)]
        sea_keys = [k for k in self._season_vars()
                    if NEEDS[self._split(k)[0]] <= set(das)]
        ann = {k: np.full((len(years), n), np.nan, dtype=np.float32) for k in ann_keys}
        sea = {k: np.full((len(years), n), np.nan, dtype=np.float32) for k in sea_keys}
        south = np.repeat(lat, nx)[cells] < 0.0     # row-major (lat, lon) cells

        def run(arrays, doy, min_valid, store, iy, suffix, spells):
            tx, tn = arrays.get("tasmax"), arrays.get("tasmin")
            for s in range(0, len(cells), _CELL_CHUNK):
                e = min(s + _CELL_CHUNK, len(cells))
                res = annual_indices(
                    None if tx is None else tx[:, s:e],
                    None if tn is None else tn[:, s:e],
                    doy, thresholds={k: v[:, s:e] for k, v in thr.items()},
                    min_valid_days=min_valid, spells=spells)
                for k, v in res.items():
                    key = f"{k}{suffix}"
                    if key in store:
                        store[key][iy, cells[s:e]] = v

        for iy, y in enumerate(years):
            arrays, months, days = self._year(das, y, cells, conv)
            doy = calendar_doy(months, days, calendar)[0]
            if check_units and iy == 0:
                self._check_units(model, arrays, y)
            run(arrays, doy, MIN_VALID_DAYS, ann, iy, "", True)
            if "gsl" in ann:
                ann["gsl"][iy, cells] = self._gsl(arrays, months, prev, south)
            if sea_keys:
                order = list(arrays)
                stacked = [arrays[v] for v in order] + [doy[:, None]]
                p_st = p_m = None
                if prev is not None and all(v in prev["arrays"] for v in order):
                    p_st = [prev["arrays"][v] for v in order] + [prev["doy"][:, None]]
                    p_m = prev["months"]
                splits = [season_blocks(a, months, self.seasons,
                                        None if p_st is None else p_st[i], p_m)
                          for i, a in enumerate(stacked)]
                for parts in zip(*splits):
                    season = parts[0][0]
                    blocks = {v: parts[i][1] for i, v in enumerate(order)}
                    sdoy = parts[-1][1][:, 0]
                    if len(sdoy):
                        run(blocks, sdoy, season_min_valid_days(season), sea, iy,
                            f"_{season.lower()}", False)
            prev = {"arrays": {v: _carry(a, months) for v, a in arrays.items()},
                    "months": months[months >= 7], "doy": doy[months >= 7],
                    "year": y}
        logger.info("  %s: indices for %d years (%d–%d)%s", model, len(years),
                    years[0], years[-1],
                    f" + seasons {', '.join(self.seasons)}" if sea_keys else "")
        return ann, sea, prev, years

    @staticmethod
    def _gsl(arrays: dict, months: np.ndarray, prev: dict | None,
             south: np.ndarray) -> np.ndarray:
        """GSL per land cell: Jan–Dec north, July(Y−1)–June(Y) south."""
        tx, tn = arrays["tasmax"], arrays["tasmin"]
        tg = 0.5 * (tx + tn)
        out = np.full(tg.shape[1], np.nan, dtype=np.float32)
        mid = int(np.searchsorted(months, 7))
        min_valid = MIN_VALID_DAYS
        north = ~south
        if north.any():
            out[north] = growing_season_length(tg[:, north], mid, min_valid)
        if south.any() and prev is not None:
            ptx, ptn = prev["arrays"].get("tasmax"), prev["arrays"].get("tasmin")
            if ptx is not None and ptn is not None and len(ptx):
                first_half = months < 7
                series = np.concatenate([0.5 * (ptx + ptn)[:, south],
                                         tg[first_half][:, south]])
                out[south] = growing_season_length(series, len(ptx), min_valid)
        return out

    def _as_ds(self, store: dict, years: list[int], ny: int, nx: int,
               lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
        coords = {"year": np.asarray(years), "lat": lat, "lon": lon}
        data_vars = {}
        for k, vals in store.items():
            label, long_name, units, _season = self._info(k)
            data_vars[k] = xr.DataArray(
                vals.reshape(len(years), ny, nx), dims=("year", "lat", "lon"),
                coords=coords,
                attrs={"long_name": f"{label}: {long_name}", "units": units})
        return xr.Dataset(data_vars)

    def _load_checkpoint(self, path: Path) -> xr.Dataset | None:
        """Annual (+ seasonal) checkpoint, or ``None`` if anything is missing."""
        if not path.exists():
            return None
        ann = xr.open_dataset(path, decode_timedelta=False)
        sea_vars = self._season_vars()
        if not sea_vars:
            return ann
        sea_path = self._season_path(path)
        if not sea_path.exists():
            return None
        sea = xr.open_dataset(sea_path, decode_timedelta=False)
        have = set(sea.data_vars)
        # Indices whose inputs the run lacked are legitimately absent; only a
        # season with no field at all means the file predates that season.
        for season in self.seasons:
            if not any(v.endswith(f"_{season.lower()}") for v in have):
                logger.info("  %s lacks %s — recomputing", sea_path.name, season)
                return None
        return self._merge(ann, sea)

    def _segment(self, model, seg, path, getters, state=None, prev_fn=None,
                 thresholds=None):
        """Load one segment's checkpoints, computing them if missing.

        *state* is shared by the segments of one model: land cells, unit
        conversions, thresholds (computed from the historical run) and the
        July–December carry.
        """
        state = {} if state is None else state
        done = self._load_checkpoint(path)
        if done is not None:
            logger.info("  %s/%s: loading indices from %s", model, seg, path.name)
            return done
        if not getters:
            return None
        das = self._open(model, seg, getters)
        if not das:
            return None
        first = next(iter(das.values()))
        lat, lon = first["lat"].values, first["lon"].values
        ny, nx = len(lat), len(lon)
        calendar = self._calendar(first)
        if "cells" not in state or state.get("shape") != (ny, nx):
            state["cells"] = self._land_cells(model, state.get("kind", "eerie"), lat, lon)
            state["shape"] = (ny, nx)
        cells = state["cells"]
        conv = state.setdefault("conv", {})

        if "thr" not in state:
            base_das = das if seg == "hist" else state.get("hist_das", lambda: {})()
            state["thr"] = self._thresholds(model, base_das, cells, calendar, conv) \
                if base_das else {}
        prev = prev_fn() if prev_fn else None
        if prev is not None:
            y0 = int(first["time"].dt.year.values[0])
            m0 = int(first["time"].dt.month.values[0])
            if prev.get("year") != y0 - 1 or m0 != 1 or len(prev["months"]) == 0 \
                    or int(prev["months"][-1]) != 12:
                logger.info("  %s/%s: previous segment does not end the December "
                            "before %d-%02d — not carried over", model, seg, y0, m0)
                prev = None
        try:
            ann, sea, carry, years = self._stream(
                model, das, cells, ny, nx, lat, state["thr"], conv, calendar, prev)
        except ValueError as exc:
            logger.error("  %s/%s: %s — skipping", model, seg, exc)
            return None
        state["carry"] = carry

        meta = {
            "model": model, "segment": seg,
            "base_period": f"{self.base_period[0]}-{self.base_period[1]}",
            "domain": "land only (land-area fraction > 50 %)",
            "daily_mean": "TG = (TX + TN) / 2",
            "inputs": ", ".join(das),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = self._as_ds(ann, years, ny, nx, lat, lon)
        ds.attrs.update(meta)
        sea_ds = None
        if sea:
            sea_ds = self._as_ds(sea, years, ny, nx, lat, lon)
            sea_ds.attrs.update(meta)
            sea_ds.attrs["djf_note"] = ("DJF of year Y = December Y-1 + January-"
                                        "February Y")
            write_netcdf(sea_ds, self._season_path(path),
                         encoding={v: {"zlib": True, "complevel": 1}
                                   for v in sea_ds.data_vars})
            logger.info("  Saved %s", self._season_path(path))
        write_netcdf(ds, path, encoding={v: {"zlib": True, "complevel": 1}
                                         for v in ds.data_vars})
        logger.info("  Saved %s", path)
        return self._merge(ds, sea_ds)

    def _model_indices(self, model, get_hist, get_fut, cmip6: bool = False,
                       kind: str | None = None):
        kind = kind or ("cmip6" if cmip6 else "eerie")
        state: dict[str, Any] = {"kind": kind}
        state["hist_das"] = lambda: self._open(model, "hist", get_hist)
        obs = kind == "obs"
        hist = self._segment(model, "hist",
                             self._nc_path(model, "hist", cmip6=cmip6, obs=obs),
                             get_hist, state)
        if hist is None:
            return None
        ssp = None
        if get_fut:
            def prev():
                # Same run, so the stream's own carry is right when the
                # historical segment was just computed; otherwise re-read
                # the last historical year.
                if state.get("carry") is not None:
                    return state["carry"]
                try:
                    das = state["hist_das"]()
                    y = max(int(v) for da in das.values()
                            for v in da["time"].dt.year.values)
                    das = {v: da.sel(time=str(y)) for v, da in das.items()}
                    cells = state["cells"]
                    arrays, months, days = self._year(
                        das, y, cells, state.setdefault("conv", {}))
                    doy = calendar_doy(months, days,
                                       self._calendar(next(iter(das.values()))))[0]
                    return {"arrays": {v: _carry(a, months) for v, a in arrays.items()},
                            "months": months[months >= 7], "doy": doy[months >= 7],
                            "year": y}
                except Exception:   # noqa: BLE001 — only DJF/SH GSL of one year
                    return None

            ssp = self._segment(model, "ssp",
                                self._nc_path(model, "ssp", cmip6=cmip6),
                                get_fut, state, prev_fn=prev)
        return {"hist": hist, "ssp": ssp}

    # ── Data sources ───────────────────────────────────────────────────

    def _eerie_sources(self) -> dict[str, tuple[dict, dict | None]]:
        out = {}
        for model in self.config.models:
            if model in self._obs_cfg:
                # e.g. ERA5 listed as a flat "model" for tropical_nights /
                # heatwave: here it is the observational reference only.
                logger.info("  %s: configured as observations — not an "
                            "ensemble member here", model)
                continue
            hist = {v: (lambda m=model, v=v: self.model_loader.load_var(
                m, v, table="day", period=self.hist_load_period)) for v in VARIABLES}
            fut = None
            if model in self._cc_models:
                ldr = self._make_fut_loader(model)
                if ldr is not None:
                    fut = {v: (lambda m=model, v=v, ldr=ldr: ldr.load_var(
                        m, v, table="day", period=self.ssp_load_period))
                        for v in VARIABLES}
            out[model] = (hist, fut)
        return out

    def _obs_sources(self) -> dict[str, dict]:
        out = {}
        for name, spec in self._obs_cfg.items():
            spec = spec or {}
            if not spec.get("data_root"):
                logger.warning("temp_extremes_indices: obs %s has no data_root — "
                               "skipped", name)
                continue
            out[name] = {v: (lambda n=name, sp=spec, v=v: self._make_obs_loader(
                n, sp).load_var(n, v, table="day", period=self.obs_load_period))
                for v in VARIABLES}
        return out

    def _obs_indices(self, name: str, get_hist: dict) -> xr.Dataset | None:
        r = self._model_indices(name, get_hist, None, kind="obs")
        return None if r is None else r["hist"]

    def _cmip6_sources(self, loader, model: str) -> tuple[dict, dict]:
        hist = {v: (lambda m=model, v=v: loader.load_var(
            m, v, table="day", period=self.hist_load_period)) for v in VARIABLES}
        fut = {v: (lambda m=model, v=v: loader.load_var(
            m, v, table="day", period=self.ssp_load_period)) for v in VARIABLES}
        return hist, fut

    def _discover_cmip6(self):
        """Daily CMIP6 models whose member has tasmax AND tasmin, hist + future."""
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
                root, experiment=hist_exp, table="day", variable="tasmax",
                member=cfg.get("member", "r1i1p1f1"),
                exclude=tuple(cfg.get("exclude", [])), require_also=(fut_exp,))
        except Exception as exc:  # noqa: BLE001 — discovery must not kill the run
            logger.warning("CMIP6 daily discovery failed (%s) — EERIE only", exc)
            return None, []
        wanted = cfg.get("models", "auto")
        if isinstance(wanted, (list, tuple, set)):
            found = [m for m in found if m.name in set(wanted)]

        def has(mc, exp, var):
            return any(_pd.variable_files(d, mc.variant, "day", var)
                       for d in experiment_dirs(root, mc.name, exp, mc.institution))

        keep, dropped = [], []
        for mc in found:
            if all(has(mc, e, v) for e in (hist_exp, fut_exp) for v in VARIABLES):
                mc.experiments = [hist_exp, fut_exp]
                keep.append(mc)
            else:
                dropped.append(f"{mc.name}({mc.variant})")
        if dropped:
            logger.info("CMIP6 daily: member lacks tasmin or tasmax in %s/%s — "
                        "skipped: %s", hist_exp, fut_exp, ", ".join(dropped))
        if cfg.get("max_models"):
            keep = keep[: int(cfg["max_models"])]
        if not keep:
            logger.warning("CMIP6 daily: no models with daily tasmax+tasmin in "
                           "%s + %s", hist_exp, fut_exp)
            return None, []
        self._cmip6_models = {m.name: m for m in keep}
        logger.info("CMIP6 daily: %d models — %s", len(keep),
                    ", ".join(f"{m.name}({m.variant})" for m in keep))
        return CMIP6NCLoader.for_models(self.config, keep, root=root), keep

    # ── Figures ────────────────────────────────────────────────────────

    def _figure_fns(self, idx: str) -> list[Callable]:
        if self._split(idx)[0] in PERCENTILE_INDICES:
            return [self._plot_change, self._plot_timeseries]
        return super()._figure_fns(idx)

    def _bias_wanted(self, idx: str) -> bool:
        return self._split(idx)[0] not in PERCENTILE_INDICES

    def _ref_cmap_for(self, idx: str) -> str:
        return _REF_CMAPS.get(self._split(idx)[0], self._ref_cmap)

    def _ref_vmin(self, idx: str, vmin: float) -> float:
        units = INDEX_INFO[self._split(idx)[0]][2]
        return vmin if units == "°C" else max(0.0, vmin)

    def _ref_definition(self, idx: str) -> str:
        return ("from daily maximum/minimum 2 m temperature over land "
                "(land-area fraction > 50 %)")

    def _more(self, units: str) -> str:
        return {"days": "days", "%": "exceedance days"}.get(
            units, "degrees (a higher index value)")

    def _change_note(self, idx: str) -> str:
        base = self._split(idx)[0]
        if base in PERCENTILE_INDICES or base in ("wsdi", "csdi"):
            return (" Exceedances are counted against each model's own "
                    f"{self.base_period[0]}–{self.base_period[1]} calendar-day "
                    "percentiles (5-day window), held fixed for the future"
                    + ("; the change is in percentage points of days."
                       if base in PERCENTILE_INDICES else "."))
        return ""

    def _obs_processing_note(self) -> str:
        return ("The observations are daily maximum/minimum temperatures on a "
                "0.25° grid, processed exactly like the models (indices from "
                "daily values over the dataset's own land mask, own base-period "
                "percentiles). ")

    def _computation_notes(self, idx: str) -> str:
        base, season = self._split(idx)
        return (
            "Daily tasmax/tasmin (K → °C) over land only (land-area fraction "
            "> 50 %: own sftlf, else the ERA5 land-sea mask). Indices computed per "
            "year on each model's native grid (ETCCDI definitions as in C3S "
            "sis-extreme-indices-cmip6; a cell-year with fewer than 350 valid days "
            "is missing). Percentile thresholds: calendar-day 10th/90th "
            f"percentiles over {self.base_period[0]}–{self.base_period[1]}, 5-day "
            "window, Hyndman-Fan type 8, no bootstrap, fixed for the future; within "
            "the base period TX10p/TX90p/TN10p/TN90p are therefore ≈ 10 % by "
            "construction. WSDI/CSDI: days in runs of ≥ 6 days, runs not "
            "continuing across years. GSL: TG = (TX + TN)/2; NH January–December, "
            "SH July–June labelled by the ending year. Period means bilinearly "
            "interpolated to 0.25° (EERIE) or 1° (CMIP6) before averaging across "
            "members."
            + (f" Seasonal index: the same definitions over the days of {season} "
               f"(cell-season missing below {season_min_valid_days(season)} valid "
               "days)."
               + (" DJF of year Y = December Y-1 + January-February Y."
                  if season == "DJF" else "") if season else "")
        )
