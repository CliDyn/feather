"""Marine heatwaves (Hobday et al., 2016) in daily SST: EERIE vs obs and CMIP6.

Follows the surface analysis of de Toma et al. (2026, EGUsphere-2026-5747,
"Comparing Surface and Subsurface Marine Heatwaves in Ocean Reanalyses";
code: github.com/vincenzodetoma/ROHAN), applied to the EERIE members instead
of the GREP reanalyses: marine heatwaves (MHWs) are detected independently
in every ocean cell of every dataset's **daily** SST, events are reduced to
per-cell maps, and the maps are compared with daily observations.

Detection
---------
:mod:`feather.util.mhw` — 90th percentile of an 11-day window (±5 days) per
calendar day over the climatology period, climatology and threshold smoothed
with a 31-day running mean, events ≥ 5 days, gaps ≤ 2 days joined, no
detrending (the paper's main set-up).  All set in
``project.marine_heatwaves``.

Per cell: event frequency, percentage of MHW days, mean duration, maximum
intensity over all events, mean peak / mean / cumulative intensity, mean
within-event intensity variability, mean onset and decline rates, maximum
category, and per-year MHW days and events (intensities relative to the
seasonal climatology, °C).

Data
----
* EERIE members: daily ``tos`` (``table="Oday"``) on their 0.25° grid —
  CMOR ``Oday`` (r1 members), the IFS-FESOM2 kerchunk ocean store and the
  ICON-ESM-ER kerchunk ``ocean_2d_daily_mean_remap025`` store (r2/r3).  A
  member without daily SST is skipped with a warning.
* Observations (``marine_heatwaves.obs``): daily SST laid out as CMOR trees
  on the same 0.25° grid (``scripts/oisst_derive_tos_daily.sh`` for NOAA
  OISST v2.1, ``scripts/era5_derive_tos_daily.sh`` for ERA5), processed
  exactly like a member.  The first (or ``reference_obs``) is the reference.
* CMIP6 (``cmip6_daily``): daily ``Oday/tos`` of the historical run, one
  member per model, box-averaged from the native ocean grid onto 1° before
  detection (empty 1° cells take the nearest source point within 1°).  So
  every dataset is evaluated at its own resolution, as the paper argues it
  should be: coarse models have fewer, weaker, longer events.

Data are read one block of latitude rows at a time (all ocean cells of the
block × all days), sized by ``memory_gb``; cells are detected in chunks of
``cell_chunk`` on ``workers`` threads (numpy releases the GIL).  Per-dataset
results are checkpointed to ``{output}/marine_heatwaves/[obs/|cmip6/]
{name}_{y0}_{y1}.nc`` and reused while the detection settings match.

Masking
-------
Cells whose reference-observation annual-mean climatology is below
``ice_mask_sst`` (default −1 °C, i.e. mostly ice-covered) are masked in
every figure: SST pinned at the freezing point carries no heatwave signal.

Figures (per metric ``m``)
--------------------------
``mhw_{m}_maps``      observations, family (or ensemble) means, CMIP6 MMM
``mhw_{m}_members``   reference observation and every member
``mhw_{m}_bias``      family means, other obs and the CMIP6 MMM minus the
                      reference; grey where a family's bias is within twice
                      its member spread (multi-member families)
``mhw_{m}_spread``    member standard deviation per family (≥ 2 members)

Plus ``mhw_zonal_means`` (zonal means of the main metrics),
``mhw_days_timeseries`` (yearly ocean-mean MHW days, 60°S–60°N) and
``mhw_regions_{region}`` (cell distributions in the paper's areas of
interest, Table 2).
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.precip_extremes import PrecipExtremesDiag, _regular_grid, _to_grid
from feather.diag.registry import register
from feather.util import mhw

logger = logging.getLogger(__name__)

_EERIE_RES: float = 0.25
_CMIP6_RES: float = 1.0

#: Plausible ocean-mean SST (°C) of one day, after the K → °C check.
_PLAUSIBLE_SST: tuple[float, float] = (-5.0, 35.0)

#: Latitude band of the ocean-mean MHW-day series (keeps sea ice out).
_SERIES_BAND: tuple[float, float] = (-60.0, 60.0)

#: Areas of interest of de Toma et al. (2026), Table 2: name → (label,
#: lat0, lat1, lon0, lon1), longitudes 0–360.
REGIONS: dict[str, tuple[str, float, float, float, float]] = {
    "western_australia": ("Western Australia", -40.0, -20.0, 90.0, 120.0),
    "ne_pacific": ("Northeast Pacific (Blob)", 15.0, 70.0, 180.0, 260.0),
    "mediterranean": ("Mediterranean Sea", 30.0, 50.0, 0.0, 30.0),
    "tasman_sea": ("Tasman Sea", -50.0, -35.0, 140.0, 160.0),
    "great_barrier_reef": ("Great Barrier Reef", -25.0, -5.0, 135.0, 155.0),
    "santa_barbara": ("Santa Barbara", 30.0, 37.0, 235.0, 245.0),
    "nw_atlantic": ("Northwest Atlantic", 35.0, 50.0, 280.0, 310.0),
}

#: Metrics in the zonal-mean and region figures.
_SUMMARY_METRICS: tuple[str, ...] = (
    "duration_mean", "intensity_max_max", "intensity_cumulative_mean",
    "rate_onset_mean", "rate_decline_mean", "mhw_days_pct",
)

#: Colour maps per metric (sequential); differences use RdBu_r.
_CMAPS: dict[str, str] = {
    "frequency": "YlOrRd", "mhw_days_pct": "YlOrRd", "duration_mean": "viridis",
    "intensity_max_max": "magma_r", "intensity_max_mean": "magma_r",
    "intensity_mean_mean": "magma_r", "intensity_cumulative_mean": "inferno_r",
    "intensity_var_mean": "PuRd", "rate_onset_mean": "YlGnBu",
    "rate_decline_mean": "YlGnBu", "category_max": "YlOrRd",
}

_CATEGORY_LABELS = ("none", "moderate", "strong", "severe", "extreme")

_DETECT_KEYS = ("pctile", "window_half_width", "smooth_width", "min_duration",
                "max_gap")


def _lat_txt(lat: float) -> str:
    return f"{abs(lat):g}°{'S' if lat < 0 else 'N'}"


@register
class MarineHeatwavesDiag(DiagnosticBase):
    """Hobday marine heatwaves in daily SST for EERIE, obs and daily CMIP6."""

    name = "marine_heatwaves"
    title = "Marine Heatwaves (Hobday et al. 2016) in Daily SST"
    domain = "o2d"
    variables = ["tos"]
    group = "marine_heatwaves"

    #: Display resolution (°) of the map panels; finer fields are averaged.
    plot_resolution: float = 0.25

    _make_obs_loader = PrecipExtremesDiag._make_obs_loader
    _robust_range = staticmethod(PrecipExtremesDiag._robust_range)

    def __init__(
        self,
        model_loader,
        obs_loader,
        config,
        *,
        cmip6_loader=None,
        experiment: str = "hist-1950",
        period: tuple[str, str] = ("1982", "2014"),
        variables=None,
        save_netcdf: bool = False,
    ):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, save_netcdf=save_netcdf)
        cfg = self._config_block(config)
        self.period: tuple[str, str] = tuple(
            str(x) for x in cfg.get("period", ["1982", "2014"]))
        self.clim_period: tuple[str, str] = tuple(
            str(x) for x in cfg.get("climatology_period", self.period))
        self.params: dict[str, float] = {
            "pctile": float(cfg.get("pctile", mhw.PCTILE)),
            "window_half_width": int(cfg.get("window_half_width",
                                             mhw.WINDOW_HALF_WIDTH)),
            "smooth_width": int(cfg.get("smooth_width", mhw.SMOOTH_WIDTH)),
            "min_duration": int(cfg.get("min_duration", mhw.MIN_DURATION)),
            "max_gap": int(cfg.get("max_gap", mhw.MAX_GAP)),
        }
        wanted = cfg.get("metrics", list(mhw.METRICS))
        unknown = sorted(set(wanted) - set(mhw.METRICS))
        if unknown:
            logger.warning("marine_heatwaves: unknown metrics %s ignored", unknown)
        self.metrics: list[str] = [m for m in mhw.METRICS if m in wanted]
        self._obs_cfg: dict[str, dict] = dict(cfg.get("obs", {}) or {})
        self.reference: str | None = cfg.get("reference_obs") or next(
            iter(self._obs_cfg), None)
        self.memory_gb: float = float(cfg.get("memory_gb", 64))
        self.workers: int = int(cfg.get("workers", 8))
        self.cell_chunk: int = int(cfg.get("cell_chunk", 5_000))
        self.ice_sst: float | None = cfg.get("ice_mask_sst", -1.0)
        self.regions: dict = {**REGIONS, **(cfg.get("regions") or {})}
        self._cmip6_cfg: dict = getattr(config, "cmip6_daily", {}) or {}
        self._cmip6_label: str = self._cmip6_cfg.get("label", "CMIP6")

    #: ``project`` blocks merged (later ones win) into the settings.
    _CONFIG_BLOCKS: tuple[str, ...] = ("marine_heatwaves",)

    @classmethod
    def _config_block(cls, config) -> dict:
        out: dict = {}
        for b in cls._CONFIG_BLOCKS:
            out.update(config.project.get(b, {}) or {})
        return out

    # ── Paths and checkpoints ──────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / "marine_heatwaves"

    def _nc_path(self, name: str, kind: str = "eerie") -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        base = {"eerie": self.nc_dir, "obs": self.nc_dir / "obs",
                "cmip6": self.nc_dir / "cmip6"}[kind]
        return base / f"{safe}_{self.period[0]}_{self.period[1]}.nc"

    def _settings(self) -> dict:
        return {**self.params, "period": "-".join(self.period),
                "climatology_period": "-".join(self.clim_period)}

    def _load_checkpoint(self, path: Path) -> xr.Dataset | None:
        if not path.exists():
            return None
        ds = xr.open_dataset(path).load()
        stale = {k: (ds.attrs.get(k), v) for k, v in self._settings().items()
                 if str(ds.attrs.get(k)) != str(v)}
        if stale:
            logger.warning("  %s: checkpoint settings differ %s — recomputing",
                           path.name, stale)
            return None
        # Coordinates of the regridded stores carry ~1e-14 noise; snap them so
        # datasets align exactly.
        return ds.assign_coords(lat=np.round(ds["lat"].values, 9),
                                lon=np.round(ds["lon"].values, 9))

    def _dataset(self, path: Path, getter: Callable, cmip6: bool = False
                 ) -> xr.Dataset | None:
        ds = self._load_checkpoint(path)
        if ds is not None:
            logger.info("  loaded %s", path.name)
            return ds
        try:
            da = getter()
        except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("  no daily SST (%s) — skipped", exc)
            return None
        try:
            ds = self._detect_regrid(da) if cmip6 else self._detect_regular(da)
        except ValueError as exc:
            logger.error("  %s — skipped", exc)
            return None
        ds.attrs.update({k: str(v) for k, v in self._settings().items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
        ds.to_netcdf(path, encoding=enc)
        logger.info("  saved %s", path)
        return ds

    # ── Daily data → per-cell maps ─────────────────────────────────────

    @staticmethod
    def _std(da: xr.DataArray, zdim: str | None = None) -> xr.DataArray:
        """``(time, lat, lon)`` — or ``(time, zdim, lat, lon)`` — south → north."""
        ren = {d: n for d, n in (("latitude", "lat"), ("longitude", "lon"))
               if d in da.dims}
        da = da.rename(ren) if ren else da
        extra = [d for d in da.dims if d not in ("time", "lat", "lon", zdim)]
        for d in extra:
            if da.sizes[d] != 1:
                raise ValueError(f"unexpected dimension {d!r} of size {da.sizes[d]}")
        da = da.isel({d: 0 for d in extra}, drop=True)
        da = da.transpose("time", *([zdim] if zdim else []), "lat", "lon")
        if da["lat"].values[0] > da["lat"].values[-1]:
            da = da.isel(lat=slice(None, None, -1))
        return da

    @staticmethod
    def _time_parts(da: xr.DataArray) -> tuple[np.ndarray, ...]:
        t = da["time"].dt
        cal = getattr(t, "calendar", "standard") or "standard"
        return (t.year.values.astype(int), t.month.values.astype(int),
                t.day.values.astype(int), str(cal))

    def _full_axis(self, cal: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every day of the period in calendar *cal*: (years, months, days)."""
        start = f"{self.period[0]}-01-01"
        last = f"{self.period[1]}-12-{30 if cal == '360_day' else 31}"
        if cal in ("standard", "gregorian", "proleptic_gregorian"):
            import pandas as pd
            t = pd.date_range(start, last, freq="D")
            return t.year.values, t.month.values, t.day.values
        t = xr.date_range(start, last, freq="D", calendar=cal, use_cftime=True)
        return (np.array([x.year for x in t]), np.array([x.month for x in t]),
                np.array([x.day for x in t]))

    @staticmethod
    def _positions(years, months, days, fy, fm, fd) -> tuple[np.ndarray, np.ndarray]:
        """Index of each data day on the full axis, and which days matched."""
        key = years * 10000 + months * 100 + days
        fkey = fy * 10000 + fm * 100 + fd
        pos = np.searchsorted(fkey, key).clip(0, len(fkey) - 1)
        ok = fkey[pos] == key
        # Keep the first of duplicated stamps.
        first = np.zeros(len(key), dtype=bool)
        _, idx = np.unique(np.where(ok, pos, -1), return_index=True)
        first[idx] = True
        return pos, ok & first

    @staticmethod
    def _offset(sample: np.ndarray) -> float:
        """0 for °C, −273.15 for K; raises on implausible values."""
        mean = float(np.nanmean(sample)) if np.isfinite(sample).any() else np.nan
        off = -273.15 if mean > 100.0 else 0.0
        lo, hi = _PLAUSIBLE_SST
        if not lo <= mean + off <= hi:
            raise ValueError(f"mean SST of the first day is {mean:.4g} — not °C or K")
        return off

    @staticmethod
    def _values(sel: xr.DataArray) -> np.ndarray:
        import dask
        with dask.config.set(scheduler="threads"):
            return np.asarray(sel.values, dtype=np.float32)

    def _detect_cells(self, arr: np.ndarray, axis: tuple) -> dict[str, np.ndarray]:
        """:func:`mhw.detect` on chunks of *arr* (cells, days), in threads."""
        fy, fm, fd, cal = axis
        base = (int(self.clim_period[0]), int(self.clim_period[1]))
        n = arr.shape[0]
        bounds = list(range(0, n, self.cell_chunk)) + [n]

        def run(i):
            _, maps = mhw.detect(arr[bounds[i]:bounds[i + 1]], fy, fm, fd,
                                 calendar=cal, base_years=base, **self.params)
            return maps

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            parts = list(ex.map(run, range(len(bounds) - 1)))
        out = {"years": parts[0]["years"]}
        for k in parts[0]:
            if k != "years":
                out[k] = np.concatenate([p[k] for p in parts], axis=-1)
        return out

    def _row_blocks(self, ocean: np.ndarray, nt: int) -> list[tuple[int, int]]:
        """Latitude-row blocks whose ocean cells × days fit ``memory_gb``.

        *ocean* is a (lat, lon) mask or count of ocean cells (summed over
        levels when several are read together).  The workers' temporaries
        (``workers × cell_chunk`` cells at :data:`mhw.BYTES_PER_CELL_DAY`)
        come off the budget first; the rest holds the block as float32.
        Each block re-reads the daily files for its rows, so fewer, larger
        blocks are faster.
        """
        temps = self.workers * self.cell_chunk * nt * mhw.BYTES_PER_CELL_DAY
        budget = self.memory_gb * 1e9 - temps
        if budget < 0.25 * self.memory_gb * 1e9:
            logger.warning("  marine_heatwaves: workers × cell_chunk use %.0f of "
                           "%.0f GB — lower them or raise memory_gb",
                           temps / 1e9, self.memory_gb)
            budget = 0.25 * self.memory_gb * 1e9
        max_cells = max(int(budget // (4 * nt)), int(ocean.shape[1]))
        per_row = ocean.sum(axis=1)
        blocks, r0, acc = [], 0, 0
        for r, c in enumerate(per_row):
            if acc + c > max_cells and r > r0:
                blocks.append((r0, r))
                r0, acc = r, 0
            acc += c
        blocks.append((r0, len(per_row)))
        return blocks

    def _detect_regular(self, da: xr.DataArray) -> xr.Dataset:
        """Per-cell maps of a (time, lat, lon) daily SST field."""
        da = self._std(da).expand_dims("_level", axis=1)
        return self._detect_levels(da, "_level", [0])[0]

    def _detect_levels(self, da: xr.DataArray, zdim: str,
                       zidx: list[int]) -> list[xr.Dataset]:
        """Per-cell maps of levels *zidx* of a (time, zdim, lat, lon) field.

        All requested levels of a block of latitude rows are read together:
        the daily files store neighbouring levels in one chunk, so reading
        them level by level would decompress every chunk once per level.
        Detection still runs on each level independently.  A level without
        any ocean cell gives an empty dataset.
        """
        ty, tm, td, cal = self._time_parts(da)
        fy, fm, fd = self._full_axis(cal)
        pos, ok = self._positions(ty, tm, td, fy, fm, fd)
        nt = len(fy)
        if ok.sum() < 0.9 * nt:
            raise ValueError(f"only {ok.sum()} of {nt} days of "
                             f"{self.period[0]}–{self.period[1]}")
        first = int(np.flatnonzero(ok)[0])
        sample = self._values(da.isel({"time": first, zdim: list(zidx)}))
        off = np.float32(self._offset(sample[0]))
        ocean = np.isfinite(sample)                         # (levels, lat, lon)
        nz, ny, nx = ocean.shape
        logger.info("  %s ocean cells, %d days (%d missing), calendar %s",
                    " + ".join(str(int(o.sum())) for o in ocean), nt,
                    nt - int(ok.sum()), cal)
        years = np.unique(fy)
        flats = [np.flatnonzero(o.ravel()) for o in ocean]
        maps: list[dict[str, np.ndarray]] = [{} for _ in range(nz)]
        blocks = self._row_blocks(ocean.sum(axis=0), nt)
        logger.info("  %d block(s) of latitude rows", len(blocks))
        for r0, r1 in blocks:
            seas = [o[r0:r1].ravel() for o in ocean]
            if not any(s.any() for s in seas):
                continue
            arrs = [np.full((int(s.sum()), nt), np.nan, dtype=np.float32)
                    for s in seas]
            for y in np.unique(ty[ok]):
                ti = np.flatnonzero((ty == y) & ok)
                block = self._values(da.isel({"time": ti, zdim: list(zidx),
                                              "lat": slice(r0, r1)}))
                block = block.reshape(len(ti), nz, -1)
                for lev, s in enumerate(seas):
                    if s.any():
                        arrs[lev][:, pos[ti]] = block[:, lev][:, s].T + off
                del block
            for lev, s in enumerate(seas):
                if not s.any():
                    continue
                logger.info("    rows %d–%d%s: %d cells", r0, r1,
                            f", level {zidx[lev]}" if nz > 1 else "",
                            arrs[lev].shape[0])
                part = self._detect_cells(arrs[lev], (fy, fm, fd, cal))
                arrs[lev] = None
                flat = flats[lev]
                sel = (flat >= r0 * nx) & (flat < r1 * nx)
                for k, v in part.items():
                    if k == "years":
                        continue
                    if k not in maps[lev]:
                        shape = v.shape[:-1] + (ny * nx,)
                        maps[lev][k] = np.full(shape, np.nan, dtype=np.float32)
                    maps[lev][k][..., flat[sel]] = v
        lat, lon = da["lat"].values, da["lon"].values
        return [self._as_dataset(m, years, lat, lon) for m in maps]

    @staticmethod
    def _as_dataset(maps: dict, years: np.ndarray, lat: np.ndarray,
                    lon: np.ndarray) -> xr.Dataset:
        ny, nx = len(lat), len(lon)
        data = {}
        for k, v in maps.items():
            if v.ndim == 2:
                data[k] = (("year", "lat", "lon"), v.reshape(len(years), ny, nx))
            else:
                data[k] = (("lat", "lon"), v.reshape(ny, nx))
        return xr.Dataset(data, coords={"year": years.astype(int),
                                        "lat": np.round(lat, 9),
                                        "lon": np.round(lon % 360.0, 9)})

    # ── CMIP6: native ocean grid → 1° → detection ──────────────────────

    @staticmethod
    def _latlon_of(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        """Lat/lon of every spatial point of *da*, in its dimension order."""
        tmpl = da.isel(time=0, drop=True)
        found = {}
        for axis, names in (("lat", ("lat", "latitude", "nav_lat")),
                            ("lon", ("lon", "longitude", "nav_lon"))):
            name = next((n for n in names if n in da.coords), None)
            if name is None:
                name = next((c for c in da.coords if da[c].attrs.get(
                    "standard_name") == ("latitude" if axis == "lat" else "longitude")),
                    None)
            if name is None:
                raise ValueError(f"no {axis} coordinate")
            c = da[name].reset_coords(drop=True)
            found[axis] = np.asarray(
                xr.broadcast(c, tmpl)[0].transpose(*tmpl.dims).values, dtype=np.float64)
        return found["lat"], found["lon"]

    @staticmethod
    def _box_matrix(lat: np.ndarray, lon: np.ndarray, res: float):
        """Sparse averaging of points into ``res``° cells, plus target cells.

        Every point falls in one target cell; target cells without a point
        take the nearest point if it lies within ``res``°.
        """
        from scipy.sparse import csr_matrix, vstack
        from scipy.spatial import cKDTree

        nlat, nlon = int(round(180 / res)), int(round(360 / res))
        iy = np.clip(np.floor((lat + 90.0) / res), 0, nlat - 1).astype(np.int64)
        ix = (np.floor((lon % 360.0) / res) % nlon).astype(np.int64)
        tgt = iy * nlon + ix
        cells, inv = np.unique(tgt, return_inverse=True)
        n = len(lat)
        m = csr_matrix((np.ones(n, np.float32), (inv, np.arange(n))),
                       shape=(len(cells), n))

        def xyz(la, lo):
            la, lo = np.radians(la), np.radians(lo)
            return np.column_stack([np.cos(la) * np.cos(lo),
                                    np.cos(la) * np.sin(lo), np.sin(la)])

        tlat = -90.0 + res * (np.arange(nlat) + 0.5)
        tlon = res * (np.arange(nlon) + 0.5)
        empty = np.setdiff1d(np.arange(nlat * nlon), cells)
        if len(empty):
            tree = cKDTree(xyz(lat, lon))
            d, j = tree.query(xyz(tlat[empty // nlon], tlon[empty % nlon]))
            near = d <= 2 * np.sin(np.radians(res) / 2)
            fill = csr_matrix((np.ones(int(near.sum()), np.float32),
                               (np.arange(int(near.sum())), j[near])),
                              shape=(int(near.sum()), n))
            m = vstack([m, fill]).tocsr()
            cells = np.concatenate([cells, empty[near]])
        return m, cells, tlat, tlon

    def _detect_regrid(self, da: xr.DataArray) -> xr.Dataset:
        """Per-cell maps of a CMIP6 daily SST field on its native grid."""
        ty, tm, td, cal = self._time_parts(da)
        fy, fm, fd = self._full_axis(cal)
        pos, ok = self._positions(ty, tm, td, fy, fm, fd)
        nt = len(fy)
        if ok.sum() < 0.9 * nt:
            raise ValueError(f"only {ok.sum()} of {nt} days of "
                             f"{self.period[0]}–{self.period[1]}")
        lat, lon = self._latlon_of(da)
        lat, lon = lat.ravel(), lon.ravel()
        first = int(np.flatnonzero(ok)[0])
        sample = self._values(da.isel(time=first)).ravel()
        good = np.isfinite(sample) & np.isfinite(lat) & np.isfinite(lon) \
            & (np.abs(lat) <= 90)
        off = np.float32(self._offset(sample[good]))
        m, cells, tlat, tlon = self._box_matrix(lat[good], lon[good], _CMIP6_RES)
        logger.info("  %d source points → %d %g° cells, %d days, calendar %s",
                    int(good.sum()), len(cells), _CMIP6_RES, nt, cal)
        arr = np.full((len(cells), nt), np.nan, dtype=np.float32)
        for y in np.unique(ty[ok]):
            ti = np.flatnonzero((ty == y) & ok)
            block = self._values(da.isel(time=ti)).reshape(len(ti), -1)[:, good].T
            fin = np.isfinite(block)
            num = m @ np.where(fin, block, 0.0)
            den = m @ fin.astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                arr[:, pos[ti]] = np.where(den > 0, num / den, np.nan) + off
        part = self._detect_cells(arr, (fy, fm, fd, cal))
        ny, nx = len(tlat), len(tlon)
        maps = {}
        for k, v in part.items():
            if k == "years":
                continue
            full = np.full(v.shape[:-1] + (ny * nx,), np.nan, dtype=np.float32)
            full[..., cells] = v
            maps[k] = full
        return self._as_dataset(maps, np.unique(fy), tlat, tlon)

    # ── Sources ────────────────────────────────────────────────────────

    def _eerie_sources(self) -> dict[str, Callable]:
        out = {}
        for model in self.config.models:
            if model in self._obs_cfg:
                continue

            def get(m=model):
                return self.model_loader.load_var(m, "tos", table="Oday",
                                                  period=self.period)
            out[model] = get
        return out

    def _obs_sources(self) -> dict[str, Callable]:
        out = {}
        for name, spec in self._obs_cfg.items():
            spec = spec or {}
            if not spec.get("data_root"):
                logger.warning("marine_heatwaves: obs %s has no data_root — skipped",
                               name)
                continue

            def get(n=name, sp=spec):
                return self._make_obs_loader(n, sp).load_var(
                    n, "tos", table="Oday", period=self.period)
            out[name] = get
        return out

    def _discover_cmip6(self):
        cfg = self._cmip6_cfg
        if not cfg.get("enabled", False):
            return None, []
        from feather.data.cmip6_nc_loader import CMIP6NCLoader, discover_daily_models

        root = cfg.get("root", "/work/ik1017/CMIP6/data/CMIP6")
        try:
            found = discover_daily_models(
                root, experiment=cfg.get("experiment", "historical"),
                table="Oday", variable="tos",
                member=cfg.get("member", "r1i1p1f1"),
                exclude=tuple(cfg.get("exclude", [])))
        except Exception as exc:  # noqa: BLE001 — discovery must not kill the run
            logger.warning("CMIP6 daily tos discovery failed (%s) — EERIE only", exc)
            return None, []
        wanted = cfg.get("models", "auto")
        if isinstance(wanted, (list, tuple, set)):
            found = [m for m in found if m.name in set(wanted)]
        if cfg.get("max_models"):
            found = found[: int(cfg["max_models"])]
        if not found:
            return None, []
        logger.info("CMIP6 daily tos: %d models — %s", len(found),
                    ", ".join(f"{m.name}({m.variant})" for m in found))
        return CMIP6NCLoader.for_models(self.config, found, root=root), found

    # ── Orchestration ──────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        logger.info("marine_heatwaves: %s–%s, climatology %s–%s, %s",
                    *self.period, *self.clim_period, self.params)
        obs: dict[str, xr.Dataset] = {}
        for name, get in self._obs_sources().items():
            logger.info("  Observations: %s", name)
            ds = self._dataset(self._nc_path(name, "obs"), get)
            if ds is not None:
                obs[name] = ds
        eerie: dict[str, xr.Dataset] = {}
        for model, get in self._eerie_sources().items():
            logger.info("  EERIE member: %s", model)
            ds = self._dataset(self._nc_path(model), get)
            if ds is not None:
                eerie[model] = ds
        cmip6: dict[str, xr.Dataset] = {}
        loader, models = self._discover_cmip6()
        for mc in models:
            logger.info("  CMIP6 model: %s", mc.name)

            def get(m=mc.name):
                return loader.load_var(m, "tos", table="Oday", period=self.period)
            ds = self._dataset(self._nc_path(mc.name, "cmip6"), get, cmip6=True)
            if ds is not None:
                cmip6[mc.name] = ds
        return self._summarise(obs, eerie, cmip6)

    def _ice_mask(self, ref: xr.Dataset | None, lat, lon) -> xr.DataArray | None:
        """True where the reference climatology is warmer than ``ice_mask_sst``."""
        if ref is None or self.ice_sst is None or "seas_mean" not in ref:
            return None
        clim = _to_grid(ref["seas_mean"], lat, lon)
        return clim > float(self.ice_sst)

    def _summarise(self, obs: dict, eerie: dict, cmip6: dict) -> dict[str, Any]:
        grid = _regular_grid(_EERIE_RES)
        cgrid = _regular_grid(_CMIP6_RES)
        ref_name = self.reference if self.reference in obs else next(iter(obs), None)
        ref = obs.get(ref_name)
        mask = self._ice_mask(ref, *grid)
        cmask = self._ice_mask(ref, *cgrid)

        def fields(ds, g, msk):
            out = {}
            for k in list(self.metrics) + ["seas_mean"]:
                if k in ds:
                    f = _to_grid(ds[k], *g)
                    out[k] = f.where(msk) if msk is not None else f
            return out

        def series(ds):
            if "mhw_days_annual" not in ds:
                return None
            band = ds["mhw_days_annual"].sel(lat=slice(*_SERIES_BAND))
            from feather.util.spatial import latlon_global_mean
            ndays = 365.25
            return (100.0 * latlon_global_mean(band) / ndays).load()

        res: dict[str, Any] = {
            "obs": {n: fields(d, grid, mask) for n, d in obs.items()},
            "obs_1deg": {n: fields(d, cgrid, cmask) for n, d in obs.items()},
            "eerie": {m: fields(d, grid, mask) for m, d in eerie.items()},
            "cmip6": {m: fields(d, cgrid, cmask) for m, d in cmip6.items()},
            "series": {
                "obs": {n: series(d) for n, d in obs.items()},
                "eerie": {m: series(d) for m, d in eerie.items()},
                "cmip6": {m: series(d) for m, d in cmip6.items()},
            },
            "reference": ref_name,
            "families": self._families(list(eerie)),
        }
        fam: dict[str, dict] = {k: {} for k in self.metrics}
        for k in self.metrics:
            for f, members in res["families"].items():
                arrs = [res["eerie"][m][k] for m in members if k in res["eerie"][m]]
                if not arrs:
                    continue
                stack = xr.concat(arrs, dim="member")
                with warnings.catch_warnings():     # land: all-NaN columns
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = stack.mean("member")
                    std = stack.std("member") if len(arrs) > 1 else None
                fam[k][f] = {"mean": mean, "std": std,
                             "n": len(arrs),
                             "members": [m for m in members
                                         if k in res["eerie"][m]]}
        res["family"] = fam
        mmm = {}
        for k in self.metrics:
            arrs = [d[k] for d in res["cmip6"].values() if k in d]
            mmm[k] = ({"mean": xr.concat(arrs, "member").mean("member"),
                       "n": len(arrs)} if arrs else None)
        res["mmm"] = mmm
        return res

    def _families(self, models: list[str]) -> dict[str, list[str]]:
        if self._per_family:
            ordered = [m for m in self.config.models if m in models]
            return self.config.get_model_families(ordered)
        return {"EERIE": list(models)} if models else {}

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = self.compute()
        saved = []
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

    def _iter_figures(self, results):
        if not results["eerie"] and not results["obs"] and not results["cmip6"]:
            logger.warning("%s: no data — no figures", self.name)
            return
        for k in self.metrics:
            for fn in (self._plot_maps, self._plot_members, self._plot_bias,
                       self._plot_spread):
                fm = fn(results, k)
                if fm is not None:
                    yield fm
        for fn in (self._plot_zonal, self._plot_series):
            fm = fn(results)
            if fm is not None:
                yield fm
        for region in self.regions:
            fm = self._plot_region(results, region)
            if fm is not None:
                yield fm

    # Labels -----------------------------------------------------------

    def _info(self, k: str) -> tuple[str, str, str]:
        return mhw.METRIC_INFO[k]

    def _family_label(self, f: str, g: dict) -> str:
        return f if g["n"] == 1 else f"{f} mean ({g['n']} members)"

    def _mmm_label(self, g: dict) -> str:
        return f"{self._cmip6_label} MMM ({g['n']} models, {_CMIP6_RES:g}°)"

    def _settings_txt(self) -> str:
        p = self.params
        return (f"Hobday et al. (2016) marine heatwaves in daily SST, "
                f"{self.period[0]}–{self.period[1]}: threshold = "
                f"{p['pctile']:g}th percentile of a ±{p['window_half_width']}-day "
                f"window over {self.clim_period[0]}–{self.clim_period[1]}, "
                f"climatology and threshold smoothed over {p['smooth_width']} days, "
                f"events ≥ {p['min_duration']} days, gaps ≤ {p['max_gap']} days "
                f"joined, not detrended. Intensities are relative to the "
                f"seasonal climatology.")

    def _mask_txt(self) -> str:
        if self.ice_sst is None:
            return ""
        return (f" Cells where the {self.reference or 'reference'} annual-mean "
                f"climatology is below {self.ice_sst:g} °C (sea ice) are masked.")

    def _computation_notes(self) -> str:
        return (self._settings_txt() + " EERIE members and observations are "
                f"detected on their common 0.25° grid; {self._cmip6_label} models "
                f"after box-averaging their native ocean grid onto {_CMIP6_RES:g}°."
                + self._mask_txt())

    # Map helper ---------------------------------------------------------

    def _map_grid(self, panels, *, cmap, vmin, vmax, cbar_label, suptitle,
                  ncols=3, greyed=None, extend="both", ticks=None,
                  ticklabels=None):
        """Robinson maps, one shared horizontal colour bar.

        *panels* is ``[(title, field)]``; a ``None`` field leaves its slot
        blank.  *greyed* maps a title to a boolean field drawn in grey under
        the data (e.g. non-significant cells).
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        n = len(panels)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.6 * ncols, 3.2 * nrows + 0.9),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=180)},
            squeeze=False)
        pc = ccrs.PlateCarree()
        mesh = None
        panels = [(t, None if f is None else self._for_plot(f)) for t, f in panels]
        if greyed:
            greyed = {t: (self._for_plot(g.astype(float)) > 0.5
                          if g is not None else None) for t, g in greyed.items()}
        for ax, (title, field) in zip(axes.flat, panels):
            if field is None:
                ax.set_visible(False)
                continue
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
            ax.coastlines(linewidth=0.3)
            lons, lats = field["lon"].values, field["lat"].values
            if greyed and greyed.get(title) is not None:
                g = greyed[title]
                ax.pcolormesh(lons, lats, g.where(g).astype(float).values,
                              transform=pc, cmap="Greys", vmin=0, vmax=2.5,
                              shading="auto", zorder=1, rasterized=True)
                field = field.where(~g)
            mesh = ax.pcolormesh(lons, lats, field.values, transform=pc, cmap=cmap,
                                 vmin=vmin, vmax=vmax, shading="auto", zorder=2,
                                 rasterized=True)
            ax.set_title(title, fontsize=9)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12,
                            wspace=0.05, hspace=0.18)
        cax = fig.add_axes([0.3, 0.06, 0.4, 0.022])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend)
        cb.set_label(cbar_label)
        if ticks is not None:
            cb.set_ticks(ticks)
            if ticklabels is not None:
                cb.set_ticklabels(ticklabels)
        return fig

    def _for_plot(self, field: xr.DataArray) -> xr.DataArray:
        """Block-average *field* to ``plot_resolution`` when it is finer.

        Only for display (a 0.25° Robinson pcolormesh costs seconds per
        panel); statistics use the full-resolution fields.
        """
        res = abs(float(field["lon"].values[1] - field["lon"].values[0]))
        n = int(round(self.plot_resolution / res))
        if n <= 1:
            return field
        return field.coarsen(lat=n, lon=n, boundary="trim").mean()

    def _style(self, k: str, fields: list) -> dict:
        """Colour map and range of metric *k* over *fields*."""
        if k == "category_max":
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0, 1, 5)))
            return dict(cmap=cmap, vmin=-0.5, vmax=4.5, extend="neither",
                        ticks=range(5), ticklabels=_CATEGORY_LABELS)
        vmin, vmax = self._robust_range(fields, symmetric=False)
        lo = 0.0 if k not in ("rate_onset_mean", "rate_decline_mean") else vmin
        return dict(cmap=_CMAPS.get(k, "viridis"), vmin=min(lo, vmin), vmax=vmax,
                    extend="max")

    def _stats(self, panels) -> dict:
        return {t: {"area_mean": self._latlon_field_mean(f)} for t, f in panels}

    def _meta(self, results, *, title, figure_id, description, units, plot_type,
              stats, models, extra=None, benchmark=True):
        n_cmip6 = len(results["cmip6"])
        return self._build_metadata(
            title=title, figure_id=figure_id, models=models,
            description=description, computation_notes=self._computation_notes(),
            period=self.period, units=units, plot_type=plot_type,
            summary_statistics=stats,
            benchmark_info=({"benchmark": self._cmip6_label, "n_models": n_cmip6,
                             "models": sorted(results["cmip6"])}
                            if benchmark and n_cmip6 else None),
            extra={"group": self.group,
                   "obs_dataset": ", ".join(results["obs"]),
                   "obs_variable": "tos (daily)" if results["obs"] else "",
                   "observations": list(results["obs"]),
                   "reference_obs": results["reference"],
                   "ensemble_mode": "per_family" if self._per_family else "pooled",
                   "model_families": results["families"], **(extra or {})},
        )

    # Per-metric figures -------------------------------------------------

    def _plot_maps(self, results, k):
        label, long_name, units = self._info(k)
        panels = [(o, d[k]) for o, d in results["obs"].items() if k in d]
        panels += [(self._family_label(f, g), g["mean"])
                   for f, g in results["family"][k].items()]
        if results["mmm"].get(k):
            panels.append((self._mmm_label(results["mmm"][k]),
                           results["mmm"][k]["mean"]))
        if not panels:
            return None
        style = self._style(k, [p[1] for p in panels])
        fig = self._map_grid(panels, cbar_label=f"{label} ({units})" if units else label,
                             suptitle=f"Marine heatwaves — {long_name}, "
                                      f"{self.period[0]}–{self.period[1]}", **style)
        meta = self._meta(
            results, title=f"Marine Heatwaves — {label}",
            figure_id=f"mhw_{k}_maps",
            description=(f"{long_name} ({units}) per grid cell: observations, "
                         + ("EERIE family means" if self._per_family
                            else "EERIE ensemble mean")
                         + f" and the {self._cmip6_label} multi-model mean."),
            units=units, plot_type="map", stats=self._stats(panels),
            models=[m for g in results["family"][k].values() for m in g["members"]],
            extra={"metric": k})
        return fig, meta

    def _plot_members(self, results, k):
        label, long_name, units = self._info(k)
        ref = results["reference"]
        panels = ([(ref, results["obs"][ref][k])]
                  if ref in results["obs"] and k in results["obs"][ref] else [])
        members = [m for m in self.config.models if m in results["eerie"]
                   and k in results["eerie"][m]]
        panels += [(m, results["eerie"][m][k]) for m in members]
        if len(members) < 2:
            return None
        style = self._style(k, [p[1] for p in panels])
        fig = self._map_grid(panels, ncols=4,
                             cbar_label=f"{label} ({units})" if units else label,
                             suptitle=f"Marine heatwaves — {long_name}, EERIE "
                                      f"members, {self.period[0]}–{self.period[1]}",
                             **style)
        meta = self._meta(
            results, title=f"Marine Heatwaves — {label} (EERIE members)",
            figure_id=f"mhw_{k}_members",
            description=(f"{long_name} ({units}) per grid cell for every EERIE "
                         f"member with daily SST, next to {ref}."),
            units=units, plot_type="map", stats=self._stats(panels),
            models=members, extra={"metric": k}, benchmark=False)
        return fig, meta

    def _plot_bias(self, results, k):
        ref = results["reference"]
        if ref is None or k not in results["obs"].get(ref, {}):
            return None
        label, long_name, units = self._info(k)
        obs_ref = results["obs"][ref][k]
        panels, greyed, stats = [], {}, {}
        for f, g in results["family"][k].items():
            t = f"{self._family_label(f, g)} − {ref}"
            bias = g["mean"] - obs_ref
            panels.append((t, bias))
            stats[t] = self._latlon_bias_stats(g["mean"], obs_ref)
            if g["std"] is not None:
                insig = (np.abs(bias) <= 2 * g["std"]) & bias.notnull()
                greyed[t] = insig
                stats[t]["frac_area_within_2std"] = self._latlon_field_mean(
                    insig.where(bias.notnull()).astype(float))
        for o, d in results["obs"].items():
            if o != ref and k in d:
                t = f"{o} − {ref}"
                panels.append((t, d[k] - obs_ref))
                stats[t] = self._latlon_bias_stats(d[k], obs_ref)
        mmm = results["mmm"].get(k)
        if mmm and k in results["obs_1deg"].get(ref, {}):
            t = f"{self._mmm_label(mmm)} − {ref}"
            ref1 = results["obs_1deg"][ref][k]
            panels.append((t, mmm["mean"] - ref1))
            stats[t] = self._latlon_bias_stats(mmm["mean"], ref1)
        if not panels:
            return None
        lo, hi = self._robust_range([p[1] for p in panels], symmetric=True)
        fig = self._map_grid(
            panels, cmap="RdBu_r", vmin=lo, vmax=hi,
            cbar_label=f"Δ {label} ({units})" if units else f"Δ {label}",
            suptitle=f"Marine heatwaves — {long_name}: difference to {ref}",
            greyed=greyed)
        meta = self._meta(
            results, title=f"Marine Heatwaves — {label} Bias vs {ref}",
            figure_id=f"mhw_{k}_bias",
            description=(f"{long_name} minus {ref} ({units}). Grey: cells where "
                         f"a multi-member family's bias is within twice its "
                         f"member standard deviation (not distinguishable from "
                         f"internal variability), as in de Toma et al. (2026) "
                         f"Fig. 1. The {self._cmip6_label} MMM is compared with "
                         f"{ref} interpolated to {_CMIP6_RES:g}°."),
            units=units, plot_type="bias_map", stats=stats,
            models=[m for g in results["family"][k].values() for m in g["members"]],
            extra={"metric": k})
        return fig, meta

    def _plot_spread(self, results, k):
        label, long_name, units = self._info(k)
        panels = [(f"{f} member std ({g['n']} members)", g["std"])
                  for f, g in results["family"][k].items() if g["std"] is not None]
        if not panels:
            return None
        _, hi = self._robust_range([p[1] for p in panels], symmetric=False)
        fig = self._map_grid(
            panels, cmap="Purples", vmin=0.0, vmax=hi, extend="max",
            cbar_label=f"σ {label} ({units})" if units else f"σ {label}",
            suptitle=f"Marine heatwaves — {long_name}: spread across members")
        meta = self._meta(
            results, title=f"Marine Heatwaves — {label} Member Spread",
            figure_id=f"mhw_{k}_spread",
            description=(f"Standard deviation of {long_name.lower()} across the "
                         f"members of each EERIE family ({units}): the internal "
                         f"variability against which biases are judged."),
            units=units, plot_type="map", stats=self._stats(panels),
            models=[m for g in results["family"][k].values() for m in g["members"]],
            extra={"metric": k}, benchmark=False)
        return fig, meta

    # Summary figures ----------------------------------------------------

    def _obs_color(self, name: str, i: int) -> str:
        default = ("black", "0.45", "0.7")[min(i, 2)]
        return (self._obs_cfg.get(name) or {}).get("color", default)

    def _plot_zonal(self, results):
        keys = [k for k in _SUMMARY_METRICS if k in self.metrics]
        if not keys:
            return None
        ncols = min(3, len(keys))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                                 squeeze=False)
        stats = {}
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._info(k)
            for i, (o, d) in enumerate(results["obs"].items()):
                if k in d:
                    ax.plot(d[k].lat, d[k].mean("lon"), color=self._obs_color(o, i),
                            lw=2, label=o, zorder=5)
            for f, g in results["family"][k].items():
                col = self.config.get_model_color(g["members"][0])
                for m in g["members"]:
                    if g["n"] > 1:
                        ax.plot(results["eerie"][m][k].lat,
                                results["eerie"][m][k].mean("lon"),
                                color=self.config.get_model_color(m), lw=0.6,
                                alpha=0.6)
                ax.plot(g["mean"].lat, g["mean"].mean("lon"), color=col, lw=1.8,
                        label=self._family_label(f, g))
            if results["mmm"].get(k):
                mm = results["mmm"][k]["mean"]
                ax.plot(mm.lat, mm.mean("lon"), color="0.3", ls="--", lw=1.5,
                        label=self._mmm_label(results["mmm"][k]))
            ax.set_xlim(-80, 80)
            ax.set_xlabel("Latitude")
            ax.set_ylabel(f"{label} ({units})" if units else label)
            ax.grid(alpha=0.3)
            stats[k] = {o: float(d[k].mean()) for o, d in results["obs"].items()
                        if k in d}
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        h, lab = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Marine heatwaves — zonal means", fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        meta = self._meta(
            results, title="Marine Heatwaves — Zonal Means",
            figure_id="mhw_zonal_means",
            description=("Zonal means of the main marine-heatwave metrics: "
                         "observations (thick), EERIE members (thin) and family "
                         "means, and the CMIP6 multi-model mean (dashed)."),
            units="", plot_type="zonal_mean", stats=stats,
            models=list(results["eerie"]))
        return fig, meta

    def _plot_series(self, results):
        s = results["series"]
        if not any(v is not None for grp in s.values() for v in grp.values()):
            return None
        fig, ax = plt.subplots(figsize=(10, 4.5))
        stats = {}
        cm = [v for v in s["cmip6"].values() if v is not None]
        if cm:
            stack = xr.concat(cm, "member")
            ax.fill_between(stack.year, stack.min("member"), stack.max("member"),
                            color="0.85", label=f"{self._cmip6_label} range ({len(cm)})")
            ax.plot(stack.year, stack.mean("member"), color="0.3", ls="--",
                    label=f"{self._cmip6_label} MMM")
            stats[f"{self._cmip6_label} MMM"] = float(stack.mean())
        for f, members in results["families"].items():
            ser = [s["eerie"][m] for m in members if s["eerie"].get(m) is not None]
            if not ser:
                continue
            for m in members:
                if s["eerie"].get(m) is not None and len(ser) > 1:
                    ax.plot(s["eerie"][m].year, s["eerie"][m],
                            color=self.config.get_model_color(m), lw=0.7, alpha=0.6)
            mean = xr.concat(ser, "member").mean("member")
            ax.plot(mean.year, mean, color=self.config.get_model_color(members[0]),
                    lw=2, label=f if len(ser) == 1 else f"{f} mean ({len(ser)})")
            stats[f] = float(mean.mean())
        for i, (o, v) in enumerate(s["obs"].items()):
            if v is not None:
                ax.plot(v.year, v, color=self._obs_color(o, i), lw=2.2, label=o,
                        zorder=5)
                stats[o] = float(v.mean())
        ax.set_ylabel("MHW days (% of days)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"Marine heatwaves — ocean-mean MHW days, "
                     f"{_lat_txt(_SERIES_BAND[0])}–{_lat_txt(_SERIES_BAND[1])}",
                     fontweight="bold")
        fig.tight_layout()
        meta = self._meta(
            results, title="Marine Heatwaves — Yearly MHW Days",
            figure_id="mhw_days_timeseries",
            description=("Yearly percentage of days in a marine heatwave, area "
                         f"mean over the ocean between {_lat_txt(_SERIES_BAND[0])} "
                         f"and {_lat_txt(_SERIES_BAND[1])}. The threshold is fixed over the "
                         f"climatology period and the data are not detrended, so "
                         f"the warming trend raises the count over time."),
            units="%", plot_type="timeseries", stats=stats,
            models=list(results["eerie"]))
        return fig, meta

    def _plot_region(self, results, region):
        name, la0, la1, lo0, lo1 = self.regions[region]
        keys = [k for k in _SUMMARY_METRICS if k in self.metrics]

        def cut(f):
            lon = f["lon"] % 360.0
            sel = f.where((f["lat"] >= la0) & (f["lat"] <= la1)
                          & (lon >= lo0) & (lon <= lo1), drop=True)
            v = sel.values.ravel()
            return v[np.isfinite(v)]

        datasets: list[tuple[str, str, dict]] = []
        for i, (o, d) in enumerate(results["obs"].items()):
            datasets.append((o, self._obs_color(o, i), d))
        for m in self.config.models:
            if m in results["eerie"]:
                datasets.append((m, self.config.get_model_color(m), results["eerie"][m]))
        if results["mmm"] and any(results["mmm"].values()):
            datasets.append((f"{self._cmip6_label} MMM", "0.5",
                             {k: g["mean"] for k, g in results["mmm"].items() if g}))
        if not datasets or not keys:
            return None
        ncols = min(3, len(keys))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(max(5, 0.55 * len(datasets)) * ncols,
                                          3.8 * nrows), squeeze=False)
        stats: dict = {}
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._info(k)
            vals, cols, names = [], [], []
            for n, c, d in datasets:
                if k in d:
                    v = cut(d[k])
                    if len(v) > 1:
                        vals.append(v)
                        cols.append(c)
                        names.append(n)
                        stats.setdefault(n, {})[k] = float(np.median(v))
            if not vals:
                ax.set_visible(False)
                continue
            parts = ax.violinplot(vals, showmedians=True, widths=0.8)
            for body, c in zip(parts["bodies"], cols):
                body.set_facecolor(c)
                body.set_alpha(0.6)
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel(f"{label} ({units})" if units else label)
            ax.grid(axis="y", alpha=0.3)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        fig.suptitle(f"Marine heatwaves — {name} ({_lat_txt(la0)}–{_lat_txt(la1)}, "
                     f"{lo0:g}–{lo1:g}°E)", fontweight="bold")
        fig.tight_layout()
        meta = self._meta(
            results, title=f"Marine Heatwaves — {name}",
            figure_id=f"mhw_regions_{region}",
            description=(f"Distribution over the ocean cells of the {name} area "
                         f"of interest (de Toma et al. 2026, Table 2) of the "
                         f"per-cell marine-heatwave metrics, per dataset (median "
                         f"marked). The {self._cmip6_label} MMM is on "
                         f"{_CMIP6_RES:g}°, so it has fewer cells."),
            units="", plot_type="violin", stats=stats,
            models=list(results["eerie"]), extra={"region": region})
        return fig, meta

    # ── NetCDF export ──────────────────────────────────────────────────

    def _export_summary(self, results) -> None:
        out = Path(self.config.output_dir) / "netcdf" / self.name
        out.mkdir(parents=True, exist_ok=True)
        for k in self.metrics:
            fields = {f"obs_{o}": d[k] for o, d in results["obs"].items() if k in d}
            fields.update({f"family_{f}_mean": g["mean"]
                           for f, g in results["family"][k].items()})
            fields.update({f"member_{m}": d[k] for m, d in results["eerie"].items()
                           if k in d})
            if fields:
                path = out / f"mhw_{k}_{self.period[0]}-{self.period[1]}.nc"
                xr.Dataset({n.replace(" ", "_"): v for n, v in fields.items()}
                           ).to_netcdf(path)
                logger.info("  wrote %s", path)
