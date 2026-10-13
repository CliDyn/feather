"""Sea-surface-height variability: EERIE members vs AVISO (and CMIP6).

Generalises the notebook ``/work/bm1344/AWI/EERIE/science/
zos_multimodel_std.ipynb`` (maps of the temporal standard deviation and of
the mean of daily SSH, models next to AVISO) to the EERIE ensemble.

Statistics (:mod:`feather.util.ssh_stats`), per grid cell over ``period``:

``mean``      time-mean SSH minus its area mean over the cells every dataset
              covers.  Models and AVISO use different reference levels
              (geoid, global-mean constraint), so only the pattern compares.
``std_raw``   standard deviation of SSH as it is (the notebook's quantity).
``std_anom``  after removing the seasonal climatology and the linear trend.
``std_hp``    (daily only) of the anomaly minus its ``hp_window_days`` running
              mean: periods shorter than ~3 months, i.e. the mesoscale.

Two comparisons:

* **daily** — the members with daily ``zos`` (``daily_tables`` are tried in
  order, so CMOR ``HROday`` and the kerchunk / HadGEM3 ``Oday`` both work)
  against daily AVISO on the 0.25° grid.  CMIP6 publishes no daily ``zos``.
* **monthly** — every member's ``Omon/zos`` against monthly AVISO, plus the
  CMIP6 historical ``Omon/zos`` ensemble (``cmip6_daily`` block: DRS root,
  member, exclusions; one member per model), box-averaged onto 1° first.
  AVISO is also box-averaged onto 1° *before* its statistics so the CMIP6
  MMM is compared with observations at its own resolution (averaging
  removes eddy variance, so a std of the 1° field is lower than the 1°
  mean of the 0.25° std).

Observations (``ssh_variability.obs``) are CMOR trees made by
``scripts/aviso_derive_zos_daily.sh`` (AVISO DT2024 allsat L4 ``adt``,
0.125° → 0.25° conservative, daily ``Oday`` + monthly ``Omon``).  Cells
where the reference covers fewer than ``min_valid_fraction`` of the steps
(seasonal sea ice) are masked in every figure.

Checkpoints: ``{output}/ssh_variability/[obs/|cmip6/]{name}_{freq}_{y0}_{y1}.nc``.

Figures (``f`` = ``daily`` | ``monthly``; ``s`` = statistic)
------------------------------------------------------------
``ssh_{s}_{f}_maps``     AVISO, family means (+ AVISO 1° and CMIP6 MMM)
``ssh_{s}_{f}_members``  AVISO and every member
``ssh_{s}_{f}_bias``     family means (and CMIP6 MMM) minus AVISO; grey where
                         a multi-member family's difference is within twice
                         its member spread
``ssh_{f}_zonal_means``  zonal means of the statistics
``ssh_{f}_regions``      area means over boundary-current and other regions
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

from feather.diag.marine_heatwaves import MarineHeatwavesDiag
from feather.diag.precip_extremes import _regular_grid, _to_grid
from feather.diag.registry import register
from feather.util import mhw, ssh_stats

logger = logging.getLogger(__name__)

_EERIE_RES: float = 0.25
_CMIP6_RES: float = 1.0

#: Plausible |area mean| (m) of one SSH field; larger means wrong units.
_MAX_ABS_SSH: float = 5.0

FREQS: tuple[str, ...] = ("daily", "monthly")

#: Statistics shown per comparison.
FREQ_STATS: dict[str, tuple[str, ...]] = {
    "daily": ("std_raw", "std_anom", "std_hp"),
    "monthly": ("mean", "std_raw", "std_anom"),
}

#: Display factor (from m) and units.
_DISPLAY: dict[str, tuple[float, str]] = {
    "mean": (1.0, "m"), "std_raw": (100.0, "cm"), "std_anom": (100.0, "cm"),
    "std_hp": (100.0, "cm"),
}

#: Regions of strong SSH variability: name → (label, lat0, lat1, lon0, lon1),
#: longitudes 0–360.
REGIONS: dict[str, tuple[str, float, float, float, float]] = {
    "gulf_stream": ("Gulf Stream", 33.0, 45.0, 285.0, 310.0),
    "kuroshio": ("Kuroshio Extension", 30.0, 42.0, 140.0, 175.0),
    "agulhas": ("Agulhas Retroflection", -45.0, -35.0, 10.0, 35.0),
    "brazil_malvinas": ("Brazil–Malvinas Confluence", -48.0, -33.0, 300.0, 315.0),
    "east_australian": ("East Australian Current", -42.0, -28.0, 148.0, 162.0),
    "drake_passage": ("Drake Passage", -62.0, -54.0, 290.0, 305.0),
    "equatorial_pacific": ("Equatorial Pacific", -10.0, 10.0, 160.0, 270.0),
    "ocean_60": ("Ocean 60°S–60°N", -60.0, 60.0, 0.0, 360.0),
}


@register
class SSHVariabilityDiag(MarineHeatwavesDiag):
    """Mean and temporal variability of SSH vs AVISO, daily and monthly."""

    name = "ssh_variability"
    title = "Sea Surface Height Variability vs AVISO"
    domain = "o2d"
    variables = ["zos"]
    group = "sea_level"

    _CONFIG_BLOCKS: tuple[str, ...] = ("ssh_variability",)

    def __init__(
        self,
        model_loader,
        obs_loader,
        config,
        *,
        cmip6_loader=None,
        experiment: str = "hist-1950",
        period: tuple[str, str] = ("1993", "2014"),
        variables=None,
        save_netcdf: bool = False,
    ):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, experiment=experiment,
                         period=period, variables=variables,
                         save_netcdf=save_netcdf)
        cfg = self._config_block(config)
        self.period = tuple(str(x) for x in cfg.get("period", ["1993", "2014"]))
        self.hp_window: int = int(cfg.get("hp_window_days", 91))
        self.smooth_width: int = int(cfg.get("smooth_width", 31))
        self.min_valid: float = float(cfg.get("min_valid_fraction", 0.9))
        self.daily_tables: list[str] = list(cfg.get("daily_tables",
                                                    ["Oday", "HROday"]))
        self.monthly_table: str = str(cfg.get("monthly_table", "Omon"))
        self.freqs: list[str] = [f for f in FREQS if cfg.get(f, True)]
        self.regions = {**REGIONS, **(cfg.get("regions") or {})}
        self.ice_sst = None

    # ── Paths and checkpoints ──────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / self.name

    def _nc_path(self, name: str, kind: str = "eerie", freq: str = "daily") -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        base = {"eerie": self.nc_dir, "obs": self.nc_dir / "obs",
                "cmip6": self.nc_dir / "cmip6"}[kind]
        return base / f"{safe}_{freq}_{self.period[0]}_{self.period[1]}.nc"

    def _settings(self) -> dict:
        return {"period": "-".join(self.period), "hp_window_days": self.hp_window,
                "smooth_width": self.smooth_width}

    def _ssh_dataset(self, path: Path, getter: Callable, freq: str,
                     regrid: bool = False) -> xr.Dataset | None:
        ds = self._load_checkpoint(path)
        if ds is not None:
            logger.info("  loaded %s", path.name)
            return ds
        try:
            da = getter()
        except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("  no %s zos (%s) — skipped", freq, exc)
            return None
        try:
            ds = (self._stats_regrid(da, freq) if regrid
                  else self._stats_regular(da, freq))
        except ValueError as exc:
            logger.error("  %s — skipped", exc)
            return None
        ds.attrs.update({k: str(v) for k, v in self._settings().items()})
        ds.attrs["freq"] = freq
        path.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
        ds.to_netcdf(path, encoding=enc)
        logger.info("  saved %s", path)
        return ds

    # ── Time axis ──────────────────────────────────────────────────────

    def _axis(self, freq: str, cal: str):
        """Full axis of the period: (years, months, days, slots, n_slots, feb29)."""
        if freq == "daily":
            fy, fm, fd = self._full_axis(cal)
            slots, n_slots, feb29 = mhw.calendar_slots(fy, fm, fd, cal)
            return fy, fm, fd, slots, n_slots, feb29
        years = np.arange(int(self.period[0]), int(self.period[1]) + 1)
        fy = np.repeat(years, 12)
        fm = np.tile(np.arange(1, 13), len(years))
        return fy, fm, np.ones_like(fy), fm - 1, 12, None

    def _place(self, da: xr.DataArray, freq: str):
        """Axis, data positions on it and which data steps are used."""
        ty, tm, td, cal = self._time_parts(da)
        fy, fm, fd, slots, n_slots, feb29 = self._axis(freq, cal)
        if freq == "monthly":
            td = np.ones_like(ty)
        elif len(ty) > 1 and np.median(np.diff(ty * 372 + tm * 31 + td)) > 2:
            raise ValueError("time step is not daily")
        pos, ok = self._positions(ty, tm, td, fy, fm, fd)
        nt = len(fy)
        if ok.sum() < 0.8 * nt:
            raise ValueError(f"only {ok.sum()} of {nt} {freq} steps of "
                             f"{self.period[0]}–{self.period[1]}")
        return (fy, slots, n_slots, feb29, cal), pos, ok

    @staticmethod
    def _check_units(sample: np.ndarray) -> None:
        mean = float(np.nanmean(sample)) if np.isfinite(sample).any() else np.nan
        if not abs(mean) <= _MAX_ABS_SSH:
            raise ValueError(f"mean SSH of the first step is {mean:.4g} — not m")

    def _cell_stats(self, arr: np.ndarray, axis: tuple, freq: str
                    ) -> dict[str, np.ndarray]:
        """:func:`ssh_stats.cell_stats` on chunks of *arr* (cells, steps)."""
        _, slots, n_slots, feb29, _ = axis
        kw = dict(feb29=feb29, smooth_width=self.smooth_width if freq == "daily" else 0,
                  hp_window=self.hp_window if freq == "daily" else 0)
        n = arr.shape[0]
        bounds = list(range(0, n, self.cell_chunk)) + [n]

        def run(i):
            return ssh_stats.cell_stats(arr[bounds[i]:bounds[i + 1]], slots,
                                        n_slots, **kw)

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            parts = list(ex.map(run, range(len(bounds) - 1)))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # ── Data → per-cell statistics ─────────────────────────────────────

    def _stats_regular(self, da: xr.DataArray, freq: str) -> xr.Dataset:
        """Statistics of a (time, lat, lon) field on its own grid."""
        da = self._std(da)
        axis, pos, ok = self._place(da, freq)
        fy = axis[0]
        nt = len(fy)
        first = int(np.flatnonzero(ok)[0])
        sample = self._values(da.isel(time=first))
        self._check_units(sample)
        ocean = np.isfinite(sample)
        ny, nx = ocean.shape
        logger.info("  %d ocean cells, %d %s steps (%d missing), calendar %s",
                    int(ocean.sum()), nt, freq, nt - int(ok.sum()), axis[4])
        flat = np.flatnonzero(ocean.ravel())
        maps: dict[str, np.ndarray] = {}
        ty = da["time"].dt.year.values
        for r0, r1 in self._row_blocks(ocean, nt):
            sea = ocean[r0:r1].ravel()
            if not sea.any():
                continue
            arr = np.full((int(sea.sum()), nt), np.nan, dtype=np.float32)
            for y in np.unique(ty[ok]):
                ti = np.flatnonzero((ty == y) & ok)
                block = self._values(da.isel(time=ti, lat=slice(r0, r1)))
                arr[:, pos[ti]] = block.reshape(len(ti), -1)[:, sea].T
                del block
            logger.info("    rows %d–%d: %d cells", r0, r1, arr.shape[0])
            part = self._cell_stats(arr, axis, freq)
            del arr
            sel = (flat >= r0 * nx) & (flat < r1 * nx)
            for k, v in part.items():
                if k not in maps:
                    maps[k] = np.full(ny * nx, np.nan, dtype=np.float32)
                maps[k][flat[sel]] = v
        return self._stats_dataset(maps, da["lat"].values, da["lon"].values)

    def _stats_regrid(self, da: xr.DataArray, freq: str) -> xr.Dataset:
        """Statistics after box-averaging a (native-grid) field onto 1°."""
        axis, pos, ok = self._place(da, freq)
        nt = len(axis[0])
        lat, lon = self._latlon_of(da)
        lat, lon = lat.ravel(), lon.ravel()
        first = int(np.flatnonzero(ok)[0])
        sample = self._values(da.isel(time=first)).ravel()
        good = np.isfinite(sample) & np.isfinite(lat) & np.isfinite(lon) \
            & (np.abs(lat) <= 90)
        self._check_units(sample[good])
        m, cells, tlat, tlon = self._box_matrix(lat[good], lon[good], _CMIP6_RES)
        logger.info("  %d source points → %d %g° cells, %d %s steps",
                    int(good.sum()), len(cells), _CMIP6_RES, nt, freq)
        arr = np.full((len(cells), nt), np.nan, dtype=np.float32)
        ty = da["time"].dt.year.values
        for y in np.unique(ty[ok]):
            ti = np.flatnonzero((ty == y) & ok)
            block = self._values(da.isel(time=ti)).reshape(len(ti), -1)[:, good].T
            fin = np.isfinite(block)
            num = m @ np.where(fin, block, 0.0)
            den = m @ fin.astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                arr[:, pos[ti]] = np.where(den > 0, num / den, np.nan)
        part = self._cell_stats(arr, axis, freq)
        maps = {}
        for k, v in part.items():
            full = np.full(len(tlat) * len(tlon), np.nan, dtype=np.float32)
            full[cells] = v
            maps[k] = full
        return self._stats_dataset(maps, tlat, tlon)

    @staticmethod
    def _stats_dataset(maps: dict, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
        ny, nx = len(lat), len(lon)
        return xr.Dataset(
            {k: (("lat", "lon"), v.reshape(ny, nx)) for k, v in maps.items()},
            coords={"lat": np.round(lat, 9), "lon": np.round(lon % 360.0, 9)})

    # ── Sources ────────────────────────────────────────────────────────

    def _load_daily(self, load: Callable) -> xr.DataArray:
        """First of ``daily_tables`` that gives daily data."""
        errors = []
        for table in self.daily_tables:
            try:
                da = load(table)
            except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
                errors.append(f"{table}: {exc}")
                continue
            t = da["time"].values
            if len(t) > 1 and np.median(np.diff(t[: min(len(t), 400)])
                                        .astype("timedelta64[h]").astype(float)) <= 36:
                return da
            errors.append(f"{table}: not daily")
        raise FileNotFoundError("; ".join(errors) or "no daily table configured")

    def _member_sources(self, freq: str) -> dict[str, Callable]:
        out = {}
        for model in self.config.models:
            if model in self._obs_cfg:
                continue

            def load(table, m=model):
                return self.model_loader.load_var(m, "zos", table=table,
                                                  period=self.period)

            if freq == "daily":
                out[model] = lambda ld=load: self._load_daily(ld)
            else:
                out[model] = lambda ld=load: ld(self.monthly_table)
        return out

    def _ref_sources(self, freq: str) -> dict[str, Callable]:
        out = {}
        for name, spec in self._obs_cfg.items():
            spec = spec or {}
            if not spec.get("data_root"):
                logger.warning("%s: obs %s has no data_root — skipped",
                               self.name, name)
                continue

            def load(table, n=name, sp=spec):
                return self._make_obs_loader(n, sp).load_var(
                    n, "zos", table=table, period=self.period)

            if freq == "daily":
                out[name] = lambda ld=load: self._load_daily(ld)
            else:
                out[name] = lambda ld=load: ld(self.monthly_table)
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
                table=self.monthly_table, variable="zos",
                member=cfg.get("member", "r1i1p1f1"),
                exclude=tuple(cfg.get("exclude", [])))
        except Exception as exc:  # noqa: BLE001 — discovery must not kill the run
            logger.warning("CMIP6 zos discovery failed (%s) — no CMIP6", exc)
            return None, []
        # A model published under two institutions comes back twice (e.g.
        # UKESM1-0-LL r1 and r13); names key the checkpoints, so keep the first.
        seen: set[str] = set()
        found = [m for m in found if not (m.name in seen or seen.add(m.name))]
        wanted = cfg.get("models", "auto")
        if isinstance(wanted, (list, tuple, set)):
            found = [m for m in found if m.name in set(wanted)]
        if cfg.get("max_models"):
            found = found[: int(cfg["max_models"])]
        if not found:
            return None, []
        logger.info("CMIP6 %s zos: %d models — %s", self.monthly_table, len(found),
                    ", ".join(f"{m.name}({m.variant})" for m in found))
        return CMIP6NCLoader.for_models(self.config, found, root=root), found

    # ── Orchestration ──────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        logger.info("%s: %s–%s, %s, high-pass %d days", self.name, *self.period,
                    "+".join(self.freqs), self.hp_window)
        out: dict[str, Any] = {}
        for freq in self.freqs:
            obs, obs1, eerie, cmip6 = {}, {}, {}, {}
            for name, get in self._ref_sources(freq).items():
                logger.info("  Observations (%s): %s", freq, name)
                ds = self._ssh_dataset(self._nc_path(name, "obs", freq), get, freq)
                if ds is not None:
                    obs[name] = ds
            for model, get in self._member_sources(freq).items():
                logger.info("  EERIE member (%s): %s", freq, model)
                ds = self._ssh_dataset(self._nc_path(model, "eerie", freq), get, freq)
                if ds is not None:
                    eerie[model] = ds
            if freq == "monthly":
                loader, models = self._discover_cmip6()
                for mc in models:
                    logger.info("  CMIP6 model: %s", mc.name)

                    def get(m=mc.name):
                        return loader.load_var(m, "zos", table=self.monthly_table,
                                               period=self.period)
                    ds = self._ssh_dataset(self._nc_path(mc.name, "cmip6", freq),
                                           get, freq, regrid=True)
                    if ds is not None:
                        cmip6[mc.name] = ds
                if cmip6:
                    for name, get in self._ref_sources(freq).items():
                        logger.info("  Observations (%s, %g°): %s", freq,
                                    _CMIP6_RES, name)
                        ds = self._ssh_dataset(
                            self._nc_path(f"{name}_{_CMIP6_RES:g}deg", "obs", freq),
                            get, freq, regrid=True)
                        if ds is not None:
                            obs1[name] = ds
            out[freq] = self._summarise_freq(freq, obs, obs1, eerie, cmip6)
        return out

    def _display(self, ds: xr.Dataset, stats, grid, mask, common) -> dict:
        """Masked display fields; ``mean`` relative to its area mean on *common*."""
        out = {}
        for k in stats:
            if k not in ds:
                continue
            f = _to_grid(ds[k], *grid)
            f = f.where(mask) if mask is not None else f
            if k == "mean":
                f = f - self._latlon_field_mean(f.where(common))
            fac, units = _DISPLAY[k]
            out[k] = (f * fac).assign_attrs(units=units)
        return out

    def _summarise_freq(self, freq, obs, obs1, eerie, cmip6) -> dict[str, Any]:
        stats = FREQ_STATS[freq]
        ref_name = self.reference if self.reference in obs else next(iter(obs), None)
        grid, cgrid = _regular_grid(_EERIE_RES), _regular_grid(_CMIP6_RES)

        def ref_mask(ds, g):
            if ds is None or "valid_frac" not in ds:
                return None
            return _to_grid(ds["valid_frac"], *g) >= self.min_valid

        mask = ref_mask(obs.get(ref_name), grid)
        cmask = ref_mask(obs1.get(ref_name), cgrid)

        def common(group, g, msk):
            c = msk if msk is not None else True
            for ds in group:
                if "mean" in ds:
                    c = c & _to_grid(ds["mean"], *g).notnull()
            return c

        com = common(list(obs.values()) + list(eerie.values()), grid, mask)
        ccom = common(list(obs1.values()) + list(cmip6.values()), cgrid, cmask)
        res: dict[str, Any] = {
            "freq": freq, "stats": stats, "reference": ref_name,
            "obs": {n: self._display(d, stats, grid, mask, com) for n, d in obs.items()},
            "obs_1deg": {n: self._display(d, stats, cgrid, cmask, ccom)
                         for n, d in obs1.items()},
            "eerie": {m: self._display(d, stats, grid, mask, com)
                      for m, d in eerie.items()},
            "cmip6": {m: self._display(d, stats, cgrid, cmask, ccom)
                      for m, d in cmip6.items()},
            "families": self._families(list(eerie)),
        }
        fam: dict[str, dict] = {k: {} for k in stats}
        for k in stats:
            for f, members in res["families"].items():
                arrs = [res["eerie"][m][k] for m in members if k in res["eerie"][m]]
                if not arrs:
                    continue
                stack = xr.concat(arrs, dim="member")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = stack.mean("member")
                    std = stack.std("member") if len(arrs) > 1 else None
                fam[k][f] = {"mean": mean, "std": std, "n": len(arrs),
                             "members": [m for m in members if k in res["eerie"][m]]}
        res["family"] = fam
        res["mmm"] = {}
        for k in stats:
            arrs = [d[k] for d in res["cmip6"].values() if k in d]
            res["mmm"][k] = ({"mean": xr.concat(arrs, "member").mean("member"),
                              "n": len(arrs)} if arrs else None)
        return res

    # ── Plotting ───────────────────────────────────────────────────────

    def _iter_figures(self, results):
        for freq, res in results.items():
            if not res["eerie"] and not res["obs"]:
                logger.warning("%s: no %s data — no figures", self.name, freq)
                continue
            for k in res["stats"]:
                for fn in (self._plot_ssh_maps, self._plot_ssh_members,
                           self._plot_ssh_bias):
                    fm = fn(res, k)
                    if fm is not None:
                        yield fm
            for fn in (self._plot_ssh_zonal, self._plot_ssh_regions):
                fm = fn(res)
                if fm is not None:
                    yield fm

    def _stat_info(self, k: str, freq: str) -> tuple[str, str, str]:
        """(short label, long description, display units) of statistic *k*."""
        step = "daily" if freq == "daily" else "monthly"
        info = {
            "mean": ("Mean SSH", "Time-mean sea-surface height minus its area mean"),
            "std_raw": ("SSH std", f"Standard deviation of {step} sea-surface height"),
            "std_anom": ("SSH anomaly std",
                         f"Standard deviation of {step} sea-surface-height anomalies "
                         f"(seasonal cycle and linear trend removed)"),
            "std_hp": (f"SSH high-pass std (<{self.hp_window} d)",
                       f"Standard deviation of high-pass sea-surface-height anomalies "
                       f"(periods shorter than ~{self.hp_window} days: mesoscale)"),
        }[k]
        return info[0], info[1], _DISPLAY[k][1]

    def _computation_notes(self) -> str:
        return (f"Per-cell statistics of sea-surface height (zos; AVISO absolute "
                f"dynamic topography) over {self.period[0]}–{self.period[1]}. "
                f"Anomalies: seasonal climatology (calendar-day mean smoothed over "
                f"{self.smooth_width} days for daily data, monthly means for monthly "
                f"data) and a linear trend removed. High-pass: anomaly minus its "
                f"centred {self.hp_window}-day running mean. Mean SSH: time mean "
                f"minus its area mean over the cells all datasets cover (models and "
                f"AVISO use different reference levels). Cells where the reference "
                f"covers fewer than {self.min_valid:.0%} of the steps (sea ice) are "
                f"masked. EERIE members and AVISO on the common 0.25° grid; "
                f"{self._cmip6_label} models and AVISO box-averaged onto "
                f"{_CMIP6_RES:g}° before the statistics.")

    def _ssh_meta(self, res, *, title, figure_id, description, units, plot_type,
                  stats, models, extra=None, colormap=None):
        n_cmip6 = len(res["cmip6"])
        return self._build_metadata(
            title=title, figure_id=figure_id, models=models,
            description=description, computation_notes=self._computation_notes(),
            period=self.period, units=units, plot_type=plot_type,
            summary_statistics=stats, variables=["zos"],
            benchmark_info=({"benchmark": self._cmip6_label, "n_models": n_cmip6,
                             "models": sorted(res["cmip6"])} if n_cmip6 else None),
            extra={"group": self.group, "frequency": res["freq"],
                   "obs_dataset": ", ".join(res["obs"]),
                   "obs_variable": f"zos ({res['freq']})" if res["obs"] else "",
                   "observations": list(res["obs"]),
                   "reference_obs": res["reference"],
                   "ensemble_mode": "per_family" if self._per_family else "pooled",
                   "model_families": res["families"],
                   **({"colormap": colormap} if colormap else {}), **(extra or {})},
        )

    def _ssh_style(self, k: str, fields: list) -> dict:
        if k == "mean":
            lo, hi = self._robust_range(fields, symmetric=True)
            return dict(cmap="RdBu_r", vmin=lo, vmax=hi, extend="both")
        _, hi = self._robust_range(fields, symmetric=False)
        return dict(cmap="Reds", vmin=0.0, vmax=hi, extend="max")

    def _models_of(self, res, k) -> list[str]:
        return [m for g in res["family"][k].values() for m in g["members"]]

    def _plot_ssh_maps(self, res, k):
        freq = res["freq"]
        label, long_name, units = self._stat_info(k, freq)
        panels = [(o, d[k]) for o, d in res["obs"].items() if k in d]
        panels += [(self._family_label(f, g), g["mean"])
                   for f, g in res["family"][k].items()]
        if res["mmm"].get(k):
            for o, d in res["obs_1deg"].items():
                if k in d:
                    panels.append((f"{o} ({_CMIP6_RES:g}° box mean)", d[k]))
            panels.append((self._mmm_label(res["mmm"][k]), res["mmm"][k]["mean"]))
        if not panels:
            return None
        style = self._ssh_style(k, [p[1] for p in panels])
        fig = self._map_grid(panels, cbar_label=f"{label} ({units})",
                             suptitle=f"{long_name}, {self.period[0]}–{self.period[1]}",
                             **style)
        meta = self._ssh_meta(
            res, title=f"{label} ({freq})", figure_id=f"ssh_{k}_{freq}_maps",
            description=(f"{long_name} ({units}): AVISO, "
                         + ("EERIE family means" if self._per_family
                            else "EERIE ensemble mean")
                         + (f", and the {self._cmip6_label} multi-model mean next to "
                            f"AVISO box-averaged onto {_CMIP6_RES:g}°"
                            if res["mmm"].get(k) else "") + "."),
            units=units, plot_type="map", stats=self._stats(panels),
            colormap=style["cmap"],
            models=self._models_of(res, k), extra={"statistic": k})
        return fig, meta

    def _plot_ssh_members(self, res, k):
        freq = res["freq"]
        label, long_name, units = self._stat_info(k, freq)
        ref = res["reference"]
        panels = ([(ref, res["obs"][ref][k])]
                  if ref in res["obs"] and k in res["obs"][ref] else [])
        members = [m for m in self.config.models if m in res["eerie"]
                   and k in res["eerie"][m]]
        if len(members) < 2:
            return None
        panels += [(m, res["eerie"][m][k]) for m in members]
        style = self._ssh_style(k, [p[1] for p in panels])
        fig = self._map_grid(panels, ncols=4, cbar_label=f"{label} ({units})",
                             suptitle=f"{long_name}, EERIE members, "
                                      f"{self.period[0]}–{self.period[1]}", **style)
        meta = self._ssh_meta(
            res, title=f"{label} ({freq}, EERIE members)",
            figure_id=f"ssh_{k}_{freq}_members",
            description=f"{long_name} ({units}) for every EERIE member, next to {ref}.",
            units=units, plot_type="map", stats=self._stats(panels), models=members,
            colormap=style["cmap"],
            extra={"statistic": k})
        return fig, meta

    def _plot_ssh_bias(self, res, k):
        freq = res["freq"]
        ref = res["reference"]
        if ref is None or k not in res["obs"].get(ref, {}):
            return None
        label, long_name, units = self._stat_info(k, freq)
        obs_ref = res["obs"][ref][k]
        panels, greyed, stats = [], {}, {}

        def add(t, field, reference):
            panels.append((t, field - reference))
            stats[t] = self._latlon_bias_stats(field, reference)
            if k != "mean":
                r = self._latlon_field_mean(reference.where(field.notnull()))
                stats[t]["area_mean_ratio"] = (
                    self._latlon_field_mean(field.where(reference.notnull())) / r
                    if r else float("nan"))

        for f, g in res["family"][k].items():
            t = f"{self._family_label(f, g)} − {ref}"
            add(t, g["mean"], obs_ref)
            if g["std"] is not None:
                bias = g["mean"] - obs_ref
                insig = (np.abs(bias) <= 2 * g["std"]) & bias.notnull()
                greyed[t] = insig
                stats[t]["frac_area_within_2std"] = self._latlon_field_mean(
                    insig.where(bias.notnull()).astype(float))
        mmm = res["mmm"].get(k)
        if mmm and k in res["obs_1deg"].get(ref, {}):
            add(f"{self._mmm_label(mmm)} − {ref} ({_CMIP6_RES:g}°)", mmm["mean"],
                res["obs_1deg"][ref][k])
        if not panels:
            return None
        lo, hi = self._robust_range([p[1] for p in panels], symmetric=True)
        fig = self._map_grid(panels, cmap="RdBu_r", vmin=lo, vmax=hi,
                             cbar_label=f"Δ {label} ({units})",
                             suptitle=f"{long_name}: difference to {ref}",
                             greyed=greyed)
        meta = self._ssh_meta(
            res, title=f"{label} ({freq}) minus {ref}",
            figure_id=f"ssh_{k}_{freq}_bias",
            description=(f"{long_name} minus {ref} ({units}). Grey: cells where a "
                         f"multi-member family's difference is within twice its "
                         f"member standard deviation."
                         + (f" The {self._cmip6_label} MMM is compared with {ref} "
                            f"box-averaged onto {_CMIP6_RES:g}° before its "
                            f"statistics." if mmm else "")),
            units=units, plot_type="bias_map", stats=stats, colormap="RdBu_r",
            models=self._models_of(res, k), extra={"statistic": k})
        return fig, meta

    def _plot_ssh_zonal(self, res):
        freq, keys = res["freq"], res["stats"]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4),
                                 squeeze=False)
        stats: dict = {}
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._stat_info(k, freq)
            for i, (o, d) in enumerate(res["obs"].items()):
                if k in d:
                    ax.plot(d[k].lat, d[k].mean("lon"), color=self._obs_color(o, i),
                            lw=2, label=o, zorder=5)
                    stats.setdefault(k, {})[o] = self._latlon_field_mean(d[k])
            for o, d in res["obs_1deg"].items():
                if k in d:
                    ax.plot(d[k].lat, d[k].mean("lon"), color="black", lw=1.2,
                            ls=":", label=f"{o} ({_CMIP6_RES:g}°)", zorder=5)
            for f, g in res["family"][k].items():
                if g["n"] > 1:
                    for m in g["members"]:
                        ax.plot(res["eerie"][m][k].lat, res["eerie"][m][k].mean("lon"),
                                color=self.config.get_model_color(m), lw=0.6, alpha=0.6)
                ax.plot(g["mean"].lat, g["mean"].mean("lon"),
                        color=self.config.get_model_color(g["members"][0]), lw=1.8,
                        label=self._family_label(f, g))
                stats.setdefault(k, {})[f] = self._latlon_field_mean(g["mean"])
            if res["mmm"].get(k):
                mm = res["mmm"][k]["mean"]
                ax.plot(mm.lat, mm.mean("lon"), color="0.3", ls="--", lw=1.5,
                        label=self._mmm_label(res["mmm"][k]))
                stats.setdefault(k, {})[f"{self._cmip6_label} MMM"] = \
                    self._latlon_field_mean(mm)
            ax.set_xlim(-80, 80)
            ax.set_xlabel("Latitude")
            ax.set_ylabel(f"{label} ({units})")
            ax.grid(alpha=0.3)
        h, lab = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Sea-surface height ({freq}) — zonal means", fontweight="bold")
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        meta = self._ssh_meta(
            res, title=f"SSH Statistics ({freq}) — Zonal Means",
            figure_id=f"ssh_{freq}_zonal_means",
            description=("Zonal means of the SSH statistics: AVISO (thick), EERIE "
                         "members (thin) and family means"
                         + (f", {self._cmip6_label} MMM (dashed) and AVISO at "
                            f"{_CMIP6_RES:g}° (dotted)" if res["cmip6"] else "")
                         + ". Summary statistics are area means."),
            units="", plot_type="zonal_mean", stats=stats, models=list(res["eerie"]))
        return fig, meta

    def _region_mean(self, field: xr.DataArray, region: str) -> float:
        _, la0, la1, lo0, lo1 = self.regions[region]
        lon = field["lon"] % 360.0
        box = field.where((field["lat"] >= la0) & (field["lat"] <= la1)
                          & (lon >= lo0) & (lon <= lo1))
        return self._latlon_field_mean(box)

    def _plot_ssh_regions(self, res):
        freq = res["freq"]
        keys = [k for k in res["stats"] if k != "mean"]
        regions = list(self.regions)
        if not keys or not regions:
            return None
        fig, axes = plt.subplots(len(keys), 1, figsize=(max(8, 1.3 * len(regions)),
                                                        3.2 * len(keys)),
                                 squeeze=False, sharex=True)
        x = np.arange(len(regions))
        stats: dict = {}
        for ax, k in zip(axes[:, 0], keys):
            label, _, units = self._stat_info(k, freq)
            fams = list(res["family"][k].items())
            ncol = len(fams) + (1 if res["mmm"].get(k) else 0)
            width = 0.8 / max(ncol, 1)
            for i, (o, d) in enumerate(res["obs"].items()):
                if k not in d:
                    continue
                v = [self._region_mean(d[k], r) for r in regions]
                ax.scatter(x, v, marker="*", s=140, color=self._obs_color(o, i),
                           zorder=6, label=o)
                for r, val in zip(regions, v):
                    stats.setdefault(r, {}).setdefault(k, {})[o] = val
            for j, (f, g) in enumerate(fams):
                xo = x - 0.4 + width * (j + 0.5)
                col = self.config.get_model_color(g["members"][0])
                for m in g["members"]:
                    v = [self._region_mean(res["eerie"][m][k], r) for r in regions]
                    ax.scatter(xo, v, s=14, color=self.config.get_model_color(m),
                               zorder=4)
                v = [self._region_mean(g["mean"], r) for r in regions]
                ax.bar(xo, v, width=width * 0.9, color=col, alpha=0.45,
                       label=self._family_label(f, g))
                for r, val in zip(regions, v):
                    stats.setdefault(r, {}).setdefault(k, {})[f] = val
            if res["mmm"].get(k):
                xo = x - 0.4 + width * (ncol - 0.5)
                per = np.array([[self._region_mean(d[k], r) for r in regions]
                                for d in res["cmip6"].values() if k in d])
                v = [self._region_mean(res["mmm"][k]["mean"], r) for r in regions]
                ax.bar(xo, v, width=width * 0.9, color="0.6", alpha=0.6,
                       label=self._mmm_label(res["mmm"][k]))
                with warnings.catch_warnings():     # region without CMIP6 cells
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ax.vlines(xo, np.nanmin(per, 0), np.nanmax(per, 0),
                              color="0.3", lw=1.2)
                for o, d in res["obs_1deg"].items():
                    if k in d:
                        ax.scatter(xo, [self._region_mean(d[k], r) for r in regions],
                                   marker="*", s=70, color="black", zorder=6,
                                   label=f"{o} ({_CMIP6_RES:g}°)")
                for r, val in zip(regions, v):
                    stats.setdefault(r, {}).setdefault(k, {})[
                        f"{self._cmip6_label} MMM"] = val
            ax.set_ylabel(f"{label} ({units})")
            ax.grid(axis="y", alpha=0.3)
        axes[-1, 0].set_xticks(x)
        axes[-1, 0].set_xticklabels([self.regions[r][0] for r in regions],
                                    rotation=30, ha="right", fontsize=8)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=min(len(lab), 5), fontsize=8,
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Sea-surface height ({freq}) — regional means",
                     fontweight="bold", y=1.04)
        fig.tight_layout()
        meta = self._ssh_meta(
            res, title=f"SSH Variability ({freq}) — Regions",
            figure_id=f"ssh_{freq}_regions",
            description=("Area means of the SSH standard deviations over regions of "
                         "strong variability (boundary currents, Drake Passage, "
                         "equatorial Pacific) and the ocean between 60°S and 60°N: "
                         "AVISO (stars), EERIE members (dots) and family means "
                         "(bars)"
                         + (f", {self._cmip6_label} MMM (grey bar, line = model "
                            f"range) next to AVISO at {_CMIP6_RES:g}°"
                            if res["cmip6"] else "") + "."),
            units="cm", plot_type="bar", stats=stats, models=list(res["eerie"]),
            extra={"regions": {r: list(v) for r, v in self.regions.items()}})
        return fig, meta

    # ── NetCDF export ──────────────────────────────────────────────────

    def _export_summary(self, results) -> None:
        out = Path(self.config.output_dir) / "netcdf" / self.name
        out.mkdir(parents=True, exist_ok=True)
        for freq, res in results.items():
            for k in res["stats"]:
                fields = {f"obs_{o}": d[k] for o, d in res["obs"].items() if k in d}
                fields.update({f"family_{f}_mean": g["mean"]
                               for f, g in res["family"][k].items()})
                fields.update({f"member_{m}": d[k] for m, d in res["eerie"].items()
                               if k in d})
                if not fields:
                    continue
                path = out / f"ssh_{k}_{freq}_{self.period[0]}-{self.period[1]}.nc"
                xr.Dataset({n.replace(" ", "_"): v for n, v in fields.items()}
                           ).to_netcdf(path)
                logger.info("  wrote %s", path)
                if res["mmm"].get(k):
                    flds = {f"cmip6_{m}": d[k] for m, d in res["cmip6"].items()
                            if k in d}
                    flds["cmip6_mmm"] = res["mmm"][k]["mean"]
                    flds.update({f"obs_{o}": d[k] for o, d in res["obs_1deg"].items()
                                 if k in d})
                    p1 = out / (f"ssh_{k}_{freq}_{_CMIP6_RES:g}deg_"
                                f"{self.period[0]}-{self.period[1]}.nc")
                    xr.Dataset({n.replace(" ", "_"): v for n, v in flds.items()}
                               ).to_netcdf(p1)
                    logger.info("  wrote %s", p1)

