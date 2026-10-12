"""Subsurface marine heatwaves: shallow (≤ 50 m) vs deep (> 50 m) events.

Subsurface part of de Toma et al. (2026, EGUsphere-2026-5747, "Comparing
Surface and Subsurface Marine Heatwaves in Ocean Reanalyses"; code
github.com/vincenzodetoma/ROHAN), applied to the EERIE members instead of
the GREP reanalyses.  The surface part is :mod:`marine_heatwaves`.

Method (paper §2.4)
-------------------
Marine heatwaves (Hobday et al. 2016) are detected **independently on every
model level** of daily ``thetao`` down to ``max_depth`` (the paper: 0.5, 10,
30, 50, 100, 150, 200 m), with the same settings as the surface run
(``project.marine_heatwaves``, overridable in
``project.marine_heatwaves_subsurface``).  Levels are then grouped into
**shallow** (depth ≤ ``shallow_max_depth``, 50 m) and **deep** (deeper)
events.  Group maps pool the events of their levels:

* event means (duration, intensities, rates) — mean over all events of the
  group's levels, i.e. the per-level means weighted by event counts;
* maxima (maximum intensity, maximum category) — maximum over the levels;
* frequency, MHW-day percentage, climatology — mean over the levels.

Each model is detected on its own levels; nothing is interpolated in the
vertical.  For the EERIE members these are (≤ 210 m)

==================  ===========================  =====================
member              shallow                      deep
==================  ===========================  =====================
IFS-FESOM2-SR r1    2.5, 12.5 m                  97.5, 195 m
ICON-ESM-ER r1–r3   1 m                          50.55, 201.8 m
==================  ===========================  =====================

so the two families sample the column differently (ICON's 50.55 m level
falls just below the 50 m divide); the per-level figures and depth profiles
show this explicitly.

There is no daily subsurface observation product on Levante for 1982–2014
(the paper validates the reanalyses against ESA CCI at the surface only).
The surface results of :mod:`marine_heatwaves` — observations and each
member's ``tos`` — are read from its checkpoints when present and anchor the
profiles at 0 m.  The mixed-layer depth (``Omon/mlotst``, annual mean and
deepest monthly climatology) is drawn on the profiles, as in the paper's
time–depth sections.  The paper's time–depth sections of named events
(Table 2) are not reproduced: free-running coupled models do not reproduce
the observed events at their dates.

Figures (per metric ``m``)
--------------------------
``mhwz_{m}_layers``              shallow and deep maps, one row per family
``mhwz_{m}_deep_minus_shallow``  deep − shallow per family
``mhwz_{m}_levels``              every level of every member

Plus ``mhwz_profiles_{region}`` (area-mean metrics against depth, global
60°S–60°N and the paper's areas of interest), ``mhwz_zonal_means``,
``mhwz_days_timeseries`` and ``mhwz_regions_{region}`` (shallow/deep cell
distributions per member, paper Fig. 17).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.marine_heatwaves import (
    _EERIE_RES,
    _SERIES_BAND,
    _SUMMARY_METRICS,
    MarineHeatwavesDiag,
    _lat_txt,
)
from feather.diag.precip_extremes import _regular_grid, _to_grid
from feather.diag.registry import register

logger = logging.getLogger(__name__)

#: Names of the vertical dimension in daily ocean temperature.
_ZDIMS: tuple[str, ...] = ("lev", "depth", "olevel", "deptht")

#: Group maps that are the maximum over the group's levels.
_MAX_METRICS = frozenset({"intensity_max_max", "category_max"})

#: Group maps that are the mean over the levels (not event-weighted).
_LEVEL_MEAN = frozenset({"frequency", "mhw_days_pct", "seas_mean",
                         "thresh_minus_seas", "mhw_days_annual",
                         "events_annual"})

_LAYERS: tuple[str, ...] = ("shallow", "deep")

#: Global profile domain (keeps sea ice out, as the MHW-day series).
_GLOBAL = ("Global ocean", _SERIES_BAND[0], _SERIES_BAND[1], 0.0, 360.0)


def pool_levels(levels: list[dict[str, xr.DataArray]]) -> dict[str, xr.DataArray]:
    """Group maps from the per-level maps of one layer.

    Event means are pooled over all events of the levels (weights: events
    per year of each level, the same years everywhere); maxima take the
    maximum over levels; counts and the climatology the mean over levels.
    Cells take the levels where they are ocean.
    """
    if not levels:
        return {}
    keys = set.intersection(*(set(lv) for lv in levels))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN cells
        freq = (xr.concat([lv["frequency"] for lv in levels], "level")
                if "frequency" in keys else None)
        out: dict[str, xr.DataArray] = {}
        for k in sorted(keys):
            stack = xr.concat([lv[k] for lv in levels], "level")
            if k in _MAX_METRICS:
                out[k] = stack.max("level")
            elif k in _LEVEL_MEAN or freq is None:
                out[k] = stack.mean("level")
            else:
                w = freq.where(stack.notnull()).fillna(0.0)
                num = (stack.fillna(0.0) * w).sum("level")
                den = w.sum("level")
                out[k] = (num / den).where(den > 0)
    return out


@register
class MarineHeatwavesSubsurfaceDiag(MarineHeatwavesDiag):
    """Hobday marine heatwaves on the model levels of daily ``thetao``."""

    name = "marine_heatwaves_subsurface"
    title = "Subsurface Marine Heatwaves (shallow vs deep) in Daily Ocean Temperature"
    domain = "o3d"
    variables = ["thetao"]
    group = "marine_heatwaves_subsurface"

    _CONFIG_BLOCKS = ("marine_heatwaves", "marine_heatwaves_subsurface")

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
                         cmip6_loader=cmip6_loader, experiment=experiment,
                         period=period, variables=variables,
                         save_netcdf=save_netcdf)
        cfg = config.project.get("marine_heatwaves_subsurface", {}) or {}
        self.table: str = cfg.get("table", "HROday")
        self.max_depth: float = float(cfg.get("max_depth", 210.0))
        self.shallow_max: float = float(cfg.get("shallow_max_depth", 50.0))
        self.mld_table: str | None = cfg.get("mld_table", "Omon")

    # ── Paths ──────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        return Path(self.config.output_dir) / self.name

    @property
    def surface_dir(self) -> Path:
        return Path(self.config.output_dir) / MarineHeatwavesDiag.name

    def _level_path(self, model: str, depth: float) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_z{depth:g}m_{self.period[0]}_{self.period[1]}.nc"

    def _mld_path(self, model: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_mld_{self.period[0]}_{self.period[1]}.nc"

    def _surface_path(self, name: str, kind: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        base = self.surface_dir / ("obs" if kind == "obs" else "")
        return base / f"{safe}_{self.period[0]}_{self.period[1]}.nc"

    def _layer(self, depth: float) -> str:
        return "shallow" if depth <= self.shallow_max else "deep"

    def _layer_txt(self, layer: str) -> str:
        return (f"shallow (≤ {self.shallow_max:g} m)" if layer == "shallow"
                else f"deep (> {self.shallow_max:g} m)")

    # ── Daily thetao → per-level maps ──────────────────────────────────

    @staticmethod
    def _depths(z: xr.DataArray) -> np.ndarray:
        d = np.asarray(z.values, dtype=np.float64)
        if str(z.attrs.get("units", "")).lower() in ("cm", "centimeters"):
            d = d / 100.0
        return np.round(d, 3)           # CMOR levels carry ~1e-14 noise

    def _member_levels(self, model: str) -> dict[float, xr.Dataset]:
        """Per-level maps of *model* (from checkpoints, else detected)."""
        try:
            da = self.model_loader.load_var(model, "thetao", table=self.table,
                                            period=self.period)
        except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("  no daily thetao (%s) — skipped", exc)
            return {}
        zdim = next((d for d in _ZDIMS if d in da.dims), None)
        if zdim is None:
            logger.warning("  thetao has no depth axis (%s) — skipped", da.dims)
            return {}
        try:
            da = self._std(da, zdim)
        except ValueError as exc:
            logger.warning("  %s — skipped", exc)
            return {}
        depth = self._depths(da[zdim])
        want = [int(i) for i in np.flatnonzero(depth <= self.max_depth)]
        out: dict[float, xr.Dataset] = {}
        todo = []
        for i in want:
            ds = self._load_checkpoint(self._level_path(model, depth[i]))
            if ds is not None:
                logger.info("  loaded %s", self._level_path(model, depth[i]).name)
                out[float(depth[i])] = ds
            else:
                todo.append(i)
        if todo:
            logger.info("  detecting levels %s m",
                        ", ".join(f"{depth[i]:g}" for i in todo))
            try:
                parts = self._detect_levels(da, zdim, todo)
            except ValueError as exc:
                logger.error("  %s — skipped", exc)
                return out
            for i, ds in zip(todo, parts):
                if not ds.data_vars:
                    continue
                ds.attrs.update({k: str(v) for k, v in self._settings().items()})
                ds.attrs["depth"] = f"{depth[i]:g}"
                path = self._level_path(model, depth[i])
                path.parent.mkdir(parents=True, exist_ok=True)
                enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
                ds.to_netcdf(path, encoding=enc)
                logger.info("  saved %s", path)
                out[float(depth[i])] = ds
        return dict(sorted(out.items()))

    def _member_mld(self, model: str) -> xr.Dataset | None:
        """Annual-mean and deepest monthly-climatology mixed-layer depth."""
        if not self.mld_table:
            return None
        path = self._mld_path(model)
        if path.exists():
            return xr.open_dataset(path).load()
        try:
            da = self.model_loader.load_var(model, "mlotst", table=self.mld_table,
                                            period=self.period)
            da = self._std(da)
            clim = da.groupby("time.month").mean("time")
            vals = self._values(clim)
        except Exception as exc:  # noqa: BLE001 — the MLD is only decoration
            logger.warning("  no mixed-layer depth (%s)", exc)
            return None
        clim = clim.copy(data=vals)
        ds = xr.Dataset({"mld_mean": clim.mean("month"),
                         "mld_max": clim.max("month")})
        ds = ds.assign_coords(lat=np.round(ds["lat"].values, 9),
                              lon=np.round(ds["lon"].values % 360.0, 9))
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        return ds

    def _surface(self) -> dict[str, dict[str, xr.Dataset]]:
        """Surface checkpoints of :mod:`marine_heatwaves`, where they exist."""
        out: dict[str, dict[str, xr.Dataset]] = {"obs": {}, "eerie": {}}
        for name in self._obs_cfg:
            ds = self._load_checkpoint(self._surface_path(name, "obs"))
            if ds is not None:
                out["obs"][name] = ds
        for model in self.config.models:
            if model in self._obs_cfg:
                continue
            ds = self._load_checkpoint(self._surface_path(model, "eerie"))
            if ds is not None:
                out["eerie"][model] = ds
        logger.info("  surface results found: obs %s, members %s",
                    sorted(out["obs"]) or "none", sorted(out["eerie"]) or "none")
        return out

    # ── Orchestration ──────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        logger.info("%s: %s–%s, climatology %s–%s, levels ≤ %g m (shallow ≤ "
                    "%g m), %s", self.name, *self.period, *self.clim_period,
                    self.max_depth, self.shallow_max, self.params)
        levels: dict[str, dict[float, xr.Dataset]] = {}
        mld: dict[str, xr.Dataset] = {}
        for model in self.config.models:
            if model in self._obs_cfg:
                continue
            logger.info("  EERIE member: %s", model)
            lv = self._member_levels(model)
            if not lv:
                continue
            levels[model] = lv
            m = self._member_mld(model)
            if m is not None:
                mld[model] = m
        return self._summarise_z(levels, mld, self._surface())

    def _summarise_z(self, levels: dict, mld: dict, surface: dict) -> dict[str, Any]:
        grid = _regular_grid(_EERIE_RES)
        obs = surface["obs"]
        ref_name = self.reference if self.reference in obs else next(iter(obs), None)
        mask = self._ice_mask(obs.get(ref_name), *grid)
        keys = list(self.metrics) + ["seas_mean"]

        def fields(ds):
            out = {}
            for k in keys + ["mhw_days_annual"]:
                if k in ds:
                    f = _to_grid(ds[k], *grid)
                    out[k] = f.where(mask) if mask is not None else f
            return out

        res: dict[str, Any] = {
            "levels": {m: {d: fields(ds) for d, ds in lv.items()}
                       for m, lv in levels.items()},
            "obs": {n: fields(d) for n, d in obs.items()},
            "surface": {m: fields(d) for m, d in surface["eerie"].items()
                        if m in levels},
            "mld": {m: {k: _to_grid(d[k], *grid) for k in ("mld_mean", "mld_max")}
                    for m, d in mld.items()},
            "cmip6": {},
            "reference": ref_name,
            "families": self._families(list(levels)),
        }
        res["layers"] = {
            m: {layer: pool_levels([f for d, f in lv.items()
                                    if self._layer(d) == layer])
                for layer in _LAYERS}
            for m, lv in res["levels"].items()}
        fam: dict[str, dict] = {}
        for f, members in res["families"].items():
            fam[f] = {"members": members, "n": len(members)}
            for layer in _LAYERS:
                fam[f][layer] = {}
                for k in keys + ["mhw_days_annual"]:
                    arrs = [res["layers"][m][layer][k] for m in members
                            if k in res["layers"][m][layer]]
                    if arrs:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            fam[f][layer][k] = xr.concat(arrs, "member").mean("member")
        res["family"] = fam
        res["column_mask"] = self._column_mask(res["levels"], mask)
        return res

    @staticmethod
    def _column_mask(levels: dict, mask) -> xr.DataArray | None:
        """Cells that are ocean on every level of every member (no shelves)."""
        out = None
        for lv in levels.values():
            for f in lv.values():
                if "seas_mean" not in f:
                    continue
                ok = f["seas_mean"].notnull()
                out = ok if out is None else (out & ok)
        if out is not None and mask is not None:
            out = out & mask
        return out

    # ── Labels and metadata ────────────────────────────────────────────

    def _computation_notes(self) -> str:
        return (self._settings_txt().replace("in daily SST", "in daily ocean "
                                             "temperature (thetao)")
                + f" Detection runs independently on every model level down to "
                f"{self.max_depth:g} m (no vertical interpolation); levels at or "
                f"above {self.shallow_max:g} m form the shallow group, deeper "
                f"levels the deep group (de Toma et al. 2026). Group event "
                f"means pool the events of their levels (per-level means "
                f"weighted by event counts), maxima are the maximum over "
                f"levels, frequency and MHW-day share the mean over levels."
                + self._mask_txt())

    def _meta(self, results, *, title, figure_id, description, units, plot_type,
              stats, models, extra=None, benchmark=False):
        depths = {m: sorted(lv) for m, lv in results["levels"].items()}
        return self._build_metadata(
            title=title, figure_id=figure_id, models=models,
            description=description, computation_notes=self._computation_notes(),
            period=self.period, units=units, plot_type=plot_type,
            summary_statistics=stats,
            extra={"group": self.group,
                   "obs_dataset": ", ".join(results["obs"]),
                   "obs_variable": "tos (daily, surface only)" if results["obs"] else "",
                   "observations": list(results["obs"]),
                   "reference_obs": results["reference"],
                   "ensemble_mode": "per_family" if self._per_family else "pooled",
                   "model_families": results["families"],
                   "model_levels_m": depths,
                   "shallow_max_depth_m": self.shallow_max,
                   "max_depth_m": self.max_depth, **(extra or {})},
        )

    def _fam_label(self, f: str, g: dict) -> str:
        return f if g["n"] == 1 else f"{f} mean ({g['n']} members)"

    def _member_depths_txt(self, results, members) -> str:
        return "; ".join(
            f"{m}: " + ", ".join(f"{d:g}" for d in results["levels"][m]) + " m"
            for m in members if m in results["levels"])

    # ── Figures ────────────────────────────────────────────────────────

    def _iter_figures(self, results):
        if not results["levels"]:
            logger.warning("%s: no member with daily thetao — no figures", self.name)
            return
        for k in self.metrics:
            for fn in (self._plot_layers, self._plot_layer_diff, self._plot_levels):
                fm = fn(results, k)
                if fm is not None:
                    yield fm
        for region in ["global", *self.regions]:
            fm = self._plot_profiles(results, region)
            if fm is not None:
                yield fm
        for fn in (self._plot_zonal_z, self._plot_series_z):
            fm = fn(results)
            if fm is not None:
                yield fm
        for region in self.regions:
            fm = self._plot_region_z(results, region)
            if fm is not None:
                yield fm

    def _all_members(self, results):
        return [m for g in results["family"].values() for m in g["members"]]

    def _plot_layers(self, results, k):
        label, long_name, units = self._info(k)
        panels = []
        for f, g in results["family"].items():
            for layer in _LAYERS:
                panels.append((f"{self._fam_label(f, g)} — {self._layer_txt(layer)}",
                               g[layer].get(k)))
        if not any(p[1] is not None for p in panels):
            return None
        style = self._style(k, [p[1] for p in panels if p[1] is not None])
        fig = self._map_grid(panels, ncols=2,
                             cbar_label=f"{label} ({units})" if units else label,
                             suptitle=f"Subsurface marine heatwaves — {long_name}, "
                                      f"{self.period[0]}–{self.period[1]}", **style)
        members = self._all_members(results)
        meta = self._meta(
            results, title=f"Subsurface Marine Heatwaves — {label}, Shallow vs Deep",
            figure_id=f"mhwz_{k}_layers",
            description=(f"{long_name} ({units}) of events shallower than or at "
                         f"{self.shallow_max:g} m (left) and deeper (right), per "
                         f"EERIE family, from daily ocean temperature on the model "
                         f"levels ({self._member_depths_txt(results, members)})."),
            units=units, plot_type="map",
            stats=self._stats([p for p in panels if p[1] is not None]),
            models=members, extra={"metric": k})
        return fig, meta

    def _plot_layer_diff(self, results, k):
        label, long_name, units = self._info(k)
        panels, stats = [], {}
        for f, g in results["family"].items():
            if k in g["shallow"] and k in g["deep"]:
                t = f"{self._fam_label(f, g)}: deep − shallow"
                d = g["deep"][k] - g["shallow"][k]
                panels.append((t, d))
                stats[t] = {"area_mean": self._latlon_field_mean(d),
                            "frac_area_deep_larger": self._latlon_field_mean(
                                (d > 0).where(d.notnull()).astype(float))}
        if not panels:
            return None
        lo, hi = self._robust_range([p[1] for p in panels], symmetric=True)
        fig = self._map_grid(
            panels, ncols=min(2, len(panels)), cmap="RdBu_r", vmin=lo, vmax=hi,
            cbar_label=f"Δ {label} ({units})" if units else f"Δ {label}",
            suptitle=f"Subsurface marine heatwaves — {long_name}: deep minus shallow")
        meta = self._meta(
            results, title=f"Subsurface Marine Heatwaves — {label}, Deep minus Shallow",
            figure_id=f"mhwz_{k}_deep_minus_shallow",
            description=(f"{long_name} of deep (> {self.shallow_max:g} m) minus "
                         f"shallow (≤ {self.shallow_max:g} m) events ({units}), per "
                         f"EERIE family; only where the column reaches the deep "
                         f"levels. Positive: larger at depth."),
            units=units, plot_type="bias_map", stats=stats,
            models=self._all_members(results), extra={"metric": k})
        return fig, meta

    def _plot_levels(self, results, k):
        label, long_name, units = self._info(k)
        members = [m for m in self.config.models if m in results["levels"]]
        ncols = max(len(results["levels"][m]) for m in members)
        panels = []
        for m in members:
            row = [(f"{m} — {d:g} m", f.get(k))
                   for d, f in results["levels"][m].items()]
            panels += row + [(f"{m} — blank {i}", None)
                             for i in range(ncols - len(row))]
        real = [p for p in panels if p[1] is not None]
        if not real:
            return None
        style = self._style(k, [p[1] for p in real])
        fig = self._map_grid(panels, ncols=ncols,
                             cbar_label=f"{label} ({units})" if units else label,
                             suptitle=f"Subsurface marine heatwaves — {long_name} "
                                      f"per model level, {self.period[0]}–"
                                      f"{self.period[1]}", **style)
        meta = self._meta(
            results, title=f"Subsurface Marine Heatwaves — {label} per Level",
            figure_id=f"mhwz_{k}_levels",
            description=(f"{long_name} ({units}) on every model level of every "
                         f"member (one row per member), detected independently "
                         f"per level."),
            units=units, plot_type="map", stats=self._stats(real),
            models=members, extra={"metric": k})
        return fig, meta

    # Profiles ---------------------------------------------------------

    def _box(self, region):
        return _GLOBAL if region == "global" else self.regions[region]

    @staticmethod
    def _in_box(f: xr.DataArray, la0, la1, lo0, lo1) -> xr.DataArray:
        lon = f["lon"] % 360.0
        return (f["lat"] >= la0) & (f["lat"] <= la1) & (lon >= lo0) & (lon <= lo1)

    def _plot_profiles(self, results, region):
        name, la0, la1, lo0, lo1 = self._box(region)
        keys = [k for k in _SUMMARY_METRICS if k in self.metrics]
        cmask = results["column_mask"]
        if cmask is None or not keys:
            return None
        sel = cmask & self._in_box(cmask, la0, la1, lo0, lo1)
        if not bool(sel.any()):
            return None

        def amean(f):
            return self._latlon_field_mean(f.where(sel))

        ncols = min(3, len(keys))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows),
                                 squeeze=False)
        stats: dict = {}
        members = [m for m in self.config.models if m in results["levels"]]
        mld = {m: (amean(results["mld"][m]["mld_mean"]),
                   amean(results["mld"][m]["mld_max"]))
               for m in members if m in results["mld"]}
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._info(k)
            for m in members:
                col = self.config.get_model_color(m)
                lv = results["levels"][m]
                z = [d for d in lv if k in lv[d]]
                x = [amean(lv[d][k]) for d in z]
                surf = results["surface"].get(m, {})
                if k in surf:
                    ax.plot([amean(surf[k])], [0.0], marker="o", mfc="none",
                            color=col, ms=7, ls="none")
                ax.plot(x, z, marker="o", color=col, lw=1.6, label=m)
                stats.setdefault(m, {})[k] = dict(zip([f"{d:g} m" for d in z], x))
            for i, (o, d) in enumerate(results["obs"].items()):
                if k in d:
                    v = amean(d[k])
                    ax.plot([v], [0.0], marker="*", color=self._obs_color(o, i),
                            ms=11, ls="none", label=f"{o} (SST)", zorder=5)
                    stats.setdefault(o, {})[k] = {"0 m": v}
            for m, (mean, deepest) in mld.items():
                col = self.config.get_model_color(m)
                ax.axhline(mean, color=col, ls="--", lw=0.9, alpha=0.8)
                ax.axhline(deepest, color=col, ls=":", lw=0.9, alpha=0.8)
            ax.axhline(self.shallow_max, color="0.4", lw=0.6)
            ax.set_ylim(self.max_depth, -5.0)
            ax.set_xlabel(f"{label} ({units})" if units else label)
            ax.set_ylabel("Depth (m)")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        h, lab = axes.flat[0].get_legend_handles_labels()
        if mld:
            from matplotlib.lines import Line2D
            h += [Line2D([], [], color="0.3", ls="--", lw=0.9),
                  Line2D([], [], color="0.3", ls=":", lw=0.9)]
            lab += ["mixed layer, annual mean", "mixed layer, deepest month"]
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), fontsize=8,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Subsurface marine heatwaves — {name} "
                     f"({_lat_txt(la0)}–{_lat_txt(la1)}, {lo0:g}–{lo1:g}°E)",
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        if mld:
            stats["mixed_layer_depth_m"] = {m: {"annual_mean": a, "deepest_month": b}
                                            for m, (a, b) in mld.items()}
        meta = self._meta(
            results, title=f"Subsurface Marine Heatwaves — Profiles, {name}",
            figure_id=f"mhwz_profiles_{region}",
            description=(f"Area mean of the marine-heatwave metrics on every model "
                         f"level against depth, {name}, over the cells that are "
                         f"ocean on every level of every member down to "
                         f"{self.max_depth:g} m (shelves excluded). Open circles at "
                         f"0 m: the members' surface (tos) results; stars: daily "
                         f"SST observations. Dashed: annual-mean mixed-layer depth "
                         f"(mlotst), dotted: deepest monthly-climatology mixed "
                         f"layer; grey line: the {self.shallow_max:g} m "
                         f"shallow/deep divide."),
            units="", plot_type="profile", stats=stats, models=members,
            extra={"region": region})
        return fig, meta

    # Zonal means and series -------------------------------------------

    def _plot_zonal_z(self, results):
        keys = [k for k in _SUMMARY_METRICS if k in self.metrics]
        if not keys:
            return None
        ncols = min(3, len(keys))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                                 squeeze=False)
        stats: dict = {}
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._info(k)
            for f, g in results["family"].items():
                col = self.config.get_model_color(g["members"][0])
                for layer, ls in zip(_LAYERS, ("-", "--")):
                    if k not in g[layer]:
                        continue
                    z = g[layer][k].mean("lon")
                    ax.plot(z.lat, z, color=col, ls=ls, lw=1.6,
                            label=f"{self._fam_label(f, g)} {layer}")
                    stats.setdefault(f"{f} {layer}", {})[k] = float(g[layer][k].mean())
            ax.set_xlim(-80, 80)
            ax.set_xlabel("Latitude")
            ax.set_ylabel(f"{label} ({units})" if units else label)
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        h, lab = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 4), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Subsurface marine heatwaves — zonal means, shallow (solid) "
                     "vs deep (dashed)", fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        meta = self._meta(
            results, title="Subsurface Marine Heatwaves — Zonal Means",
            figure_id="mhwz_zonal_means",
            description=(f"Zonal means of the main marine-heatwave metrics for "
                         f"shallow (≤ {self.shallow_max:g} m, solid) and deep "
                         f"(> {self.shallow_max:g} m, dashed) events, per EERIE "
                         f"family."),
            units="", plot_type="zonal_mean", stats=stats,
            models=self._all_members(results))
        return fig, meta

    def _plot_series_z(self, results):
        from feather.util.spatial import latlon_global_mean

        def series(f):
            band = f["mhw_days_annual"].sel(lat=slice(*_SERIES_BAND))
            return (100.0 * latlon_global_mean(band) / 365.25).load()

        fig, ax = plt.subplots(figsize=(10, 4.5))
        stats, drawn = {}, False
        for f, g in results["family"].items():
            col = self.config.get_model_color(g["members"][0])
            for layer, ls in zip(_LAYERS, ("-", "--")):
                if "mhw_days_annual" not in g[layer]:
                    continue
                s = series(g[layer])
                ax.plot(s.year, s, color=col, ls=ls, lw=1.8,
                        label=f"{self._fam_label(f, g)} {layer}")
                stats[f"{f} {layer}"] = float(s.mean())
                drawn = True
        for i, (o, d) in enumerate(results["obs"].items()):
            if "mhw_days_annual" in d:
                s = series(d)
                ax.plot(s.year, s, color=self._obs_color(o, i), lw=1.2,
                        label=f"{o} (SST)", zorder=5)
                stats[o] = float(s.mean())
        if not drawn:
            plt.close(fig)
            return None
        ax.set_ylabel("MHW days (% of days)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"Subsurface marine heatwaves — ocean-mean MHW days, "
                     f"{_lat_txt(_SERIES_BAND[0])}–{_lat_txt(_SERIES_BAND[1])}",
                     fontweight="bold")
        fig.tight_layout()
        meta = self._meta(
            results, title="Subsurface Marine Heatwaves — Yearly MHW Days",
            figure_id="mhwz_days_timeseries",
            description=(f"Yearly percentage of days in a marine heatwave, ocean "
                         f"area mean between {_lat_txt(_SERIES_BAND[0])} and "
                         f"{_lat_txt(_SERIES_BAND[1])}, averaged over the shallow "
                         f"(solid) and deep (dashed) levels of each EERIE family; "
                         f"observed surface values for reference. Not detrended, "
                         f"so warming raises the count over time."),
            units="%", plot_type="timeseries", stats=stats,
            models=self._all_members(results))
        return fig, meta

    # Region distributions ---------------------------------------------

    def _plot_region_z(self, results, region):
        name, la0, la1, lo0, lo1 = self.regions[region]
        keys = [k for k in _SUMMARY_METRICS if k in self.metrics]
        members = [m for m in self.config.models if m in results["layers"]]
        if not keys or not members:
            return None

        def cut(f):
            v = f.where(self._in_box(f, la0, la1, lo0, lo1)).values.ravel()
            return v[np.isfinite(v)]

        ncols = min(3, len(keys))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(max(5, 1.1 * len(members)) * ncols,
                                          3.8 * nrows), squeeze=False)
        stats: dict = {}
        drawn = False
        for ax, k in zip(axes.flat, keys):
            label, _, units = self._info(k)
            for j, m in enumerate(members):
                col = self.config.get_model_color(m)
                for layer, dx in zip(_LAYERS, (-0.2, 0.2)):
                    f = results["layers"][m][layer].get(k)
                    if f is None:
                        continue
                    v = cut(f)
                    if len(v) < 2:
                        continue
                    parts = ax.violinplot([v], positions=[j + 1 + dx], widths=0.38,
                                          showmedians=True)
                    for line in ("cbars", "cmins", "cmaxes", "cmedians"):
                        if line in parts:
                            parts[line].set_color(col)
                            parts[line].set_linewidth(1.0)
                    for body in parts["bodies"]:
                        body.set_facecolor(col)
                        body.set_alpha(0.75 if layer == "shallow" else 0.35)
                        if layer == "deep":
                            body.set_hatch("//")
                    stats.setdefault(f"{m} {layer}", {})[k] = float(np.median(v))
                    drawn = True
            ax.set_xticks(range(1, len(members) + 1))
            ax.set_xticklabels(members, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(f"{label} ({units})" if units else label)
            ax.grid(axis="y", alpha=0.3)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        if not drawn:
            plt.close(fig)
            return None
        fig.suptitle(f"Subsurface marine heatwaves — {name}: shallow (left, solid) "
                     f"vs deep (right, hatched)", fontweight="bold")
        fig.tight_layout()
        meta = self._meta(
            results, title=f"Subsurface Marine Heatwaves — {name}",
            figure_id=f"mhwz_regions_{region}",
            description=(f"Distribution over the ocean cells of the {name} area of "
                         f"interest (de Toma et al. 2026, Table 2) of the per-cell "
                         f"metrics of shallow (≤ {self.shallow_max:g} m, left) and "
                         f"deep (> {self.shallow_max:g} m, right) events, per "
                         f"member (median marked), as in the paper's Fig. 17."),
            units="", plot_type="violin", stats=stats, models=members,
            extra={"region": region})
        return fig, meta

    # ── NetCDF export ──────────────────────────────────────────────────

    def _export_summary(self, results) -> None:
        out = Path(self.config.output_dir) / "netcdf" / self.name
        out.mkdir(parents=True, exist_ok=True)
        for k in self.metrics:
            fields = {}
            for f, g in results["family"].items():
                for layer in _LAYERS:
                    if k in g[layer]:
                        fields[f"family_{f}_{layer}_mean"] = g[layer][k]
            for m, lay in results["layers"].items():
                for layer in _LAYERS:
                    if k in lay[layer]:
                        fields[f"member_{m}_{layer}"] = lay[layer][k]
            if fields:
                path = out / f"mhwz_{k}_{self.period[0]}-{self.period[1]}.nc"
                xr.Dataset({n.replace(" ", "_"): v for n, v in fields.items()}
                           ).to_netcdf(path)
                logger.info("  wrote %s", path)
