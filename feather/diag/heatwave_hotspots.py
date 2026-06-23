"""Heatwave Hotspots — extreme-heat tail-widening trends (PNAS Fig 2–4).

Reproduces Figures 2–4 of Sambartusek, Kornhuber et al. (2024, PNAS,
doi:10.1073/pnas.2411258121), "A global emergence of regional heatwave
hotspots", within the Feather framework.

Core metric ("tail-widening").  At each grid point and each year, the 99th and
87.5th percentiles of *daily* maximum 2 m temperature (CMOR ``tasmax``, ``day``
table) are computed over all days of that year:

- **P99**   — the median day of the hottest 2 % of days (the extreme tail).
- **P87.5** — the median of the hottest quarter of days (an "average summer
  day").  Deliberately calendar-season agnostic, so it generalises globally.

The yearly **tail width** ``D = P99 − P87.5`` measures how far the extreme tail
sits above moderate-hot days.  Its **linear trend over the analysis period**
(°C/decade) is the heatwave-hotspot metric: a positive trend means the hottest
extremes are warming *faster* than typical summer heat — the distribution tail
is widening.

All per-year percentile fields are regridded to a common 0.25° grid, land-masked
(land fraction > 25 %), and masked grey where the P87.5 trend is negative
(following the paper, regions where moderate summer heat is *not* warming are
excluded).

Reference = ERA5 (the configured ``extremes_obs_reference``).  The "model
ensemble" is the set of evaluated models (EERIE high-resolution members).  A
future CMIP6 / HighResMIP benchmark slots into the same ensemble once a daily
percentile cache is available.

Figures (group ``extremes``):
  - ``heatwave_hotspots_trend_map``     (Fig 2A) global ERA5 D-trend map with
        non-significance stippling, grey P87.5-negative mask, region boxes.
  - ``heatwave_hotspots_regions``       (Fig 2B–K) per-region yearly-D time
        series with fitted trends (obs + models).
  - ``heatwave_hotspots_discrepancy``   (Fig 3A) map where the observed trend
        exceeds the model-ensemble range (model underestimation, dark red).
  - ``heatwave_hotspots_boxwhisker``    (Fig 3B–K) per-region distribution of
        modelled D-trends vs the observed trend.
  - ``heatwave_hotspots_pdf``           (Fig 4) area-weighted PDF of land
        D-trends (ERA5 vs models) + cumulative difference.

Per-model yearly percentile fields are cached to NetCDF (outside the figures
tree) so re-runs skip the heavy daily-data recompute:
``{output_dir}/heatwave_hotspots/{model}_percs_{start}_{end}.nc``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.diag._extremes_obs import (
    obs_ref_label,
    obs_ref_model_name,
    _open_era5_sftlf,
)
from feather.util.spatial import compute_latlon_areas
from feather.util.temporal import linear_trend

logger = logging.getLogger(__name__)

#: Upper / lower percentiles defining the extreme-heat tail width (paper values).
_UPPER_Q = 99.0
_LOWER_Q = 87.5

_BE_LANDMASK_PATH = Path(
    "/work/bm1344/AWI/OBS/berkeleyearth/Land_TMAX_Gridded_0p25deg.nc"
)

# Paper region set (8 objective hotspots + 2 cold-spots).  Boxes are
# (lat_min, lat_max, lon_min, lon_max) in -180..180 longitude.
_REGIONS: list[dict] = [
    {"key": "europe",    "title": "Northwest Europe",   "lat": (45, 58),  "lon": (-5, 12)},
    {"key": "china",     "title": "Central China",      "lat": (27, 37),  "lon": (95, 110)},
    {"key": "argentina", "title": "Southern S. America","lat": (-53, -35),"lon": (-77, -66)},
    {"key": "oman",      "title": "Arabian Peninsula",  "lat": (15, 25),  "lon": (45, 60)},
    {"key": "australia", "title": "Eastern Australia",  "lat": (-40, -30),"lon": (135, 150)},
    {"key": "japan",     "title": "Japan/Korea",        "lat": (30, 39),  "lon": (125, 142)},
    {"key": "arctic",    "title": "High Arctic",        "lat": (78, 85),  "lon": (-95, -10)},
    {"key": "nwcanada",  "title": "Northwest Canada",   "lat": (62, 72),  "lon": (-135, -115)},
    {"key": "nafrica",   "title": "North Africa",       "lat": (20, 35),  "lon": (10, 35)},
    {"key": "siberia",   "title": "Siberia",            "lat": (55, 75),  "lon": (75, 140)},
]


@register
class HeatwaveHotspotsDiag(DiagnosticBase):
    """Extreme-heat tail-widening trends (Kornhuber et al. 2024 Fig 2–4).

    Uses daily maximum 2 m temperature (CMOR ``tasmax``, ``day`` table).
    Models without daily tasmax are skipped gracefully.  ERA5 (the configured
    extremes obs reference) is the reanalysis benchmark; the remaining models
    form the evaluated ensemble.
    """

    name = "heatwave_hotspots"
    title = "Heatwave Hotspots (Tail-Widening)"
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
        period: tuple[str, str] = ("1980", "2014"),
        n_bootstrap: int = 10000,
    ):
        super().__init__(model_loader, obs_loader, config, cmip6_loader=cmip6_loader)
        self.experiment = experiment
        self.period = period
        self.n_bootstrap = n_bootstrap
        self._land_thresh = 25.0  # land fraction % (paper: 0.25)
        self._resolution = self.config.nereus.get("resolution", 0.25)
        self._influence_radius = self.config.nereus.get("influence_radius", 80_000.0)

    # ── Paths ─────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-model percentile NetCDF files (outside figures)."""
        return Path(self.config.output_dir) / "heatwave_hotspots"

    def _nc_path(self, model: str) -> Path:
        start, end = self.period
        safe = model.replace("/", "_").replace(" ", "_")
        return self.nc_dir / f"{safe}_percs_{start}_{end}.nc"

    @staticmethod
    def _grid_signature(lon, lat) -> tuple:
        """Cache key capturing a source grid's size AND axis orientation."""
        lon_r = np.asarray(lon).ravel()
        lat_r = np.asarray(lat).ravel()
        return (
            lon_r.shape[0],
            round(float(lat_r[0]), 4), round(float(lat_r[-1]), 4),
            round(float(lon_r[0]), 4), round(float(lon_r[-1]), 4),
        )

    # ── Orchestration ─────────────────────────────────────────────────

    _FIG_IDS = [
        "heatwave_hotspots_trend_map",
        "heatwave_hotspots_regions",
        "heatwave_hotspots_discrepancy",
        "heatwave_hotspots_boxwhisker",
        "heatwave_hotspots_pdf",
    ]

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute per model → regrid → trends → plot → save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing and all(self._figure_exists(f) for f in self._FIG_IDS):
            logger.info("  All heatwave-hotspots figures exist — skipping")
            return saved

        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Load daily tasmax per model, compute yearly P99/P87.5, regrid, trends.

        Returns a dict with per-model common-grid tail-width fields, trends,
        the shared common grid, masks, and per-region series/trends.
        """
        target_lats: np.ndarray | None = None
        target_lons: np.ndarray | None = None
        interp_cache: dict[tuple, Any] = {}

        d_common: dict[str, xr.DataArray] = {}        # model → D(year, lat, lon)
        trend_d: dict[str, xr.DataArray] = {}         # model → D-trend (lat, lon)
        trend_d_pval: dict[str, xr.DataArray] = {}    # model → D-trend p-value
        trend_lower: dict[str, xr.DataArray] = {}     # model → P87.5-trend (lat, lon)

        for model in self.config.models:
            try:
                percs = self._load_or_compute_percs(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue

            lon = np.asarray(percs["lon"])
            lat = np.asarray(percs["lat"])
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon2d, lat2d = np.meshgrid(lon, lat)
            else:
                lon2d, lat2d = lon, lat

            grid_key = self._grid_signature(lon2d, lat2d)
            if grid_key not in interp_cache:
                logger.info("  %s: building nereus interpolator (%d pts)",
                            model, np.asarray(lon2d).size)
                _, interp = nr_regrid_probe(
                    percs["p99"].isel(year=0).values.ravel(),
                    np.asarray(lon2d).ravel(), np.asarray(lat2d).ravel(),
                    self._resolution, self._influence_radius,
                )
                interp_cache[grid_key] = interp
                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]
            interp = interp_cache[grid_key]

            d_native = (percs["p99"] - percs["p875"])  # (year, lat, lon)
            d_reg = self._regrid_stack(d_native, interp, target_lats, target_lons)
            lower_reg = self._regrid_stack(
                percs["p875"], interp, target_lats, target_lons,
            )

            d_common[model] = d_reg
            trend, pval = self._ols_trend_stats(d_reg)
            trend_d[model] = trend
            trend_d_pval[model] = pval
            trend_lower[model] = linear_trend(lower_reg, dim="year") * 10.0

        if not d_common:
            logger.warning("HeatwaveHotspotsDiag: no model data loaded")
            return {"models": [], "ref_model": None}

        # Shared land mask on the common grid (ERA5 sftlf, else Berkeley mask).
        land_mask = self._common_land_mask(target_lats, target_lons)
        areas = compute_latlon_areas(target_lats, target_lons)

        ref_model = obs_ref_model_name(self.config)
        if ref_model not in d_common:  # ERA5 not loaded — fall back to first
            ref_model = next(iter(d_common))
        models = [m for m in d_common if m != ref_model]

        # Per-region observed/model trends and bootstrap CIs.
        regions = self._compute_regions(
            d_common, trend_d, ref_model, models, target_lats, target_lons, areas,
        )

        return {
            "models": models,
            "ref_model": ref_model,
            "lat": target_lats,
            "lon": target_lons,
            "areas": areas,
            "land_mask": land_mask,
            "trend_d": trend_d,
            "trend_d_pval": trend_d_pval,
            "trend_lower": trend_lower,
            "d_common": d_common,
            "regions": regions,
        }

    # ── Per-model percentiles (cached) ─────────────────────────────────

    def _load_or_compute_percs(self, model: str) -> xr.Dataset:
        """Return Dataset(p99, p875) with dims (year, lat, lon), from NC or fresh."""
        nc_path = self._nc_path(model)
        if nc_path.exists():
            logger.info("  %s: loading percentiles from %s", model, nc_path.name)
            ds = xr.open_dataset(nc_path)
            if {"p99", "p875"}.issubset(ds.data_vars):
                return ds
            logger.info("  %s: NC incomplete — recomputing", model)

        logger.info("  %s: computing yearly P99/P87.5 from daily tasmax", model)
        da = self.model_loader.load_var(
            model, "tasmax", table="day", period=self.period,
        )
        ds = self._compute_percentiles(da)
        self._save_nc(model, ds)
        return ds

    def _compute_percentiles(self, da: xr.DataArray) -> xr.Dataset:
        """Year-by-year P99/P87.5 of daily tasmax (bounds memory like heatwave)."""
        start_year, end_year = int(self.period[0]), int(self.period[1])
        if "time" not in da.dims or da.sizes["time"] == 0:
            raise ValueError("no daily tasmax timesteps in period")
        all_years = da.time.dt.year.values

        # Spatial coordinate metadata
        spatial_dims = [d for d in da.dims if d != "time"]
        coords = {d: np.asarray(da[d]) for d in spatial_dims if d in da.coords}

        years: list[int] = []
        p99_list: list[np.ndarray] = []
        p875_list: list[np.ndarray] = []

        for year in range(start_year, end_year + 1):
            tidx = np.where(all_years == year)[0]
            if len(tidx) == 0:
                continue
            block = da.isel(time=tidx).values.astype(np.float32)
            q = np.nanpercentile(block, [_LOWER_Q, _UPPER_Q], axis=0)
            p875_list.append(q[0].astype(np.float32))
            p99_list.append(q[1].astype(np.float32))
            years.append(year)

        if not years:
            raise ValueError("no complete years of daily tasmax in period")

        def _stack(arrays, name):
            data = np.stack(arrays, axis=0)
            return xr.DataArray(
                data, dims=["year"] + spatial_dims,
                coords={"year": years, **coords}, name=name,
            )

        return xr.Dataset({
            "p99": _stack(p99_list, "p99"),
            "p875": _stack(p875_list, "p875"),
        })

    def _save_nc(self, model: str, ds: xr.Dataset) -> None:
        nc_path = self._nc_path(model)
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        start, end = self.period
        ds = ds.assign_attrs(
            Conventions="CF-1.8",
            title=f"Yearly tasmax percentiles (P99, P87.5) — {model} ({start}–{end})",
            institution="Feather climate evaluation framework",
            source="feather/diag/heatwave_hotspots.py",
            model=model, period_start=start, period_end=end,
            method="yearly P99 and P87.5 of daily tasmax (Kornhuber et al. 2024)",
        )
        ds.to_netcdf(nc_path)
        logger.info("  Saved percentile NetCDF: %s", nc_path)

    # ── Regridding / trends ────────────────────────────────────────────

    @staticmethod
    def _regrid_stack(
        stack: xr.DataArray, interp, target_lats, target_lons,
    ) -> xr.DataArray:
        """Apply a nereus interpolator to each year of a (year, …) stack."""
        years = np.asarray(stack["year"])
        out = np.empty((len(years), len(target_lats), len(target_lons)),
                       dtype=np.float32)
        for i in range(len(years)):
            out[i] = interp(np.asarray(stack.isel(year=i).values).ravel())
        return xr.DataArray(
            out, dims=("year", "lat", "lon"),
            coords={"year": years, "lat": target_lats, "lon": target_lons},
        )

    @staticmethod
    def _ols_trend_stats(
        d_stack: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Per-grid-point OLS trend (°C/decade) and two-sided slope p-value.

        Parametric Wald test on the slope (t-distribution, n−2 df) — equivalent
        to the paper's SI significance test, far cheaper than a full per-grid
        bootstrap for the global map.
        """
        from scipy import stats as _st

        years = np.asarray(d_stack["year"], dtype=np.float64)
        y = np.asarray(d_stack.values, dtype=np.float64)  # (n, lat, lon)
        n = len(years)
        x = years - years.mean()
        sxx = float((x * x).sum())

        sxy = np.einsum("i,ijk->jk", x, y)
        slope = sxy / sxx                                   # units/year
        intercept = y.mean(axis=0) - slope * x.mean()       # x.mean()==0
        pred = intercept[None] + slope[None] * x[:, None, None]
        ss_res = ((y - pred) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(ss_res / (n - 2) / sxx)
            tval = np.where(se > 0, slope / se, 0.0)
        pval = 2.0 * _st.t.sf(np.abs(tval), df=n - 2)

        lat = d_stack["lat"]; lon = d_stack["lon"]
        slope_da = xr.DataArray(slope * 10.0, dims=("lat", "lon"),
                                coords={"lat": lat, "lon": lon})
        pval_da = xr.DataArray(pval, dims=("lat", "lon"),
                               coords={"lat": lat, "lon": lon})
        return slope_da, pval_da

    # ── Masks ──────────────────────────────────────────────────────────

    def _common_land_mask(self, lats, lons) -> xr.DataArray:
        """Boolean land mask (True=land) on the common grid, > land threshold."""
        target = xr.DataArray(
            np.zeros((len(lats), len(lons))), dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        sftlf = _open_era5_sftlf(self.config)
        if sftlf is not None:
            frac = sftlf.interp(lat=target.lat, lon=target.lon, method="nearest",
                                kwargs={"fill_value": 0.0})
            return frac > self._land_thresh
        # Fall back to Berkeley land mask (0..1 fraction)
        if _BE_LANDMASK_PATH.exists():
            try:
                ds = xr.open_dataset(_BE_LANDMASK_PATH)
                mask = ds["land_mask"].rename({"latitude": "lat", "longitude": "lon"})
                if float(mask.lon.min()) < 0:
                    mask = mask.assign_coords(lon=((mask.lon + 360) % 360)).sortby("lon")
                frac = mask.interp(lat=target.lat, lon=target.lon, method="nearest",
                                   kwargs={"fill_value": 0.0})
                return frac > (self._land_thresh / 100.0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("  Could not load Berkeley land mask: %s", exc)
        logger.warning("  No land mask available — using all grid points")
        return target.astype(bool) | True

    # ── Regional aggregation ───────────────────────────────────────────

    def _region_box_mask(self, lats, lons, region: dict) -> xr.DataArray:
        """Boolean mask for a region box; longitudes handled in 0..360."""
        lat_min, lat_max = region["lat"]
        lon_min, lon_max = region["lon"]
        lon_min360 = lon_min % 360
        lon_max360 = lon_max % 360
        lat2d = xr.DataArray(lats, dims="lat", coords={"lat": lats})
        lon2d = xr.DataArray(lons, dims="lon", coords={"lon": lons})
        lat_ok = (lat2d >= lat_min) & (lat2d <= lat_max)
        if lon_min360 <= lon_max360:
            lon_ok = (lon2d >= lon_min360) & (lon2d <= lon_max360)
        else:  # box crosses the prime meridian
            lon_ok = (lon2d >= lon_min360) | (lon2d <= lon_max360)
        return lat_ok & lon_ok

    def _regional_series(self, d_field, region_mask, land_mask, areas) -> np.ndarray:
        """Area-weighted yearly D over a region (land only) → (year,) array."""
        weights = xr.DataArray(areas, dims=("lat", "lon"),
                               coords={"lat": d_field.lat, "lon": d_field.lon})
        w = weights.where(region_mask & land_mask)
        series = d_field.weighted(w.fillna(0.0)).mean(("lat", "lon"))
        return np.asarray(series.values, dtype=np.float64)

    def _compute_regions(
        self, d_common, trend_d, ref_model, models, lats, lons, areas,
    ) -> list[dict]:
        """Per-region observed trend (+ bootstrap CI) and model-ensemble trends."""
        land_mask = self._common_land_mask(lats, lons)
        years = np.asarray(d_common[ref_model]["year"], dtype=np.float64)
        rng = np.random.default_rng(0)
        out: list[dict] = []

        for region in _REGIONS:
            rmask = self._region_box_mask(lats, lons, region)
            obs_series = self._regional_series(
                d_common[ref_model], rmask, land_mask, areas,
            )
            obs_trend = self._slope(years, obs_series) * 10.0
            ci_lo, ci_hi = self._bootstrap_ci(years, obs_series, rng)

            model_series: dict[str, np.ndarray] = {}
            model_trends: dict[str, float] = {}
            for m in models:
                s = self._regional_series(d_common[m], rmask, land_mask, areas)
                model_series[m] = s
                model_trends[m] = self._slope(years, s) * 10.0

            out.append({
                "key": region["key"],
                "title": region["title"],
                "lat": region["lat"],
                "lon": region["lon"],
                "years": years,
                "obs_series": obs_series,
                "obs_trend": obs_trend,
                "obs_ci": (ci_lo * 10.0, ci_hi * 10.0),
                "model_series": model_series,
                "model_trends": model_trends,
            })
        return out

    @staticmethod
    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        """OLS slope (units/year) of y vs x, NaN-safe."""
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            return float("nan")
        return float(np.polyfit(x[finite], y[finite], 1)[0])

    def _bootstrap_ci(self, x, y, rng) -> tuple[float, float]:
        """2.5–97.5 % bootstrap CI of the slope (resample years w/ replacement)."""
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        n = len(x)
        if n < 3:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, n, size=(self.n_bootstrap, n))
        xb = x[idx]; yb = y[idx]
        xbar = xb.mean(axis=1, keepdims=True)
        ybar = yb.mean(axis=1, keepdims=True)
        num = ((xb - xbar) * (yb - ybar)).sum(axis=1)
        den = ((xb - xbar) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.where(den > 0, num / den, np.nan)
        return (float(np.nanpercentile(slopes, 2.5)),
                float(np.nanpercentile(slopes, 97.5)))

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        if not results.get("models") and results.get("ref_model") is None:
            logger.warning("HeatwaveHotspotsDiag: no data — no figures produced")
            return figs

        figs.append(self._plot_trend_map(results))
        figs.append(self._plot_regions(results))
        if results["models"]:
            figs.append(self._plot_discrepancy(results))
            figs.append(self._plot_boxwhisker(results))
            figs.append(self._plot_pdf(results))
        return figs

    def _masked_obs_trend(self, results):
        """ERA5 D-trend, land-masked and grey-masked where P87.5 trend < 0."""
        ref = results["ref_model"]
        trend = results["trend_d"][ref]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0
        shown = trend.where(land & summer_pos)
        greyed = land & ~summer_pos
        return shown, greyed

    def _plot_trend_map(self, results):
        """Fig 2A: global ERA5 D-trend map with stippling + region boxes."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ref = results["ref_model"]
        lats, lons = results["lat"], results["lon"]
        shown, greyed = self._masked_obs_trend(results)
        nonsig = (results["trend_d_pval"][ref] >= 0.05) & results["land_mask"]

        vmax = float(np.nanpercentile(np.abs(shown.values), 98)) or 0.5
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

        lon_p = np.where(lons > 180, lons - 360, lons)
        order = np.argsort(lon_p)
        lon_s = lon_p[order]
        pc = ccrs.PlateCarree()

        # Grey land where the 87.5th-pctile trend is negative
        grey = greyed.isel(lon=order).astype(float).where(greyed.isel(lon=order))
        ax.pcolormesh(lon_s, lats, grey.values, transform=pc,
                      cmap="Greys", vmin=0, vmax=1.5, shading="auto", zorder=1)
        mesh = ax.pcolormesh(
            lon_s, lats, shown.isel(lon=order).values, transform=pc,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", zorder=2,
        )
        # Non-significance stippling (subsampled for legibility)
        latg, long = np.meshgrid(lats, lon_s, indexing="ij")
        ns = nonsig.isel(lon=order).values
        step = max(1, len(lats) // 90)
        sel = np.zeros_like(ns, dtype=bool)
        sel[::step, ::step] = True
        pts = ns & sel
        ax.scatter(long[pts], latg[pts], s=0.6, c="white", alpha=0.5,
                   transform=pc, zorder=3, linewidths=0)

        for region in results["regions"]:
            self._draw_region_box(ax, region, pc)

        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                          shrink=0.6, pad=0.05, extend="both")
        cb.set_label(f"Trend in yearly P{_UPPER_Q:g}−P{_LOWER_Q:g} of daily "
                     f"Tx [°C/decade]")
        obs_label = obs_ref_label(self.config, "tasmax")
        ax.set_title(f"{self.title} — {obs_label} extreme-heat tail-widening "
                     f"({self.period[0]}–{self.period[1]})")
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — {obs_label} tail-widening trend map",
            figure_id="heatwave_hotspots_trend_map",
            models=[ref],
            description=(
                "Linear trend in the yearly difference between the 99th and "
                "87.5th percentiles of daily maximum temperature (°C/decade) "
                f"in {obs_label}.  Grey: land where the 87.5th-percentile trend "
                "is negative.  White stippling: trend not significant at p<0.05."
            ),
            period=self.period,
            plot_type="map",
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
        )
        return fig, meta

    @staticmethod
    def _draw_region_box(ax, region, pc):
        import matplotlib.patches as mpatches

        lat_min, lat_max = region["lat"]
        lon_min, lon_max = region["lon"]
        ax.add_patch(mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            transform=pc, fill=False, edgecolor="black", linewidth=1.0,
            zorder=4,
        ))

    def _plot_regions(self, results):
        """Fig 2B–K: per-region yearly-D time series with fitted trends."""
        regions = results["regions"]
        ref = results["ref_model"]
        obs_label = obs_ref_label(self.config, "tasmax")
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)

        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            years = region["years"]
            # Models
            for m, s in region["model_series"].items():
                color = self.config.get_model_color(m)
                ax.plot(years, s, color=color, lw=1.0, alpha=0.8, label=m)
            # Obs (reference) on top
            ax.plot(years, region["obs_series"], color="tab:red", lw=2.0,
                    label=obs_label, zorder=5)
            self._add_trend_line(ax, years, region["obs_series"], "tab:red")
            ax.set_title(f"{chr(98 + i)}) {region['title']}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if i % ncols == 0:
                ax.set_ylabel(f"P{_UPPER_Q:g}−P{_LOWER_Q:g} Tx [°C]", fontsize=8)

        # Hide unused axes
        for j in range(len(regions), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{self.title} — regional tail-width time series", y=1.0)
        fig.tight_layout()

        stats = {
            r["title"]: {
                "obs_trend_per_decade": r["obs_trend"],
                "obs_ci_low": r["obs_ci"][0], "obs_ci_high": r["obs_ci"][1],
            }
            for r in regions
        }
        meta = self._build_metadata(
            title=f"{self.title} — regional tail-width time series",
            figure_id="heatwave_hotspots_regions",
            models=[ref] + results["models"],
            description=(
                "Area-weighted yearly P99−P87.5 of daily Tx for each region "
                "(land only); reference reanalysis in red with its fitted trend."
            ),
            period=self.period,
            plot_type="timeseries",
            summary_statistics=stats,
        )
        return fig, meta

    @staticmethod
    def _add_trend_line(ax, x, y, color):
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            return
        coef = np.polyfit(x[finite], y[finite], 1)
        ax.plot(x, np.polyval(coef, x), color=color, lw=1.0, ls="--", alpha=0.7)

    def _plot_discrepancy(self, results):
        """Fig 3A: where the observed trend exceeds the model-ensemble range."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ref = results["ref_model"]
        models = results["models"]
        lats, lons = results["lat"], results["lon"]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0

        obs_trend = results["trend_d"][ref]
        ens = xr.concat([results["trend_d"][m] for m in models], dim="member")
        ens_max = ens.max("member")
        ens_min = ens.min("member")
        # Positive = obs above ensemble max (model underestimation)
        disc = xr.where(obs_trend > ens_max, obs_trend - ens_max,
                        xr.where(obs_trend < ens_min, obs_trend - ens_min, 0.0))
        disc = disc.where(land & summer_pos)

        vmax = float(np.nanpercentile(np.abs(disc.values), 98)) or 0.5
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        lon_p = np.where(lons > 180, lons - 360, lons)
        order = np.argsort(lon_p)
        mesh = ax.pcolormesh(
            lon_p[order], lats, disc.isel(lon=order).values,
            transform=ccrs.PlateCarree(), cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, shading="auto",
        )
        for region in results["regions"]:
            self._draw_region_box(ax, region, ccrs.PlateCarree())
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                          shrink=0.6, pad=0.05, extend="both")
        cb.set_label("Observed minus model-ensemble range [°C/decade]")
        ax.set_title(f"{self.title} — observed trend vs model-ensemble range "
                     f"(red = model underestimation)")
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — observed vs model-ensemble discrepancy",
            figure_id="heatwave_hotspots_discrepancy",
            models=models,
            description=(
                "Difference between the observed tail-widening trend and the "
                "model-ensemble range: positive (red) where the observed trend "
                "exceeds every model (model underestimation).  Land only; grey-"
                "masked region (P87.5 trend < 0) excluded."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_boxwhisker(self, results):
        """Fig 3B–K: per-region distribution of model trends vs observed."""
        regions = results["regions"]
        models = results["models"]
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)

        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            model_vals = np.array(
                [region["model_trends"][m] for m in models], dtype=float,
            )
            model_vals = model_vals[np.isfinite(model_vals)]
            if model_vals.size:
                ax.boxplot(model_vals, vert=True, widths=0.5,
                           positions=[0], showfliers=False,
                           whis=(5, 95), patch_artist=True,
                           boxprops=dict(facecolor="lightsteelblue"))
                for m in models:
                    ax.scatter(0, region["model_trends"][m],
                               color=self.config.get_model_color(m),
                               s=18, zorder=4)
            # Observed trend + CI
            ci_lo, ci_hi = region["obs_ci"]
            ax.errorbar(0.0, region["obs_trend"],
                        yerr=[[region["obs_trend"] - ci_lo],
                              [ci_hi - region["obs_trend"]]],
                        fmt="D", color="tab:red", capsize=4, zorder=5,
                        label="ERA5")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xticks([])
            ax.set_title(f"{chr(98 + i)}) {region['title']}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            if i % ncols == 0:
                ax.set_ylabel("Trend [°C/decade]", fontsize=8)

        for j in range(len(regions), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"{self.title} — model trends vs observed (per region)",
                     y=1.0)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — model vs observed regional trends",
            figure_id="heatwave_hotspots_boxwhisker",
            models=models,
            description=(
                "Per-region distribution of modelled tail-widening trends "
                "(box: 25–75 %, whiskers: 5–95 %, scatter: individual models) "
                "versus the observed trend (red diamond, 2.5–97.5 % bootstrap CI)."
            ),
            period=self.period,
            plot_type="distribution",
        )
        return fig, meta

    def _plot_pdf(self, results):
        """Fig 4: area-weighted PDF of land D-trends, ERA5 vs models."""
        ref = results["ref_model"]
        models = results["models"]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0
        mask = land & summer_pos
        areas = xr.DataArray(results["areas"], dims=("lat", "lon"),
                             coords={"lat": results["lat"], "lon": results["lon"]})

        def _vals_weights(trend):
            v = trend.where(mask).values.ravel()
            w = areas.where(mask).values.ravel()
            ok = np.isfinite(v) & np.isfinite(w)
            return v[ok], w[ok]

        bins = np.linspace(-1.0, 1.0, 61)
        centers = 0.5 * (bins[:-1] + bins[1:])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        def _density(trend):
            v, w = _vals_weights(trend)
            if v.size == 0 or w.sum() <= 0:
                return None
            h, _ = np.histogram(v, bins=bins, weights=w, density=True)
            return h

        obs_hist = _density(results["trend_d"][ref])
        obs_label = obs_ref_label(self.config, "tasmax")
        if obs_hist is not None:
            ax1.step(centers, obs_hist, where="mid", color="tab:red", lw=2.0,
                     label=obs_label, zorder=5)

        model_hists = []
        for m in models:
            h = _density(results["trend_d"][m])
            if h is None:
                continue
            model_hists.append(h)
            ax1.step(centers, h, where="mid", color=self.config.get_model_color(m),
                     lw=1.0, alpha=0.7, label=m)
        ax1.set_yscale("log")
        ax1.set_xlabel("Trend in yearly P99−P87.5 of Tx [°C/decade]")
        ax1.set_ylabel("Area-weighted probability density")
        ax1.set_title("a) Distribution of land tail-widening trends")
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color="grey", lw=0.6)

        # Cumulative difference: obs CDF − ensemble-mean CDF
        if model_hists and obs_hist is not None:
            ens_mean = np.mean(model_hists, axis=0)
            dx = np.diff(bins)
            obs_cdf = np.cumsum(obs_hist * dx)
            ens_cdf = np.cumsum(ens_mean * dx)
            ax2.plot(centers, obs_cdf - ens_cdf, color="black", lw=2.0)
            ax2.axhline(0, color="grey", lw=0.6)
            ax2.set_xlabel("Trend [°C/decade]")
            ax2.set_ylabel("CDF(obs) − CDF(model ensemble mean)")
            ax2.set_title("b) Cumulative obs−model difference")
            ax2.grid(True, alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — land tail-widening trend distribution",
            figure_id="heatwave_hotspots_pdf",
            models=[ref] + models,
            description=(
                "Area-weighted probability density of land tail-widening trends "
                "(yearly P99−P87.5 of daily Tx) for the reanalysis reference and "
                "each model (log y-axis), with the cumulative obs−model "
                "difference.  Models underestimating the extreme tail produce a "
                "positive cumulative difference at high trend values."
            ),
            period=self.period,
            plot_type="distribution",
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
        )
        return fig, meta


def nr_regrid_probe(values, lon, lat, resolution, influence_radius):
    """Build a nereus interpolator for a source grid (returns regridded, interp).

    Thin wrapper so the import of ``nereus`` is localised and easy to stub in
    unit tests.
    """
    import nereus as nr

    return nr.regrid(
        np.asarray(values),
        lon=np.asarray(lon), lat=np.asarray(lat),
        resolution=resolution, influence_radius=influence_radius,
        lon_bounds=(0.0, 360.0), as_xarray=True,
    )
