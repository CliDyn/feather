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
  - ``heatwave_hotspots_trend_maps_members`` D-trend maps of ERA5, every
        evaluated member and the CMIP6 MMM on one colour scale.
  - ``heatwave_hotspots_trend_maps_family``  ERA5, the model-family means and
        the CMIP6 MMM on one colour scale.
  - ``heatwave_hotspots_trend_diff_family``  ERA5 (own scale), then each
        family mean and the CMIP6 MMM minus ERA5 on a shared difference scale.
  All three use ERA5's land + P87.5-trend > 0 mask in every panel.

With ``project.ensemble_mode: per_family`` the four figures that summarise the
evaluated models as one ensemble (regions, discrepancy, boxwhisker, pdf) and
``heatwave_hotspots_eerie_vs_cmip6`` are drawn per model family instead and
their ids gain a ``_family`` suffix: family-mean lines over thin members;
obs minus each family's member range (top row) and family mean (bottom row);
member dots + family-mean bar per family; family-mean densities and one
cumulative obs − family curve per family; family mean − CMIP6 MMM per family.

Per-model yearly percentile fields are cached to NetCDF (outside the figures
tree) so re-runs skip the heavy daily-data recompute:
``{output_dir}/heatwave_hotspots/{model}_percs_{start}_{end}.nc``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.diag._families import family_label
from feather.diag._extremes_obs import (
    obs_ref_label,
    obs_ref_model_name,
    _open_era5_sftlf,
)
from feather.util.spatial import compute_latlon_areas
from feather.util.temporal import linear_trend
from feather.diag.netcdf_export import write_netcdf

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
        variables=None,
    ):
        super().__init__(model_loader, obs_loader, config, cmip6_loader=cmip6_loader)
        # Single-variable diagnostic (tasmax); accept the pipeline's --variables
        # selection so the CLI can target it, but only honour the overlap.
        if variables:
            self.variables = [v for v in variables if v in type(self).variables] \
                or list(type(self).variables)
        self.experiment = experiment
        self.period = period
        self.n_bootstrap = n_bootstrap
        self._land_thresh = 25.0  # land fraction % (paper: 0.25)
        self._resolution = self.config.nereus.get("resolution", 0.25)
        self._influence_radius = self.config.nereus.get("influence_radius", 80_000.0)
        # Shared target grid + per-source-grid interpolator cache (filled by the
        # first ensemble member computed; reused by every subsequent member,
        # including the CMIP6 envelope, since the target grid is resolution-fixed).
        self._target_lats: np.ndarray | None = None
        self._target_lons: np.ndarray | None = None
        self._interp_cache: dict[tuple, Any] = {}
        # Optional daily CMIP6 envelope (read straight from the DRS NetCDF tree).
        self._cmip6_loader = None
        self._cmip6_models: list[str] = []
        self._cmip6_label = "CMIP6"
        self._init_cmip6_daily()

    # ── Daily CMIP6 setup ─────────────────────────────────────────────

    def _init_cmip6_daily(self) -> None:
        """Auto-discover the daily-CMIP6 ensemble from ``config.cmip6_daily``.

        Never raises: any discovery failure logs a warning and leaves the
        diagnostic running ERA5+EERIE only.
        """
        cfg = getattr(self.config, "cmip6_daily", {}) or {}
        if not cfg.get("enabled", False):
            return
        from feather.data.cmip6_nc_loader import (
            discover_daily_models, CMIP6NCLoader,
        )

        root = cfg.get("root", "/work/ik1017/CMIP6/data/CMIP6")
        experiment = cfg.get("experiment", "historical")
        table = cfg.get("table", "day")
        member = cfg.get("member", "r1i1p1f1")
        exclude = tuple(cfg.get("exclude", []))
        max_models = cfg.get("max_models")
        variable = self.variables[0] if self.variables else "tasmax"
        self._cmip6_label = cfg.get("label", "CMIP6")

        try:
            discovered = discover_daily_models(
                root, experiment=experiment, table=table, variable=variable,
                member=member, exclude=exclude, max_models=max_models,
            )
            wanted = cfg.get("models", "auto")
            if isinstance(wanted, (list, tuple, set)):
                names = set(wanted)
                discovered = [m for m in discovered if m.name in names]
        except Exception as exc:  # pragma: no cover - defensive (FS walk)
            logger.warning(
                "CMIP6 daily discovery failed (%s) — skipping CMIP6 envelope",
                exc,
            )
            return

        if not discovered:
            logger.warning(
                "CMIP6 daily: no models found for %s/%s %s — skipping envelope",
                experiment, table, variable,
            )
            return

        self._cmip6_loader = CMIP6NCLoader.for_models(
            self.config, discovered, root=root,
        )
        self._cmip6_models = [m.name for m in discovered]
        logger.info(
            "CMIP6 daily envelope: %d model(s) — %s",
            len(self._cmip6_models), ", ".join(self._cmip6_models),
        )

    # ── Paths ─────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-model percentile NetCDF files (outside figures)."""
        return Path(self.config.output_dir) / "heatwave_hotspots"

    def _nc_path(self, model: str, tag: str = "") -> Path:
        start, end = self.period
        safe = model.replace("/", "_").replace(" ", "_")
        prefix = f"{tag}_" if tag else ""
        return self.nc_dir / f"{prefix}{safe}_percs_{start}_{end}.nc"

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
        "heatwave_hotspots_trend_maps_members",
        "heatwave_hotspots_trend_maps_family",
        "heatwave_hotspots_trend_diff_family",
    ]

    #: Extra figures rendered only when a daily CMIP6 envelope is configured.
    _CMIP6_FIG_IDS = [
        "heatwave_hotspots_cmip6_mmm_map",
        "heatwave_hotspots_cmip6_discrepancy",
        "heatwave_hotspots_eerie_vs_cmip6",
    ]

    #: Figures that summarise the evaluated models as one ensemble.  With
    #: ``project.ensemble_mode: per_family`` they are drawn per model family
    #: instead and their ids gain a ``_family`` suffix.
    _ENSEMBLE_FIG_IDS = frozenset({
        "heatwave_hotspots_regions",
        "heatwave_hotspots_discrepancy",
        "heatwave_hotspots_boxwhisker",
        "heatwave_hotspots_pdf",
        "heatwave_hotspots_eerie_vs_cmip6",
    })

    def _fig_id(self, base: str) -> str:
        """Figure id for *base*, with ``_family`` in per-family mode."""
        if self._per_family and base in self._ENSEMBLE_FIG_IDS:
            return f"{base}_family"
        return base

    @property
    def _expected_fig_ids(self) -> list[str]:
        ids = list(self._FIG_IDS)
        if self._cmip6_models:
            ids += self._CMIP6_FIG_IDS
        return [self._fig_id(i) for i in ids]

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute per model → regrid → trends → plot → save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing and all(
            self._figure_exists(f) for f in self._expected_fig_ids
        ):
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
        self._target_lats = None
        self._target_lons = None
        self._interp_cache = {}

        # ERA5 reference + EERIE high-resolution ensemble.
        d_common, trend_d, trend_d_pval, trend_lower = self._ensemble_fields(
            list(self.config.models), self.model_loader,
        )

        if not d_common:
            logger.warning("HeatwaveHotspotsDiag: no model data loaded")
            return {"models": [], "ref_model": None}

        # Daily CMIP6 envelope (separate ensemble, separate NetCDF cache).
        cmip6_d_common, cmip6_trend_d, _c_pval, _c_lower = (
            self._ensemble_fields(
                self._cmip6_models, self._cmip6_loader, tag="cmip6",
                grid_type="latlon",
            )
            if self._cmip6_loader is not None else ({}, {}, {}, {})
        )

        target_lats, target_lons = self._target_lats, self._target_lons

        # Shared land mask on the common grid (ERA5 sftlf, else Berkeley mask).
        land_mask = self._common_land_mask(target_lats, target_lons)
        areas = compute_latlon_areas(target_lats, target_lons)

        ref_model = obs_ref_model_name(self.config)
        if ref_model not in d_common:  # ERA5 not loaded — fall back to first
            ref_model = next(iter(d_common))
        models = [m for m in d_common if m != ref_model]

        # Per-region observed/model trends and bootstrap CIs (+ CMIP6 series).
        regions = self._compute_regions(
            d_common, trend_d, ref_model, models, target_lats, target_lons, areas,
            cmip6_d_common=cmip6_d_common,
        )

        # CMIP6 spatial aggregates: MMM + min/max envelope of the D-trend.
        cmip6_mmm_trend = cmip6_env_min = cmip6_env_max = None
        cmip6_models = list(cmip6_trend_d)
        if cmip6_models:
            ens = xr.concat([cmip6_trend_d[m] for m in cmip6_models],
                            dim="member")
            cmip6_mmm_trend = ens.mean("member")
            cmip6_env_min = ens.min("member")
            cmip6_env_max = ens.max("member")

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
            "cmip6_models": cmip6_models,
            "cmip6_label": self._cmip6_label,
            "cmip6_trend_d": cmip6_trend_d,
            "cmip6_mmm_trend": cmip6_mmm_trend,
            "cmip6_env_min": cmip6_env_min,
            "cmip6_env_max": cmip6_env_max,
        }

    def _ensemble_fields(
        self, models, loader, *, tag: str = "", grid_type: str | None = None,
    ) -> tuple[dict, dict, dict, dict]:
        """Compute regridded tail-width fields + trends for one ensemble.

        Returns ``(d_common, trend_d, trend_d_pval, trend_lower)`` keyed by
        model.  Populates the shared target grid / interpolator cache on first
        use.  ``grid_type`` forces the source-grid kind (CMIP6 models are not in
        ``model_configs`` so the per-model config lookup would misfire — pass
        ``"latlon"`` for them).
        """
        d_common: dict[str, xr.DataArray] = {}        # model → D(year, lat, lon)
        trend_d: dict[str, xr.DataArray] = {}         # model → D-trend (lat, lon)
        trend_d_pval: dict[str, xr.DataArray] = {}    # model → D-trend p-value
        trend_lower: dict[str, xr.DataArray] = {}     # model → P87.5-trend

        for model in models:
            try:
                percs = self._load_or_compute_percs(model, loader, tag=tag)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue

            lon = np.asarray(percs["lon"])
            lat = np.asarray(percs["lat"])
            gtype = grid_type or self.config.get_grid_type(model, self.domain)
            if gtype != "healpix":
                lon2d, lat2d = np.meshgrid(lon, lat)
            else:
                lon2d, lat2d = lon, lat

            grid_key = self._grid_signature(lon2d, lat2d)
            if grid_key not in self._interp_cache:
                logger.info("  %s: building nereus interpolator (%d pts)",
                            model, np.asarray(lon2d).size)
                _, interp = nr_regrid_probe(
                    percs["p99"].isel(year=0).values.ravel(),
                    np.asarray(lon2d).ravel(), np.asarray(lat2d).ravel(),
                    self._resolution, self._influence_radius,
                )
                self._interp_cache[grid_key] = interp
                if self._target_lats is None:
                    self._target_lats = interp.target_lat[:, 0]
                    self._target_lons = interp.target_lon[0, :]
            interp = self._interp_cache[grid_key]
            t_lats, t_lons = self._target_lats, self._target_lons

            d_native = (percs["p99"] - percs["p875"])  # (year, lat, lon)
            d_reg = self._regrid_stack(d_native, interp, t_lats, t_lons)
            lower_reg = self._regrid_stack(percs["p875"], interp, t_lats, t_lons)

            d_common[model] = d_reg
            trend, pval = self._ols_trend_stats(d_reg)
            trend_d[model] = trend
            trend_d_pval[model] = pval
            trend_lower[model] = linear_trend(lower_reg, dim="year") * 10.0

        return d_common, trend_d, trend_d_pval, trend_lower

    # ── Per-model percentiles (cached) ─────────────────────────────────

    def _load_or_compute_percs(
        self, model: str, loader=None, *, tag: str = "",
    ) -> xr.Dataset:
        """Return Dataset(p99, p875) with dims (year, lat, lon), from NC or fresh.

        ``loader`` defaults to the main EERIE model loader; pass the daily CMIP6
        loader (with ``tag="cmip6"``) to cache CMIP6 percentiles separately.
        """
        loader = loader or self.model_loader
        nc_path = self._nc_path(model, tag=tag)
        if nc_path.exists():
            logger.info("  %s: loading percentiles from %s", model, nc_path.name)
            ds = xr.open_dataset(nc_path)
            if {"p99", "p875"}.issubset(ds.data_vars):
                return ds
            logger.info("  %s: NC incomplete — recomputing", model)

        logger.info("  %s: computing yearly P99/P87.5 from daily tasmax", model)
        da = loader.load_var(
            model, "tasmax", table="day", period=self.period,
        )
        ds = self._compute_percentiles(da)
        self._save_nc(model, ds, tag=tag)
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

    def _save_nc(self, model: str, ds: xr.Dataset, tag: str = "") -> None:
        nc_path = self._nc_path(model, tag=tag)
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
        write_netcdf(ds, nc_path)
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
        *, cmip6_d_common=None,
    ) -> list[dict]:
        """Per-region observed trend (+ bootstrap CI) and model-ensemble trends.

        When ``cmip6_d_common`` is given, also stores the per-CMIP6-model
        regional series and trends so the envelope can be drawn in the
        time-series and box-whisker figures.
        """
        cmip6_d_common = cmip6_d_common or {}
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

            cmip6_series: dict[str, np.ndarray] = {}
            cmip6_trends: dict[str, float] = {}
            for m, field in cmip6_d_common.items():
                s = self._regional_series(field, rmask, land_mask, areas)
                cmip6_series[m] = s
                cmip6_trends[m] = self._slope(years, s) * 10.0

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
                "cmip6_series": cmip6_series,
                "cmip6_trends": cmip6_trends,
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

        fam = self._per_family
        figs.append(self._plot_trend_map(results))
        figs.append(self._plot_regions_family(results) if fam
                    else self._plot_regions(results))
        if results["models"]:
            figs.append(self._plot_trend_maps_members(results))
            figs.append(self._plot_trend_maps_family(results))
            figs.append(self._plot_trend_diff_family(results))
            if fam:
                figs.append(self._plot_discrepancy_family(results))
                figs.append(self._plot_boxwhisker_family(results))
                figs.append(self._plot_pdf_family(results))
            else:
                figs.append(self._plot_discrepancy(results))
                figs.append(self._plot_boxwhisker(results))
                figs.append(self._plot_pdf(results))
        if results.get("cmip6_models"):
            figs.append(self._plot_cmip6_mmm_map(results))
            figs.append(self._plot_cmip6_discrepancy(results))
            if results["models"]:
                figs.append(self._plot_eerie_vs_cmip6_family(results) if fam
                            else self._plot_eerie_vs_cmip6(results))
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
        cmip6_label = results.get("cmip6_label", "CMIP6")
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)

        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            years = region["years"]
            # CMIP6 envelope (min/max band + dashed MMM) in the background
            cmip6_series = region.get("cmip6_series", {})
            if cmip6_series:
                arr = np.vstack([cmip6_series[m] for m in cmip6_series])
                if np.isfinite(arr).any():
                    with np.errstate(invalid="ignore"):
                        ax.fill_between(years, np.nanmin(arr, axis=0),
                                        np.nanmax(arr, axis=0), color="grey",
                                        alpha=0.25, lw=0, zorder=1,
                                        label=f"{cmip6_label} range")
                        ax.plot(years, np.nanmean(arr, axis=0), color="grey",
                                lw=1.5, ls="--", zorder=2,
                                label=f"{cmip6_label} MMM")
            # Models
            for m, s in region["model_series"].items():
                color = self.config.get_model_color(m)
                ax.plot(years, s, color=color, lw=1.0, alpha=0.8, label=m,
                        zorder=3)
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
        fig.suptitle(f"{self.title} — regional tail-width time series", y=0.99)
        # Reserve a band at the bottom for the legend and the top for the
        # suptitle *before* placing the legend, so tight_layout does not
        # reclaim the space and overlap the legend onto the bottom row.
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, 0.0))

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
            models=[ref] + results["models"] + results.get("cmip6_models", []),
            description=(
                "Area-weighted yearly P99−P87.5 of daily Tx for each region "
                "(land only); reference reanalysis in red with its fitted trend, "
                "EERIE members as coloured lines, and the CMIP6 ensemble as a "
                "grey min–max band with dashed MMM."
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
        cmip6_label = results.get("cmip6_label", "CMIP6")
        obs_label = obs_ref_label(self.config, "tasmax")
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)

        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            ticks: list[float] = []
            tick_labels: list[str] = []

            # EERIE box + per-member scatter at x=0
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
                ticks.append(0); tick_labels.append("EERIE")

            # CMIP6 envelope box at x=1
            cmip6_vals = np.array(
                [v for v in region.get("cmip6_trends", {}).values()],
                dtype=float,
            )
            cmip6_vals = cmip6_vals[np.isfinite(cmip6_vals)]
            if cmip6_vals.size:
                ax.boxplot(cmip6_vals, vert=True, widths=0.5,
                           positions=[1], showfliers=False,
                           whis=(5, 95), patch_artist=True,
                           boxprops=dict(facecolor="lightgrey"))
                ax.scatter(np.ones_like(cmip6_vals), cmip6_vals,
                           color="grey", s=10, alpha=0.6, zorder=3)
                ticks.append(1); tick_labels.append(cmip6_label)

            # Observed trend + CI as a horizontal reference band
            ci_lo, ci_hi = region["obs_ci"]
            ax.axhspan(ci_lo, ci_hi, color="tab:red", alpha=0.12, zorder=1)
            ax.axhline(region["obs_trend"], color="tab:red", lw=1.5, ls="--",
                       zorder=5, label=obs_label)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlim(-0.6, 1.6)
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
            models=models + results.get("cmip6_models", []),
            description=(
                "Per-region distribution of modelled tail-widening trends "
                "(box: 25–75 %, whiskers: 5–95 %, scatter: individual models) "
                "for the EERIE ensemble and the CMIP6 envelope, versus the "
                f"observed trend ({obs_label}, red dashed line with shaded "
                "2.5–97.5 % bootstrap CI)."
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

    # ── Multi-panel trend maps (ERA5 / members or families / CMIP6 MMM) ───

    def _trend_panel_figure(self, results, panels, scales, *, ncols, suptitle):
        """Robinson trend panels sharing the ERA5 land + P87.5 > 0 mask.

        *panels* is a list of ``(title, field, scale)``; *scales* maps a scale
        name to ``(cmap, vmax, colourbar label)``.  The first scale gets a
        horizontal colourbar under the figure; any other scale a vertical one
        beside its first panel.
        """
        import cartopy.crs as ccrs

        ref = results["ref_model"]
        lats, lons = results["lat"], results["lon"]
        land = results["land_mask"]
        greyed = land & ~(results["trend_lower"][ref] > 0)
        nrows = int(np.ceil(len(panels) / ncols))
        side = len(scales) > 1
        fig = plt.figure(figsize=(4.6 * ncols, 2.6 * nrows + 1.0))
        fig.subplots_adjust(left=0.02, right=0.96 if side else 0.98,
                            top=0.90 if nrows > 1 else 0.84,
                            bottom=0.13 if nrows > 1 else 0.2,
                            wspace=0.2 if side else 0.05, hspace=0.25)
        first: dict[str, tuple] = {}
        for i, (title, field, scale) in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, i + 1,
                                 projection=ccrs.Robinson(central_longitude=0))
            cmap, vmax, _ = scales[scale]
            mesh = self._draw_map_panel(ax, field, lats, lons, cmap=cmap,
                                        vmax=vmax, regions=results["regions"],
                                        greyed=greyed)
            ax.set_title(title, fontsize=9)
            first.setdefault(scale, (ax, mesh))

        for k, scale in enumerate(s for s in scales if s in first):
            ax, mesh = first[scale]
            label = scales[scale][2]
            if k == 0:
                cax = fig.add_axes([0.3, 0.05 if nrows > 1 else 0.08, 0.4, 0.018])
                cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                                  extend="both")
            else:
                ax.apply_aspect()
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1 + 0.006, pos.y0 + 0.1 * pos.height,
                                    0.007, 0.8 * pos.height])
                cb = fig.colorbar(mesh, cax=cax, orientation="vertical",
                                  extend="both")
                cb.ax.tick_params(labelsize=7)
            cb.set_label(label, fontsize=8)
        fig.suptitle(suptitle, y=0.99)
        return fig

    def _trend_shown(self, results, field):
        """*field* on land where ERA5's P87.5 trend is positive."""
        ref = results["ref_model"]
        return field.where(results["land_mask"]
                           & (results["trend_lower"][ref] > 0))

    @staticmethod
    def _sym_vmax(fields, default=0.5) -> float:
        vals = np.concatenate([np.ravel(np.asarray(f)) for f in fields])
        vals = np.abs(vals[np.isfinite(vals)])
        return (float(np.nanpercentile(vals, 98)) if vals.size else 0.0) or default

    def _area_mean(self, results, field) -> float | None:
        w = xr.DataArray(results["areas"], dims=("lat", "lon"),
                         coords={"lat": results["lat"], "lon": results["lon"]})
        w = w.where(np.isfinite(field))
        total = float(w.sum())
        return float((w * field).sum()) / total if total > 0 else None

    def _pattern_corr(self, results, a, b) -> float | None:
        """Area-weighted centred pattern correlation of *a* and *b*."""
        w = xr.DataArray(results["areas"], dims=("lat", "lon"),
                         coords={"lat": results["lat"], "lon": results["lon"]})
        ok = np.isfinite(a) & np.isfinite(b)
        w = w.where(ok, 0.0)
        if float(w.sum()) <= 0:
            return None
        am = float((w * a.fillna(0)).sum() / w.sum())
        bm = float((w * b.fillna(0)).sum() / w.sum())
        da, db = (a - am).where(ok, 0.0), (b - bm).where(ok, 0.0)
        cov = float((w * da * db).sum())
        var = float((w * da ** 2).sum()) * float((w * db ** 2).sum())
        return cov / np.sqrt(var) if var > 0 else None

    def _trend_panel_stats(self, results, shown, era5_shown) -> dict:
        st = {"land_mean_trend_per_decade": self._area_mean(results, shown)}
        if shown is not era5_shown:
            diff = shown - era5_shown
            st["mean_diff_to_era5"] = self._area_mean(results, diff)
            msq = self._area_mean(results, diff ** 2)
            st["rmse_to_era5"] = float(np.sqrt(msq)) if msq is not None else None
            st["pattern_corr_with_era5"] = self._pattern_corr(
                results, shown, era5_shown)
        return st

    def _cmip6_panel(self, results):
        """``(label, shown MMM trend)`` or ``None`` without daily CMIP6."""
        mmm = results.get("cmip6_mmm_trend")
        if mmm is None or not results.get("cmip6_models"):
            return None
        label = (f"{results.get('cmip6_label', 'CMIP6')} MMM "
                 f"({len(results['cmip6_models'])})")
        return label, self._trend_shown(results, mmm)

    _TREND_CBLABEL = (f"Trend in yearly P{_UPPER_Q:g}−P{_LOWER_Q:g} of daily Tx "
                      f"[°C/decade]")

    def _plot_trend_maps_members(self, results):
        """ERA5, every evaluated member and the CMIP6 MMM on one trend scale."""
        ref = results["ref_model"]
        obs_label = obs_ref_label(self.config, "tasmax")
        era5 = self._trend_shown(results, results["trend_d"][ref])
        ordered = [m for f in self._families(results).values() for m in f]
        entries = [(obs_label, era5)] + [
            (m, self._trend_shown(results, results["trend_d"][m]))
            for m in ordered]
        cm = self._cmip6_panel(results)
        if cm:
            entries.append(cm)
        vmax = self._sym_vmax([f for _, f in entries])
        fig = self._trend_panel_figure(
            results, [(t, f, "trend") for t, f in entries],
            {"trend": ("RdBu_r", vmax, self._TREND_CBLABEL)},
            ncols=4,
            suptitle=(f"{self.title} — tail-widening trend, {obs_label}, EERIE "
                      f"members{' and CMIP6 MMM' if cm else ''} "
                      f"({self.period[0]}–{self.period[1]})"),
        )
        stats = {t: self._trend_panel_stats(results, f, era5)
                 for t, f in entries}
        meta = self._build_metadata(
            title=f"{self.title} — trend maps: {obs_label}, EERIE members, CMIP6 MMM",
            figure_id="heatwave_hotspots_trend_maps_members",
            models=[ref] + ordered + results.get("cmip6_models", []),
            description=(
                "Linear trend in the yearly P99−P87.5 of daily maximum "
                f"temperature (°C/decade) for {obs_label} (first panel), each "
                f"of the {len(ordered)} EERIE members with daily tasmax"
                + (f" and the {cm[0]}" if cm else "") + ", on one colour "
                "scale.  Land only; grey: land where the ERA5 87.5th-percentile "
                "trend is negative (excluded in every panel)."
            ),
            period=self.period,
            plot_type="map",
            summary_statistics=stats,
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
        )
        return fig, meta

    def _plot_trend_maps_family(self, results):
        """ERA5, the family-mean trends and the CMIP6 MMM on one trend scale."""
        ref = results["ref_model"]
        obs_label = obs_ref_label(self.config, "tasmax")
        families = self._families(results)
        era5 = self._trend_shown(results, results["trend_d"][ref])
        entries = [(obs_label, era5)] + [
            (family_label(f, len(m), bold=False),
             self._trend_shown(results, self._family_trend(results, m)))
            for f, m in families.items()]
        cm = self._cmip6_panel(results)
        if cm:
            entries.append(cm)
        vmax = self._sym_vmax([f for _, f in entries])
        fig = self._trend_panel_figure(
            results, [(t, f, "trend") for t, f in entries],
            {"trend": ("RdBu_r", vmax, self._TREND_CBLABEL)},
            ncols=3,
            suptitle=(f"{self.title} — tail-widening trend, {obs_label}, model-"
                      f"family means{' and CMIP6 MMM' if cm else ''} "
                      f"({self.period[0]}–{self.period[1]})"),
        )
        stats = {t: self._trend_panel_stats(results, f, era5)
                 for t, f in entries}
        meta = self._build_metadata(
            title=f"{self.title} — trend maps: {obs_label}, family means, CMIP6 MMM",
            figure_id="heatwave_hotspots_trend_maps_family",
            models=[ref] + results["models"] + results.get("cmip6_models", []),
            description=(
                "Linear trend in the yearly P99−P87.5 of daily maximum "
                f"temperature (°C/decade) for {obs_label} (first panel), the "
                "mean of each EERIE model family's members"
                + (f" and the {cm[0]}" if cm else "") + ", on one colour "
                "scale.  Land only; grey: land where the ERA5 87.5th-percentile "
                "trend is negative." + self._families_note(families)
            ),
            period=self.period,
            plot_type="map",
            summary_statistics=stats,
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
            extra=self._family_meta(families),
        )
        return fig, meta

    def _plot_trend_diff_family(self, results):
        """ERA5 trend, then family means and CMIP6 MMM minus ERA5."""
        ref = results["ref_model"]
        obs_label = obs_ref_label(self.config, "tasmax")
        families = self._families(results)
        era5 = self._trend_shown(results, results["trend_d"][ref])
        models = [(family_label(f, len(m), bold=False),
                   self._trend_shown(results, self._family_trend(results, m)))
                  for f, m in families.items()]
        cm = self._cmip6_panel(results)
        if cm:
            models.append(cm)
        diffs = [(f"{t} − {obs_label}", f - era5) for t, f in models]
        fig = self._trend_panel_figure(
            results,
            [(obs_label, era5, "trend")] + [(t, d, "diff") for t, d in diffs],
            {"diff": ("RdBu_r", self._sym_vmax([d for _, d in diffs]),
                      f"Trend difference to {obs_label} [°C/decade]"),
             "trend": ("RdBu_r", self._sym_vmax([era5]), "Trend [°C/decade]")},
            ncols=3,
            suptitle=(f"{self.title} — tail-widening trend: model-family means"
                      f"{' and CMIP6 MMM' if cm else ''} minus {obs_label} "
                      f"({self.period[0]}–{self.period[1]})"),
        )
        stats = {obs_label: {"land_mean_trend_per_decade":
                             self._area_mean(results, era5)}}
        for (t, f), (dt, d) in zip(models, diffs):
            st = self._latlon_bias_stats(f, era5)
            st["pattern_corr_with_era5"] = self._pattern_corr(results, f, era5)
            st["land_mean_trend_per_decade"] = self._area_mean(results, f)
            stats[dt] = st
        meta = self._build_metadata(
            title=f"{self.title} — family-mean and CMIP6 MMM trend differences "
                  f"to {obs_label}",
            figure_id="heatwave_hotspots_trend_diff_family",
            models=[ref] + results["models"] + results.get("cmip6_models", []),
            description=(
                f"First panel: {obs_label} tail-widening trend (yearly "
                "P99−P87.5 of daily Tx, °C/decade, own colour scale).  Other "
                f"panels: each EERIE model family's mean trend minus {obs_label}"
                + (f", and the {cm[0]} minus {obs_label}" if cm else "")
                + ", on one shared difference scale; red = model trend stronger "
                "than observed.  Land only; grey: land where the ERA5 87.5th-"
                "percentile trend is negative.  Statistics per difference panel: "
                "area-weighted mean difference, RMSE, t-test, variance ratio and "
                "pattern correlation with ERA5." + self._families_note(families)
            ),
            period=self.period,
            plot_type="map",
            summary_statistics=stats,
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
            extra=self._family_meta(families),
        )
        return fig, meta

    # ── Per-family summaries (project.ensemble_mode: per_family) ─────────

    def _families(self, results) -> dict[str, list[str]]:
        """``{family: [members]}`` of the evaluated models that loaded.

        Config order, so a family's first member (whose colour it takes) is
        r1.  Independent of ``ensemble_mode``: the trend-map panels group by
        family in both modes.
        """
        models = results["models"]
        ordered = [m for m in self.config.models if m in models]
        ordered += [m for m in models if m not in ordered]
        return self.config.get_model_families(ordered)

    def _family_trend(self, results, members):
        """Mean D-trend of *members* (= trend of their mean, OLS is linear)."""
        return xr.concat([results["trend_d"][m] for m in members],
                         dim="member").mean("member")

    def _family_meta(self, families) -> dict[str, Any]:
        return {"ensemble_mode": "per_family",
                "model_families": {f: list(m) for f, m in families.items()}}

    @staticmethod
    def _families_note(families) -> str:
        fams = ", ".join(
            f"{f} ({len(m)} member{'s' if len(m) != 1 else ''})"
            for f, m in families.items())
        return (f" Model families are summarised separately and never pooled: "
                f"{fams}.")

    def _family_map_figure(self, rows, families, lats, lons, regions, *,
                           cblabels, suptitle):
        """Grid of Robinson maps: one row per entry of *rows*, one column per
        family.  ``rows`` is a list of ``(row_title, {family: field})``; each
        row gets its own symmetric colour scale and colourbar."""
        import cartopy.crs as ccrs

        nrow, ncol = len(rows), len(families)
        fig = plt.figure(figsize=(5.2 * ncol, 3.6 * nrow + 0.6))
        for r, (row_title, fields) in enumerate(rows):
            vals = np.concatenate([np.ravel(f.values) for f in fields.values()])
            vals = np.abs(vals[np.isfinite(vals)])
            vmax = (float(np.nanpercentile(vals, 98)) if vals.size else 0.0) or 0.5
            axes, mesh = [], None
            for c, (family, members) in enumerate(families.items()):
                ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1,
                                     projection=ccrs.Robinson(central_longitude=0))
                mesh = self._draw_map_panel(ax, fields[family], lats, lons,
                                            cmap="RdBu_r", vmax=vmax,
                                            regions=regions)
                ax.set_title(f"{row_title}\n"
                             f"{family_label(family, len(members), bold=False)}",
                             fontsize=9)
                axes.append(ax)
            cb = fig.colorbar(mesh, ax=axes, orientation="horizontal",
                              shrink=0.5, pad=0.04, extend="both")
            cb.set_label(cblabels[r], fontsize=8)
        fig.suptitle(suptitle, y=0.995)
        return fig

    def _plot_regions_family(self, results):
        """Fig 2B–K, per family: thick family-mean lines over thin members."""
        regions = results["regions"]
        ref = results["ref_model"]
        families = self._families(results)
        obs_label = obs_ref_label(self.config, "tasmax")
        cmip6_label = results.get("cmip6_label", "CMIP6")
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)

        stats: dict[str, dict] = {}
        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            years = region["years"]
            cmip6_series = region.get("cmip6_series", {})
            if cmip6_series:
                arr = np.vstack([cmip6_series[m] for m in cmip6_series])
                if np.isfinite(arr).any():
                    with np.errstate(invalid="ignore"):
                        ax.fill_between(years, np.nanmin(arr, axis=0),
                                        np.nanmax(arr, axis=0), color="grey",
                                        alpha=0.25, lw=0, zorder=1,
                                        label=f"{cmip6_label} range")
                        ax.plot(years, np.nanmean(arr, axis=0), color="grey",
                                lw=1.5, ls="--", zorder=2,
                                label=f"{cmip6_label} MMM")
            fam_trends: dict[str, float] = {}
            for family, members in families.items():
                color = self.config.get_model_color(members[0])
                arr = np.vstack([region["model_series"][m] for m in members])
                for m in members:
                    ax.plot(years, region["model_series"][m], color=color,
                            lw=0.7, alpha=0.45, zorder=3)
                with warnings.catch_warnings():  # all-NaN years stay NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(arr, axis=0)
                ax.plot(years, mean, color=color, lw=2.2, zorder=4,
                        label=family_label(family, len(members), bold=False))
                fam_trends[family] = self._slope(years, mean) * 10.0
            ax.plot(years, region["obs_series"], color="tab:red", lw=2.0,
                    label=obs_label, zorder=5)
            self._add_trend_line(ax, years, region["obs_series"], "tab:red")
            ax.set_title(f"{chr(98 + i)}) {region['title']}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if i % ncols == 0:
                ax.set_ylabel(f"P{_UPPER_Q:g}−P{_LOWER_Q:g} Tx [°C]", fontsize=8)
            stats[region["title"]] = {
                "obs_trend_per_decade": region["obs_trend"],
                "obs_ci_low": region["obs_ci"][0],
                "obs_ci_high": region["obs_ci"][1],
                "family_mean_trend_per_decade": fam_trends,
            }

        for j in range(len(regions), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(f"{self.title} — regional tail-width time series "
                     f"(family means)", y=0.99)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, 0.0))

        meta = self._build_metadata(
            title=f"{self.title} — regional tail-width time series (family means)",
            figure_id=self._fig_id("heatwave_hotspots_regions"),
            models=[ref] + results["models"] + results.get("cmip6_models", []),
            description=(
                "Area-weighted yearly P99−P87.5 of daily Tx for each region "
                "(land only); reference reanalysis in red with its fitted trend, "
                "one thick line per model family (mean of its members' regional "
                "series) over its members as thin lines, and the CMIP6 ensemble "
                "as a grey min–max band with dashed MMM."
                + self._families_note(families)
            ),
            period=self.period,
            plot_type="timeseries",
            summary_statistics=stats,
            extra=self._family_meta(families),
        )
        return fig, meta

    def _plot_discrepancy_family(self, results):
        """Fig 3A per family: obs vs the family's member range, and vs its mean."""
        ref = results["ref_model"]
        families = self._families(results)
        lats, lons = results["lat"], results["lon"]
        keep = results["land_mask"] & (results["trend_lower"][ref] > 0)
        obs_trend = results["trend_d"][ref]
        obs_label = obs_ref_label(self.config, "tasmax")

        by_range, by_mean, stats = {}, {}, {}
        areas = xr.DataArray(results["areas"], dims=("lat", "lon"),
                             coords={"lat": lats, "lon": lons})
        for family, members in families.items():
            ens = xr.concat([results["trend_d"][m] for m in members], dim="member")
            hi, lo = ens.max("member"), ens.min("member")
            by_range[family] = xr.where(
                obs_trend > hi, obs_trend - hi,
                xr.where(obs_trend < lo, obs_trend - lo, 0.0)).where(keep)
            by_mean[family] = (obs_trend - ens.mean("member")).where(keep)
            # Area fractions of the shown land where obs leaves the range.
            w = areas.where(np.isfinite(by_range[family]))
            total = float(w.sum())
            if total > 0:
                stats[family] = {
                    "frac_area_obs_above_range":
                        float((w * (by_range[family] > 0)).sum()) / total,
                    "frac_area_obs_below_range":
                        float((w * (by_range[family] < 0)).sum()) / total,
                    "mean_obs_minus_family_mean": float(
                        (w * by_mean[family]).sum()) / total,
                }

        fig = self._family_map_figure(
            [("Obs − family member range", by_range),
             ("Obs − family mean", by_mean)],
            families, lats, lons, results["regions"],
            cblabels=[f"{obs_label} minus family member range [°C/decade]",
                      f"{obs_label} minus family mean [°C/decade]"],
            suptitle=(f"{self.title} — observed trend vs model families "
                      f"(red = model underestimation)"),
        )
        meta = self._build_metadata(
            title=f"{self.title} — observed vs model-family discrepancy",
            figure_id=self._fig_id("heatwave_hotspots_discrepancy"),
            models=results["models"],
            description=(
                "One column per model family.  Top row: observed tail-widening "
                "trend minus the family's member range — positive (red) where "
                "the observed trend exceeds every member of the family (model "
                "underestimation), 0 inside the range; with three members the "
                "range is narrow, so read it together with the bottom row.  "
                "Bottom row: observed trend minus the family-mean trend.  Land "
                "only; region with a negative P87.5 trend excluded."
                + self._families_note(families)
            ),
            period=self.period,
            plot_type="map",
            summary_statistics=stats,
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
            extra=self._family_meta(families),
        )
        return fig, meta

    def _plot_boxwhisker_family(self, results):
        """Fig 3B–K per family: member dots + family-mean bar, CMIP6 box."""
        regions = results["regions"]
        families = self._families(results)
        cmip6_label = results.get("cmip6_label", "CMIP6")
        obs_label = obs_ref_label(self.config, "tasmax")
        nfam = len(families)
        ncols = 5
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(max(4, 0.9 * (nfam + 1) + 1.5) * ncols,
                                          3.2 * nrows), squeeze=False)

        stats: dict[str, dict] = {}
        for i, region in enumerate(regions):
            ax = axes[i // ncols][i % ncols]
            ticks, tick_labels = [], []
            fam_stats = {}
            for x, (family, members) in enumerate(families.items()):
                vals = np.array([region["model_trends"][m] for m in members],
                                dtype=float)
                for m, v in zip(members, vals):
                    ax.scatter(x, v, color=self.config.get_model_color(m),
                               s=22, zorder=4, edgecolors="k", linewidths=0.3)
                if np.isfinite(vals).any():
                    mean = float(np.nanmean(vals))
                    ax.hlines(mean, x - 0.28, x + 0.28,
                              color=self.config.get_model_color(members[0]),
                              lw=2.5, zorder=3)
                    fam_stats[family] = {
                        "mean": mean,
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                        "n_members": int(np.isfinite(vals).sum()),
                    }
                ticks.append(x); tick_labels.append(f"{family}\n({len(members)})")

            cmip6_vals = np.array(list(region.get("cmip6_trends", {}).values()),
                                  dtype=float)
            cmip6_vals = cmip6_vals[np.isfinite(cmip6_vals)]
            if cmip6_vals.size:
                ax.boxplot(cmip6_vals, vert=True, widths=0.5,
                           positions=[nfam], showfliers=False,
                           whis=(5, 95), patch_artist=True,
                           boxprops=dict(facecolor="lightgrey"))
                ax.scatter(np.full_like(cmip6_vals, nfam), cmip6_vals,
                           color="grey", s=10, alpha=0.6, zorder=3)
                ticks.append(nfam); tick_labels.append(cmip6_label)

            ci_lo, ci_hi = region["obs_ci"]
            ax.axhspan(ci_lo, ci_hi, color="tab:red", alpha=0.12, zorder=1)
            ax.axhline(region["obs_trend"], color="tab:red", lw=1.5, ls="--",
                       zorder=5, label=obs_label)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_xlim(-0.6, nfam + 0.6)
            ax.set_title(f"{chr(98 + i)}) {region['title']}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            if i % ncols == 0:
                ax.set_ylabel("Trend [°C/decade]", fontsize=8)
            stats[region["title"]] = {
                "obs_trend_per_decade": region["obs_trend"],
                "families": fam_stats,
            }

        for j in range(len(regions), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"{self.title} — model-family trends vs observed "
                     f"(per region)", y=1.0)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — model-family vs observed regional trends",
            figure_id=self._fig_id("heatwave_hotspots_boxwhisker"),
            models=results["models"] + results.get("cmip6_models", []),
            description=(
                "Per-region modelled tail-widening trends, one column per model "
                "family: each member as a dot in its own colour, the family mean "
                f"as a horizontal bar; the {cmip6_label} ensemble as a box "
                "(25–75 %, whiskers 5–95 %).  Observed trend "
                f"({obs_label}) as a red dashed line with its shaded 2.5–97.5 % "
                "bootstrap CI." + self._families_note(families)
            ),
            period=self.period,
            plot_type="distribution",
            summary_statistics=stats,
            extra=self._family_meta(families),
        )
        return fig, meta

    def _plot_pdf_family(self, results):
        """Fig 4 per family: family-mean densities + per-family CDF differences."""
        ref = results["ref_model"]
        families = self._families(results)
        mask = results["land_mask"] & (results["trend_lower"][ref] > 0)
        areas = xr.DataArray(results["areas"], dims=("lat", "lon"),
                             coords={"lat": results["lat"], "lon": results["lon"]})
        bins = np.linspace(-1.0, 1.0, 61)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dx = np.diff(bins)

        def _density(trend):
            v = trend.where(mask).values.ravel()
            w = areas.where(mask).values.ravel()
            ok = np.isfinite(v) & np.isfinite(w)
            if not ok.any() or w[ok].sum() <= 0:
                return None
            h, _ = np.histogram(v[ok], bins=bins, weights=w[ok], density=True)
            return h

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        obs_label = obs_ref_label(self.config, "tasmax")
        obs_hist = _density(results["trend_d"][ref])
        if obs_hist is not None:
            ax1.step(centers, obs_hist, where="mid", color="tab:red", lw=2.0,
                     label=obs_label, zorder=5)
        obs_cdf = None if obs_hist is None else np.cumsum(obs_hist * dx)

        stats = {}
        for family, members in families.items():
            color = self.config.get_model_color(members[0])
            hists = [h for h in (_density(results["trend_d"][m]) for m in members)
                     if h is not None]
            if not hists:
                continue
            for h in hists:
                ax1.step(centers, h, where="mid", color=color, lw=0.7, alpha=0.4)
            fam_hist = np.mean(hists, axis=0)
            label = family_label(family, len(hists), bold=False)
            ax1.step(centers, fam_hist, where="mid", color=color, lw=2.0,
                     label=label, zorder=4)
            if obs_cdf is not None:
                diff = obs_cdf - np.cumsum(fam_hist * dx)
                ax2.plot(centers, diff, color=color, lw=2.0, label=label)
                stats[family] = {"max_cdf_obs_minus_family": float(np.max(diff)),
                                 "min_cdf_obs_minus_family": float(np.min(diff))}

        ax1.set_yscale("log")
        ax1.set_xlabel("Trend in yearly P99−P87.5 of Tx [°C/decade]")
        ax1.set_ylabel("Area-weighted probability density")
        ax1.set_title("a) Distribution of land tail-widening trends")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color="grey", lw=0.6)
        ax2.axhline(0, color="grey", lw=0.6)
        ax2.set_xlabel("Trend [°C/decade]")
        ax2.set_ylabel("CDF(obs) − CDF(family mean)")
        ax2.set_title("b) Cumulative obs − model-family difference")
        ax2.grid(True, alpha=0.3)
        if stats:
            ax2.legend(fontsize=8)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — land tail-widening trend distribution "
                  f"(model families)",
            figure_id=self._fig_id("heatwave_hotspots_pdf"),
            models=[ref] + results["models"],
            description=(
                "Area-weighted probability density of land tail-widening trends "
                "(yearly P99−P87.5 of daily Tx) for the reanalysis reference and "
                "each model family (thick: mean of its members' densities; thin: "
                "members; log y-axis), with the cumulative obs − family "
                "difference per family.  A family underestimating the extreme "
                "tail gives a positive cumulative difference at high trend "
                "values." + self._families_note(families)
            ),
            period=self.period,
            plot_type="distribution",
            summary_statistics=stats,
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
            extra=self._family_meta(families),
        )
        return fig, meta

    def _plot_eerie_vs_cmip6_family(self, results):
        """Family-mean trend minus the CMIP6 MMM, one panel per family."""
        ref = results["ref_model"]
        families = self._families(results)
        lats, lons = results["lat"], results["lon"]
        keep = results["land_mask"] & (results["trend_lower"][ref] > 0)
        label = results["cmip6_label"]
        diffs = {
            f: (self._family_trend(results, m) - results["cmip6_mmm_trend"]).where(keep)
            for f, m in families.items()
        }
        fig = self._family_map_figure(
            [(f"Family mean − {label} MMM", diffs)],
            families, lats, lons, results["regions"],
            cblabels=[f"Family mean − {label} MMM tail-widening trend "
                      f"[°C/decade]"],
            suptitle=f"{self.title} — model-family means minus {label} MMM",
        )
        meta = self._build_metadata(
            title=f"{self.title} — model-family means minus {label} MMM",
            figure_id=self._fig_id("heatwave_hotspots_eerie_vs_cmip6"),
            models=results["models"] + results["cmip6_models"],
            description=(
                "Difference between each EERIE model family's mean tail-widening "
                f"trend and the coarse {label} MMM (°C/decade), one panel per "
                "family.  Red: family trend stronger than CMIP6; blue: weaker.  "
                "Land only; region with a negative P87.5 trend excluded."
                + self._families_note(families)
            ),
            period=self.period,
            plot_type="map",
            extra=self._family_meta(families),
        )
        return fig, meta

    # ── CMIP6 comparison maps ──────────────────────────────────────────

    def _render_map(
        self, field, lats, lons, *, cmap, vmax, title, cblabel, regions,
        greyed=None,
    ):
        """Render a single Robinson trend/difference map (shared layout)."""
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        mesh = self._draw_map_panel(ax, field, lats, lons, cmap=cmap,
                                    vmax=vmax, regions=regions, greyed=greyed)
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                          shrink=0.6, pad=0.05, extend="both")
        cb.set_label(cblabel)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _draw_map_panel(self, ax, field, lats, lons, *, cmap, vmax, regions,
                        greyed=None):
        """Draw one Robinson trend/difference panel on *ax*; returns the mesh."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

        lon_p = np.where(lons > 180, lons - 360, lons)
        order = np.argsort(lon_p)
        lon_s = lon_p[order]
        pc = ccrs.PlateCarree()

        if greyed is not None:
            grey = greyed.isel(lon=order).astype(float).where(
                greyed.isel(lon=order))
            ax.pcolormesh(lon_s, lats, grey.values, transform=pc, cmap="Greys",
                          vmin=0, vmax=1.5, shading="auto", zorder=1)
        mesh = ax.pcolormesh(
            lon_s, lats, field.isel(lon=order).values, transform=pc,
            cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", zorder=2,
        )
        for region in regions:
            self._draw_region_box(ax, region, pc)
        return mesh

    def _plot_cmip6_mmm_map(self, results):
        """CMIP6 MMM tail-widening trend map (same style as the ERA5 Fig 2A)."""
        ref = results["ref_model"]
        lats, lons = results["lat"], results["lon"]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0
        label = results["cmip6_label"]

        field = results["cmip6_mmm_trend"].where(land & summer_pos)
        greyed = land & ~summer_pos
        vmax = float(np.nanpercentile(np.abs(field.values), 98)) or 0.5
        fig = self._render_map(
            field, lats, lons, cmap="RdBu_r", vmax=vmax,
            title=(f"{self.title} — {label} MMM extreme-heat tail-widening "
                   f"({self.period[0]}–{self.period[1]})"),
            cblabel=(f"Trend in yearly P{_UPPER_Q:g}−P{_LOWER_Q:g} of daily "
                     f"Tx [°C/decade]"),
            regions=results["regions"], greyed=greyed,
        )
        meta = self._build_metadata(
            title=f"{self.title} — {label} MMM tail-widening trend map",
            figure_id="heatwave_hotspots_cmip6_mmm_map",
            models=results["cmip6_models"],
            description=(
                f"Multi-model-mean ({label}, {len(results['cmip6_models'])} "
                "models) linear trend in the yearly P99−P87.5 of daily Tx "
                "(°C/decade).  Grey: land where the 87.5th-percentile trend is "
                "negative."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_cmip6_discrepancy(self, results):
        """Map where ERA5's observed trend falls outside the CMIP6 envelope."""
        ref = results["ref_model"]
        lats, lons = results["lat"], results["lon"]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0
        label = results["cmip6_label"]
        obs_label = obs_ref_label(self.config, "tasmax")

        obs_trend = results["trend_d"][ref]
        env_max = results["cmip6_env_max"]
        env_min = results["cmip6_env_min"]
        disc = xr.where(obs_trend > env_max, obs_trend - env_max,
                        xr.where(obs_trend < env_min, obs_trend - env_min, 0.0))
        disc = disc.where(land & summer_pos)
        vmax = float(np.nanpercentile(np.abs(disc.values), 98)) or 0.5
        fig = self._render_map(
            disc, lats, lons, cmap="RdBu_r", vmax=vmax,
            title=(f"{self.title} — {obs_label} observed trend vs {label} "
                   f"envelope (red = {label} underestimation)"),
            cblabel=f"{obs_label} minus {label} envelope [°C/decade]",
            regions=results["regions"],
        )
        meta = self._build_metadata(
            title=f"{self.title} — {obs_label} vs {label} envelope discrepancy",
            figure_id="heatwave_hotspots_cmip6_discrepancy",
            models=results["cmip6_models"],
            description=(
                f"Difference between the {obs_label} observed tail-widening "
                f"trend and the {label} ensemble envelope: positive (red) where "
                f"the observed trend exceeds every {label} model ({label} "
                "underestimation), negative (blue) where it falls below.  Land "
                "only; grey-masked region (P87.5 trend < 0) excluded."
            ),
            period=self.period,
            plot_type="map",
            obs_dataset="ERA5_TMINMAX",
            obs_variable="tasmax",
        )
        return fig, meta

    def _plot_eerie_vs_cmip6(self, results):
        """Map of the EERIE ensemble-mean trend minus the CMIP6 MMM (added value)."""
        ref = results["ref_model"]
        models = results["models"]
        lats, lons = results["lat"], results["lon"]
        land = results["land_mask"]
        summer_pos = results["trend_lower"][ref] > 0
        label = results["cmip6_label"]

        eerie_mean = xr.concat(
            [results["trend_d"][m] for m in models], dim="member",
        ).mean("member")
        diff = (eerie_mean - results["cmip6_mmm_trend"]).where(land & summer_pos)
        vmax = float(np.nanpercentile(np.abs(diff.values), 98)) or 0.5
        fig = self._render_map(
            diff, lats, lons, cmap="RdBu_r", vmax=vmax,
            title=f"{self.title} — EERIE ensemble mean minus {label} MMM",
            cblabel=f"EERIE mean − {label} MMM tail-widening trend [°C/decade]",
            regions=results["regions"],
        )
        meta = self._build_metadata(
            title=f"{self.title} — EERIE mean minus {label} MMM",
            figure_id="heatwave_hotspots_eerie_vs_cmip6",
            models=models + results["cmip6_models"],
            description=(
                "Difference between the high-resolution EERIE ensemble-mean "
                f"tail-widening trend and the coarse {label} MMM (°C/decade).  "
                "Red: EERIE trend stronger than CMIP6; blue: weaker.  Land only; "
                "grey-masked region (P87.5 trend < 0) excluded."
            ),
            period=self.period,
            plot_type="map",
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
