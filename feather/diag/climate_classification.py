"""Köppen–Trewartha (KT14) climate classification diagnostic.

Classifies the monthly climatology of near-surface temperature (``tas``)
and precipitation (``pr``) into the 14 Köppen–Trewartha climate types
(see :mod:`feather.util.koeppen_trewartha`).  The classification is
restricted to **land** using the Berkeley Earth land mask and is computed
on a shared regular lat/lon grid (``nereus.resolution``, default 0.25°) so
that the per-type area fractions are directly comparable across sources.

Sources classified:

- every model in the config (model ``tas`` + ``pr``),
- **ERA5** (``t2m`` + ``tp``),
- **Berkeley Earth HR + MSWEP** (BE-HR temperature + MSWEP precipitation),
- the **EERIE ensemble mean and median** (per-cell mean/median of the model
  ``tas`` and ``pr`` climatologies, then classified),
- the **CMIP6 multi-model mean** (when ``cmip6.enabled``).

Outputs (all in ``{output_dir}/climate_classification/``):

- ``{source}_clim_{period}.nc`` — monthly ``tas`` (°C) and ``pr`` (mm/month)
  climatology per source (global + land-only), on the common grid, for ERA5,
  Berkeley Earth HR, MSWEP, every EERIE model, every CMIP6 model, the EERIE
  ensemble mean/median, and the CMIP6 MMM,
- ``{source}_kt_{period}.nc`` — integer ``kt_code`` field (land only) per
  classified source,
- a CSV of % land area per KT type per source,
- a multi-panel discrete classification map, an ensemble-comparison map, a
  grouped bar chart, and a summary table.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.util.koeppen_trewartha import (
    KT_CODES,
    KT_COLORS,
    KT_DESCRIPTIONS,
    KT_LABELS,
    area_percent_by_type,
    classify_kt,
)
from feather.util.spatial import compute_latlon_areas

logger = logging.getLogger(__name__)

# Well-known Levante Berkeley Earth file carrying a 0.25° ``land_mask``.
_BE_LAND_PATHS = [
    Path("/work/bm1344/AWI/OBS/berkeleyearth/Global_TAVG_Gridded_0p25deg.nc"),
    Path("/work/bm1344/AWI/OBS/berkeleyearth/Land_TMIN_Gridded_0p25deg.nc"),
]

_K_TO_C = 273.15
# kg m-2 s-1 → mm/month uses seconds-per-day (days_in_month applied per step)
_SEC_PER_DAY = 86400.0

# Source labels
_ERA5 = "ERA5"
_BE_HR = "Berkeley Earth HR"
_MSWEP = "MSWEP"
_BE_MSWEP = "BE-HR + MSWEP"
_ENS_MEAN = "EERIE Ensemble Mean"
_ENS_MEDIAN = "EERIE Ensemble Median"
_CMIP6_MMM = "CMIP6 MMM"

# Regional target-grid bounds (lon in -180..180 when min < 0).
_REGIONS: dict[str, dict] = {
    "africa": {"lon_bounds": (-26.0, 60.0), "lat_bounds": (-48.0, 44.0)},
}


@register
class KTClimateClassification(DiagnosticBase):
    """Köppen–Trewartha (KT14) land climate classification."""

    name = "climate_classification"
    title = "Köppen–Trewartha Climate Classification"
    domain = "sfc"
    variables = ["tas", "pr"]
    group = "climate_classification"

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
        self._res = float(self.config.nereus.get("resolution", 0.25))
        self._ir = float(self.config.nereus.get("influence_radius", 80_000.0))
        # Region → target-grid bounds.  Global default keeps the historical
        # 0..360 convention; a named region uses its (possibly negative) bounds.
        self._region = str(self.config.project.get("region", "")).lower()
        bounds = _REGIONS.get(self._region)
        if bounds:
            self._lon_bounds = tuple(bounds["lon_bounds"])
            self._lat_bounds = tuple(bounds["lat_bounds"])
        else:
            self._lon_bounds = (0.0, 360.0)
            self._lat_bounds = (-90.0, 90.0)
        # Common target grid (populated on first regrid)
        self._tlat: np.ndarray | None = None
        self._tlon: np.ndarray | None = None
        self._land_mask: xr.DataArray | None = None  # cached once grid is known
        self._reuse_cache: bool = True  # reuse saved NetCDFs when re-plotting

    # ── Paths ─────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-source KT NetCDF files (outside figures tree)."""
        return Path(self.config.output_dir) / "climate_classification"

    @staticmethod
    def _safe(source: str) -> str:
        s = source.replace("/", "_").replace(" ", "_").replace("+", "")
        return "_".join(filter(None, s.split("_")))

    def _nc_path(self, source: str) -> Path:
        start, end = self.period
        return self.nc_dir / f"{self._safe(source)}_kt_{start}_{end}.nc"

    def _clim_path(self, source: str) -> Path:
        start, end = self.period
        return self.nc_dir / f"{self._safe(source)}_clim_{start}_{end}.nc"

    def _csv_path(self) -> Path:
        start, end = self.period
        return self.nc_dir / f"kt_area_percent_{start}_{end}.csv"

    def _be_land_file(self) -> Path | None:
        cfg = self.config.obs_datasets.get("BERKELEY_EARTH_HR", {})
        path = cfg.get("path", "")
        fname = cfg.get("variables", {}).get("temperature", "")
        if path and fname:
            p = Path(path.replace("{obs_root}", self.config.obs_root)) / fname
            if p.exists():
                return p
        for p in _BE_LAND_PATHS:
            if p.exists():
                return p
        return None

    # ── Orchestration ─────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig_ids = [
            "kt_classification_maps",
            "kt_classification_ensemble",
            "kt_area_bar",
            "kt_area_table",
        ]
        if skip_existing and all(self._figure_exists(f) for f in fig_ids):
            logger.info("  All KT figures exist — skipping")
            return saved

        self._reuse_cache = skip_existing
        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Collect per-source climatologies, classify, persist, return results.

        Returns
        -------
        dict with keys:

        - ``codes``     : dict[source → DataArray(lat, lon)] (land-only KT code)
        - ``area_pct``  : dict[source → dict[label → percent]]
        - ``models``    : list[str] — EERIE model sources only
        - ``sources``   : list[str] — classified sources
        - ``ens_mean``/``ens_median`` : DataArray | None — ensemble KT codes
        - ``lat``/``lon``: common grid coordinates
        """
        # Fast path: reuse the saved classification NetCDFs (e.g. when only
        # re-plotting after a figure tweak) instead of recomputing from data.
        if self._reuse_cache:
            cached = self._load_cached_results()
            if cached is not None:
                logger.info(
                    "  Reusing %d cached classification NetCDF(s) — skipping "
                    "recomputation (delete %s to force a full recompute)",
                    len(cached["codes"]), self.nc_dir,
                )
                return cached

        # Regridded monthly climatology per source: tas in °C, pr in cm/month.
        clims_tas: dict[str, xr.DataArray] = {}
        clims_pr: dict[str, xr.DataArray] = {}

        # ── EERIE models (tas + pr) ────────────────────────────────────
        model_sources: list[str] = []
        for model in self.config.models:
            try:
                tmon, pmon = self._model_clim(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue
            clims_tas[model] = tmon
            clims_pr[model] = pmon
            model_sources.append(model)

        # ── ERA5 (tas + pr) ────────────────────────────────────────────
        try:
            tas = self._load_obs_var("tas", self.period)
            pr = self._load_obs_var("pr", self.period)
            clims_tas[_ERA5] = self._regrid_monthly(self._clim_tas(tas), self._ir)
            clims_pr[_ERA5] = self._regrid_monthly(self._clim_pr(pr), self._ir)
        except Exception as exc:
            logger.warning("  ERA5: skipping — %s", exc)

        # ── Berkeley Earth HR (tas only) ───────────────────────────────
        try:
            clims_tas[_BE_HR] = self._regrid_monthly(
                self._be_hr_monthly_clim(), self._ir,
            )
        except Exception as exc:
            logger.warning("  Berkeley Earth HR: skipping — %s", exc)

        # ── MSWEP (pr only) ────────────────────────────────────────────
        try:
            pr = self.obs_loader.load_mswep(self.period)
            clims_pr[_MSWEP] = self._regrid_monthly(self._clim_pr(pr), self._ir)
        except Exception as exc:
            logger.warning("  MSWEP: skipping — %s", exc)

        if self._tlat is None:
            logger.warning("KTClimateClassification: no data classified")
            return {"codes": {}, "area_pct": {}, "models": [], "sources": [],
                    "ens_mean": None, "ens_median": None, "lat": None, "lon": None}

        self.nc_dir.mkdir(parents=True, exist_ok=True)

        # ── CMIP6 members (tas + pr) — save each, accumulate MMM ────────
        cmip6_sum_t = cmip6_sum_p = None
        cmip6_n = 0
        if self.cmip6_enabled:
            cmip6_sum_t, cmip6_sum_p, cmip6_n = self._collect_cmip6_clims()

        # ── Derived climatologies ──────────────────────────────────────
        if len(model_sources) >= 2:
            stack_t = xr.concat([clims_tas[m] for m in model_sources], dim="member")
            stack_p = xr.concat([clims_pr[m] for m in model_sources], dim="member")
            clims_tas[_ENS_MEAN] = stack_t.mean("member")
            clims_pr[_ENS_MEAN] = stack_p.mean("member")
            clims_tas[_ENS_MEDIAN] = stack_t.median("member")
            clims_pr[_ENS_MEDIAN] = stack_p.median("member")
        if cmip6_n > 0:
            clims_tas[_CMIP6_MMM] = cmip6_sum_t / cmip6_n
            clims_pr[_CMIP6_MMM] = cmip6_sum_p / cmip6_n

        # ── Classification (maps/table set) ────────────────────────────
        codes: dict[str, xr.DataArray] = {}
        for src in model_sources:
            codes[src] = self._classify_from(clims_tas[src], clims_pr[src])
        if _ERA5 in clims_tas and _ERA5 in clims_pr:
            codes[_ERA5] = self._classify_from(clims_tas[_ERA5], clims_pr[_ERA5])
        if _BE_HR in clims_tas and _MSWEP in clims_pr:
            codes[_BE_MSWEP] = self._classify_from(clims_tas[_BE_HR], clims_pr[_MSWEP])
        if _CMIP6_MMM in clims_tas:
            codes[_CMIP6_MMM] = self._classify_from(
                clims_tas[_CMIP6_MMM], clims_pr[_CMIP6_MMM],
            )

        # Ensemble KT codes (classified from the ensemble climatology)
        ens_mean_code = ens_median_code = None
        if _ENS_MEAN in clims_tas:
            ens_mean_code = self._classify_from(
                clims_tas[_ENS_MEAN], clims_pr[_ENS_MEAN],
            )
            ens_median_code = self._classify_from(
                clims_tas[_ENS_MEDIAN], clims_pr[_ENS_MEDIAN],
            )

        # ── Area percentages ───────────────────────────────────────────
        area_np = compute_latlon_areas(self._tlat, self._tlon)
        area_da = xr.DataArray(
            area_np, dims=("lat", "lon"),
            coords={"lat": self._tlat, "lon": self._tlon},
        )
        area_pct = {
            src: area_percent_by_type(code, area_da) for src, code in codes.items()
        }
        # Include the EERIE ensemble mean/median in the bar chart and table
        # (they are shown on the ensemble map, not the per-source maps).
        if ens_mean_code is not None:
            area_pct[_ENS_MEAN] = area_percent_by_type(ens_mean_code, area_da)
            area_pct[_ENS_MEDIAN] = area_percent_by_type(ens_median_code, area_da)

        # ── Persistence ────────────────────────────────────────────────
        for src in set(clims_tas) | set(clims_pr):
            self._save_clim_nc(src, clims_tas.get(src), clims_pr.get(src))
        for src, code in codes.items():
            self._save_nc(src, code)
        if ens_mean_code is not None:
            self._save_nc(_ENS_MEAN, ens_mean_code)
            self._save_nc(_ENS_MEDIAN, ens_median_code)
        self._write_csv(area_pct)

        return {
            "codes": codes,
            "area_pct": area_pct,
            "models": model_sources,
            "sources": list(codes.keys()),
            "ens_mean": ens_mean_code,
            "ens_median": ens_median_code,
            "lat": self._tlat,
            "lon": self._tlon,
        }

    def _load_cached_results(self) -> dict[str, Any] | None:
        """Reconstruct results from saved ``{source}_kt`` NetCDFs + the CSV.

        Returns None when the cache is absent or incomplete, so the caller
        falls back to a full recomputation.
        """
        csv = self._csv_path()
        if not csv.exists():
            return None

        codes: dict[str, xr.DataArray] = {}
        for src in list(self.config.models) + [_ERA5, _BE_MSWEP, _CMIP6_MMM]:
            p = self._nc_path(src)
            if p.exists():
                codes[src] = xr.open_dataset(p)["kt_code"]
        if not codes:
            return None

        df = pd.read_csv(csv, index_col=0)
        area_pct = {
            col: {lbl: float(df.loc[lbl, col]) for lbl in KT_LABELS if lbl in df.index}
            for col in df.columns
        }

        ens_mean = ens_median = None
        pm, pmed = self._nc_path(_ENS_MEAN), self._nc_path(_ENS_MEDIAN)
        if pm.exists() and pmed.exists():
            ens_mean = xr.open_dataset(pm)["kt_code"]
            ens_median = xr.open_dataset(pmed)["kt_code"]

        ref = next(iter(codes.values()))
        self._tlat = np.asarray(ref["lat"].values)
        self._tlon = np.asarray(ref["lon"].values)

        return {
            "codes": codes,
            "area_pct": area_pct,
            "models": [m for m in self.config.models if m in codes],
            "sources": list(codes.keys()),
            "ens_mean": ens_mean,
            "ens_median": ens_median,
            "lat": self._tlat,
            "lon": self._tlon,
        }

    # ── Per-source climatology collection ──────────────────────────────

    def _model_clim(self, model: str) -> tuple[xr.DataArray, xr.DataArray]:
        """Regridded monthly tas (°C) + pr (cm/month) climatology for a model."""
        logger.info("  %s: climatology", model)
        tas = self._load_model_var(model, "tas", period=self.period)
        pr = self._load_model_var(model, "pr", period=self.period)
        tmon = self._regrid_monthly(self._clim_tas(tas), self._ir)
        pmon = self._regrid_monthly(self._clim_pr(pr), self._ir)
        return tmon, pmon

    def _collect_cmip6_clims(self):
        """Save each CMIP6 member's climatology; return (sum_tas, sum_pr, n).

        The running sums (in classification units) feed the CMIP6 MMM, while
        each member's climatology is persisted to NetCDF as it is computed so
        member fields need not all be held in memory.
        """
        from feather.data.variables import get_var

        ir = max(self._ir, 250_000.0)  # coarse-grid floor for CMIP6
        tas_info = get_var("tas")
        pr_info = get_var("pr")

        sum_t = sum_p = None
        n = 0
        for model in self.cmip6_loader.models:
            tas = self.cmip6_loader.load_var(
                tas_info.cmip6_variable, model,
                table=tas_info.cmip6_table or None,
                period=self.period, time_mean=False,
            )
            pr = self.cmip6_loader.load_var(
                pr_info.cmip6_variable, model,
                table=pr_info.cmip6_table or None,
                period=self.period, time_mean=False,
            )
            if tas is None or pr is None:
                continue
            tmon = self._regrid_monthly(self._clim_tas(tas), ir)
            pmon = self._regrid_monthly(self._clim_pr(pr), ir)
            self._save_clim_nc(f"CMIP6 {model}", tmon, pmon)
            sum_t = tmon if sum_t is None else sum_t + tmon
            sum_p = pmon if sum_p is None else sum_p + pmon
            n += 1

        if n == 0:
            logger.info("  CMIP6: no members available")
        return sum_t, sum_p, n

    def _classify_from(
        self, tmon: xr.DataArray, pmon: xr.DataArray,
    ) -> xr.DataArray:
        """Classify a (tas °C, pr cm) climatology pair, return land-only code."""
        lat_da = xr.DataArray(self._tlat, dims="lat", coords={"lat": self._tlat})
        code = classify_kt(tmon, pmon, lat_da)
        return self._apply_land_mask(code)

    # ── Climatology helpers ────────────────────────────────────────────

    @staticmethod
    def _clim_tas(da: xr.DataArray) -> xr.DataArray:
        """Monthly climatology of temperature in °C, dims (month, lat, lon)."""
        if "time" not in da.dims:
            raise ValueError("tas has no time dimension")
        clim = da.groupby("time.month").mean("time")
        # Auto-detect Kelvin (BE-HR is already °C and handled separately).
        if float(clim.max()) > 100.0:
            clim = clim - _K_TO_C
        return clim.compute()

    @staticmethod
    def _clim_pr(da: xr.DataArray) -> xr.DataArray:
        """Monthly climatology of precipitation in cm/month.

        Accepts ``pr`` in kg m-2 s-1 (CF standard, used by ERA5/MSWEP/models
        in this framework) and converts to cm/month using calendar-aware
        month lengths.
        """
        if "time" not in da.dims:
            raise ValueError("pr has no time dimension")
        # kg m-2 s-1 → mm/month (per timestep) → cm
        pr_mm = da * (_SEC_PER_DAY * da["time"].dt.days_in_month)
        clim_mm = pr_mm.groupby("time.month").mean("time")
        return (clim_mm / 10.0).compute()

    def _be_hr_monthly_clim(self) -> xr.DataArray:
        """Berkeley Earth HR monthly temperature climatology (°C).

        Reconstructs absolute monthly temperature as ``anomaly +
        climatology[month]`` over the analysis period, returns the 12-month
        climatology with dims ``(month, lat, lon)`` on the native BE grid.
        """
        path = self._be_land_file()
        if path is None:
            raise FileNotFoundError("Berkeley Earth HR file not found")
        ds = xr.open_dataset(path, chunks="auto")

        dec = ds["time"].values
        years = dec.astype(int)
        months = np.clip(np.floor((dec - years) * 12).astype(int) + 1, 1, 12)
        times = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)])
        ds = ds.assign_coords(time=times)

        start, end = self.period
        anom = ds["temperature"].sel(time=slice(start, end))
        clim = ds["climatology"]                       # (month_number, lat, lon)
        midx = anom.time.dt.month.values - 1
        clim_matched = clim.values[midx]
        abs_t = anom + xr.DataArray(clim_matched, dims=anom.dims, coords=anom.coords)

        tmon = abs_t.groupby("time.month").mean("time")
        rename = {}
        if "latitude" in tmon.dims:
            rename["latitude"] = "lat"
        if "longitude" in tmon.dims:
            rename["longitude"] = "lon"
        if rename:
            tmon = tmon.rename(rename)
        return tmon.compute()

    # ── Regridding / masking ───────────────────────────────────────────

    def _regrid_monthly(self, clim: xr.DataArray, ir: float) -> xr.DataArray:
        """Regrid a (month, lat, lon) climatology to the common grid.

        All sources are regridded with the same ``resolution`` and
        ``lon_bounds`` so they land on an identical target grid.
        """
        import nereus as nr

        lat_name = "lat" if "lat" in clim.coords else "latitude"
        lon_name = "lon" if "lon" in clim.coords else "longitude"
        latv = np.asarray(clim[lat_name].values)
        lonv = self._to_target_lon_convention(np.asarray(clim[lon_name].values))

        if lonv.ndim == 1:
            # Regular grid: sort columns by longitude (so the source matches
            # the target lon convention and no hemisphere is dropped).
            order = np.argsort(lonv)
            lonv = lonv[order]
            clim = clim.isel({lon_name: order})
            lon2d, lat2d = np.meshgrid(lonv, latv)
        else:
            # 2-D (rotated-pole) grid: pass coordinate arrays as-is.
            lon2d, lat2d = lonv, latv

        interp = None
        out = []
        for i in range(clim.sizes["month"]):
            field = np.asarray(clim.isel(month=i).values).ravel()
            if interp is None:
                _, interp = nr.regrid(
                    field, lon=lon2d, lat=lat2d,
                    resolution=self._res, influence_radius=ir,
                    lon_bounds=self._lon_bounds, lat_bounds=self._lat_bounds,
                    as_xarray=True,
                )
                if self._tlat is None:
                    self._tlat = np.asarray(interp.target_lat[:, 0])
                    self._tlon = np.asarray(interp.target_lon[0, :])
            out.append(np.asarray(interp(field)))

        arr = np.stack(out, axis=0)
        return xr.DataArray(
            arr, dims=("month", "lat", "lon"),
            coords={
                "month": np.asarray(clim["month"].values),
                "lat": self._tlat, "lon": self._tlon,
            },
        )

    def _to_target_lon_convention(self, lon: np.ndarray) -> np.ndarray:
        """Convert longitudes to the target grid's convention.

        Regional grids use -180..180 (so a negative ``lon_bounds`` like Africa
        maps the western part correctly); the global grid keeps 0..360.
        """
        if self._lon_bounds[0] < 0:
            return ((lon + 180.0) % 360.0) - 180.0
        return np.mod(lon, 360.0)

    def _apply_land_mask(self, code: xr.DataArray) -> xr.DataArray:
        mask = self._get_land_mask()
        if mask is not None:
            code = code.where(mask)
        return code

    def _get_land_mask(self) -> xr.DataArray | None:
        """Berkeley Earth land mask on the common grid (bool), cached."""
        if self._land_mask is not None:
            return self._land_mask
        path = self._be_land_file()
        if path is None:
            return None
        try:
            ds = xr.open_dataset(path)
            mask = ds["land_mask"]
            rename = {}
            if "latitude" in mask.dims:
                rename["latitude"] = "lat"
            if "longitude" in mask.dims:
                rename["longitude"] = "lon"
            if rename:
                mask = mask.rename(rename)
            # Match the target grid's longitude convention before interpolation.
            mask = mask.assign_coords(
                lon=self._to_target_lon_convention(np.asarray(mask.lon.values)),
            ).sortby("lon")
            mask_i = mask.interp(
                lat=xr.DataArray(self._tlat, dims="lat"),
                lon=xr.DataArray(self._tlon, dims="lon"),
                method="nearest", kwargs={"fill_value": 0.0},
            )
            self._land_mask = mask_i > 0.5
            return self._land_mask
        except Exception as exc:
            logger.warning("  Could not load BE land mask: %s", exc)
            return None

    # ── Persistence ────────────────────────────────────────────────────

    def _save_nc(self, source: str, code: xr.DataArray) -> None:
        start, end = self.period
        out = code.astype("float32").rename("kt_code")
        ds = xr.Dataset(
            {"kt_code": out},
            attrs={
                "Conventions": "CF-1.8",
                "title": f"Köppen–Trewartha classification — {source} ({start}–{end})",
                "institution": "Feather climate evaluation framework",
                "source": "feather/diag/climate_classification.py",
                "source_dataset": source,
                "period_start": start,
                "period_end": end,
                "land_only": "True — ocean pixels are NaN",
                "kt_mapping_json": json.dumps(KT_CODES),
                "reference": "Trewartha & Horn (1980); Belda et al. (2014)",
            },
        )
        path = self._nc_path(source)
        ds.to_netcdf(path)
        logger.info("  Saved KT NetCDF: %s", path)

    def _save_clim_nc(
        self,
        source: str,
        tas: xr.DataArray | None,
        pr: xr.DataArray | None,
    ) -> None:
        """Persist monthly tas (°C) / pr (mm/month) climatology for one source.

        Saves both the full global field and a land-only version, on the
        common grid.  ``tas`` is expected in °C and ``pr`` in cm/month
        (classification units); pr is converted to mm/month on write.
        """
        start, end = self.period
        mask = self._get_land_mask()
        data: dict[str, xr.DataArray] = {}

        if tas is not None:
            t = tas.astype("float32")
            data["tas_clim"] = t.assign_attrs(
                long_name="Monthly climatology of 2m air temperature", units="degC",
            )
            if mask is not None:
                data["tas_clim_land"] = t.where(mask).assign_attrs(
                    long_name="Monthly tas climatology (land only)", units="degC",
                )
        if pr is not None:
            p = (pr * 10.0).astype("float32")  # cm/month → mm/month
            data["pr_clim"] = p.assign_attrs(
                long_name="Monthly climatology of precipitation", units="mm/month",
            )
            if mask is not None:
                data["pr_clim_land"] = p.where(mask).assign_attrs(
                    long_name="Monthly pr climatology (land only)", units="mm/month",
                )
        if not data:
            return

        ds = xr.Dataset(
            data,
            attrs={
                "Conventions": "CF-1.8",
                "title": f"Monthly climatology — {source} ({start}–{end})",
                "institution": "Feather climate evaluation framework",
                "source": "feather/diag/climate_classification.py",
                "source_dataset": source,
                "period_start": start,
                "period_end": end,
                "grid": "common regular lat/lon (nereus.resolution)",
                "note": (
                    "tas_clim/pr_clim are global; *_land are masked to land "
                    "(Berkeley Earth land mask)."
                ),
            },
        )
        path = self._clim_path(source)
        ds.to_netcdf(path)
        logger.info("  Saved climatology NetCDF: %s", path)

    def _write_csv(self, area_pct: dict[str, dict[str, float]]) -> None:
        df = pd.DataFrame(area_pct).reindex(KT_LABELS)
        df.index.name = "kt_type"
        df.to_csv(self._csv_path())
        logger.info("  Saved KT area-percent CSV: %s", self._csv_path())

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        if not results["sources"]:
            logger.warning("KTClimateClassification: no sources — no figures")
            return figs
        figs.append(self._plot_maps(results))
        ensemble = self._plot_ensemble_maps(results)
        if ensemble is not None:
            figs.append(ensemble)
        figs.append(self._plot_bar(results))
        figs.append(self._plot_table(results))
        return figs

    def _ordered_sources(self, results: dict) -> list[str]:
        """Panel/source order: observations, then EERIE models, then CMIP6 MMM."""
        codes = results["codes"]
        order = [s for s in (_ERA5, _BE_MSWEP) if s in codes]
        order += [m for m in results["models"] if m in codes]
        if _CMIP6_MMM in codes:
            order.append(_CMIP6_MMM)
        order += [s for s in codes if s not in order]   # any stragglers
        return order

    def _table_sources(self, results: dict) -> list[str]:
        """Source order for the bar chart / table: like the maps, plus the
        EERIE ensemble mean/median inserted ahead of the CMIP6 MMM."""
        order = self._ordered_sources(results)
        pct = results["area_pct"]
        extra = [s for s in (_ENS_MEAN, _ENS_MEDIAN) if s in pct]
        if not extra:
            return order
        if _CMIP6_MMM in order:
            i = order.index(_CMIP6_MMM)
            return order[:i] + extra + order[i:]
        return order + extra

    @staticmethod
    def _kt_cmap_norm():
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap([KT_COLORS[lbl] for lbl in KT_LABELS])
        cmap.set_bad("0.85")
        norm = BoundaryNorm(np.arange(0.5, 15.5, 1.0), cmap.N)
        return cmap, norm

    def _render_kt_maps(
        self,
        panels: list[tuple[str, xr.DataArray]],
        lon: np.ndarray,
        lat: np.ndarray,
        suptitle: str,
    ) -> plt.Figure:
        """Render labelled KT-code panels with a shared discrete colorbar."""
        import math

        import cartopy.crs as ccrs

        cmap, norm = self._kt_cmap_norm()
        # Remap longitudes from 0..360 to -180..180 (and reorder columns) so
        # that pcolormesh on a Robinson axis centred at 0° renders the full
        # globe — a 0..360 array otherwise drops the eastern hemisphere.
        lon = np.asarray(lon)
        lon_plot = np.where(lon > 180.0, lon - 360.0, lon)
        col_order = np.argsort(lon_plot)
        lon_plot = lon_plot[col_order]

        n = len(panels)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        # Zoom to the region when one is configured; else global Robinson.
        regional = bool(self._region)
        if regional:
            proj = ccrs.PlateCarree()
            extent = [self._lon_bounds[0], self._lon_bounds[1],
                      self._lat_bounds[0], self._lat_bounds[1]]
            panel_w, panel_h = 4.6, 4.6
        else:
            proj = ccrs.Robinson(central_longitude=0)
            extent = None
            panel_w, panel_h = 6.5, 3.6
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
            subplot_kw={"projection": proj},
        )
        axes = np.atleast_1d(axes).ravel()

        mesh = None
        for ax, (label, code) in zip(axes, panels):
            mesh = ax.pcolormesh(
                lon_plot, lat, code.values[:, col_order],
                transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                shading="auto",
            )
            ax.coastlines(linewidth=0.4)
            if regional:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            else:
                ax.set_global()
            ax.set_title(label, fontsize=10)
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(suptitle, fontsize=13)
        cbar = fig.colorbar(
            mesh, ax=axes.tolist(), orientation="horizontal",
            fraction=0.04, pad=0.04, ticks=np.arange(1, 15),
        )
        cbar.ax.set_xticklabels(KT_LABELS, fontsize=8)
        return fig

    def _plot_maps(self, results: dict) -> tuple[plt.Figure, dict]:
        """Multi-panel discrete KT classification maps (one panel per source)."""
        sources = self._ordered_sources(results)
        panels = [(s, results["codes"][s]) for s in sources]
        fig = self._render_kt_maps(
            panels, results["lon"], results["lat"],
            f"{self.title} (land only, {self.period[0]}–{self.period[1]})",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Maps",
            figure_id="kt_classification_maps",
            models=results["models"],
            description=(
                "Köppen–Trewartha (KT14) climate type per grid cell, land only, "
                "on a common 0.25° grid. Panels are ordered ERA5, Berkeley Earth "
                "HR + MSWEP, the EERIE models, then the CMIP6 multi-model mean."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_ensemble_maps(self, results: dict) -> tuple[plt.Figure, dict] | None:
        """Observations vs EERIE ensemble mean/median vs CMIP6 MMM (5 panels)."""
        if results.get("ens_mean") is None:
            return None
        codes = results["codes"]
        panels: list[tuple[str, xr.DataArray]] = []
        for s in (_ERA5, _BE_MSWEP):
            if s in codes:
                panels.append((s, codes[s]))
        panels.append((_ENS_MEAN, results["ens_mean"]))
        panels.append((_ENS_MEDIAN, results["ens_median"]))
        if _CMIP6_MMM in codes:
            panels.append((_CMIP6_MMM, codes[_CMIP6_MMM]))

        fig = self._render_kt_maps(
            panels, results["lon"], results["lat"],
            f"{self.title} — Observations vs EERIE Ensemble "
            f"({self.period[0]}–{self.period[1]})",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Ensemble Comparison",
            figure_id="kt_classification_ensemble",
            models=results["models"],
            description=(
                "Köppen–Trewartha classification: ERA5 and Berkeley Earth HR + "
                "MSWEP observations, the EERIE ensemble mean and median "
                "(classified from the per-cell mean/median of model tas and pr "
                "climatologies), and the CMIP6 multi-model mean. Land only, "
                "common 0.25° grid."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_bar(self, results: dict) -> tuple[plt.Figure, dict]:
        """Grouped bar chart of % land area per KT type per source."""
        sources = self._table_sources(results)
        area_pct = results["area_pct"]
        x = np.arange(len(KT_LABELS))
        width = 0.8 / max(len(sources), 1)

        fig, ax = plt.subplots(figsize=(max(12, len(KT_LABELS)), 5))
        for i, src in enumerate(sources):
            vals = [area_pct[src][lbl] for lbl in KT_LABELS]
            color = self.config.get_model_color(src) if src in results["models"] else None
            ax.bar(x + i * width, vals, width, label=src, color=color)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(KT_LABELS)
        ax.set_ylabel("% of land area")
        ax.set_xlabel("Köppen–Trewartha type")
        ax.set_title(f"{self.title} — Land Area Fraction by Type")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — Area Fraction Bar Chart",
            figure_id="kt_area_bar",
            models=results["models"],
            description=(
                "Percentage of classified land area in each Köppen–Trewartha "
                "type, grouped by type with one bar per source."
            ),
            period=self.period,
            plot_type="bar",
        )
        return fig, meta

    def _plot_table(self, results: dict) -> tuple[plt.Figure, dict]:
        """Heatmap-style summary table of % land area per type per source."""
        sources = self._table_sources(results)
        area_pct = results["area_pct"]
        data = np.array(
            [[area_pct[src][lbl] for lbl in KT_LABELS] for src in sources]
        )

        fig, ax = plt.subplots(
            figsize=(1.0 * len(KT_LABELS) + 3, 0.5 * len(sources) + 2)
        )
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0)
        ax.set_xticks(np.arange(len(KT_LABELS)))
        ax.set_xticklabels(
            [f"{lbl}\n{KT_DESCRIPTIONS[lbl]}" for lbl in KT_LABELS],
            fontsize=7, rotation=90,
        )
        ax.set_yticks(np.arange(len(sources)))
        ax.set_yticklabels(sources, fontsize=8)
        for i in range(len(sources)):
            for j in range(len(KT_LABELS)):
                v = data[i, j]
                ax.text(
                    j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if v > data.max() * 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="% of land area")
        ax.set_title(f"{self.title} — % Land Area per Type ({self.period[0]}–{self.period[1]})")
        fig.tight_layout()

        meta = self._build_metadata(
            title=f"{self.title} — Area Summary Table",
            figure_id="kt_area_table",
            models=results["models"],
            description=(
                "Summary table: percentage of classified land area in each "
                "Köppen–Trewartha type for every source (each source sums to "
                "100 %). Also written to kt_area_percent CSV."
            ),
            period=self.period,
            plot_type="table",
            extra={"area_percent": area_pct},
        )
        return fig, meta
