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
- **CMIP6 multi-model mean** (when ``cmip6.enabled``).

Outputs:

- per-source NetCDF of the integer ``kt_code`` field (land only), written
  to ``{output_dir}/climate_classification/`` for later regional analysis,
- a CSV of % land area per KT type per source,
- a multi-panel discrete classification **map** figure,
- a grouped **bar chart** of % land area per KT type,
- a **summary table** (heatmap) of % land area per type per source.
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
# Display order for source panels: models first, then obs, then CMIP6 MMM.
_ERA5 = "ERA5"
_BE_MSWEP = "BE-HR + MSWEP"
_CMIP6_MMM = "CMIP6 MMM"


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
        # Common target grid (populated on first regrid)
        self._tlat: np.ndarray | None = None
        self._tlon: np.ndarray | None = None

    # ── Paths ─────────────────────────────────────────────────────────

    @property
    def nc_dir(self) -> Path:
        """Directory for per-source KT NetCDF files (outside figures tree)."""
        return Path(self.config.output_dir) / "climate_classification"

    def _nc_path(self, source: str) -> Path:
        start, end = self.period
        safe = source.replace("/", "_").replace(" ", "_").replace("+", "")
        safe = "_".join(filter(None, safe.split("_")))
        return self.nc_dir / f"{safe}_kt_{start}_{end}.nc"

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
            "kt_area_bar",
            "kt_area_table",
        ]
        if skip_existing and all(self._figure_exists(f) for f in fig_ids):
            logger.info("  All KT figures exist — skipping")
            return saved

        results = self.compute()
        for fig, meta in self.plot(results):
            saved.append(self._save(fig, meta, meta["figure_id"]))
            plt.close(fig)

        logger.info("Diagnostic %s complete — %d figure(s)", self.name, len(saved))
        return saved

    # ── Compute ────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Classify every source, save NetCDF + CSV, return results dict.

        Returns
        -------
        dict with keys:

        - ``codes``     : ordered dict[source → DataArray(lat, lon)] (land only)
        - ``area_pct``  : dict[source → dict[label → percent]]
        - ``models``    : list[str] — model sources only
        - ``sources``   : list[str] — full panel order
        - ``lat``/``lon``: common grid coordinates
        """
        codes: dict[str, xr.DataArray] = {}

        # ── Models ─────────────────────────────────────────────────────
        for model in self.config.models:
            try:
                code = self._classify_model(model)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                logger.warning("  %s: skipping — %s", model, exc)
                continue
            codes[model] = code

        model_sources = list(codes.keys())

        # ── ERA5 ───────────────────────────────────────────────────────
        try:
            codes[_ERA5] = self._classify_era5()
        except Exception as exc:  # obs failures shouldn't kill the diagnostic
            logger.warning("  ERA5: skipping — %s", exc)

        # ── Berkeley Earth HR + MSWEP ──────────────────────────────────
        try:
            codes[_BE_MSWEP] = self._classify_be_mswep()
        except Exception as exc:
            logger.warning("  BE-HR + MSWEP: skipping — %s", exc)

        # ── CMIP6 multi-model mean ─────────────────────────────────────
        if self.cmip6_enabled:
            try:
                code = self._classify_cmip6_mmm()
                if code is not None:
                    codes[_CMIP6_MMM] = code
            except Exception as exc:
                logger.warning("  CMIP6 MMM: skipping — %s", exc)

        if self._tlat is None:
            logger.warning("KTClimateClassification: no data classified")
            return {"codes": {}, "area_pct": {}, "models": [],
                    "sources": [], "lat": None, "lon": None}

        # ── Area percentages + persistence ─────────────────────────────
        area_np = compute_latlon_areas(self._tlat, self._tlon)
        area_da = xr.DataArray(
            area_np, dims=("lat", "lon"),
            coords={"lat": self._tlat, "lon": self._tlon},
        )
        area_pct = {
            src: area_percent_by_type(code, area_da)
            for src, code in codes.items()
        }

        self.nc_dir.mkdir(parents=True, exist_ok=True)
        for src, code in codes.items():
            self._save_nc(src, code)
        self._write_csv(area_pct)

        return {
            "codes": codes,
            "area_pct": area_pct,
            "models": model_sources,
            "sources": list(codes.keys()),
            "lat": self._tlat,
            "lon": self._tlon,
        }

    # ── Per-source classification ──────────────────────────────────────

    def _classify_model(self, model: str) -> xr.DataArray:
        """Classify one model from its monthly tas + pr."""
        logger.info("  %s: classifying", model)
        tas = self._load_model_var(model, "tas", period=self.period)
        pr = self._load_model_var(model, "pr", period=self.period)
        tmon = self._clim_tas(tas)
        pmon = self._clim_pr(pr)
        return self._regrid_and_classify(tmon, pmon, self._ir)

    def _classify_era5(self) -> xr.DataArray:
        logger.info("  ERA5: classifying")
        tas = self._load_obs_var("tas", self.period)
        pr = self._load_obs_var("pr", self.period)
        tmon = self._clim_tas(tas)
        pmon = self._clim_pr(pr)
        return self._regrid_and_classify(tmon, pmon, self._ir)

    def _classify_be_mswep(self) -> xr.DataArray:
        logger.info("  BE-HR + MSWEP: classifying")
        tmon = self._be_hr_monthly_clim()          # already °C
        pr = self.obs_loader.load_mswep(self.period)
        pmon = self._clim_pr(pr)
        return self._regrid_and_classify(tmon, pmon, self._ir)

    def _classify_cmip6_mmm(self) -> xr.DataArray | None:
        logger.info("  CMIP6 MMM: classifying")
        ir = max(self._ir, 250_000.0)              # coarse-grid floor
        from feather.data.variables import get_var

        tas_info = get_var("tas")
        pr_info = get_var("pr")

        tmon_members: list[xr.DataArray] = []
        pmon_members: list[xr.DataArray] = []
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
            tmon_members.append(tmon)
            pmon_members.append(pmon)

        if not tmon_members:
            logger.info("  CMIP6 MMM: no members available")
            return None

        tmon_mmm = xr.concat(tmon_members, dim="member").mean("member")
        pmon_mmm = xr.concat(pmon_members, dim="member").mean("member")
        lat_da = xr.DataArray(self._tlat, dims="lat", coords={"lat": self._tlat})
        code = classify_kt(tmon_mmm, pmon_mmm, lat_da)
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

    # ── Regridding ─────────────────────────────────────────────────────

    def _regrid_and_classify(
        self, tmon: xr.DataArray, pmon: xr.DataArray, ir: float,
    ) -> xr.DataArray:
        tmon_c = self._regrid_monthly(tmon, ir)
        pmon_c = self._regrid_monthly(pmon, ir)
        lat_da = xr.DataArray(self._tlat, dims="lat", coords={"lat": self._tlat})
        code = classify_kt(tmon_c, pmon_c, lat_da)
        return self._apply_land_mask(code)

    def _regrid_monthly(self, clim: xr.DataArray, ir: float) -> xr.DataArray:
        """Regrid a (month, lat, lon) climatology to the common grid.

        All sources are regridded with the same ``resolution`` and
        ``lon_bounds`` so they land on an identical target grid.
        """
        import nereus as nr

        lat_name = "lat" if "lat" in clim.coords else "latitude"
        lon_name = "lon" if "lon" in clim.coords else "longitude"
        lat = np.asarray(clim[lat_name].values)
        lon = np.asarray(clim[lon_name].values)
        lon2d, lat2d = np.meshgrid(lon, lat)

        interp = None
        out = []
        for i in range(clim.sizes["month"]):
            field = np.asarray(clim.isel(month=i).values).ravel()
            if interp is None:
                _, interp = nr.regrid(
                    field, lon=lon2d, lat=lat2d,
                    resolution=self._res, influence_radius=ir,
                    lon_bounds=(0.0, 360.0), as_xarray=True,
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

    def _apply_land_mask(self, code: xr.DataArray) -> xr.DataArray:
        mask = self._load_land_mask()
        if mask is not None:
            code = code.where(mask)
        return code

    def _load_land_mask(self) -> xr.DataArray | None:
        """Berkeley Earth land mask interpolated to the common grid (bool)."""
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
            if float(mask.lon.min()) < 0:
                mask = mask.assign_coords(lon=((mask.lon + 360) % 360)).sortby("lon")
            mask_i = mask.interp(
                lat=xr.DataArray(self._tlat, dims="lat"),
                lon=xr.DataArray(self._tlon, dims="lon"),
                method="nearest", kwargs={"fill_value": 0.0},
            )
            return mask_i > 0.5
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
        figs.append(self._plot_bar(results))
        figs.append(self._plot_table(results))
        return figs

    @staticmethod
    def _kt_cmap_norm():
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap([KT_COLORS[lbl] for lbl in KT_LABELS])
        cmap.set_bad("0.85")
        norm = BoundaryNorm(np.arange(0.5, 15.5, 1.0), cmap.N)
        return cmap, norm

    def _plot_maps(self, results: dict) -> tuple[plt.Figure, dict]:
        """Multi-panel discrete KT classification maps."""
        import cartopy.crs as ccrs
        import math

        sources = results["sources"]
        lon = results["lon"]
        lat = results["lat"]
        cmap, norm = self._kt_cmap_norm()

        n = len(sources)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        proj = ccrs.Robinson(central_longitude=0)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 3.6 * nrows),
            subplot_kw={"projection": proj},
        )
        axes = np.atleast_1d(axes).ravel()

        mesh = None
        for ax, src in zip(axes, sources):
            code = results["codes"][src]
            mesh = ax.pcolormesh(
                lon, lat, code.values,
                transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                shading="auto",
            )
            ax.coastlines(linewidth=0.4)
            ax.set_global()
            ax.set_title(src, fontsize=10)
        for ax in axes[len(sources):]:
            ax.axis("off")

        fig.suptitle(
            f"{self.title} (land only, {self.period[0]}–{self.period[1]})",
            fontsize=13,
        )
        # Discrete colorbar with type labels
        cbar = fig.colorbar(
            mesh, ax=axes.tolist(), orientation="horizontal",
            fraction=0.04, pad=0.04, ticks=np.arange(1, 15),
        )
        cbar.ax.set_xticklabels(KT_LABELS, fontsize=8)

        meta = self._build_metadata(
            title=f"{self.title} — Maps",
            figure_id="kt_classification_maps",
            models=results["models"],
            description=(
                "Köppen–Trewartha (KT14) climate type per grid cell, land only, "
                "on a common 0.25° grid. One panel per model, ERA5, the Berkeley "
                "Earth HR + MSWEP observational combination, and the CMIP6 "
                "multi-model mean (when available)."
            ),
            period=self.period,
            plot_type="map",
        )
        return fig, meta

    def _plot_bar(self, results: dict) -> tuple[plt.Figure, dict]:
        """Grouped bar chart of % land area per KT type per source."""
        sources = results["sources"]
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
        sources = results["sources"]
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
