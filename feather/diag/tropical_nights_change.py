"""Tropical Nights climate change signal diagnostic.

Compares mean annual Tropical Nights (TN > 20 °C) between a historical
reference period (hist-1950) and a future period (ssp245), quantifying the
climate change signal for each model.

Produces five figure groups:

A (×1): Mean TN20 map — reference period (one panel per model).
B (×1): Mean TN20 map — future period (models with SSP2-4.5 data only).
C (×1): Change map ΔTN = future − reference (symmetric diverging colormap).
D (×1): Stitched annual time series hist+ssp245 with vertical line at 2015
         and shaded reference / future windows.
E (×1): Mean daily Tmin bias vs Berkeley Earth Land TMIN (reference period).
         Skipped when BE data is unavailable.

Per-model NetCDF checkpoints (outside figures tree):
  ``{output_dir}/tropical_nights_change/{model}_tn_hist_{start}_{end}.nc``
  ``{output_dir}/tropical_nights_change/{model}_tn_ssp_{start}_{end}.nc``

Configuration lives in ``config.project["climate_change"]``:

.. code-block:: yaml

    project:
      climate_change:
        reference_period: ["1981", "2000"]
        future_period:    ["2031", "2050"]
        hist_load_period: ["1981", "2014"]   # years loaded for time series
        ssp_load_period:  ["2015", "2050"]   # years attempted from ssp runs
        models:
          IFS-FESOM2-SR:
            hist_experiment:    hist-1950
            hist_data_source:   cmor          # cmor | kerchunk_native
            future_experiment:  highres-future-ssp245
            future_data_source: cmor
          IFS-FESOM2-SR-r2:
            hist_data_source:   kerchunk_native
            hist_data_root:     /work/.../hist-1950
            future_data_source: kerchunk_native
            future_data_root:   /work/.../ssp245
            future_only_to:     "2030"        # truncates future if run incomplete
          ICON-ESM-ER:
            hist_experiment:    hist-1950
            hist_data_source:   cmor
            future_experiment:  highres-future-ssp245
            future_data_source: cmor
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
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.util.spatial import compute_latlon_areas, latlon_global_mean

logger = logging.getLogger(__name__)

_TN_THRESHOLD_K: float = 293.15      # 20 °C
_K_TO_C: float = 273.15
_HIST_BOUNDARY_YEAR: int = 2015      # dashed line in timeseries plot
_BE_TMIN_LAND_PATH = Path(
    "/work/bm1344/AWI/OBS/berkeleyearth/Land_TMIN_Gridded_0p25deg.nc"
)


@register
class TropicalNightsChangeDiag(DiagnosticBase):
    """Tropical Nights climate change signal under SSP2-4.5.

    Loads daily tasmin from hist-1950 and highres-future-ssp245 experiments,
    computes annual TN20 counts for both, and produces reference/future
    climatology maps, a change map, and a stitched time series.
    """

    name = "tropical_nights_change"
    title = "Tropical Nights — Climate Change Signal (SSP2-4.5)"
    domain = "sfc"
    variables = ["tasmin"]
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
        """Directory for NC checkpoints (outside figures tree)."""
        return Path(self.config.output_dir) / "tropical_nights_change"

    def _nc_hist_path(self, model: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return (
            self.nc_dir
            / f"{safe}_tn_hist_{self.hist_load_period[0]}_{self.hist_load_period[1]}.nc"
        )

    def _nc_ssp_path(self, model: str) -> Path:
        safe = model.replace("/", "_").replace(" ", "_")
        return (
            self.nc_dir
            / f"{safe}_tn_ssp_{self.ssp_load_period[0]}_{self.ssp_load_period[1]}.nc"
        )

    # ── Loader factories ───────────────────────────────────────────────

    def _make_cmor_loader(self, model: str, experiment: str, data_root: str = ""):
        """Build a CMORLoader with the given experiment/root for *model*."""
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
        """Build a KerchunkParquetLoader using the native daily store."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        mc_orig = self.config.model_configs.get(model) or ModelConfig(name=model)
        mc = copy.copy(mc_orig)
        mc.data_source_type = "kerchunk_native"
        mc.variant = variant
        mc.data_root = ""       # use global root from data_source, not per-model
        cfg = copy.copy(self.config)
        cfg.model_configs = {**self.config.model_configs, model: mc}
        cfg.data_source = {"type": "kerchunk_parquet", "root": data_root}
        return KerchunkParquetLoader(cfg)

    def _make_hist_loader(self, model: str):
        """Return loader for the historical experiment of *model*."""
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
        """Return loader for the SSP2-4.5 experiment, or None if unavailable."""
        cc_cfg = self._cc_models.get(model, {})

        # Respect declared run end: skip future if run ends before fut_period starts
        fut_only_to = cc_cfg.get("future_only_to")
        if fut_only_to and int(fut_only_to) < int(self.fut_period[0]):
            logger.info(
                "  %s: future run ends %s < %s — future period not available",
                model, fut_only_to, self.fut_period[0],
            )
            return None

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
        return self._make_cmor_loader(
            model, fut_exp, cc_cfg.get("future_data_root", "")
        )

    # ── Static / shared helpers ────────────────────────────────────────

    @staticmethod
    def _count_tn_days(da: xr.DataArray) -> xr.DataArray:
        """Annual count of days with tasmin > 20 °C, dims (year, lat, lon)."""
        exceed = (da > _TN_THRESHOLD_K).astype(np.int16)
        annual = exceed.groupby("time.year").sum("time")
        annual.name = "tn_count"
        annual.attrs = {
            "long_name": "Tropical Nights count (TN > 20 °C)",
            "units": "days/year",
            "threshold_K": _TN_THRESHOLD_K,
        }
        return annual

    def _be_tmin_path(self) -> Path | None:
        ds_cfg = self.config.obs_datasets.get("BERKELEY_EARTH_TMIN", {})
        if ds_cfg:
            p = Path(ds_cfg.get("path", ""))
            if p.exists():
                return p
        if _BE_TMIN_LAND_PATH.exists():
            return _BE_TMIN_LAND_PATH
        return None

    def _load_land_mask(
        self, model_lat: np.ndarray, model_lon: np.ndarray
    ) -> xr.DataArray | None:
        """BE land mask (True = land) interpolated to model lat/lon."""
        path = self._be_tmin_path()
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

    def _load_be_mean_tmin(
        self,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
        period: tuple[str, str],
    ) -> xr.DataArray | None:
        """BE Land TMIN period mean interpolated to model grid (K)."""
        import pandas as pd

        path = self._be_tmin_path()
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
            logger.warning("  Could not load BE Land TMIN: %s", exc)
            return None

    def _land_mean_series(
        self, tn_annual: xr.DataArray, land_mask: xr.DataArray | None
    ) -> xr.DataArray:
        """Area-weighted global land-mean of annual TN count (year,)."""
        areas = compute_latlon_areas(
            np.asarray(tn_annual["lat"]), np.asarray(tn_annual["lon"])
        )
        areas_da = xr.DataArray(areas, dims=("lat", "lon"))
        if land_mask is not None:
            areas_da = areas_da.where(land_mask, 0.0)
        return latlon_global_mean(tn_annual, areas_da)

    # ── NC I/O ─────────────────────────────────────────────────────────

    def _load_from_loader(
        self, loader, model: str, period: tuple[str, str]
    ) -> xr.DataArray:
        """Call load_var, handling both CMORLoader (needs table=) and Kerchunk."""
        from feather.data.kerchunk_loader import KerchunkParquetLoader

        if isinstance(loader, KerchunkParquetLoader):
            return loader.load_var(model, "tasmin", period=period)
        return loader.load_var(model, "tasmin", table="day", period=period)

    def _load_and_save_tn(
        self,
        model: str,
        loader,
        period: tuple[str, str],
        nc_path: Path,
        *,
        land_mask: xr.DataArray | None = None,
        ref_period: tuple[str, str] | None = None,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        """Load annual TN count + optional period-mean Tmin from NC or raw data.

        Returns
        -------
        (tn_annual, tmin_mean)
            ``tn_annual``: DataArray(year, lat, lon), or None on failure.
            ``tmin_mean``: DataArray(lat, lon) mean over *ref_period* if provided
                and the NC was freshly computed; None otherwise.
        """
        if nc_path.exists():
            logger.info("  %s: loading TN from %s", model, nc_path.name)
            ds = xr.open_dataset(nc_path)
            tmin_mean = ds["tmin_mean"] if "tmin_mean" in ds else None
            return ds["tn_count"], tmin_mean

        logger.info("  %s: computing TN %s–%s", model, *period)
        try:
            da = self._load_from_loader(loader, model, period)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.warning("  %s: cannot load tasmin (%s) — skipping", model, exc)
            return None, None

        tn_raw = self._count_tn_days(da)

        # Compute mean Tmin over ref_period for obs comparison (Group E)
        tmin_raw: xr.DataArray | None = None
        if ref_period is not None:
            ref_slice = da.sel(time=slice(*ref_period))
            if len(ref_slice.time) > 0:
                tmin_raw = ref_slice.mean("time")

        # Apply land mask
        if land_mask is not None:
            tn_raw = tn_raw.where(land_mask)
            if tmin_raw is not None:
                tmin_raw = tmin_raw.where(land_mask)

        # Compute (trigger dask graph)
        try:
            import dask
            to_compute = [tn_raw, tmin_raw] if tmin_raw is not None else [tn_raw]
            computed = dask.compute(*to_compute)
            tn_annual = computed[0]
            tmin_mean = computed[1] if tmin_raw is not None else None
        except (ImportError, AttributeError):
            tn_annual = tn_raw.compute() if hasattr(tn_raw, "compute") else tn_raw
            tmin_mean = (
                tmin_raw.compute() if (tmin_raw is not None and hasattr(tmin_raw, "compute"))
                else tmin_raw
            )

        nc_path.parent.mkdir(parents=True, exist_ok=True)
        ds_vars: dict[str, xr.DataArray] = {"tn_count": tn_annual}
        if tmin_mean is not None:
            ds_vars["tmin_mean"] = tmin_mean.assign_attrs({
                "long_name": (
                    f"Period-mean daily minimum temperature ({ref_period[0]}–{ref_period[1]})"
                ),
                "units": "K",
            })
        xr.Dataset(
            ds_vars,
            attrs={
                "model": model,
                "period_start": period[0],
                "period_end": period[1],
                "threshold": f"tasmin > {_TN_THRESHOLD_K} K (20 °C)",
            },
        ).to_netcdf(nc_path)
        logger.info("  Saved TN NC: %s", nc_path)
        return tn_annual, tmin_mean

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
        """Load/compute TN20 for ref and future periods for all models.

        Returns
        -------
        dict with keys:
          models          : list[str] — models with successful hist data
          ref_clim        : {model: DataArray(lat, lon)} — mean TN, ref period
          fut_clim        : {model: DataArray(lat, lon)} — mean TN, future period
          change          : {model: DataArray(lat, lon)} — fut_clim − ref_clim
          hist_series     : {model: DataArray(year)}     — land-mean TN, hist
          ssp_series      : {model: DataArray(year)}     — land-mean TN, ssp
          model_mean_tmin : {model: DataArray(lat, lon)} — period-mean Tmin (K)
          obs_mean_tmin   : DataArray | None             — BE Tmin (K)
          lat / lon       : shared coordinates
        """
        models: list[str] = []
        ref_clim: dict[str, xr.DataArray] = {}
        fut_clim: dict[str, xr.DataArray] = {}
        change: dict[str, xr.DataArray] = {}
        hist_series: dict[str, xr.DataArray] = {}
        ssp_series: dict[str, xr.DataArray] = {}
        model_mean_tmin: dict[str, xr.DataArray] = {}
        lat_coord: xr.DataArray | None = None
        lon_coord: xr.DataArray | None = None
        shared_land_mask: xr.DataArray | None = None

        for model in self.config.models:
            logger.info("  Processing: %s", model)

            # ── Historical ────────────────────────────────────────────
            hist_loader = self._make_hist_loader(model)
            hist_tn, hist_tmin = self._load_and_save_tn(
                model,
                hist_loader,
                self.hist_load_period,
                self._nc_hist_path(model),
                ref_period=self.ref_period,
            )
            if hist_tn is None:
                logger.warning("  %s: no hist TN — skipping", model)
                continue

            models.append(model)
            if lat_coord is None:
                lat_coord = hist_tn["lat"]
                lon_coord = hist_tn["lon"]
                shared_land_mask = self._load_land_mask(
                    lat_coord.values, lon_coord.values
                )

            ref_slice = hist_tn.sel(year=slice(*self.ref_period))
            ref_clim[model] = ref_slice.mean("year")
            hist_series[model] = self._land_mean_series(hist_tn, shared_land_mask)
            if hist_tmin is not None:
                model_mean_tmin[model] = hist_tmin

            # ── Future (SSP) ──────────────────────────────────────────
            fut_loader = self._make_fut_loader(model)
            if fut_loader is None:
                logger.info("  %s: no future loader — reference-only model", model)
                continue

            ssp_tn, _ = self._load_and_save_tn(
                model,
                fut_loader,
                self.ssp_load_period,
                self._nc_ssp_path(model),
            )
            if ssp_tn is None:
                logger.warning("  %s: SSP TN load failed — reference only", model)
                continue

            ssp_series[model] = self._land_mean_series(ssp_tn, shared_land_mask)

            fut_slice = ssp_tn.sel(year=slice(*self.fut_period))
            if len(fut_slice.year) > 0:
                fut_clim[model] = fut_slice.mean("year")
                change[model] = fut_clim[model] - ref_clim[model]
            else:
                logger.info(
                    "  %s: SSP run ends before future period (%s) — "
                    "excluded from change map",
                    model, self.fut_period[0],
                )

        # Berkeley Earth TMIN for Group E
        obs_mean_tmin = None
        if lat_coord is not None:
            obs_mean_tmin = self._load_be_mean_tmin(
                lat_coord.values, lon_coord.values, self.ref_period
            )

        return {
            "models": models,
            "ref_clim": ref_clim,
            "fut_clim": fut_clim,
            "change": change,
            "hist_series": hist_series,
            "ssp_series": ssp_series,
            "model_mean_tmin": model_mean_tmin,
            "obs_mean_tmin": obs_mean_tmin,
            "lat": lat_coord,
            "lon": lon_coord,
        }

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        figs: list[tuple[plt.Figure, dict]] = []
        if not results["models"]:
            logger.warning("%s: no model data — no figures", self.name)
            return figs

        figs.append(self._plot_ref_map(results))

        if results["fut_clim"]:
            figs.append(self._plot_fut_map(results))

        if results["change"]:
            figs.append(self._plot_change_map(results))

        figs.append(self._plot_timeseries(results))

        if results.get("obs_mean_tmin") is not None and results["model_mean_tmin"]:
            figs.append(self._plot_tmin_bias(results))

        return figs

    @staticmethod
    def _safe_vmax(data_dict: dict) -> float:
        """98th-percentile vmax across all arrays; fallback 1.0."""
        if not data_dict:
            return 1.0
        all_finite = np.concatenate([
            np.asarray(d).ravel()[np.isfinite(np.asarray(d).ravel())]
            for d in data_dict.values()
        ])
        if len(all_finite) == 0:
            return 1.0
        return max(float(np.percentile(all_finite, 98)), 1.0)

    def _plot_ref_map(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group A: mean TN20 map, reference period."""
        models = results["models"]
        data_dict = {m: results["ref_clim"][m] for m in models}
        fig, _ = plot_combined_map(
            data_dict,
            title=(
                f"{self.title}\n"
                f"Reference Period {self.ref_period[0]}–{self.ref_period[1]} (land only)"
            ),
            cmap="YlOrRd",
            vmin=0,
            vmax=self._safe_vmax(data_dict),
            units="days/year",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Reference Period",
            figure_id="tropical_nights_change_ref_map",
            models=models,
            description=(
                f"Mean annual Tropical Nights (TN > 20 °C) over the reference "
                f"period {self.ref_period[0]}–{self.ref_period[1]} from hist-1950 runs. "
                "Land-only; computed from CMOR daily tasmin."
            ),
            period=self.ref_period,
            plot_type="map",
        )
        return fig, meta

    def _plot_fut_map(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group B: mean TN20 map, future period."""
        fut_models = list(results["fut_clim"].keys())
        data_dict = {m: results["fut_clim"][m] for m in fut_models}
        fig, _ = plot_combined_map(
            data_dict,
            title=(
                f"{self.title}\n"
                f"Future Period {self.fut_period[0]}–{self.fut_period[1]} (land only)"
            ),
            cmap="YlOrRd",
            vmin=0,
            vmax=self._safe_vmax(data_dict),
            units="days/year",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Future Period",
            figure_id="tropical_nights_change_fut_map",
            models=fut_models,
            description=(
                f"Mean annual Tropical Nights (TN > 20 °C) over the future "
                f"period {self.fut_period[0]}–{self.fut_period[1]} from SSP2-4.5 runs. "
                "Land-only; computed from CMOR daily tasmin."
            ),
            period=self.fut_period,
            plot_type="map",
        )
        return fig, meta

    def _plot_change_map(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group C: ΔTN change map (future − reference)."""
        change_models = list(results["change"].keys())
        change_dict = {m: results["change"][m] for m in change_models}

        # Symmetric diverging range
        all_finite = np.concatenate([
            np.asarray(d).ravel()[np.isfinite(np.asarray(d).ravel())]
            for d in change_dict.values()
        ])
        vlim = (float(np.percentile(np.abs(all_finite), 98)) or 1.0) if len(all_finite) > 0 else 10.0

        fig, _ = plot_combined_map(
            change_dict,
            title=(
                f"{self.title}\n"
                f"Change: {self.fut_period[0]}–{self.fut_period[1]} minus "
                f"{self.ref_period[0]}–{self.ref_period[1]} (land only)"
            ),
            cmap="RdYlBu_r",
            vmin=-vlim,
            vmax=vlim,
            units="days/year",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Change Map",
            figure_id="tropical_nights_change_delta_map",
            models=change_models,
            description=(
                f"Change in mean annual Tropical Nights (days/year): future period "
                f"({self.fut_period[0]}–{self.fut_period[1]}) minus reference period "
                f"({self.ref_period[0]}–{self.ref_period[1]}). "
                "Positive: more tropical nights in the future (warming signal)."
            ),
            period=(self.ref_period[0], self.fut_period[1]),
            plot_type="map",
        )
        return fig, meta

    def _plot_timeseries(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group D: stitched hist+ssp245 annual time series with period shading."""
        fig, ax = plt.subplots(figsize=(13, 5))

        # Shade reference and future 20-year windows
        ax.axvspan(
            int(self.ref_period[0]), int(self.ref_period[1]) + 1,
            alpha=0.10, color="#1f77b4", zorder=0,
        )
        ax.axvspan(
            int(self.fut_period[0]), int(self.fut_period[1]) + 1,
            alpha=0.10, color="#d62728", zorder=0,
        )

        # Vertical dashed line at hist → ssp boundary
        ax.axvline(
            _HIST_BOUNDARY_YEAR, color="k", lw=1.2, ls="--", alpha=0.55,
            label=f"hist | ssp245 ({_HIST_BOUNDARY_YEAR})",
        )

        for model in results["models"]:
            color = self.config.get_model_color(model)
            labeled = False

            if model in results["hist_series"]:
                s = results["hist_series"][model]
                ax.plot(
                    np.asarray(s["year"]), np.asarray(s),
                    color=color, lw=1.5, label=model,
                )
                labeled = True

            if model in results["ssp_series"]:
                s = results["ssp_series"][model]
                ax.plot(
                    np.asarray(s["year"]), np.asarray(s),
                    color=color, lw=1.5,
                    **({"label": "_nolegend_"} if labeled else {"label": model}),
                )

        ax.set_xlabel("Year")
        ax.set_ylabel("Tropical Nights (days/year)")
        ax.set_title(
            f"{self.title} — Global Land Mean\n"
            f"Blue shading: ref {self.ref_period[0]}–{self.ref_period[1]}; "
            f"red shading: future {self.fut_period[0]}–{self.fut_period[1]}"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        all_models = results["models"]
        meta = self._build_metadata(
            title=f"{self.title} — Global Land Mean Time Series",
            figure_id="tropical_nights_change_timeseries",
            models=all_models,
            description=(
                "Area-weighted global land-mean annual Tropical Nights (TN > 20 °C). "
                f"Continuous line: hist-1950 ({self.hist_load_period[0]}–"
                f"{self.hist_load_period[1]}) joining SSP2-4.5 from {_HIST_BOUNDARY_YEAR}. "
                f"Blue shading: reference period ({self.ref_period[0]}–{self.ref_period[1]}); "
                f"red shading: future period ({self.fut_period[0]}–{self.fut_period[1]}). "
                "Models without future data show only the historical segment."
            ),
            period=(self.hist_load_period[0], self.ssp_load_period[1]),
            plot_type="timeseries",
        )
        return fig, meta

    def _plot_tmin_bias(self, results: dict) -> tuple[plt.Figure, dict]:
        """Group E: model mean Tmin bias vs Berkeley Earth (reference period)."""
        models = [m for m in results["models"] if m in results["model_mean_tmin"]]
        obs_k = results["obs_mean_tmin"]
        obs_c = obs_k - _K_TO_C
        bias_dict = {
            m: results["model_mean_tmin"][m] - obs_k
            for m in models
        }
        fig, _ = plot_combined_bias_map(
            obs_c,
            bias_dict,
            title=(
                f"{self.title}\n"
                f"Mean Tmin Bias vs Berkeley Earth "
                f"({self.ref_period[0]}–{self.ref_period[1]})"
            ),
            obs_title="Berkeley Earth Land TMIN",
            cmap="cmo.thermal",
            bias_cmap="RdBu_r",
            units="°C",
        )
        meta = self._build_metadata(
            title=f"{self.title} — Mean Tmin Bias (reference period)",
            figure_id="tropical_nights_change_tmin_bias",
            models=models,
            description=(
                f"Bias in climatological mean daily minimum temperature "
                f"(model − Berkeley Earth Land TMIN, °C) for the reference period "
                f"{self.ref_period[0]}–{self.ref_period[1]}. "
                "Land-only. Model: CMOR daily tasmin mean over reference period; "
                "obs: Berkeley Earth monthly Land TMIN (anomaly + climatology)."
            ),
            obs_dataset="BERKELEY_EARTH_TMIN",
            obs_variable="temperature",
            period=self.ref_period,
            plot_type="bias_map",
        )
        return fig, meta
