"""2m Temperature evaluation against Berkeley Earth Land+Ocean.

Produces 10 figures across 8 groups:
A (x3): Bias maps (annual, DJF, JJA) vs Berkeley Earth
B (x1): Global-mean time series (monthly + annual)
C (x1): Seasonal cycle (12-month climatology)
D (x1): Zonal mean profile
E (x3): Warming trend maps — global (°C/decade), Arctic, Antarctic
F (x1): Taylor diagram (pattern corr vs normalised STD)
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import xarray as xr

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.lines import plot_taylor_diagram
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.plot.styles import OBS_COLOR
from feather.util.spatial import (
    compute_latlon_areas,
    latlon_global_mean,
    zonal_mean,
)
from feather.util.temporal import (
    annual_mean,
    climatology,
    linear_trend,
    monthly_climatology,
    seasonal_climatology,
)

logger = logging.getLogger(__name__)

_K_TO_C = 273.15   # subtract from K values to get °C for display


@register
class TemperatureBerkeley(DiagnosticBase):
    """2m temperature evaluation against Berkeley Earth Land+Ocean.

    Uses Berkeley Earth (station-based, independent of reanalysis) as
    the primary reference dataset instead of ERA5.  Provides bias maps,
    time series, seasonal cycle, zonal mean, warming trends (global +
    polar), and a Taylor diagram.
    """

    name = "temperature_berkeley"
    title = "2m Temperature (Berkeley Earth)"
    domain = "sfc"
    variables = ["tas"]
    group = "temperature"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # ── Orchestration (per-group incremental) ─────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-group: compute -> plot -> save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        out = self.output_dir

        # Determine which groups need computing
        bias_ids = [
            f"tas_{p}_bias_combined"
            for p in ["annual", "DJF", "MAM", "JJA", "SON"]
        ]
        need_a = not skip_existing or not all(
            self._figure_exists(fid) for fid in bias_ids
        )
        need_b = not skip_existing or not self._figure_exists("tas_timeseries")
        need_c = not skip_existing or not self._figure_exists("tas_seasonal_cycle")
        need_d = not skip_existing or not self._figure_exists("tas_zonal_mean")
        trend_ids = [
            "tas_trend_combined", "tas_trend_arctic", "tas_trend_antarctic",
        ]
        need_e = not skip_existing or not all(
            self._figure_exists(fid) for fid in trend_ids
        )
        need_f = not skip_existing or not self._figure_exists("tas_taylor")

        # Collect existing paths
        if not need_a:
            logger.info("Skipping bias maps -- figures exist")
            saved.extend([
                (out / f"{fid}.png", out / f"{fid}.json")
                for fid in bias_ids
            ])
        if not need_b:
            logger.info("Skipping timeseries -- figure exists")
            saved.append((out / "tas_timeseries.png", out / "tas_timeseries.json"))
        if not need_c:
            logger.info("Skipping seasonal cycle -- figure exists")
            saved.append((out / "tas_seasonal_cycle.png", out / "tas_seasonal_cycle.json"))
        if not need_d:
            logger.info("Skipping zonal mean -- figure exists")
            saved.append((out / "tas_zonal_mean.png", out / "tas_zonal_mean.json"))
        if not need_e:
            logger.info("Skipping trend maps -- figures exist")
            saved.extend([
                (out / f"{fid}.png", out / f"{fid}.json")
                for fid in trend_ids
            ])
        if not need_f:
            logger.info("Skipping Taylor diagram -- figure exists")
            saved.append((out / "tas_taylor.png", out / "tas_taylor.json"))

        if not any([need_a, need_b, need_c, need_d, need_e, need_f]):
            logger.info("Diagnostic %s complete -- all figures exist", self.name)
            return saved

        # Load shared data
        try:
            shared = self._load_shared_data()
        except Exception:
            logger.warning(
                "Failed to load data for temperature_berkeley", exc_info=True,
            )
            return saved

        # Group A: Bias maps
        if need_a:
            try:
                results = self._compute_bias_maps(shared)
                for fig, meta in self._plot_bias_maps(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group A (bias maps) failed", exc_info=True)

        # Group B: Timeseries
        if need_b:
            try:
                results = self._compute_timeseries(shared)
                for fig, meta in self._plot_timeseries(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group B (timeseries) failed", exc_info=True)

        # Group C: Seasonal cycle
        if need_c:
            try:
                results = self._compute_seasonal_cycle(shared)
                for fig, meta in self._plot_seasonal_cycle(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group C (seasonal cycle) failed", exc_info=True)

        # Group D: Zonal mean
        if need_d:
            try:
                results = self._compute_zonal_mean(shared)
                for fig, meta in self._plot_zonal_mean(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group D (zonal mean) failed", exc_info=True)

        # Group E: Trend maps
        if need_e:
            try:
                results = self._compute_trends(shared)
                for fig, meta in self._plot_trends(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group E (trends) failed", exc_info=True)

        # Group F: Taylor diagram
        if need_f:
            try:
                results = self._compute_taylor(shared)
                for fig, meta in self._plot_taylor(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("Group F (Taylor diagram) failed", exc_info=True)

        logger.info(
            "Diagnostic %s complete -- %d figure(s)", self.name, len(saved),
        )
        return saved

    def compute(self) -> dict[str, Any]:
        """Compute all results (backward compat wrapper)."""
        shared = self._load_shared_data()
        return {
            "bias_maps": self._compute_bias_maps(shared),
            "timeseries": self._compute_timeseries(shared),
            "seasonal_cycle": self._compute_seasonal_cycle(shared),
            "zonal_mean": self._compute_zonal_mean(shared),
            "trends": self._compute_trends(shared),
            "taylor": self._compute_taylor(shared),
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figures (backward compat wrapper)."""
        figures = []
        figures.extend(self._plot_bias_maps(results["bias_maps"]))
        figures.extend(self._plot_timeseries(results["timeseries"]))
        figures.extend(self._plot_seasonal_cycle(results["seasonal_cycle"]))
        figures.extend(self._plot_zonal_mean(results["zonal_mean"]))
        figures.extend(self._plot_trends(results["trends"]))
        figures.extend(self._plot_taylor(results["taylor"]))
        return figures

    # ── Berkeley Earth loading ───────────────────────────────────────

    def _load_berkeley_earth(
        self, period: tuple[str, str] | None = None,
    ) -> xr.DataArray:
        """Load Berkeley Earth Land+Ocean data.

        Prefers the high-resolution 0.25° dataset (``BERKELEY_EARTH_HR``)
        when present in the config, falling back to the legacy 1° dataset
        (``BERKELEY_EARTH``) otherwise.
        """
        if "BERKELEY_EARTH_HR" in self.config.obs_datasets:
            return self._load_berkeley_earth_hr(period)
        return self._load_berkeley_earth_legacy(period)

    def _load_berkeley_earth_hr(
        self, period: tuple[str, str] | None = None,
    ) -> xr.DataArray:
        """Load Berkeley Earth 0.25° gridded dataset.

        The file stores monthly *anomalies* (°C, re: 1951-1980 climatology)
        and a separate ``climatology`` array (12 × lat × lon, °C).
        Absolute temperature = anomaly + climatology[month_of_year].

        Time is encoded as decimal years (float); this method converts it to
        a proper ``pandas.DatetimeIndex`` before slicing.
        """
        import pandas as pd

        ds_cfg = self.config.obs_datasets["BERKELEY_EARTH_HR"]
        filepath = Path(ds_cfg["path"]) / ds_cfg["variables"]["temperature"]
        ds_full = xr.open_dataset(filepath, chunks="auto")

        # Convert decimal-year time → DatetimeIndex
        dec_years = ds_full["time"].values
        years = dec_years.astype(int)
        months = np.floor((dec_years - years) * 12).astype(int) + 1
        months = np.clip(months, 1, 12)
        datetimes = pd.to_datetime(
            [f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)]
        )
        ds_full = ds_full.assign_coords(time=datetimes)

        # Slice to requested period
        start, end = period if period else (None, None)
        anom = ds_full["temperature"].sel(time=slice(start, end))
        clim = ds_full["climatology"]  # (month_number, latitude, longitude)

        # Reconstruct absolute temperature: anomaly + climatology[month_of_year]
        month_idx = anom.time.dt.month.values - 1  # 0-based
        clim_np = clim.values  # (12, nlat, nlon)
        clim_matched = clim_np[month_idx]  # (ntime, nlat, nlon)
        abs_temp = anom + xr.DataArray(
            clim_matched, dims=anom.dims, coords=anom.coords,
        )

        # Rename dims latitude/longitude → lat/lon
        rename = {}
        if "latitude" in abs_temp.dims:
            rename["latitude"] = "lat"
        if "longitude" in abs_temp.dims:
            rename["longitude"] = "lon"
        if rename:
            abs_temp = abs_temp.rename(rename)

        # Shift −180..180 → 0..360
        if float(abs_temp.lon.min()) < 0:
            abs_temp = abs_temp.assign_coords(
                lon=((abs_temp.lon + 360) % 360),
            ).sortby("lon")

        # degC → K
        return abs_temp + 273.15

    def _load_berkeley_earth_legacy(
        self, period: tuple[str, str] | None = None,
    ) -> xr.DataArray:
        """Load legacy Berkeley Earth 1° Land+Ocean file (absolute °C → K)."""
        da = self.obs_loader.load("BERKELEY_EARTH", "2t", period=period)

        # Rename dims
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)

        # Shift lons from -180..180 to 0..360
        if float(da.lon.min()) < 0:
            da = da.assign_coords(
                lon=((da.lon + 360) % 360),
            ).sortby("lon")

        # Convert degC to K
        return da + 273.15

    # ── Shared data loading ──────────────────────────────────────────

    def _load_shared_data(self) -> dict[str, Any]:
        """Load model and Berkeley Earth data used across groups."""
        logger.info("Loading shared data for temperature_berkeley...")

        model_monthly: dict[str, xr.DataArray] = {}
        model_coords: dict[str, tuple] = {}

        for model in self.config.models:
            try:
                da = self._load_model_var(model, "tas", period=self.period)
                model_monthly[model] = da
                lon, lat = self._load_model_coords(model, "tas")
                model_coords[model] = (lon, lat)
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "Variable tas not available for %s -- skipping", model,
                )

        if not model_monthly:
            raise RuntimeError("No models have tas data")

        # Load Berkeley Earth
        logger.info("  Loading Berkeley Earth observations...")
        berkeley = self._load_berkeley_earth(period=self.period)

        return {
            "model_monthly": model_monthly,
            "model_coords": model_coords,
            "berkeley": berkeley,
        }

    # ── Group A: Bias maps ───────────────────────────────────────────

    def _compute_bias_maps(self, shared: dict) -> dict[str, Any]:
        """Compute annual and seasonal T2m bias maps vs Berkeley Earth."""
        logger.info("Computing temperature bias maps...")
        model_monthly = shared["model_monthly"]
        model_coords = shared["model_coords"]
        berkeley = shared["berkeley"]

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )

        # Obs climatologies (on Berkeley Earth obs grid)
        obs_clim = climatology(berkeley, self.period).compute()
        obs_seasonal = seasonal_climatology(berkeley, self.period)

        obs_lats = obs_clim.lat.values
        obs_lons = obs_clim.lon.values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # Cache nereus interpolator per source grid size
        _interp_cache: dict[int, Any] = {}
        obs_clim_common = None
        obs_seasonal_common = {}
        common_area = None
        target_lats = None
        target_lons = None

        model_results: dict[str, dict] = {}

        for model in self.config.models:
            if model not in model_monthly:
                continue
            logger.info("  Computing biases for %s...", model)

            model_data = model_monthly[model]
            lon, lat = model_coords[model]

            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            model_clim = climatology(model_data, self.period).compute()
            model_gmean = float(model_clim.mean().values)

            model_seas = seasonal_climatology(model_data, self.period)
            model_seas = {
                s: model_seas[s].compute() for s in model_seas.data_vars
            }

            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _interp_cache:
                logger.info("  Building nereus interpolator (%d pts)...", n_src)
                annual_regrid, interp = nr.regrid(
                    model_clim.values.ravel(),
                    lon=np.asarray(lon).ravel(),
                    lat=np.asarray(lat).ravel(),
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]

                    # Regrid obs to common grid
                    obs_lons_2d, obs_lats_2d = np.meshgrid(obs_lons, obs_lats)
                    _, obs_interp = nr.regrid(
                        obs_clim.values.ravel(),
                        lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                        resolution=obs_res,
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    obs_clim_common = xr.DataArray(
                        obs_interp(obs_clim.values.ravel()),
                        dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
                    common_area = compute_latlon_areas(target_lats, target_lons)

                    for season in obs_seasonal:
                        s_np = obs_interp(obs_seasonal[season].values.ravel())
                        obs_seasonal_common[season] = xr.DataArray(
                            s_np, dims=("lat", "lon"),
                            coords={"lat": target_lats, "lon": target_lons},
                        )
            else:
                interp = _interp_cache[n_src]
                regridded_np = interp(model_clim.values.ravel())
                annual_regrid = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            # Annual bias + statistics
            annual_bias = annual_regrid - obs_clim_common
            stats = self._compute_summary_stats(
                annual_regrid, obs_clim_common, annual_bias,
                target_lats, common_area,
            )

            # Seasonal biases
            seasonal_biases: dict[str, Any] = {}
            seasonal_regrids: dict[str, Any] = {}
            interp = _interp_cache[n_src]
            for season in ["DJF", "MAM", "JJA", "SON"]:
                if season in model_seas:
                    s_np = interp(model_seas[season].values.ravel())
                    s_regrid = xr.DataArray(
                        s_np, dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
                    seasonal_regrids[season] = s_regrid
                    if season in obs_seasonal_common:
                        seasonal_biases[season] = (
                            s_regrid - obs_seasonal_common[season]
                        )

            model_results[model] = {
                "annual_regrid": annual_regrid,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": model_gmean,
                "annual_bias": annual_bias,
                "stats": stats,
                "seasonal_biases": seasonal_biases,
            }

        # Benchmark biases (CMIP6, HighResMIP, …) — one MMM per benchmark.
        cmip6_data = {}
        cmip6_info = {}
        cmip6_individual_data: dict[str, dict] = {}
        benchmark_data: dict[str, dict] = {}
        benchmark_info: dict[str, dict] = {}
        if self.cmip6_enabled and target_lats is not None:
            for i, bench in enumerate(self.benchmarks):
                label = getattr(bench, "label", "CMIP6 MMM")
                if i == 0 and self.cmip6_individual:
                    cmip6_individual_data = self._compute_cmip6_individual(
                        target_lats, target_lons,
                        obs_clim_common, obs_seasonal_common, common_area,
                    )
                    b_data, b_info = self._mmm_from_individual(
                        cmip6_individual_data,
                        obs_clim_common, obs_seasonal_common, common_area,
                    )
                else:
                    b_data, b_info = self._compute_cmip6_mmm(
                        target_lats, target_lons,
                        obs_clim_common, obs_seasonal_common, common_area,
                        loader=bench,
                    )
                if b_data:
                    benchmark_data[label] = b_data
                    benchmark_info[label] = b_info

            if benchmark_data:
                primary_label = next(iter(benchmark_data))
                cmip6_data = benchmark_data[primary_label]
                cmip6_info = benchmark_info[primary_label]

        # Shared colorbar ranges
        colorbar_ranges = self._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
            cmip6_data=cmip6_data,
            cmip6_individual_data=cmip6_individual_data,
            benchmark_data=benchmark_data,
        )

        return {
            "models": model_results,
            "obs": {
                "clim": obs_clim_common,
                "seasonal_clim": obs_seasonal_common,
            },
            "colorbar_ranges": colorbar_ranges,
            "cmip6_data": cmip6_data,
            "cmip6_info": cmip6_info,
            "cmip6_individual_data": cmip6_individual_data,
            "benchmark_data": benchmark_data,
            "benchmark_info": benchmark_info,
        }

    def _plot_bias_maps(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot bias maps for annual, DJF, JJA."""
        figures = []
        obs_clim = results["obs"]["clim"]
        cb = results["colorbar_ranges"]
        cmip6_info = results.get("cmip6_info", {})
        cmip6_individual_data = results.get("cmip6_individual_data", {})
        benchmark_data = results.get("benchmark_data", {})

        periods = [("annual", "Annual Mean")]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in cb:
                periods.append((season, season))

        for period_key, period_label in periods:
            bias_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in results["models"].items():
                if period_key == "annual":
                    bias_field = mdata["annual_bias"]
                    summary_stats[model] = mdata["stats"]
                else:
                    bias_field = mdata["seasonal_biases"].get(period_key)
                    if bias_field is None:
                        continue
                bias_dict[model] = bias_field
                all_models.append(model)

            # Benchmark MMMs (CMIP6, HighResMIP, …)
            for b_label, b_data in benchmark_data.items():
                if period_key in b_data:
                    bias_dict[b_label] = b_data[period_key]["bias"]
                    all_models.append(b_label)

            # CMIP6 individual
            if period_key in cmip6_individual_data:
                for label, c_data in cmip6_individual_data[period_key].items():
                    bias_dict[label] = c_data["bias"]
                    all_models.append(label)

            if not bias_dict:
                continue

            # Get obs data for period
            if period_key == "annual":
                obs_period = obs_clim
            else:
                obs_period = results["obs"]["seasonal_clim"].get(period_key)
                if obs_period is None:
                    continue

            p_cb = cb.get(period_key, cb.get("annual", {}))

            fig, axes = plot_combined_bias_map(
                obs_period - _K_TO_C, bias_dict,
                title=f"2m Temperature {period_label}",
                obs_title="Berkeley Earth",
                cmap="cmo.thermal",
                bias_cmap="RdBu_r",
                vmin=(p_cb["vmin"] - _K_TO_C if p_cb.get("vmin") is not None else None),
                vmax=(p_cb["vmax"] - _K_TO_C if p_cb.get("vmax") is not None else None),
                bias_vmax=p_cb.get("bias_vmax"),
                units="°C",
                method=self._regrid_method,
            )

            meta = self._build_metadata(
                title=f"2m Temperature {period_label} Bias",
                figure_id=f"tas_{period_key.lower()}_bias_combined",
                models=all_models,
                variables=["tas"],
                description=(
                    f"{period_label} 2m temperature bias maps "
                    f"(model - Berkeley Earth)."
                ),
                obs_dataset="Berkeley Earth",
                obs_variable="2m temperature",
                plot_type="combined_bias_map",
                period=self.period,
                cmip6_info=cmip6_info or None,
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures

    # ── Group B: Timeseries ──────────────────────────────────────────

    def _compute_timeseries(self, shared: dict) -> dict[str, Any]:
        """Compute global-mean T2m time series."""
        logger.info("Computing temperature time series...")
        model_ts: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            ts = self._model_global_mean(da, model).compute()
            model_ts[model] = ts

        berkeley = shared["berkeley"]
        obs_ts = latlon_global_mean(berkeley)

        # Benchmark global-mean series (CMIP6, HighResMIP, …)
        benchmarks_ts = self._benchmark_timeseries("tas")
        primary = benchmarks_ts[0] if benchmarks_ts else None

        return {
            "models": model_ts,
            "obs": obs_ts,
            "benchmarks_ts": benchmarks_ts,
            "cmip6_ts": primary["ts"] if primary else None,
            "cmip6_info": primary["info"] if primary else {},
            "cmip6_individual_ts": primary["individual"] if primary else {},
        }

    def _plot_timeseries(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot global-mean T2m time series."""
        import pandas as pd

        fig, ax = plt.subplots(figsize=(12, 5))
        all_models = list(self.config.models)

        benchmarks = results.get("benchmarks_ts", [])
        for bench in benchmarks:
            all_models.append(bench["label"])
            all_models.extend(bench["individual"].keys())

        # Monthly pass (background)
        for bench in benchmarks:
            b_color = bench["color"]
            for ts in bench["individual"].values():
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values - _K_TO_C,
                        color=b_color, alpha=0.2, linewidth=0.5)
            b_ts = bench["ts"]
            time_vals = _to_plot_time(b_ts.time.values)
            ax.plot(time_vals, b_ts.values - _K_TO_C,
                    color=b_color, alpha=0.3, linewidth=0.7,
                    linestyle="--")

        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values - _K_TO_C,
                    color=color, alpha=0.3, linewidth=0.7)

        obs_ts = results["obs"]
        obs_time = _to_plot_time(obs_ts.time.values)
        ax.plot(obs_time, obs_ts.values - _K_TO_C,
                color=OBS_COLOR, alpha=0.3, linewidth=0.7)

        # Annual pass (foreground)
        for bench in benchmarks:
            b_color = bench["color"]
            b_label = bench["label"]
            members_name = b_label[:-4] if b_label.endswith(" MMM") else b_label
            for i, ts in enumerate(bench["individual"].values()):
                label = f"{members_name} members" if i == 0 else "_nolegend_"
                ts_annual = annual_mean(ts)
                time_vals = _to_plot_time(ts_annual.time.values)
                ax.plot(time_vals, ts_annual.values - _K_TO_C,
                        color=b_color, alpha=0.35, linewidth=0.8,
                        label=label)
            b_annual = annual_mean(bench["ts"])
            time_vals = _to_plot_time(b_annual.time.values)
            ax.plot(time_vals, b_annual.values - _K_TO_C,
                    label=b_label, color=b_color,
                    linewidth=2.0, linestyle="--")

        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values - _K_TO_C,
                    label=model, color=color, linewidth=2.0)

        obs_annual = annual_mean(obs_ts)
        obs_annual_time = _to_plot_time(obs_annual.time.values)
        ax.plot(obs_annual_time, obs_annual.values - _K_TO_C,
                label="Berkeley Earth", color=OBS_COLOR, linewidth=2.5)

        ax.set_title("2m Temperature \u2014 Global Mean")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="2m Temperature Global Mean Time Series",
            figure_id="tas_timeseries",
            models=all_models,
            variables=["tas"],
            description=(
                "Area-weighted global mean monthly 2m temperature "
                "time series for all models vs Berkeley Earth."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="timeseries",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group C: Seasonal cycle ──────────────────────────────────────

    def _compute_seasonal_cycle(self, shared: dict) -> dict[str, Any]:
        """Compute monthly climatological cycle of global-mean T2m."""
        logger.info("Computing temperature seasonal cycle...")
        model_monthly_clim: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            ts = self._model_global_mean(da, model).compute()
            model_monthly_clim[model] = monthly_climatology(ts, self.period)

        berkeley = shared["berkeley"]
        obs_ts = latlon_global_mean(berkeley)
        obs_monthly = monthly_climatology(obs_ts, self.period)

        # Benchmark monthly climatologies (CMIP6, HighResMIP, …)
        benchmarks_monthly = []
        for bench in self._benchmark_timeseries("tas"):
            benchmarks_monthly.append({
                "label": bench["label"],
                "color": bench["color"],
                "monthly": monthly_climatology(bench["ts"]),
                "info": bench["info"],
                "individual": {
                    mname: monthly_climatology(mts)
                    for mname, mts in bench["individual"].items()
                },
            })
        primary = benchmarks_monthly[0] if benchmarks_monthly else None

        return {
            "models": model_monthly_clim,
            "obs": obs_monthly,
            "benchmarks_monthly": benchmarks_monthly,
            "cmip6_monthly": primary["monthly"] if primary else None,
            "cmip6_info": primary["info"] if primary else {},
            "cmip6_individual_monthly": primary["individual"] if primary else {},
        }

    def _plot_seasonal_cycle(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot 12-month seasonal cycle."""
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]

        fig, ax = plt.subplots(figsize=(8, 5))
        months = np.arange(1, 13)
        all_models = list(self.config.models)

        # Layers 1-2: Per-benchmark individual members + MMM
        for bench in results.get("benchmarks_monthly", []):
            b_color = bench["color"]
            b_label = bench["label"]
            members_name = b_label[:-4] if b_label.endswith(" MMM") else b_label
            for i, monthly in enumerate(bench["individual"].values()):
                label = f"{members_name} members" if i == 0 else "_nolegend_"
                ax.plot(months, monthly.values - _K_TO_C,
                        color=b_color, alpha=0.35, linewidth=0.8, label=label)
            all_models.extend(bench["individual"].keys())
            ax.plot(months, bench["monthly"].values - _K_TO_C,
                    marker="d", label=b_label, color=b_color,
                    linewidth=1.5, linestyle="--")
            all_models.append(b_label)

        # Layer 3: Model lines
        for model, monthly in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(months, monthly.values - _K_TO_C,
                    marker="o", label=model, color=color)

        # Layer 4: Observations
        ax.plot(months, results["obs"].values - _K_TO_C,
                marker="s", label="Berkeley Earth", color=OBS_COLOR, linewidth=2)

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title("2m Temperature \u2014 Seasonal Cycle")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="2m Temperature Seasonal Cycle",
            figure_id="tas_seasonal_cycle",
            models=all_models,
            variables=["tas"],
            description=(
                "Monthly climatological cycle (Jan-Dec) of global mean "
                "2m temperature for all models vs Berkeley Earth."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="seasonal_cycle",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group D: Zonal mean ──────────────────────────────────────────

    def _compute_zonal_mean(self, shared: dict) -> dict[str, Any]:
        """Compute zonal mean T2m profiles."""
        logger.info("Computing temperature zonal mean profiles...")
        model_zonal: dict[str, xr.DataArray] = {}

        for model, da in shared["model_monthly"].items():
            lon, lat = shared["model_coords"][model]
            clim = climatology(da, self.period).compute()
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                zm = zonal_mean(clim, lat)
            else:
                lon_dim = "lon" if "lon" in clim.dims else "longitude"
                zm = clim.mean(lon_dim)
                if "latitude" in zm.dims:
                    zm = zm.rename({"latitude": "lat"})
            model_zonal[model] = zm

        # Obs zonal mean
        berkeley = shared["berkeley"]
        obs_clim = climatology(berkeley, self.period)
        obs_zonal = obs_clim.mean("lon")

        # Per-benchmark zonal means (CMIP6, HighResMIP, …)
        from feather.plot.styles import benchmark_color
        benchmarks_zonal = []
        if self.cmip6_enabled:
            for i, bench in enumerate(self.benchmarks):
                zm = self._compute_cmip6_zonal_mean(loader=bench)
                if zm is None:
                    continue
                benchmarks_zonal.append({
                    "label": getattr(bench, "label", "CMIP6 MMM"),
                    "color": getattr(bench, "color", None) or benchmark_color(i),
                    "zonal": zm,
                })
        primary = benchmarks_zonal[0] if benchmarks_zonal else None

        return {
            "models": model_zonal,
            "obs": obs_zonal,
            "benchmarks_zonal": benchmarks_zonal,
            "cmip6_zonal": primary["zonal"] if primary else None,
            "cmip6_info": {},
        }

    def _compute_cmip6_zonal_mean(self, loader=None):
        """Compute benchmark MMM zonal mean temperature (per-benchmark loader)."""
        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return None

        member_pairs = loader.get_member_pairs()
        zonal_fields = []

        for model, variant in member_pairs:
            da = loader.load_var_for_model_var(
                "tas", model, variant=variant, period=self.period,
            )
            if da is None:
                continue
            lon_dim = "lon" if "lon" in da.dims else "longitude"
            zm = da.mean(lon_dim)
            if "latitude" in zm.dims:
                zm = zm.rename({"latitude": "lat"})
            zonal_fields.append(zm)

        if not zonal_fields:
            return None

        aligned = xr.align(*zonal_fields, join="inner")
        return sum(aligned) / len(aligned)

    def _plot_zonal_mean(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot zonal mean temperature profile."""
        fig, ax = plt.subplots(figsize=(6, 8))
        all_models = []

        # Benchmark MMMs (CMIP6, HighResMIP, …)
        for bench in results.get("benchmarks_zonal", []):
            zm = bench["zonal"]
            ax.plot(zm.values - _K_TO_C, zm.lat.values,
                    label=bench["label"], color=bench["color"],
                    linewidth=1.5, linestyle="--")
            all_models.append(bench["label"])

        # Models
        for model, zm in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(zm.values - _K_TO_C, zm.lat.values,
                    label=model, color=color, linewidth=1.5)
            all_models.append(model)

        # Observations
        obs_zm = results["obs"]
        ax.plot(obs_zm.values - _K_TO_C, obs_zm.lat.values,
                label="Berkeley Earth", color=OBS_COLOR, linewidth=2.5)

        ax.set_ylabel("Latitude")
        ax.set_xlabel("Temperature (°C)")
        ax.set_title("2m Temperature \u2014 Zonal Mean")
        ax.set_ylim(-90, 90)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        meta = self._build_metadata(
            title="2m Temperature Zonal Mean Profile",
            figure_id="tas_zonal_mean",
            models=all_models,
            variables=["tas"],
            description=(
                "Zonal mean 2m temperature profile for all models, "
                "Berkeley Earth, and CMIP6 MMM."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="zonal_profile",
            period=self.period,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── Group E: Warming trends ──────────────────────────────────────

    def _compute_trends(self, shared: dict) -> dict[str, Any]:
        """Compute linear T2m trends (°C/decade) on a common 0.25° grid."""
        logger.info("Computing temperature trends...")
        model_coords = shared["model_coords"]
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = self.config.nereus.get("resolution", 0.25)

        # Raw trends on native grids (kept for polar maps)
        model_trends_native: dict[str, xr.DataArray] = {}
        for model, da in shared["model_monthly"].items():
            trend = linear_trend(da.compute()) * 10  # °C/decade
            model_trends_native[model] = trend

        berkeley = shared["berkeley"]
        obs_trend_native = linear_trend(berkeley.compute()) * 10  # °C/decade

        # Regrid everything to common nereus grid
        _trend_interp_cache: dict[int, Any] = {}
        target_lats = None
        target_lons = None

        model_trends_common: dict[str, xr.DataArray] = {}
        for model, trend in model_trends_native.items():
            lon, lat = model_coords[model]
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            n_src = np.asarray(lon).ravel().shape[0]
            if n_src not in _trend_interp_cache:
                regridded, interp = nr.regrid(
                    trend.values.ravel(),
                    lon=np.asarray(lon).ravel(),
                    lat=np.asarray(lat).ravel(),
                    resolution=resolution,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _trend_interp_cache[n_src] = interp
                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]
            else:
                regridded = _trend_interp_cache[n_src](trend.values.ravel())

            model_trends_common[model] = xr.DataArray(
                regridded, dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )

        # Regrid obs to same common grid
        obs_lons_2d, obs_lats_2d = np.meshgrid(
            obs_trend_native.lon.values, obs_trend_native.lat.values,
        )
        if target_lats is None:
            # No models loaded — build from obs
            obs_regridded, obs_interp = nr.regrid(
                obs_trend_native.values.ravel(),
                lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                resolution=resolution,
                influence_radius=influence_radius,
                lon_bounds=(0.0, 360.0),
                as_xarray=True,
            )
            target_lats = obs_interp.target_lat[:, 0]
            target_lons = obs_interp.target_lon[0, :]
        else:
            _, obs_interp = nr.regrid(
                obs_trend_native.values.ravel(),
                lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                resolution=resolution,
                influence_radius=influence_radius,
                lon_bounds=(0.0, 360.0),
                as_xarray=True,
            )
            obs_regridded = obs_interp(obs_trend_native.values.ravel())

        obs_trend_common = xr.DataArray(
            obs_regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

        # Benchmark trends on the common grid (CMIP6, HighResMIP, …)
        from feather.plot.styles import benchmark_color
        benchmark_trends: dict[str, Any] = {}
        benchmark_trend_colors: dict[str, Any] = {}
        cmip6_info = {}
        if self.cmip6_enabled:
            for i, bench in enumerate(self.benchmarks):
                label = getattr(bench, "label", "CMIP6 MMM")
                trend, info = self._compute_cmip6_trends(
                    target_lats, target_lons, resolution, influence_radius,
                    loader=bench,
                )
                if trend is not None:
                    benchmark_trends[label] = trend
                    benchmark_trend_colors[label] = (
                        getattr(bench, "color", None) or benchmark_color(i)
                    )
                    if not cmip6_info:
                        cmip6_info = info

        primary_trend = (
            next(iter(benchmark_trends.values())) if benchmark_trends else None
        )

        return {
            "model_trends": model_trends_common,
            "model_trends_native": model_trends_native,
            "model_coords": model_coords,
            "obs_trend": obs_trend_common,
            "obs_trend_native": obs_trend_native,
            "benchmark_trends": benchmark_trends,
            "benchmark_trend_colors": benchmark_trend_colors,
            "cmip6_trend": primary_trend,
            "cmip6_info": cmip6_info,
        }

    def _compute_cmip6_trends(self, target_lats, target_lons,
                              resolution, influence_radius, loader=None):
        """Compute benchmark MMM trend on the common grid (per-benchmark)."""
        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return None, {}

        from feather.diag.global_biases import GlobalBiases

        member_pairs = loader.get_member_pairs()
        trend_fields = []
        models_used = []
        interp_cache: dict = {}

        for model, variant in member_pairs:
            da = loader.load_var_for_model_var(
                "tas", model, variant=variant,
                period=self.period, time_mean=False,
            )
            if da is None:
                continue
            trend = linear_trend(da.compute()) * 10
            # Regrid to common grid before averaging
            regridded = GlobalBiases._regrid_to_target(
                trend, target_lats, target_lons,
                resolution, influence_radius, interp_cache,
                method=self._regrid_method,
            )
            trend_fields.append(regridded)
            models_used.append(f"{model}/{variant}")

        if not trend_fields:
            return None, {}

        mmm_trend = sum(trend_fields) / len(trend_fields)
        info = {"n_members": len(models_used), "models_used": models_used}
        return mmm_trend, info

    def _plot_trends(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot global + polar trend maps."""
        figures = []
        model_trends = results["model_trends"]
        obs_trend = results["obs_trend"]
        all_models = list(model_trends.keys())

        benchmark_trends = results.get("benchmark_trends", {})

        # Global trend map (combined: obs + model-obs diffs on common grid)
        bias_dict = {}
        for model, trend in model_trends.items():
            bias_dict[model] = trend - obs_trend

        for b_label, b_trend in benchmark_trends.items():
            bias_dict[b_label] = b_trend - obs_trend
            all_models.append(b_label)

        # Symmetric colorbar for obs trend panel (centered on zero)
        obs_vals = np.asarray(obs_trend).ravel()
        obs_vals = obs_vals[np.isfinite(obs_vals)]
        obs_vmax = float(np.percentile(np.abs(obs_vals), 98)) or 0.5

        fig, axes = plot_combined_bias_map(
            obs_trend, bias_dict,
            title="2m Temperature Trends",
            obs_title="Berkeley Earth (°C/decade)",
            cmap="RdBu_r",
            bias_cmap="RdBu_r",
            vmin=-obs_vmax, vmax=obs_vmax,
            units="°C/decade",
            method=self._regrid_method,
        )

        meta = self._build_metadata(
            title="2m Temperature Warming Trends",
            figure_id="tas_trend_combined",
            models=all_models,
            variables=["tas"],
            description=(
                "Linear trends in 2m temperature (°C/decade) over the "
                "analysis period. Obs panel shows Berkeley Earth trends; "
                "bias panels show model-obs trend differences."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="combined_bias_map",
            period=self.period,
            computation_notes="Linear OLS regression per grid point, x10 for °C/decade",
        )
        figures.append((fig, meta))

        # Polar trend maps (Arctic and Antarctic)
        for pole, extent, proj_str, lat_range, fig_id, pole_name in [
            ("Arctic", (-180, 180, 50, 90), "np", (50, 90),
             "tas_trend_arctic", "Arctic (>50\u00b0N)"),
            ("Antarctic", (-180, 180, -90, -50), "sp", (-90, -50),
             "tas_trend_antarctic", "Antarctic (<50\u00b0S)"),
        ]:
            try:
                fig_polar, meta_polar = self._plot_polar_trend(
                    results, pole_name, proj_str, extent, lat_range, fig_id,
                    all_models,
                )
                figures.append((fig_polar, meta_polar))
            except Exception:
                logger.warning(
                    "Polar trend map (%s) failed", pole, exc_info=True,
                )

        return figures

    def _plot_polar_trend(
        self, results, pole_name, proj_str, extent, lat_range, fig_id,
        all_models,
    ):
        """Plot a single polar stereographic trend map."""
        import cartopy.crs as ccrs

        obs_trend = results["obs_trend"]
        model_trends = results["model_trends"]

        # Collect panels: obs + models
        panels = {"Berkeley Earth": obs_trend}
        for model, trend in model_trends.items():
            panels[model] = trend
        for b_label, b_trend in results.get("benchmark_trends", {}).items():
            panels[b_label] = b_trend

        n_panels = len(panels)
        ncols = min(n_panels, 3)
        nrows = max(1, (n_panels + ncols - 1) // ncols)

        if proj_str == "np":
            proj = ccrs.NorthPolarStereo()
        else:
            proj = ccrs.SouthPolarStereo()

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 5 * nrows),
            subplot_kw={"projection": proj},
        )
        if nrows == 1 and ncols == 1:
            axes_flat = [axes]
        else:
            axes_flat = np.asarray(axes).ravel().tolist()

        # Compute shared color range from all panels
        all_vals = []
        for panel_data in panels.values():
            v = np.asarray(panel_data).ravel()
            all_vals.append(v[np.isfinite(v)])
        if all_vals:
            combined = np.concatenate(all_vals)
            vmax = float(np.percentile(np.abs(combined), 98)) or 0.5
        else:
            vmax = 0.5

        interpolator = None
        for i, (label, panel_data) in enumerate(panels.items()):
            if i >= len(axes_flat):
                break

            if "lat" in panel_data.dims and "lon" in panel_data.dims:
                lons_2d, lats_2d = np.meshgrid(
                    panel_data.lon.values, panel_data.lat.values,
                )
                vals = panel_data.values.ravel()
                lons = lons_2d.ravel()
                lats = lats_2d.ravel()
            else:
                vals = np.asarray(panel_data).ravel()
                lons = np.asarray(panel_data.longitude).ravel() if hasattr(panel_data, "longitude") else None
                lats = np.asarray(panel_data.latitude).ravel() if hasattr(panel_data, "latitude") else None
                if lons is None:
                    continue

            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[i], projection=proj_str,
                extent=extent,
                land=True, colorbar=False,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            axes_flat[i].set_title(label)

        # Hide unused axes
        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Shared colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax),
        )
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                     label="°C/decade")

        fig.suptitle(
            f"2m Temperature Trends \u2014 {pole_name}",
            fontsize=14, fontweight="bold", y=0.98,
        )
        fig.subplots_adjust(bottom=0.12, top=0.92)

        meta = self._build_metadata(
            title=f"2m Temperature Trends ({pole_name})",
            figure_id=fig_id,
            models=all_models,
            variables=["tas"],
            description=(
                f"Polar stereographic map of 2m temperature linear trends "
                f"(°C/decade) in the {pole_name} region."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="polar_map",
            spatial_extent=pole_name,
            period=self.period,
            computation_notes="Linear OLS regression per grid point, x10 for °C/decade",
        )
        return fig, meta

    # ── Group F: Taylor diagram ──────────────────────────────────────

    def _compute_taylor(self, shared: dict) -> dict[str, Any]:
        """Compute pattern statistics for Taylor diagram."""
        logger.info("Computing Taylor diagram statistics...")
        berkeley = shared["berkeley"]
        model_monthly = shared["model_monthly"]
        model_coords = shared["model_coords"]

        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )

        # Obs climatologies
        obs_annual = climatology(berkeley, self.period).compute()
        obs_seasonal = seasonal_climatology(berkeley, self.period)
        obs_lats = obs_annual.lat.values
        obs_lons = obs_annual.lon.values
        obs_area = compute_latlon_areas(obs_lats, obs_lons)

        model_stats: dict[str, dict[str, dict]] = {}

        for model in self.config.models:
            if model not in model_monthly:
                continue

            model_data = model_monthly[model]
            lon, lat = model_coords[model]

            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lon, lat = np.meshgrid(lon, lat)

            # Annual
            model_annual = climatology(model_data, self.period).compute()
            regridded, _ = nr.regrid(
                model_annual.values.ravel(),
                lon=np.asarray(lon).ravel(),
                lat=np.asarray(lat).ravel(),
                resolution=abs(float(obs_lats[1] - obs_lats[0])),
                influence_radius=influence_radius,
                lon_bounds=(0.0, 360.0),
                as_xarray=True,
            )
            regridded_da = xr.DataArray(
                regridded, dims=("lat", "lon"),
                coords={"lat": obs_lats, "lon": obs_lons},
            )

            seasons_data = {"ANN": (regridded_da, obs_annual)}

            # Seasonal
            model_seas = seasonal_climatology(model_data, self.period)
            for season in ["DJF", "MAM", "JJA", "SON"]:
                if season in model_seas.data_vars and season in obs_seasonal.data_vars:
                    ms = model_seas[season].compute()
                    ms_regridded, _ = nr.regrid(
                        ms.values.ravel(),
                        lon=np.asarray(lon).ravel(),
                        lat=np.asarray(lat).ravel(),
                        resolution=abs(float(obs_lats[1] - obs_lats[0])),
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    ms_da = xr.DataArray(
                        ms_regridded, dims=("lat", "lon"),
                        coords={"lat": obs_lats, "lon": obs_lons},
                    )
                    seasons_data[season] = (ms_da, obs_seasonal[season].compute())

            model_stats[model] = {}
            for season, (m_field, o_field) in seasons_data.items():
                corr = self._pattern_correlation(m_field, o_field, obs_area)
                std_r = self._std_ratio(m_field, o_field, obs_area)
                model_stats[model][season] = {
                    "corr": corr, "std_ratio": std_r,
                }

        # CMIP6 stats
        cmip6_stats = None
        cmip6_info = {}
        if self.cmip6_enabled:
            cmip6_stats, cmip6_info = self._compute_cmip6_taylor_stats(
                obs_annual, obs_seasonal, obs_lats, obs_lons, obs_area,
            )

        return {
            "model_stats": model_stats,
            "cmip6_stats": cmip6_stats,
            "cmip6_info": cmip6_info,
        }

    def _compute_cmip6_taylor_stats(
        self, obs_annual, obs_seasonal, obs_lats, obs_lons, obs_area,
    ):
        """Compute CMIP6 MMM Taylor stats."""
        if not self.cmip6_enabled:
            return None, {}

        from feather.diag.global_biases import GlobalBiases

        member_pairs = self.cmip6_loader.get_member_pairs()
        influence_radius = self.config.nereus.get("influence_radius", 80_000.0)
        resolution = abs(float(obs_lats[1] - obs_lats[0]))
        interp_cache: dict[tuple, Any] = {}

        annual_fields = []
        seasonal_fields: dict[str, list] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        models_used = []

        for model, variant in member_pairs:
            da = self.cmip6_loader.load_var_for_model_var(
                "tas", model, variant=variant, period=self.period,
            )
            if da is None:
                continue

            regridded = GlobalBiases._regrid_to_target(
                da, obs_lats, obs_lons,
                resolution, influence_radius, interp_cache,
                method=self._regrid_method,
            )
            annual_fields.append(regridded)
            models_used.append(f"{model}/{variant}")

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    "tas", model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is not None:
                    s_regridded = GlobalBiases._regrid_to_target(
                        da_s, obs_lats, obs_lons,
                        resolution, influence_radius, interp_cache,
                        method=self._regrid_method,
                    )
                    seasonal_fields[season].append(s_regridded)

        if not annual_fields:
            return None, {}

        mmm_annual = xr.concat(annual_fields, dim="member").mean("member")
        stats = {}
        stats["ANN"] = {
            "corr": self._pattern_correlation(mmm_annual, obs_annual, obs_area),
            "std_ratio": self._std_ratio(mmm_annual, obs_annual, obs_area),
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if seasonal_fields[season] and season in obs_seasonal.data_vars:
                mmm_s = xr.concat(
                    seasonal_fields[season], dim="member",
                ).mean("member")
                stats[season] = {
                    "corr": self._pattern_correlation(
                        mmm_s, obs_seasonal[season].compute(), obs_area,
                    ),
                    "std_ratio": self._std_ratio(
                        mmm_s, obs_seasonal[season].compute(), obs_area,
                    ),
                }

        info = {"n_members": len(models_used), "models_used": models_used}
        return stats, info

    def _plot_taylor(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        """Plot Taylor diagram."""
        model_stats = results["model_stats"]
        if not model_stats:
            return []

        model_colors = {
            model: self.config.get_model_color(model)
            for model in model_stats
        }

        fig, ax = plot_taylor_diagram(
            model_stats,
            title="Taylor Diagram \u2014 2m Temperature vs Berkeley Earth",
            obs_label="Berkeley Earth",
            cmip6_stats=results.get("cmip6_stats"),
            model_colors=model_colors,
        )

        all_models = list(model_stats.keys())
        if results.get("cmip6_stats"):
            all_models.append("CMIP6 MMM")

        # Flatten stats for metadata
        summary_stats = {}
        for model, seasons in model_stats.items():
            for season, stats in seasons.items():
                summary_stats[f"{model}_{season}"] = stats

        meta = self._build_metadata(
            title="Taylor Diagram \u2014 2m Temperature",
            figure_id="tas_taylor",
            models=all_models,
            variables=["tas"],
            description=(
                "Taylor diagram comparing spatial patterns of 2m "
                "temperature climatology (annual, DJF, JJA) for all "
                "models against Berkeley Earth. Shows pattern correlation "
                "(angular axis) vs normalised standard deviation (radial "
                "axis)."
            ),
            obs_dataset="Berkeley Earth",
            obs_variable="2m temperature",
            plot_type="taylor_diagram",
            period=self.period,
            summary_statistics=summary_stats,
            cmip6_info=results.get("cmip6_info") or None,
        )
        return [(fig, meta)]

    # ── CMIP6 helpers ────────────────────────────────────────────────

    def _compute_cmip6_mmm(self, target_lats, target_lons,
                            obs_clim_common, obs_seasonal_common,
                            common_area, loader=None):
        """Compute benchmark MMM temperature biases (per-benchmark loader)."""
        from feather.diag.global_biases import GlobalBiases

        loader = loader or self.cmip6_loader
        cmip6_data = {}
        cmip6_info = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, Any] = {}

        logger.info("  Computing %s MMM for tas...",
                    getattr(loader, "label", "CMIP6"))
        member_pairs = loader.get_member_pairs()

        annual_fields = []
        seasonal_fields: dict[str, list] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        models_used = []

        for model, variant in member_pairs:
            da = loader.load_var_for_model_var(
                "tas", model, variant=variant, period=self.period,
            )
            if da is None:
                continue

            regridded = GlobalBiases._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            annual_fields.append(regridded)
            models_used.append(f"{model}/{variant}")

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = loader.load_var_for_model_var(
                    "tas", model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is not None:
                    s_regridded = GlobalBiases._regrid_to_target(
                        da_s, target_lats, target_lons,
                        resolution, influence_radius, cmip6_interp_cache,
                        method=self._regrid_method,
                    )
                    seasonal_fields[season].append(s_regridded)

        if not annual_fields:
            return cmip6_data, cmip6_info

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            ),
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if not seasonal_fields[season]:
                continue
            if season not in obs_seasonal_common:
                continue
            s_mmm = xr.concat(
                seasonal_fields[season], dim="member",
            ).mean("member")
            s_bias = s_mmm - obs_seasonal_common[season]
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(s_bias, area=common_area).values
                ),
            }

        return cmip6_data, cmip6_info

    def _compute_cmip6_individual(self, target_lats, target_lons,
                                   obs_clim_common, obs_seasonal_common,
                                   common_area):
        """Compute individual CMIP6 model temperature biases."""
        from feather.diag.global_biases import GlobalBiases

        cmip6_individual_data: dict[str, dict] = {}
        influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        resolution = abs(float(target_lats[1] - target_lats[0]))
        cmip6_interp_cache: dict[tuple, Any] = {}

        logger.info("  Loading individual CMIP6 models for tas...")
        member_pairs = self.cmip6_loader.get_member_pairs()

        for model, variant in member_pairs:
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                "tas", model, variant=variant, period=self.period,
            )
            if da is None:
                continue

            cmip6_common = GlobalBiases._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            cmip6_bias = cmip6_common - obs_clim_common
            bias_gmean = float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            )

            cmip6_individual_data.setdefault("annual", {})[label] = {
                "regrid": cmip6_common,
                "bias": cmip6_bias,
                "bias_gmean": bias_gmean,
            }

            for season in ["DJF", "MAM", "JJA", "SON"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    "tas", model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is None or season not in obs_seasonal_common:
                    continue
                cmip6_s = GlobalBiases._regrid_to_target(
                    da_s, target_lats, target_lons,
                    resolution, influence_radius, cmip6_interp_cache,
                    method=self._regrid_method,
                )
                cmip6_s_bias = cmip6_s - obs_seasonal_common[season]
                cmip6_individual_data.setdefault(season, {})[label] = {
                    "regrid": cmip6_s,
                    "bias": cmip6_s_bias,
                    "bias_gmean": float(
                        latlon_global_mean(
                            cmip6_s_bias, area=common_area,
                        ).values
                    ),
                }

        return cmip6_individual_data

    @staticmethod
    def _mmm_from_individual(cmip6_individual_data, obs_clim_common,
                              obs_seasonal_common, common_area):
        """Derive MMM from already-regridded individual CMIP6 fields."""
        cmip6_data = {}

        if "annual" not in cmip6_individual_data:
            return cmip6_data, {}

        annual_members = cmip6_individual_data["annual"]
        models_used = list(annual_members.keys())
        annual_fields = [m["regrid"] for m in annual_members.values()]

        mmm = xr.concat(annual_fields, dim="member").mean("member")
        cmip6_bias = mmm - obs_clim_common
        cmip6_data["annual"] = {
            "regrid": mmm,
            "bias": cmip6_bias,
            "bias_gmean": float(
                latlon_global_mean(cmip6_bias, area=common_area).values
            ),
        }

        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season not in cmip6_individual_data:
                continue
            if season not in obs_seasonal_common:
                continue
            s_fields = [
                m["regrid"]
                for m in cmip6_individual_data[season].values()
            ]
            s_mmm = xr.concat(s_fields, dim="member").mean("member")
            s_bias = s_mmm - obs_seasonal_common[season]
            cmip6_data[season] = {
                "regrid": s_mmm,
                "bias": s_bias,
                "bias_gmean": float(
                    latlon_global_mean(s_bias, area=common_area).values
                ),
            }

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }
        return cmip6_data, cmip6_info

    # ── Colorbar range computation ───────────────────────────────────

    @staticmethod
    def _compute_colorbar_ranges(
        model_results, obs_clim_common, obs_seasonal_common,
        cmip6_data=None, cmip6_individual_data=None, benchmark_data=None,
    ):
        """Compute shared colorbar ranges across all models per period."""
        ranges: dict[str, dict] = {}
        cmip6_data = cmip6_data or {}
        cmip6_individual_data = cmip6_individual_data or {}
        benchmark_data = benchmark_data or {}

        def _finite_vals(arrays):
            parts = []
            for a in arrays:
                v = np.asarray(a).ravel()
                parts.append(v[np.isfinite(v)])
            return np.concatenate(parts)

        def _percentile_range(arrays):
            vals = _finite_vals(arrays)
            return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

        def _bias_max(arrays):
            vals = _finite_vals(arrays)
            return float(np.percentile(np.abs(vals), 98)) or 1.0

        # Annual
        field_arrays = [mr["annual_regrid"] for mr in model_results.values()]
        field_arrays.append(obs_clim_common)
        bias_arrays = [mr["annual_bias"] for mr in model_results.values()]
        if "annual" in cmip6_data:
            field_arrays.append(cmip6_data["annual"]["regrid"])
            bias_arrays.append(cmip6_data["annual"]["bias"])
        if "annual" in cmip6_individual_data:
            for member_data in cmip6_individual_data["annual"].values():
                bias_arrays.append(member_data["bias"])
        for b_data in benchmark_data.values():
            if "annual" in b_data:
                field_arrays.append(b_data["annual"]["regrid"])
                bias_arrays.append(b_data["annual"]["bias"])
        vmin, vmax = _percentile_range(field_arrays)
        ranges["annual"] = {
            "vmin": vmin, "vmax": vmax,
            "bias_vmax": _bias_max(bias_arrays),
        }

        # Seasonal
        for season in ["DJF", "MAM", "JJA", "SON"]:
            s_fields = [
                mr["seasonal_regrids"][season]
                for mr in model_results.values()
                if season in mr["seasonal_regrids"]
            ]
            s_biases = [
                mr["seasonal_biases"][season]
                for mr in model_results.values()
                if season in mr["seasonal_biases"]
            ]
            if not s_fields:
                continue
            if season in obs_seasonal_common:
                s_fields.append(obs_seasonal_common[season])
            if season in cmip6_data:
                s_fields.append(cmip6_data[season]["regrid"])
                s_biases.append(cmip6_data[season]["bias"])
            if season in cmip6_individual_data:
                for member_data in cmip6_individual_data[season].values():
                    s_biases.append(member_data["bias"])
            for b_data in benchmark_data.values():
                if season in b_data:
                    s_fields.append(b_data[season]["regrid"])
                    s_biases.append(b_data[season]["bias"])
            vmin, vmax = _percentile_range(s_fields)
            ranges[season] = {
                "vmin": vmin, "vmax": vmax,
                "bias_vmax": _bias_max(s_biases) if s_biases else 1.0,
            }

        return ranges

    # ── Statistics helpers ───────────────────────────────────────────

    @staticmethod
    def _pattern_correlation(model_field, obs_field, area):
        """Compute area-weighted spatial pattern correlation."""
        area_da = xr.DataArray(
            area, dims=("lat", "lon"),
            coords={"lat": model_field.lat, "lon": model_field.lon},
        )
        m = model_field.values.ravel()
        o = obs_field.values.ravel()
        w = np.asarray(area_da).ravel()

        valid = np.isfinite(m) & np.isfinite(o)
        m, o, w = m[valid], o[valid], w[valid]

        m_mean = np.average(m, weights=w)
        o_mean = np.average(o, weights=w)
        m_anom = m - m_mean
        o_anom = o - o_mean

        cov = np.average(m_anom * o_anom, weights=w)
        m_std = np.sqrt(np.average(m_anom ** 2, weights=w))
        o_std = np.sqrt(np.average(o_anom ** 2, weights=w))

        if m_std == 0 or o_std == 0:
            return 0.0
        return float(cov / (m_std * o_std))

    @staticmethod
    def _std_ratio(model_field, obs_field, area):
        """Compute ratio of model to obs spatial standard deviation."""
        area_flat = np.asarray(area).ravel()
        m = model_field.values.ravel()
        o = obs_field.values.ravel()

        valid = np.isfinite(m) & np.isfinite(o)
        m, o, w = m[valid], o[valid], area_flat[valid]

        m_std = np.sqrt(np.average((m - np.average(m, weights=w)) ** 2, weights=w))
        o_std = np.sqrt(np.average((o - np.average(o, weights=w)) ** 2, weights=w))

        if o_std == 0:
            return float("inf")
        return float(m_std / o_std)

    @staticmethod
    def _rmse(model_field, obs_field, area):
        """Compute area-weighted RMSE."""
        area_flat = np.asarray(area).ravel()
        m = model_field.values.ravel()
        o = obs_field.values.ravel()

        valid = np.isfinite(m) & np.isfinite(o)
        m, o, w = m[valid], o[valid], area_flat[valid]

        return float(np.sqrt(np.average((m - o) ** 2, weights=w)))

    @staticmethod
    def _regional_mean_bias(bias_field, lats, area, lat_min, lat_max):
        """Compute area-weighted mean bias in a latitude band."""
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if not lat_mask.any():
            return 0.0
        bias_sub = bias_field.isel(lat=lat_mask)
        area_sub = area[lat_mask, :]
        area_da = xr.DataArray(
            area_sub, dims=("lat", "lon"),
            coords={"lat": bias_sub.lat, "lon": bias_sub.lon},
        )
        return float(bias_sub.weighted(area_da).mean().values)

    @staticmethod
    def _compute_summary_stats(model_field, obs_field, bias_field, lats, area):
        """Compute all statistics for metadata."""
        corr = TemperatureBerkeley._pattern_correlation(
            model_field, obs_field, area,
        )
        std_r = TemperatureBerkeley._std_ratio(model_field, obs_field, area)
        rmse = TemperatureBerkeley._rmse(model_field, obs_field, area)
        global_bias = float(
            latlon_global_mean(bias_field, area=area).values
        )
        arctic_bias = TemperatureBerkeley._regional_mean_bias(
            bias_field, lats, area, lat_min=60, lat_max=90,
        )
        tropical_bias = TemperatureBerkeley._regional_mean_bias(
            bias_field, lats, area, lat_min=-30, lat_max=30,
        )
        antarctic_bias = TemperatureBerkeley._regional_mean_bias(
            bias_field, lats, area, lat_min=-90, lat_max=-60,
        )
        return {
            "pattern_correlation": corr,
            "std_ratio": std_r,
            "rmse": rmse,
            "global_mean_bias": global_bias,
            "arctic_bias": arctic_bias,
            "tropical_bias": tropical_bias,
            "antarctic_bias": antarctic_bias,
        }


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    import pandas as pd

    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
