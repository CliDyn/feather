"""Ocean SST evaluation diagnostic.

Compares DestinE high-resolution models against ESA-CCI L4 v3.0.1 SST
satellite observations.  Produces bias maps (annual, DJF, JJA), global-
mean time series, seasonal cycle, and zonal mean profile figures.

All data is converted from Kelvin to degrees Celsius for display.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.styles import OBS_COLOR
from feather.util.spatial import (
    latlon_global_mean,
    zonal_mean,
)
from feather.util.temporal import (
    annual_mean,
    climatology,
    monthly_climatology,
    seasonal_climatology,
)

logger = logging.getLogger(__name__)

_K_TO_C = 273.15


def _to_celsius(da):
    """Convert Kelvin DataArray to Celsius."""
    return da - _K_TO_C


def _ocean_global_mean(da):
    """Cosine-latitude-weighted mean for regular lat/lon ocean data.

    NaN (land) cells are automatically excluded by xarray's
    ``weighted().mean()``.
    """
    lat_name = "lat" if "lat" in da.dims else "latitude"
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean([lat_name, _lon_name(da)])


def _lon_name(da):
    """Return the longitude dimension name."""
    return "lon" if "lon" in da.dims else "longitude"


@register
class OceanSST(DiagnosticBase):
    """Ocean SST evaluation against ESA-CCI L4 v3.0.1.

    Produces 6 figures across 4 groups:
    A) Bias maps (3): annual, DJF, JJA
    B) Time series (1): global-mean SST over time
    C) Seasonal cycle (1): 12-month climatological cycle
    D) Zonal mean (1): latitude profile of SST
    """

    name = "ocean_sst"
    title = "Ocean SST Evaluation"
    domain = "o2d"
    variables = ["tos"]
    group = "ocean_surface"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self.ocean_influence_radius = self.config.nereus.get(
            "ocean_influence_radius", 20_000.0,
        )

    # ── Orchestration (per-figure-group incremental) ──────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-figure-group: compute -> plot -> save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        out = self.output_dir

        # Determine which groups need computation
        bias_ids = [f"sst_{p}_bias_combined" for p in ("annual", "djf", "jja")]
        need_a = not skip_existing or not all(
            self._figure_exists(f) for f in bias_ids
        )
        need_b = not skip_existing or not self._figure_exists("sst_timeseries")
        need_c = not skip_existing or not self._figure_exists(
            "sst_seasonal_cycle"
        )
        need_d = not skip_existing or not self._figure_exists("sst_zonal_mean")

        # Collect already-existing paths
        if not need_a:
            logger.info("Skipping bias maps -- figures exist")
            saved.extend([
                (out / f"{f}.png", out / f"{f}.json") for f in bias_ids
            ])
        if not need_b:
            logger.info("Skipping time series -- figure exists")
            saved.append((
                out / "sst_timeseries.png", out / "sst_timeseries.json",
            ))
        if not need_c:
            logger.info("Skipping seasonal cycle -- figure exists")
            saved.append((
                out / "sst_seasonal_cycle.png",
                out / "sst_seasonal_cycle.json",
            ))
        if not need_d:
            logger.info("Skipping zonal mean -- figure exists")
            saved.append((
                out / "sst_zonal_mean.png", out / "sst_zonal_mean.json",
            ))

        if not any([need_a, need_b, need_c, need_d]):
            logger.info(
                "Diagnostic %s complete -- all figures exist", self.name,
            )
            return saved

        # Load shared model data (once)
        model_monthly, model_coords = self._load_model_data()

        # Group A: Bias maps
        if need_a:
            results = self._compute_bias_maps(model_monthly, model_coords)
            for fig, meta in self._plot_bias_maps(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group B: Time series
        if need_b:
            results = self._compute_timeseries(model_monthly)
            for fig, meta in self._plot_timeseries(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group C: Seasonal cycle
        if need_c:
            results = self._compute_seasonal_cycle(model_monthly)
            for fig, meta in self._plot_seasonal_cycle(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group D: Zonal mean
        if need_d:
            results = self._compute_zonal_mean(model_monthly, model_coords)
            for fig, meta in self._plot_zonal_mean(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        logger.info(
            "Diagnostic %s complete -- %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Abstract interface (thin wrappers for backward compat) ────────

    def compute(self) -> dict[str, Any]:
        """Compute all results (backward compat wrapper)."""
        model_monthly, model_coords = self._load_model_data()
        return {
            "bias_maps": self._compute_bias_maps(model_monthly, model_coords),
            "timeseries": self._compute_timeseries(model_monthly),
            "seasonal_cycle": self._compute_seasonal_cycle(model_monthly),
            "zonal_mean": self._compute_zonal_mean(
                model_monthly, model_coords,
            ),
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figures (backward compat wrapper)."""
        figures = []
        figures.extend(self._plot_bias_maps(results["bias_maps"]))
        figures.extend(self._plot_timeseries(results["timeseries"]))
        figures.extend(self._plot_seasonal_cycle(results["seasonal_cycle"]))
        figures.extend(self._plot_zonal_mean(results["zonal_mean"]))
        return figures

    # ── Shared data loading ───────────────────────────────────────────

    def _load_model_data(self):
        """Load model data for all available models.

        Returns
        -------
        model_monthly : dict[str, xr.DataArray]
            Model name -> monthly SST in Celsius (time, values).
        model_coords : dict[str, tuple]
            Model name -> (lon, lat) coordinate arrays.
        """
        # CMOR (EERIE) tos is already in °C; DestinE tos is in Kelvin
        is_cmor = self.config.get_data_source_type() == "cmor"
        model_monthly = {}
        model_coords = {}
        for model in self.config.models:
            try:
                da = self._load_model_var(
                    model, "tos", period=self.period,
                )
                lon, lat = self._load_model_coords(model, "tos")
            except (KeyError, FileNotFoundError):
                logger.warning("tos not available for %s", model)
                continue
            model_monthly[model] = da if is_cmor else _to_celsius(da)
            model_coords[model] = (lon, lat)
        return model_monthly, model_coords

    def _load_obs_timemean(self):
        """Load ESA-CCI annual time-mean SST in Celsius."""
        da = self.obs_loader.load_esa_cci("timemean")
        # Squeeze singleton dims (e.g. length-1 time in pre-computed file)
        da = da.squeeze(drop=True)
        return _to_celsius(da)

    def _load_obs_ymonmean(self):
        """Load ESA-CCI monthly climatology SST in Celsius."""
        da = self.obs_loader.load_esa_cci("ymonmean")
        return _to_celsius(da)

    def _load_obs_monthly(self):
        """Load ESA-CCI full monthly series SST in Celsius."""
        da = self.obs_loader.load_esa_cci("analysed_sst", period=self.period)
        return _to_celsius(da)

    # ── Group A: Bias maps ────────────────────────────────────────────

    def _compute_bias_maps(self, model_monthly, model_coords):
        """Compute annual/DJF/JJA bias maps on a common grid."""
        import nereus as nr

        from feather.util.spatial import compute_latlon_areas

        logger.info("Computing SST bias maps...")

        # Load obs
        obs_timemean = self._load_obs_timemean()  # annual mean
        obs_ymonmean = self._load_obs_ymonmean()  # monthly clim

        # Compute DJF/JJA obs from monthly climatology
        # ymonmean has month dimension (1-12)
        if "time" in obs_ymonmean.dims:
            # Group by month for seasonal extraction
            obs_djf = obs_ymonmean.sel(
                time=obs_ymonmean["time.month"].isin([12, 1, 2])
            ).mean("time")
            obs_jja = obs_ymonmean.sel(
                time=obs_ymonmean["time.month"].isin([6, 7, 8])
            ).mean("time")
        elif "month" in obs_ymonmean.dims:
            obs_djf = obs_ymonmean.sel(month=[12, 1, 2]).mean("month")
            obs_jja = obs_ymonmean.sel(month=[6, 7, 8]).mean("month")
        else:
            # Squeeze single-time
            obs_djf = obs_timemean
            obs_jja = obs_timemean

        resolution = self.config.nereus.get("resolution", 0.25)

        # Build interpolators once
        model_interpolator = None
        obs_interpolator = None
        target_lats = None
        target_lons = None
        common_area = None

        # Results per period
        periods_data = {
            "annual": {"obs": obs_timemean},
            "djf": {"obs": obs_djf},
            "jja": {"obs": obs_jja},
        }
        model_results = {}

        for model in model_monthly:
            logger.info("  Regridding model: %s", model)
            da = model_monthly[model]
            lon, lat = model_coords[model]

            # For latlon grids, meshgrid 1D coord arrays to per-pixel arrays
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                regrid_lon, regrid_lat = np.meshgrid(lon, lat)
            else:
                regrid_lon, regrid_lat = np.asarray(lon), np.asarray(lat)

            # Compute model climatologies
            model_clim_annual = climatology(da, self.period).compute()
            model_seasonal = seasonal_climatology(da, self.period)
            model_djf = (
                model_seasonal["DJF"].compute()
                if "DJF" in model_seasonal
                else model_clim_annual
            )
            model_jja = (
                model_seasonal["JJA"].compute()
                if "JJA" in model_seasonal
                else model_clim_annual
            )

            # Build model interpolator once
            if model_interpolator is None:
                _, model_interpolator = nr.regrid(
                    model_clim_annual.values.ravel(),
                    lon=regrid_lon.ravel(), lat=regrid_lat.ravel(),
                    resolution=resolution,
                    influence_radius=self.ocean_influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                target_lats = model_interpolator.target_lat[:, 0]
                target_lons = model_interpolator.target_lon[0, :]

                # Compute common area weights
                common_area = compute_latlon_areas(target_lats, target_lons)

                # Regrid obs to common grid (once)
                lat_name = (
                    "lat" if "lat" in obs_timemean.coords else "latitude"
                )
                lon_name = (
                    "lon" if "lon" in obs_timemean.coords else "longitude"
                )
                obs_lats = obs_timemean[lat_name].values
                obs_lons = obs_timemean[lon_name].values
                obs_lons_2d, obs_lats_2d = np.meshgrid(obs_lats, obs_lons)
                # Note: meshgrid(lats, lons) creates (nlon, nlat) so we
                # need meshgrid(lons, lats) with indexing='ij' for (nlat, nlon)
                obs_lons_2d, obs_lats_2d = np.meshgrid(
                    obs_lons, obs_lats,
                )

                _, obs_interpolator = nr.regrid(
                    obs_timemean.values.ravel(),
                    lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                    resolution=resolution,
                    influence_radius=self.ocean_influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )

                # Regrid each obs period to common grid
                for pkey in periods_data:
                    obs_field = periods_data[pkey]["obs"]
                    obs_common = xr.DataArray(
                        obs_interpolator(obs_field.values.ravel()),
                        dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    )
                    periods_data[pkey]["obs_common"] = obs_common

            # Regrid model to common grid
            annual_regrid = xr.DataArray(
                model_interpolator(model_clim_annual.values.ravel()),
                dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )
            djf_regrid = xr.DataArray(
                model_interpolator(model_djf.values.ravel()),
                dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )
            jja_regrid = xr.DataArray(
                model_interpolator(model_jja.values.ravel()),
                dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )

            model_results[model] = {}
            for pkey, regrid in [
                ("annual", annual_regrid),
                ("djf", djf_regrid),
                ("jja", jja_regrid),
            ]:
                obs_common = periods_data[pkey]["obs_common"]
                bias = regrid - obs_common
                bias_gmean = float(
                    latlon_global_mean(bias, area=common_area).values
                )
                rmse = float(np.sqrt(
                    latlon_global_mean(bias ** 2, area=common_area).values
                ))
                model_results[model][pkey] = {
                    "regrid": regrid,
                    "bias": bias,
                    "bias_gmean": bias_gmean,
                    "rmse": rmse,
                }

        return {
            "models": model_results,
            "periods": periods_data,
            "common_area": common_area,
        }

    def _plot_bias_maps(self, results):
        """Plot combined bias maps for annual/DJF/JJA."""
        from feather.plot.maps import plot_combined_bias_map

        figures = []
        periods_data = results["periods"]

        for pkey, plabel in [
            ("annual", "Annual Mean"),
            ("djf", "DJF"),
            ("jja", "JJA"),
        ]:
            obs_common = periods_data[pkey].get("obs_common")
            if obs_common is None:
                continue

            bias_dict = {}
            summary_stats = {}
            all_models = []

            for model, mdata in results["models"].items():
                if pkey in mdata:
                    bias_dict[model] = mdata[pkey]["bias"]
                    summary_stats[model] = {
                        "global_mean_bias": mdata[pkey]["bias_gmean"],
                        "rmse": mdata[pkey]["rmse"],
                    }
                    all_models.append(model)

            if not bias_dict:
                continue

            try:
                import cmocean
                obs_cmap = cmocean.cm.thermal
            except ImportError:
                obs_cmap = "RdYlBu_r"

            fig, axes = plot_combined_bias_map(
                obs_common, bias_dict,
                title=f"Sea Surface Temperature {plabel}",
                obs_title="ESA-CCI",
                cmap=obs_cmap,
                bias_cmap="RdBu_r",
                units="\u00b0C",
                land=True,
            )

            meta = self._build_metadata(
                title=f"SST {plabel} Bias",
                figure_id=f"sst_{pkey}_bias_combined",
                models=all_models,
                description=(
                    f"{plabel} SST climatology and model biases relative to "
                    f"ESA-CCI L4 v3.0.1 satellite observations."
                ),
                plot_type="combined_bias_map",
                period=self.period,
                obs_dataset="ESA-CCI L4 v3.0.1",
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures

    # ── Group B: Time series ──────────────────────────────────────────

    def _compute_timeseries(self, model_monthly):
        """Compute global-mean SST time series for models and obs."""
        logger.info("Computing SST time series...")
        model_ts = {}

        for model, da in model_monthly.items():
            ts = self._model_global_mean(da, model).compute()
            model_ts[model] = ts

        # Obs time series (cos-lat weighted)
        obs_ts = None
        try:
            obs_monthly = self._load_obs_monthly()
            obs_ts = _ocean_global_mean(obs_monthly)
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("ESA-CCI monthly time series failed: %s", e)

        return {"models": model_ts, "obs": obs_ts}

    def _plot_timeseries(self, results):
        """Plot SST global-mean time series."""
        fig, ax = plt.subplots(figsize=(12, 5))
        all_models = []

        # Monthly semi-transparent background
        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            time_vals = _to_plot_time(ts.time.values)
            ax.plot(time_vals, ts.values, color=color, alpha=0.3,
                    linewidth=0.7)

        if results.get("obs") is not None:
            obs_ts = results["obs"]
            time_vals = _to_plot_time(obs_ts.time.values)
            ax.plot(time_vals, obs_ts.values, color=OBS_COLOR, alpha=0.3,
                    linewidth=0.7)

        # Annual thick foreground
        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values, label=model, color=color,
                    linewidth=2.0)
            all_models.append(model)

        if results.get("obs") is not None:
            obs_annual = annual_mean(results["obs"])
            time_vals = _to_plot_time(obs_annual.time.values)
            ax.plot(time_vals, obs_annual.values, label="ESA-CCI",
                    color=OBS_COLOR, linewidth=2.5)

        ax.set_title("Global Mean Sea Surface Temperature")
        ax.set_ylabel("SST (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append("ESA-CCI")

        meta = self._build_metadata(
            title="SST Global Mean Time Series",
            figure_id="sst_timeseries",
            models=all_models,
            description=(
                "Global-mean SST time series for DestinE models and ESA-CCI "
                "observations. Monthly values as semi-transparent lines, "
                "annual means as thick lines. Units: degrees Celsius."
            ),
            plot_type="timeseries",
            period=self.period,
            obs_dataset="ESA-CCI L4 v3.0.1",
        )
        return [(fig, meta)]

    # ── Group C: Seasonal cycle ───────────────────────────────────────

    def _compute_seasonal_cycle(self, model_monthly):
        """Compute 12-month climatological cycle for models and obs."""
        logger.info("Computing SST seasonal cycle...")
        model_cycles = {}

        for model, da in model_monthly.items():
            # Monthly climatology then global mean
            mon_clim = monthly_climatology(da, self.period)
            cycle = self._model_global_mean(mon_clim, model).compute()
            model_cycles[model] = cycle

        # Obs seasonal cycle (cos-lat weighted per month)
        obs_cycle = None
        try:
            obs_ymonmean = self._load_obs_ymonmean()
            if "time" in obs_ymonmean.dims:
                months = obs_ymonmean["time.month"].values
                vals = []
                for m in range(1, 13):
                    month_da = obs_ymonmean.sel(
                        time=obs_ymonmean["time.month"] == m,
                    ).squeeze("time", drop=True)
                    vals.append(float(_ocean_global_mean(month_da).values))
                obs_cycle = xr.DataArray(
                    vals, dims="month",
                    coords={"month": np.arange(1, 13)},
                )
            elif "month" in obs_ymonmean.dims:
                vals = []
                for m in obs_ymonmean.month.values:
                    month_da = obs_ymonmean.sel(month=m)
                    vals.append(float(_ocean_global_mean(month_da).values))
                obs_cycle = xr.DataArray(
                    vals, dims="month",
                    coords={"month": obs_ymonmean.month.values},
                )
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("ESA-CCI seasonal cycle failed: %s", e)

        return {"models": model_cycles, "obs": obs_cycle}

    def _plot_seasonal_cycle(self, results):
        """Plot SST seasonal cycle."""
        fig, ax = plt.subplots(figsize=(8, 5))
        months = np.arange(1, 13)
        month_labels = [
            "J", "F", "M", "A", "M", "J",
            "J", "A", "S", "O", "N", "D",
        ]
        all_models = []

        for model, cycle in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(months, cycle.values, marker="o", label=model,
                    color=color)
            all_models.append(model)

        if results.get("obs") is not None:
            obs_cycle = results["obs"]
            ax.plot(months, obs_cycle.values, marker="s", label="ESA-CCI",
                    color=OBS_COLOR, linewidth=2)

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title("SST Seasonal Cycle")
        ax.set_ylabel("SST (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append("ESA-CCI")

        meta = self._build_metadata(
            title="SST Seasonal Cycle",
            figure_id="sst_seasonal_cycle",
            models=all_models,
            description=(
                "Monthly climatological cycle of global-mean SST for DestinE "
                "models and ESA-CCI observations. Units: degrees Celsius."
            ),
            plot_type="seasonal_cycle",
            period=self.period,
            obs_dataset="ESA-CCI L4 v3.0.1",
        )
        return [(fig, meta)]

    # ── Group D: Zonal mean ───────────────────────────────────────────

    def _compute_zonal_mean(self, model_monthly, model_coords):
        """Compute zonal mean SST profiles for models and obs."""
        logger.info("Computing SST zonal mean profiles...")
        model_zonal = {}

        for model, da in model_monthly.items():
            lon, lat = model_coords[model]
            clim = climatology(da, self.period).compute()
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type == "healpix":
                zm = zonal_mean(clim, lat)
            else:
                # Latlon: simple longitude mean
                lon_dim = "lon" if "lon" in clim.dims else "longitude"
                zm = clim.mean(lon_dim)
                # Rename lat dim for consistent plotting
                lat_dim = "lat" if "lat" in zm.dims else "latitude"
                if lat_dim != "lat":
                    zm = zm.rename({lat_dim: "lat"})
            model_zonal[model] = zm

        # Obs zonal mean (mean over lon, NaN excluded)
        obs_zonal = None
        try:
            obs_timemean = self._load_obs_timemean()
            lon_name_obs = _lon_name(obs_timemean)
            obs_zonal = obs_timemean.mean(lon_name_obs)
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("ESA-CCI zonal mean failed: %s", e)

        return {"models": model_zonal, "obs": obs_zonal}

    def _plot_zonal_mean(self, results):
        """Plot SST zonal mean profile."""
        fig, ax = plt.subplots(figsize=(6, 8))
        all_models = []

        for model, zm in results["models"].items():
            color = self.config.get_model_color(model)
            ax.plot(zm.values, zm.lat.values, label=model, color=color)
            all_models.append(model)

        if results.get("obs") is not None:
            obs_zm = results["obs"]
            lat_name = "lat" if "lat" in obs_zm.coords else "latitude"
            ax.plot(obs_zm.values, obs_zm[lat_name].values,
                    label="ESA-CCI", color=OBS_COLOR, linewidth=2)

        ax.set_title("SST Zonal Mean")
        ax.set_xlabel("SST (\u00b0C)")
        ax.set_ylabel("Latitude")
        ax.set_ylim(-90, 90)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append("ESA-CCI")

        meta = self._build_metadata(
            title="SST Zonal Mean Profile",
            figure_id="sst_zonal_mean",
            models=all_models,
            description=(
                "Zonal mean SST profile for DestinE models and ESA-CCI "
                "observations. Latitude on y-axis, SST (degrees Celsius) on "
                "x-axis."
            ),
            plot_type="zonal_profile",
            period=self.period,
            obs_dataset="ESA-CCI L4 v3.0.1",
        )
        return [(fig, meta)]


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
