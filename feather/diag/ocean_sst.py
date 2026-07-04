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


def _needs_celsius_conversion(da, model_src: str) -> bool:
    """Return True if *da* needs K→°C conversion before comparison.

    Decision order:
    1. ``units`` attribute says "kelvin" / "K"  → True
    2. ``units`` attribute says "degC" / "°C"   → False
    3. No usable units attr: fall back to data-source heuristic
       (CMOR stores tos in °C; all other backends store it in K).
    """
    units = da.attrs.get("units", "").strip().lower()
    if units in ("k", "kelvin"):
        return True
    if units in ("c", "celsius", "degc", "°c"):
        return False
    # Heuristic fallback: CMOR tos is in °C, everything else in K
    return model_src != "cmor"


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

    #: Obs labels/keys — overridden by subclasses (e.g. sst_hadisst) to
    #: evaluate the same SST fields against a different reference dataset.
    _obs_label = "ESA-CCI"
    _obs_dataset_name = "ESA-CCI L4 v3.0.1"
    #: Obs key passed to ocean_bias for the benchmark-bias NetCDF (None →
    #: OCEAN_OBS["tos"] = ESA_CCI).
    _ocean_bias_obs = None

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks,
                         save_netcdf=save_netcdf)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self.ocean_influence_radius = self.config.nereus.get(
            "ocean_influence_radius", 20_000.0,
        )
        self._influence_radius = self.config.nereus.get(
            "influence_radius", 80_000.0,
        )
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # ── Orchestration (per-figure-group incremental) ──────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-figure-group: compute -> plot -> save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []
        out = self.output_dir

        # Benchmark (CMIP6/HighResMIP) bias NetCDFs for Added Value reuse.
        from feather.diag import ocean_bias
        ocean_bias.maybe_export_ocean_bias(
            self, ["tos"], want_individual=self.cmip6_individual,
            skip_existing=skip_existing, obs_name=self._ocean_bias_obs,
        )

        # Determine which groups need computation
        bias_ids = [
            f"sst_{p}_{suffix}" for p in ("annual", "djf", "jja")
            for suffix in ("bias_combined", "ens_bias_combined")
        ]
        need_a = not skip_existing or not all(
            self._figure_exists(f) for f in bias_ids
        )
        need_b = not skip_existing or not self._figure_exists("sst_timeseries")
        need_c = not skip_existing or not self._figure_exists(
            "sst_seasonal_cycle"
        )
        need_d = not skip_existing or not self._figure_exists("sst_zonal_mean")
        extra_ids = self._extra_figure_ids()
        need_extra = bool(extra_ids) and (
            not skip_existing
            or not all(self._figure_exists(f) for f in extra_ids)
        )
        if extra_ids and not need_extra:
            logger.info("Skipping extra groups -- figures exist")
            saved.extend([
                (out / f"{f}.png", out / f"{f}.json") for f in extra_ids
            ])

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

        if not any([need_a, need_b, need_c, need_d, need_extra]):
            logger.info(
                "Diagnostic %s complete -- all figures exist", self.name,
            )
            return saved

        # Load shared model data (once)
        model_monthly, model_coords = self._load_model_data()

        # Group A: Bias maps (per-model + ensemble mean/median)
        if need_a:
            results = self._compute_bias_maps(model_monthly, model_coords)
            self._maybe_export_netcdf(results, "sst_bias")
            for fig, meta in self._plot_bias_maps(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))
            for fig, meta in self._plot_ens_bias_maps(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group B: Time series
        if need_b:
            results = self._compute_timeseries(model_monthly)
            self._maybe_export_netcdf(results, "sst_timeseries")
            for fig, meta in self._plot_timeseries(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group C: Seasonal cycle
        if need_c:
            results = self._compute_seasonal_cycle(model_monthly)
            self._maybe_export_netcdf(results, "sst_seasonal_cycle")
            for fig, meta in self._plot_seasonal_cycle(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group D: Zonal mean
        if need_d:
            results = self._compute_zonal_mean(model_monthly, model_coords)
            self._maybe_export_netcdf(results, "sst_zonal_mean")
            for fig, meta in self._plot_zonal_mean(results):
                saved.append(self._save(fig, meta, meta["figure_id"]))

        # Extra groups (subclass hook: e.g. sst_hadisst trends + Taylor)
        if need_extra:
            saved.extend(self._extra_groups(
                model_monthly, model_coords, skip_existing))

        logger.info(
            "Diagnostic %s complete -- %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Subclass extension hook (extra figure groups) ─────────────────

    def _extra_figure_ids(self) -> list[str]:
        """Figure IDs for subclass-added groups (default none)."""
        return []

    def _extra_groups(self, model_monthly, model_coords, skip_existing):
        """Compute + plot subclass-added figure groups (default none)."""
        return []

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
        # CMOR (EERIE) tos is already in °C; DestinE/GRIB tos is in Kelvin
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
            model_src = self.config.get_model_data_source_type(model)
            convert = _needs_celsius_conversion(da, model_src)
            da_c = _to_celsius(da) if convert else da
            logger.info(
                "  %s tos: src=%s units_attr=%s K→°C=%s",
                model, model_src,
                da.attrs.get("units", "?"),
                convert,
            )
            model_monthly[model] = da_c
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
        obs_timemean = self._load_obs_timemean()  # annual mean, in °C
        obs_ymonmean = self._load_obs_ymonmean()  # monthly clim, in °C
        logger.info(
            "  ESA-CCI timemean after K→°C: min=%.2f max=%.2f",
            float(np.nanmin(obs_timemean.values)),
            float(np.nanmax(obs_timemean.values)),
        )
        if float(np.nanmin(obs_timemean.values)) > 100:
            logger.warning(
                "  ESA-CCI timemean min=%.1f looks like Kelvin — "
                "_to_celsius may not have been applied!",
                float(np.nanmin(obs_timemean.values)),
            )

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

        # Cache interpolators per source grid size
        _model_interp_cache: dict[int, Any] = {}
        obs_interpolator = None
        target_lats = None
        target_lons = None
        common_area = None
        n_roll = 0

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

            # Convert to -180..180 so Delaunay triangulation does not
            # produce a NaN stripe at the prime meridian (0°/360° gap).
            regrid_lon = np.where(
                regrid_lon > 180, regrid_lon - 360, regrid_lon,
            )

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

            # Build/reuse interpolator keyed by source grid size.
            # Use the regular influence_radius for model data — the
            # ocean_influence_radius (20 km) is too small for coarse
            # grids (e.g. GRIB TCO399 ~25 km) and leaves NaN holes.
            n_src = regrid_lon.ravel().shape[0]
            if n_src not in _model_interp_cache:
                _, interp = nr.regrid(
                    model_clim_annual.values.ravel(),
                    lon=regrid_lon.ravel(), lat=regrid_lat.ravel(),
                    resolution=resolution,
                    method=self._regrid_method,
                    influence_radius=self._influence_radius,
                    lon_bounds=(-180.0, 180.0),
                    as_xarray=True,
                )
                _model_interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    # Roll target lons from -180..180 to 0..360
                    target_lons_raw = interp.target_lon[0, :]
                    n_roll = len(target_lons_raw) // 2
                    target_lons = np.roll(target_lons_raw, -n_roll)
                    target_lons = np.where(
                        target_lons < 0, target_lons + 360, target_lons,
                    )

                    # Compute common area weights
                    common_area = compute_latlon_areas(
                        target_lats, target_lons,
                    )

                    # Regrid obs to common grid (once)
                    lat_name = (
                        "lat" if "lat" in obs_timemean.coords else "latitude"
                    )
                    lon_name = (
                        "lon" if "lon" in obs_timemean.coords else "longitude"
                    )
                    obs_lats = obs_timemean[lat_name].values
                    obs_lons = obs_timemean[lon_name].values
                    obs_lons_180 = np.where(
                        obs_lons > 180, obs_lons - 360, obs_lons,
                    )
                    obs_lons_2d, obs_lats_2d = np.meshgrid(
                        obs_lons_180, obs_lats,
                    )

                    _, obs_interpolator = nr.regrid(
                        obs_timemean.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=resolution,
                        method=self._regrid_method,
                        influence_radius=self.ocean_influence_radius,
                        lon_bounds=(-180.0, 180.0),
                        as_xarray=True,
                    )

                    # Regrid each obs period to common grid
                    # (roll from -180..180 to 0..360 output order)
                    for pkey in periods_data:
                        obs_field = periods_data[pkey]["obs"]
                        regridded = np.roll(
                            obs_interpolator(obs_field.values.ravel()),
                            -n_roll, axis=1,
                        )
                        obs_common = xr.DataArray(
                            regridded,
                            dims=("lat", "lon"),
                            coords={
                                "lat": target_lats, "lon": target_lons,
                            },
                        )
                        periods_data[pkey]["obs_common"] = obs_common

            model_interpolator = _model_interp_cache[n_src]

            # Regrid model to common grid (roll from -180..180 to 0..360)
            annual_regrid = xr.DataArray(
                np.roll(
                    model_interpolator(model_clim_annual.values.ravel()),
                    -n_roll, axis=1,
                ),
                dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )
            djf_regrid = xr.DataArray(
                np.roll(
                    model_interpolator(model_djf.values.ravel()),
                    -n_roll, axis=1,
                ),
                dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )
            jja_regrid = xr.DataArray(
                np.roll(
                    model_interpolator(model_jja.values.ravel()),
                    -n_roll, axis=1,
                ),
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
                bias_vals = bias.values[np.isfinite(bias.values)]
                logger.info(
                    "  %s %s: model_regrid=[%.2f,%.2f] "
                    "obs_common=[%.2f,%.2f] bias=[%.2f,%.2f]",
                    model, pkey,
                    float(np.nanmin(regrid.values)),
                    float(np.nanmax(regrid.values)),
                    float(np.nanmin(obs_common.values)),
                    float(np.nanmax(obs_common.values)),
                    float(bias_vals.min()) if len(bias_vals) else float("nan"),
                    float(bias_vals.max()) if len(bias_vals) else float("nan"),
                )
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

        # EERIE ensemble mean/median bias (evaluated models only), computed
        # from the already-regridded model climatologies on the common grid.
        ensemble: dict[str, dict] = {}
        eerie = [m for m in model_results if m in self.config.models]
        for pkey in periods_data:
            regrids = [
                model_results[m][pkey]["regrid"]
                for m in eerie if pkey in model_results[m]
            ]
            if not regrids:
                continue
            obs_common = periods_data[pkey].get("obs_common")
            if obs_common is None:
                continue
            stack = xr.concat(regrids, dim="member")
            mean = stack.mean("member")
            median = stack.median("member")
            ensemble[pkey] = {
                "mean_regrid": mean, "median_regrid": median,
                "mean_bias": mean - obs_common,
                "median_bias": median - obs_common,
                "n_members": len(regrids),
            }

        # Benchmark (CMIP6/HighResMIP) MMM bias panel(s), on the same grid.
        if target_lats is not None:
            model_results.update(self._benchmark_bias_maps(
                target_lats, target_lons, resolution, periods_data,
                common_area))

        return {
            "models": model_results,
            "ensemble": ensemble,
            "periods": periods_data,
            "common_area": common_area,
        }

    def _benchmark_bias_maps(self, target_lats, target_lons, resolution,
                             periods_data, common_area):
        """Benchmark MMM SST bias per period on the common grid.

        Loads each benchmark (CMIP6/HighResMIP) member's ``tos``, regrids to
        the figure's common grid via the curvilinear-safe scattered regrid
        (ORCA/tripolar ocean grids), averages to the MMM, and takes the bias
        vs the same obs.  Returned as extra entries so ``_plot_bias_maps``
        renders one panel per benchmark alongside the evaluated models.
        """
        from feather.diag import ocean_bias
        out: dict[str, dict] = {}
        if not (getattr(self, "benchmarks", None) and self.cmip6_enabled):
            return out
        ir = max(self._influence_radius, ocean_bias._BENCH_IR_FLOOR)
        season_key = {"annual": None, "djf": "DJF", "jja": "JJA"}
        for bench in self.benchmarks:
            label = getattr(bench, "label", "CMIP6 MMM")
            cache: dict = {}
            members: dict[str, list] = {pk: [] for pk in periods_data}
            for model, variant in bench.get_member_pairs():
                for pkey, season in season_key.items():
                    try:
                        da = bench.load_var_for_model_var(
                            "tos", model, variant=variant,
                            period=self.period, season=season)
                    except Exception:  # noqa: BLE001
                        da = None
                    if da is None:
                        continue
                    try:
                        reg = ocean_bias.regrid_scatter(
                            ocean_bias.prep_ocean_field(self.config, da, "tos"),
                            target_lats, target_lons, resolution, ir, cache,
                            method=self._regrid_method)
                    except Exception:  # noqa: BLE001
                        continue
                    members[pkey].append(reg)
            if not members.get("annual"):
                logger.warning(
                    "No benchmark tos members for %s — SST bias panel skipped "
                    "(check the ocean zarr cache is complete)", label)
                continue
            out[label] = {}
            for pkey in periods_data:
                if not members[pkey]:
                    continue
                mmm = xr.concat(members[pkey], dim="member").mean("member")
                bias = mmm - periods_data[pkey]["obs_common"]
                out[label][pkey] = {
                    "regrid": mmm, "bias": bias,
                    "n_members": len(members[pkey]),
                    "bias_gmean": float(
                        latlon_global_mean(bias, area=common_area).values),
                    "rmse": float(np.sqrt(
                        latlon_global_mean(bias ** 2, area=common_area).values)),
                }
            logger.info("  Added %s MMM SST bias panel (%d members)",
                        label, len(members["annual"]))
        return out

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
                obs_title=self._obs_label,
                cmap=obs_cmap,
                bias_cmap="RdBu_r",
                units="\u00b0C",
                land=True,
                method=self._regrid_method,
            )

            meta = self._build_metadata(
                title=f"SST {plabel} Bias",
                figure_id=f"sst_{pkey}_bias_combined",
                models=all_models,
                description=(
                    f"{plabel} SST climatology and model biases relative to "
                    f"{self._obs_dataset_name} satellite observations."
                ),
                plot_type="combined_bias_map",
                period=self.period,
                obs_dataset=self._obs_dataset_name,
                summary_statistics=summary_stats,
            )
            figures.append((fig, meta))

        return figures

    def _plot_ens_bias_maps(self, results):
        """Ensemble mean/median bias maps (+ benchmark MMM) per period."""
        from feather.plot.maps import plot_combined_bias_map

        figures = []
        periods_data = results["periods"]
        ensemble = results.get("ensemble", {})
        benchmarks = [
            m for m in results["models"] if m not in self.config.models]

        for pkey, plabel in [
            ("annual", "Annual Mean"), ("djf", "DJF"), ("jja", "JJA"),
        ]:
            obs_common = periods_data[pkey].get("obs_common")
            ens = ensemble.get(pkey)
            if obs_common is None or ens is None:
                continue

            # Harmonised panel labels with member counts in bold parentheses,
            # matching temperature_berkeley / precipitation_mswep.
            proj = self.config.project.get("name", "Ensemble")
            n = ens["n_members"]
            bias_dict = {
                rf"{proj} ens. median $\mathbf{{({n})}}$": ens["median_bias"],
                rf"{proj} ens. mean $\mathbf{{({n})}}$": ens["mean_bias"],
            }
            for label in benchmarks:
                mdata = results["models"][label]
                if pkey in mdata:
                    m = mdata[pkey].get("n_members", 0)
                    panel = rf"{label} $\mathbf{{({m})}}$" if m else label
                    bias_dict[panel] = mdata[pkey]["bias"]

            try:
                import cmocean
                obs_cmap = cmocean.cm.thermal
            except ImportError:
                obs_cmap = "RdYlBu_r"

            fig, _ = plot_combined_bias_map(
                obs_common, bias_dict,
                title=f"Sea Surface Temperature {plabel} — Ensemble",
                obs_title=self._obs_label,
                cmap=obs_cmap, bias_cmap="RdBu_r", units="°C",
                land=True, method=self._regrid_method,
            )
            meta = self._build_metadata(
                title=f"SST {plabel} Ensemble Bias",
                figure_id=f"sst_{pkey}_ens_bias_combined",
                models=list(self.config.models),
                description=(
                    f"{plabel} SST ensemble mean/median bias "
                    f"(n={ens['n_members']}) and benchmark MMM bias relative "
                    f"to {self._obs_dataset_name}."
                ),
                plot_type="combined_bias_map", period=self.period,
                obs_dataset=self._obs_dataset_name,
                summary_statistics={
                    "ensemble_mean_global_bias": float(_ocean_global_mean(
                        ens["mean_bias"]).values),
                    "ensemble_median_global_bias": float(_ocean_global_mean(
                        ens["median_bias"]).values),
                    "n_members": ens["n_members"],
                },
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
            ax.plot(time_vals, obs_annual.values, label=self._obs_label,
                    color=OBS_COLOR, linewidth=2.5)

        ax.set_title("Global Mean Sea Surface Temperature")
        ax.set_ylabel("SST (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append(self._obs_label)

        meta = self._build_metadata(
            title="SST Global Mean Time Series",
            figure_id="sst_timeseries",
            models=all_models,
            description=(
                "Global-mean SST time series for DestinE models and {self._obs_label} "
                "observations. Monthly values as semi-transparent lines, "
                "annual means as thick lines. Units: degrees Celsius."
            ),
            plot_type="timeseries",
            period=self.period,
            obs_dataset=self._obs_dataset_name,
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
            ax.plot(months, obs_cycle.values, marker="s", label=self._obs_label,
                    color=OBS_COLOR, linewidth=2)

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title("SST Seasonal Cycle")
        ax.set_ylabel("SST (\u00b0C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append(self._obs_label)

        meta = self._build_metadata(
            title="SST Seasonal Cycle",
            figure_id="sst_seasonal_cycle",
            models=all_models,
            description=(
                "Monthly climatological cycle of global-mean SST for DestinE "
                "models and {self._obs_label} observations. Units: degrees Celsius."
            ),
            plot_type="seasonal_cycle",
            period=self.period,
            obs_dataset=self._obs_dataset_name,
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
                    label=self._obs_label, color=OBS_COLOR, linewidth=2)

        ax.set_title("SST Zonal Mean")
        ax.set_xlabel("SST (\u00b0C)")
        ax.set_ylabel("Latitude")
        ax.set_ylim(-90, 90)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append(self._obs_label)

        meta = self._build_metadata(
            title="SST Zonal Mean Profile",
            figure_id="sst_zonal_mean",
            models=all_models,
            description=(
                "Zonal mean SST profile for DestinE models and {self._obs_label} "
                "observations. Latitude on y-axis, SST (degrees Celsius) on "
                "x-axis."
            ),
            plot_type="zonal_profile",
            period=self.period,
            obs_dataset=self._obs_dataset_name,
        )
        return [(fig, meta)]


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values
