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
from feather.plot.styles import ENS_COLOR, OBS_COLOR
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

#: Months making up each meteorological season (for obs seasonal means).
_SEASON_MONTHS = {
    "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
}
#: Human-readable panel/period labels keyed by lowercase period key.
_SEASON_LABEL = {
    "annual": "Annual Mean", "djf": "DJF", "mam": "MAM",
    "jja": "JJA", "son": "SON",
}


def _to_celsius(da):
    """Convert Kelvin DataArray to Celsius."""
    return da - _K_TO_C


def _normalize_monthly_time(da):
    """Normalise a time series' ``time`` coord to first-of-month timestamps.

    Different models use different calendars (360_day, noleap, standard) with
    differing mid-month day conventions, so identical months carry different
    raw timestamps and an inner join finds no overlap.  Mapping every step to
    ``YYYY-MM-01`` pandas timestamps makes members on any calendar align by
    month.  Returns ``None`` if the series has no usable time axis.
    """
    if "time" not in getattr(da, "dims", ()):
        return da
    try:
        times = da.time.values
        if len(times) == 0:
            return None
        t0 = times[0]
        if hasattr(t0, "year") and not isinstance(t0, np.datetime64):
            new_times = pd.to_datetime(
                [f"{t.year:04d}-{t.month:02d}-01" for t in times]
            )
        else:
            new_times = pd.to_datetime(times).to_period("M").to_timestamp()
        return da.assign_coords(time=new_times)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to normalize time coordinate: %s", e)
        return None


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


def _ocean_grid_global_mean(da, area=None):
    """Grid-agnostic area-weighted global mean over a field's spatial dims.

    Works for rectilinear (``lat``/``lon``), curvilinear (2-D ``latitude`` on
    ``j``/``i``) and unstructured (``ncells``) ocean grids alike — whatever
    dimensions remain after ``time``.  Weights come from *area* (``areacello``)
    when its shape matches the field's spatial grid; otherwise cos-lat weights
    are derived from the latitude coordinate.  NaN weights (common in
    ``areacello``) are zeroed so ``weighted().mean()`` does not raise, and NaN
    (land) data cells are excluded via ``skipna``.  Returns ``None`` if no
    usable weights can be built.
    """
    from feather.diag.ocean_bias import latlon_names

    spatial = [d for d in da.dims if d != "time"]
    if not spatial:
        return None
    spatial_shape = tuple(da.sizes[d] for d in spatial)

    weights = None
    if area is not None:
        av = np.nan_to_num(np.asarray(area.values), nan=0.0)
        if av.shape == spatial_shape:
            weights = xr.DataArray(av, dims=spatial)

    if weights is None:
        lat_name, _ = latlon_names(da)
        if lat_name in da.coords:
            cosw = np.nan_to_num(
                np.cos(np.deg2rad(np.asarray(da[lat_name].values))), nan=0.0,
            )
            if cosw.shape == spatial_shape:
                weights = xr.DataArray(cosw, dims=spatial)
            elif cosw.ndim == 1:
                for d in spatial:
                    if da.sizes[d] == cosw.size:
                        weights = xr.DataArray(cosw, dims=(d,))
                        break

    if weights is None:
        return None
    return da.weighted(weights).mean(spatial, skipna=True)


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

    def _season_pkeys(self) -> list[str]:
        """Lowercase period keys for the configured seasons (annual first)."""
        return [s.lower() for s in self.config.get_seasons()]

    def _season_plot_list(self) -> list[tuple[str, str]]:
        """Ordered ``(period_key, panel_label)`` for the configured seasons."""
        return [(pk, _SEASON_LABEL.get(pk, pk.upper()))
                for pk in self._season_pkeys()]

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
            f"sst_{p}_{suffix}" for p in self._season_pkeys()
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

        # Configured seasonal breakdown (annual + subset of DJF/MAM/JJA/SON).
        seasons = self.config.get_seasons()
        seasonal_keys = [s for s in seasons if s != "annual"]

        # Obs per configured season from the monthly climatology (month dim
        # 1-12, or a "time" axis of monthly means).
        periods_data: dict[str, dict] = {"annual": {"obs": obs_timemean}}
        for s in seasonal_keys:
            months = list(_SEASON_MONTHS[s])
            if "time" in obs_ymonmean.dims:
                obs_s = obs_ymonmean.sel(
                    time=obs_ymonmean["time.month"].isin(months)).mean("time")
            elif "month" in obs_ymonmean.dims:
                obs_s = obs_ymonmean.sel(month=months).mean("month")
            else:
                obs_s = obs_timemean
            periods_data[s.lower()] = {"obs": obs_s}

        resolution = self.config.nereus.get("resolution", 0.25)

        # Cache interpolators per source grid size
        _model_interp_cache: dict[int, Any] = {}
        obs_interpolator = None
        target_lats = None
        target_lons = None
        common_area = None
        n_roll = 0

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

            # Compute model climatologies (annual + each configured season).
            model_clim_annual = climatology(da, self.period).compute()
            model_seasonal = seasonal_climatology(da, self.period)
            model_fields = {"annual": model_clim_annual}
            for s in seasonal_keys:
                model_fields[s.lower()] = (
                    model_seasonal[s].compute()
                    if s in model_seasonal else model_clim_annual
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

                    # Adapt the obs influence radius to the obs grid spacing.
                    # ocean_influence_radius (~20 km) is tuned for ESA-CCI's
                    # 0.05° grid; a coarse obs like HadISST (1°, ~111 km) needs
                    # a radius comparable to its spacing or the regrid leaves
                    # NaN holes (speckled/striped panel) on the finer common
                    # grid.  Take the larger of the configured radius and ~1.5×
                    # the obs cell size in metres.
                    obs_dlat = (
                        abs(float(obs_lats[1] - obs_lats[0]))
                        if len(obs_lats) > 1 else resolution
                    )
                    obs_ir = max(
                        self.ocean_influence_radius,
                        obs_dlat * 111_000.0 * 1.5,
                    )

                    _, obs_interpolator = nr.regrid(
                        obs_timemean.values.ravel(),
                        lon=obs_lons_2d.ravel(),
                        lat=obs_lats_2d.ravel(),
                        resolution=resolution,
                        method=self._regrid_method,
                        influence_radius=obs_ir,
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

            # Regrid model to common grid (roll from -180..180 to 0..360) for
            # each configured period.
            model_results[model] = {}
            for pkey, field in model_fields.items():
                regrid = xr.DataArray(
                    np.roll(
                        model_interpolator(field.values.ravel()),
                        -n_roll, axis=1,
                    ),
                    dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )
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
        # period key → benchmark season arg (annual → None; seasons → CMOR name)
        season_key = {
            pk: (None if pk == "annual" else pk.upper())
            for pk in periods_data
        }
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
        """Plot combined bias maps for each configured season."""
        from feather.plot.maps import plot_combined_bias_map

        figures = []
        periods_data = results["periods"]

        for pkey, plabel in self._season_plot_list():
            obs_common = periods_data.get(pkey, {}).get("obs_common")
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

        for pkey, plabel in self._season_plot_list():
            obs_common = periods_data.get(pkey, {}).get("obs_common")
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
        """Compute global-mean SST time series for models and obs.

        In addition to the per-model and obs series, this derives the
        evaluated-ensemble mean/median and the benchmark (CMIP6/HighResMIP)
        MMM series with a min/max envelope band across benchmark members.
        Benchmark ``tos`` is normalised to °C (via ``prep_ocean_field``) to
        match the °C model/obs series.
        """
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

        # Benchmark (CMIP6/HighResMIP) MMM series + min/max envelope band.
        # Uses an ocean-aware, grid-agnostic global mean (areacello-weighted
        # with a cos-lat fallback) so curvilinear ORCA/tripolar/ICON grids —
        # the majority of CMIP6 ocean models — are included, not skipped.
        benchmarks_ts = self._benchmark_ocean_timeseries()

        # Evaluated-ensemble mean/median across the model series.
        ens_mean, ens_median = self._compute_ensemble_stats(model_ts)

        return {
            "models": model_ts,
            "obs": obs_ts,
            "benchmarks_ts": benchmarks_ts,
            "ens_mean": ens_mean,
            "ens_median": ens_median,
        }

    # ── Ocean-aware benchmark global-mean time series ─────────────────

    def _benchmark_ocean_timeseries(self) -> list[dict]:
        """Per-benchmark global-mean ``tos`` MMM series + min/max envelope.

        Mirrors :meth:`DiagnosticBase._benchmark_timeseries` but computes each
        member's global mean with :func:`_ocean_grid_global_mean`, which weights
        by ``areacello`` (falling back to cos-lat) over whatever spatial dims a
        model uses.  This keeps curvilinear ocean grids — which the rectilinear
        base helper drops — in the ensemble.  Member time axes are normalised to
        first-of-month so mixed calendars still align.
        """
        from feather.diag.ocean_bias import prep_ocean_field
        from feather.plot.styles import benchmark_color

        if not (getattr(self, "benchmarks", None) and self.cmip6_enabled):
            return []

        out: list[dict] = []
        for i, bench in enumerate(self.benchmarks):
            label = getattr(bench, "label", "CMIP6 MMM")
            series: dict[str, Any] = {}
            for model, variant in bench.get_member_pairs():
                try:
                    da = bench.load_var(
                        "tos", model, table="Omon",
                        period=self.period, time_mean=False,
                    )
                except Exception:  # noqa: BLE001
                    da = None
                if da is None or "time" not in getattr(da, "dims", ()):
                    continue
                try:
                    da = prep_ocean_field(self.config, da, "tos")
                    area = bench.load_area(model, variant=variant, table="Omon")
                    ts = _ocean_grid_global_mean(da, area)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "    Skipping %s for benchmark tos: %s", model, e)
                    continue
                if ts is None:
                    continue
                ts = _normalize_monthly_time(ts.compute())
                if ts is not None:
                    series[model] = ts.reset_coords(drop=True)

            if not series:
                logger.warning(
                    "No benchmark tos members for %s — MMM series skipped",
                    label)
                continue

            aligned = xr.align(*series.values(), join="outer")
            mmm = xr.concat(aligned, dim="member").mean("member", skipna=True)
            env_min, env_max = self._envelope_from_series(series)
            logger.info(
                "  %s tos MMM time series: %d members, %d timesteps",
                label, len(series), len(mmm.time))
            out.append({
                "label": label,
                "color": getattr(bench, "color", None) or benchmark_color(i),
                "ts": mmm,
                "info": {"n_members": len(series),
                         "models_used": list(series)},
                "env_min": env_min,
                "env_max": env_max,
                "individual": dict(series) if self.cmip6_individual else {},
            })
        return out

    @staticmethod
    def _compute_ensemble_stats(model_ts):
        """Ensemble mean/median across the evaluated model time series.

        Time coordinates are first normalised to first-of-month timestamps so
        members on different calendars (e.g. HadGEM3's 360-day cftime vs the
        IFS models' ``datetime64``) and differing mid-month day conventions
        still overlap.  Series are then aligned on the inner time union so
        models of differing lengths are reduced to their common period before
        averaging.  Returns ``(None, None)`` when fewer than 2 members remain
        or the members share no common month.
        """
        series = list(model_ts.values())
        if len(series) < 2:
            return None, None
        # Drop non-dimension scalar coords (e.g. depth) that some models
        # carry and others don't — otherwise xr.concat raises on mismatch.
        series = [
            _normalize_monthly_time(s.reset_coords(drop=True)) for s in series
        ]
        series = [s for s in series if s is not None]
        if len(series) < 2:
            return None, None
        aligned = xr.align(*series, join="inner")
        if aligned[0].sizes.get("time", 0) == 0:
            logger.warning(
                "Ensemble members share no common month — "
                "skipping SST ensemble mean/median",
            )
            return None, None
        stacked = xr.concat(list(aligned), dim="member")
        return stacked.mean("member"), stacked.median("member")

    def _plot_timeseries(self, results):
        """Plot SST global-mean time series.

        Layering (back to front): benchmark min/max envelope band + MMM
        (dashed) \u2192 evaluated models \u2192 ensemble median (dashed) / mean (solid)
        \u2192 observations.  All series are in \u00b0C.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        all_models = []
        benchmarks = results.get("benchmarks_ts", []) or []
        n_eerie = len(results["models"])

        # --- Monthly pass (background, washed-out) ---

        # Benchmark envelope band + MMM (dashed)
        for bench in benchmarks:
            b_color = bench["color"]
            emin, emax = bench.get("env_min"), bench.get("env_max")
            if emin is not None and emax is not None:
                t = _to_plot_time(emin.time.values)
                ax.fill_between(t, emin.values, emax.values,
                                color=b_color, alpha=0.12, lw=0)
            b_ts = bench["ts"]
            t = _to_plot_time(b_ts.time.values)
            ax.plot(t, b_ts.values, color=b_color, alpha=0.3,
                    linewidth=0.7, linestyle="--")

        # DestinE model monthly
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

        # --- Annual pass (foreground, thick with labels) ---

        # Benchmark min/max envelope + MMM (dashed)
        for bench in benchmarks:
            b_color = bench["color"]
            b_label = bench["label"]
            n_mmm = bench["info"].get("n_members", 0)
            emin, emax = bench.get("env_min"), bench.get("env_max")
            if emin is not None and emax is not None:
                amin, amax = annual_mean(emin), annual_mean(emax)
                band = b_label[:-4] if b_label.endswith(" MMM") else b_label
                t = _to_plot_time(amin.time.values)
                ax.fill_between(t, amin.values, amax.values, color=b_color,
                                alpha=0.25, lw=0,
                                label=f"{band} min\u2013max ({n_mmm})")
            b_annual = annual_mean(bench["ts"])
            t = _to_plot_time(b_annual.time.values)
            ax.plot(t, b_annual.values, label=f"{b_label} ({n_mmm})",
                    color=b_color, linewidth=2.0, linestyle="--")
            all_models.append(b_label)

        # DestinE model annual
        for model, ts in results["models"].items():
            color = self.config.get_model_color(model)
            ts_annual = annual_mean(ts)
            time_vals = _to_plot_time(ts_annual.time.values)
            ax.plot(time_vals, ts_annual.values, label=model, color=color,
                    linewidth=2.0)
            all_models.append(model)

        # Ensemble median (dashed) / mean (solid)
        proj = self.config.project.get("name", "Ensemble")
        if results.get("ens_median") is not None:
            ens_med_annual = annual_mean(results["ens_median"])
            t = _to_plot_time(ens_med_annual.time.values)
            ax.plot(t, ens_med_annual.values, color=ENS_COLOR,
                    linewidth=2.5, linestyle="--",
                    label=f"{proj} ensemble median ({n_eerie})")
        if results.get("ens_mean") is not None:
            ens_mean_annual = annual_mean(results["ens_mean"])
            t = _to_plot_time(ens_mean_annual.time.values)
            ax.plot(t, ens_mean_annual.values, color=ENS_COLOR,
                    linewidth=2.5,
                    label=f"{proj} ensemble mean ({n_eerie})")

        if results.get("obs") is not None:
            obs_annual = annual_mean(results["obs"])
            time_vals = _to_plot_time(obs_annual.time.values)
            ax.plot(time_vals, obs_annual.values, label=self._obs_label,
                    color=OBS_COLOR, linewidth=2.5)

        ax.set_title("Global Mean Sea Surface Temperature")
        ax.set_ylabel("SST (\u00b0C)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if results.get("obs") is not None:
            all_models.append(self._obs_label)

        meta = self._build_metadata(
            title="SST Global Mean Time Series",
            figure_id="sst_timeseries",
            models=all_models,
            description=(
                f"Global-mean SST time series for models, benchmark MMM "
                f"(with min\u2013max envelope), ensemble mean/median, and "
                f"{self._obs_label} observations. Monthly values as "
                f"semi-transparent lines, annual means as thick lines. "
                f"Units: degrees Celsius."
            ),
            plot_type="timeseries",
            period=self.period,
            obs_dataset=self._obs_dataset_name,
            benchmark_info=self._benchmark_meta_from_list(
                benchmarks) or None,
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
