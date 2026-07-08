"""Shared ocean benchmark-bias computation + NetCDF export.

The ocean diagnostics (``ocean_sst``, ``ocean_en4``, ``sea_ice``) historically
had no CMIP6/benchmark integration.  This module gives them a single, shared
way to compute — on a common grid (``nereus.resolution``, 0.25° in the EERIE
configs) — the evaluated-model, ensemble (mean/median) and benchmark (CMIP6 /
HighResMIP MMM) biases against the same observational reference each diagnostic
already uses, and to persist them with
:func:`feather.diag.netcdf_export.export_biasmap_netcdf` in exactly the schema
:class:`feather.diag.added_value.AddedValueDiag` consumes.

Fine sources (obs, evaluated models) are regridded with the config
``influence_radius``; coarse benchmark ocean members get a ~250 km floor so
they still fill a fine (0.25°) target grid without gaps.

Ocean references
----------------
- ``tos``            → ESA-CCI L4 (0.05°)          [ocean_sst]
- ``thetao``/``so``  → EN4 v4.2.2 surface (1°)      [ocean_en4]
- ``siconc``         → OSI-SAF (EASE2, NH+SH)        [sea_ice]

Everything is regridded to the common global grid via :func:`regrid_scatter`,
which treats every source as scattered points and therefore handles
rectilinear, curvilinear (ORCA/tripolar) and polar (EASE2) grids uniformly.

The functions here take a *diagnostic instance* (any :class:`DiagnosticBase`
subclass) so they can reuse its ``config``, ``obs_loader``, ``benchmarks`` and
grid-agnostic ``_load_model_var`` helper.
"""

import logging
from typing import Any

import numpy as np
import nereus as nr
import xarray as xr

from feather.diag.netcdf_export import (
    export_biasmap_individual_netcdf,
    export_biasmap_netcdf,
)
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)

#: Ocean variable → observational reference dataset key (in ``obs_datasets``).
OCEAN_OBS: dict[str, str] = {
    "tos": "ESA_CCI",
    "thetao": "EN4",
    "so": "EN4",
    "siconc": "OSI_SAF",
}

#: Obs dataset → the diagnostic that owns its benchmark-bias NetCDF.  ``tos``
#: has two references: ESA-CCI (ocean_sst, 0.05°/1990–2014) and HadISST
#: (sst_hadisst, 1°/full period), both consumed by the ocean Added Value.
OCEAN_OBS_DIAG: dict[str, str] = {
    "ESA_CCI": "ocean_sst",
    "HADISST": "sst_hadisst",
    "EN4": "ocean_en4",
    "OSI_SAF": "sea_ice",
}

#: Display units for the exported bias fields (AV is dimensionless, but the
#: NetCDF records physical units for the climatology/bias fields).
OCEAN_UNITS: dict[str, str] = {
    "tos": "degC", "thetao": "degC", "so": "PSU", "siconc": "1",
}

#: Depth dimension names collapsed to the surface (shallowest) level.
DEPTH_DIMS = (
    "lev", "depth", "deptht", "olevel", "lev_2", "z", "nav_lev", "level",
)

#: Fallback common ocean bias/AV grid resolution (degrees) when the config
#: does not specify ``nereus.resolution``.
OCEAN_RES = 0.25

#: Influence-radius floor (m) applied to *coarse* benchmark (CMIP6/HighResMIP)
#: ocean members so they fill a fine target grid without gaps.  Fine sources
#: (obs, evaluated models) use the config ``influence_radius`` unfloored so a
#: 0.25° grid keeps their detail instead of being over-smoothed.
_BENCH_IR_FLOOR = 250_000.0

#: Periods computed (annual + the two solstitial seasons).
PERIODS = ("annual", "DJF", "JJA")


def grid_resolution(config) -> float:
    """Common-grid resolution (deg): ``nereus.resolution`` or the fallback."""
    return float(config.nereus.get("resolution", OCEAN_RES))


def common_grid(config=None, res: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the common (lat, lon) 1-D axes for the ocean analysis grid.

    Resolution precedence: explicit *res* → ``config.nereus.resolution`` →
    :data:`OCEAN_RES` (0.25°).
    """
    if res is None:
        res = grid_resolution(config) if config is not None else OCEAN_RES
    return (
        np.arange(-90.0 + res / 2, 90.0, res),
        np.arange(res / 2, 360.0, res),
    )


def latlon_names(da: xr.DataArray) -> tuple[str, str]:
    """Return the (lat, lon) coordinate names of *da*."""
    lat = next(
        (c for c in da.coords
         if str(c).lower() in ("lat", "latitude", "nav_lat", "y")),
        "lat",
    )
    lon = next(
        (c for c in da.coords
         if str(c).lower() in ("lon", "longitude", "nav_lon", "x")),
        "lon",
    )
    return lat, lon


def surface_slice(da: xr.DataArray) -> xr.DataArray:
    """Collapse any depth dimension to the surface (index 0)."""
    for d in da.dims:
        if str(d).lower() in DEPTH_DIMS:
            return da.isel({d: 0})
    return da


def to_celsius_if_needed(da: xr.DataArray) -> xr.DataArray:
    """Convert a temperature field to °C when it is clearly in Kelvin."""
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("k", "kelvin"):
        return da - 273.15
    if units in ("c", "celsius", "degc", "°c", "degrees_c"):
        return da
    try:
        if float(np.nanmean(np.asarray(da.values))) > 150.0:
            return da - 273.15
    except (ValueError, TypeError):
        pass
    return da


def siconc_to_fraction(da: xr.DataArray) -> xr.DataArray:
    """Normalise sea-ice concentration to a 0–1 fraction (from %)."""
    try:
        if float(np.nanmax(np.asarray(da.values))) > 1.5:
            return da / 100.0
    except (ValueError, TypeError):
        pass
    return da


def surface_sa_to_sp(config, da2d: xr.DataArray, model: str) -> xr.DataArray:
    """Convert surface absolute salinity → practical salinity for a model.

    No-op unless the model has ``absolute_salinity: true`` in config.
    """
    mc = config.model_configs.get(model)
    if not (mc and getattr(mc, "absolute_salinity", False)):
        return da2d
    import gsw

    lat_name, lon_name = latlon_names(da2d)
    lat = np.asarray(da2d[lat_name].values)
    lon = np.asarray(da2d[lon_name].values)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat
    sp = gsw.SP_from_SA(np.asarray(da2d.values), 0.0, lon2d, lat2d)
    return da2d.copy(data=sp)


def prep_ocean_field(
    config, da: xr.DataArray, var: str, *, model: str | None = None,
) -> xr.DataArray:
    """Surface-slice + unit-normalise an ocean field for comparison."""
    da = surface_slice(da)
    if var in ("tos", "thetao"):
        da = to_celsius_if_needed(da)
    elif var == "so" and model is not None:
        da = surface_sa_to_sp(config, da, model)
    elif var == "siconc":
        da = siconc_to_fraction(da)
    return da


def regrid_scatter(
    da: xr.DataArray, target_lats, target_lons, resolution,
    influence_radius, cache: dict, method: str = "nearest",
) -> xr.DataArray:
    """Regrid any 2-D ocean field to the common grid via nereus.

    Treats the source as scattered points so it works for rectilinear,
    curvilinear (2-D nav_lat/nav_lon) and polar (EASE2) grids alike.
    Any singleton non-spatial dimension (e.g. a length-1 ``time`` on the
    ESA-CCI ``timemean`` field) is dropped first so the data is genuinely 2-D.
    """
    da = da.squeeze(drop=True)
    lat_name, lon_name = latlon_names(da)
    lat = np.asarray(da[lat_name].values)
    lon = np.asarray(da[lon_name].values)
    data = np.asarray(da.values)

    if (lat.ndim == 1 and lon.ndim == 1 and data.ndim == 2
            and data.shape == (lat.size, lon.size)):
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat

    src_lon = np.where(lon2d > 180, lon2d - 360, lon2d).ravel()
    src_lat = np.asarray(lat2d).ravel()
    vals = data.ravel()

    key = (
        int(vals.shape[0]),
        round(float(np.nanmin(src_lat)), 3),
        round(float(np.nanmax(src_lat)), 3),
    )
    if key not in cache:
        _, cache[key] = nr.regrid(
            vals, lon=src_lon, lat=src_lat,
            resolution=resolution, method=method,
            influence_radius=influence_radius, lon_bounds=(-180.0, 180.0),
            as_xarray=True,
        )
    regridded = cache[key](vals)
    n_roll = regridded.shape[1] // 2
    regridded = np.roll(regridded, -n_roll, axis=1)
    return xr.DataArray(
        regridded, dims=("lat", "lon"),
        coords={"lat": target_lats, "lon": target_lons},
    )


# -- Observation loading (returns climatologies already on the common grid) --


def _coarsen_rectilinear(da: xr.DataArray, target_res: float) -> xr.DataArray:
    """Block-average a fine rectilinear field toward *target_res* (degrees).

    Keeps very high-resolution obs (e.g. ESA-CCI at 0.05°, 25.9M cells)
    tractable for the scattered-point regrid: coarsening to ~the analysis
    grid resolution avoids building a KDTree over tens of millions of points.
    No-op for curvilinear grids (lat/lon are 2-D coords, not dims) or when the
    source is already at/above the target resolution.  NaN (land) cells are
    skipped by the block mean.
    """
    da = da.squeeze(drop=True)
    lat_name, lon_name = latlon_names(da)
    if lat_name not in da.dims or lon_name not in da.dims:
        return da
    lat = np.asarray(da[lat_name].values)
    if lat.size < 2:
        return da
    src_res = abs(float(lat[1] - lat[0]))
    if src_res <= 0:
        return da
    k = int(target_res / src_res)
    if k >= 2:
        da = da.coarsen({lat_name: k, lon_name: k}, boundary="trim").mean()
    return da


def _esa_cci_native(
    obs_loader, target_res: float = OCEAN_RES,
) -> dict[str, xr.DataArray]:
    annual = obs_loader.load_esa_cci("timemean") - 273.15
    ymon = obs_loader.load_esa_cci("ymonmean") - 273.15
    if "time" in ymon.dims:
        mon = ymon["time.month"]
        djf = ymon.sel(time=mon.isin([12, 1, 2])).mean("time")
        jja = ymon.sel(time=mon.isin([6, 7, 8])).mean("time")
    elif "month" in ymon.dims:
        djf = ymon.sel(month=[12, 1, 2]).mean("month")
        jja = ymon.sel(month=[6, 7, 8]).mean("month")
    else:
        djf = jja = annual
    out = {"annual": annual, "DJF": djf, "JJA": jja}
    return {pk: _coarsen_rectilinear(v, target_res) for pk, v in out.items()}


def _esa_cci_coverage(obs_loader) -> tuple[str, str] | None:
    """Actual (start_year, end_year) covered by the ESA-CCI monthly SST file.

    The pre-averaged ``timemean``/``ymonmean`` climatologies carry no usable
    period (their ``time_coverage_*`` attributes report the single collapsed
    timestamp), so the real window is read from the monthly ``analysed_sst``
    time axis.  Only the time coordinate is materialised, not the data.
    Returns ``None`` if the coverage cannot be determined.
    """
    try:
        da = obs_loader.load_esa_cci("analysed_sst")
    except Exception:  # noqa: BLE001
        return None
    if "time" not in getattr(da, "dims", ()):
        return None
    t = da["time"].values
    if t.size == 0:
        return None
    return (str(t.min())[:4], str(t.max())[:4])


def obs_clim_period(obs_loader, config, var: str, period, obs_name=None):
    """Period the model/benchmark climatology should use to match the obs.

    Only ESA-CCI needs alignment: it is a *fixed* pre-averaged climatology
    over the ESA-CCI file's own coverage, so the model/benchmark climatology
    is aligned to that window (∩ analysis *period*).  Every other obs (EN4,
    OSI-SAF, HadISST) is sliced to the analysis period on load, so *period* is
    returned unchanged.
    """
    period = (str(period[0]), str(period[1]))
    obs_name = obs_name or OCEAN_OBS.get(var)
    if obs_name != "ESA_CCI":
        return period
    cov = _esa_cci_coverage(obs_loader)
    if not cov:
        return period
    aligned = (max(period[0], cov[0]), min(period[1], cov[1]))
    if aligned[0] > aligned[1]:  # no overlap → fall back to config period
        return period
    return aligned


def _hadisst_native(obs_loader, period) -> dict[str, xr.DataArray]:
    """HadISST SST climatologies (annual/DJF/JJA) in °C on the native 1° grid.

    ``ObsLoader.load_hadisst`` returns K with land/ice masked; climatologies
    are computed over the analysis *period* (HadISST spans the full window).
    """
    da = to_celsius_if_needed(obs_loader.load_hadisst(period=period))
    out = {"annual": climatology(da, period)}
    seasonal = seasonal_climatology(da, period)
    for s in ("DJF", "JJA"):
        if s in seasonal:
            out[s] = seasonal[s]
    return out


def _en4_native(obs_loader, var: str, period) -> dict[str, xr.DataArray]:
    da = surface_slice(obs_loader.load_en4(var, period=period))
    if var == "thetao":
        da = to_celsius_if_needed(da)  # EN4 thetao is stored in K
    out = {"annual": climatology(da, period)}
    seasonal = seasonal_climatology(da, period)
    for s in ("DJF", "JJA"):
        if s in seasonal:
            out[s] = seasonal[s]
    return out


def _osisaf_native_hemis(obs_loader, period) -> dict[str, dict[str, xr.DataArray]]:
    hemis: dict[str, dict[str, xr.DataArray]] = {}
    for hemi in ("nh", "sh"):
        ds = obs_loader.load_osisaf(hemi, period=period)
        da = siconc_to_fraction(ds["ice_conc"])
        clim = {"annual": da.mean("time") if "time" in da.dims else da}
        if "time" in da.dims:
            mon = da["time.month"]
            clim["DJF"] = da.sel(time=mon.isin([12, 1, 2])).mean("time")
            clim["JJA"] = da.sel(time=mon.isin([6, 7, 8])).mean("time")
        hemis[hemi] = clim
    return hemis


def load_ocean_obs_on_target(
    obs_loader, config, var: str, period,
    target_lats, target_lons, resolution, influence_radius, cache, method,
    obs_name=None,
) -> dict[str, xr.DataArray] | None:
    """Load the ocean obs reference for *var* as {period: field-on-grid}.

    *obs_name* selects the reference dataset (defaults to ``OCEAN_OBS[var]``);
    pass it explicitly to evaluate a variable against an alternative obs (e.g.
    ``tos`` vs ``HADISST`` instead of ``ESA_CCI``).
    """
    obs_name = obs_name or OCEAN_OBS.get(var)
    if obs_name not in config.obs_datasets:
        logger.warning("Ocean obs %s not configured — skipping %s", obs_name, var)
        return None

    if obs_name == "ESA_CCI":
        native = _esa_cci_native(obs_loader, target_res=resolution)
        merged_hemis = None
    elif obs_name == "HADISST":
        native = _hadisst_native(obs_loader, period)
        merged_hemis = None
    elif obs_name == "EN4":
        native = _en4_native(obs_loader, var, period)
        merged_hemis = None
    elif obs_name == "OSI_SAF":
        hemis = _osisaf_native_hemis(obs_loader, period)
        native = hemis["nh"]
        merged_hemis = hemis
    else:
        return None

    out: dict[str, xr.DataArray] = {}
    for pk, field in native.items():
        out[pk] = regrid_scatter(
            field, target_lats, target_lons, resolution,
            influence_radius, cache, method=method,
        )
    # Merge the OSI-SAF SH hemisphere onto the (NH) fields.
    if merged_hemis is not None:
        for pk, sh_field in merged_hemis["sh"].items():
            if pk not in out:
                continue
            sh_on_grid = regrid_scatter(
                sh_field, target_lats, target_lons, resolution,
                influence_radius, cache, method=method,
            )
            out[pk] = out[pk].combine_first(sh_on_grid)
    return out


# -- Field computation (export_biasmap schema) --------------------------------


def compute_ocean_fields(
    diag, var: str, period, method: str = "nearest",
    *, benchmarks=None, want_individual: bool = False, obs_name=None,
) -> dict[str, Any] | None:
    """Compute obs/model/ensemble/benchmark ocean fields on the common grid.

    Returns a ``results`` dict in the schema understood by
    :func:`export_biasmap_netcdf` (``obs`` / ``models`` / ``benchmark_data`` /
    ``ens_data`` / ``benchmark_individual_data``), or ``None`` when the obs
    reference, the evaluated models or the benchmark members are unavailable.

    *benchmarks* defaults to ``diag.benchmarks`` (all configured); pass an
    explicit list (e.g. a single active benchmark) to restrict it.
    *obs_name* selects the obs reference (defaults to ``OCEAN_OBS[var]``).
    """
    config = diag.config
    benches = benchmarks if benchmarks is not None else diag.benchmarks
    if not benches:
        return None
    obs_name = obs_name or OCEAN_OBS.get(var)

    res = grid_resolution(config)
    target_lats, target_lons = common_grid(config)
    # Fine sources (obs, evaluated models) keep the config influence radius;
    # coarse benchmark members get a floor so they fill a fine target grid.
    ir = config.nereus.get("influence_radius", 80_000.0)
    bench_ir = max(ir, _BENCH_IR_FLOOR)

    obs_cache: dict = {}
    obs_fields = load_ocean_obs_on_target(
        diag.obs_loader, config, var, period,
        target_lats, target_lons, res, ir, obs_cache, method,
        obs_name=obs_name,
    )
    if not obs_fields:
        return None
    periods = [p for p in PERIODS if p in obs_fields]

    # Climatology window for the model/benchmark side.  For ESA-CCI tos this
    # is the ESA-CCI coverage (∩ analysis period) so the SST bias is
    # like-for-like; every other obs is sliced to the analysis period.
    mperiod = obs_clim_period(diag.obs_loader, config, var, period, obs_name)
    if tuple(mperiod) != (str(period[0]), str(period[1])):
        logger.info(
            "  %s: aligning model/benchmark climatology to obs window %s-%s",
            var, mperiod[0], mperiod[1],
        )

    # -- Evaluated models --------------------------------------------------
    model_cache: dict = {}
    models: dict[str, dict] = {}
    ens_regrids: dict[str, list[xr.DataArray]] = {p: [] for p in periods}
    for model in config.models:
        try:
            da = diag._load_model_var(model, var, period=mperiod)
        except (KeyError, FileNotFoundError):
            logger.warning("  %s not available for %s — skipping", var, model)
            continue
        da = prep_ocean_field(config, da, var, model=model)
        annual = regrid_scatter(
            climatology(da, mperiod).compute(), target_lats, target_lons,
            res, ir, model_cache, method=method,
        )
        entry = {
            "annual_regrid": annual,
            "annual_bias": annual - obs_fields["annual"],
            "seasonal_regrids": {},
            "seasonal_biases": {},
        }
        ens_regrids["annual"].append(annual)
        seasonal = seasonal_climatology(da, mperiod)
        for s in ("DJF", "JJA"):
            if s in periods and s in seasonal:
                sr = regrid_scatter(
                    seasonal[s].compute(), target_lats, target_lons,
                    res, ir, model_cache, method=method,
                )
                entry["seasonal_regrids"][s] = sr
                entry["seasonal_biases"][s] = sr - obs_fields[s]
                ens_regrids[s].append(sr)
        models[model] = entry

    if not models:
        logger.warning("No evaluated models for ocean %s — skipping", var)
        return None

    # -- Evaluated ensemble mean/median ------------------------------------
    ens_data: dict[str, dict] = {}
    for pk in periods:
        if not ens_regrids[pk]:
            continue
        stack = xr.concat(ens_regrids[pk], dim="member")
        mean = stack.mean("member")
        median = stack.median("member")
        ens_data[pk] = {
            "mean": mean, "median": median,
            "mean_bias": mean - obs_fields[pk],
            "median_bias": median - obs_fields[pk],
        }

    # -- Benchmark MMMs (+ optional individual members) --------------------
    benchmark_data: dict[str, dict] = {}
    benchmark_individual: dict[str, dict] = {}
    for bench in benches:
        label = getattr(bench, "label", "CMIP6 MMM") or "CMIP6 MMM"
        bench_cache: dict = {}
        members: dict[str, list[xr.DataArray]] = {p: [] for p in periods}
        indiv: dict[str, dict] = {p: {} for p in periods}
        for model, variant in bench.get_member_pairs():
            member_label = f"{model}/{variant}"
            try:
                da = bench.load_var_for_model_var(
                    var, model, variant=variant, period=mperiod)
            except Exception:  # noqa: BLE001
                da = None
            if da is None:
                continue
            try:
                reg = regrid_scatter(
                    prep_ocean_field(config, da, var),
                    target_lats, target_lons, res, bench_ir, bench_cache,
                    method=method,
                )
            except Exception:  # noqa: BLE001
                logger.debug("  benchmark %s regrid failed", member_label)
                continue
            members["annual"].append(reg)
            if want_individual:
                indiv["annual"][member_label] = {
                    "regrid": reg, "bias": reg - obs_fields["annual"]}
            for s in ("DJF", "JJA"):
                if s not in periods:
                    continue
                try:
                    da_s = bench.load_var_for_model_var(
                        var, model, variant=variant, period=mperiod, season=s)
                except Exception:  # noqa: BLE001
                    da_s = None
                if da_s is None:
                    continue
                reg_s = regrid_scatter(
                    prep_ocean_field(config, da_s, var),
                    target_lats, target_lons, res, bench_ir, bench_cache,
                    method=method,
                )
                members[s].append(reg_s)
                if want_individual:
                    indiv[s][member_label] = {
                        "regrid": reg_s, "bias": reg_s - obs_fields[s]}

        if not members["annual"]:
            logger.warning(
                "No benchmark members for %s / %s — skipping", label, var)
            continue
        bd: dict[str, dict] = {}
        for pk in periods:
            if not members[pk]:
                continue
            mmm = xr.concat(members[pk], dim="member").mean("member")
            bd[pk] = {"regrid": mmm, "bias": mmm - obs_fields[pk]}
        benchmark_data[label] = bd
        if want_individual:
            benchmark_individual[label] = indiv

    if not benchmark_data:
        return None

    obs_block = {
        "clim": obs_fields["annual"],
        "seasonal_clim": {p: obs_fields[p] for p in periods if p != "annual"},
    }
    results: dict[str, Any] = {
        "obs": obs_block,
        "models": models,
        "ens_data": ens_data,
        "benchmark_data": benchmark_data,
    }
    if want_individual and benchmark_individual:
        results["benchmark_individual_data"] = benchmark_individual
    return results


def save_ocean_bias_netcdf(
    diag, var: str, period, method: str = "nearest",
    *, want_individual: bool = False, obs_name=None,
) -> list:
    """Compute and persist ocean benchmark/model/ensemble bias NetCDFs.

    Writes to ``{output}/netcdf/{diag.name}/`` in the
    :func:`export_biasmap_netcdf` schema so :class:`AddedValueDiag` can reuse
    the fields for ocean Added Value.  *obs_name* selects the obs reference
    (defaults to ``OCEAN_OBS[var]``).  Returns the list of written paths (may
    be empty).  Never raises — failures are logged.
    """
    try:
        results = compute_ocean_fields(
            diag, var, period, method, want_individual=want_individual,
            obs_name=obs_name,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Ocean bias compute failed for %s", var, exc_info=True)
        return []
    if not results:
        return []

    units = OCEAN_UNITS.get(var, "")
    # Label the files with the actual climatology window used for the fields.
    # For ESA-CCI tos this is the obs coverage (∩ analysis period), i.e.
    # 1990-2014 rather than the config 1980-2014; every other obs is unchanged.
    fperiod = obs_clim_period(diag.obs_loader, diag.config, var, period, obs_name)
    written = export_biasmap_netcdf(
        diag._netcdf_dir, var, results, tuple(fperiod), units=units,
    )
    if want_individual and results.get("benchmark_individual_data"):
        try:
            export_biasmap_individual_netcdf(
                diag._netcdf_dir, var, results, tuple(fperiod), units=units,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Ocean individual-bias export failed for %s", var,
                exc_info=True,
            )
    return written


def maybe_export_ocean_bias(
    diag, ocean_vars, *, want_individual: bool = False,
    skip_existing: bool = True, obs_name=None,
) -> list:
    """Export ocean benchmark-bias NetCDFs for *ocean_vars*, if applicable.

    No-op unless the diagnostic has ``save_netcdf`` on and at least one
    benchmark loader.  Each variable is skipped when its annual/DJF/JJA bias
    NetCDFs already exist (cheap re-runs).  *obs_name* selects the obs
    reference (defaults to ``OCEAN_OBS[var]``).  Never raises.
    """
    if not (getattr(diag, "save_netcdf", False)
            and getattr(diag, "benchmarks", None)):
        return []

    period = tuple(getattr(diag, "period", None) or diag.config.get_period())
    method = diag.config.nereus.get("method", "nearest")
    nc_dir = diag._netcdf_dir
    written: list = []
    for var in ocean_vars:
        this_obs = obs_name or OCEAN_OBS.get(var)
        if this_obs not in diag.config.obs_datasets:
            continue
        # Files are labelled with the actual climatology window (obs coverage
        # ∩ period; only ESA-CCI differs from the config period).
        fperiod = obs_clim_period(diag.obs_loader, diag.config, var, period,
                                  this_obs)
        if skip_existing and all(
            (nc_dir / f"{var}_{pk}_{fperiod[0]}-{fperiod[1]}.nc").exists()
            for pk in PERIODS
        ):
            logger.info(
                "Ocean benchmark bias for %s already saved — skipping", var)
            continue
        logger.info("Saving ocean benchmark bias for %s (%s, obs %s)",
                    var, diag.name, this_obs)
        written += save_ocean_bias_netcdf(
            diag, var, period, method, want_individual=want_individual,
            obs_name=this_obs,
        )
    return written
