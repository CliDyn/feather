"""Added Value diagnostic (Dosio et al. 2015).

Quantifies the added value of the EERIE model ensemble (mean and median)
over the CMIP6 multi-model mean, relative to ERA5 observations.

For each variable and period the AV metric is computed grid-point-wise:

    AV = [ (X_EERIE - X_ref)² - (X_CMIP6 - X_ref)² ]
         / max( (X_EERIE - X_ref)², (X_CMIP6 - X_ref)² )

AV ∈ [-1, 1]:
  AV > 0  →  EERIE reduces squared error over CMIP6
  AV < 0  →  CMIP6 performs better
  AV = 0  →  no difference

Reference
---------
Dosio, A., Panitz, H.-J., Schubert-Frisius, M., & Lüthi, D. (2015).
Dynamical downscaling of CMIP5 global circulation models over CORDEX-Africa
with COSMO-CLM: evaluation over the present climate and analysis of the
added value. Climate Dynamics, 44(9-10), 2637-2661.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cmocean
import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import pandas as pd
import xarray as xr

from feather.data.variables import get_var
from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import plot_combined_map
from feather.util.spatial import compute_latlon_areas, latlon_global_mean
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)

_AV_CMAP = cmocean.tools.crop_by_percent(cmocean.cm.tarn, 50, which="both", N=None)

_OBS_DISPLAY_NAMES: dict[str, str] = {
    "ERA5": "ERA5",
    "BERKELEY_EARTH_HR": "Berkeley Earth HR",
    "MSWEP": "MSWEP v2.8",
}


@register
class AddedValueDiag(DiagnosticBase):
    """Added Value of EERIE ensemble vs CMIP6 MMM (Dosio et al. 2015).

    Produces combined two-panel figures per variable per period
    (annual, DJF, JJA) showing the spatial AV field for the EERIE
    ensemble mean and ensemble median.  AV fields are also saved as
    CMORized NetCDF files for offline analysis.
    """

    name = "added_value"
    title = "Added Value (EERIE vs CMIP6)"
    domain = "sfc"
    variables = [
        # Temperature & pressure
        "tas", "psl",
        # Wind
        "uas", "vas",
        # Cloud cover
        "clt",
        # Precipitation
        "pr",
        # Surface heat fluxes
        "hfss", "hfls",
        # Surface downwelling radiation
        "rsds", "rlds",
        # Surface net radiation (all-sky + clear-sky)
        "rss", "rls",
        "rsscs", "rlscs",
        # TOA net radiation (all-sky + clear-sky)
        "rst", "rlt",
        "rstcs", "rltcs",
    ]
    group = "evaluation"

    def __init__(
        self,
        model_loader,
        obs_loader,
        config,
        *,
        cmip6_loader=None,
        variables=None,
        experiment="baseline_hist",
        period=("1990", "2014"),
        cmip6_individual=False,
    ):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        if variables is not None:
            self.variables = list(variables)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual
        self._regrid_method = self.config.nereus.get("method", "nearest")

    # -- Output paths -------------------------------------------------------

    @property
    def nc_dir(self) -> Path:
        """Directory for saved AV NetCDF files."""
        return Path(self.config.output_dir) / "added_value"

    def _nc_path(self, var: str, period: str, ensemble_type: str) -> Path:
        """Return path for a single AV NetCDF file."""
        return self.nc_dir / f"{var}_{period}_{ensemble_type}_av.nc"

    def _all_nc_exist(self, var: str) -> bool:
        """True when all 6 NC files (3 periods × 2 ensemble types) exist."""
        for period in ("annual", "djf", "jja"):
            for etype in ("ensemble_mean", "ensemble_median"):
                if not self._nc_path(var, period, etype).exists():
                    return False
        return True

    def _all_figures_exist(self, var: str) -> bool:
        """True when all figures for primary-obs ensemble and per-model plots exist."""
        primary_obs = self._OBS_ALT_DATASETS.get(var, "ERA5")
        secondary_obs = [
            d for d in self._MULTI_OBS_DATASETS.get(var, [])
            if d != primary_obs
        ]
        for p in ("annual", "djf", "jja"):
            if not self._figure_exists(f"{var}_{p}_added_value"):
                return False
            if not self._figure_exists(f"{var}_{p}_added_value_models"):
                return False
            for sec_obs in secondary_obs:
                suffix = self._OBS_FIGURE_SUFFIX.get(sec_obs, sec_obs.lower())
                if not self._figure_exists(f"{var}_{p}_added_value_{suffix}"):
                    return False
                if not self._figure_exists(f"{var}_{p}_added_value_models_{suffix}"):
                    return False
        return True

    # -- Orchestration -------------------------------------------------------

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-variable: compute → save NC → plot → save figures.

        NC files are the durable checkpoint.  If all NC files for a
        variable already exist, computation is skipped and figures are
        regenerated directly from the saved NetCDF (or also skipped if
        figures are present too).

        After the per-variable loop, summary bar chart figures are generated
        (one set per temporal period: annual, DJF, JJA).
        """
        logger.info("Running diagnostic: %s", self.name)
        self.nc_dir.mkdir(parents=True, exist_ok=True)
        saved: list[tuple[Path, Path]] = []
        all_obs_stats: dict[str, dict] = {}  # {var: obs_stats}

        for var in self.variables:
            if skip_existing and self._all_figures_exist(var):
                logger.info("Skipping %s — all figures exist", var)
                for p in ("annual", "djf", "jja"):
                    for suffix in ("added_value", "added_value_models"):
                        fid = f"{var}_{p}_{suffix}"
                        saved.append((
                            self.output_dir / f"{fid}.png",
                            self.output_dir / f"{fid}.json",
                        ))
                # Still load obs_stats for bar charts
                json_path = self._obs_stats_path(var)
                if json_path.exists():
                    try:
                        with open(json_path) as fh:
                            jd = json.load(fh)
                        var_stats: dict[str, dict] = {}
                        for pk, pd in jd.get("periods", {}).items():
                            for on, od in pd.items():
                                var_stats.setdefault(on, {})[pk] = od
                        if var_stats:
                            all_obs_stats[var] = var_stats
                    except Exception:
                        pass
                continue

            try:
                if skip_existing and self._all_nc_exist(var):
                    logger.info(
                        "%s — NC files exist, loading for plotting", var,
                    )
                    var_result = self._load_variable_from_nc(var)
                else:
                    var_result = self._compute_variable(var)

                if var_result is None:
                    continue

                if var_result.get("obs_stats"):
                    all_obs_stats[var] = var_result["obs_stats"]

                figures = self._plot_variable(var, var_result)
                for fig, meta in figures:
                    paths = self._save(fig, meta, meta["figure_id"])
                    saved.append(paths)

            except Exception:
                logger.warning(
                    "Variable %s failed — skipping", var, exc_info=True,
                )

        # ── Summary bar charts (cross-variable, one per period) ─────────────
        if all_obs_stats:
            for period_key in ("annual", "djf", "jja"):
                for bar_fn, fn_name in (
                    (self._plot_summary_bars_ensemble, "ensemble"),
                    (self._plot_summary_bars_models,   "models"),
                ):
                    bar_id = f"added_value_bars_{fn_name}_{period_key}"
                    if skip_existing and self._figure_exists(bar_id):
                        saved.append((
                            self.output_dir / f"{bar_id}.png",
                            self.output_dir / f"{bar_id}.json",
                        ))
                        continue
                    try:
                        for fig, meta in bar_fn(all_obs_stats, period_key):
                            paths = self._save(fig, meta, meta["figure_id"])
                            saved.append(paths)
                    except Exception:
                        logger.warning(
                            "Bar chart %s failed", bar_id, exc_info=True,
                        )

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # -- Observation loading ------------------------------------------------

    #: Variables that use alternative obs datasets instead of ERA5.
    _OBS_ALT_DATASETS: dict[str, str] = {
        "tas": "BERKELEY_EARTH_HR",
        "pr": "MSWEP",
    }

    #: Short suffix used in figure IDs for each obs dataset.
    #: Primary obs keeps no suffix (backward compat); secondary obs gets one.
    _OBS_FIGURE_SUFFIX: dict[str, str] = {
        "ERA5": "era5",
        "BERKELEY_EARTH_HR": "berkeleyearth",
        "MSWEP": "mswep",
    }

    #: All obs datasets to evaluate against, per variable.
    #: Used to compute per-obs improvement/neutral/deterioration statistics.
    #: ERA5 is always listed first (primary for most variables).
    _MULTI_OBS_DATASETS: dict[str, list[str]] = {
        "tas": ["ERA5", "BERKELEY_EARTH_HR"],
        "pr":  ["ERA5", "MSWEP"],
    }

    #: Unit conversion factors applied to both obs and model data so that
    #: climatologies are in display-friendly units before the AV formula.
    #: AV is dimensionless so this does not affect the final values, but
    #: it ensures all inputs are consistent and matches user-facing units.
    _UNIT_FACTORS: dict[str, float] = {
        "pr": 86400.0,  # kg/m²/s → mm/day
    }

    def _load_obs_for_var(
        self, var: str, period: tuple[str, str],
    ) -> xr.DataArray:
        """Load observations for *var*, routing to the best available dataset.

        - ``tas``: Berkeley Earth 0.25° HR (falls back to ERA5 if not in config)
        - ``pr``:  MSWEP v2.8 (falls back to ERA5 if not in config)
        - others: ERA5 via standard ``_load_obs_var``
        """
        alt = self._OBS_ALT_DATASETS.get(var)
        if alt == "BERKELEY_EARTH_HR" and alt in self.config.obs_datasets:
            logger.info("  Using Berkeley Earth HR for %s", var)
            return self._load_berkeley_earth_for_av(period)
        if alt == "MSWEP" and "MSWEP" in self.config.obs_datasets:
            logger.info("  Using MSWEP for %s", var)
            return self._load_mswep_for_av(period)
        return self._load_obs_var(var, period)

    def _load_obs_by_name(
        self, var: str, obs_name: str, period: tuple[str, str],
    ) -> xr.DataArray:
        """Load a specific obs dataset by name for AV statistics.

        Parameters
        ----------
        obs_name : str
            Dataset name: ``"ERA5"``, ``"BERKELEY_EARTH_HR"``, or
            ``"MSWEP"``.

        Notes
        -----
        Unit conventions match the primary-obs loaders:
        - ERA5 ``pr`` is returned raw in kg/m²/s (caller applies ×86400).
        - MSWEP is returned in mm/day (``_load_mswep_for_av`` applies ×86400
          internally).
        - Berkeley Earth and ERA5 ``tas`` are in K.
        """
        if obs_name == "BERKELEY_EARTH_HR":
            return self._load_berkeley_earth_for_av(period)
        if obs_name == "MSWEP":
            return self._load_mswep_for_av(period)  # already mm/day
        return self._load_obs_var(var, period)  # ERA5 (raw units)

    def _load_berkeley_earth_for_av(
        self, period: tuple[str, str],
    ) -> xr.DataArray:
        """Load Berkeley Earth 0.25° gridded temperature.

        The file stores monthly anomalies (°C re 1951-1980) plus a separate
        ``climatology`` array (12 × lat × lon, °C).
        Absolute temperature = anomaly + climatology[month_of_year].
        Time is encoded as decimal years; this method converts it to proper
        datetime coordinates before slicing.
        Returns a DataArray in Kelvin with dims (time, lat, lon),
        lons 0..360.
        """
        ds_cfg = self.config.obs_datasets["BERKELEY_EARTH_HR"]
        filepath = Path(ds_cfg["path"]) / ds_cfg["variables"]["temperature"]
        ds_full = xr.open_dataset(filepath, chunks="auto")

        # Convert decimal-year time → DatetimeIndex
        dec_years = ds_full["time"].values
        years = dec_years.astype(int)
        months = np.round((dec_years - years) * 12).astype(int) + 1
        months = np.clip(months, 1, 12)
        datetimes = pd.to_datetime(
            [f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)]
        )
        ds_full = ds_full.assign_coords(time=datetimes)

        start, end = period
        anom = ds_full["temperature"].sel(time=slice(start, end))
        clim = ds_full["climatology"]  # (month_number, latitude, longitude)

        month_idx = anom.time.dt.month.values - 1  # 0-based
        clim_np = clim.values  # (12, nlat, nlon)
        clim_matched = clim_np[month_idx]
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

        return abs_temp + 273.15  # degC → K

    def _load_mswep_for_av(self, period: tuple[str, str]) -> xr.DataArray:
        """Load MSWEP v2.8 precipitation for the AV reference.

        Returns a DataArray in mm/day with dims (time, lat, lon).
        ``ObsLoader.load_mswep()`` delivers kg/m²/s; multiply by 86400.
        """
        da = self.obs_loader.load_mswep(period=period)
        return da * self._UNIT_FACTORS["pr"]

    # -- Computation --------------------------------------------------------

    def compute(self) -> dict[str, Any]:
        """Compute AV for all variables.

        Returns
        -------
        dict keyed by variable name.
        """
        results: dict[str, Any] = {}
        self.nc_dir.mkdir(parents=True, exist_ok=True)
        for var in self.variables:
            var_result = self._compute_variable(var)
            if var_result is not None:
                results[var] = var_result
        return results

    def _compute_variable(self, var: str) -> dict[str, Any] | None:
        """Compute AV fields for a single variable.

        Steps
        -----
        1. Load ERA5 climatology (annual + DJF/JJA) on its native grid.
        2. Regrid each EERIE model to the common grid; stack → mean + median.
        3. Regrid each CMIP6 member to the same common grid; stack → MMM.
        4. Apply Dosio AV formula grid-point-wise for each period.
        5. Save AV fields as CMORized NetCDF files.

        Returns None if EERIE models or CMIP6 data are not available.
        """
        if not self.cmip6_enabled:
            logger.warning(
                "CMIP6 loader not available — cannot compute AV for %s", var,
            )
            return None

        var_info = get_var(var)
        logger.info(
            "Computing Added Value for %s (%s)", var, var_info.long_name,
        )

        influence_radius = self.config.nereus.get("influence_radius", 80_000.0)

        # -- Observations ---------------------------------------------------
        logger.info("  Loading observations for %s", var)
        unit_factor = self._UNIT_FACTORS.get(var, 1.0)
        obs_data = self._load_obs_for_var(var, self.period)
        obs_clim = climatology(obs_data)
        obs_seasonal = seasonal_climatology(obs_data)

        lat_name = "lat" if "lat" in obs_clim.coords else "latitude"
        lon_name = "lon" if "lon" in obs_clim.coords else "longitude"
        obs_lats = obs_clim[lat_name].values
        obs_lons = obs_clim[lon_name].values
        obs_res = abs(float(obs_lats[1] - obs_lats[0]))

        # -- EERIE models ---------------------------------------------------
        _interp_cache: dict[int, Any] = {}
        target_lats = None
        target_lons = None
        obs_clim_common = None
        obs_seasonal_common: dict[str, xr.DataArray] = {}
        common_area = None

        eerie_annual_fields: list[xr.DataArray] = []
        eerie_seasonal_fields: dict[str, list[xr.DataArray]] = {
            "DJF": [], "JJA": [],
        }
        eerie_seasonal_models: dict[str, list[str]] = {"DJF": [], "JJA": []}
        eerie_models_used: list[str] = []

        for model in self.config.models:
            logger.info("  Loading EERIE model %s / %s", var, model)
            try:
                model_data = self._load_model_var(model, var, period=self.period)
            except (KeyError, FileNotFoundError):
                logger.warning(
                    "  Variable %s not available for %s — skipping",
                    var, model,
                )
                continue

            lon, lat = self._load_model_coords(model, var)
            grid_type = self.config.get_grid_type(model, self.domain)

            model_clim = climatology(model_data, self.period).compute()
            if unit_factor != 1.0:
                model_clim = model_clim * unit_factor
            model_seasonal = seasonal_climatology(model_data, self.period)
            model_seasonal = {
                s: model_seasonal[s].compute() * unit_factor
                for s in model_seasonal.data_vars
            }

            # nereus expects:
            #   HEALPix → 1-D scattered lon/lat + 1-D data
            #   regular lat/lon → 1-D lon/lat axes + 2-D (lat×lon) data
            lon_1d = np.asarray(lon)
            lat_1d = np.asarray(lat)
            if grid_type == "healpix":
                data_for_regrid = model_clim.values.ravel()
                n_src = lon_1d.shape[0]
            else:
                data_for_regrid = model_clim.values  # 2-D (lat, lon)
                n_src = lat_1d.shape[0] * lon_1d.shape[0]

            if n_src not in _interp_cache:
                logger.info(
                    "  Building nereus interpolator (grid size %d)...", n_src,
                )
                annual_regrid, interp = nr.regrid(
                    data_for_regrid,
                    lon=lon_1d, lat=lat_1d,
                    resolution=obs_res,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                _interp_cache[n_src] = interp

                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]

                    # Regrid obs to common grid (once) — obs is always lat/lon
                    _, obs_interp = nr.regrid(
                        obs_clim.values,  # 2-D
                        lon=obs_lons, lat=obs_lats,
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

                    for season in ["DJF", "JJA"]:
                        if season in obs_seasonal:
                            obs_s = obs_seasonal[season]
                            obs_seasonal_common[season] = xr.DataArray(
                                obs_interp(obs_s.values.ravel()),
                                dims=("lat", "lon"),
                                coords={
                                    "lat": target_lats, "lon": target_lons,
                                },
                            )
            else:
                interp = _interp_cache[n_src]
                annual_regrid = xr.DataArray(
                    interp(np.asarray(data_for_regrid).ravel()),
                    dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )

            eerie_annual_fields.append(annual_regrid)
            eerie_models_used.append(model)

            for season in ["DJF", "JJA"]:
                if season in model_seasonal:
                    s_data = model_seasonal[season].values.ravel()
                    s_vals = _interp_cache[n_src](s_data)
                    eerie_seasonal_fields[season].append(xr.DataArray(
                        s_vals, dims=("lat", "lon"),
                        coords={"lat": target_lats, "lon": target_lons},
                    ))
                    eerie_seasonal_models[season].append(model)

        if not eerie_annual_fields:
            logger.warning("No EERIE models found for %s — skipping", var)
            return None

        # EERIE ensemble mean and median (annual)
        eerie_stack = xr.concat(eerie_annual_fields, dim="member")
        eerie_mean = eerie_stack.mean("member")
        eerie_median = eerie_stack.median("member")

        # EERIE seasonal
        eerie_seasonal_mean: dict[str, xr.DataArray] = {}
        eerie_seasonal_median: dict[str, xr.DataArray] = {}
        for season, fields in eerie_seasonal_fields.items():
            if fields:
                s_stack = xr.concat(fields, dim="member")
                eerie_seasonal_mean[season] = s_stack.mean("member")
                eerie_seasonal_median[season] = s_stack.median("member")

        # -- CMIP6 MMM -------------------------------------------------------
        logger.info("  Computing CMIP6 MMM for %s...", var)
        cmip6_annual_fields: list[xr.DataArray] = []
        cmip6_seasonal_fields: dict[str, list[xr.DataArray]] = {
            "DJF": [], "JJA": [],
        }
        cmip6_seasonal_models: dict[str, list[str]] = {"DJF": [], "JJA": []}
        cmip6_models_used: list[str] = []
        cmip6_interp_cache: dict[tuple, Any] = {}

        resolution = abs(float(target_lats[1] - target_lats[0]))

        for model, variant in self.cmip6_loader.get_member_pairs():
            label = f"{model}/{variant}"
            da = self.cmip6_loader.load_var_for_model_var(
                var, model, variant=variant, period=self.period,
            )
            if da is None:
                logger.debug("  CMIP6 %s — no data", label)
                continue

            regridded = self._regrid_to_target(
                da, target_lats, target_lons,
                resolution, influence_radius, cmip6_interp_cache,
                method=self._regrid_method,
            )
            cmip6_annual_fields.append(regridded * unit_factor)
            cmip6_models_used.append(label)

            for season in ["DJF", "JJA"]:
                da_s = self.cmip6_loader.load_var_for_model_var(
                    var, model, variant=variant,
                    period=self.period, season=season,
                )
                if da_s is not None:
                    s_r = self._regrid_to_target(
                        da_s, target_lats, target_lons,
                        resolution, influence_radius, cmip6_interp_cache,
                        method=self._regrid_method,
                    )
                    cmip6_seasonal_fields[season].append(s_r * unit_factor)
                    cmip6_seasonal_models[season].append(label)

        if not cmip6_annual_fields:
            logger.warning("No CMIP6 models found for %s — skipping", var)
            return None

        cmip6_stack = xr.concat(cmip6_annual_fields, dim="member")
        cmip6_mmm = cmip6_stack.mean("member")

        cmip6_seasonal_mmm: dict[str, xr.DataArray] = {}
        for season, fields in cmip6_seasonal_fields.items():
            if fields:
                cmip6_seasonal_mmm[season] = xr.concat(
                    fields, dim="member",
                ).mean("member")

        # -- AV computation -------------------------------------------------
        # Model1 = CMIP6 MMM, Model2 = EERIE  →  AV > 0 means EERIE adds value
        #
        # Two ensemble statistics computed:
        #   ensemble_mean  : AV(CMIP6 MMM, EERIE ens. mean,   ERA5)
        #   ensemble_median: AV(CMIP6 MMM, EERIE ens. median, ERA5)
        #
        # Per-model individual AV maps:
        #   per_eerie_av:  {model: AV(CMIP6 MMM, EERIE_i, ERA5)}
        #   per_cmip6_av:  {label: AV(EERIE ens. mean, CMIP6_j, ERA5)}

        av_results: dict[str, dict] = {}

        def _period_av(cmip6_field, eerie_ens_mean, eerie_ens_median,
                       eerie_fields, eerie_labels,
                       cmip6_fields, cmip6_labels,
                       obs_field):
            """Compute ensemble and per-model AV for one period."""
            av_ens_mean = self._compute_av(cmip6_field, eerie_ens_mean,
                                           obs_field)
            av_ens_median = self._compute_av(cmip6_field, eerie_ens_median,
                                             obs_field)

            per_eerie_av = {
                lbl: self._compute_av(cmip6_field, ef, obs_field)
                for lbl, ef in zip(eerie_labels, eerie_fields)
            }
            per_cmip6_av = {
                lbl: self._compute_av(eerie_ens_mean, cf, obs_field)
                for lbl, cf in zip(cmip6_labels, cmip6_fields)
            }

            return {
                "ensemble_mean": av_ens_mean,
                "ensemble_median": av_ens_median,
                "per_eerie_av": per_eerie_av,
                "per_cmip6_av": per_cmip6_av,
                "ensemble_mean_domain_av": self._domain_mean_av(
                    av_ens_mean, common_area),
                "ensemble_median_domain_av": self._domain_mean_av(
                    av_ens_median, common_area),
                "ensemble_mean_frac_positive": self._frac_positive(
                    av_ens_mean, common_area),
                "ensemble_median_frac_positive": self._frac_positive(
                    av_ens_median, common_area),
            }

        # Annual
        av_results["annual"] = _period_av(
            cmip6_mmm, eerie_mean, eerie_median,
            eerie_annual_fields, eerie_models_used,
            cmip6_annual_fields, cmip6_models_used,
            obs_clim_common,
        )

        # Seasonal
        for season in ["DJF", "JJA"]:
            if (
                season in eerie_seasonal_fields
                and eerie_seasonal_fields[season]
                and season in cmip6_seasonal_mmm
                and season in obs_seasonal_common
            ):
                av_results[season] = _period_av(
                    cmip6_seasonal_mmm[season],
                    eerie_seasonal_mean[season],
                    eerie_seasonal_median[season],
                    eerie_seasonal_fields[season],
                    eerie_seasonal_models[season],
                    cmip6_seasonal_fields[season],
                    cmip6_seasonal_models[season],
                    obs_seasonal_common[season],
                )

        # -- Save NetCDF files ----------------------------------------------
        obs_dataset_name = self._OBS_ALT_DATASETS.get(var, "ERA5")
        nc_meta = {
            "eerie_models": eerie_models_used,
            "n_eerie_models": len(eerie_models_used),
            "cmip6_models": cmip6_models_used,
            "n_cmip6_models": len(cmip6_models_used),
            "period_start": self.period[0],
            "period_end": self.period[1],
            "variable": var,
            "long_name": var_info.long_name,
            "units": var_info.units,
            "obs_dataset": obs_dataset_name,
        }
        for period_key, period_data in av_results.items():
            for etype in ("ensemble_mean", "ensemble_median"):
                nc_path = self._nc_path(var, period_key.lower(), etype)
                if not nc_path.exists():
                    self._save_av_to_nc(
                        period_data[etype], var,
                        period_key.lower(), etype, nc_meta,
                    )

        # -- Per-obs improvement/neutral/deterioration statistics ------------
        eerie_individual_annual = dict(zip(eerie_models_used, eerie_annual_fields))
        eerie_seasonal_individual: dict[str, dict[str, xr.DataArray]] = {
            season: dict(zip(eerie_seasonal_models[season], fields))
            for season, fields in eerie_seasonal_fields.items()
            if fields
        }
        obs_stats: dict[str, dict] = {}
        if target_lats is not None:
            obs_stats = self._compute_multi_obs_stats(
                var,
                eerie_mean, eerie_median, cmip6_mmm,
                eerie_seasonal_mean, eerie_seasonal_median,
                cmip6_seasonal_mmm,
                target_lats, target_lons, influence_radius,
                eerie_individual=eerie_individual_annual,
                eerie_seasonal_individual=eerie_seasonal_individual,
            )
            if obs_stats:
                self._save_obs_stats_json(var, obs_stats, nc_meta)

        # -- Secondary-obs AV maps ------------------------------------------
        primary_obs_name = self._OBS_ALT_DATASETS.get(var, "ERA5")
        av_by_obs: dict[str, dict] = {primary_obs_name: av_results}

        secondary_obs_names = [
            d for d in self._MULTI_OBS_DATASETS.get(var, [])
            if d != primary_obs_name
            and (d == "ERA5" or d in self.config.obs_datasets)
        ]
        for sec_obs_name in secondary_obs_names:
            try:
                logger.info(
                    "  Computing secondary-obs AV maps: %s / %s", var, sec_obs_name,
                )
                sec_da = self._load_obs_by_name(var, sec_obs_name, self.period)
                if sec_obs_name == "ERA5" and unit_factor != 1.0:
                    sec_da = sec_da * unit_factor
                sec_clim = climatology(sec_da)
                sec_seasonal = seasonal_climatology(sec_da)
                sec_common = self._regrid_obs_to_common_grid(
                    sec_clim, target_lats, target_lons, influence_radius,
                )
                sec_seasonal_common: dict[str, xr.DataArray] = {}
                for season in ["DJF", "JJA"]:
                    if season in sec_seasonal:
                        sec_seasonal_common[season] = self._regrid_obs_to_common_grid(
                            sec_seasonal[season], target_lats, target_lons,
                            influence_radius,
                        )
                sec_av: dict[str, dict] = {}
                sec_av["annual"] = _period_av(
                    cmip6_mmm, eerie_mean, eerie_median,
                    eerie_annual_fields, eerie_models_used,
                    cmip6_annual_fields, cmip6_models_used,
                    sec_common,
                )
                for season in ["DJF", "JJA"]:
                    if (
                        season in eerie_seasonal_fields
                        and eerie_seasonal_fields[season]
                        and season in cmip6_seasonal_mmm
                        and season in sec_seasonal_common
                    ):
                        sec_av[season] = _period_av(
                            cmip6_seasonal_mmm[season],
                            eerie_seasonal_mean[season],
                            eerie_seasonal_median[season],
                            eerie_seasonal_fields[season],
                            eerie_seasonal_models[season],
                            cmip6_seasonal_fields[season],
                            cmip6_seasonal_models[season],
                            sec_seasonal_common[season],
                        )
                av_by_obs[sec_obs_name] = sec_av
            except Exception:
                logger.warning(
                    "  Secondary-obs AV failed for %s / %s", var, sec_obs_name,
                    exc_info=True,
                )

        return {
            "var_info": var_info,
            "av": av_results,
            "av_by_obs": av_by_obs,
            "obs_dataset_name": primary_obs_name,
            "obs_stats": obs_stats,
            "n_eerie_models": len(eerie_models_used),
            "eerie_models": eerie_models_used,
            "n_cmip6_models": len(cmip6_models_used),
            "cmip6_models": cmip6_models_used,
        }

    def _load_variable_from_nc(self, var: str) -> dict[str, Any] | None:
        """Load previously saved AV NetCDF files for plotting.

        Used when NC files exist but figures are missing.
        """
        var_info = get_var(var)
        av_results: dict[str, dict] = {}
        _etypes = ("ensemble_mean", "ensemble_median")

        for period in ("annual", "djf", "jja"):
            paths = {
                et: self._nc_path(var, period, et) for et in _etypes
            }
            if not all(p.exists() for p in paths.values()):
                continue

            datasets = {et: xr.open_dataset(p) for et, p in paths.items()}
            av_fields = {et: ds["av"] for et, ds in datasets.items()}

            lat_vals = av_fields["ensemble_mean"]["lat"].values
            lon_vals = av_fields["ensemble_mean"]["lon"].values
            area = compute_latlon_areas(lat_vals, lon_vals)

            period_data: dict[str, Any] = {}
            for et, av_f in av_fields.items():
                period_data[et] = av_f
                period_data[f"{et}_domain_av"] = self._domain_mean_av(
                    av_f, area)
                period_data[f"{et}_frac_positive"] = self._frac_positive(
                    av_f, area)
            # Per-model AV maps are not stored in NC; skip per-model figure
            period_data["per_eerie_av"] = {}
            period_data["per_cmip6_av"] = {}
            av_results[period] = period_data

            for ds in datasets.values():
                ds.close()

        if not av_results:
            return None

        # Recover model lists from NC attributes
        ds = xr.open_dataset(
            self._nc_path(var, "annual", "ensemble_mean"))
        n_eerie = int(ds["av"].attrs.get("n_eerie_models", 0))
        n_cmip6 = int(ds["av"].attrs.get("n_cmip6_models", 0))
        eerie_models = ds["av"].attrs.get("eerie_models", "").split(",")
        cmip6_models = ds["av"].attrs.get("cmip6_models", "").split(",")
        obs_dataset_name = ds["av"].attrs.get("reference_dataset", "ERA5")
        ds.close()

        # Restore obs_stats from JSON checkpoint if available
        obs_stats: dict[str, dict] = {}
        json_path = self._obs_stats_path(var)
        if json_path.exists():
            with open(json_path) as fh:
                json_data = json.load(fh)
            for period_key, period_data in json_data.get("periods", {}).items():
                for _obs_name, _obs_data in period_data.items():
                    obs_stats.setdefault(_obs_name, {})[period_key] = _obs_data

        return {
            "var_info": var_info,
            "av": av_results,
            "av_by_obs": {obs_dataset_name: av_results},
            "obs_dataset_name": obs_dataset_name,
            "obs_stats": obs_stats,
            "n_eerie_models": n_eerie,
            "eerie_models": [m for m in eerie_models if m],
            "n_cmip6_models": n_cmip6,
            "cmip6_models": [m for m in cmip6_models if m],
        }

    # -- AV formula helpers -------------------------------------------------

    @staticmethod
    def _compute_av(
        m1: xr.DataArray,
        m2: xr.DataArray,
        ref: xr.DataArray,
    ) -> xr.DataArray:
        """Dosio et al. (2015) Added Value metric.

        Parameters
        ----------
        m1 : xr.DataArray
            Reference model climatology (CMIP6 MMM).
        m2 : xr.DataArray
            Candidate model climatology (EERIE mean, median, or individual).
        ref : xr.DataArray
            Observation climatology (ERA5).

        Notes
        -----
        With m1=CMIP6 and m2=EERIE, AV > 0 means EERIE reduces the
        squared error relative to CMIP6, i.e. EERIE adds value.

        Returns
        -------
        xr.DataArray
            AV in [-1, 1] on the same grid.
        """
        sq1 = (m1.values - ref.values) ** 2
        sq2 = (m2.values - ref.values) ** 2
        denom = np.maximum(sq1, sq2)
        with np.errstate(invalid="ignore", divide="ignore"):
            av_vals = np.where(denom > 0, (sq1 - sq2) / denom, 0.0)
        return xr.DataArray(
            av_vals, dims=m1.dims, coords=m1.coords,
        )

    @staticmethod
    def _domain_mean_av(
        av: xr.DataArray, area: np.ndarray | None,
    ) -> float:
        """Area-weighted domain-mean AV score.

        When *area* is None, falls back to a simple arithmetic mean.
        """
        if area is None:
            return float(np.nanmean(np.asarray(av)))
        return float(latlon_global_mean(av, area=area).values)

    @staticmethod
    def _frac_positive(
        av: xr.DataArray, area: np.ndarray | None = None,
    ) -> float:
        """Area-weighted fraction where AV > 0 (EERIE adds value).

        When *area* (m², shape matching *av*) is provided the result is
        weighted by cell area; otherwise falls back to a simple cell count.
        """
        vals = np.asarray(av).ravel()
        mask = np.isfinite(vals)
        if not np.any(mask):
            return float("nan")
        if area is not None:
            w = np.asarray(area).ravel()
            w_finite = w[mask]
            total = np.sum(w_finite)
            if total == 0:
                return float("nan")
            return float(np.sum(w_finite * (vals[mask] > 0)) / total)
        finite = vals[mask]
        return float(np.sum(finite > 0) / len(finite))

    @staticmethod
    def _frac_categories(
        av: xr.DataArray,
        threshold: float = 0.001,
        area: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Area-weighted percentage in each AV category.

        Parameters
        ----------
        av : xr.DataArray
            AV field in [-1, 1].
        threshold : float
            Half-width of the neutral zone (default 0.001).
        area : np.ndarray, optional
            Cell areas (m², same shape as *av*).  When provided the
            percentages are area-weighted; otherwise plain cell counts.

        Returns
        -------
        dict with keys ``pct_improvement``, ``pct_neutral``,
        ``pct_deterioration`` in [0, 100].  Values sum to 100.

        Categories
        ----------
        improvement:    AV >  +threshold  (candidate reduces error)
        neutral:       -threshold ≤ AV ≤ +threshold
        deterioration:  AV <  -threshold  (baseline is better)
        """
        vals = np.asarray(av).ravel()
        mask = np.isfinite(vals)
        _nan = {
            "pct_improvement": float("nan"),
            "pct_neutral": float("nan"),
            "pct_deterioration": float("nan"),
        }
        if not np.any(mask):
            return _nan
        if area is not None:
            w = np.asarray(area).ravel()
            w_f = w[mask]
            v_f = vals[mask]
            total = np.sum(w_f)
            if total == 0:
                return _nan
            pct_imp = float(np.sum(w_f[v_f > threshold]) / total * 100)
            pct_det = float(np.sum(w_f[v_f < -threshold]) / total * 100)
            pct_neu = 100.0 - pct_imp - pct_det
        else:
            finite = vals[mask]
            n = len(finite)
            pct_imp = float(np.sum(finite > threshold) / n * 100)
            pct_det = float(np.sum(finite < -threshold) / n * 100)
            pct_neu = 100.0 - pct_imp - pct_det
        return {
            "pct_improvement": pct_imp,
            "pct_neutral": pct_neu,
            "pct_deterioration": pct_det,
        }

    # -- NetCDF I/O ---------------------------------------------------------

    def _save_av_to_nc(
        self,
        av: xr.DataArray,
        var: str,
        period: str,
        ensemble_type: str,
        meta: dict[str, Any],
    ) -> None:
        """Save a single AV field as a CMORized NetCDF file.

        Parameters
        ----------
        av : xr.DataArray
            AV field on a regular lat/lon grid.
        var : str
            Feather variable name (e.g. ``"tas"``).
        period : str
            Period label (``"annual"``, ``"djf"``, ``"jja"``).
        ensemble_type : str
            ``"mean"`` or ``"median"``.
        meta : dict
            Provenance metadata written to variable attributes.
        """
        nc_path = self._nc_path(var, period, ensemble_type)
        ds = xr.Dataset(
            {
                "av": xr.DataArray(
                    av.values,
                    dims=("lat", "lon"),
                    coords={
                        "lat": av["lat"].values,
                        "lon": av["lon"].values,
                    },
                    attrs={
                        "long_name": (
                            f"Added Value: EERIE ({ensemble_type}) vs "
                            f"CMIP6 MMM for {meta['long_name']} "
                            f"(AV>0 means EERIE adds value)"
                        ),
                        "units": "1",
                        "valid_range": np.array([-1.0, 1.0]),
                        "reference": (
                            "Dosio et al. (2015), doi:10.1007/s00382-015-2869-x"
                        ),
                        "model1": "CMIP6 multi-model mean",
                        "model2": f"EERIE {ensemble_type}",
                        "reference_dataset": meta.get("obs_dataset", "ERA5"),
                        "ensemble_type": ensemble_type,
                        "n_eerie_models": meta["n_eerie_models"],
                        "n_cmip6_models": meta["n_cmip6_models"],
                        "eerie_models": ",".join(meta["eerie_models"]),
                        "cmip6_models": ",".join(meta["cmip6_models"]),
                        "period_start": meta["period_start"],
                        "period_end": meta["period_end"],
                        "source_variable": var,
                        "source_variable_units": meta["units"],
                        "period": period,
                    },
                ),
            }
        )
        ds["lat"].attrs = {"units": "degrees_north", "axis": "Y"}
        ds["lon"].attrs = {"units": "degrees_east", "axis": "X"}
        ds.attrs = {
            "Conventions": "CF-1.8",
            "title": (
                f"Added Value: EERIE {ensemble_type} vs CMIP6 MMM — "
                f"{meta['long_name']} ({period})"
            ),
            "institution": "Feather climate evaluation framework",
            "source": "feather/diag/added_value.py",
        }
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(nc_path)
        logger.info("  Saved AV NetCDF: %s", nc_path)

    # -- Regridding helper (shared with GlobalBiases/GlobalTrends) ----------

    @staticmethod
    def _regrid_to_target(
        da, target_lats, target_lons,
        resolution, influence_radius,
        interp_cache, method="nearest",
    ):
        """Regrid a regular lat/lon DataArray to the target grid via nereus.

        250 km floor on influence_radius ensures CMIP6 coarse grids (~2°)
        are covered.  Source lons shifted to -180..180 to avoid a NaN
        stripe at the prime meridian with Delaunay triangulation.
        """
        ir = max(influence_radius, 250_000.0)

        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lat_arr = da[lat_name].values
        lon_arr = da[lon_name].values

        lon_arr = np.where(lon_arr > 180, lon_arr - 360, lon_arr)
        sort_idx = np.argsort(lon_arr)
        lon_arr = lon_arr[sort_idx]

        grid_key = (len(lat_arr), len(lon_arr))
        if grid_key not in interp_cache:
            lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
            _, interp_cache[grid_key] = nr.regrid(
                da.values[:, sort_idx].ravel(),
                lon=lon_2d.ravel(), lat=lat_2d.ravel(),
                resolution=resolution,
                method=method,
                influence_radius=ir,
                lon_bounds=(-180.0, 180.0),
                as_xarray=True,
            )

        regridded = interp_cache[grid_key](da.values[:, sort_idx].ravel())
        n_roll = regridded.shape[1] // 2
        regridded = np.roll(regridded, -n_roll, axis=1)

        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    # -- Multi-obs statistics -----------------------------------------------

    def _regrid_obs_to_common_grid(
        self,
        obs_da: xr.DataArray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        influence_radius: float,
    ) -> xr.DataArray:
        """Regrid a lat/lon obs DataArray to the common target grid.

        Used to evaluate AV statistics against secondary obs datasets
        without changing the primary obs used for map plots.

        The nereus interpolator is built with the **common grid's**
        resolution (derived from target_lats), not the source resolution,
        so the output always has shape (len(target_lats), len(target_lons)).
        """
        lat_name = "lat" if "lat" in obs_da.coords else "latitude"
        lon_name = "lon" if "lon" in obs_da.coords else "longitude"
        obs_lats = obs_da[lat_name].values
        obs_lons = obs_da[lon_name].values

        # Use the common (target) grid resolution, not the source resolution.
        # nr.regrid() defines its output grid from this parameter; using the
        # source's own resolution would create a mismatched grid when the
        # secondary obs is coarser than the primary obs (e.g. ERA5 at 0.25°
        # vs MSWEP common grid at 0.1°).
        target_res = abs(float(target_lats[1] - target_lats[0]))

        _, interp = nr.regrid(
            obs_da.values,
            lon=obs_lons, lat=obs_lats,
            resolution=target_res,
            influence_radius=influence_radius,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        regridded = interp(obs_da.values.ravel())
        return xr.DataArray(
            regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons},
        )

    @staticmethod
    def _compute_period_category_stats(
        eerie_mean: xr.DataArray,
        eerie_median: xr.DataArray,
        cmip6_mmm: xr.DataArray,
        obs: xr.DataArray,
        eerie_individual: dict[str, xr.DataArray] | None = None,
        area: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute area-weighted improvement/neutral/degradation fractions.

        Returns a dict with keys ``eerie_mean``, ``eerie_median``,
        ``cmip6_mean``, each containing ``pct_improvement``,
        ``pct_neutral``, ``pct_deterioration``.

        ``cmip6_mean`` uses the EERIE ensemble mean as the baseline
        (AV > 0 means CMIP6 reduces error vs EERIE).

        When ``eerie_individual`` is supplied, also adds
        ``per_eerie_models`` with per-model fractions.

        When *area* is supplied, percentages are area-weighted.
        """
        av_em = AddedValueDiag._compute_av(cmip6_mmm, eerie_mean, obs)
        av_emd = AddedValueDiag._compute_av(cmip6_mmm, eerie_median, obs)
        av_c = AddedValueDiag._compute_av(eerie_mean, cmip6_mmm, obs)
        result: dict[str, Any] = {
            "eerie_mean":   AddedValueDiag._frac_categories(av_em, area=area),
            "eerie_median": AddedValueDiag._frac_categories(av_emd, area=area),
            "cmip6_mean":   AddedValueDiag._frac_categories(av_c, area=area),
        }
        if eerie_individual:
            result["per_eerie_models"] = {
                name: AddedValueDiag._frac_categories(
                    AddedValueDiag._compute_av(cmip6_mmm, field, obs),
                    area=area,
                )
                for name, field in eerie_individual.items()
            }
        return result

    def _compute_multi_obs_stats(
        self,
        var: str,
        eerie_mean: xr.DataArray,
        eerie_median: xr.DataArray,
        cmip6_mmm: xr.DataArray,
        eerie_seasonal_mean: dict[str, xr.DataArray],
        eerie_seasonal_median: dict[str, xr.DataArray],
        cmip6_seasonal_mmm: dict[str, xr.DataArray],
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        influence_radius: float,
        eerie_individual: dict[str, xr.DataArray] | None = None,
        eerie_seasonal_individual: dict[str, dict[str, xr.DataArray]] | None = None,
    ) -> dict[str, dict]:
        """Compute category stats (improvement/neutral/deterioration) for each
        available obs dataset.

        Parameters
        ----------
        var : str
            Variable name (CMOR canonical).
        eerie_mean, eerie_median, cmip6_mmm : xr.DataArray
            Annual-mean fields on the common grid.
        eerie_seasonal_mean/median, cmip6_seasonal_mmm : dict
            Season-keyed fields (``"DJF"``, ``"JJA"``) on the common grid.
        target_lats, target_lons : np.ndarray
            Common grid coordinates.
        influence_radius : float
            nereus influence radius in metres (for obs regridding).

        Returns
        -------
        dict keyed by obs dataset name (e.g. ``"ERA5"``,
        ``"BERKELEY_EARTH_HR"``, ``"MSWEP"``). Each value is a dict
        keyed by period (``"annual"``, ``"djf"``, ``"jja"``), containing
        ``eerie_mean``, ``eerie_median``, ``cmip6_mean`` sub-dicts of
        ``pct_improvement``, ``pct_neutral``, ``pct_deterioration``.
        """
        obs_names = self._MULTI_OBS_DATASETS.get(var, ["ERA5"])
        obs_names = [
            n for n in obs_names
            if n == "ERA5" or n in self.config.obs_datasets
        ]
        unit_factor = self._UNIT_FACTORS.get(var, 1.0)
        common_area = compute_latlon_areas(target_lats, target_lons)

        stats: dict[str, dict] = {}
        for obs_name in obs_names:
            try:
                logger.info("  Computing multi-obs stats: %s / %s", var, obs_name)
                obs_data = self._load_obs_by_name(var, obs_name, self.period)
                # ERA5 raw units need conversion (MSWEP/BE handle internally)
                if obs_name == "ERA5" and unit_factor != 1.0:
                    obs_data = obs_data * unit_factor

                from feather.util.temporal import climatology as _clim
                from feather.util.temporal import seasonal_climatology as _sclim
                obs_clim = _clim(obs_data)
                obs_common = self._regrid_obs_to_common_grid(
                    obs_clim, target_lats, target_lons, influence_radius,
                )

                periods_stats: dict[str, dict] = {}
                periods_stats["annual"] = self._compute_period_category_stats(
                    eerie_mean, eerie_median, cmip6_mmm, obs_common,
                    eerie_individual=eerie_individual,
                    area=common_area,
                )

                obs_seasonal = _sclim(obs_data)
                for season in ["DJF", "JJA"]:
                    if (
                        season in obs_seasonal
                        and season in eerie_seasonal_mean
                        and season in eerie_seasonal_median
                        and season in cmip6_seasonal_mmm
                    ):
                        obs_s_common = self._regrid_obs_to_common_grid(
                            obs_seasonal[season],
                            target_lats, target_lons, influence_radius,
                        )
                        sea_ind = (
                            eerie_seasonal_individual.get(season)
                            if eerie_seasonal_individual else None
                        )
                        periods_stats[season.lower()] = (
                            self._compute_period_category_stats(
                                eerie_seasonal_mean[season],
                                eerie_seasonal_median[season],
                                cmip6_seasonal_mmm[season],
                                obs_s_common,
                                eerie_individual=sea_ind,
                                area=common_area,
                            )
                        )

                stats[obs_name] = periods_stats

            except Exception:
                logger.warning(
                    "  Multi-obs stats failed for %s / %s — skipping",
                    obs_name, var, exc_info=True,
                )

        return stats

    def _obs_stats_path(self, var: str) -> Path:
        """Path for the per-variable obs-comparison stats JSON."""
        return self.nc_dir / f"{var}_obs_stats.json"

    def _save_obs_stats_json(
        self, var: str, obs_stats: dict, nc_meta: dict,
    ) -> None:
        """Persist per-obs improvement/neutral/deterioration stats to JSON.

        The file is written to ``{nc_dir}/{var}_obs_stats.json`` and is
        suitable for direct use in barplot comparisons across obs datasets.

        Schema
        ------
        .. code-block:: json

            {
              "variable": "tas",
              "period": ["1980", "2014"],
              "threshold": 0.001,
              "periods": {
                "annual": {
                  "ERA5": {
                    "eerie_mean":   {"pct_improvement": 62.1, ...},
                    "eerie_median": {"pct_improvement": 63.4, ...},
                    "cmip6_mean":   {"pct_improvement": 38.2, ...}
                  },
                  "BERKELEY_EARTH_HR": { ... }
                },
                "djf": { ... },
                "jja": { ... }
              }
            }
        """
        # Transpose obs_stats from {obs_name: {period: ...}}
        # to {period: {obs_name: ...}} — friendlier for barplot consumers.
        by_period: dict[str, dict] = {}
        for obs_name, periods in obs_stats.items():
            for period_key, period_data in periods.items():
                by_period.setdefault(period_key, {})[obs_name] = period_data

        payload = {
            "variable": var,
            "period": [nc_meta["period_start"], nc_meta["period_end"]],
            "threshold": 0.001,
            "eerie_models": nc_meta.get("eerie_models", []),
            "n_eerie_models": nc_meta.get("n_eerie_models", 0),
            "cmip6_models": nc_meta.get("cmip6_models", []),
            "n_cmip6_models": nc_meta.get("n_cmip6_models", 0),
            "periods": by_period,
        }
        out_path = self._obs_stats_path(var)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("  Saved obs stats JSON: %s", out_path)

    # -- Plotting -----------------------------------------------------------

    def plot(
        self, results: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Generate AV map figures for all variables."""
        figures: list[tuple[plt.Figure, dict]] = []
        for var, vr in results.items():
            figures.extend(self._plot_variable(var, vr))
        return figures

    def _plot_variable(
        self, var: str, vr: dict[str, Any],
    ) -> list[tuple[plt.Figure, dict]]:
        """Generate AV map figures for a single variable.

        Iterates over all available obs datasets in ``vr["av_by_obs"]``.
        For each obs dataset × period:

        Figure 1 — ensemble summary (two panels):
          - AV(CMIP6 MMM, EERIE ensemble mean, <obs>)
          - AV(CMIP6 MMM, EERIE ensemble median, <obs>)

        Figure 2 — individual model panels:
          - One panel per EERIE model:  AV(CMIP6 MMM, EERIE_i, <obs>)
          - One panel per CMIP6 model:  AV(EERIE ens. mean, CMIP6_j, <obs>)

        The primary obs dataset keeps the legacy figure IDs (no suffix).
        Secondary obs datasets (e.g. ERA5 for tas/pr) get an ``_{suffix}`` suffix.
        """
        figures: list[tuple[plt.Figure, dict]] = []
        var_info = vr["var_info"]
        primary_obs_name = vr.get("obs_dataset_name", "ERA5")
        av_by_obs: dict[str, dict] = vr.get("av_by_obs", {primary_obs_name: vr["av"]})

        for obs_name, av in av_by_obs.items():
            is_primary = obs_name == primary_obs_name
            obs_suffix = (
                ""
                if is_primary
                else f"_{self._OBS_FIGURE_SUFFIX.get(obs_name, obs_name.lower())}"
            )
            obs_label = _OBS_DISPLAY_NAMES.get(obs_name, obs_name)

            # Per-obs category stats for barplot metadata (primary obs only)
            obs_stats_all: dict[str, dict] = vr.get("obs_stats", {}) if is_primary else {}

            period_labels = [("annual", "Annual")]
            for s in ("DJF", "JJA"):
                if s in av:
                    period_labels.append((s, s))

            for period_key, period_label in period_labels:
                period_data = av.get(period_key)
                if period_data is None:
                    continue

                pk_lower = period_key.lower()

                # ── Figure 1: ensemble mean + median ───────────────────────
                summary_stats: dict[str, Any] = {}
                data_dict: dict[str, xr.DataArray] = {}
                for etype, label in (
                    ("ensemble_mean",   "AV(EERIE Ens. Mean)"),
                    ("ensemble_median", "AV(EERIE Ens. Median)"),
                ):
                    dom_av = period_data[f"{etype}_domain_av"]
                    frac = period_data[f"{etype}_frac_positive"]
                    panel_title = (
                        f"{label}\n"
                        f"domain mean={dom_av:+.3f}, AV>0: {frac:.0%}"
                    )
                    data_dict[panel_title] = period_data[etype]
                    summary_stats[etype] = {
                        "domain_mean_av": dom_av,
                        "frac_positive": frac,
                        "n_eerie_models": vr["n_eerie_models"],
                        "n_cmip6_models": vr["n_cmip6_models"],
                    }

                per_obs_period: dict[str, dict] = {}
                for _obs, _periods in obs_stats_all.items():
                    _ps = _periods.get(pk_lower)
                    if _ps is not None:
                        per_obs_period[_obs] = _ps
                if per_obs_period:
                    summary_stats["per_obs_stats"] = per_obs_period

                fig1, _ = plot_combined_map(
                    data_dict,
                    title=(
                        f"{var_info.long_name} {period_label} Added Value"
                        f" — EERIE ensemble vs CMIP6 MMM"
                        f"  (vs {obs_label}, green = EERIE better)"
                    ),
                    cmap=_AV_CMAP,
                    vmin=-1.0, vmax=1.0,
                    units="AV [ ]",
                    method=self._regrid_method,
                )
                meta1 = self._build_metadata(
                    title=(
                        f"{var_info.long_name} {period_label} Added Value "
                        f"(EERIE ensemble vs CMIP6 MMM, obs: {obs_label})"
                    ),
                    figure_id=f"{var}_{pk_lower}_added_value{obs_suffix}",
                    models=vr["eerie_models"],
                    variables=[var],
                    description=(
                        f"Dosio et al. (2015) Added Value metric for "
                        f"{var_info.long_name} ({period_label}), "
                        f"{self.period[0]}-{self.period[1]}. "
                        f"Reference obs: {obs_label}. "
                        f"AV > 0: EERIE ensemble reduces squared error over "
                        f"CMIP6 MMM. "
                        f"EERIE n={vr['n_eerie_models']}, "
                        f"CMIP6 n={vr['n_cmip6_models']}."
                    ),
                    plot_type="added_value_map",
                    period=self.period,
                    summary_statistics=summary_stats,
                    extra={
                        "eerie_models": vr["eerie_models"],
                        "cmip6_models": vr["cmip6_models"],
                        "obs_dataset": obs_name,
                        "reference": "Dosio et al. (2015)",
                    },
                )
                figures.append((fig1, meta1))

                # ── Figure 2: individual model panels ─────────────────────
                per_eerie = period_data.get("per_eerie_av", {})
                per_cmip6 = period_data.get("per_cmip6_av", {})
                if not per_eerie and not per_cmip6:
                    continue

                # Derive common area from the first available AV field's grid.
                _all_av_fields = list(per_eerie.values()) + list(per_cmip6.values())
                _panel_area = compute_latlon_areas(
                    _all_av_fields[0]["lat"].values,
                    _all_av_fields[0]["lon"].values,
                ) if _all_av_fields else None

                models_data_dict: dict[str, xr.DataArray] = {}
                for model_name, av_field in per_eerie.items():
                    dom_av = self._domain_mean_av(av_field, _panel_area)
                    frac = self._frac_positive(av_field, _panel_area)
                    title_str = (
                        f"EERIE: {model_name}\n"
                        f"vs CMIP6 MMM — mean={dom_av:+.3f}, AV>0: {frac:.0%}"
                    )
                    models_data_dict[title_str] = av_field
                for cmip6_label, av_field in per_cmip6.items():
                    dom_av = self._domain_mean_av(av_field, _panel_area)
                    frac = self._frac_positive(av_field, _panel_area)
                    title_str = (
                        f"CMIP6: {cmip6_label}\n"
                        f"vs EERIE mean — mean={dom_av:+.3f}, AV>0: {frac:.0%}"
                    )
                    models_data_dict[title_str] = av_field

                fig2, _ = plot_combined_map(
                    models_data_dict,
                    title=(
                        f"{var_info.long_name} {period_label} Added Value"
                        f" — Individual Models"
                        f"  (vs {obs_label}, green = model better)"
                    ),
                    cmap=_AV_CMAP,
                    vmin=-1.0, vmax=1.0,
                    units="AV [ ]",
                    method=self._regrid_method,
                )
                eerie_stats = {
                    m: {
                        "domain_mean_av": self._domain_mean_av(av_f, _panel_area),
                        "frac_positive": self._frac_positive(av_f, _panel_area),
                    }
                    for m, av_f in per_eerie.items()
                }
                cmip6_stats = {
                    lbl: {
                        "domain_mean_av": self._domain_mean_av(av_f, _panel_area),
                        "frac_positive": self._frac_positive(av_f, _panel_area),
                    }
                    for lbl, av_f in per_cmip6.items()
                }
                meta2 = self._build_metadata(
                    title=(
                        f"{var_info.long_name} {period_label} Added Value "
                        f"— Individual Models (obs: {obs_label})"
                    ),
                    figure_id=f"{var}_{pk_lower}_added_value_models{obs_suffix}",
                    models=vr["eerie_models"],
                    variables=[var],
                    description=(
                        f"Per-model Dosio et al. (2015) Added Value for "
                        f"{var_info.long_name} ({period_label}), "
                        f"{self.period[0]}-{self.period[1]}. "
                        f"Reference obs: {obs_label}. "
                        f"EERIE panels: AV(CMIP6 MMM, EERIE_i, {obs_label}). "
                        f"CMIP6 panels: AV(EERIE mean, CMIP6_j, {obs_label}). "
                        f"Green = model better than its baseline."
                    ),
                    plot_type="added_value_map",
                    period=self.period,
                    summary_statistics={
                        "per_eerie": eerie_stats,
                        "per_cmip6": cmip6_stats,
                    },
                    extra={
                        "eerie_models": vr["eerie_models"],
                        "cmip6_models": vr["cmip6_models"],
                        "obs_dataset": obs_name,
                        "reference": "Dosio et al. (2015)",
                    },
                )
                figures.append((fig2, meta2))

        return figures

    # -- Summary bar charts -------------------------------------------------

    def _plot_summary_bars_ensemble(
        self,
        all_obs_stats: dict[str, dict],
        period_key: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Grouped horizontal stacked bar chart — ensemble view.

        3 panels: ERA5 (all variables), Berkeley Earth HR (tas only),
        MSWEP (pr only).  For each variable within a panel:
          3 bars (EERIE mean / EERIE median / CMIP6 mean),
        each stacked [improvement | neutral | deterioration].

        Colors: EERIE improvement = blue palette, CMIP6 = green,
        neutral = light gray, deterioration = always red.
        """
        from matplotlib import gridspec as mgs
        from matplotlib.patches import Patch

        period_label = {"annual": "Annual", "djf": "DJF", "jja": "JJA"}.get(
            period_key, period_key.upper()
        )

        # ── Gather data per obs panel ───────────────────────────────────────
        obs_panels = [
            ("ERA5",             "ERA5"),
            ("BERKELEY_EARTH_HR", "Berkeley Earth HR"),
            ("MSWEP",             "MSWEP v2.8"),
        ]
        panel_rows: dict[str, list[tuple[str, str, dict]]] = {}
        for obs_name, _ in obs_panels:
            rows: list[tuple[str, str, dict]] = []
            for var, vr_stats in all_obs_stats.items():
                period_stats = vr_stats.get(obs_name, {}).get(period_key, {})
                if period_stats:
                    rows.append((var, get_var(var).long_name, period_stats))
            if rows:
                panel_rows[obs_name] = rows

        if not panel_rows:
            return []

        active = [(n, lbl) for n, lbl in obs_panels if n in panel_rows]
        n_panels = len(active)

        bar_h = 0.22
        grp_pad = 0.15  # gap between variable groups
        n_bars = 3       # EERIE mean, EERIE median, CMIP6 mean

        def _panel_height(n_vars: int) -> float:
            return n_vars * (n_bars * bar_h + grp_pad) + 0.8

        heights = [_panel_height(len(panel_rows[n])) for n, _ in active]
        fig = plt.figure(figsize=(13, max(5, sum(heights) + 1.2)))
        gs = mgs.GridSpec(
            n_panels, 1,
            height_ratios=heights,
            hspace=0.55,
            figure=fig,
        )

        etype_colors = {
            "eerie_mean":   "#1f77b4",
            "eerie_median": "#6aaed6",
            "cmip6_mean":   "#2ca02c",
        }
        etype_labels = {
            "eerie_mean":   "EERIE mean",
            "eerie_median": "EERIE median",
            "cmip6_mean":   "CMIP6 mean",
        }
        neutral_color = "#d5d5d5"
        det_color = "#c0392b"
        etypes = ["eerie_mean", "eerie_median", "cmip6_mean"]

        for pi, (obs_name, obs_label) in enumerate(active):
            rows = panel_rows[obs_name]
            n_vars = len(rows)
            ax = fig.add_subplot(gs[pi])

            grp_height = n_bars * bar_h + grp_pad
            grp_centers = np.arange(n_vars) * grp_height
            y_offsets = np.array([(i - (n_bars - 1) / 2) * bar_h
                                  for i in range(n_bars)])

            for ei, etype in enumerate(etypes):
                imp = np.array([
                    r[2].get(etype, {}).get("pct_improvement", 0.0) for r in rows
                ])
                neu = np.array([
                    r[2].get(etype, {}).get("pct_neutral", 0.0) for r in rows
                ])
                det = np.array([
                    r[2].get(etype, {}).get("pct_deterioration", 0.0) for r in rows
                ])
                ys = grp_centers + y_offsets[ei]
                ec = etype_colors[etype]
                ax.barh(ys, imp, height=bar_h, color=ec,
                        label=etype_labels[etype])
                ax.barh(ys, neu, height=bar_h, left=imp,
                        color=neutral_color, label="_")
                ax.barh(ys, det, height=bar_h, left=imp + neu,
                        color=det_color, label="_")

            ax.set_yticks(grp_centers)
            ax.set_yticklabels([r[1] for r in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of area", fontsize=9)
            ax.axvline(50, color="k", lw=0.5, ls="--", alpha=0.35)
            ax.set_title(obs_label, fontsize=10, fontweight="bold", pad=4)
            ax.tick_params(axis="x", labelsize=8)

        # Shared legend on first panel
        first_ax = fig.axes[0]
        legend_handles = [
            Patch(facecolor=etype_colors[e], label=etype_labels[e])
            for e in etypes
        ] + [
            Patch(facecolor=neutral_color, label="Neutral"),
            Patch(facecolor=det_color, label="Degradation"),
        ]
        first_ax.legend(
            handles=legend_handles, fontsize=8,
            loc="lower right", framealpha=0.85,
        )

        fig.suptitle(
            f"Added Value — {period_label}: area-weighted % improvement / neutral / degradation\n"
            f"EERIE ensemble vs CMIP6 MMM",
            fontsize=11, fontweight="bold", y=1.01,
        )

        figure_id = f"added_value_bars_ensemble_{period_key}"
        meta = self._build_metadata(
            title=f"Added Value Summary — {period_label} (ensemble view)",
            figure_id=figure_id,
            models=list(self.config.models),
            variables=list(all_obs_stats.keys()),
            description=(
                f"Summary bar chart of area-weighted improvement/neutral/degradation "
                f"fractions ({period_label}) for all variables and obs datasets. "
                f"Blue = EERIE improves, green = CMIP6 reference, red = degradation."
            ),
            plot_type="added_value_bars",
            period=self.period,
        )
        return [(fig, meta)]

    def _plot_summary_bars_models(
        self,
        all_obs_stats: dict[str, dict],
        period_key: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Grouped horizontal stacked bar chart — per-EERIE-model view.

        Same layout as ``_plot_summary_bars_ensemble`` but each variable
        group shows one bar per EERIE model (blue palette) plus CMIP6 mean
        (green).  Requires ``per_eerie_models`` in obs_stats.
        """
        from matplotlib import gridspec as mgs
        from matplotlib.patches import Patch

        period_label = {"annual": "Annual", "djf": "DJF", "jja": "JJA"}.get(
            period_key, period_key.upper()
        )

        # Collect model names from first available entry
        eerie_model_names: list[str] = []
        for vr_stats in all_obs_stats.values():
            for _obs, pdata in vr_stats.items():
                pm = pdata.get(period_key, {}).get("per_eerie_models", {})
                if pm:
                    eerie_model_names = list(pm.keys())
                    break
            if eerie_model_names:
                break

        if not eerie_model_names:
            logger.debug("No per_eerie_models data — skipping bars_models figure")
            return []

        obs_panels = [
            ("ERA5",             "ERA5"),
            ("BERKELEY_EARTH_HR", "Berkeley Earth HR"),
            ("MSWEP",             "MSWEP v2.8"),
        ]
        panel_rows: dict[str, list[tuple[str, str, dict]]] = {}
        for obs_name, _ in obs_panels:
            rows: list[tuple[str, str, dict]] = []
            for var, vr_stats in all_obs_stats.items():
                period_stats = vr_stats.get(obs_name, {}).get(period_key, {})
                if period_stats and period_stats.get("per_eerie_models"):
                    rows.append((var, get_var(var).long_name, period_stats))
            if rows:
                panel_rows[obs_name] = rows

        if not panel_rows:
            return []

        active = [(n, lbl) for n, lbl in obs_panels if n in panel_rows]
        n_panels = len(active)
        n_bars = len(eerie_model_names) + 1  # models + CMIP6 mean

        # Blue palette: darker shades for more models
        _blue_palette = ["#08519c", "#2171b5", "#4292c6", "#6baed6",
                         "#9ecae1", "#c6dbef"]
        eerie_colors = {m: _blue_palette[i % len(_blue_palette)]
                        for i, m in enumerate(eerie_model_names)}
        cmip6_color = "#2ca02c"
        neutral_color = "#d5d5d5"
        det_color = "#c0392b"

        bar_h = 0.20
        grp_pad = 0.18

        def _panel_height(n_vars: int) -> float:
            return n_vars * (n_bars * bar_h + grp_pad) + 0.8

        heights = [_panel_height(len(panel_rows[n])) for n, _ in active]
        fig = plt.figure(figsize=(13, max(5, sum(heights) + 1.2)))
        gs = mgs.GridSpec(
            n_panels, 1,
            height_ratios=heights,
            hspace=0.55,
            figure=fig,
        )

        for pi, (obs_name, obs_label) in enumerate(active):
            rows = panel_rows[obs_name]
            n_vars = len(rows)
            ax = fig.add_subplot(gs[pi])

            grp_height = n_bars * bar_h + grp_pad
            grp_centers = np.arange(n_vars) * grp_height
            y_offsets = np.array([(i - (n_bars - 1) / 2) * bar_h
                                  for i in range(n_bars)])

            # EERIE individual models
            for mi, model_name in enumerate(eerie_model_names):
                imp = np.array([
                    r[2].get("per_eerie_models", {})
                       .get(model_name, {})
                       .get("pct_improvement", 0.0)
                    for r in rows
                ])
                neu = np.array([
                    r[2].get("per_eerie_models", {})
                       .get(model_name, {})
                       .get("pct_neutral", 0.0)
                    for r in rows
                ])
                det = np.array([
                    r[2].get("per_eerie_models", {})
                       .get(model_name, {})
                       .get("pct_deterioration", 0.0)
                    for r in rows
                ])
                ys = grp_centers + y_offsets[mi]
                ec = eerie_colors[model_name]
                ax.barh(ys, imp, height=bar_h, color=ec, label=model_name)
                ax.barh(ys, neu, height=bar_h, left=imp,
                        color=neutral_color, label="_")
                ax.barh(ys, det, height=bar_h, left=imp + neu,
                        color=det_color, label="_")

            # CMIP6 mean bar (last in group)
            imp_c = np.array([
                r[2].get("cmip6_mean", {}).get("pct_improvement", 0.0)
                for r in rows
            ])
            neu_c = np.array([
                r[2].get("cmip6_mean", {}).get("pct_neutral", 0.0)
                for r in rows
            ])
            det_c = np.array([
                r[2].get("cmip6_mean", {}).get("pct_deterioration", 0.0)
                for r in rows
            ])
            ys_c = grp_centers + y_offsets[len(eerie_model_names)]
            ax.barh(ys_c, imp_c, height=bar_h, color=cmip6_color,
                    label="CMIP6 mean")
            ax.barh(ys_c, neu_c, height=bar_h, left=imp_c,
                    color=neutral_color, label="_")
            ax.barh(ys_c, det_c, height=bar_h, left=imp_c + neu_c,
                    color=det_color, label="_")

            ax.set_yticks(grp_centers)
            ax.set_yticklabels([r[1] for r in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of area", fontsize=9)
            ax.axvline(50, color="k", lw=0.5, ls="--", alpha=0.35)
            ax.set_title(obs_label, fontsize=10, fontweight="bold", pad=4)
            ax.tick_params(axis="x", labelsize=8)

        from matplotlib.patches import Patch
        first_ax = fig.axes[0]
        legend_handles = (
            [Patch(facecolor=eerie_colors[m], label=m) for m in eerie_model_names]
            + [Patch(facecolor=cmip6_color, label="CMIP6 mean")]
            + [
                Patch(facecolor=neutral_color, label="Neutral"),
                Patch(facecolor=det_color, label="Degradation"),
            ]
        )
        first_ax.legend(
            handles=legend_handles, fontsize=8,
            loc="lower right", framealpha=0.85,
        )

        fig.suptitle(
            f"Added Value — {period_label}: per-model area-weighted % improvement / neutral / degradation\n"
            f"EERIE models vs CMIP6 MMM",
            fontsize=11, fontweight="bold", y=1.01,
        )

        figure_id = f"added_value_bars_models_{period_key}"
        meta = self._build_metadata(
            title=f"Added Value Summary — {period_label} (per-model view)",
            figure_id=figure_id,
            models=list(self.config.models),
            variables=list(all_obs_stats.keys()),
            description=(
                f"Per-model summary bar chart of area-weighted improvement/neutral/degradation "
                f"fractions ({period_label}). "
                f"Blue shades = EERIE models vs CMIP6 MMM, "
                f"green = CMIP6 mean vs EERIE mean, red = degradation."
            ),
            plot_type="added_value_bars",
            period=self.period,
        )
        return [(fig, meta)]
