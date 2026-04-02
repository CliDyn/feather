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
from feather.plot.maps import plot_combined_map
from feather.util.spatial import compute_latlon_areas, latlon_global_mean
from feather.util.temporal import climatology, seasonal_climatology

logger = logging.getLogger(__name__)


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
            for etype in ("mean", "median"):
                if not self._nc_path(var, period, etype).exists():
                    return False
        return True

    def _all_figures_exist(self, var: str) -> bool:
        """True when all 3 period figures exist."""
        return all(
            self._figure_exists(f"{var}_{p}_added_value")
            for p in ("annual", "djf", "jja")
        )

    # -- Orchestration -------------------------------------------------------

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-variable: compute → save NC → plot → save figures.

        NC files are the durable checkpoint.  If all NC files for a
        variable already exist, computation is skipped and figures are
        regenerated directly from the saved NetCDF (or also skipped if
        figures are present too).
        """
        logger.info("Running diagnostic: %s", self.name)
        self.nc_dir.mkdir(parents=True, exist_ok=True)
        saved: list[tuple[Path, Path]] = []

        for var in self.variables:
            if skip_existing and self._all_figures_exist(var):
                logger.info("Skipping %s — all figures exist", var)
                for p in ("annual", "djf", "jja"):
                    fid = f"{var}_{p}_added_value"
                    saved.append((
                        self.output_dir / f"{fid}.png",
                        self.output_dir / f"{fid}.json",
                    ))
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

                figures = self._plot_variable(var, var_result)
                for fig, meta in figures:
                    paths = self._save(fig, meta, meta["figure_id"])
                    saved.append(paths)

            except Exception:
                logger.warning(
                    "Variable %s failed — skipping", var, exc_info=True,
                )

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

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
        obs_data = self._load_obs_var(var, self.period)
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
            model_seasonal = seasonal_climatology(model_data, self.period)
            model_seasonal = {
                s: model_seasonal[s].compute()
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
            cmip6_annual_fields.append(regridded)
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
                    cmip6_seasonal_fields[season].append(s_r)

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
        av_results: dict[str, dict] = {}

        # Annual
        av_mean_annual = self._compute_av(
            eerie_mean, cmip6_mmm, obs_clim_common,
        )
        av_median_annual = self._compute_av(
            eerie_median, cmip6_mmm, obs_clim_common,
        )
        av_results["annual"] = {
            "mean": av_mean_annual,
            "median": av_median_annual,
            "mean_domain_av": self._domain_mean_av(
                av_mean_annual, common_area,
            ),
            "median_domain_av": self._domain_mean_av(
                av_median_annual, common_area,
            ),
            "mean_frac_positive": self._frac_positive(av_mean_annual),
            "median_frac_positive": self._frac_positive(av_median_annual),
        }

        # Seasonal
        for season in ["DJF", "JJA"]:
            if (
                season in eerie_seasonal_mean
                and season in cmip6_seasonal_mmm
                and season in obs_seasonal_common
            ):
                av_m = self._compute_av(
                    eerie_seasonal_mean[season],
                    cmip6_seasonal_mmm[season],
                    obs_seasonal_common[season],
                )
                av_med = self._compute_av(
                    eerie_seasonal_median[season],
                    cmip6_seasonal_mmm[season],
                    obs_seasonal_common[season],
                )
                av_results[season] = {
                    "mean": av_m,
                    "median": av_med,
                    "mean_domain_av": self._domain_mean_av(
                        av_m, common_area,
                    ),
                    "median_domain_av": self._domain_mean_av(
                        av_med, common_area,
                    ),
                    "mean_frac_positive": self._frac_positive(av_m),
                    "median_frac_positive": self._frac_positive(av_med),
                }

        # -- Save NetCDF files ----------------------------------------------
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
        }
        for period_key, period_data in av_results.items():
            for etype in ("mean", "median"):
                nc_path = self._nc_path(var, period_key.lower(), etype)
                if not nc_path.exists():
                    self._save_av_to_nc(
                        period_data[etype], var,
                        period_key.lower(), etype, nc_meta,
                    )

        return {
            "var_info": var_info,
            "av": av_results,
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

        for period in ("annual", "djf", "jja"):
            mean_path = self._nc_path(var, period, "mean")
            median_path = self._nc_path(var, period, "median")
            if not mean_path.exists() or not median_path.exists():
                continue

            ds_mean = xr.open_dataset(mean_path)
            ds_median = xr.open_dataset(median_path)
            av_m = ds_mean["av"]
            av_med = ds_median["av"]

            # Reload area weights from coordinates
            lat_vals = av_m["lat"].values
            lon_vals = av_m["lon"].values
            area = compute_latlon_areas(lat_vals, lon_vals)

            av_results[period] = {
                "mean": av_m,
                "median": av_med,
                "mean_domain_av": self._domain_mean_av(av_m, area),
                "median_domain_av": self._domain_mean_av(av_med, area),
                "mean_frac_positive": self._frac_positive(av_m),
                "median_frac_positive": self._frac_positive(av_med),
            }
            ds_mean.close()
            ds_median.close()

        if not av_results:
            return None

        # Recover model lists from NC attributes
        ds = xr.open_dataset(self._nc_path(var, "annual", "mean"))
        n_eerie = int(ds["av"].attrs.get("n_eerie_models", 0))
        n_cmip6 = int(ds["av"].attrs.get("n_cmip6_models", 0))
        eerie_models = ds["av"].attrs.get("eerie_models", "").split(",")
        cmip6_models = ds["av"].attrs.get("cmip6_models", "").split(",")
        ds.close()

        return {
            "var_info": var_info,
            "av": av_results,
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
            EERIE ensemble mean or median climatology.
        m2 : xr.DataArray
            CMIP6 multi-model mean climatology.
        ref : xr.DataArray
            Observation climatology (ERA5).

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
        av: xr.DataArray, area: np.ndarray,
    ) -> float:
        """Area-weighted domain-mean AV score."""
        return float(latlon_global_mean(av, area=area).values)

    @staticmethod
    def _frac_positive(av: xr.DataArray) -> float:
        """Fraction of grid points where AV > 0 (EERIE adds value)."""
        vals = np.asarray(av).ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return float("nan")
        return float(np.sum(finite > 0) / len(finite))

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
                            f"Added Value of EERIE {ensemble_type} over "
                            f"CMIP6 MMM for {meta['long_name']}"
                        ),
                        "units": "1",
                        "valid_range": np.array([-1.0, 1.0]),
                        "reference": (
                            "Dosio et al. (2015), doi:10.1007/s00382-015-2869-x"
                        ),
                        "model1": f"EERIE ensemble {ensemble_type}",
                        "model2": "CMIP6 multi-model mean",
                        "reference_dataset": "ERA5",
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

        Produces one figure per period (annual, DJF, JJA), each with
        two panels: EERIE ensemble mean AV | EERIE ensemble median AV.
        """
        figures: list[tuple[plt.Figure, dict]] = []
        var_info = vr["var_info"]
        av = vr["av"]

        period_labels = [("annual", "Annual")]
        for s in ("DJF", "JJA"):
            if s in av:
                period_labels.append((s, s))

        for period_key, period_label in period_labels:
            period_data = av.get(period_key)
            if period_data is None:
                continue

            mean_domain_av = period_data["mean_domain_av"]
            median_domain_av = period_data["median_domain_av"]
            mean_frac = period_data["mean_frac_positive"]
            median_frac = period_data["median_frac_positive"]

            panel_label_mean = (
                f"EERIE Mean AV\n"
                f"domain mean={mean_domain_av:+.3f}, "
                f"AV>0: {mean_frac:.0%}"
            )
            panel_label_median = (
                f"EERIE Median AV\n"
                f"domain mean={median_domain_av:+.3f}, "
                f"AV>0: {median_frac:.0%}"
            )

            data_dict = {
                panel_label_mean: period_data["mean"],
                panel_label_median: period_data["median"],
            }

            fig, axes = plot_combined_map(
                data_dict,
                title=(
                    f"{var_info.long_name} {period_label} Added Value"
                    f" — EERIE vs CMIP6 MMM"
                ),
                cmap="RdYlGn",
                vmin=-1.0,
                vmax=1.0,
                units="AV [ ]",
                method=self._regrid_method,
            )

            figure_id = f"{var}_{period_key.lower()}_added_value"
            summary_stats = {
                "EERIE_mean": {
                    "domain_mean_av": mean_domain_av,
                    "frac_positive": mean_frac,
                    "n_eerie_models": vr["n_eerie_models"],
                    "n_cmip6_models": vr["n_cmip6_models"],
                },
                "EERIE_median": {
                    "domain_mean_av": median_domain_av,
                    "frac_positive": median_frac,
                    "n_eerie_models": vr["n_eerie_models"],
                    "n_cmip6_models": vr["n_cmip6_models"],
                },
            }
            meta = self._build_metadata(
                title=(
                    f"{var_info.long_name} {period_label} Added Value "
                    f"(EERIE vs CMIP6 MMM)"
                ),
                figure_id=figure_id,
                models=vr["eerie_models"],
                variables=[var],
                description=(
                    f"Dosio et al. (2015) Added Value metric for "
                    f"{var_info.long_name} ({period_label}), "
                    f"{self.period[0]}-{self.period[1]}. "
                    f"AV > 0: EERIE reduces squared error over CMIP6 MMM. "
                    f"EERIE n={vr['n_eerie_models']}, "
                    f"CMIP6 n={vr['n_cmip6_models']}."
                ),
                plot_type="added_value_map",
                period=self.period,
                summary_statistics=summary_stats,
                extra={
                    "eerie_models": vr["eerie_models"],
                    "cmip6_models": vr["cmip6_models"],
                    "reference": "Dosio et al. (2015)",
                },
            )
            figures.append((fig, meta))

        return figures
