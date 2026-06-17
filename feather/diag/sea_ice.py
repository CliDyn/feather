"""Sea ice diagnostic.

Evaluates DestinE sea ice simulations against OSI-SAF (concentration/extent)
and PIOMAS/GIOMAS (thickness/volume) observations using nereus ice metric
functions.  Produces time series, seasonal cycles, March & September trends,
and polar spatial maps.
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
from feather.plot.styles import CMIP6_COLOR, OBS_COLOR
from feather.util.spatial import compute_latlon_areas
from feather.util.temporal import annual_mean

logger = logging.getLogger(__name__)

# Metrics computed per hemisphere
_METRICS = {
    "area": {"long_name": "Sea Ice Area", "units": "10⁶ km²", "scale": 1e-12},
    "extent": {"long_name": "Sea Ice Extent", "units": "10⁶ km²", "scale": 1e-12},
    "volume": {"long_name": "Sea Ice Volume", "units": "10³ km³", "scale": 1e-12},
}

# CMIP6 models excluded from sea ice computations due to known
# unrealistic results (e.g. implausible hemispheric ice area).
_CMIP6_SEA_ICE_EXCLUDE = {
    "FGOALS-g3",  # NH area ~2×10⁶ km² (too low), SH area ~30×10⁶ km² (too high)
}

# Annual max/min months for each hemisphere
_MINMAX_MONTHS = {
    "nh": {"max": 3, "min": 9, "max_label": "March", "min_label": "September"},
    "sh": {"max": 9, "min": 3, "max_label": "September", "min_label": "March"},
}


@register
class SeaIceDiag(DiagnosticBase):
    """Sea ice diagnostic.

    Produces 6 groups of figures (19 total):
    A) Time series — ice area/extent/volume (NH+SH subplots)
    B) Seasonal cycles — 12-month climatology (NH+SH subplots)
    C) March & September trends — annual max/min months (2×2 subplots)
    D) Spatial maps — polar stereographic maps of siconc/sithick (absolute)
    E) Bias maps — obs climatology + per-model bias (model − obs)
    F) Ensemble summary — obs climatology + ensemble mean/median bias
    """

    name = "sea_ice"
    title = "Sea Ice"
    domain = "o2d"
    variables = ["siconc", "sithick"]
    group = "sea_ice"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, benchmarks=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False, save_netcdf=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader, benchmarks=benchmarks, save_netcdf=save_netcdf)
        self.experiment = experiment
        self.period = period
        self.cmip6_individual = cmip6_individual

    # ── Orchestration (per-figure-group incremental) ──────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute per-figure-group: compute → plot → save."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        # Pre-compute model time series (shared across groups A, B, C)
        model_ts = self._compute_model_timeseries()
        obs_ts = self._compute_obs_timeseries()
        cmip6_ts, cmip6_info, cmip6_indiv, benchmarks = (
            self._compute_benchmark_timeseries()
        )

        cmip6_kw = dict(
            cmip6_ts=cmip6_ts or None,
            cmip6_individual_ts=cmip6_indiv or None,
            cmip6_info=cmip6_info or None,
            benchmarks=benchmarks or None,
        )

        self._maybe_export_netcdf(
            {"model": model_ts, "obs": obs_ts, "benchmarks": benchmarks,
             "cmip6": cmip6_ts, "cmip6_individual": cmip6_indiv},
            "sea_ice_timeseries",
        )
        # Group A: Time series (3 figures)
        for metric in _METRICS:
            fid = f"sea_ice_{metric}_timeseries"
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_timeseries(
                    metric, model_ts, obs_ts, **cmip6_kw,
                )
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group B: Seasonal cycles (3 figures)
        for metric in _METRICS:
            fid = f"sea_ice_{metric}_seasonal_cycle"
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_seasonal_cycle(
                    metric, model_ts, obs_ts, **cmip6_kw,
                )
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group C: Extreme month trends (3 figures)
        for metric in _METRICS:
            fid = f"sea_ice_{metric}_extremes"
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_extremes(
                    metric, model_ts, obs_ts, **cmip6_kw,
                )
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group D: Spatial maps (4 figures)
        spatial_figs = [
            ("siconc_nh_spatial", "siconc", "np"),
            ("siconc_sh_spatial", "siconc", "sp"),
            ("sithick_nh_spatial", "sithick", "np"),
            ("sithick_sh_spatial", "sithick", "sp"),
        ]
        for fid, var, pole in spatial_figs:
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_spatial(fid, var, pole)
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group E: Bias spatial maps (4 figures — obs clim + per-model biases)
        bias_figs = [
            ("siconc_nh_bias", "siconc", "np"),
            ("siconc_sh_bias", "siconc", "sp"),
            ("sithick_nh_bias", "sithick", "np"),
            ("sithick_sh_bias", "sithick", "sp"),
        ]
        for fid, var, pole in bias_figs:
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_bias_spatial(fid, var, pole)
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        # Group F: Ensemble summary (2 figures — obs clim + ens mean/median bias)
        for var in ("siconc", "sithick"):
            fid = f"{var}_ens_summary"
            if skip_existing and self._figure_exists(fid):
                logger.info("Skipping %s — figure exists", fid)
                saved.append((
                    self.output_dir / f"{fid}.png",
                    self.output_dir / f"{fid}.json",
                ))
            else:
                figs = self._plot_ensemble_summary(var)
                for fig, meta in figs:
                    saved.append(self._save(fig, meta, meta["figure_id"]))

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Abstract interface (thin wrappers for backward compat) ────────

    def _compute_benchmark_timeseries(self):
        """Compute primary + per-benchmark sea-ice MMM time series.

        Returns ``(primary_ts, primary_info, primary_indiv, benchmarks)``
        where *benchmarks* is a list of ``{label, color, ts}`` (one MMM per
        configured benchmark loader).  The primary tuple is kept for
        back-compat and for individual-member overlay.
        """
        from feather.plot.styles import benchmark_color

        benchmarks = []
        for i, bench in enumerate(self.benchmarks):
            mmm_ts, info, _ = self._compute_cmip6_timeseries(loader=bench)
            if not mmm_ts:
                continue
            benchmarks.append({
                "label": getattr(bench, "label", "CMIP6 MMM"),
                "color": getattr(bench, "color", None) or benchmark_color(i),
                "ts": mmm_ts,
                "info": info,
            })

        # Primary benchmark individual members (only when requested).
        primary_ts = benchmarks[0]["ts"] if benchmarks else {}
        primary_info = benchmarks[0]["info"] if benchmarks else {}
        primary_indiv = {}
        if self.cmip6_individual and self.benchmarks:
            _, _, primary_indiv = self._compute_cmip6_timeseries(
                loader=self.benchmarks[0],
            )
        return primary_ts, primary_info, primary_indiv, benchmarks

    def compute(self) -> dict[str, Any]:
        """Compute all sea ice results."""
        cmip6_ts, cmip6_info, cmip6_indiv, benchmarks = (
            self._compute_benchmark_timeseries()
        )
        return {
            "model_ts": self._compute_model_timeseries(),
            "obs_ts": self._compute_obs_timeseries(),
            "cmip6_ts": cmip6_ts,
            "cmip6_info": cmip6_info,
            "cmip6_individual_ts": cmip6_indiv,
            "benchmarks": benchmarks,
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figure types from precomputed results."""
        figures = []
        model_ts = results["model_ts"]
        obs_ts = results["obs_ts"]
        cmip6_kw = dict(
            cmip6_ts=results.get("cmip6_ts") or None,
            cmip6_individual_ts=results.get("cmip6_individual_ts") or None,
            cmip6_info=results.get("cmip6_info") or None,
            benchmarks=results.get("benchmarks") or None,
        )
        for metric in _METRICS:
            figures.extend(self._plot_timeseries(
                metric, model_ts, obs_ts, **cmip6_kw,
            ))
            figures.extend(self._plot_seasonal_cycle(
                metric, model_ts, obs_ts, **cmip6_kw,
            ))
            figures.extend(self._plot_extremes(
                metric, model_ts, obs_ts, **cmip6_kw,
            ))
        for fid, var, pole in [
            ("siconc_nh_spatial", "siconc", "np"),
            ("siconc_sh_spatial", "siconc", "sp"),
            ("sithick_nh_spatial", "sithick", "np"),
            ("sithick_sh_spatial", "sithick", "sp"),
        ]:
            figures.extend(self._plot_spatial(fid, var, pole))
        for fid, var, pole in [
            ("siconc_nh_bias", "siconc", "np"),
            ("siconc_sh_bias", "siconc", "sp"),
            ("sithick_nh_bias", "sithick", "np"),
            ("sithick_sh_bias", "sithick", "sp"),
        ]:
            figures.extend(self._plot_bias_spatial(fid, var, pole))
        for var in ("siconc", "sithick"):
            figures.extend(self._plot_ensemble_summary(var))
        return figures

    # ── Computation: Model time series ────────────────────────────────

    def _compute_model_timeseries(self) -> dict[str, dict]:
        """Compute ice area/extent/volume time series for all models.

        Returns
        -------
        dict
            {model_name: {"area_nh": DataArray, "area_sh": DataArray,
                          "extent_nh": ..., "volume_nh": ..., ...}}
        """
        import nereus as nr

        result: dict[str, dict] = {}

        for model in self.config.models:
            logger.info("Computing sea ice metrics for model: %s", model)
            try:
                siconc = self._load_model_var(
                    model, "siconc", period=self.period,
                )
            except (KeyError, FileNotFoundError):
                logger.warning("Model %s siconc not available — skipping", model)
                continue

            siconc = self._validate_sea_ice_units(model, siconc, "siconc")

            # Get coordinates and compute areas based on grid type
            lon, lat = self._load_model_coords(model, "siconc")
            grid_type = self.config.get_grid_type(model, self.domain)

            if grid_type == "healpix":
                npoints = len(siconc.isel(time=0))
                mesh = nr.healpix.load_mesh(npoints)
                area = mesh.area
                lat_1d = mesh.lat
            else:
                # Latlon grid: compute areas and flatten
                lat_2d, _ = np.meshgrid(lat, lon, indexing="ij")
                area = compute_latlon_areas(lat, lon).ravel()
                lat_1d = lat_2d.ravel()
                # Flatten spatial dims to 1D
                spatial_dims = [d for d in siconc.dims if d != "time"]
                if len(spatial_dims) == 2:
                    siconc = siconc.stack(
                        points=tuple(spatial_dims),
                    ).reset_index("points")

            model_data: dict[str, Any] = {}

            # Concentration metrics (area + extent)
            for hemi in ("nh", "sh"):
                area_fn = getattr(nr, f"ice_area_{hemi}")
                extent_fn = getattr(nr, f"ice_extent_{hemi}")
                model_data[f"area_{hemi}"] = area_fn(
                    siconc, area, lat_1d, as_xarray=True,
                ).compute()
                model_data[f"extent_{hemi}"] = extent_fn(
                    siconc, area, lat_1d, as_xarray=True,
                ).compute()

            # Volume (requires sithick)
            try:
                sithick = self._load_model_var(
                    model, "sithick", period=self.period,
                )
            except (KeyError, FileNotFoundError):
                sithick = None

            if sithick is not None:
                sithick = self._validate_sea_ice_units(model, sithick, "sithick")
                if grid_type != "healpix":
                    spatial_dims = [d for d in sithick.dims if d != "time"]
                    if len(spatial_dims) == 2:
                        sithick = sithick.stack(
                            points=tuple(spatial_dims),
                        ).reset_index("points")
                for hemi in ("nh", "sh"):
                    vol_fn = getattr(nr, f"ice_volume_{hemi}")
                    model_data[f"volume_{hemi}"] = vol_fn(
                        sithick, area, lat_1d, as_xarray=True,
                    ).compute()

            result[model] = model_data

        return result

    # ── Computation: Obs time series ──────────────────────────────────

    def _compute_obs_timeseries(self) -> dict[str, Any]:
        """Compute obs ice metrics from OSI-SAF and PIOMAS/GIOMAS.

        Returns
        -------
        dict
            {"area_nh": DataArray, "extent_nh": DataArray, ...,
             "volume_nh": DataArray, "volume_sh": DataArray}
        """
        import nereus as nr

        obs: dict[str, Any] = {}

        # OSI-SAF concentration → area + extent
        for hemi in ("nh", "sh"):
            try:
                ds = self.obs_loader.load_osisaf(hemi, self.period)
                ice_conc = ds["ice_conc"]

                # Stack 2D EASE2 grid to 1D
                spatial_dims = [d for d in ice_conc.dims if d != "time"]
                if len(spatial_dims) == 2:
                    ice_conc = ice_conc.stack(
                        points=tuple(spatial_dims),
                    ).reset_index("points")

                # Convert percentage to fraction
                ice_conc = ice_conc / 100.0

                # Get latitude
                if "lat" in ds:
                    lat = ds["lat"]
                elif "latitude" in ds:
                    lat = ds["latitude"]
                else:
                    # Try extracting from the variable's coordinates
                    lat = ice_conc.coords.get(
                        "lat", ice_conc.coords.get("latitude")
                    )

                if lat is not None:
                    if len(lat.dims) > 1 or (
                        lat.dims and lat.dims[0] != "points"
                    ):
                        lat = lat.stack(
                            points=tuple(
                                d for d in lat.dims if d != "time"
                            ),
                        ).reset_index("points")
                    # Use first timestep if lat has time dim
                    if "time" in lat.dims:
                        lat = lat.isel(time=0)

                # EASE2 cell area: 25km × 25km = 625,000,000 m²
                n_points = ice_conc.shape[-1]
                cell_area = np.full(n_points, 625_000_000.0)

                area_fn = getattr(nr, f"ice_area_{hemi}")
                extent_fn = getattr(nr, f"ice_extent_{hemi}")
                obs[f"area_{hemi}"] = area_fn(
                    ice_conc, cell_area, lat, as_xarray=True,
                )
                obs[f"extent_{hemi}"] = extent_fn(
                    ice_conc, cell_area, lat, as_xarray=True,
                )

            except (KeyError, FileNotFoundError, AttributeError) as e:
                logger.warning(
                    "OSI-SAF %s loading failed: %s", hemi.upper(), e,
                )

        # PIOMAS (NH volume) and GIOMAS (SH volume)
        psc_map = {"piomas": "nh", "giomas": "sh"}
        for product, hemi in psc_map.items():
            try:
                ds = self.obs_loader.load_psc(product, self.period)
                sithick = ds["sithick"]
                areacello = ds["areacello"]
                lat = ds["latitude"]

                # Stack curvilinear grid to 1D
                spatial_dims = [d for d in sithick.dims if d != "time"]
                if len(spatial_dims) == 2:
                    sithick = sithick.stack(
                        points=tuple(spatial_dims),
                    ).reset_index("points")
                    areacello = areacello.stack(
                        points=tuple(
                            d for d in areacello.dims if d != "time"
                        ),
                    ).reset_index("points")
                    lat = lat.stack(
                        points=tuple(d for d in lat.dims if d != "time"),
                    ).reset_index("points")

                vol_fn = getattr(nr, f"ice_volume_{hemi}")
                obs[f"volume_{hemi}"] = vol_fn(
                    sithick, areacello, lat, as_xarray=True,
                )

            except (KeyError, FileNotFoundError, AttributeError) as e:
                logger.warning(
                    "%s loading failed: %s", product.upper(), e,
                )

        return obs

    # ── Computation: CMIP6 time series ────────────────────────────────

    def _compute_cmip6_timeseries(
        self, loader=None,
    ) -> tuple[dict, dict, dict]:
        """Compute benchmark ice area/extent/volume per model, then MMM.

        *loader* defaults to the primary benchmark; pass another benchmark
        loader (e.g. HighResMIP) to compute its MMM.

        Returns
        -------
        (mmm_ts, cmip6_info, individual_ts) : tuple
            mmm_ts: dict of metric_hemi → DataArray (ensemble mean).
            cmip6_info: dict with n_members, models_used.
            individual_ts: dict of model → {metric_hemi → DataArray}
                (populated only when cmip6_individual is True).
        """
        import nereus as nr

        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return {}, {}, {}

        logger.info("Computing %s sea ice time series",
                    getattr(loader, "label", "CMIP6"))

        all_model_ts: dict[str, dict] = {}
        models_used: list[str] = []

        for model in loader.models:
            if model in _CMIP6_SEA_ICE_EXCLUDE:
                logger.warning(
                    "  EXCLUDING %s from sea ice computations — known "
                    "unrealistic sea ice results (see _CMIP6_SEA_ICE_EXCLUDE)",
                    model,
                )
                continue

            logger.info("  Loading CMIP6 sea ice for %s", model)

            siconc = loader.load_var(
                "siconc", model, table="SImon", time_mean=False,
                period=self.period,
            )
            if siconc is None:
                logger.info(
                    "    siconc not available for %s — skipping", model,
                )
                continue

            # Find lat/lon coordinates
            lat_arr = lon_arr = None
            for lname, loname in [
                ("lat", "lon"), ("latitude", "longitude"),
            ]:
                if lname in siconc.coords and loname in siconc.coords:
                    lat_arr = np.asarray(siconc.coords[lname])
                    lon_arr = np.asarray(siconc.coords[loname])
                    break
            if lat_arr is None:
                logger.warning(
                    "    Cannot find lat/lon for %s — skipping", model,
                )
                continue

            # Flatten spatial dims to 1D for nereus ice functions
            ntime = siconc.sizes["time"]
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                lat_2d, _ = np.meshgrid(lat_arr, lon_arr, indexing="ij")
            else:
                lat_2d = lat_arr
            lat_flat = lat_2d.ravel()
            npoints = len(lat_flat)

            # Sanitize siconc.  After _normalise_siconc() in load_var,
            # valid values should be 0-1 fraction.  However:
            #   - Non-NaN fill values (e.g. 1e20) become ~1e18 after
            #     /100 normalisation.  These must be ZEROED, not clipped
            #     to 1.0 (which would create fake 100% ice at land).
            #   - Some datasets may still be in 0-100% (if the loader
            #     didn't trigger normalisation).  Re-normalise if needed.
            siconc_vals = np.nan_to_num(
                siconc.values.reshape(ntime, npoints), nan=0.0,
            )
            # Detect still-in-percentage: valid max in 1-100 range
            valid_mask = siconc_vals < 1e10  # ignore obvious fill values
            valid_max = float(siconc_vals[valid_mask].max()) if valid_mask.any() else 0
            if valid_max > 1.0:
                logger.info(
                    "    siconc for %s appears to be in %% (max valid=%.1f)"
                    " — dividing by 100", model, valid_max,
                )
                siconc_vals = siconc_vals / 100.0
            # Zero fill values (anything > 1 after normalisation)
            siconc_vals = np.where(siconc_vals > 1.0, 0.0, siconc_vals)
            siconc_vals = np.where(siconc_vals < 0.0, 0.0, siconc_vals)

            siconc_1d = xr.DataArray(
                siconc_vals,
                dims=("time", "points"),
                coords={"time": siconc.time},
            )

            # Load or compute cell areas
            # Max physical cell area: ~1.2e10 m² for a 1° cell at equator.
            # Use 1e12 m² (1M km²) as generous upper bound — anything
            # above is a fill value (e.g. 9.97e36 for float32 netCDF fill).
            _MAX_CELL_AREA = 1e12  # m²
            area = loader.load_area(model, table="SImon")
            if area is not None:
                area_flat = np.nan_to_num(
                    np.asarray(area).ravel(), nan=0.0,
                )
                area_flat = np.where(
                    area_flat > _MAX_CELL_AREA, 0.0, area_flat,
                )
                area_flat = np.clip(area_flat, 0.0, _MAX_CELL_AREA)
                if len(area_flat) != npoints:
                    logger.warning(
                        "    areacello shape mismatch for %s — computing "
                        "from grid", model,
                    )
                    area = None
            if area is None:
                if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                    area_flat = compute_latlon_areas(
                        lat_arr, lon_arr,
                    ).ravel()
                else:
                    logger.warning(
                        "    Cannot compute areas for %s — skipping", model,
                    )
                    continue

            area_da = xr.DataArray(area_flat, dims="points")
            lat_da = xr.DataArray(lat_flat, dims="points")

            model_data: dict[str, Any] = {}

            # Concentration metrics (area + extent)
            for hemi in ("nh", "sh"):
                area_fn = getattr(nr, f"ice_area_{hemi}")
                extent_fn = getattr(nr, f"ice_extent_{hemi}")
                model_data[f"area_{hemi}"] = area_fn(
                    siconc_1d, area_da, lat_da, as_xarray=True,
                ).compute()
                model_data[f"extent_{hemi}"] = extent_fn(
                    siconc_1d, area_da, lat_da, as_xarray=True,
                ).compute()

            # Volume (requires sithick)
            sithick = loader.load_var(
                "sithick", model, table="SImon", time_mean=False,
                period=self.period,
            )
            if sithick is not None:
                sithick_nt = sithick.sizes["time"]
                # Sanitize thickness: NaN → 0, zero fill values.
                # Fill values (e.g. 1e20) must be zeroed, not clipped to
                # 100m — that would create absurd ice volume.  Physical
                # max thickness is ~20m for multi-year ridged ice.
                sithick_vals = np.nan_to_num(
                    sithick.values.reshape(sithick_nt, npoints), nan=0.0,
                )
                sithick_vals = np.where(
                    sithick_vals > 100.0, 0.0, sithick_vals,
                )
                sithick_vals = np.clip(sithick_vals, 0.0, 100.0)

                sithick_1d = xr.DataArray(
                    sithick_vals,
                    dims=("time", "points"),
                    coords={"time": sithick.time},
                )
                for hemi in ("nh", "sh"):
                    vol_fn = getattr(nr, f"ice_volume_{hemi}")
                    model_data[f"volume_{hemi}"] = vol_fn(
                        sithick_1d, area_da, lat_da, as_xarray=True,
                    ).compute()

            # Per-model diagnostics: log metrics in plot units so
            # outliers are easy to spot.
            for mkey, mval in model_data.items():
                metric_name = mkey.rsplit("_", 1)[0]  # area, extent, volume
                hemi_name = mkey.rsplit("_", 1)[1].upper()
                scale = _METRICS.get(metric_name, {}).get("scale", 1e-12)
                units = _METRICS.get(metric_name, {}).get("units", "?")
                scaled_max = float(mval.max()) * scale
                scaled_mean = float(mval.mean()) * scale
                logger.info(
                    "    %s %s %s: mean=%.2f max=%.2f %s",
                    model, hemi_name, metric_name,
                    scaled_mean, scaled_max, units,
                )

            # Post-computation validation: skip models with unreasonable
            # metrics.  Even after sanitising inputs, some CMIP6 models may
            # produce absurd values due to grid/metadata issues.
            # Physical maxima: NH/SH ice extent ~20e6 km² = 2e13 m²,
            # ice volume ~30e3 km³ ≈ 3e13 m³.  Use 1e14 as generous cap.
            _MAX_METRIC = 1e14
            bad_metric = False
            for mkey, mval in model_data.items():
                if float(mval.max()) > _MAX_METRIC:
                    logger.warning(
                        "    %s has unreasonable %s (max=%.2e) — "
                        "excluding from MMM",
                        model, mkey, float(mval.max()),
                    )
                    bad_metric = True
                    break
            if bad_metric:
                continue

            all_model_ts[model] = model_data
            models_used.append(model)

        if not models_used:
            logger.info("    No CMIP6 models available for sea ice")
            return {}, {}, {}

        # Compute MMM by aligning to common time axis
        mmm_ts: dict[str, xr.DataArray] = {}
        for key in ("area_nh", "area_sh", "extent_nh", "extent_sh",
                     "volume_nh", "volume_sh"):
            series = [
                all_model_ts[m][key]
                for m in models_used
                if key in all_model_ts[m]
            ]
            if series:
                aligned = xr.align(*series, join="inner")
                mmm_ts[key] = sum(aligned) / len(aligned)

        cmip6_info = {
            "n_members": len(models_used),
            "models_used": models_used,
        }
        logger.info(
            "    CMIP6 sea ice MMM: %d models, keys: %s",
            len(models_used), list(mmm_ts.keys()),
        )

        individual_ts = all_model_ts if self.cmip6_individual else {}
        return mmm_ts, cmip6_info, individual_ts

    # ── Plotting: Time series ─────────────────────────────────────────

    def _plot_timeseries(
        self, metric: str, model_ts: dict, obs_ts: dict,
        cmip6_ts=None, cmip6_individual_ts=None, cmip6_info=None,
        benchmarks=None,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH time series for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        obs_label = "PIOMAS" if metric == "volume" else "OSI-SAF"
        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}
        benchmarks = benchmarks or []
        # Back-compat: a lone cmip6_ts (no benchmarks list) → one MMM.
        if not benchmarks and cmip6_ts:
            benchmarks = [{"label": "CMIP6 MMM", "color": CMIP6_COLOR,
                           "ts": cmip6_ts}]

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())
        all_models.extend(b["label"] for b in benchmarks)

        for ax, hemi, hemi_label in [
            (ax_nh, "nh", "Northern Hemisphere"),
            (ax_sh, "sh", "Southern Hemisphere"),
        ]:
            key = f"{metric}_{hemi}"

            # --- Monthly pass (background, semi-transparent) ---

            # CMIP6 individual monthly
            for _mname, mdata in cmip6_individual_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    time_vals = _to_plot_time(ts.time.values)
                    ax.plot(time_vals, ts.values * scale,
                            color=CMIP6_COLOR, alpha=0.2, linewidth=0.5)

            # Benchmark MMM monthly (CMIP6, HighResMIP, …)
            for bench in benchmarks:
                if key in bench["ts"]:
                    ts = bench["ts"][key]
                    time_vals = _to_plot_time(ts.time.values)
                    ax.plot(time_vals, ts.values * scale,
                            color=bench["color"], alpha=0.3, linewidth=0.7,
                            linestyle="--")

            # Model monthly (background)
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    color = self.config.get_model_color(model)
                    time_vals = _to_plot_time(ts.time.values)
                    ax.plot(time_vals, ts.values * scale,
                            color=color, alpha=0.3, linewidth=0.7)

            # Obs monthly (background)
            if key in obs_ts:
                ts = obs_ts[key]
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values * scale,
                        color=OBS_COLOR, alpha=0.3, linewidth=0.7)

            # --- Annual pass (foreground, thick with labels) ---

            # CMIP6 individual annual
            for i, (_mname, mdata) in enumerate(
                cmip6_individual_ts.items()
            ):
                if key in mdata:
                    label = "CMIP6 members" if i == 0 else "_nolegend_"
                    ts_annual = annual_mean(mdata[key])
                    time_vals = _to_plot_time(ts_annual.time.values)
                    ax.plot(time_vals, ts_annual.values * scale,
                            color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                            label=label)

            # Benchmark MMM annual (CMIP6, HighResMIP, …)
            for bench in benchmarks:
                if key in bench["ts"]:
                    ts_annual = annual_mean(bench["ts"][key])
                    time_vals = _to_plot_time(ts_annual.time.values)
                    # Legend label only on the first axis to avoid duplicates
                    lbl = bench["label"] if hemi == "nh" else "_nolegend_"
                    ax.plot(time_vals, ts_annual.values * scale,
                            label=lbl, color=bench["color"],
                            linewidth=2.0, linestyle="--")

            # Model annual (foreground)
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    color = self.config.get_model_color(model)
                    annual = annual_mean(ts)
                    time_vals = _to_plot_time(annual.time.values)
                    ax.plot(time_vals, annual.values * scale,
                            label=model, color=color, linewidth=2.0)
                    if model not in all_models:
                        all_models.append(model)

            # Obs annual (foreground)
            if key in obs_ts:
                ts = obs_ts[key]
                annual = annual_mean(ts)
                time_vals = _to_plot_time(annual.time.values)
                ax.plot(time_vals, annual.values * scale,
                        label=obs_label, color=OBS_COLOR, linewidth=2.5)

            ax.set_title(f"{hemi_label}")
            ax.set_ylabel(f"{info['long_name']} ({info['units']})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{info['long_name']} — Time Series", fontsize=13)
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{info['long_name']} Time Series",
            figure_id=f"sea_ice_{metric}_timeseries",
            models=all_models,
            description=(
                f"Monthly and annual-mean {info['long_name'].lower()} "
                f"for NH (left) and SH (right)."
            ),
            plot_type="timeseries",
            period=self.period,
            obs_dataset="OSI_SAF" if metric != "volume" else "PSC",
            cmip6_info=cmip6_info,
            benchmark_info=self._benchmark_meta_from_list(benchmarks) or None,
        )
        return [(fig, meta)]

    # ── Plotting: Seasonal cycle ──────────────────────────────────────

    def _plot_seasonal_cycle(
        self, metric: str, model_ts: dict, obs_ts: dict,
        cmip6_ts=None, cmip6_individual_ts=None, cmip6_info=None,
        benchmarks=None,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH seasonal cycle for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        obs_label = "PIOMAS" if metric == "volume" else "OSI-SAF"
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]
        months = np.arange(1, 13)

        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}
        benchmarks = benchmarks or []
        # Back-compat: a lone cmip6_ts (no benchmarks list) → one MMM.
        if not benchmarks and cmip6_ts:
            benchmarks = [{"label": "CMIP6 MMM", "color": CMIP6_COLOR,
                           "ts": cmip6_ts}]

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())
        all_models.extend(b["label"] for b in benchmarks)

        for ax, hemi, hemi_label in [
            (ax_nh, "nh", "Northern Hemisphere"),
            (ax_sh, "sh", "Southern Hemisphere"),
        ]:
            key = f"{metric}_{hemi}"

            # CMIP6 individual
            for i, (_mname, mdata) in enumerate(
                cmip6_individual_ts.items()
            ):
                if key in mdata:
                    clim = mdata[key].groupby("time.month").mean("time")
                    label = "CMIP6 members" if i == 0 else "_nolegend_"
                    ax.plot(months, clim.values * scale,
                            color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                            label=label)

            # Benchmark MMMs (CMIP6, HighResMIP, …)
            for bench in benchmarks:
                if key in bench["ts"]:
                    clim = bench["ts"][key].groupby("time.month").mean("time")
                    lbl = bench["label"] if hemi == "nh" else "_nolegend_"
                    ax.plot(months, clim.values * scale,
                            marker="d", label=lbl, color=bench["color"],
                            linewidth=1.5, linestyle="--")

            # Models
            for model, mdata in model_ts.items():
                if key in mdata:
                    clim = mdata[key].groupby("time.month").mean("time")
                    color = self.config.get_model_color(model)
                    ax.plot(months, clim.values * scale,
                            marker="o", label=model, color=color)
                    if model not in all_models:
                        all_models.append(model)

            # Obs
            if key in obs_ts:
                clim = obs_ts[key].groupby("time.month").mean("time")
                ax.plot(months, clim.values * scale,
                        marker="s", label=obs_label, color=OBS_COLOR,
                        linewidth=2)

            ax.set_xticks(months)
            ax.set_xticklabels(month_labels)
            ax.set_title(f"{hemi_label}")
            ax.set_ylabel(f"{info['long_name']} ({info['units']})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{info['long_name']} — Seasonal Cycle", fontsize=13)
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{info['long_name']} Seasonal Cycle",
            figure_id=f"sea_ice_{metric}_seasonal_cycle",
            models=all_models,
            description=(
                f"Monthly climatological cycle of {info['long_name'].lower()} "
                f"for NH (left) and SH (right)."
            ),
            plot_type="seasonal_cycle",
            period=self.period,
            obs_dataset="OSI_SAF" if metric != "volume" else "PSC",
            cmip6_info=cmip6_info,
            benchmark_info=self._benchmark_meta_from_list(benchmarks) or None,
        )
        return [(fig, meta)]

    # ── Plotting: Extreme month trends ────────────────────────────────

    def _plot_extremes(
        self, metric: str, model_ts: dict, obs_ts: dict,
        cmip6_ts=None, cmip6_individual_ts=None, cmip6_info=None,
        benchmarks=None,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot 2×2 extreme month trends for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        obs_label = "PIOMAS" if metric == "volume" else "OSI-SAF"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}
        benchmarks = benchmarks or []
        # Back-compat: a lone cmip6_ts (no benchmarks list) → one MMM.
        if not benchmarks and cmip6_ts:
            benchmarks = [{"label": "CMIP6 MMM", "color": CMIP6_COLOR,
                           "ts": cmip6_ts}]

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())
        all_models.extend(b["label"] for b in benchmarks)

        panels = [
            (axes[0, 0], "nh", "max", _MINMAX_MONTHS["nh"]),
            (axes[0, 1], "nh", "min", _MINMAX_MONTHS["nh"]),
            (axes[1, 0], "sh", "max", _MINMAX_MONTHS["sh"]),
            (axes[1, 1], "sh", "min", _MINMAX_MONTHS["sh"]),
        ]

        for ax, hemi, extreme, months_info in panels:
            key = f"{metric}_{hemi}"
            month = months_info[extreme]
            month_label = months_info[f"{extreme}_label"]
            hemi_label = "NH" if hemi == "nh" else "SH"

            # CMIP6 individual
            for i, (_mname, mdata) in enumerate(
                cmip6_individual_ts.items()
            ):
                if key in mdata:
                    ts = mdata[key]
                    monthly = ts.where(
                        ts["time.month"] == month, drop=True,
                    )
                    if len(monthly) > 0:
                        label = (
                            "CMIP6 members" if i == 0 else "_nolegend_"
                        )
                        time_vals = _to_plot_time(monthly.time.values)
                        ax.plot(
                            time_vals, monthly.values * scale,
                            color=CMIP6_COLOR, alpha=0.35, linewidth=0.8,
                            label=label,
                        )

            # Benchmark MMMs (CMIP6, HighResMIP, …)
            for bench in benchmarks:
                if key not in bench["ts"]:
                    continue
                ts = bench["ts"][key]
                monthly = ts.where(ts["time.month"] == month, drop=True)
                if len(monthly) > 0:
                    time_vals = _to_plot_time(monthly.time.values)
                    # Label once (first panel) to avoid legend duplicates
                    lbl = (bench["label"]
                           if (hemi == "nh" and extreme == "max")
                           else "_nolegend_")
                    ax.plot(
                        time_vals, monthly.values * scale,
                        label=lbl, color=bench["color"],
                        linewidth=1.5, linestyle="--",
                    )

            # Models
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    monthly = ts.where(ts["time.month"] == month, drop=True)
                    if len(monthly) > 0:
                        color = self.config.get_model_color(model)
                        time_vals = _to_plot_time(monthly.time.values)
                        ax.plot(time_vals, monthly.values * scale,
                                label=model, color=color, linewidth=1.5)
                        if model not in all_models:
                            all_models.append(model)

            # Obs
            if key in obs_ts:
                ts = obs_ts[key]
                monthly = ts.where(ts["time.month"] == month, drop=True)
                if len(monthly) > 0:
                    time_vals = _to_plot_time(monthly.time.values)
                    ax.plot(time_vals, monthly.values * scale,
                            label=obs_label, color=OBS_COLOR, linewidth=2)

            ax.set_title(f"{hemi_label} {month_label}")
            ax.set_ylabel(f"{info['long_name']} ({info['units']})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{info['long_name']} — March & September Trends", fontsize=13,
        )
        plt.tight_layout()

        meta = self._build_metadata(
            title=f"{info['long_name']} March & September Trends",
            figure_id=f"sea_ice_{metric}_extremes",
            models=all_models,
            description=(
                f"Annual {info['long_name'].lower()} for March and September: "
                f"NH March (annual max) / September (annual min), "
                f"SH September (annual max) / March (annual min)."
            ),
            plot_type="monthly_trends",
            period=self.period,
            obs_dataset="OSI_SAF" if metric != "volume" else "PSC",
            cmip6_info=cmip6_info,
            benchmark_info=self._benchmark_meta_from_list(benchmarks) or None,
        )
        return [(fig, meta)]

    # ── Plotting: Spatial maps ────────────────────────────────────────

    def _plot_spatial(
        self, figure_id: str, var: str, pole: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot polar spatial maps for March and September."""
        import cartopy.crs as ccrs
        import nereus as nr

        if pole == "np":
            proj = ccrs.NorthPolarStereo()
            extent = (-180, 180, 50, 90)
            months = [3, 9]
            month_labels = ["March", "September"]
            hemi_label = "Arctic"
        else:
            proj = ccrs.SouthPolarStereo()
            extent = (-180, 180, -90, -50)
            months = [9, 3]
            month_labels = ["September", "March"]
            hemi_label = "Antarctic"

        is_conc = "siconc" in var
        obs_label = "OSI-SAF" if is_conc else "PIOMAS"
        if is_conc:
            cmap = "Blues_r"
            vmin, vmax = 0, 1
            var_label = "Sea Ice Concentration"
        else:
            import cmocean
            cmap = cmocean.cm.tempo
            vmin, vmax = 0, 4
            var_label = "Sea Ice Thickness (m)"
        resolution = self.config.nereus.get("resolution", 0.25)

        # Collect panels: (month_label, model/obs label, data, lon, lat)
        panels = []
        models_used = []
        nrows = len(months)

        # Load obs spatial data for each month
        obs_panels = []
        for month, mlabel in zip(months, month_labels):
            obs_data = self._load_obs_spatial(var, pole, month)
            if obs_data is not None:
                obs_panels.append((mlabel, obs_label, *obs_data))

        # Load model spatial data for each month
        model_panel_data: dict[str, list] = {}
        for model in self.config.models:
            try:
                da = self._load_model_var(model, var, period=self.period)
            except (KeyError, FileNotFoundError):
                continue

            if is_conc:
                da = self._validate_sea_ice_units(model, da, var)

            lon, lat = self._load_model_coords(model, var)
            grid_type = self.config.get_grid_type(model, self.domain)

            # For latlon grids, meshgrid and ravel for nr.plot()
            if grid_type != "healpix":
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                lon = lon_2d.ravel()
                lat = lat_2d.ravel()

            model_months = []
            for month, mlabel in zip(months, month_labels):
                clim = da.sel(time=da.time.dt.month == month).mean("time")
                clim_vals = clim.values
                if grid_type != "healpix":
                    clim_vals = clim_vals.ravel()
                model_months.append(
                    (mlabel, model, clim_vals, lon, lat)
                )
            model_panel_data[model] = model_months
            models_used.append(model)

        if not models_used and not obs_panels:
            logger.warning("No data available for %s — skipping", figure_id)
            return []

        # Layout: nrows (months) × ncols (obs + models)
        ncols = (1 if obs_panels else 0) + len(models_used)
        if ncols == 0:
            return []

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 5 * nrows),
            subplot_kw={"projection": proj},
        )
        # Ensure axes is 2D
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        col = 0
        # Plot obs columns
        if obs_panels:
            for row, (mlabel, label, data, lon, lat) in enumerate(obs_panels):
                ax = axes[row, col]
                nr.plot(data, lon, lat, ax=ax, projection=pole,
                        extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                        land=True, resolution=resolution,
                        title=f"{label} {mlabel}", colorbar=False)
            col += 1

        # Plot model columns
        for model in models_used:
            for row, (mlabel, mname, data, lon, lat) in enumerate(
                model_panel_data[model]
            ):
                ax = axes[row, col]
                nr.plot(data, lon, lat, ax=ax, projection=pole,
                        extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                        land=True, resolution=resolution,
                        title=f"{model} {mlabel}", colorbar=False)
            col += 1

        # Shared horizontal colorbar at bottom
        fig.suptitle(f"{hemi_label} {var_label}", fontsize=13, y=0.98)
        fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.12)
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                     label=var_label)

        meta = self._build_metadata(
            title=f"{hemi_label} {var_label}",
            figure_id=figure_id,
            models=models_used,
            variables=[var],
            description=(
                f"Polar stereographic maps of {var_label.lower()} for "
                f"{hemi_label} in March and September."
            ),
            plot_type="spatial_map",
            period=self.period,
            spatial_extent="NH" if pole == "np" else "SH",
            obs_dataset="OSI_SAF" if is_conc else "PSC",
        )
        return [(fig, meta)]

    # ── Unit validation ──────────────────────────────────────────────────

    def _validate_sea_ice_units(
        self, model: str, da: xr.DataArray, var: str,
    ) -> xr.DataArray:
        """Log statistics and auto-correct unit issues in siconc/sithick.

        Called on raw model output to catch inconsistencies between runs
        (e.g. run2 siconc in fraction vs run3 in percentage).

        siconc expected in fraction [0, 1].
        sithick expected in metres [0, ~20].
        """
        try:
            sample_vals = da.isel(time=slice(0, 12)).values.ravel()
        except Exception:
            return da
        finite = sample_vals[np.isfinite(sample_vals)]
        if len(finite) == 0:
            logger.warning("Unit check %s %s: no finite values found", model, var)
            return da

        vmin = float(finite.min())
        vmean = float(finite.mean())
        vmax = float(finite.max())
        logger.info(
            "Unit check %-20s %-10s  min=%8.4g  mean=%8.4g  max=%8.4g",
            model, var, vmin, vmean, vmax,
        )

        if var == "siconc":
            if vmax > 1.5:
                logger.warning(
                    "  %s siconc max=%.2f — values appear to be in %% (0–100),"
                    " dividing by 100 for unit consistency",
                    model, vmax,
                )
                return da / 100.0
            if vmax > 1.01:
                logger.warning(
                    "  %s siconc max=%.4f slightly > 1 — clipping to [0, 1]",
                    model, vmax,
                )
                return da.clip(0.0, 1.0)
        elif var == "sithick":
            if vmax > 50.0:
                logger.warning(
                    "  %s sithick max=%.2f m — fill values suspected,"
                    " masking values > 50 m",
                    model, vmax,
                )
                return da.where(da < 50.0)
        return da

    # ── Helpers for polar bias regridding ────────────────────────────────

    def _build_polar_bias_data(
        self,
        var: str,
        pole: str,
        months: list[int],
    ) -> tuple[
        dict[int, np.ndarray],   # obs_common  {month → 1-D polar array}
        dict[str, dict[int, np.ndarray]],  # model_biases {model → {month → 1-D}}
        np.ndarray,              # polar lon (1-D)
        np.ndarray,              # polar lat (1-D)
        list[str],               # models_used
    ]:
        """Regrid obs and models to a common grid; return polar-subset biases.

        Uses nr.regrid() to build a common regular lat/lon grid, then restricts
        to the polar cap (lat ≥ 50° for NH, lat ≤ -50° for SH) for efficiency.
        Model interpolators are cached per source grid size.
        """
        import nereus as nr

        resolution = float(self.config.nereus.get("resolution", 0.25))
        influence_radius = float(
            self.config.nereus.get("ocean_influence_radius", 200_000.0)
        )
        lat_thresh = 50.0 if pole == "np" else -50.0

        # --- Regrid obs to common grid ---
        obs_interp = None
        polar_mask: np.ndarray | None = None
        common_lon_polar: np.ndarray | None = None
        common_lat_polar: np.ndarray | None = None
        obs_common: dict[int, np.ndarray] = {}

        for month in months:
            raw = self._load_obs_spatial(var, pole, month)
            if raw is None:
                continue
            obs_vals, obs_lon, obs_lat = raw
            obs_lon = np.where(obs_lon < 0, obs_lon + 360.0, obs_lon)

            if obs_interp is None:
                _, obs_interp = nr.regrid(
                    obs_vals.ravel(),
                    lon=obs_lon.ravel(),
                    lat=obs_lat.ravel(),
                    resolution=resolution,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                tgt_lat_1d = obs_interp.target_lat[:, 0]
                tgt_lon_1d = obs_interp.target_lon[0, :]
                lon_2d_g, lat_2d_g = np.meshgrid(tgt_lon_1d, tgt_lat_1d)
                lon_flat = lon_2d_g.ravel()
                lat_flat = lat_2d_g.ravel()
                if pole == "np":
                    polar_mask = lat_flat >= lat_thresh
                else:
                    polar_mask = lat_flat <= lat_thresh
                common_lon_polar = lon_flat[polar_mask]
                common_lat_polar = lat_flat[polar_mask]

            regridded = np.asarray(obs_interp(obs_vals.ravel())).ravel()
            obs_common[month] = regridded[polar_mask]

        if not obs_common or polar_mask is None:
            return {}, {}, np.array([]), np.array([]), []

        # --- Regrid models to same common grid and compute biases ---
        _model_interp_cache: dict[int, Any] = {}
        model_biases: dict[str, dict[int, np.ndarray]] = {}
        models_used: list[str] = []

        for model in self.config.models:
            try:
                da = self._load_model_var(model, var, period=self.period)
            except (KeyError, FileNotFoundError):
                continue
            da = self._validate_sea_ice_units(model, da, var)

            lon, lat = self._load_model_coords(model, var)
            grid_type = self.config.get_grid_type(model, self.domain)
            if grid_type != "healpix":
                lm2, la2 = np.meshgrid(lon, lat)
                src_lon = lm2.ravel()
                src_lat = la2.ravel()
            else:
                src_lon = np.asarray(lon)
                src_lat = np.asarray(lat)
            src_lon = np.where(src_lon < 0, src_lon + 360.0, src_lon)
            n_src = len(src_lon)

            biases: dict[int, np.ndarray] = {}
            for month in months:
                if month not in obs_common:
                    continue
                clim = da.sel(time=da.time.dt.month == month).mean("time")
                clim_vals = clim.values.ravel()

                if n_src not in _model_interp_cache:
                    _, m_interp = nr.regrid(
                        clim_vals,
                        lon=src_lon,
                        lat=src_lat,
                        resolution=resolution,
                        influence_radius=influence_radius,
                        lon_bounds=(0.0, 360.0),
                        as_xarray=True,
                    )
                    _model_interp_cache[n_src] = m_interp
                else:
                    m_interp = _model_interp_cache[n_src]

                model_common = np.asarray(m_interp(clim_vals)).ravel()[polar_mask]
                biases[month] = model_common - obs_common[month]

            model_biases[model] = biases
            models_used.append(model)

        return (
            obs_common, model_biases,
            common_lon_polar, common_lat_polar,  # type: ignore[return-value]
            models_used,
        )

    # ── Plotting: Bias spatial maps (Group E) ────────────────────────────

    def _plot_bias_spatial(
        self, figure_id: str, var: str, pole: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot polar bias maps: obs climatology column + per-model bias columns."""
        import cartopy.crs as ccrs

        is_conc = var == "siconc"
        if pole == "np":
            proj = ccrs.NorthPolarStereo()
            extent = (-180, 180, 50, 90)
            months = [3, 9]
            month_labels = ["March", "September"]
            hemi_label = "Arctic"
        else:
            proj = ccrs.SouthPolarStereo()
            extent = (-180, 180, -90, -50)
            months = [9, 3]
            month_labels = ["September", "March"]
            hemi_label = "Antarctic"

        if is_conc:
            obs_cmap, bias_cmap = "Blues_r", "RdBu_r"
            vmin_obs, vmax_obs, vmax_bias = 0.0, 1.0, 0.3
            var_label, obs_units = "Sea Ice Concentration", "fraction"
        else:
            import cmocean
            obs_cmap = cmocean.cm.tempo
            bias_cmap = "RdBu_r"
            vmin_obs, vmax_obs, vmax_bias = 0.0, 4.0, 1.5
            var_label, obs_units = "Sea Ice Thickness", "m"

        resolution = float(self.config.nereus.get("resolution", 0.25))

        obs_common, model_biases, plon, plat, models_used = (
            self._build_polar_bias_data(var, pole, months)
        )
        if not obs_common or not models_used:
            logger.warning("No data for %s — skipping", figure_id)
            return []

        nrows = len(months)
        ncols = 1 + len(models_used)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            subplot_kw={"projection": proj},
        )
        if nrows == 1:
            axes = axes[np.newaxis, :]
        if ncols == 1:
            axes = axes[:, np.newaxis]

        import nereus as nr

        for row, (month, mlabel) in enumerate(zip(months, month_labels)):
            ax = axes[row, 0]
            if month in obs_common:
                nr.plot(
                    obs_common[month], plon, plat,
                    ax=ax, projection=pole, extent=extent,
                    cmap=obs_cmap, vmin=vmin_obs, vmax=vmax_obs,
                    land=True, resolution=resolution,
                    title=f"Obs {mlabel}", colorbar=False,
                )
            for col, model in enumerate(models_used, start=1):
                ax = axes[row, col]
                bias = model_biases.get(model, {}).get(month)
                if bias is not None:
                    nr.plot(
                        bias, plon, plat,
                        ax=ax, projection=pole, extent=extent,
                        cmap=bias_cmap, vmin=-vmax_bias, vmax=vmax_bias,
                        land=True, resolution=resolution,
                        title=f"{model} − Obs {mlabel}", colorbar=False,
                    )

        fig.suptitle(
            f"{hemi_label} {var_label} — Climatology & Bias",
            fontsize=13, y=0.98,
        )
        fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.12)

        cbar_obs_ax = fig.add_axes([0.05, 0.04, 0.18, 0.025])
        sm_obs = plt.cm.ScalarMappable(
            cmap=obs_cmap, norm=plt.Normalize(vmin=vmin_obs, vmax=vmax_obs),
        )
        sm_obs.set_array([])
        fig.colorbar(sm_obs, cax=cbar_obs_ax, orientation="horizontal",
                     label=f"Climatology ({obs_units})")

        cbar_bias_ax = fig.add_axes([0.30, 0.04, 0.55, 0.025])
        sm_bias = plt.cm.ScalarMappable(
            cmap=bias_cmap, norm=plt.Normalize(vmin=-vmax_bias, vmax=vmax_bias),
        )
        sm_bias.set_array([])
        fig.colorbar(sm_bias, cax=cbar_bias_ax, orientation="horizontal",
                     label=f"Bias ({obs_units})")

        meta = self._build_metadata(
            title=f"{hemi_label} {var_label} Bias",
            figure_id=figure_id,
            models=models_used,
            variables=[var],
            description=(
                f"Polar stereographic bias maps ({hemi_label}) of "
                f"{var_label.lower()} for March and September. "
                f"Column 1: obs climatology. Remaining columns: model − obs."
            ),
            plot_type="bias_map",
            period=self.period,
            spatial_extent="NH" if pole == "np" else "SH",
            obs_dataset="OSI_SAF" if is_conc else "PSC",
        )
        return [(fig, meta)]

    # ── Plotting: Ensemble summary (Group F) ─────────────────────────────

    def _plot_ensemble_summary(
        self, var: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Ensemble summary: obs climatology + mean and median biases, NH and SH.

        Layout: 4 rows (Arctic March, Arctic September, Antarctic September,
        Antarctic March) × 3 columns (obs clim | ens mean bias | ens median bias).
        """
        import cartopy.crs as ccrs
        from matplotlib.gridspec import GridSpec

        is_conc = var == "siconc"
        if is_conc:
            obs_cmap, bias_cmap = "Blues_r", "RdBu_r"
            vmin_obs, vmax_obs, vmax_bias = 0.0, 1.0, 0.3
            var_label, obs_units = "Sea Ice Concentration", "fraction"
        else:
            import cmocean
            obs_cmap = cmocean.cm.tempo
            bias_cmap = "RdBu_r"
            vmin_obs, vmax_obs, vmax_bias = 0.0, 4.0, 1.5
            var_label, obs_units = "Sea Ice Thickness", "m"

        resolution = float(self.config.nereus.get("resolution", 0.25))

        # hemisphere_configs: (pole, proj, extent, months, month_labels, label)
        hemi_configs = [
            (
                "np", ccrs.NorthPolarStereo(), (-180, 180, 50, 90),
                [3, 9], ["March", "September"], "Arctic",
            ),
            (
                "sp", ccrs.SouthPolarStereo(), (-180, 180, -90, -50),
                [9, 3], ["September", "March"], "Antarctic",
            ),
        ]

        nrows = 4  # 2 hemispheres × 2 months
        ncols = 3  # obs clim | ens mean bias | ens median bias

        import nereus as nr

        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.15)

        all_models_used: list[str] = []
        global_row = 0

        for pole, proj, extent, months, month_labels, hemi_label in hemi_configs:
            obs_common, model_biases, plon, plat, models_used = (
                self._build_polar_bias_data(var, pole, months)
            )
            for m in models_used:
                if m not in all_models_used:
                    all_models_used.append(m)

            for month_idx, (month, mlabel) in enumerate(zip(months, month_labels)):
                row = global_row + month_idx
                ax_obs = fig.add_subplot(gs[row, 0], projection=proj)
                ax_mean = fig.add_subplot(gs[row, 1], projection=proj)
                ax_med = fig.add_subplot(gs[row, 2], projection=proj)

                if month in obs_common:
                    nr.plot(
                        obs_common[month], plon, plat,
                        ax=ax_obs, projection=pole, extent=extent,
                        cmap=obs_cmap, vmin=vmin_obs, vmax=vmax_obs,
                        land=True, resolution=resolution,
                        title=f"{hemi_label} {mlabel} — Obs",
                        colorbar=False,
                    )

                per_month_biases = [
                    model_biases[m][month]
                    for m in models_used
                    if month in model_biases.get(m, {})
                ]
                if per_month_biases:
                    bias_stack = np.stack(per_month_biases, axis=0)
                    if bias_stack.size == 0:
                        continue
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        ens_mean = np.nanmean(bias_stack, axis=0)
                        ens_median = np.nanmedian(bias_stack, axis=0)

                    nr.plot(
                        ens_mean, plon, plat,
                        ax=ax_mean, projection=pole, extent=extent,
                        cmap=bias_cmap, vmin=-vmax_bias, vmax=vmax_bias,
                        land=True, resolution=resolution,
                        title=f"{hemi_label} {mlabel} — Ens Mean Bias",
                        colorbar=False,
                    )
                    nr.plot(
                        ens_median, plon, plat,
                        ax=ax_med, projection=pole, extent=extent,
                        cmap=bias_cmap, vmin=-vmax_bias, vmax=vmax_bias,
                        land=True, resolution=resolution,
                        title=f"{hemi_label} {mlabel} — Ens Median Bias",
                        colorbar=False,
                    )

            global_row += len(months)

        fig.suptitle(f"{var_label} — Ensemble Summary", fontsize=14, y=0.99)
        fig.subplots_adjust(bottom=0.07)

        cbar_obs_ax = fig.add_axes([0.05, 0.03, 0.22, 0.015])
        sm_obs = plt.cm.ScalarMappable(
            cmap=obs_cmap, norm=plt.Normalize(vmin=vmin_obs, vmax=vmax_obs),
        )
        sm_obs.set_array([])
        fig.colorbar(sm_obs, cax=cbar_obs_ax, orientation="horizontal",
                     label=f"Climatology ({obs_units})")

        cbar_bias_ax = fig.add_axes([0.38, 0.03, 0.52, 0.015])
        sm_bias = plt.cm.ScalarMappable(
            cmap=bias_cmap, norm=plt.Normalize(vmin=-vmax_bias, vmax=vmax_bias),
        )
        sm_bias.set_array([])
        fig.colorbar(sm_bias, cax=cbar_bias_ax, orientation="horizontal",
                     label=f"Bias ({obs_units})")

        figure_id = f"{var}_ens_summary"
        meta = self._build_metadata(
            title=f"{var_label} Ensemble Summary",
            figure_id=figure_id,
            models=all_models_used,
            variables=[var],
            description=(
                f"Summary of {var_label.lower()} for Arctic and Antarctic "
                f"(March and September). Column 1: obs climatology. "
                f"Column 2: ensemble mean bias (model − obs). "
                f"Column 3: ensemble median bias."
            ),
            plot_type="bias_map",
            period=self.period,
            obs_dataset="OSI_SAF" if is_conc else "PSC",
        )
        return [(fig, meta)]

    def _load_obs_spatial(
        self, var: str, pole: str, month: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Load obs spatial data for a single month climatology.

        Returns (data_1d, lon_1d, lat_1d) or None.
        """
        try:
            if "siconc" in var:
                hemi = "nh" if pole == "np" else "sh"
                ds = self.obs_loader.load_osisaf(hemi, self.period)
                da = ds["ice_conc"] / 100.0  # percentage → fraction
            else:
                product = "piomas" if pole == "np" else "giomas"
                ds = self.obs_loader.load_psc(product, self.period)
                da = ds["sithick"]

            # Select month and mean over time
            clim = da.sel(time=da.time.dt.month == month).mean("time")

            # Get lon/lat
            if "lon" in ds:
                lon = ds["lon"]
            elif "longitude" in ds:
                lon = ds["longitude"]
            else:
                lon = clim.coords.get("lon", clim.coords.get("longitude"))

            if "lat" in ds:
                lat = ds["lat"]
            elif "latitude" in ds:
                lat = ds["latitude"]
            else:
                lat = clim.coords.get("lat", clim.coords.get("latitude"))

            if lon is None or lat is None:
                return None

            return (
                np.asarray(clim).ravel(),
                np.asarray(lon).ravel(),
                np.asarray(lat).ravel(),
            )
        except (KeyError, FileNotFoundError, AttributeError) as e:
            logger.warning("Obs spatial data for %s not available: %s", var, e)
            return None


# ── Helpers ───────────────────────────────────────────────────────────


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert time values to matplotlib-compatible format."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values


