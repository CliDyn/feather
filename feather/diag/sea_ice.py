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
from feather.plot.styles import CMIP6_COLOR, MODEL_COLORS, OBS_COLOR
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

    Produces 4 groups of figures (13 total):
    A) Time series — ice area/extent/volume (NH+SH subplots)
    B) Seasonal cycles — 12-month climatology (NH+SH subplots)
    C) March & September trends — annual max/min months (2×2 subplots)
    D) Spatial maps — polar stereographic maps of siconc/sithick
    """

    name = "sea_ice"
    title = "Sea Ice"
    domain = "o2d"
    variables = ["avg_siconc", "avg_sithick"]
    group = "sea_ice"

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1990", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
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
        cmip6_ts, cmip6_info, cmip6_indiv = (
            self._compute_cmip6_timeseries()
        )

        cmip6_kw = dict(
            cmip6_ts=cmip6_ts or None,
            cmip6_individual_ts=cmip6_indiv or None,
            cmip6_info=cmip6_info or None,
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
            ("siconc_nh_spatial", "avg_siconc", "np"),
            ("siconc_sh_spatial", "avg_siconc", "sp"),
            ("sithick_nh_spatial", "avg_sithick", "np"),
            ("sithick_sh_spatial", "avg_sithick", "sp"),
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

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Abstract interface (thin wrappers for backward compat) ────────

    def compute(self) -> dict[str, Any]:
        """Compute all sea ice results."""
        cmip6_ts, cmip6_info, cmip6_indiv = (
            self._compute_cmip6_timeseries()
        )
        return {
            "model_ts": self._compute_model_timeseries(),
            "obs_ts": self._compute_obs_timeseries(),
            "cmip6_ts": cmip6_ts,
            "cmip6_info": cmip6_info,
            "cmip6_individual_ts": cmip6_indiv,
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
            ("siconc_nh_spatial", "avg_siconc", "np"),
            ("siconc_sh_spatial", "avg_siconc", "sp"),
            ("sithick_nh_spatial", "avg_sithick", "np"),
            ("sithick_sh_spatial", "avg_sithick", "sp"),
        ]:
            figures.extend(self._plot_spatial(fid, var, pole))
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
        from feather.data.loader import DataLoader

        result: dict[str, dict] = {}

        for model in self.config.models:
            logger.info("Computing sea ice metrics for model: %s", model)
            key = DataLoader.make_key(self.experiment, model, "o2d")
            try:
                ds = self.model_loader.load(key)
            except KeyError:
                logger.warning("Model %s not available — skipping", model)
                continue

            if "avg_siconc" not in ds:
                logger.warning(
                    "avg_siconc not in %s dataset — skipping", model,
                )
                continue

            npoints = len(ds["avg_siconc"].isel(time=0))
            mesh = nr.healpix.load_mesh(npoints)

            model_data: dict[str, Any] = {}

            # Concentration metrics (area + extent)
            siconc = ds["avg_siconc"]
            if self.period and "time" in siconc.dims:
                siconc = siconc.sel(time=slice(self.period[0], self.period[1]))

            for hemi in ("nh", "sh"):
                area_fn = getattr(nr, f"ice_area_{hemi}")
                extent_fn = getattr(nr, f"ice_extent_{hemi}")
                model_data[f"area_{hemi}"] = area_fn(
                    siconc, mesh.area, mesh.lat, as_xarray=True,
                ).compute()
                model_data[f"extent_{hemi}"] = extent_fn(
                    siconc, mesh.area, mesh.lat, as_xarray=True,
                ).compute()

            # Volume (requires sithick)
            if "avg_sithick" in ds:
                sithick = ds["avg_sithick"]
                if self.period and "time" in sithick.dims:
                    sithick = sithick.sel(
                        time=slice(self.period[0], self.period[1]),
                    )
                for hemi in ("nh", "sh"):
                    vol_fn = getattr(nr, f"ice_volume_{hemi}")
                    model_data[f"volume_{hemi}"] = vol_fn(
                        sithick, mesh.area, mesh.lat, as_xarray=True,
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
        self,
    ) -> tuple[dict, dict, dict]:
        """Compute CMIP6 ice area/extent/volume per model, then MMM.

        Returns
        -------
        (mmm_ts, cmip6_info, individual_ts) : tuple
            mmm_ts: dict of metric_hemi → DataArray (ensemble mean).
            cmip6_info: dict with n_members, models_used.
            individual_ts: dict of model → {metric_hemi → DataArray}
                (populated only when cmip6_individual is True).
        """
        import nereus as nr

        if not self.cmip6_enabled:
            return {}, {}, {}

        logger.info("Computing CMIP6 sea ice time series")

        all_model_ts: dict[str, dict] = {}
        models_used: list[str] = []

        for model in self.cmip6_loader.models:
            if model in _CMIP6_SEA_ICE_EXCLUDE:
                logger.warning(
                    "  EXCLUDING %s from sea ice computations — known "
                    "unrealistic sea ice results (see _CMIP6_SEA_ICE_EXCLUDE)",
                    model,
                )
                continue

            logger.info("  Loading CMIP6 sea ice for %s", model)

            siconc = self.cmip6_loader.load_var(
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
            area = self.cmip6_loader.load_area(model, table="SImon")
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
            sithick = self.cmip6_loader.load_var(
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
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH time series for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())

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

            # CMIP6 MMM monthly
            if cmip6_ts and key in cmip6_ts:
                ts = cmip6_ts[key]
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values * scale,
                        color=CMIP6_COLOR, alpha=0.3, linewidth=0.7,
                        linestyle="--")

            # Model monthly (background)
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    color = MODEL_COLORS.get(model)
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

            # CMIP6 MMM annual
            if cmip6_ts and key in cmip6_ts:
                ts_annual = annual_mean(cmip6_ts[key])
                time_vals = _to_plot_time(ts_annual.time.values)
                ax.plot(time_vals, ts_annual.values * scale,
                        label="CMIP6 MMM", color=CMIP6_COLOR,
                        linewidth=2.0, linestyle="--")

            # Model annual (foreground)
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    color = MODEL_COLORS.get(model)
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
                        label="Obs", color=OBS_COLOR, linewidth=2.5)

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
        )
        return [(fig, meta)]

    # ── Plotting: Seasonal cycle ──────────────────────────────────────

    def _plot_seasonal_cycle(
        self, metric: str, model_ts: dict, obs_ts: dict,
        cmip6_ts=None, cmip6_individual_ts=None, cmip6_info=None,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH seasonal cycle for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]
        months = np.arange(1, 13)

        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())

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

            # CMIP6 MMM
            if cmip6_ts and key in cmip6_ts:
                clim = cmip6_ts[key].groupby("time.month").mean("time")
                ax.plot(months, clim.values * scale,
                        marker="d", label="CMIP6 MMM", color=CMIP6_COLOR,
                        linewidth=1.5, linestyle="--")

            # Models
            for model, mdata in model_ts.items():
                if key in mdata:
                    clim = mdata[key].groupby("time.month").mean("time")
                    color = MODEL_COLORS.get(model)
                    ax.plot(months, clim.values * scale,
                            marker="o", label=model, color=color)
                    if model not in all_models:
                        all_models.append(model)

            # Obs
            if key in obs_ts:
                clim = obs_ts[key].groupby("time.month").mean("time")
                ax.plot(months, clim.values * scale,
                        marker="s", label="Obs", color=OBS_COLOR,
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
        )
        return [(fig, meta)]

    # ── Plotting: Extreme month trends ────────────────────────────────

    def _plot_extremes(
        self, metric: str, model_ts: dict, obs_ts: dict,
        cmip6_ts=None, cmip6_individual_ts=None, cmip6_info=None,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot 2×2 extreme month trends for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        all_models = []
        cmip6_individual_ts = cmip6_individual_ts or {}

        if cmip6_individual_ts:
            all_models.extend(cmip6_individual_ts.keys())

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

            # CMIP6 MMM
            if cmip6_ts and key in cmip6_ts:
                ts = cmip6_ts[key]
                monthly = ts.where(
                    ts["time.month"] == month, drop=True,
                )
                if len(monthly) > 0:
                    time_vals = _to_plot_time(monthly.time.values)
                    ax.plot(
                        time_vals, monthly.values * scale,
                        label="CMIP6 MMM", color=CMIP6_COLOR,
                        linewidth=1.5, linestyle="--",
                    )

            # Models
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    monthly = ts.where(ts["time.month"] == month, drop=True)
                    if len(monthly) > 0:
                        color = MODEL_COLORS.get(model)
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
                            label="Obs", color=OBS_COLOR, linewidth=2)

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
        )
        return [(fig, meta)]

    # ── Plotting: Spatial maps ────────────────────────────────────────

    def _plot_spatial(
        self, figure_id: str, var: str, pole: str,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot polar spatial maps for March and September."""
        import cartopy.crs as ccrs
        import nereus as nr
        from feather.data.loader import DataLoader

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
                obs_panels.append((mlabel, "Obs", *obs_data))

        # Load model spatial data for each month
        model_panel_data: dict[str, list] = {}
        for model in self.config.models:
            key = DataLoader.make_key(self.experiment, model, "o2d")
            try:
                ds = self.model_loader.load(key)
            except KeyError:
                continue

            if var not in ds:
                continue

            da = ds[var]
            if self.period and "time" in da.dims:
                da = da.sel(time=slice(self.period[0], self.period[1]))
            lon = np.asarray(ds["longitude"])
            lat = np.asarray(ds["latitude"])

            model_months = []
            for month, mlabel in zip(months, month_labels):
                clim = da.sel(time=da.time.dt.month == month).mean("time")
                model_months.append(
                    (mlabel, model, clim.values, lon, lat)
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
                        title=f"Obs {mlabel}", colorbar=False)
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


