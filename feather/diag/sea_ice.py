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
from feather.plot.styles import MODEL_COLORS, OBS_COLOR
from feather.util.temporal import annual_mean

logger = logging.getLogger(__name__)

# Metrics computed per hemisphere
_METRICS = {
    "area": {"long_name": "Sea Ice Area", "units": "10⁶ km²", "scale": 1e-12},
    "extent": {"long_name": "Sea Ice Extent", "units": "10⁶ km²", "scale": 1e-12},
    "volume": {"long_name": "Sea Ice Volume", "units": "10³ km³", "scale": 1e-12},
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
                figs = self._plot_timeseries(metric, model_ts, obs_ts)
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
                figs = self._plot_seasonal_cycle(metric, model_ts, obs_ts)
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
                figs = self._plot_extremes(metric, model_ts, obs_ts)
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
        return {
            "model_ts": self._compute_model_timeseries(),
            "obs_ts": self._compute_obs_timeseries(),
        }

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Plot all figure types from precomputed results."""
        figures = []
        model_ts = results["model_ts"]
        obs_ts = results["obs_ts"]
        for metric in _METRICS:
            figures.extend(self._plot_timeseries(metric, model_ts, obs_ts))
            figures.extend(self._plot_seasonal_cycle(metric, model_ts, obs_ts))
            figures.extend(self._plot_extremes(metric, model_ts, obs_ts))
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

    # ── Plotting: Time series ─────────────────────────────────────────

    def _plot_timeseries(
        self, metric: str, model_ts: dict, obs_ts: dict,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH time series for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []

        for ax, hemi, hemi_label in [
            (ax_nh, "nh", "Northern Hemisphere"),
            (ax_sh, "sh", "Southern Hemisphere"),
        ]:
            key = f"{metric}_{hemi}"

            # Obs monthly (background)
            if key in obs_ts:
                ts = obs_ts[key]
                time_vals = _to_plot_time(ts.time.values)
                ax.plot(time_vals, ts.values * scale,
                        color=OBS_COLOR, alpha=0.3, linewidth=0.7)

            # Model monthly (background)
            for model, mdata in model_ts.items():
                if key in mdata:
                    ts = mdata[key]
                    color = MODEL_COLORS.get(model)
                    time_vals = _to_plot_time(ts.time.values)
                    ax.plot(time_vals, ts.values * scale,
                            color=color, alpha=0.3, linewidth=0.7)

            # Obs annual (foreground)
            if key in obs_ts:
                ts = obs_ts[key]
                annual = annual_mean(ts)
                time_vals = _to_plot_time(annual.time.values)
                ax.plot(time_vals, annual.values * scale,
                        label="Obs", color=OBS_COLOR, linewidth=2.5)

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
        )
        return [(fig, meta)]

    # ── Plotting: Seasonal cycle ──────────────────────────────────────

    def _plot_seasonal_cycle(
        self, metric: str, model_ts: dict, obs_ts: dict,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot NH/SH seasonal cycle for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]
        month_labels = ["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"]
        months = np.arange(1, 13)

        fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(14, 5))
        all_models = []

        for ax, hemi, hemi_label in [
            (ax_nh, "nh", "Northern Hemisphere"),
            (ax_sh, "sh", "Southern Hemisphere"),
        ]:
            key = f"{metric}_{hemi}"

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
        )
        return [(fig, meta)]

    # ── Plotting: Extreme month trends ────────────────────────────────

    def _plot_extremes(
        self, metric: str, model_ts: dict, obs_ts: dict,
    ) -> list[tuple[plt.Figure, dict]]:
        """Plot 2×2 extreme month trends for a given metric."""
        info = _METRICS[metric]
        scale = info["scale"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        all_models = []

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


