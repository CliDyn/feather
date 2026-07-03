"""Sea-surface temperature evaluation against HadISST.

A second SST observational reference alongside ESA-CCI.  HadISST is coarser
(1°) but spans the full 1980–2014 analysis window (ESA-CCI covers only
1990–2014), so it enables a full-period SST evaluation and Added Value.

The diagnostic reuses the :class:`~feather.diag.ocean_sst.OceanSST` figure
machinery (bias maps, global-mean time series, seasonal cycle, zonal-mean
profile) with HadISST as the reference, and — like ``ocean_sst`` — writes the
benchmark/model/ensemble bias NetCDF (via :mod:`feather.diag.ocean_bias`) so
the ocean Added Value diagnostic can use HadISST as a second ``tos`` reference
(``added_value._OBS_NETCDF_SOURCE["HADISST"] == "sst_hadisst"``).
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import xarray as xr

from feather.diag.ocean_sst import OceanSST, _to_celsius
from feather.diag.registry import register
from feather.diag.temperature_berkeley import TemperatureBerkeley
from feather.plot.lines import plot_taylor_diagram
from feather.plot.maps import plot_combined_bias_map
from feather.plot.styles import benchmark_color
from feather.util.spatial import compute_latlon_areas
from feather.util.temporal import (
    climatology,
    linear_trend,
    monthly_climatology,
    seasonal_climatology,
)

logger = logging.getLogger(__name__)

# Reuse the well-tested pattern-statistics/grid helpers from the Berkeley
# temperature diagnostic rather than duplicating them.
_grid_signature = TemperatureBerkeley._grid_signature
_pattern_correlation = TemperatureBerkeley._pattern_correlation
_std_ratio = TemperatureBerkeley._std_ratio


@register
class SSTHadISST(OceanSST):
    """SST evaluation against HadISST (1°, full 1980–2014 record).

    Same figures as :class:`OceanSST` but referenced to HadISST, plus the
    ``tos``-vs-HadISST benchmark-bias NetCDF used by Added Value.
    """

    name = "sst_hadisst"
    title = "Sea Surface Temperature (HadISST)"
    domain = "o2d"
    variables = ["tos"]
    group = "ocean_surface"

    _obs_label = "HadISST"
    _obs_dataset_name = "HadISST"
    _ocean_bias_obs = "HADISST"

    # -- Obs loading overrides (HadISST instead of ESA-CCI) -----------------

    def _load_obs_monthly(self):
        """Full monthly HadISST SST series in °C over the analysis period."""
        da = self.obs_loader.load_hadisst(period=self.period)
        return _to_celsius(da)

    def _load_obs_timemean(self):
        """Annual-mean HadISST SST in °C (period mean of the monthly series)."""
        return self._load_obs_monthly().mean("time")

    def _load_obs_ymonmean(self):
        """Monthly-climatology HadISST SST in °C (``month`` dim 1–12)."""
        return monthly_climatology(self._load_obs_monthly())

    # -- Extra groups: warming-trend maps + Taylor diagram ------------------

    def _extra_figure_ids(self) -> list[str]:
        return [
            "sst_trend_combined", "sst_trend_arctic", "sst_trend_antarctic",
            "sst_taylor",
        ]

    def _extra_groups(self, model_monthly, model_coords, skip_existing):
        """Group E (SST trend maps) + Group F (Taylor diagram) vs HadISST."""
        saved = []
        shared = {
            "model_monthly": model_monthly,
            "model_coords": model_coords,
            "obs": self._load_obs_monthly(),  # HadISST monthly °C
        }

        trend_ids = [
            "sst_trend_combined", "sst_trend_arctic", "sst_trend_antarctic"]
        if not skip_existing or not all(
            self._figure_exists(f) for f in trend_ids
        ):
            try:
                results = self._compute_trends(shared)
                for fig, meta in self._plot_trends(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("SST trend maps failed", exc_info=True)
        else:
            saved.extend([
                (self.output_dir / f"{f}.png", self.output_dir / f"{f}.json")
                for f in trend_ids
            ])

        if not skip_existing or not self._figure_exists("sst_taylor"):
            try:
                results = self._compute_taylor(shared)
                for fig, meta in self._plot_taylor(results):
                    saved.append(self._save(fig, meta, meta["figure_id"]))
            except Exception:
                logger.warning("SST Taylor diagram failed", exc_info=True)
        else:
            saved.append((
                self.output_dir / "sst_taylor.png",
                self.output_dir / "sst_taylor.json",
            ))
        return saved

    # ── Group E: SST warming trends ───────────────────────────────────

    def _compute_trends(self, shared: dict) -> dict[str, Any]:
        """Linear SST trends (°C/decade) on a common grid vs HadISST."""
        logger.info("Computing SST trends (HadISST)...")
        model_coords = shared["model_coords"]
        ir = self.config.nereus.get("influence_radius", 80_000.0)
        resolution = self.config.nereus.get("resolution", 0.25)

        model_trends_native = {
            m: linear_trend(da.compute()) * 10
            for m, da in shared["model_monthly"].items()
        }
        obs_trend_native = linear_trend(shared["obs"].compute()) * 10

        cache: dict = {}
        target_lats = target_lons = None
        model_trends_common: dict[str, xr.DataArray] = {}
        for model, trend in model_trends_native.items():
            lon, lat = model_coords[model]
            if self.config.get_grid_type(model, self.domain) != "healpix":
                lon, lat = np.meshgrid(lon, lat)
            key = _grid_signature(lon, lat)
            if key not in cache:
                regridded, interp = nr.regrid(
                    trend.values.ravel(),
                    lon=np.asarray(lon).ravel(), lat=np.asarray(lat).ravel(),
                    resolution=resolution, influence_radius=ir,
                    lon_bounds=(0.0, 360.0), as_xarray=True,
                )
                cache[key] = interp
                if target_lats is None:
                    target_lats = interp.target_lat[:, 0]
                    target_lons = interp.target_lon[0, :]
            else:
                regridded = cache[key](trend.values.ravel())
            model_trends_common[model] = xr.DataArray(
                regridded, dims=("lat", "lon"),
                coords={"lat": target_lats, "lon": target_lons},
            )

        obs_lons_2d, obs_lats_2d = np.meshgrid(
            obs_trend_native.lon.values, obs_trend_native.lat.values)
        if target_lats is None:
            obs_regridded, interp = nr.regrid(
                obs_trend_native.values.ravel(),
                lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                resolution=resolution, influence_radius=ir,
                lon_bounds=(0.0, 360.0), as_xarray=True,
            )
            target_lats = interp.target_lat[:, 0]
            target_lons = interp.target_lon[0, :]
        else:
            _, interp = nr.regrid(
                obs_trend_native.values.ravel(),
                lon=obs_lons_2d.ravel(), lat=obs_lats_2d.ravel(),
                resolution=resolution, influence_radius=ir,
                lon_bounds=(0.0, 360.0), as_xarray=True,
            )
            obs_regridded = interp(obs_trend_native.values.ravel())
        obs_trend_common = xr.DataArray(
            obs_regridded, dims=("lat", "lon"),
            coords={"lat": target_lats, "lon": target_lons})

        benchmark_trends: dict[str, Any] = {}
        if self.cmip6_enabled:
            for i, bench in enumerate(self.benchmarks):
                label = getattr(bench, "label", "CMIP6 MMM")
                trend = self._compute_cmip6_trends(
                    target_lats, target_lons, resolution, ir, loader=bench)
                if trend is not None:
                    benchmark_trends[label] = trend

        return {
            "model_trends": model_trends_common,
            "obs_trend": obs_trend_common,
            "benchmark_trends": benchmark_trends,
        }

    def _compute_cmip6_trends(self, target_lats, target_lons, resolution,
                              influence_radius, loader=None):
        """Benchmark MMM SST trend on the common grid."""
        from feather.diag.global_biases import GlobalBiases
        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return None
        fields = []
        interp_cache: dict = {}
        for model, variant in loader.get_member_pairs():
            da = loader.load_var_for_model_var(
                "tos", model, variant=variant,
                period=self.period, time_mean=False)
            if da is None:
                continue
            trend = linear_trend(da.compute()) * 10
            fields.append(GlobalBiases._regrid_to_target(
                trend, target_lats, target_lons, resolution,
                influence_radius, interp_cache, method=self._regrid_method))
        if not fields:
            return None
        return sum(fields) / len(fields)

    def _plot_trends(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        model_trends = results["model_trends"]
        obs_trend = results["obs_trend"]
        all_models = list(model_trends.keys())
        bias_dict = {m: t - obs_trend for m, t in model_trends.items()}
        for b_label, b_trend in results.get("benchmark_trends", {}).items():
            bias_dict[b_label] = b_trend - obs_trend
            all_models.append(b_label)

        obs_vals = np.asarray(obs_trend).ravel()
        obs_vals = obs_vals[np.isfinite(obs_vals)]
        obs_vmax = float(np.percentile(np.abs(obs_vals), 98)) or 0.5

        fig, _ = plot_combined_bias_map(
            obs_trend, bias_dict,
            title="Sea Surface Temperature Trends",
            obs_title=f"{self._obs_label} (°C/decade)",
            cmap="RdBu_r", bias_cmap="RdBu_r",
            vmin=-obs_vmax, vmax=obs_vmax, units="°C/decade",
            bias_title_prefix="Trend Diff", land=True,
            method=self._regrid_method,
        )
        meta = self._build_metadata(
            title="Sea Surface Temperature Warming Trends",
            figure_id="sst_trend_combined", models=all_models,
            variables=["tos"],
            description=(
                f"Linear SST trends (°C/decade) over the analysis period. "
                f"Obs panel: {self._obs_label}; difference panels show "
                f"model−obs trend differences."
            ),
            obs_dataset=self._obs_dataset_name, obs_variable="SST",
            plot_type="combined_bias_map", period=self.period,
            computation_notes="Linear OLS per grid point, ×10 for °C/decade",
        )
        figures = [(fig, meta)]

        for proj_str, extent, fig_id, pole_name in [
            ("np", (-180, 180, 50, 90), "sst_trend_arctic", "Arctic (>50°N)"),
            ("sp", (-180, 180, -90, -50), "sst_trend_antarctic",
             "Antarctic (<50°S)"),
        ]:
            try:
                figures.append(self._plot_polar_trend(
                    results, pole_name, proj_str, extent, fig_id, all_models))
            except Exception:
                logger.warning(
                    "Polar SST trend (%s) failed", pole_name, exc_info=True)
        return figures

    def _plot_polar_trend(self, results, pole_name, proj_str, extent,
                          fig_id, all_models):
        import cartopy.crs as ccrs

        panels = {self._obs_label: results["obs_trend"]}
        panels.update(results["model_trends"])
        panels.update(results.get("benchmark_trends", {}))

        n = len(panels)
        ncols = min(n, 3)
        nrows = max(1, (n + ncols - 1) // ncols)
        proj = ccrs.NorthPolarStereo() if proj_str == "np" else ccrs.SouthPolarStereo()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows),
            subplot_kw={"projection": proj})
        axes_flat = [axes] if n == 1 else np.asarray(axes).ravel().tolist()

        vals_all = [np.asarray(v).ravel()[np.isfinite(np.asarray(v).ravel())]
                    for v in panels.values()]
        vmax = (float(np.percentile(np.abs(np.concatenate(vals_all)), 98))
                if vals_all else 0.5) or 0.5

        for i, (label, pdata) in enumerate(panels.items()):
            if i >= len(axes_flat):
                break
            lons_2d, lats_2d = np.meshgrid(pdata.lon.values, pdata.lat.values)
            nr.plot(
                pdata.values.ravel(), lons_2d.ravel(), lats_2d.ravel(),
                ax=axes_flat[i], projection=proj_str, extent=extent,
                land=True, colorbar=False, cmap="RdBu_r",
                vmin=-vmax, vmax=vmax)
            axes_flat[i].set_title(label)
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                     label="°C/decade")
        fig.suptitle(f"Sea Surface Temperature Trends — {pole_name}",
                     fontsize=14, fontweight="bold", y=0.98)
        fig.subplots_adjust(bottom=0.12, top=0.92)

        meta = self._build_metadata(
            title=f"SST Trends ({pole_name})", figure_id=fig_id,
            models=all_models, variables=["tos"],
            description=(
                f"Polar stereographic map of SST linear trends (°C/decade) "
                f"in the {pole_name} region vs {self._obs_label}."),
            obs_dataset=self._obs_dataset_name, obs_variable="SST",
            plot_type="polar_map", spatial_extent=pole_name,
            period=self.period,
            computation_notes="Linear OLS per grid point, ×10 for °C/decade")
        return fig, meta

    # ── Group F: Taylor diagram ───────────────────────────────────────

    def _compute_taylor(self, shared: dict) -> dict[str, Any]:
        logger.info("Computing SST Taylor statistics (HadISST)...")
        obs = shared["obs"]
        model_monthly = shared["model_monthly"]
        model_coords = shared["model_coords"]
        ir = self.config.nereus.get("influence_radius", 80_000.0)

        obs_annual = climatology(obs, self.period).compute()
        obs_seasonal = seasonal_climatology(obs, self.period)
        obs_lats = obs_annual.lat.values
        obs_lons = obs_annual.lon.values
        obs_area = compute_latlon_areas(obs_lats, obs_lons)
        res = abs(float(obs_lats[1] - obs_lats[0]))

        def _regrid(field, lon, lat):
            out, _ = nr.regrid(
                field.values.ravel(),
                lon=np.asarray(lon).ravel(), lat=np.asarray(lat).ravel(),
                resolution=res, influence_radius=ir,
                lon_bounds=(0.0, 360.0), as_xarray=True)
            return xr.DataArray(out, dims=("lat", "lon"),
                                coords={"lat": obs_lats, "lon": obs_lons})

        model_stats: dict[str, dict[str, dict]] = {}
        for model in self.config.models:
            if model not in model_monthly:
                continue
            lon, lat = model_coords[model]
            if self.config.get_grid_type(model, self.domain) != "healpix":
                lon, lat = np.meshgrid(lon, lat)
            m_annual = _regrid(
                climatology(model_monthly[model], self.period).compute(),
                lon, lat)
            fields = {"ANN": (m_annual, obs_annual)}
            m_seas = seasonal_climatology(model_monthly[model], self.period)
            for s in ("DJF", "MAM", "JJA", "SON"):
                if s in m_seas.data_vars and s in obs_seasonal.data_vars:
                    fields[s] = (_regrid(m_seas[s].compute(), lon, lat),
                                 obs_seasonal[s].compute())
            model_stats[model] = {
                s: {"corr": _pattern_correlation(mf, of, obs_area),
                    "std_ratio": _std_ratio(mf, of, obs_area)}
                for s, (mf, of) in fields.items()
            }

        cmip6_stats = None
        if self.cmip6_enabled:
            cmip6_stats = self._compute_cmip6_taylor_stats(
                obs_annual, obs_seasonal, obs_lats, obs_lons, obs_area, res, ir)
        return {"model_stats": model_stats, "cmip6_stats": cmip6_stats}

    def _compute_cmip6_taylor_stats(self, obs_annual, obs_seasonal,
                                    obs_lats, obs_lons, obs_area, res, ir):
        from feather.diag.global_biases import GlobalBiases
        if not self.cmip6_enabled:
            return None
        interp_cache: dict = {}
        annual_fields = []
        seasonal_fields: dict[str, list] = {
            "DJF": [], "MAM": [], "JJA": [], "SON": []}
        for model, variant in self.cmip6_loader.get_member_pairs():
            da = self.cmip6_loader.load_var_for_model_var(
                "tos", model, variant=variant, period=self.period)
            if da is None:
                continue
            annual_fields.append(GlobalBiases._regrid_to_target(
                da, obs_lats, obs_lons, res, ir, interp_cache,
                method=self._regrid_method))
            for s in ("DJF", "MAM", "JJA", "SON"):
                da_s = self.cmip6_loader.load_var_for_model_var(
                    "tos", model, variant=variant,
                    period=self.period, season=s)
                if da_s is not None:
                    seasonal_fields[s].append(GlobalBiases._regrid_to_target(
                        da_s, obs_lats, obs_lons, res, ir, interp_cache,
                        method=self._regrid_method))
        if not annual_fields:
            return None
        mmm = xr.concat(annual_fields, dim="member").mean("member")
        stats = {"ANN": {"corr": _pattern_correlation(mmm, obs_annual, obs_area),
                         "std_ratio": _std_ratio(mmm, obs_annual, obs_area)}}
        for s in ("DJF", "MAM", "JJA", "SON"):
            if seasonal_fields[s] and s in obs_seasonal.data_vars:
                mmm_s = xr.concat(seasonal_fields[s], dim="member").mean("member")
                of = obs_seasonal[s].compute()
                stats[s] = {"corr": _pattern_correlation(mmm_s, of, obs_area),
                            "std_ratio": _std_ratio(mmm_s, of, obs_area)}
        return stats

    def _plot_taylor(self, results: dict) -> list[tuple[plt.Figure, dict]]:
        model_stats = results["model_stats"]
        if not model_stats:
            return []
        model_colors = {m: self.config.get_model_color(m) for m in model_stats}
        fig, _ = plot_taylor_diagram(
            model_stats,
            title=f"Taylor Diagram — SST vs {self._obs_label}",
            obs_label=self._obs_label,
            cmip6_stats=results.get("cmip6_stats"),
            model_colors=model_colors,
        )
        all_models = list(model_stats.keys())
        if results.get("cmip6_stats"):
            all_models.append("CMIP6 MMM")
        summary_stats = {
            f"{m}_{s}": st
            for m, seas in model_stats.items() for s, st in seas.items()
        }
        meta = self._build_metadata(
            title="Taylor Diagram — Sea Surface Temperature",
            figure_id="sst_taylor", models=all_models, variables=["tos"],
            description=(
                f"Taylor diagram comparing SST climatology spatial patterns "
                f"(annual, DJF, JJA) for all models against {self._obs_label}."),
            obs_dataset=self._obs_dataset_name, obs_variable="SST",
            plot_type="taylor_diagram", period=self.period,
            summary_statistics=summary_stats,
        )
        return [(fig, meta)]
