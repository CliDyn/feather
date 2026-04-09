"""Precipitation observational dataset comparison: ERA5 vs MSWEP v2.8.

Produces 13 figures across 5 groups:
A (×3): Trend maps (ERA5 + MSWEP, both periods) — annual, DJF, JJA
B (×3): Trend difference maps (period diffs + dataset diffs) — annual, DJF, JJA
C (×1): Global mean annual time series
D (×3): 5-panel climatological maps — annual, DJF, JJA
E (×3): Relative bias maps (ERA5 − MSWEP) / MSWEP × 100 % — annual, DJF, JJA

Periods compared:
- Short: 1980–2014  (model-comparable period)
- Long:  1980–2023  (MSWEP near-real-time coverage)

Common grid: ERA5 native 0.25° (MSWEP 0.1° regridded to ERA5 coordinates).
Units:       mm/day for all displayed quantities (stored as kg/m²/s internally).
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.diag.base import DiagnosticBase
from feather.diag.registry import register
from feather.plot.maps import _flatten_latlon, plot_combined_map
from feather.plot.styles import OBS_COLOR
from feather.util.spatial import compute_latlon_areas, latlon_global_mean
from feather.util.temporal import (
    annual_mean,
    climatology,
    linear_trend,
    seasonal_annual_mean,
)

logger = logging.getLogger(__name__)

# kg/m²/s → mm/day display conversion
_PR_MMDAY = 86400.0

# Relative bias masking threshold: below 0.1 mm/day → mask to avoid
# division artefacts in arid regions.
_REL_BIAS_THRESHOLD = 0.1 / _PR_MMDAY  # kg/m²/s

# MSWEP line colour (sea-green, contrasts with OBS_COLOR which is teal/blue)
_MSWEP_COLOR = "#2e8b57"


@register
class PrecipObsComparisonDiag(DiagnosticBase):
    """ERA5 vs MSWEP precipitation comparison.

    Obs-only diagnostic: no model data is loaded or compared.
    Evaluates total precipitation for two periods to assess dataset
    disagreement and sensitivity to record length.

    Figures
    -------
    A — Trend maps: ERA5 and MSWEP for both periods (3 figures)
    B — Trend differences: period extension + dataset disagreement (3)
    C — Global mean annual time series (1)
    D — 5-panel climatological maps (ERA5 ref, MSWEP ref, diffs) (3)
    E — Relative bias (ERA5 − MSWEP) / MSWEP × 100 % (3)
    """

    name = "precip_obs_comparison"
    title = "Precipitation Dataset Comparison (ERA5 vs MSWEP 0.1°)"
    domain = "sfc"
    variables = ["pr"]
    group = "precipitation"

    PERIOD_SHORT = ("1980", "2014")
    PERIOD_LONG = ("1980", "2023")

    def __init__(self, model_loader, obs_loader, config, *,
                 cmip6_loader=None, variables=None,
                 experiment="baseline_hist", period=("1980", "2014"),
                 cmip6_individual=False):
        super().__init__(model_loader, obs_loader, config,
                         cmip6_loader=cmip6_loader)
        self.experiment = experiment
        # period arg kept for API compatibility; PERIOD_SHORT/LONG are used

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute: compute → plot → save all figures."""
        logger.info("Running diagnostic: %s", self.name)
        saved: list[tuple[Path, Path]] = []

        all_ids = ["pr_obs_timeseries"]
        for pk in ["annual", "djf", "jja"]:
            all_ids += [
                f"pr_obs_{pk}_trends",
                f"pr_obs_{pk}_trend_diffs",
                f"pr_obs_{pk}_clim",
                f"pr_obs_{pk}_relative_bias",
            ]

        if skip_existing and all(self._figure_exists(fid) for fid in all_ids):
            logger.info("Skipping %s — all figures exist", self.name)
            return [
                (self.output_dir / f"{fid}.png",
                 self.output_dir / f"{fid}.json")
                for fid in all_ids
            ]

        try:
            results = self.compute()
            figures = self.plot(results)
            for fig, meta in figures:
                paths = self._save(fig, meta, meta["figure_id"])
                saved.append(paths)
                plt.close(fig)
        except Exception:
            logger.error("Diagnostic %s failed", self.name, exc_info=True)

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Computation ────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Load ERA5 and MSWEP, compute trends, climatologies, time series.

        All intermediate fields are in kg/m²/s.  Conversion to mm/day
        happens in the plot methods.

        Returns
        -------
        dict with keys:
            trends       — per period-key trend fields and diffs
            clim         — per period-key clim means, biases, relative biases
            timeseries   — global-mean annual series (kg/m²/s)
            target_lats  — ERA5 latitude array (common grid)
            target_lons  — ERA5 longitude array (0..360)
            area         — area weights on common grid
        """
        logger.info("Loading ERA5 precipitation for both periods...")
        era5_s_raw = self._load_era5_pr(self.PERIOD_SHORT)
        era5_l_raw = self._load_era5_pr(self.PERIOD_LONG)

        # Normalise ERA5 lons to 0..360, rename dims → lat/lon
        era5_s = self._normalise_lons(era5_s_raw)
        era5_l = self._normalise_lons(era5_l_raw)

        target_lats = era5_s.lat.values
        target_lons = era5_s.lon.values

        logger.info("Loading MSWEP precipitation for both periods...")
        mswep_s_raw = self._load_mswep_pr(self.PERIOD_SHORT)
        mswep_l_raw = self._load_mswep_pr(self.PERIOD_LONG)

        logger.info("Regridding MSWEP to ERA5 grid...")
        mswep_s = self._interp_to_era5(mswep_s_raw, target_lats, target_lons)
        mswep_l = self._interp_to_era5(mswep_l_raw, target_lats, target_lons)

        area = compute_latlon_areas(target_lats, target_lons)

        trends = self._compute_all_trends(era5_s, era5_l, mswep_s, mswep_l)
        clim = self._compute_clim(era5_s, era5_l, mswep_s, mswep_l)
        ts = self._compute_timeseries(era5_l, mswep_l, area)

        return {
            "trends": trends,
            "clim": clim,
            "timeseries": ts,
            "target_lats": target_lats,
            "target_lons": target_lons,
            "area": area,
        }

    def _compute_all_trends(
        self,
        era5_s: xr.DataArray,
        era5_l: xr.DataArray,
        mswep_s: xr.DataArray,
        mswep_l: xr.DataArray,
    ) -> dict[str, dict]:
        """Compute annual/DJF/JJA trends (kg/m²/s/decade) and diffs.

        Returns
        -------
        dict keyed by "annual", "DJF", "JJA".  Each value contains:
            era5_short, era5_long, mswep_short, mswep_long — trend maps
            era5_period_diff  — ERA5 trend change when extending to 2023
            mswep_period_diff — MSWEP trend change, same extension
            dataset_diff_short — ERA5 − MSWEP trend for 1980–2014
            dataset_diff_long  — ERA5 − MSWEP trend for 1980–2023
        """
        results: dict[str, dict] = {}

        for pk in ["annual", "DJF", "JJA"]:
            era5_s_t = self._compute_trend(era5_s, pk)
            era5_l_t = self._compute_trend(era5_l, pk)
            mswep_s_t = self._compute_trend(mswep_s, pk)
            mswep_l_t = self._compute_trend(mswep_l, pk)

            if any(t is None for t in [era5_s_t, era5_l_t, mswep_s_t, mswep_l_t]):
                logger.warning(
                    "Insufficient data for %s trends — skipping", pk,
                )
                continue

            results[pk] = {
                "era5_short": era5_s_t,
                "era5_long": era5_l_t,
                "mswep_short": mswep_s_t,
                "mswep_long": mswep_l_t,
                "era5_period_diff": era5_l_t - era5_s_t,
                "mswep_period_diff": mswep_l_t - mswep_s_t,
                "dataset_diff_short": era5_s_t - mswep_s_t,
                "dataset_diff_long": era5_l_t - mswep_l_t,
            }

        return results

    def _compute_clim(
        self,
        era5_s: xr.DataArray,
        era5_l: xr.DataArray,
        mswep_s: xr.DataArray,
        mswep_l: xr.DataArray,
    ) -> dict[str, dict]:
        """Compute climatological means, absolute biases, and relative biases.

        Returns
        -------
        dict keyed by "annual", "DJF", "JJA".  Each value contains:
            era5_short, era5_long, mswep_short, mswep_long — time means (kg/m²/s)
            diff_short      — ERA5 − MSWEP for 1980–2014 (kg/m²/s)
            diff_long       — ERA5 − MSWEP for 1980–2023 (kg/m²/s)
            rel_bias_short  — (ERA5 − MSWEP) / MSWEP × 100 % for short period
            rel_bias_long   — same for long period
        """
        results: dict[str, dict] = {}

        for pk in ["annual", "DJF", "JJA"]:
            era5_s_c = self._seasonal_mean(era5_s, pk)
            era5_l_c = self._seasonal_mean(era5_l, pk)
            mswep_s_c = self._seasonal_mean(mswep_s, pk)
            mswep_l_c = self._seasonal_mean(mswep_l, pk)

            diff_short = era5_s_c - mswep_s_c
            diff_long = era5_l_c - mswep_l_c

            # Relative bias: masked where MSWEP < 0.1 mm/day threshold
            rel_bias_short = xr.where(
                mswep_s_c > _REL_BIAS_THRESHOLD,
                (era5_s_c - mswep_s_c) / mswep_s_c * 100.0,
                np.nan,
            )
            rel_bias_long = xr.where(
                mswep_l_c > _REL_BIAS_THRESHOLD,
                (era5_l_c - mswep_l_c) / mswep_l_c * 100.0,
                np.nan,
            )

            results[pk] = {
                "era5_short": era5_s_c,
                "era5_long": era5_l_c,
                "mswep_short": mswep_s_c,
                "mswep_long": mswep_l_c,
                "diff_short": diff_short,
                "diff_long": diff_long,
                "rel_bias_short": rel_bias_short,
                "rel_bias_long": rel_bias_long,
            }

        return results

    def _compute_timeseries(
        self,
        era5_l: xr.DataArray,
        mswep_l: xr.DataArray,
        area: xr.DataArray,
    ) -> dict[str, xr.DataArray]:
        """Global mean annual time series for both datasets (full period)."""
        era5_annual = annual_mean(era5_l)
        if hasattr(era5_annual, "compute"):
            era5_annual = era5_annual.compute()
        mswep_annual = annual_mean(mswep_l)
        if hasattr(mswep_annual, "compute"):
            mswep_annual = mswep_annual.compute()
        return {
            "era5": latlon_global_mean(era5_annual, area=area),
            "mswep": latlon_global_mean(mswep_annual, area=area),
        }

    # ── Static computation helpers ──────────────────────────────────────

    @staticmethod
    def _compute_trend(
        da: xr.DataArray, period_key: str,
    ) -> xr.DataArray | None:
        """Linear trend (kg/m²/s/decade) for annual or a specific season."""
        if period_key == "annual":
            grouped = annual_mean(da)
            if hasattr(grouped, "compute"):
                grouped = grouped.compute()
            n = grouped.sizes.get("year", grouped.sizes.get("time", 0))
            if n < 2:
                return None
            return linear_trend(grouped) * 10

        grouped = seasonal_annual_mean(da, period_key)
        if hasattr(grouped, "compute"):
            grouped = grouped.compute()
        n = grouped.sizes.get("year", grouped.sizes.get("time", 0))
        if n < 2:
            return None
        return linear_trend(grouped, dim="year") * 10

    @staticmethod
    def _seasonal_mean(da: xr.DataArray, period_key: str) -> xr.DataArray:
        """Time mean for annual or a specific season (DJF/JJA)."""
        if period_key == "annual":
            result = climatology(da)
        else:
            result = da.where(da.time.dt.season == period_key).mean("time")
        if hasattr(result, "compute"):
            result = result.compute()
        return result

    # ── Data loading helpers ────────────────────────────────────────────

    def _load_era5_pr(self, period: tuple[str, str]) -> xr.DataArray:
        """Load ERA5 total precipitation (kg/m²/s) via VARIABLE_REGISTRY."""
        return self._load_obs_var("pr", period)

    def _load_mswep_pr(self, period: tuple[str, str]) -> xr.DataArray:
        """Load MSWEP v2.8 precipitation (kg/m²/s) via obs_loader."""
        return self.obs_loader.load_mswep(period)

    @staticmethod
    def _normalise_lons(da: xr.DataArray) -> xr.DataArray:
        """Rename latitude/longitude → lat/lon and shift lons to 0..360."""
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)

        if float(da.lon.min()) < 0:
            da = da.assign_coords(
                lon=((da.lon + 360) % 360),
            ).sortby("lon")
        return da

    @staticmethod
    def _interp_to_era5(
        da: xr.DataArray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
    ) -> xr.DataArray:
        """Interpolate MSWEP to ERA5 grid with longitude wraparound padding.

        Padding avoids NaN at the 0°/360° seam which would appear as a
        white stripe at the centre of Robinson projection maps.
        """
        lat_name = "lat" if "lat" in da.dims else "latitude"
        lon_name = "lon" if "lon" in da.dims else "longitude"

        da_lon_vals = da[lon_name].values
        if float(da_lon_vals.max()) > 180:  # 0..360 convention
            n_pad = 10
            left_pad = da.isel({lon_name: slice(-n_pad, None)}).assign_coords(
                {lon_name: da_lon_vals[-n_pad:] - 360.0}
            )
            right_pad = da.isel({lon_name: slice(None, n_pad)}).assign_coords(
                {lon_name: da_lon_vals[:n_pad] + 360.0}
            )
            da = xr.concat([left_pad, da, right_pad], dim=lon_name)

        result = da.interp(
            {lat_name: target_lats, lon_name: target_lons},
            method="linear",
        )

        rename = {}
        if lat_name != "lat":
            rename[lat_name] = "lat"
        if lon_name != "lon":
            rename[lon_name] = "lon"
        if rename:
            result = result.rename(rename)

        return result

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate all 13 figures."""
        figures = []
        figures.extend(self._plot_trend_maps(results))
        figures.extend(self._plot_trend_diffs(results))
        figures.extend(self._plot_clim(results))
        figures.extend(self._plot_relative_bias(results))
        figures.extend(self._plot_timeseries(results))
        return figures

    def _plot_trend_maps(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group A: 4-panel trend maps per seasonal period (3 figures).

        Colormap: cmo.tarn (brown=drying, white=no trend, blue=wetting).
        """
        out = []
        trends = results["trends"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for pk, pdata in trends.items():
            pk_lower = pk.lower()

            data_dict = {
                f"ERA5 {short_lbl}": pdata["era5_short"] * _PR_MMDAY,
                f"ERA5 {long_lbl}": pdata["era5_long"] * _PR_MMDAY,
                f"MSWEP {short_lbl}": pdata["mswep_short"] * _PR_MMDAY,
                f"MSWEP {long_lbl}": pdata["mswep_long"] * _PR_MMDAY,
            }

            all_vals = _finite_concat(data_dict.values())
            vmax = float(np.percentile(np.abs(all_vals), 98)) or 1.0

            fig, _ = plot_combined_map(
                data_dict,
                title=f"pr {pk} Warming Trend (mm/day/decade)",
                cmap="cmo.tarn",
                vmin=-vmax, vmax=vmax,
                units="mm/day/decade",
                max_cols=2,
            )

            meta = self._build_metadata(
                title=f"pr {pk} Trend Maps — ERA5 vs MSWEP",
                figure_id=f"pr_obs_{pk_lower}_trends",
                models=[],
                variables=["pr"],
                description=(
                    f"{pk} precipitation trends (mm/day/decade) from ERA5 and "
                    f"MSWEP for {short_lbl} and {long_lbl}."
                ),
                plot_type="combined_trend_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, MSWEP",
            )
            out.append((fig, meta))

        return out

    def _plot_trend_diffs(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group B: 4-panel trend difference maps (3 figures).

        Colormap: BrBG (brown=ERA5 drier, green=ERA5 wetter than MSWEP).
        """
        out = []
        trends = results["trends"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for pk, pdata in trends.items():
            pk_lower = pk.lower()

            data_dict = {
                f"ERA5 period diff\n({self.PERIOD_LONG[1]}−{self.PERIOD_SHORT[1]})": (
                    pdata["era5_period_diff"] * _PR_MMDAY
                ),
                f"MSWEP period diff\n({self.PERIOD_LONG[1]}−{self.PERIOD_SHORT[1]})": (
                    pdata["mswep_period_diff"] * _PR_MMDAY
                ),
                f"ERA5 − MSWEP\n({short_lbl})": pdata["dataset_diff_short"] * _PR_MMDAY,
                f"ERA5 − MSWEP\n({long_lbl})": pdata["dataset_diff_long"] * _PR_MMDAY,
            }

            all_vals = _finite_concat(data_dict.values())
            vmax = float(np.percentile(np.abs(all_vals), 98)) or 1.0

            fig, _ = plot_combined_map(
                data_dict,
                title=f"pr {pk} Trend Differences (mm/day/decade)",
                cmap="BrBG",
                vmin=-vmax, vmax=vmax,
                units="mm/day/decade",
                max_cols=2,
            )

            meta = self._build_metadata(
                title=f"pr {pk} Trend Differences — Period and Dataset",
                figure_id=f"pr_obs_{pk_lower}_trend_diffs",
                models=[],
                variables=["pr"],
                description=(
                    f"{pk} trend differences (mm/day/decade): period extension "
                    f"({short_lbl} → {long_lbl}) and ERA5 − MSWEP disagreement."
                ),
                plot_type="combined_trend_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, MSWEP",
            )
            out.append((fig, meta))

        return out

    def _plot_clim(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group D: 5-panel climatological maps (3 figures).

        Layout (row 0): ERA5 ref | MSWEP ref | ERA5 period diff
        Layout (row 1): ERA5 − MSWEP (short) | ERA5 − MSWEP (long) | [hidden]

        Absolute panels: cmo.tarn (brown=dry, blue=wet).
        Difference panels: BrBG.
        """
        import math

        import nereus as nr
        from nereus.plotting import get_projection

        out = []
        clim = results["clim"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for pk, cdata in clim.items():
            pk_lower = pk.lower()

            era5_ref = cdata["era5_short"] * _PR_MMDAY
            mswep_ref = cdata["mswep_short"] * _PR_MMDAY
            era5_period_diff = (cdata["era5_long"] - cdata["era5_short"]) * _PR_MMDAY
            diff_short = cdata["diff_short"] * _PR_MMDAY
            diff_long = cdata["diff_long"] * _PR_MMDAY

            # Absolute range — precipitation is non-negative
            field_vals = _finite_concat([era5_ref, mswep_ref])
            field_vals_pos = field_vals[field_vals >= 0]
            f_vmax = (
                float(np.percentile(field_vals_pos, 98))
                if len(field_vals_pos) > 0 else 10.0
            )

            # Bias range from all difference fields
            diff_vals = _finite_concat([diff_short, diff_long, era5_period_diff])
            bias_vmax = float(np.percentile(np.abs(diff_vals), 98)) or 1.0

            n_panels = 5
            ncols = 3
            nrows = math.ceil(n_panels / ncols)

            proj = get_projection("rob")
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(7 * ncols, 5 * nrows),
                subplot_kw={"projection": proj},
            )
            axes_flat = np.asarray(axes).ravel().tolist()

            interpolator = None

            # Panel 0: ERA5 reference (absolute, cmo.tarn)
            vals, lons, lats = _flatten_latlon(era5_ref)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[0], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="cmo.tarn",
                vmin=0.0, vmax=f_vmax,
                colorbar=True, colorbar_label="mm/day",
                title=f"ERA5 {short_lbl}",
            )

            # Panel 1: MSWEP reference (absolute, cmo.tarn, reuse interpolator)
            vals, lons, lats = _flatten_latlon(mswep_ref)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[1], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="cmo.tarn",
                vmin=0.0, vmax=f_vmax,
                colorbar=True, colorbar_label="mm/day",
                title=f"MSWEP {short_lbl}",
            )

            # Panel 2: ERA5 period difference (BrBG)
            vals, lons, lats = _flatten_latlon(era5_period_diff)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[2], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="BrBG",
                vmin=-bias_vmax, vmax=bias_vmax,
                colorbar=True, colorbar_label="mm/day",
                title=f"Difference: ERA5 periods\n({short_lbl} → {long_lbl})",
            )

            # Panel 3: ERA5 − MSWEP short period (BrBG)
            vals, lons, lats = _flatten_latlon(diff_short)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[3], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="BrBG",
                vmin=-bias_vmax, vmax=bias_vmax,
                colorbar=True, colorbar_label="mm/day",
                title=f"Difference: ERA5 − MSWEP\n({short_lbl})",
            )

            # Panel 4: ERA5 − MSWEP long period (BrBG)
            vals, lons, lats = _flatten_latlon(diff_long)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[4], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="BrBG",
                vmin=-bias_vmax, vmax=bias_vmax,
                colorbar=True, colorbar_label="mm/day",
                title=f"Difference: ERA5 − MSWEP\n({long_lbl})",
            )

            for j in range(n_panels, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(
                f"pr {pk} Climatological Mean (mm/day)",
                fontsize=14, fontweight="bold", y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            meta = self._build_metadata(
                title=f"pr {pk} Climatology — ERA5 vs MSWEP",
                figure_id=f"pr_obs_{pk_lower}_clim",
                models=[],
                variables=["pr"],
                description=(
                    f"{pk} climatological precipitation (mm/day): "
                    f"ERA5 {short_lbl} and MSWEP {short_lbl} references, "
                    f"ERA5 period difference, and ERA5 − MSWEP bias for "
                    f"both periods."
                ),
                plot_type="combined_bias_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, MSWEP",
            )
            out.append((fig, meta))

        return out

    def _plot_relative_bias(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group E: 2-panel relative bias maps per season (3 figures).

        Shows (ERA5 − MSWEP) / MSWEP × 100 % for the short and long period.
        Masked where MSWEP < 0.1 mm/day to avoid artefacts in arid regions.
        Colormap: BrBG (brown=ERA5 drier, green=ERA5 wetter than MSWEP).
        """
        import nereus as nr
        from nereus.plotting import get_projection

        out = []
        clim = results["clim"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for pk, cdata in clim.items():
            pk_lower = pk.lower()

            rel_short = cdata["rel_bias_short"]
            rel_long = cdata["rel_bias_long"]

            all_vals = _finite_concat([rel_short, rel_long])
            vmax = float(np.percentile(np.abs(all_vals), 98)) if len(all_vals) > 0 else 50.0
            vmax = max(vmax, 1.0)

            proj = get_projection("rob")
            fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 5),
                subplot_kw={"projection": proj},
            )
            axes_flat = list(axes)

            interpolator = None

            # Panel 0: relative bias short period
            vals, lons, lats = _flatten_latlon(rel_short)
            _, _, interpolator = nr.plot(
                vals, lons, lats,
                ax=axes_flat[0], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="BrBG",
                vmin=-vmax, vmax=vmax,
                colorbar=True, colorbar_label="%",
                title=f"ERA5 / MSWEP relative bias\n({short_lbl})",
            )

            # Panel 1: relative bias long period
            vals, lons, lats = _flatten_latlon(rel_long)
            _, _, _ = nr.plot(
                vals, lons, lats,
                ax=axes_flat[1], projection="rob", resolution=0.25,
                interpolator=interpolator, cmap="BrBG",
                vmin=-vmax, vmax=vmax,
                colorbar=True, colorbar_label="%",
                title=f"ERA5 / MSWEP relative bias\n({long_lbl})",
            )

            fig.suptitle(
                f"pr {pk} Relative Bias (ERA5 − MSWEP) / MSWEP × 100 %",
                fontsize=14, fontweight="bold", y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            meta = self._build_metadata(
                title=f"pr {pk} Relative Bias — ERA5 vs MSWEP",
                figure_id=f"pr_obs_{pk_lower}_relative_bias",
                models=[],
                variables=["pr"],
                description=(
                    f"{pk} relative precipitation bias "
                    f"(ERA5 − MSWEP) / MSWEP × 100 % "
                    f"for {short_lbl} and {long_lbl}. "
                    f"Masked where MSWEP < 0.1 mm/day."
                ),
                plot_type="combined_bias_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, MSWEP",
            )
            out.append((fig, meta))

        return out

    def _plot_timeseries(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group C: global mean annual time series (1 figure)."""
        ts = results["timeseries"]
        era5_ts = ts["era5"]
        mswep_ts = ts["mswep"]

        fig, ax = plt.subplots(figsize=(12, 5))

        def _year_axis(da: xr.DataArray) -> np.ndarray:
            if "year" in da.coords:
                return da.year.values
            return np.arange(
                int(self.PERIOD_LONG[0]),
                int(self.PERIOD_LONG[0]) + len(da),
            )

        ax.plot(
            _year_axis(era5_ts), era5_ts.values * _PR_MMDAY,
            color=OBS_COLOR, linewidth=2.0,
            label=f"ERA5 ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})",
        )
        ax.plot(
            _year_axis(mswep_ts), mswep_ts.values * _PR_MMDAY,
            color=_MSWEP_COLOR, linewidth=2.0, linestyle="--",
            label=f"MSWEP ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})",
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Global mean pr (mm/day)")
        ax.set_title(
            "Precipitation: ERA5 vs MSWEP — Global Mean Annual Series",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title="Precipitation Global Mean Annual Series — ERA5 vs MSWEP",
            figure_id="pr_obs_timeseries",
            models=[],
            variables=["pr"],
            description=(
                f"Global-mean annual precipitation (mm/day) from ERA5 and MSWEP "
                f"({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}). "
                "Shows long-term trends and inter-dataset agreement."
            ),
            plot_type="timeseries",
            period=self.PERIOD_LONG,
            obs_dataset="ERA5, MSWEP",
        )
        return [(fig, meta)]


# ── Module-level helpers ────────────────────────────────────────────────────


def _finite_concat(arrays) -> np.ndarray:
    """Concatenate finite values from multiple array-like objects."""
    parts = []
    for a in arrays:
        v = np.asarray(a).ravel()
        parts.append(v[np.isfinite(v)])
    return np.concatenate(parts) if parts else np.array([0.0])
