"""Observational dataset comparison: ERA5 vs Berkeley Earth trends and biases.

Produces 10 figures across 4 groups:
A (x3): Trend maps (ERA5 + Berkeley Earth, both periods) — annual, DJF, JJA
B (x3): Trend difference maps (period diffs + dataset diffs) — annual, DJF, JJA
C (x1): Global mean time series for both datasets
D (x3): Climatological bias maps (ERA5 − Berkeley Earth) — annual, DJF, JJA

Periods compared:
- Short: 1980–2014  (model-comparable period)
- Long:  1980–2024  (full available record)
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
from feather.plot.maps import plot_combined_bias_map, plot_combined_map
from feather.plot.styles import OBS_COLOR
from feather.util.spatial import compute_latlon_areas, latlon_global_mean
from feather.util.temporal import (
    annual_mean,
    climatology,
    linear_trend,
    seasonal_annual_mean,
)

logger = logging.getLogger(__name__)

# Amber/orange that contrasts well with OBS_COLOR (usually teal/blue)
_BERKELEY_COLOR = "#e07b39"


@register
class ObsComparisonDiag(DiagnosticBase):
    """ERA5 vs Berkeley Earth comparison for global trends and biases.

    Obs-only diagnostic: no model data is loaded or compared.
    Evaluates 2m temperature (tas) for two periods to assess how trend
    estimates depend on the record length and the choice of observational
    dataset.

    Figures
    -------
    A — Trend maps: ERA5 and Berkeley Earth for both periods (3 figures)
    B — Trend differences: period extension and dataset disagreement (3)
    C — Global mean time series for both datasets (1)
    D — Climatological bias maps ERA5 − Berkeley Earth (3)
    """

    name = "obs_comparison"
    title = "Observational Dataset Comparison (ERA5 vs Berkeley Earth 0.25°)"
    domain = "sfc"
    variables = ["tas"]
    group = "evaluation"

    PERIOD_SHORT = ("1980", "2014")
    PERIOD_LONG = ("1980", "2024")

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

        all_ids = ["tas_obs_timeseries"]
        for pk in ["annual", "djf", "jja"]:
            all_ids += [
                f"tas_{pk}_obs_trends",
                f"tas_{pk}_obs_trend_diffs",
                f"tas_{pk}_obs_clim_bias",
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
        """Load ERA5 and Berkeley Earth, compute trends and climatologies.

        Returns
        -------
        dict with keys:
            trends    — per period-key (annual/DJF/JJA) trend fields and diffs
            clim      — per period-key climatological means and biases
            timeseries — global mean annual time series for both datasets
            common_lats, common_lons — 1° common grid coordinates
            area      — area weights for global mean computation
        """
        logger.info("Loading ERA5 data for both periods...")
        era5_short = self._load_era5(self.PERIOD_SHORT)
        era5_long = self._load_era5(self.PERIOD_LONG)

        logger.info("Loading Berkeley Earth data for both periods...")
        be_short = self._load_berkeley_earth(self.PERIOD_SHORT)
        be_long = self._load_berkeley_earth(self.PERIOD_LONG)

        logger.info("Regridding to common 0.25° grid...")
        common_lats, common_lons = self._common_grid()
        era5_s = self._interp_to_common(era5_short, common_lats, common_lons)
        era5_l = self._interp_to_common(era5_long, common_lats, common_lons)
        be_s = self._interp_to_common(be_short, common_lats, common_lons)
        be_l = self._interp_to_common(be_long, common_lats, common_lons)

        area = compute_latlon_areas(common_lats, common_lons)

        trends = self._compute_all_trends(era5_s, era5_l, be_s, be_l)
        clim = self._compute_clim_bias(era5_s, era5_l, be_s, be_l)
        ts = self._compute_timeseries(era5_l, be_l, area)

        return {
            "trends": trends,
            "clim": clim,
            "timeseries": ts,
            "common_lats": common_lats,
            "common_lons": common_lons,
            "area": area,
        }

    def _compute_all_trends(
        self,
        era5_s: xr.DataArray,
        era5_l: xr.DataArray,
        be_s: xr.DataArray,
        be_l: xr.DataArray,
    ) -> dict[str, dict]:
        """Compute annual/DJF/JJA trends and their differences.

        Returns
        -------
        dict keyed by "annual", "DJF", "JJA".  Each value contains:
            era5_short, era5_long, be_short, be_long — trend maps (K/decade)
            era5_period_diff   — ERA5 trend change when extending to 2024
            be_period_diff     — Berkeley Earth trend change, same extension
            dataset_diff_short — ERA5 − BE trend for 1980–2014
            dataset_diff_long  — ERA5 − BE trend for 1980–2024
        """
        results: dict[str, dict] = {}

        for period_key in ["annual", "DJF", "JJA"]:
            era5_s_t = self._compute_trend(era5_s, period_key)
            era5_l_t = self._compute_trend(era5_l, period_key)
            be_s_t = self._compute_trend(be_s, period_key)
            be_l_t = self._compute_trend(be_l, period_key)

            if any(t is None for t in [era5_s_t, era5_l_t, be_s_t, be_l_t]):
                logger.warning(
                    "Insufficient data for %s trends — skipping", period_key,
                )
                continue

            results[period_key] = {
                "era5_short": era5_s_t,
                "era5_long": era5_l_t,
                "be_short": be_s_t,
                "be_long": be_l_t,
                "era5_period_diff": era5_l_t - era5_s_t,
                "be_period_diff": be_l_t - be_s_t,
                "dataset_diff_short": era5_s_t - be_s_t,
                "dataset_diff_long": era5_l_t - be_l_t,
            }

        return results

    def _compute_clim_bias(
        self,
        era5_s: xr.DataArray,
        era5_l: xr.DataArray,
        be_s: xr.DataArray,
        be_l: xr.DataArray,
    ) -> dict[str, dict]:
        """Compute climatological means and ERA5 − Berkeley Earth biases.

        Returns
        -------
        dict keyed by "annual", "DJF", "JJA". Each value contains:
            era5_short, era5_long, be_short, be_long — time means (K)
            diff_short — ERA5 − BE for 1980–2014
            diff_long  — ERA5 − BE for 1980–2024
        """
        results: dict[str, dict] = {}

        for period_key in ["annual", "DJF", "JJA"]:
            era5_s_c = self._seasonal_mean(era5_s, period_key)
            era5_l_c = self._seasonal_mean(era5_l, period_key)
            be_s_c = self._seasonal_mean(be_s, period_key)
            be_l_c = self._seasonal_mean(be_l, period_key)

            results[period_key] = {
                "era5_short": era5_s_c,
                "era5_long": era5_l_c,
                "be_short": be_s_c,
                "be_long": be_l_c,
                "diff_short": era5_s_c - be_s_c,
                "diff_long": era5_l_c - be_l_c,
            }

        return results

    def _compute_timeseries(
        self,
        era5_l: xr.DataArray,
        be_l: xr.DataArray,
        area: xr.DataArray,
    ) -> dict[str, xr.DataArray]:
        """Global mean annual time series for both datasets (full period)."""
        era5_annual = annual_mean(era5_l)
        if hasattr(era5_annual, "compute"):
            era5_annual = era5_annual.compute()
        be_annual = annual_mean(be_l)
        if hasattr(be_annual, "compute"):
            be_annual = be_annual.compute()

        return {
            "era5": latlon_global_mean(era5_annual, area=area),
            "be": latlon_global_mean(be_annual, area=area),
        }

    # ── Static computation helpers ─────────────────────────────────────

    @staticmethod
    def _compute_trend(
        da: xr.DataArray, period_key: str,
    ) -> xr.DataArray | None:
        """Linear trend (K/decade) for annual or a specific season."""
        if period_key == "annual":
            grouped = annual_mean(da)
            if hasattr(grouped, "compute"):
                grouped = grouped.compute()
            n = grouped.sizes.get("year", grouped.sizes.get("time", 0))
            if n < 2:
                return None
            return linear_trend(grouped) * 10

        # Seasonal: one value per calendar year
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

    # ── Data loading helpers ───────────────────────────────────────────

    def _load_era5(self, period: tuple[str, str]) -> xr.DataArray:
        """Load ERA5 2m temperature for the given period via VARIABLE_REGISTRY."""
        return self._load_obs_var("tas", period)

    def _load_berkeley_earth(self, period: tuple[str, str]) -> xr.DataArray:
        """Load Berkeley Earth 2m temperature.

        Prefers the high-resolution 0.25° dataset (``BERKELEY_EARTH_HR``)
        when it is present in the config, falling back to the legacy 1°
        dataset (``BERKELEY_EARTH``) otherwise.
        """
        if "BERKELEY_EARTH_HR" in self.config.obs_datasets:
            return self._load_berkeley_earth_hr(period)
        return self._load_berkeley_earth_legacy(period)

    def _load_berkeley_earth_hr(self, period: tuple[str, str]) -> xr.DataArray:
        """Load Berkeley Earth 0.25° gridded dataset.

        The file stores monthly *anomalies* (°C, re: 1951-1980 climatology)
        and a separate ``climatology`` array (12 × lat × lon, °C).
        Absolute temperature = anomaly + climatology[month_of_year].

        Time is encoded as decimal years (float); this method converts it to
        a proper ``pandas.DatetimeIndex`` before slicing.
        """
        # Open as dataset to access both 'temperature' and 'climatology' variables
        ds_cfg = self.config.obs_datasets["BERKELEY_EARTH_HR"]
        filepath = Path(ds_cfg["path"]) / ds_cfg["variables"]["temperature"]
        ds_full = xr.open_dataset(filepath, chunks="auto")

        # Convert decimal-year time → DatetimeIndex
        dec_years = ds_full["time"].values  # e.g. 1850.042, 1850.125, ...
        years = dec_years.astype(int)
        # Month derived from fractional part: 12 evenly-spaced values per year
        months = np.round((dec_years - years) * 12).astype(int) + 1
        months = np.clip(months, 1, 12)
        datetimes = pd.to_datetime(
            [f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)]
        )
        ds_full = ds_full.assign_coords(time=datetimes)

        # Slice to requested period
        start, end = period
        anom = ds_full["temperature"].sel(time=slice(start, end))
        clim = ds_full["climatology"]  # (month_number, latitude, longitude)

        # Reconstruct absolute temperature: anomaly + climatology[month_of_year]
        month_idx = anom.time.dt.month.values - 1  # 0-based index
        clim_np = clim.values  # (12, nlat, nlon)
        clim_matched = clim_np[month_idx]  # (ntime, nlat, nlon)
        abs_temp = anom + xr.DataArray(
            clim_matched,
            dims=anom.dims,
            coords=anom.coords,
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

        # degC → K
        return abs_temp + 273.15

    def _load_berkeley_earth_legacy(self, period: tuple[str, str]) -> xr.DataArray:
        """Load legacy Berkeley Earth 1° Land+Ocean file (absolute °C → K)."""
        da = self.obs_loader.load("BERKELEY_EARTH", "2t", period=period)

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

        return da + 273.15

    @staticmethod
    def _common_grid() -> tuple[np.ndarray, np.ndarray]:
        """0.25° common grid: lats −89.875..89.875, lons 0.125..359.875.

        Matches the native resolution of ERA5 and Berkeley Earth 0.25°,
        so both datasets are interpolated without loss of spatial detail.
        """
        lats = np.arange(-89.875, 90.0, 0.25)
        lons = np.arange(0.125, 360.0, 0.25)
        return lats, lons

    @staticmethod
    def _interp_to_common(
        da: xr.DataArray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> xr.DataArray:
        """Bilinear interpolation of a regular lat/lon field to the common grid.

        Handles both 'lat'/'lon' and 'latitude'/'longitude' dim names and
        renames the output dims to 'lat'/'lon'.
        """
        lat_name = "lat" if "lat" in da.dims else "latitude"
        lon_name = "lon" if "lon" in da.dims else "longitude"

        result = da.interp(
            {lat_name: lats, lon_name: lons},
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
        """Generate all 10 figures."""
        figures = []
        figures.extend(self._plot_trend_maps(results))
        figures.extend(self._plot_trend_diffs(results))
        figures.extend(self._plot_clim_bias(results))
        figures.extend(self._plot_timeseries(results))
        return figures

    def _plot_trend_maps(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group A: 4-panel trend maps per seasonal period (3 figures)."""
        out = []
        trends = results["trends"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for period_key, pdata in trends.items():
            pk = period_key.lower()

            data_dict = {
                f"ERA5 {short_lbl}": pdata["era5_short"],
                f"ERA5 {long_lbl}": pdata["era5_long"],
                f"Berkeley Earth {short_lbl}": pdata["be_short"],
                f"Berkeley Earth {long_lbl}": pdata["be_long"],
            }

            all_vals = _finite_concat(data_dict.values())
            vmax = float(np.percentile(np.abs(all_vals), 98)) or 1.0

            fig, _ = plot_combined_map(
                data_dict,
                title=f"tas {period_key} Warming Trend (K/decade)",
                cmap="coolwarm",
                vmin=-vmax, vmax=vmax,
                units="K/decade",
                max_cols=2,
            )

            meta = self._build_metadata(
                title=f"tas {period_key} Trend Maps — ERA5 vs Berkeley Earth",
                figure_id=f"tas_{pk}_obs_trends",
                models=[],
                variables=["tas"],
                description=(
                    f"{period_key} warming trends (K/decade) from ERA5 and "
                    f"Berkeley Earth for {short_lbl} and {long_lbl}."
                ),
                plot_type="combined_trend_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, Berkeley Earth",
            )
            out.append((fig, meta))

        return out

    def _plot_trend_diffs(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group B: 4-panel trend difference maps (3 figures)."""
        out = []
        trends = results["trends"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for period_key, pdata in trends.items():
            pk = period_key.lower()

            data_dict = {
                f"ERA5 period diff\n({self.PERIOD_LONG[1]}−{self.PERIOD_SHORT[1]})": pdata["era5_period_diff"],
                f"Berkeley Earth period diff\n({self.PERIOD_LONG[1]}−{self.PERIOD_SHORT[1]})": pdata["be_period_diff"],
                f"ERA5 − Berkeley Earth\n({short_lbl})": pdata["dataset_diff_short"],
                f"ERA5 − Berkeley Earth\n({long_lbl})": pdata["dataset_diff_long"],
            }

            all_vals = _finite_concat(data_dict.values())
            vmax = float(np.percentile(np.abs(all_vals), 98)) or 1.0

            fig, _ = plot_combined_map(
                data_dict,
                title=f"tas {period_key} Trend Differences (K/decade)",
                cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                units="K/decade",
                max_cols=2,
            )

            meta = self._build_metadata(
                title=f"tas {period_key} Trend Differences — Period and Dataset",
                figure_id=f"tas_{pk}_obs_trend_diffs",
                models=[],
                variables=["tas"],
                description=(
                    f"{period_key} trend differences (K/decade): period "
                    f"extension effect ({short_lbl} → {long_lbl}) and "
                    f"ERA5 − Berkeley Earth dataset disagreement."
                ),
                plot_type="combined_trend_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, Berkeley Earth",
            )
            out.append((fig, meta))

        return out

    def _plot_clim_bias(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group D: climatological bias maps ERA5 − Berkeley Earth (3 figures).

        Layout: ERA5 clim (short period) as reference + 3 difference
        panels (ERA5 period change, ERA5-BE short, ERA5-BE long).
        """
        out = []
        clim = results["clim"]
        short_lbl = f"{self.PERIOD_SHORT[0]}–{self.PERIOD_SHORT[1]}"
        long_lbl = f"{self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}"

        for period_key, cdata in clim.items():
            pk = period_key.lower()

            # Absolute field range from ERA5 short period
            field_vals = _finite_concat([cdata["era5_short"]])
            f_vmin = float(np.percentile(field_vals, 2))
            f_vmax = float(np.percentile(field_vals, 98))

            # Symmetric bias range across all difference fields
            diff_vals = _finite_concat([
                cdata["diff_short"],
                cdata["diff_long"],
                cdata["era5_long"] - cdata["era5_short"],
            ])
            bias_vmax = float(np.percentile(np.abs(diff_vals), 98)) or 1.0

            diff_dict = {
                f"ERA5 clim change\n({short_lbl} → {long_lbl})": (
                    cdata["era5_long"] - cdata["era5_short"]
                ),
                f"ERA5 − Berkeley Earth\n({short_lbl})": cdata["diff_short"],
                f"ERA5 − Berkeley Earth\n({long_lbl})": cdata["diff_long"],
            }

            fig, _ = plot_combined_bias_map(
                cdata["era5_short"],
                diff_dict,
                title=f"tas {period_key} Climatological Mean",
                obs_title=f"ERA5 {short_lbl}",
                cmap="RdBu_r",
                bias_cmap="RdBu_r",
                vmin=f_vmin, vmax=f_vmax,
                bias_vmax=bias_vmax,
                units="K",
                bias_title_prefix="Difference",
            )

            meta = self._build_metadata(
                title=f"tas {period_key} Climatology — ERA5 vs Berkeley Earth",
                figure_id=f"tas_{pk}_obs_clim_bias",
                models=[],
                variables=["tas"],
                description=(
                    f"{period_key} climatological mean 2m temperature (K): "
                    f"ERA5 {short_lbl} reference, ERA5 period change, and "
                    f"ERA5 − Berkeley Earth dataset bias for both periods."
                ),
                plot_type="combined_bias_map",
                period=self.PERIOD_LONG,
                obs_dataset="ERA5, Berkeley Earth",
            )
            out.append((fig, meta))

        return out

    def _plot_timeseries(self, results) -> list[tuple[plt.Figure, dict]]:
        """Group C: global mean annual time series (1 figure)."""
        ts = results["timeseries"]
        era5_ts = ts["era5"]
        be_ts = ts["be"]

        fig, ax = plt.subplots(figsize=(12, 5))

        def _year_axis(da: xr.DataArray) -> np.ndarray:
            if "year" in da.coords:
                return da.year.values
            return np.arange(
                int(self.PERIOD_LONG[0]),
                int(self.PERIOD_LONG[0]) + len(da),
            )

        ax.plot(
            _year_axis(era5_ts), era5_ts.values,
            color=OBS_COLOR, linewidth=2.0,
            label=f"ERA5 ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})",
        )
        ax.plot(
            _year_axis(be_ts), be_ts.values,
            color=_BERKELEY_COLOR, linewidth=2.0, linestyle="--",
            label=f"Berkeley Earth ({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]})",
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Global mean tas (K)")
        ax.set_title(
            "2m Temperature: ERA5 vs Berkeley Earth — Global Mean Annual Series",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        meta = self._build_metadata(
            title="2m Temperature Global Mean Annual Series — ERA5 vs Berkeley Earth",
            figure_id="tas_obs_timeseries",
            models=[],
            variables=["tas"],
            description=(
                f"Global-mean annual 2m temperature (K) from ERA5 and Berkeley Earth "
                f"({self.PERIOD_LONG[0]}–{self.PERIOD_LONG[1]}). "
                "Shows long-term warming and inter-dataset agreement."
            ),
            plot_type="timeseries",
            period=self.PERIOD_LONG,
            obs_dataset="ERA5, Berkeley Earth",
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
