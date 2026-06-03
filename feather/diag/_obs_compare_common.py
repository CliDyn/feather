"""Shared engine for pairwise observational-dataset comparisons.

The four obs-comparison diagnostics (temperature, precipitation, cloud cover,
temperature extremes) all compare a *reference* gridded observational dataset
against one or more *secondary* datasets over two time windows.  This module
factors out the common machinery so each diagnostic only has to declare a
:class:`CompareSpec` and supply the loaded fields.

For each (reference, secondary, variable) pair the engine produces a fixed set
of figures on a common **0.5° land-masked** grid:

* a 4-panel trend map      (ref/sec × short/long period),
* a 4-panel trend-difference map,
* a 5-panel climatology + bias map,
* optionally a 2-panel relative-bias map (precipitation only),

and exports the climatology and difference fields as NetCDF for reuse.

Fields handed to the engine must already be in framework-canonical units
(temperatures in K, precipitation in kg/m²/s, cloud cover in %), with dims
named ``lat``/``lon`` (or ``latitude``/``longitude``) and longitudes in 0..360.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from feather.plot.maps import _flatten_latlon, plot_combined_map
from feather.util.temporal import (
    annual_mean,
    climatology,
    linear_trend,
    seasonal_annual_mean,
)

logger = logging.getLogger(__name__)

#: Seasonal windows produced for every comparison.
PERIOD_KEYS = ["annual", "DJF", "JJA"]


@dataclass
class CompareSpec:
    """Display/comparison configuration for one variable + secondary dataset."""

    var: str                 # canonical variable name (for metadata/registry)
    ref_name: str            # reference dataset display name, e.g. "ERA5"
    sec_name: str            # secondary dataset display name, e.g. "CHIRPS"
    sec_token: str           # short token used in figure ids / filenames
    units_label: str         # display units, e.g. "mm/day", "°C", "%"
    display_factor: float = 1.0   # stored → display multiplier
    display_offset: float = 0.0   # added after the multiplier
    abs_cmap: str = "cmo.thermal"     # absolute / climatology colormap
    diff_cmap: str = "RdBu_r"         # difference / bias colormap
    trend_cmap: str = "coolwarm"      # trend-map colormap
    abs_nonneg: bool = False          # clamp absolute vmin to 0 (precip)
    relative_bias: bool = False       # add a relative-bias figure (precip)
    land_only: bool = True            # mask reference to secondary coverage
    sec_color: str = "#2e8b57"        # timeseries line colour
    bias_vmax: float | None = None    # fixed |bias| range (None → percentile)
    extent: tuple[float, float] | None = None  # (lat_min, lat_max) clip


# ── Common grid + regridding ────────────────────────────────────────────────


def common_grid_05() -> tuple[np.ndarray, np.ndarray]:
    """0.5° common grid: lats −89.75..89.75, lons 0.25..359.75."""
    lats = np.arange(-89.75, 90.0, 0.5)
    lons = np.arange(0.25, 360.0, 0.5)
    return lats, lons


def interp_to_grid(
    da: xr.DataArray, lats: np.ndarray, lons: np.ndarray,
) -> xr.DataArray:
    """Bilinear regrid a regular lat/lon field to a common grid.

    Handles ``lat``/``lon`` or ``latitude``/``longitude`` dim names, renames
    the result to ``lat``/``lon``, and adds longitude wraparound padding so
    points near the 0°/360° seam interpolate correctly.  NaNs (e.g. ocean
    cells of a land-only dataset) propagate, preserving the land mask.
    """
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

    lon_vals = da[lon_name].values
    if float(lon_vals.max()) > 180:  # 0..360 convention
        n_pad = 10
        left = da.isel({lon_name: slice(-n_pad, None)}).assign_coords(
            {lon_name: lon_vals[-n_pad:] - 360.0}
        )
        right = da.isel({lon_name: slice(None, n_pad)}).assign_coords(
            {lon_name: lon_vals[:n_pad] + 360.0}
        )
        da = xr.concat([left, da, right], dim=lon_name)

    result = da.interp({lat_name: lats, lon_name: lons}, method="linear")

    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if rename:
        result = result.rename(rename)
    return result


# ── Time aggregation ────────────────────────────────────────────────────────


def compute_trend(da: xr.DataArray, period_key: str) -> xr.DataArray | None:
    """Linear trend per decade for annual or a specific season."""
    if period_key == "annual":
        grouped = annual_mean(da)   # → 'time' dim (year-end)
        dim = "time"
    else:
        grouped = seasonal_annual_mean(da, period_key)  # → 'year' dim
        dim = "year"
    if hasattr(grouped, "compute"):
        grouped = grouped.compute()
    n = grouped.sizes.get("year", grouped.sizes.get("time", 0))
    if n < 2:
        return None
    return linear_trend(grouped, dim=dim) * 10


def seasonal_mean(da: xr.DataArray, period_key: str) -> xr.DataArray:
    """Time mean for annual or a specific season (DJF/JJA)."""
    if period_key == "annual":
        result = climatology(da)
    else:
        result = da.where(da.time.dt.season == period_key).mean("time")
    if hasattr(result, "compute"):
        result = result.compute()
    return result


def finite_concat(arrays) -> np.ndarray:
    """Concatenate the finite values of several array-likes."""
    parts = []
    for a in arrays:
        v = np.asarray(a).ravel()
        parts.append(v[np.isfinite(v)])
    return np.concatenate(parts) if parts else np.array([0.0])


def area_weighted_annual_series(da: xr.DataArray) -> xr.DataArray:
    """cos(lat)-weighted spatial mean of each annual mean (NaN-aware).

    Suitable for land-only fields: ocean NaNs are excluded from the weighted
    average per timestep.  Returns a 1-D series indexed by ``year``.
    """
    annual = annual_mean(da)  # 'time' dim at year-end
    if hasattr(annual, "compute"):
        annual = annual.compute()
    # Re-index the time axis to integer years for plotting
    annual = annual.assign_coords(time=annual.time.dt.year.values).rename(
        {"time": "year"})
    lat_name = "lat" if "lat" in annual.dims else "latitude"
    weights = np.cos(np.deg2rad(annual[lat_name]))
    valid = np.isfinite(annual)
    w = weights.broadcast_like(annual).where(valid)
    spatial = [d for d in annual.dims if d != "year"]
    num = (annual.where(valid) * w).sum(dim=spatial)
    den = w.sum(dim=spatial)
    return num / den


# ── The engine ──────────────────────────────────────────────────────────────


class PairwiseObsComparison:
    """Build the figure set + NetCDF for one reference/secondary pair.

    Parameters
    ----------
    diag : DiagnosticBase
        Owning diagnostic (used for ``_build_metadata``).
    spec : CompareSpec
        Variable/dataset display configuration.
    period_short, period_long : tuple of str
        (start, end) windows; long must extend the short one.
    """

    def __init__(self, diag, spec: CompareSpec, period_short, period_long):
        self.diag = diag
        self.spec = spec
        self.period_short = period_short
        self.period_long = period_long

    # -- computation ----------------------------------------------------------

    def compute(self, ref_short, ref_long, sec_short, sec_long) -> dict:
        """Regrid, mask, and compute trends + climatologies for both periods."""
        spec = self.spec
        lats, lons = common_grid_05()

        ref_s = interp_to_grid(ref_short, lats, lons)
        ref_l = interp_to_grid(ref_long, lats, lons)
        sec_s = interp_to_grid(sec_short, lats, lons)
        sec_l = interp_to_grid(sec_long, lats, lons)

        if spec.extent is not None:
            lo, hi = spec.extent
            sel = {"lat": slice(lo, hi)} if lats[0] < lats[-1] else {"lat": slice(hi, lo)}
            ref_s, ref_l = ref_s.sel(sel), ref_l.sel(sel)
            sec_s, sec_l = sec_s.sel(sel), sec_l.sel(sel)

        trends, clim = {}, {}
        for pk in PERIOD_KEYS:
            r_s_t = compute_trend(ref_s, pk)
            r_l_t = compute_trend(ref_l, pk)
            s_s_t = compute_trend(sec_s, pk)
            s_l_t = compute_trend(sec_l, pk)
            if any(t is None for t in (r_s_t, r_l_t, s_s_t, s_l_t)):
                logger.warning("Insufficient data for %s trends — skipping", pk)
            else:
                trends[pk] = {
                    "ref_short": r_s_t, "ref_long": r_l_t,
                    "sec_short": s_s_t, "sec_long": s_l_t,
                    "ref_period_diff": r_l_t - r_s_t,
                    "sec_period_diff": s_l_t - s_s_t,
                    "dataset_diff_short": r_s_t - s_s_t,
                    "dataset_diff_long": r_l_t - s_l_t,
                }

            r_s_c = seasonal_mean(ref_s, pk)
            r_l_c = seasonal_mean(ref_l, pk)
            s_s_c = seasonal_mean(sec_s, pk)
            s_l_c = seasonal_mean(sec_l, pk)

            # Mask the (possibly global) reference to the land-only secondary
            if spec.land_only:
                r_s_c = r_s_c.where(np.isfinite(s_s_c))
                r_l_c = r_l_c.where(np.isfinite(s_l_c))

            entry = {
                "ref_short": r_s_c, "ref_long": r_l_c,
                "sec_short": s_s_c, "sec_long": s_l_c,
                "diff_short": r_s_c - s_s_c,
                "diff_long": r_l_c - s_l_c,
            }
            if spec.relative_bias:
                thr = 0.1 / spec.display_factor if spec.display_factor else 0.0
                entry["rel_bias_short"] = xr.where(
                    s_s_c > thr, (r_s_c - s_s_c) / s_s_c * 100.0, np.nan)
                entry["rel_bias_long"] = xr.where(
                    s_l_c > thr, (r_l_c - s_l_c) / s_l_c * 100.0, np.nan)
            clim[pk] = entry

        return {
            "trends": trends, "clim": clim,
            "fields": {"ref_long": ref_l, "sec_long": sec_l},
        }

    def _disp(self, da):
        """Convert a stored field to display units."""
        return da * self.spec.display_factor + self.spec.display_offset

    # -- figures --------------------------------------------------------------

    def figures(self, results) -> list[tuple[plt.Figure, dict]]:
        out = []
        out += self._trend_maps(results["trends"])
        out += self._trend_diffs(results["trends"])
        out += self._clim_maps(results["clim"])
        if self.spec.relative_bias:
            out += self._relative_bias(results["clim"])
        return out

    def _ids(self, pk: str, kind: str) -> str:
        s = self.spec
        return f"{s.var}_{s.sec_token}_{pk.lower()}_{kind}"

    def _labels(self):
        s = self.spec
        short = f"{self.period_short[0]}–{self.period_short[1]}"
        long = f"{self.period_long[0]}–{self.period_long[1]}"
        return short, long

    def _trend_maps(self, trends):
        s = self.spec
        short, long = self._labels()
        out = []
        for pk, p in trends.items():
            tu = f"{s.units_label}/decade"
            data = {
                f"{s.ref_name} {short}": self._disp(p["ref_short"]),
                f"{s.ref_name} {long}": self._disp(p["ref_long"]),
                f"{s.sec_name} {short}": self._disp(p["sec_short"]),
                f"{s.sec_name} {long}": self._disp(p["sec_long"]),
            }
            vmax = float(np.percentile(np.abs(finite_concat(data.values())), 98)) or 1.0
            fig, _ = plot_combined_map(
                data, title=f"{s.var} {pk} Trend ({tu})",
                cmap=s.trend_cmap, vmin=-vmax, vmax=vmax,
                units=tu, max_cols=2,
            )
            meta = self.diag._build_metadata(
                title=f"{s.var} {pk} Trend Maps — {s.ref_name} vs {s.sec_name}",
                figure_id=self._ids(pk, "trends"), models=[],
                variables=[s.var],
                description=(
                    f"{pk} {s.var} trends ({tu}) from {s.ref_name} and "
                    f"{s.sec_name} for {short} and {long} (0.5° land-masked)."
                ),
                plot_type="combined_trend_map", period=self.period_long,
                obs_dataset=f"{s.ref_name}, {s.sec_name}",
            )
            out.append((fig, meta))
        return out

    def _trend_diffs(self, trends):
        s = self.spec
        short, long = self._labels()
        out = []
        for pk, p in trends.items():
            tu = f"{s.units_label}/decade"
            data = {
                f"{s.ref_name} period diff\n({self.period_long[1]}−{self.period_short[1]})": self._disp(p["ref_period_diff"]),
                f"{s.sec_name} period diff\n({self.period_long[1]}−{self.period_short[1]})": self._disp(p["sec_period_diff"]),
                f"{s.ref_name} − {s.sec_name}\n({short})": self._disp(p["dataset_diff_short"]),
                f"{s.ref_name} − {s.sec_name}\n({long})": self._disp(p["dataset_diff_long"]),
            }
            vmax = float(np.percentile(np.abs(finite_concat(data.values())), 98)) or 1.0
            fig, _ = plot_combined_map(
                data, title=f"{s.var} {pk} Trend Differences ({tu})",
                cmap=s.diff_cmap, vmin=-vmax, vmax=vmax,
                units=tu, max_cols=2,
            )
            meta = self.diag._build_metadata(
                title=f"{s.var} {pk} Trend Differences — {s.ref_name} vs {s.sec_name}",
                figure_id=self._ids(pk, "trend_diffs"), models=[],
                variables=[s.var],
                description=(
                    f"{pk} {s.var} trend differences ({tu}): period extension "
                    f"and {s.ref_name} − {s.sec_name} disagreement."
                ),
                plot_type="combined_trend_map", period=self.period_long,
                obs_dataset=f"{s.ref_name}, {s.sec_name}",
            )
            out.append((fig, meta))
        return out

    def _clim_maps(self, clim):
        import nereus as nr
        from nereus.plotting import get_projection

        s = self.spec
        short, long = self._labels()
        out = []
        for pk, c in clim.items():
            ref_ref = self._disp(c["ref_short"])
            sec_ref = self._disp(c["sec_short"])
            ref_period_diff = self._disp(c["ref_long"]) - self._disp(c["ref_short"])
            diff_short = self._disp(c["ref_short"]) - self._disp(c["sec_short"])
            diff_long = self._disp(c["ref_long"]) - self._disp(c["sec_long"])

            field_vals = finite_concat([ref_ref, sec_ref])
            f_vmin = 0.0 if s.abs_nonneg else float(np.percentile(field_vals, 2))
            f_vmax = float(np.percentile(field_vals, 98))
            if s.bias_vmax is not None:
                bias_vmax = s.bias_vmax
            else:
                bias_vmax = float(np.percentile(
                    np.abs(finite_concat([diff_short, diff_long, ref_period_diff])), 98,
                )) or 1.0

            n_panels, ncols = 5, 3
            nrows = math.ceil(n_panels / ncols)
            proj = get_projection("rob")
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                subplot_kw={"projection": proj},
            )
            axf = np.asarray(axes).ravel().tolist()
            itp = None
            panels = [
                (ref_ref, s.abs_cmap, f_vmin, f_vmax, f"{s.ref_name} {short}"),
                (sec_ref, s.abs_cmap, f_vmin, f_vmax, f"{s.sec_name} {short}"),
                (ref_period_diff, s.diff_cmap, -bias_vmax, bias_vmax,
                 f"Difference: {s.ref_name} periods\n({short} → {long})"),
                (diff_short, s.diff_cmap, -bias_vmax, bias_vmax,
                 f"Difference: {s.ref_name} − {s.sec_name}\n({short})"),
                (diff_long, s.diff_cmap, -bias_vmax, bias_vmax,
                 f"Difference: {s.ref_name} − {s.sec_name}\n({long})"),
            ]
            for i, (field, cmap, vmin, vmax, ttl) in enumerate(panels):
                vals, lon, lat = _flatten_latlon(field)
                _, _, itp = nr.plot(
                    vals, lon, lat, ax=axf[i], projection="rob", resolution=0.5,
                    interpolator=itp, cmap=cmap, vmin=vmin, vmax=vmax,
                    colorbar=True, colorbar_label=s.units_label, title=ttl,
                )
            for j in range(n_panels, len(axf)):
                axf[j].set_visible(False)
            fig.suptitle(
                f"{s.var} {pk} Climatological Mean ({s.units_label})",
                fontsize=14, fontweight="bold", y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            meta = self.diag._build_metadata(
                title=f"{s.var} {pk} Climatology — {s.ref_name} vs {s.sec_name}",
                figure_id=self._ids(pk, "clim"), models=[],
                variables=[s.var],
                description=(
                    f"{pk} climatological {s.var} ({s.units_label}): "
                    f"{s.ref_name} and {s.sec_name} references, {s.ref_name} "
                    f"period difference, and {s.ref_name} − {s.sec_name} bias."
                ),
                plot_type="combined_bias_map", period=self.period_long,
                obs_dataset=f"{s.ref_name}, {s.sec_name}",
            )
            out.append((fig, meta))
        return out

    def _relative_bias(self, clim):
        import nereus as nr
        from nereus.plotting import get_projection

        s = self.spec
        short, long = self._labels()
        out = []
        for pk, c in clim.items():
            rel_short, rel_long = c["rel_bias_short"], c["rel_bias_long"]
            vals = finite_concat([rel_short, rel_long])
            vmax = max(float(np.percentile(np.abs(vals), 98)) if len(vals) else 50.0, 1.0)
            proj = get_projection("rob")
            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5), subplot_kw={"projection": proj})
            axf = list(axes)
            itp = None
            for i, (field, lbl) in enumerate(
                [(rel_short, short), (rel_long, long)]
            ):
                v, lon, lat = _flatten_latlon(field)
                _, _, itp = nr.plot(
                    v, lon, lat, ax=axf[i], projection="rob", resolution=0.5,
                    interpolator=itp, cmap=s.diff_cmap, vmin=-vmax, vmax=vmax,
                    colorbar=True, colorbar_label="%",
                    title=f"{s.ref_name} / {s.sec_name} relative bias\n({lbl})",
                )
            fig.suptitle(
                f"{s.var} {pk} Relative Bias "
                f"({s.ref_name} − {s.sec_name}) / {s.sec_name} × 100 %",
                fontsize=14, fontweight="bold", y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            meta = self.diag._build_metadata(
                title=f"{s.var} {pk} Relative Bias — {s.ref_name} vs {s.sec_name}",
                figure_id=self._ids(pk, "relative_bias"), models=[],
                variables=[s.var],
                description=(
                    f"{pk} relative {s.var} bias ({s.ref_name} − {s.sec_name}) "
                    f"/ {s.sec_name} × 100 % for {short} and {long}."
                ),
                plot_type="combined_bias_map", period=self.period_long,
                obs_dataset=f"{s.ref_name}, {s.sec_name}",
            )
            out.append((fig, meta))
        return out

    # -- NetCDF export --------------------------------------------------------

    def export_netcdf(self, results, netcdf_dir: Path) -> list[Path]:
        """Write climatology + difference fields to NetCDF.

        One file per period-key holds the reference/secondary climatologies and
        their difference (annual/DJF/JJA), in *stored* (canonical) units.
        """
        s = self.spec
        netcdf_dir = Path(netcdf_dir)
        netcdf_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for pk, c in results["clim"].items():
            ds = xr.Dataset(
                {
                    f"{s.ref_name.lower()}_clim": c["ref_short"],
                    f"{s.sec_name.lower()}_clim": c["sec_short"],
                    "diff_short": c["diff_short"],
                    "diff_long": c["diff_long"],
                }
            )
            ds.attrs.update(
                variable=s.var, reference=s.ref_name, secondary=s.sec_name,
                period_short=f"{self.period_short[0]}-{self.period_short[1]}",
                period_long=f"{self.period_long[0]}-{self.period_long[1]}",
                units=f"canonical ({s.units_label} after display factor "
                      f"{s.display_factor}, offset {s.display_offset})",
                grid="0.5deg land-masked",
            )
            path = netcdf_dir / f"{s.var}_{s.sec_token}_{pk.lower()}_clim_diff.nc"
            ds.to_netcdf(path)
            written.append(path)
        return written
