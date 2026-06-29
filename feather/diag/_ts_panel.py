"""Shared rendering for envelope / anomaly global-mean time-series figures.

Used by the ``timeseries``, ``temperature_berkeley`` and
``precipitation_mswep`` diagnostics so the three produce a consistent
"absolute with benchmark min/max envelope" and "anomaly relative to the
full-period mean" pair of figures.

The display transform is ``value * factor + offset`` for absolute figures
(e.g. K→°C uses ``offset=-273.15``; kg/m²/s→mm/day uses ``factor=86400``).  For
anomalies the additive *offset* cancels, so each series is shown as
``(value - value.mean("time")) * factor``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feather.plot.styles import ENS_COLOR, OBS_COLOR
from feather.util.temporal import annual_mean

#: Color/style for an auxiliary ERA5 reference line.
_ERA5_COLOR = "black"


# ── NetCDF persistence / reconstruction for the envelope figures ────────

def export_timeseries_netcdf(
    netcdf_dir, token: str, results: dict, period, *, skip_existing: bool = True,
):
    """Write a timeseries result dict to NetCDF, tagging benchmark metadata.

    Extends the generic per-source export with ``ts_benchmark_*`` global
    attributes (label, color, member count, indexed by benchmark) so the
    figures can later be rebuilt from the NetCDF alone via
    :func:`load_timeseries_netcdf`.  *token* names the file
    (``{token}_{start}-{end}.nc``); pass the bare variable for the standalone
    timeseries diagnostic or ``{var}_timeseries`` for diagnostics that also
    write bias-map NetCDF for the same variable.
    """
    import json

    from feather.diag import netcdf_export

    benches = results.get("benchmarks_ts", []) or []
    extra = {
        "ts_benchmark_labels": json.dumps([b.get("label", "") for b in benches]),
        "ts_benchmark_colors": json.dumps(
            [b.get("color") or "" for b in benches]),
        "ts_benchmark_n_members": json.dumps(
            [int((b.get("info") or {}).get("n_members", 0)) for b in benches]),
    }
    return netcdf_export.export_generic_netcdf(
        netcdf_dir, token, results, period,
        skip_existing=skip_existing, extra_attrs=extra,
    )


def load_timeseries_netcdf(nc_path, name_map: dict, benchmarks, benchmark_color):
    """Reconstruct a timeseries result dict from a NetCDF written by
    :func:`export_timeseries_netcdf`.

    Returns a dict with ``models``, ``obs``, ``era5_ts``, ``benchmarks_ts``
    (each with ``env_min``/``env_max``), plus ``ens_mean``/``ens_median`` and
    the back-compat ``cmip6_*`` keys.  Returns ``None`` when the file has no
    ``obs`` field.  *name_map* maps sanitised model names back to display
    names; *benchmark_color(i)`` supplies a fallback color.
    """
    import json

    import xarray as xr

    ds = xr.open_dataset(nc_path, decode_timedelta=False).load()
    dv = set(ds.data_vars)
    if "obs" not in dv:
        return None

    models = {
        name_map.get(f[len("models_"):], f[len("models_"):]): ds[f]
        for f in dv if f.startswith("models_")
    }

    labels = json.loads(ds.attrs.get("ts_benchmark_labels", "[]"))
    colors = json.loads(ds.attrs.get("ts_benchmark_colors", "[]"))
    counts = json.loads(ds.attrs.get("ts_benchmark_n_members", "[]"))
    indices = sorted({
        int(f.split("_")[2]) for f in dv
        if f.startswith("benchmarks_ts_") and f.endswith("_ts")
    })
    benchmarks_ts: list[dict] = []
    for i in indices:
        ts = ds.get(f"benchmarks_ts_{i}_ts")
        if ts is None:
            continue
        if i < len(labels) and labels[i]:
            label = labels[i]
        elif i < len(benchmarks):
            label = getattr(benchmarks[i], "label", f"Benchmark {i}")
        else:
            label = f"Benchmark {i}"
        color = (colors[i] if i < len(colors) and colors[i] else None) \
            or benchmark_color(i)
        n_members = int(counts[i]) if i < len(counts) else 0
        benchmarks_ts.append({
            "label": label,
            "color": color,
            "ts": ts,
            "info": {"n_members": n_members},
            "env_min": ds.get(f"benchmarks_ts_{i}_env_min"),
            "env_max": ds.get(f"benchmarks_ts_{i}_env_max"),
            "individual": {},
        })

    primary = benchmarks_ts[0] if benchmarks_ts else None
    return {
        "models": models,
        "obs": ds["obs"],
        "era5_ts": ds.get("era5_ts"),
        "benchmarks_ts": benchmarks_ts,
        "ens_mean": ds.get("ens_mean"),
        "ens_median": ds.get("ens_median"),
        "cmip6_ts": primary["ts"] if primary else ds.get("cmip6_ts"),
        "cmip6_info": dict(primary["info"]) if primary else {},
        "cmip6_individual_ts": {},
    }


def _to_plot_time(time_values: np.ndarray) -> np.ndarray:
    """Convert (possibly cftime) time values to a matplotlib-friendly array."""
    if len(time_values) == 0:
        return time_values
    if hasattr(time_values[0], "strftime"):
        return pd.to_datetime([str(t) for t in time_values])
    return time_values


def build_envelope_timeseries(
    *,
    anomaly: bool,
    models: dict,
    model_color,
    obs,
    obs_label: str,
    long_name: str,
    units: str,
    benchmarks: list[dict] | None = None,
    ens_mean=None,
    ens_median=None,
    ens_prefix: str = "Ensemble",
    extra_obs: list[tuple] | None = None,
    offset: float = 0.0,
    factor: float = 1.0,
) -> plt.Figure:
    """Render an envelope (or anomaly) global-mean time-series figure.

    Parameters
    ----------
    anomaly : bool
        When True, every series is shown relative to its own full-period mean
        and the additive *offset* is dropped.
    models : dict
        Mapping of model name → monthly DataArray (raw units).
    model_color : callable
        ``model_color(name) -> color``.
    obs : DataArray
        Primary observation monthly series (raw units).
    obs_label : str
        Legend label for the primary observation line.
    long_name, units : str
        Variable long name and display units (for title/ylabel).
    benchmarks : list of dict
        Each with ``label``, ``color``, ``ts`` (MMM), ``info`` (``n_members``)
        and optional ``env_min``/``env_max`` for the shaded band.
    ens_mean, ens_median : DataArray, optional
        Evaluated-ensemble mean/median series (omitted when None).
    extra_obs : list of (DataArray, label)
        Auxiliary reference series rendered as dashed black lines (e.g. ERA5).
    offset, factor : float
        Display transform ``value * factor + offset``.
    """
    benchmarks = benchmarks or []
    extra_obs = extra_obs or []
    n_eval = len(models)

    def xf(ts):
        if anomaly:
            return (ts - ts.mean("time")) * factor
        return ts * factor + offset

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Monthly pass (background, washed-out) ---
    for bench in benchmarks:
        c = bench["color"]
        emin, emax = bench.get("env_min"), bench.get("env_max")
        if emin is not None and emax is not None:
            a, b = xf(emin), xf(emax)
            t = _to_plot_time(a.time.values)
            ax.fill_between(t, a.values, b.values, color=c, alpha=0.12, lw=0)
        bt = xf(bench["ts"])
        t = _to_plot_time(bt.time.values)
        ax.plot(t, bt.values, color=c, alpha=0.3, lw=0.7, ls="--")

    for model, ts in models.items():
        v = xf(ts)
        t = _to_plot_time(v.time.values)
        ax.plot(t, v.values, color=model_color(model), alpha=0.3, lw=0.7)

    ov = xf(obs)
    t = _to_plot_time(ov.time.values)
    ax.plot(t, ov.values, color=OBS_COLOR, alpha=0.3, lw=0.7)

    for eo_ts, _ in extra_obs:
        ev = xf(eo_ts)
        t = _to_plot_time(ev.time.values)
        ax.plot(t, ev.values, color=_ERA5_COLOR, alpha=0.25, lw=0.7, ls="--")

    # --- Annual pass (foreground, thick with labels) ---
    for bench in benchmarks:
        c, lbl = bench["color"], bench["label"]
        n = bench["info"].get("n_members", 0)
        emin, emax = bench.get("env_min"), bench.get("env_max")
        if emin is not None and emax is not None:
            amin, amax = annual_mean(xf(emin)), annual_mean(xf(emax))
            band = lbl[:-4] if lbl.endswith(" MMM") else lbl
            t = _to_plot_time(amin.time.values)
            ax.fill_between(t, amin.values, amax.values, color=c, alpha=0.25,
                            lw=0, label=f"{band} min–max ({n})")
        ba = annual_mean(xf(bench["ts"]))
        t = _to_plot_time(ba.time.values)
        ax.plot(t, ba.values, color=c, lw=2.0, ls="--", label=f"{lbl} ({n})")

    for model, ts in models.items():
        a = annual_mean(xf(ts))
        t = _to_plot_time(a.time.values)
        ax.plot(t, a.values, color=model_color(model), lw=2.0, label=model)

    if ens_median is not None:
        a = annual_mean(xf(ens_median))
        t = _to_plot_time(a.time.values)
        ax.plot(t, a.values, color=ENS_COLOR, lw=2.5, ls="--",
                label=f"{ens_prefix} ensemble median ({n_eval})")
    if ens_mean is not None:
        a = annual_mean(xf(ens_mean))
        t = _to_plot_time(a.time.values)
        ax.plot(t, a.values, color=ENS_COLOR, lw=2.5,
                label=f"{ens_prefix} ensemble mean ({n_eval})")

    oa = annual_mean(xf(obs))
    t = _to_plot_time(oa.time.values)
    ax.plot(t, oa.values, color=OBS_COLOR, lw=2.5, label=obs_label)

    for eo_ts, eo_label in extra_obs:
        ea = annual_mean(xf(eo_ts))
        t = _to_plot_time(ea.time.values)
        ax.plot(t, ea.values, color=_ERA5_COLOR, lw=2.0, ls="--",
                label=eo_label)

    if anomaly:
        ax.axhline(0.0, color="0.5", lw=0.8, alpha=0.6)

    kind = " Anomaly" if anomaly else ""
    ulabel = f"Δ {units}" if anomaly else units
    ax.set_title(f"{long_name}{kind} — Global Mean")
    ax.set_ylabel(f"{long_name} ({ulabel})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
