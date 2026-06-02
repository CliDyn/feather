"""Global Warming Level (GWL) window computation.

Given a GCM's global-mean near-surface temperature, find the 30-year window
in which the model reaches a target warming level (e.g. +2 °C) relative to a
pre-industrial baseline (default 1850–1900).  This is the standard
CORDEX-CORE / IPCC "time-sampling" approach: the warming level is defined per
model from its own global-mean temperature, and regional models inherit the
window of their driving GCM.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def global_mean_annual_tas(tas: xr.DataArray) -> xr.DataArray:
    """Area-weighted global-mean annual temperature, indexed by integer year.

    Parameters
    ----------
    tas : xr.DataArray
        Temperature with ``time`` and regular ``lat``/``lon`` dims (any units;
        units are preserved so the baseline difference is unit-consistent).

    Returns
    -------
    xr.DataArray
        Annual global mean with an integer ``year`` coordinate.
    """
    lat_name = "lat" if "lat" in tas.coords else "latitude"
    lon_name = "lon" if "lon" in tas.coords else "longitude"
    weights = np.cos(np.deg2rad(tas[lat_name]))
    weights.name = "weights"
    gm = tas.weighted(weights).mean((lat_name, lon_name))
    annual = gm.groupby("time.year").mean("time")
    return annual.compute()


def warming_level_window(
    annual_tas: xr.DataArray,
    level: float = 2.0,
    baseline: tuple[int, int] = (1850, 1900),
    window: int = 30,
) -> tuple[int, int] | None:
    """Find the *window*-year period reaching *level* warming above *baseline*.

    A centred running mean of the warming (annual − baseline mean) is scanned
    for the first year whose window-mean reaches ``level``; the returned period
    is that window.

    Parameters
    ----------
    annual_tas : xr.DataArray
        Annual global-mean temperature with a ``year`` coordinate.
    level : float
        Target warming level (°C/K above baseline).
    baseline : (int, int)
        Inclusive baseline year range (pre-industrial; default 1850–1900).
    window : int
        Window length in years (default 30).

    Returns
    -------
    (start_year, end_year) inclusive, or None if the level is never reached.
    """
    base = annual_tas.sel(year=slice(baseline[0], baseline[1])).mean()
    warming = annual_tas - base
    roll = warming.rolling(year=window, center=True, min_periods=window).mean()

    years = np.asarray(roll["year"].values)
    vals = np.asarray(roll.values)
    hit = np.where(np.isfinite(vals) & (vals >= level))[0]
    if hit.size == 0:
        return None
    center = int(years[hit[0]])
    half = window // 2
    start = center - half
    end = start + window - 1
    return start, end
