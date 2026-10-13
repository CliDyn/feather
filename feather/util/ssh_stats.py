"""Per-cell sea-surface-height statistics: mean and three standard deviations.

Used by :class:`feather.diag.ssh_variability.SSHVariabilityDiag`.  Every
function works on a ``(cells, time)`` block whose time axis is complete
(missing steps are NaN), so calendar slots and running windows line up.

Standard deviations (population, ``ddof=0`` as in ``xarray.std``):

``std_raw``   of the series as it is: seasonal cycle, trend, interannual and
              mesoscale variability together.
``std_anom``  after removing the seasonal climatology (mean per calendar slot,
              smoothed with a periodic running mean of ``smooth_width`` slots)
              and a least-squares linear trend.
``std_hp``    of the anomaly minus its centred ``hp_window``-step running mean,
              i.e. periods shorter than about ``hp_window`` steps (mesoscale
              eddies for daily data).

A running-mean step is defined where at least half of its window is valid;
cells with fewer than ``min_valid`` (fraction) valid steps are NaN throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

#: Bytes of temporaries per cell and time step (float32/float64 mix).
BYTES_PER_CELL_STEP: int = 40

STATS: tuple[str, ...] = ("valid_frac", "mean", "std_raw", "std_anom", "std_hp")


def seasonal_climatology(x: np.ndarray, slots: np.ndarray, n_slots: int, *,
                         feb29: int | None = None,
                         smooth_width: int = 0) -> np.ndarray:
    """Mean of each calendar slot, ``(cells, n_slots)`` float32.

    Sums and counts are smoothed separately, so a slot with gaps is a
    count-weighted mean of its neighbours.  *feb29* (Gregorian) takes the
    mean of its two neighbours before smoothing.
    """
    fin = np.isfinite(x)
    xs = np.where(fin, x, 0.0)
    ncell = x.shape[0]
    sums = np.zeros((ncell, n_slots), dtype=np.float64)
    cnts = np.zeros((ncell, n_slots), dtype=np.float64)
    for d in range(n_slots):
        cols = np.flatnonzero(slots == d)
        if len(cols):
            sums[:, d] = xs[:, cols].sum(axis=1, dtype=np.float64)
            cnts[:, d] = fin[:, cols].sum(axis=1)
    if feb29 is not None and 0 < feb29 < n_slots - 1:
        for a in (sums, cnts):
            a[:, feb29] = 0.5 * (a[:, feb29 - 1] + a[:, feb29 + 1])
    if smooth_width and smooth_width > 1:
        sums = uniform_filter1d(sums, size=smooth_width, axis=1, mode="wrap")
        cnts = uniform_filter1d(cnts, size=smooth_width, axis=1, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts > 0, sums / cnts, np.nan).astype(np.float32)


def detrend(a: np.ndarray) -> np.ndarray:
    """*a* minus its least-squares line in time (NaN skipped), per cell."""
    fin = np.isfinite(a)
    t = np.arange(a.shape[1], dtype=np.float64)
    n = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = (fin * t).sum(axis=1) / n
        am = np.where(fin, a, 0.0).sum(axis=1, dtype=np.float64) / n
        dt = np.where(fin, t[None, :] - tm[:, None], 0.0)
        slope = (dt * np.where(fin, a - am[:, None], 0.0)).sum(axis=1) \
            / (dt * dt).sum(axis=1)
    slope = np.where(np.isfinite(slope), slope, 0.0)
    fit = am[:, None] + slope[:, None] * (t[None, :] - tm[:, None])
    return (a - fit).astype(np.float32)


def running_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean over ``window`` steps, NaN-aware.

    A step is NaN where fewer than half of its window is valid.  The window
    is truncated at the ends of the record.
    """
    fin = np.isfinite(a)
    zero = np.zeros((a.shape[0], 1))
    cs = np.concatenate([zero, np.cumsum(np.where(fin, a, 0.0), axis=1,
                                         dtype=np.float64)], axis=1)
    cc = np.concatenate([zero, np.cumsum(fin, axis=1, dtype=np.float64)], axis=1)
    nt = a.shape[1]
    half = window // 2
    lo = np.clip(np.arange(nt) - half, 0, nt)
    hi = np.clip(np.arange(nt) + window - half, 0, nt)
    s = cs[:, hi] - cs[:, lo]
    c = cc[:, hi] - cc[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= 0.5 * window, s / c, np.nan).astype(np.float32)


def _std(a: np.ndarray) -> np.ndarray:
    fin = np.isfinite(a)
    n = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(fin, a, 0.0).sum(axis=1, dtype=np.float64) / n
        v = (np.where(fin, a - m[:, None], 0.0) ** 2).sum(axis=1) / n
    return np.sqrt(v)


def cell_stats(x: np.ndarray, slots: np.ndarray, n_slots: int, *,
               feb29: int | None = None, smooth_width: int = 0,
               hp_window: int = 0, min_valid: float = 0.5
               ) -> dict[str, np.ndarray]:
    """Statistics of every row of *x* ``(cells, time)``, as float32 arrays.

    ``hp_window = 0`` leaves ``std_hp`` out (monthly data).
    """
    x = np.asarray(x, dtype=np.float32)
    nt = x.shape[1]
    fin = np.isfinite(x)
    frac = fin.sum(axis=1) / max(nt, 1)
    out: dict[str, np.ndarray] = {"valid_frac": frac.astype(np.float32)}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mean"] = (np.where(fin, x, 0.0).sum(axis=1, dtype=np.float64)
                       / fin.sum(axis=1)).astype(np.float32)
    out["std_raw"] = _std(x).astype(np.float32)
    clim = seasonal_climatology(x, slots, n_slots, feb29=feb29,
                                smooth_width=smooth_width)
    anom = detrend(x - clim[:, slots])
    out["std_anom"] = _std(anom).astype(np.float32)
    if hp_window and hp_window > 1:
        out["std_hp"] = _std(anom - running_mean(anom, hp_window)).astype(np.float32)
    bad = frac < min_valid
    for k, v in out.items():
        if k != "valid_frac":
            v[bad] = np.nan
    return out
