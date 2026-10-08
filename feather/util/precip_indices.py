"""ETCCDI daily-precipitation extreme indices.

Definitions follow the ETCCDI set as distributed in the C3S dataset
*Climate extreme indices and heat stress indicators derived from CMIP6
global climate projections* (``sis-extreme-indices-cmip6``).  ``RR`` is the
daily precipitation amount in mm (1 kg m⁻² = 1 mm of water), and a *wet day*
is a day with ``RR ≥ 1 mm``.

=========  ==============================================================  ======
Index      Definition (per year)                                           Units
=========  ==============================================================  ======
R1mm       Number of wet days, ``RR ≥ 1 mm``                               days
R10mm      Number of heavy-precipitation days, ``RR ≥ 10 mm``              days
R20mm      Number of very-heavy-precipitation days, ``RR ≥ 20 mm``         days
PRCPTOT    Total precipitation on wet days                                 mm
SDII       Simple daily intensity index: PRCPTOT / R1mm                    mm/day
R95p       Total precipitation on wet days with ``RR > RR95``             mm
R99p       Total precipitation on wet days with ``RR > RR99``             mm
Rx1day     Maximum 1-day precipitation                                     mm
Rx5day     Maximum consecutive 5-day precipitation                         mm
=========  ==============================================================  ======

Seasonal indices (DJF, MAM, JJA, SON) apply the same definitions to the days
of one season; DJF of year *Y* is December *Y−1* plus January–February *Y*.
R95p/R99p keep the annual base-period thresholds, and an Rx5day window
belongs to the season of its last day (the four days before the season
start complete the first windows), exactly as for the annual index.

``RR95``/``RR99`` are the 95th/99th percentiles of wet-day precipitation over
a base period, per grid cell (numpy's default linear interpolation between
order statistics).  ETCCDI uses 1961–1990; here the base period is whatever
the caller streams through :class:`WetDayPercentiles` (feather uses the
climate-change reference period, because most EERIE members start in 1975).

Everything here is plain numpy on ``(time, cell)`` arrays — no xarray, no
dask — so the diagnostic can stream one year at a time and keep the memory
bounded on 0.25° grids (1,038,240 cells).
"""

from __future__ import annotations

import numpy as np

#: Wet-day threshold (mm/day).
WET_DAY_MM: float = 1.0

#: kg m⁻² s⁻¹ → mm/day.
KG_M2_S_TO_MM_DAY: float = 86400.0

#: Index order used throughout feather.
INDICES: tuple[str, ...] = (
    "r1mm", "r10mm", "r20mm", "prcptot", "sdii",
    "r95p", "r99p", "rx1day", "rx5day",
)

#: ``{index: (ETCCDI label, long name, units)}``.
INDEX_INFO: dict[str, tuple[str, str, str]] = {
    "r1mm":    ("R1mm",    "Number of wet days (RR ≥ 1 mm)",                     "days"),
    "r10mm":   ("R10mm",   "Heavy precipitation days (RR ≥ 10 mm)",              "days"),
    "r20mm":   ("R20mm",   "Very heavy precipitation days (RR ≥ 20 mm)",         "days"),
    "prcptot": ("PRCPTOT", "Total wet-day precipitation",                        "mm"),
    "sdii":    ("SDII",    "Simple daily intensity index",                       "mm/day"),
    "r95p":    ("R95p",    "Very wet day precipitation (RR > 95th percentile)",  "mm"),
    "r99p":    ("R99p",    "Extremely wet day precipitation (RR > 99th percentile)", "mm"),
    "rx1day":  ("Rx1day",  "Maximum 1-day precipitation",                        "mm"),
    "rx5day":  ("Rx5day",  "Maximum consecutive 5-day precipitation",            "mm"),
}

#: Minimum number of valid days for a cell-year to count (climdex drops a
#: year with more than 15 missing days; 350 also admits 360-day calendars).
MIN_VALID_DAYS: int = 350

#: Season → calendar months, in plotting order.
SEASONS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
}

#: Nominal (365-day) season lengths; the valid-day minimum scales with them.
_SEASON_DAYS: dict[str, int] = {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91}

#: Days of the previous year to carry: all of December plus the four days
#: before it (the first DJF Rx5day windows).
CARRY_DAYS: int = 35


def season_min_valid_days(season: str) -> int:
    """Valid-day minimum for a cell-season, scaled like the annual 350/365."""
    return int(np.floor(_SEASON_DAYS[season] * MIN_VALID_DAYS / 365.0))


def season_blocks(
    rr: np.ndarray,
    months: np.ndarray,
    seasons,
    prev: np.ndarray | None = None,
    prev_months: np.ndarray | None = None,
):
    """Split one year of daily data into season blocks.

    Parameters
    ----------
    rr : ndarray, shape (n_days, ...)
        Daily precipitation of year *Y*.
    months : ndarray, shape (n_days,)
        Calendar month of every day in *rr*.
    seasons : iterable of str
        Keys of :data:`SEASONS`.
    prev, prev_months : ndarray, optional
        The last :data:`CARRY_DAYS` days of year *Y−1* and their months.
        Without them DJF has no December and comes out missing.

    Yields
    ------
    (season, block, tail)
        *block* holds the season's days, *tail* the (up to) four days before
        the season start for Rx5day, or ``None``.
    """
    months = np.asarray(months)
    for season in seasons:
        if season == "DJF":
            jf = np.flatnonzero(np.isin(months, (1, 2)))
            if prev is not None and prev_months is not None:
                dec = np.flatnonzero(np.asarray(prev_months) == 12)
            else:
                dec = np.array([], dtype=int)
            if len(dec):
                block = np.concatenate([prev[dec], rr[jf]])
                tail = prev[: dec[0]][-4:] if dec[0] > 0 else None
            else:
                block, tail = rr[jf], None
        else:
            idx = np.flatnonzero(np.isin(months, SEASONS[season]))
            block = rr[idx]
            tail = rr[max(idx[0] - 4, 0): idx[0]] if len(idx) and idx[0] > 0 else None
        yield season, block, tail


def annual_indices(
    rr: np.ndarray,
    *,
    p95: np.ndarray | None = None,
    p99: np.ndarray | None = None,
    prev_tail: np.ndarray | None = None,
    min_valid_days: int = MIN_VALID_DAYS,
) -> dict[str, np.ndarray]:
    """All nine indices for one year of daily precipitation.

    Parameters
    ----------
    rr : ndarray, shape (n_days, ...)
        Daily precipitation in **mm/day** for a single year.
    p95, p99 : ndarray, shape (...), optional
        Base-period wet-day percentiles.  Without them R95p/R99p are NaN.
    prev_tail : ndarray, shape (k ≤ 4, ...), optional
        The last days of the previous year, so 5-day windows that start in
        December count towards the year of their last day.
    min_valid_days : int
        Cells with fewer finite days are set to NaN for every index.

    Returns
    -------
    dict
        ``{index: ndarray(...)}`` in :data:`INDICES` order, float32.
    """
    rr = np.asarray(rr, dtype=np.float32)
    valid = np.isfinite(rr)
    n_valid = valid.sum(axis=0)
    x = np.where(valid, rr, np.float32(0.0))

    wet = x >= WET_DAY_MM
    n_wet = wet.sum(axis=0)
    prcptot = np.where(wet, x, 0.0).sum(axis=0, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        sdii = np.where(n_wet > 0, prcptot / np.maximum(n_wet, 1), np.nan)

    out: dict[str, np.ndarray] = {
        "r1mm": n_wet,
        "r10mm": (x >= 10.0).sum(axis=0),
        "r20mm": (x >= 20.0).sum(axis=0),
        "prcptot": prcptot,
        "sdii": sdii,
        "r95p": _sum_above(x, wet, p95),
        "r99p": _sum_above(x, wet, p99),
        "rx1day": np.where(valid, rr, -np.inf).max(axis=0),
        "rx5day": _rx5day(rr, prev_tail),
    }

    bad = n_valid < min_valid_days
    return {
        k: np.where(bad, np.nan, out[k]).astype(np.float32) for k in INDICES
    }


def _sum_above(x: np.ndarray, wet: np.ndarray, thr: np.ndarray | None) -> np.ndarray:
    """Sum of wet-day amounts strictly above the per-cell threshold *thr*."""
    if thr is None:
        return np.full(x.shape[1:], np.nan)
    thr = np.asarray(thr, dtype=np.float32)
    above = wet & (x > thr)
    total = np.where(above, x, 0.0).sum(axis=0, dtype=np.float64)
    # No base-period wet days → threshold undefined → index undefined.
    return np.where(np.isfinite(thr), total, np.nan)


def _rx5day(rr: np.ndarray, prev_tail: np.ndarray | None) -> np.ndarray:
    """Maximum 5-day running total, windows assigned to their last day.

    A window containing a missing day is skipped.  The first four days of a
    year are completed from *prev_tail* when given.
    """
    if prev_tail is not None and len(prev_tail):
        series = np.concatenate([np.asarray(prev_tail, np.float32)[-4:], rr])
    else:
        series = rr
    if series.shape[0] < 5:
        return np.full(rr.shape[1:], np.nan, dtype=np.float32)
    csum = np.cumsum(np.where(np.isfinite(series), series, 0.0),
                     axis=0, dtype=np.float64)
    csum = np.concatenate([np.zeros((1,) + series.shape[1:]), csum])
    window = csum[5:] - csum[:-5]
    bad = np.cumsum(~np.isfinite(series), axis=0)
    bad = np.concatenate([np.zeros((1,) + series.shape[1:], dtype=bad.dtype), bad])
    has_gap = (bad[5:] - bad[:-5]) > 0
    window = np.where(has_gap, -np.inf, window)
    best = window.max(axis=0)
    return np.where(np.isfinite(best), best, np.nan)


class WetDayPercentiles:
    """Streaming, exact per-cell percentiles of wet-day precipitation.

    The base period is fed one year at a time with :meth:`update`.  Only the
    largest ``k`` wet-day amounts per cell are kept, where ``k`` covers every
    order statistic the requested percentiles can need, so the result equals
    ``np.percentile`` over all wet days (linear interpolation) while memory
    stays at ``k × n_cells`` instead of ``n_days × n_cells``.

    Parameters
    ----------
    n_cells : int
        Number of grid cells (the flattened spatial shape).
    max_days : int
        Upper bound on the number of base-period days (e.g. 20 × 366).
    quantiles : sequence of float
        Percentiles to provide, as fractions (default 0.95 and 0.99).
    chunk : int
        Cells processed per partition step, bounding temporary memory.
    """

    def __init__(self, n_cells: int, max_days: int,
                 quantiles=(0.95, 0.99), chunk: int = 131_072):
        self.quantiles = tuple(float(q) for q in quantiles)
        q_min = min(self.quantiles)
        self.k = int(np.ceil((1.0 - q_min) * max(max_days - 1, 0))) + 2
        self.max_days = int(max_days)
        self.n_cells = int(n_cells)
        self.chunk = int(chunk)
        self._top = np.full((self.k, self.n_cells), -np.inf, dtype=np.float32)
        self._count = np.zeros(self.n_cells, dtype=np.int64)
        self._days = 0

    def update(self, rr: np.ndarray) -> None:
        """Add one block of daily precipitation, shape ``(n_days, n_cells)``."""
        rr = np.asarray(rr, dtype=np.float32).reshape(rr.shape[0], -1)
        if rr.shape[1] != self.n_cells:
            raise ValueError(
                f"expected {self.n_cells} cells, got {rr.shape[1]}")
        self._days += rr.shape[0]
        if self._days > self.max_days:
            raise ValueError(
                f"fed {self._days} days, more than max_days={self.max_days}; "
                "the percentile would no longer be exact")
        wet = np.isfinite(rr) & (rr >= WET_DAY_MM)
        self._count += wet.sum(axis=0)
        vals = np.where(wet, rr, -np.inf)
        k = self.k
        for s in range(0, self.n_cells, self.chunk):
            e = min(s + self.chunk, self.n_cells)
            both = np.concatenate([self._top[:, s:e], vals[:, s:e]], axis=0)
            if both.shape[0] > k:
                both = -np.partition(-both, k - 1, axis=0)[:k]
            self._top[:, s:e] = both

    def result(self) -> dict[float, np.ndarray]:
        """``{q: ndarray(n_cells)}``; NaN where a cell had no wet day."""
        desc = -np.sort(-self._top, axis=0)            # largest first
        n = self._count
        out: dict[float, np.ndarray] = {}
        cells = np.arange(self.n_cells)
        for q in self.quantiles:
            pos = q * np.maximum(n - 1, 0)              # ascending position
            lo = np.floor(pos).astype(np.int64)
            frac = (pos - lo).astype(np.float32)
            # ascending index i ↔ descending index n-1-i
            i_lo = np.clip(n - 1 - lo, 0, self.k - 1)
            i_hi = np.clip(n - 2 - lo, 0, self.k - 1)
            v_lo = desc[i_lo, cells]
            v_hi = np.where(n - 2 - lo >= 0, desc[i_hi, cells], v_lo)
            with np.errstate(invalid="ignore"):   # -inf - -inf when n < 2
                val = v_lo + frac * (v_hi - v_lo)
            out[q] = np.where(n > 0, val, np.nan).astype(np.float32)
        return out
