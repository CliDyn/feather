"""Marine heatwave detection (Hobday et al., 2016) vectorised over many cells.

A marine heatwave (MHW) is a run of at least ``min_duration`` days on which
the temperature exceeds a seasonally varying threshold, the ``pctile``-th
percentile of the base-period values within ``±window_half_width`` days of
the calendar day.  Events separated by at most ``max_gap`` days are joined.

The algorithm and its defaults follow Oliver's ``marineHeatWaves.detect``,
which de Toma et al. (2026, EGUsphere-2026-5747; github.com/vincenzodetoma/ROHAN)
use for their GREP comparison, with the paper's settings: 90th percentile,
an 11-day window (±5 days), the climatology and threshold smoothed with a
31-day running mean, no detrending, events ≥ 5 days, gaps ≤ 2 days joined.
Differences to the reference implementation:

* All cells of a block are processed at once with numpy (no per-cell Python
  loop); events are located with run-length arithmetic and their metrics
  reduced with ``ufunc.reduceat`` over the flattened (cell, day) array.
* Percentiles use linear interpolation between order statistics, as
  ``np.nanpercentile`` does, computed by sorting so NaN days are skipped
  without ``nanpercentile``'s per-row Python loop.
* Calendars: 366 calendar-day slots for Gregorian calendars, with 29 February
  filled with the mean of its neighbours as Oliver does; ``noleap``/``365_day``
  use 365 slots, ``360_day`` 360 and ``all_leap``/``366_day`` 366 (no fill).
* A day with a missing temperature never exceeds the threshold and counts
  as the climatology inside an event (Oliver fills it the same way).
* At the very start/end of the record the onset/decline rate is divided by
  at least one day (Oliver's guard misses the case of a peak on an event's
  last day when the event ends the record, which divides by zero).

Event metrics (Oliver's names; intensities relative to the seasonal
climatology, °C, rates °C/day):

``duration``                days, ``end − start + 1``
``intensity_max``           peak (temperature − climatology)
``intensity_mean``          mean over the event
``intensity_var``           standard deviation over the event
``intensity_cumulative``    sum over the event (°C days)
``rate_onset``              from the half-day before the start to the peak
``rate_decline``            from the peak to the half-day after the end
``category``                1 Moderate … 4 Extreme: ⌊1 + max (T−thresh)/(thresh−clim)⌋
                            over the event, capped at 4

:func:`cell_metrics` reduces the events of every cell to the maps used by
the paper (mean duration, maximum intensity, mean onset/decline rate,
maximum category, mean cumulative intensity, mean intensity variability,
percentage of MHW days) plus event frequency and annual MHW-day counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

#: Approximate peak memory of :func:`detect` per cell-day (bytes), for sizing
#: blocks: the input, expanded climatology and threshold, exceedance and run
#: masks, the event-day arrays and the MHW-day mask.
BYTES_PER_CELL_DAY: int = 40

#: Defaults of Hobday et al. (2016) / de Toma et al. (2026).
PCTILE: float = 90.0
WINDOW_HALF_WIDTH: int = 5
SMOOTH_WIDTH: int = 31
MIN_DURATION: int = 5
MAX_GAP: int = 2

#: Per-cell output maps: ``name -> (label, long name, units)``.
METRIC_INFO: dict[str, tuple[str, str, str]] = {
    "frequency": ("Frequency", "MHW events per year", "events/yr"),
    "mhw_days_pct": ("MHW days", "Percentage of days in a MHW", "%"),
    "duration_mean": ("Mean duration", "Mean MHW duration", "days"),
    "intensity_max_max": ("Max intensity", "Maximum intensity over all events",
                          "°C"),
    "intensity_max_mean": ("Mean peak intensity",
                           "Mean of the event peak intensity", "°C"),
    "intensity_mean_mean": ("Mean intensity", "Mean MHW intensity", "°C"),
    "intensity_cumulative_mean": ("Cumulative intensity",
                                  "Mean cumulative intensity per event",
                                  "°C days"),
    "intensity_var_mean": ("Intensity variability",
                           "Mean within-event intensity standard deviation",
                           "°C"),
    "rate_onset_mean": ("Onset rate", "Mean onset rate", "°C/day"),
    "rate_decline_mean": ("Decline rate", "Mean decline rate", "°C/day"),
    "category_max": ("Max category",
                     "Maximum category reached (1 moderate … 4 extreme)", ""),
}

METRICS: tuple[str, ...] = tuple(METRIC_INFO)


# ── Calendar handling ────────────────────────────────────────────────────

def calendar_slots(years: np.ndarray, months: np.ndarray, days: np.ndarray,
                   calendar: str = "standard") -> tuple[np.ndarray, int, int | None]:
    """Calendar-day slot (0-based) of every day, slot count and Feb-29 slot.

    Gregorian calendars map every year onto a 366-day template, so 1 March is
    slot 60 in leap and common years alike and 29 February (slot 59) only
    occurs in leap years; its climatology is later filled from its
    neighbours.  ``noleap`` uses 365 slots, ``360_day`` 360 and
    ``all_leap``/``366_day`` 366 without filling.
    """
    cal = (calendar or "standard").lower()
    months = np.asarray(months)
    days = np.asarray(days)
    if cal in ("360_day",):
        return ((months - 1) * 30 + days - 1).astype(np.int32), 360, None
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    if cal in ("noleap", "365_day"):
        return (cum[months - 1] + days - 1).astype(np.int32), 365, None
    cum_leap = cum + (np.arange(12) >= 2)
    slot = (cum_leap[months - 1] + days - 1).astype(np.int32)
    if cal in ("all_leap", "366_day"):
        return slot, 366, None
    return slot, 366, 59


# ── Percentiles ignoring NaN ─────────────────────────────────────────────

def _nan_percentile_rows(a: np.ndarray, q: float) -> np.ndarray:
    """``np.nanpercentile(a, q, axis=1)`` (linear) without its per-row loop.

    Rows without NaN (nearly all ocean cells) take a partition-based path;
    the others are sorted so their NaN go last and are skipped.
    """
    full = np.isfinite(a).all(axis=1)
    out = np.empty(a.shape[0], dtype=np.float64)
    if full.any():
        out[full] = np.percentile(a[full], q, axis=1)
    if (~full).any():
        out[~full] = _sorted_percentile_rows(a[~full], q)
    return out


def _sorted_percentile_rows(a: np.ndarray, q: float) -> np.ndarray:
    s = np.sort(a, axis=1)                       # NaN sort last
    n = np.isfinite(s).sum(axis=1)
    pos = (n - 1).clip(min=0) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, (n - 1).clip(min=0))
    frac = (pos - lo).astype(s.dtype)
    rows = np.arange(len(s))
    out = s[rows, lo] * (1 - frac) + s[rows, hi] * frac
    # When frac is 0 the upper neighbour may be NaN (hi == n); 0·NaN is NaN.
    out = np.where(frac == 0, s[rows, lo], out)
    out[n == 0] = np.nan
    return out


def climatology(temp: np.ndarray, slots: np.ndarray, n_slots: int,
                base: np.ndarray, *, feb29: int | None = None,
                pctile: float = PCTILE,
                window_half_width: int = WINDOW_HALF_WIDTH,
                smooth_width: int = SMOOTH_WIDTH,
                min_base_fraction: float = 0.5,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal climatology and threshold per cell and calendar slot.

    Parameters
    ----------
    temp : (cells, days) float array
    slots : (days,) calendar slot of every day (:func:`calendar_slots`)
    n_slots : number of calendar slots
    base : (days,) bool, True for days of the base period
    feb29 : slot filled with the mean of its neighbours (Gregorian), or None
    min_base_fraction : a cell needs this fraction of valid base-period days,
        otherwise its climatology and threshold are NaN

    Returns
    -------
    (seas, thresh) : (cells, n_slots) float32 arrays
    """
    temp = np.asarray(temp)
    ncell = temp.shape[0]
    nt = temp.shape[1]
    tb = np.flatnonzero(base)
    seas = np.full((ncell, n_slots), np.nan, dtype=np.float32)
    thresh = np.full((ncell, n_slots), np.nan, dtype=np.float32)
    offsets = np.arange(-window_half_width, window_half_width + 1)
    for d in range(n_slots):
        if d == feb29:
            continue
        # Base-period days of this slot, ±window days of the whole record
        # (as Oliver's detect: the window may reach just outside the base).
        t0 = tb[slots[tb] == d]
        if not len(t0):
            continue
        idx = (t0[:, None] + offsets[None, :]).ravel()
        idx = idx[(idx >= 0) & (idx < nt)]
        win = temp[:, idx]
        with np.errstate(invalid="ignore"):
            thresh[:, d] = _nan_percentile_rows(win, pctile)
            cnt = np.isfinite(win).sum(axis=1)
            seas[:, d] = np.where(cnt > 0, np.nansum(win, axis=1)
                                  / np.maximum(cnt, 1), np.nan)
    if feb29 is not None:
        for arr in (seas, thresh):
            arr[:, feb29] = 0.5 * (arr[:, feb29 - 1] + arr[:, feb29 + 1])
    if smooth_width and smooth_width > 1:
        seas = _smooth(seas, smooth_width)
        thresh = _smooth(thresh, smooth_width)
    valid = np.isfinite(temp[:, tb]).sum(axis=1) >= min_base_fraction * len(tb)
    seas[~valid] = np.nan
    thresh[~valid] = np.nan
    return seas, thresh


def _smooth(a: np.ndarray, width: int) -> np.ndarray:
    """Periodic running mean over the slot axis (Oliver's ``runavg``)."""
    out = uniform_filter1d(a.astype(np.float64), size=width, axis=1, mode="wrap")
    return out.astype(np.float32)


# ── Event detection ──────────────────────────────────────────────────────

@dataclass
class Events:
    """All events of a block of cells, sorted by cell then start day."""

    cell: np.ndarray
    start: np.ndarray
    end: np.ndarray                      # inclusive
    metrics: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cell)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cell, start, end)`` of every run of True along axis 1."""
    ncell, nt = mask.shape
    pad = np.zeros((ncell, nt + 2), dtype=np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    c0, s = np.nonzero(d == 1)
    _, e = np.nonzero(d == -1)
    return c0, s, e - 1


def detect_events(temp: np.ndarray, seas_d: np.ndarray, thresh_d: np.ndarray,
                  *, min_duration: int = MIN_DURATION,
                  max_gap: int = MAX_GAP) -> Events:
    """Find MHWs and their metrics in every cell.

    *seas_d*/*thresh_d* are the climatology and threshold expanded to the
    days of *temp* (all ``(cells, days)``).
    """
    temp = np.asarray(temp, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        exceed = temp > thresh_d
    cell, start, end = _runs(exceed)
    keep = (end - start + 1) >= min_duration
    cell, start, end = cell[keep], start[keep], end[keep]
    if max_gap and len(cell) > 1:
        join = ((cell[1:] == cell[:-1])
                & (start[1:] - end[:-1] - 1 <= max_gap))
        first = np.flatnonzero(np.concatenate([[True], ~join]))
        last = np.concatenate([first[1:] - 1, [len(cell) - 1]])
        cell, start, end = cell[first], start[first], end[last]
    ev = Events(cell=cell.astype(np.int64), start=start.astype(np.int64),
                end=end.astype(np.int64))
    if len(ev):
        ev.metrics = _event_metrics(temp, seas_d, thresh_d, ev)
    else:
        ev.metrics = {k: np.array([], dtype=np.float32) for k in (
            "duration", "intensity_max", "intensity_mean", "intensity_var",
            "intensity_cumulative", "rate_onset", "rate_decline", "category")}
    return ev


def _event_metrics(temp, seas_d, thresh_d, ev: Events) -> dict[str, np.ndarray]:
    ncell, nt = temp.shape
    # Missing days take the climatology, as in Oliver's detect (they can sit
    # inside an event only through a joined gap).
    # float32 throughout: events are at most a few hundred days, and the
    # (cells × days) temporaries dominate the memory of a block.
    rel_flat = np.zeros(ncell * nt + 1, dtype=np.float32)
    np.subtract(temp.ravel(), seas_d.ravel(), out=rel_flat[:-1])
    np.nan_to_num(rel_flat, copy=False, nan=0.0)
    s_flat = ev.cell * nt + ev.start
    e_flat = ev.cell * nt + ev.end
    dur = ev.end - ev.start + 1
    bounds = np.empty(2 * len(ev), dtype=np.int64)
    bounds[0::2] = s_flat
    bounds[1::2] = e_flat + 1

    def reduce(ufunc, arr):
        return ufunc.reduceat(arr, bounds)[0::2]

    total = reduce(np.add, rel_flat).astype(np.float64)
    peak = reduce(np.maximum, rel_flat).astype(np.float64)
    mean = total / dur
    # Second moment about the event mean (one pass would lose precision).
    eid = np.repeat(np.arange(len(ev)), dur)
    starts_e = np.cumsum(dur) - dur
    off = np.arange(int(dur.sum())) - starts_e[eid]
    day_flat = s_flat[eid] + off
    dev = rel_flat[day_flat] - mean[eid].astype(np.float32)
    var = np.sqrt(np.add.reduceat(dev.astype(np.float64) ** 2, starts_e) / dur)
    del dev

    # Category from the largest (T − thresh)/(thresh − clim) of the event.
    # Only event days matter, so evaluate the norm on those alone.
    th = thresh_d.ravel()[day_flat]
    se = seas_d.ravel()[day_flat]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (rel_flat[day_flat] + se - th) / (th - se)
    norm = np.nan_to_num(norm, nan=-np.inf)
    category = np.minimum(np.floor(1.0 + np.maximum.reduceat(norm, starts_e)
                                   .astype(np.float64)), 4.0)
    del norm, th, se

    # First day of the intensity peak: compare each event day with its peak.
    hit = np.where(rel_flat[day_flat] == peak[eid].astype(np.float32), off,
                   np.iinfo(np.int64).max)
    tt_peak = np.minimum.reduceat(hit, starts_e)
    del eid, off, hit, day_flat

    # Onset/decline from/to the half day outside the event; at the ends of
    # the record from/to the first/last event day, over at least one day.
    rel_s = rel_flat[s_flat].astype(np.float64)
    rel_e = rel_flat[e_flat].astype(np.float64)
    has_prev = ev.start > 0
    has_next = ev.end < nt - 1
    prev = rel_flat[np.maximum(s_flat - 1, 0)].astype(np.float64)
    nxt = rel_flat[np.minimum(e_flat + 1, ncell * nt - 1)].astype(np.float64)
    tt_end = dur - 1
    onset = np.where(has_prev,
                     (peak - 0.5 * (rel_s + prev)) / (tt_peak + 0.5),
                     (peak - rel_s) / np.maximum(tt_peak, 1))
    decline = np.where(has_next,
                       (peak - 0.5 * (rel_e + nxt)) / (tt_end - tt_peak + 0.5),
                       (peak - rel_e) / np.maximum(tt_end - tt_peak, 1))
    f = np.float32
    return {
        "duration": dur.astype(f),
        "intensity_max": peak.astype(f),
        "intensity_mean": mean.astype(f),
        "intensity_var": var.astype(f),
        "intensity_cumulative": total.astype(f),
        "rate_onset": onset.astype(f),
        "rate_decline": decline.astype(f),
        "category": category.astype(f),
    }


# ── Per-cell reduction ───────────────────────────────────────────────────

def mhw_day_mask(ev: Events, ncell: int, nt: int) -> np.ndarray:
    """(cells, days) bool: True on days inside an event."""
    acc = np.zeros(ncell * (nt + 1), dtype=np.int16)
    np.add.at(acc, ev.cell * (nt + 1) + ev.start, 1)
    np.add.at(acc, ev.cell * (nt + 1) + ev.end + 1, -1)
    return np.cumsum(acc.reshape(ncell, nt + 1), axis=1, dtype=np.int16)[:, :nt] > 0


def cell_metrics(ev: Events, ncell: int, nt: int, years: np.ndarray,
                 valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-cell maps (:data:`METRICS`) and annual counts from *ev*.

    Parameters
    ----------
    years : (days,) year of every day
    valid : (cells,) bool, cells with a climatology; others are NaN

    Returns a dict of ``(cells,)`` float32 maps plus ``mhw_days_annual``
    and ``events_annual`` of shape ``(n_years, cells)`` (events counted in
    their start year), and ``years`` (the year axis).
    """
    f = np.float32
    n = np.bincount(ev.cell, minlength=ncell).astype(np.float64)
    out: dict[str, np.ndarray] = {}
    uyears = np.unique(years)
    n_years = len(uyears)

    def mean_of(x):
        good = np.isfinite(x)
        s = np.bincount(ev.cell[good], weights=x[good], minlength=ncell)
        c = np.bincount(ev.cell[good], minlength=ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan).astype(f)

    def max_of(x):
        m = np.full(ncell, -np.inf)
        good = np.isfinite(x)
        np.maximum.at(m, ev.cell[good], x[good].astype(np.float64))
        return np.where(np.isfinite(m), m, np.nan).astype(f)

    M = ev.metrics
    out["frequency"] = (n / n_years).astype(f)
    out["duration_mean"] = mean_of(M["duration"])
    out["intensity_max_max"] = max_of(M["intensity_max"])
    out["intensity_max_mean"] = mean_of(M["intensity_max"])
    out["intensity_mean_mean"] = mean_of(M["intensity_mean"])
    out["intensity_cumulative_mean"] = mean_of(M["intensity_cumulative"])
    out["intensity_var_mean"] = mean_of(M["intensity_var"])
    out["rate_onset_mean"] = mean_of(M["rate_onset"])
    out["rate_decline_mean"] = mean_of(M["rate_decline"])
    out["category_max"] = max_of(M["category"])
    # Cells without events have no event means but a well-defined zero
    # frequency, MHW-day share and category.
    out["category_max"] = np.where(n == 0, 0.0, out["category_max"]).astype(f)

    mask = mhw_day_mask(ev, ncell, nt)
    out["mhw_days_pct"] = (100.0 * mask.mean(axis=1)).astype(f)
    ystart = np.searchsorted(years, uyears)
    out["mhw_days_annual"] = np.add.reduceat(mask, ystart, axis=1,
                                             dtype=np.int32).T.astype(f)
    ev_year = np.searchsorted(uyears, years[ev.start]) if len(ev) else ev.start
    ann = np.zeros((n_years, ncell), dtype=f)
    np.add.at(ann, (ev_year, ev.cell), 1.0)
    out["events_annual"] = ann
    out["years"] = uyears
    if valid is not None:
        for k, v in out.items():
            if k == "years":
                continue
            v[..., ~valid] = np.nan
    return out


def detect(temp: np.ndarray, years: np.ndarray, months: np.ndarray,
           days: np.ndarray, *, calendar: str = "standard",
           base_years: tuple[int, int] | None = None,
           pctile: float = PCTILE, window_half_width: int = WINDOW_HALF_WIDTH,
           smooth_width: int = SMOOTH_WIDTH, min_duration: int = MIN_DURATION,
           max_gap: int = MAX_GAP) -> tuple[Events, dict[str, np.ndarray]]:
    """Climatology, events and per-cell maps for a block of cells.

    *temp* is ``(cells, days)``; *years*/*months*/*days* describe the time
    axis.  *base_years* defaults to the whole record.  Returns ``(events,
    maps)`` with *maps* from :func:`cell_metrics` plus ``seas_mean`` and
    ``thresh_minus_seas`` (annual means of the climatology and of the
    threshold's height above it).
    """
    years = np.asarray(years)
    temp = np.asarray(temp, dtype=np.float32)
    slots, n_slots, feb29 = calendar_slots(years, months, days, calendar)
    if base_years is None:
        base = np.ones(len(years), dtype=bool)
    else:
        base = (years >= base_years[0]) & (years <= base_years[1])
    seas, thresh = climatology(temp, slots, n_slots, base, feb29=feb29,
                               pctile=pctile, window_half_width=window_half_width,
                               smooth_width=smooth_width)
    seas_d = seas[:, slots]
    thresh_d = thresh[:, slots]
    ev = detect_events(temp, seas_d, thresh_d, min_duration=min_duration,
                       max_gap=max_gap)
    valid = np.isfinite(seas).all(axis=1)
    maps = cell_metrics(ev, temp.shape[0], temp.shape[1], years, valid)
    for key, arr in (("seas_mean", seas), ("thresh_minus_seas", thresh - seas)):
        out = np.full(temp.shape[0], np.nan, dtype=np.float32)
        out[valid] = arr[valid].mean(axis=1)
        maps[key] = out
    return ev, maps
