"""ETCCDI daily-temperature extreme indices.

Definitions follow the ETCCDI set as distributed in the C3S dataset
*Climate extreme indices and heat stress indicators derived from CMIP6
global climate projections* (``sis-extreme-indices-cmip6``).  ``TX``/``TN``
are the daily maximum/minimum near-surface air temperature in °C and ``TG``
the daily mean, taken here as ``(TX + TN) / 2`` (as climdex does when no mean
is given — ERA5 and most EERIE members publish no daily ``tas`` next to the
extremes).

=======  ===========================================================  ======
Index    Definition (per year)                                        Units
=======  ===========================================================  ======
TXx      Maximum of daily TX                                          °C
TNx      Maximum of daily TN                                          °C
TXn      Minimum of daily TX                                          °C
TNn      Minimum of daily TN                                          °C
DTR      Mean diurnal temperature range, mean(TX − TN)                °C
FD       Frost days, TN < 0 °C                                        days
ID       Ice days, TX < 0 °C                                          days
TX10p    Cold days, share of days with TX < TX10 (calendar day)       %
TX90p    Warm days, share of days with TX > TX90                      %
TN10p    Cold nights, share of days with TN < TN10                    %
TN90p    Warm nights, share of days with TN > TN90                    %
WSDI     Warm spell duration: days in runs of ≥ 6 days TX > TX90      days
CSDI     Cold spell duration: days in runs of ≥ 6 days TN < TN10      days
GSL      Growing season length (see :func:`growing_season_length`)    days
=======  ===========================================================  ======

``TX10``/``TX90``/``TN10``/``TN90`` are calendar-day percentiles over a base
period, from a 5-day window centred on each day (``5 × n_years`` samples),
with the Hyndman–Fan type-8 estimator (numpy ``median_unbiased``), as in
climdex.  ETCCDI bootstraps the in-base years to remove an inhomogeneity at
the base-period edges; that is **not** done here, so inside the base period
the four percentile indices are slightly too close to 10 % — and a model
evaluated over its own base period scores ≈ 10 % everywhere by construction.

Spells (WSDI, CSDI) do not continue across years (climdex default
``spells.can.span.years = FALSE``).  Seasonal indices (DJF, MAM, JJA, SON)
exist for everything except WSDI, CSDI and GSL, which are annual only.

Everything here is plain numpy on ``(time, cell)`` arrays, so the diagnostic
can stream one year at a time over the land cells of a 0.25° grid.
"""

from __future__ import annotations

import numpy as np

#: Freezing point (°C) for FD/ID.
FREEZE_C: float = 0.0

#: GSL temperature threshold (°C) and spell length (days).
GSL_THRESHOLD_C: float = 5.0
GSL_SPELL_DAYS: int = 6

#: Minimum run length (days) for WSDI/CSDI.
SPELL_DAYS: int = 6

#: Calendar-day window (days, centred) for the base-period percentiles.
PERCENTILE_WINDOW: int = 5

#: Index order used throughout feather.
INDICES: tuple[str, ...] = (
    "txx", "tnx", "txn", "tnn", "dtr", "fd", "id",
    "tx10p", "tx90p", "tn10p", "tn90p", "wsdi", "csdi", "gsl",
)

#: ``{index: (ETCCDI label, long name, units)}``.
INDEX_INFO: dict[str, tuple[str, str, str]] = {
    "txx":   ("TXx",   "Maximum of daily maximum temperature",          "°C"),
    "tnx":   ("TNx",   "Maximum of daily minimum temperature",          "°C"),
    "txn":   ("TXn",   "Minimum of daily maximum temperature",          "°C"),
    "tnn":   ("TNn",   "Minimum of daily minimum temperature",          "°C"),
    "dtr":   ("DTR",   "Mean diurnal temperature range",                "°C"),
    "fd":    ("FD",    "Frost days (TN < 0 °C)",                        "days"),
    "id":    ("ID",    "Ice days (TX < 0 °C)",                          "days"),
    "tx10p": ("TX10p", "Cold days (TX < 10th percentile)",              "%"),
    "tx90p": ("TX90p", "Warm days (TX > 90th percentile)",              "%"),
    "tn10p": ("TN10p", "Cold nights (TN < 10th percentile)",            "%"),
    "tn90p": ("TN90p", "Warm nights (TN > 90th percentile)",            "%"),
    "wsdi":  ("WSDI",  "Warm spell duration index",                     "days"),
    "csdi":  ("CSDI",  "Cold spell duration index",                     "days"),
    "gsl":   ("GSL",   "Growing season length",                         "days"),
}

#: Indices that need TX, TN or both.
NEEDS: dict[str, frozenset[str]] = {
    "txx": frozenset({"tasmax"}), "txn": frozenset({"tasmax"}),
    "tnx": frozenset({"tasmin"}), "tnn": frozenset({"tasmin"}),
    "dtr": frozenset({"tasmax", "tasmin"}),
    "fd": frozenset({"tasmin"}), "id": frozenset({"tasmax"}),
    "tx10p": frozenset({"tasmax"}), "tx90p": frozenset({"tasmax"}),
    "tn10p": frozenset({"tasmin"}), "tn90p": frozenset({"tasmin"}),
    "wsdi": frozenset({"tasmax"}), "csdi": frozenset({"tasmin"}),
    "gsl": frozenset({"tasmax", "tasmin"}),
}

#: Indices defined per season as well as per year.
SEASONAL_INDICES: tuple[str, ...] = tuple(
    i for i in INDICES if i not in ("wsdi", "csdi", "gsl"))

#: Percentile-exceedance indices (≈ 10 % over the base period by construction).
PERCENTILE_INDICES: frozenset[str] = frozenset({"tx10p", "tx90p", "tn10p", "tn90p"})

#: Threshold keys: ``{name: (variable, quantile)}``.
THRESHOLDS: dict[str, tuple[str, float]] = {
    "tx10": ("tasmax", 0.10), "tx90": ("tasmax", 0.90),
    "tn10": ("tasmin", 0.10), "tn90": ("tasmin", 0.90),
}

# Day-of-year offsets of the month starts in a 365-day year.
_MONTH_START_365 = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def calendar_doy(months, days, calendar: str = "standard") -> tuple[np.ndarray, int]:
    """0-based calendar-day index of each date and the number of calendar days.

    360-day calendars keep their 360 days.  Every other calendar is mapped
    onto 365 days, 29 February sharing the slot of 28 February (it then
    takes that day's thresholds, and joins its percentile window).
    """
    months = np.asarray(months, dtype=np.int64)
    days = np.asarray(days, dtype=np.int64)
    if str(calendar).lower() in ("360_day", "360"):
        return (months - 1) * 30 + (days - 1), 360
    doy = _MONTH_START_365[months - 1] + np.minimum(days, np.where(months == 2, 28, 31)) - 1
    return doy.astype(np.int64), 365


def calendar_day_percentiles(
    x: np.ndarray,
    doy: np.ndarray,
    n_doy: int,
    quantiles=(0.1, 0.9),
    window: int = PERCENTILE_WINDOW,
) -> dict[float, np.ndarray]:
    """Calendar-day percentiles of *x* over all base-period days.

    Parameters
    ----------
    x : ndarray, shape (n_days, n_cells)
        Every base-period day (any number of years), °C.
    doy : ndarray, shape (n_days,)
        0-based calendar day of each row (:func:`calendar_doy`).
    n_doy : int
        Calendar days per year (365 or 360).
    quantiles : sequence of float
        As fractions.
    window : int
        Odd window length in days, centred and wrapping round the year end.

    Returns
    -------
    dict
        ``{q: ndarray(n_doy, n_cells)}`` float32.  A cell whose window holds
        a missing day uses the remaining days.
    """
    x = np.asarray(x, dtype=np.float32)
    doy = np.asarray(doy)
    half = window // 2
    rows_by_day = [np.flatnonzero(doy == d) for d in range(n_doy)]
    out = {q: np.full((n_doy, x.shape[1]), np.nan, dtype=np.float32)
           for q in quantiles}
    qs = list(quantiles)
    for d in range(n_doy):
        rows = np.concatenate([rows_by_day[(d + k) % n_doy]
                               for k in range(-half, half + 1)])
        if not len(rows):
            continue
        block = x[rows]
        if np.isnan(block).any():
            with np.errstate(invalid="ignore"), _ignore_all_nan():
                res = np.nanquantile(block, qs, axis=0, method="median_unbiased")
        else:
            res = np.quantile(block, qs, axis=0, method="median_unbiased")
        for q, r in zip(qs, res):
            out[q][d] = r
    return out


class _ignore_all_nan:
    """Silence numpy's all-NaN-slice warning (ocean/missing cells)."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.filterwarnings("ignore", message="All-NaN slice")
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _spell_days(flag: np.ndarray, min_len: int = SPELL_DAYS) -> np.ndarray:
    """Per cell, the number of days that belong to a run of ≥ *min_len* flags."""
    n = flag.shape[0]
    if n < min_len:
        return np.zeros(flag.shape[1:], dtype=np.float32)
    c = np.concatenate([np.zeros((1,) + flag.shape[1:], np.int32),
                        np.cumsum(flag, axis=0, dtype=np.int32)])
    full = (c[min_len:] - c[:-min_len]) == min_len          # run starts at s
    # Day i is in a long run when some full window starts in [i-min_len+1, i].
    f = np.concatenate([np.zeros((1,) + flag.shape[1:], np.int32),
                        np.cumsum(full, axis=0, dtype=np.int32)])
    idx = np.arange(n)
    hi = np.minimum(idx, n - min_len) + 1
    lo = np.clip(idx - min_len + 1, 0, n - min_len + 1)
    covered = (f[hi] - f[lo]) > 0
    return covered.sum(axis=0).astype(np.float32)


def _first_run_start(flag: np.ndarray, min_len: int, start: np.ndarray) -> np.ndarray:
    """First index ``s ≥ start`` (per cell) opening a run of ≥ *min_len* flags.

    Returns ``n`` where there is none.
    """
    n = flag.shape[0]
    if n < min_len:
        return np.full(flag.shape[1:], n, dtype=np.int64)
    c = np.concatenate([np.zeros((1,) + flag.shape[1:], np.int32),
                        np.cumsum(flag, axis=0, dtype=np.int32)])
    full = (c[min_len:] - c[:-min_len]) == min_len          # (n-min_len+1, cells)
    pos = np.arange(full.shape[0])[:, None]
    full &= pos >= np.asarray(start)[None, ...]
    has = full.any(axis=0)
    return np.where(has, full.argmax(axis=0), n).astype(np.int64)


def growing_season_length(
    tg: np.ndarray, mid: int, min_valid_days: int,
) -> np.ndarray:
    """ETCCDI GSL for one growing year of daily mean temperature (°C).

    Days from the first run of ≥ 6 days with ``TG > 5 °C`` to the first run
    of ≥ 6 days with ``TG < 5 °C`` that starts on or after day *mid* (1 July
    in the northern hemisphere, 1 January in the southern one, whose growing
    year runs July–June).  No warm run → 0; no cold run → until the end of
    the growing year.  Missing days count as neither warm nor cold.
    """
    tg = np.asarray(tg, dtype=np.float32)
    n = tg.shape[0]
    valid = np.isfinite(tg)
    with np.errstate(invalid="ignore"):
        warm = valid & (tg > GSL_THRESHOLD_C)
        cold = valid & (tg < GSL_THRESHOLD_C)
    zero = np.zeros(tg.shape[1:], dtype=np.int64)
    s = _first_run_start(warm, GSL_SPELL_DAYS, zero)
    e = _first_run_start(cold, GSL_SPELL_DAYS, np.maximum(s, mid))
    gsl = np.where(s < n, e - s, 0).astype(np.float32)
    return np.where(valid.sum(axis=0) < min_valid_days, np.nan, gsl)


def annual_indices(
    tx: np.ndarray | None,
    tn: np.ndarray | None,
    doy: np.ndarray,
    *,
    thresholds: dict[str, np.ndarray] | None = None,
    min_valid_days: int,
    spells: bool = True,
) -> dict[str, np.ndarray]:
    """Every index computable from one year (or season) of daily data.

    Parameters
    ----------
    tx, tn : ndarray, shape (n_days, n_cells), or None
        Daily maximum / minimum temperature in **°C** on the same days.
        Indices that need a missing variable are left out of the result.
    doy : ndarray, shape (n_days,)
        Calendar-day index of each day, selecting the threshold row.
    thresholds : dict, optional
        ``{"tx10" | "tx90" | "tn10" | "tn90": ndarray(n_doy, n_cells)}``.
        Missing keys leave the matching percentile and spell indices out.
    min_valid_days : int
        Cells with fewer finite days (of the variable an index uses) are NaN.
    spells : bool
        Also return WSDI/CSDI (annual blocks only).

    Returns
    -------
    dict
        ``{index: ndarray(n_cells)}`` float32, GSL excluded (see
        :func:`growing_season_length`).
    """
    thr = thresholds or {}
    out: dict[str, np.ndarray] = {}
    doy = np.asarray(doy)

    def stats(x, key_max, key_min, frost_key, p10, p90, k10, k90, spell_key, low):
        x = np.asarray(x, dtype=np.float32)
        valid = np.isfinite(x)
        nv = valid.sum(axis=0)
        bad = nv < min_valid_days
        res = {
            key_max: np.where(valid, x, -np.inf).max(axis=0),
            key_min: np.where(valid, x, np.inf).min(axis=0),
        }
        with np.errstate(invalid="ignore"):
            res[frost_key] = (valid & (x < FREEZE_C)).sum(axis=0).astype(np.float32)
            nv_safe = np.maximum(nv, 1)
            if p10 in thr:
                below = valid & (x < thr[p10][doy])
                res[k10] = 100.0 * below.sum(axis=0) / nv_safe
                if spells and low:
                    res[spell_key] = _spell_days(below)
            if p90 in thr:
                above = valid & (x > thr[p90][doy])
                res[k90] = 100.0 * above.sum(axis=0) / nv_safe
                if spells and not low:
                    res[spell_key] = _spell_days(above)
        for k, v in res.items():
            out[k] = np.where(bad, np.nan, v).astype(np.float32)

    if tx is not None:
        stats(tx, "txx", "txn", "id", "tx10", "tx90", "tx10p", "tx90p", "wsdi", low=False)
    if tn is not None:
        stats(tn, "tnx", "tnn", "fd", "tn10", "tn90", "tn10p", "tn90p", "csdi", low=True)
    if tx is not None and tn is not None:
        rng = np.asarray(tx, np.float32) - np.asarray(tn, np.float32)
        ok = np.isfinite(rng)
        n_ok = ok.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dtr = np.where(ok, rng, 0.0).sum(axis=0, dtype=np.float64) / n_ok
        out["dtr"] = np.where(n_ok < min_valid_days, np.nan, dtr).astype(np.float32)
    return out
