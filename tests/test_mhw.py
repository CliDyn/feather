"""Tests for the vectorised marine heatwave detection (feather.util.mhw)."""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from feather.util import mhw


# ── A per-cell reference, following Oliver's marineHeatWaves.detect ──────

def _runavg(ts, w):
    n = len(ts)
    ts3 = np.concatenate([ts, ts, ts])
    return np.convolve(ts3, np.ones(w) / w, mode="same")[n:2 * n]


def _reference(temp, dates, base, pctile=90, hw=5, smooth=31, min_dur=5,
               max_gap=2):
    """Oliver's algorithm for one series (Gregorian calendar)."""
    temp = temp.astype(float).copy()
    t_n = len(temp)
    leap = pd.date_range("2012-01-01", "2012-12-31")
    doy_of = {(d.month, d.day): i + 1 for i, d in enumerate(leap)}
    doy = np.array([doy_of[(d.month, d.day)] for d in dates])
    years = np.array([d.year for d in dates])
    c0 = np.where(years == base[0])[0][0]
    c1 = np.where(years == base[1])[0][-1]
    thresh_y = np.full(366, np.nan)
    seas_y = np.full(366, np.nan)
    for d in range(1, 367):
        if d == 60:
            continue
        tt0 = np.where(doy[c0:c1 + 1] == d)[0]
        if not len(tt0):
            continue
        tt = np.concatenate([c0 + tt0 + w for w in range(-hw, hw + 1)])
        tt = tt[(tt >= 0) & (tt < t_n)]
        thresh_y[d - 1] = np.nanpercentile(temp[tt], pctile)
        seas_y[d - 1] = np.nanmean(temp[tt])
    thresh_y[59] = 0.5 * (thresh_y[58] + thresh_y[60])
    seas_y[59] = 0.5 * (seas_y[58] + seas_y[60])
    thresh_y = _runavg(thresh_y, smooth)
    seas_y = _runavg(seas_y, smooth)
    thresh = thresh_y[doy - 1]
    seas = seas_y[doy - 1]
    missing = np.isnan(temp)
    temp[missing] = seas[missing]
    exceed = temp - thresh > 0
    lab, n = ndimage.label(exceed)
    starts, ends = [], []
    for ev in range(1, n + 1):
        w = np.where(lab == ev)[0]
        if len(w) >= min_dur:
            starts.append(w[0])
            ends.append(w[-1])
    gaps = np.array(starts[1:]) - np.array(ends[:-1]) - 1
    while len(gaps) and gaps.min() <= max_gap:
        e = np.where(gaps <= max_gap)[0][0]
        ends[e] = ends[e + 1]
        del starts[e + 1], ends[e + 1]
        gaps = np.array(starts[1:]) - np.array(ends[:-1]) - 1
    out = []
    for s, e in zip(starts, ends):
        rel = temp[s:e + 1] - seas[s:e + 1]
        norm = (temp[s:e + 1] - thresh[s:e + 1]) / (thresh[s:e + 1] - seas[s:e + 1])
        tp = int(np.argmax(rel))
        cat = min(np.floor(1 + norm[np.argmax(norm)]), 4)
        if s > 0:
            st = 0.5 * (rel[0] + temp[s - 1] - seas[s - 1])
            onset = (rel[tp] - st) / (tp + 0.5)
        else:
            onset = (rel[tp] - rel[0]) / max(tp, 1)
        te = e - s
        if e < t_n - 1:
            en = 0.5 * (rel[-1] + temp[e + 1] - seas[e + 1])
            decline = (rel[tp] - en) / (te - tp + 0.5)
        else:
            decline = (rel[tp] - rel[-1]) / max(te - tp, 1)
        out.append(dict(start=s, end=e, duration=e - s + 1,
                        intensity_max=rel[tp], intensity_mean=rel.mean(),
                        intensity_var=np.sqrt(rel.var()),
                        intensity_cumulative=rel.sum(), rate_onset=onset,
                        rate_decline=decline, category=cat))
    return seas, thresh, out


def _synthetic(ncell=6, years=(2000, 2009), seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
    t = np.arange(len(dates))
    seasonal = 15 + 5 * np.sin(2 * np.pi * t / 365.25)
    # AR(1) anomalies give realistic multi-day exceedances.
    noise = np.zeros((ncell, len(t)))
    eps = rng.normal(0, 0.4, (ncell, len(t)))
    for i in range(1, len(t)):
        noise[:, i] = 0.9 * noise[:, i - 1] + eps[:, i]
    temp = (seasonal + noise + 0.02 * np.arange(ncell)[:, None]).astype(np.float32)
    if nan_frac:
        temp[rng.random(temp.shape) < nan_frac] = np.nan
    return temp, dates


def _detect(temp, dates, **kw):
    return mhw.detect(temp, dates.year.values, dates.month.values,
                      dates.day.values, **kw)


@pytest.mark.parametrize("nan_frac", [0.0, 0.01])
def test_matches_reference(nan_frac):
    temp, dates = _synthetic(nan_frac=nan_frac)
    base = (2001, 2008)
    ev, maps = _detect(temp, dates, base_years=base)
    for c in range(temp.shape[0]):
        seas, thresh, ref = _reference(temp[c], dates, base)
        mine = np.flatnonzero(ev.cell == c)
        assert len(mine) == len(ref)
        for k, e in zip(mine, ref):
            assert ev.start[k] == e["start"] and ev.end[k] == e["end"]
            for key in ("duration", "intensity_max", "intensity_mean",
                        "intensity_var", "intensity_cumulative",
                        "rate_onset", "rate_decline", "category"):
                np.testing.assert_allclose(ev.metrics[key][k], e[key],
                                           rtol=2e-4, atol=2e-5, err_msg=key)


def test_climatology_matches_reference():
    temp, dates = _synthetic(ncell=2)
    slots, n, feb29 = mhw.calendar_slots(dates.year.values, dates.month.values,
                                         dates.day.values)
    base = (dates.year.values >= 2001) & (dates.year.values <= 2008)
    seas, thresh = mhw.climatology(temp, slots, n, base, feb29=feb29)
    for c in range(2):
        s_ref, t_ref, _ = _reference(temp[c], dates, (2001, 2008))
        np.testing.assert_allclose(seas[c, slots], s_ref, rtol=1e-5)
        np.testing.assert_allclose(thresh[c, slots], t_ref, rtol=1e-5)


def test_cell_metrics_aggregate_events():
    temp, dates = _synthetic(ncell=4)
    ev, maps = _detect(temp, dates)
    nyears = len(np.unique(dates.year))
    for c in range(4):
        k = ev.cell == c
        assert maps["frequency"][c] == pytest.approx(k.sum() / nyears)
        assert maps["duration_mean"][c] == pytest.approx(ev.metrics["duration"][k].mean())
        assert maps["intensity_max_max"][c] == pytest.approx(
            ev.metrics["intensity_max"][k].max())
        assert maps["category_max"][c] == ev.metrics["category"][k].max()
        days = ev.metrics["duration"][k].sum()
        assert maps["mhw_days_pct"][c] == pytest.approx(100 * days / len(dates), rel=1e-5)
        assert maps["mhw_days_annual"][:, c].sum() == pytest.approx(days)
        assert maps["events_annual"][:, c].sum() == k.sum()
    assert maps["mhw_days_annual"].shape == (nyears, 4)


def test_join_across_short_gap():
    dates = pd.date_range("2000-01-01", "2003-12-31")
    temp = np.zeros((1, len(dates)), dtype=np.float32)
    temp += np.random.default_rng(1).normal(0, 0.01, temp.shape).astype(np.float32)
    i0 = 500
    temp[0, i0:i0 + 6] = 5.0           # 6 days
    temp[0, i0 + 8:i0 + 14] = 5.0      # gap of 2 days, then 6 days
    temp[0, i0 + 30:i0 + 33] = 5.0     # 3 days: too short
    ev, _ = _detect(temp, dates)
    big = ev.metrics["intensity_max"] > 1
    assert big.sum() == 1
    k = np.flatnonzero(big)[0]
    assert (ev.start[k], ev.end[k]) == (i0, i0 + 13)
    assert ev.metrics["duration"][k] == 14


def test_constant_series_has_no_events():
    dates = pd.date_range("2000-01-01", "2002-12-31")
    temp = np.full((3, len(dates)), 10.0, dtype=np.float32)
    ev, maps = _detect(temp, dates)
    assert len(ev) == 0
    np.testing.assert_array_equal(maps["frequency"], 0)
    np.testing.assert_array_equal(maps["mhw_days_pct"], 0)
    assert np.isnan(maps["duration_mean"]).all()
    np.testing.assert_array_equal(maps["category_max"], 0)


def test_all_nan_cell_is_nan():
    temp, dates = _synthetic(ncell=2)
    temp[1] = np.nan
    _, maps = _detect(temp, dates)
    assert np.isfinite(maps["frequency"][0])
    for k in ("frequency", "mhw_days_pct", "category_max", "duration_mean"):
        assert np.isnan(maps[k][1]), k


@pytest.mark.parametrize("cal,n", [("360_day", 360), ("noleap", 365),
                                   ("standard", 366), ("all_leap", 366)])
def test_calendar_slots(cal, n):
    months = np.array([1, 2, 3, 12])
    days = np.array([1, 28, 1, 30])
    slots, n_slots, feb29 = mhw.calendar_slots(np.zeros(4), months, days, cal)
    assert n_slots == n
    assert slots[0] == 0
    if cal == "360_day":
        assert list(slots) == [0, 57, 60, 359]
    elif cal == "noleap":
        assert list(slots) == [0, 58, 59, 363]
        assert feb29 is None
    else:
        assert list(slots) == [0, 58, 60, 364]
        assert feb29 == (59 if cal == "standard" else None)


def test_360_day_calendar_runs():
    import cftime
    times = [cftime.Datetime360Day(y, m, d) for y in range(2000, 2006)
             for m in range(1, 13) for d in range(1, 31)]
    years = np.array([t.year for t in times])
    months = np.array([t.month for t in times])
    days = np.array([t.day for t in times])
    rng = np.random.default_rng(3)
    temp = np.cumsum(rng.normal(0, 0.3, (2, len(times))), axis=1).astype(np.float32)
    ev, maps = mhw.detect(temp, years, months, days, calendar="360_day")
    assert np.isfinite(maps["frequency"]).all()
    assert maps["mhw_days_annual"].shape == (6, 2)


def test_percentile_rows_matches_nanpercentile():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 37))
    a[rng.random(a.shape) < 0.1] = np.nan
    a[3] = np.nan
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanpercentile(a, 90, axis=1)
    np.testing.assert_allclose(mhw._nan_percentile_rows(a, 90), expected)
