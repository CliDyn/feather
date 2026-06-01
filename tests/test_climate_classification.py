"""Tests for the Köppen–Trewartha climate classification.

Focus on the pure classifier (:mod:`feather.util.koeppen_trewartha`), which
is deterministic and login-node safe.  Each test builds a hand-crafted
monthly climatology that should fall into a specific KT type.
"""

import numpy as np
import pytest
import xarray as xr

from feather.util.koeppen_trewartha import (
    KT_CODES,
    KT_LABELS,
    area_percent_by_type,
    classify_kt,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _single_cell(tmon_vals, pmon_vals, lat=45.0):
    """Build (month, lat, lon) climatology for one cell.

    Parameters
    ----------
    tmon_vals, pmon_vals : sequence of length 12
        Monthly temperature (°C) and precipitation (cm/month).
    lat : float
        Latitude of the single cell.
    """
    months = np.arange(1, 13)
    tmon = xr.DataArray(
        np.asarray(tmon_vals, dtype=float).reshape(12, 1, 1),
        dims=("month", "lat", "lon"),
        coords={"month": months, "lat": [lat], "lon": [0.0]},
    )
    pmon = xr.DataArray(
        np.asarray(pmon_vals, dtype=float).reshape(12, 1, 1),
        dims=("month", "lat", "lon"),
        coords={"month": months, "lat": [lat], "lon": [0.0]},
    )
    lat_da = xr.DataArray([lat], dims="lat", coords={"lat": [lat]})
    return tmon, pmon, lat_da


def _classify_one(tmon_vals, pmon_vals, lat=45.0) -> str:
    tmon, pmon, lat_da = _single_cell(tmon_vals, pmon_vals, lat)
    code = classify_kt(tmon, pmon, lat_da)
    val = int(code.isel(lat=0, lon=0).values)
    return KT_CODES.get(val, "UNCLASSIFIED")


# ── Group F (polar) ───────────────────────────────────────────────────


def test_polar_ice_fi():
    # All months below 0 °C → Fi
    assert _classify_one([-30] * 12, [3] * 12, lat=80) == "Fi"


def test_polar_tundra_ft():
    # Warmest month between 0 and 10 °C → Ft
    t = [-20, -18, -12, -5, 2, 6, 8, 7, 3, -4, -12, -18]
    assert _classify_one(t, [3] * 12, lat=75) == "Ft"


# ── Group E (boreal) ──────────────────────────────────────────────────


def test_boreal_continental_ec():
    # 1–3 months ≥10 °C, coldest ≤ −10 °C → Ec
    t = [-25, -22, -15, -3, 6, 12, 14, 11, 4, -5, -15, -22]
    assert _classify_one(t, [5] * 12, lat=65) == "Ec"


def test_boreal_oceanic_eo():
    # 1–3 months ≥10 °C, coldest > −10 °C → Eo
    t = [-5, -4, -1, 3, 7, 11, 13, 11, 6, 1, -2, -4]
    assert _classify_one(t, [8] * 12, lat=60) == "Eo"


# ── Group D (temperate/continental) ───────────────────────────────────


def test_temperate_continental_dc():
    # 4–7 months ≥10 °C, coldest ≤ 0 °C → Dc
    t = [-5, -3, 2, 8, 14, 18, 20, 19, 14, 8, 2, -3]
    # Wet enough to avoid the dry threshold
    assert _classify_one(t, [10] * 12, lat=50) == "Dc"


def test_temperate_oceanic_do():
    # 4–7 months ≥10 °C, coldest > 0 °C → Do
    t = [4, 4, 6, 9, 12, 15, 17, 17, 14, 10, 6, 5]
    assert _classify_one(t, [12] * 12, lat=48) == "Do"


# ── Group C (subtropical) ─────────────────────────────────────────────


def test_subtropical_humid_cf():
    # ≥8 months ≥10 °C, coldest ≤18 °C, no dry season → Cf
    t = [9, 10, 13, 16, 20, 24, 26, 26, 23, 18, 13, 10]
    assert _classify_one(t, [12] * 12, lat=33) == "Cf"


def test_mediterranean_cs():
    # Dry summer (NH summer = Apr–Sep dry), wet winter → Cs
    # months: J F M A M J J A S O N D
    p = [10, 9, 7, 2, 1, 0.5, 0.3, 0.5, 2, 8, 10, 11]
    t = [9, 10, 12, 15, 19, 23, 26, 26, 22, 17, 12, 10]
    assert _classify_one(t, p, lat=37) == "Cs"


# ── Group B (dry) ─────────────────────────────────────────────────────


def test_desert_bw():
    # Hot, essentially no rain → BW
    assert _classify_one([20, 22, 26, 30, 34, 38, 39, 38, 34, 29, 24, 20],
                         [0.1] * 12, lat=23) == "BW"


def test_steppe_bs():
    # Semi-arid: precip between halfR and R → BS
    t = [12, 14, 18, 22, 26, 30, 32, 31, 27, 22, 17, 13]
    # Tune annual precip to sit in the steppe band
    p = [3.0] * 12
    assert _classify_one(t, p, lat=35) == "BS"


# ── Group A (tropical) ────────────────────────────────────────────────


def test_tropical_rainforest_ar():
    # Coldest ≥18 °C, ≥10 wet months (>6 cm), ≤2 dry → Ar
    assert _classify_one([26] * 12, [20] * 12, lat=2) == "Ar"


def test_tropical_wet_dry_aw():
    # Coldest ≥18 °C, pronounced dry season (>2 dry months) → Aw
    p = [0.5, 0.5, 2, 8, 15, 20, 22, 20, 15, 8, 2, 0.5]
    assert _classify_one([25] * 12, p, lat=10) == "Aw"


# ── Hemisphere symmetry ───────────────────────────────────────────────


def test_southern_hemisphere_mediterranean():
    # SH Mediterranean: dry SH summer (Oct–Mar), wet SH winter (Apr–Sep)
    # months: J F M A M J J A S O N D  (J=SH summer)
    p = [1, 0.5, 2, 8, 10, 11, 10, 9, 7, 2, 1, 0.5]
    t = [24, 24, 21, 17, 13, 10, 9, 11, 14, 18, 21, 23]
    assert _classify_one(t, p, lat=-35) == "Cs"


# ── Area percentages ──────────────────────────────────────────────────


def test_area_percent_sums_to_100():
    code = xr.DataArray(
        np.array([[1, 1, 4], [5, 8, np.nan]]),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1, 2]},
    )
    area = xr.ones_like(code)
    pct = area_percent_by_type(code, area)
    assert set(pct.keys()) == set(KT_LABELS)
    # 5 valid cells: 2×Ar, 1×BW, 1×BS, 1×Cf
    assert pct["Ar"] == pytest.approx(40.0)
    assert pct["BW"] == pytest.approx(20.0)
    assert pct["Cf"] == pytest.approx(20.0)
    assert sum(pct.values()) == pytest.approx(100.0)


def test_area_percent_empty():
    code = xr.DataArray(
        np.full((2, 2), np.nan),
        dims=("lat", "lon"), coords={"lat": [0, 1], "lon": [0, 1]},
    )
    area = xr.ones_like(code)
    pct = area_percent_by_type(code, area)
    assert all(v == 0.0 for v in pct.values())


def test_missing_input_is_nan():
    tmon, pmon, lat_da = _single_cell([np.nan] * 12, [5] * 12)
    code = classify_kt(tmon, pmon, lat_da)
    assert np.isnan(float(code.isel(lat=0, lon=0).values))


def test_area_weighting_respected():
    # Two cells of different KT type but unequal area
    code = xr.DataArray(
        np.array([[1, 4]]), dims=("lat", "lon"),
        coords={"lat": [0], "lon": [0, 1]},
    )
    area = xr.DataArray(
        np.array([[3.0, 1.0]]), dims=("lat", "lon"),
        coords={"lat": [0], "lon": [0, 1]},
    )
    pct = area_percent_by_type(code, area)
    assert pct["Ar"] == pytest.approx(75.0)
    assert pct["BW"] == pytest.approx(25.0)
