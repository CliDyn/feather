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


# ── Figure summary_statistics (sidecar numbers for LLM / audit) ─────────

def _stats_diag():
    from feather.diag.climate_classification import KTClimateClassification
    return KTClimateClassification.__new__(KTClimateClassification)


def _stats_results():
    lat = np.array([-10.0, 10.0])
    lon = np.array([0.0, 10.0, 20.0])
    def da(v):
        return xr.DataArray(np.array(v, dtype=float), dims=("lat", "lon"),
                            coords={"lat": lat, "lon": lon})
    codes = {
        "ERA5": da([[1, 1, 2], [3, 3, np.nan]]),
        "M1": da([[1, 1, 2], [3, 3, np.nan]]),        # identical
        "M2": da([[2, 2, 2], [2, 2, np.nan]]),        # agrees on 1 of 5 cells
    }
    from feather.util.spatial import compute_latlon_areas
    area = xr.DataArray(compute_latlon_areas(lat, lon), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})
    area_pct = {k: area_percent_by_type(v, area) for k, v in codes.items()}
    return {"codes": codes, "area_pct": area_pct, "lat": lat, "lon": lon,
            "models": ["M1", "M2"]}


def test_area_stats_pct_and_minus_era5():
    res = _stats_results()
    stats = _stats_diag()._area_stats(res, ["ERA5", "M1", "M2"])
    assert set(stats) == {"ERA5", "M1", "M2"}
    assert not any(k.endswith("_minus_ERA5_pct") for k in stats["ERA5"])
    assert sum(v for k, v in stats["M2"].items() if k.endswith("_pct")
               and "_minus_" not in k) == pytest.approx(100.0, abs=0.01)
    for lbl in KT_LABELS:
        assert stats["M1"][f"{lbl}_minus_ERA5_pct"] == pytest.approx(0.0)
        assert stats["M2"][f"{lbl}_minus_ERA5_pct"] == pytest.approx(
            res["area_pct"]["M2"][lbl] - res["area_pct"]["ERA5"][lbl], abs=1e-3)


def test_area_stats_without_era5_has_no_differences():
    res = _stats_results()
    del res["area_pct"]["ERA5"]
    stats = _stats_diag()._area_stats(res, ["M1"])
    assert not any("_minus_" in k for k in stats["M1"])


def test_agreement_stats():
    res = _stats_results()
    panels = [(k, v) for k, v in res["codes"].items()]
    stats = _stats_diag()._agreement_stats(res, panels)
    assert "ERA5" not in stats
    assert stats["M1"]["agreement_with_ERA5_pct"] == pytest.approx(100.0)
    assert stats["M1"]["classified_land_pct_of_ERA5"] == pytest.approx(100.0)
    # All cells have equal area here (symmetric lats): 1 of 5 agree.
    assert stats["M2"]["agreement_with_ERA5_pct"] == pytest.approx(20.0, abs=0.01)


def test_agreement_stats_empty_without_era5():
    res = _stats_results()
    del res["codes"]["ERA5"]
    assert _stats_diag()._agreement_stats(res, list(res["codes"].items())) == {}
