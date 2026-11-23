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


# ── CMIP6 benchmark + reuse of saved climatologies ──────────────────────

class _Cfg:
    def __init__(self, out, cmip6_enabled=False):
        self.output_dir = str(out)
        self.models = ["M1", "M2"]
        self.model_configs = {}
        self.project = {}
        self.obs_datasets = {}
        self.obs_root = ""
        self.nereus = {"resolution": 30.0}
        self.cmip6 = {"enabled": cmip6_enabled}

    def get_model_families(self, models):
        return {"M": list(models)}

    def get_ensemble_mode(self):
        return "per_family"

    def get_model_data_source_type(self, model):
        return "cmor"


class _Bench:
    def __init__(self, label, models=("CM1",)):
        self.label = label
        self.models = {m: {} for m in models}
        self.calls = []

    def load_var(self, var, model, **kw):
        self.calls.append((var, model))
        return None


class _NoLoad:
    """Model/obs loader that fails the test if real data is requested."""
    def __getattr__(self, name):
        raise AssertionError(f"data loader used: {name}")


def _kt_diag(tmp_path, *, benchmarks=None, cmip6_loader=None, enabled=False):
    from feather.diag.climate_classification import KTClimateClassification
    d = KTClimateClassification(
        _NoLoad(), _NoLoad(), _Cfg(tmp_path, enabled),
        cmip6_loader=cmip6_loader, benchmarks=benchmarks,
        period=("1980", "2014"),
    )
    d._region = ""
    return d


def _grid():
    lat = np.array([-75.0, -45.0, -15.0, 15.0, 45.0, 75.0])
    lon = np.arange(15.0, 360.0, 30.0)
    return lat, lon


def _clim_fields(lat, lon, t_off=0.0):
    month = np.arange(1, 13)
    t = 25.0 - 0.4 * np.abs(lat)[None, :, None] + t_off + 0 * lon[None, None, :] \
        + 0 * month[:, None, None]
    p = np.full((12, len(lat), len(lon)), 8.0)
    coords = {"month": month, "lat": lat, "lon": lon}
    return (xr.DataArray(t, dims=("month", "lat", "lon"), coords=coords),
            xr.DataArray(p, dims=("month", "lat", "lon"), coords=coords))


def _prime(d, sources):
    lat, lon = _grid()
    d._tlat, d._tlon = lat, lon
    d._land_mask = xr.DataArray(np.ones((len(lat), len(lon)), bool),
                                dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    d.nc_dir.mkdir(parents=True, exist_ok=True)
    for src, off in sources.items():
        t, p = _clim_fields(lat, lon, off)
        d._save_clim_nc(src, t if src != "MSWEP" else None,
                        p if src != "Berkeley Earth HR" else None)
    d._tlat = d._tlon = None


def test_cmip6_source_selection(tmp_path):
    cm, hr = _Bench("CMIP6 MMM"), _Bench("HighResMIP MMM")
    assert _kt_diag(tmp_path, benchmarks=[hr, cm], cmip6_loader=hr)._cmip6_source() is cm
    assert _kt_diag(tmp_path, benchmarks=[hr], cmip6_loader=hr)._cmip6_source() is None
    legacy = _Bench("CMIP6 MMM")
    assert _kt_diag(tmp_path, cmip6_loader=legacy, enabled=True)._cmip6_source() is legacy
    assert _kt_diag(tmp_path, cmip6_loader=legacy)._cmip6_source() is None


def test_run_passes_benchmarks_to_constructor():
    import inspect
    from feather.diag.climate_classification import KTClimateClassification
    assert "benchmarks" in inspect.signature(KTClimateClassification.__init__).parameters


def test_cached_clim_roundtrip_and_guards(tmp_path):
    d = _kt_diag(tmp_path)
    _prime(d, {"M1": 0.0})
    d._reused_clims = set()
    t, p = d._cached_clim("M1")
    lat, lon = _grid()
    t0, p0 = _clim_fields(lat, lon)
    np.testing.assert_allclose(t.values, t0.values, rtol=1e-6)
    np.testing.assert_allclose(p.values, p0.values, rtol=1e-6)   # back to cm/month
    np.testing.assert_array_equal(d._tlat, lat)
    assert "M1" in d._reused_clims
    assert d._cached_clim("missing") == (None, None)
    d._reuse_cache = False                          # --no-skip-existing
    assert d._cached_clim("M1") == (None, None)
    d._reuse_cache = True
    d._res = 1.0                                    # resolution changed
    d._tlat = d._tlon = None
    assert d._cached_clim("M1") == (None, None)


def test_cache_without_cmip6_mmm_is_refused(tmp_path):
    cm = _Bench("CMIP6 MMM")
    d = _kt_diag(tmp_path, benchmarks=[cm], cmip6_loader=cm)
    _prime(d, {})
    lat, lon = _grid()
    code = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})
    d._save_nc("M1", code)
    d._write_csv({"M1": {lbl: 0.0 for lbl in KT_LABELS}})
    assert d._load_cached_results() is None
    assert _kt_diag(tmp_path)._load_cached_results() is not None   # no benchmark


def test_compute_reuses_saved_climatologies(tmp_path):
    cm = _Bench("CMIP6 MMM", models=("CM1", "CM2"))
    d = _kt_diag(tmp_path, benchmarks=[cm], cmip6_loader=cm)
    _prime(d, {"M1": 0.0, "M2": 1.0, "ERA5": 0.0, "Berkeley Earth HR": 0.5,
               "MSWEP": 0.0, "CMIP6 CM1": -1.0})
    res = d.compute()                    # _NoLoad raises if data is read
    assert set(res["codes"]) == {"M1", "M2", "ERA5", "BE-HR + MSWEP", "CMIP6 MMM"}
    assert cm.calls == [("tas", "CM2"), ("pr", "CM2")]   # only the uncached member
    assert d._nc_path("CMIP6 MMM").exists()
    assert d._clim_path("CMIP6 MMM").exists()
    assert {"M1", "M2", "ERA5", "Berkeley Earth HR", "MSWEP",
            "CMIP6 CM1"} <= d._reused_clims


def _capture_provenance(monkeypatch):
    from feather import provenance
    events = []
    monkeypatch.setattr(provenance, "current_run", lambda: object())
    monkeypatch.setattr(provenance, "emit",
                        lambda step, **f: events.append({"step": step, **f}))
    return events


def test_reused_cmip6_members_are_recorded(tmp_path, monkeypatch):
    events = _capture_provenance(monkeypatch)
    cm = _Bench("CMIP6 MMM", models=("CM1", "CM2"))
    cm.models = {"CM1": {"variants": ["r1i1p1f1"]}, "CM2": {}}
    d = _kt_diag(tmp_path, benchmarks=[cm], cmip6_loader=cm)
    _prime(d, {"M1": 0.0, "M2": 1.0, "ERA5": 0.0, "Berkeley Earth HR": 0.5,
               "MSWEP": 0.0, "CMIP6 CM1": -1.0})
    res = d.compute()
    assert res["cmip6_members"] == ["CM1"]     # CM2 has no data (_Bench → None)
    rec = [e for e in events if e["step"] == "benchmark_member"]
    assert {(e["model"], e["variable"]) for e in rec} == {("CM1", "tas"), ("CM1", "pr")}
    assert all(e["status"] == "used" and e["source"] == "saved_climatology"
               and e["variant"] == "r1i1p1f1" and e["benchmark"] == "CMIP6 MMM"
               for e in rec)
    with xr.open_dataset(d._nc_path("CMIP6 MMM")) as ds:
        assert ds.attrs["cmip6_members"] == "CM1"

    # Replotting from the cached classification re-records the members.
    events.clear()
    cached = _kt_diag(tmp_path, benchmarks=[cm], cmip6_loader=cm)
    cached._reuse_cache = True
    out = cached._load_cached_results()
    assert out is not None and out["cmip6_members"] == ["CM1"]
    assert {(e["model"], e["variable"]) for e in events} == {("CM1", "tas"), ("CM1", "pr")}


def test_cached_mmm_without_member_list_is_refused(tmp_path):
    cm = _Bench("CMIP6 MMM")
    d = _kt_diag(tmp_path, benchmarks=[cm], cmip6_loader=cm)
    _prime(d, {})
    lat, lon = _grid()
    code = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})
    d._save_nc("M1", code)
    d._save_nc("CMIP6 MMM", code)                 # old cache: no member list
    d._write_csv({"M1": {lbl: 0.0 for lbl in KT_LABELS}})
    assert d._load_cached_results() is None
    d._save_nc("CMIP6 MMM", code, extra_attrs={"cmip6_members": "CM1,CM2"})
    assert d._load_cached_results()["cmip6_members"] == ["CM1", "CM2"]


def test_member_counts_in_stats():
    res = _stats_results()
    res["codes"]["CMIP6 MMM"] = res["codes"]["M2"]
    res["area_pct"]["CMIP6 MMM"] = res["area_pct"]["M2"]
    res["cmip6_members"] = ["A", "B", "C"]
    res["ensembles"] = {"F": {"mean": "F Mean", "members": ["M1", "M2"]}}
    res["codes"]["F Mean"] = res["codes"]["M1"]
    res["area_pct"]["F Mean"] = res["area_pct"]["M1"]
    d = _stats_diag()
    agree = d._agreement_stats(res, list(res["codes"].items()))
    area = d._area_stats(res, list(res["codes"]))
    for st in (agree, area):
        assert st["CMIP6 MMM"]["n_members"] == 3
        assert st["F Mean"]["n_members"] == 2
        assert "n_members" not in st["M1"]


def test_ensemble_maps_show_single_member_family(tmp_path, monkeypatch):
    from feather.diag.climate_classification import KTClimateClassification
    cfg = _Cfg(tmp_path)
    cfg.models = ["M1", "M2", "H1"]
    cfg.get_model_families = lambda models: {
        k: [m for m in v if m in models]
        for k, v in {"M": ["M1", "M2"], "H": ["H1"]}.items()
        if any(m in models for m in v)}
    d = _kt_diag(tmp_path)
    d.config = cfg
    res = _stats_results()
    res["codes"]["H1"] = res["codes"]["M2"]
    res["models"] = ["M1", "M2", "H1"]
    res["ensembles"] = {"M": {"mean": "M Mean", "members": ["M1", "M2"]}}
    res["ensemble_codes"] = {"M Mean": res["codes"]["M1"]}
    seen = {}
    monkeypatch.setattr(KTClimateClassification, "_render_kt_maps",
                        lambda self, panels, *a: seen.setdefault("p", panels))
    monkeypatch.setattr(KTClimateClassification, "_build_metadata",
                        lambda self, **kw: kw)
    _, meta = d._plot_ensemble_maps(res)
    assert [n for n, _ in seen["p"]] == ["ERA5", "M Mean", "H1"]
    assert "H1" in meta["summary_statistics"]
    assert "(H1) are shown as that member" in meta["description"]


def test_regrid_monthly_unstructured_points(tmp_path):
    """ICON-ESM-LR-style field: one cell dim with 1-D lat/lon coords."""
    d = _kt_diag(tmp_path)
    rng = np.random.default_rng(0)
    n = 4000
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    vals = np.broadcast_to(20.0 - 0.3 * np.abs(lat), (12, n)).copy()
    clim = xr.DataArray(vals, dims=("month", "i"),
                        coords={"month": np.arange(1, 13),
                                "latitude": ("i", lat), "longitude": ("i", lon)})
    out = d._regrid_monthly(clim, 2_000_000.0)
    assert out.dims == ("month", "lat", "lon")
    assert out.shape[0] == 12
    eq = float(out.isel(month=0).sel(lat=15.0, method="nearest").mean())
    assert abs(eq - (20.0 - 0.3 * 15.0)) < 3.0
