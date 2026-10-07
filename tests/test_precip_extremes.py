"""Tests for the ETCCDI precipitation indices and the precip_extremes diagnostic."""

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.util.precip_indices import (
    INDICES,
    WetDayPercentiles,
    annual_indices,
)


def _col(values):
    """A single-cell (n_days, 1) series."""
    return np.asarray(values, dtype=np.float32)[:, None]


# ── Index definitions ────────────────────────────────────────────────────


class TestAnnualIndices:
    def _one(self, rr, **kw):
        res = annual_indices(_col(rr), min_valid_days=1, **kw)
        return {k: float(v[0]) for k, v in res.items()}

    def test_counts_use_inclusive_thresholds(self):
        r = self._one([0.0, 0.99, 1.0, 9.99, 10.0, 19.9, 20.0, 50.0])
        assert r["r1mm"] == 6          # 1.0, 9.99, 10, 19.9, 20, 50
        assert r["r10mm"] == 4         # 10, 19.9, 20, 50
        assert r["r20mm"] == 2         # 20, 50

    def test_prcptot_ignores_dry_days(self):
        r = self._one([0.5, 0.9, 2.0, 3.0])
        assert r["prcptot"] == pytest.approx(5.0)

    def test_sdii_is_prcptot_over_wet_days(self):
        r = self._one([0.5, 2.0, 4.0, 6.0])
        assert r["sdii"] == pytest.approx(4.0)

    def test_sdii_nan_without_wet_days(self):
        assert np.isnan(self._one([0.0, 0.5, 0.9])["sdii"])

    def test_rx1day(self):
        assert self._one([1.0, 7.5, 3.0])["rx1day"] == pytest.approx(7.5)

    def test_rx5day_max_consecutive_window(self):
        rr = [1, 1, 1, 1, 1, 10, 10, 0, 0, 0, 0, 30]
        # best window: days 5-9 → 1+10+10+0+0 = 21 vs days 7-11 → 0..30 = 30
        assert self._one(rr)["rx5day"] == pytest.approx(30.0)

    def test_rx5day_uses_previous_year_tail(self):
        tail = _col([20.0, 20.0, 20.0, 20.0])
        rr = [20.0] + [0.0] * 20
        with_tail = annual_indices(_col(rr), prev_tail=tail, min_valid_days=1)
        without = annual_indices(_col(rr), min_valid_days=1)
        assert float(with_tail["rx5day"][0]) == pytest.approx(100.0)
        assert float(without["rx5day"][0]) == pytest.approx(20.0)

    def test_r95p_sums_wet_days_strictly_above_threshold(self):
        r = self._one([1.0, 5.0, 10.0, 12.0, 30.0],
                      p95=np.array([10.0]), p99=np.array([20.0]))
        assert r["r95p"] == pytest.approx(42.0)    # 12 + 30 (10 is not > 10)
        assert r["r99p"] == pytest.approx(30.0)

    def test_r95p_nan_without_thresholds(self):
        r = self._one([5.0, 6.0])
        assert np.isnan(r["r95p"]) and np.isnan(r["r99p"])

    def test_r95p_nan_where_threshold_undefined(self):
        r = self._one([5.0, 6.0], p95=np.array([np.nan]), p99=np.array([1.0]))
        assert np.isnan(r["r95p"])

    def test_cell_year_with_too_many_missing_days_is_missing(self):
        rr = np.full((365, 2), 2.0, dtype=np.float32)
        rr[:20, 1] = np.nan                        # 345 valid < 350
        res = annual_indices(rr)
        assert res["r1mm"][0] == 365
        assert all(np.isnan(res[k][1]) for k in INDICES)

    def test_missing_days_do_not_count_as_dry_or_wet(self):
        rr = [2.0] * 360 + [np.nan] * 5
        r = annual_indices(_col(rr))
        assert float(r["r1mm"][0]) == 360
        assert float(r["prcptot"][0]) == pytest.approx(720.0)

    def test_rx5day_skips_windows_with_gaps(self):
        # Treating the gap as 0 mm would make days 1-5 win with 40 mm.
        rr = [10.0, 10.0, np.nan, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        assert self._one(rr)["rx5day"] == pytest.approx(32.0)

    def test_all_indices_returned_float32(self):
        res = annual_indices(np.ones((365, 3), dtype=np.float32) * 2)
        assert list(res) == list(INDICES)
        assert all(v.dtype == np.float32 and v.shape == (3,) for v in res.values())


class TestWetDayPercentiles:
    def test_matches_numpy_percentile(self):
        rng = np.random.default_rng(1)
        rr = rng.gamma(0.4, 8.0, size=(4 * 365, 40)).astype(np.float32)
        rr[:, 0] = 0.0                                   # never wet
        rr[:, 1] = 0.0
        rr[10, 1] = 4.0                                  # one wet day
        acc = WetDayPercentiles(40, max_days=4 * 365, chunk=7)
        for y in range(4):
            acc.update(rr[y * 365:(y + 1) * 365])
        res = acc.result()
        for c in range(40):
            wet = rr[:, c][rr[:, c] >= 1.0]
            for q in (0.95, 0.99):
                got = res[q][c]
                if len(wet) == 0:
                    assert np.isnan(got)
                else:
                    assert got == pytest.approx(np.percentile(wet, q * 100), rel=1e-5)

    def test_more_days_than_declared_raises(self):
        acc = WetDayPercentiles(1, max_days=10)
        with pytest.raises(ValueError, match="max_days"):
            acc.update(np.ones((11, 1)))

    def test_cell_count_mismatch_raises(self):
        acc = WetDayPercentiles(3, max_days=10)
        with pytest.raises(ValueError, match="cells"):
            acc.update(np.ones((5, 2)))


# ── Diagnostic ───────────────────────────────────────────────────────────

_LAT = np.arange(-87.5, 90.0, 5.0)
_LON = np.arange(2.5, 360.0, 5.0)


def _daily(model_seed: int, start: str, end: str, *, scale=1.0,
           units="kg m-2 s-1") -> xr.DataArray:
    """Gamma-distributed daily pr (kg m-2 s-1 unless *units* says otherwise)."""
    time = pd.date_range(f"{start}-01-01T12:00", f"{end}-12-31T12:00", freq="D")
    rng = np.random.default_rng(model_seed)
    mm = rng.gamma(0.5, 6.0 * scale, size=(len(time), len(_LAT), len(_LON)))
    vals = mm if units == "mm/day" else mm / 86400.0
    return xr.DataArray(vals.astype(np.float32), dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": _LAT, "lon": _LON},
                        name="pr", attrs={"units": units})


class _FakeLoader:
    """load_var(model, 'pr', table='day', period=…) from in-memory series."""

    def __init__(self, series: dict[str, xr.DataArray]):
        self.series = series
        self.calls = []

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        self.calls.append((model, variable, table, period))
        assert table == "day"
        if model not in self.series:
            raise FileNotFoundError(model)
        da = self.series[model]
        if period:
            da = da.sel(time=slice(period[0], period[1]))
        return da


def _config(tmp_path, models, *, cc_models=None, cmip6=False, mode="per_family",
            indices=("r1mm", "prcptot", "r95p", "rx5day")):
    mcs = {name: ModelConfig(name=name, family=fam, grids={"sfc": "latlon"},
                             color=col)
           for name, fam, col in models}
    return FeatherConfig(
        model_catalogs={}, models=list(mcs), obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs=mcs,
        project={
            "name": "TEST", "ensemble_mode": mode,
            "precip_extremes": {"indices": list(indices)},
            "climate_change": {
                "reference_period": ["1981", "1982"],
                "future_period": ["2031", "2032"],
                "hist_load_period": ["1981", "1983"],
                "ssp_load_period": ["2031", "2032"],
                "models": cc_models or {},
            },
        },
        cmip6_daily={"enabled": cmip6, "label": "CMIP6"},
    )


_MODELS = [("A", "FamA", "#1f77b4"), ("A-r2", "FamA", "#5aabdf"),
           ("B", "FamB", "#d62728")]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from feather.diag.precip_extremes import PrecipExtremesDiag

    hist = _FakeLoader({"A": _daily(1, "1981", "1983"),
                        "A-r2": _daily(2, "1981", "1983"),
                        "B": _daily(3, "1981", "1983", scale=2.0)})
    fut = _FakeLoader({"A": _daily(11, "2031", "2032", scale=1.2),
                       "B": _daily(13, "2031", "2032", scale=2.0)})
    cmip = _FakeLoader({"C1": _daily(21, "1981", "1983"),
                        "C2": _daily(22, "1981", "1983")})
    for name, seed in (("C1", 31), ("C2", 32)):
        cmip.series[name] = xr.concat(
            [cmip.series[name], _daily(seed, "2031", "2032", scale=1.1)], "time")

    cfg = _config(tmp_path, _MODELS, cc_models={"A": {}, "B": {}}, cmip6=True)
    diag = PrecipExtremesDiag(hist, None, cfg)
    diag.plot_resolution = 2.0      # keep map rasterisation cheap in tests
    monkeypatch.setattr(diag, "_make_fut_loader",
                        lambda m: fut if m in ("A", "B") else None)
    monkeypatch.setattr(
        diag, "_discover_cmip6",
        lambda: (cmip, [ModelConfig(name="C1"), ModelConfig(name="C2")]))
    return diag, hist, fut, cmip, tmp_path


class TestPrecipExtremesCompute:
    def test_checkpoints_written_per_model_and_segment(self, setup):
        diag, *_ , tmp = setup
        diag.compute()
        d = tmp / "out" / "precip_extremes"
        names = sorted(p.name for p in d.glob("*.nc"))
        assert names == ["A-r2_hist_1981_1983.nc", "A_hist_1981_1983.nc",
                         "A_ssp_2031_2032.nc", "B_hist_1981_1983.nc",
                         "B_ssp_2031_2032.nc"]
        assert sorted(p.name for p in (d / "cmip6").glob("*.nc")) == [
            "C1_hist_1981_1983.nc", "C1_ssp_2031_2032.nc",
            "C2_hist_1981_1983.nc", "C2_ssp_2031_2032.nc"]

    def test_checkpoint_contents_and_units(self, setup):
        diag, *_ , tmp = setup
        diag.compute()
        ds = xr.open_dataset(tmp / "out" / "precip_extremes" / "A_hist_1981_1983.nc",
                             decode_timedelta=False)
        assert list(ds["year"].values) == [1981, 1982, 1983]
        assert set(INDICES) <= set(ds.data_vars)
        assert ds["r1mm"].attrs["units"] == "days"
        assert ds["prcptot"].attrs["units"] == "mm"
        assert ds["sdii"].attrs["units"] == "mm/day"
        assert ds["rr95"].attrs["units"] == "mm/day"
        # Gamma(0.5, 6) mm/day: mean 3 mm/day → PRCPTOT a few hundred mm/yr.
        assert 300 < float(ds["prcptot"].mean()) < 1500
        assert 0 < float(ds["r1mm"].mean()) < 366

    def test_indices_match_direct_computation(self, setup):
        diag, hist, *_ , tmp = setup
        diag.compute()
        ds = xr.open_dataset(tmp / "out" / "precip_extremes" / "A_hist_1981_1983.nc",
                             decode_timedelta=False)
        mm = hist.series["A"].sel(time="1982").values * 86400.0
        cell = mm[:, 10, 20]
        wet = cell[cell >= 1.0]
        assert float(ds["r1mm"].sel(year=1982)[10, 20]) == len(wet)
        assert float(ds["prcptot"].sel(year=1982)[10, 20]) == pytest.approx(
            wet.sum(), rel=1e-4)
        base = hist.series["A"].sel(time=slice("1981", "1982")).values[:, 10, 20] * 86400
        p95 = np.percentile(base[base >= 1.0], 95)
        assert float(ds["rr95"][10, 20]) == pytest.approx(p95, rel=1e-4)
        assert float(ds["r95p"].sel(year=1982)[10, 20]) == pytest.approx(
            wet[wet > p95].sum(), rel=1e-4)

    def test_future_uses_reference_thresholds(self, setup):
        diag, *_ , tmp = setup
        diag.compute()
        d = tmp / "out" / "precip_extremes"
        ssp = xr.open_dataset(d / "A_ssp_2031_2032.nc", decode_timedelta=False)
        assert "rr95" not in ssp
        # Wetter future (scale 1.2) → more rain above the present-day p95.
        hist = xr.open_dataset(d / "A_hist_1981_1983.nc", decode_timedelta=False)
        assert float(ssp["r95p"].mean()) > float(hist["r95p"].mean())

    def test_rerun_reads_checkpoints_not_daily(self, setup):
        diag, hist, fut, cmip, _ = setup
        first = diag.compute()
        n = len(hist.calls), len(fut.calls), len(cmip.calls)
        again = diag.compute()
        assert (len(hist.calls), len(fut.calls), len(cmip.calls)) == n
        # Counts read back as numbers, not timedeltas ("days" units).
        r1 = again["eerie"]["A"]["ref"]["r1mm"]
        assert np.issubdtype(r1.dtype, np.floating)
        np.testing.assert_allclose(r1.values, first["eerie"]["A"]["ref"]["r1mm"].values)

    def test_families_and_future_members(self, setup):
        diag, *_ = setup
        res = diag.compute()
        assert res["families"] == {"FamA": ["A", "A-r2"], "FamB": ["B"]}
        g = res["family"]["r1mm"]["FamA"]
        assert g["n"] == 2 and g["n_fut"] == 1       # A-r2 has no future
        assert res["mmm"]["r1mm"]["n"] == 2
        assert g["ref"].shape == (721, 1440)          # EERIE common 0.25° grid
        assert res["mmm"]["r1mm"]["ref"].shape == (180, 360)

    def test_percent_change_for_amounts_absolute_for_counts(self, setup):
        diag, *_ = setup
        res = diag.compute()
        a = res["eerie"]["A"]
        pr = res["family"]["prcptot"]["FamA"]["change"]
        expect = 100 * (a["fut"]["prcptot"] - a["ref"]["prcptot"]) / a["ref"]["prcptot"]
        np.testing.assert_allclose(pr.values, expect.values, rtol=1e-4)
        cnt = res["family"]["r1mm"]["FamA"]["change"]
        np.testing.assert_allclose(
            cnt.values, (a["fut"]["r1mm"] - a["ref"]["r1mm"]).values, rtol=1e-5)

    def test_wrong_units_model_skipped(self, tmp_path, monkeypatch):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        # Values in mm/day but labelled kg m-2 s-1 → ×86400 → absurd.
        bad = _daily(5, "1981", "1983", units="mm/day")
        bad.attrs["units"] = "kg m-2 s-1"
        hist = _FakeLoader({"A": _daily(1, "1981", "1983"), "B": bad})
        cfg = _config(tmp_path, [("A", "FamA", "#000"), ("B", "FamB", "#111")])
        diag = PrecipExtremesDiag(hist, None, cfg)
        res = diag.compute()
        assert list(res["eerie"]) == ["A"]
        assert not (tmp_path / "out" / "precip_extremes" / "B_hist_1981_1983.nc").exists()

    def test_mm_day_units_attr_not_rescaled(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({"A": _daily(1, "1981", "1983", units="mm/day")})
        diag = PrecipExtremesDiag(hist, None, _config(tmp_path, [("A", "FamA", "#000")]))
        res = diag.compute()
        assert "A" in res["eerie"]

    def test_missing_member_skipped(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({"A": _daily(1, "1981", "1983")})
        cfg = _config(tmp_path, [("A", "FamA", "#000"), ("Z", "FamZ", "#111")])
        res = PrecipExtremesDiag(hist, None, cfg).compute()
        assert list(res["eerie"]) == ["A"]

    def test_pooled_mode_single_group(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({m: _daily(i, "1981", "1983")
                            for i, (m, *_r) in enumerate(_MODELS)})
        cfg = _config(tmp_path, _MODELS, mode="pooled")
        res = PrecipExtremesDiag(hist, None, cfg).compute()
        assert res["families"] == {"EERIE": ["A", "A-r2", "B"]}

    def test_unknown_index_ignored(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        cfg = _config(tmp_path, _MODELS, indices=("r1mm", "nosuch"))
        diag = PrecipExtremesDiag(_FakeLoader({}), None, cfg)
        assert diag.indices == ["r1mm"]


class TestPrecipExtremesRun:
    def test_figures_and_sidecars(self, setup):
        diag, *_ , tmp = setup
        diag.indices = ["r1mm", "prcptot"]
        saved = diag.run()
        ids = sorted(p.stem for p, _ in saved)
        assert ids == sorted(f"{i}_{k}" for i in ("r1mm", "prcptot")
                             for k in ("reference", "change", "diff_cmip6",
                                       "timeseries"))
        meta = json.loads((tmp / "out" / "figures" / "precip_extremes"
                           / "prcptot_change.json").read_text())
        assert meta["units"] == "%"
        assert meta["group"] == "precipitation_extremes"
        assert meta["obs_dataset"] == "" and meta["obs_variable"] == ""
        ref = json.loads((tmp / "out" / "figures" / "precip_extremes"
                          / "r1mm_reference.json").read_text())
        assert ref["units"] == "days"

    def test_save_netcdf_summary(self, setup):
        diag, *_ , tmp = setup
        diag.indices = ["rx5day"]
        diag.save_netcdf = True
        diag.run()
        ds = xr.open_dataset(tmp / "out" / "netcdf" / "precip_extremes"
                             / "rx5day_1981-2032_summary.nc")
        assert {"family_FamA_ref", "family_FamA_change", "cmip6_mmm_ref",
                "cmip6_mmm_change"} <= set(ds.data_vars)
        assert ds.attrs["change_units"] == "%"

    def test_no_change_figure_without_future(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({"A": _daily(1, "1981", "1983")})
        cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=("r1mm",))
        diag = PrecipExtremesDiag(hist, None, cfg)
        diag.plot_resolution = 2.0
        saved = diag.run()
        assert sorted(p.stem for p, _ in saved) == ["r1mm_reference", "r1mm_timeseries"]


class TestRegistry:
    def test_registered(self):
        from feather.diag.registry import get_diagnostic
        assert get_diagnostic("precip_extremes").name == "precip_extremes"
