"""Tests for the ETCCDI precipitation indices and the precip_extremes diagnostic."""

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            indices=("r1mm", "prcptot", "r95p", "rx5day"), obs=None, seasons=()):
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
            "precip_extremes": {"indices": list(indices), "obs": obs or {},
                                "seasons": list(seasons)},
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

    def test_figures_closed_one_at_a_time(self, setup, monkeypatch):
        diag, *_ = setup
        diag.indices = ["r1mm", "prcptot"]
        diag.seasons = ["DJF", "JJA"]
        peak = []
        orig = diag._save

        def save(fig, meta, fid):
            peak.append(len(plt.get_fignums()))
            return orig(fig, meta, fid)

        monkeypatch.setattr(diag, "_save", save)
        plt.close("all")
        saved = diag.run()
        assert len(saved) == 2 * 3 * 4            # indices × (ann+2 seasons) × figs
        assert max(peak) == 1
        assert plt.get_fignums() == []

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


_OBS = {"ERA5": {"data_root": "/fake/era5", "experiment": "era5"},
        "MSWEP": {"data_root": "/fake/mswep", "experiment": "mswep"}}


@pytest.fixture
def setup_obs(tmp_path, monkeypatch):
    """Two EERIE families, a CMIP6 pair and ERA5 + MSWEP observations."""
    from feather.diag.precip_extremes import PrecipExtremesDiag

    hist = _FakeLoader({"A": _daily(1, "1981", "1983"),
                        "B": _daily(3, "1981", "1983", scale=2.0)})
    cmip = _FakeLoader({"C1": _daily(21, "1981", "1983")})
    obs = _FakeLoader({"ERA5": _daily(41, "1981", "1983"),
                       "MSWEP": _daily(42, "1981", "1983", scale=1.3)})
    cfg = _config(tmp_path, [("A", "FamA", "#1f77b4"), ("B", "FamB", "#d62728")],
                  cmip6=True, obs=_OBS)
    diag = PrecipExtremesDiag(hist, None, cfg)
    diag.plot_resolution = 2.0
    built = []

    def make_obs_loader(name, spec):
        built.append((name, spec["data_root"]))
        return obs

    monkeypatch.setattr(diag, "_make_obs_loader", make_obs_loader)
    monkeypatch.setattr(diag, "_discover_cmip6",
                        lambda: (cmip, [ModelConfig(name="C1")]))
    return diag, obs, built, tmp_path


class TestPrecipExtremesObs:
    def test_obs_checkpoints_and_load_period(self, setup_obs):
        diag, obs, built, tmp = setup_obs
        res = diag.compute()
        d = tmp / "out" / "precip_extremes" / "obs"
        assert sorted(p.name for p in d.glob("*.nc")) == [
            "ERA5_hist_1981_1983.nc", "MSWEP_hist_1981_1983.nc"]
        assert set(built) == {("ERA5", "/fake/era5"), ("MSWEP", "/fake/mswep")}
        assert {c[0] for c in obs.calls} == {"ERA5", "MSWEP"}
        assert all(c[3] == ("1981", "1983") for c in obs.calls)
        assert set(res["obs"]) == {"ERA5", "MSWEP"}
        # Obs are not ensemble members.
        assert "ERA5" not in res["eerie"] and "ERA5" not in res["families"].get("FamA", [])

    def test_obs_indices_use_own_percentiles(self, setup_obs):
        diag, *_ , tmp = setup_obs
        diag.compute()
        d = tmp / "out" / "precip_extremes"
        era5 = xr.open_dataset(d / "obs" / "ERA5_hist_1981_1983.nc",
                               decode_timedelta=False)
        mswep = xr.open_dataset(d / "obs" / "MSWEP_hist_1981_1983.nc",
                                decode_timedelta=False)
        # MSWEP is scaled ×1.3, so its wet-day percentiles must be higher.
        assert float(mswep["rr95"].mean()) > float(era5["rr95"].mean()) * 1.2

    def test_obs_load_period_override(self, setup_obs):
        diag, obs, *_ = setup_obs
        diag.obs_load_period = ("1981", "1982")
        diag.compute()
        assert all(c[3] == ("1981", "1982") for c in obs.calls)

    def test_bias_figures_per_obs(self, setup_obs):
        diag, *_ , tmp = setup_obs
        diag.indices = ["prcptot"]
        saved = diag.run()
        ids = sorted(p.stem for p, _ in saved)
        # No future runs in this fixture, hence no change map.
        assert ids == sorted(["prcptot_reference",
                              "prcptot_diff_cmip6", "prcptot_timeseries",
                              "prcptot_bias_era5", "prcptot_bias_mswep"])
        fdir = tmp / "out" / "figures" / "precip_extremes"
        meta = json.loads((fdir / "prcptot_bias_era5.json").read_text())
        assert meta["obs_dataset"] == "ERA5" and meta["obs_variable"] == "pr"
        stats = meta["summary_statistics"]
        assert {"FamA minus ERA5", "FamB minus ERA5", "CMIP6 MMM (1 models) minus ERA5",
                "MSWEP minus ERA5", "ERA5 (reference)"} == set(stats)
        # B is scaled ×2, A is not: B must be wetter than A relative to ERA5.
        assert (stats["FamB minus ERA5"]["global_mean_bias"]
                > stats["FamA minus ERA5"]["global_mean_bias"])
        assert stats["MSWEP minus ERA5"]["global_mean_bias"] > 0
        assert stats["FamA minus ERA5"]["rmse"] >= abs(
            stats["FamA minus ERA5"]["global_mean_bias"])

    def test_bias_against_itself_is_zero(self, tmp_path, monkeypatch):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        same = _daily(7, "1981", "1983")
        hist = _FakeLoader({"A": same})
        obs = _FakeLoader({"ERA5": same})
        cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=("rx1day",),
                      obs={"ERA5": _OBS["ERA5"]})
        diag = PrecipExtremesDiag(hist, None, cfg)
        diag.plot_resolution = 2.0
        monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: obs)
        res = diag.compute()
        _, meta = diag._plot_bias(res, "rx1day", "ERA5")
        st = meta["summary_statistics"]["FamA minus ERA5"]
        assert st["global_mean_bias"] == pytest.approx(0.0, abs=1e-5)
        assert st["rmse"] == pytest.approx(0.0, abs=1e-5)

    def test_reference_and_timeseries_include_obs(self, setup_obs):
        diag, *_ , tmp = setup_obs
        diag.indices = ["r1mm"]
        diag.run()
        fdir = tmp / "out" / "figures" / "precip_extremes"
        ref = json.loads((fdir / "r1mm_reference.json").read_text())
        assert {"ERA5", "MSWEP"} <= set(ref["summary_statistics"])
        assert ref["obs_dataset"] == "ERA5, MSWEP"
        ts = json.loads((fdir / "r1mm_timeseries.json").read_text())
        assert {"ERA5", "MSWEP"} <= set(ts["summary_statistics"])
        # The CMIP6-difference map has no observational reference.
        diff = json.loads((fdir / "r1mm_diff_cmip6.json").read_text())
        assert diff["obs_dataset"] == ""

    def test_missing_obs_dataset_skipped(self, tmp_path, monkeypatch):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({"A": _daily(1, "1981", "1983")})
        obs = _FakeLoader({"ERA5": _daily(41, "1981", "1983")})   # no MSWEP
        cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=("r1mm",),
                      obs={**_OBS, "GPCC": {}})                  # GPCC: no root
        diag = PrecipExtremesDiag(hist, None, cfg)
        diag.plot_resolution = 2.0
        monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: obs)
        saved = diag.run()
        assert sorted(p.stem for p, _ in saved) == [
            "r1mm_bias_era5", "r1mm_reference", "r1mm_timeseries"]

    def test_no_obs_configured_unchanged(self, setup):
        diag, *_ = setup
        res = diag.compute()
        assert res["obs"] == {}

    def test_save_netcdf_includes_obs(self, setup_obs):
        diag, *_ , tmp = setup_obs
        diag.indices = ["sdii"]
        diag.save_netcdf = True
        diag.run()
        ds = xr.open_dataset(tmp / "out" / "netcdf" / "precip_extremes"
                             / "sdii_1981-2032_summary.nc")
        assert {"obs_ERA5_ref", "obs_MSWEP_ref"} <= set(ds.data_vars)

    def test_make_obs_loader_reads_cmor_tree(self, tmp_path):
        """The real loader path: a CMOR day/pr tree under data_root."""
        from feather.diag.precip_extremes import PrecipExtremesDiag

        d = tmp_path / "obs" / "era5" / "r1i1p1f1" / "day" / "pr" / "gr" / "v1"
        d.mkdir(parents=True)
        _daily(5, "1981", "1981").to_dataset().to_netcdf(
            d / "pr_day_ERA5_era5_r1i1p1f1_gr_1981.nc")
        cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=("r1mm",))
        diag = PrecipExtremesDiag(_FakeLoader({}), None, cfg)
        ldr = diag._make_obs_loader(
            "ERA5", {"data_root": str(tmp_path / "obs"), "experiment": "era5"})
        da = ldr.load_var("ERA5", "pr", table="day", period=("1981", "1981"))
        assert da.sizes["time"] == 365


# ── Seasons ──────────────────────────────────────────────────────────────


class TestSeasonBlocks:
    def _year(self, year=1982):
        t = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return np.arange(len(t), dtype=np.float32)[:, None], t.month.values

    def test_mam_block_and_tail(self):
        from feather.util.precip_indices import season_blocks

        rr, months = self._year()
        (_, block, tail), = season_blocks(rr, months, ["MAM"])
        assert block.shape[0] == 92 and block[0, 0] == 59     # 1 March
        assert tail[:, 0].tolist() == [55, 56, 57, 58]        # 25–28 Feb

    def test_djf_uses_previous_december(self):
        from feather.util.precip_indices import CARRY_DAYS, season_blocks

        rr, months = self._year()
        prev = np.full((CARRY_DAYS, 1), -1.0, np.float32)
        prev[4:] = 100.0                                       # Dec of prev year
        prev_months = np.array([11] * 4 + [12] * 31)
        (_, block, tail), = season_blocks(rr, months, ["DJF"], prev, prev_months)
        assert block.shape[0] == 31 + 59
        assert (block[:31, 0] == 100.0).all() and block[31, 0] == 0   # 1 Jan
        assert tail[:, 0].tolist() == [-1.0] * 4                       # late Nov

    def test_djf_without_previous_year_is_short(self):
        from feather.util.precip_indices import season_blocks, season_min_valid_days

        rr, months = self._year()
        (_, block, tail), = season_blocks(rr, months, ["DJF"])
        assert block.shape[0] == 59 < season_min_valid_days("DJF") and tail is None

    def test_min_valid_days(self):
        from feather.util.precip_indices import season_min_valid_days

        assert [season_min_valid_days(x) for x in ("DJF", "MAM", "JJA", "SON")] == [
            86, 88, 88, 87]


def _seasonal_setup(tmp_path, seasons=("DJF", "MAM"), indices=("r1mm", "prcptot")):
    from feather.diag.precip_extremes import PrecipExtremesDiag

    hist = _FakeLoader({"A": _daily(1, "1981", "1983")})
    cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=indices,
                  seasons=seasons)
    diag = PrecipExtremesDiag(hist, None, cfg)
    diag.plot_resolution = 2.0
    return diag, hist


class TestPrecipExtremesSeasons:
    def test_seasonal_checkpoint_values(self, tmp_path):
        diag, hist = _seasonal_setup(tmp_path)
        diag.compute()
        d = tmp_path / "out" / "precip_extremes"
        assert (d / "A_hist_1981_1983.nc").exists()
        sea = xr.open_dataset(d / "A_hist_1981_1983_seasons.nc",
                              decode_timedelta=False)
        assert set(sea.data_vars) == {"r1mm_djf", "prcptot_djf",
                                      "r1mm_mam", "prcptot_mam"}
        mm = hist.series["A"] * 86400.0
        wet = (mm >= 1.0)
        mam82 = wet.sel(time=slice("1982-03-01", "1982-05-31")).sum("time")
        np.testing.assert_array_equal(sea["r1mm_mam"].sel(year=1982).values,
                                      mam82.values)
        djf82 = wet.sel(time=slice("1981-12-01", "1982-02-28")).sum("time")
        np.testing.assert_array_equal(sea["r1mm_djf"].sel(year=1982).values,
                                      djf82.values)
        # No December 1980 → DJF 1981 is missing.
        assert np.isnan(sea["r1mm_djf"].sel(year=1981).values).all()

    def test_seasons_added_to_existing_annual_checkpoint(self, tmp_path):
        diag, hist = _seasonal_setup(tmp_path, seasons=())
        diag.compute()
        ann = tmp_path / "out" / "precip_extremes" / "A_hist_1981_1983.nc"
        before = ann.stat().st_mtime_ns
        rr95 = xr.open_dataset(ann, decode_timedelta=False)["rr95"].values

        diag2, hist2 = _seasonal_setup(tmp_path, seasons=("JJA",))
        res = diag2.compute()
        assert ann.stat().st_mtime_ns == before             # annual kept
        assert len(hist2.calls) == 1                        # daily read once
        assert ann.with_name("A_hist_1981_1983_seasons.nc").exists()
        assert "r1mm_jja" in res["eerie"]["A"]["ref"]
        assert "r1mm" in res["eerie"]["A"]["ref"]
        np.testing.assert_array_equal(
            xr.open_dataset(ann, decode_timedelta=False)["rr95"].values, rr95)

        # Third run: everything from checkpoints, no daily read.
        diag3, hist3 = _seasonal_setup(tmp_path, seasons=("JJA",))
        diag3.compute()
        assert hist3.calls == []

    def test_new_season_triggers_recompute_of_seasons_only(self, tmp_path):
        diag, _ = _seasonal_setup(tmp_path, seasons=("MAM",))
        diag.compute()
        diag2, hist2 = _seasonal_setup(tmp_path, seasons=("MAM", "SON"))
        diag2.compute()
        sea = xr.open_dataset(tmp_path / "out" / "precip_extremes"
                              / "A_hist_1981_1983_seasons.nc", decode_timedelta=False)
        assert {"r1mm_mam", "r1mm_son"} <= set(sea.data_vars)
        assert len(hist2.calls) == 1

    def test_ssp_djf_uses_contiguous_hist_december(self, tmp_path, monkeypatch):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        hist = _FakeLoader({"A": _daily(1, "2012", "2014")})
        fut = _FakeLoader({"A": _daily(2, "2015", "2016")})
        cfg = _config(tmp_path, [("A", "FamA", "#000")], indices=("r1mm",),
                      seasons=("DJF",), cc_models={"A": {}})
        cfg.project["climate_change"].update({
            "reference_period": ["2012", "2014"], "future_period": ["2015", "2016"],
            "hist_load_period": ["2012", "2014"], "ssp_load_period": ["2015", "2016"]})
        diag = PrecipExtremesDiag(hist, None, cfg)
        monkeypatch.setattr(diag, "_make_fut_loader", lambda m: fut)
        diag.compute()
        sea = xr.open_dataset(tmp_path / "out" / "precip_extremes"
                              / "A_ssp_2015_2016_seasons.nc", decode_timedelta=False)
        both = xr.concat([hist.series["A"], fut.series["A"]], "time") * 86400.0
        djf15 = (both.sel(time=slice("2014-12-01", "2015-02-28")) >= 1.0).sum("time")
        np.testing.assert_array_equal(sea["r1mm_djf"].sel(year=2015).values,
                                      djf15.values)

    def test_non_contiguous_previous_segment_not_carried(self, setup):
        # Fixture: hist ends 1983, SSP starts 2031 → DJF 2031 must be missing.
        diag, *_ , tmp = setup
        diag.seasons = ["DJF"]
        diag.indices = ["r1mm"]
        diag.compute()
        sea = xr.open_dataset(tmp / "out" / "precip_extremes"
                              / "A_ssp_2031_2032_seasons.nc", decode_timedelta=False)
        assert np.isnan(sea["r1mm_djf"].sel(year=2031).values).all()
        assert np.isfinite(sea["r1mm_djf"].sel(year=2032).values).all()

    def test_seasonal_figures_and_metadata(self, setup_obs):
        diag, *_ , tmp = setup_obs
        diag.indices = ["prcptot"]
        diag.seasons = ["JJA"]
        saved = diag.run()
        ids = sorted(p.stem for p, _ in saved)
        for kind in ("reference", "diff_cmip6", "timeseries", "bias_era5",
                     "bias_mswep"):
            assert f"prcptot_{kind}" in ids
            assert f"prcptot_jja_{kind}" in ids
        meta = json.loads((tmp / "out" / "figures" / "precip_extremes"
                           / "prcptot_jja_bias_era5.json").read_text())
        assert meta["season"] == "JJA" and meta["index"] == "PRCPTOT JJA"
        assert "Mean JJA PRCPTOT" in meta["title"] or "JJA" in meta["title"]
        assert "JJA" in meta["description"]
        ann = json.loads((tmp / "out" / "figures" / "precip_extremes"
                          / "prcptot_reference.json").read_text())
        assert ann["season"] == "annual"

    def test_seasonal_percent_floor_scaled(self):
        from feather.diag.precip_extremes import PrecipExtremesDiag as D

        assert D._percent_floor("prcptot") == 10.0
        assert D._percent_floor("prcptot_djf") == 2.5
        assert D._percent_floor("rx1day_djf") == 1.0

    def test_unknown_season_ignored(self, tmp_path):
        diag, _ = _seasonal_setup(tmp_path, seasons=("JJA", "XYZ", "djf"))
        assert diag.seasons == ["DJF", "JJA"]


# ── CMIP6 member selection ───────────────────────────────────────────────


def _drs(root, activity, inst, model, exp, member, var="pr"):
    d = root / activity / inst / model / exp / member / "day" / var / "gn" / "v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{var}.nc").touch()


class TestCmip6MemberSelection:
    def test_member_with_both_experiments_is_chosen(self, tmp_path):
        from feather.data.cmip6_nc_loader import discover_daily_models

        root = tmp_path / "CMIP6"
        # r1 has historical day/pr only; r4 has both (CESM2-like).
        _drs(root, "CMIP", "NCAR", "M1", "historical", "r1i1p1f1")
        _drs(root, "CMIP", "NCAR", "M1", "historical", "r4i1p1f1")
        _drs(root, "ScenarioMIP", "NCAR", "M1", "ssp245", "r4i1p1f1")
        # Preferred member lacks day/pr, r2 has it (CAMS-CSM1-0-like).
        _drs(root, "CMIP", "CAMS", "M2", "historical", "r1i1p1f1", var="tas")
        _drs(root, "CMIP", "CAMS", "M2", "historical", "r2i1p1f1")
        _drs(root, "ScenarioMIP", "CAMS", "M2", "ssp245", "r2i1p1f1")
        # Future under another institution (MPI-ESM1-2-HR-like).
        _drs(root, "CMIP", "MPI-M", "M3", "historical", "r1i1p1f1")
        _drs(root, "ScenarioMIP", "DKRZ", "M3", "ssp245", "r1i1p1f1")
        # No member with both → dropped.
        _drs(root, "CMIP", "X", "M4", "historical", "r1i1p1f1")

        old = discover_daily_models(root, experiment="historical", table="day",
                                    variable="pr")
        assert {m.name: m.variant for m in old} == {
            "M1": "r1i1p1f1", "M3": "r1i1p1f1", "M4": "r1i1p1f1"}

        new = discover_daily_models(root, experiment="historical", table="day",
                                    variable="pr", require_also=("ssp245",))
        assert {m.name: m.variant for m in new} == {
            "M1": "r4i1p1f1", "M2": "r2i1p1f1", "M3": "r1i1p1f1"}

    def test_loader_finds_future_under_other_institution(self, tmp_path):
        from feather.data.cmip6_nc_loader import CMIP6NCLoader

        root = tmp_path / "CMIP6"
        _drs(root, "CMIP", "MPI-M", "M3", "historical", "r1i1p1f1")
        _drs(root, "ScenarioMIP", "DKRZ", "M3", "ssp245", "r1i1p1f1")
        cfg = _config(tmp_path, [("A", "FamA", "#000")])
        cfg.data_source = {"type": "cmor", "cmip6_root": str(root)}
        ldr = CMIP6NCLoader(cfg)
        mc = ModelConfig(name="M3", institution="MPI-M", variant="r1i1p1f1")
        assert "DKRZ" in str(ldr._var_dir(mc, "ssp245", "pr", "day"))
        assert "MPI-M" in str(ldr._var_dir(mc, "historical", "pr", "day"))

    def test_diag_discovery_keeps_all_three(self, tmp_path):
        from feather.diag.precip_extremes import PrecipExtremesDiag

        root = tmp_path / "CMIP6"
        _drs(root, "CMIP", "NCAR", "M1", "historical", "r1i1p1f1")
        _drs(root, "CMIP", "NCAR", "M1", "historical", "r4i1p1f1")
        _drs(root, "ScenarioMIP", "NCAR", "M1", "ssp245", "r4i1p1f1")
        _drs(root, "CMIP", "MPI-M", "M3", "historical", "r1i1p1f1")
        _drs(root, "ScenarioMIP", "DKRZ", "M3", "ssp245", "r1i1p1f1")
        cfg = _config(tmp_path, [("A", "FamA", "#000")], cmip6=True)
        cfg.cmip6_daily["root"] = str(root)
        diag = PrecipExtremesDiag(_FakeLoader({}), None, cfg)
        _, keep = diag._discover_cmip6()
        assert {m.name: m.variant for m in keep} == {"M1": "r4i1p1f1",
                                                     "M3": "r1i1p1f1"}


class TestRegistry:
    def test_registered(self):
        from feather.diag.registry import get_diagnostic
        assert get_diagnostic("precip_extremes").name == "precip_extremes"
