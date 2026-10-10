"""Tests for the ETCCDI temperature indices and the temp_extremes_indices diagnostic."""

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.util.temp_indices import (
    INDICES,
    SEASONAL_INDICES,
    annual_indices,
    calendar_day_percentiles,
    calendar_doy,
    growing_season_length,
)
from feather.util.temp_indices import _spell_days


def _col(values):
    """A single-cell (n_days, 1) series."""
    return np.asarray(values, dtype=np.float32)[:, None]


# ── Index definitions ────────────────────────────────────────────────────


class TestAnnualIndices:
    def _one(self, tx=None, tn=None, thresholds=None, doy=None):
        n = len(tx if tx is not None else tn)
        res = annual_indices(
            None if tx is None else _col(tx), None if tn is None else _col(tn),
            np.arange(n) if doy is None else np.asarray(doy),
            thresholds=thresholds, min_valid_days=1)
        return {k: float(v[0]) for k, v in res.items()}

    def test_extremes_counts_and_dtr(self):
        r = self._one(tx=[-2.0, 5.0, 30.0], tn=[-5.0, -1.0, 10.0])
        assert (r["txx"], r["txn"], r["tnx"], r["tnn"]) == (30.0, -2.0, 10.0, -5.0)
        assert r["id"] == 1 and r["fd"] == 2
        assert r["dtr"] == pytest.approx((3 + 6 + 20) / 3)

    def test_frost_and_ice_strictly_below_zero(self):
        r = self._one(tx=[0.0, -0.1], tn=[0.0, -0.1])
        assert r["id"] == 1 and r["fd"] == 1

    def test_percentiles_use_the_calendar_day_threshold(self):
        tx = [10.0, 10.0, 10.0, 10.0]
        thr = {"tx90": _col([9.0, 11.0, 9.0, 10.0]),     # per calendar day
               "tx10": _col([11.0, 9.0, 9.0, 9.0])}
        r = self._one(tx=tx, thresholds=thr)
        assert r["tx90p"] == pytest.approx(50.0)          # days 0 and 2; day 3 equal
        assert r["tx10p"] == pytest.approx(25.0)          # day 0 only

    def test_threshold_row_follows_doy(self):
        thr = {"tn10": _col([0.0, 100.0])}
        r = self._one(tn=[5.0, 5.0], thresholds=thr, doy=[1, 1])
        assert r["tn10p"] == pytest.approx(100.0)

    def test_percentile_indices_absent_without_thresholds(self):
        r = self._one(tx=[1.0, 2.0])
        assert "tx90p" not in r and "wsdi" not in r

    def test_missing_variable_leaves_its_indices_out(self):
        r = self._one(tx=[1.0, 2.0])
        assert set(r) == {"txx", "txn", "id"}

    def test_wsdi_counts_days_in_runs_of_six(self):
        above = [1] * 7 + [0] + [1] * 5 + [0] + [1] * 6
        tx = [20.0 if a else 0.0 for a in above]
        thr = {"tx90": np.full((len(tx), 1), 10.0, np.float32)}
        assert self._one(tx=tx, thresholds=thr)["wsdi"] == 13

    def test_csdi_counts_cold_runs(self):
        below = [0] + [1] * 8 + [0] * 3
        tn = [-20.0 if b else 0.0 for b in below]
        thr = {"tn10": np.full((len(tn), 1), -10.0, np.float32)}
        assert self._one(tn=tn, thresholds=thr)["csdi"] == 8

    def test_too_few_valid_days_is_missing(self):
        res = annual_indices(_col([1.0, np.nan, np.nan]), None, np.arange(3),
                             min_valid_days=2)
        assert np.isnan(res["txx"][0]) and np.isnan(res["id"][0])

    def test_missing_days_do_not_count(self):
        r = self._one(tx=[-1.0, np.nan, 3.0])
        assert r["id"] == 1 and r["txn"] == -1.0

    def test_float32_outputs(self):
        res = annual_indices(_col([1, 2]), _col([0, 1]), np.arange(2),
                             min_valid_days=1)
        assert all(v.dtype == np.float32 for v in res.values())


class TestSpellDays:
    @pytest.mark.parametrize("flags, expected", [
        ([1] * 6, 6), ([1] * 5, 0), ([0, 1, 1, 1, 1, 1, 1, 1, 0], 7),
        ([1] * 6 + [0] + [1] * 6, 12), ([1, 1, 0] * 4, 0),
    ])
    def test_runs(self, flags, expected):
        assert _spell_days(np.asarray(flags, bool)[:, None])[0] == expected

    def test_short_series(self):
        assert _spell_days(np.ones((3, 2), bool)).tolist() == [0, 0]


class TestGrowingSeasonLength:
    def _gsl(self, tg, mid=181):
        return float(growing_season_length(_col(tg), mid, 1)[0])

    def test_warm_then_cold_after_mid(self):
        tg = [0.0] * 100 + [10.0] * 150 + [0.0] * 115
        assert self._gsl(tg) == 150

    def test_never_warm_is_zero(self):
        assert self._gsl([0.0] * 365) == 0

    def test_always_warm_is_full_year(self):
        assert self._gsl([20.0] * 365) == 365

    def test_cold_run_before_mid_is_ignored(self):
        tg = [10.0] * 50 + [0.0] * 10 + [10.0] * 200 + [0.0] * 105
        assert self._gsl(tg) == 260

    def test_short_warm_spells_do_not_start_season(self):
        tg = ([10.0] * 5 + [0.0]) * 20 + [10.0] * 100 + [0.0] * 145
        assert self._gsl(tg) == 100

    def test_missing_days_break_runs(self):
        tg = [0.0] * 100 + [10.0, 10.0, np.nan] * 2 + [0.0] * 259
        assert self._gsl(tg) == 0


class TestCalendar:
    def test_feb29_shares_feb28(self):
        doy, n = calendar_doy([2, 2, 3, 12], [28, 29, 1, 31])
        assert n == 365 and doy.tolist() == [58, 58, 59, 364]

    def test_360_day(self):
        doy, n = calendar_doy([1, 12], [1, 30], "360_day")
        assert n == 360 and doy.tolist() == [0, 359]

    def test_percentiles_match_numpy_window(self):
        rng = np.random.default_rng(0)
        years, n_doy = 4, 10
        doy = np.tile(np.arange(n_doy), years)
        x = rng.normal(size=(len(doy), 3)).astype(np.float32)
        q = calendar_day_percentiles(x, doy, n_doy, quantiles=(0.1, 0.9))
        # day 0 window wraps to days 8, 9, 0, 1, 2
        rows = np.flatnonzero(np.isin(doy, [8, 9, 0, 1, 2]))
        ref = np.quantile(x[rows], [0.1, 0.9], axis=0, method="median_unbiased")
        np.testing.assert_allclose(q[0.1][0], ref[0], rtol=1e-6)
        np.testing.assert_allclose(q[0.9][0], ref[1], rtol=1e-6)

    def test_percentiles_ignore_missing_days(self):
        doy = np.zeros(10, int)
        x = np.arange(10, dtype=np.float32)[:, None].repeat(2, 1)
        x[0, 1] = np.nan
        q = calendar_day_percentiles(x, doy, 1, quantiles=(0.9,), window=1)
        ref = np.quantile(np.arange(1, 10), 0.9, method="median_unbiased")
        assert q[0.9][0, 1] == pytest.approx(ref)


# ── ICON native-grid fallback ────────────────────────────────────────────


def _scrip(path, n_src=4):
    """Weights: 2×3 lat-lon target, each target the mean of two sources."""
    nlat, nlon = 2, 3
    dst = np.repeat(np.arange(nlat * nlon) + 1, 2)
    src = (np.arange(nlat * nlon * 2) % n_src) + 1
    lat = np.radians(np.repeat([-45.0, 45.0], nlon))
    lon = np.radians(np.tile([0.0, 120.0, 240.0], nlat))
    xr.Dataset({
        "src_address": ("num_links", src.astype(np.int32)),
        "dst_address": ("num_links", dst.astype(np.int32)),
        "remap_matrix": (("num_links", "num_wgts"), np.full((len(dst), 1), 0.5)),
        "dst_grid_dims": ("dst_grid_rank", np.array([nlon, nlat], np.int32)),
        "dst_grid_center_lat": ("dst_grid_size", lat),
        "dst_grid_center_lon": ("dst_grid_size", lon),
        "src_grid_center_lat": ("src_grid_size", np.zeros(n_src)),
    }).to_netcdf(path)


class TestIconNativeFallback:
    def _loader(self, tmp_path, monkeypatch, remapped=False):
        from feather.data import icon_kerchunk_loader as mod

        root = tmp_path / "v1"
        (root / "2" / mod._STORE_FILES["atmos2d_daymax_native"]).mkdir(parents=True)
        if remapped:
            (root / "2" / mod._STORE_FILES["atmos2d_daymax"]).mkdir()
        _scrip(tmp_path / "w.nc")
        mc = ModelConfig(name="I-r2", data_root=str(root), member=2,
                         grids={"sfc": "latlon"})
        cfg = FeatherConfig(model_catalogs={}, models=["I-r2"], obs_root="",
                            obs_datasets={}, cmip6={}, dask={}, nereus={},
                            output_dir=str(tmp_path),
                            data_source={"type": "cmor",
                                         "icon_remap_weights": str(tmp_path / "w.nc")},
                            model_configs={"I-r2": mc}, project={})
        time = pd.date_range("1981-01-01T23:59:59", periods=3, freq="D")
        native = xr.Dataset({"tas": (("time", "height", "ncells"),
                                     np.arange(12, dtype=np.float32).reshape(3, 1, 4)
                                     + 270.0)},
                            coords={"time": time, "height": [2.0]})
        ldr = mod.ICONKerchunkLoader(cfg)
        opened = []

        def fake_open(model, store):
            opened.append(store)
            return native.chunk({"time": 1})
        monkeypatch.setattr(ldr, "_open_store", fake_open)
        mod._remap_weights.cache_clear()
        return ldr, opened

    def test_regrids_native_when_remap_store_missing(self, tmp_path, monkeypatch):
        ldr, opened = self._loader(tmp_path, monkeypatch)
        da = ldr.load_var("I-r2", "tasmax", table="day")
        assert opened == ["atmos2d_daymax_native"]
        assert da.dims == ("time", "lat", "lon") and da.shape == (3, 2, 3)
        np.testing.assert_allclose(da["lat"], [-45.0, 45.0])
        np.testing.assert_allclose(da["lon"], [0.0, 120.0, 240.0])
        # target k averages sources (2k % 4, 2k+1 % 4); day 0 sources are 270..273
        expected = np.array([270.5, 272.5, 270.5, 272.5, 270.5, 272.5]).reshape(2, 3)
        np.testing.assert_allclose(da.isel(time=0).values, expected)
        assert da.attrs["units"] == "" or da.attrs["regridded_from"]

    def test_native_stamps_centred_on_their_own_day(self, tmp_path, monkeypatch):
        ldr, _ = self._loader(tmp_path, monkeypatch)
        da = ldr.load_var("I-r2", "tasmax", table="day")
        assert str(da["time"].values[0])[:19] == "1981-01-01T12:00:00"

    def test_remapped_store_preferred(self, tmp_path, monkeypatch):
        ldr, opened = self._loader(tmp_path, monkeypatch, remapped=True)
        assert ldr._native_fallback("I-r2", "atmos2d_daymax") is None

    def test_no_native_store_raises(self, tmp_path):
        from feather.data import icon_kerchunk_loader as mod

        mc = ModelConfig(name="I", data_root=str(tmp_path), member=2)
        cfg = FeatherConfig(model_catalogs={}, models=["I"], obs_root="",
                            obs_datasets={}, cmip6={}, dask={}, nereus={},
                            output_dir=str(tmp_path), data_source={},
                            model_configs={"I": mc}, project={})
        with pytest.raises(FileNotFoundError):
            mod.ICONKerchunkLoader(cfg).load_var("I", "tasmax", table="day")


# ── Diagnostic ───────────────────────────────────────────────────────────

_LAT = np.arange(-75.0, 90.0, 15.0)        # 11 rows, south and north
_LON = np.arange(0.0, 360.0, 30.0)         # 12 columns


def _temps(seed, start, end, *, warm=0.0, units="K", lat=_LAT, lon=_LON):
    """Daily (tasmax, tasmin) with a seasonal cycle, latitude gradient and noise."""
    time = pd.date_range(f"{start}-01-01T12:00", f"{end}-12-31T12:00", freq="D")
    rng = np.random.default_rng(seed)
    doy = time.dayofyear.values[:, None, None]
    la = np.asarray(lat)[None, :, None]
    season = -np.sign(la + 1e-9) * 12.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    base = 27.0 - 0.5 * np.abs(la) + season + warm
    noise = rng.normal(0.0, 3.0, size=(len(time), len(lat), len(lon)))
    tx = base + 5.0 + noise
    tn = base - 5.0 + noise + rng.normal(0.0, 1.0, size=noise.shape)
    off = 273.15 if units == "K" else 0.0
    out = {}
    for name, v in (("tasmax", tx), ("tasmin", tn)):
        out[name] = xr.DataArray((v + off).astype(np.float32),
                                 dims=("time", "lat", "lon"),
                                 coords={"time": time, "lat": lat, "lon": lon},
                                 name=name, attrs={"units": units})
    return out


class _FakeLoader:
    def __init__(self, series):
        self.series = series                 # {model: {var: DataArray}}
        self.calls = []

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        self.calls.append((model, variable, table, period))
        if table != "day" or variable not in self.series.get(model, {}):
            raise FileNotFoundError(f"{model}/{variable}/{table}")
        da = self.series[model][variable]
        return da.sel(time=slice(period[0], period[1])) if period else da


def _land_mask_file(tmp_path, lat=_LAT, lon=_LON):
    """ERA5-style sftlf (%): land west of 180°E."""
    frac = np.where(np.asarray(lon)[None, :] < 180.0, 100.0, 0.0) * np.ones((len(lat), 1))
    d = tmp_path / "sftlf"
    d.mkdir(exist_ok=True)
    xr.Dataset({"sftlf": (("lat", "lon"), frac.astype(np.float32))},
               coords={"lat": lat, "lon": lon}).to_netcdf(d / "sftlf.nc")
    return {"path": str(d), "variables": {"sftlf": "sftlf.nc"}}


def _config(tmp_path, models, *, cc_models=None, cmip6=False, indices=None,
            obs=None, seasons=(), mask=True):
    mcs = {name: ModelConfig(name=name, family=fam, grids={"sfc": "latlon"},
                             color=col) for name, fam, col in models}
    te = {"obs": obs or {}, "seasons": list(seasons)}
    if indices is not None:
        te["indices"] = list(indices)
    return FeatherConfig(
        model_catalogs={}, models=list(mcs), obs_root="",
        obs_datasets={"ERA5_SFTLF": _land_mask_file(tmp_path)} if mask else {},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs=mcs,
        project={
            "name": "TEST", "ensemble_mode": "per_family",
            "temp_extremes_indices": te,
            "climate_change": {
                "reference_period": ["1981", "1983"],
                "future_period": ["2031", "2032"],
                "hist_load_period": ["1981", "1984"],
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
    from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

    hist = _FakeLoader({"A": _temps(1, "1981", "1984"),
                        "A-r2": _temps(2, "1981", "1984"),
                        # B publishes tasmax only.
                        "B": {"tasmax": _temps(3, "1981", "1984")["tasmax"]}})
    fut = _FakeLoader({"A": _temps(11, "2031", "2032", warm=3.0)})
    c1 = _temps(21, "1981", "1984")
    c1f = _temps(31, "2031", "2032", warm=2.0)
    cmip = _FakeLoader({"C1": {v: xr.concat([c1[v], c1f[v]], "time") for v in c1}})
    obs = _FakeLoader({"ERA5": _temps(41, "1981", "1984")})
    cfg = _config(tmp_path, _MODELS, cc_models={"A": {}}, cmip6=True,
                  obs={"ERA5": {"data_root": "/fake", "experiment": "era5"}})
    diag = TempExtremesIndicesDiag(hist, None, cfg)
    diag.plot_resolution = 5.0
    monkeypatch.setattr(diag, "_make_fut_loader", lambda m: fut if m == "A" else None)
    monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: obs)
    monkeypatch.setattr(diag, "_discover_cmip6",
                        lambda: (cmip, [ModelConfig(name="C1")]))
    return diag, hist, fut, cmip, obs, tmp_path


def _ckpt(tmp, *parts):
    return xr.open_dataset(tmp.joinpath("out", "temp_extremes_indices", *parts),
                           decode_timedelta=False)


class TestTempExtremesCompute:
    def test_checkpoints_per_model_and_segment(self, setup):
        diag, *_, tmp = setup
        diag.compute()
        d = tmp / "out" / "temp_extremes_indices"
        assert sorted(p.name for p in d.glob("*.nc")) == [
            "A-r2_hist_1981_1984.nc", "A_hist_1981_1984.nc", "A_ssp_2031_2032.nc",
            "B_hist_1981_1984.nc"]
        assert sorted(p.name for p in (d / "cmip6").glob("*.nc")) == [
            "C1_hist_1981_1984.nc", "C1_ssp_2031_2032.nc"]
        assert [p.name for p in (d / "obs").glob("*.nc")] == ["ERA5_hist_1981_1984.nc"]

    def test_all_indices_and_land_only(self, setup):
        diag, *_, tmp = setup
        diag.compute()
        ds = _ckpt(tmp, "A_hist_1981_1984.nc")
        assert set(INDICES) <= set(ds.data_vars)
        land = ds["lon"] < 180.0
        txx = ds["txx"].isel(year=1)
        assert bool(txx.where(land).notnull().sum() == land.sum() * ds.sizes["lat"])
        assert bool(txx.where(~land).isnull().all())
        assert ds["txx"].attrs["units"] == "°C"

    def test_kelvin_converted_to_celsius(self, setup):
        diag, hist, *_, tmp = setup
        diag.compute()
        ds = _ckpt(tmp, "A_hist_1981_1984.nc")
        tx = hist.series["A"]["tasmax"].sel(time="1982") - 273.15
        ref = tx.max("time").where(tx["lon"] < 180.0)
        np.testing.assert_allclose(ds["txx"].sel(year=1982).values, ref.values,
                                   rtol=1e-5, equal_nan=True)

    def test_celsius_input_not_shifted(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        k = _temps(1, "1981", "1983")
        c = _temps(1, "1981", "1983", units="degC")
        cfg = _config(tmp_path, [("K", "F", "#000"), ("C", "G", "#111")],
                      indices=("txx",))
        diag = TempExtremesIndicesDiag(_FakeLoader({"K": k, "C": c}), None, cfg)
        res = diag.compute()
        np.testing.assert_allclose(res["eerie"]["K"]["ref"]["txx"],
                                   res["eerie"]["C"]["ref"]["txx"], rtol=1e-4)

    def test_implausible_units_skip_model(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        bad = {v: da * 1000.0 for v, da in _temps(1, "1981", "1983").items()}
        cfg = _config(tmp_path, [("X", "F", "#000")], indices=("txx",))
        res = TempExtremesIndicesDiag(_FakeLoader({"X": bad}), None, cfg).compute()
        assert res["eerie"] == {}

    def test_tasmax_only_member_gets_tasmax_indices(self, setup):
        diag, *_, tmp = setup
        diag.compute()
        ds = _ckpt(tmp, "B_hist_1981_1984.nc")
        assert {"txx", "txn", "id", "tx10p", "tx90p", "wsdi"} == set(ds.data_vars)

    def test_base_period_exceedances_match_window_order_statistics(self, setup):
        # 3 base years × 5-day window = 15 samples per calendar day.  The
        # type-8 10th/90th percentiles fall between the 1st/2nd and 14th/15th
        # order statistics, so exactly 1 sample in 15 lies beyond them: the
        # in-base inhomogeneity that ETCCDI's (omitted) bootstrap corrects.
        diag, *_, tmp = setup
        diag.compute()
        ds = _ckpt(tmp, "A_hist_1981_1984.nc")
        base = ds.sel(year=slice(1981, 1983))
        for idx in ("tx90p", "tx10p", "tn90p", "tn10p"):
            assert float(base[idx].mean()) == pytest.approx(100 / 15, abs=0.5), idx

    def test_future_uses_reference_thresholds(self, setup):
        diag, *_, tmp = setup
        res = diag.compute()
        ssp = _ckpt(tmp, "A_ssp_2031_2032.nc")
        # +3 °C everywhere: far more warm days, far fewer frost-free… cold days.
        assert float(ssp["tx90p"].mean()) > 30.0
        assert float(ssp["tx10p"].mean()) < 3.0
        ch = res["family"]["tx90p"]["FamA"]["change"]
        assert float(ch.mean()) > 20.0 and res["family"]["tx90p"]["FamA"]["n_fut"] == 1

    def test_gsl_southern_cells_need_previous_half_year(self, setup):
        diag, *_, tmp = setup
        diag.compute()
        ds = _ckpt(tmp, "A_hist_1981_1984.nc")
        land = ds["lon"] < 180.0
        south = ds["gsl"].where(land & (ds["lat"] < 0))
        north = ds["gsl"].where(land & (ds["lat"] > 0))
        assert bool(south.sel(year=1981).isnull().all())
        assert bool(south.sel(year=1982).notnull().any())
        assert bool(north.sel(year=1981).notnull().any())
        assert float(ds["gsl"].max()) <= 366

    def test_ssp_gsl_south_continues_only_from_contiguous_hist(self, setup):
        # The fixture's hist run ends 1984, the SSP run starts 2031: no carry.
        diag, *_, tmp = setup
        diag.compute()
        ssp = _ckpt(tmp, "A_ssp_2031_2032.nc")
        south = ssp["gsl"].where((ssp["lon"] < 180.0) & (ssp["lat"] < 0))
        assert bool(south.sel(year=2031).isnull().all())

    def test_flat_model_configured_as_obs_is_not_a_member(self, tmp_path, monkeypatch):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        data = {"A": _temps(1, "1981", "1983"), "ERA5": _temps(41, "1981", "1983")}
        ldr = _FakeLoader(data)
        cfg = _config(tmp_path, [("A", "F", "#000"), ("ERA5", "ERA5", "#111")],
                      indices=("txx",), obs={"ERA5": {"data_root": "/fake"}})
        diag = TempExtremesIndicesDiag(ldr, None, cfg)
        monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: ldr)
        res = diag.compute()
        assert set(res["eerie"]) == {"A"} and set(res["obs"]) == {"ERA5"}
        assert "ERA5" not in res["families"]

    def test_rerun_reads_checkpoints(self, setup):
        diag, hist, fut, cmip, obs, _ = setup
        diag.compute()
        n = (len(hist.calls), len(fut.calls), len(cmip.calls), len(obs.calls))
        diag.compute()
        assert (len(hist.calls), len(fut.calls), len(cmip.calls),
                len(obs.calls)) == n

    def test_ssp_recomputed_alone_from_hist_thresholds(self, setup):
        diag, *_, tmp = setup
        diag.compute()
        p = tmp / "out" / "temp_extremes_indices" / "A_ssp_2031_2032.nc"
        before = _ckpt(tmp, "A_ssp_2031_2032.nc")["tx90p"].values.copy()
        p.unlink()
        diag.compute()
        np.testing.assert_allclose(_ckpt(tmp, "A_ssp_2031_2032.nc")["tx90p"].values,
                                   before, equal_nan=True)

    def test_without_any_mask_every_cell_is_used(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        cfg = _config(tmp_path, [("A", "F", "#000")], indices=("txx",), mask=False)
        res = TempExtremesIndicesDiag(
            _FakeLoader({"A": _temps(1, "1981", "1983")}), None, cfg).compute()
        assert bool(res["eerie"]["A"]["ref"]["txx"].notnull().all())

    def test_own_sftlf_preferred_over_era5(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        data = _temps(1, "1981", "1983")
        own = xr.DataArray(np.where(_LAT[:, None] > 0, 100.0, 0.0) * np.ones((1, len(_LON))),
                           dims=("lat", "lon"), coords={"lat": _LAT, "lon": _LON})

        class Ldr(_FakeLoader):
            def load_var(self, model, variable, **kw):
                if variable == "sftlf":
                    return own
                return super().load_var(model, variable, **kw)

        cfg = _config(tmp_path, [("A", "F", "#000")], indices=("txx",))
        TempExtremesIndicesDiag(Ldr({"A": data}), None, cfg).compute()
        ds = _ckpt(tmp_path, "A_hist_1981_1984.nc")
        assert bool(ds["txx"].sel(lat=ds["lat"] < 0).isnull().all())
        assert bool(ds["txx"].sel(lat=ds["lat"] > 0).notnull().all())


class TestTempExtremesSeasons:
    def test_seasonal_fields_exclude_annual_only_indices(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        cfg = _config(tmp_path, [("A", "F", "#000")], seasons=("DJF", "JJA"))
        TempExtremesIndicesDiag(_FakeLoader({"A": _temps(1, "1981", "1983")}),
                                None, cfg).compute()
        sea = _ckpt(tmp_path, "A_hist_1981_1984_seasons.nc")
        expected = {f"{i}_{s}" for i in SEASONAL_INDICES for s in ("djf", "jja")}
        assert set(sea.data_vars) == expected
        # DJF of the first year has no December → missing.
        assert bool(sea["txx_djf"].sel(year=1981).isnull().all())
        assert bool(sea["txx_djf"].sel(year=1982).notnull().any())

    def test_seasonal_values(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        data = _temps(1, "1981", "1983")
        cfg = _config(tmp_path, [("A", "F", "#000")], seasons=("JJA",),
                      indices=("txx", "fd"))
        TempExtremesIndicesDiag(_FakeLoader({"A": data}), None, cfg).compute()
        sea = _ckpt(tmp_path, "A_hist_1981_1984_seasons.nc")
        tx = data["tasmax"].sel(time=slice("1982-06", "1982-08")) - 273.15
        ref = tx.max("time").where(tx["lon"] < 180.0)
        np.testing.assert_allclose(sea["txx_jja"].sel(year=1982).values, ref.values,
                                   rtol=1e-5, equal_nan=True)

    def test_adding_a_season_recomputes(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        ldr = _FakeLoader({"A": _temps(1, "1981", "1983")})
        cfg = _config(tmp_path, [("A", "F", "#000")], seasons=("JJA",),
                      indices=("txx",))
        TempExtremesIndicesDiag(ldr, None, cfg).compute()
        n = len(ldr.calls)
        cfg.project["temp_extremes_indices"]["seasons"] = ["JJA", "DJF"]
        TempExtremesIndicesDiag(ldr, None, cfg).compute()
        assert len(ldr.calls) > n
        sea = _ckpt(tmp_path, "A_hist_1981_1984_seasons.nc")
        assert {"txx_jja", "txx_djf"} == set(sea.data_vars)


class TestTempExtremesRun:
    def test_figure_set(self, setup):
        diag, *_, tmp = setup
        diag.indices = ["txx", "tx90p"]
        saved = diag.run()
        ids = sorted(p.stem for p, _ in saved)
        assert ids == sorted([
            "txx_reference", "txx_change", "txx_diff_cmip6", "txx_timeseries",
            "txx_bias_era5",
            # Percentile index: ≈ 10 % by construction over the base period.
            "tx90p_change", "tx90p_timeseries"])

    def test_metadata(self, setup):
        diag, *_, tmp = setup
        diag.indices = ["fd"]
        diag.run()
        fdir = tmp / "out" / "figures" / "temp_extremes_indices"
        bias = json.loads((fdir / "fd_bias_era5.json").read_text())
        assert bias["obs_dataset"] == "ERA5"
        assert bias["obs_variable"] == "tasmax, tasmin"
        assert bias["group"] == "temperature_extremes"
        assert "land" in bias["computation_notes"]
        ts = json.loads((fdir / "fd_timeseries.json").read_text())
        assert "land-only" in ts["description"]
        assert {"FamA mean", "ERA5", "CMIP6 MMM"} <= set(ts["summary_statistics"])

    def test_save_netcdf_summary(self, setup):
        diag, *_, tmp = setup
        diag.indices = ["gsl"]
        diag.save_netcdf = True
        diag.run()
        ds = xr.open_dataset(tmp / "out" / "netcdf" / "temp_extremes_indices"
                             / "gsl_1981-2032_summary.nc")
        assert {"family_FamA_ref", "family_FamA_change", "cmip6_mmm_ref",
                "obs_ERA5_ref"} <= set(ds.data_vars)
        assert ds.attrs["change_units"] == "days"


def _drs(root, activity, inst, model, exp, member, var):
    d = root / activity / inst / model / exp / member / "day" / var / "gn" / "v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{var}.nc").touch()


class TestCmip6Discovery:
    def test_requires_tasmax_and_tasmin_in_both_experiments(self, tmp_path):
        from feather.diag.temp_extremes_indices import TempExtremesIndicesDiag

        root = tmp_path / "CMIP6"
        for v in ("tasmax", "tasmin"):
            _drs(root, "CMIP", "I", "M1", "historical", "r1i1p1f1", v)
            _drs(root, "ScenarioMIP", "I", "M1", "ssp245", "r1i1p1f1", v)
        # M2 lacks ssp245 tasmin.
        _drs(root, "CMIP", "I", "M2", "historical", "r1i1p1f1", "tasmax")
        _drs(root, "CMIP", "I", "M2", "historical", "r1i1p1f1", "tasmin")
        _drs(root, "ScenarioMIP", "I", "M2", "ssp245", "r1i1p1f1", "tasmax")
        cfg = _config(tmp_path, [("A", "F", "#000")], cmip6=True)
        cfg.cmip6_daily["root"] = str(root)
        _, keep = TempExtremesIndicesDiag(_FakeLoader({}), None, cfg)._discover_cmip6()
        assert [m.name for m in keep] == ["M1"]
        assert keep[0].experiments == ["historical", "ssp245"]


class TestRegistry:
    def test_registered(self):
        from feather.diag.registry import get_diagnostic
        assert get_diagnostic("temp_extremes_indices").name == "temp_extremes_indices"
