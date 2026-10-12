"""Tests for the marine_heatwaves_subsurface diagnostic (synthetic daily thetao)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig

_LAT = np.arange(-60.0, 61.0, 15.0)        # 9 rows
_LON = np.arange(0.0, 360.0, 30.0)         # 12 columns
_P0, _P1 = "2000", "2005"


def _thetao(seed, depths, *, zdim="lev", units="degC", drop_days=0):
    """Daily temperature on *depths*: cooler and smoother with depth.

    Land west of 60°E on every level, and west of 120°E below 100 m (a shelf).
    """
    time = pd.date_range(f"{_P0}-01-01T12:00", f"{_P1}-12-31T12:00", freq="D")
    rng = np.random.default_rng(seed)
    nt, nz = len(time), len(depths)
    shape = (nt, nz, len(_LAT), len(_LON))
    eps = rng.normal(0.0, 0.25, size=shape)
    anom = np.zeros(shape)
    for i in range(1, nt):
        anom[i] = 0.9 * anom[i - 1] + eps[i]
    doy = time.dayofyear.values[:, None, None, None]
    z = np.asarray(depths, dtype=float)[None, :, None, None]
    la = _LAT[None, None, :, None]
    clim = (28.0 - 0.3 * np.abs(la) - 0.03 * z
            + 3 * np.exp(-z / 80.0) * np.cos(2 * np.pi * doy / 365.25))
    v = clim + anom
    v[:, :, :, _LON < 60.0] = np.nan
    deep = np.flatnonzero(np.asarray(depths) > 100.0)
    for k in deep:
        v[:, k][:, :, _LON < 120.0] = np.nan
    if units == "K":
        v = v + 273.15
    da = xr.DataArray(v.astype(np.float32), dims=("time", zdim, "lat", "lon"),
                      coords={"time": time, zdim: np.asarray(depths, float),
                              "lat": _LAT, "lon": _LON},
                      name="thetao", attrs={"units": units})
    if drop_days:
        da = da.isel(time=np.setdiff1d(np.arange(nt), np.arange(100, 100 + drop_days)))
    return da


def _mld(seed):
    time = pd.date_range(f"{_P0}-01-01", periods=72, freq="MS") + pd.Timedelta(days=14)
    rng = np.random.default_rng(seed)
    season = 40.0 + 30.0 * np.cos(2 * np.pi * (time.month.values - 1) / 12)
    v = season[:, None, None] + rng.normal(0, 2, size=(72, len(_LAT), len(_LON)))
    v[:, :, _LON < 60.0] = np.nan
    return xr.DataArray(v.astype(np.float32), dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": _LAT, "lon": _LON},
                        name="mlotst")


class _FakeLoader:
    def __init__(self, thetao, mld=None):
        self.thetao = thetao
        self.mld = mld or {}
        self.calls = []

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        self.calls.append((model, variable, table))
        if variable == "thetao" and table == "HROday" and model in self.thetao:
            da = self.thetao[model]
        elif variable == "mlotst" and table == "Omon" and model in self.mld:
            da = self.mld[model]
        else:
            raise FileNotFoundError(f"{model}/{variable}/{table}")
        return da.sel(time=slice(period[0], period[1])) if period else da


_A_DEPTHS = [2.5, 12.5, 97.5, 195.0, 600.0]
_B_DEPTHS = [1.0, 50.55, 201.8, 617.1]
_MODELS = [("A", "FamA", "#1f77b4"), ("A-r2", "FamA", "#5aabdf"),
           ("B", "FamB", "#2ca02c"), ("NoThetao", "FamB", "#aa0000")]


def _config(tmp_path, *, sub=None):
    mcs = {name: ModelConfig(name=name, family=fam, grids={"o3d": "latlon"},
                             color=col) for name, fam, col in _MODELS}
    return FeatherConfig(
        model_catalogs={}, models=list(mcs), obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs=mcs,
        project={"name": "TEST", "ensemble_mode": "per_family",
                 "marine_heatwaves": {
                     "period": [_P0, _P1],
                     "obs": {"OISST": {"data_root": "/fake", "experiment": "oisst"}},
                     "workers": 2, "cell_chunk": 40, "memory_gb": 1},
                 "marine_heatwaves_subsurface": {"max_depth": 210, **(sub or {})}},
        cmip6_daily={"enabled": False},
    )


def _loader():
    return _FakeLoader(
        {"A": _thetao(1, _A_DEPTHS), "A-r2": _thetao(2, _A_DEPTHS, units="K"),
         "B": _thetao(3, _B_DEPTHS, zdim="depth", drop_days=5)},
        mld={"A": _mld(7), "B": _mld(8)})


@pytest.fixture
def setup(tmp_path):
    from feather.diag.marine_heatwaves_subsurface import MarineHeatwavesSubsurfaceDiag

    loader = _loader()
    diag = MarineHeatwavesSubsurfaceDiag(loader, None, _config(tmp_path))
    diag.plot_resolution = 5.0
    return diag, loader, tmp_path


def _out(tmp, *parts):
    return tmp.joinpath("out", "marine_heatwaves_subsurface", *parts)


class TestPoolLevels:
    def _lv(self, freq, dur, imax, days):
        f = lambda v: xr.DataArray(np.asarray(v, dtype=float), dims=("x",))  # noqa: E731
        return {"frequency": f(freq), "duration_mean": f(dur),
                "intensity_max_max": f(imax), "mhw_days_pct": f(days)}

    def test_event_weighted_mean_max_and_level_mean(self):
        from feather.diag.marine_heatwaves_subsurface import pool_levels
        out = pool_levels([self._lv([1.0, 0.0, 2.0], [10.0, np.nan, 5.0],
                                    [1.0, np.nan, 3.0], [2.0, 0.0, 4.0]),
                           self._lv([3.0, 2.0, np.nan], [20.0, 8.0, np.nan],
                                    [2.0, 0.5, np.nan], [6.0, 2.0, np.nan])])
        # Cell 0: (1·10 + 3·20)/4; cell 1: only level 2 has events; cell 2:
        # land on level 2.
        np.testing.assert_allclose(out["duration_mean"], [17.5, 8.0, 5.0])
        np.testing.assert_allclose(out["intensity_max_max"], [2.0, 0.5, 3.0])
        np.testing.assert_allclose(out["frequency"], [2.0, 1.0, 2.0])
        np.testing.assert_allclose(out["mhw_days_pct"], [4.0, 1.0, 4.0])

    def test_no_events_anywhere_is_nan(self):
        from feather.diag.marine_heatwaves_subsurface import pool_levels
        out = pool_levels([self._lv([0.0], [np.nan], [np.nan], [0.0]),
                           self._lv([0.0], [np.nan], [np.nan], [0.0])])
        assert np.isnan(out["duration_mean"][0])
        assert out["frequency"][0] == 0.0

    def test_empty(self):
        from feather.diag.marine_heatwaves_subsurface import pool_levels
        assert pool_levels([]) == {}


class TestCompute:
    def test_levels_checkpoints_and_layers(self, setup):
        diag, loader, tmp = setup
        res = diag.compute()
        assert set(res["levels"]) == {"A", "A-r2", "B"}          # NoThetao skipped
        assert list(res["levels"]["A"]) == [2.5, 12.5, 97.5, 195.0]   # 600 m dropped
        assert list(res["levels"]["B"]) == [1.0, 50.55, 201.8]
        assert _out(tmp, "A_z2.5m_2000_2005.nc").exists()
        assert _out(tmp, "B_z50.55m_2000_2005.nc").exists()
        assert not _out(tmp, "A_z600m_2000_2005.nc").exists()
        ds = xr.open_dataset(_out(tmp, "A_z195m_2000_2005.nc"))
        assert ds.attrs["depth"] == "195"
        assert ds.attrs["pctile"] == "90.0"
        # Shelf: land below 100 m west of 120°E, ocean above.
        assert np.isnan(ds["seas_mean"].sel(lon=90.0)).all()
        ds2 = xr.open_dataset(_out(tmp, "A_z12.5m_2000_2005.nc"))
        assert np.isfinite(ds2["seas_mean"].sel(lon=90.0)).all()
        # Layers per member, families with their members.
        assert set(res["layers"]["B"]) == {"shallow", "deep"}
        assert res["family"]["FamA"]["n"] == 2
        assert res["family"]["FamB"]["members"] == ["B"]
        for g in res["family"].values():
            assert "duration_mean" in g["shallow"] and "duration_mean" in g["deep"]
        # On the shelf the deep layer holds only the 97.5 m level.
        lv = res["levels"]["A"]
        deep = res["layers"]["A"]["deep"]["seas_mean"].sel(lon=90.0)
        assert np.isnan(lv[195.0]["seas_mean"].sel(lon=90.0)).all()
        xr.testing.assert_allclose(deep, lv[97.5]["seas_mean"].sel(lon=90.0))
        # Column mask excludes the shelf.
        cm = res["column_mask"]
        assert not bool(cm.sel(lon=90.0, method="nearest").any())
        assert bool(cm.sel(lon=200.0, method="nearest").any())
        # MLD read for members that publish it.
        assert set(res["mld"]) == {"A", "B"}

    def test_checkpoints_reused(self, setup, monkeypatch):
        diag, loader, _ = setup
        diag.compute()

        def boom(*a, **k):
            raise AssertionError("re-detected")
        monkeypatch.setattr(diag, "_detect_levels", boom)
        res = diag.compute()
        assert set(res["levels"]) == {"A", "A-r2", "B"}

    def test_only_missing_levels_recomputed(self, setup, monkeypatch):
        diag, _, tmp = setup
        diag.compute()
        _out(tmp, "B_z201.8m_2000_2005.nc").unlink()
        seen = []
        orig = diag._detect_levels

        def spy(da, zdim, zidx):
            seen.append(list(zidx))
            return orig(da, zdim, zidx)
        monkeypatch.setattr(diag, "_detect_levels", spy)
        diag.compute()
        assert seen == [[2]]

    def test_multilevel_equals_single_level(self, setup):
        """Reading levels together gives the same maps as one level alone."""
        diag, loader, _ = setup
        da = diag._std(loader.thetao["A"], "lev")
        multi = diag._detect_levels(da, "lev", [0, 2, 3])
        single = diag._detect_regular(da.isel(lev=2))
        for k in ("duration_mean", "frequency", "mhw_days_annual", "seas_mean"):
            xr.testing.assert_allclose(multi[1][k], single[k])

    def test_kelvin_converted(self, setup):
        diag, _, _ = setup
        res = diag.compute()
        sm = res["levels"]["A-r2"][2.5]["seas_mean"]
        assert float(sm.max()) < 40.0

    def test_surface_checkpoints_used(self, setup):
        from feather.diag.marine_heatwaves import MarineHeatwavesDiag

        diag, loader, tmp = setup
        sst = loader.thetao["A"].isel(lev=0, drop=True).rename("tos")

        class _SST:
            def load_var(self, model, variable, *, table=None, period=None,
                         time_mean=False):
                if variable != "tos":
                    raise FileNotFoundError(variable)
                return sst

        surf = MarineHeatwavesDiag(_SST(), None, diag.config)
        surf._eerie_sources = lambda: {"A": lambda: sst}
        surf._make_obs_loader = lambda n, s: _SST()
        surf.compute()
        res = diag.compute()
        assert set(res["obs"]) == {"OISST"}
        assert set(res["surface"]) == {"A"}
        assert res["reference"] == "OISST"

    def test_no_surface_checkpoints(self, setup):
        diag, _, _ = setup
        res = diag.compute()
        assert res["obs"] == {} and res["surface"] == {}
        assert res["reference"] is None

    def test_settings_override(self, tmp_path):
        from feather.diag.marine_heatwaves_subsurface import MarineHeatwavesSubsurfaceDiag
        cfg = _config(tmp_path, sub={"max_depth": 60, "shallow_max_depth": 20,
                                     "min_duration": 7})
        diag = MarineHeatwavesSubsurfaceDiag(_loader(), None, cfg)
        assert diag.params["min_duration"] == 7
        assert diag.period == (_P0, _P1)          # inherited from marine_heatwaves
        res = diag.compute()
        assert list(res["levels"]["B"]) == [1.0, 50.55]
        assert diag._layer(12.5) == "shallow" and diag._layer(50.55) == "deep"


class TestFigures:
    def test_run_writes_figures(self, setup):
        diag, _, tmp = setup
        diag.metrics = ["duration_mean", "category_max", "mhw_days_pct"]
        diag.regions = {"box": ("Box", -20.0, 20.0, 150.0, 250.0)}
        diag.save_netcdf = True
        saved = diag.run()
        ids = {p[0].stem for p in saved}
        for k in diag.metrics:
            assert f"mhwz_{k}_layers" in ids
            assert f"mhwz_{k}_deep_minus_shallow" in ids
            assert f"mhwz_{k}_levels" in ids
        assert {"mhwz_profiles_global", "mhwz_profiles_box", "mhwz_zonal_means",
                "mhwz_days_timeseries", "mhwz_regions_box"} <= ids
        meta = json.loads(next(p[1] for p in saved
                               if p[0].stem == "mhwz_profiles_global").read_text())
        assert meta["model_levels_m"]["B"] == [1.0, 50.55, 201.8]
        assert meta["shallow_max_depth_m"] == 50.0
        assert "mixed_layer_depth_m" in meta["summary_statistics"]
        nc = tmp / "out" / "netcdf" / "marine_heatwaves_subsurface" / \
            f"mhwz_duration_mean_{_P0}-{_P1}.nc"
        ds = xr.open_dataset(nc)
        assert {"family_FamA_shallow_mean", "family_FamB_deep_mean",
                "member_A_deep"} <= set(ds.data_vars)

    def test_no_members_no_figures(self, tmp_path):
        from feather.diag.marine_heatwaves_subsurface import MarineHeatwavesSubsurfaceDiag
        diag = MarineHeatwavesSubsurfaceDiag(_FakeLoader({}), None, _config(tmp_path))
        assert diag.run() == []


class TestICONLoader:
    def test_daily_thetao_store(self):
        from feather.data.icon_kerchunk_loader import ICONKerchunkLoader
        store, (name, scale, off) = ICONKerchunkLoader._resolve_store("thetao", "HROday")
        assert (store, name, scale, off) == ("ocean5lev_day", "to", 1.0, 0.0)
        assert ICONKerchunkLoader._resolve_store("thetao", "Oday")[0] == "ocean5lev_day"
        assert ICONKerchunkLoader._resolve_store("tos", "HROday")[0] == "ocean2d_day"

    def test_monthly_thetao_not_served(self):
        from feather.data.icon_kerchunk_loader import ICONKerchunkLoader
        with pytest.raises(KeyError):
            ICONKerchunkLoader._resolve_store("thetao", "Omon")
