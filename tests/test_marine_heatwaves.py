"""Tests for the marine_heatwaves diagnostic (synthetic daily SST)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig

_LAT = np.arange(-60.0, 61.0, 15.0)        # 9 rows
_LON = np.arange(0.0, 360.0, 30.0)         # 12 columns
_P0, _P1 = "2000", "2005"


def _sst(seed, *, warm=0.0, units="degC", lat=_LAT, lon=_LON, start=_P0,
         end=_P1, land=True, drop_days=0):
    """Daily SST: seasonal cycle, AR(1) anomalies, land NaN west of 60°E."""
    time = pd.date_range(f"{start}-01-01T12:00", f"{end}-12-31T12:00", freq="D")
    rng = np.random.default_rng(seed)
    nt = len(time)
    shape = (nt, len(lat), len(lon))
    eps = rng.normal(0.0, 0.25, size=shape)
    anom = np.zeros(shape)
    for i in range(1, nt):
        anom[i] = 0.9 * anom[i - 1] + eps[i]
    doy = time.dayofyear.values[:, None, None]
    la = np.asarray(lat)[None, :, None]
    clim = 28.0 - 0.3 * np.abs(la) + 3 * np.cos(2 * np.pi * doy / 365.25)
    trend = warm * np.linspace(0, 1, nt)[:, None, None]
    v = clim + anom + trend
    if land:
        v[:, :, np.asarray(lon) < 60.0] = np.nan
    if units == "K":
        v = v + 273.15
    da = xr.DataArray(v.astype(np.float32), dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lat, "lon": lon}, name="tos",
                      attrs={"units": units})
    if drop_days:
        da = da.isel(time=np.setdiff1d(np.arange(nt), np.arange(100, 100 + drop_days)))
    return da


def _curvilinear(seed):
    """CMIP6-like SST on a (j, i) grid with 2-D lat/lon and a 360-day calendar."""
    import cftime
    nj, ni = 20, 36
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing="ij")
    lat2 = -57.0 + 6.0 * jj + 0.3 * np.sin(ii)
    lon2 = 5.0 + 10.0 * ii
    times = [cftime.Datetime360Day(y, m, d, 12) for y in range(int(_P0), int(_P1) + 1)
             for m in range(1, 13) for d in range(1, 31)]
    rng = np.random.default_rng(seed)
    nt = len(times)
    anom = np.zeros((nt, nj, ni))
    eps = rng.normal(0, 0.25, size=anom.shape)
    for i in range(1, nt):
        anom[i] = 0.9 * anom[i - 1] + eps[i]
    v = 300.0 - 0.3 * np.abs(lat2)[None] + anom       # K
    v[:, :, :3] = np.nan                               # land
    return xr.DataArray(
        v.astype(np.float32), dims=("time", "j", "i"),
        coords={"time": times, "latitude": (("j", "i"), lat2),
                "longitude": (("j", "i"), lon2)},
        name="tos", attrs={"units": "K"})


class _FakeLoader:
    def __init__(self, series):
        self.series = series
        self.calls = []

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        self.calls.append((model, variable, table, period))
        if table != "Oday" or model not in self.series:
            raise FileNotFoundError(f"{model}/{variable}/{table}")
        da = self.series[model]
        return da.sel(time=slice(period[0], period[1])) if period else da


def _config(tmp_path, models, *, cmip6=False, obs=None, mode="per_family", mhw=None):
    mcs = {name: ModelConfig(name=name, family=fam, grids={"o2d": "latlon"},
                             color=col) for name, fam, col in models}
    return FeatherConfig(
        model_catalogs={}, models=list(mcs), obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs=mcs,
        project={"name": "TEST", "ensemble_mode": mode,
                 "marine_heatwaves": {
                     "period": [_P0, _P1], "obs": obs or {},
                     "workers": 2, "cell_chunk": 40, "memory_gb": 1,
                     **(mhw or {})}},
        cmip6_daily={"enabled": cmip6, "label": "CMIP6"},
    )


_MODELS = [("A", "FamA", "#1f77b4"), ("A-r2", "FamA", "#5aabdf"),
           ("B", "FamB", "#d62728"), ("NoSST", "FamB", "#aa0000")]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from feather.diag.marine_heatwaves import MarineHeatwavesDiag

    models = _FakeLoader({"A": _sst(1), "A-r2": _sst(2, units="K"),
                          "B": _sst(3, warm=1.0, drop_days=5)})
    obs = _FakeLoader({"OISST": _sst(41), "ERA5": _sst(42)})
    cmip = _FakeLoader({"C1": _curvilinear(51)})
    cfg = _config(tmp_path, _MODELS, cmip6=True,
                  obs={"OISST": {"data_root": "/fake", "experiment": "oisst"},
                       "ERA5": {"data_root": "/fake", "experiment": "era5"}})
    diag = MarineHeatwavesDiag(models, None, cfg)
    diag.plot_resolution = 5.0
    monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: obs)
    monkeypatch.setattr(diag, "_discover_cmip6",
                        lambda: (cmip, [ModelConfig(name="C1")]))
    return diag, models, obs, cmip, tmp_path


def _out(tmp, *parts):
    return tmp.joinpath("out", "marine_heatwaves", *parts)


class TestCompute:
    def test_checkpoints(self, setup):
        diag, models, _, _, tmp = setup
        res = diag.compute()
        assert set(res["eerie"]) == {"A", "A-r2", "B"}       # NoSST skipped
        assert set(res["obs"]) == {"OISST", "ERA5"}
        assert set(res["cmip6"]) == {"C1"}
        assert _out(tmp, "A_2000_2005.nc").exists()
        assert _out(tmp, "obs", "OISST_2000_2005.nc").exists()
        assert _out(tmp, "cmip6", "C1_2000_2005.nc").exists()
        ds = xr.open_dataset(_out(tmp, "A_2000_2005.nc"))
        assert ds.attrs["pctile"] == "90.0"
        assert ds["mhw_days_annual"].sizes["year"] == 6
        # Land stays NaN, ocean is filled.
        assert ds["frequency"].sel(lon=0.0).isnull().all()
        assert ds["frequency"].sel(lon=90.0).notnull().all()
        assert ("A", "tos", "Oday", (_P0, _P1)) in models.calls

    def test_matches_util_detect(self, setup):
        """The streamed, chunked, threaded path equals one detect() call."""
        from feather.util import mhw
        diag, models, *_ = setup
        da = models.series["A"]
        ds = diag._detect_regular(da)
        lat, lon = 30.0, 120.0
        ser = da.sel(lat=lat, lon=lon).values[None, :]
        t = da.time.dt
        _, ref = mhw.detect(ser, t.year.values, t.month.values, t.day.values,
                            base_years=(2000, 2005))
        for k in ("frequency", "duration_mean", "intensity_max_max",
                  "category_max", "mhw_days_pct"):
            np.testing.assert_allclose(ds[k].sel(lat=lat, lon=lon), ref[k][0],
                                       rtol=1e-5, err_msg=k)

    def test_kelvin_and_missing_days(self, setup):
        diag, models, *_ = setup
        ds_k = diag._detect_regular(models.series["A-r2"])
        # Kelvin input → same intensities as the °C field would give.
        ds_c = diag._detect_regular(models.series["A-r2"] - 273.15)
        np.testing.assert_allclose(ds_k["intensity_max_max"], ds_c["intensity_max_max"],
                                   rtol=1e-4)
        ds_b = diag._detect_regular(models.series["B"])     # 5 days dropped
        assert np.isfinite(ds_b["frequency"].sel(lon=90.0)).all()

    def test_row_blocks_split(self, setup):
        diag, models, *_ = setup
        one = diag._detect_regular(models.series["A"])
        diag.memory_gb = 1e-9          # forces one latitude row per block
        many = diag._detect_regular(models.series["A"])
        xr.testing.assert_allclose(one, many)

    def test_checkpoint_reused_and_settings_checked(self, setup):
        diag, models, *_ = setup
        def n_loaded():
            return sum(c[0] != "NoSST" for c in models.calls)

        diag.compute()
        n = n_loaded()
        diag.compute()
        assert n_loaded() == n                   # all from checkpoints
        diag.params["min_duration"] = 6
        diag.compute()
        assert n_loaded() > n                    # settings changed → recompute

    def test_cmip6_regridded_to_1deg(self, setup):
        diag, *_ = setup
        res = diag.compute()
        c1 = res["cmip6"]["C1"]["frequency"]
        assert c1.sizes == {"lat": 180, "lon": 360}
        assert np.isfinite(c1).sum() > 100

    def test_family_means(self, setup):
        diag, *_ = setup
        res = diag.compute()
        assert res["families"] == {"FamA": ["A", "A-r2"], "FamB": ["B"]}
        g = res["family"]["duration_mean"]["FamA"]
        expected = (res["eerie"]["A"]["duration_mean"]
                    + res["eerie"]["A-r2"]["duration_mean"]) / 2
        xr.testing.assert_allclose(g["mean"], expected)
        assert g["std"] is not None
        assert res["family"]["duration_mean"]["FamB"]["std"] is None

    def test_pooled_mode(self, tmp_path, monkeypatch):
        from feather.diag.marine_heatwaves import MarineHeatwavesDiag
        cfg = _config(tmp_path, _MODELS[:2], mode="pooled",
                      obs={"OISST": {"data_root": "/f"}})
        diag = MarineHeatwavesDiag(_FakeLoader({"A": _sst(1), "A-r2": _sst(2)}),
                                   None, cfg)
        monkeypatch.setattr(diag, "_make_obs_loader",
                            lambda n, s: _FakeLoader({"OISST": _sst(9)}))
        res = diag.compute()
        assert res["families"] == {"EERIE": ["A", "A-r2"]}

    def test_box_matrix(self):
        from feather.diag.marine_heatwaves import MarineHeatwavesDiag
        lat = np.array([0.2, 0.7, 0.4, 10.5])
        lon = np.array([0.3, 0.6, 359.9, 20.5])
        m, cells, tlat, tlon = MarineHeatwavesDiag._box_matrix(lat, lon, 1.0)
        vals = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        out = dict(zip(cells, (m @ vals) / np.asarray(m.sum(axis=1)).ravel()))
        assert out[90 * 360 + 0] == pytest.approx(2.0)      # first two points
        assert out[90 * 360 + 359] == pytest.approx(5.0)
        assert out[100 * 360 + 20] == pytest.approx(7.0)
        # Neighbouring empty cells within 1° take the nearest point.
        assert 90 * 360 + 1 in cells


class TestRun:
    def test_figure_set(self, setup):
        diag, *_, tmp = setup
        diag.metrics = ["duration_mean", "category_max"]
        saved = diag.run()
        ids = sorted(p.stem for p, _ in saved)
        expected = []
        for k in diag.metrics:
            expected += [f"mhw_{k}_maps", f"mhw_{k}_members", f"mhw_{k}_bias",
                         f"mhw_{k}_spread"]
        expected += ["mhw_zonal_means", "mhw_days_timeseries"]
        expected += [f"mhw_regions_{r}" for r in diag.regions]
        assert ids == sorted(expected)

    def test_metadata(self, setup):
        diag, *_, tmp = setup
        diag.metrics = ["intensity_max_max"]
        diag.run()
        fdir = tmp / "out" / "figures" / "marine_heatwaves"
        bias = json.loads((fdir / "mhw_intensity_max_max_bias.json").read_text())
        assert bias["group"] == "marine_heatwaves"
        assert bias["obs_dataset"] == "OISST, ERA5"
        assert bias["reference_obs"] == "OISST"
        stats = bias["summary_statistics"]
        assert "FamA mean (2 members) − OISST" in stats
        assert "frac_area_within_2std" in stats["FamA mean (2 members) − OISST"]
        assert "ERA5 − OISST" in stats
        assert any(k.startswith("CMIP6 MMM (1 models") for k in stats)
        assert "Hobday" in bias["computation_notes"]
        ts = json.loads((fdir / "mhw_days_timeseries.json").read_text())
        assert {"OISST", "FamA", "CMIP6 MMM"} <= set(ts["summary_statistics"])

    def test_save_netcdf(self, setup):
        diag, *_, tmp = setup
        diag.metrics = ["frequency"]
        diag.save_netcdf = True
        diag.run()
        ds = xr.open_dataset(tmp / "out" / "netcdf" / "marine_heatwaves"
                             / "mhw_frequency_2000-2005.nc")
        assert {"obs_OISST", "family_FamA_mean", "member_A"} <= set(ds.data_vars)


class TestDiscovery:
    def test_finds_oday_tos(self, tmp_path):
        from feather.diag.marine_heatwaves import MarineHeatwavesDiag
        root = tmp_path / "CMIP6"
        for model in ("M1", "M2"):
            d = (root / "CMIP" / "I" / model / "historical" / "r1i1p1f1" / "Oday"
                 / "tos" / "gn" / "v1")
            d.mkdir(parents=True)
            (d / "tos.nc").touch()
        cfg = _config(tmp_path, [("A", "F", "#000")], cmip6=True)
        cfg.cmip6_daily.update(root=str(root), exclude=["M2"])
        _, found = MarineHeatwavesDiag(_FakeLoader({}), None, cfg)._discover_cmip6()
        assert [m.name for m in found] == ["M1"]


def test_registered():
    from feather.diag.registry import get_diagnostic
    assert get_diagnostic("marine_heatwaves").name == "marine_heatwaves"
