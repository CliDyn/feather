"""Tests for the ssh_variability diagnostic and feather.util.ssh_stats."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.util import mhw, ssh_stats

_LAT = np.arange(-60.0, 61.0, 15.0)        # 9 rows
_LON = np.arange(0.0, 360.0, 30.0)         # 12 columns
_P0, _P1 = "2000", "2003"


# ── feather.util.ssh_stats ───────────────────────────────────────────────

class TestStats:
    def test_running_mean_matches_convolution(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 200)).astype(np.float32)
        rm = ssh_stats.running_mean(a, 11)
        ref = np.array([np.convolve(r, np.ones(11) / 11, mode="valid") for r in a])
        np.testing.assert_allclose(rm[:, 5:-5], ref, rtol=1e-5, atol=1e-6)
        # Truncated window at the ends: mean of the 6 available steps.
        np.testing.assert_allclose(rm[:, 0], a[:, :6].mean(axis=1), rtol=1e-5)

    def test_running_mean_needs_half_window(self):
        a = np.ones((1, 50), dtype=np.float32)
        a[0, 20:28] = np.nan                     # 8 gaps in an 11-step window
        rm = ssh_stats.running_mean(a, 11)
        assert np.isnan(rm[0, 24])               # only 3 valid of 11
        assert rm[0, 10] == pytest.approx(1.0)

    def test_detrend_removes_line(self):
        t = np.arange(100.0)
        a = (2.0 + 0.03 * t + np.sin(t))[None].astype(np.float32)
        a[0, ::7] = np.nan
        d = ssh_stats.detrend(a)
        fin = np.isfinite(d[0])
        slope = np.polyfit(t[fin], d[0, fin], 1)[0]
        assert abs(slope) < 1e-5
        assert abs(np.nanmean(d)) < 1e-5

    def test_climatology_recovers_cycle(self):
        days = pd.date_range("2000-01-01", "2007-12-31", freq="D")
        slots, n, feb29 = mhw.calendar_slots(days.year.values, days.month.values,
                                             days.day.values)
        cyc = np.cos(2 * np.pi * np.arange(n) / n)
        x = cyc[slots][None].astype(np.float32)
        x[0, 100:130] = np.nan                   # a gap is filled by smoothing
        clim = ssh_stats.seasonal_climatology(x, slots, n, feb29=feb29,
                                              smooth_width=31)
        assert np.isfinite(clim).all()
        np.testing.assert_allclose(clim[0], cyc, atol=0.02)

    def test_cell_stats_separates_scales(self):
        days = pd.date_range("2000-01-01", "2009-12-31", freq="D")
        nt = len(days)
        slots, n, feb29 = mhw.calendar_slots(days.year.values, days.month.values,
                                             days.day.values)
        t = np.arange(nt)
        seas = 0.10 * np.cos(2 * np.pi * slots / n)
        trend = 0.05 * t / nt
        low = 0.03 * np.sin(2 * np.pi * t / (3 * 365.25))     # interannual
        high = 0.02 * np.sin(2 * np.pi * t / 20.0)            # 20-day "eddies"
        x = np.stack([seas + trend + low + high, np.full(nt, np.nan)])
        out = ssh_stats.cell_stats(x.astype(np.float32), slots, n, feb29=feb29,
                                   smooth_width=31, hp_window=91)
        assert set(out) == set(ssh_stats.STATS)
        assert out["std_raw"][0] == pytest.approx(np.std(x[0]), rel=1e-4)
        anom = np.sqrt(0.03 ** 2 / 2 + 0.02 ** 2 / 2)
        assert out["std_anom"][0] == pytest.approx(anom, rel=0.1)
        assert out["std_hp"][0] == pytest.approx(0.02 / np.sqrt(2), rel=0.1)
        assert out["valid_frac"][0] == 1.0 and out["valid_frac"][1] == 0.0
        assert np.isnan(out["std_raw"][1]) and np.isnan(out["mean"][1])

    def test_to_grid_stamps_exact_coords(self):
        """A field on the grid within float tolerance gets the exact axes,
        so arithmetic with other fields does not inner-join to a subset."""
        from feather.diag.precip_extremes import _regular_grid, _to_grid
        lat, lon = _regular_grid(0.25)
        a = xr.DataArray(np.ones((len(lat), len(lon)), dtype=np.float32),
                         coords={"lat": lat.astype(np.float32),
                                 "lon": lon.astype(np.float32) + 1e-6},
                         dims=("lat", "lon"))
        b = xr.DataArray(np.ones((len(lat), len(lon))),
                         coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        assert (a - b).size < a.size                 # the failure mode
        assert (_to_grid(a, lat, lon) - _to_grid(b, lat, lon)).shape == a.shape

    def test_monthly_has_no_high_pass(self):
        x = np.random.default_rng(1).normal(size=(4, 48)).astype(np.float32)
        slots = np.tile(np.arange(12), 4)
        out = ssh_stats.cell_stats(x, slots, 12)
        assert "std_hp" not in out
        assert np.isfinite(out["std_anom"]).all()


# ── Diagnostic ───────────────────────────────────────────────────────────

def _ssh(seed, *, freq="D", amp=0.1, offset=0.0, start=_P0, end=_P1,
         land=True, lat=_LAT, lon=_LON):
    """SSH (m): latitude pattern + seasonal cycle + AR(1) noise, land NaN."""
    if freq == "D":
        time = pd.date_range(f"{start}-01-01T12:00", f"{end}-12-31T12:00", freq="D")
    else:
        time = pd.date_range(f"{start}-01-16", f"{end}-12-16", freq="MS") \
            + pd.Timedelta(days=15)
    rng = np.random.default_rng(seed)
    nt = len(time)
    shape = (nt, len(lat), len(lon))
    eps = rng.normal(0.0, amp, size=shape)
    anom = np.zeros(shape)
    for i in range(1, nt):
        anom[i] = 0.8 * anom[i - 1] + eps[i]
    doy = time.dayofyear.values[:, None, None]
    la = np.asarray(lat)[None, :, None]
    v = offset + 0.5 * np.cos(np.radians(la) * 2) + 0.05 * np.cos(
        2 * np.pi * doy / 365.25) + anom
    if land:
        v[:, :, np.asarray(lon) < 60.0] = np.nan
    return xr.DataArray(v.astype(np.float32), dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": lat, "lon": lon}, name="zos",
                        attrs={"units": "m"})


def _curvilinear(seed):
    """CMIP6-like monthly zos on a (j, i) grid with 2-D lat/lon."""
    nj, ni = 20, 36
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing="ij")
    lat2 = -57.0 + 6.0 * jj + 0.3 * np.sin(ii)
    lon2 = 5.0 + 10.0 * ii
    time = pd.date_range(f"{_P0}-01-01", f"{_P1}-12-01", freq="MS") \
        + pd.Timedelta(days=15)
    rng = np.random.default_rng(seed)
    v = 0.4 * np.cos(np.radians(lat2) * 2)[None] + rng.normal(
        0, 0.05, size=(len(time), nj, ni))
    v[:, :, :3] = np.nan
    return xr.DataArray(
        v.astype(np.float32), dims=("time", "j", "i"),
        coords={"time": time, "latitude": (("j", "i"), lat2),
                "longitude": (("j", "i"), lon2)}, name="zos")


class _FakeLoader:
    """Serves ``{(model, table): DataArray}``."""

    def __init__(self, series):
        self.series = series
        self.calls = []

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        self.calls.append((model, variable, table))
        if (model, table) not in self.series:
            raise FileNotFoundError(f"{model}/{variable}/{table}")
        da = self.series[(model, table)]
        return da.sel(time=slice(period[0], period[1])) if period else da


def _config(tmp_path, models, *, cmip6=False, obs=None, extra=None):
    mcs = {name: ModelConfig(name=name, family=fam, grids={"o2d": "latlon"},
                             color=col) for name, fam, col in models}
    return FeatherConfig(
        model_catalogs={}, models=list(mcs), obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs=mcs,
        project={"name": "TEST", "ensemble_mode": "per_family",
                 "ssh_variability": {
                     "period": [_P0, _P1], "obs": obs or {},
                     "workers": 2, "cell_chunk": 40, "memory_gb": 1,
                     **(extra or {})}},
        cmip6_daily={"enabled": cmip6, "label": "CMIP6"},
    )


_MODELS = [("A", "FamA", "#1f77b4"), ("A-r2", "FamA", "#5aabdf"),
           ("B", "FamB", "#d62728"), ("Mon", "FamB", "#aa0000")]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from feather.diag.ssh_variability import SSHVariabilityDiag

    models = _FakeLoader({
        ("A", "HROday"): _ssh(1), ("A", "Omon"): _ssh(1, freq="M"),
        # Like the FESOM kerchunk store: "HROday" silently gives monthly data.
        ("A-r2", "HROday"): _ssh(2, freq="M"), ("A-r2", "Oday"): _ssh(2),
        ("A-r2", "Omon"): _ssh(2, freq="M"),
        ("B", "Oday"): _ssh(3, amp=0.2, offset=-0.6),
        ("B", "Omon"): _ssh(3, freq="M", amp=0.2, offset=-0.6),
        ("Mon", "Omon"): _ssh(4, freq="M"),               # monthly only
    })
    obs = _FakeLoader({("AVISO", "Oday"): _ssh(41, offset=0.3),
                       ("AVISO", "Omon"): _ssh(41, freq="M", offset=0.3)})
    cmip = _FakeLoader({("C1", "Omon"): _curvilinear(51)})
    cfg = _config(tmp_path, _MODELS, cmip6=True,
                  obs={"AVISO": {"data_root": "/fake", "experiment": "aviso"}},
                  extra={"daily_tables": ["HROday", "Oday"]})
    diag = SSHVariabilityDiag(models, None, cfg)
    diag.plot_resolution = 5.0
    monkeypatch.setattr(diag, "_make_obs_loader", lambda n, s: obs)
    monkeypatch.setattr(diag, "_discover_cmip6",
                        lambda: (cmip, [ModelConfig(name="C1")]))
    return diag, models, obs, cmip, tmp_path


def _out(tmp, *parts):
    return tmp.joinpath("out", "ssh_variability", *parts)


class TestCompute:
    def test_members_per_frequency(self, setup):
        diag, models, _, _, tmp = setup
        res = diag.compute()
        assert set(res) == {"daily", "monthly"}
        assert set(res["daily"]["eerie"]) == {"A", "A-r2", "B"}
        assert set(res["monthly"]["eerie"]) == {"A", "A-r2", "B", "Mon"}
        assert set(res["monthly"]["cmip6"]) == {"C1"}
        assert set(res["monthly"]["obs_1deg"]) == {"AVISO"}
        assert res["daily"]["cmip6"] == {} and res["daily"]["obs_1deg"] == {}
        # A-r2 daily came from Oday after HROday turned out monthly.
        assert ("A-r2", "zos", "HROday") in models.calls
        assert ("A-r2", "zos", "Oday") in models.calls
        for p in ("A_daily_2000_2003.nc", "Mon_monthly_2000_2003.nc",
                  "obs/AVISO_daily_2000_2003.nc", "obs/AVISO_1deg_monthly_2000_2003.nc",
                  "cmip6/C1_monthly_2000_2003.nc"):
            assert _out(tmp, *p.split("/")).exists(), p
        ds = xr.open_dataset(_out(tmp, "A_daily_2000_2003.nc"))
        assert ds.attrs["freq"] == "daily" and ds.attrs["hp_window_days"] == "91"
        assert {"mean", "std_raw", "std_anom", "std_hp", "valid_frac"} <= set(ds)
        assert ds["std_raw"].sel(lon=0.0).isnull().all()
        assert ds["std_raw"].sel(lon=90.0).notnull().all()
        dm = xr.open_dataset(_out(tmp, "A_monthly_2000_2003.nc"))
        assert "std_hp" not in dm

    def test_display_units_and_mean_offset(self, setup):
        diag, *_ = setup
        res = diag.compute()["monthly"]
        b = res["eerie"]["B"]
        assert b["std_raw"].attrs["units"] == "cm"
        # std in cm: AR(1) noise of 0.2 m has a std of several tens of cm.
        assert 10 < float(b["std_raw"].mean()) < 80
        # Mean SSH is relative to its own area mean: B's −0.6 m offset and
        # AVISO's +0.3 m are gone.
        for d in (b, res["obs"]["AVISO"]):
            assert abs(diag._latlon_field_mean(d["mean"])) < 0.02
        assert "std_hp" not in b

    def test_checkpoints_reused(self, setup):
        diag, models, *_ = setup
        diag.compute()
        n = len(models.calls)
        diag.compute()
        # Only the member without daily data is looked up again.
        assert {c[0] for c in models.calls[n:]} == {"Mon"}

    def test_stale_checkpoint_recomputed(self, setup):
        diag, models, *_ = setup
        diag.compute()
        n = len(models.calls)
        diag.hp_window = 61
        diag.compute()
        assert len(models.calls) > n

    def test_wrong_units_skipped(self, setup, caplog):
        diag, models, *_ = setup
        models.series[("B", "Oday")] = _ssh(3) * 100.0 + 50.0     # cm, offset
        res = diag.compute()
        assert "B" not in res["daily"]["eerie"]
        assert "not m" in caplog.text

    def test_daily_only(self, tmp_path, monkeypatch):
        from feather.diag.ssh_variability import SSHVariabilityDiag
        models = _FakeLoader({("A", "HROday"): _ssh(1)})
        cfg = _config(tmp_path, [("A", "FamA", "#1f77b4")],
                      extra={"monthly": False})
        diag = SSHVariabilityDiag(models, None, cfg)
        res = diag.compute()
        assert set(res) == {"daily"}


class TestFigures:
    def test_figures_and_metadata(self, setup):
        diag, _, _, _, tmp = setup
        saved = diag.run()
        ids = {p[0].stem for p in saved}
        for k in ("std_raw", "std_anom", "std_hp"):
            for kind in ("maps", "members", "bias"):
                assert f"ssh_{k}_daily_{kind}" in ids
        for k in ("mean", "std_raw", "std_anom"):
            assert f"ssh_{k}_monthly_maps" in ids
        assert "ssh_mean_daily_maps" not in ids
        assert "ssh_std_hp_monthly_maps" not in ids
        assert {"ssh_daily_zonal_means", "ssh_monthly_zonal_means",
                "ssh_daily_regions", "ssh_monthly_regions"} <= ids
        meta = json.loads(next(p[1] for p in saved
                               if p[0].stem == "ssh_std_raw_monthly_bias").read_text())
        assert meta["frequency"] == "monthly"
        assert meta["units"] == "cm"
        assert any("CMIP6 MMM" in t for t in meta["summary_statistics"])
        fam = next(t for t in meta["summary_statistics"] if t.startswith("FamA"))
        assert "area_mean_ratio" in meta["summary_statistics"][fam]
        assert "frac_area_within_2std" in meta["summary_statistics"][fam]
        assert meta["colormap"] == "RdBu_r"
        maps = json.loads(next(p[1] for p in saved
                               if p[0].stem == "ssh_std_raw_daily_maps").read_text())
        assert maps["colormap"] == "Reds"
        reg = json.loads(next(p[1] for p in saved
                              if p[0].stem == "ssh_daily_regions").read_text())
        assert "gulf_stream" in reg["summary_statistics"]

    def test_netcdf_export(self, setup):
        diag, _, _, _, tmp = setup
        diag.save_netcdf = True
        diag.run()
        out = tmp / "out" / "netcdf" / "ssh_variability"
        ds = xr.open_dataset(out / "ssh_std_hp_daily_2000-2003.nc")
        assert {"obs_AVISO", "family_FamA_mean", "member_A"} <= set(ds)
        d1 = xr.open_dataset(out / "ssh_std_raw_monthly_1deg_2000-2003.nc")
        assert {"cmip6_C1", "cmip6_mmm", "obs_AVISO"} <= set(d1)


class TestLoaderFiles:
    def test_stray_files_ignored(self, tmp_path):
        from feather.data.cmor_loader import CMORLoader

        vdir = tmp_path / "BSC" / "M" / "hist-1950" / "r1i1p1f1" / "HROday" / "zos" \
            / "gr" / "v1"
        vdir.mkdir(parents=True)
        for y in (2000, 2001):
            _ssh(y, start=str(y), end=str(y)).to_dataset().to_netcdf(
                vdir / f"zos_HROday_M_hist-1950_r1i1p1f1_gr_{y}0101-{y}1231.nc")
        (vdir / "zos_merged_2000-2001.nc").write_text("broken")
        mc = ModelConfig(name="M", institution="BSC", experiment="hist-1950",
                         variant="r1i1p1f1", grids={"o2d": "latlon"})
        cfg = FeatherConfig(model_catalogs={}, models=["M"], obs_root="",
                            obs_datasets={}, cmip6={"enabled": False}, dask={},
                            nereus={}, output_dir=str(tmp_path / "o"),
                            data_source={"type": "cmor", "root": str(tmp_path)},
                            model_configs={"M": mc}, project={})
        da = CMORLoader(cfg).load_var("M", "zos", table="HROday")
        assert da.sizes["time"] == 731

    def test_config_parses(self):
        from feather.diag.ssh_variability import SSHVariabilityDiag

        cfg = FeatherConfig.from_yaml("configs/eerie_10_mems_ssh.yaml")
        assert len(cfg.models) == 10
        diag = SSHVariabilityDiag(None, None, cfg)
        assert diag.period == ("1993", "2014")
        assert diag.freqs == ["daily", "monthly"]
        assert diag.reference == "AVISO"
        assert cfg.model_configs["HadGEM3-GC5"].data_root.startswith(
            "/work/bm1344/DKRZ/MOHC/")
