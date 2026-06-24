"""Tests for HeatwaveHotspotsDiag — extreme-heat tail-widening (PNAS Fig 2–4)."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.diag.registry import get_diagnostic, list_diagnostics
from feather.diag.heatwave_hotspots import HeatwaveHotspotsDiag, _REGIONS


# ── Fixtures / builders ──────────────────────────────────────────────────────

def _make_config(tmp_path, models=("ERA5", "ModelA", "ModelB")):
    return FeatherConfig(
        model_catalogs={},
        models=list(models),
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000, "resolution": 1.0},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        project={"extremes_obs_reference": "ERA5", "period": ["1980", "1984"]},
        model_configs={
            m: ModelConfig(name=m, grids={"sfc": "latlon"}, color="#1f77b4")
            for m in models
        },
    )


def _make_daily_tasmax(n_years=5, base_k=300.0, seed=0):
    """Synthetic daily tasmax on a small lat/lon grid with a widening tail."""
    rng = np.random.default_rng(seed)
    lat = np.array([-45.0, -15.0, 15.0, 45.0], dtype=np.float32)
    lon = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], dtype=np.float32)
    nlat, nlon = len(lat), len(lon)
    n_days = n_years * 365
    time = xr.cftime_range("1980-01-01", periods=n_days, freq="D", calendar="noleap")
    years = np.asarray(time.year)
    data = np.empty((n_days, nlat, nlon), dtype=np.float32)
    for i, yr in enumerate(years):
        spread = 5.0 + 2.0 * (yr - 1980)  # widening upper tail over time
        data[i] = base_k + rng.normal(0, spread, size=(nlat, nlon))
    return xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )


class MockCMORLoader:
    def __init__(self, da):
        self._da = da

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        if variable == "sftlf":
            raise FileNotFoundError("no sftlf")
        da = self._da
        if period:
            da = da.sel(time=slice(period[0], period[1]))
        return da


class _FakeInterp:
    """Identity interpolator: source grid == target grid (small lat/lon)."""

    def __init__(self, lat, lon):
        self._shape = (len(lat), len(lon))
        self.target_lon, self.target_lat = np.meshgrid(lon, lat)

    def __call__(self, raveled):
        return np.asarray(raveled).reshape(self._shape)


@pytest.fixture
def diag(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    da = _make_daily_tasmax()
    d = HeatwaveHotspotsDiag(
        MockCMORLoader(da), None, cfg,
        period=("1980", "1984"), n_bootstrap=200,
    )
    # Stub nereus regridding with an identity interpolator on the small grid.
    lat = np.asarray(da["lat"]); lon = np.asarray(da["lon"])

    def _fake_probe(values, slon, slat, resolution, influence_radius):
        interp = _FakeInterp(lat, lon)
        return interp(values), interp

    monkeypatch.setattr(
        "feather.diag.heatwave_hotspots.nr_regrid_probe", _fake_probe,
    )
    # Force land-mask fallback to "all land" (avoid host-dependent Berkeley file).
    monkeypatch.setattr(
        "feather.diag.heatwave_hotspots._BE_LANDMASK_PATH",
        tmp_path / "no_such_landmask.nc",
    )
    return d


# ── Registration ─────────────────────────────────────────────────────────────

def test_registered():
    names = [d["name"] for d in list_diagnostics()]
    assert "heatwave_hotspots" in names
    cls = get_diagnostic("heatwave_hotspots")
    assert cls.variables == ["tasmax"]
    assert cls.group == "extremes"


def test_region_set_count():
    assert len(_REGIONS) == 10
    assert {r["key"] for r in _REGIONS} >= {"europe", "siberia", "nafrica"}


# ── Percentile computation ───────────────────────────────────────────────────

def test_compute_percentiles_orders(diag):
    da = _make_daily_tasmax()
    ds = diag._compute_percentiles(da)
    assert set(ds.data_vars) == {"p99", "p875"}
    assert ds["p99"].dims == ("year", "lat", "lon")
    assert ds.sizes["year"] == 5
    # The 99th percentile is always >= the 87.5th percentile.
    assert bool((ds["p99"] >= ds["p875"]).all())


def test_compute_percentiles_empty_raises(diag):
    da = _make_daily_tasmax()
    diag.period = ("2050", "2051")  # no data in this window
    with pytest.raises(ValueError):
        diag._compute_percentiles(da.sel(time=slice("2050", "2051")))


# ── Trend statistics ─────────────────────────────────────────────────────────

def test_ols_trend_stats_recovers_slope():
    years = np.arange(1980, 1990)
    lat = np.array([0.0, 10.0]); lon = np.array([0.0, 20.0, 40.0])
    # D increases by 0.3 per year everywhere → 3.0 per decade.
    base = (years - 1980) * 0.3
    data = np.broadcast_to(base[:, None, None], (len(years), 2, 3)).astype(float)
    stack = xr.DataArray(
        data, dims=("year", "lat", "lon"),
        coords={"year": years, "lat": lat, "lon": lon},
    )
    slope, pval = HeatwaveHotspotsDiag._ols_trend_stats(stack)
    assert slope.dims == ("lat", "lon")
    np.testing.assert_allclose(slope.values, 3.0, atol=1e-6)
    assert bool((pval.values < 0.05).all())  # perfect trend → significant


def test_slope_nan_safe():
    assert np.isnan(HeatwaveHotspotsDiag._slope(np.array([1.0]), np.array([2.0])))
    s = HeatwaveHotspotsDiag._slope(np.arange(5.0), 2 * np.arange(5.0))
    assert abs(s - 2.0) < 1e-9


def test_bootstrap_ci_brackets_trend(diag):
    rng = np.random.default_rng(1)
    x = np.arange(1980.0, 2000.0)
    y = (x - 1980) * 0.5 + rng.normal(0, 0.05, size=len(x))
    lo, hi = diag._bootstrap_ci(x, y, rng)
    assert lo <= hi
    assert lo <= 0.5 <= hi  # CI brackets the true slope (~0.5/yr)


# ── Region geometry ──────────────────────────────────────────────────────────

def test_region_box_mask_prime_meridian(diag):
    lats = np.array([50.0, 55.0])
    lons = np.array([2.0, 8.0, 200.0, 358.0])  # 0..360 convention
    europe = next(r for r in _REGIONS if r["key"] == "europe")  # lon (-5, 12)
    mask = diag._region_box_mask(lats, lons, europe)
    # 2°E, 8°E, 358°E(=-2) inside; 200°E outside.
    assert bool(mask.sel(lon=2.0).all())
    assert bool(mask.sel(lon=358.0).all())
    assert not bool(mask.sel(lon=200.0).any())


def test_region_box_mask_simple(diag):
    lats = np.array([30.0, 60.0])
    lons = np.array([100.0, 300.0])
    china = next(r for r in _REGIONS if r["key"] == "china")  # lat(27,37) lon(95,110)
    mask = diag._region_box_mask(lats, lons, china)
    assert bool(mask.sel(lat=30.0, lon=100.0))
    assert not bool(mask.sel(lat=60.0, lon=300.0))


def test_regional_series_area_weighted(diag):
    lats = np.array([10.0, 20.0]); lons = np.array([100.0, 105.0])
    years = np.arange(1980, 1983)
    d = xr.DataArray(
        np.ones((3, 2, 2)) * 4.0, dims=("year", "lat", "lon"),
        coords={"year": years, "lat": lats, "lon": lons},
    )
    region_mask = xr.DataArray(np.ones((2, 2), bool), dims=("lat", "lon"),
                               coords={"lat": lats, "lon": lons})
    land = region_mask
    areas = np.ones((2, 2))
    series = diag._regional_series(d, region_mask, land, areas)
    np.testing.assert_allclose(series, 4.0)


# ── Land mask fallback ───────────────────────────────────────────────────────

def test_common_land_mask_all_land_fallback(diag):
    lats = np.array([0.0, 30.0]); lons = np.array([10.0, 200.0])
    mask = diag._common_land_mask(lats, lons)
    assert bool(mask.all())  # no sftlf, no Berkeley → all land


# ── End-to-end compute + plot (stubbed regridding) ───────────────────────────

def test_compute_and_plot_smoke(diag):
    results = diag.compute()
    assert results["ref_model"] == "ERA5"
    assert set(results["models"]) == {"ModelA", "ModelB"}
    assert len(results["regions"]) == 10
    assert results["trend_d"]["ERA5"].dims == ("lat", "lon")
    # tail-widening trend should be positive (we built a widening distribution)
    assert float(np.nanmean(results["trend_d"]["ERA5"].values)) > 0

    figs = diag.plot(results)
    fig_ids = {meta["figure_id"] for _, meta in figs}
    assert fig_ids == set(HeatwaveHotspotsDiag._FIG_IDS)
    import matplotlib.pyplot as plt
    for fig, _ in figs:
        plt.close(fig)


def test_nc_checkpoint_roundtrip(diag, tmp_path):
    da = _make_daily_tasmax()
    ds = diag._compute_percentiles(da)
    diag._save_nc("ModelA", ds)
    assert diag._nc_path("ModelA").exists()
    reloaded = diag._load_or_compute_percs("ModelA")
    assert {"p99", "p875"}.issubset(reloaded.data_vars)
    np.testing.assert_allclose(reloaded["p99"].values, ds["p99"].values)
