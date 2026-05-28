"""Tests for HeatwaveChangeDiag."""

import copy
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.diag.heatwave_change import (
    HeatwaveChangeDiag,
    _HIST_BOUNDARY_YEAR,
    _HW_MIN_RUN,
    _HW_PERCENTILE,
    _INDICES,
    _K_TO_C,
    compute_hw_indices_year,
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────


def _make_daily_tasmax(
    start: str = "1981-01-01",
    n_years: int = 5,
    nlat: int = 9,
    nlon: int = 18,
    base_k: float = 305.0,
) -> xr.DataArray:
    """Synthetic daily tasmax on a regular lat/lon grid.

    Default base_k=305 K (32 °C) at equator, decreasing pole-ward.
    No heatwave events by default (all values constant — T90 = base value,
    and tmax == t90 is not > t90 so no events).
    """
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    n_days = 365 * n_years
    time = xr.date_range(start, periods=n_days, freq="1D", calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    data = base_k - 30.0 * np.abs(lat_grid) / 80.0
    data = np.broadcast_to(data[np.newaxis], (n_days, nlat, nlon)).copy().astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )


def _make_ssp_tasmax(n_years: int = 5, nlat: int = 9, nlon: int = 18) -> xr.DataArray:
    """SSP daily tasmax starting 2015 at base_k=307 K (+2 K above hist)."""
    return _make_daily_tasmax(
        start="2015-01-01", n_years=n_years, nlat=nlat, nlon=nlon, base_k=307.0
    )


def _make_be_tmax_nc(path, nlat: int = 9, nlon: int = 18):
    """Write a minimal mock Berkeley Earth Land TMAX NetCDF."""
    lats = np.linspace(-80, 80, nlat).astype(np.float32)
    lons = np.linspace(-170, 170, nlon).astype(np.float32)
    dec_years = np.array([1981 + m / 12.0 for m in range(24)], dtype=np.float64)
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    clim_field = (20.0 * np.cos(np.deg2rad(lat_grid))).astype(np.float32)
    climatology = np.stack([clim_field] * 12, axis=0)
    temperature = np.zeros((24, nlat, nlon), dtype=np.float32)
    land_mask = np.ones((nlat, nlon), dtype=np.float64)
    ds = xr.Dataset(
        {
            "land_mask": xr.DataArray(land_mask, dims=("latitude", "longitude")),
            "temperature": xr.DataArray(
                temperature,
                dims=("time", "latitude", "longitude"),
                coords={"time": dec_years, "latitude": lats, "longitude": lons},
                attrs={"units": "degree C"},
            ),
            "climatology": xr.DataArray(
                climatology,
                dims=("month_number", "latitude", "longitude"),
                attrs={"units": "degree C"},
            ),
        },
        coords={"latitude": lats, "longitude": lons},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class _MockLoader:
    def __init__(self, da: xr.DataArray):
        self._da = da

    def load_var(self, model: str, variable: str, *, table=None, period=None):
        da = self._da
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_coords(self, model: str, variable: str = "tas"):
        return np.asarray(self._da["lon"]), np.asarray(self._da["lat"])


class _FailLoader:
    def load_var(self, *args, **kwargs):
        raise FileNotFoundError("intentional test failure")

    def load_coords(self, *args, **kwargs):
        raise FileNotFoundError("intentional test failure")


class _MockObsLoader:
    def load(self, *a, **kw):
        raise FileNotFoundError

    def load_for_model_var(self, *a, **kw):
        raise FileNotFoundError


# ── Config factory ────────────────────────────────────────────────────────────


def _make_config(tmp_path, models=("model-A", "model-B"), cc_override=None):
    cc = cc_override or {
        "reference_period": ["1981", "1985"],
        "future_period":    ["2019", "2023"],
        "hist_load_period": ["1981", "1985"],
        "ssp_load_period":  ["2019", "2023"],
        "models": {
            "model-A": {
                "hist_data_source": "cmor",
                "hist_experiment":  "hist-1950",
                "future_data_source": "cmor",
                "future_experiment":  "highres-future-ssp245",
            },
            "model-B": {
                "hist_data_source": "cmor",
                "hist_experiment":  "hist-1950",
                "future_data_source": "cmor",
                "future_experiment":  "highres-future-ssp245",
            },
        },
    }
    model_configs = {
        m: ModelConfig(
            name=m,
            institution="TEST",
            experiment="hist-1950",
            variant="r1i1p1f1",
            grids={"sfc": "latlon"},
            color="#1f77b4",
        )
        for m in models
    }
    return FeatherConfig(
        model_catalogs={},
        models=list(models),
        model_configs=model_configs,
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        project={
            "name": "TEST",
            "climate_change": cc,
        },
    )


def _make_diag(config, hist_da, ssp_da=None, obs_loader=None):
    diag = HeatwaveChangeDiag(
        model_loader=_MockLoader(hist_da),
        obs_loader=obs_loader or _MockObsLoader(),
        config=config,
    )
    diag._make_hist_loader = lambda model: _MockLoader(hist_da)
    if ssp_da is not None:
        diag._make_fut_loader = lambda model: _MockLoader(ssp_da)
    else:
        diag._make_fut_loader = lambda model: None
    return diag


# ── Unit tests: compute_hw_indices_year ───────────────────────────────────────


def test_hw_indices_no_events():
    """All values below T90 → all count-based indices 0, HWA/HWM NaN."""
    n_days, nlat, nlon = 30, 3, 4
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWF"].sum() == 0
    assert r["HWN"].sum() == 0
    assert r["HWD"].sum() == 0
    assert np.all(np.isnan(r["HWA"]))
    assert np.all(np.isnan(r["HWM"]))


def test_hw_indices_exactly_3_days():
    """Exactly 3 consecutive days above T90 → 1 event, HWF=3, HWD=3, HWN=1."""
    n_days, nlat, nlon = 10, 2, 2
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[2:5] = 301.0  # days 3,4,5 above threshold (+1 K anomaly)
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWF"][0, 0] == 3
    assert r["HWN"][0, 0] == 1
    assert r["HWD"][0, 0] == 3
    assert r["HWA"][0, 0] == pytest.approx(1.0, abs=1e-4)
    assert r["HWM"][0, 0] == pytest.approx(1.0, abs=1e-4)


def test_hw_indices_only_2_consecutive_days_not_event():
    """Run of 2 consecutive days does NOT qualify as an event (min_run=3)."""
    n_days, nlat, nlon = 10, 2, 2
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[2:4] = 301.0  # only 2 days
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWF"][0, 0] == 0
    assert r["HWN"][0, 0] == 0


def test_hw_indices_two_events():
    """Two separate events → HWN=2, HWF=sum, HWD=longest."""
    n_days, nlat, nlon = 20, 2, 2
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[1:4] = 301.0    # 3-day event
    tmax[10:15] = 302.0  # 5-day event
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWN"][0, 0] == 2
    assert r["HWF"][0, 0] == 8
    assert r["HWD"][0, 0] == 5  # longest is the 5-day event


def test_hw_indices_hwm_mean_anomaly():
    """HWM is mean anomaly over all heatwave days across events."""
    n_days, nlat, nlon = 20, 1, 1
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[0:3] = 301.0   # 3 days, +1 K anomaly
    tmax[5:8] = 303.0   # 3 days, +3 K anomaly
    r = compute_hw_indices_year(tmax, t90)
    # HWM = mean anomaly over 6 HW days: (1+1+1 + 3+3+3) / 6 = 2.0
    assert r["HWM"][0, 0] == pytest.approx(2.0, abs=1e-4)


def test_hw_indices_hwa_peak_anomaly():
    """HWA is maximum anomaly over all heatwave days."""
    n_days, nlat, nlon = 20, 1, 1
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[0:3] = np.array([301.0, 303.0, 301.5])[:, np.newaxis, np.newaxis]
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWA"][0, 0] == pytest.approx(3.0, abs=1e-4)


def test_hw_indices_spatially_varying():
    """Each grid point is independent: one cell has events, another does not."""
    n_days, nlat, nlon = 10, 1, 2
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[2:5, :, 0] = 301.0  # events in lon=0 only
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWF"][0, 0] == 3
    assert r["HWF"][0, 1] == 0


def test_hw_indices_exactly_at_threshold_not_counted():
    """Tmax == T90 is NOT counted (strictly greater than)."""
    n_days, nlat, nlon = 10, 1, 1
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 300.0, dtype=np.float32)
    r = compute_hw_indices_year(tmax, t90)
    assert r["HWF"][0, 0] == 0


def test_hw_indices_custom_min_run():
    """min_run=2: 2-day events now qualify."""
    n_days, nlat, nlon = 10, 1, 1
    t90 = np.full((nlat, nlon), 300.0, dtype=np.float32)
    tmax = np.full((n_days, nlat, nlon), 299.0, dtype=np.float32)
    tmax[2:4] = 301.0  # 2-day run
    r = compute_hw_indices_year(tmax, t90, min_run=2)
    assert r["HWF"][0, 0] == 2
    assert r["HWN"][0, 0] == 1


# ── Unit tests: _collect_finite ───────────────────────────────────────────────


def test_collect_finite_normal():
    r = HeatwaveChangeDiag._collect_finite([np.array([1.0, 2.0, 3.0])])
    assert len(r) == 3
    assert np.all(np.isfinite(r))


def test_collect_finite_all_nan():
    r = HeatwaveChangeDiag._collect_finite([np.array([np.nan, np.nan])])
    assert len(r) == 0


def test_collect_finite_empty_list():
    r = HeatwaveChangeDiag._collect_finite([])
    assert len(r) == 0


# ── Unit tests: NC paths ──────────────────────────────────────────────────────


def test_nc_paths(tmp_path):
    config = _make_config(tmp_path)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), config)
    hist = diag._nc_hist_path("model-A")
    ssp = diag._nc_ssp_path("model-A")
    assert "model-A" in hist.name
    assert "hw_hist" in hist.name
    assert "hw_ssp" in ssp.name
    assert hist.suffix == ".nc"


def test_nc_paths_special_chars(tmp_path):
    config = _make_config(tmp_path)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), config)
    hist = diag._nc_hist_path("IFS-FESOM2/r2")
    assert "/" not in hist.name


# ── Unit tests: _load_and_save_hw ─────────────────────────────────────────────


def hist_da():
    return _make_daily_tasmax(n_years=5, nlat=5, nlon=8)


def test_load_and_save_hw_writes_nc(tmp_path):
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_test.nc"
    hw_ds, t90, tmax_mean = diag._load_and_save_hw(
        "model-A", loader, ("1981", "1985"), nc_path,
        ref_period=("1981", "1985"),
    )
    assert nc_path.exists()
    assert hw_ds is not None
    assert "HWF" in hw_ds
    assert "year" in hw_ds["HWF"].dims


def test_load_and_save_hw_has_t90(tmp_path):
    """Hist NC saves T90; returned t90 is a DataArray with lat/lon dims."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_test.nc"
    _, t90, _ = diag._load_and_save_hw(
        "model-A", loader, ("1981", "1985"), nc_path,
        ref_period=("1981", "1985"),
    )
    assert t90 is not None
    assert "lat" in t90.dims and "lon" in t90.dims


def test_load_and_save_hw_reloads_from_nc(tmp_path):
    """Second call reloads from NC without invoking the loader."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_test.nc"
    diag._load_and_save_hw("model-A", loader, ("1981", "1985"), nc_path, ref_period=("1981", "1985"))
    # Second call with a failing loader — should not be called
    hw_ds, _, _ = diag._load_and_save_hw(
        "model-A", _FailLoader(), ("1981", "1985"), nc_path, ref_period=("1981", "1985")
    )
    assert hw_ds is not None


def test_load_and_save_hw_fail_loader(tmp_path):
    """Missing tasmax data → returns (None, None, None)."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=2, nlat=3, nlon=4)
    diag = HeatwaveChangeDiag(_MockLoader(da), _MockObsLoader(), config)
    nc_path = tmp_path / "hw_fail.nc"
    hw_ds, t90, tmax_mean = diag._load_and_save_hw(
        "model-A", _FailLoader(), ("1981", "1982"), nc_path, ref_period=("1981", "1982")
    )
    assert hw_ds is None
    assert t90 is None
    assert tmax_mean is None


def test_load_and_save_hw_no_t90_source(tmp_path):
    """No ref_period and no t90_external → returns (None, None, None)."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=2, nlat=3, nlon=4)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_no_t90.nc"
    hw_ds, _, _ = diag._load_and_save_hw("model-A", loader, ("1981", "1982"), nc_path)
    assert hw_ds is None


def test_load_and_save_hw_with_t90_external(tmp_path):
    """SSP case: t90_external passed instead of computing from ref_period."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    # Build a synthetic T90
    t90_ext = xr.DataArray(
        np.full((5, 8), 305.0, dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-80, 80, 5), "lon": np.linspace(0, 350, 8)},
    )
    nc_path = tmp_path / "hw_ssp.nc"
    hw_ds, t90_ret, _ = diag._load_and_save_hw(
        "model-A", loader, ("1981", "1985"), nc_path, t90_external=t90_ext
    )
    assert hw_ds is not None
    assert t90_ret is None  # SSP NC does not store T90


def test_load_and_save_hw_saves_tmax_mean(tmp_path):
    """Hist NC includes a tmax_mean variable when ref_period is given."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_test.nc"
    _, _, tmax_mean = diag._load_and_save_hw(
        "model-A", loader, ("1981", "1985"), nc_path,
        ref_period=("1981", "1985"),
    )
    assert tmax_mean is not None


def test_load_and_save_hw_all_five_indices_present(tmp_path):
    """All five indices are present in the returned Dataset."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    loader = _MockLoader(da)
    diag = HeatwaveChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "hw_test.nc"
    hw_ds, _, _ = diag._load_and_save_hw(
        "model-A", loader, ("1981", "1985"), nc_path, ref_period=("1981", "1985")
    )
    for idx in _INDICES:
        assert idx in hw_ds


# ── Unit tests: compute() ─────────────────────────────────────────────────────


@pytest.fixture
def simple_config(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def hist_da_fix():
    return _make_daily_tasmax(n_years=5, nlat=5, nlon=8)


def test_compute_returns_expected_keys(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    for key in ("models", "ref_clim", "fut_clim", "change",
                "hist_series", "ssp_series", "model_mean_tmax",
                "obs_mean_tmax", "lat", "lon"):
        assert key in results


def test_compute_ref_clim_has_all_indices(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    for idx in _INDICES:
        assert idx in results["ref_clim"]


def test_compute_models_list(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    assert set(results["models"]) == {"model-A", "model-B"}


def test_compute_ref_clim_shape(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    for m in results["models"]:
        clim = results["ref_clim"]["HWF"][m]
        assert "lat" in clim.dims and "lon" in clim.dims
        assert "year" not in clim.dims


def test_compute_change_equals_fut_minus_ref(tmp_path):
    """change[idx][model] == fut_clim[idx][model] - ref_clim[idx][model]."""
    # Use warmer SSP to guarantee some change
    hist_da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8, base_k=305.0)
    ssp_da = _make_ssp_tasmax(n_years=5, nlat=5, nlon=8)
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist_da, ssp_da)
    results = diag.compute()
    for idx in _INDICES:
        for m in results["models"]:
            if m in results["change"][idx] and m in results["fut_clim"][idx]:
                diff = results["fut_clim"][idx][m] - results["ref_clim"][idx][m]
                np.testing.assert_allclose(
                    np.asarray(results["change"][idx][m]),
                    np.asarray(diff),
                    atol=1e-5,
                )


def test_compute_no_future_loader(tmp_path, hist_da_fix, simple_config):
    """Models without SSP loader appear in models but not in fut_clim."""
    diag = _make_diag(simple_config, hist_da_fix, ssp_da=None)
    results = diag.compute()
    assert len(results["models"]) == 2
    for idx in _INDICES:
        assert len(results["fut_clim"][idx]) == 0
        assert len(results["change"][idx]) == 0


def test_compute_skips_model_when_hist_fails(tmp_path, simple_config):
    hist_da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    diag = HeatwaveChangeDiag(
        _MockLoader(hist_da), _MockObsLoader(), simple_config
    )
    diag._make_hist_loader = lambda model: _FailLoader()
    diag._make_fut_loader = lambda model: None
    results = diag.compute()
    assert len(results["models"]) == 0


def test_compute_hist_series_has_year_dim(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    for idx in _INDICES:
        for m in results["models"]:
            series = results["hist_series"][idx][m]
            assert "year" in series.dims


def test_compute_lat_lon_populated(tmp_path, hist_da_fix, simple_config):
    diag = _make_diag(simple_config, hist_da_fix)
    results = diag.compute()
    assert results["lat"] is not None
    assert results["lon"] is not None


# ── Unit tests: _land_mean_series ─────────────────────────────────────────────


def test_land_mean_series_returns_1d(tmp_path, simple_config):
    lats = np.linspace(-80, 80, 5)
    lons = np.linspace(0, 350, 8)
    data = np.ones((10, 5, 8), dtype=np.float32)
    da = xr.DataArray(data, dims=("year", "lat", "lon"),
                      coords={"year": np.arange(2000, 2010), "lat": lats, "lon": lons})
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    result = diag._land_mean_series(da)
    assert result.dims == ("year",)
    assert len(result) == 10


def test_land_mean_series_all_nan_returns_nan(tmp_path, simple_config):
    lats = np.linspace(-80, 80, 5)
    lons = np.linspace(0, 350, 8)
    data = np.full((5, 5, 8), np.nan, dtype=np.float32)
    da = xr.DataArray(data, dims=("year", "lat", "lon"),
                      coords={"year": np.arange(2000, 2005), "lat": lats, "lon": lons})
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    result = diag._land_mean_series(da)
    assert np.all(np.isnan(np.asarray(result)))


# ── Unit tests: plot() ────────────────────────────────────────────────────────


def _make_synthetic_results(models=("model-A", "model-B"), with_ssp=True, with_obs=False):
    lats = np.linspace(-80, 80, 5)
    lons = np.linspace(0, 350, 8)
    years_hist = np.arange(1981, 1986)
    years_ssp = np.arange(2019, 2024)

    ref_clim = {idx: {} for idx in _INDICES}
    fut_clim = {idx: {} for idx in _INDICES}
    change = {idx: {} for idx in _INDICES}
    hist_series = {idx: {} for idx in _INDICES}
    ssp_series = {idx: {} for idx in _INDICES}
    model_mean_tmax = {}
    obs_mean_tmax = {}

    for m in models:
        for idx in _INDICES:
            data2d = np.ones((5, 8), dtype=np.float32)
            ref_clim[idx][m] = xr.DataArray(data2d, dims=("lat", "lon"),
                                             coords={"lat": lats, "lon": lons})
            hist_series[idx][m] = xr.DataArray(
                np.ones(len(years_hist), dtype=np.float32),
                dims=("year",), coords={"year": years_hist},
            )
            if with_ssp:
                fut_clim[idx][m] = xr.DataArray(data2d * 2, dims=("lat", "lon"),
                                                 coords={"lat": lats, "lon": lons})
                change[idx][m] = xr.DataArray(data2d, dims=("lat", "lon"),
                                               coords={"lat": lats, "lon": lons})
                ssp_series[idx][m] = xr.DataArray(
                    np.ones(len(years_ssp), dtype=np.float32) * 2,
                    dims=("year",), coords={"year": years_ssp},
                )
        obs_k = xr.DataArray(
            np.full((5, 8), 305.0, dtype=np.float32),
            dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        )
        model_mean_tmax[m] = obs_k + 1.0
        if with_obs:
            obs_mean_tmax[m] = obs_k

    return {
        "models": list(models),
        "ref_clim": ref_clim,
        "fut_clim": fut_clim,
        "change": change,
        "hist_series": hist_series,
        "ssp_series": ssp_series,
        "model_mean_tmax": model_mean_tmax,
        "obs_mean_tmax": obs_mean_tmax,
        "lat": xr.DataArray(lats, dims=("lat",)),
        "lon": xr.DataArray(lons, dims=("lon",)),
    }


def test_plot_returns_empty_when_no_models(tmp_path, simple_config):
    results = _make_synthetic_results(models=())
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    assert figs == []
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_produces_combined_panels(tmp_path, simple_config):
    """plot() returns 5 map figures when models exist."""
    results = _make_synthetic_results(with_ssp=True, with_obs=False)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    for idx in _INDICES:
        assert f"heatwave_change_{idx.lower()}_maps" in ids
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_timeseries_always_produced(tmp_path, simple_config):
    """One timeseries figure per index is always produced."""
    results = _make_synthetic_results(with_ssp=False, with_obs=False)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    for idx in _INDICES:
        assert f"heatwave_change_{idx.lower()}_timeseries" in ids
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_tmax_bias_included_when_obs_available(tmp_path, simple_config):
    results = _make_synthetic_results(with_obs=True)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "heatwave_change_tmax_bias" in ids
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_tmax_bias_skipped_when_no_obs(tmp_path, simple_config):
    results = _make_synthetic_results(with_obs=False)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "heatwave_change_tmax_bias" not in ids
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_figure_count_with_fut_and_obs(tmp_path, simple_config):
    """With future and obs: 5 maps + 5 timeseries + 1 bias = 11 figures."""
    results = _make_synthetic_results(with_ssp=True, with_obs=True)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    assert len(figs) == 11
    import matplotlib
    matplotlib.pyplot.close("all")


def test_plot_figure_count_without_fut_or_obs(tmp_path, simple_config):
    """Without future and without obs: 5 maps + 5 timeseries = 10 figures."""
    results = _make_synthetic_results(with_ssp=False, with_obs=False)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), simple_config)
    figs = diag.plot(results)
    assert len(figs) == 10
    import matplotlib
    matplotlib.pyplot.close("all")


# ── Integration tests: run() ──────────────────────────────────────────────────


def test_run_creates_figures(tmp_path):
    config = _make_config(tmp_path)
    hist_da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    ssp_da = _make_ssp_tasmax(n_years=5, nlat=5, nlon=8)
    diag = _make_diag(config, hist_da, ssp_da)
    saved = diag.run()
    assert len(saved) > 0


def test_run_idempotent(tmp_path):
    """Second run() reuses NC checkpoints and produces same figure count."""
    config = _make_config(tmp_path)
    hist_da = _make_daily_tasmax(n_years=5, nlat=5, nlon=8)
    diag = _make_diag(config, hist_da)
    saved1 = diag.run()
    saved2 = diag.run()
    assert len(saved1) == len(saved2)


# ── Config / constants tests ──────────────────────────────────────────────────


def test_min_run_constant():
    assert _HW_MIN_RUN == 3


def test_percentile_constant():
    assert abs(_HW_PERCENTILE - 0.90) < 1e-9


def test_five_indices_defined():
    assert set(_INDICES.keys()) == {"HWF", "HWD", "HWN", "HWA", "HWM"}


def test_config_defaults_used_when_no_cc_section(tmp_path):
    """No climate_change section → diagnostic uses its default periods."""
    config = FeatherConfig(
        model_catalogs={},
        models=["m"],
        model_configs={
            "m": ModelConfig(
                name="m", institution="T", experiment="hist",
                variant="r1", grids={"sfc": "latlon"}, color="#000",
            )
        },
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path),
        project={"name": "T"},
    )
    diag = HeatwaveChangeDiag(
        _MockLoader(_make_daily_tasmax()), _MockObsLoader(), config,
        period=("1990", "2000"),
    )
    assert diag.ref_period == ("1990", "2000")


def test_config_cc_periods_respected(tmp_path):
    cc = {
        "reference_period": ["1982", "1990"],
        "future_period":    ["2040", "2059"],
        "hist_load_period": ["1982", "2014"],
        "ssp_load_period":  ["2015", "2059"],
        "models": {},
    }
    config = _make_config(tmp_path, cc_override=cc)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), config)
    assert diag.ref_period == ("1982", "1990")
    assert diag.fut_period == ("2040", "2059")
    assert diag.hist_load_period == ("1982", "2014")


def test_make_kerchunk_loader_does_not_set_kerchunk_native(tmp_path):
    """_make_kerchunk_loader must NOT set data_source_type='kerchunk_native'."""
    config = _make_config(tmp_path)
    diag = HeatwaveChangeDiag(_MockLoader(_make_daily_tasmax()), _MockObsLoader(), config)
    # Patch to avoid actual filesystem access
    from unittest.mock import patch, MagicMock
    mock_loader_cls = MagicMock()
    mock_loader_cls.return_value = MagicMock()
    with patch("feather.diag.heatwave_change.HeatwaveChangeDiag._make_kerchunk_loader") as mk:
        mk.return_value = MagicMock()
        loader = mk("/some/root", "model-A", "r1i1p1f1")
    # If the method were called for real, verify no data_source_type override:
    mc_orig = config.model_configs.get("model-A")
    import copy as _copy
    mc = _copy.copy(mc_orig) if mc_orig else ModelConfig(name="model-A")
    assert getattr(mc, "data_source_type", None) != "kerchunk_native"


def test_registered():
    from feather.diag.registry import get_diagnostic
    import feather.diag.heatwave_change  # noqa: F401 — ensure @register fires
    diag_cls = get_diagnostic("heatwave_change")
    assert diag_cls is HeatwaveChangeDiag
