"""Tests for TropicalNightsChangeDiag."""

import copy
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.diag.tropical_nights_change import (
    TropicalNightsChangeDiag,
    _HIST_BOUNDARY_YEAR,
    _K_TO_C,
    _TN_THRESHOLD_K,
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────


def _make_daily_tasmin(
    start: str = "1981-01-01",
    n_years: int = 5,
    nlat: int = 9,
    nlon: int = 18,
    base_k: float = 293.0,
) -> xr.DataArray:
    """Synthetic daily tasmin on a regular lat/lon grid.

    Default base_k=293.0 places the equator 0.15 K *below* the 20 °C threshold,
    so hist has 0 tropical nights.  Use a higher base_k (e.g. 294.5) to push
    equatorial cells above threshold (useful for SSP data).
    """
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    n_days = 365 * n_years
    time = xr.date_range(start, periods=n_days, freq="1D", calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    # Gradient: tropics ~base_k, poles much colder
    data = base_k - 34.0 * np.abs(lat_grid) / 80.0
    data = np.broadcast_to(data[np.newaxis], (n_days, nlat, nlon)).copy().astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )


def _make_ssp_tasmin(n_years: int = 5, nlat: int = 9, nlon: int = 18) -> xr.DataArray:
    """SSP daily tasmin starting 2015: equator at 294.5 K (above 293.15 threshold).

    Combined with default hist (equator=293.0 K, below threshold), the change map
    will have positive values at equatorial cells.
    """
    return _make_daily_tasmin(
        start="2015-01-01", n_years=n_years, nlat=nlat, nlon=nlon, base_k=294.5
    )


def _make_be_tmin_nc(path, nlat: int = 9, nlon: int = 18):
    """Write a minimal mock Berkeley Earth Land TMIN NetCDF."""
    lats = np.linspace(-80, 80, nlat).astype(np.float32)
    lons = np.linspace(-170, 170, nlon).astype(np.float32)
    dec_years = np.array([1981 + m / 12.0 for m in range(24)], dtype=np.float64)
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    clim_field = (10.0 * np.cos(np.deg2rad(lat_grid))).astype(np.float32)
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
    """Minimal loader that returns a pre-loaded DataArray for any model/variable."""

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
    """Loader that always raises FileNotFoundError."""

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
    """Build a TropicalNightsChangeDiag with injected mock loaders."""
    diag = TropicalNightsChangeDiag(
        model_loader=_MockLoader(hist_da),
        obs_loader=obs_loader or _MockObsLoader(),
        config=config,
    )
    # Patch the loader factories so they return our controlled loaders
    diag._make_hist_loader = lambda model: _MockLoader(hist_da)
    if ssp_da is not None:
        diag._make_fut_loader = lambda model: _MockLoader(ssp_da)
    else:
        diag._make_fut_loader = lambda model: None
    return diag


# ── Unit tests: _count_tn_days ────────────────────────────────────────────────


def test_count_tn_days_basic():
    """Days above threshold are counted per year."""
    lats = np.array([0.0, 60.0])
    lons = np.array([0.0, 90.0])
    n_days = 365 * 2
    time = xr.date_range("1981-01-01", periods=n_days, freq="1D", calendar="standard")
    # All values above threshold
    data = np.full((n_days, 2, 2), _TN_THRESHOLD_K + 1.0, dtype=np.float32)
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsChangeDiag._count_tn_days(da)
    assert "year" in result.dims
    assert result.dims == ("year", "lat", "lon")
    # All days count — expect 365 per year
    assert int(result.isel(year=0, lat=0, lon=0).values) == 365


def test_count_tn_days_none_above_threshold():
    """Returns zero when all values are below threshold."""
    lats = np.array([60.0])
    lons = np.array([0.0])
    time = xr.date_range("1981-01-01", periods=365, freq="1D", calendar="standard")
    data = np.full((365, 1, 1), _TN_THRESHOLD_K - 5.0, dtype=np.float32)
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsChangeDiag._count_tn_days(da)
    assert int(result.sum().values) == 0


def test_count_tn_days_exactly_at_threshold():
    """Values exactly equal to threshold are NOT counted (strictly greater)."""
    lats = np.array([0.0])
    lons = np.array([0.0])
    time = xr.date_range("1981-01-01", periods=365, freq="1D", calendar="standard")
    data = np.full((365, 1, 1), _TN_THRESHOLD_K, dtype=np.float32)
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsChangeDiag._count_tn_days(da)
    assert int(result.sum().values) == 0


def test_count_tn_days_threshold_value():
    """_TN_THRESHOLD_K is 293.15 (20 °C)."""
    assert abs(_TN_THRESHOLD_K - (_K_TO_C + 20.0)) < 1e-6


# ── Unit tests: _safe_vmax ────────────────────────────────────────────────────


def test_safe_vmax_normal():
    data = {"m": np.array([0.0, 10.0, 20.0, 100.0])}
    result = TropicalNightsChangeDiag._safe_vmax(data)
    assert result > 0


def test_safe_vmax_all_nan():
    data = {"m": np.array([np.nan, np.nan])}
    result = TropicalNightsChangeDiag._safe_vmax(data)
    assert result == 1.0


def test_safe_vmax_empty_dict():
    result = TropicalNightsChangeDiag._safe_vmax({})
    assert result == 1.0


# ── Unit tests: NC paths ──────────────────────────────────────────────────────


def test_nc_paths(tmp_path):
    config = _make_config(tmp_path)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    hist = diag._nc_hist_path("model-A")
    ssp = diag._nc_ssp_path("model-A")
    assert "model-A" in hist.name
    assert "tn_hist" in hist.name
    assert "tn_ssp" in ssp.name
    assert hist.suffix == ".nc"
    assert ssp.suffix == ".nc"


def test_nc_paths_special_chars(tmp_path):
    """Slashes and spaces in model names are sanitised."""
    config = _make_config(tmp_path)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    hist = diag._nc_hist_path("IFS-FESOM2/r2")
    assert "/" not in hist.name


# ── Unit tests: _load_and_save_tn checkpoint ──────────────────────────────────


def test_load_and_save_tn_writes_nc(tmp_path):
    config = _make_config(tmp_path)
    da = _make_daily_tasmin(n_years=5)
    loader = _MockLoader(da)
    diag = TropicalNightsChangeDiag(
        loader, _MockObsLoader(), config
    )
    nc_path = tmp_path / "tn_test.nc"
    tn, tmin = diag._load_and_save_tn(
        "model-A", loader, ("1981", "1985"), nc_path,
        ref_period=("1981", "1985"),
    )
    assert nc_path.exists()
    assert tn is not None
    assert "year" in tn.dims


def test_load_and_save_tn_reloads_from_nc(tmp_path):
    """Second call should reload from checkpoint without calling loader."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmin(n_years=5)
    loader = _MockLoader(da)
    diag = TropicalNightsChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "tn_test.nc"
    # First call: compute + save
    tn1, _ = diag._load_and_save_tn("model-A", loader, ("1981", "1985"), nc_path)
    # Second call: should reload from NC (use a failing loader to prove it)
    tn2, _ = diag._load_and_save_tn("model-A", _FailLoader(), ("1981", "1985"), nc_path)
    assert tn2 is not None
    np.testing.assert_array_equal(tn1.values, tn2.values)


def test_load_and_save_tn_fail_loader(tmp_path):
    """Returns (None, None) when loader raises FileNotFoundError."""
    config = _make_config(tmp_path)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    nc_path = tmp_path / "tn_missing.nc"
    tn, tmin = diag._load_and_save_tn("model-X", _FailLoader(), ("1981", "1985"), nc_path)
    assert tn is None
    assert tmin is None


def test_load_and_save_tn_saves_tmin_mean(tmp_path):
    """tmin_mean is saved when ref_period is provided."""
    config = _make_config(tmp_path)
    da = _make_daily_tasmin(n_years=5)
    loader = _MockLoader(da)
    diag = TropicalNightsChangeDiag(loader, _MockObsLoader(), config)
    nc_path = tmp_path / "tn_tmin.nc"
    _, tmin = diag._load_and_save_tn(
        "model-A", loader, ("1981", "1985"), nc_path,
        ref_period=("1981", "1985"),
    )
    assert tmin is not None
    ds = xr.open_dataset(nc_path)
    assert "tmin_mean" in ds


# ── Unit tests: loader factories ─────────────────────────────────────────────


def test_make_cmor_loader_experiment(tmp_path):
    """_make_cmor_loader clones config with overridden experiment."""
    from feather.data.cmor_loader import CMORLoader

    config = _make_config(tmp_path, models=("IFS-FESOM2-SR",))
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    ldr = diag._make_cmor_loader("IFS-FESOM2-SR", "highres-future-ssp245")
    assert isinstance(ldr, CMORLoader)
    # The model's experiment in the cloned config should be the new experiment
    mc = ldr._config.model_configs.get("IFS-FESOM2-SR")
    assert mc is not None
    assert mc.experiment == "highres-future-ssp245"


def test_make_cmor_loader_does_not_mutate_original(tmp_path):
    """Cloning must not mutate the original config."""
    config = _make_config(tmp_path, models=("IFS-FESOM2-SR",))
    original_exp = config.model_configs["IFS-FESOM2-SR"].experiment
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    diag._make_cmor_loader("IFS-FESOM2-SR", "highres-future-ssp245")
    assert config.model_configs["IFS-FESOM2-SR"].experiment == original_exp


# ── Unit tests: _make_fut_loader with future_only_to ─────────────────────────


def test_make_fut_loader_returns_none_when_run_ends_early(tmp_path):
    """future_only_to < fut_period[0] means no future loader."""
    cc = {
        "reference_period": ["1981", "1985"],
        "future_period":    ["2031", "2050"],
        "hist_load_period": ["1981", "1985"],
        "ssp_load_period":  ["2015", "2050"],
        "models": {
            "model-A": {
                "hist_data_source":   "cmor",
                "hist_experiment":    "hist-1950",
                "future_data_source": "cmor",
                "future_experiment":  "highres-future-ssp245",
                "future_only_to":     "2030",  # ends before 2031
            },
        },
    }
    config = _make_config(tmp_path, models=("model-A",), cc_override=cc)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    assert diag._make_fut_loader("model-A") is None


def test_make_fut_loader_returns_loader_when_run_reaches_future(tmp_path):
    """future_only_to >= fut_period[0] means future loader is created."""
    from feather.data.cmor_loader import CMORLoader

    cc = {
        "reference_period": ["1981", "1985"],
        "future_period":    ["2031", "2050"],
        "hist_load_period": ["1981", "1985"],
        "ssp_load_period":  ["2015", "2050"],
        "models": {
            "model-A": {
                "hist_data_source":   "cmor",
                "hist_experiment":    "hist-1950",
                "future_data_source": "cmor",
                "future_experiment":  "highres-future-ssp245",
                "future_only_to":     "2050",  # covers future period
            },
        },
    }
    config = _make_config(tmp_path, models=("model-A",), cc_override=cc)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    ldr = diag._make_fut_loader("model-A")
    assert isinstance(ldr, CMORLoader)


def test_make_fut_loader_returns_none_when_no_experiment(tmp_path):
    """No future_experiment → future loader is None."""
    cc = {
        "reference_period": ["1981", "1985"],
        "future_period":    ["2031", "2050"],
        "hist_load_period": ["1981", "1985"],
        "ssp_load_period":  ["2015", "2050"],
        "models": {
            "model-A": {
                "hist_data_source": "cmor",
                "hist_experiment":  "hist-1950",
                # no future_experiment
            },
        },
    }
    config = _make_config(tmp_path, models=("model-A",), cc_override=cc)
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    assert diag._make_fut_loader("model-A") is None


# ── Unit tests: compute() ─────────────────────────────────────────────────────


@pytest.fixture
def hist_da():
    return _make_daily_tasmin(start="1981-01-01", n_years=5)


@pytest.fixture
def ssp_da():
    return _make_ssp_tasmin(n_years=5)


@pytest.fixture
def simple_config(tmp_path):
    return _make_config(tmp_path, models=("model-A", "model-B"))


def test_compute_returns_expected_keys(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    for key in ("models", "ref_clim", "fut_clim", "change",
                 "hist_series", "ssp_series", "model_mean_tmin",
                 "obs_mean_tmin", "lat", "lon"):
        assert key in results, f"Missing key: {key}"


def test_compute_models_list(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    assert set(results["models"]) == {"model-A", "model-B"}


def test_compute_ref_clim_shape(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    for m in ("model-A", "model-B"):
        rc = results["ref_clim"][m]
        assert set(rc.dims) == {"lat", "lon"}


def test_compute_change_equals_fut_minus_ref(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    for m in results["change"]:
        expected = results["fut_clim"][m] - results["ref_clim"][m]
        np.testing.assert_allclose(
            results["change"][m].values,
            expected.values,
            rtol=1e-5,
        )


def test_compute_change_positive_with_warmer_ssp(tmp_path, hist_da, ssp_da, simple_config):
    """SSP is +1 K warmer → change map should have positive values in tropics."""
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    for m in results["change"]:
        # At least some positive values (tropical cells)
        assert float(results["change"][m].max()) > 0


def test_compute_hist_series_has_year_dim(tmp_path, hist_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    for m in results["hist_series"]:
        assert "year" in results["hist_series"][m].dims


def test_compute_no_future_loader_skips_fut_clim(tmp_path, hist_da, simple_config):
    """Models without a future loader must not appear in fut_clim / change."""
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    assert len(results["fut_clim"]) == 0
    assert len(results["change"]) == 0
    # But should still have hist data
    assert len(results["models"]) == 2


def test_compute_skips_model_when_hist_fails(tmp_path, ssp_da, simple_config):
    """Model that fails hist load is absent from all result dicts."""
    diag = _make_diag(simple_config, _make_daily_tasmin(), ssp_da)
    diag._make_hist_loader = lambda model: (
        _FailLoader() if model == "model-A" else _MockLoader(_make_daily_tasmin())
    )
    results = diag.compute()
    assert "model-A" not in results["models"]
    assert "model-A" not in results["ref_clim"]


def test_compute_lat_lon_populated(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    assert results["lat"] is not None
    assert results["lon"] is not None


# ── Unit tests: _land_mean_series ─────────────────────────────────────────────


def test_land_mean_series_returns_1d(tmp_path, hist_da, simple_config):
    diag = _make_diag(simple_config, hist_da)
    loader = _MockLoader(hist_da)
    tn, _ = diag._load_and_save_tn(
        "model-A", loader, ("1981", "1985"),
        diag._nc_hist_path("model-A"),
    )
    series = diag._land_mean_series(tn, land_mask=None)
    assert "year" in series.dims
    assert series.ndim == 1


def test_land_mean_series_with_land_mask_all_zeros(tmp_path, hist_da, simple_config):
    """All-zero land mask → series all NaN/zero (no land area)."""
    diag = _make_diag(simple_config, hist_da)
    loader = _MockLoader(hist_da)
    tn, _ = diag._load_and_save_tn(
        "model-A", loader, ("1981", "1985"),
        diag._nc_hist_path("model-A"),
    )
    nlat = len(tn["lat"])
    nlon = len(tn["lon"])
    mask = xr.DataArray(
        np.zeros((nlat, nlon), dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": tn["lat"], "lon": tn["lon"]},
    )
    series = diag._land_mean_series(tn, land_mask=mask)
    # All-zero weight mean → should be 0.0 or NaN
    assert np.all(np.isfinite(series.values) | np.isnan(series.values))


# ── Unit tests: plot() ────────────────────────────────────────────────────────


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_plot_returns_empty_when_no_models(tmp_path, simple_config):
    diag = _make_diag(simple_config, _make_daily_tasmin())
    results = {
        "models": [],
        "ref_clim": {}, "fut_clim": {}, "change": {},
        "hist_series": {}, "ssp_series": {},
        "model_mean_tmin": {}, "obs_mean_tmin": None,
        "lat": None, "lon": None,
    }
    figs = diag.plot(results)
    assert len(figs) == 0
    plt.close("all")


def test_plot_produces_ref_map(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_ref_map" in ids
    plt.close("all")


def test_plot_produces_fut_and_change_when_available(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_fut_map" in ids
    assert "tropical_nights_change_delta_map" in ids
    plt.close("all")


def test_plot_no_fut_when_no_fut_data(tmp_path, hist_da, simple_config):
    """When fut_clim is empty, no fut/change figures are produced."""
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_fut_map" not in ids
    assert "tropical_nights_change_delta_map" not in ids
    plt.close("all")


def test_plot_timeseries_always_produced(tmp_path, hist_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_timeseries" in ids
    plt.close("all")


def test_plot_tmin_bias_included_when_obs_available(tmp_path, hist_da, simple_config):
    """Group E appears when obs_mean_tmin is present and model tmin was computed."""
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    # Inject synthetic obs tmin
    lats = results["lat"]
    lons = results["lon"]
    obs_k = xr.DataArray(
        np.full((len(lats), len(lons)), 290.0),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    results["obs_mean_tmin"] = obs_k
    # Also ensure model tmin is populated
    results["model_mean_tmin"] = {
        m: xr.DataArray(
            np.full((len(lats), len(lons)), 292.0),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        for m in results["models"]
    }
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_tmin_bias" in ids
    plt.close("all")


def test_plot_tmin_bias_skipped_when_no_obs(tmp_path, hist_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    results["obs_mean_tmin"] = None
    figs = diag.plot(results)
    ids = [meta["figure_id"] for _, meta in figs]
    assert "tropical_nights_change_tmin_bias" not in ids
    plt.close("all")


def test_plot_figure_count_with_fut_data(tmp_path, hist_da, ssp_da, simple_config):
    """With future data: ref + fut + change + timeseries = 4 figures (no BE obs)."""
    diag = _make_diag(simple_config, hist_da, ssp_da)
    results = diag.compute()
    results["obs_mean_tmin"] = None
    figs = diag.plot(results)
    assert len(figs) == 4
    plt.close("all")


def test_plot_figure_count_without_fut_data(tmp_path, hist_da, simple_config):
    """Without future data: ref + timeseries = 2 figures (no BE obs)."""
    diag = _make_diag(simple_config, hist_da, ssp_da=None)
    results = diag.compute()
    results["obs_mean_tmin"] = None
    figs = diag.plot(results)
    assert len(figs) == 2
    plt.close("all")


# ── Integration: full run() with NC checkpoints ───────────────────────────────


def test_run_creates_figures(tmp_path, hist_da, ssp_da, simple_config):
    diag = _make_diag(simple_config, hist_da, ssp_da)
    saved = diag.run()
    assert len(saved) > 0
    for png_path, json_path in saved:
        assert png_path.exists()
        assert json_path.exists()
    plt.close("all")


def test_run_idempotent(tmp_path, hist_da, ssp_da, simple_config):
    """Second run reloads NC checkpoints and produces the same figures."""
    diag1 = _make_diag(simple_config, hist_da, ssp_da)
    saved1 = diag1.run()
    plt.close("all")

    diag2 = _make_diag(simple_config, hist_da, ssp_da)
    saved2 = diag2.run()
    plt.close("all")

    assert len(saved1) == len(saved2)


# ── Constants ─────────────────────────────────────────────────────────────────


def test_hist_boundary_year():
    assert _HIST_BOUNDARY_YEAR == 2015


def test_threshold_is_20_celsius_in_kelvin():
    assert abs(_TN_THRESHOLD_K - 293.15) < 1e-9


# ── Config reading ────────────────────────────────────────────────────────────


def test_config_defaults_used_when_no_cc_section(tmp_path):
    """When no climate_change section, default periods are set."""
    config = FeatherConfig(
        model_catalogs={},
        models=["m"],
        model_configs={
            "m": ModelConfig(
                name="m", institution="X", experiment="hist-1950",
                variant="r1i1p1f1", grids={"sfc": "latlon"}, color="#ff0000",
            )
        },
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path),
        project={"name": "NOCLIMCHANGE"},
    )
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), config
    )
    assert len(diag.ref_period) == 2
    assert len(diag.fut_period) == 2
    assert int(diag.fut_period[0]) > int(diag.ref_period[1])


def test_config_cc_periods_respected(tmp_path, simple_config):
    diag = TropicalNightsChangeDiag(
        _MockLoader(_make_daily_tasmin()), _MockObsLoader(), simple_config
    )
    assert diag.ref_period == ("1981", "1985")
    assert diag.fut_period == ("2019", "2023")
    assert diag.hist_load_period == ("1981", "1985")
    assert diag.ssp_load_period == ("2019", "2023")


# ── Registration ──────────────────────────────────────────────────────────────


def test_registered():
    """Diagnostic must be discoverable via the registry."""
    from feather.diag.registry import get_diagnostic
    import feather.diag  # noqa: F401 — ensures @register fires

    cls = get_diagnostic("tropical_nights_change")
    assert cls is TropicalNightsChangeDiag
