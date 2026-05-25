"""Tests for the TropicalNightsDiag diagnostic."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.tropical_nights import TropicalNightsDiag, _TN_THRESHOLD_K


# ── Synthetic daily data ─────────────────────────────────────────────


def _make_daily_tasmin(n_years: int = 3, nlat: int = 9, nlon: int = 18) -> xr.DataArray:
    """Synthetic daily tasmin on a regular lat/lon grid.

    Values are set so that tropical latitudes (|lat| < 23.5°) have
    ~200 days/year above 20 °C (293.15 K) and polar regions have none.
    """
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    n_days = 365 * n_years

    time = xr.date_range("1990-01-01", periods=n_days, freq="1D", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    # Base temperature: 310 K in tropics, 255 K at poles
    base = 255 + 55 * np.cos(np.deg2rad(lat_grid))

    # Broadcast to (time, lat, lon): no seasonal variation for simplicity
    data = np.broadcast_to(base[np.newaxis], (n_days, nlat, nlon)).copy().astype(np.float32)

    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
        attrs={"units": "K", "long_name": "Daily Minimum Near-Surface Air Temperature"},
    )


class MockCMORLoader:
    """Minimal mock for CMORLoader that serves daily tasmin."""

    def __init__(self, da: xr.DataArray):
        self._da = da

    def load_var(self, model: str, variable: str, *, table=None, period=None, time_mean=False):
        da = self._da
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_coords(self, model: str, variable: str):
        return np.asarray(self._da["lon"]), np.asarray(self._da["lat"])


class MockObsLoader:
    """Minimal mock obs loader (no daily tasmin obs)."""

    def load(self, dataset, variable, period=None):
        raise FileNotFoundError("no daily tasmin obs")

    def load_for_model_var(self, model_var, period=None):
        raise FileNotFoundError("no daily tasmin obs")


@pytest.fixture
def daily_tasmin():
    return _make_daily_tasmin(n_years=3)


@pytest.fixture
def tn_config(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["model-A", "model-B"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def tn_diag(tn_config, daily_tasmin):
    loader = MockCMORLoader(daily_tasmin)
    obs = MockObsLoader()
    return TropicalNightsDiag(
        loader, obs, tn_config,
        period=("1990", "1992"),
    )


# ── Unit tests: threshold counting ───────────────────────────────────


def test_threshold_constant():
    """_TN_THRESHOLD_K should be exactly 293.15 (20 °C)."""
    assert _TN_THRESHOLD_K == pytest.approx(293.15)


def test_count_tn_days_all_above():
    """When all values exceed threshold, count equals n_days per year."""
    lats = np.array([-10.0, 0.0, 10.0])
    lons = np.array([0.0, 90.0])
    time = xr.date_range("2000-01-01", periods=365, freq="1D", calendar="standard")
    data = np.full((365, 3, 2), 300.0, dtype=np.float32)  # well above 293.15 K
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsDiag._count_tn_days(da)
    assert result.dims == ("year", "lat", "lon")
    assert int(result.sel(year=2000).values[0, 0]) == 365


def test_count_tn_days_none_above():
    """When no values exceed threshold, count is zero everywhere."""
    lats = np.array([-70.0])
    lons = np.array([0.0])
    time = xr.date_range("2000-01-01", periods=365, freq="1D", calendar="standard")
    data = np.full((365, 1, 1), 250.0, dtype=np.float32)  # well below threshold
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsDiag._count_tn_days(da)
    assert int(result.values.sum()) == 0


def test_count_tn_days_exact_threshold():
    """Days exactly at threshold (not strictly greater) are not counted."""
    lats = np.array([0.0])
    lons = np.array([0.0])
    time = xr.date_range("2000-01-01", periods=10, freq="1D", calendar="standard")
    data = np.full((10, 1, 1), _TN_THRESHOLD_K, dtype=np.float64)
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsDiag._count_tn_days(da)
    assert int(result.values.sum()) == 0


def test_count_tn_days_multi_year():
    """Counting over multiple years produces correct per-year sums."""
    lats = np.array([0.0])
    lons = np.array([0.0])
    # 2 years of daily data
    time = xr.date_range("2000-01-01", periods=730, freq="1D", calendar="standard")
    # First year: all above, second year: all below.
    # 2000 is a leap year (366 days), so indices 0..365 belong to year 2000.
    data = np.empty((730, 1, 1), dtype=np.float32)
    data[:366] = 300.0    # 2000: all 366 days above threshold
    data[366:] = 250.0    # 2001: none above
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsDiag._count_tn_days(da)
    assert int(result.sel(year=2000).values[0, 0]) == 366
    assert int(result.sel(year=2001).values[0, 0]) == 0


def test_count_tn_days_output_attrs():
    """Output DataArray has the expected name and units attribute."""
    lats = np.array([0.0])
    lons = np.array([0.0])
    time = xr.date_range("2000-01-01", periods=10, freq="1D", calendar="standard")
    data = np.full((10, 1, 1), 300.0, dtype=np.float32)
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lats, "lon": lons})
    result = TropicalNightsDiag._count_tn_days(da)
    assert result.name == "tn_count"
    assert result.attrs["units"] == "days/year"


def test_count_tn_days_tropical_gradient(daily_tasmin):
    """Tropical latitudes should have more TN days than polar ones."""
    result = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tropical_mean = float(result.sel(lat=0.0, method="nearest").mean())
    polar_mean = float(result.sel(lat=80.0, method="nearest").mean())
    assert tropical_mean > polar_mean


# ── NC checkpoint tests ───────────────────────────────────────────────


def test_nc_path_format(tn_diag):
    """NC path embeds model name and period."""
    path = tn_diag._nc_path("model-A")
    assert "model-A" in path.name
    assert "1990" in path.name
    assert "1992" in path.name
    assert path.suffix == ".nc"


def test_nc_path_sanitises_slashes(tn_diag):
    """Forward slashes in model name are replaced with underscores."""
    path = tn_diag._nc_path("org/model")
    assert "/" not in path.name


def test_save_nc_creates_file(tn_diag, daily_tasmin, tmp_path):
    """_save_nc() writes a readable NetCDF file with tn_count variable."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual)
    nc_path = tn_diag._nc_path("model-A")
    assert nc_path.exists()
    ds = xr.open_dataset(nc_path)
    assert "tn_count" in ds
    ds.close()


def test_save_nc_correct_dims(tn_diag, daily_tasmin):
    """Saved NetCDF has (year, lat, lon) dims."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert set(ds["tn_count"].dims) == {"year", "lat", "lon"}
    ds.close()


def test_save_nc_global_attrs(tn_diag, daily_tasmin):
    """Saved NetCDF carries CF Conventions and threshold metadata."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert ds.attrs.get("Conventions", "").startswith("CF")
    assert "threshold" in ds.attrs
    ds.close()


def test_load_or_compute_uses_nc_when_exists(tn_diag, daily_tasmin, tmp_path):
    """_load_or_compute() reads from NC if the checkpoint exists."""
    # Pre-compute and save
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual)

    # Swap out loader so that any actual load would fail
    class FailLoader:
        def load_var(self, *a, **kw):
            raise RuntimeError("should not have been called")

    tn_diag.model_loader = FailLoader()
    result = tn_diag._load_or_compute("model-A")
    assert "tn_count" in result.name


# ── Compute tests ─────────────────────────────────────────────────────


def test_compute_returns_expected_keys(tn_diag):
    """compute() returns the required result keys."""
    results = tn_diag.compute()
    for key in ("tn_clim", "tn_series", "tn_zonal", "models"):
        assert key in results


def test_compute_models_list(tn_diag):
    """compute() populates 'models' with successful models."""
    results = tn_diag.compute()
    assert set(results["models"]) == {"model-A", "model-B"}


def test_compute_clim_shape(tn_diag):
    """Climatology fields have (lat, lon) shape."""
    results = tn_diag.compute()
    for model in results["models"]:
        clim = results["tn_clim"][model]
        assert clim.dims == ("lat", "lon")


def test_compute_series_has_year_dim(tn_diag):
    """Time series DataArrays have a 'year' dimension."""
    results = tn_diag.compute()
    for model in results["models"]:
        series = results["tn_series"][model]
        assert "year" in series.dims


def test_compute_zonal_is_1d(tn_diag):
    """Zonal mean results are 1-D with lat coordinate."""
    results = tn_diag.compute()
    for model in results["models"]:
        zonal = results["tn_zonal"][model]
        assert zonal.ndim == 1


def test_compute_graceful_skip_on_missing_data(tn_config):
    """Models that raise KeyError/FileNotFoundError are silently skipped."""

    class AlwaysFailLoader:
        def load_var(self, *a, **kw):
            raise FileNotFoundError("no daily data")

    diag = TropicalNightsDiag(
        AlwaysFailLoader(), MockObsLoader(), tn_config,
        period=("1990", "1992"),
    )
    results = diag.compute()
    assert results["models"] == []
    assert results["tn_clim"] == {}


def test_compute_partial_skip(tn_config, daily_tasmin):
    """When only one model has data, the other is skipped gracefully."""

    class SelectiveLoader:
        def __init__(self, da):
            self._da = da

        def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
            if model == "model-A":
                return self._da
            raise FileNotFoundError("no data for model-B")

    diag = TropicalNightsDiag(
        SelectiveLoader(daily_tasmin), MockObsLoader(), tn_config,
        period=("1990", "1992"),
    )
    results = diag.compute()
    assert results["models"] == ["model-A"]


# ── Plot tests ────────────────────────────────────────────────────────


def test_plot_returns_three_figures(tn_diag):
    """plot() produces exactly 3 figures (A, C, D)."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    assert len(pairs) == 3


def test_plot_figure_ids_are_unique(tn_diag):
    """Each figure has a distinct figure_id."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = [meta["figure_id"] for _, meta in pairs]
    assert len(ids) == len(set(ids))


def test_plot_climatology_id(tn_diag):
    """Climatology figure has the expected figure_id."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = {meta["figure_id"] for _, meta in pairs}
    assert "tropical_nights_climatology" in ids


def test_plot_timeseries_id(tn_diag):
    """Time series figure has the expected figure_id."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = {meta["figure_id"] for _, meta in pairs}
    assert "tropical_nights_timeseries" in ids


def test_plot_zonal_mean_id(tn_diag):
    """Zonal mean figure has the expected figure_id."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = {meta["figure_id"] for _, meta in pairs}
    assert "tropical_nights_zonal_mean" in ids


def test_plot_no_models_returns_empty(tn_config):
    """plot() with empty models list returns no figures."""

    class AlwaysFailLoader:
        def load_var(self, *a, **kw):
            raise FileNotFoundError("no data")

    diag = TropicalNightsDiag(
        AlwaysFailLoader(), MockObsLoader(), tn_config,
        period=("1990", "1992"),
    )
    results = diag.compute()
    pairs = diag.plot(results)
    assert pairs == []


def test_plot_metadata_contains_period(tn_diag):
    """Figure metadata includes the configured period."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    for _, meta in pairs:
        assert "period" in meta


# ── run() integration tests ───────────────────────────────────────────


def test_run_creates_png_and_json(tn_diag):
    """run() produces PNG + JSON sidecar for each figure."""
    import matplotlib
    matplotlib.use("Agg")

    saved = tn_diag.run(skip_existing=False)
    assert len(saved) == 3
    for png_path, json_path in saved:
        assert png_path.exists()
        assert json_path.exists()


def test_run_creates_nc_checkpoints(tn_diag):
    """run() writes per-model NC files to the output directory."""
    import matplotlib
    matplotlib.use("Agg")

    tn_diag.run(skip_existing=False)
    for model in tn_diag.config.models:
        assert tn_diag._nc_path(model).exists()


def test_run_skip_existing(tn_diag):
    """run() skips figure generation when all figures already exist."""
    import json
    import matplotlib
    matplotlib.use("Agg")

    # First pass — generate everything
    saved_first = tn_diag.run(skip_existing=False)

    # Second pass — should skip (no new saves)
    saved_second = tn_diag.run(skip_existing=True)
    assert len(saved_second) == 0


def test_run_force_regenerates(tn_diag):
    """run(skip_existing=False) regenerates even when figures exist."""
    import matplotlib
    matplotlib.use("Agg")

    tn_diag.run(skip_existing=False)
    saved_second = tn_diag.run(skip_existing=False)
    assert len(saved_second) == 3


# ── Registration test ────────────────────────────────────────────────


def test_diagnostic_is_registered():
    """TropicalNightsDiag is accessible via the registry."""
    from feather.diag.registry import get_diagnostic
    diag_cls = get_diagnostic("tropical_nights")
    assert diag_cls is TropicalNightsDiag


def test_diagnostic_name():
    assert TropicalNightsDiag.name == "tropical_nights"


def test_diagnostic_group():
    assert TropicalNightsDiag.group == "extremes"


def test_diagnostic_variables():
    assert "tasmin" in TropicalNightsDiag.variables


# ── Variable registry test ───────────────────────────────────────────


def test_tasmin_in_variable_registry():
    """tasmin is present in VARIABLE_REGISTRY with correct table."""
    from feather.data.variables import get_var
    vinfo = get_var("tasmin")
    assert vinfo.cmip6_table == "day"
    assert vinfo.group == "extremes"
