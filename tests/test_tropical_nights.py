"""Tests for the TropicalNightsDiag diagnostic."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.tropical_nights import TropicalNightsDiag, _TN_THRESHOLD_K


# ── Synthetic daily data ─────────────────────────────────────────────


def _make_daily_tasmin(n_years: int = 3, nlat: int = 9, nlon: int = 18) -> xr.DataArray:
    """Synthetic daily tasmin on a regular lat/lon grid.

    Values set so tropical latitudes (|lat| < ~23°) are well above 20 °C
    and polar regions are well below, giving a clear tropical-nights signal.
    """
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    n_days = 365 * n_years

    time = xr.date_range("1990-01-01", periods=n_days, freq="1D", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    # Base temperature: 310 K in tropics, 255 K at poles
    base = 255 + 55 * np.cos(np.deg2rad(lat_grid))
    data = np.broadcast_to(base[np.newaxis], (n_days, nlat, nlon)).copy().astype(np.float32)

    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K", "long_name": "Daily Minimum Near-Surface Air Temperature"},
    )


def _make_be_tmin_nc(path, nlat=9, nlon=18):
    """Write a minimal mock Berkeley Earth Land TMIN NetCDF to *path*."""
    lats = np.linspace(-80, 80, nlat).astype(np.float32)
    lons = np.linspace(-170, 170, nlon).astype(np.float32)

    # Build decimal-year time axis: 36 months covering 1990-1992
    dec_years = np.array(
        [1990 + m / 12.0 for m in range(36)], dtype=np.float64,
    )
    # Land mask: 1 everywhere for simplicity
    land_mask = np.ones((nlat, nlon), dtype=np.float64)
    # Temperature anomaly: zero for all months
    temperature = np.zeros((36, nlat, nlon), dtype=np.float32)
    # Climatology: 25 °C in tropics, -20 °C at poles
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    clim_field = (-20 + 45 * np.cos(np.deg2rad(lat_grid))).astype(np.float32)
    climatology = np.stack([clim_field] * 12, axis=0)

    ds = xr.Dataset(
        {
            "land_mask": xr.DataArray(land_mask, dims=("latitude", "longitude")),
            "temperature": xr.DataArray(
                temperature, dims=("time", "latitude", "longitude"),
                coords={"time": dec_years, "latitude": lats, "longitude": lons},
                attrs={"units": "degree C"},
            ),
            "climatology": xr.DataArray(
                climatology, dims=("month_number", "latitude", "longitude"),
                attrs={"units": "degree C"},
            ),
            "areal_weight": xr.DataArray(
                np.ones((nlat, nlon)), dims=("latitude", "longitude"),
            ),
        },
        coords={"latitude": lats, "longitude": lons},
    )
    ds["time"].attrs = {"units": "year A.D."}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class MockCMORLoader:
    """Minimal mock loader that serves daily tasmin."""

    def __init__(self, da: xr.DataArray):
        self._da = da

    def load_var(self, model: str, variable: str, *,
                 table=None, period=None, time_mean=False):
        if variable == "sftlf":
            raise FileNotFoundError("mock model has no sftlf land mask")
        da = self._da
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_coords(self, model: str, variable: str):
        return np.asarray(self._da["lon"]), np.asarray(self._da["lat"])


class MockObsLoader:
    """Minimal mock obs loader."""

    def load(self, dataset, variable, period=None):
        raise FileNotFoundError("no obs")

    def load_for_model_var(self, model_var, period=None):
        raise FileNotFoundError("no obs")


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
    return TropicalNightsDiag(loader, obs, tn_config, period=("1990", "1992"))


# ── Unit tests: threshold counting ───────────────────────────────────


def test_threshold_constant():
    assert _TN_THRESHOLD_K == pytest.approx(293.15)


def test_count_tn_days_all_above():
    """When all values exceed threshold, count equals n_days per year."""
    lats = np.array([-10.0, 0.0, 10.0])
    lons = np.array([0.0, 90.0])
    time = xr.date_range("2000-01-01", periods=365, freq="1D", calendar="standard")
    data = np.full((365, 3, 2), 300.0, dtype=np.float32)
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
    data = np.full((365, 1, 1), 250.0, dtype=np.float32)
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
    # 730 days starting 2000-01-01; year 2000 is a leap year (366 days)
    time = xr.date_range("2000-01-01", periods=730, freq="1D", calendar="standard")
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


# ── NC path and directory tests ───────────────────────────────────────


def test_nc_dir_is_outside_figures(tn_diag):
    """NC output directory should NOT be inside the figures tree."""
    nc_dir = tn_diag.nc_dir
    figures_dir = tn_diag.output_dir
    # nc_dir should not be a subdirectory of output_dir (figures)
    assert not str(nc_dir).startswith(str(figures_dir))


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


def test_nc_dir_parent_is_output_dir(tn_diag):
    """NC directory is a direct child of the configured output_dir."""
    from pathlib import Path
    assert tn_diag.nc_dir.parent == Path(tn_diag.config.output_dir)


# ── NC save / load tests ──────────────────────────────────────────────


def test_save_nc_creates_file(tn_diag, daily_tasmin):
    """_save_nc() writes a readable NetCDF with tn_count variable."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)
    assert tn_diag._nc_path("model-A").exists()


def test_save_nc_contains_both_variables(tn_diag, daily_tasmin):
    """Saved NC has both tn_count and tmin_mean variables."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert "tn_count" in ds
    assert "tmin_mean" in ds
    ds.close()


def test_save_nc_tn_count_dims(tn_diag, daily_tasmin):
    """tn_count variable has (year, lat, lon) dims."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert set(ds["tn_count"].dims) == {"year", "lat", "lon"}
    ds.close()


def test_save_nc_tmin_mean_dims(tn_diag, daily_tasmin):
    """tmin_mean variable has (lat, lon) dims."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert set(ds["tmin_mean"].dims) == {"lat", "lon"}
    ds.close()


def test_save_nc_global_attrs(tn_diag, daily_tasmin):
    """Saved NC carries CF Conventions and threshold metadata."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)
    ds = xr.open_dataset(tn_diag._nc_path("model-A"))
    assert ds.attrs.get("Conventions", "").startswith("CF")
    assert "threshold" in ds.attrs
    assert "land_only" in ds.attrs
    ds.close()


def test_load_or_compute_uses_nc_when_exists(tn_diag, daily_tasmin):
    """_load_or_compute() reads from NC if checkpoint exists."""
    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    mean_tmin = daily_tasmin.mean("time")
    tn_diag.nc_dir.mkdir(parents=True, exist_ok=True)
    tn_diag._save_nc("model-A", tn_annual, mean_tmin)

    class FailLoader:
        def load_var(self, *a, **kw):
            raise RuntimeError("should not have been called")

    tn_diag.model_loader = FailLoader()
    tn_out, tmin_out = tn_diag._load_or_compute("model-A")
    assert tn_out.name == "tn_count"


def test_load_or_compute_returns_tuple(tn_diag):
    """_load_or_compute() returns a (tn_annual, mean_tmin) tuple."""
    result = tn_diag._load_or_compute("model-A")
    assert isinstance(result, tuple) and len(result) == 2


# ── Land mask tests ───────────────────────────────────────────────────


def test_land_mask_returns_none_when_path_overridden(tn_diag, daily_tasmin, monkeypatch):
    """_load_land_mask() returns None when _be_tmin_path() returns None."""
    monkeypatch.setattr(tn_diag, "_be_tmin_path", lambda: None)
    mask = tn_diag._load_land_mask(
        "model-A",
        np.asarray(daily_tasmin["lat"]),
        np.asarray(daily_tasmin["lon"]),
    )
    assert mask is None


def test_land_mask_applied_from_mock_file(tn_config, daily_tasmin, tmp_path):
    """Land mask from a mock BE file is applied: ocean pixels become NaN."""
    be_path = tmp_path / "be_tmin.nc"
    _make_be_tmin_nc(be_path, nlat=daily_tasmin.sizes["lat"],
                     nlon=daily_tasmin.sizes["lon"])

    # Config with the mock path
    obs_datasets = {"BERKELEY_EARTH_TMIN": {"path": str(be_path.parent),
                                             "variables": {"temperature": be_path.name}}}
    cfg = FeatherConfig(
        model_catalogs={},
        models=["model-A"],
        obs_root="",
        obs_datasets=obs_datasets,
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )
    diag = TropicalNightsDiag(
        MockCMORLoader(daily_tasmin), MockObsLoader(), cfg,
        period=("1990", "1992"),
    )
    # Override _be_tmin_path to return the mock file directly
    diag._be_tmin_path = lambda: be_path

    mask = diag._load_land_mask(
        "model-A",
        np.asarray(daily_tasmin["lat"]),
        np.asarray(daily_tasmin["lon"]),
    )
    assert mask is not None
    assert mask.dtype == bool or mask.dtype == np.bool_


def test_land_mask_ocean_pixels_nan(tn_config, daily_tasmin, tmp_path):
    """After applying land mask, ocean pixels in TN count should be NaN."""
    be_path = tmp_path / "be_tmin.nc"
    _make_be_tmin_nc(be_path, nlat=daily_tasmin.sizes["lat"],
                     nlon=daily_tasmin.sizes["lon"])

    # Manually apply a mask where only the first lat band is land
    lats = np.asarray(daily_tasmin["lat"])
    lons = np.asarray(daily_tasmin["lon"])

    # Build a land mask: land for lat < 0 only
    mask_vals = (lats[:, np.newaxis] < 0).astype(float) * np.ones((1, len(lons)))
    land_mask = xr.DataArray(
        mask_vals > 0.5, dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )

    tn_annual = TropicalNightsDiag._count_tn_days(daily_tasmin)
    tn_masked = tn_annual.where(land_mask)
    # Ocean lats (lat >= 0) should be NaN in mean
    ocean_lat = float(lats[lats >= 0][0])
    assert np.all(np.isnan(tn_masked.sel(lat=ocean_lat, method="nearest").values))


def test_model_sftlf_land_mask_preferred(tn_config, daily_tasmin, monkeypatch):
    """When the model provides sftlf (fx, %), it is used as the land mask."""
    lats = np.asarray(daily_tasmin["lat"])
    lons = np.asarray(daily_tasmin["lon"])
    # sftlf: land (80 %) for lat < 0, ocean (10 %) elsewhere
    sftlf_vals = np.where(lats[:, None] < 0, 80.0, 10.0) * np.ones((1, len(lons)))
    sftlf = xr.DataArray(sftlf_vals, dims=("lat", "lon"),
                         coords={"lat": lats, "lon": lons}, name="sftlf")

    class SftlfLoader(MockCMORLoader):
        def load_var(self, model, variable, *, table=None, period=None,
                     time_mean=False):
            if variable == "sftlf":
                return sftlf
            return super().load_var(model, variable, table=table,
                                    period=period, time_mean=time_mean)

    diag = TropicalNightsDiag(
        SftlfLoader(daily_tasmin), MockObsLoader(), tn_config,
        period=("1990", "1992"))
    # BE path would raise if reached; ensure we don't fall back
    monkeypatch.setattr(diag, "_be_tmin_path", lambda: None)

    mask = diag._load_land_mask("model-A", lats, lons)
    assert mask is not None
    assert bool(mask.sel(lat=float(lats[lats < 0][0]), method="nearest").all())
    assert not bool(mask.sel(lat=float(lats[lats >= 0][0]), method="nearest").any())


# ── Compute tests ─────────────────────────────────────────────────────


def test_compute_returns_expected_keys(tn_diag):
    """compute() returns all required keys including model_mean_tmin."""
    results = tn_diag.compute()
    for key in ("tn_clim", "tn_series", "tn_zonal", "model_mean_tmin", "models"):
        assert key in results


def test_compute_models_list(tn_diag):
    """compute() populates 'models' with successful models."""
    results = tn_diag.compute()
    assert set(results["models"]) == {"model-A", "model-B"}


def test_compute_clim_shape(tn_diag):
    """Climatology fields have (lat, lon) shape."""
    results = tn_diag.compute()
    for model in results["models"]:
        assert results["tn_clim"][model].dims == ("lat", "lon")


def test_compute_model_mean_tmin_shape(tn_diag):
    """model_mean_tmin fields have (lat, lon) shape."""
    results = tn_diag.compute()
    for model in results["models"]:
        assert results["model_mean_tmin"][model].dims == ("lat", "lon")


def test_compute_series_has_year_dim(tn_diag):
    """Time series DataArrays have a 'year' dimension."""
    results = tn_diag.compute()
    for model in results["models"]:
        assert "year" in results["tn_series"][model].dims


def test_compute_zonal_is_1d(tn_diag):
    """Zonal mean results are 1-D."""
    results = tn_diag.compute()
    for model in results["models"]:
        assert results["tn_zonal"][model].ndim == 1


def test_compute_obs_mean_tmin_none_when_path_overridden(tn_diag, monkeypatch):
    """obs_mean_tmin is None when _be_tmin_path() returns None."""
    monkeypatch.setattr(tn_diag, "_be_tmin_path", lambda: None)
    results = tn_diag.compute()
    assert results["obs_mean_tmin"] is None


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


def test_plot_three_figures_without_obs(tn_diag, monkeypatch):
    """Without BE obs, plot() produces exactly 3 figures (A, C, D)."""
    monkeypatch.setattr(tn_diag, "_be_tmin_path", lambda: None)
    results = tn_diag.compute()
    assert results["obs_mean_tmin"] is None
    pairs = tn_diag.plot(results)
    assert len(pairs) == 3


def test_plot_four_figures_with_obs(tn_diag, daily_tasmin):
    """With mock BE obs, plot() produces 4 figures (A, B, C, D)."""
    results = tn_diag.compute()
    # Inject a mock obs DataArray
    lats = np.asarray(daily_tasmin["lat"])
    lons = np.asarray(daily_tasmin["lon"])
    mock_obs = xr.DataArray(
        np.full((len(lats), len(lons)), 295.0),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    results["obs_mean_tmin"] = mock_obs
    pairs = tn_diag.plot(results)
    assert len(pairs) == 4


def test_plot_figure_ids_are_unique(tn_diag):
    """Each figure has a distinct figure_id."""
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = [meta["figure_id"] for _, meta in pairs]
    assert len(ids) == len(set(ids))


def test_plot_expected_ids_without_obs(tn_diag, monkeypatch):
    """Figure IDs without obs: climatology, timeseries, zonal_mean, no bias."""
    monkeypatch.setattr(tn_diag, "_be_tmin_path", lambda: None)
    results = tn_diag.compute()
    pairs = tn_diag.plot(results)
    ids = {meta["figure_id"] for _, meta in pairs}
    assert "tropical_nights_climatology" in ids
    assert "tropical_nights_timeseries" in ids
    assert "tropical_nights_zonal_mean" in ids
    assert "tropical_nights_tmin_bias" not in ids


def test_plot_bias_id_present_with_obs(tn_diag, daily_tasmin):
    """With obs injected, tmin_bias figure_id is present."""
    results = tn_diag.compute()
    lats = np.asarray(daily_tasmin["lat"])
    lons = np.asarray(daily_tasmin["lon"])
    results["obs_mean_tmin"] = xr.DataArray(
        np.full((len(lats), len(lons)), 295.0),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    pairs = tn_diag.plot(results)
    ids = {meta["figure_id"] for _, meta in pairs}
    assert "tropical_nights_tmin_bias" in ids


def test_plot_no_models_returns_empty(tn_config):
    """plot() with empty models list returns no figures."""

    class AlwaysFailLoader:
        def load_var(self, *a, **kw):
            raise FileNotFoundError("no data")

    diag = TropicalNightsDiag(
        AlwaysFailLoader(), MockObsLoader(), tn_config, period=("1990", "1992"),
    )
    results = diag.compute()
    assert diag.plot(results) == []


def test_plot_metadata_contains_period(tn_diag):
    """Figure metadata includes the configured period."""
    results = tn_diag.compute()
    for _, meta in tn_diag.plot(results):
        assert "period" in meta


# ── run() integration tests ───────────────────────────────────────────


def test_run_creates_png_and_json(tn_diag):
    """run() produces PNG + JSON sidecar for each figure (3 without obs, 4 with)."""
    import matplotlib
    matplotlib.use("Agg")
    saved = tn_diag.run(skip_existing=False)
    assert len(saved) >= 3
    for png_path, json_path in saved:
        assert png_path.exists()
        assert json_path.exists()


def test_run_nc_in_correct_directory(tn_diag):
    """run() places NC files in {output_dir}/tropical_nights/, not figures."""
    import matplotlib
    matplotlib.use("Agg")
    tn_diag.run(skip_existing=False)
    for model in tn_diag.config.models:
        nc = tn_diag._nc_path(model)
        assert nc.exists()
        assert nc.parent == tn_diag.nc_dir


def test_run_skip_existing(tn_diag):
    """run() skips figure generation when all figures already exist."""
    import matplotlib
    matplotlib.use("Agg")
    tn_diag.run(skip_existing=False)
    saved_second = tn_diag.run(skip_existing=True)
    assert len(saved_second) == 0


def test_run_force_regenerates(tn_diag):
    """run(skip_existing=False) regenerates even when figures exist."""
    import matplotlib
    matplotlib.use("Agg")
    saved_first = tn_diag.run(skip_existing=False)
    saved_second = tn_diag.run(skip_existing=False)
    assert len(saved_second) == len(saved_first)


# ── Registration and registry tests ─────────────────────────────────


def test_diagnostic_is_registered():
    from feather.diag.registry import get_diagnostic
    assert get_diagnostic("tropical_nights") is TropicalNightsDiag


def test_diagnostic_name():
    assert TropicalNightsDiag.name == "tropical_nights"


def test_diagnostic_group():
    assert TropicalNightsDiag.group == "extremes"


def test_diagnostic_variables():
    assert "tasmin" in TropicalNightsDiag.variables


def test_tasmin_in_variable_registry():
    from feather.data.variables import get_var
    vinfo = get_var("tasmin")
    assert vinfo.cmip6_table == "day"
    assert vinfo.group == "extremes"
