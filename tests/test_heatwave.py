"""Tests for HeatwaveDiag — TX90 heatwave indices."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig


# ── Synthetic data builders ──────────────────────────────────────────────────

def _make_config(tmp_path, models=("ModelA", "ModelB")):
    return FeatherConfig(
        model_catalogs={},
        models=list(models),
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor", "root": str(tmp_path)},
        model_configs={
            m: ModelConfig(name=m, grids={"sfc": "latlon"}, color="#1f77b4")
            for m in models
        },
    )


def _make_daily_tasmax(n_years=3, nlat=4, nlon=6, base_k=300.0):
    """Synthetic daily tasmax on a small lat/lon grid.

    lat: -45, -15, 15, 45 (two SH, two NH)
    lon: 0, 60, 120, 180, 240, 300
    Values are base_k for all days (below 90th pct by design unless overridden).
    """
    lat = np.array([-45.0, -15.0, 15.0, 45.0], dtype=np.float32)
    lon = np.linspace(0.0, 300.0, nlon, dtype=np.float32)
    n_days = n_years * 365
    time = xr.cftime_range("1980-01-01", periods=n_days, freq="D", calendar="noleap")
    data = np.full((n_days, nlat, nlon), base_k, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _make_be_tmax_nc(path, nlat=4, nlon=6):
    """Write a minimal BE Land TMAX NetCDF fixture."""
    lat = np.array([-45.0, -15.0, 15.0, 45.0], dtype=np.float32)
    lon = np.linspace(-180.0, 120.0, nlon, dtype=np.float32)
    n_months = 36
    # Decimal-year time axis starting ~1980
    dec_times = np.array([1980.0 + i / 12 for i in range(n_months)])
    land_mask = np.ones((nlat, nlon), dtype=np.float64)
    anom = np.zeros((n_months, nlat, nlon), dtype=np.float32)
    clim = np.full((12, nlat, nlon), 25.0, dtype=np.float32)  # 25°C climatology

    ds = xr.Dataset(
        {
            "land_mask": (["latitude", "longitude"], land_mask),
            "temperature": (["time", "latitude", "longitude"], anom),
            "climatology": (["month_number", "latitude", "longitude"], clim),
            "areal_weight": (["latitude", "longitude"], np.ones((nlat, nlon))),
        },
        coords={
            "time": dec_times,
            "latitude": lat,
            "longitude": lon,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


class MockCMORLoader:
    """Mock loader that returns synthetic daily tasmax."""

    def __init__(self, tasmax_da):
        self._da = tasmax_da

    def load_var(self, model, variable, *, table=None, period=None, time_mean=False):
        if variable == "sftlf":
            raise FileNotFoundError("mock model has no sftlf land mask")
        da = self._da
        if period:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_coords(self, model, variable):
        return np.asarray(self._da["lon"]), np.asarray(self._da["lat"])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def hw_diag(tmp_path):
    """HeatwaveDiag with synthetic loader, no obs."""
    from feather.diag.heatwave import HeatwaveDiag

    da = _make_daily_tasmax(n_years=5)
    config = _make_config(tmp_path, models=("ModelA",))
    loader = MockCMORLoader(da)
    diag = HeatwaveDiag(loader, None, config, period=("1980", "1984"))
    return diag


@pytest.fixture
def hw_diag_with_obs(tmp_path):
    """HeatwaveDiag with synthetic loader + BE TMAX obs file."""
    from feather.diag.heatwave import HeatwaveDiag

    da = _make_daily_tasmax(n_years=5)
    config = _make_config(tmp_path, models=("ModelA",))
    loader = MockCMORLoader(da)
    diag = HeatwaveDiag(loader, None, config, period=("1980", "1984"))

    obs_path = tmp_path / "be_tmax.nc"
    _make_be_tmax_nc(obs_path)
    diag._be_tmax_path = lambda: obs_path
    return diag


# ── Unit tests: _compute_hw_indices_numpy ────────────────────────────────────

class TestComputeHwIndicesNumpy:
    from feather.diag.heatwave import HeatwaveDiag as _D

    def _run(self, hot, tasmax=None):
        from feather.diag.heatwave import HeatwaveDiag
        if tasmax is None:
            tasmax = hot.astype(np.float32) * 305.0 + 300.0
        return HeatwaveDiag._compute_hw_indices_numpy(tasmax, hot)

    def test_no_hot_days_gives_zero_counts(self):
        hot = np.zeros((30, 2, 3), dtype=bool)
        tasmax = np.full_like(hot, 300.0, dtype=np.float32)
        r = self._run(hot, tasmax)
        assert np.all(r["hwn"] == 0)
        assert np.all(r["hwf"] == 0)
        assert np.all(r["hwd"] == 0)

    def test_no_hot_days_gives_nan_temperatures(self):
        hot = np.zeros((30, 2, 3), dtype=bool)
        tasmax = np.full_like(hot, 300.0, dtype=np.float32)
        r = self._run(hot, tasmax)
        assert np.all(np.isnan(r["hwm"]))
        assert np.all(np.isnan(r["hwa"]))

    def test_two_consecutive_hot_days_not_a_heatwave(self):
        hot = np.zeros((30, 1, 1), dtype=bool)
        hot[5:7, 0, 0] = True  # 2-day run
        tasmax = np.where(hot, 310.0, 300.0).astype(np.float32)
        r = self._run(hot, tasmax)
        assert r["hwn"][0, 0] == 0
        assert r["hwf"][0, 0] == 0

    def test_three_consecutive_hot_days_is_one_heatwave(self):
        hot = np.zeros((30, 1, 1), dtype=bool)
        hot[5:8, 0, 0] = True  # 3-day run
        tasmax = np.where(hot, 310.0, 300.0).astype(np.float32)
        r = self._run(hot, tasmax)
        assert r["hwn"][0, 0] == pytest.approx(1.0)
        assert r["hwf"][0, 0] == pytest.approx(3.0)
        assert r["hwd"][0, 0] == pytest.approx(3.0)

    def test_five_day_run_counts_as_one_event_five_days(self):
        hot = np.zeros((30, 1, 1), dtype=bool)
        hot[10:15, 0, 0] = True  # 5-day run
        tasmax = np.where(hot, 310.0, 300.0).astype(np.float32)
        r = self._run(hot, tasmax)
        assert r["hwn"][0, 0] == pytest.approx(1.0)
        assert r["hwf"][0, 0] == pytest.approx(5.0)
        assert r["hwd"][0, 0] == pytest.approx(5.0)

    def test_two_separate_heatwaves(self):
        hot = np.zeros((40, 1, 1), dtype=bool)
        hot[3:6, 0, 0] = True   # 3-day run
        hot[20:24, 0, 0] = True  # 4-day run
        tasmax = np.where(hot, 310.0, 300.0).astype(np.float32)
        r = self._run(hot, tasmax)
        assert r["hwn"][0, 0] == pytest.approx(2.0)
        assert r["hwf"][0, 0] == pytest.approx(7.0)
        assert r["hwd"][0, 0] == pytest.approx(4.0)

    def test_hwm_is_mean_of_heatwave_days(self):
        hot = np.zeros((10, 1, 1), dtype=bool)
        hot[2:5, 0, 0] = True  # days 2,3,4
        tasmax = np.zeros((10, 1, 1), dtype=np.float32)
        tasmax[2, 0, 0] = 305.0
        tasmax[3, 0, 0] = 307.0
        tasmax[4, 0, 0] = 309.0
        r = self._run(hot, tasmax)
        expected_hwm = (305.0 + 307.0 + 309.0) / 3.0
        assert r["hwm"][0, 0] == pytest.approx(expected_hwm, abs=0.01)

    def test_hwa_is_peak_of_heatwave_days(self):
        hot = np.zeros((10, 1, 1), dtype=bool)
        hot[2:5, 0, 0] = True
        tasmax = np.zeros((10, 1, 1), dtype=np.float32)
        tasmax[2, 0, 0] = 305.0
        tasmax[3, 0, 0] = 312.0  # peak
        tasmax[4, 0, 0] = 308.0
        r = self._run(hot, tasmax)
        assert r["hwa"][0, 0] == pytest.approx(312.0, abs=0.01)

    def test_empty_array_returns_zeros_and_nans(self):
        from feather.diag.heatwave import HeatwaveDiag
        hot = np.zeros((0, 2, 3), dtype=bool)
        tasmax = np.zeros((0, 2, 3), dtype=np.float32)
        r = HeatwaveDiag._compute_hw_indices_numpy(tasmax, hot)
        assert r["hwn"].shape == (2, 3)
        assert np.all(r["hwn"] == 0)
        assert np.all(np.isnan(r["hwm"]))

    def test_vectorised_across_spatial_dims(self):
        """Indices computed correctly for all cells simultaneously."""
        hot = np.zeros((20, 2, 2), dtype=bool)
        # cell (0,0): 3-day run → 1 event
        hot[5:8, 0, 0] = True
        # cell (0,1): no heatwave
        # cell (1,0): 4-day run → 1 event
        hot[10:14, 1, 0] = True
        # cell (1,1): two runs of 3 → 2 events
        hot[2:5, 1, 1] = True
        hot[15:18, 1, 1] = True
        tasmax = np.where(hot, 310.0, 300.0).astype(np.float32)
        r = self._run(hot, tasmax)
        assert r["hwn"][0, 0] == pytest.approx(1.0)
        assert r["hwn"][0, 1] == pytest.approx(0.0)
        assert r["hwn"][1, 0] == pytest.approx(1.0)
        assert r["hwn"][1, 1] == pytest.approx(2.0)


# ── Unit tests: _compute_threshold ───────────────────────────────────────────

class TestComputeThreshold:
    def test_threshold_has_dayofyear_dim(self, hw_diag):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        assert "dayofyear" in thresh.dims

    def test_threshold_has_lat_lon_dims(self, hw_diag):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        assert "lat" in thresh.dims
        assert "lon" in thresh.dims

    def test_threshold_doy_366_always_present(self, hw_diag):
        da = _make_daily_tasmax(n_years=3)  # noleap calendar — no DOY 366
        thresh = hw_diag._compute_threshold(da)
        assert 366 in thresh["dayofyear"].values

    def test_threshold_values_are_quantile_90(self, hw_diag):
        """When all values are constant, 90th pct = that constant."""
        da = _make_daily_tasmax(n_years=3, base_k=305.0)
        thresh = hw_diag._compute_threshold(da)
        assert float(thresh.isel(dayofyear=0, lat=0, lon=0)) == pytest.approx(305.0, abs=0.1)

    def test_threshold_above_hot_days_gives_no_heatwave(self, hw_diag):
        """If threshold > all data, no hot days → no heatwaves."""
        da = _make_daily_tasmax(n_years=3, base_k=295.0)
        # Push threshold very high
        thresh = hw_diag._compute_threshold(da) + 20.0
        # Verify: all tasmax < thresh → no hot days
        doy_vals = da.time.dt.dayofyear.values
        thresh_vals = thresh.sel(dayofyear=xr.DataArray(doy_vals, dims="time")).values
        assert np.all(da.values < thresh_vals)


# ── Unit tests: summer mask / hemisphere logic ────────────────────────────────

class TestHemisphereSplit:
    def _get_indices(self, tasmax_da, period, hot_window):
        """Return indices for a tasmax where hot days only in hot_window (month list)."""
        from feather.diag.heatwave import HeatwaveDiag

        config_obj = type("C", (), {
            "models": [], "output_dir": "/tmp",
            "obs_datasets": {},
            "get_model_color": lambda self, m: "#000",
        })()

        # Set base DA values below threshold, bump up during hot_window months
        base_k = 300.0
        thresh_val = 302.0
        da = tasmax_da.copy()
        hot_months = da.time.dt.month.isin(hot_window)
        data = da.values.copy()
        data[hot_months.values] = thresh_val + 5.0
        da.values[:] = data

        # Compute threshold (constant → base_k everywhere except hot months)
        # Build threshold manually: < thresh_val everywhere
        thresh_da = xr.DataArray(
            np.full((366,) + da.shape[1:], thresh_val, dtype=np.float32),
            dims=["dayofyear", "lat", "lon"],
            coords={
                "dayofyear": np.arange(1, 367),
                "lat": da["lat"].values,
                "lon": da["lon"].values,
            },
        )
        diag = HeatwaveDiag.__new__(HeatwaveDiag)
        diag.period = period
        indices_ds, _ = diag._compute_indices(da, thresh_da)
        return indices_ds

    def test_nh_summer_only_in_may_sep(self):
        """Hot days only in May-Sep → NH cells (lat > 0) should have heatwaves."""
        da = _make_daily_tasmax(n_years=3)
        # Make May-Sep days very hot at NH cells
        hot = da.time.dt.month.isin([5, 6, 7, 8, 9]).values
        data = da.values.copy()
        # Force a 5-day run in every May-Sep for all cells
        may_idx = np.where(
            (da.time.dt.month == 6).values
        )[0]
        if len(may_idx) >= 5:
            data[may_idx[:5]] = 320.0  # very hot — well above any threshold

        da2 = da.copy(data=data)
        from feather.diag.heatwave import HeatwaveDiag
        thresh = HeatwaveDiag._compute_threshold(da2)
        # Set threshold above base but below 320
        thresh_high = thresh + 0  # use actual 90th pct; for constant data it equals base
        # Most of these days are at base_k = 300, so 90th pct ≈ 300
        # The 5 days at 320 will exceed it
        diag = HeatwaveDiag.__new__(HeatwaveDiag)
        diag.period = ("1980", "1982")
        indices_ds, _ = diag._compute_indices(da2, thresh_high)
        nh_idx = indices_ds["hwf"].values[:, 2:, :]  # lat >= 15 (NH)
        # NH should have some heatwave frequency (June days triggered)
        assert nh_idx.sum() > 0

    def test_nh_cells_use_may_sep_window(self):
        """NH cells should be zero when only Dec-Feb is hot (SH summer)."""
        da = _make_daily_tasmax(n_years=3)
        data = da.values.copy()
        # Make only December days hot
        dec_idx = np.where(da.time.dt.month.values == 12)[0]
        if len(dec_idx) >= 5:
            data[dec_idx[:5]] = 320.0
        da2 = da.copy(data=data)
        from feather.diag.heatwave import HeatwaveDiag
        thresh = HeatwaveDiag._compute_threshold(da2)
        diag = HeatwaveDiag.__new__(HeatwaveDiag)
        diag.period = ("1980", "1982")
        indices_ds, _ = diag._compute_indices(da2, thresh)
        # NH cells (lat > 0, indices 2 and 3): Dec is not NH summer → hwf should be 0
        nh_hwf = indices_ds["hwf"].values[:, 2:, :]
        assert nh_hwf.sum() == pytest.approx(0.0)


# ── Unit tests: NC checkpoint ─────────────────────────────────────────────────

class TestNcCheckpoint:
    def test_nc_path_uses_model_name(self, hw_diag, tmp_path):
        path = hw_diag._nc_path("ModelA")
        assert "ModelA" in path.name

    def test_nc_path_uses_period(self, hw_diag, tmp_path):
        path = hw_diag._nc_path("ModelA")
        assert "1980" in path.name and "1984" in path.name

    def test_nc_dir_is_outside_figures_dir(self, hw_diag, tmp_path):
        nc_dir = hw_diag.nc_dir
        figures_dir = hw_diag.output_dir
        assert not str(nc_dir).startswith(str(figures_dir))

    def test_save_nc_creates_file(self, hw_diag, tmp_path):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        indices_ds, mean_tmax = hw_diag._compute_indices(da, thresh)
        hw_diag._save_nc("ModelA", indices_ds, mean_tmax)
        assert hw_diag._nc_path("ModelA").exists()

    def test_save_nc_contains_all_variables(self, hw_diag, tmp_path):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        indices_ds, mean_tmax = hw_diag._compute_indices(da, thresh)
        hw_diag._save_nc("ModelA", indices_ds, mean_tmax)
        ds = xr.open_dataset(hw_diag._nc_path("ModelA"))
        for var in ("hwn", "hwf", "hwd", "hwm", "hwa", "tmax_mean"):
            assert var in ds.data_vars, f"Missing: {var}"

    def test_nc_has_method_attribute(self, hw_diag, tmp_path):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        indices_ds, mean_tmax = hw_diag._compute_indices(da, thresh)
        hw_diag._save_nc("ModelA", indices_ds, mean_tmax)
        ds = xr.open_dataset(hw_diag._nc_path("ModelA"))
        assert ds.attrs.get("method") == "tx90"

    def test_nc_indices_have_year_dim(self, hw_diag, tmp_path):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        indices_ds, mean_tmax = hw_diag._compute_indices(da, thresh)
        hw_diag._save_nc("ModelA", indices_ds, mean_tmax)
        ds = xr.open_dataset(hw_diag._nc_path("ModelA"))
        assert "year" in ds["hwn"].dims

    def test_load_from_nc_checkpoint(self, hw_diag, tmp_path):
        da = _make_daily_tasmax(n_years=3)
        thresh = hw_diag._compute_threshold(da)
        indices_ds, mean_tmax = hw_diag._compute_indices(da, thresh)
        hw_diag._save_nc("ModelA", indices_ds, mean_tmax)
        # Load via _load_or_compute — should load from NC, not recompute
        returned_ds, returned_tmax = hw_diag._load_or_compute("ModelA")
        assert "hwn" in returned_ds


# ── Unit tests: BE TMAX obs ───────────────────────────────────────────────────

class TestBeObsLoading:
    def test_be_tmax_path_returns_none_without_obs(self, hw_diag, monkeypatch):
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        assert hw_diag._be_tmax_path() is None

    def test_land_mask_returns_none_without_obs(self, hw_diag, monkeypatch):
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        mask = hw_diag._load_land_mask("model-A", np.array([-45.0, 15.0]), np.array([0.0, 90.0]))
        assert mask is None

    def test_be_mean_tmax_returns_none_without_obs(self, hw_diag, monkeypatch):
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        result = hw_diag._load_be_mean_tmax(np.array([-45.0, 15.0]), np.array([0.0, 90.0]))
        assert result is None

    def test_land_mask_loaded_from_be_file(self, hw_diag_with_obs):
        lat = np.array([-45.0, 15.0])
        lon = np.array([0.0, 90.0])
        mask = hw_diag_with_obs._load_land_mask("model-A", lat, lon)
        assert mask is not None
        assert mask.dtype == bool

    def test_be_mean_tmax_loaded_and_in_kelvin(self, hw_diag_with_obs):
        lat = np.array([-45.0, 15.0])
        lon = np.array([0.0, 90.0])
        result = hw_diag_with_obs._load_be_mean_tmax(lat, lon)
        assert result is not None
        # BE clim is 25°C → expect ~298 K
        assert float(result.mean()) == pytest.approx(273.15 + 25.0, abs=2.0)


# ── Unit tests: plot functions ────────────────────────────────────────────────

class TestPlot:
    def _make_results(self, hw_diag):
        import matplotlib
        matplotlib.use("Agg")
        return hw_diag.compute()

    def test_plot_returns_11_figures_with_obs(self, hw_diag_with_obs):
        import matplotlib
        matplotlib.use("Agg")
        results = hw_diag_with_obs.compute()
        figs = hw_diag_with_obs.plot(results)
        # 5 maps + 5 timeseries + 1 bias = 11
        assert len(figs) == 11

    def test_plot_returns_10_figures_without_obs(self, hw_diag, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        results = hw_diag.compute()
        figs = hw_diag.plot(results)
        # 5 maps + 5 timeseries, no bias
        assert len(figs) == 10

    def test_plot_figure_ids_are_unique(self, hw_diag, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        results = hw_diag.compute()
        figs = hw_diag.plot(results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert len(ids) == len(set(ids))

    def test_plot_expected_map_ids(self, hw_diag, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        results = hw_diag.compute()
        figs = hw_diag.plot(results)
        ids = [meta["figure_id"] for _, meta in figs]
        for idx in ("hwn", "hwf", "hwd", "hwm", "hwa"):
            assert f"heatwave_{idx}_map" in ids
            assert f"heatwave_{idx}_timeseries" in ids

    def test_bias_figure_id_present_with_obs(self, hw_diag_with_obs):
        import matplotlib
        matplotlib.use("Agg")
        results = hw_diag_with_obs.compute()
        figs = hw_diag_with_obs.plot(results)
        ids = [meta["figure_id"] for _, meta in figs]
        assert "heatwave_tmax_bias" in ids

    def test_plot_no_models_returns_empty(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from feather.diag.heatwave import HeatwaveDiag

        config = _make_config(tmp_path, models=())
        diag = HeatwaveDiag(None, None, config, period=("1980", "1984"))
        figs = diag.plot({"models": [], "hw_clim": {}, "hw_series": {},
                          "model_mean_tmax": {}, "obs_mean_tmax": None,
                          "lat": None, "lon": None})
        assert figs == []

    def test_plot_metadata_contains_period(self, hw_diag, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        results = hw_diag.compute()
        figs = hw_diag.plot(results)
        for _, meta in figs:
            assert "period" in meta or "period_start" in str(meta)


# ── Unit tests: run() ─────────────────────────────────────────────────────────

class TestRun:
    def test_run_creates_png_and_json(self, hw_diag, monkeypatch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        saved = hw_diag.run(skip_existing=False)
        assert len(saved) == 10
        for png, json_path in saved:
            assert png.suffix == ".png"
            assert json_path.suffix == ".json"

    def test_run_nc_in_correct_directory(self, hw_diag, monkeypatch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        hw_diag.run(skip_existing=False)
        nc_path = hw_diag._nc_path("ModelA")
        assert nc_path.exists()
        assert str(nc_path).startswith(str(hw_diag.nc_dir))

    def test_run_skip_existing(self, hw_diag, monkeypatch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        hw_diag.run(skip_existing=False)
        saved_second = hw_diag.run(skip_existing=True)
        assert saved_second == []

    def test_run_force_regenerates(self, hw_diag, monkeypatch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(hw_diag, "_be_tmax_path", lambda: None)
        saved1 = hw_diag.run(skip_existing=False)
        saved2 = hw_diag.run(skip_existing=False)
        assert len(saved2) == len(saved1)


# ── Unit tests: registry ──────────────────────────────────────────────────────

def test_diagnostic_is_registered():
    from feather.diag.registry import get_diagnostic
    diag_cls = get_diagnostic("heatwave")
    assert diag_cls is not None


def test_diagnostic_name():
    from feather.diag.heatwave import HeatwaveDiag
    assert HeatwaveDiag.name == "heatwave"


def test_diagnostic_group():
    from feather.diag.heatwave import HeatwaveDiag
    assert HeatwaveDiag.group == "extremes"


def test_diagnostic_variables():
    from feather.diag.heatwave import HeatwaveDiag
    assert "tasmax" in HeatwaveDiag.variables


def test_tasmax_in_variable_registry():
    from feather.data.variables import get_var
    vinfo = get_var("tasmax")
    assert vinfo.name == "tasmax"
    assert vinfo.cmip6_table == "day"


def test_kerchunk_daily_max_has_tasmax():
    from feather.data.kerchunk_loader import _ATMOS2D_DAILY_MAX
    assert "tasmax" in _ATMOS2D_DAILY_MAX


def test_kerchunk_tasmax_maps_to_mx2t24():
    from feather.data.kerchunk_loader import _ATMOS2D_DAILY_MAX
    kname, scale = _ATMOS2D_DAILY_MAX["tasmax"]
    assert kname == "mx2t24"
    assert scale == pytest.approx(1.0)
