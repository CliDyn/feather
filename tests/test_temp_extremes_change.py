"""Tests for TempExtremesChangeDiag (daily Tmin/Tmax climate change signal)."""

import numpy as np
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.diag.temp_extremes_change import (
    TempExtremesChangeDiag,
    _K_TO_C,
    _PERIODS,
    _SEASONS,
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────


def _make_daily(
    start: str = "1981-01-01",
    n_years: int = 4,
    nlat: int = 9,
    nlon: int = 18,
    base_k: float = 285.0,
) -> xr.DataArray:
    """Synthetic daily temperature field, constant in time (pole→equator gradient)."""
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 350, nlon)
    n_days = 365 * n_years
    time = xr.date_range(start, periods=n_days, freq="1D", calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    field = base_k - 34.0 * np.abs(lat_grid) / 80.0
    data = np.broadcast_to(field[np.newaxis], (n_days, nlat, nlon)).astype(np.float32)
    return xr.DataArray(
        np.ascontiguousarray(data),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )


class _MockLoader:
    """Loader returning a per-variable DataArray; supports table= and period=."""

    def __init__(self, per_var: dict[str, xr.DataArray]):
        self._per_var = per_var

    def load_var(self, model, variable, *, table=None, period=None):
        da = self._per_var[variable]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_coords(self, model, variable="tas"):
        da = next(iter(self._per_var.values()))
        return np.asarray(da["lon"]), np.asarray(da["lat"])


class _MockObsLoader:
    def load(self, *a, **kw):
        raise FileNotFoundError

    def load_for_model_var(self, *a, **kw):
        raise FileNotFoundError


def _make_config(tmp_path, models=("model-A", "model-B")):
    cc = {
        "reference_period": ["1981", "1984"],
        "future_period":    ["2015", "2018"],
        "hist_load_period": ["1981", "1984"],
        "ssp_load_period":  ["2015", "2018"],
        "models": {
            m: {
                "hist_data_source": "cmor",
                "hist_experiment":  "hist-1950",
                "future_data_source": "cmor",
                "future_experiment":  "highres-future-ssp245",
            }
            for m in models
        },
    }
    model_configs = {
        m: ModelConfig(
            name=m, institution="TEST", experiment="hist-1950",
            variant="r1i1p1f1", grids={"sfc": "latlon"}, color="#1f77b4",
        )
        for m in models
    }
    return FeatherConfig(
        model_catalogs={}, models=list(models), model_configs=model_configs,
        obs_root="", obs_datasets={}, cmip6={"enabled": False}, dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        project={"name": "TEST", "climate_change": cc},
    )


def _hist_vars():
    return {"tasmin": _make_daily(base_k=285.0), "tasmax": _make_daily(base_k=295.0)}


def _ssp_vars():
    return {
        "tasmin": _make_daily(start="2015-01-01", base_k=288.0),   # +3 K
        "tasmax": _make_daily(start="2015-01-01", base_k=298.0),   # +3 K
    }


def _make_diag(config, hist=None, ssp=None):
    diag = TempExtremesChangeDiag(
        model_loader=_MockLoader(hist or _hist_vars()),
        obs_loader=_MockObsLoader(),
        config=config,
    )
    diag._make_hist_loader = lambda model: _MockLoader(hist or _hist_vars())
    if ssp is not None:
        diag._make_fut_loader = lambda model: _MockLoader(ssp)
    else:
        diag._make_fut_loader = lambda model: None
    return diag


# ── Registration / config ─────────────────────────────────────────────────────


def test_registered():
    from feather.diag.registry import get_diagnostic, registered_names
    assert "temp_extremes_change" in registered_names()
    assert get_diagnostic("temp_extremes_change") is TempExtremesChangeDiag


def test_variables_and_group():
    assert TempExtremesChangeDiag.variables == ["tasmin", "tasmax"]
    assert TempExtremesChangeDiag.group == "extremes"


def test_variable_selection_overlap(tmp_path):
    config = _make_config(tmp_path)
    diag = TempExtremesChangeDiag(
        _MockLoader(_hist_vars()), _MockObsLoader(), config, variables=["tasmax", "pr"]
    )
    assert diag.variables == ["tasmax"]


def test_variable_selection_empty_overlap_keeps_both(tmp_path):
    config = _make_config(tmp_path)
    diag = TempExtremesChangeDiag(
        _MockLoader(_hist_vars()), _MockObsLoader(), config, variables=["pr"]
    )
    assert diag.variables == ["tasmin", "tasmax"]


# ── NC paths ──────────────────────────────────────────────────────────────────


def test_nc_paths(tmp_path):
    diag = _make_diag(_make_config(tmp_path))
    hist = diag._nc_path("model-A", "tasmin", "hist")
    ssp = diag._nc_path("model-A", "tasmax", "ssp")
    assert "model-A_tasmin_hist" in hist.name and hist.suffix == ".nc"
    assert "model-A_tasmax_ssp" in ssp.name
    assert diag._nc_path("IFS/r2", "tasmin", "hist").name.count("/") == 0


# ── Derivation helpers ────────────────────────────────────────────────────────


def test_annual_series_dims():
    da = _make_daily(n_years=3)
    out = TempExtremesChangeDiag._annual_series(da)
    assert out.dims == ("year", "lat", "lon")
    assert list(out["year"].values) == [1981, 1982, 1983]
    # Field constant in time → annual mean equals the field value
    assert np.allclose(out.isel(year=0).values, da.isel(time=0).values, atol=1e-3)


def test_period_mean_slices_and_means():
    da = _make_daily(n_years=4)
    series = {"annual": TempExtremesChangeDiag._annual_series(da)}
    out = TempExtremesChangeDiag._period_mean(series, ("1982", "1983"))
    assert "annual" in out
    assert out["annual"].dims == ("lat", "lon")


def test_period_mean_out_of_range_empty():
    da = _make_daily(n_years=2)
    series = {"annual": TempExtremesChangeDiag._annual_series(da)}
    out = TempExtremesChangeDiag._period_mean(series, ("2100", "2110"))
    assert out == {}


# ── Checkpoint round-trip ─────────────────────────────────────────────────────


def test_load_and_save_means_writes_nc(tmp_path):
    diag = _make_diag(_make_config(tmp_path))
    loader = _MockLoader(_hist_vars())
    nc = diag._nc_path("model-A", "tasmin", "hist")
    series = diag._load_and_save_means(
        "model-A", "tasmin", loader, ("1981", "1984"), nc
    )
    assert nc.exists()
    assert set(series) == set(_PERIODS)
    # File holds one variable per season/annual key.
    ds = xr.open_dataset(nc)
    for s in _PERIODS:
        assert f"tasmin_{s}" in ds
        assert "year" in ds[f"tasmin_{s}"].dims


def test_load_and_save_means_reads_existing_nc(tmp_path):
    diag = _make_diag(_make_config(tmp_path))
    nc = diag._nc_path("model-A", "tasmin", "hist")
    diag._load_and_save_means("model-A", "tasmin", _MockLoader(_hist_vars()),
                              ("1981", "1984"), nc)
    # Second call must NOT touch the loader (raises if used).
    class _Boom:
        def load_var(self, *a, **k):
            raise AssertionError("loader should not be called on cache hit")
    series = diag._load_and_save_means("model-A", "tasmin", _Boom(),
                                       ("1981", "1984"), nc)
    assert set(series) == set(_PERIODS)


def test_load_and_save_means_returns_none_on_failure(tmp_path):
    diag = _make_diag(_make_config(tmp_path))
    class _Fail:
        def load_var(self, *a, **k):
            raise FileNotFoundError("no data")
    out = diag._load_and_save_means("model-A", "tasmin", _Fail(),
                                    ("1981", "1984"), diag._nc_path("model-A", "tasmin", "hist"))
    assert out is None


# ── compute() end-to-end ──────────────────────────────────────────────────────


def test_compute_populates_change(tmp_path):
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist=_hist_vars(), ssp=_ssp_vars())
    res = diag.compute()
    assert res["models"]["tasmin"] == ["model-A", "model-B"]
    assert res["models"]["tasmax"] == ["model-A", "model-B"]
    for var in ("tasmin", "tasmax"):
        for season in _PERIODS:
            chg = res["change"][var]["model-A"][season]
            # +3 K warming imposed uniformly.
            assert np.allclose(np.asarray(chg), 3.0, atol=1e-2)


def test_compute_reference_only_when_no_future(tmp_path):
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist=_hist_vars(), ssp=None)
    res = diag.compute()
    assert res["models"]["tasmin"] == ["model-A", "model-B"]
    assert res["change"]["tasmin"] == {}
    assert res["fut"]["tasmin"] == {}
    assert "model-A" in res["hist_series"]["tasmin"]


def test_compute_global_mean_series(tmp_path):
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist=_hist_vars(), ssp=_ssp_vars())
    res = diag.compute()
    h = res["hist_series"]["tasmin"]["model-A"]
    s = res["ssp_series"]["tasmin"]["model-A"]
    assert "year" in h.dims and "year" in s.dims
    # SSP global-mean warmer than hist by ~3 K.
    assert float(s.mean()) - float(h.mean()) > 2.5


# ── plot() ────────────────────────────────────────────────────────────────────


def test_plot_figure_count_and_ids(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist=_hist_vars(), ssp=_ssp_vars())
    figs = diag.plot(diag.compute())
    ids = {meta["figure_id"] for _f, meta in figs}
    # Per variable: 5 map panels (annual + 4 seasons) + 1 timeseries = 6; 2 vars = 12
    assert len(figs) == 12
    for var in ("tasmin", "tasmax"):
        for season in _PERIODS:
            assert f"{var}_{season}_change_panels" in ids
        assert f"{var}_change_timeseries" in ids
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_skips_variable_without_models(tmp_path):
    config = _make_config(tmp_path)
    diag = _make_diag(config, hist=_hist_vars(), ssp=_ssp_vars())
    res = diag.compute()
    res["models"]["tasmax"] = []          # simulate all-tasmax loads failing
    figs = diag.plot(res)
    ids = {meta["figure_id"] for _f, meta in figs}
    assert not any(i.startswith("tasmax_") for i in ids)
    assert f"tasmin_annual_change_panels" in ids
    import matplotlib.pyplot as plt
    plt.close("all")


def test_seasons_constant():
    assert _SEASONS == ["DJF", "MAM", "JJA", "SON"]
    assert _PERIODS == ["annual", "DJF", "MAM", "JJA", "SON"]
