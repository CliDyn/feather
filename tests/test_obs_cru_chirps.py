"""Tests for CHIRPS/CRU/Berkeley-Land loaders and the obs-comparison engine.

Covers:
* ``ObsLoader.load_cru`` / ``load_chirps`` / ``load_berkeley_hr``
* the shared ``_obs_compare_common`` engine
* the ``cloud_obs_comparison`` and ``temp_extremes_obs_comparison`` diagnostics

All synthetic — safe on a login node.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.obs import ObsLoader
from feather.diag import _obs_compare_common as occ

# ── Synthetic data builders ─────────────────────────────────────────────────

_LATS = np.arange(-87.5, 90.0, 7.5)        # 24 lats
_LONS_180 = np.arange(-177.5, 180.0, 7.5)  # 48 lons, -180..180


def _months(start, n):
    return xr.date_range(start, periods=n, freq="MS", calendar="standard")


def _cru_file(path, var, units, start, n, fill_ocean=True, seed=0):
    """Write a CRU-style decadal NetCDF (lat/lon/time, land-only)."""
    rng = np.random.default_rng(seed)
    time = _months(start, n)
    base = np.cos(np.deg2rad(_LATS))[:, None] * np.ones(len(_LONS_180))[None, :]
    data = (10.0 * base[None] + rng.normal(0, 0.5, (n, len(_LATS), len(_LONS_180))))
    if fill_ocean:
        # make a deterministic "ocean" band NaN (every 3rd lon)
        data[:, :, ::3] = np.nan
    da = xr.DataArray(
        data.astype(np.float32), dims=("time", "lat", "lon"),
        coords={"time": time, "lat": _LATS, "lon": _LONS_180}, name=var,
    )
    da.encoding["_FillValue"] = 9.96921e36
    ds = da.to_dataset()
    # auxiliary variables that must be ignored
    ds["stn"] = (("time", "lat", "lon"), np.ones_like(data, dtype=np.int32))
    ds[var].attrs["units"] = units
    ds.to_netcdf(path)


def _chirps_file(path, start, n, seed=0):
    rng = np.random.default_rng(seed)
    time = _months(start, n)
    lats = np.arange(-57.5, 60.0, 7.5)   # 60N-60S
    base = 80.0 * np.exp(-np.abs(lats) / 30.0)  # mm/month tropical max
    data = (base[None, :, None] + rng.normal(0, 2, (n, len(lats), len(_LONS_180)))).clip(0)
    da = xr.DataArray(
        data.astype(np.float32), dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lats, "longitude": _LONS_180},
        name="precip",
    )
    da.to_dataset().to_netcdf(path)


def _berkeley_file(path, start_year, n_years, seed=0):
    """Berkeley HR format: decimal-year time, anomaly + 12-month climatology."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    dec_time = start_year + (np.arange(n) % 12) / 12.0 + np.arange(n) // 12
    anom = rng.normal(0, 1.5, (n, len(_LATS), len(_LONS_180))).astype(np.float32)
    clim = (15.0 * np.cos(np.deg2rad(_LATS))[None, :, None]
            * np.ones((12, 1, len(_LONS_180)))).astype(np.float32)
    ds = xr.Dataset(
        {
            "temperature": (("time", "latitude", "longitude"), anom),
            "climatology": (("month_number", "latitude", "longitude"), clim),
        },
        coords={
            "time": dec_time, "latitude": _LATS, "longitude": _LONS_180,
            "month_number": np.arange(1, 13),
        },
    )
    ds.to_netcdf(path)


@pytest.fixture
def obs_config(tmp_path):
    cru_dir = tmp_path / "CRU"
    chirps_dir = tmp_path / "CHIRPS"
    be_dir = tmp_path / "BE"
    for d in (cru_dir, chirps_dir, be_dir):
        d.mkdir()
    # Two CRU decades per variable to exercise mfdataset globbing
    for var, units in [("tmp", "degrees Celsius"), ("pre", "mm/month"),
                       ("tmn", "degrees Celsius"), ("tmx", "degrees Celsius"),
                       ("cld", "percentage")]:
        _cru_file(cru_dir / f"cru_ts4.09.2001.2010.{var}.dat.nc",
                  var, units, "2001-01", 120, seed=1)
        _cru_file(cru_dir / f"cru_ts4.09.2011.2020.{var}.dat.nc",
                  var, units, "2011-01", 120, seed=2)
    _chirps_file(chirps_dir / "chirps.nc", "2001-01", 120)
    _berkeley_file(be_dir / "tmax.nc", 2001, 10)
    return FeatherConfig(
        model_catalogs={}, models=[], obs_root=str(tmp_path),
        obs_datasets={
            "CRU": {"path": str(cru_dir),
                    "variables": {v: f"cru_ts4.09.*.{v}.dat.nc"
                                  for v in ["tmp", "pre", "tmn", "tmx", "cld"]}},
            "CHIRPS": {"path": str(chirps_dir),
                       "variables": {"precip": "chirps.nc"}},
            "BERKELEY_EARTH_LAND_TMAX": {"path": str(be_dir),
                                         "variables": {"temperature": "tmax.nc"}},
        },
        cmip6={"enabled": False}, dask={},
        output_dir=str(tmp_path / "out"), nereus={"method": "nearest"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadCRU:
    def test_concatenates_decades(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp")
        assert da.sizes["time"] == 240  # 2 decades × 120 months

    def test_temp_kelvin(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp")
        # land cells (finite) should be ~283 K (10 °C base)
        assert float(da.max()) > 200  # clearly K, not °C

    def test_precip_to_flux(self, obs_config):
        da = ObsLoader(obs_config).load_cru("pre")
        # mm/month → kg/m²/s is a small number
        assert float(da.max(skipna=True)) < 1e-2

    def test_cloud_unchanged_percent(self, obs_config):
        da = ObsLoader(obs_config).load_cru("cld")
        assert float(da.max(skipna=True)) < 100

    def test_lons_shifted_to_360(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp")
        assert float(da.lon.min()) >= 0
        assert float(da.lon.max()) < 360

    def test_aux_vars_dropped(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp")
        assert da.name == "tmp"

    def test_land_only_has_nan(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp")
        assert bool(np.isnan(da).any())

    def test_period_slice(self, obs_config):
        da = ObsLoader(obs_config).load_cru("tmp", ("2011", "2020"))
        assert da.sizes["time"] == 120

    def test_missing_dataset_raises(self, obs_config):
        obs_config.obs_datasets.pop("CRU")
        with pytest.raises(KeyError):
            ObsLoader(obs_config).load_cru("tmp")


class TestLoadCHIRPS:
    def test_returns_flux(self, obs_config):
        da = ObsLoader(obs_config).load_chirps()
        assert float(da.max(skipna=True)) < 1e-2  # kg/m²/s

    def test_dims_renamed(self, obs_config):
        da = ObsLoader(obs_config).load_chirps()
        assert "lat" in da.dims and "lon" in da.dims

    def test_lons_360(self, obs_config):
        da = ObsLoader(obs_config).load_chirps()
        assert float(da.lon.min()) >= 0

    def test_extent_60ns(self, obs_config):
        da = ObsLoader(obs_config).load_chirps()
        assert float(da.lat.max()) <= 60 and float(da.lat.min()) >= -60

    def test_period_slice(self, obs_config):
        da = ObsLoader(obs_config).load_chirps(("2005", "2010"))
        assert da.sizes["time"] <= 72


class TestLoadBerkeleyHR:
    def test_reconstructs_kelvin(self, obs_config):
        da = ObsLoader(obs_config).load_berkeley_hr("BERKELEY_EARTH_LAND_TMAX")
        assert float(da.max()) > 200  # K

    def test_dims_and_lons(self, obs_config):
        da = ObsLoader(obs_config).load_berkeley_hr("BERKELEY_EARTH_LAND_TMAX")
        assert "lat" in da.dims and "lon" in da.dims
        assert float(da.lon.min()) >= 0

    def test_time_is_datetime(self, obs_config):
        da = ObsLoader(obs_config).load_berkeley_hr("BERKELEY_EARTH_LAND_TMAX")
        assert np.issubdtype(da.time.dtype, np.datetime64)

    def test_period_slice(self, obs_config):
        da = ObsLoader(obs_config).load_berkeley_hr(
            "BERKELEY_EARTH_LAND_TMAX", ("2005", "2010"))
        assert da.sizes["time"] <= 72


# ══════════════════════════════════════════════════════════════════════════════
# Engine helpers
# ══════════════════════════════════════════════════════════════════════════════


def _latlon_series(start, n, lats, lons, value=283.0, trend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    time = _months(start, n)
    base = value * np.ones((len(lats), len(lons)))
    tr = (trend / (10 * 12)) * np.arange(n)
    data = base[None] + tr[:, None, None] + rng.normal(0, 0.1, (n, len(lats), len(lons)))
    return xr.DataArray(
        data.astype(np.float32), dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons})


class TestEngineHelpers:
    def test_common_grid_shape(self):
        lats, lons = occ.common_grid_05()
        assert lats.size == 360 and lons.size == 720

    def test_interp_to_grid_renames(self):
        da = _latlon_series("2001-01", 12, _LATS, _LONS_180).isel(time=0)
        da = da.rename({"lat": "latitude", "lon": "longitude"})
        lats = np.arange(-80, 81, 20.0)
        lons = np.arange(10, 360, 40.0)
        out = occ.interp_to_grid(da, lats, lons)
        assert "lat" in out.dims and "lon" in out.dims

    def test_compute_trend_decadal(self):
        lons360 = (_LONS_180 + 360) % 360
        da = _latlon_series("2001-01", 120, _LATS, np.sort(lons360), trend=2.0)
        tr = occ.compute_trend(da, "annual")
        assert tr is not None
        assert abs(float(tr.mean()) - 2.0) < 0.5

    def test_compute_trend_too_short(self):
        da = _latlon_series("2001-01", 6, _LATS, _LONS_180)
        assert occ.compute_trend(da, "annual") is None

    def test_finite_concat_drops_nan(self):
        out = occ.finite_concat([np.array([1.0, np.nan]), np.array([3.0])])
        assert list(out) == [1.0, 3.0]

    def test_area_weighted_series(self):
        da = _latlon_series("2001-01", 36, _LATS, _LONS_180, value=5.0)
        s = occ.area_weighted_annual_series(da)
        assert s.sizes["year"] == 3
        assert abs(float(s.mean()) - 5.0) < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# PairwiseObsComparison engine
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine_diag(obs_config):
    from feather.diag.cloud_obs_comparison import CloudObsComparisonDiag
    return CloudObsComparisonDiag(MagicMock(), MagicMock(), obs_config)


@pytest.fixture
def engine_results(engine_diag):
    lons360 = np.sort((_LONS_180 + 360) % 360)
    ref_s = _latlon_series("2001-01", 60, _LATS, lons360, value=60.0, trend=1.0, seed=1)
    ref_l = _latlon_series("2001-01", 96, _LATS, lons360, value=60.0, trend=1.0, seed=2)
    sec_s = _latlon_series("2001-01", 60, _LATS, lons360, value=55.0, trend=0.5, seed=3)
    sec_l = _latlon_series("2001-01", 96, _LATS, lons360, value=55.0, trend=0.5, seed=4)
    spec = occ.CompareSpec(
        var="clt", ref_name="ERA5", sec_name="CRU", sec_token="cru",
        units_label="%", land_only=True)
    eng = occ.PairwiseObsComparison(engine_diag, spec, ("2001", "2004"), ("2001", "2008"))
    return eng, eng.compute(ref_s, ref_l, sec_s, sec_l)


class TestEngine:
    def test_compute_has_trends_clim(self, engine_results):
        _, res = engine_results
        assert set(res["trends"]) == {"annual", "DJF", "JJA"}
        assert set(res["clim"]) == {"annual", "DJF", "JJA"}

    def test_compute_diffs(self, engine_results):
        _, res = engine_results
        d = res["clim"]["annual"]["diff_short"]
        # ERA5 (60) − CRU (55) ≈ 5 over land
        assert abs(float(d.mean(skipna=True)) - 5.0) < 1.0

    def test_figures_count(self, engine_results):
        eng, res = engine_results
        figs = eng.figures(res)
        # 3 periods × (trends + trend_diffs + clim) = 9 (no relative bias)
        assert len(figs) == 9
        import matplotlib.pyplot as plt
        for fig, meta in figs:
            assert "figure_id" in meta
            plt.close(fig)

    def test_export_netcdf(self, engine_results, tmp_path):
        eng, res = engine_results
        written = eng.export_netcdf(res, tmp_path / "nc")
        assert len(written) == 3
        for p in written:
            assert Path(p).exists()
            ds = xr.open_dataset(p)
            assert "diff_short" in ds and "diff_long" in ds


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics end-to-end (mock loaders)
# ══════════════════════════════════════════════════════════════════════════════


def _mock_obs_loader_for_cloud():
    lons360 = np.sort((_LONS_180 + 360) % 360)
    loader = MagicMock()

    def _era5(var, period=None):
        n = 60 if (period and period[1] <= "2014") else 96
        return _latlon_series("1981-01", n, _LATS, lons360, value=0.6, seed=7) \
            .rename({"lat": "latitude", "lon": "longitude"})

    def _cru(var, period=None):
        n = 60 if (period and period[1] <= "2014") else 96
        d = _latlon_series("1981-01", n, _LATS, lons360, value=55.0, seed=8)
        d.values[:, :, ::3] = np.nan
        return d

    loader.load_for_model_var.side_effect = _era5
    loader.load_cru.side_effect = _cru
    return loader


class TestCloudDiag:
    def test_run_produces_figures(self, obs_config, tmp_path):
        from feather.diag.cloud_obs_comparison import CloudObsComparisonDiag
        # ERA5 obs_unit_factor for clt is 100 (fraction→%) applied by real
        # loader; mock returns % directly via load_for_model_var.
        diag = CloudObsComparisonDiag(
            MagicMock(), _mock_obs_loader_for_cloud(), obs_config)
        saved = diag.run(skip_existing=False)
        ids = {p[0].stem for p in saved}
        assert "clt_obs_timeseries" in ids
        assert "clt_cru_annual_clim" in ids
        # NetCDF exported
        nc = list((Path(obs_config.output_dir) / "netcdf" / "cloud_obs_comparison").glob("*.nc"))
        assert len(nc) == 3


class TestTempExtremesDiag:
    def test_run_one_variable(self, obs_config):
        from feather.diag.temp_extremes_obs_comparison import (
            TempExtremesObsComparisonDiag,
        )
        lons360 = np.sort((_LONS_180 + 360) % 360)
        loader = MagicMock()

        def _be(key, period=None):
            n = 60 if (period and period[1] <= "2014") else 96
            return _latlon_series("1981-01", n, _LATS, lons360, value=290.0, seed=11)

        def _cru(var, period=None):
            n = 60 if (period and period[1] <= "2014") else 96
            d = _latlon_series("1981-01", n, _LATS, lons360, value=289.0, seed=12)
            d.values[:, :, ::3] = np.nan
            return d

        loader.load_berkeley_hr.side_effect = _be
        loader.load_cru.side_effect = _cru
        diag = TempExtremesObsComparisonDiag(
            MagicMock(), loader, obs_config, variables=["tasmax"])
        saved = diag.run(skip_existing=False)
        ids = {p[0].stem for p in saved}
        assert "tasmax_obs_timeseries" in ids
        assert "tasmax_cru_annual_clim" in ids

    def test_era5_secondary_added(self, obs_config):
        """When ERA5_TMINMAX is configured, ERA5 appears as a 2nd secondary."""
        from feather.diag.temp_extremes_obs_comparison import (
            TempExtremesObsComparisonDiag,
        )
        obs_config.obs_datasets["ERA5_TMINMAX"] = {
            "path": "/tmp", "variables": {"tasmax": "x.nc", "tasmin": "y.nc"}}
        lons360 = np.sort((_LONS_180 + 360) % 360)
        loader = MagicMock()

        def _be(key, period=None):
            n = 60 if (period and period[1] <= "2014") else 96
            return _latlon_series("1981-01", n, _LATS, lons360, value=290.0, seed=11)

        def _cru(var, period=None):
            n = 60 if (period and period[1] <= "2014") else 96
            d = _latlon_series("1981-01", n, _LATS, lons360, value=289.0, seed=12)
            d.values[:, :, ::3] = np.nan
            return d

        def _load(ds_key, var, period=None):
            n = 60 if (period and period[1] <= "2014") else 96
            return _latlon_series("1981-01", n, _LATS, lons360, value=290.5, seed=13)

        loader.load_berkeley_hr.side_effect = _be
        loader.load_cru.side_effect = _cru
        loader.load.side_effect = _load
        diag = TempExtremesObsComparisonDiag(
            MagicMock(), loader, obs_config, variables=["tasmax"])
        saved = diag.run(skip_existing=False)
        ids = {p[0].stem for p in saved}
        assert "tasmax_cru_annual_clim" in ids
        assert "tasmax_era5_annual_clim" in ids
