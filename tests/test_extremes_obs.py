"""Tests for the configurable ERA5/Berkeley extremes obs reference."""

import numpy as np
import xarray as xr

from feather.config import FeatherConfig
from feather.diag import _extremes_obs as eo


def _make_config(tmp_path, *, era5=True, with_sftlf=True, ref="ERA5"):
    lats = np.arange(-87.5, 90.0, 7.5)
    lons = np.arange(1.25, 360.0, 7.5)
    time = xr.date_range("1981-01", periods=48, freq="MS", calendar="standard")
    obs_datasets = {}
    if era5:
        mon = tmp_path / "mon"
        mon.mkdir(exist_ok=True)
        for var, base in [("tasmin", 285.0), ("tasmax", 295.0)]:
            data = base + 5 * np.cos(np.deg2rad(lats))[None, :, None] * np.ones(
                (len(time), 1, len(lons)))
            da = xr.DataArray(
                data.astype("float32"), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons}, name=var)
            da.to_dataset().to_netcdf(mon / f"{var}.nc")
        obs_datasets["ERA5_TMINMAX"] = {
            "path": str(mon),
            "variables": {"tasmin": "tasmin.nc", "tasmax": "tasmax.nc"}}
        if with_sftlf:
            sm = np.where(lats[:, None] < 0, 80.0, 10.0) * np.ones((1, len(lons)))
            xr.DataArray(sm, dims=("lat", "lon"),
                         coords={"lat": lats, "lon": lons}, name="sftlf"
                         ).to_dataset().to_netcdf(tmp_path / "sftlf.nc")
            obs_datasets["ERA5_SFTLF"] = {
                "path": str(tmp_path), "variables": {"sftlf": "sftlf.nc"}}
    cfg = FeatherConfig(
        model_catalogs={}, models=[], obs_root=str(tmp_path),
        obs_datasets=obs_datasets, cmip6={"enabled": False}, dask={},
        output_dir=str(tmp_path / "out"), nereus={"method": "nearest"})
    cfg.project["extremes_obs_reference"] = ref
    return cfg, lats, lons


class TestSelection:
    def test_use_era5_true(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, ref="ERA5")
        assert eo.use_era5_obs(cfg) is True

    def test_use_era5_false_when_not_requested(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, ref="Berkeley")
        assert eo.use_era5_obs(cfg) is False

    def test_use_era5_false_when_dataset_missing(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, era5=False, ref="ERA5")
        assert eo.use_era5_obs(cfg) is False

    def test_label(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, ref="ERA5")
        assert eo.obs_ref_label(cfg, "tasmin") == "ERA5"
        cfg2, *_ = _make_config(tmp_path, ref="Berkeley")
        assert eo.obs_ref_label(cfg2, "tasmin") == "Berkeley Earth Land TMIN"
        assert eo.obs_ref_label(cfg2, "tasmax") == "Berkeley Earth Land TMAX"

    def test_mean_available(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, ref="ERA5")
        assert eo.era5_mean_available(cfg, "tasmin") is True

    def test_obs_ref_model_name(self, tmp_path):
        # ERA5 reference doubles as a model → excluded from bias panels.
        cfg, *_ = _make_config(tmp_path, ref="ERA5")
        assert eo.obs_ref_model_name(cfg) == "ERA5"
        # Berkeley reference is never a model → nothing to exclude.
        cfg2, *_ = _make_config(tmp_path, ref="Berkeley")
        assert eo.obs_ref_model_name(cfg2) is None


class TestLoadMean:
    def test_period_mean_kelvin_on_model_grid(self, tmp_path):
        cfg, lats, lons = _make_config(tmp_path)
        mlat = np.arange(-80, 81, 20.0)
        mlon = np.arange(10, 360, 40.0)
        da = eo.load_era5_mean(cfg, "tasmin", ("1981", "1982"), mlat, mlon)
        assert da is not None
        assert da.sizes["lat"] == len(mlat) and da.sizes["lon"] == len(mlon)
        assert 270 < float(da.mean()) < 300  # K

    def test_returns_none_without_dataset(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, era5=False)
        assert eo.load_era5_mean(cfg, "tasmin", ("1981", "1982"),
                                 np.array([0.0]), np.array([0.0])) is None


class TestApproxSeries:
    def test_land_masked_series(self, tmp_path):
        cfg, *_ = _make_config(tmp_path)
        s = eo.era5_approx_exceedance_series(cfg, "tasmin", "1981", "1984", 5.0)
        assert s is not None
        assert "year" in s.dims
        assert s.sizes["year"] == 4

    def test_none_without_sftlf(self, tmp_path):
        cfg, *_ = _make_config(tmp_path, with_sftlf=False)
        s = eo.era5_approx_exceedance_series(cfg, "tasmin", "1981", "1984", 5.0)
        assert s is None
