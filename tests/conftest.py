"""Shared test fixtures for feather tests."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.variables import VARIABLE_REGISTRY


# ── Synthetic data fixtures ──────────────────────────────────────────


@pytest.fixture
def synth_healpix():
    """Small synthetic HEALPix dataset (nside=8, 768 cells, 12 timesteps).

    Has a temperature gradient from pole to equator for testing:
    T = 300 - 40 * abs(lat/90)  (warm equator, cold poles).
    """
    nside = 8
    ncells = 12 * nside**2  # 768

    # Generate HEALPix pixel centres
    import healpy as hp
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    # Synthetic temperature: gradient from equator (300K) to poles (260K)
    temp_base = 300 - 40 * np.abs(lat / 90.0)

    # Create monthly time axis (12 months)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Add seasonal cycle: +5K in summer, -5K in winter (NH convention)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    # Build 2D field: (time, values)
    temp_2d = temp_base[np.newaxis, :] + seasonal[:, np.newaxis]

    # HEALPix cells are equal area
    area = np.ones(ncells) * (4 * np.pi / ncells)

    ds = xr.Dataset(
        {
            "avg_2t": xr.DataArray(
                temp_2d, dims=("time", "values"),
                coords={"time": time},
            ),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(area, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_obs():
    """Synthetic observation on regular lat/lon grid (5-degree resolution).

    Same temperature field as synth_healpix for bias testing.
    """
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    # Temperature gradient matching synth_healpix
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    ds = xr.Dataset(
        {
            "t2m": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        }
    )
    return ds


# ── Configuration fixture ────────────────────────────────────────────


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal FeatherConfig for testing diagnostics."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
    )


# ── Mock data loaders ────────────────────────────────────────────────


class MockModelLoader:
    """Mock DataLoader that returns the given dataset for any key."""

    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset

    def load(self, key: str) -> xr.Dataset:
        return self._ds

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        return self._ds[variable]

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        return DataLoader.make_key(experiment, model, domain, member)


class MockObsLoader:
    """Mock ObsLoader that returns a fixed DataArray for any request."""

    def __init__(self, dataset: xr.Dataset, var_name: str = "t2m"):
        self._ds = dataset
        self._var_name = var_name

    def load(self, dataset, variable, period=None):
        da = self._ds[self._var_name]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_for_model_var(self, model_var, period=None):
        return self.load(None, None, period=period)


@pytest.fixture
def mock_model_loader(synth_healpix):
    """MockModelLoader backed by synth_healpix."""
    return MockModelLoader(synth_healpix)


@pytest.fixture
def mock_obs_loader(synth_obs):
    """MockObsLoader backed by synth_obs."""
    return MockObsLoader(synth_obs)


# ── CMIP6 synthetic data & mock loader ──────────────────────────────


@pytest.fixture
def synth_cmip6():
    """Synthetic CMIP6-like dataset on a 5-degree regular lat/lon grid.

    Has 12 monthly timesteps, ``tas`` and ``areacella`` variables.
    Temperature follows the same gradient as synth_obs (300K equator, 260K poles).
    """
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    # Approximate cell areas (cos-lat scaling, arbitrary units)
    area_2d = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "areacella": xr.DataArray(
                area_2d, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        }
    )
    return ds


class MockCMIP6Loader:
    """Mock CMIP6Loader backed by synthetic data.

    Follows the real CMIP6Loader API so diagnostics can be tested
    without real CMIP6 zarr data.
    """

    def __init__(self, dataset: xr.Dataset, models: dict | None = None):
        self._ds = dataset
        self._models = models or {
            "ModelA": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
            "ModelB": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
        }

    @property
    def models(self) -> dict:
        return self._models

    def load_var(self, cmip6_var, model, *, variant=None, table=None,
                 period=None, season=None):
        if cmip6_var not in self._ds.data_vars:
            return None
        da = self._ds[cmip6_var]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if season and "time" in da.dims:
            da = da.sel(time=da["time.season"] == season)
        if "time" in da.dims:
            da = da.mean("time")
        return da

    def load_var_for_model_var(self, model_var, model, **kwargs):
        vinfo = VARIABLE_REGISTRY.get(model_var)
        if vinfo is None or not vinfo.cmip6_variable:
            return None
        return self.load_var(vinfo.cmip6_variable, model, **kwargs)

    def load_multi_model_mean(self, cmip6_var, **kwargs):
        da = self.load_var(cmip6_var, list(self._models)[0])
        if da is None:
            return None, {"n_members": 0, "models_used": [], "models_skipped": []}
        info = {
            "n_members": sum(
                len(cfg.get("variants", [cfg.get("variant", "r1i1p1f1")]))
                for cfg in self._models.values()
            ),
            "models_used": [
                f"{m}/{v}"
                for m, cfg in self._models.items()
                for v in cfg.get("variants", [cfg.get("variant", "r1i1p1f1")])
            ],
            "models_skipped": [],
        }
        return da, info

    def load_mmm_for_model_var(self, model_var, **kwargs):
        vinfo = VARIABLE_REGISTRY.get(model_var)
        if vinfo is None or not vinfo.cmip6_variable:
            return None, {"n_members": 0, "models_used": [], "models_skipped": []}
        return self.load_multi_model_mean(vinfo.cmip6_variable, **kwargs)

    def load_area(self, model, variant=None, table="Amon"):
        if "areacella" in self._ds.data_vars:
            return self._ds["areacella"]
        return None

    def available_models(self, cmip6_var, table=None):
        if cmip6_var in self._ds.data_vars:
            return list(self._models)
        return []

    def available_members(self, cmip6_var, table=None):
        if cmip6_var not in self._ds.data_vars:
            return []
        return [
            (m, v)
            for m, cfg in self._models.items()
            for v in cfg.get("variants", [cfg.get("variant", "r1i1p1f1")])
        ]

    def available_models_for_model_var(self, model_var):
        vinfo = VARIABLE_REGISTRY.get(model_var)
        if vinfo is None or not vinfo.cmip6_variable:
            return []
        return self.available_models(vinfo.cmip6_variable)


@pytest.fixture
def mock_cmip6_loader(synth_cmip6):
    """MockCMIP6Loader backed by synth_cmip6."""
    return MockCMIP6Loader(synth_cmip6)


@pytest.fixture
def cmip6_config(tmp_path):
    """FeatherConfig with CMIP6 enabled and two fake models."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={
            "enabled": True,
            "catalog_path": str(tmp_path / "fake_catalog.yaml"),
            "regrid_resolution": 1.0,
            "ensemble_mode": "one_per_model",
            "models": {
                "ModelA": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
                "ModelB": {"variants": ["r1i1p1f1", "r2i1p1f1"]},
            },
        },
        dask={},
        nereus={},
        output_dir=str(tmp_path / "output"),
    )
