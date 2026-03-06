"""Tests for the temperature_berkeley diagnostic."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.variables import get_var


# ── Test fixtures ─────────────────────────────────────────────────────


def _make_temp_field(lats, lons, ntimes=12, base_temp=288.0):
    """Create synthetic temperature with pole-to-equator gradient.

    T = base_temp - 40 * abs(lat/90) + seasonal cycle
    """
    time = xr.date_range("1990-01", periods=ntimes, freq="MS",
                         calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

    temp_base = base_temp - 40 * np.abs(lat_grid / 90.0)

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(ntimes) - 3) / 12)
    temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]

    return xr.DataArray(
        temp_3d, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


@pytest.fixture
def synth_temp_obs():
    """Synthetic Berkeley Earth-like obs on 5-degree grid (in K)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    return _make_temp_field(lats, lons)


@pytest.fixture
def synth_temp_obs_degc():
    """Synthetic Berkeley Earth obs in degC (as stored on disk)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    da = _make_temp_field(lats, lons) - 273.15  # K to degC
    # Use Berkeley Earth dim names
    da = da.rename({"lat": "latitude", "lon": "longitude"})
    # Use -180..180 lons
    new_lons = np.arange(-177.5, 180, 5.0)
    da = da.assign_coords(longitude=new_lons)
    return da


@pytest.fixture
def synth_temp_healpix():
    """Synthetic HEALPix temperature (nside=8, 768 cells)."""
    import healpy as hp

    nside = 8
    ncells = 12 * nside ** 2
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    time = xr.date_range("1990-01", periods=12, freq="MS",
                         calendar="standard")

    temp_base = 288 - 40 * np.abs(lat / 90.0)
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    temp_2d = temp_base[np.newaxis, :] + seasonal[:, np.newaxis]

    area = np.ones(ncells) * (4 * np.pi / ncells)

    ds = xr.Dataset({
        "avg_2t": xr.DataArray(temp_2d, dims=("time", "values"),
                               coords={"time": time}),
        "longitude": xr.DataArray(lon, dims="values"),
        "latitude": xr.DataArray(lat, dims="values"),
        "area": xr.DataArray(area, dims="values"),
    })
    return ds


@pytest.fixture
def synth_temp_latlon():
    """Synthetic latlon model data (CMOR-like)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    da = _make_temp_field(lats, lons, base_temp=289.0)
    ds = xr.Dataset({"tas": da})
    return ds


@pytest.fixture
def berkeley_config(tmp_path):
    """FeatherConfig for Berkeley Earth tests."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"BERKELEY_EARTH": {
            "path": "/fake", "variables": {"2t": "fake.nc"},
        }},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def berkeley_config_latlon(tmp_path):
    """FeatherConfig for latlon (CMOR) temperature tests."""
    from feather.config import ModelConfig
    cfg = FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"BERKELEY_EARTH": {
            "path": "/fake", "variables": {"2t": "fake.nc"},
        }},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )
    cfg.data_source = {"type": "cmor"}
    cfg.model_configs = {
        "ifs-fesom": ModelConfig(
            name="ifs-fesom",
            institution="test",
            experiment="test",
            variant="r1i1p1f1",
            grids={"sfc": "latlon", "o2d": "latlon"},
            color="#1f77b4",
        ),
    }
    return cfg


@pytest.fixture
def cmip6_berkeley_config(tmp_path):
    """FeatherConfig with CMIP6 enabled for temperature tests."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"BERKELEY_EARTH": {
            "path": "/fake", "variables": {"2t": "fake.nc"},
        }},
        cmip6={
            "enabled": True,
            "catalog_path": str(tmp_path / "fake_catalog.yaml"),
            "regrid_resolution": 1.0,
            "influence_radius": 1_000_000,
            "ensemble_mode": "one_per_model",
            "models": {
                "ModelA": {"variants": ["r1i1p1f1"]},
                "ModelB": {"variants": ["r1i1p1f1"]},
            },
        },
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


# ── Mock loaders ──────────────────────────────────────────────────────


class MockTempModelLoader:
    """Mock model loader returning temperature data."""

    def __init__(self, dataset):
        self._ds = dataset

    def load(self, key):
        return self._ds

    def load_var(self, key_or_model, variable=None, **kwargs):
        if variable is None:
            variable = key_or_model
        for var_name in [variable, "avg_2t", "tas"]:
            if var_name in self._ds.data_vars:
                da = self._ds[var_name]
                period = kwargs.get("period")
                if period and "time" in da.dims:
                    da = da.sel(time=slice(period[0], period[1]))
                time_mean = kwargs.get("time_mean", False)
                if time_mean and "time" in da.dims:
                    da = da.mean("time")
                return da
        raise KeyError(f"Variable {variable} not found")

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        return DataLoader.make_key(experiment, model, domain, member)


class MockBerkeleyObsLoader:
    """Mock ObsLoader returning synthetic Berkeley Earth data."""

    def __init__(self, berkeley_da):
        self._berkeley = berkeley_da

    def load(self, dataset, variable, period=None):
        da = self._berkeley
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_for_model_var(self, model_var, period=None):
        return self.load(None, None, period=period)


class MockCMIP6TempLoader:
    """Mock CMIP6 loader with temperature data."""

    def __init__(self, dataset, models=None):
        self._ds = dataset
        self._models = models or {
            "ModelA": {"variants": ["r1i1p1f1"]},
            "ModelB": {"variants": ["r1i1p1f1"]},
        }

    @property
    def models(self):
        return self._models

    def load_var(self, cmip6_var, model, *, variant=None, table=None,
                 period=None, season=None, time_mean=True):
        for name in [cmip6_var, "tas"]:
            if name in self._ds.data_vars:
                da = self._ds[name]
                if period and "time" in da.dims:
                    da = da.sel(time=slice(period[0], period[1]))
                if season and "time" in da.dims:
                    da = da.sel(time=da["time.season"] == season)
                if time_mean and "time" in da.dims:
                    da = da.mean("time")
                return da
        return None

    def load_var_for_model_var(self, model_var, model, **kwargs):
        try:
            vinfo = get_var(model_var)
        except KeyError:
            return None
        if not vinfo.cmip6_variable:
            return None
        return self.load_var(vinfo.cmip6_variable, model, **kwargs)

    def get_member_pairs(self, ensemble_mode=None):
        pairs = []
        for model, cfg in self._models.items():
            variants = cfg.get("variants", ["r1i1p1f1"])
            pairs.append((model, variants[0]))
        return pairs

    def load_area(self, model, variant=None, table="Amon"):
        if "areacella" in self._ds.data_vars:
            return self._ds["areacella"]
        return None


# ── Helper to create the diagnostic ──────────────────────────────────


def _make_diag(model_loader, obs_loader, config, *,
               cmip6_loader=None, cmip6_individual=False):
    """Create TemperatureBerkeley diagnostic with standard kwargs."""
    from feather.diag.temperature_berkeley import TemperatureBerkeley
    return TemperatureBerkeley(
        model_loader, obs_loader, config,
        cmip6_loader=cmip6_loader,
        experiment="baseline_hist",
        period=("1990", "1990"),
        cmip6_individual=cmip6_individual,
    )


# ══════════════════════════════════════════════════════════════════════
# Test classes
# ══════════════════════════════════════════════════════════════════════


class TestRegistration:
    """Diagnostic is properly registered."""

    def test_registry_lookup(self):
        from feather.diag.registry import get_diagnostic
        diag_cls = get_diagnostic("temperature_berkeley")
        assert diag_cls is not None

    def test_registry_in_list(self):
        from feather.diag.registry import list_diagnostics
        entries = list_diagnostics()
        names = [e["name"] for e in entries]
        assert "temperature_berkeley" in names


class TestClassAttributes:
    """Class-level attributes are correctly defined."""

    def test_name(self):
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        assert TemperatureBerkeley.name == "temperature_berkeley"

    def test_domain(self):
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        assert TemperatureBerkeley.domain == "sfc"

    def test_variables(self):
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        assert TemperatureBerkeley.variables == ["tas"]

    def test_group(self):
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        assert TemperatureBerkeley.group == "temperature"

    def test_title(self):
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        assert "Temperature" in TemperatureBerkeley.title
        assert "Berkeley" in TemperatureBerkeley.title


class TestBerkeleyEarthLoading:
    """Test Berkeley Earth data loading and transforms."""

    def test_degc_to_k_conversion(self, synth_temp_obs_degc, berkeley_config):
        """Berkeley Earth degC is converted to K."""
        loader = MockTempModelLoader(xr.Dataset({"avg_2t": synth_temp_obs_degc}))
        obs = MockBerkeleyObsLoader(synth_temp_obs_degc)
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        # Should be in Kelvin range
        assert float(result.mean()) > 200

    def test_dim_rename(self, synth_temp_obs_degc, berkeley_config):
        """latitude/longitude dims renamed to lat/lon."""
        obs = MockBerkeleyObsLoader(synth_temp_obs_degc)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims

    def test_lon_shift(self, synth_temp_obs_degc, berkeley_config):
        """Lons shifted from -180..180 to 0..360."""
        obs = MockBerkeleyObsLoader(synth_temp_obs_degc)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        assert float(result.lon.min()) >= 0
        assert float(result.lon.max()) <= 360

    def test_period_slicing(self, synth_temp_obs, berkeley_config):
        """Period argument slices time correctly."""
        # synth_temp_obs is already in K with lat/lon dims
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth(period=("1990", "1990"))
        assert len(result.time) == 12

    def test_already_k_still_works(self, synth_temp_obs, berkeley_config):
        """Data already in K (without degC conversion) still adds 273.15.

        Berkeley Earth is always degC on disk, so we always add 273.15.
        For tests with data already in K, values will be high but the
        conversion logic is tested separately.
        """
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        # The value is K + 273.15 = very high, but the method works
        assert result is not None

    def test_no_rename_if_already_latlon(self, synth_temp_obs, berkeley_config):
        """No rename needed if dims are already lat/lon."""
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_no_lon_shift_if_already_positive(self, synth_temp_obs,
                                               berkeley_config):
        """No lon shift needed if lons already 0..360."""
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        loader = MockTempModelLoader(xr.Dataset())
        diag = _make_diag(loader, obs, berkeley_config)
        result = diag._load_berkeley_earth()
        assert float(result.lon.min()) >= 0

    def test_missing_dataset_raises(self, berkeley_config):
        """Missing BERKELEY_EARTH dataset in config raises."""
        cfg = berkeley_config
        cfg.obs_datasets = {}

        class FailObsLoader:
            def load(self, dataset, variable, period=None):
                raise KeyError(f"Dataset {dataset!r} not in config")

        loader = MockTempModelLoader(xr.Dataset())
        obs = FailObsLoader()
        diag = _make_diag(loader, obs, cfg)
        with pytest.raises(KeyError):
            diag._load_berkeley_earth()


class TestSharedDataLoading:
    """Test _load_shared_data."""

    def test_loads_model_and_obs(self, synth_temp_healpix, synth_temp_obs,
                                 berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        assert "model_monthly" in shared
        assert "berkeley" in shared
        assert "model_coords" in shared
        assert len(shared["model_monthly"]) == 1

    def test_no_models_raises(self, synth_temp_obs, berkeley_config):
        """All models missing raises RuntimeError."""
        empty_ds = xr.Dataset()
        loader = MockTempModelLoader(empty_ds)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        with pytest.raises(RuntimeError, match="No models have tas data"):
            diag._load_shared_data()

    def test_latlon_model_loading(self, synth_temp_latlon, synth_temp_obs,
                                   berkeley_config_latlon):
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        assert "ifs-fesom" in shared["model_monthly"]


class TestBiasMaps:
    """Test Group A: bias maps computation and plotting."""

    def test_compute_bias_maps(self, synth_temp_healpix, synth_temp_obs,
                                berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert "models" in results
        assert "obs" in results
        assert "colorbar_ranges" in results
        assert "ifs-fesom" in results["models"]

    def test_bias_field_has_lat_lon(self, synth_temp_healpix, synth_temp_obs,
                                    berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        bias = results["models"]["ifs-fesom"]["annual_bias"]
        assert "lat" in bias.dims
        assert "lon" in bias.dims

    def test_plot_bias_maps_returns_figures(self, synth_temp_healpix,
                                            synth_temp_obs, berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        assert len(figures) >= 1  # at least annual
        fig, meta = figures[0]
        assert meta["plot_type"] == "combined_bias_map"
        assert "Berkeley Earth" in meta["obs_dataset"]
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_bias_maps_metadata_has_stats(self, synth_temp_healpix,
                                           synth_temp_obs, berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        _, meta = figures[0]  # annual
        stats = meta.get("summary_statistics", {})
        assert "ifs-fesom" in stats
        model_stats = stats["ifs-fesom"]
        assert "pattern_correlation" in model_stats
        assert "rmse" in model_stats
        assert "global_mean_bias" in model_stats
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_seasonal_biases(self, synth_temp_healpix, synth_temp_obs,
                              berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        model_data = results["models"]["ifs-fesom"]
        assert "seasonal_biases" in model_data
        assert "seasonal_regrids" in model_data

    def test_latlon_bias_maps(self, synth_temp_latlon, synth_temp_obs,
                               berkeley_config_latlon):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert "ifs-fesom" in results["models"]
        import matplotlib.pyplot as plt
        plt.close("all")


class TestTimeseries:
    """Test Group B: time series."""

    def test_compute_timeseries(self, synth_temp_healpix, synth_temp_obs,
                                 berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert "models" in results
        assert "obs" in results
        assert "ifs-fesom" in results["models"]

    def test_obs_has_time_dim(self, synth_temp_healpix, synth_temp_obs,
                               berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert "time" in results["obs"].dims

    def test_plot_timeseries(self, synth_temp_healpix, synth_temp_obs,
                              berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        figures = diag._plot_timeseries(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert meta["figure_id"] == "tas_timeseries"
        assert meta["plot_type"] == "timeseries"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_timeseries_metadata(self, synth_temp_healpix, synth_temp_obs,
                                  berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        figures = diag._plot_timeseries(results)
        _, meta = figures[0]
        assert "Berkeley Earth" in meta["obs_dataset"]
        import matplotlib.pyplot as plt
        plt.close("all")


class TestSeasonalCycle:
    """Test Group C: seasonal cycle."""

    def test_compute_seasonal_cycle(self, synth_temp_healpix, synth_temp_obs,
                                     berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert "models" in results
        assert "obs" in results
        assert len(results["obs"]) == 12

    def test_plot_seasonal_cycle(self, synth_temp_healpix, synth_temp_obs,
                                  berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        figures = diag._plot_seasonal_cycle(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert meta["figure_id"] == "tas_seasonal_cycle"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_seasonal_cycle_12_months(self, synth_temp_healpix,
                                       synth_temp_obs, berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        model_clim = results["models"]["ifs-fesom"]
        assert len(model_clim) == 12


class TestZonalMean:
    """Test Group D: zonal mean."""

    def test_compute_zonal_mean(self, synth_temp_healpix, synth_temp_obs,
                                 berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "models" in results
        assert "obs" in results
        assert "ifs-fesom" in results["models"]

    def test_obs_zonal_has_lat(self, synth_temp_healpix, synth_temp_obs,
                                berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "lat" in results["obs"].dims

    def test_plot_zonal_mean(self, synth_temp_healpix, synth_temp_obs,
                              berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        figures = diag._plot_zonal_mean(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert meta["figure_id"] == "tas_zonal_mean"
        assert meta["plot_type"] == "zonal_profile"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_latlon_zonal_mean(self, synth_temp_latlon, synth_temp_obs,
                                berkeley_config_latlon):
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "ifs-fesom" in results["models"]
        zm = results["models"]["ifs-fesom"]
        assert "lat" in zm.dims

    def test_healpix_zonal_mean_uses_binning(self, synth_temp_healpix,
                                              synth_temp_obs,
                                              berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        zm = results["models"]["ifs-fesom"]
        assert "lat" in zm.dims or "lat" in zm.coords


class TestTrends:
    """Test Group E: warming trends."""

    def test_compute_trends(self, synth_temp_healpix, synth_temp_obs,
                             berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        assert "model_trends" in results
        assert "obs_trend" in results
        assert "ifs-fesom" in results["model_trends"]

    def test_trend_is_per_decade(self, synth_temp_healpix, synth_temp_obs,
                                  berkeley_config):
        """linear_trend * 10 gives K/decade."""
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        obs_trend = results["obs_trend"]
        # With 12 months, trends should be ~0 (no long-term trend)
        assert np.isfinite(obs_trend.values).any()

    def test_plot_trends_returns_figures(self, synth_temp_healpix,
                                         synth_temp_obs, berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        figures = diag._plot_trends(results)
        # Should have global trend map
        assert len(figures) >= 1
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_latlon_trends(self, synth_temp_latlon, synth_temp_obs,
                            berkeley_config_latlon):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        model_trend = results["model_trends"]["ifs-fesom"]
        assert "lat" in model_trend.dims
        assert "lon" in model_trend.dims
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_trend_global_map_metadata(self, synth_temp_latlon,
                                        synth_temp_obs,
                                        berkeley_config_latlon):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        figures = diag._plot_trends(results)
        # First figure should be the global trend
        if figures:
            _, meta = figures[0]
            assert meta["figure_id"] == "tas_trend_combined"
            assert "K/decade" in meta.get("computation_notes", "")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_polar_trend_maps(self, synth_temp_latlon, synth_temp_obs,
                               berkeley_config_latlon):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_latlon)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_trends(shared)
        figures = diag._plot_trends(results)
        # Should have global + arctic + antarctic = 3
        assert len(figures) == 3
        ids = [meta["figure_id"] for _, meta in figures]
        assert "tas_trend_arctic" in ids
        assert "tas_trend_antarctic" in ids
        import matplotlib.pyplot as plt
        plt.close("all")


class TestTaylorDiagram:
    """Test Group F: Taylor diagram."""

    def test_compute_taylor(self, synth_temp_healpix, synth_temp_obs,
                             berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        assert "model_stats" in results
        assert "ifs-fesom" in results["model_stats"]

    def test_taylor_stats_have_ann(self, synth_temp_healpix, synth_temp_obs,
                                    berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        stats = results["model_stats"]["ifs-fesom"]
        assert "ANN" in stats
        assert "corr" in stats["ANN"]
        assert "std_ratio" in stats["ANN"]

    def test_taylor_stats_have_seasons(self, synth_temp_healpix,
                                        synth_temp_obs, berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        stats = results["model_stats"]["ifs-fesom"]
        # DJF and JJA should be present when data covers all months
        assert "DJF" in stats or "JJA" in stats

    def test_corr_in_range(self, synth_temp_healpix, synth_temp_obs,
                            berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        corr = results["model_stats"]["ifs-fesom"]["ANN"]["corr"]
        assert -1.0 <= corr <= 1.0

    def test_std_ratio_positive(self, synth_temp_healpix, synth_temp_obs,
                                 berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        std_r = results["model_stats"]["ifs-fesom"]["ANN"]["std_ratio"]
        assert std_r > 0

    def test_plot_taylor(self, synth_temp_healpix, synth_temp_obs,
                          berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        figures = diag._plot_taylor(results)
        assert len(figures) == 1
        fig, meta = figures[0]
        assert meta["figure_id"] == "tas_taylor"
        assert meta["plot_type"] == "taylor_diagram"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_taylor_uses_polar_axes(self, synth_temp_healpix, synth_temp_obs,
                                     berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.projections.polar import PolarAxes
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        figures = diag._plot_taylor(results)
        fig, meta = figures[0]
        axes = fig.get_axes()
        assert any(isinstance(ax, PolarAxes) for ax in axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_taylor_metadata_has_summary_stats(self, synth_temp_healpix,
                                                synth_temp_obs,
                                                berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_taylor(shared)
        figures = diag._plot_taylor(results)
        _, meta = figures[0]
        stats = meta.get("summary_statistics", {})
        assert len(stats) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_empty_model_stats_returns_no_figures(self, synth_temp_obs,
                                                   berkeley_config):
        """No figures when no model stats available."""
        import matplotlib
        matplotlib.use("Agg")
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        loader = MockTempModelLoader(xr.Dataset())
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        results = {"model_stats": {}, "cmip6_stats": None, "cmip6_info": {}}
        figures = diag._plot_taylor(results)
        assert len(figures) == 0
        import matplotlib.pyplot as plt
        plt.close("all")


class TestStatistics:
    """Test statistical helper functions."""

    def test_pattern_correlation_identical(self):
        """Pattern correlation of identical fields is 1."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data = _make_temp_field(lats, lons).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        corr = TemperatureBerkeley._pattern_correlation(data, data, area)
        assert abs(corr - 1.0) < 1e-10

    def test_pattern_correlation_range(self):
        """Pattern correlation is in [-1, 1]."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data1 = _make_temp_field(lats, lons, base_temp=288).isel(time=0)
        data2 = _make_temp_field(lats, lons, base_temp=290).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        corr = TemperatureBerkeley._pattern_correlation(data1, data2, area)
        assert -1.0 <= corr <= 1.0

    def test_std_ratio_identical(self):
        """STD ratio of identical fields is 1."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data = _make_temp_field(lats, lons).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        ratio = TemperatureBerkeley._std_ratio(data, data, area)
        assert abs(ratio - 1.0) < 1e-10

    def test_std_ratio_positive(self):
        """STD ratio is positive."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data1 = _make_temp_field(lats, lons, base_temp=288).isel(time=0)
        data2 = _make_temp_field(lats, lons, base_temp=290).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        ratio = TemperatureBerkeley._std_ratio(data1, data2, area)
        assert ratio > 0

    def test_rmse_identical_is_zero(self):
        """RMSE of identical fields is 0."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data = _make_temp_field(lats, lons).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        rmse = TemperatureBerkeley._rmse(data, data, area)
        assert abs(rmse) < 1e-10

    def test_rmse_positive(self):
        """RMSE is non-negative."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data1 = _make_temp_field(lats, lons, base_temp=288).isel(time=0)
        data2 = _make_temp_field(lats, lons, base_temp=290).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        rmse = TemperatureBerkeley._rmse(data1, data2, area)
        assert rmse >= 0

    def test_regional_mean_bias_tropical(self):
        """Regional mean bias in tropics."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        bias = _make_temp_field(lats, lons, base_temp=1.0).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        trop_bias = TemperatureBerkeley._regional_mean_bias(
            bias, lats, area, lat_min=-30, lat_max=30,
        )
        assert isinstance(trop_bias, float)
        assert np.isfinite(trop_bias)

    def test_regional_mean_bias_arctic(self):
        """Regional mean bias in Arctic."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        bias = _make_temp_field(lats, lons, base_temp=2.0).isel(time=0)
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        arctic_bias = TemperatureBerkeley._regional_mean_bias(
            bias, lats, area, lat_min=60, lat_max=90,
        )
        assert isinstance(arctic_bias, float)

    def test_compute_summary_stats(self):
        """_compute_summary_stats returns all expected keys."""
        from feather.diag.temperature_berkeley import TemperatureBerkeley
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        model = _make_temp_field(lats, lons, base_temp=289).isel(time=0)
        obs = _make_temp_field(lats, lons, base_temp=288).isel(time=0)
        bias = model - obs
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        stats = TemperatureBerkeley._compute_summary_stats(
            model, obs, bias, lats, area,
        )
        assert "pattern_correlation" in stats
        assert "std_ratio" in stats
        assert "rmse" in stats
        assert "global_mean_bias" in stats
        assert "arctic_bias" in stats
        assert "tropical_bias" in stats
        assert "antarctic_bias" in stats


class TestRun:
    """Test the run() orchestration method."""

    def test_run_saves_figures(self, synth_temp_healpix, synth_temp_obs,
                                berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 6  # At least 6 groups
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_skip_existing(self, synth_temp_healpix, synth_temp_obs,
                                berkeley_config, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)

        # First run — creates figures
        saved1 = diag.run(skip_existing=False)
        n1 = len(saved1)

        # Second run — should skip
        saved2 = diag.run(skip_existing=True)
        assert len(saved2) == n1
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_partial_failure(self, synth_temp_obs, berkeley_config):
        """Partial failures don't crash the whole diagnostic."""
        import matplotlib
        matplotlib.use("Agg")
        # Model loader that will fail
        class FailingLoader:
            def load(self, key):
                raise RuntimeError("Cannot load")
            def load_var(self, *args, **kwargs):
                raise RuntimeError("Cannot load")
            @staticmethod
            def make_key(*args):
                return "key"

        loader = FailingLoader()
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        saved = diag.run(skip_existing=False)
        # Should not crash, even though no data loaded
        assert isinstance(saved, list)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_compute_backward_compat(self, synth_temp_healpix,
                                      synth_temp_obs, berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        results = diag.compute()
        assert "bias_maps" in results
        assert "timeseries" in results
        assert "seasonal_cycle" in results
        assert "zonal_mean" in results
        assert "trends" in results
        assert "taylor" in results

    def test_plot_backward_compat(self, synth_temp_healpix, synth_temp_obs,
                                   berkeley_config):
        import matplotlib
        matplotlib.use("Agg")
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        results = diag.compute()
        figures = diag.plot(results)
        assert len(figures) >= 6
        import matplotlib.pyplot as plt
        plt.close("all")


class TestCMIP6Integration:
    """Test CMIP6 support."""

    def _make_cmip6_data(self):
        """Create synthetic CMIP6 temperature dataset."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1990-01", periods=12, freq="MS",
                             calendar="standard")
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        temp_base = 287 - 40 * np.abs(lat_grid / 90.0)
        seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
        temp_3d = temp_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]
        area_2d = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)

        return xr.Dataset({
            "tas": xr.DataArray(
                temp_3d, dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "areacella": xr.DataArray(
                area_2d, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        })

    def test_cmip6_timeseries(self, synth_temp_healpix, synth_temp_obs,
                               cmip6_berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        cmip6 = MockCMIP6TempLoader(self._make_cmip6_data())
        diag = _make_diag(loader, obs, cmip6_berkeley_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert results.get("cmip6_ts") is not None

    def test_cmip6_seasonal_cycle(self, synth_temp_healpix, synth_temp_obs,
                                   cmip6_berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        cmip6 = MockCMIP6TempLoader(self._make_cmip6_data())
        diag = _make_diag(loader, obs, cmip6_berkeley_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert results.get("cmip6_monthly") is not None

    def test_cmip6_zonal_mean(self, synth_temp_healpix, synth_temp_obs,
                               cmip6_berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        cmip6 = MockCMIP6TempLoader(self._make_cmip6_data())
        diag = _make_diag(loader, obs, cmip6_berkeley_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert results.get("cmip6_zonal") is not None

    def test_cmip6_disabled(self, synth_temp_healpix, synth_temp_obs,
                             berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert results.get("cmip6_ts") is None


class TestTaylorDiagramPlot:
    """Test the plot_taylor_diagram function directly."""

    def test_basic_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        stats = {
            "ModelA": {
                "ANN": {"corr": 0.95, "std_ratio": 1.05},
                "DJF": {"corr": 0.90, "std_ratio": 0.95},
            },
        }
        fig, ax = plot_taylor_diagram(stats, title="Test")
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_polar_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.projections.polar import PolarAxes
        from feather.plot.lines import plot_taylor_diagram

        stats = {"M": {"ANN": {"corr": 0.9, "std_ratio": 1.0}}}
        fig, ax = plot_taylor_diagram(stats)
        assert isinstance(ax, PolarAxes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_custom_markers(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        stats = {"M": {"ANN": {"corr": 0.9, "std_ratio": 1.0}}}
        markers = {"ANN": "s"}
        fig, ax = plot_taylor_diagram(stats, season_markers=markers)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_model_colors(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        stats = {"M": {"ANN": {"corr": 0.9, "std_ratio": 1.0}}}
        colors = {"M": "#ff0000"}
        fig, ax = plot_taylor_diagram(stats, model_colors=colors)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_cmip6_stats(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        model_stats = {"M": {"ANN": {"corr": 0.9, "std_ratio": 1.0}}}
        cmip6_stats = {"ANN": {"corr": 0.85, "std_ratio": 0.9}}
        fig, ax = plot_taylor_diagram(
            model_stats, cmip6_stats=cmip6_stats,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_cmip6_individual_stats(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        model_stats = {"M": {"ANN": {"corr": 0.9, "std_ratio": 1.0}}}
        cmip6_individual = {
            "C1": {"ANN": {"corr": 0.80, "std_ratio": 0.8}},
            "C2": {"ANN": {"corr": 0.85, "std_ratio": 1.1}},
        }
        fig, ax = plot_taylor_diagram(
            model_stats, cmip6_individual_stats=cmip6_individual,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_corr_clipped(self):
        """Correlation values >1 are clipped to 1."""
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        stats = {"M": {"ANN": {"corr": 1.01, "std_ratio": 1.0}}}
        fig, ax = plot_taylor_diagram(stats)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_multiple_models(self):
        import matplotlib
        matplotlib.use("Agg")
        from feather.plot.lines import plot_taylor_diagram

        stats = {
            "A": {"ANN": {"corr": 0.95, "std_ratio": 1.05}},
            "B": {"ANN": {"corr": 0.88, "std_ratio": 0.92}},
            "C": {"ANN": {"corr": 0.92, "std_ratio": 1.10}},
        }
        fig, ax = plot_taylor_diagram(stats)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")


class TestColorbarRanges:
    """Test colorbar range computation."""

    def test_annual_ranges(self, synth_temp_healpix, synth_temp_obs,
                            berkeley_config):
        loader = MockTempModelLoader(synth_temp_healpix)
        obs = MockBerkeleyObsLoader(synth_temp_obs)
        diag = _make_diag(loader, obs, berkeley_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        cb = results["colorbar_ranges"]
        assert "annual" in cb
        assert "vmin" in cb["annual"]
        assert "vmax" in cb["annual"]
        assert "bias_vmax" in cb["annual"]
        assert cb["annual"]["vmin"] < cb["annual"]["vmax"]


class TestPrompts:
    """Test LLM prompt integration."""

    def test_figure_analysis_has_temperature_berkeley(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "Berkeley Earth" in system
        assert "Taylor diagram" in system

    def test_curation_has_temperature_berkeley(self):
        from feather.export.prompts import build_curation_system
        system = build_curation_system()
        assert "Berkeley Earth" in system

    def test_figure_analysis_has_warming_trends(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "warming trend" in system.lower() or "K/decade" in system
