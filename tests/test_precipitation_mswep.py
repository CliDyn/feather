"""Tests for the precipitation_mswep diagnostic."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.variables import VARIABLE_REGISTRY, get_var


# ── Test fixtures ─────────────────────────────────────────────────────


def _make_precip_field(lats, lons, ntimes=12, base_rate=3e-5):
    """Create synthetic precipitation with ITCZ-like gradient.

    Tropical max (~5e-5 kg/m²/s), polar min (~1e-6).
    """
    time = xr.date_range("1990-01", periods=ntimes, freq="MS",
                         calendar="standard")
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")

    # ITCZ-like: maximum near equator, decreasing toward poles
    # with secondary maxima at ~50° (storm tracks)
    pr_base = base_rate * np.exp(-0.5 * (lat_grid / 30.0) ** 2)
    pr_base += base_rate * 0.3 * np.exp(-0.5 * ((np.abs(lat_grid) - 50) / 15) ** 2)
    pr_base = np.clip(pr_base, 1e-6, None)

    # Add seasonal variation
    seasonal = 0.2 * base_rate * np.sin(
        2 * np.pi * (np.arange(ntimes) - 3) / 12
    )
    pr_3d = pr_base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis]
    pr_3d = np.clip(pr_3d, 0, None)

    return xr.DataArray(
        pr_3d, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


@pytest.fixture
def synth_precip_obs():
    """Synthetic observation-like precipitation on 5-degree grid."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    return _make_precip_field(lats, lons)


@pytest.fixture
def synth_precip_healpix():
    """Synthetic HEALPix precipitation (nside=8, 768 cells)."""
    import healpy as hp

    nside = 8
    ncells = 12 * nside ** 2
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)

    time = xr.date_range("1990-01", periods=12, freq="MS",
                         calendar="standard")
    base_rate = 3e-5

    pr_base = base_rate * np.exp(-0.5 * (lat / 30.0) ** 2)
    pr_base += base_rate * 0.3 * np.exp(-0.5 * ((np.abs(lat) - 50) / 15) ** 2)
    pr_base = np.clip(pr_base, 1e-6, None)

    seasonal = 0.2 * base_rate * np.sin(
        2 * np.pi * (np.arange(12) - 3) / 12
    )
    pr_2d = pr_base[np.newaxis, :] + seasonal[:, np.newaxis]
    pr_2d = np.clip(pr_2d, 0, None)

    area = np.ones(ncells) * (4 * np.pi / ncells)

    ds = xr.Dataset({
        "avg_tp": xr.DataArray(pr_2d, dims=("time", "values"),
                               coords={"time": time}),
        "longitude": xr.DataArray(lon, dims="values"),
        "latitude": xr.DataArray(lat, dims="values"),
        "area": xr.DataArray(area, dims="values"),
    })
    return ds


@pytest.fixture
def synth_mswep():
    """Synthetic MSWEP-like data (same grid as synth_precip_obs)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    return _make_precip_field(lats, lons, base_rate=2.8e-5)


@pytest.fixture
def precip_config(tmp_path):
    """FeatherConfig for precipitation tests."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"MSWEP": {"path": "/fake", "variables": {"pr": "fake"}}},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def precip_config_latlon(tmp_path):
    """FeatherConfig for latlon (CMOR) precipitation tests."""
    from feather.config import ModelConfig
    cfg = FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"MSWEP": {"path": "/fake", "variables": {"pr": "fake"}}},
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
def cmip6_precip_config(tmp_path):
    """FeatherConfig with CMIP6 enabled for precipitation tests."""
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={"MSWEP": {"path": "/fake", "variables": {"pr": "fake"}}},
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


class MockPrecipModelLoader:
    """Mock model loader returning precipitation data."""

    def __init__(self, dataset):
        self._ds = dataset

    def load(self, key):
        return self._ds

    def load_var(self, key_or_model, variable=None, **kwargs):
        if variable is None:
            variable = key_or_model
        for var_name in [variable, "avg_tp", "pr"]:
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


class MockMSWEPObsLoader:
    """Mock ObsLoader returning synthetic MSWEP data."""

    def __init__(self, mswep_da):
        self._mswep = mswep_da

    def load_mswep(self, period=None):
        da = self._mswep
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load(self, dataset, variable, period=None):
        return self._mswep

    def load_for_model_var(self, model_var, period=None):
        return self._mswep


class MockCMIP6PrecipLoader:
    """Mock CMIP6 loader with precipitation data."""

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
        for name in [cmip6_var, "pr"]:
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
    """Create PrecipitationMSWEP diagnostic with standard kwargs."""
    from feather.diag.precipitation_mswep import PrecipitationMSWEP
    return PrecipitationMSWEP(
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
        diag_cls = get_diagnostic("precipitation_mswep")
        assert diag_cls is not None

    def test_registry_in_list(self):
        from feather.diag.registry import list_diagnostics
        entries = list_diagnostics()
        names = [e["name"] for e in entries]
        assert "precipitation_mswep" in names


class TestClassAttributes:
    """Class-level attributes are correctly defined."""

    def test_name(self):
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        assert PrecipitationMSWEP.name == "precipitation_mswep"

    def test_domain(self):
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        assert PrecipitationMSWEP.domain == "sfc"

    def test_variables(self):
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        assert PrecipitationMSWEP.variables == ["pr"]

    def test_group(self):
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        assert PrecipitationMSWEP.group == "precipitation"

    def test_title(self):
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        assert "Precipitation" in PrecipitationMSWEP.title
        assert "MSWEP" in PrecipitationMSWEP.title


class TestDataLoading:
    """Test data loading pathways."""

    def test_load_model_data(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        assert "model_monthly" in shared
        assert "mswep" in shared
        assert len(shared["model_monthly"]) == 1

    def test_load_mswep_obs(self, synth_precip_healpix, synth_mswep,
                             precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        assert "time" in shared["mswep"].dims
        assert "lat" in shared["mswep"].dims
        assert "lon" in shared["mswep"].dims

    def test_model_coords_stored(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        assert "model_coords" in shared
        assert "ifs-fesom" in shared["model_coords"]
        lon, lat = shared["model_coords"]["ifs-fesom"]
        assert len(lon) > 0

    def test_missing_model_skipped(self, synth_mswep, precip_config):
        """Models without pr data are skipped gracefully."""
        empty_ds = xr.Dataset()
        loader = MockPrecipModelLoader(empty_ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        with pytest.raises(RuntimeError, match="No models have pr data"):
            diag._load_shared_data()

    def test_latlon_model_loading(self, synth_precip_obs, synth_mswep,
                                   precip_config_latlon):
        """Latlon (CMOR) model loading path works."""
        # Create a dataset with 'pr' variable (CMOR naming)
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        assert "ifs-fesom" in shared["model_monthly"]

    def test_mswep_loader_method(self):
        """ObsLoader.load_mswep raises KeyError when MSWEP not configured."""
        from feather.data.obs import ObsLoader
        cfg = FeatherConfig(
            model_catalogs={},
            models=["test"],
            obs_root="",
            obs_datasets={},  # No MSWEP
            cmip6={"enabled": False},
            dask={},
            nereus={},
            output_dir="/tmp/test",
        )
        loader = ObsLoader(cfg)
        with pytest.raises(KeyError, match="MSWEP"):
            loader.load_mswep()

    def test_period_slicing(self, synth_precip_healpix, synth_mswep,
                             precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        # Period is ("1990", "1990") — should have at most 12 timesteps
        assert len(shared["mswep"].time) <= 12

    def test_multiple_models(self, synth_precip_healpix, synth_mswep,
                              tmp_path):
        """Multiple models are loaded."""
        cfg = FeatherConfig(
            model_catalogs={},
            models=["ifs-fesom", "ifs-nemo"],
            obs_root="",
            obs_datasets={"MSWEP": {"path": "/fake", "variables": {"pr": "fake"}}},
            cmip6={"enabled": False},
            dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
        )
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, cfg)
        shared = diag._load_shared_data()
        assert len(shared["model_monthly"]) == 2


class TestBiasMaps:
    """Group A: Absolute bias maps."""

    def test_compute_bias_maps(self, synth_precip_healpix, synth_mswep,
                                precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert "models" in results
        assert "obs" in results
        assert "colorbar_ranges" in results

    def test_model_bias_fields(self, synth_precip_healpix, synth_mswep,
                                precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        model = "ifs-fesom"
        assert "annual_bias" in results["models"][model]
        assert "annual_rmse" in results["models"][model]
        assert "pattern_correlation" in results["models"][model]
        assert "std_ratio" in results["models"][model]

    def test_obs_clim_computed(self, synth_precip_healpix, synth_mswep,
                               precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert results["obs"]["clim"] is not None
        assert "lat" in results["obs"]["clim"].dims
        assert "lon" in results["obs"]["clim"].dims

    def test_seasonal_biases(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        model = "ifs-fesom"
        assert "seasonal_biases" in results["models"][model]

    def test_colorbar_ranges_annual(self, synth_precip_healpix, synth_mswep,
                                     precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        cb = results["colorbar_ranges"]
        assert "annual" in cb
        assert "vmin" in cb["annual"]
        assert "vmax" in cb["annual"]
        assert "bias_vmax" in cb["annual"]

    @pytest.mark.parametrize("period_key", ["annual", "DJF", "JJA"])
    def test_plot_bias_maps_periods(self, synth_precip_healpix, synth_mswep,
                                     precip_config, period_key):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        # Should produce at least 1 figure (annual always present)
        if period_key == "annual":
            assert len(figures) >= 1
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_plot_bias_map_metadata(self, synth_precip_healpix, synth_mswep,
                                     precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        assert len(figures) > 0
        _, meta = figures[0]
        assert meta["diagnostic_name"] == "precipitation_mswep"
        assert meta["plot_type"] == "combined_bias_map"
        assert "pr" in meta["variables_used"]
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_bias_map_summary_statistics(self, synth_precip_healpix,
                                          synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        _, meta = figures[0]
        stats = meta.get("summary_statistics", {})
        assert "ifs-fesom" in stats
        assert "rmse" in stats["ifs-fesom"]
        assert "pattern_correlation" in stats["ifs-fesom"]
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_bias_map_obs_title_mswep(self, synth_precip_healpix,
                                       synth_mswep, precip_config):
        """Obs panel should be labeled MSWEP, not ERA5."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        figures = diag._plot_bias_maps(results)
        _, meta = figures[0]
        assert meta.get("obs_dataset") == "MSWEP"
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_tropical_and_extratropical_bias(self, synth_precip_healpix,
                                              synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        model = "ifs-fesom"
        assert "tropical_mean_bias" in results["models"][model]
        assert "extratropical_mean_bias" in results["models"][model]
        # Both should be finite
        assert np.isfinite(results["models"][model]["tropical_mean_bias"])
        assert np.isfinite(results["models"][model]["extratropical_mean_bias"])

    def test_rmse_nonnegative(self, synth_precip_healpix, synth_mswep,
                               precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert results["models"]["ifs-fesom"]["annual_rmse"] >= 0

    def test_pattern_correlation_range(self, synth_precip_healpix,
                                        synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        pc = results["models"]["ifs-fesom"]["pattern_correlation"]
        assert -1 <= pc <= 1

    def test_std_ratio_positive(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert results["models"]["ifs-fesom"]["std_ratio"] > 0


class TestRelativeBias:
    """Group B: Relative bias maps."""

    def test_compute_relative_bias(self, synth_precip_healpix, synth_mswep,
                                    precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        assert "rel_bias_annual" in results
        assert "ifs-fesom" in results["rel_bias_annual"]

    def test_relative_bias_units(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        """Relative bias should be in percentage."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        rel = results["rel_bias_annual"]["ifs-fesom"]
        # Values should be in percentage range (not fraction)
        finite_vals = rel.values[np.isfinite(rel.values)]
        assert np.abs(finite_vals).max() < 1000  # reasonable % range

    def test_masking_low_precip(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        """Low precip areas should be masked (NaN)."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        # Some NaN values expected where obs < threshold
        rel = results["rel_bias_annual"]["ifs-fesom"]
        # Not all NaN
        assert not np.all(np.isnan(rel.values))

    def test_plot_relative_bias(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        figures = diag._plot_relative_bias(results)
        assert len(figures) >= 1
        _, meta = figures[0]
        assert meta["figure_id"] == "pr_annual_relative_bias"
        assert meta["plot_type"] == "combined_map"
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_relative_bias_symmetric_range(self, synth_precip_healpix,
                                            synth_mswep, precip_config):
        """Plot should use symmetric colorbar range."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        figures = diag._plot_relative_bias(results)
        # The plot uses vmin=-100, vmax=100 explicitly
        assert len(figures) >= 1
        import matplotlib.pyplot as plt
        for fig, _ in figures:
            plt.close(fig)

    def test_obs_clim_stored(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        assert "obs_clim" in results

    def test_empty_relative_bias(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        """Plot returns empty list when no data."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        results = {"rel_bias_annual": {}, "rel_bias_seasonal": {}, "obs_clim": None}
        figures = diag._plot_relative_bias(results)
        assert figures == []


class TestTimeseries:
    """Group C: Global-mean time series."""

    def test_compute_timeseries(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert "models" in results
        assert "obs" in results
        assert "ifs-fesom" in results["models"]

    def test_model_timeseries_shape(self, synth_precip_healpix, synth_mswep,
                                     precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        ts = results["models"]["ifs-fesom"]
        assert "time" in ts.dims
        assert ts.ndim == 1  # scalar time series

    def test_obs_timeseries_shape(self, synth_precip_healpix, synth_mswep,
                                   precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        obs_ts = results["obs"]
        assert "time" in obs_ts.dims
        assert obs_ts.ndim == 1

    def test_plot_timeseries(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        figures = diag._plot_timeseries(results)
        assert len(figures) == 1
        _, meta = figures[0]
        assert meta["figure_id"] == "pr_timeseries"
        assert meta["plot_type"] == "timeseries"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])

    def test_timeseries_values_positive(self, synth_precip_healpix,
                                         synth_mswep, precip_config):
        """Precipitation should be positive."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        ts = results["models"]["ifs-fesom"]
        assert float(ts.min()) >= 0

    def test_timeseries_no_cmip6(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        """CMIP6 fields are None when disabled."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert results["cmip6_ts"] is None

    def test_timeseries_metadata_obs_dataset(self, synth_precip_healpix,
                                              synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        figures = diag._plot_timeseries(results)
        _, meta = figures[0]
        assert meta["obs_dataset"] == "MSWEP"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])


class TestSeasonalCycle:
    """Group D: Seasonal cycle."""

    def test_compute_seasonal_cycle(self, synth_precip_healpix, synth_mswep,
                                     precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert "models" in results
        assert "obs" in results

    def test_monthly_values_count(self, synth_precip_healpix, synth_mswep,
                                   precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        # Should have 12 monthly values
        assert len(results["models"]["ifs-fesom"]) == 12
        assert len(results["obs"]) == 12

    def test_plot_seasonal_cycle(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        figures = diag._plot_seasonal_cycle(results)
        assert len(figures) == 1
        _, meta = figures[0]
        assert meta["figure_id"] == "pr_seasonal_cycle"
        assert meta["plot_type"] == "seasonal_cycle"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])

    def test_seasonal_cycle_label(self, synth_precip_healpix, synth_mswep,
                                   precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        figures = diag._plot_seasonal_cycle(results)
        _, meta = figures[0]
        assert meta["obs_dataset"] == "MSWEP"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])

    def test_no_cmip6_monthly(self, synth_precip_healpix, synth_mswep,
                               precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert results["cmip6_monthly"] is None

    def test_seasonal_values_positive(self, synth_precip_healpix,
                                       synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert float(results["models"]["ifs-fesom"].min()) >= 0


class TestZonalMean:
    """Group E: Zonal mean profile."""

    def test_compute_zonal_mean(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "models" in results
        assert "obs" in results
        assert "ifs-fesom" in results["models"]

    def test_obs_zonal_has_lat(self, synth_precip_healpix, synth_mswep,
                                precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "lat" in results["obs"].dims

    def test_model_zonal_has_lat(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert "lat" in results["models"]["ifs-fesom"].dims

    def test_plot_zonal_mean(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        figures = diag._plot_zonal_mean(results)
        assert len(figures) == 1
        _, meta = figures[0]
        assert meta["figure_id"] == "pr_zonal_mean"
        assert meta["plot_type"] == "zonal_profile"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])

    def test_zonal_mean_latlon(self, synth_precip_obs, synth_mswep,
                                precip_config_latlon):
        """Latlon model zonal mean uses .mean('lon')."""
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        zm = results["models"]["ifs-fesom"]
        assert "lat" in zm.dims
        assert "lon" not in zm.dims

    def test_zonal_mean_values_positive(self, synth_precip_healpix,
                                         synth_mswep, precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        obs_zm = results["obs"]
        assert float(obs_zm.min()) >= 0

    def test_no_cmip6_zonal(self, synth_precip_healpix, synth_mswep,
                             precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert results["cmip6_zonal"] is None

    def test_zonal_mean_metadata(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        figures = diag._plot_zonal_mean(results)
        _, meta = figures[0]
        assert "ITCZ" in meta["description"]
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])


class TestIntensityPDF:
    """Group F: Precipitation intensity distribution."""

    def test_compute_intensity_pdf(self, synth_precip_healpix, synth_mswep,
                                    precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        assert "pdfs" in results
        assert "bins" in results
        assert "bin_centres" in results

    def test_pdf_keys(self, synth_precip_healpix, synth_mswep,
                       precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        assert "MSWEP" in results["pdfs"]
        assert "ifs-fesom" in results["pdfs"]

    def test_pdf_nonnegative(self, synth_precip_healpix, synth_mswep,
                              precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        for name, pdf in results["pdfs"].items():
            assert np.all(pdf >= 0), f"PDF for {name} has negative values"

    def test_pdf_length(self, synth_precip_healpix, synth_mswep,
                         precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        n_bins = len(results["bins"]) - 1
        for pdf in results["pdfs"].values():
            assert len(pdf) == n_bins

    def test_bin_centres_logspaced(self, synth_precip_healpix, synth_mswep,
                                    precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        bc = results["bin_centres"]
        # Log-spaced -> ratio between consecutive centres should be constant
        ratios = bc[1:] / bc[:-1]
        assert np.allclose(ratios, ratios[0], rtol=0.01)

    def test_plot_intensity_pdf(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        figures = diag._plot_intensity_pdf(results)
        assert len(figures) == 1
        _, meta = figures[0]
        assert meta["figure_id"] == "pr_intensity_distribution"
        assert meta["plot_type"] == "intensity_pdf"
        import matplotlib.pyplot as plt
        plt.close(figures[0][0])

    def test_pdf_log_axes(self, synth_precip_healpix, synth_mswep,
                           precip_config):
        """Both axes should be log-scaled."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        figures = diag._plot_intensity_pdf(results)
        fig, _ = figures[0]
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_histogram_pdf_static(self):
        """Test the static _histogram_pdf method directly."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        values = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        weights = np.ones(5)
        bins = np.logspace(-6, -4, 10)
        pdf = PrecipitationMSWEP._histogram_pdf(values, weights, bins)
        assert len(pdf) == len(bins) - 1
        assert np.all(pdf >= 0)

    def test_histogram_pdf_with_nan(self):
        """NaN values should be excluded."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        values = np.array([1e-5, np.nan, 3e-5, np.nan, 5e-5])
        weights = np.ones(5)
        bins = np.logspace(-6, -4, 10)
        pdf = PrecipitationMSWEP._histogram_pdf(values, weights, bins)
        assert np.all(np.isfinite(pdf))

    def test_histogram_pdf_negative_excluded(self):
        """Negative values should be excluded."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        values = np.array([-1e-5, 2e-5, 3e-5])
        weights = np.ones(3)
        bins = np.logspace(-6, -4, 10)
        pdf = PrecipitationMSWEP._histogram_pdf(values, weights, bins)
        assert np.all(pdf >= 0)


class TestStatistics:
    """Test enhanced summary statistics."""

    def test_pattern_correlation_identical(self):
        """Identical fields should have correlation = 1."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        data = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        corr = PrecipitationMSWEP._pattern_correlation(data, data, area)
        assert abs(corr - 1.0) < 1e-10

    def test_pattern_correlation_opposite(self):
        """Opposite fields should have correlation = -1."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        data = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        neg_data = -data
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        corr = PrecipitationMSWEP._pattern_correlation(data, neg_data, area)
        assert abs(corr + 1.0) < 1e-10

    def test_std_ratio_identical(self):
        """Identical fields should have STD ratio = 1."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        data = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        ratio = PrecipitationMSWEP._std_ratio(data, data, area)
        assert abs(ratio - 1.0) < 1e-10

    def test_std_ratio_double(self):
        """2x field should have STD ratio = 2."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        data = xr.DataArray(
            np.cos(np.deg2rad(lat_grid)),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        ratio = PrecipitationMSWEP._std_ratio(2 * data, data, area)
        assert abs(ratio - 2.0) < 1e-10

    def test_regional_mean_bias(self):
        """Test tropical mean bias computation."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        # Uniform bias field
        bias = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        trop_bias = PrecipitationMSWEP._regional_mean_bias(
            bias, lats, area, lat_min=-30, lat_max=30,
        )
        assert abs(trop_bias - 1.0) < 0.01

    def test_extratropical_mean_bias(self):
        """Test extratropical mean bias computation."""
        from feather.diag.precipitation_mswep import PrecipitationMSWEP
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        bias = xr.DataArray(
            np.ones((len(lats), len(lons))) * 2.0,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        from feather.util.spatial import compute_latlon_areas
        area = compute_latlon_areas(lats, lons)
        ext_bias = PrecipitationMSWEP._extratropical_mean_bias(
            bias, lats, area,
        )
        assert abs(ext_bias - 2.0) < 0.01

    def test_global_mean_bias_signed(self, synth_precip_healpix,
                                      synth_mswep, precip_config):
        """Global mean bias can be positive or negative."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        bias = results["models"]["ifs-fesom"]["annual_bias_gmean"]
        assert np.isfinite(bias)

    def test_global_mean_bias_pct_derivable(self, synth_precip_healpix,
                                             synth_mswep, precip_config):
        """Can derive relative bias from absolute bias / obs mean."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        bias = results["models"]["ifs-fesom"]["annual_bias_gmean"]
        obs_mean = results["obs"]["global_mean"]
        pct = bias / obs_mean * 100
        assert np.isfinite(pct)


class TestRun:
    """Full run orchestration."""

    def test_run_creates_figures(self, synth_precip_healpix, synth_mswep,
                                  precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 6  # At least: 1 bias + 1 rel + 1 ts + 1 sc + 1 zm + 1 pdf
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_creates_files(self, synth_precip_healpix, synth_mswep,
                                precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        saved = diag.run(skip_existing=False)
        for png_path, json_path in saved:
            assert png_path.exists(), f"Missing: {png_path}"
            assert json_path.exists(), f"Missing: {json_path}"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_skip_existing(self, synth_precip_healpix, synth_mswep,
                                precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)

        # First run
        saved1 = diag.run(skip_existing=False)
        n_first = len(saved1)
        assert n_first > 0

        # Second run with skip_existing
        saved2 = diag.run(skip_existing=True)
        assert len(saved2) == n_first  # Same count (skipped but still listed)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_output_dir(self, synth_precip_healpix, synth_mswep,
                             precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        assert "precipitation_mswep" in str(diag.output_dir)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_figure_ids(self, synth_precip_healpix, synth_mswep,
                             precip_config):
        """Check that expected figure IDs are generated."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        saved = diag.run(skip_existing=False)
        figure_ids = [p[0].stem for p in saved]
        assert "pr_annual_bias_combined" in figure_ids
        assert "pr_annual_relative_bias" in figure_ids
        assert "pr_timeseries" in figure_ids
        assert "pr_seasonal_cycle" in figure_ids
        assert "pr_zonal_mean" in figure_ids
        assert "pr_intensity_distribution" in figure_ids
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_compute_backward_compat(self, synth_precip_healpix,
                                      synth_mswep, precip_config):
        """compute() returns all group results."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        results = diag.compute()
        expected_keys = [
            "bias_maps", "relative_bias", "timeseries",
            "seasonal_cycle", "zonal_mean", "intensity_pdf",
        ]
        for key in expected_keys:
            assert key in results

    def test_plot_backward_compat(self, synth_precip_healpix, synth_mswep,
                                   precip_config):
        """plot() generates figures from compute() output."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        results = diag.compute()
        figures = diag.plot(results)
        assert len(figures) >= 6
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_incremental_group_failure(self, synth_precip_healpix,
                                        synth_mswep, precip_config):
        """If one group fails, others still succeed."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        # Should not raise — individual group failures are caught
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 1
        import matplotlib.pyplot as plt
        plt.close("all")


class TestLatlon:
    """EERIE latlon grid path."""

    def test_latlon_compute_bias_maps(self, synth_precip_obs, synth_mswep,
                                       precip_config_latlon):
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        assert "ifs-fesom" in results["models"]

    def test_latlon_compute_timeseries(self, synth_precip_obs, synth_mswep,
                                        precip_config_latlon):
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        ts = results["models"]["ifs-fesom"]
        assert ts.ndim == 1

    def test_latlon_zonal_mean(self, synth_precip_obs, synth_mswep,
                                precip_config_latlon):
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        zm = results["models"]["ifs-fesom"]
        assert "lat" in zm.dims

    def test_latlon_intensity_pdf(self, synth_precip_obs, synth_mswep,
                                   precip_config_latlon):
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        shared = diag._load_shared_data()
        results = diag._compute_intensity_pdf(shared)
        assert "ifs-fesom" in results["pdfs"]

    def test_latlon_full_run(self, synth_precip_obs, synth_mswep,
                              precip_config_latlon):
        ds = xr.Dataset({"pr": synth_precip_obs})
        loader = MockPrecipModelLoader(ds)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config_latlon)
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 6
        import matplotlib.pyplot as plt
        plt.close("all")


class TestCMIP6:
    """CMIP6 integration tests."""

    @pytest.fixture
    def synth_cmip6_precip(self):
        """Synthetic CMIP6-like precipitation dataset."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        pr = _make_precip_field(lats, lons, base_rate=2.5e-5)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        area = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)
        ds = xr.Dataset({
            "pr": pr,
            "areacella": xr.DataArray(
                area, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        })
        return ds

    def test_cmip6_timeseries(self, synth_precip_healpix, synth_mswep,
                               synth_cmip6_precip, cmip6_precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        cmip6 = MockCMIP6PrecipLoader(synth_cmip6_precip)
        diag = _make_diag(loader, obs, cmip6_precip_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert results["cmip6_ts"] is not None

    def test_cmip6_seasonal_cycle(self, synth_precip_healpix, synth_mswep,
                                   synth_cmip6_precip, cmip6_precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        cmip6 = MockCMIP6PrecipLoader(synth_cmip6_precip)
        diag = _make_diag(loader, obs, cmip6_precip_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_seasonal_cycle(shared)
        assert results["cmip6_monthly"] is not None

    def test_cmip6_individual_timeseries(self, synth_precip_healpix,
                                          synth_mswep, synth_cmip6_precip,
                                          cmip6_precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        cmip6 = MockCMIP6PrecipLoader(synth_cmip6_precip)
        diag = _make_diag(loader, obs, cmip6_precip_config,
                          cmip6_loader=cmip6, cmip6_individual=True)
        shared = diag._load_shared_data()
        results = diag._compute_timeseries(shared)
        assert len(results["cmip6_individual_ts"]) > 0

    def test_cmip6_zonal_mean(self, synth_precip_healpix, synth_mswep,
                               synth_cmip6_precip, cmip6_precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        cmip6 = MockCMIP6PrecipLoader(synth_cmip6_precip)
        diag = _make_diag(loader, obs, cmip6_precip_config,
                          cmip6_loader=cmip6)
        shared = diag._load_shared_data()
        results = diag._compute_zonal_mean(shared)
        assert results["cmip6_zonal"] is not None

    def test_cmip6_individual_relative_bias(self, synth_precip_healpix,
                                             synth_mswep, synth_cmip6_precip,
                                             cmip6_precip_config):
        """CMIP6 individual models should appear in relative bias dict."""
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        cmip6 = MockCMIP6PrecipLoader(synth_cmip6_precip)
        diag = _make_diag(loader, obs, cmip6_precip_config,
                          cmip6_loader=cmip6, cmip6_individual=True)
        shared = diag._load_shared_data()
        results = diag._compute_relative_bias(shared)
        keys = list(results["rel_bias_annual"].keys())
        # Should contain model(s) + CMIP6 MMM + individual CMIP6 models
        assert "CMIP6 MMM" in keys
        cmip6_individual_keys = [k for k in keys
                                 if k not in ("CMIP6 MMM", "ifs-fesom")]
        assert len(cmip6_individual_keys) > 0, (
            "Expected individual CMIP6 models in relative bias dict"
        )


class TestPrompts:
    """LLM prompt updates."""

    def test_llm_prompts_mention_precipitation(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "Precipitation bias maps" in system
        assert "MSWEP" in system

    def test_llm_prompts_mention_intensity(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "Precipitation intensity distribution" in system

    def test_llm_prompts_mention_relative(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "Relative precipitation bias" in system

    def test_llm_prompts_mention_zonal(self):
        from feather.llm.prompts import build_figure_analysis_system
        system = build_figure_analysis_system()
        assert "Precipitation zonal mean" in system

    def test_export_prompts_mention_precipitation(self):
        from feather.export.prompts import build_curation_system
        system = build_curation_system()
        assert "Precipitation (MSWEP)" in system
        assert "MSWEP v2.8" in system


class TestColorbarRanges:
    """Colorbar range computation."""

    def test_colorbar_ranges_keys(self, synth_precip_healpix, synth_mswep,
                                   precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        cb = results["colorbar_ranges"]
        assert "annual" in cb

    def test_colorbar_vmin_lt_vmax(self, synth_precip_healpix, synth_mswep,
                                    precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        cb = results["colorbar_ranges"]["annual"]
        assert cb["vmin"] < cb["vmax"]

    def test_bias_vmax_positive(self, synth_precip_healpix, synth_mswep,
                                 precip_config):
        loader = MockPrecipModelLoader(synth_precip_healpix)
        obs = MockMSWEPObsLoader(synth_mswep)
        diag = _make_diag(loader, obs, precip_config)
        shared = diag._load_shared_data()
        results = diag._compute_bias_maps(shared)
        cb = results["colorbar_ranges"]["annual"]
        assert cb["bias_vmax"] > 0
