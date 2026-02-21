"""Tests for the radiation budget diagnostic."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.variables import VARIABLE_REGISTRY
from feather.diag.radiation_budget import RadiationBudget, _BUDGET_COMPONENTS
from feather.plot.lines import plot_budget_bars, plot_gregory
from tests.conftest import MockCMIP6Loader, MockModelLoader, MockObsLoader


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synth_radiation_healpix():
    """Synthetic HEALPix dataset with all radiation variables."""
    import healpy as hp

    nside = 8
    ncells = 12 * nside**2
    lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    # Temperature: 300K equator, 260K poles
    temp_base = 300 - 40 * np.abs(lat / 90.0)
    temp_2d = temp_base[np.newaxis, :] + seasonal[:, np.newaxis]

    # Radiation fields: realistic-ish global-mean values
    # TOA net SW: ~240 W/m² (positive down = absorbed solar)
    toa_sw = np.full_like(temp_base, 240.0)
    toa_sw_2d = toa_sw[np.newaxis, :] + seasonal[:, np.newaxis] * 2

    # TOA net LW: ~-240 W/m² (negative = outgoing)
    toa_lw = np.full_like(temp_base, -238.0)
    toa_lw_2d = toa_lw[np.newaxis, :] + seasonal[:, np.newaxis] * 1.5

    # Clear-sky versions (slightly different)
    toa_sw_cs_2d = toa_sw_2d + 50  # Clear sky absorbs more SW (no cloud reflection)
    toa_lw_cs_2d = toa_lw_2d - 30  # Clear sky emits more LW (no greenhouse from clouds)

    # Surface net radiation
    sfc_sw = np.full_like(temp_base, 170.0)
    sfc_sw_2d = sfc_sw[np.newaxis, :] + seasonal[:, np.newaxis] * 1.5
    sfc_lw = np.full_like(temp_base, -60.0)
    sfc_lw_2d = sfc_lw[np.newaxis, :] + seasonal[:, np.newaxis] * 0.5
    sfc_sw_cs_2d = sfc_sw_2d + 30
    sfc_lw_cs_2d = sfc_lw_2d - 15

    # Surface downwelling
    sd_sw = np.full_like(temp_base, 200.0)
    sd_sw_2d = sd_sw[np.newaxis, :] + seasonal[:, np.newaxis]
    sd_lw = np.full_like(temp_base, 340.0)
    sd_lw_2d = sd_lw[np.newaxis, :] + seasonal[:, np.newaxis]

    area = np.ones(ncells) * (4 * np.pi / ncells)

    ds = xr.Dataset(
        {
            "avg_2t": xr.DataArray(temp_2d, dims=("time", "values"), coords={"time": time}),
            "avg_tnswrf": xr.DataArray(toa_sw_2d, dims=("time", "values"), coords={"time": time}),
            "avg_tnlwrf": xr.DataArray(toa_lw_2d, dims=("time", "values"), coords={"time": time}),
            "avg_tnswrfcs": xr.DataArray(toa_sw_cs_2d, dims=("time", "values"), coords={"time": time}),
            "avg_tnlwrfcs": xr.DataArray(toa_lw_cs_2d, dims=("time", "values"), coords={"time": time}),
            "avg_snswrf": xr.DataArray(sfc_sw_2d, dims=("time", "values"), coords={"time": time}),
            "avg_snlwrf": xr.DataArray(sfc_lw_2d, dims=("time", "values"), coords={"time": time}),
            "avg_snswrfcs": xr.DataArray(sfc_sw_cs_2d, dims=("time", "values"), coords={"time": time}),
            "avg_snlwrfcs": xr.DataArray(sfc_lw_cs_2d, dims=("time", "values"), coords={"time": time}),
            "avg_sdswrf": xr.DataArray(sd_sw_2d, dims=("time", "values"), coords={"time": time}),
            "avg_sdlwrf": xr.DataArray(sd_lw_2d, dims=("time", "values"), coords={"time": time}),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
            "area": xr.DataArray(area, dims="values"),
        }
    )
    return ds


@pytest.fixture
def synth_ceres():
    """Synthetic CERES-like dataset on 5-degree lat/lon grid."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    def _make_field(base_val, seasonal_amp=2.0):
        base = np.full_like(lat_grid, base_val, dtype=float)
        return base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis] * seasonal_amp

    ds = xr.Dataset(
        {
            # TOA variables
            "toa_sw_all_mon": xr.DataArray(
                _make_field(240.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "toa_lw_all_mon": xr.DataArray(
                _make_field(-238.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "toa_net_all_mon": xr.DataArray(
                _make_field(2.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "toa_cre_sw_mon": xr.DataArray(
                _make_field(-50.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "toa_cre_lw_mon": xr.DataArray(
                _make_field(30.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            # Surface variables
            "sfc_net_sw_all_mon": xr.DataArray(
                _make_field(170.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "sfc_net_lw_all_mon": xr.DataArray(
                _make_field(-60.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "sfc_net_tot_all_mon": xr.DataArray(
                _make_field(110.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
        }
    )
    return ds


@pytest.fixture
def synth_obs_for_rad(synth_ceres):
    """Synthetic obs dataset for radiation (ERA5 T2m + CERES)."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
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


class MockObsLoaderWithCeres(MockObsLoader):
    """Mock ObsLoader that also supports load_ceres()."""

    def __init__(self, era5_ds, ceres_ds, var_name="t2m"):
        super().__init__(era5_ds, var_name)
        self._ceres = ceres_ds

    def load_ceres(self, ceres_var, period=None, file_key="toa"):
        if ceres_var not in self._ceres.data_vars:
            raise KeyError(f"{ceres_var} not in synthetic CERES data")
        da = self._ceres[ceres_var]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da


@pytest.fixture
def rad_model_loader(synth_radiation_healpix):
    return MockModelLoader(synth_radiation_healpix)


@pytest.fixture
def rad_obs_loader(synth_obs_for_rad, synth_ceres):
    return MockObsLoaderWithCeres(synth_obs_for_rad, synth_ceres)


@pytest.fixture
def rad_config(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def rad_cmip6_config(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
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


@pytest.fixture
def rad_cmip6_loader():
    """MockCMIP6Loader with radiation variables for budget testing."""
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=12, freq="MS", calendar="standard")

    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    seasonal = 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)

    def _make(val, amp=2.0):
        base = np.full_like(lat_grid, val, dtype=float)
        return base[np.newaxis, :, :] + seasonal[:, np.newaxis, np.newaxis] * amp

    area = np.cos(np.deg2rad(lat_grid)) * np.ones_like(lat_grid)

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                _make(288.0, 3.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rsdt": xr.DataArray(
                _make(340.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rsut": xr.DataArray(
                _make(100.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rlut": xr.DataArray(
                _make(238.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rsds": xr.DataArray(
                _make(200.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rsus": xr.DataArray(
                _make(30.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rlds": xr.DataArray(
                _make(340.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "rlus": xr.DataArray(
                _make(400.0), dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lats, "lon": lons},
            ),
            "areacella": xr.DataArray(
                area, dims=("lat", "lon"),
                coords={"lat": lats, "lon": lons},
            ),
        }
    )
    return MockCMIP6Loader(ds)


# ── Test Classes ─────────────────────────────────────────────────────


class TestCeresLoading:
    """Tests for CERES loading via MockObsLoaderWithCeres."""

    def test_load_toa_variable(self, rad_obs_loader):
        """load_ceres returns correct CERES TOA variable."""
        da = rad_obs_loader.load_ceres("toa_net_all_mon")
        assert "time" in da.dims
        assert "lat" in da.dims
        assert "lon" in da.dims

    def test_load_surface_variable(self, rad_obs_loader):
        """load_ceres returns correct CERES surface variable."""
        da = rad_obs_loader.load_ceres("sfc_net_tot_all_mon", file_key="surface")
        assert da is not None
        assert "time" in da.dims

    def test_load_ceres_missing_var(self, rad_obs_loader):
        """load_ceres raises KeyError for missing variable."""
        with pytest.raises(KeyError, match="nonexistent_var"):
            rad_obs_loader.load_ceres("nonexistent_var")

    def test_load_ceres_with_period(self, rad_obs_loader):
        """load_ceres slices time when period is given."""
        da_full = rad_obs_loader.load_ceres("toa_net_all_mon")
        da_sliced = rad_obs_loader.load_ceres(
            "toa_net_all_mon", period=("2000-03", "2000-06"),
        )
        assert len(da_sliced.time) <= len(da_full.time)


class TestBudgetComputation:
    """Tests for _compute_budget()."""

    def test_budget_structure(self, rad_model_loader, rad_obs_loader, rad_config):
        """Budget returns models, obs, cmip6 keys."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()
        assert "models" in results
        assert "obs" in results
        assert "cmip6" in results

    def test_model_budget_has_components(self, rad_model_loader, rad_obs_loader,
                                          rad_config):
        """Model budget has expected component names."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        mbud = results["models"]["ifs-fesom"]
        for comp_name, _, _, _, _ in _BUDGET_COMPONENTS:
            assert comp_name in mbud, f"Missing {comp_name}"

    def test_toa_net_is_sum_of_sw_lw(self, rad_model_loader, rad_obs_loader,
                                       rad_config):
        """TOA Net = TOA SW + TOA LW."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        mbud = results["models"]["ifs-fesom"]
        np.testing.assert_allclose(
            mbud["TOA Net"], mbud["TOA SW"] + mbud["TOA LW"], atol=0.01,
        )

    def test_sfc_net_is_sum(self, rad_model_loader, rad_obs_loader, rad_config):
        """Sfc Net = Sfc SW + Sfc LW."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        mbud = results["models"]["ifs-fesom"]
        np.testing.assert_allclose(
            mbud["Sfc Net"], mbud["Sfc SW"] + mbud["Sfc LW"], atol=0.01,
        )

    def test_atm_absorption(self, rad_model_loader, rad_obs_loader, rad_config):
        """Atm Abs = TOA Net - Sfc Net."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        mbud = results["models"]["ifs-fesom"]
        np.testing.assert_allclose(
            mbud["Atm Abs"], mbud["TOA Net"] - mbud["Sfc Net"], atol=0.01,
        )

    def test_obs_budget_from_ceres(self, rad_model_loader, rad_obs_loader,
                                     rad_config):
        """Obs budget is populated from CERES data."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        obs = results["obs"]
        assert len(obs) > 0
        assert "TOA Net" in obs
        assert "Sfc Net" in obs

    def test_budget_values_reasonable(self, rad_model_loader, rad_obs_loader,
                                       rad_config):
        """Budget values are in physically reasonable ranges."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()

        mbud = results["models"]["ifs-fesom"]
        assert 100 < mbud["TOA SW"] < 350  # ~240 W/m²
        assert -300 < mbud["TOA LW"] < 0   # ~-240 W/m²
        assert abs(mbud["TOA Net"]) < 50    # ~0-2 W/m²


class TestBudgetBarsPlot:
    """Tests for plot_budget_bars()."""

    def test_returns_fig_ax(self):
        """plot_budget_bars returns (fig, ax)."""
        data = {"TOA SW": {"Model": 240.0, "Obs": 241.0}}
        fig, ax = plot_budget_bars(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_data(self):
        """plot_budget_bars handles empty data."""
        fig, ax = plot_budget_bars({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_components(self):
        """Bars are created for multiple components."""
        data = {
            "TOA SW": {"A": 240, "B": 235},
            "TOA LW": {"A": -238, "B": -240},
        }
        fig, ax = plot_budget_bars(data)
        # Should have bars rendered
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_budget_plot_integration(self, rad_model_loader, rad_obs_loader,
                                      rad_config):
        """Full budget compute → plot pipeline."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()
        figs = diag._plot_budget(results)
        assert len(figs) == 1
        fig, meta = figs[0]
        assert meta["figure_id"] == "radiation_budget_bars"
        assert meta["plot_type"] == "budget_bars"
        plt.close(fig)


class TestGregoryComputation:
    """Tests for _compute_gregory()."""

    def test_gregory_structure(self, rad_model_loader, rad_obs_loader, rad_config):
        """Gregory returns models, obs, cmip6."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_gregory()
        assert "models" in results
        assert "obs" in results
        assert "cmip6" in results

    def test_model_has_monthly_and_annual(self, rad_model_loader, rad_obs_loader,
                                           rad_config):
        """Model data has monthly and annual time series."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_gregory()

        mdata = results["models"]["ifs-fesom"]
        assert "t2m_monthly" in mdata
        assert "toa_monthly" in mdata
        assert "t2m_annual" in mdata
        assert "toa_annual" in mdata
        assert len(mdata["t2m_monthly"]) == 12

    def test_obs_has_data(self, rad_model_loader, rad_obs_loader, rad_config):
        """Obs data is populated."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_gregory()

        obs = results["obs"]
        assert obs is not None
        assert "t2m_monthly" in obs
        assert "toa_monthly" in obs


class TestGregoryPlot:
    """Tests for plot_gregory()."""

    def test_returns_fig_ax(self):
        """plot_gregory returns (fig, ax)."""
        data = [{
            "label": "Test",
            "t2m_monthly": np.array([288, 289, 290, 291]),
            "toa_monthly": np.array([1.0, 0.5, 2.0, 1.5]),
        }]
        fig, ax = plot_gregory(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_sources(self):
        """Multiple scatter entries are plotted."""
        data = [
            {
                "label": "Model A",
                "t2m_monthly": np.array([288, 289, 290]),
                "toa_monthly": np.array([1.0, 0.5, 2.0]),
                "color": "blue",
            },
            {
                "label": "Model B",
                "t2m_monthly": np.array([287, 288, 289]),
                "toa_monthly": np.array([0.5, 1.0, 1.5]),
                "color": "red",
            },
        ]
        fig, ax = plot_gregory(data)
        # Should have scatter collections
        assert len(ax.collections) >= 2
        plt.close(fig)

    def test_regression_line(self):
        """Regression line appears in legend."""
        data = [{
            "label": "Test",
            "t2m_monthly": np.linspace(285, 295, 24),
            "toa_monthly": np.linspace(-1, 3, 24),
            "show_regression": True,
        }]
        fig, ax = plot_gregory(data)
        labels = [l.get_label() for l in ax.get_lines()]
        assert any("W/m" in lab for lab in labels)
        plt.close(fig)

    def test_annual_markers(self):
        """Annual mean markers are plotted."""
        data = [{
            "label": "Test",
            "t2m_monthly": np.array([288, 289, 290, 291]),
            "toa_monthly": np.array([1.0, 0.5, 2.0, 1.5]),
            "t2m_annual": np.array([289.5]),
            "toa_annual": np.array([1.25]),
        }]
        fig, ax = plot_gregory(data)
        # Should have scatter collections for both monthly and annual
        assert len(ax.collections) >= 2
        plt.close(fig)

    def test_gregory_plot_integration(self, rad_model_loader, rad_obs_loader,
                                       rad_config):
        """Full Gregory compute → plot pipeline."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_gregory()
        figs = diag._plot_gregory(results)
        assert len(figs) == 1
        fig, meta = figs[0]
        assert meta["figure_id"] == "gregory_plot"
        assert meta["plot_type"] == "gregory"
        plt.close(fig)


class TestImbalanceTimeseries:
    """Tests for radiation imbalance time series."""

    def test_imbalance_structure(self, rad_model_loader, rad_obs_loader,
                                  rad_config):
        """Imbalance returns models, obs, cmip6_ts."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_imbalance_timeseries()
        assert "models" in results
        assert "obs" in results
        assert "cmip6_ts" in results

    def test_model_imbalance_has_time(self, rad_model_loader, rad_obs_loader,
                                       rad_config):
        """Model imbalance time series has time dimension."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_imbalance_timeseries()

        ts = results["models"]["ifs-fesom"]
        assert "time" in ts.dims
        assert len(ts) == 12

    def test_obs_imbalance(self, rad_model_loader, rad_obs_loader, rad_config):
        """Obs imbalance time series is populated from CERES."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_imbalance_timeseries()

        assert results["obs"] is not None
        assert "time" in results["obs"].dims

    def test_imbalance_reasonable_values(self, rad_model_loader, rad_obs_loader,
                                          rad_config):
        """Net TOA imbalance is in a reasonable range."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_imbalance_timeseries()

        ts = results["models"]["ifs-fesom"]
        # Net TOA should be small (few W/m²) in equilibrium
        assert np.all(np.abs(ts.values) < 100)

    def test_imbalance_plot(self, rad_model_loader, rad_obs_loader, rad_config):
        """Imbalance plot produces figure with correct metadata."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_imbalance_timeseries()
        figs = diag._plot_imbalance_timeseries(results)
        assert len(figs) == 1
        fig, meta = figs[0]
        assert meta["figure_id"] == "radiation_imbalance_timeseries"
        assert meta["plot_type"] == "timeseries"
        plt.close(fig)


class TestBiasMaps:
    """Tests for derived-quantity bias maps."""

    def test_compute_bias_map_structure(self, rad_model_loader, rad_obs_loader,
                                         rad_config):
        """Bias map computation returns expected structure."""
        from feather.diag.radiation_budget import _DERIVED_QUANTITIES
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        dq_key = "toa_net"
        dq_info = _DERIVED_QUANTITIES[dq_key]
        results = diag._compute_bias_map(dq_key, dq_info)

        assert results is not None
        assert "models" in results
        assert "obs_clim" in results
        assert "ifs-fesom" in results["models"]
        assert "bias" in results["models"]["ifs-fesom"]

    def test_bias_map_has_stats(self, rad_model_loader, rad_obs_loader,
                                  rad_config):
        """Bias map results include global-mean bias and RMSE."""
        from feather.diag.radiation_budget import _DERIVED_QUANTITIES
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        results = diag._compute_bias_map("toa_net", _DERIVED_QUANTITIES["toa_net"])
        mdata = results["models"]["ifs-fesom"]
        assert "bias_gmean" in mdata
        assert "rmse" in mdata
        assert isinstance(mdata["bias_gmean"], float)

    def test_bias_map_plot(self, rad_model_loader, rad_obs_loader, rad_config):
        """Bias map plot produces figure with correct metadata."""
        from feather.diag.radiation_budget import _DERIVED_QUANTITIES
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        dq_key = "toa_net"
        dq_info = _DERIVED_QUANTITIES[dq_key]
        results = diag._compute_bias_map(dq_key, dq_info)
        figs = diag._plot_bias_map(dq_key, dq_info, results)

        assert len(figs) == 1
        fig, meta = figs[0]
        assert meta["figure_id"] == "toa_net_annual_bias"
        assert meta["plot_type"] == "combined_bias_map"
        assert meta["obs_dataset"] == "CERES_EBAF"
        plt.close(fig)

    def test_cre_bias_map(self, rad_model_loader, rad_obs_loader, rad_config):
        """CRE SW bias map computes correctly (subtraction operation)."""
        from feather.diag.radiation_budget import _DERIVED_QUANTITIES
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        results = diag._compute_bias_map("toa_cre_sw", _DERIVED_QUANTITIES["toa_cre_sw"])
        assert results is not None
        assert "ifs-fesom" in results["models"]
        plt.close("all")


class TestRunOrchestration:
    """Tests for run() orchestration."""

    def test_run_produces_figures(self, rad_model_loader, rad_obs_loader,
                                   rad_config):
        """run() produces multiple figure files."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        saved = diag.run(skip_existing=False)

        assert len(saved) >= 3  # bars + gregory + imbalance + bias maps
        for png, json_f in saved:
            assert png.exists()
            assert json_f.exists()
        plt.close("all")

    def test_run_skip_existing(self, rad_model_loader, rad_obs_loader,
                                rad_config):
        """run() skips figures that already exist."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        # First run creates files
        saved1 = diag.run(skip_existing=False)
        n_first = len(saved1)

        # Record file sizes
        sizes = {}
        for png, _ in saved1:
            sizes[png.name] = png.stat().st_size

        # Second run should skip
        saved2 = diag.run(skip_existing=True)
        assert len(saved2) == n_first

        # Verify files not overwritten (same size)
        for png, _ in saved2:
            if png.name in sizes:
                assert png.stat().st_size == sizes[png.name]
        plt.close("all")

    def test_run_no_skip(self, rad_model_loader, rad_obs_loader, rad_config):
        """run(skip_existing=False) regenerates all figures."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        saved1 = diag.run(skip_existing=False)
        # Tamper with a file
        png1 = saved1[0][0]
        png1.write_bytes(b"fake")

        saved2 = diag.run(skip_existing=False)
        # File should have been regenerated
        assert saved2[0][0].read_bytes() != b"fake"
        plt.close("all")


class TestSkipExistingPartial:
    """Tests for partial skip_existing behavior."""

    def test_skip_bars_only(self, rad_model_loader, rad_obs_loader, rad_config):
        """Pre-existing bars figure is skipped; others are computed."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)

        # Pre-create only the bars figure
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "radiation_budget_bars.png").write_bytes(b"fake")
        (diag.output_dir / "radiation_budget_bars.json").write_text("{}")

        saved = diag.run(skip_existing=True)
        # Bars should be in saved list with original content
        bars_png = diag.output_dir / "radiation_budget_bars.png"
        assert bars_png.read_bytes() == b"fake"
        # Other figures should also be in the list
        assert len(saved) >= 3
        plt.close("all")


class TestCMIP6Integration:
    """Tests for CMIP6 integration in radiation budget."""

    def test_cmip6_budget(self, rad_model_loader, rad_obs_loader,
                           rad_cmip6_config, rad_cmip6_loader):
        """CMIP6 budget is populated when enabled."""
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
        )
        results = diag._compute_budget()
        assert len(results["cmip6"]) > 0
        assert "TOA SW" in results["cmip6"]
        assert "TOA Net" in results["cmip6"]

    def test_cmip6_budget_values(self, rad_model_loader, rad_obs_loader,
                                   rad_cmip6_config, rad_cmip6_loader):
        """CMIP6 budget values are physically consistent."""
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
        )
        results = diag._compute_budget()
        cb = results["cmip6"]

        # Net TOA = TOA SW + TOA LW
        if "TOA Net" in cb and "TOA SW" in cb and "TOA LW" in cb:
            np.testing.assert_allclose(
                cb["TOA Net"], cb["TOA SW"] + cb["TOA LW"], atol=0.01,
            )

    def test_cmip6_disabled_empty(self, rad_model_loader, rad_obs_loader,
                                    rad_config):
        """CMIP6 budget is empty when disabled."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()
        assert results["cmip6"] == {}

    def test_cmip6_imbalance_timeseries(self, rad_model_loader, rad_obs_loader,
                                          rad_cmip6_config, rad_cmip6_loader):
        """CMIP6 MMM net TOA time series is populated."""
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
        )
        results = diag._compute_imbalance_timeseries()
        assert results["cmip6_ts"] is not None
        assert "time" in results["cmip6_ts"].dims

    def test_cmip6_gregory(self, rad_model_loader, rad_obs_loader,
                            rad_cmip6_config, rad_cmip6_loader):
        """CMIP6 MMM Gregory data is populated."""
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
        )
        results = diag._compute_gregory()
        assert results["cmip6"].get("mmm") is not None

    def test_cmip6_individual_gregory(self, rad_model_loader, rad_obs_loader,
                                        rad_cmip6_config, rad_cmip6_loader):
        """Individual CMIP6 models appear in Gregory data."""
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
            cmip6_individual=True,
        )
        results = diag._compute_gregory()
        indiv = results["cmip6"].get("individual", {})
        assert len(indiv) > 0

    def test_cmip6_bias_map(self, rad_model_loader, rad_obs_loader,
                              rad_cmip6_config, rad_cmip6_loader):
        """CMIP6 MMM bias panel appears in bias maps."""
        from feather.diag.radiation_budget import _DERIVED_QUANTITIES
        diag = RadiationBudget(
            rad_model_loader, rad_obs_loader, rad_cmip6_config,
            cmip6_loader=rad_cmip6_loader,
        )
        results = diag._compute_bias_map("toa_net", _DERIVED_QUANTITIES["toa_net"])
        if results is not None:
            assert "CMIP6 MMM" in results.get("cmip6_bias", {})
        plt.close("all")


class TestMissingData:
    """Tests for graceful handling of missing data."""

    def test_missing_ceres_graceful(self, rad_model_loader, rad_config):
        """Diagnostic handles missing CERES data gracefully."""
        # ObsLoader without CERES
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1990-01", periods=12, freq="MS")
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        temp = (300 - 40 * np.abs(lat_grid / 90.0))[np.newaxis, :, :] * np.ones((12, 1, 1))
        ds = xr.Dataset({
            "t2m": xr.DataArray(temp, dims=("time", "lat", "lon"),
                                coords={"time": time, "lat": lats, "lon": lons}),
        })
        obs = MockObsLoader(ds)

        diag = RadiationBudget(rad_model_loader, obs, rad_config)
        # Budget should still work for models (obs budget will be empty)
        results = diag._compute_budget()
        assert len(results["models"]) > 0
        assert results["obs"] == {}

    def test_missing_model_var_skipped(self, rad_obs_loader, rad_config):
        """Models missing a variable are skipped gracefully."""
        # Model loader with only avg_2t (no radiation vars)
        import healpy as hp
        nside = 8
        ncells = 12 * nside**2
        lon, lat = hp.pix2ang(nside, np.arange(ncells), nest=True, lonlat=True)
        time = xr.date_range("1990-01", periods=12, freq="MS")
        temp = (300 * np.ones(ncells))[np.newaxis, :] * np.ones((12, 1))
        ds = xr.Dataset({
            "avg_2t": xr.DataArray(temp, dims=("time", "values"), coords={"time": time}),
            "longitude": xr.DataArray(lon, dims="values"),
            "latitude": xr.DataArray(lat, dims="values"),
        })
        model_loader = MockModelLoader(ds)

        diag = RadiationBudget(model_loader, rad_obs_loader, rad_config)
        results = diag._compute_budget()
        # Should have partial budget (only model vars that exist)
        mbud = results["models"].get("ifs-fesom", {})
        assert "TOA SW" not in mbud  # Missing from model data


class TestRegistration:
    """Tests for diagnostic registration."""

    def test_registered(self):
        """RadiationBudget is registered in the diagnostic registry."""
        from feather.diag.registry import get_diagnostic
        cls = get_diagnostic("radiation_budget")
        assert cls is RadiationBudget

    def test_class_attributes(self):
        """Class attributes are correctly set."""
        assert RadiationBudget.name == "radiation_budget"
        assert RadiationBudget.title == "Radiation Budget"
        assert RadiationBudget.group == "radiation"
        assert RadiationBudget.domain == "sfc"
        assert len(RadiationBudget.variables) == 11

    def test_listed_in_diagnostics(self):
        """radiation_budget appears in list_diagnostics()."""
        from feather.diag.registry import list_diagnostics
        names = [d["name"] for d in list_diagnostics()]
        assert "radiation_budget" in names


class TestComputePlotInterface:
    """Tests for compute() → plot() backward-compat interface."""

    def test_compute_returns_dict(self, rad_model_loader, rad_obs_loader,
                                   rad_config):
        """compute() returns dict with budget, gregory, imbalance."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag.compute()
        assert "budget" in results
        assert "gregory" in results
        assert "imbalance" in results

    def test_plot_returns_figures(self, rad_model_loader, rad_obs_loader,
                                  rad_config):
        """plot() returns list of (fig, meta) pairs."""
        diag = RadiationBudget(rad_model_loader, rad_obs_loader, rad_config)
        results = diag.compute()
        figures = diag.plot(results)
        assert len(figures) >= 3  # bars + gregory + imbalance
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert isinstance(meta, dict)
            assert "figure_id" in meta
        plt.close("all")
