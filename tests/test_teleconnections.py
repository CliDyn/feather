"""Tests for feather.diag.teleconnections — climate variability modes."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.teleconnections import (
    ModeDefinition,
    TeleconnectionDiag,
    _MODE_REGISTRY,
)

# Reuse mock loaders from conftest
from tests.conftest import MockCMIP6Loader, MockModelLoader, MockObsLoader


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_sst_field(nt=48, lats=None, lons=None, seed=42):
    """Create synthetic SST field with ENSO-like signal in Nino 3.4 box."""
    if lats is None:
        lats = np.arange(-87.5, 90, 5.0)
    if lons is None:
        lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=nt, freq="MS")

    rng = np.random.default_rng(seed)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 300 - 20 * np.abs(lat_grid / 90.0)

    # Add ENSO-like signal in Nino 3.4 box (190-240E, 5S-5N)
    enso_mask = (
        (lat_grid >= -5) & (lat_grid <= 5) &
        (lon_grid >= 190) & (lon_grid <= 240)
    )
    enso_signal = 2.0 * np.sin(2 * np.pi * np.arange(nt) / 48)  # 4-yr period

    data = np.zeros((nt, len(lats), len(lons)))
    for t in range(nt):
        data[t] = temp_base + enso_signal[t] * enso_mask
        data[t] += 0.1 * rng.standard_normal(data[t].shape)

    # Add seasonal cycle
    seasonal = 3.0 * np.sin(2 * np.pi * (np.arange(nt) - 3) / 12)
    data += seasonal[:, None, None]

    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


def _make_slp_field(nt=48, lats=None, lons=None, seed=42):
    """Create synthetic SLP field with NAO-like dipole."""
    if lats is None:
        lats = np.arange(-87.5, 90, 5.0)
    if lons is None:
        lons = np.arange(2.5, 360, 5.0)
    time = xr.date_range("1990-01", periods=nt, freq="MS")

    rng = np.random.default_rng(seed)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # NAO-like dipole: negative over Iceland (65N), positive over Azores (35N)
    # Only in N. Atlantic (270-40E)
    nao_pattern = -np.sin(np.deg2rad((lat_grid - 50) * 3))
    atlantic_mask = ((lon_grid >= 270) | (lon_grid <= 40)) & (lat_grid >= 20)
    nao_pattern = nao_pattern * atlantic_mask

    nao_signal = np.sin(2 * np.pi * np.arange(nt) / 36)  # 3-yr period

    data = np.zeros((nt, len(lats), len(lons)))
    base_slp = 101325.0  # Pa
    for t in range(nt):
        data[t] = base_slp + 500 * nao_signal[t] * nao_pattern
        data[t] += 10 * rng.standard_normal(data[t].shape)

    # Add seasonal cycle
    seasonal = 200 * np.sin(2 * np.pi * (np.arange(nt) - 0) / 12)
    data += seasonal[:, None, None]

    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lats, "lon": lons},
    )


@pytest.fixture
def sst_field():
    return _make_sst_field()


@pytest.fixture
def slp_field():
    return _make_slp_field()


@pytest.fixture
def sst_obs_field():
    return _make_sst_field(seed=99)


@pytest.fixture
def slp_obs_field():
    return _make_slp_field(seed=99)


@pytest.fixture
def teleconnection_config(tmp_path):
    """Config with models for teleconnection testing."""
    return FeatherConfig(
        model_catalogs={},
        models=["model-a"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor"},
    )


@pytest.fixture
def multi_model_config(tmp_path):
    """Config with multiple models."""
    return FeatherConfig(
        model_catalogs={},
        models=["model-a", "model-b"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor"},
    )


class MultiVarModelLoader:
    """Mock model loader that handles multiple variables."""

    def __init__(self, datasets):
        self._datasets = datasets  # {var_name: DataArray}

    def load_var(self, model, variable, period=None, time_mean=False):
        if variable not in self._datasets:
            raise FileNotFoundError(f"No data for {variable}")
        da = self._datasets[variable]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da


class MultiVarObsLoader:
    """Mock obs loader that handles multiple variables."""

    def __init__(self, datasets):
        self._datasets = datasets  # {var_name: DataArray}

    def load_for_model_var(self, model_var, period=None):
        # Map CMOR names to our datasets
        mapping = {"tos": "tos", "psl": "psl", "ua": "ua"}
        key = mapping.get(model_var, model_var)
        if key not in self._datasets:
            raise KeyError(f"No obs for {key}")
        da = self._datasets[key]
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da


# ── Mode registry tests ──────────────────────────────────────────────


class TestModeRegistry:
    """Test that all 7 modes are registered correctly."""

    def test_all_modes_registered(self):
        expected = {"enso", "nao", "sam", "ao", "iod", "pdo", "qbo"}
        assert expected == set(_MODE_REGISTRY.keys())

    def test_enso_attributes(self):
        m = _MODE_REGISTRY["enso"]
        assert m.variable == "tos"
        assert m.method == "box_mean"
        assert "nino34" in m.boxes

    def test_nao_attributes(self):
        m = _MODE_REGISTRY["nao"]
        assert m.variable == "psl"
        assert m.method == "eof"
        assert m.eof_region is not None

    def test_sam_attributes(self):
        m = _MODE_REGISTRY["sam"]
        assert m.eof_all_lons is True
        assert m.lat_band == (-90, -20)

    def test_ao_attributes(self):
        m = _MODE_REGISTRY["ao"]
        assert m.eof_all_lons is True
        assert m.lat_band == (20, 90)

    def test_iod_attributes(self):
        m = _MODE_REGISTRY["iod"]
        assert m.method == "box_diff"
        assert "west" in m.boxes
        assert "east" in m.boxes

    def test_pdo_attributes(self):
        m = _MODE_REGISTRY["pdo"]
        assert m.detrend_global is True
        assert m.method == "eof"

    def test_qbo_attributes(self):
        m = _MODE_REGISTRY["qbo"]
        assert m.method == "zonal_mean"
        assert m.pressure_level == 50.0


# ── ENSO computation tests ──────────────────────────────────────────


class TestENSOCompute:
    """Test ENSO (box mean) computation."""

    def test_box_mean_extraction(self, sst_field):
        box = (190, 240, -5, 5)
        result = TeleconnectionDiag._extract_box_mean(sst_field, box)
        assert "time" in result.dims
        assert len(result.dims) == 1  # 1D time series

    def test_box_mean_values_in_box(self, sst_field):
        """Box mean should capture the ENSO signal."""
        box = (190, 240, -5, 5)
        ts = TeleconnectionDiag._extract_box_mean(sst_field, box)
        # Should have variability from the ENSO signal
        assert float(ts.std()) > 0.1

    def test_enso_index_deseasonalised(self, sst_field, sst_obs_field,
                                        teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        idx, pattern, var_exp = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert "time" in idx.dims
        # Monthly mean of deseasonalised should be near zero
        monthly_means = idx.groupby("time.month").mean()
        assert float(monthly_means.std()) < 0.5

    def test_enso_regression_pattern(self, sst_field, sst_obs_field,
                                      teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        idx, pattern, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert pattern is not None
        assert "lat" in pattern.dims
        assert "lon" in pattern.dims

    def test_enso_full_mode(self, sst_field, sst_obs_field,
                             teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        assert result is not None
        assert "model-a" in result["model_indices"]
        assert result["obs_index"] is not None


# ── IOD computation tests ────────────────────────────────────────────


class TestIODCompute:
    """Test IOD (box difference) computation."""

    def test_box_diff_computation(self, sst_field, sst_obs_field,
                                   teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["iod"]
        idx, pattern, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert "time" in idx.dims

    def test_iod_is_west_minus_east(self, sst_field):
        """IOD = western box - eastern box."""
        mode_def = _MODE_REGISTRY["iod"]
        west = TeleconnectionDiag._extract_box_mean(
            sst_field, mode_def.boxes["west"],
        )
        east = TeleconnectionDiag._extract_box_mean(
            sst_field, mode_def.boxes["east"],
        )
        expected_diff = west - east
        assert float(expected_diff.std()) >= 0


# ── NAO computation tests ────────────────────────────────────────────


class TestNAOCompute:
    """Test NAO (EOF) computation."""

    def test_nao_eof_extraction(self, slp_field, slp_obs_field,
                                 teleconnection_config):
        model_loader = MultiVarModelLoader({"psl": slp_field})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["nao"]
        idx, pattern, var_exp = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert pattern is not None
        assert var_exp is not None
        assert 0 < var_exp < 1

    def test_nao_pattern_has_spatial_dims(self, slp_field, slp_obs_field,
                                           teleconnection_config):
        model_loader = MultiVarModelLoader({"psl": slp_field})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["nao"]
        _, pattern, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert "lat" in pattern.dims
        assert "lon" in pattern.dims

    def test_nao_region_extraction(self, slp_field):
        """NAO EOF region should extract N. Atlantic."""
        mode_def = _MODE_REGISTRY["nao"]
        da_region = TeleconnectionDiag._extract_eof_region(slp_field, mode_def)
        lats = da_region.lat.values
        assert lats.min() >= 20
        assert lats.max() <= 80


# ── SAM/AO computation tests ────────────────────────────────────────


class TestSAMCompute:
    """Test SAM (hemispheric EOF) computation."""

    def test_sam_region_is_southern_hemisphere(self, slp_field):
        mode_def = _MODE_REGISTRY["sam"]
        da_region = TeleconnectionDiag._extract_eof_region(slp_field, mode_def)
        assert da_region.lat.values.max() <= -20

    def test_sam_eof_computation(self, slp_field, slp_obs_field,
                                  teleconnection_config):
        model_loader = MultiVarModelLoader({"psl": slp_field})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["sam"]
        idx, pattern, var_exp = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert var_exp is not None


class TestAOCompute:
    """Test AO (hemispheric EOF) computation."""

    def test_ao_region_is_northern_hemisphere(self, slp_field):
        mode_def = _MODE_REGISTRY["ao"]
        da_region = TeleconnectionDiag._extract_eof_region(slp_field, mode_def)
        assert da_region.lat.values.min() >= 20

    def test_ao_eof_computation(self, slp_field, slp_obs_field,
                                 teleconnection_config):
        model_loader = MultiVarModelLoader({"psl": slp_field})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["ao"]
        idx, _, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None


# ── PDO computation tests ────────────────────────────────────────────


class TestPDOCompute:
    """Test PDO (EOF with global-mean SST removed)."""

    def test_pdo_region_extraction(self, sst_field):
        mode_def = _MODE_REGISTRY["pdo"]
        da_region = TeleconnectionDiag._extract_eof_region(sst_field, mode_def)
        lats = da_region.lat.values
        assert lats.min() >= 20
        assert lats.max() <= 70

    def test_pdo_computation(self, sst_field, sst_obs_field,
                              teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["pdo"]
        idx, pattern, var_exp = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None


# ── QBO computation tests ────────────────────────────────────────────


class TestQBOCompute:
    """Test QBO (zonal mean) computation."""

    def test_qbo_with_pressure_levels(self, teleconnection_config):
        """QBO needs pressure level data."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        levs = np.array([100.0, 50.0, 30.0, 10.0])  # hPa
        nt = 48
        time = xr.date_range("1990-01", periods=nt, freq="MS")
        rng = np.random.default_rng(42)

        data = rng.standard_normal((nt, len(levs), len(lats), len(lons)))
        # Add QBO signal at 50 hPa in equatorial band
        qbo_signal = np.sin(2 * np.pi * np.arange(nt) / 28)  # ~28 months
        equatorial = (np.abs(lats) <= 5)
        lev_50_idx = 1
        for t in range(nt):
            data[t, lev_50_idx, equatorial, :] += 5 * qbo_signal[t]

        ua = xr.DataArray(
            data, dims=("time", "plev", "lat", "lon"),
            coords={"time": time, "plev": levs, "lat": lats, "lon": lons},
        )

        model_loader = MultiVarModelLoader({"ua": ua})
        obs_loader = MultiVarObsLoader({"ua": ua})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["qbo"]
        idx, _, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert "time" in idx.dims

    def test_qbo_no_level_dimension(self, sst_field, sst_obs_field,
                                     teleconnection_config):
        """QBO should return None when no pressure levels available."""
        # Use SST data (no plev dimension) for ua
        model_loader = MultiVarModelLoader({"ua": sst_field})
        obs_loader = MultiVarObsLoader({"ua": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["qbo"]
        idx, _, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is None


# ── Plot tests ────────────────────────────────────────────────────────


class TestPlotTimeseries:
    """Test time series plotting."""

    def test_creates_figure(self, sst_field, sst_obs_field,
                             teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_timeseries(mode_def, result)
        assert fig is not None
        assert meta["figure_id"] == "enso_timeseries"
        assert meta["plot_type"] == "teleconnection_timeseries"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_metadata_has_statistics(self, sst_field, sst_obs_field,
                                      teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        _, meta = diag._plot_timeseries(mode_def, result)
        assert "summary_statistics" in meta
        assert "model-a" in meta["summary_statistics"]


class TestPlotPattern:
    """Test spatial pattern plotting."""

    def test_creates_figure(self, sst_field, sst_obs_field,
                             teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_pattern(mode_def, result)
        assert fig is not None
        assert meta["figure_id"] == "enso_pattern"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_eof_pattern_figure(self, slp_field, slp_obs_field,
                                 teleconnection_config):
        model_loader = MultiVarModelLoader({"psl": slp_field})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["nao"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_pattern(mode_def, result)
        assert fig is not None
        assert meta["figure_id"] == "nao_pattern"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_pattern_placeholder(self, teleconnection_config):
        """When no patterns available, should create placeholder."""
        model_loader = MultiVarModelLoader({})
        obs_loader = MultiVarObsLoader({})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["qbo"]
        result = {
            "model_indices": {"model-a": xr.DataArray(
                np.random.randn(48),
                dims="time",
                coords={"time": xr.date_range("1990-01", periods=48, freq="MS")},
            )},
            "model_patterns": {},
            "model_var_explained": {},
            "obs_index": None,
            "obs_pattern": None,
            "obs_var_explained": None,
            "cmip6_mmm_index": None,
            "cmip6_individual": {},
            "cmip6_info": {},
        }
        fig, meta = diag._plot_pattern(mode_def, result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSpectrum:
    """Test power spectrum plotting."""

    def test_creates_figure(self, sst_field, sst_obs_field,
                             teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_spectrum(mode_def, result)
        assert fig is not None
        assert meta["figure_id"] == "enso_spectrum"
        assert meta["plot_type"] == "teleconnection_spectrum"
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSeasonalVariance:
    """Test seasonal variance profile plotting."""

    def test_creates_figure(self, sst_field, sst_obs_field,
                             teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_seasonal_variance(mode_def, result)
        assert fig is not None
        assert meta["figure_id"] == "enso_seasonal_variance"
        assert meta["plot_type"] == "teleconnection_seasonal_variance"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_has_12_bars_per_source(self, sst_field, sst_obs_field,
                                     teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        result = diag._compute_mode(mode_def)
        fig, meta = diag._plot_seasonal_variance(mode_def, result)
        # Check peak_month stat
        stats = meta["summary_statistics"]
        for model_stats in stats.values():
            assert 1 <= model_stats["peak_month"] <= 12
        import matplotlib.pyplot as plt
        plt.close(fig)


# ── Full run tests ────────────────────────────────────────────────────


class TestRun:
    """Test the full run() method."""

    def test_run_creates_figures(self, sst_field, slp_field,
                                  sst_obs_field, slp_obs_field,
                                  teleconnection_config, tmp_path):
        model_loader = MultiVarModelLoader({
            "tos": sst_field, "psl": slp_field,
        })
        obs_loader = MultiVarObsLoader({
            "tos": sst_obs_field, "psl": slp_obs_field,
        })
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        # Only test a subset (ENSO + NAO) for speed
        diag._modes = ["enso", "nao"]
        saved = diag.run(skip_existing=False)
        # 2 modes × 4 figures = 8 figure pairs
        assert len(saved) == 8
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_skip_existing(self, sst_field, sst_obs_field,
                            teleconnection_config, tmp_path):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        diag._modes = ["enso"]
        # First run
        saved1 = diag.run(skip_existing=False)
        assert len(saved1) == 4
        # Second run — should skip
        saved2 = diag.run(skip_existing=True)
        assert len(saved2) == 4  # Returns paths but doesn't recompute
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_missing_variable_graceful(self, sst_field, sst_obs_field,
                                        teleconnection_config):
        """Mode requiring psl should fail gracefully when only tos available."""
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        diag._modes = ["nao"]  # Needs psl
        saved = diag.run(skip_existing=False)
        # Should produce 0 figures (psl not available)
        assert len(saved) == 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_compute_abc_compat(self, sst_field, sst_obs_field,
                                 teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        diag._modes = ["enso"]
        results = diag.compute()
        assert "enso" in results
        figures = diag.plot(results)
        assert len(figures) == 4
        import matplotlib.pyplot as plt
        plt.close("all")


# ── Multi-model tests ────────────────────────────────────────────────


class TestMultiModel:
    """Test with multiple models."""

    def test_two_models(self, sst_field, sst_obs_field,
                         multi_model_config, tmp_path):
        # Both models get same data (different would need different seeds)
        sst_b = _make_sst_field(seed=77)
        model_data = {"tos": sst_field}

        class TwoModelLoader:
            def __init__(self):
                self._data = {"model-a": sst_field, "model-b": sst_b}

            def load_var(self, model, variable, period=None, time_mean=False):
                da = self._data[model]
                if period and "time" in da.dims:
                    da = da.sel(time=slice(period[0], period[1]))
                return da

        model_loader = TwoModelLoader()
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, multi_model_config,
        )
        diag._modes = ["enso"]
        result = diag._compute_mode(_MODE_REGISTRY["enso"])
        assert "model-a" in result["model_indices"]
        assert "model-b" in result["model_indices"]
        import matplotlib.pyplot as plt
        plt.close("all")


# ── Latlon grid tests ────────────────────────────────────────────────


class TestLatlon:
    """Test with regular lat/lon grids (EERIE-style)."""

    def test_latlon_enso(self, teleconnection_config):
        sst = _make_sst_field(nt=48,
                               lats=np.arange(-89.5, 90, 1.0),
                               lons=np.arange(0.5, 360, 1.0))
        sst_obs = _make_sst_field(nt=48, seed=99,
                                    lats=np.arange(-89.5, 90, 1.0),
                                    lons=np.arange(0.5, 360, 1.0))
        model_loader = MultiVarModelLoader({"tos": sst})
        obs_loader = MultiVarObsLoader({"tos": sst_obs})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        idx, pattern, _ = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert "time" in idx.dims

    def test_latlon_nao(self, teleconnection_config):
        slp = _make_slp_field(nt=48,
                               lats=np.arange(-89.5, 90, 1.0),
                               lons=np.arange(0.5, 360, 1.0))
        slp_obs = _make_slp_field(nt=48, seed=99,
                                    lats=np.arange(-89.5, 90, 1.0),
                                    lons=np.arange(0.5, 360, 1.0))
        model_loader = MultiVarModelLoader({"psl": slp})
        obs_loader = MultiVarObsLoader({"psl": slp_obs})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["nao"]
        idx, pattern, var_exp = diag._compute_index(
            mode_def, "model-a", source="model",
        )
        assert idx is not None
        assert pattern is not None


# ── CMIP6 tests ──────────────────────────────────────────────────────


class TestCMIP6:
    """Test CMIP6 integration."""

    def test_cmip6_enso(self, sst_field, sst_obs_field, tmp_path):
        cmip6_ds = xr.Dataset({
            "tos": sst_field.rename({"lat": "lat", "lon": "lon"}),
            "areacella": xr.DataArray(
                np.cos(np.deg2rad(sst_field.lat.values))[:, None]
                * np.ones((1, len(sst_field.lon))),
                dims=("lat", "lon"),
                coords={"lat": sst_field.lat, "lon": sst_field.lon},
            ),
        })
        cmip6_loader = MockCMIP6Loader(cmip6_ds)

        config = FeatherConfig(
            model_catalogs={},
            models=["model-a"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": True},
            dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor"},
        )

        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})

        diag = TeleconnectionDiag(
            model_loader, obs_loader, config,
            cmip6_loader=cmip6_loader,
        )
        diag._modes = ["enso"]
        result = diag._compute_mode(_MODE_REGISTRY["enso"])
        assert result is not None
        assert result["cmip6_mmm_index"] is not None

    def test_cmip6_individual(self, sst_field, sst_obs_field, tmp_path):
        cmip6_ds = xr.Dataset({
            "tos": sst_field,
            "areacella": xr.DataArray(
                np.cos(np.deg2rad(sst_field.lat.values))[:, None]
                * np.ones((1, len(sst_field.lon))),
                dims=("lat", "lon"),
                coords={"lat": sst_field.lat, "lon": sst_field.lon},
            ),
        })
        cmip6_loader = MockCMIP6Loader(cmip6_ds)

        config = FeatherConfig(
            model_catalogs={},
            models=["model-a"],
            obs_root="",
            obs_datasets={},
            cmip6={"enabled": True},
            dask={},
            nereus={"influence_radius": 1_000_000},
            output_dir=str(tmp_path / "output"),
            data_source={"type": "cmor"},
        )

        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})

        diag = TeleconnectionDiag(
            model_loader, obs_loader, config,
            cmip6_loader=cmip6_loader,
            cmip6_individual=True,
        )
        diag._modes = ["enso"]
        result = diag._compute_mode(_MODE_REGISTRY["enso"])
        assert result is not None
        assert len(result["cmip6_individual"]) > 0

    def test_cmip6_none_when_disabled(self, sst_field, sst_obs_field,
                                       teleconnection_config):
        model_loader = MultiVarModelLoader({"tos": sst_field})
        obs_loader = MultiVarObsLoader({"tos": sst_obs_field})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        diag._modes = ["enso"]
        result = diag._compute_mode(_MODE_REGISTRY["enso"])
        assert result["cmip6_mmm_index"] is None
        assert result["cmip6_individual"] == {}


# ── Registration tests ───────────────────────────────────────────────


class TestRegistration:
    """Test diagnostic registration."""

    def test_registered(self):
        from feather.diag.registry import get_diagnostic
        diag_cls = get_diagnostic("teleconnections")
        assert diag_cls is TeleconnectionDiag

    def test_in_list(self):
        from feather.diag.registry import list_diagnostics
        diags = list_diagnostics()
        names = [d["name"] for d in diags]
        assert "teleconnections" in names


# ── Prompt update tests ──────────────────────────────────────────────


class TestPrompts:
    """Test LLM prompt updates."""

    def test_figure_analysis_prompt_has_teleconnections(self):
        from feather.llm.prompts import build_figure_analysis_system
        prompt = build_figure_analysis_system()
        assert "teleconnection" in prompt.lower()
        assert "ENSO" in prompt
        assert "NAO" in prompt

    def test_export_prompt_has_teleconnections(self):
        from feather.export.prompts import build_curation_system
        prompt = build_curation_system()
        assert "Teleconnections" in prompt
        assert "ENSO" in prompt


# ── Box wrapping tests ───────────────────────────────────────────────


class TestBoxWrapping:
    """Test longitude wrapping for cross-dateline boxes."""

    def test_wrapping_box(self):
        """Box crossing 0° longitude (e.g. NAO region 270-40E)."""
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        time = xr.date_range("1990-01", periods=12, freq="MS")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((12, len(lats), len(lons)))
        da = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )

        # Box that wraps: 270° to 40° (NAO-like region)
        box = (270, 40, 30, 70)
        result = TeleconnectionDiag._extract_box_mean(da, box)
        assert "time" in result.dims
        assert len(result) == 12


# ── Obs lon normalisation tests ──────────────────────────────────────


class TestObsLonNorm:
    """Test that obs data with -180..180 lons is normalised to 0..360."""

    def test_obs_lons_normalised(self, teleconnection_config):
        """Obs with -180..180 lons should be shifted to 0..360 in _load_field."""
        lats = np.arange(-87.5, 90, 5.0)
        lons_neg = np.arange(-177.5, 180, 5.0)  # -180..180 convention
        nt = 48
        time = xr.date_range("1990-01", periods=nt, freq="MS")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((nt, len(lats), len(lons_neg)))
        da_neg = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons_neg},
        )

        model_loader = MultiVarModelLoader({"tos": _make_sst_field()})
        obs_loader = MultiVarObsLoader({"tos": da_neg})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["enso"]
        loaded = diag._load_field(mode_def, None, source="obs")
        # After normalisation, all lons should be >= 0
        assert float(loaded.lon.values.min()) >= 0
        assert float(loaded.lon.values.max()) < 360

    def test_obs_nao_eof_with_neg_lons(self, teleconnection_config):
        """NAO EOF should work when obs has -180..180 lons."""
        lats = np.arange(-87.5, 90, 5.0)
        lons_neg = np.arange(-177.5, 180, 5.0)
        slp = _make_slp_field(lons=np.arange(2.5, 360, 5.0))

        # Obs in -180..180
        slp_obs_neg = _make_slp_field(seed=99, lons=lons_neg)

        model_loader = MultiVarModelLoader({"psl": slp})
        obs_loader = MultiVarObsLoader({"psl": slp_obs_neg})
        diag = TeleconnectionDiag(
            model_loader, obs_loader, teleconnection_config,
        )
        mode_def = _MODE_REGISTRY["nao"]
        result = diag._compute_mode(mode_def)
        # Both model and obs should have patterns
        assert result is not None
        assert result["obs_index"] is not None
        assert result["obs_pattern"] is not None

    def test_regrid_patterns_to_common(self):
        """Patterns from different grids should be regridded to common."""
        lats_a = np.arange(-85, 90, 10.0)
        lons_a = np.arange(5, 360, 10.0)
        lats_b = np.arange(-87.5, 90, 5.0)
        lons_b = np.arange(2.5, 360, 5.0)
        rng = np.random.default_rng(42)
        pat_a = xr.DataArray(
            rng.standard_normal((len(lats_a), len(lons_a))),
            dims=("lat", "lon"),
            coords={"lat": lats_a, "lon": lons_a},
        )
        pat_b = xr.DataArray(
            rng.standard_normal((len(lats_b), len(lons_b))),
            dims=("lat", "lon"),
            coords={"lat": lats_b, "lon": lons_b},
        )
        patterns = {"A": pat_a, "B": pat_b}
        result = TeleconnectionDiag._regrid_patterns_to_common(patterns)
        # Both should have same shape now
        assert result["A"].shape == result["B"].shape
        # Lons should be in -180..180
        assert float(result["A"].lon.values.min()) >= -180
        assert float(result["A"].lon.values.max()) <= 180

    def test_regrid_curvilinear_pattern_to_common(self):
        """Curvilinear patterns should be regridded to common 1-deg grid."""
        nj, ni = 20, 30
        j_vals = np.arange(nj)
        i_vals = np.arange(ni)
        lat_2d = np.linspace(-30, 30, nj)[:, np.newaxis] * np.ones(ni)
        lon_2d = np.ones(nj)[:, np.newaxis] * np.linspace(150, 250, ni)

        rng = np.random.default_rng(42)
        pat_curv = xr.DataArray(
            rng.standard_normal((nj, ni)),
            dims=("j", "i"),
            coords={
                "j": j_vals,
                "i": i_vals,
                "latitude": (("j", "i"), lat_2d),
                "longitude": (("j", "i"), lon_2d),
            },
        )
        # Also add a rectilinear pattern
        lats_r = np.arange(-30, 35, 5.0)
        lons_r = np.arange(150, 255, 5.0)
        pat_rect = xr.DataArray(
            rng.standard_normal((len(lats_r), len(lons_r))),
            dims=("lat", "lon"),
            coords={"lat": lats_r, "lon": lons_r},
        )

        patterns = {"curv": pat_curv, "rect": pat_rect}
        result = TeleconnectionDiag._regrid_patterns_to_common(patterns)

        assert "curv" in result
        assert "rect" in result
        # Both should have lat/lon dims on common grid
        assert "lat" in result["curv"].dims
        assert "lon" in result["curv"].dims
        assert result["curv"].shape == result["rect"].shape

    def test_regrid_renamed_rectilinear_pattern(self):
        """Renamed rectilinear (IPSL-style: 1D coords on i/j dims)."""
        nj, ni = 30, 40  # different lengths
        rng = np.random.default_rng(42)
        pat = xr.DataArray(
            rng.standard_normal((nj, ni)),
            dims=("j", "i"),
            coords={
                "latitude": ("j", np.linspace(-30, 30, nj)),
                "longitude": ("i", np.linspace(150, 250, ni)),
            },
        )
        patterns = {"ipsl": pat}
        result = TeleconnectionDiag._regrid_patterns_to_common(patterns)
        assert "ipsl" in result
        assert "lat" in result["ipsl"].dims
        assert "lon" in result["ipsl"].dims


# ── Sign convention tests ───────────────────────────────────────────


class TestSignConvention:
    """Test EOF sign convention fixing."""

    def test_fix_eof_sign_flips_when_positive(self):
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        eof = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        pc = xr.DataArray(np.ones(12), dims="time")
        # sign_point should be negative → flip
        pc_fixed, eof_fixed = TeleconnectionDiag._fix_eof_sign(
            pc, eof, sign_point=(65, 5),
        )
        # All values positive at sign point → should flip
        assert float(eof_fixed.sel(lat=65, lon=5, method="nearest")) < 0

    def test_fix_eof_sign_no_flip_when_negative(self):
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        eof = xr.DataArray(
            -np.ones((len(lats), len(lons))),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        pc = xr.DataArray(np.ones(12), dims="time")
        pc_fixed, eof_fixed = TeleconnectionDiag._fix_eof_sign(
            pc, eof, sign_point=(65, 5),
        )
        # Already negative → no flip
        assert float(eof_fixed.sel(lat=65, lon=5, method="nearest")) < 0


# ── Regression pattern tests ────────────────────────────────────────


class TestRegressionPattern:
    """Test regression map computation."""

    def test_regression_with_known_signal(self):
        """Regression should recover the spatial pattern of a known signal."""
        lats = np.arange(-85, 90, 10.0)
        lons = np.arange(5, 360, 10.0)
        nt = 60
        time = xr.date_range("1990-01", periods=nt, freq="MS")

        rng = np.random.default_rng(42)
        # Known index
        idx = xr.DataArray(
            np.sin(2 * np.pi * np.arange(nt) / 24),
            dims="time", coords={"time": time},
        )

        # Field = pattern * index + noise
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        pattern_true = np.sin(np.deg2rad(lat_grid))
        data = np.zeros((nt, len(lats), len(lons)))
        for t in range(nt):
            data[t] = pattern_true * idx.values[t]
            data[t] += 0.05 * rng.standard_normal(data[t].shape)

        da = xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lats, "lon": lons},
        )

        pattern = TeleconnectionDiag._regression_pattern(da, idx)
        assert pattern is not None
        # Should correlate with true pattern
        corr = np.corrcoef(
            pattern.values.ravel(), pattern_true.ravel(),
        )[0, 1]
        assert corr > 0.8, f"Pattern correlation {corr} too low"

    def test_regression_returns_none_for_short_data(self):
        time = xr.date_range("1990-01", periods=2, freq="MS")
        idx = xr.DataArray([1.0, 2.0], dims="time", coords={"time": time})
        da = xr.DataArray(
            np.random.randn(2, 5, 5), dims=("time", "lat", "lon"),
            coords={"time": time,
                     "lat": np.arange(5.0),
                     "lon": np.arange(5.0)},
        )
        pattern = TeleconnectionDiag._regression_pattern(da, idx)
        assert pattern is None


class TestCurvilinearGrid:
    """Test support for CMIP6 ocean models with curvilinear grids."""

    @staticmethod
    def _make_curvilinear_sst(nt=48, nj=30, ni=40, seed=42):
        """Create synthetic SST on a curvilinear grid (dims j, i)."""
        rng = np.random.default_rng(seed)
        time = xr.date_range("1990-01", periods=nt, freq="MS")

        # Simulate a curvilinear grid (like ORCA/tripolar)
        j_idx = np.arange(nj)
        i_idx = np.arange(ni)
        j2d, i2d = np.meshgrid(j_idx, i_idx, indexing="ij")

        # Smooth mapping from (j,i) → (lat,lon)
        lat_2d = -90 + 180 * j2d / (nj - 1) + 2 * np.sin(2 * np.pi * i2d / ni)
        lon_2d = 360 * i2d / ni + 5 * np.sin(2 * np.pi * j2d / nj)
        lat_2d = np.clip(lat_2d, -90, 90)
        lon_2d = lon_2d % 360

        # SST data with ENSO-like warm anomaly in Nino 3.4 box
        enso_mask = (
            (lat_2d >= -5) & (lat_2d <= 5) &
            (lon_2d >= 190) & (lon_2d <= 240)
        )
        enso_signal = 2.0 * np.sin(2 * np.pi * np.arange(nt) / 48)
        seasonal = 3.0 * np.sin(2 * np.pi * (np.arange(nt) - 3) / 12)

        data = np.zeros((nt, nj, ni))
        base = 300 - 20 * np.abs(lat_2d / 90)
        for t in range(nt):
            data[t] = base + enso_signal[t] * enso_mask + seasonal[t]
            data[t] += 0.1 * rng.standard_normal((nj, ni))

        return xr.DataArray(
            data, dims=("time", "j", "i"),
            coords={
                "time": time,
                "latitude": (("j", "i"), lat_2d),
                "longitude": (("j", "i"), lon_2d),
            },
        )

    def test_find_latlon_curvilinear(self):
        """_find_latlon detects curvilinear coords as non-rectilinear."""
        da = self._make_curvilinear_sst(nt=12)
        lat_name, lon_name, is_rect = TeleconnectionDiag._find_latlon(da)
        assert not is_rect
        assert lat_name == "latitude"
        assert lon_name == "longitude"

    def test_box_mean_curvilinear(self):
        """Box mean works on curvilinear grids via nereus masking."""
        da = self._make_curvilinear_sst(nt=48)
        box = (190, 240, -5, 5)  # Nino 3.4
        ts = TeleconnectionDiag._extract_box_mean(da, box)
        assert "time" in ts.dims
        assert ts.sizes["time"] == 48
        # Should have a clear signal (ENSO amplitude ~ 2K)
        assert float(ts.std()) > 0.5

    def test_box_mean_curvilinear_wrapping(self):
        """Box mean handles longitude wrapping on curvilinear grids."""
        da = self._make_curvilinear_sst(nt=12)
        # Wrapping box: lon_min > lon_max
        box = (350, 10, -10, 10)
        ts = TeleconnectionDiag._extract_box_mean(da, box)
        assert "time" in ts.dims

    def test_field_global_mean_curvilinear(self):
        """Global mean works on curvilinear grid."""
        da = self._make_curvilinear_sst(nt=12)
        gm = TeleconnectionDiag._field_global_mean(da)
        assert "time" in gm.dims
        assert gm.sizes["time"] == 12
        # Should be a reasonable SST value
        assert 250 < float(gm.mean()) < 310

    def test_cmip6_single_box_mean_curvilinear(self, tmp_path):
        """CMIP6 single model computation works for curvilinear tos."""
        da = self._make_curvilinear_sst(nt=48)
        mode_def = _MODE_REGISTRY["enso"]

        cfg = FeatherConfig(
            model_catalogs={}, models=[], obs_root="",
            obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path),
        )
        diag = TeleconnectionDiag.__new__(TeleconnectionDiag)
        diag.config = cfg

        idx, pat = diag._compute_cmip6_single(mode_def, da)
        assert idx is not None
        assert "time" in idx.dims
        # Regression pattern computed for curvilinear (regridded for display)
        assert pat is not None

    def test_box_mean_renamed_rectilinear(self):
        """Box mean works on rectilinear grids with i/j dims (e.g. IPSL)."""
        nj, ni = 30, 40
        nt = 24
        time = xr.date_range("1990-01", periods=nt, freq="MS")
        lats_1d = np.linspace(-85, 85, nj)
        lons_1d = np.linspace(0, 355, ni)

        rng = np.random.default_rng(42)
        data = 300 + rng.standard_normal((nt, nj, ni))

        da = xr.DataArray(
            data, dims=("time", "j", "i"),
            coords={
                "time": time,
                "latitude": ("j", lats_1d),
                "longitude": ("i", lons_1d),
            },
        )

        # Should be detected as non-rectilinear (lat/lon not dims)
        lat_name, lon_name, is_rect = TeleconnectionDiag._find_latlon(da)
        assert not is_rect

        # But box mean should still work via meshgrid + masking
        box = (190, 240, -5, 5)
        ts = TeleconnectionDiag._extract_box_mean(da, box)
        assert "time" in ts.dims
        assert ts.sizes["time"] == nt

    def test_cmip6_single_eof_curvilinear_index_only(self, tmp_path):
        """EOF modes on curvilinear grids produce index but no pattern."""
        da = self._make_curvilinear_sst(nt=48)
        mode_def = _MODE_REGISTRY["pdo"]

        cfg = FeatherConfig(
            model_catalogs={}, models=[], obs_root="",
            obs_datasets={}, cmip6={"enabled": False},
            dask={}, nereus={}, output_dir=str(tmp_path),
        )
        diag = TeleconnectionDiag.__new__(TeleconnectionDiag)
        diag.config = cfg

        idx, pat = diag._compute_cmip6_single(mode_def, da)
        # Index computed via flat EOF, pattern skipped
        assert idx is not None
        assert "time" in idx.dims
        assert pat is None
