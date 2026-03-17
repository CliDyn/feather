"""Tests for GRIBLoader — per-month GRIB files on HEALPix grid."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.grib_loader import GRIBLoader, _SQUEEZE_DIMS


# ── Fixtures ────────────────────────────────────────────────────────────

NCELLS = 196608  # nside=128, but for tests we use small synthetic data
SMALL_CELLS = 768  # nside=8


def _make_config(
    tmp_path,
    *,
    variable_aliases=None,
    scale_factors=None,
    depth_levels=None,
    models=None,
    model_configs=None,
    data_root="",
):
    """Create a FeatherConfig for testing."""
    aliases = (
        variable_aliases if variable_aliases is not None
        else {"tas": "sfc_mean2t", "tos": "o2d_avg_tos",
              "thetao": "o3d_avg_thetao"}
    )
    data_source = {
        "type": "grib_healpix",
        "root": str(tmp_path / "grib"),
    }

    ocean_3d = {}
    if depth_levels is not None:
        ocean_3d["depth_levels"] = depth_levels

    if models is None:
        models = ["IFS-FESOM-TCO399"]
    if model_configs is None:
        model_configs = {
            "IFS-FESOM-TCO399": ModelConfig(
                name="IFS-FESOM-TCO399",
                institution="AWI",
                grids={"sfc": "healpix", "o2d": "healpix", "o3d": "healpix"},
                color="#1f77b4",
                variable_aliases=aliases,
                scale_factors=scale_factors or {},
                data_root=data_root,
            ),
        }

    return FeatherConfig(
        model_catalogs={},
        models=models,
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source=data_source,
        model_configs=model_configs,
        ocean_3d=ocean_3d,
    )


def _make_sfc_grib_dataset(ncells=SMALL_CELLS, n_months=12, year=1990):
    """Create a synthetic surface xr.Dataset mimicking cfgrib output.

    cfgrib produces: dims=(time, values), coords=(latitude, longitude,
    step, valid_time, heightAboveGround).
    """
    lon = np.linspace(0, 360, ncells, endpoint=False)
    lat = np.linspace(-90, 90, ncells)
    time = xr.date_range(f"{year}-01", periods=n_months, freq="MS")

    temp = 300 - 40 * np.abs(lat / 90.0)
    data = temp[np.newaxis, :] + np.random.default_rng(year).normal(
        0, 1, (n_months, 1)
    )

    return xr.Dataset({
        "mean2t": xr.DataArray(
            data,
            dims=["time", "values"],
            coords={
                "time": time,
                "latitude": ("values", lat),
                "longitude": ("values", lon),
                "heightAboveGround": 2.0,
                "step": np.timedelta64(0, "h"),
            },
        ),
    })


def _make_ocean3d_grib_dataset(ncells=SMALL_CELLS, n_levels=28, year=1990):
    """Create a synthetic ocean 3D xr.Dataset mimicking cfgrib output.

    cfgrib produces: dims=(oceanModelLayer, time, values).
    """
    lon = np.linspace(0, 360, ncells, endpoint=False)
    lat = np.linspace(-90, 90, ncells)
    time = xr.date_range(f"{year}-01", periods=12, freq="MS")
    layers = np.arange(2, 2 * n_levels + 1, 2, dtype=np.float64)

    data = np.random.default_rng(year).normal(
        280, 5, (n_levels, 12, ncells),
    )

    return xr.Dataset({
        "avg_thetao": xr.DataArray(
            data,
            dims=["oceanModelLayer", "time", "values"],
            coords={
                "oceanModelLayer": layers,
                "time": time,
                "latitude": ("values", lat),
                "longitude": ("values", lon),
            },
        ),
    })


def _make_ocean2d_grib_dataset(ncells=SMALL_CELLS, year=1990):
    """Create a synthetic ocean 2D xr.Dataset mimicking cfgrib output."""
    lon = np.linspace(0, 360, ncells, endpoint=False)
    lat = np.linspace(-90, 90, ncells)
    time = xr.date_range(f"{year}-01", periods=12, freq="MS")

    data = np.random.default_rng(year).normal(290, 5, (12, ncells))

    return xr.Dataset({
        "avg_tos": xr.DataArray(
            data,
            dims=["time", "values"],
            coords={
                "time": time,
                "latitude": ("values", lat),
                "longitude": ("values", lon),
                "oceanSurface": 0.0,
            },
        ),
    })


# ── Test: file prefix derivation ─────────────────────────────────────


class TestFilePrefix:
    """Test _get_file_prefix static method."""

    def test_sfc_prefix(self):
        assert GRIBLoader._get_file_prefix("sfc_mean2t") == "mean2t"

    def test_o2d_prefix(self):
        assert GRIBLoader._get_file_prefix("o2d_avg_tos") == "avg_tos"

    def test_o3d_prefix(self):
        assert GRIBLoader._get_file_prefix("o3d_avg_thetao") == "avg_thetao"

    def test_pl_prefix(self):
        assert GRIBLoader._get_file_prefix("pl_t") == "t"

    def test_sol_prefix(self):
        assert GRIBLoader._get_file_prefix("sol_sot") == "sot"

    def test_misc_prefix(self):
        assert GRIBLoader._get_file_prefix("misc_msd") == "msd"

    def test_no_prefix(self):
        """Dir name without known prefix is returned as-is."""
        assert GRIBLoader._get_file_prefix("thetao") == "thetao"

    def test_unknown_prefix(self):
        """Unknown prefix is NOT stripped."""
        assert GRIBLoader._get_file_prefix("atm_mean2t") == "atm_mean2t"


# ── Test: alias resolution ───────────────────────────────────────────


class TestAliasResolution:
    """Test _get_dir_name alias lookup order."""

    def test_alias_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, variable_aliases={"tas": "custom_dir"})
        loader = GRIBLoader(cfg)
        assert loader._get_dir_name("IFS-FESOM-TCO399", "tas") == "custom_dir"

    def test_fallback_to_destine_variable(self, tmp_path):
        cfg = _make_config(tmp_path, variable_aliases={})
        loader = GRIBLoader(cfg)
        # "tas" has destine_variable="avg_2t" in the registry
        assert loader._get_dir_name("IFS-FESOM-TCO399", "tas") == "avg_2t"

    def test_fallback_to_canonical(self, tmp_path):
        cfg = _make_config(tmp_path, variable_aliases={})
        loader = GRIBLoader(cfg)
        name = loader._get_dir_name("IFS-FESOM-TCO399", "nonexistent_var")
        assert name == "nonexistent_var"


# ── Test: data_root override ─────────────────────────────────────────


class TestDataRoot:
    """Test per-model data_root override."""

    def test_default_root(self, tmp_path):
        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        root = loader._get_data_root("IFS-FESOM-TCO399")
        assert root == tmp_path / "grib"

    def test_per_model_override(self, tmp_path):
        cfg = _make_config(tmp_path, data_root="/custom/path")
        loader = GRIBLoader(cfg)
        from pathlib import Path
        assert loader._get_data_root("IFS-FESOM-TCO399") == Path("/custom/path")

    def test_two_models_different_roots(self, tmp_path):
        """Two models can have different data_root paths."""
        mc = {
            "ModelA": ModelConfig(
                name="ModelA",
                data_root="/path/a",
                variable_aliases={"tas": "sfc_mean2t"},
            ),
            "ModelB": ModelConfig(
                name="ModelB",
                data_root="/path/b",
                variable_aliases={"tas": "sfc_mean2t"},
            ),
        }
        cfg = _make_config(
            tmp_path,
            models=["ModelA", "ModelB"],
            model_configs=mc,
        )
        loader = GRIBLoader(cfg)
        from pathlib import Path
        assert loader._get_data_root("ModelA") == Path("/path/a")
        assert loader._get_data_root("ModelB") == Path("/path/b")


# ── Test: file discovery ─────────────────────────────────────────────


class TestFileDiscovery:
    """Test _discover_grib_files."""

    def test_finds_files(self, tmp_path):
        """Discovers GRIB files across year directories."""
        root = tmp_path / "grib"
        for year in [1990, 1991]:
            var_dir = root / str(year) / "sfc_mean2t"
            var_dir.mkdir(parents=True)
            for month in range(1, 13):
                (var_dir / f"mean2t_{year}{month:02d}.grib").touch()

        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        files = loader._discover_grib_files(root, "sfc_mean2t", "mean2t")
        assert len(files) == 24

    def test_empty_root_returns_empty(self, tmp_path):
        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        files = loader._discover_grib_files(
            tmp_path / "nonexistent", "sfc_mean2t", "mean2t",
        )
        assert files == []

    def test_ignores_non_year_dirs(self, tmp_path):
        """Non-numeric directories are ignored."""
        root = tmp_path / "grib"
        # Create a non-year directory
        (root / "README").mkdir(parents=True)
        # Create a year directory with data
        var_dir = root / "1990" / "sfc_mean2t"
        var_dir.mkdir(parents=True)
        (var_dir / "mean2t_199001.grib").touch()

        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        files = loader._discover_grib_files(root, "sfc_mean2t", "mean2t")
        assert len(files) == 1

    def test_sorted_by_path(self, tmp_path):
        """Files are returned in sorted order."""
        root = tmp_path / "grib"
        for year in [1991, 1990]:
            var_dir = root / str(year) / "sfc_mean2t"
            var_dir.mkdir(parents=True)
            (var_dir / f"mean2t_{year}01.grib").touch()

        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        files = loader._discover_grib_files(root, "sfc_mean2t", "mean2t")
        assert "1990" in str(files[0])
        assert "1991" in str(files[1])


# ── Test: find_data_var ──────────────────────────────────────────────


class TestFindDataVar:
    """Test _find_data_var static method."""

    def test_strip_sfc_prefix(self):
        ds = xr.Dataset({"mean2t": xr.DataArray([1, 2])})
        assert GRIBLoader._find_data_var(ds, "sfc_mean2t", "tas") == "mean2t"

    def test_strip_o3d_prefix(self):
        ds = xr.Dataset({"avg_thetao": xr.DataArray([1, 2])})
        assert GRIBLoader._find_data_var(ds, "o3d_avg_thetao", "thetao") == "avg_thetao"

    def test_canonical_fallback(self):
        ds = xr.Dataset({"tas": xr.DataArray([1, 2])})
        assert GRIBLoader._find_data_var(ds, "sfc_mean2t", "tas") == "tas"

    def test_first_var_fallback(self):
        ds = xr.Dataset({"mystery": xr.DataArray([1, 2])})
        assert GRIBLoader._find_data_var(ds, "sfc_mean2t", "tas") == "mystery"

    def test_empty_raises(self):
        with pytest.raises(KeyError, match="No data variable"):
            GRIBLoader._find_data_var(xr.Dataset(), "sfc_mean2t", "tas")


# ── Test: loading via monkeypatched open_mfdataset ───────────────────


class TestLoadVar:
    """Test full load_var pipeline with monkeypatched xr.open_mfdataset."""

    @pytest.fixture
    def grib_setup(self, tmp_path, monkeypatch):
        """Set up GRIB file structure and patch open_mfdataset."""
        root = tmp_path / "grib"
        for year in [1990, 1991]:
            var_dir = root / str(year) / "sfc_mean2t"
            var_dir.mkdir(parents=True)
            for month in range(1, 13):
                (var_dir / f"mean2t_{year}{month:02d}.grib").touch()

        ds1 = _make_sfc_grib_dataset(year=1990)
        ds2 = _make_sfc_grib_dataset(year=1991)
        combined = xr.concat([ds1, ds2], dim="time")

        def mock_open_mfdataset(paths, **kwargs):
            assert kwargs.get("engine") == "cfgrib"
            return combined

        monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)
        return tmp_path

    def test_basic_load(self, grib_setup):
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas")
        assert "time" in da.dims
        assert "values" in da.dims
        assert da.sizes["time"] == 24

    def test_singleton_squeezed(self, grib_setup):
        """heightAboveGround singleton coord is dropped."""
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas")
        assert "heightAboveGround" not in da.dims
        assert "heightAboveGround" not in da.coords

    def test_step_squeezed(self, grib_setup):
        """Step scalar coord is dropped."""
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas")
        assert "step" not in da.coords

    def test_has_lat_lon(self, grib_setup):
        """latitude/longitude coords preserved from cfgrib."""
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas")
        assert "latitude" in da.coords
        assert "longitude" in da.coords

    def test_period_slicing(self, grib_setup):
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas",
                             period=("1990", "1990"))
        assert da.sizes["time"] == 12

    def test_time_mean(self, grib_setup):
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas", time_mean=True)
        assert "time" not in da.dims

    def test_period_plus_time_mean(self, grib_setup):
        cfg = _make_config(grib_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas",
                             period=("1990", "1990"), time_mean=True)
        assert "time" not in da.dims
        assert "values" in da.dims


# ── Test: scale factors ──────────────────────────────────────────────


class TestScaleFactors:
    """Test scale factor application."""

    @pytest.fixture
    def scale_setup(self, tmp_path, monkeypatch):
        root = tmp_path / "grib"
        var_dir = root / "1990" / "sfc_mean2t"
        var_dir.mkdir(parents=True)
        for m in range(1, 13):
            (var_dir / f"mean2t_1990{m:02d}.grib").touch()

        ds = _make_sfc_grib_dataset(year=1990)

        def mock_open(paths, **kwargs):
            return ds

        monkeypatch.setattr(xr, "open_mfdataset", mock_open)
        return tmp_path

    def test_scale_applied(self, scale_setup):
        cfg_scaled = _make_config(scale_setup, scale_factors={"tas": 100.0})
        cfg_raw = _make_config(scale_setup, scale_factors={})

        loader_s = GRIBLoader(cfg_scaled)
        loader_r = GRIBLoader(cfg_raw)

        da_s = loader_s.load_var("IFS-FESOM-TCO399", "tas", time_mean=True)
        da_r = loader_r.load_var("IFS-FESOM-TCO399", "tas", time_mean=True)

        np.testing.assert_allclose(da_s.values, da_r.values * 100.0, rtol=1e-5)

    def test_no_scale_is_identity(self, scale_setup):
        cfg = _make_config(scale_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tas", time_mean=True)
        assert np.isfinite(da.values).all()
        assert da.values.mean() > 250


# ── Test: caching ────────────────────────────────────────────────────


class TestCaching:
    """Test dataset caching."""

    @pytest.fixture
    def cache_setup(self, tmp_path, monkeypatch):
        root = tmp_path / "grib"
        for vdir in ["sfc_mean2t", "sfc_msl"]:
            d = root / "1990" / vdir
            d.mkdir(parents=True)
            prefix = GRIBLoader._get_file_prefix(vdir)
            (d / f"{prefix}_199001.grib").touch()

        ds_tas = _make_sfc_grib_dataset(year=1990, n_months=1)
        ds_psl = xr.Dataset({
            "msl": ds_tas["mean2t"].rename("msl"),
        })
        # Preserve coords
        for c in ds_tas.coords:
            if c not in ds_psl.coords:
                ds_psl.coords[c] = ds_tas.coords[c]

        call_count = {"n": 0}

        def mock_open(paths, **kwargs):
            call_count["n"] += 1
            # Determine which variable based on file paths
            path_str = str(paths[0])
            if "msl" in path_str:
                return ds_psl
            return ds_tas

        monkeypatch.setattr(xr, "open_mfdataset", mock_open)
        return tmp_path, call_count

    def test_repeated_load_uses_cache(self, cache_setup):
        tmp_path, call_count = cache_setup
        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        loader.load_var("IFS-FESOM-TCO399", "tas")
        loader.load_var("IFS-FESOM-TCO399", "tas")
        assert call_count["n"] == 1  # only opened once

    def test_different_vars_separate(self, cache_setup):
        tmp_path, call_count = cache_setup
        aliases = {"tas": "sfc_mean2t", "psl": "sfc_msl"}
        cfg = _make_config(tmp_path, variable_aliases=aliases)
        loader = GRIBLoader(cfg)
        loader.load_var("IFS-FESOM-TCO399", "tas")
        loader.load_var("IFS-FESOM-TCO399", "psl")
        assert ("IFS-FESOM-TCO399", "tas") in loader._cache
        assert ("IFS-FESOM-TCO399", "psl") in loader._cache


# ── Test: ocean 3D depth remapping ───────────────────────────────────


class TestOceanDepthRemap:
    """Test oceanModelLayer → lev depth remapping."""

    @pytest.fixture
    def ocean_setup(self, tmp_path, monkeypatch):
        root = tmp_path / "grib"
        d = root / "1990" / "o3d_avg_thetao"
        d.mkdir(parents=True)
        for m in range(1, 13):
            (d / f"avg_thetao_1990{m:02d}.grib").touch()

        ds = _make_ocean3d_grib_dataset(year=1990, n_levels=5)

        def mock_open(paths, **kwargs):
            return ds

        monkeypatch.setattr(xr, "open_mfdataset", mock_open)
        return tmp_path

    def test_lev_dim_created(self, ocean_setup):
        """oceanModelLayer is renamed to lev."""
        cfg = _make_config(ocean_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao")
        assert "lev" in da.dims
        assert "oceanModelLayer" not in da.dims

    def test_depth_values_from_config(self, ocean_setup):
        """When depth_levels configured, lev has real depth values."""
        depths = [7.5, 25.0, 45.0, 65.0, 85.0]
        cfg = _make_config(ocean_setup, depth_levels=depths)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao")
        np.testing.assert_array_equal(da["lev"].values, depths)

    def test_no_depth_config_keeps_indices(self, ocean_setup):
        """Without depth_levels, lev has the raw GRIB layer indices."""
        cfg = _make_config(ocean_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao")
        expected = np.arange(2, 12, 2, dtype=np.float64)  # 5 levels
        np.testing.assert_array_equal(da["lev"].values, expected)

    def test_28_levels_full_remap(self, tmp_path, monkeypatch):
        """Full 28-level depth remapping works."""
        root = tmp_path / "grib"
        d = root / "1990" / "o3d_avg_thetao"
        d.mkdir(parents=True)
        for m in range(1, 13):
            (d / f"avg_thetao_1990{m:02d}.grib").touch()

        ds = _make_ocean3d_grib_dataset(year=1990, n_levels=28)
        monkeypatch.setattr(xr, "open_mfdataset", lambda *a, **kw: ds)

        depths = [
            7.5, 25.0, 45.0, 65.0, 85.0, 107.5, 137.5, 170.0,
            212.5, 262.5, 312.5, 362.5, 425.0, 525.0, 625.0, 750.0,
            970.0, 1255.0, 1600.0, 2035.0, 2525.0, 3025.0, 3525.0,
            4025.0, 4525.0, 5025.0, 5525.0, 6125.0,
        ]
        cfg = _make_config(tmp_path, depth_levels=depths)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao")
        assert da.sizes["lev"] == 28
        np.testing.assert_array_almost_equal(da["lev"].values, depths)

    def test_mismatched_depth_count_keeps_indices(self, ocean_setup):
        """If depth_levels count != n_layers, keep raw indices."""
        cfg = _make_config(ocean_setup, depth_levels=[10.0, 50.0])  # wrong count
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao")
        # Should keep raw indices since count doesn't match
        expected = np.arange(2, 12, 2, dtype=np.float64)
        np.testing.assert_array_equal(da["lev"].values, expected)


# ── Test: ocean 2D singleton squeeze ─────────────────────────────────


class TestOcean2DSqueeze:
    """Test that ocean 2D singleton coords are squeezed."""

    @pytest.fixture
    def ocean2d_setup(self, tmp_path, monkeypatch):
        root = tmp_path / "grib"
        d = root / "1990" / "o2d_avg_tos"
        d.mkdir(parents=True)
        for m in range(1, 13):
            (d / f"avg_tos_1990{m:02d}.grib").touch()

        ds = _make_ocean2d_grib_dataset(year=1990)

        def mock_open(paths, **kwargs):
            return ds

        monkeypatch.setattr(xr, "open_mfdataset", mock_open)
        return tmp_path

    def test_ocean_surface_squeezed(self, ocean2d_setup):
        cfg = _make_config(ocean2d_setup)
        loader = GRIBLoader(cfg)
        da = loader.load_var("IFS-FESOM-TCO399", "tos")
        assert "oceanSurface" not in da.dims
        assert "oceanSurface" not in da.coords


# ── Test: missing data errors ────────────────────────────────────────


class TestMissingData:
    """Test error handling for missing files/variables."""

    def test_no_files_raises(self, tmp_path, monkeypatch):
        """No GRIB files → FileNotFoundError."""
        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        with pytest.raises(FileNotFoundError, match="No GRIB files"):
            loader.load_var("IFS-FESOM-TCO399", "tas")

    def test_missing_model_no_data_root(self, tmp_path):
        """Model without config still uses default root."""
        cfg = _make_config(tmp_path)
        loader = GRIBLoader(cfg)
        # UNKNOWN model has no data_root override, falls back to cfg root
        with pytest.raises(FileNotFoundError):
            loader.load_var("UNKNOWN", "tas")


# ── Test: squeeze dim set completeness ───────────────────────────────


class TestSqueezeDims:
    """Test that _SQUEEZE_DIMS covers expected GRIB singleton dims."""

    def test_known_dims_in_set(self):
        for dim in [
            "heightAboveGround", "step", "valid_time", "surface",
            "entireAtmosphere", "meanSea", "iceTopOnWater",
            "iceLayerOnWater", "oceanSurface",
        ]:
            assert dim in _SQUEEZE_DIMS

    def test_data_dims_not_in_set(self):
        """Data dimensions should NOT be in the squeeze set."""
        for dim in ["time", "values", "oceanModelLayer", "isobaricInhPa"]:
            assert dim not in _SQUEEZE_DIMS


# ── Test: run.py dispatch ────────────────────────────────────────────


class TestRunDispatch:
    """Test that _create_model_loader dispatches to GRIBLoader."""

    def test_grib_healpix_dispatch(self, tmp_path):
        from feather.run import _create_model_loader
        cfg = _make_config(tmp_path)
        loader = _create_model_loader(cfg)
        assert isinstance(loader, GRIBLoader)


# ── Test: base class integration ─────────────────────────────────────


class TestBaseClassIntegration:
    """Test config helpers for grib_healpix source type."""

    def test_data_source_type(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.get_data_source_type() == "grib_healpix"

    def test_grid_type_healpix(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.get_grid_type("IFS-FESOM-TCO399", "sfc") == "healpix"

    def test_model_color(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.get_model_color("IFS-FESOM-TCO399") == "#1f77b4"


# ── Test: config YAML loading ────────────────────────────────────────


class TestConfigYAML:
    """Test tco_grib.yaml config loads correctly."""

    @pytest.mark.integration
    def test_load_config(self):
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "tco_grib.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
        cfg = FeatherConfig.from_yaml(config_path)
        assert "IFS-FESOM-TCO399" in cfg.models
        assert "IFS-FESOM-TCO319" in cfg.models
        assert cfg.get_data_source_type() == "grib_healpix"
        assert cfg.get_grid_type("IFS-FESOM-TCO399") == "healpix"
        assert cfg.get_period() == ("1990", "2014")
        # Variable aliases
        m399 = cfg.model_configs["IFS-FESOM-TCO399"]
        assert m399.variable_aliases["tas"] == "sfc_mean2t"
        assert m399.data_root != ""
        # Different data_root for each model
        m319 = cfg.model_configs["IFS-FESOM-TCO319"]
        assert m399.data_root != m319.data_root
        # Depth levels
        depth_levels = cfg.ocean_3d["depth_levels"]
        assert len(depth_levels) == 28
        assert depth_levels[0] == 7.5
        assert depth_levels[-1] == 6125.0


# ── Test: integration (real GRIB files on Levante) ───────────────────


class TestRealGRIBFiles:
    """Integration tests using real GRIB data on Levante."""

    TCO399_ROOT = "/work/ab0995/a270135/MN5/projt399/a3bo_grib/varyears"

    @pytest.fixture
    def real_config(self):
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "tco_grib.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Config not found")
        return FeatherConfig.from_yaml(config_path)

    @pytest.mark.integration
    def test_load_surface_var(self, real_config):
        """Load a surface variable from real GRIB files."""
        import os
        if not os.path.exists(self.TCO399_ROOT):
            pytest.skip("TCO399 data not available")
        loader = GRIBLoader(real_config)
        da = loader.load_var("IFS-FESOM-TCO399", "tas",
                             period=("1990", "1990"))
        assert "time" in da.dims
        assert "values" in da.dims
        assert da.sizes["values"] == 196608  # nside=128
        assert "latitude" in da.coords
        assert "longitude" in da.coords

    @pytest.mark.integration
    def test_load_ocean2d_var(self, real_config):
        """Load an ocean 2D variable from real GRIB files."""
        import os
        if not os.path.exists(self.TCO399_ROOT):
            pytest.skip("TCO399 data not available")
        loader = GRIBLoader(real_config)
        da = loader.load_var("IFS-FESOM-TCO399", "tos",
                             period=("1990", "1990"))
        assert "time" in da.dims
        assert "values" in da.dims

    @pytest.mark.integration
    def test_load_ocean3d_var(self, real_config):
        """Load an ocean 3D variable with depth remapping."""
        import os
        if not os.path.exists(self.TCO399_ROOT):
            pytest.skip("TCO399 data not available")
        loader = GRIBLoader(real_config)
        da = loader.load_var("IFS-FESOM-TCO399", "thetao",
                             period=("1990", "1990"))
        assert "lev" in da.dims
        assert da.sizes["lev"] == 28
        # Check depth values are remapped
        assert da["lev"].values[0] == pytest.approx(7.5)

    @pytest.mark.integration
    def test_time_axis_correct(self, real_config):
        """Time axis starts in January (not February like raw IFS)."""
        import os
        if not os.path.exists(self.TCO399_ROOT):
            pytest.skip("TCO399 data not available")
        import pandas as pd
        loader = GRIBLoader(real_config)
        da = loader.load_var("IFS-FESOM-TCO399", "tas",
                             period=("1990", "1990"))
        t0 = pd.Timestamp(da["time"].values[0])
        assert t0.month == 1
        assert t0.year == 1990
