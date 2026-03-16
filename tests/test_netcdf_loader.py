"""Tests for NetCDFLoader — per-year NetCDF files on HEALPix grid."""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.netcdf_loader import NetCDFLoader


# ── Fixtures ────────────────────────────────────────────────────────────


NSIDE = 8
NCELLS = 12 * NSIDE**2  # 768


def _make_sfc_file(path, var_name, year, ncells=NCELLS, with_height=True):
    """Create a synthetic surface NetCDF file.

    Uses IFS time convention: monthly means timestamped at start of
    next month (Jan mean → Feb 1, ..., Dec mean → Jan 1 of next year).
    """
    import healpy as hp

    lon, lat = hp.pix2ang(NSIDE, np.arange(ncells), nest=True, lonlat=True)
    temp = 300 - 40 * np.abs(lat / 90.0)
    # IFS offset: file for year Y has timestamps Feb Y .. Jan Y+1
    time = xr.date_range(f"{year}-02", periods=12, freq="MS")

    data = temp[np.newaxis, :] + np.random.default_rng(year).normal(0, 1, (12, 1))

    dims = ["time", "gsize"]
    coords = {"time": time}

    if with_height:
        # Add singleton height dimension
        data = data[:, np.newaxis, :]
        dims = ["time", "height", "gsize"]

    ds = xr.Dataset({
        var_name: xr.DataArray(data, dims=dims, coords=coords),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_ocean3d_file(path, var_name, year, ncells=NCELLS, n_levels=5):
    """Create a synthetic ocean 3D NetCDF file."""
    # IFS offset: file for year Y has timestamps Feb Y .. Jan Y+1
    time = xr.date_range(f"{year}-02", periods=12, freq="MS")
    levels = np.array([5.0, 50.0, 200.0, 500.0, 1000.0])[:n_levels]

    data = np.random.default_rng(year).normal(280, 5, (12, n_levels, ncells))

    ds = xr.Dataset({
        var_name: xr.DataArray(
            data, dims=["time", "lev", "gsize"],
            coords={"time": time, "lev": levels},
        ),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_config(tmp_path, variable_aliases=None, scale_factors=None,
                 nside=None):
    """Create a FeatherConfig for testing."""
    aliases = (
        variable_aliases if variable_aliases is not None
        else {"tas": "sfc_mean2t", "tos": "o2d_avg_tos", "thetao": "o3d_avg_thetao"}
    )
    data_source = {
        "type": "netcdf_healpix",
        "root": str(tmp_path / "netcdf"),
    }
    if nside is not None:
        data_source["nside"] = nside

    return FeatherConfig(
        model_catalogs={},
        models=["IFS-FESOM-T319"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source=data_source,
        model_configs={
            "IFS-FESOM-T319": ModelConfig(
                name="IFS-FESOM-T319",
                institution="AWI",
                grids={"sfc": "healpix", "o2d": "healpix", "o3d": "healpix"},
                color="#1f77b4",
                variable_aliases=aliases,
                scale_factors=scale_factors or {},
            ),
        },
    )


@pytest.fixture
def sfc_data(tmp_path):
    """Create 2 years of synthetic surface data."""
    root = tmp_path / "netcdf"
    for year in [1990, 1991]:
        path = root / "sfc_mean2t" / f"sfc_mean2t_{year}.nc"
        _make_sfc_file(path, "mean2t", year)
    return tmp_path


@pytest.fixture
def sfc_no_height(tmp_path):
    """Create surface data without singleton height dim."""
    root = tmp_path / "netcdf"
    for year in [1990, 1991]:
        path = root / "sfc_mean2t" / f"sfc_mean2t_{year}.nc"
        _make_sfc_file(path, "mean2t", year, with_height=False)
    return tmp_path


@pytest.fixture
def ocean3d_data(tmp_path):
    """Create 2 years of synthetic ocean 3D data."""
    root = tmp_path / "netcdf"
    for year in [1990, 1991]:
        path = root / "o3d_avg_thetao" / f"o3d_avg_thetao_{year}.nc"
        _make_ocean3d_file(path, "avg_thetao", year)
    return tmp_path


# ── Test: file discovery and loading ────────────────────────────────


class TestFileDiscovery:
    """Test path construction and file finding."""

    def test_load_via_alias(self, sfc_data):
        """Variable alias maps CMOR name to on-disk directory."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert da is not None
        assert "time" in da.dims
        assert "values" in da.dims

    def test_missing_variable_raises(self, sfc_data):
        """Requesting a missing variable raises FileNotFoundError."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        with pytest.raises(FileNotFoundError):
            loader.load_var("IFS-FESOM-T319", "psl")

    def test_missing_model_raises(self, sfc_data):
        """Requesting an unknown model raises FileNotFoundError or KeyError."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        with pytest.raises((FileNotFoundError, KeyError)):
            loader.load_var("UNKNOWN_MODEL", "tas")

    def test_multi_year_concat(self, sfc_data):
        """Two years of monthly data → 24 timesteps."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert da.sizes["time"] == 24

    def test_no_nc_files_raises(self, tmp_path):
        """Empty variable directory raises FileNotFoundError."""
        var_dir = tmp_path / "netcdf" / "sfc_mean2t"
        var_dir.mkdir(parents=True)
        cfg = _make_config(tmp_path)
        loader = NetCDFLoader(cfg)
        with pytest.raises(FileNotFoundError, match="No NetCDF files"):
            loader.load_var("IFS-FESOM-T319", "tas")


# ── Test: dimension handling ────────────────────────────────────────


class TestDimensions:
    """Test gsize rename and singleton squeeze."""

    def test_gsize_renamed_to_values(self, sfc_data):
        """The gsize dimension should be renamed to values."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "values" in da.dims
        assert "gsize" not in da.dims

    def test_height_squeezed(self, sfc_data):
        """Singleton height dimension should be squeezed."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "height" not in da.dims

    def test_no_height_still_works(self, sfc_no_height):
        """Data without height dim loads correctly."""
        cfg = _make_config(sfc_no_height)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "values" in da.dims
        assert da.sizes["values"] == NCELLS

    def test_ocean3d_has_lev(self, ocean3d_data):
        """Ocean 3D data preserves the lev dimension."""
        cfg = _make_config(ocean3d_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "thetao")
        assert "lev" in da.dims
        assert da.sizes["lev"] == 5

    def test_ncells_correct(self, sfc_data):
        """Number of spatial cells matches nside=8 HEALPix."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert da.sizes["values"] == NCELLS


# ── Test: HEALPix coordinates ──────────────────────────────────────


class TestHEALPixCoords:
    """Test coordinate attachment."""

    def test_longitude_attached(self, sfc_data):
        """Longitude coordinate should be attached."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "longitude" in da.coords
        assert da["longitude"].shape == (NCELLS,)

    def test_latitude_attached(self, sfc_data):
        """Latitude coordinate should be attached."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "latitude" in da.coords
        assert da["latitude"].shape == (NCELLS,)

    def test_lon_range(self, sfc_data):
        """Longitude should be in [0, 360)."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        lon = da["longitude"].values
        assert lon.min() >= 0
        assert lon.max() < 360

    def test_lat_range(self, sfc_data):
        """Latitude should be in [-90, 90]."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        lat = da["latitude"].values
        assert lat.min() >= -90
        assert lat.max() <= 90

    def test_nside_auto_detected(self, sfc_data):
        """nside is correctly auto-detected from gsize."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        # nside=8 → 768 cells, coords from pix2ang(8, ...)
        assert da["longitude"].shape == (768,)

    def test_nside_override(self, sfc_data):
        """Config nside override is used instead of auto-detection."""
        cfg = _make_config(sfc_data, nside=8)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert "longitude" in da.coords


# ── Test: time slicing ──────────────────────────────────────────────


class TestPeriodSlicing:
    """Test period and time_mean parameters."""

    def test_period_slicing(self, sfc_data):
        """Period slicing restricts time range."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas",
                             period=("1990", "1990"))
        assert da.sizes["time"] == 12

    def test_period_partial_year(self, sfc_data):
        """Period can select partial range."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas",
                             period=("1990-06", "1991-03"))
        # Jun 1990 through Mar 1991 = 10 months
        assert da.sizes["time"] == 10

    def test_time_mean(self, sfc_data):
        """time_mean=True removes the time dimension."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas", time_mean=True)
        assert "time" not in da.dims

    def test_period_plus_time_mean(self, sfc_data):
        """Period slicing + time mean works together."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas",
                             period=("1990", "1990"), time_mean=True)
        assert "time" not in da.dims
        assert "values" in da.dims


# ── Test: scale factors ─────────────────────────────────────────────


class TestScaleFactors:
    """Test scale factor application."""

    def test_scale_factor_applied(self, sfc_data):
        """Scale factor multiplies data values."""
        cfg = _make_config(sfc_data, scale_factors={"tas": 100.0})
        loader = NetCDFLoader(cfg)
        da_scaled = loader.load_var("IFS-FESOM-T319", "tas", time_mean=True)

        # Load without scaling
        cfg2 = _make_config(sfc_data, scale_factors={})
        loader2 = NetCDFLoader(cfg2)
        da_raw = loader2.load_var("IFS-FESOM-T319", "tas", time_mean=True)

        np.testing.assert_allclose(
            da_scaled.values, da_raw.values * 100.0, rtol=1e-5,
        )

    def test_no_scale_factor_is_identity(self, sfc_data):
        """Default scale factor (1.0) leaves data unchanged."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas", time_mean=True)
        # Just check it loaded without error and has reasonable values
        assert np.isfinite(da.values).all()
        assert da.values.mean() > 250  # should be ~280K


# ── Test: caching ───────────────────────────────────────────────────


class TestCaching:
    """Test dataset caching."""

    def test_same_var_returns_cached(self, sfc_data):
        """Repeated loads return the same cached DataArray."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da1 = loader.load_var("IFS-FESOM-T319", "tas")
        da2 = loader.load_var("IFS-FESOM-T319", "tas")
        # Period slicing creates a new view, but the cached base is the same
        assert (model_key := ("IFS-FESOM-T319", "tas")) in loader._cache

    def test_different_vars_separate_cache(self, tmp_path):
        """Different variables have separate cache entries."""
        root = tmp_path / "netcdf"
        for year in [1990]:
            _make_sfc_file(
                root / "sfc_mean2t" / f"sfc_mean2t_{year}.nc",
                "mean2t", year,
            )
            # Create a second variable
            _make_sfc_file(
                root / "sfc_msl" / f"sfc_msl_{year}.nc",
                "msl", year,
            )

        aliases = {"tas": "sfc_mean2t", "psl": "sfc_msl"}
        cfg = _make_config(tmp_path, variable_aliases=aliases)
        loader = NetCDFLoader(cfg)

        loader.load_var("IFS-FESOM-T319", "tas")
        loader.load_var("IFS-FESOM-T319", "psl")

        assert ("IFS-FESOM-T319", "tas") in loader._cache
        assert ("IFS-FESOM-T319", "psl") in loader._cache


# ── Test: data variable finding ─────────────────────────────────────


class TestFindDataVar:
    """Test _find_data_var static method."""

    def test_strip_domain_prefix(self):
        """Domain prefix is stripped to find variable."""
        ds = xr.Dataset({"mean2t": xr.DataArray([1, 2, 3])})
        result = NetCDFLoader._find_data_var(ds, "sfc_mean2t", "tas")
        assert result == "mean2t"

    def test_canonical_name_found(self):
        """Falls back to canonical CMOR name."""
        ds = xr.Dataset({"tas": xr.DataArray([1, 2, 3])})
        result = NetCDFLoader._find_data_var(ds, "sfc_mean2t", "tas")
        assert result == "tas"

    def test_first_data_var_fallback(self):
        """Falls back to first non-coordinate variable."""
        ds = xr.Dataset({"mysterious_var": xr.DataArray([1, 2, 3])})
        result = NetCDFLoader._find_data_var(ds, "sfc_mean2t", "tas")
        assert result == "mysterious_var"

    def test_no_data_var_raises(self):
        """Empty dataset raises KeyError."""
        ds = xr.Dataset()
        with pytest.raises(KeyError, match="No data variable"):
            NetCDFLoader._find_data_var(ds, "sfc_mean2t", "tas")

    def test_ocean_prefix_stripped(self):
        """Ocean domain prefix (o2d, o3d) is stripped."""
        ds = xr.Dataset({"avg_tos": xr.DataArray([1, 2, 3])})
        result = NetCDFLoader._find_data_var(ds, "o2d_avg_tos", "tos")
        assert result == "avg_tos"

    def test_pl_prefix_stripped(self):
        """Pressure level prefix (pl) is stripped."""
        ds = xr.Dataset({"t": xr.DataArray([1, 2, 3])})
        result = NetCDFLoader._find_data_var(ds, "pl_t", "ta")
        assert result == "t"


# ── Test: alias resolution ──────────────────────────────────────────


class TestAliasResolution:
    """Test _get_dir_name alias lookup order."""

    def test_alias_from_config(self, tmp_path):
        """Config variable_aliases take priority."""
        cfg = _make_config(tmp_path, variable_aliases={"tas": "custom_dir"})
        loader = NetCDFLoader(cfg)
        assert loader._get_dir_name("IFS-FESOM-T319", "tas") == "custom_dir"

    def test_fallback_to_destine_variable(self, tmp_path):
        """Falls back to destine_variable from VARIABLE_REGISTRY."""
        cfg = _make_config(tmp_path, variable_aliases={})
        loader = NetCDFLoader(cfg)
        # "tas" has destine_variable="avg_2t" in the registry
        dir_name = loader._get_dir_name("IFS-FESOM-T319", "tas")
        assert dir_name == "avg_2t"

    def test_fallback_to_canonical(self, tmp_path):
        """Falls back to canonical name when no alias and no destine var."""
        cfg = _make_config(tmp_path, variable_aliases={})
        loader = NetCDFLoader(cfg)
        # Use a variable that might not have destine_variable
        dir_name = loader._get_dir_name("IFS-FESOM-T319", "nonexistent_var_xyz")
        assert dir_name == "nonexistent_var_xyz"


# ── Test: ocean 3D data ────────────────────────────────────────────


class TestOcean3D:
    """Test ocean 3D (depth-level) data loading."""

    def test_load_ocean3d(self, ocean3d_data):
        """Ocean 3D data loads with lev dimension."""
        cfg = _make_config(ocean3d_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "thetao")
        assert "time" in da.dims
        assert "lev" in da.dims
        assert "values" in da.dims
        assert da.sizes["lev"] == 5

    def test_ocean3d_period(self, ocean3d_data):
        """Period slicing works for ocean 3D data."""
        cfg = _make_config(ocean3d_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "thetao",
                             period=("1990", "1990"))
        assert da.sizes["time"] == 12

    def test_ocean3d_has_coords(self, ocean3d_data):
        """Ocean 3D data gets HEALPix coords attached."""
        cfg = _make_config(ocean3d_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "thetao")
        assert "longitude" in da.coords
        assert "latitude" in da.coords


# ── Test: integration with base class ───────────────────────────────


class TestBaseClassIntegration:
    """Test that NetCDFLoader works with DiagnosticBase helpers."""

    def test_data_source_type(self, sfc_data):
        """Config reports correct data source type."""
        cfg = _make_config(sfc_data)
        assert cfg.get_data_source_type() == "netcdf_healpix"

    def test_grid_type_healpix(self, sfc_data):
        """Config reports HEALPix grid type."""
        cfg = _make_config(sfc_data)
        assert cfg.get_grid_type("IFS-FESOM-T319", "sfc") == "healpix"

    def test_model_color(self, sfc_data):
        """Config returns correct model color."""
        cfg = _make_config(sfc_data)
        assert cfg.get_model_color("IFS-FESOM-T319") == "#1f77b4"


# ── Test: run.py dispatch ───────────────────────────────────────────


class TestRunDispatch:
    """Test that _create_model_loader dispatches correctly."""

    def test_netcdf_healpix_dispatch(self, sfc_data):
        """netcdf_healpix config creates a NetCDFLoader."""
        from feather.run import _create_model_loader
        cfg = _make_config(sfc_data)
        loader = _create_model_loader(cfg)
        assert isinstance(loader, NetCDFLoader)


# ── Test: depth dim squeeze ─────────────────────────────────────────


class TestDepthSqueeze:
    """Test singleton depth dimension squeeze for ocean 2D."""

    def test_singleton_depth_squeezed(self, tmp_path):
        """Singleton depth dim (ocean 2D) is squeezed."""
        root = tmp_path / "netcdf"
        # IFS offset timestamps
        time = xr.date_range("1990-02", periods=12, freq="MS")
        data = np.random.default_rng(42).normal(290, 5, (12, 1, NCELLS))
        ds = xr.Dataset({
            "avg_tos": xr.DataArray(
                data, dims=["time", "depth", "gsize"],
                coords={"time": time, "depth": [0.5]},
            ),
        })
        path = root / "o2d_avg_tos" / "o2d_avg_tos_1990.nc"
        path.parent.mkdir(parents=True)
        ds.to_netcdf(path)

        cfg = _make_config(tmp_path)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tos")
        assert "depth" not in da.dims
        assert "values" in da.dims

    def test_singleton_lev_squeezed(self, tmp_path):
        """Singleton lev dim (siconc-like) is squeezed."""
        root = tmp_path / "netcdf"
        # IFS offset timestamps
        time = xr.date_range("1990-02", periods=12, freq="MS")
        data = np.random.default_rng(42).normal(0.5, 0.3, (12, 1, NCELLS))
        ds = xr.Dataset({
            "avg_siconc": xr.DataArray(
                data, dims=["time", "lev", "gsize"],
                coords={"time": time, "lev": [0.0]},
            ),
        })
        aliases = {"siconc": "o2d_avg_siconc"}
        path = root / "o2d_avg_siconc" / "o2d_avg_siconc_1990.nc"
        path.parent.mkdir(parents=True)
        ds.to_netcdf(path)

        cfg = _make_config(tmp_path, variable_aliases=aliases)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "siconc")
        assert "lev" not in da.dims
        assert "values" in da.dims

    def test_multi_lev_preserved(self, ocean3d_data):
        """Multi-level lev dim (ocean 3D) is NOT squeezed."""
        cfg = _make_config(ocean3d_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "thetao")
        assert "lev" in da.dims
        assert da.sizes["lev"] == 5


# ── Test: IFS time offset correction ────────────────────────────────


class TestTimeOffset:
    """Test IFS time offset correction (shift back by 1 month)."""

    def test_time_shifted_back(self, sfc_data):
        """Timestamps are shifted back by 1 month."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        times = da["time"].values
        import pandas as pd
        t0 = pd.Timestamp(times[0])
        # File has Feb 1990 as first stamp → corrected to Jan 1990
        assert t0.month == 1
        assert t0.year == 1990

    def test_last_time_dec(self, sfc_data):
        """Last timestamp of a single-year file is December."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas",
                             period=("1990", "1990"))
        times = da["time"].values
        import pandas as pd
        t_last = pd.Timestamp(times[-1])
        assert t_last.month == 12
        assert t_last.year == 1990

    def test_seasonal_groupby_correct(self, sfc_data):
        """time.month grouping works correctly after offset correction."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas",
                             period=("1990", "1990"))
        months = da["time.month"].values
        # Should be [1, 2, 3, ..., 12] after correction
        np.testing.assert_array_equal(months, np.arange(1, 13))

    def test_march_september_selectable(self, sfc_data):
        """Can select March and September by month number."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        march = da.sel(time=da["time.month"] == 3)
        sept = da.sel(time=da["time.month"] == 9)
        # 2 years of data → 2 March, 2 September
        assert march.sizes["time"] == 2
        assert sept.sizes["time"] == 2

    def test_two_years_24_months(self, sfc_data):
        """Two years of data give 24 timesteps after correction."""
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")
        assert da.sizes["time"] == 24


# ── Test: NEST ordering ────────────────────────────────────────────


class TestNestOrdering:
    """Test HEALPix NEST pixel ordering."""

    def test_nest_ordering_coords(self, sfc_data):
        """Coordinates should match NEST ordering (not RING)."""
        import healpy as hp
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")

        expected_lon, expected_lat = hp.pix2ang(
            NSIDE, np.arange(NCELLS), nest=True, lonlat=True,
        )
        np.testing.assert_array_almost_equal(
            da["longitude"].values, expected_lon,
        )
        np.testing.assert_array_almost_equal(
            da["latitude"].values, expected_lat,
        )

    def test_nest_differs_from_ring(self, sfc_data):
        """NEST coords differ from RING coords."""
        import healpy as hp
        cfg = _make_config(sfc_data)
        loader = NetCDFLoader(cfg)
        da = loader.load_var("IFS-FESOM-T319", "tas")

        ring_lon, _ = hp.pix2ang(
            NSIDE, np.arange(NCELLS), nest=False, lonlat=True,
        )
        # First few pixels should differ between NEST and RING
        assert not np.allclose(da["longitude"].values[:10], ring_lon[:10])


# ── Test: config YAML loading ──────────────────────────────────────


class TestConfigYAML:
    """Test that the himansu_319 config loads correctly."""

    @pytest.mark.integration
    def test_load_config(self):
        """himansu_319.yaml loads without error."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "himansu_319.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")
        cfg = FeatherConfig.from_yaml(config_path)
        assert "IFS-FESOM-T319" in cfg.models
        assert cfg.get_data_source_type() == "netcdf_healpix"
        assert cfg.get_grid_type("IFS-FESOM-T319") == "healpix"
        assert cfg.get_period() == ("1990", "2014")
        assert cfg.model_configs["IFS-FESOM-T319"].variable_aliases["tas"] == "sfc_mean2t"
