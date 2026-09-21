"""Tests for Berkeley Earth land-only masking.

The Berkeley Earth Land+Ocean product reports SST over the ocean rather
than 2 m air temperature, so ``temperature_berkeley`` and ``added_value``
restrict the comparison to land.  These tests cover the shared helpers in
``feather.diag._berkeley``, the ``ObsLoader`` entry point, and the two
diagnostics' use of them.
"""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag import _berkeley


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def be_hr_nc(tmp_path):
    """Berkeley Earth HR-style file with a land_mask (land east of 0degE)."""
    lats = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
    lons = np.array([-180.0, -90.0, 0.0, 90.0], dtype=np.float32)
    dec_years = np.array([1990 + (m - 0.5) / 12 for m in range(1, 13)])

    anom = np.zeros((12, len(lats), len(lons)), dtype=np.float32)
    clim_base = 15.0 - 40.0 * np.abs(lats[:, None] / 90.0)
    clim = np.broadcast_to(
        clim_base[None, :, :], (12, len(lats), len(lons)),
    ).copy().astype(np.float32)

    # Land where lon >= 0 (i.e. the last two columns)
    land = np.zeros((len(lats), len(lons)), dtype=np.float32)
    land[:, 2:] = 1.0

    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                anom, dims=("time", "latitude", "longitude")),
            "climatology": xr.DataArray(
                clim, dims=("month_number", "latitude", "longitude")),
            "land_mask": xr.DataArray(land, dims=("latitude", "longitude")),
        },
        coords={"latitude": lats, "longitude": lons, "time": dec_years},
    )
    path = tmp_path / "Global_TAVG_Gridded_0p25deg.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def be_hr_nc_no_mask(tmp_path):
    """Same file but without the land_mask variable."""
    lats = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
    lons = np.array([-180.0, -90.0, 0.0, 90.0], dtype=np.float32)
    dec_years = np.array([1990 + (m - 0.5) / 12 for m in range(1, 13)])
    anom = np.zeros((12, len(lats), len(lons)), dtype=np.float32)
    clim = np.zeros((12, len(lats), len(lons)), dtype=np.float32)
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                anom, dims=("time", "latitude", "longitude")),
            "climatology": xr.DataArray(
                clim, dims=("month_number", "latitude", "longitude")),
        },
        coords={"latitude": lats, "longitude": lons, "time": dec_years},
    )
    path = tmp_path / "no_mask.nc"
    ds.to_netcdf(path)
    return path


def _config(nc_path, tmp_path, **project):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={
            "BERKELEY_EARTH_HR": {
                "path": str(nc_path.parent),
                "variables": {"temperature": nc_path.name},
            },
        },
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        project=project,
    )


# ── Config knobs ──────────────────────────────────────────────────────


class TestConfigKnobs:

    def test_land_only_defaults_true(self, be_hr_nc, tmp_path):
        assert _berkeley.land_only_enabled(_config(be_hr_nc, tmp_path))

    def test_land_only_can_be_disabled(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_only=False)
        assert not _berkeley.land_only_enabled(cfg)

    def test_threshold_defaults_to_half(self, be_hr_nc, tmp_path):
        assert _berkeley.land_threshold(_config(be_hr_nc, tmp_path)) == 0.5

    def test_threshold_override(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_threshold=0.9)
        assert _berkeley.land_threshold(cfg) == 0.9


# ── Shared loader ─────────────────────────────────────────────────────


class TestLoadBerkeleyHR:

    def test_unmasked_by_default(self, be_hr_nc, tmp_path):
        da = _berkeley.load_berkeley_hr(_config(be_hr_nc, tmp_path))
        assert np.isfinite(da.values).all()

    def test_land_only_masks_ocean(self, be_hr_nc, tmp_path):
        da = _berkeley.load_berkeley_hr(
            _config(be_hr_nc, tmp_path), land_only=True,
        )
        # After the 0..360 shift, lons are [0, 90, 180, 270]; land is the
        # original lon >= 0 columns, i.e. 0 and 90.
        land = da.sel(lon=[0.0, 90.0])
        ocean = da.sel(lon=[180.0, 270.0])
        assert np.isfinite(land.values).all()
        assert np.isnan(ocean.values).all()

    def test_returns_kelvin(self, be_hr_nc, tmp_path):
        da = _berkeley.load_berkeley_hr(_config(be_hr_nc, tmp_path))
        # climatology at the equator is 15 degC -> 288.15 K
        assert float(da.sel(lat=0.0).isel(time=0, lon=0)) == pytest.approx(288.15)

    def test_dims_renamed_and_lons_shifted(self, be_hr_nc, tmp_path):
        da = _berkeley.load_berkeley_hr(_config(be_hr_nc, tmp_path))
        assert "lat" in da.dims and "lon" in da.dims
        assert float(da.lon.min()) >= 0.0

    def test_missing_land_mask_warns_not_raises(
        self, be_hr_nc_no_mask, tmp_path, caplog,
    ):
        cfg = _config(be_hr_nc_no_mask, tmp_path)
        da = _berkeley.load_berkeley_hr(cfg, land_only=True)
        assert np.isfinite(da.values).all()
        assert "no land_mask" in caplog.text

    def test_period_slicing(self, be_hr_nc, tmp_path):
        da = _berkeley.load_berkeley_hr(
            _config(be_hr_nc, tmp_path), period=("1990-01", "1990-03"),
        )
        assert da.sizes["time"] == 3


class TestLoadLandFraction:

    def test_returns_fraction_on_0_360(self, be_hr_nc, tmp_path):
        frac = _berkeley.load_land_fraction(_config(be_hr_nc, tmp_path))
        assert frac is not None
        assert "lat" in frac.dims and "lon" in frac.dims
        assert float(frac.lon.min()) >= 0.0

    def test_none_when_dataset_absent(self, tmp_path):
        cfg = FeatherConfig(
            model_catalogs={}, models=[], obs_root="", obs_datasets={},
            cmip6={"enabled": False}, dask={}, nereus={},
            output_dir=str(tmp_path),
        )
        assert _berkeley.load_land_fraction(cfg) is None

    def test_none_when_variable_absent(self, be_hr_nc_no_mask, tmp_path):
        cfg = _config(be_hr_nc_no_mask, tmp_path)
        assert _berkeley.load_land_fraction(cfg) is None


# ── Masking helpers ───────────────────────────────────────────────────


def _latlon_field(lats, lons):
    return xr.DataArray(
        np.ones((len(lats), len(lons))),
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
    )


class TestMaskLatlonToLand:

    def test_noop_without_fraction(self):
        da = _latlon_field([0.0, 45.0], [0.0, 180.0])
        assert _berkeley.mask_latlon_to_land(da, None) is da

    def test_masks_ocean_cells(self, be_hr_nc, tmp_path):
        frac = _berkeley.load_land_fraction(_config(be_hr_nc, tmp_path))
        da = _latlon_field(
            np.array([-45.0, 0.0, 45.0]),
            np.array([0.0, 90.0, 180.0, 270.0]),
        )
        out = _berkeley.mask_latlon_to_land(da, frac)
        assert np.isfinite(out.sel(lon=[0.0, 90.0]).values).all()
        assert np.isnan(out.sel(lon=[180.0, 270.0]).values).all()


class _NoSftlfLoader:
    def load_var(self, *a, **k):
        raise KeyError("sftlf")


class _SftlfLoader:
    """Model loader publishing an sftlf that calls everything land."""

    def __init__(self, lats, lons):
        self._sftlf = xr.DataArray(
            np.full((len(lats), len(lons)), 100.0),
            dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        )

    def load_var(self, model, variable, **kwargs):
        if variable != "sftlf":
            raise KeyError(variable)
        return self._sftlf


class TestModelLandMask:

    def test_falls_back_to_berkeley_mask(self, be_hr_nc, tmp_path):
        frac = _berkeley.load_land_fraction(_config(be_hr_nc, tmp_path))
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0])
        da = _latlon_field(lats, lons)
        mask = _berkeley.model_land_mask(
            _NoSftlfLoader(), "M", da, lons, lats, "latlon", frac,
        )
        assert mask is not None
        assert mask.sel(lon=0.0).values.all()
        assert not mask.sel(lon=180.0).values.any()

    def test_prefers_model_sftlf(self, be_hr_nc, tmp_path):
        frac = _berkeley.load_land_fraction(_config(be_hr_nc, tmp_path))
        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0])
        da = _latlon_field(lats, lons)
        mask = _berkeley.model_land_mask(
            _SftlfLoader(lats, lons), "M", da, lons, lats, "latlon", frac,
        )
        # sftlf says 100 % land everywhere, overriding the Berkeley mask.
        assert mask.values.all()

    def test_none_without_any_source(self):
        lats = np.array([-45.0, 0.0])
        lons = np.array([0.0, 180.0])
        da = _latlon_field(lats, lons)
        assert _berkeley.model_land_mask(
            _NoSftlfLoader(), "M", da, lons, lats, "latlon", None,
        ) is None

    def test_healpix_pointwise_sampling(self, be_hr_nc, tmp_path):
        frac = _berkeley.load_land_fraction(_config(be_hr_nc, tmp_path))
        lon = np.array([0.0, 90.0, 180.0, 270.0])
        lat = np.array([0.0, 0.0, 0.0, 0.0])
        da = xr.DataArray(np.ones(4), dims=("values",))
        mask = _berkeley.model_land_mask(
            _NoSftlfLoader(), "M", da, lon, lat, "healpix", frac,
        )
        assert mask.dims == ("values",)
        assert list(mask.values) == [True, True, False, False]


# ── ObsLoader entry point ─────────────────────────────────────────────


class TestObsLoaderLandOnly:

    def test_land_only_masks_ocean(self, be_hr_nc, tmp_path):
        from feather.data.obs import ObsLoader

        loader = ObsLoader(_config(be_hr_nc, tmp_path))
        da = loader.load_berkeley_hr("BERKELEY_EARTH_HR", land_only=True)
        assert np.isnan(da.sel(lon=180.0).values).all()
        assert np.isfinite(da.sel(lon=0.0).values).all()

    def test_default_is_unmasked(self, be_hr_nc, tmp_path):
        from feather.data.obs import ObsLoader

        loader = ObsLoader(_config(be_hr_nc, tmp_path))
        da = loader.load_berkeley_hr("BERKELEY_EARTH_HR")
        assert np.isfinite(da.values).all()


# ── Diagnostic wiring ─────────────────────────────────────────────────


class _StubModelLoader:
    def load_var(self, *a, **k):
        raise KeyError("no data")


class _StubObsLoader:
    def load(self, *a, **k):
        raise KeyError("no obs")

    def load_for_model_var(self, *a, **k):
        raise KeyError("no obs")


def _temp_diag(cfg):
    from feather.diag.temperature_berkeley import TemperatureBerkeley

    return TemperatureBerkeley(
        _StubModelLoader(), _StubObsLoader(), cfg,
        experiment="baseline_hist", period=("1990", "1990"),
    )


class TestTemperatureBerkeleyWiring:

    def test_land_only_on_by_default(self, be_hr_nc, tmp_path):
        assert _temp_diag(_config(be_hr_nc, tmp_path)).land_only

    def test_land_only_respects_config(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_only=False)
        assert not _temp_diag(cfg).land_only

    def test_label_marks_land(self, be_hr_nc, tmp_path):
        diag = _temp_diag(_config(be_hr_nc, tmp_path))
        assert diag._berkeley_label == "Berkeley Earth HR (land)"

    def test_label_plain_when_global(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_only=False)
        assert _temp_diag(cfg)._berkeley_label == "Berkeley Earth HR"

    def test_obs_is_land_masked(self, be_hr_nc, tmp_path):
        diag = _temp_diag(_config(be_hr_nc, tmp_path))
        da = diag._load_berkeley_earth()
        assert np.isnan(da.sel(lon=180.0).values).all()

    def test_mask_model_noop_when_disabled(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_only=False)
        diag = _temp_diag(cfg)
        da = _latlon_field(np.array([0.0, 45.0]), np.array([0.0, 180.0]))
        assert diag._mask_model_to_land("M", da) is da

    def test_field_hook_none_when_disabled(self, be_hr_nc, tmp_path):
        cfg = _config(be_hr_nc, tmp_path, berkeley_land_only=False)
        assert _temp_diag(cfg)._land_field_hook is None

    def test_field_hook_masks_ocean(self, be_hr_nc, tmp_path):
        diag = _temp_diag(_config(be_hr_nc, tmp_path))
        hook = diag._land_field_hook
        assert hook is not None
        da = _latlon_field(
            np.array([-45.0, 0.0, 45.0]),
            np.array([0.0, 90.0, 180.0, 270.0]),
        )
        assert np.isnan(hook(da).sel(lon=270.0).values).all()

    def test_common_grid_mask(self, be_hr_nc, tmp_path):
        diag = _temp_diag(_config(be_hr_nc, tmp_path))
        da = _latlon_field(
            np.array([-45.0, 0.0, 45.0]),
            np.array([0.0, 90.0, 180.0, 270.0]),
        )
        out = diag._mask_common_to_land(da)
        assert np.isfinite(out.sel(lon=0.0).values).all()
        assert np.isnan(out.sel(lon=180.0).values).all()


# ── Added Value NaN handling ──────────────────────────────────────────


class TestAddedValueNaN:

    def test_av_ratio_preserves_nan(self):
        from feather.diag.added_value import AddedValueDiag

        sq1 = np.array([4.0, np.nan, 1.0])
        sq2 = np.array([1.0, 2.0, np.nan])
        av = AddedValueDiag._av_ratio(sq1, sq2)
        assert av[0] == pytest.approx(0.75)
        assert np.isnan(av[1])
        assert np.isnan(av[2])

    def test_av_ratio_zero_when_both_perfect(self):
        from feather.diag.added_value import AddedValueDiag

        av = AddedValueDiag._av_ratio(np.array([0.0]), np.array([0.0]))
        assert av[0] == 0.0

    def test_compute_av_nan_where_ref_is_nan(self):
        from feather.diag.added_value import AddedValueDiag

        coords = {"lat": [0.0, 1.0]}
        m1 = xr.DataArray([290.0, 290.0], dims="lat", coords=coords)
        m2 = xr.DataArray([288.0, 288.0], dims="lat", coords=coords)
        ref = xr.DataArray([288.0, np.nan], dims="lat", coords=coords)
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert float(av[0]) == pytest.approx(1.0)
        assert np.isnan(float(av[1]))

    def test_av_from_biases_nan_propagates(self):
        from feather.diag.added_value import AddedValueDiag

        coords = {"lat": [0.0, 1.0]}
        b1 = xr.DataArray([2.0, np.nan], dims="lat", coords=coords)
        b2 = xr.DataArray([1.0, 1.0], dims="lat", coords=coords)
        av = AddedValueDiag._av_from_biases(b1, b2)
        assert float(av[0]) == pytest.approx(0.75)
        assert np.isnan(float(av[1]))
