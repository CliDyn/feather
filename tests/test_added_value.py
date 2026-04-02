"""Tests for AddedValueDiag (Dosio et al. 2015).

All tests use small synthetic data (nside=8 HEALPix / 5° lat-lon) and
the MockCMIP6Loader from conftest.py — no real data needed.
"""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.added_value import AddedValueDiag


# ── Helpers / fixtures ───────────────────────────────────────────────


def _make_latlon(value, lats=None, lons=None):
    """Return a constant 2-D DataArray on a coarse lat/lon grid."""
    if lats is None:
        lats = np.arange(-87.5, 90, 5.0)
    if lons is None:
        lons = np.arange(2.5, 360, 5.0)
    data = np.full((len(lats), len(lons)), value, dtype=float)
    return xr.DataArray(
        data, dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )


@pytest.fixture
def eerie_config(tmp_path):
    """Minimal FeatherConfig with two EERIE-style latlon models."""
    from feather.config import ModelConfig
    mc_a = ModelConfig(
        name="ModelA", institution="INS", experiment="hist-1950",
        variant="r1i1p1f1",
        grids={"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
        color="#1f77b4", data_root="", grid_label="gr",
        variable_aliases={}, scale_factors={},
        absolute_salinity=False, data_source_type="cmor",
        catalog_key="", member=1,
    )
    mc_b = ModelConfig(
        name="ModelB", institution="INS", experiment="hist-1950",
        variant="r1i1p1f1",
        grids={"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
        color="#ff7f0e", data_root="", grid_label="gr",
        variable_aliases={}, scale_factors={},
        absolute_salinity=False, data_source_type="cmor",
        catalog_key="", member=1,
    )
    return FeatherConfig(
        model_catalogs={},
        models={"ModelA": mc_a, "ModelB": mc_b},
        model_configs={"ModelA": mc_a, "ModelB": mc_b},
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": True},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
        data_source={"type": "cmor"},
    )


class MockCMORLoader:
    """Minimal loader returning a constant lat/lon climatology per model."""

    def __init__(self, synth_obs_ds):
        self._ds = synth_obs_ds

    def load_var(self, model, variable, *, period=None, time_mean=False):
        da = self._ds["t2m"]
        if period and "time" in da.dims:
            da = da.sel(time=slice(*period))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da


class MockObsLoaderLatlon:
    def __init__(self, ds):
        self._ds = ds

    def load_for_model_var(self, model_var, period=None):
        da = self._ds["t2m"]
        if period and "time" in da.dims:
            da = da.sel(time=slice(*period))
        return da


# ── Unit tests for _compute_av ──────────────────────────────────────


class TestComputeAv:
    def test_perfect_cmip6_returns_minus_one(self):
        """When m1=CMIP6=ref (perfect CMIP6), AV=-1 (CMIP6 better)."""
        ref = _make_latlon(300.0)
        m1 = ref.copy()           # CMIP6 = perfect
        m2 = _make_latlon(302.0)  # EERIE has error
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert np.allclose(av.values, -1.0)

    def test_perfect_eerie_returns_plus_one(self):
        """When m2=EERIE=ref (perfect EERIE), AV=+1 (EERIE adds value)."""
        ref = _make_latlon(300.0)
        m1 = _make_latlon(302.0)  # CMIP6 has error
        m2 = ref.copy()           # EERIE = perfect
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert np.allclose(av.values, 1.0)

    def test_both_same_error_returns_zero(self):
        """When |m1-ref| = |m2-ref|, AV = 0."""
        ref = _make_latlon(300.0)
        m1 = _make_latlon(302.0)
        m2 = _make_latlon(298.0)  # same absolute error, different sign
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert np.allclose(av.values, 0.0)

    def test_both_equal_ref_returns_zero(self):
        """When both m1 and m2 equal ref, denom=0 → AV=0 (no division by zero)."""
        ref = _make_latlon(300.0)
        m1 = ref.copy()
        m2 = ref.copy()
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert np.all(np.isfinite(av.values))
        assert np.allclose(av.values, 0.0)

    def test_av_within_bounds(self):
        """AV must lie strictly in [-1, 1]."""
        rng = np.random.default_rng(42)
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        shape = (len(lats), len(lons))
        ref = xr.DataArray(
            rng.normal(300, 5, shape), dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        m1 = xr.DataArray(
            ref.values + rng.normal(0, 3, shape), dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        m2 = xr.DataArray(
            ref.values + rng.normal(0, 3, shape), dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert float(av.min()) >= -1.0 - 1e-10
        assert float(av.max()) <= 1.0 + 1e-10

    def test_output_coords_match_input(self):
        """AV DataArray has the same dims/coords as m1."""
        ref = _make_latlon(300.0)
        m1 = _make_latlon(301.0)
        m2 = _make_latlon(302.0)
        av = AddedValueDiag._compute_av(m1, m2, ref)
        assert av.dims == m1.dims
        assert set(av.coords) == set(m1.coords)


# ── Unit tests for domain statistics ─────────────────────────────────


class TestDomainStats:
    def test_frac_positive_all_positive(self):
        av = _make_latlon(0.5)
        assert AddedValueDiag._frac_positive(av) == pytest.approx(1.0)

    def test_frac_positive_all_negative(self):
        av = _make_latlon(-0.5)
        assert AddedValueDiag._frac_positive(av) == pytest.approx(0.0)

    def test_frac_positive_half(self):
        lats = np.array([-2.5, 2.5])
        lons = np.array([2.5])
        data = np.array([[0.5], [-0.5]])
        av = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        frac = AddedValueDiag._frac_positive(av)
        assert frac == pytest.approx(0.5)

    def test_domain_mean_av_constant_field(self):
        """Domain-mean of a constant field equals that constant."""
        from feather.util.spatial import compute_latlon_areas
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        av = _make_latlon(0.3, lats, lons)
        area = compute_latlon_areas(lats, lons)
        result = AddedValueDiag._domain_mean_av(av, area)
        assert result == pytest.approx(0.3, rel=1e-4)


# ── Integration test: compute + plot ─────────────────────────────────


class TestAddedValueDiagIntegration:
    """End-to-end test using synthetic data and mock loaders."""

    @pytest.fixture
    def diag(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        cmip6_loader = MockCMIP6Loader(synth_cmip6)
        obs_loader = MockObsLoaderLatlon(synth_obs)
        model_loader = MockCMORLoader(synth_obs)
        return AddedValueDiag(
            model_loader, obs_loader, eerie_config,
            cmip6_loader=cmip6_loader,
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_compute_returns_expected_keys(self, diag):
        results = diag.compute()
        assert "tas" in results
        vr = results["tas"]
        assert "av" in vr
        assert "annual" in vr["av"]
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            assert etype in vr["av"]["annual"], f"missing key: {etype}"

    def test_av_dims_and_bounds(self, diag):
        results = diag.compute()
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            av = results["tas"]["av"]["annual"][etype]
            assert set(av.dims) == {"lat", "lon"}, etype
            assert float(av.min()) >= -1.0 - 1e-6, etype
            assert float(av.max()) <= 1.0 + 1e-6, etype

    def test_summary_stats_in_result(self, diag):
        results = diag.compute()
        av = results["tas"]["av"]["annual"]
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            assert f"{etype}_domain_av" in av
            assert f"{etype}_frac_positive" in av
            assert np.isfinite(av[f"{etype}_domain_av"])
            assert 0.0 <= av[f"{etype}_frac_positive"] <= 1.0

    def test_nc_files_saved(self, diag, tmp_path):
        diag.compute()
        nc_dir = diag.nc_dir
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            assert (nc_dir / f"tas_annual_{etype}_av.nc").exists(), etype

    def test_nc_file_contents(self, diag, tmp_path):
        diag.compute()
        import xarray as xr
        ds = xr.open_dataset(diag.nc_dir / "tas_annual_ensemble_mean_av.nc")
        assert "av" in ds
        av = ds["av"]
        assert "lat" in av.dims and "lon" in av.dims
        assert av.attrs["units"] == "1"
        assert "n_eerie_models" in av.attrs
        assert "n_cmip6_models" in av.attrs
        assert "reference" in av.attrs
        assert "Dosio" in av.attrs["reference"]
        ds.close()

    def test_plot_returns_two_panels_per_period(self, diag):
        results = diag.compute()
        figures = diag.plot(results)
        assert len(figures) >= 1
        for fig, meta in figures:
            import matplotlib.pyplot as plt
            assert isinstance(fig, plt.Figure)
            assert "figure_id" in meta
            assert "added_value" in meta["figure_id"]

    def test_skip_when_nc_exists(self, diag, tmp_path, monkeypatch):
        """_all_nc_exist returns True only when all 6 NC files are present."""
        var = "tas"
        # Before computation no NC files exist
        assert not diag._all_nc_exist(var)
        diag.compute()
        assert diag._all_nc_exist(var)

    def test_load_from_nc(self, diag):
        diag.compute()
        loaded = diag._load_variable_from_nc("tas")
        assert loaded is not None
        assert "av" in loaded
        assert "annual" in loaded["av"]
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            av = loaded["av"]["annual"][etype]
            assert float(av.min()) >= -1.0 - 1e-6, etype

    def test_no_cmip6_returns_none(self, synth_obs, eerie_config):
        """Without CMIP6 loader, compute returns None for all variables."""
        obs_loader = MockObsLoaderLatlon(synth_obs)
        model_loader = MockCMORLoader(synth_obs)
        diag = AddedValueDiag(
            model_loader, obs_loader, eerie_config,
            cmip6_loader=None,
            variables=["tas"],
        )
        result = diag._compute_variable("tas")
        assert result is None

    def test_metadata_fields(self, diag):
        results = diag.compute()
        figures = diag.plot(results)
        _, meta = figures[0]
        assert meta["diagnostic_name"] == "added_value"
        assert "summary_statistics" in meta
        stats = meta["summary_statistics"]
        for etype in ("ensemble_mean", "ensemble_median",
                      "individual_mean", "individual_median"):
            assert etype in stats, f"missing {etype}"
            assert "domain_mean_av" in stats[etype]
            assert "frac_positive" in stats[etype]


# ── run() orchestration tests ─────────────────────────────────────────


class TestAddedValueRun:
    @pytest.fixture
    def diag(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        cmip6_loader = MockCMIP6Loader(synth_cmip6)
        obs_loader = MockObsLoaderLatlon(synth_obs)
        model_loader = MockCMORLoader(synth_obs)
        return AddedValueDiag(
            model_loader, obs_loader, eerie_config,
            cmip6_loader=cmip6_loader,
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_run_returns_figure_paths(self, diag):
        saved = diag.run(skip_existing=False)
        assert len(saved) >= 1
        png_path, json_path = saved[0]
        assert png_path.suffix == ".png"
        assert json_path.suffix == ".json"
        assert png_path.exists()
        assert json_path.exists()

    def test_run_skip_existing(self, diag):
        diag.run(skip_existing=False)
        # Second run should skip (figures exist)
        call_count = [0]
        orig = diag._compute_variable

        def counting_compute(var):
            call_count[0] += 1
            return orig(var)

        diag._compute_variable = counting_compute
        diag.run(skip_existing=True)
        assert call_count[0] == 0  # no recomputation

    def test_run_nc_checkpoint_skips_compute(self, diag, tmp_path):
        """If NC files exist but figures don't, loading from NC is used."""
        diag.compute()  # saves NC files
        # Remove figures to simulate interrupted run
        for p in diag.output_dir.glob("*.png"):
            p.unlink()
        for p in diag.output_dir.glob("*.json"):
            p.unlink()

        compute_called = [False]
        orig = diag._compute_variable

        def spy(var):
            compute_called[0] = True
            return orig(var)

        diag._compute_variable = spy
        saved = diag.run(skip_existing=True)
        assert not compute_called[0]  # loaded from NC, not recomputed
        assert len(saved) >= 1
