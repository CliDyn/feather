"""Tests for AddedValueDiag (Dosio et al. 2015).

All tests use small synthetic data (nside=8 HEALPix / 5° lat-lon) and
the MockCMIP6Loader from conftest.py — no real data needed.
"""

from pathlib import Path

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
        for etype in ("ensemble_mean", "ensemble_median"):
            assert etype in vr["av"]["annual"], f"missing key: {etype}"
        assert "per_eerie_av" in vr["av"]["annual"]
        assert "per_cmip6_av" in vr["av"]["annual"]

    def test_av_dims_and_bounds(self, diag):
        results = diag.compute()
        for etype in ("ensemble_mean", "ensemble_median"):
            av = results["tas"]["av"]["annual"][etype]
            assert set(av.dims) == {"lat", "lon"}, etype
            assert float(av.min()) >= -1.0 - 1e-6, etype
            assert float(av.max()) <= 1.0 + 1e-6, etype

    def test_per_model_av_in_result(self, diag):
        results = diag.compute()
        annual = results["tas"]["av"]["annual"]
        # per_eerie_av: one entry per EERIE model (ModelA, ModelB)
        assert len(annual["per_eerie_av"]) == 2
        for model_name, av_field in annual["per_eerie_av"].items():
            assert set(av_field.dims) == {"lat", "lon"}
            assert float(av_field.min()) >= -1.0 - 1e-6
            assert float(av_field.max()) <= 1.0 + 1e-6
        # per_cmip6_av: one entry per CMIP6 member
        assert len(annual["per_cmip6_av"]) >= 1
        for label, av_field in annual["per_cmip6_av"].items():
            assert set(av_field.dims) == {"lat", "lon"}

    def test_summary_stats_in_result(self, diag):
        results = diag.compute()
        av = results["tas"]["av"]["annual"]
        for etype in ("ensemble_mean", "ensemble_median"):
            assert f"{etype}_domain_av" in av
            assert f"{etype}_frac_positive" in av
            assert np.isfinite(av[f"{etype}_domain_av"])
            assert 0.0 <= av[f"{etype}_frac_positive"] <= 1.0

    def test_nc_files_saved(self, diag, tmp_path):
        diag.compute()
        nc_dir = diag.nc_dir
        for etype in ("ensemble_mean", "ensemble_median"):
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

    def test_plot_returns_figures_per_period(self, diag):
        results = diag.compute()
        figures = diag.plot(results)
        figure_ids = [meta["figure_id"] for _, meta in figures]
        import matplotlib.pyplot as plt
        for fig, meta in figures:
            assert isinstance(fig, plt.Figure)
            assert "figure_id" in meta
            assert "added_value" in meta["figure_id"]
        # Both ensemble and per-model figures should be present
        assert any("added_value_models" not in fid for fid in figure_ids), \
            "expected at least one ensemble figure"
        assert any("added_value_models" in fid for fid in figure_ids), \
            "expected at least one per-model figure"

    def test_skip_when_nc_exists(self, diag, tmp_path, monkeypatch):
        """_all_nc_exist returns True only when all 10 NC files are present."""
        var = "tas"
        # Before computation no NC files exist
        assert not diag._all_nc_exist(var)
        diag.compute()
        # After computation: 2 etypes × 5 periods = 10 NC files
        assert diag._all_nc_exist(var)

    def test_load_from_nc(self, diag):
        diag.compute()
        loaded = diag._load_variable_from_nc("tas")
        assert loaded is not None
        assert "av" in loaded
        assert "annual" in loaded["av"]
        for etype in ("ensemble_mean", "ensemble_median"):
            av = loaded["av"]["annual"][etype]
            assert float(av.min()) >= -1.0 - 1e-6, etype
        # Per-model dicts are empty when loading from NC
        assert loaded["av"]["annual"]["per_eerie_av"] == {}
        assert loaded["av"]["annual"]["per_cmip6_av"] == {}

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
        # Figure 1 is ensemble summary
        ensemble_figs = [
            meta for _, meta in figures
            if "models" not in meta["figure_id"].split("_added_value")[-1]
        ]
        assert ensemble_figs, "expected at least one ensemble figure"
        _, meta = figures[0]
        assert meta["diagnostic_name"] == "added_value"
        assert "summary_statistics" in meta
        stats = meta["summary_statistics"]
        for etype in ("ensemble_mean", "ensemble_median"):
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

    def test_run_iterates_all_benchmarks(self, synth_obs, synth_cmip6,
                                         eerie_config):
        """--benchmarks cmip6 HighResMIP → both AV sets in one run.

        Regression: added_value did not accept ``benchmarks=`` so
        cmip6_enabled was False (no AV computed), and it only ever used the
        primary benchmark. It now iterates every benchmark, tokenising the
        non-CMIP6 outputs.
        """
        from tests.conftest import MockCMIP6Loader
        cmip6 = MockCMIP6Loader(synth_cmip6)          # no label → "cmip6"
        highres = MockCMIP6Loader(synth_cmip6)
        highres.label = "HighResMIP MMM"              # → "_highresmip"
        diag = AddedValueDiag(
            MockCMORLoader(synth_obs), MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=cmip6, benchmarks=[cmip6, highres],
            variables=["tas"], period=("1990", "1990"),
        )
        diag.run(skip_existing=False)

        nc_names = {p.name for p in diag.nc_dir.glob("*.nc")}
        # CMIP6 (token-less) and HighResMIP (tokenised) checkpoints both exist.
        assert "tas_annual_ensemble_mean_av.nc" in nc_names
        assert "tas_annual_ensemble_mean_av_highresmip.nc" in nc_names

        fig_ids = {p.stem for p in diag.output_dir.glob("*added_value*.png")}
        assert any(f.endswith("_highresmip") for f in fig_ids)
        assert any(not f.endswith("_highresmip") for f in fig_ids)

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


# ── Unit tests for _frac_categories ──────────────────────────────────


class TestFracCategories:
    def test_all_improvement(self):
        av = _make_latlon(0.5)
        cats = AddedValueDiag._frac_categories(av)
        assert cats["pct_improvement"] == pytest.approx(100.0)
        assert cats["pct_neutral"] == pytest.approx(0.0)
        assert cats["pct_deterioration"] == pytest.approx(0.0)

    def test_all_deterioration(self):
        av = _make_latlon(-0.5)
        cats = AddedValueDiag._frac_categories(av)
        assert cats["pct_improvement"] == pytest.approx(0.0)
        assert cats["pct_deterioration"] == pytest.approx(100.0)
        assert cats["pct_neutral"] == pytest.approx(0.0)

    def test_all_neutral(self):
        av = _make_latlon(0.0)  # exactly 0 → neutral (|AV| ≤ threshold)
        cats = AddedValueDiag._frac_categories(av)
        assert cats["pct_neutral"] == pytest.approx(100.0)
        assert cats["pct_improvement"] == pytest.approx(0.0)
        assert cats["pct_deterioration"] == pytest.approx(0.0)

    def test_percentages_sum_to_100(self):
        rng = np.random.default_rng(7)
        lats = np.arange(-87.5, 90, 5.0)
        lons = np.arange(2.5, 360, 5.0)
        data = rng.uniform(-1, 1, (len(lats), len(lons)))
        av = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        cats = AddedValueDiag._frac_categories(av)
        total = cats["pct_improvement"] + cats["pct_neutral"] + cats["pct_deterioration"]
        assert total == pytest.approx(100.0, abs=1e-10)

    def test_custom_threshold(self):
        lats = np.array([-2.5, 2.5])
        lons = np.array([2.5, 7.5])
        data = np.array([[0.5, -0.5], [0.005, -0.005]])
        av = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        cats_001 = AddedValueDiag._frac_categories(av, threshold=0.001)
        cats_01 = AddedValueDiag._frac_categories(av, threshold=0.01)
        # With tighter threshold 0.001: ±0.005 counts as improvement/deterioration
        assert cats_001["pct_neutral"] < cats_01["pct_neutral"]

    def test_nan_field(self):
        lats = np.array([-2.5, 2.5])
        lons = np.array([2.5])
        data = np.full((2, 1), np.nan)
        av = xr.DataArray(data, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        cats = AddedValueDiag._frac_categories(av)
        assert np.isnan(cats["pct_improvement"])
        assert np.isnan(cats["pct_neutral"])
        assert np.isnan(cats["pct_deterioration"])


# ── Multi-obs statistics tests ────────────────────────────────────────


class TestMultiObsStats:
    """Tests for per-obs improvement/neutral/deterioration statistics.

    Uses ``tas`` throughout because MockCMIP6Loader has synthetic data for it.
    The eerie_config has an empty obs_datasets dict, so BERKELEY_EARTH_HR is
    not available and only ERA5 is queried — perfect for testing the single-obs
    path without mocking additional loaders.
    """

    @pytest.fixture
    def diag_multi(self, synth_obs, synth_cmip6, eerie_config):
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

    def test_obs_stats_in_compute_result(self, diag_multi):
        results = diag_multi.compute()
        assert "tas" in results
        assert "obs_stats" in results["tas"]
        # ERA5 always present; BERKELEY_EARTH_HR absent (not in config)
        assert "ERA5" in results["tas"]["obs_stats"]

    def test_obs_stats_has_annual_period(self, diag_multi):
        results = diag_multi.compute()
        era5_stats = results["tas"]["obs_stats"]["ERA5"]
        assert "annual" in era5_stats

    def test_obs_stats_category_keys(self, diag_multi):
        results = diag_multi.compute()
        annual = results["tas"]["obs_stats"]["ERA5"]["annual"]
        for etype in ("eerie_mean", "eerie_median", "cmip6_mean"):
            assert etype in annual, f"missing key: {etype}"
            cats = annual[etype]
            assert "pct_improvement" in cats
            assert "pct_neutral" in cats
            assert "pct_deterioration" in cats

    def test_obs_stats_percentages_sum_to_100(self, diag_multi):
        results = diag_multi.compute()
        annual = results["tas"]["obs_stats"]["ERA5"]["annual"]
        for etype in ("eerie_mean", "eerie_median", "cmip6_mean"):
            cats = annual[etype]
            total = (
                cats["pct_improvement"]
                + cats["pct_neutral"]
                + cats["pct_deterioration"]
            )
            assert total == pytest.approx(100.0, abs=1e-9), (
                f"{etype}: categories don't sum to 100 ({total})"
            )

    def test_obs_stats_json_saved(self, diag_multi):
        import json as _json
        diag_multi.compute()
        json_path = diag_multi._obs_stats_path("tas")
        assert json_path.exists(), f"Expected {json_path} to exist"
        with open(json_path) as fh:
            data = _json.load(fh)
        assert data["variable"] == "tas"
        assert "periods" in data
        assert "threshold" in data
        assert data["threshold"] == pytest.approx(0.005)
        assert "ERA5" in data["periods"]["annual"]

    def test_obs_stats_in_figure_metadata(self, diag_multi):
        results = diag_multi.compute()
        figures = diag_multi.plot(results)
        # Ensemble figures (not _models) should carry per_obs_stats
        ensemble_figs = [
            meta for _, meta in figures
            if not meta["figure_id"].endswith("_models")
        ]
        assert ensemble_figs, "expected at least one ensemble figure"
        meta = ensemble_figs[0]
        assert "summary_statistics" in meta
        assert "per_obs_stats" in meta["summary_statistics"], (
            "per_obs_stats not found in summary_statistics"
        )
        per_obs = meta["summary_statistics"]["per_obs_stats"]
        assert "ERA5" in per_obs

    def test_no_alternative_obs_in_empty_config(self, synth_obs, synth_cmip6, eerie_config):
        """BERKELEY_EARTH_HR is filtered out when not in config.obs_datasets."""
        from tests.conftest import MockCMIP6Loader
        diag = AddedValueDiag(
            MockCMORLoader(synth_obs),
            MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=MockCMIP6Loader(synth_cmip6),
            variables=["tas"],
            period=("1990", "1990"),
        )
        results = diag.compute()
        obs_stats = results["tas"]["obs_stats"]
        # Only ERA5; BERKELEY_EARTH_HR not in config → filtered
        assert "ERA5" in obs_stats
        assert "BERKELEY_EARTH_HR" not in obs_stats


class TestBenchmarkToken:
    """Benchmark token disambiguates AV filenames across reference ensembles."""

    def _make_diag(self, synth_obs, synth_cmip6, eerie_config, label=None):
        from tests.conftest import MockCMIP6Loader
        loader = MockCMIP6Loader(synth_cmip6)
        if label is not None:
            loader.label = label  # instance attr read in __init__
        return AddedValueDiag(
            MockCMORLoader(synth_obs),
            MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=loader,
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_default_cmip6_is_token_less(self, synth_obs, synth_cmip6, eerie_config):
        """No label / CMIP6 label → empty suffix (backward compatible)."""
        diag = self._make_diag(synth_obs, synth_cmip6, eerie_config)
        assert diag._benchmark_token == "cmip6"
        assert diag._bench_suffix == ""
        assert diag._nc_path("tas", "annual", "ensemble_mean").name == (
            "tas_annual_ensemble_mean_av.nc"
        )
        assert diag._obs_stats_path("tas").name == "tas_obs_stats.json"

    def test_highresmip_label_adds_token(self, synth_obs, synth_cmip6, eerie_config):
        """HighResMIP MMM label → '_highresmip' on NC + obs-stats filenames."""
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        assert diag._benchmark_token == "highresmip"
        assert diag._bench_suffix == "_highresmip"
        assert diag._nc_path("tas", "annual", "ensemble_mean").name == (
            "tas_annual_ensemble_mean_av_highresmip.nc"
        )
        assert diag._obs_stats_path("tas").name == (
            "tas_obs_stats_highresmip.json"
        )
        assert diag._bars_models_eerie_id("annual").endswith("_highresmip")

    def test_figure_ids_carry_token(self, synth_obs, synth_cmip6, eerie_config):
        """Saved AV map figure IDs include the benchmark token."""
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])
        fids = [meta["figure_id"] for _, meta in figures]
        assert fids, "expected at least one figure"
        assert all(fid.endswith("_highresmip") for fid in fids), fids

    def test_nc_files_written_with_token(
        self, synth_obs, synth_cmip6, eerie_config,
    ):
        """HighResMIP run writes tokenized NC files that don't collide."""
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        diag.compute()
        assert (
            diag.nc_dir / "tas_annual_ensemble_mean_av_highresmip.nc"
        ).exists()
        # The token-less (CMIP6) name must NOT be produced by this run.
        assert not (diag.nc_dir / "tas_annual_ensemble_mean_av.nc").exists()

    def test_bench_label_and_name(self, synth_obs, synth_cmip6, eerie_config):
        """Label/name fields and instance title reflect the benchmark."""
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        assert diag._bench_label == "HighResMIP MMM"
        assert diag._bench_name == "HighResMIP"
        assert diag.title == "Added Value (ensemble vs HighResMIP)"

    def test_figure_text_uses_benchmark(
        self, synth_obs, synth_cmip6, eerie_config,
    ):
        """Figure titles/descriptions say HighResMIP, never CMIP6."""
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])
        for _, meta in figures:
            blob = f"{meta.get('title', '')} {meta.get('description', '')}"
            assert "HighResMIP" in blob, meta.get("figure_id")
            assert "CMIP6" not in blob, meta.get("figure_id")

    def test_nc_attrs_use_benchmark(self, synth_obs, synth_cmip6, eerie_config):
        """NC long_name/model1 reference the active benchmark."""
        import xarray as xr
        diag = self._make_diag(
            synth_obs, synth_cmip6, eerie_config, label="HighResMIP MMM",
        )
        diag.compute()
        ds = xr.open_dataset(
            diag.nc_dir / "tas_annual_ensemble_mean_av_highresmip.nc"
        )
        assert "HighResMIP MMM" in ds["av"].attrs["long_name"]
        assert "HighResMIP" in ds["av"].attrs["model1"]
        ds.close()


# ── NetCDF fast-path tests ────────────────────────────────────────────


def _write_bias_nc(path, models_bias, bench_bias, ens_mean_bias,
                   ens_median_bias, bench_field="CMIP6_MMM"):
    """Write a synthetic bias-map NetCDF like the bias-map diagnostics do."""
    import xarray as xr
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = np.arange(-87.5, 90, 5.0)
    lons = np.arange(2.5, 360, 5.0)

    def _f(val):
        return xr.DataArray(
            np.full((len(lats), len(lons)), val, dtype=float),
            dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        )

    fields = {f"{bench_field}_bias": _f(bench_bias),
              "ens_mean_bias": _f(ens_mean_bias),
              "ens_median_bias": _f(ens_median_bias)}
    for m, b in models_bias.items():
        fields[f"{m}_bias"] = _f(b)
    xr.Dataset(fields).to_netcdf(path)


class TestAddedValueNetCDFFastPath:
    """Added Value reuses precomputed biases from the bias-map NetCDFs."""

    def _diag(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        return AddedValueDiag(
            MockCMORLoader(synth_obs), MockObsLoaderLatlon(synth_obs),
            eerie_config, cmip6_loader=MockCMIP6Loader(synth_cmip6),
            variables=["psl"], period=("1990", "1990"),
        )

    def _seed(self, diag, var="psl"):
        # global_biases is the source for psl (ERA5). Benchmark bias is large
        # (worse), EERIE model biases small (better) → AV > 0.
        out = Path(diag.config.output_dir) / "netcdf" / "global_biases"
        for pk in ("annual", "DJF", "MAM", "JJA", "SON"):
            _write_bias_nc(
                out / f"{var}_{pk}_1990-1990.nc",
                models_bias={"ModelA": 1.0, "ModelB": -1.0},
                bench_bias=4.0, ens_mean_bias=1.0, ens_median_bias=1.0,
            )

    def test_av_from_biases_sign(self):
        small = _make_latlon(0.5)   # candidate bias (better)
        big = _make_latlon(4.0)     # reference bias (worse)
        av = AddedValueDiag._av_from_biases(big, small)
        # AV>0 everywhere: candidate reduces squared error vs reference.
        assert float(av.min()) > 0
        assert float(av.max()) <= 1.0 + 1e-9

    def test_reads_from_netcdf(self, synth_obs, synth_cmip6, eerie_config):
        diag = self._diag(synth_obs, synth_cmip6, eerie_config)
        self._seed(diag)
        res = diag._compute_variable_from_netcdf("psl")
        assert res is not None
        assert set(res["eerie_models"]) == {"ModelA", "ModelB"}
        annual = res["av"]["annual"]
        assert "ModelA" in annual["per_eerie_av"]
        # bench worse than EERIE → ensemble-mean AV positive.
        assert annual["ensemble_mean_domain_av"] > 0
        # per_cmip6 panels empty without --cmip6-individual.
        assert annual["per_cmip6_av"] == {}
        # secondary obs: psl has none → only ERA5.
        assert set(res["av_by_obs"]) == {"ERA5"}
        # obs_stats populated for the bars.
        assert "ERA5" in res["obs_stats"]
        assert "per_eerie_models" in res["obs_stats"]["ERA5"]["annual"]

    def test_missing_netcdf_returns_none(self, synth_obs, synth_cmip6,
                                         eerie_config):
        """No bias NetCDF → fast path returns None (caller falls back)."""
        diag = self._diag(synth_obs, synth_cmip6, eerie_config)
        assert diag._compute_variable_from_netcdf("psl") is None

    def test_compute_variable_prefers_netcdf(self, synth_obs, synth_cmip6,
                                             eerie_config):
        """_compute_variable uses the NetCDF path without touching raw data."""
        diag = self._diag(synth_obs, synth_cmip6, eerie_config)
        self._seed(diag)
        called = {"recompute": False}
        diag._compute_variable_recompute = lambda v: called.__setitem__(
            "recompute", True)
        res = diag._compute_variable("psl")
        assert res is not None
        assert called["recompute"] is False

    def test_individual_members_when_opted_in(self, synth_obs, synth_cmip6,
                                              eerie_config):
        diag = self._diag(synth_obs, synth_cmip6, eerie_config)
        diag.cmip6_individual = True
        self._seed(diag)
        # Add an individual-member bias file for annual.
        out = Path(diag.config.output_dir) / "netcdf" / "global_biases"
        _write_bias_nc(
            out / "psl_annual_individual_1990-1990.nc",
            models_bias={"CMIP6__ACCESS_CM2_r1i1p1f1": 3.0},
            bench_bias=4.0, ens_mean_bias=1.0, ens_median_bias=1.0,
        )
        res = diag._compute_variable_from_netcdf("psl")
        per_cmip6 = res["av"]["annual"]["per_cmip6_av"]
        assert "ACCESS_CM2_r1i1p1f1" in per_cmip6


class TestCordexRegionStats:
    """Per-CORDEX-region category stats and region-scoped bar charts."""

    @pytest.fixture
    def diag_multi(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        return AddedValueDiag(
            MockCMORLoader(synth_obs),
            MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=MockCMIP6Loader(synth_cmip6),
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_regions_present_in_obs_stats(self, diag_multi):
        from feather.util.regions import list_regions
        annual = diag_multi.compute()["tas"]["obs_stats"]["ERA5"]["annual"]
        assert "regions" in annual
        # Every catalogued region has an entry.
        for r in list_regions():
            assert r in annual["regions"]

    def test_region_block_has_category_keys(self, diag_multi):
        annual = diag_multi.compute()["tas"]["obs_stats"]["ERA5"]["annual"]
        eur = annual["regions"]["EUR"]
        for etype in ("eerie_mean", "eerie_median", "cmip6_mean"):
            cats = eur[etype]
            assert set(cats) >= {
                "pct_improvement", "pct_neutral", "pct_deterioration",
            }

    def test_region_fractions_sum_to_100(self, diag_multi):
        annual = diag_multi.compute()["tas"]["obs_stats"]["ERA5"]["annual"]
        eur = annual["regions"]["EUR"]["eerie_mean"]
        total = (
            eur["pct_improvement"]
            + eur["pct_neutral"]
            + eur["pct_deterioration"]
        )
        # EUR has cells on the synthetic grid → finite, summing to 100.
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_global_and_region_differ_or_match_schema(self, diag_multi):
        """Global block keeps legacy keys; regions is additive."""
        annual = diag_multi.compute()["tas"]["obs_stats"]["ERA5"]["annual"]
        # Legacy keys still at top level (backward compat).
        for etype in ("eerie_mean", "eerie_median", "cmip6_mean"):
            assert etype in annual

    def test_region_bar_chart_builds(self, diag_multi):
        results = diag_multi.compute()
        all_obs_stats = {"tas": results["tas"]["obs_stats"]}
        figs = diag_multi._plot_summary_bars_ensemble(
            all_obs_stats, "annual", region="EUR")
        assert len(figs) == 1
        fig, meta = figs[0]
        assert meta["figure_id"].startswith("added_value_bars_ensemble_EUR_")
        assert meta.get("region") == "EUR"
        # Region bars carry the dedicated nav group so the website surfaces
        # them as their own entry.
        assert meta.get("group") == "regions"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_global_bar_chart_keeps_default_group(self, diag_multi):
        results = diag_multi.compute()
        all_obs_stats = {"tas": results["tas"]["obs_stats"]}
        fig, meta = diag_multi._plot_summary_bars_ensemble(
            all_obs_stats, "annual", region=None)[0]
        assert meta.get("group") != "regions"
        assert "region" not in meta
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_region_output_dir_is_separate_nav_page(self, diag_multi):
        d = diag_multi.region_output_dir
        assert d.name == "added_value_regions"
        # Distinct from the main Added Value figures directory.
        assert d != diag_multi.output_dir
        assert d.parent == diag_multi.output_dir.parent

    def test_region_figure_ids_unique_from_global(self, diag_multi):
        assert (
            diag_multi._bars_ensemble_id("annual", None)
            != diag_multi._bars_ensemble_id("annual", "EUR")
        )
        assert diag_multi._bars_models_eerie_id("annual", "AFR").startswith(
            "added_value_bars_models_eerie_AFR_"
        )


# ── Ocean Added Value page ───────────────────────────────────────────


class TestOceanAddedValue:
    """Unit tests for the ocean AV helpers (pure functions, no real data)."""

    @pytest.fixture
    def diag(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        return AddedValueDiag(
            MockCMORLoader(synth_obs),
            MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=MockCMIP6Loader(synth_cmip6),
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_ocean_variables_list(self, diag):
        assert diag._OCEAN_VARIABLES == ["tos", "thetao", "so", "siconc"]

    def test_rsds_excluded_from_atmospheric(self, diag):
        assert "rsds" not in diag.variables

    def test_ocean_output_dir_is_separate_nav_page(self, diag):
        d = diag.ocean_output_dir
        assert d.name == "added_value_ocean"
        assert d != diag.output_dir
        assert d.parent == diag.output_dir.parent

    def test_ocean_fig_ids_cover_periods(self, diag):
        ids = diag._ocean_fig_ids("tos")
        assert len(ids) == 6  # 3 periods x (ensemble + models)
        assert any("annual_1990_1990_added_value" in i for i in ids)
        assert any("_added_value_models" in i for i in ids)

    def test_surface_slice_collapses_depth(self, diag):
        da = xr.DataArray(
            np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float),
            dims=("lev", "lat", "lon"),
            coords={"lev": [0.0, 100.0], "lat": [-30, 0, 30],
                    "lon": [0, 90, 180, 270]},
        )
        surf = diag._surface_slice(da)
        assert "lev" not in surf.dims
        # Surface = shallowest level (index 0)
        assert float(surf.isel(lat=0, lon=0)) == 0.0

    def test_surface_slice_noop_without_depth(self, diag):
        da = _make_latlon(1.0)
        assert diag._surface_slice(da).dims == da.dims

    def test_to_celsius_from_kelvin_units(self, diag):
        da = _make_latlon(300.0)
        da.attrs["units"] = "K"
        assert float(diag._to_celsius_if_needed(da).isel(lat=0, lon=0)) == \
            pytest.approx(26.85, abs=1e-2)

    def test_to_celsius_skips_celsius(self, diag):
        da = _make_latlon(15.0)
        da.attrs["units"] = "degC"
        assert float(diag._to_celsius_if_needed(da).isel(lat=0, lon=0)) == 15.0

    def test_to_celsius_heuristic_kelvin(self, diag):
        da = _make_latlon(290.0)  # no units, magnitude > 150 => Kelvin
        assert float(diag._to_celsius_if_needed(da).max()) < 100.0

    def test_siconc_percent_to_fraction(self, diag):
        da = _make_latlon(80.0)  # percent
        assert float(diag._siconc_to_fraction(da).max()) == pytest.approx(0.8)

    def test_siconc_fraction_unchanged(self, diag):
        da = _make_latlon(0.8)
        assert float(diag._siconc_to_fraction(da).max()) == pytest.approx(0.8)

    def test_sa_to_sp_noop_when_not_absolute(self, diag):
        da = _make_latlon(35.0)
        out = diag._surface_sa_to_sp(da, "ModelA")  # absolute_salinity False
        assert float(out.max()) == 35.0

    def test_regrid_scatter_rectilinear(self, diag):
        src = _make_latlon(
            5.0, lats=np.arange(-80, 81, 20.0), lons=np.arange(10, 360, 20.0),
        )
        tlat = np.arange(-89.5, 90, 1.0)
        tlon = np.arange(0.5, 360, 1.0)
        out = diag._regrid_scatter(
            src, tlat, tlon, 1.0, 1_000_000, {}, method="nearest",
        )
        assert out.dims == ("lat", "lon")
        assert out.shape == (len(tlat), len(tlon))
        # constant field stays constant where covered
        assert float(np.nanmax(out.values)) == pytest.approx(5.0, abs=1e-6)

    def test_regrid_scatter_curvilinear_2d_coords(self, diag):
        ny, nx = 10, 12
        lat2d = np.tile(np.linspace(-80, 80, ny)[:, None], (1, nx))
        lon2d = np.tile(np.linspace(0, 340, nx)[None, :], (ny, 1))
        src = xr.DataArray(
            np.full((ny, nx), 3.0),
            dims=("y", "x"),
            coords={"nav_lat": (("y", "x"), lat2d),
                    "nav_lon": (("y", "x"), lon2d)},
        )
        tlat = np.arange(-89.5, 90, 2.0)
        tlon = np.arange(0.5, 360, 2.0)
        out = diag._regrid_scatter(
            src, tlat, tlon, 2.0, 2_000_000, {}, method="nearest",
        )
        assert out.shape == (len(tlat), len(tlon))


# ── ERA5-based AV NetCDF persistence (tas/pr secondary obs) ──────────


class TestAvNetcdfObsToken:
    """The ERA5 AV field for tas/pr must be persisted alongside the primary."""

    @pytest.fixture
    def diag(self, synth_obs, synth_cmip6, eerie_config):
        from tests.conftest import MockCMIP6Loader
        return AddedValueDiag(
            MockCMORLoader(synth_obs),
            MockObsLoaderLatlon(synth_obs),
            eerie_config,
            cmip6_loader=MockCMIP6Loader(synth_cmip6),
            variables=["tas"],
            period=("1990", "1990"),
        )

    def test_primary_obs_nc_is_token_less(self, diag):
        # tas primary obs is Berkeley Earth HR → token-less (backward compat)
        p = diag._nc_path(
            "tas", "annual", "ensemble_mean", obs_name="BERKELEY_EARTH_HR")
        assert p.name == "tas_annual_ensemble_mean_av.nc"

    def test_era5_secondary_nc_has_token(self, diag):
        p = diag._nc_path("tas", "annual", "ensemble_mean", obs_name="ERA5")
        assert p.name == "tas_annual_ensemble_mean_av_era5.nc"

    def test_era5_primary_var_stays_token_less(self, diag):
        # For a var whose primary obs IS ERA5 (e.g. psl), no token is added.
        p = diag._nc_path("psl", "annual", "ensemble_mean", obs_name="ERA5")
        assert p.name == "psl_annual_ensemble_mean_av.nc"

    def test_compute_saves_era5_nc_for_tas(self, diag):
        diag._compute_variable("tas")
        primary = diag._nc_path("tas", "annual", "ensemble_mean")
        era5 = diag._nc_path("tas", "annual", "ensemble_mean", obs_name="ERA5")
        assert primary.exists(), "primary AV NetCDF missing"
        assert era5.exists(), "ERA5-based AV NetCDF missing for tas"
