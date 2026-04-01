"""Tests for the global_biases diagnostic."""

import json
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import nereus as nr
import numpy as np
import pytest
import xarray as xr

from feather.diag.global_biases import GlobalBiases
from feather.util.spatial import latlon_global_mean
from tests.conftest import MockCMIP6Loader


# -- Utility tests ----------------------------------------------------------

# Large influence radius for nside=8 test data (~815 km spacing)
_TEST_INFLUENCE_RADIUS = 1_000_000


class TestNereusRegrid:
    """Tests for NN regridding via nereus from scattered to regular grid."""

    def test_basic_regrid(self, synth_healpix, synth_obs):
        """Regrid HEALPix to obs grid -- result has lat/lon dims."""
        model_clim = synth_healpix["avg_2t"].isel(time=0)
        lon = np.asarray(synth_healpix["longitude"])
        lat = np.asarray(synth_healpix["latitude"])
        obs = synth_obs["t2m"].isel(time=0)
        obs_res = abs(float(obs.lat.values[1] - obs.lat.values[0]))

        regridded, _ = nr.regrid(
            model_clim.values.ravel(),
            lon=lon, lat=lat,
            resolution=obs_res,
            influence_radius=_TEST_INFLUENCE_RADIUS,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )

        assert "lat" in regridded.dims
        assert "lon" in regridded.dims

    def test_regrid_preserves_mean(self, synth_healpix, synth_obs):
        """Regridded global mean is close to the original."""
        model_clim = synth_healpix["avg_2t"].isel(time=0)
        lon = np.asarray(synth_healpix["longitude"])
        lat = np.asarray(synth_healpix["latitude"])
        area = synth_healpix["area"]
        obs = synth_obs["t2m"].isel(time=0)
        obs_res = abs(float(obs.lat.values[1] - obs.lat.values[0]))

        from feather.util.spatial import global_mean

        original_mean = float(global_mean(model_clim, area).values)

        regridded, _ = nr.regrid(
            model_clim.values.ravel(),
            lon=lon, lat=lat,
            resolution=obs_res,
            influence_radius=_TEST_INFLUENCE_RADIUS,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        regridded_mean = float(latlon_global_mean(regridded).values)

        # At nside=8, NN regridding introduces some error
        assert abs(original_mean - regridded_mean) < 5.0

    def test_matching_data_gives_small_bias(self, synth_healpix, synth_obs):
        """When model & obs have same pattern, regridded bias is small."""
        model_clim = synth_healpix["avg_2t"].isel(time=0)
        lon = np.asarray(synth_healpix["longitude"])
        lat = np.asarray(synth_healpix["latitude"])
        obs_clim = synth_obs["t2m"].isel(time=0)
        obs_res = abs(float(obs_clim.lat.values[1] - obs_clim.lat.values[0]))

        regridded, interpolator = nr.regrid(
            model_clim.values.ravel(),
            lon=lon, lat=lat,
            resolution=obs_res,
            influence_radius=_TEST_INFLUENCE_RADIUS,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        target_lats = interpolator.target_lat[:, 0]
        target_lons = interpolator.target_lon[0, :]
        obs_common = obs_clim.interp(lat=target_lats, lon=target_lons)
        bias = regridded - obs_common

        # With matching synthetic data, bias should be small
        assert abs(float(latlon_global_mean(bias).values)) < 3.0


class TestLatlonGlobalMean:
    """Tests for cos-lat weighted global mean."""

    def test_returns_scalar_for_2d(self, synth_obs):
        """Global mean of a 2D field is a scalar."""
        field = synth_obs["t2m"].isel(time=0)
        result = latlon_global_mean(field)
        assert result.shape == ()

    def test_returns_timeseries_for_3d(self, synth_obs):
        """Global mean of (time, lat, lon) is a time series."""
        result = latlon_global_mean(synth_obs["t2m"])
        assert "time" in result.dims
        assert result.shape == (12,)

    def test_reasonable_value(self, synth_obs):
        """Global mean of synthetic temperature is ~280K."""
        field = synth_obs["t2m"].isel(time=0)
        result = float(latlon_global_mean(field).values)
        assert 270 < result < 300


# -- GlobalBiases compute tests --------------------------------------------


class TestGlobalBiasesCompute:
    """Tests for GlobalBiases.compute()."""

    def test_compute_structure(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """compute() returns expected nested structure."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert "tas" in results
        vr = results["tas"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr
        assert "ifs-fesom" in vr["models"]

    def test_model_results_fields(self, mock_model_loader, mock_obs_loader,
                                   minimal_config):
        """Each model entry has required fields."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert "annual_regrid" in mdata
        assert "annual_bias" in mdata
        assert "annual_bias_gmean" in mdata
        assert "annual_rmse" in mdata
        assert "global_mean" in mdata
        assert "seasonal_biases" in mdata
        assert "seasonal_regrids" in mdata

    def test_obs_results_fields(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Obs entry has required fields."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        obs = results["tas"]["obs"]
        assert "clim" in obs
        assert "global_mean" in obs
        assert isinstance(obs["global_mean"], float)

    def test_bias_is_small_for_matching_data(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """With matching synth data, global mean bias is small."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        # Synth model and obs have the same pattern -> small bias
        assert abs(mdata["annual_bias_gmean"]) < 3.0

    def test_rmse_is_small_for_matching_data(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """RMSE is small when model matches obs."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert mdata["annual_rmse"] < 5.0

    def test_seasonal_biases_computed(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Seasonal biases are computed for DJF and JJA."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        seasonal = results["tas"]["models"]["ifs-fesom"]["seasonal_biases"]
        assert "DJF" in seasonal
        assert "JJA" in seasonal

    def test_custom_variables(self, mock_model_loader, mock_obs_loader,
                               minimal_config):
        """Constructor variables= overrides class default."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        assert diag.variables == ["tas"]


# -- GlobalBiases skip_existing tests --------------------------------------


class TestGlobalBiasesSkipExisting:
    """Tests for skip_existing and per-variable incremental saving."""

    def test_skip_when_all_figures_exist(self, mock_model_loader,
                                          mock_obs_loader, minimal_config):
        """run() skips variable when all 3 period figures exist."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        # Pre-create all 3 period figures
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        for period in ["annual", "djf", "jja"]:
            fid = f"tas_{period}_bias_combined"
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        saved = diag.run(skip_existing=True)

        assert len(saved) == 3
        # Files should not have been overwritten
        for png_path, _ in saved:
            assert png_path.read_bytes() == b"fake"

    def test_no_skip_when_disabled(self, mock_model_loader, mock_obs_loader,
                                    minimal_config):
        """run(skip_existing=False) recomputes even when figures exist."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        for period in ["annual", "djf", "jja"]:
            fid = f"tas_{period}_bias_combined"
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            saved = diag.run(skip_existing=False)

        assert len(saved) == 3

    def test_no_skip_when_partial_files(self, mock_model_loader,
                                         mock_obs_loader, minimal_config):
        """run() recomputes when only some period figures exist."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        # Only create annual, not DJF and JJA
        fid = "tas_annual_bias_combined"
        (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
        (diag.output_dir / f"{fid}.json").write_text("{}")

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            saved = diag.run(skip_existing=True)

        # Should have recomputed (3 new figures)
        assert len(saved) == 3

    def test_incremental_save_per_variable(self, mock_model_loader,
                                            mock_obs_loader, minimal_config):
        """run() saves figures after each variable (incremental)."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )

        def _make_saveable_fig():
            """Create a mock figure whose savefig creates a real file."""
            mock_fig = MagicMock(spec=plt.Figure)
            mock_fig.savefig = lambda path, **kw: Path(path).write_bytes(b"png")
            return mock_fig

        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
        ) as mock_plot:
            mock_plot.return_value = (_make_saveable_fig(), [None, None])
            saved = diag.run(skip_existing=False)

        # 3 period figures for 1 variable
        assert len(saved) == 3
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()


# -- GlobalBiases plot tests (mocked) --------------------------------------


class TestGlobalBiasesPlot:
    """Tests for GlobalBiases.plot() -- mocks map plotting."""

    def _make_mock_results(self, synth_healpix, synth_obs):
        """Produce results dict from compute() for testing plot()."""
        from feather.data.variables import get_var
        from feather.util.spatial import global_mean
        from feather.util.temporal import climatology, seasonal_climatology

        var = "tas"
        var_info = get_var(var)
        ds = synth_healpix
        model_clim = climatology(ds["avg_2t"])
        lon = np.asarray(ds["longitude"])
        lat = np.asarray(ds["latitude"])
        area = ds["area"]
        model_gmean = float(global_mean(model_clim, area).values)
        model_seasonal = seasonal_climatology(ds["avg_2t"])

        obs_clim = climatology(synth_obs["t2m"])
        obs_res = abs(float(obs_clim.lat.values[1] - obs_clim.lat.values[0]))

        regridded, interpolator = nr.regrid(
            model_clim.values.ravel(),
            lon=lon, lat=lat,
            resolution=obs_res,
            influence_radius=_TEST_INFLUENCE_RADIUS,
            lon_bounds=(0.0, 360.0),
            as_xarray=True,
        )
        target_lats = interpolator.target_lat[:, 0]
        target_lons = interpolator.target_lon[0, :]

        # Regrid obs to common nereus grid
        obs_clim_common = obs_clim.interp(lat=target_lats, lon=target_lons)
        annual_bias = regridded - obs_clim_common

        seasonal_biases = {}
        seasonal_regrids = {}
        obs_seasonal = seasonal_climatology(synth_obs["t2m"])
        for season in ["DJF", "JJA"]:
            if season in model_seasonal and season in obs_seasonal:
                ms = model_seasonal[season]
                s_np = interpolator(ms.values.ravel())
                s_regrid = xr.DataArray(
                    s_np, dims=("lat", "lon"),
                    coords={"lat": target_lats, "lon": target_lons},
                )
                seasonal_regrids[season] = s_regrid
                obs_s = obs_seasonal[season].interp(
                    lat=target_lats, lon=target_lons,
                )
                seasonal_biases[season] = s_regrid - obs_s

        from feather.util.spatial import spatial_ttest, spatial_variance_ratio

        t_stat, t_pval = spatial_ttest(regridded, obs_clim_common)
        f_stat, f_pval = spatial_variance_ratio(regridded, obs_clim_common)

        obs_seasonal_common = {}
        for season in ["DJF", "JJA"]:
            if season in obs_seasonal:
                obs_seasonal_common[season] = obs_seasonal[season].interp(
                    lat=target_lats, lon=target_lons,
                )

        seasonal_ttest_stats = {}
        for season in ["DJF", "JJA"]:
            if season in seasonal_regrids and season in obs_seasonal_common:
                s_t, s_p = spatial_ttest(
                    seasonal_regrids[season], obs_seasonal_common[season],
                )
                s_f, s_fp = spatial_variance_ratio(
                    seasonal_regrids[season], obs_seasonal_common[season],
                )
                seasonal_ttest_stats[season] = {
                    "ttest_statistic": s_t, "ttest_pvalue": s_p,
                    "ftest_statistic": s_f, "ftest_pvalue": s_fp,
                }

        model_results = {
            "ifs-fesom": {
                "annual_regrid": regridded,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": model_gmean,
                "annual_bias": annual_bias,
                "annual_bias_gmean": float(
                    latlon_global_mean(annual_bias).values
                ),
                "annual_rmse": float(np.sqrt(
                    latlon_global_mean(annual_bias ** 2).values
                )),
                "seasonal_biases": seasonal_biases,
                "ttest_statistic": t_stat,
                "ttest_pvalue": t_pval,
                "ftest_statistic": f_stat,
                "ftest_pvalue": f_pval,
                "seasonal_ttest": seasonal_ttest_stats,
            },
        }

        colorbar_ranges = GlobalBiases._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
        )

        return {
            "tas": {
                "models": model_results,
                "obs": {
                    "clim": obs_clim_common,
                    "seasonal_clim": obs_seasonal_common,
                    "global_mean": float(
                        latlon_global_mean(obs_clim).values
                    ),
                },
                "var_info": var_info,
                "colorbar_ranges": colorbar_ranges,
                "cmip6_data": {},
                "cmip6_info": {},
                "cmip6_individual_data": {},
            },
        }

    def test_plot_returns_correct_count(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """plot() returns 3 combined figures: annual + DJF + JJA."""
        results = self._make_mock_results(synth_healpix, synth_obs)

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        # 3 combined figures: annual + DJF + JJA
        assert len(pairs) == 3
        for fig, meta in pairs:
            assert isinstance(meta, dict)
            assert "figure_id" in meta

    def test_plot_metadata_content(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Metadata from plot() has expected fields."""
        results = self._make_mock_results(synth_healpix, synth_obs)

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        # Check annual bias metadata
        _, meta = pairs[0]
        assert meta["figure_id"] == "tas_annual_bias_combined"
        assert meta["plot_type"] == "combined_bias_map"
        assert "ifs-fesom" in meta["summary_statistics"]
        assert "global_mean_bias" in meta["summary_statistics"]["ifs-fesom"]
        assert "rmse" in meta["summary_statistics"]["ifs-fesom"]

    def test_plot_combined_figure_ids(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Combined figure IDs follow naming convention."""
        results = self._make_mock_results(synth_healpix, synth_obs)

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        figure_ids = [meta["figure_id"] for _, meta in pairs]
        assert "tas_annual_bias_combined" in figure_ids
        assert "tas_djf_bias_combined" in figure_ids
        assert "tas_jja_bias_combined" in figure_ids

    def test_plot_models_list_in_metadata(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Metadata models list contains all models in the figure."""
        results = self._make_mock_results(synth_healpix, synth_obs)

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        _, meta = pairs[0]
        assert "ifs-fesom" in meta["models"]


# -- CMIP6 integration tests -----------------------------------------------


class TestGlobalBiasesCMIP6:
    """Tests for CMIP6 integration in global_biases diagnostic."""

    def test_cmip6_disabled_no_data(self, mock_model_loader, mock_obs_loader,
                                     minimal_config):
        """When CMIP6 is disabled, cmip6_data is empty."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["cmip6_data"] == {}
        assert results["tas"]["cmip6_info"] == {}

    def test_cmip6_enabled_has_bias_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_data has annual bias."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_data = results["tas"]["cmip6_data"]
        assert "annual" in cmip6_data
        assert "regrid" in cmip6_data["annual"]
        assert "bias" in cmip6_data["annual"]
        assert "bias_gmean" in cmip6_data["annual"]

    def test_cmip6_bias_is_small(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 bias is small with matching synthetic data."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_data = results["tas"]["cmip6_data"]
        assert abs(cmip6_data["annual"]["bias_gmean"]) < 5.0

    def test_cmip6_seasonal_biases(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 seasonal biases (DJF, JJA) are computed."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_data = results["tas"]["cmip6_data"]
        assert "DJF" in cmip6_data
        assert "JJA" in cmip6_data
        assert "bias" in cmip6_data["DJF"]
        assert "bias" in cmip6_data["JJA"]

    def test_cmip6_info_populated(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_info has n_members and models_used."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        info = results["tas"]["cmip6_info"]
        assert info["n_members"] >= 1
        assert len(info["models_used"]) >= 1

    def test_cmip6_included_in_combined_figure(
        self, synth_healpix, synth_obs, cmip6_config,
        mock_model_loader, mock_obs_loader, mock_cmip6_loader,
    ):
        """CMIP6 MMM appears in the combined figure (3 total figures)."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None, None]),
        ):
            pairs = diag.plot(results)

        # Still 3 combined figures (annual + DJF + JJA), CMIP6 is inside them
        assert len(pairs) == 3
        figure_ids = [meta["figure_id"] for _, meta in pairs]
        assert "tas_annual_bias_combined" in figure_ids

        # CMIP6 MMM should appear in models list
        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        assert "CMIP6 MMM" in annual_meta["models"]
        assert "CMIP6 MMM" in annual_meta["summary_statistics"]

    def test_cmip6_plot_metadata_has_info(
        self, synth_healpix, synth_obs, cmip6_config,
        mock_model_loader, mock_obs_loader, mock_cmip6_loader,
    ):
        """CMIP6 figure metadata includes cmip6_info."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None, None]),
        ):
            pairs = diag.plot(results)

        # Annual combined figure should have cmip6_info
        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        assert annual_meta.get("cmip6_info") is not None
        assert annual_meta["cmip6_info"]["n_members"] >= 1


# -- Combined bias map plot function tests ----------------------------------


class TestPlotCombinedBiasMap:
    """Tests for plot_combined_bias_map() layout and behavior."""

    def _make_obs_and_biases(self, synth_obs, n_biases=3):
        """Create obs field and N bias fields for testing."""
        obs = synth_obs["t2m"].isel(time=0)
        biases = {}
        for i in range(n_biases):
            # Small offset per model to make distinct biases
            biases[f"Model-{i+1}"] = obs * 0.0 + (i + 1) * 0.5
        return obs, biases

    def test_correct_panel_count(self, synth_obs):
        """Number of returned axes matches 1 (obs) + N (biases)."""
        obs, biases = self._make_obs_and_biases(synth_obs, n_biases=3)

        with patch(
            "nereus.plot",
            return_value=(None, None, None),
        ):
            from feather.plot.maps import plot_combined_bias_map
            fig, axes = plot_combined_bias_map(obs, biases)

        # 1 obs + 3 biases = 4 panels
        assert len(axes) == 4
        plt.close("all")

    def test_max_cols_overflow_to_rows(self, synth_obs):
        """When panels > max_cols, layout wraps to multiple rows."""
        obs, biases = self._make_obs_and_biases(synth_obs, n_biases=4)

        with patch(
            "nereus.plot",
            return_value=(None, None, None),
        ):
            from feather.plot.maps import plot_combined_bias_map
            fig, axes = plot_combined_bias_map(
                obs, biases, max_cols=3,
            )

        # 1 obs + 4 biases = 5 panels, max_cols=3 -> 2 rows
        assert len(axes) == 5
        plt.close("all")

    def test_unused_axes_hidden(self, synth_obs):
        """Axes beyond N panels are hidden."""
        obs, biases = self._make_obs_and_biases(synth_obs, n_biases=1)

        with patch(
            "nereus.plot",
            return_value=(None, None, None),
        ):
            from feather.plot.maps import plot_combined_bias_map
            # 2 panels in a 3-col layout -> 1 row, 3 axes, 1 hidden
            fig, axes = plot_combined_bias_map(
                obs, biases, max_cols=3,
            )

        # axes has 2 visible panels; the underlying figure has 3 axes
        assert len(axes) == 2
        plt.close("all")

    def test_single_bias_panel(self, synth_obs):
        """Works with a single bias panel (obs + 1 bias = 2 panels)."""
        obs, biases = self._make_obs_and_biases(synth_obs, n_biases=1)

        with patch(
            "nereus.plot",
            return_value=(None, None, None),
        ):
            from feather.plot.maps import plot_combined_bias_map
            fig, axes = plot_combined_bias_map(obs, biases)

        assert len(axes) == 2
        plt.close("all")


# -- CMIP6 individual model tests ------------------------------------------


class TestGlobalBiasesCMIP6Individual:
    """Tests for cmip6_individual=True mode."""

    def test_individual_populates_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_individual=True populates cmip6_individual_data."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        ind_data = results["tas"]["cmip6_individual_data"]
        assert "annual" in ind_data
        # Should have entries for individual models
        assert len(ind_data["annual"]) >= 1

    def test_individual_also_has_mmm(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When cmip6_individual=True, both individual and MMM are computed."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        # MMM is also computed alongside individual models
        assert "annual" in results["tas"]["cmip6_data"]
        # Individual data is present too
        assert results["tas"]["cmip6_individual_data"] != {}

    def test_individual_models_in_combined_figure(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual CMIP6 models appear in combined figure metadata."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None] * 5),
        ):
            pairs = diag.plot(results)

        # Still 3 combined figures
        assert len(pairs) == 3

        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        # Should include DestinE model + individual CMIP6 models
        assert "ifs-fesom" in annual_meta["models"]
        # At least one CMIP6 individual model in models list
        cmip6_models = [
            m for m in annual_meta["models"] if "/" in m
        ]
        assert len(cmip6_models) >= 1

    def test_individual_figure_count_is_3(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Figure count remains 3 even with individual CMIP6 models."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None] * 5),
        ):
            pairs = diag.plot(results)

        assert len(pairs) == 3

    def test_individual_seasonal_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual CMIP6 seasonal biases are computed."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        ind_data = results["tas"]["cmip6_individual_data"]
        assert "DJF" in ind_data
        assert "JJA" in ind_data


# -- Precipitation (pr) special handling -------------------------------------


class TestPrecipitationBias:
    """pr gets BrBG colormaps, mm/day units, and relative bias figures."""

    def _make_pr_vr(self, synth_obs):
        """Build a minimal ``vr`` dict for ``pr`` with realistic units (kg/m²/s)."""
        from feather.data.variables import get_var
        from feather.util.temporal import climatology, seasonal_climatology

        var_info = get_var("pr")

        # Synthetic precipitation obs on the synth_obs grid (use t2m rescaled)
        # Typical precip ~3e-5 kg/m²/s; keep all values positive
        obs_da = synth_obs["t2m"] * 1e-7 + 3e-5  # ~3e-5 kg/m²/s, positive
        obs_clim = climatology(obs_da)
        obs_seasonal = seasonal_climatology(obs_da)

        lats = obs_clim.lat.values
        lons = obs_clim.lon.values
        obs_clim_common = obs_clim
        obs_seasonal_common = {s: obs_seasonal[s] for s in obs_seasonal}

        # Model bias field (small positive and negative values)
        bias = obs_clim * 0.1  # 10% wet bias
        seasonal_biases = {
            season: obs_seasonal_common[season] * 0.1
            for season in obs_seasonal_common
        }
        seasonal_regrids = {
            season: obs_seasonal_common[season] * 1.1
            for season in obs_seasonal_common
        }

        model_results = {
            "ifs-fesom": {
                "annual_regrid": obs_clim_common * 1.1,
                "seasonal_regrids": seasonal_regrids,
                "global_mean": float(obs_clim.mean()),
                "annual_bias": bias,
                "annual_bias_gmean": float(bias.mean()),
                "annual_rmse": float((bias ** 2).mean() ** 0.5),
                "seasonal_biases": seasonal_biases,
                "ttest_statistic": 1.5,
                "ttest_pvalue": 0.13,
                "ftest_statistic": 1.02,
                "ftest_pvalue": 0.45,
            }
        }
        from feather.util.spatial import compute_latlon_areas
        common_area = compute_latlon_areas(lats, lons)
        colorbar_ranges = GlobalBiases._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
        )
        return {
            "models": model_results,
            "obs": {
                "clim": obs_clim_common,
                "seasonal_clim": obs_seasonal_common,
                "global_mean": float(obs_clim.mean()),
            },
            "var_info": var_info,
            "colorbar_ranges": colorbar_ranges,
            "cmip6_data": {},
            "cmip6_info": {},
            "cmip6_individual_data": {},
        }

    def test_pr_generates_six_figures(
        self, synth_obs, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """_plot_variable('pr') returns 6 figures: 3 absolute + 3 relative."""
        vr = self._make_pr_vr(synth_obs)
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        figures = diag._plot_variable("pr", vr)
        plt.close("all")
        assert len(figures) == 6

    def test_pr_absolute_bias_figure_ids(
        self, synth_obs, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Absolute bias figures follow pr_{period}_bias_combined naming."""
        vr = self._make_pr_vr(synth_obs)
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        figures = diag._plot_variable("pr", vr)
        plt.close("all")
        ids = {meta["figure_id"] for _, meta in figures}
        for period in ["annual", "djf", "jja"]:
            assert f"pr_{period}_bias_combined" in ids

    def test_pr_relative_bias_figure_ids(
        self, synth_obs, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Relative bias figures follow pr_{period}_relative_bias_combined naming."""
        vr = self._make_pr_vr(synth_obs)
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        figures = diag._plot_variable("pr", vr)
        plt.close("all")
        ids = {meta["figure_id"] for _, meta in figures}
        for period in ["annual", "djf", "jja"]:
            assert f"pr_{period}_relative_bias_combined" in ids

    def test_pr_relative_bias_plot_type(
        self, synth_obs, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Relative bias figures have plot_type='combined_map'."""
        vr = self._make_pr_vr(synth_obs)
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        figures = diag._plot_variable("pr", vr)
        plt.close("all")
        rel_metas = [
            meta for _, meta in figures
            if "relative_bias" in meta["figure_id"]
        ]
        assert len(rel_metas) == 3
        for meta in rel_metas:
            assert meta["plot_type"] == "combined_map"

    def test_pr_non_pr_still_three_figures(
        self, synth_healpix, synth_obs, mock_model_loader,
        mock_obs_loader, minimal_config,
    ):
        """tas (non-pr) still produces only 3 absolute bias figures."""
        from feather.data.variables import get_var
        from feather.util.temporal import climatology, seasonal_climatology

        # Reuse _make_mock_results logic inline
        var_info = get_var("tas")
        obs_clim = climatology(synth_obs["t2m"])
        obs_seasonal_common = {
            s: v
            for s, v in seasonal_climatology(synth_obs["t2m"]).items()
        }
        bias = obs_clim * 0.01
        model_results = {
            "ifs-fesom": {
                "annual_regrid": obs_clim,
                "seasonal_regrids": obs_seasonal_common,
                "global_mean": float(obs_clim.mean()),
                "annual_bias": bias,
                "annual_bias_gmean": float(bias.mean()),
                "annual_rmse": float((bias ** 2).mean() ** 0.5),
                "seasonal_biases": {s: bias for s in obs_seasonal_common},
                "ttest_statistic": 1.5,
                "ttest_pvalue": 0.13,
                "ftest_statistic": 1.02,
                "ftest_pvalue": 0.45,
            }
        }
        colorbar_ranges = GlobalBiases._compute_colorbar_ranges(
            model_results, obs_clim, obs_seasonal_common,
        )
        vr = {
            "models": model_results,
            "obs": {
                "clim": obs_clim,
                "seasonal_clim": obs_seasonal_common,
                "global_mean": float(obs_clim.mean()),
            },
            "var_info": var_info,
            "colorbar_ranges": colorbar_ranges,
            "cmip6_data": {},
            "cmip6_info": {},
            "cmip6_individual_data": {},
        }

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            figures = diag._plot_variable("tas", vr)
        plt.close("all")
        assert len(figures) == 3

    def test_pr_skip_requires_six_figure_ids(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """run() only skips pr when all 6 figure IDs (abs + rel) exist."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)

        # Create only the 3 absolute bias files — relative are missing
        for period in ["annual", "djf", "jja"]:
            fid = f"pr_{period}_bias_combined"
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        # Should NOT skip — relative bias files are absent.
        # Track whether _compute_variable is called (= not skipped).
        with patch.object(
            diag, "_compute_variable", wraps=diag._compute_variable,
        ) as spy:
            saved = diag.run(skip_existing=True)

        # _compute_variable was called (variable was NOT skipped)
        spy.assert_called_once_with("pr")
        # Mock loader lacks pr so compute returns None — no new figures
        assert len(saved) == 0

    def test_pr_skip_when_all_six_exist(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """run() skips pr when all 6 figure IDs exist."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["pr"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)

        all_ids = (
            [f"pr_{p}_bias_combined" for p in ["annual", "djf", "jja"]]
            + [f"pr_{p}_relative_bias_combined" for p in ["annual", "djf", "jja"]]
        )
        for fid in all_ids:
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        saved = diag.run(skip_existing=True)

        assert len(saved) == 6
        for png_path, _ in saved:
            assert png_path.read_bytes() == b"fake"


# -- Statistical significance tests ----------------------------------------


class TestGlobalBiasesStatistics:
    """Tests for t-test and ANOVA statistics in GlobalBiases."""

    def test_compute_has_ttest_fields(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """compute() results contain ttest_statistic and ttest_pvalue per model."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert "ttest_statistic" in mdata
        assert "ttest_pvalue" in mdata
        assert isinstance(mdata["ttest_statistic"], float)
        assert isinstance(mdata["ttest_pvalue"], float)

    def test_compute_has_seasonal_ttest(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """compute() results contain seasonal t-test stats."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert "seasonal_ttest" in mdata
        for season in ["DJF", "JJA"]:
            if season in mdata["seasonal_ttest"]:
                s_tt = mdata["seasonal_ttest"][season]
                assert "ttest_statistic" in s_tt
                assert "ttest_pvalue" in s_tt

    def test_plot_metadata_has_ttest(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Plot metadata summary_statistics contains t-test fields."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        # Check annual metadata
        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        stats = annual_meta["summary_statistics"]
        assert "ifs-fesom" in stats
        assert "t_test_statistic" in stats["ifs-fesom"]
        assert "t_test_p_value" in stats["ifs-fesom"]

    def test_compute_has_ftest_fields(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """compute() results contain ftest_statistic and ftest_pvalue per model."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mdata = results["tas"]["models"]["ifs-fesom"]
        assert "ftest_statistic" in mdata
        assert "ftest_pvalue" in mdata
        assert isinstance(mdata["ftest_statistic"], float)
        assert isinstance(mdata["ftest_pvalue"], float)

    def test_plot_metadata_has_variance_ratio(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Plot metadata summary_statistics contains variance_ratio fields."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        stats = annual_meta["summary_statistics"]
        assert "variance_ratio" in stats["ifs-fesom"]
        assert "variance_ratio_p_value" in stats["ifs-fesom"]

    def test_no_aggregate_anova_in_metadata(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Aggregate ANOVA is not present (replaced by per-model variance ratio)."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "tas_annual_bias_combined"
        )
        assert "ANOVA (models)" not in annual_meta["summary_statistics"]

    def test_cmip6_mmm_has_ttest(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 MMM compute results contain t-test stats."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_annual = results["tas"]["cmip6_data"]["annual"]
        assert "ttest_statistic" in cmip6_annual
        assert "ttest_pvalue" in cmip6_annual

    def test_cmip6_individual_has_ttest(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual CMIP6 models have t-test stats."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        ind_data = results["tas"]["cmip6_individual_data"]["annual"]
        for label, entry in ind_data.items():
            assert "ttest_statistic" in entry
            assert "ttest_pvalue" in entry

    def test_cmip6_mmm_has_ftest(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 MMM compute results contain variance ratio stats."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_annual = results["tas"]["cmip6_data"]["annual"]
        assert "ftest_statistic" in cmip6_annual
        assert "ftest_pvalue" in cmip6_annual

    def test_cmip6_individual_has_ftest(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual CMIP6 models have variance ratio stats."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["tas"],
        )
        results = diag.compute()

        ind_data = results["tas"]["cmip6_individual_data"]["annual"]
        for label, entry in ind_data.items():
            assert "ftest_statistic" in entry
            assert "ftest_pvalue" in entry


class TestGlobalBiasesEnsemble:
    """Tests for ensemble mean/median bias maps in GlobalBiases."""

    def test_single_model_no_ens_data(self, mock_model_loader, mock_obs_loader,
                                       minimal_config):
        """With one model, ens_data is empty."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["ens_data"] == {}

    def test_multi_model_has_ens_data(self, mock_multi_model_loader,
                                       mock_obs_loader, multi_model_config):
        """With multiple models, ens_data has annual and seasonal entries."""
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        ens_data = results["tas"]["ens_data"]
        assert "annual" in ens_data
        assert "DJF" in ens_data
        assert "JJA" in ens_data

    def test_ens_data_annual_keys(self, mock_multi_model_loader,
                                   mock_obs_loader, multi_model_config):
        """Annual ens_data contains mean/median bias fields."""
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        annual = results["tas"]["ens_data"]["annual"]
        for key in ("mean", "median", "mean_bias", "median_bias",
                    "mean_bias_gmean", "median_bias_gmean",
                    "mean_rmse", "median_rmse"):
            assert key in annual, f"Missing key: {key}"

    def test_ens_bias_shape_matches_obs(self, mock_multi_model_loader,
                                         mock_obs_loader, multi_model_config):
        """Ensemble bias arrays have the same shape as obs common grid."""
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        vr = results["tas"]
        obs_shape = vr["obs"]["clim"].shape
        annual = vr["ens_data"]["annual"]
        assert annual["mean_bias"].shape == obs_shape
        assert annual["median_bias"].shape == obs_shape

    def test_ens_bias_gmean_is_scalar(self, mock_multi_model_loader,
                                       mock_obs_loader, multi_model_config):
        """mean_bias_gmean and median_bias_gmean are finite floats."""
        import math
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        annual = results["tas"]["ens_data"]["annual"]
        assert math.isfinite(annual["mean_bias_gmean"])
        assert math.isfinite(annual["median_bias_gmean"])

    def test_compute_ens_stats_static(self):
        """Static _compute_ens_stats returns correct mean for identical fields."""
        import numpy as np
        import xarray as xr
        from feather.util.spatial import compute_latlon_areas

        lats = np.array([-45.0, 0.0, 45.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0])
        data_a = xr.DataArray(
            np.ones((3, 4)) * 290.0, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        data_b = xr.DataArray(
            np.ones((3, 4)) * 292.0, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        obs = xr.DataArray(
            np.ones((3, 4)) * 288.0, dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
        )
        model_results = {
            "m1": {"annual_regrid": data_a, "seasonal_regrids": {}},
            "m2": {"annual_regrid": data_b, "seasonal_regrids": {}},
        }
        area = compute_latlon_areas(lats, lons)
        ens = GlobalBiases._compute_ens_stats(
            model_results, obs, {}, area,
        )

        assert "annual" in ens
        np.testing.assert_allclose(
            ens["annual"]["mean_bias"].values, 3.0, atol=1e-6,
        )
        np.testing.assert_allclose(
            ens["annual"]["mean_bias_gmean"], 3.0, atol=1e-3,
        )

    def test_multi_model_generates_ens_figures(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """With multiple models, _plot_variable produces ens figures per period."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])
        figure_ids = [meta["figure_id"] for _, meta in figures]

        assert "tas_annual_ens_bias_combined" in figure_ids
        assert "tas_djf_ens_bias_combined" in figure_ids
        assert "tas_jja_ens_bias_combined" in figure_ids
        plt.close("all")

    def test_single_model_no_ens_figures(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """With one model, no ensemble figure is produced."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])
        figure_ids = [meta["figure_id"] for _, meta in figures]

        assert not any("ens_bias_combined" in fid for fid in figure_ids)
        plt.close("all")

    def test_ens_figure_metadata(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """Ensemble figure metadata has correct figure_id and plot_type."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])

        ens_metas = [
            meta for _, meta in figures
            if "ens_bias_combined" in meta["figure_id"]
        ]
        assert len(ens_metas) == 3  # annual, DJF, JJA
        for meta in ens_metas:
            assert meta["plot_type"] == "combined_bias_map"
            assert meta.get("summary_statistics") is not None
        plt.close("all")

    def test_ens_figure_summary_stats_keys(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """Ensemble figure summary_statistics includes ens. mean and median entries."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])

        annual_ens = next(
            meta for _, meta in figures
            if meta["figure_id"] == "tas_annual_ens_bias_combined"
        )
        stats = annual_ens["summary_statistics"]
        # Keys include bold mathtext member count, e.g. "EERIE ens. mean $\mathbf{(3)}$"
        assert any("EERIE ens. mean" in k for k in stats)
        assert any("EERIE ens. median" in k for k in stats)
        plt.close("all")

    def test_ens_panel_labels_include_member_count(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """Ensemble panel labels contain the member count and mathtext bold."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        figures = diag._plot_variable("tas", results["tas"])

        annual_ens = next(
            meta for _, meta in figures
            if meta["figure_id"] == "tas_annual_ens_bias_combined"
        )
        stats = annual_ens["summary_statistics"]

        # Each EERIE label should contain "(N)" for the member count
        eerie_mean_key = next(k for k in stats if "EERIE ens. mean" in k)
        eerie_med_key = next(k for k in stats if "EERIE ens. median" in k)
        assert "(" in eerie_mean_key and ")" in eerie_mean_key
        assert "(" in eerie_med_key and ")" in eerie_med_key

        # Member count should equal number of configured models (all have tas)
        n = results["tas"]["ens_data"]["annual"]["n_members"]
        assert str(n) in eerie_mean_key
        assert str(n) in eerie_med_key

        # n_members stored in summary stats
        assert stats[eerie_mean_key]["n_members"] == n
        assert stats[eerie_med_key]["n_members"] == n
        plt.close("all")

    def test_ens_n_members_matches_available_models(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """n_members in ens_data equals the number of models that loaded."""
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        n_models_loaded = len(results["tas"]["models"])
        assert results["tas"]["ens_data"]["annual"]["n_members"] == n_models_loaded

    def test_skip_includes_ens_figure_ids_multi_model(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """run() skips variable only when all ens figures also exist."""
        import matplotlib.pyplot as plt
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)

        # Create all per-model figures but NOT the ens figures
        for p in ["annual", "djf", "jja"]:
            (diag.output_dir / f"tas_{p}_bias_combined.png").write_bytes(b"x")
            (diag.output_dir / f"tas_{p}_bias_combined.json").write_text("{}")

        saved = diag.run(skip_existing=True)

        # Should NOT skip because ens figures are missing → should regenerate
        figure_ids = [paths[0].stem for paths in saved]
        assert any("ens_bias_combined" in fid for fid in figure_ids)
        plt.close("all")

    def test_skip_with_all_ens_figures_present(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """run() skips variable when per-model AND ens figures all exist."""
        diag = GlobalBiases(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)

        # Create all required figure files
        for p in ["annual", "djf", "jja"]:
            for suffix in [".png", ".json"]:
                (diag.output_dir / f"tas_{p}_bias_combined{suffix}").write_bytes(b"x")
                (diag.output_dir / f"tas_{p}_ens_bias_combined{suffix}").write_bytes(b"x")

        saved = diag.run(skip_existing=True)

        # All pre-existing → should skip and return 6 paths (3 per-model + 3 ens)
        assert len(saved) == 6
        assert all(paths[0].read_bytes() == b"x" for paths in saved)
