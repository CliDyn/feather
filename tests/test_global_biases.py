"""Tests for the global_biases diagnostic."""

import json
from collections import OrderedDict
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        assert "avg_2t" in results
        vr = results["avg_2t"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr
        assert "ifs-fesom" in vr["models"]

    def test_model_results_fields(self, mock_model_loader, mock_obs_loader,
                                   minimal_config):
        """Each model entry has required fields."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        results = diag.compute()

        mdata = results["avg_2t"]["models"]["ifs-fesom"]
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        obs = results["avg_2t"]["obs"]
        assert "clim" in obs
        assert "global_mean" in obs
        assert isinstance(obs["global_mean"], float)

    def test_bias_is_small_for_matching_data(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """With matching synth data, global mean bias is small."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        results = diag.compute()

        mdata = results["avg_2t"]["models"]["ifs-fesom"]
        # Synth model and obs have the same pattern -> small bias
        assert abs(mdata["annual_bias_gmean"]) < 3.0

    def test_rmse_is_small_for_matching_data(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """RMSE is small when model matches obs."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        results = diag.compute()

        mdata = results["avg_2t"]["models"]["ifs-fesom"]
        assert mdata["annual_rmse"] < 5.0

    def test_seasonal_biases_computed(
        self, mock_model_loader, mock_obs_loader, minimal_config,
    ):
        """Seasonal biases are computed for DJF and JJA."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        results = diag.compute()

        seasonal = results["avg_2t"]["models"]["ifs-fesom"]["seasonal_biases"]
        assert "DJF" in seasonal
        assert "JJA" in seasonal

    def test_custom_variables(self, mock_model_loader, mock_obs_loader,
                               minimal_config):
        """Constructor variables= overrides class default."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        assert diag.variables == ["avg_2t"]


# -- GlobalBiases plot tests (mocked) --------------------------------------


class TestGlobalBiasesPlot:
    """Tests for GlobalBiases.plot() -- mocks map plotting."""

    def _make_mock_results(self, synth_healpix, synth_obs):
        """Produce results dict from compute() for testing plot()."""
        from feather.data.variables import get_var
        from feather.util.spatial import global_mean
        from feather.util.temporal import climatology, seasonal_climatology

        var = "avg_2t"
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
            },
        }
        obs_seasonal_common = {}
        for season in ["DJF", "JJA"]:
            if season in obs_seasonal:
                obs_seasonal_common[season] = obs_seasonal[season].interp(
                    lat=target_lats, lon=target_lons,
                )

        colorbar_ranges = GlobalBiases._compute_colorbar_ranges(
            model_results, obs_clim_common, obs_seasonal_common,
        )

        return {
            "avg_2t": {
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
            variables=["avg_2t"],
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
            variables=["avg_2t"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        # Check annual bias metadata
        _, meta = pairs[0]
        assert meta["figure_id"] == "avg_2t_annual_bias_combined"
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
            variables=["avg_2t"],
        )

        mock_fig = MagicMock(spec=plt.Figure)
        with patch(
            "feather.diag.global_biases.plot_combined_bias_map",
            return_value=(mock_fig, [None, None]),
        ):
            pairs = diag.plot(results)

        figure_ids = [meta["figure_id"] for _, meta in pairs]
        assert "avg_2t_annual_bias_combined" in figure_ids
        assert "avg_2t_djf_bias_combined" in figure_ids
        assert "avg_2t_jja_bias_combined" in figure_ids

    def test_plot_models_list_in_metadata(
        self, synth_healpix, synth_obs, minimal_config,
        mock_model_loader, mock_obs_loader,
    ):
        """Metadata models list contains all models in the figure."""
        results = self._make_mock_results(synth_healpix, synth_obs)

        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        assert results["avg_2t"]["cmip6_data"] == {}
        assert results["avg_2t"]["cmip6_info"] == {}

    def test_cmip6_enabled_has_bias_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_data has annual bias."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["avg_2t"],
        )
        results = diag.compute()

        cmip6_data = results["avg_2t"]["cmip6_data"]
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        cmip6_data = results["avg_2t"]["cmip6_data"]
        assert abs(cmip6_data["annual"]["bias_gmean"]) < 5.0

    def test_cmip6_seasonal_biases(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 seasonal biases (DJF, JJA) are computed."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["avg_2t"],
        )
        results = diag.compute()

        cmip6_data = results["avg_2t"]["cmip6_data"]
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        info = results["avg_2t"]["cmip6_info"]
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
            variables=["avg_2t"],
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
        assert "avg_2t_annual_bias_combined" in figure_ids

        # CMIP6 MMM should appear in models list
        annual_meta = next(
            meta for _, meta in pairs
            if meta["figure_id"] == "avg_2t_annual_bias_combined"
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
            variables=["avg_2t"],
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
            if meta["figure_id"] == "avg_2t_annual_bias_combined"
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        ind_data = results["avg_2t"]["cmip6_individual_data"]
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        # MMM is also computed alongside individual models
        assert "annual" in results["avg_2t"]["cmip6_data"]
        # Individual data is present too
        assert results["avg_2t"]["cmip6_individual_data"] != {}

    def test_individual_models_in_combined_figure(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual CMIP6 models appear in combined figure metadata."""
        diag = GlobalBiases(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            cmip6_individual=True,
            variables=["avg_2t"],
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
            if meta["figure_id"] == "avg_2t_annual_bias_combined"
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
            variables=["avg_2t"],
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
            variables=["avg_2t"],
        )
        results = diag.compute()

        ind_data = results["avg_2t"]["cmip6_individual_data"]
        assert "DJF" in ind_data
        assert "JJA" in ind_data
