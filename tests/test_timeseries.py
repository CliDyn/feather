"""Tests for the timeseries diagnostic."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.diag.timeseries import TimeseriesDiag
from tests.conftest import MockCMIP6Loader


class TestTimeseriesCompute:
    """Tests for TimeseriesDiag.compute()."""

    def test_compute_structure(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """compute() returns expected nested structure."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert "tas" in results
        vr = results["tas"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr

    def test_model_timeseries_length(self, mock_model_loader, mock_obs_loader,
                                      minimal_config):
        """Model time series has correct length (12 months in synth data)."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        ts = results["tas"]["models"]["ifs-fesom"]
        assert "time" in ts.dims
        assert len(ts) == 12

    def test_obs_timeseries_length(self, mock_model_loader, mock_obs_loader,
                                    minimal_config):
        """Obs time series has correct length."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        obs_ts = results["tas"]["obs"]
        assert "time" in obs_ts.dims
        assert len(obs_ts) == 12

    def test_reasonable_values(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """Global mean values are in a reasonable temperature range."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        model_ts = results["tas"]["models"]["ifs-fesom"]
        assert np.all(model_ts.values > 260)
        assert np.all(model_ts.values < 310)

    def test_seasonal_variation(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Time series shows seasonal variation (synth data has +-5K cycle)."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        ts = results["tas"]["models"]["ifs-fesom"]
        ts_range = float(ts.max() - ts.min())
        assert ts_range > 1.0  # Should see seasonal cycle

    def test_custom_variables(self, mock_model_loader, mock_obs_loader,
                               minimal_config):
        """Constructor variables= overrides class default."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        assert diag.variables == ["tas"]


class TestTimeseriesPlot:
    """Tests for TimeseriesDiag.plot()."""

    def test_plot_returns_figures(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """plot() returns (fig, meta) pairs."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        # 1 variable → main + envelope + anomaly figures.
        assert len(pairs) == 3
        fig, meta = pairs[0]
        assert isinstance(fig, plt.Figure)
        assert isinstance(meta, dict)
        assert meta["figure_id"] == "tas_timeseries"
        ids = {m["figure_id"] for _, m in pairs}
        assert ids == {
            "tas_timeseries", "tas_timeseries_envelope", "tas_timeseries_anomaly",
        }
        for f, _ in pairs:
            plt.close(f)

    def test_plot_metadata(self, mock_model_loader, mock_obs_loader,
                            minimal_config):
        """Metadata has correct fields."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta["figure_id"] == "tas_timeseries"
        assert meta["plot_type"] == "timeseries"
        assert meta["diagnostic_name"] == "timeseries"
        assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_run_saves_files(self, mock_model_loader, mock_obs_loader,
                              minimal_config, tmp_path):
        """run() saves PNG + JSON."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        saved = diag.run()

        # main + envelope + anomaly
        assert len(saved) == 3
        for png_path, json_path in saved:
            assert png_path.exists()
            assert json_path.exists()
            assert png_path.suffix == ".png"
            assert json_path.suffix == ".json"


class TestTimeseriesSkipExisting:
    """Tests for skip_existing behavior in timeseries diagnostic."""

    def test_skip_when_figure_exists(self, mock_model_loader, mock_obs_loader,
                                      minimal_config, tmp_path):
        """run() skips computation when figure already exists on disk."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        # Pre-create all three output figures (main + envelope + anomaly)
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        for fid in ("tas_timeseries", "tas_timeseries_envelope",
                    "tas_timeseries_anomaly"):
            (diag.output_dir / f"{fid}.png").write_bytes(b"fake")
            (diag.output_dir / f"{fid}.json").write_text("{}")

        saved = diag.run(skip_existing=True)

        assert len(saved) == 3
        png_path, json_path = saved[0]
        assert png_path == diag.output_dir / "tas_timeseries.png"
        # File should not have been overwritten (still "fake")
        assert png_path.read_bytes() == b"fake"

    def test_no_skip_when_disabled(self, mock_model_loader, mock_obs_loader,
                                    minimal_config, tmp_path):
        """run(skip_existing=False) recomputes even when figure exists."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "tas_timeseries.png").write_bytes(b"fake")
        (diag.output_dir / "tas_timeseries.json").write_text("{}")

        saved = diag.run(skip_existing=False)

        assert len(saved) == 3
        png_path, _ = saved[0]
        # File should have been overwritten (no longer "fake")
        assert png_path.read_bytes() != b"fake"

    def test_skip_partial_files_not_skipped(self, mock_model_loader,
                                             mock_obs_loader, minimal_config,
                                             tmp_path):
        """run() does NOT skip when only PNG exists (JSON missing)."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "tas_timeseries.png").write_bytes(b"fake")
        # JSON is missing

        saved = diag.run(skip_existing=True)

        assert len(saved) == 3
        png_path, _ = saved[0]
        # Should have been regenerated
        assert png_path.read_bytes() != b"fake"


class TestTimeseriesCMIP6:
    """Tests for CMIP6 integration in timeseries diagnostic."""

    def test_cmip6_disabled_no_data(self, mock_model_loader, mock_obs_loader,
                                     minimal_config):
        """When CMIP6 is disabled, cmip6_ts is None."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["cmip6_ts"] is None
        assert results["tas"]["cmip6_info"] == {}

    def test_two_benchmarks_two_mmm_series(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, synth_cmip6,
    ):
        """Two benchmark loaders yield two MMM series; primary aliased."""
        b1 = MockCMIP6Loader(synth_cmip6)
        b1.label, b1.color = "CMIP6 MMM", "#7f7f7f"
        b2 = MockCMIP6Loader(synth_cmip6)
        b2.label, b2.color = "HighResMIP MMM", "#9467bd"

        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            benchmarks=[b1, b2], variables=["tas"],
        )
        results = diag.compute()
        benches = results["tas"]["benchmarks_ts"]
        assert [b["label"] for b in benches] == ["CMIP6 MMM", "HighResMIP MMM"]
        assert [b["color"] for b in benches] == ["#7f7f7f", "#9467bd"]
        # Back-compat alias points at the primary benchmark
        assert results["tas"]["cmip6_ts"] is benches[0]["ts"]
        # Plotting both benchmarks must not raise
        figs = diag._plot_single("tas", results["tas"])
        assert len(figs) >= 1
        plt.close("all")

    def test_cmip6_enabled_has_timeseries(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_ts is a DataArray with time dim."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_ts = results["tas"]["cmip6_ts"]
        assert cmip6_ts is not None
        assert "time" in cmip6_ts.dims
        assert len(cmip6_ts) == 12

    def test_cmip6_timeseries_reasonable_values(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 MMM time series values are in reasonable range."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_ts = results["tas"]["cmip6_ts"]
        assert np.all(cmip6_ts.values > 260)
        assert np.all(cmip6_ts.values < 310)

    def test_cmip6_info_populated(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_info has n_members and models_used."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        info = results["tas"]["cmip6_info"]
        assert info["n_members"] >= 1
        assert len(info["models_used"]) >= 1

    def test_cmip6_plot_has_line(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Plot includes CMIP6 MMM line when data is available."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, meta = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("CMIP6 MMM" in lbl for lbl in labels)
        plt.close(fig)

    def test_cmip6_metadata_includes_info(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Metadata includes cmip6_info when CMIP6 is enabled."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta.get("cmip6_info") is not None
        assert meta["cmip6_info"]["n_members"] >= 1
        plt.close("all")


class TestTimeseriesCMIP6Individual:
    """Tests for individual CMIP6 model lines in timeseries."""

    def test_individual_populates_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_individual_ts is non-empty, each model has time dim."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        indiv = results["tas"]["cmip6_individual_ts"]
        assert len(indiv) > 0
        for mname, ts in indiv.items():
            assert "time" in ts.dims
            assert len(ts) == 12

    def test_individual_also_has_mmm(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Both individual and MMM data are present."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        assert results["tas"]["cmip6_ts"] is not None
        assert len(results["tas"]["cmip6_individual_ts"]) > 0

    def test_individual_false_no_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_individual_ts is empty when flag is False."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=False,
        )
        results = diag.compute()

        assert results["tas"]["cmip6_individual_ts"] == {}

    def test_individual_plot_has_member_legend(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Legend includes 'CMIP6 members' and 'CMIP6 MMM'."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("CMIP6 members" in lbl for lbl in labels)
        assert any("CMIP6 MMM" in lbl for lbl in labels)
        plt.close(fig)

    def test_individual_line_count(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """More lines with individual=True than without."""
        diag_no = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=False,
        )
        results_no = diag_no.compute()
        pairs_no = diag_no.plot(results_no)
        n_lines_no = len(pairs_no[0][0].axes[0].get_lines())

        diag_yes = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results_yes = diag_yes.compute()
        pairs_yes = diag_yes.plot(results_yes)
        n_lines_yes = len(pairs_yes[0][0].axes[0].get_lines())

        assert n_lines_yes > n_lines_no
        plt.close("all")

    def test_individual_reasonable_values(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Individual model values are in 260-310K range."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        for mname, ts in results["tas"]["cmip6_individual_ts"].items():
            assert np.all(ts.values > 260)
            assert np.all(ts.values < 310)

    def test_individual_metadata(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Metadata models list includes CMIP6 model names."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        cmip6_models = list(results["tas"]["cmip6_individual_ts"].keys())
        for cm in cmip6_models:
            assert cm in meta["models"]
        plt.close("all")


class TestTimeseriesEnsemble:
    """Tests for ensemble mean/median computation and plotting."""

    def test_single_model_returns_none(self, mock_model_loader, mock_obs_loader,
                                       minimal_config):
        """With only one model, ensemble stats are None."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["ens_mean"] is None
        assert results["tas"]["ens_median"] is None

    def test_multi_model_returns_arrays(self, mock_multi_model_loader,
                                        mock_obs_loader, multi_model_config):
        """With multiple models, ens_mean and ens_median are DataArrays."""
        import xarray as xr

        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["ens_mean"] is not None
        assert results["tas"]["ens_median"] is not None
        assert isinstance(results["tas"]["ens_mean"], xr.DataArray)
        assert isinstance(results["tas"]["ens_median"], xr.DataArray)

    def test_ensemble_has_time_dim(self, mock_multi_model_loader,
                                    mock_obs_loader, multi_model_config):
        """Ensemble arrays have a time dimension."""
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert "time" in results["tas"]["ens_mean"].dims
        assert "time" in results["tas"]["ens_median"].dims

    def test_ensemble_length_matches_models(self, mock_multi_model_loader,
                                             mock_obs_loader, multi_model_config):
        """Ensemble time axis length equals model time axis length."""
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        model_len = len(list(results["tas"]["models"].values())[0])
        assert len(results["tas"]["ens_mean"]) == model_len
        assert len(results["tas"]["ens_median"]) == model_len

    def test_ensemble_values_in_range(self, mock_multi_model_loader,
                                       mock_obs_loader, multi_model_config):
        """Ensemble mean/median values are within individual model range."""
        import numpy as np

        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()

        model_vals = np.concatenate(
            [ts.values for ts in results["tas"]["models"].values()]
        )
        ens_mean = results["tas"]["ens_mean"].values
        assert np.all(ens_mean >= model_vals.min() - 1e-6)
        assert np.all(ens_mean <= model_vals.max() + 1e-6)

    def test_compute_ensemble_stats_static_two_models(self):
        """Static method returns mean and median for two identical series."""
        import numpy as np
        import xarray as xr

        time = xr.date_range("1990-01", periods=12, freq="MS")
        ts1 = xr.DataArray(np.ones(12) * 290.0, dims=["time"],
                            coords={"time": time})
        ts2 = xr.DataArray(np.ones(12) * 292.0, dims=["time"],
                            coords={"time": time})

        mean, median = TimeseriesDiag._compute_ensemble_stats(
            {"m1": ts1, "m2": ts2}
        )

        assert mean is not None
        assert median is not None
        np.testing.assert_allclose(mean.values, 291.0)
        np.testing.assert_allclose(median.values, 291.0)

    def test_compute_ensemble_stats_mismatched_scalar_coords(self):
        """Members with mismatched scalar coords (e.g. ``height`` on tas for
        only some models) still concat — coords are dropped before stacking."""
        import numpy as np
        import xarray as xr

        time = xr.date_range("1990-01", periods=12, freq="MS")
        ts1 = xr.DataArray(np.ones(12) * 290.0, dims=["time"],
                            coords={"time": time, "height": 2.0})
        ts2 = xr.DataArray(np.ones(12) * 292.0, dims=["time"],
                            coords={"time": time})

        mean, median = TimeseriesDiag._compute_ensemble_stats(
            {"m1": ts1, "m2": ts2}
        )

        assert mean is not None
        assert median is not None
        np.testing.assert_allclose(mean.values, 291.0)
        np.testing.assert_allclose(median.values, 291.0)

    def test_compute_ensemble_stats_single_returns_none(self):
        """Static method returns (None, None) for a single-model dict."""
        import numpy as np
        import xarray as xr

        time = xr.date_range("1990-01", periods=12, freq="MS")
        ts = xr.DataArray(np.ones(12) * 290.0, dims=["time"],
                           coords={"time": time})

        mean, median = TimeseriesDiag._compute_ensemble_stats({"m1": ts})

        assert mean is None
        assert median is None

    def test_compute_ensemble_stats_empty_returns_none(self):
        """Static method returns (None, None) for an empty dict."""
        mean, median = TimeseriesDiag._compute_ensemble_stats({})

        assert mean is None
        assert median is None

    def test_plot_ensemble_lines_in_legend(self, mock_multi_model_loader,
                                            mock_obs_loader, multi_model_config):
        """Plot legend includes '<project> ensemble mean' and median."""
        multi_model_config.project["name"] = "EERIE"
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("EERIE ensemble mean" in lbl for lbl in labels)
        assert any("EERIE ensemble median" in lbl for lbl in labels)
        plt.close(fig)

    def test_plot_no_ensemble_lines_single_model(self, mock_model_loader,
                                                  mock_obs_loader, minimal_config):
        """Plot does not include ensemble legend entries for a single model."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert not any("EERIE ensemble mean" in lbl for lbl in labels)
        assert not any("EERIE ensemble median" in lbl for lbl in labels)
        plt.close(fig)

    def test_ensemble_line_colors(self, mock_multi_model_loader,
                                   mock_obs_loader, multi_model_config):
        """Ensemble lines use ENS_COLOR."""
        from feather.plot.styles import ENS_COLOR
        import matplotlib.colors as mcolors

        multi_model_config.project["name"] = "EERIE"
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        ens_lines = [
            line for line in ax.get_lines()
            if "EERIE ensemble mean" in line.get_label()
            or "EERIE ensemble median" in line.get_label()
        ]
        assert len(ens_lines) == 2
        expected = mcolors.to_rgba(ENS_COLOR)
        for line in ens_lines:
            assert mcolors.to_rgba(line.get_color()) == expected
        plt.close(fig)

    def test_ensemble_median_is_dashed(self, mock_multi_model_loader,
                                        mock_obs_loader, multi_model_config):
        """Ensemble median line uses dashed linestyle."""
        multi_model_config.project["name"] = "EERIE"
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        median_line = next(
            line for line in ax.get_lines()
            if "EERIE ensemble median" in line.get_label()
        )
        assert median_line.get_linestyle() == "--"
        plt.close(fig)

    def test_ensemble_mean_is_solid(self, mock_multi_model_loader,
                                     mock_obs_loader, multi_model_config):
        """Ensemble mean line uses solid linestyle."""
        multi_model_config.project["name"] = "EERIE"
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        mean_line = next(
            line for line in ax.get_lines()
            if "EERIE ensemble mean" in line.get_label()
            and "median" not in line.get_label()
        )
        assert mean_line.get_linestyle() == "-"
        plt.close(fig)

    def test_legend_labels_include_member_count(
        self, mock_multi_model_loader, mock_obs_loader, multi_model_config,
    ):
        """Ensemble legend labels include the member count in parentheses."""
        multi_model_config.project["name"] = "EERIE"
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]

        n = len(results["tas"]["models"])
        assert f"EERIE ensemble mean ({n})" in labels
        assert f"EERIE ensemble median ({n})" in labels
        plt.close(fig)

    def test_cmip6_mmm_label_includes_count(
        self, mock_multi_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 MMM legend label includes the member count."""
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]

        m = results["tas"]["cmip6_info"]["n_members"]
        assert f"CMIP6 MMM ({m})" in labels
        plt.close(fig)

    def test_cmip6_individual_label_includes_count(
        self, mock_multi_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """'CMIP6 members' legend label includes the count of individual series."""
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, _ = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]

        k = len(results["tas"]["cmip6_individual_ts"])
        assert f"CMIP6 members ({k})" in labels
        plt.close(fig)


class TestTimeseriesMultiVariable:
    """Tests for expanded variable list in timeseries."""

    def test_default_variables_expanded(self):
        """Class-level variable list has 18 entries."""
        assert len(TimeseriesDiag.variables) == 18
        assert "tas" in TimeseriesDiag.variables
        assert "psl" in TimeseriesDiag.variables
        assert "rltcs" in TimeseriesDiag.variables

    def test_custom_variables_override(self, mock_model_loader, mock_obs_loader,
                                        minimal_config):
        """Constructor variables= overrides the default list."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        assert diag.variables == ["tas"]
