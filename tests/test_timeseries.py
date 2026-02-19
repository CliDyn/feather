"""Tests for the timeseries diagnostic."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.diag.timeseries import TimeseriesDiag


class TestTimeseriesCompute:
    """Tests for TimeseriesDiag.compute()."""

    def test_compute_structure(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """compute() returns expected nested structure."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        assert "avg_2t" in results
        vr = results["avg_2t"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr

    def test_model_timeseries_length(self, mock_model_loader, mock_obs_loader,
                                      minimal_config):
        """Model time series has correct length (12 months in synth data)."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        ts = results["avg_2t"]["models"]["ifs-fesom"]
        assert "time" in ts.dims
        assert len(ts) == 12

    def test_obs_timeseries_length(self, mock_model_loader, mock_obs_loader,
                                    minimal_config):
        """Obs time series has correct length."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        obs_ts = results["avg_2t"]["obs"]
        assert "time" in obs_ts.dims
        assert len(obs_ts) == 12

    def test_reasonable_values(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """Global mean values are in a reasonable temperature range."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        model_ts = results["avg_2t"]["models"]["ifs-fesom"]
        assert np.all(model_ts.values > 260)
        assert np.all(model_ts.values < 310)

    def test_seasonal_variation(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Time series shows seasonal variation (synth data has +-5K cycle)."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        ts = results["avg_2t"]["models"]["ifs-fesom"]
        ts_range = float(ts.max() - ts.min())
        assert ts_range > 1.0  # Should see seasonal cycle

    def test_custom_variables(self, mock_model_loader, mock_obs_loader,
                               minimal_config):
        """Constructor variables= overrides class default."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["avg_2t"],
        )
        assert diag.variables == ["avg_2t"]


class TestTimeseriesPlot:
    """Tests for TimeseriesDiag.plot()."""

    def test_plot_returns_figures(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """plot() returns (fig, meta) pairs."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        assert len(pairs) == 1  # 1 variable
        fig, meta = pairs[0]
        assert isinstance(fig, plt.Figure)
        assert isinstance(meta, dict)
        plt.close(fig)

    def test_plot_metadata(self, mock_model_loader, mock_obs_loader,
                            minimal_config):
        """Metadata has correct fields."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta["figure_id"] == "avg_2t_timeseries"
        assert meta["plot_type"] == "timeseries"
        assert meta["diagnostic_name"] == "timeseries"
        assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_run_saves_files(self, mock_model_loader, mock_obs_loader,
                              minimal_config, tmp_path):
        """run() saves PNG + JSON."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        saved = diag.run()

        assert len(saved) == 1
        png_path, json_path = saved[0]
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.suffix == ".png"
        assert json_path.suffix == ".json"
