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


class TestTimeseriesCMIP6:
    """Tests for CMIP6 integration in timeseries diagnostic."""

    def test_cmip6_disabled_no_data(self, mock_model_loader, mock_obs_loader,
                                     minimal_config):
        """When CMIP6 is disabled, cmip6_ts is None."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        assert results["avg_2t"]["cmip6_ts"] is None
        assert results["avg_2t"]["cmip6_info"] == {}

    def test_cmip6_enabled_has_timeseries(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_ts is a DataArray with time dim."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
        )
        results = diag.compute()

        cmip6_ts = results["avg_2t"]["cmip6_ts"]
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
        )
        results = diag.compute()

        cmip6_ts = results["avg_2t"]["cmip6_ts"]
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
        )
        results = diag.compute()

        info = results["avg_2t"]["cmip6_info"]
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
        )
        results = diag.compute()
        pairs = diag.plot(results)

        fig, meta = pairs[0]
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert "CMIP6 MMM" in labels
        plt.close(fig)

    def test_cmip6_metadata_includes_info(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Metadata includes cmip6_info when CMIP6 is enabled."""
        diag = TimeseriesDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta.get("cmip6_info") is not None
        assert meta["cmip6_info"]["n_members"] >= 1
        plt.close("all")
