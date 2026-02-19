"""Tests for the seasonal_cycle diagnostic."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.diag.seasonal_cycle import SeasonalCycleDiag


class TestSeasonalCycleCompute:
    """Tests for SeasonalCycleDiag.compute()."""

    def test_compute_structure(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """compute() returns expected nested structure."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        assert "avg_2t" in results
        vr = results["avg_2t"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr

    def test_model_monthly_length(self, mock_model_loader, mock_obs_loader,
                                   minimal_config):
        """Model monthly climatology has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        monthly = results["avg_2t"]["models"]["ifs-fesom"]
        assert len(monthly) == 12
        assert "month" in monthly.dims

    def test_obs_monthly_length(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Obs monthly climatology has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        obs_monthly = results["avg_2t"]["obs"]
        assert len(obs_monthly) == 12

    def test_seasonal_variation(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Monthly climatology shows seasonal variation."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        monthly = results["avg_2t"]["models"]["ifs-fesom"]
        monthly_range = float(monthly.max() - monthly.min())
        assert monthly_range > 1.0  # Synth data has +-5K cycle

    def test_reasonable_values(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """Monthly values are in reasonable temperature range."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        monthly = results["avg_2t"]["models"]["ifs-fesom"]
        assert np.all(monthly.values > 260)
        assert np.all(monthly.values < 310)


class TestSeasonalCyclePlot:
    """Tests for SeasonalCycleDiag.plot()."""

    def test_plot_returns_figures(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """plot() returns (fig, meta) pairs."""
        diag = SeasonalCycleDiag(
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
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta["figure_id"] == "avg_2t_seasonal_cycle"
        assert meta["plot_type"] == "seasonal_cycle"
        assert meta["diagnostic_name"] == "seasonal_cycle"
        assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_run_saves_files(self, mock_model_loader, mock_obs_loader,
                              minimal_config, tmp_path):
        """run() saves PNG + JSON."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        saved = diag.run()

        assert len(saved) == 1
        png_path, json_path = saved[0]
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.suffix == ".png"
        assert json_path.suffix == ".json"
