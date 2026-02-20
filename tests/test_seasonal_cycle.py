"""Tests for the seasonal_cycle diagnostic."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from feather.diag.seasonal_cycle import SeasonalCycleDiag
from tests.conftest import MockCMIP6Loader


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


class TestSeasonalCycleCMIP6:
    """Tests for CMIP6 integration in seasonal_cycle diagnostic."""

    def test_cmip6_disabled_no_data(self, mock_model_loader, mock_obs_loader,
                                     minimal_config):
        """When CMIP6 is disabled, cmip6_monthly is None."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
        )
        results = diag.compute()

        assert results["avg_2t"]["cmip6_monthly"] is None
        assert results["avg_2t"]["cmip6_info"] == {}

    def test_cmip6_enabled_has_monthly(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_monthly has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
        )
        results = diag.compute()

        cmip6_monthly = results["avg_2t"]["cmip6_monthly"]
        assert cmip6_monthly is not None
        assert len(cmip6_monthly) == 12

    def test_cmip6_monthly_reasonable_values(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """CMIP6 MMM monthly values are in reasonable range."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
        )
        results = diag.compute()

        cmip6_monthly = results["avg_2t"]["cmip6_monthly"]
        assert np.all(cmip6_monthly.values > 260)
        assert np.all(cmip6_monthly.values < 310)

    def test_cmip6_info_populated(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_info has n_members and models_used."""
        diag = SeasonalCycleDiag(
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
        diag = SeasonalCycleDiag(
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
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta.get("cmip6_info") is not None
        assert meta["cmip6_info"]["n_members"] >= 1
        plt.close("all")
