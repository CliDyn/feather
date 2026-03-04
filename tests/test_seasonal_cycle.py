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
            variables=["tas"],
        )
        results = diag.compute()

        assert "tas" in results
        vr = results["tas"]
        assert "models" in vr
        assert "obs" in vr
        assert "var_info" in vr

    def test_model_monthly_length(self, mock_model_loader, mock_obs_loader,
                                   minimal_config):
        """Model monthly climatology has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        monthly = results["tas"]["models"]["ifs-fesom"]
        assert len(monthly) == 12
        assert "month" in monthly.dims

    def test_obs_monthly_length(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Obs monthly climatology has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        obs_monthly = results["tas"]["obs"]
        assert len(obs_monthly) == 12

    def test_seasonal_variation(self, mock_model_loader, mock_obs_loader,
                                 minimal_config):
        """Monthly climatology shows seasonal variation."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        monthly = results["tas"]["models"]["ifs-fesom"]
        monthly_range = float(monthly.max() - monthly.min())
        assert monthly_range > 1.0  # Synth data has +-5K cycle

    def test_reasonable_values(self, mock_model_loader, mock_obs_loader,
                                minimal_config):
        """Monthly values are in reasonable temperature range."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        monthly = results["tas"]["models"]["ifs-fesom"]
        assert np.all(monthly.values > 260)
        assert np.all(monthly.values < 310)


class TestSeasonalCyclePlot:
    """Tests for SeasonalCycleDiag.plot()."""

    def test_plot_returns_figures(self, mock_model_loader, mock_obs_loader,
                                  minimal_config):
        """plot() returns (fig, meta) pairs."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
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
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta["figure_id"] == "tas_seasonal_cycle"
        assert meta["plot_type"] == "seasonal_cycle"
        assert meta["diagnostic_name"] == "seasonal_cycle"
        assert "ifs-fesom" in meta["models"]
        plt.close("all")

    def test_run_saves_files(self, mock_model_loader, mock_obs_loader,
                              minimal_config, tmp_path):
        """run() saves PNG + JSON."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        saved = diag.run()

        assert len(saved) == 1
        png_path, json_path = saved[0]
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.suffix == ".png"
        assert json_path.suffix == ".json"


class TestSeasonalCycleSkipExisting:
    """Tests for skip_existing behavior in seasonal_cycle diagnostic."""

    def test_skip_when_figure_exists(self, mock_model_loader, mock_obs_loader,
                                      minimal_config, tmp_path):
        """run() skips computation when figure already exists on disk."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "tas_seasonal_cycle.png").write_bytes(b"fake")
        (diag.output_dir / "tas_seasonal_cycle.json").write_text("{}")

        saved = diag.run(skip_existing=True)

        assert len(saved) == 1
        png_path, json_path = saved[0]
        assert png_path == diag.output_dir / "tas_seasonal_cycle.png"
        assert png_path.read_bytes() == b"fake"

    def test_no_skip_when_disabled(self, mock_model_loader, mock_obs_loader,
                                    minimal_config, tmp_path):
        """run(skip_existing=False) recomputes even when figure exists."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "tas_seasonal_cycle.png").write_bytes(b"fake")
        (diag.output_dir / "tas_seasonal_cycle.json").write_text("{}")

        saved = diag.run(skip_existing=False)

        assert len(saved) == 1
        png_path, _ = saved[0]
        assert png_path.read_bytes() != b"fake"

    def test_skip_partial_files_not_skipped(self, mock_model_loader,
                                             mock_obs_loader, minimal_config,
                                             tmp_path):
        """run() does NOT skip when only PNG exists (JSON missing)."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        diag.output_dir.mkdir(parents=True, exist_ok=True)
        (diag.output_dir / "tas_seasonal_cycle.png").write_bytes(b"fake")

        saved = diag.run(skip_existing=True)

        assert len(saved) == 1
        png_path, _ = saved[0]
        assert png_path.read_bytes() != b"fake"


class TestSeasonalCycleCMIP6:
    """Tests for CMIP6 integration in seasonal_cycle diagnostic."""

    def test_cmip6_disabled_no_data(self, mock_model_loader, mock_obs_loader,
                                     minimal_config):
        """When CMIP6 is disabled, cmip6_monthly is None."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        results = diag.compute()

        assert results["tas"]["cmip6_monthly"] is None
        assert results["tas"]["cmip6_info"] == {}

    def test_cmip6_enabled_has_monthly(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """When CMIP6 is enabled, cmip6_monthly has 12 values."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_monthly = results["tas"]["cmip6_monthly"]
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
            variables=["tas"],
        )
        results = diag.compute()

        cmip6_monthly = results["tas"]["cmip6_monthly"]
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
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
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
            variables=["tas"],
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        assert meta.get("cmip6_info") is not None
        assert meta["cmip6_info"]["n_members"] >= 1
        plt.close("all")


class TestSeasonalCycleCMIP6Individual:
    """Tests for individual CMIP6 model lines in seasonal_cycle."""

    def test_individual_populates_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_individual_monthly is non-empty, each model has 12 months."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        indiv = results["tas"]["cmip6_individual_monthly"]
        assert len(indiv) > 0
        for mname, monthly in indiv.items():
            assert len(monthly) == 12

    def test_individual_also_has_mmm(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Both individual and MMM data are present."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        assert results["tas"]["cmip6_monthly"] is not None
        assert len(results["tas"]["cmip6_individual_monthly"]) > 0

    def test_individual_false_no_data(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """cmip6_individual_monthly is empty when flag is False."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=False,
        )
        results = diag.compute()

        assert results["tas"]["cmip6_individual_monthly"] == {}

    def test_individual_plot_has_member_legend(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Legend includes 'CMIP6 members' and 'CMIP6 MMM'."""
        diag = SeasonalCycleDiag(
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
        assert "CMIP6 members" in labels
        assert "CMIP6 MMM" in labels
        plt.close(fig)

    def test_individual_line_count(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """More lines with individual=True than without."""
        diag_no = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=False,
        )
        results_no = diag_no.compute()
        pairs_no = diag_no.plot(results_no)
        n_lines_no = len(pairs_no[0][0].axes[0].get_lines())

        diag_yes = SeasonalCycleDiag(
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
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()

        for mname, monthly in results["tas"]["cmip6_individual_monthly"].items():
            assert np.all(monthly.values > 260)
            assert np.all(monthly.values < 310)

    def test_individual_metadata(
        self, mock_model_loader, mock_obs_loader,
        cmip6_config, mock_cmip6_loader,
    ):
        """Metadata models list includes CMIP6 model names."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, cmip6_config,
            cmip6_loader=mock_cmip6_loader,
            variables=["tas"],
            cmip6_individual=True,
        )
        results = diag.compute()
        pairs = diag.plot(results)

        _, meta = pairs[0]
        cmip6_models = list(results["tas"]["cmip6_individual_monthly"].keys())
        for cm in cmip6_models:
            assert cm in meta["models"]
        plt.close("all")


class TestSeasonalCycleMultiVariable:
    """Tests for expanded variable list in seasonal_cycle."""

    def test_default_variables_expanded(self):
        """Class-level variable list has 18 entries."""
        assert len(SeasonalCycleDiag.variables) == 18
        assert "tas" in SeasonalCycleDiag.variables
        assert "psl" in SeasonalCycleDiag.variables
        assert "rltcs" in SeasonalCycleDiag.variables

    def test_custom_variables_override(self, mock_model_loader, mock_obs_loader,
                                        minimal_config):
        """Constructor variables= overrides the default list."""
        diag = SeasonalCycleDiag(
            mock_model_loader, mock_obs_loader, minimal_config,
            variables=["tas"],
        )
        assert diag.variables == ["tas"]
