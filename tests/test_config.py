"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from feather.config import FeatherConfig, ModelConfig


def test_from_yaml_minimal():
    """Load a minimal YAML config."""
    cfg_data = {
        "model_catalogs": {"2d": "/tmp/cat.yaml"},
        "models": ["ifs-fesom"],
        "obs_root": "/data/obs",
        "obs_datasets": {
            "ERA5": {
                "path": "{obs_root}/ERA5",
                "variables": {"t2m": "era5_t2m.nc"},
            }
        },
        "cmip6": {"enabled": False},
        "dask": {"n_workers": 2},
        "nereus": {"projection": "rob"},
        "output_dir": "/tmp/output",
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        f.flush()
        cfg = FeatherConfig.from_yaml(f.name)

    assert cfg.models == ["ifs-fesom"]
    assert cfg.model_catalogs["2d"] == "/tmp/cat.yaml"
    assert cfg.obs_root == "/data/obs"
    assert cfg.output_dir == "/tmp/output"
    # obs_root should be resolved in paths
    assert cfg.obs_datasets["ERA5"]["path"] == "/data/obs/ERA5"


def test_from_yaml_default_config():
    """Load the default config file (if present)."""
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not default_path.exists():
        pytest.skip("Default config not found")

    cfg = FeatherConfig.from_yaml(str(default_path))
    assert len(cfg.models) == 3
    assert "ifs-fesom" in cfg.models
    assert "ERA5" in cfg.obs_datasets
    assert cfg.cmip6["enabled"] is True
    assert cfg.llm["figure_analysis"]["provider"] == "vertex"


def test_obs_root_resolved():
    """Verify {obs_root} placeholder is resolved in obs paths."""
    cfg_data = {
        "obs_root": "/base/path",
        "obs_datasets": {
            "TEST": {"path": "{obs_root}/subdir"},
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        f.flush()
        cfg = FeatherConfig.from_yaml(f.name)

    assert cfg.obs_datasets["TEST"]["path"] == "/base/path/subdir"


# ── Phase 2: ModelConfig and dual-format parsing ──────────────────────


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig(name="test-model")
        assert mc.institution == ""
        assert mc.experiment == ""
        assert mc.variant == ""
        assert mc.grids == {}
        assert mc.color == ""

    def test_full_init(self):
        mc = ModelConfig(
            name="IFS-FESOM2-SR",
            institution="AWI",
            experiment="hist-1950",
            variant="r1i1p1f1",
            grids={"sfc": "latlon", "o2d": "latlon"},
            color="#1f77b4",
        )
        assert mc.name == "IFS-FESOM2-SR"
        assert mc.grids["sfc"] == "latlon"


class TestLegacyFormat:
    """Legacy DestinE format: models is a list of strings."""

    def test_models_list_populates_model_configs(self):
        cfg_data = {
            "models": ["ifs-fesom", "ifs-nemo", "icon"],
            "obs_root": "",
            "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {},
            "nereus": {},
            "output_dir": "/tmp/out",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.models == ["ifs-fesom", "ifs-nemo", "icon"]
        assert len(cfg.model_configs) == 3
        assert "ifs-fesom" in cfg.model_configs
        assert cfg.model_configs["ifs-fesom"].grids["sfc"] == "healpix"

    def test_legacy_default_period(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        assert cfg.get_period() == ("1990", "2014")

    def test_legacy_default_experiment(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        assert cfg.get_experiment() == "baseline_hist"

    def test_legacy_default_data_source(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        assert cfg.get_data_source_type() == "destine_catalog"

    def test_legacy_grid_type_defaults_healpix(self):
        cfg_data = {
            "models": ["ifs-fesom"],
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_grid_type("ifs-fesom", "sfc") == "healpix"
        assert cfg.get_grid_type("ifs-fesom", "o2d") == "healpix"


class TestStructuredFormat:
    """New structured format: models is a dict of ModelConfig dicts."""

    def test_models_dict_populates_list(self):
        cfg_data = {
            "models": {
                "IFS-FESOM2-SR": {
                    "institution": "AWI",
                    "experiment": "hist-1950",
                    "variant": "r1i1p1f1",
                    "grids": {"sfc": "latlon", "o2d": "latlon"},
                    "color": "#1f77b4",
                },
                "ICON-ESM-ER": {
                    "institution": "MPI-M",
                    "grids": {"sfc": "latlon"},
                    "color": "#2ca02c",
                },
            },
            "project": {
                "name": "EERIE",
                "period": ["1990", "2014"],
                "experiment": "hist-1950",
            },
            "data_source": {"type": "cmor", "root": "/data/eerie"},
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        # models list derived from dict keys (order may vary via yaml roundtrip)
        assert set(cfg.models) == {"IFS-FESOM2-SR", "ICON-ESM-ER"}
        assert len(cfg.model_configs) == 2
        assert cfg.model_configs["IFS-FESOM2-SR"].institution == "AWI"
        assert cfg.model_configs["ICON-ESM-ER"].color == "#2ca02c"

    def test_structured_grid_type(self):
        cfg_data = {
            "models": {
                "MyModel": {
                    "grids": {"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
                },
            },
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_grid_type("MyModel", "sfc") == "latlon"
        assert cfg.get_grid_type("MyModel", "o3d") == "latlon"

    def test_structured_period_from_project(self):
        cfg_data = {
            "models": {"M": {}},
            "project": {"period": ["2000", "2020"]},
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_period() == ("2000", "2020")

    def test_structured_experiment_from_project(self):
        cfg_data = {
            "models": {"M": {}},
            "project": {"experiment": "hist-1950"},
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_experiment() == "hist-1950"

    def test_data_source_type_cmor(self):
        cfg_data = {
            "models": {"M": {}},
            "data_source": {"type": "cmor", "root": "/data"},
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_data_source_type() == "cmor"


class TestModelColor:
    def test_color_from_model_config(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            model_configs={"m1": ModelConfig(name="m1", color="#abc123")},
        )
        assert cfg.get_model_color("m1") == "#abc123"

    def test_color_fallback_palette(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1", "m2", "m3"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        c1 = cfg.get_model_color("m1")
        c2 = cfg.get_model_color("m2")
        assert c1 != c2  # different colors from palette

    def test_unknown_model_gets_first_color(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=[],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        color = cfg.get_model_color("unknown")
        assert color.startswith("#")


class TestEerieConfig:
    def test_load_eerie_config(self):
        """EERIE config loads with structured model format."""
        eerie_path = Path(__file__).parent.parent / "configs" / "eerie.yaml"
        if not eerie_path.exists():
            pytest.skip("EERIE config not found")

        cfg = FeatherConfig.from_yaml(str(eerie_path))
        assert cfg.models == ["IFS-FESOM2-SR", "IFS-NEMO-ER", "ICON-ESM-ER"]
        assert cfg.project["name"] == "EERIE"
        assert cfg.get_period() == ("1990", "2014")
        assert cfg.get_experiment() == "hist-1950"
        assert cfg.get_data_source_type() == "cmor"
        assert cfg.get_grid_type("IFS-FESOM2-SR", "sfc") == "latlon"
        assert cfg.model_configs["IFS-FESOM2-SR"].institution == "AWI"

    def test_default_config_backward_compat(self):
        """Default config with project section still loads as legacy."""
        default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if not default_path.exists():
            pytest.skip("Default config not found")

        cfg = FeatherConfig.from_yaml(str(default_path))
        assert cfg.models == ["ifs-fesom", "ifs-nemo", "icon"]
        assert cfg.get_grid_type("ifs-fesom", "sfc") == "healpix"
        assert cfg.get_period() == ("1990", "2014")
        assert cfg.get_experiment() == "baseline_hist"
        assert cfg.project["name"] == "DestinE"
