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

    def test_legacy_nemo_absolute_salinity(self):
        """IFS-NEMO in legacy DestinE config gets absolute_salinity=True."""
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

        assert cfg.model_configs["ifs-nemo"].absolute_salinity is True
        assert cfg.model_configs["ifs-fesom"].absolute_salinity is False
        assert cfg.model_configs["icon"].absolute_salinity is False

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


class TestMemberField:
    """Tests for the per-model member field."""

    def test_model_config_default_member(self):
        mc = ModelConfig(name="test")
        assert mc.member == 1

    def test_model_config_custom_member(self):
        mc = ModelConfig(name="test", member=2)
        assert mc.member == 2

    def test_structured_format_parses_member(self):
        cfg_data = {
            "models": {
                "ifs-nemo": {
                    "grids": {"sfc": "healpix"},
                    "member": 2,
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

        assert cfg.model_configs["ifs-nemo"].member == 2

    def test_structured_format_member_defaults_to_1(self):
        cfg_data = {
            "models": {
                "ifs-fesom": {
                    "grids": {"sfc": "healpix"},
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

        assert cfg.model_configs["ifs-fesom"].member == 1

    def test_legacy_format_defaults_to_member_1(self):
        cfg_data = {
            "models": ["ifs-fesom", "ifs-nemo"],
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.model_configs["ifs-fesom"].member == 1
        assert cfg.model_configs["ifs-nemo"].member == 1


class TestMultiSource:
    """Tests for per-model data_source_type and is_multi_source()."""

    def test_model_config_new_fields_defaults(self):
        mc = ModelConfig(name="test")
        assert mc.data_source_type == ""
        assert mc.catalog_key == ""

    def test_model_config_new_fields_set(self):
        mc = ModelConfig(
            name="test",
            data_source_type="grib_healpix",
            catalog_key="ifs-fesom",
        )
        assert mc.data_source_type == "grib_healpix"
        assert mc.catalog_key == "ifs-fesom"

    def test_get_model_data_source_type_global_fallback(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            data_source={"type": "cmor"},
            model_configs={"m1": ModelConfig(name="m1")},
        )
        assert cfg.get_model_data_source_type("m1") == "cmor"

    def test_get_model_data_source_type_per_model(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            data_source={"type": "destine_catalog"},
            model_configs={
                "m1": ModelConfig(
                    name="m1", data_source_type="grib_healpix",
                ),
            },
        )
        assert cfg.get_model_data_source_type("m1") == "grib_healpix"

    def test_is_multi_source_false_single_type(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1", "m2"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            data_source={"type": "grib_healpix"},
            model_configs={
                "m1": ModelConfig(name="m1"),
                "m2": ModelConfig(name="m2"),
            },
        )
        assert cfg.is_multi_source() is False

    def test_is_multi_source_true_mixed(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m1", "m2"],
            obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            data_source={"type": "destine_catalog"},
            model_configs={
                "m1": ModelConfig(name="m1"),  # inherits destine_catalog
                "m2": ModelConfig(
                    name="m2", data_source_type="grib_healpix",
                ),
            },
        )
        assert cfg.is_multi_source() is True

    def test_yaml_parses_new_fields(self):
        cfg_data = {
            "models": {
                "ModelA": {
                    "institution": "ECMWF",
                    "catalog_key": "ifs-fesom",
                    "grids": {"sfc": "healpix"},
                },
                "ModelB": {
                    "institution": "AWI",
                    "data_source_type": "grib_healpix",
                    "data_root": "/data/grib",
                    "grids": {"sfc": "healpix"},
                },
            },
            "data_source": {"type": "destine_catalog"},
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.model_configs["ModelA"].catalog_key == "ifs-fesom"
        assert cfg.model_configs["ModelB"].data_source_type == "grib_healpix"
        assert cfg.is_multi_source() is True
        assert cfg.get_model_data_source_type("ModelA") == "destine_catalog"
        assert cfg.get_model_data_source_type("ModelB") == "grib_healpix"


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


class TestComparisonType:
    """Tests for comparison_type and comparison_description getters."""

    def test_default_multi_model(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        assert cfg.get_comparison_type() == "multi_model"

    def test_default_empty_description(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
        )
        assert cfg.get_comparison_description() == ""

    def test_resolution_sensitivity(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            project={
                "comparison_type": "resolution_sensitivity",
                "comparison_description": "Same model at 3 resolutions.",
            },
        )
        assert cfg.get_comparison_type() == "resolution_sensitivity"
        assert cfg.get_comparison_description() == "Same model at 3 resolutions."

    def test_single_model(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            project={"comparison_type": "single_model"},
        )
        assert cfg.get_comparison_type() == "single_model"

    def test_baseline_evaluation(self):
        cfg = FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp",
            project={"comparison_type": "baseline_evaluation"},
        )
        assert cfg.get_comparison_type() == "baseline_evaluation"


class TestGetSeasons:
    """Tests for the project.seasons toggle."""

    def _cfg(self, project=None):
        return FeatherConfig(
            model_catalogs={}, models=["m"], obs_root="",
            obs_datasets={}, cmip6={}, dask={}, nereus={},
            output_dir="/tmp", project=project or {},
        )

    def test_default_annual_djf_jja(self):
        assert self._cfg().get_seasons() == ["annual", "DJF", "JJA"]

    def test_all_five(self):
        cfg = self._cfg(
            {"seasons": ["annual", "DJF", "MAM", "JJA", "SON"]})
        assert cfg.get_seasons() == ["annual", "DJF", "MAM", "JJA", "SON"]

    def test_annual_prepended_and_case_insensitive(self):
        # "annual" omitted and seasons lower-case → still normalised.
        assert self._cfg({"seasons": ["djf", "son"]}).get_seasons() == [
            "annual", "DJF", "SON"]

    def test_unknown_season_ignored(self):
        assert self._cfg({"seasons": ["DJF", "BOGUS", "JJA"]}).get_seasons() == [
            "annual", "DJF", "JJA"]

    def test_order_preserved(self):
        assert self._cfg({"seasons": ["SON", "MAM"]}).get_seasons() == [
            "annual", "SON", "MAM"]

    def test_yaml_roundtrip(self):
        cfg_data = {
            "models": {"M": {}},
            "project": {
                "comparison_type": "resolution_sensitivity",
                "comparison_description": "Three resolutions",
            },
            "obs_root": "", "obs_datasets": {},
            "cmip6": {"enabled": False},
            "dask": {}, "nereus": {}, "output_dir": "/tmp",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            f.flush()
            cfg = FeatherConfig.from_yaml(f.name)

        assert cfg.get_comparison_type() == "resolution_sensitivity"
        assert cfg.get_comparison_description() == "Three resolutions"

    def test_ifs_fesom_combined_config(self):
        """IFS-FESOM combined config has resolution_sensitivity."""
        combined_path = Path(__file__).parent.parent / "configs" / "ifs_fesom_combined.yaml"
        if not combined_path.exists():
            pytest.skip("ifs_fesom_combined config not found")

        cfg = FeatherConfig.from_yaml(str(combined_path))
        assert cfg.get_comparison_type() == "resolution_sensitivity"
        assert "IFS-FESOM" in cfg.get_comparison_description()


class TestEerieConfig:
    def test_load_eerie_config(self):
        """EERIE config loads with structured model format."""
        eerie_path = Path(__file__).parent.parent / "configs" / "eerie.yaml"
        if not eerie_path.exists():
            pytest.skip("EERIE config not found")

        cfg = FeatherConfig.from_yaml(str(eerie_path))
        assert cfg.models == [
            "IFS-FESOM2-SR", "IFS-NEMO-ER", "ICON-ESM-ER",
            "HadGEM3-GC5",
        ]
        assert cfg.project["name"] == "EERIE"
        assert cfg.get_period() == ("1980", "2014")
        assert cfg.get_experiment() == "hist-1950"
        assert cfg.get_data_source_type() == "cmor"
        assert cfg.get_grid_type("IFS-FESOM2-SR", "sfc") == "latlon"
        assert cfg.model_configs["IFS-FESOM2-SR"].institution == "AWI"

        # HadGEM3 per-model overrides
        hg = cfg.model_configs["HadGEM3-GC5"]
        assert hg.data_root != ""
        assert hg.grid_label == "gr1"
        assert hg.variable_aliases["thetao"] == "thetao-con"
        assert hg.scale_factors["clt"] == 100
        assert hg.absolute_salinity is True

        # IFS-NEMO-ER also uses absolute salinity
        assert cfg.model_configs["IFS-NEMO-ER"].absolute_salinity is True

        # Models without NEMO don't use absolute salinity
        assert cfg.model_configs["IFS-FESOM2-SR"].absolute_salinity is False
        assert cfg.model_configs["ICON-ESM-ER"].absolute_salinity is False

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


class TestTerraDTConfig:
    def test_load_terradt_config(self):
        """TerraDT config loads with structured model format and member field."""
        terradt_path = Path(__file__).parent.parent / "configs" / "terradt.yaml"
        if not terradt_path.exists():
            pytest.skip("TerraDT config not found")

        cfg = FeatherConfig.from_yaml(str(terradt_path))
        assert set(cfg.models) == {"ifs-fesom", "ifs-nemo", "icon"}
        assert cfg.project["name"] == "TerraDT"
        assert cfg.get_period() == ("1990", "2014")
        assert cfg.get_experiment() == "baseline_hist"
        assert cfg.get_comparison_type() == "baseline_evaluation"
        assert "TerraDT" in cfg.get_comparison_description()
        assert cfg.get_data_source_type() == "destine_catalog"

        # Member field
        assert cfg.model_configs["ifs-fesom"].member == 1
        assert cfg.model_configs["ifs-nemo"].member == 2
        assert cfg.model_configs["icon"].member == 1

        # Grid types
        assert cfg.get_grid_type("ifs-fesom", "sfc") == "healpix"
        assert cfg.get_grid_type("ifs-nemo", "o2d") == "healpix"

        # Absolute salinity
        assert cfg.model_configs["ifs-nemo"].absolute_salinity is True
        assert cfg.model_configs["ifs-fesom"].absolute_salinity is False
