"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from feather.config import FeatherConfig


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
    assert cfg.cmip6["enabled"] is False


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
