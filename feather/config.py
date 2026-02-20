"""YAML configuration loading."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FeatherConfig:
    """Feather configuration loaded from YAML."""

    model_catalogs: dict[str, str]
    models: list[str]
    obs_root: str
    obs_datasets: dict
    cmip6: dict
    dask: dict
    nereus: dict
    output_dir: str
    llm: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "FeatherConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        obs_root = raw.get("obs_root", "")

        # Resolve {obs_root} placeholders in obs_datasets paths
        obs_datasets = raw.get("obs_datasets", {})
        for ds_cfg in obs_datasets.values():
            if "path" in ds_cfg and "{obs_root}" in ds_cfg["path"]:
                ds_cfg["path"] = ds_cfg["path"].replace("{obs_root}", obs_root)

        return cls(
            model_catalogs=raw.get("model_catalogs", {}),
            models=raw.get("models", []),
            obs_root=obs_root,
            obs_datasets=obs_datasets,
            cmip6=raw.get("cmip6", {"enabled": False}),
            dask=raw.get("dask", {}),
            nereus=raw.get("nereus", {}),
            output_dir=raw.get("output_dir", "./output"),
            llm=raw.get("llm", {}),
        )
