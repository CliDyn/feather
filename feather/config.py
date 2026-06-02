"""YAML configuration loading."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Default palette for models without explicit colors
_DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


@dataclass
class ModelConfig:
    """Per-model metadata.

    Attributes
    ----------
    name : str
        Display name for the model.
    institution : str
        Modelling centre (e.g. ``"ECMWF"``, ``"AWI"``).
    experiment : str
        Experiment identifier (e.g. ``"hist-1950"``).
    variant : str
        Variant label (e.g. ``"r1i1p1f1"``).
    grids : dict[str, str]
        Mapping of domain → grid type (``"healpix"`` or ``"latlon"``).
    color : str
        Hex color for plot lines/bars.
    """

    name: str
    institution: str = ""
    experiment: str = ""
    variant: str = ""
    grids: dict[str, str] = field(default_factory=dict)
    color: str = ""
    data_root: str = ""
    grid_label: str = ""
    variable_aliases: dict[str, str] = field(default_factory=dict)
    scale_factors: dict[str, float] = field(default_factory=dict)
    absolute_salinity: bool = False
    data_source_type: str = ""
    catalog_key: str = ""
    member: int = 1
    # Regional / CMIP5 / CMIP6-tree fields
    gcm: str = ""            # driving GCM directory name (CORDEX) or GCM name
    rcm: str = ""            # regional model directory name (CORDEX)
    rcm_version: str = ""    # RCM version dir (e.g. v1/v0/r2); auto-detected if ""
    driving_gcm: str = ""    # short driving-GCM label (GWL lookup, Phase 2)
    experiments: list = field(default_factory=list)  # experiments to stitch
    grid_dir: str = ""       # CMOR grid label dir for CMIP6 tree (e.g. "gn"/"gr")


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
    ocean_3d: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    website: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    # New structured fields (Phase 2)
    project: dict = field(default_factory=dict)
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)
    data_source: dict = field(default_factory=dict)

    # ── Helper methods ─────────────────────────────────────────────────

    def get_grid_type(self, model: str, domain: str = "sfc") -> str:
        """Return grid type for a model/domain pair.

        Returns ``"healpix"`` or ``"latlon"``.  Falls back to
        ``"healpix"`` for legacy DestinE configs.
        """
        mc = self.model_configs.get(model)
        if mc and mc.grids:
            return mc.grids.get(domain, mc.grids.get("sfc", "healpix"))
        return "healpix"

    def get_model_color(self, model: str) -> str:
        """Return hex color for a model.

        Checks ``model_configs`` first, then assigns from a default
        palette based on position in ``models`` list.
        """
        mc = self.model_configs.get(model)
        if mc and mc.color:
            return mc.color
        # Fallback: index into default palette
        try:
            idx = self.models.index(model)
        except ValueError:
            idx = 0
        return _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]

    def get_period(self) -> tuple[str, str]:
        """Return the default analysis period from project config.

        Falls back to ``("1990", "2014")`` for legacy configs.
        """
        period = self.project.get("period")
        if period and len(period) == 2:
            return (str(period[0]), str(period[1]))
        return ("1990", "2014")

    def get_experiment(self) -> str:
        """Return the default experiment from project config.

        Falls back to ``"baseline_hist"`` for legacy DestinE configs.
        """
        return self.project.get("experiment", "baseline_hist")

    def get_comparison_type(self) -> str:
        """Return the comparison type for LLM prompt framing.

        One of ``"multi_model"`` (default), ``"resolution_sensitivity"``,
        ``"single_model"``, or ``"baseline_evaluation"``.
        """
        return self.project.get("comparison_type", "multi_model")

    def get_comparison_description(self) -> str:
        """Return optional free-text comparison description.

        Appended to LLM prompts for project-specific context.
        """
        return self.project.get("comparison_description", "")

    def get_data_source_type(self) -> str:
        """Return global data source type.

        One of ``"destine_catalog"``, ``"cmor"``, ``"netcdf_healpix"``,
        or ``"grib_healpix"``.
        """
        return self.data_source.get("type", "destine_catalog")

    def get_model_data_source_type(self, model: str) -> str:
        """Return data source type for a specific model.

        Uses per-model ``data_source_type`` if set, otherwise falls
        back to the global ``data_source.type``.
        """
        mc = self.model_configs.get(model)
        if mc and mc.data_source_type:
            return mc.data_source_type
        return self.get_data_source_type()

    def is_multi_source(self) -> bool:
        """True when models use different data source backends.

        This triggers the :class:`CompositeModelLoader` to route
        ``load_var()`` calls to the correct backend per model.
        """
        types = set()
        for model in self.models:
            types.add(self.get_model_data_source_type(model))
        return len(types) > 1

    # ── YAML loading ───────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "FeatherConfig":
        """Load configuration from a YAML file.

        Supports two formats:

        **Legacy** (DestinE): ``models`` is a list of strings.
        Auto-populates ``model_configs`` with HEALPix grid defaults.

        **New** (structured): ``models`` is a dict mapping model names
        to metadata.  Builds ``ModelConfig`` instances and derives the
        ``models`` list from the dict keys.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        obs_root = raw.get("obs_root", "")

        # Resolve {obs_root} placeholders in obs_datasets paths
        obs_datasets = raw.get("obs_datasets", {})
        for ds_cfg in obs_datasets.values():
            if "path" in ds_cfg and "{obs_root}" in ds_cfg["path"]:
                ds_cfg["path"] = ds_cfg["path"].replace("{obs_root}", obs_root)

        # Detect config format: list → legacy, dict → new structured
        raw_models = raw.get("models", [])
        project = raw.get("project", {})
        data_source = raw.get("data_source", {})

        if isinstance(raw_models, list):
            # Legacy format: models is a list of strings
            models = raw_models
            model_configs = _build_legacy_model_configs(models)
        else:
            # New structured format: models is a dict
            models = list(raw_models.keys())
            model_configs = _build_model_configs(raw_models)

        return cls(
            model_catalogs=raw.get("model_catalogs", {}),
            models=models,
            obs_root=obs_root,
            obs_datasets=obs_datasets,
            cmip6=raw.get("cmip6", {"enabled": False}),
            dask=raw.get("dask", {}),
            nereus=raw.get("nereus", {}),
            output_dir=raw.get("output_dir", "./output"),
            ocean_3d=raw.get("ocean_3d", {}),
            llm=raw.get("llm", {}),
            website=raw.get("website", {}),
            report=raw.get("report", {}),
            project=project,
            model_configs=model_configs,
            data_source=data_source,
        )


def _build_legacy_model_configs(
    models: list[str],
) -> dict[str, ModelConfig]:
    """Build ModelConfig entries for legacy DestinE format.

    Assumes HEALPix grids for all domains.
    """
    from feather.plot.styles import MODEL_COLORS

    # NEMO-based models output absolute salinity (TEOS-10) — need SA→SP
    _NEMO_MODELS = {"ifs-nemo"}

    configs = {}
    for model in models:
        configs[model] = ModelConfig(
            name=model,
            grids={"sfc": "healpix", "o2d": "healpix",
                   "pl": "healpix", "o3d": "healpix"},
            color=MODEL_COLORS.get(model, ""),
            absolute_salinity=model in _NEMO_MODELS,
        )
    return configs


def _build_model_configs(
    raw_models: dict,
) -> dict[str, ModelConfig]:
    """Build ModelConfig entries from new structured YAML format."""
    configs = {}
    for name, cfg in raw_models.items():
        if cfg is None:
            cfg = {}
        grids = cfg.get("grids", {})
        configs[name] = ModelConfig(
            name=name,
            institution=cfg.get("institution", ""),
            experiment=cfg.get("experiment", ""),
            variant=cfg.get("variant", ""),
            grids=grids,
            color=cfg.get("color", ""),
            data_root=cfg.get("data_root", ""),
            grid_label=cfg.get("grid_label", ""),
            variable_aliases=cfg.get("variable_aliases", {}),
            scale_factors=cfg.get("scale_factors", {}),
            absolute_salinity=cfg.get("absolute_salinity", False),
            data_source_type=cfg.get("data_source_type", ""),
            catalog_key=cfg.get("catalog_key", ""),
            member=cfg.get("member", 1),
            gcm=cfg.get("gcm", ""),
            rcm=cfg.get("rcm", ""),
            rcm_version=cfg.get("rcm_version", ""),
            driving_gcm=cfg.get("driving_gcm", ""),
            experiments=cfg.get("experiments", []),
            grid_dir=cfg.get("grid_dir", ""),
        )
    return configs
