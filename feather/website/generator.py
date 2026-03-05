"""Static site generator for feather climate diagnostics.

Scans ``{output_dir}/figures/`` and ``{output_dir}/analysis/``, renders
Jinja2 templates, and writes a self-contained static HTML site to
``{output_dir}/site/``.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)


def _humanize(name: str) -> str:
    """Convert 'global_biases' -> 'Global Biases'."""
    return name.replace("_", " ").title()


# Legacy DestinE model names that should be displayed in uppercase
_LEGACY_UPPERCASE_MODELS = {"ifs-fesom", "ifs-nemo", "icon"}


def _model_display_name(name: str, config: "FeatherConfig | None" = None) -> str:
    """Return display-friendly model name.

    If *config* provides a ``ModelConfig`` for *name*, uses its stored
    display name.  Otherwise falls back to uppercasing known legacy
    DestinE model names and keeping everything else as-is.
    """
    if config and config.model_configs:
        mc = config.model_configs.get(name)
        if mc:
            return mc.name
    # Legacy fallback
    if name.lower() in _LEGACY_UPPERCASE_MODELS:
        return name.upper()
    return name


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.debug("Could not load JSON: %s", path)
        return {}


class SiteGenerator:
    """Build a static HTML site from diagnostic output.

    Parameters
    ----------
    config : FeatherConfig
        Feather configuration (for output paths and website settings).
    """

    def __init__(self, config: FeatherConfig) -> None:
        self.config = config
        self.figures_dir = Path(config.output_dir) / "figures"
        self.analysis_dir = Path(config.output_dir) / "analysis"
        self.site_dir = Path(config.output_dir) / "site"

        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        # Bind config to the display name filter so templates get
        # config-driven model names automatically.
        self.env.filters["model_display_name"] = (
            lambda name: _model_display_name(name, config)
        )

    # ── Registry lookup ───────────────────────────────────────────────

    def _get_registry_info(self) -> dict[str, dict[str, Any]]:
        """Get title and group for each registered diagnostic."""
        info: dict[str, dict[str, Any]] = {}
        try:
            from feather.diag.registry import list_diagnostics

            for entry in list_diagnostics():
                info[entry["name"]] = {
                    "title": entry["title"],
                    "group": entry["group"],
                }
        except Exception:
            logger.warning(
                "Could not load diagnostic registry; using fallback names"
            )
        return info

    # ── Data collection ───────────────────────────────────────────────

    def collect_diagnostics(self) -> list[dict[str, Any]]:
        """Scan figures + analysis directories and build diagnostic dicts.

        Returns
        -------
        list of dict
            One dict per diagnostic with keys: name, title, group,
            figures, synthesis, thumbnail, has_analysis, has_cmip6,
            cmip6_info.
        """
        registry = self._get_registry_info()
        diagnostics = []

        if not self.figures_dir.is_dir():
            logger.warning("No figures directory: %s", self.figures_dir)
            return diagnostics

        for diag_dir in sorted(self.figures_dir.iterdir()):
            if not diag_dir.is_dir():
                continue

            name = diag_dir.name
            reg = registry.get(name, {})

            pngs = sorted(diag_dir.glob("*.png"))
            if not pngs:
                continue

            figures = []
            first_meta_group = None

            for png_path in pngs:
                stem = png_path.stem
                meta_path = diag_dir / f"{stem}.json"
                metadata = _load_json(meta_path) if meta_path.exists() else {}

                if first_meta_group is None and metadata.get("group"):
                    first_meta_group = metadata["group"]

                analysis_path = (
                    self.analysis_dir / name / f"{stem}_analysis.json"
                )
                analysis = (
                    _load_json(analysis_path)
                    if analysis_path.exists()
                    else {}
                )

                figures.append(
                    {
                        "filename": png_path.name,
                        "stem": stem,
                        "title": metadata.get("title", _humanize(stem)),
                        "metadata": metadata,
                        "analysis": analysis,
                    }
                )

            # Load synthesis
            synth_path = self.analysis_dir / name / "synthesis.json"
            synthesis = (
                _load_json(synth_path) if synth_path.exists() else {}
            )

            # Extract CMIP6 info from figure metadata
            cmip6_info: dict[str, Any] = {}
            for fig in figures:
                if fig["metadata"].get("cmip6_info"):
                    cmip6_info = fig["metadata"]["cmip6_info"]
                    break

            # Extract unique CMIP6 model names
            cmip6_model_names: list[str] = []
            if cmip6_info:
                models_used = cmip6_info.get("models_used", {})
                if isinstance(models_used, dict):
                    first_var = next(iter(models_used), None)
                    if first_var:
                        cmip6_model_names = sorted(
                            set(
                                m.split("/")[0]
                                for m in models_used.get(first_var, [])
                            )
                        )
                elif isinstance(models_used, list):
                    cmip6_model_names = sorted(
                        set(m.split("/")[0] for m in models_used)
                    )

            # Determine group: registry > first figure metadata > "other"
            group = reg.get("group") or first_meta_group or "other"

            diagnostics.append(
                {
                    "name": name,
                    "title": reg.get("title", _humanize(name)),
                    "group": group,
                    "figures": figures,
                    "synthesis": synthesis,
                    "thumbnail": pngs[0].name,
                    "has_analysis": bool(synthesis),
                    "has_cmip6": bool(cmip6_info),
                    "cmip6_info": cmip6_info,
                    "cmip6_model_names": cmip6_model_names,
                }
            )

        return diagnostics

    def _group_diagnostics(
        self, diagnostics: list[dict[str, Any]]
    ) -> list[tuple[str, str, list[dict[str, Any]]]]:
        """Group diagnostics by group field.

        Returns
        -------
        list of (group_key, group_label, diag_list)
            Ordered by ``config.website["group_order"]``, with any
            remaining groups appended alphabetically.
        """
        group_order = self.config.website.get("group_order", [])
        group_labels = self.config.website.get("group_labels", {})

        # Bucket diagnostics by group
        buckets: dict[str, list[dict[str, Any]]] = {}
        for diag in diagnostics:
            g = diag["group"]
            buckets.setdefault(g, []).append(diag)

        result: list[tuple[str, str, list[dict[str, Any]]]] = []
        seen: set[str] = set()

        # Ordered groups first
        for g in group_order:
            if g in buckets:
                label = group_labels.get(g, _humanize(g))
                result.append((g, label, buckets[g]))
                seen.add(g)

        # Remaining groups alphabetically
        for g in sorted(buckets):
            if g not in seen:
                label = group_labels.get(g, _humanize(g))
                result.append((g, label, buckets[g]))

        return result

    # ── Site building ─────────────────────────────────────────────────

    def build(self) -> Path:
        """Generate the static site.

        Returns
        -------
        Path
            The site output directory.
        """
        logger.info("Collecting diagnostics for website...")
        diagnostics = self.collect_diagnostics()
        logger.info("Found %d diagnostics with figures", len(diagnostics))

        groups = self._group_diagnostics(diagnostics)

        # Prepare output directory
        site_dir = self.site_dir
        if site_dir.exists():
            shutil.rmtree(site_dir)
        site_dir.mkdir(parents=True, exist_ok=True)

        # Copy static assets
        static_src = Path(__file__).parent / "static"
        static_dst = site_dir / "static"
        if static_src.is_dir():
            shutil.copytree(static_src, static_dst)
        else:
            static_dst.mkdir(parents=True, exist_ok=True)

        # Copy figure PNGs into site
        figures_dst = site_dir / "figures"
        figures_dst.mkdir(parents=True, exist_ok=True)
        for diag in diagnostics:
            diag_fig_src = self.figures_dir / diag["name"]
            diag_fig_dst = figures_dst / diag["name"]
            diag_fig_dst.mkdir(parents=True, exist_ok=True)
            for fig in diag["figures"]:
                src = diag_fig_src / fig["filename"]
                dst = diag_fig_dst / fig["filename"]
                shutil.copy2(src, dst)

        # Website config
        ws = self.config.website
        site_title = ws.get(
            "title", "Feather — Climate Model Evaluation"
        )
        site_subtitle = ws.get(
            "subtitle",
            "Climate Model Evaluation",
        )

        # Period from config
        period = self.config.get_period()
        period_str = f"{period[0]}\u2013{period[1]}"

        # Display-friendly model names
        display_models = [
            _model_display_name(m, self.config) for m in self.config.models
        ]

        # Common template context
        base_ctx = {
            "diagnostics": diagnostics,
            "groups": groups,
            "models": display_models,
            "period": period_str,
            "site_title": site_title,
            "site_subtitle": site_subtitle,
        }

        # Render index.html
        index_tpl = self.env.get_template("index.html")
        index_html = index_tpl.render(**base_ctx)
        (site_dir / "index.html").write_text(index_html, encoding="utf-8")
        logger.info("Generated index.html")

        # Render per-diagnostic pages
        diag_tpl = self.env.get_template("diagnostic.html")
        for diag in diagnostics:
            html = diag_tpl.render(diag=diag, **base_ctx)
            (site_dir / f"{diag['name']}.html").write_text(
                html, encoding="utf-8"
            )

        logger.info(
            "Site generated: %s (%d pages)", site_dir, len(diagnostics) + 1
        )
        return site_dir
