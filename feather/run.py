"""Pipeline orchestration — run diagnostics, analysis, report, and website.

Provides :func:`run_pipeline` which ties together all feather stages
into a single callable.
"""

import logging
from pathlib import Path
from typing import Any

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)


def run_pipeline(
    config: FeatherConfig,
    *,
    steps: str | list[str] = "all",
    diagnostics: list[str] | None = None,
    variables: list[str] | None = None,
    experiment: str = "baseline_hist",
    period: tuple[str, str] = ("1990", "2014"),
    api_key: str | None = None,
    openai_api_key: str | None = None,
    skip_existing: bool = True,
    compile_pdf: bool = False,
) -> dict[str, Any]:
    """Run the feather pipeline (diagnostics -> analyze -> report -> website).

    Parameters
    ----------
    config : FeatherConfig
        Pipeline configuration.
    steps : str or list of str
        Which steps to run. One or more of:
        ``"diagnostics"``, ``"analyze"``, ``"report"``, ``"website"``, ``"all"``.
    diagnostics : list of str, optional
        Only run these diagnostics (by name). Default: all registered.
    variables : list of str, optional
        Only run diagnostics that use these variables.
    experiment : str
        Model experiment key (default ``"baseline_hist"``).
    period : tuple of str
        (start_year, end_year) for time slicing.
    api_key : str, optional
        Gemini API key for the ``"analyze"`` step.
    openai_api_key : str, optional
        OpenAI API key for the ``"report"`` step.
    skip_existing : bool
        Skip already-existing outputs where supported.
    compile_pdf : bool
        Whether to compile the LaTeX report to PDF.

    Returns
    -------
    dict
        Summary with keys: ``figures``, ``analyses``, ``syntheses``,
        ``report``, ``site_dir``.
    """
    if isinstance(steps, str):
        steps = [steps]
    if "all" in steps:
        steps = ["diagnostics", "analyze", "report", "website"]

    summary: dict[str, Any] = {
        "figures": 0,
        "analyses": 0,
        "syntheses": 0,
        "report": None,
        "site_dir": None,
    }

    # ── Step 1: Diagnostics ──────────────────────────────────────────
    if "diagnostics" in steps:
        summary["figures"] = _run_diagnostics(
            config,
            diagnostics=diagnostics,
            variables=variables,
            experiment=experiment,
            period=period,
        )

    # ── Step 2: LLM analysis ────────────────────────────────────────
    if "analyze" in steps:
        from feather.llm.analyzer import FigureAnalyzer

        analyzer = FigureAnalyzer(config, api_key=api_key)
        result = analyzer.run(
            skip_existing=skip_existing,
            diagnostics=diagnostics,
        )
        summary["analyses"] = result["figure_analyses"]
        summary["syntheses"] = result["syntheses"]

    # ── Step 3: Report generation ───────────────────────────────────
    if "report" in steps:
        from feather.export.report import ReportGenerator

        gen = ReportGenerator(
            config, compile_pdf=compile_pdf, api_key=openai_api_key,
        )
        tex_path = gen.run(skip_existing=skip_existing)
        summary["report"] = tex_path

    # ── Step 4: Website ─────────────────────────────────────────────
    if "website" in steps:
        from feather.website.generator import SiteGenerator

        gen = SiteGenerator(config)
        summary["site_dir"] = gen.build()

    return summary


def _run_diagnostics(
    config: FeatherConfig,
    *,
    diagnostics: list[str] | None = None,
    variables: list[str] | None = None,
    experiment: str = "baseline_hist",
    period: tuple[str, str] = ("1990", "2014"),
) -> int:
    """Run registered diagnostics and return the number of figures generated."""
    from feather.data.loader import DataLoader
    from feather.data.obs import ObsLoader
    from feather.diag.registry import get_diagnostic, registered_names

    # Determine which diagnostics to run
    if diagnostics:
        names = diagnostics
    else:
        names = registered_names()

    # Filter by variables if requested
    if variables:
        filtered = []
        for name in names:
            cls = get_diagnostic(name)
            if any(v in cls.variables for v in variables):
                filtered.append(name)
        names = filtered

    if not names:
        logger.warning("No diagnostics to run")
        return 0

    logger.info("Running %d diagnostic(s): %s", len(names), names)

    # Create loaders
    model_loader = _create_model_loader(config)
    obs_loader = ObsLoader(config)

    cmip6_loader = None
    if config.cmip6.get("enabled", False):
        from feather.data.cmip6 import CMIP6Loader
        cmip6_loader = CMIP6Loader(config)

    total_figures = 0
    for name in names:
        cls = get_diagnostic(name)
        # Build constructor kwargs — intersect user variables with diagnostic's
        kwargs: dict[str, Any] = {"cmip6_loader": cmip6_loader}
        if variables:
            supported = set(cls.variables)
            overlap = [v for v in variables if v in supported]
            unsupported = [v for v in variables if v not in supported]
            if unsupported:
                logger.warning(
                    "%s: requested variables %s not supported "
                    "(supported: %s)",
                    name, unsupported, cls.variables,
                )
            if not overlap:
                logger.warning(
                    "Skipping %s: none of requested variables %s "
                    "are in its supported list %s",
                    name, variables, cls.variables,
                )
                continue
            logger.info(
                "%s: running with variables %s (of %s requested)",
                name, overlap, variables,
            )
            kwargs["variables"] = overlap
        diag = cls(
            model_loader, obs_loader, config,
            **kwargs,
        )
        try:
            saved = diag.run()
            total_figures += len(saved)
        except Exception:
            logger.exception("Diagnostic %s failed", name)

    return total_figures


def _create_model_loader(config: FeatherConfig):
    """Create a model data loader from catalog paths in config.

    Opens all intake catalogs listed in ``config.model_catalogs``
    (typically ``2d`` and ``3d``) and returns a loader that searches
    across all of them.
    """
    from feather.data.loader import DataLoader

    catalogs = config.model_catalogs
    if not catalogs:
        logger.warning("No model_catalogs configured")
        return DataLoader()

    return _MultiCatalogLoader(catalogs)


class _MultiCatalogLoader:
    """DataLoader that searches across multiple intake catalogs."""

    def __init__(self, catalog_paths: dict[str, str]):
        import intake

        self._catalogs = {}
        self._cache: dict[str, "xr.Dataset"] = {}
        for label, path in catalog_paths.items():
            self._catalogs[label] = intake.open_catalog(path)

    def load(self, key: str) -> "xr.Dataset":
        import xarray as xr

        if key in self._cache:
            return self._cache[key]

        for label, cat in self._catalogs.items():
            if key in cat:
                ds = cat[key].to_dask()
                self._cache[key] = ds
                return ds

        available = self.list_entries()[:10]
        raise KeyError(
            f"Entry {key!r} not found in any catalog. "
            f"First entries: {available}"
        )

    def load_var(self, key: str, variable: str) -> "xr.DataArray":
        ds = self.load(key)
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        return ds[variable]

    def list_entries(self) -> list[str]:
        entries = []
        for cat in self._catalogs.values():
            entries.extend(list(cat))
        return entries

    @staticmethod
    def make_key(experiment: str, model: str, domain: str,
                 member: int = 1) -> str:
        return DataLoader.make_key(experiment, model, domain, member)
