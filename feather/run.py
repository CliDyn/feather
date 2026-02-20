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
    model_loader = DataLoader(config)
    obs_loader = ObsLoader(config)

    cmip6_loader = None
    if config.cmip6.get("enabled", False):
        from feather.data.cmip6 import CMIP6Loader
        cmip6_loader = CMIP6Loader(config)

    total_figures = 0
    for name in names:
        cls = get_diagnostic(name)
        diag = cls(
            model_loader, obs_loader, config,
            cmip6_loader=cmip6_loader,
        )
        try:
            saved = diag.run()
            total_figures += len(saved)
        except Exception:
            logger.exception("Diagnostic %s failed", name)

    return total_figures
