"""Pipeline orchestration — run diagnostics, analysis, report, and website.

Provides :func:`run_pipeline` which ties together all feather stages
into a single callable.
"""

import logging
from pathlib import Path
from typing import Any

from feather import provenance
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
    cmip6_individual: bool = False,
    added_value_regions: bool | list[str] | None = False,
    benchmarks: list[str] | None = None,
    save_netcdf: bool = False,
    individual_netcdf_only: bool = False,
    ensemble_only: bool = False,
    replot_from_netcdf: bool = False,
    no_llm: bool = False,
    provenance_hash: str = "stat",
    provenance_hash_obs: str | None = None,
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
    cmip6_individual : bool
        Plot individual CMIP6 model lines/biases (plus MMM) instead of
        MMM only.  Affects diagnostics that accept the parameter
        (``GlobalBiases``, ``SeasonalCycleDiag``, ``TimeseriesDiag``).
    no_llm : bool
        When True, skip the ``"analyze"`` step and generate a
        figures-only website without LLM content.
    provenance_hash, provenance_hash_obs : str
        Input fingerprint policy (``"none"``, ``"stat"``, ``"content"``)
        for the provenance record; the obs policy defaults to the general
        one.  See :mod:`feather.provenance`.

    Returns
    -------
    dict
        Summary with keys: ``figures``, ``analyses``, ``syntheses``,
        ``report``, ``site_dir``, ``run_id``.
    """
    options = {k: v for k, v in locals().items()
               if k not in ("config", "provenance_hash", "provenance_hash_obs")}
    with provenance.run_scope(
        config, options=options, extra=_regrid_environment(config),
        hash_policy=provenance_hash, hash_policy_obs=provenance_hash_obs,
    ) as recorder:
        summary = _pipeline(config, **options)
        summary["run_id"] = recorder.run_id if recorder is not None else None
        return summary


def _regrid_environment(config: FeatherConfig) -> dict[str, Any]:
    """Regridding capabilities and settings for the run record.

    The nereus conservative capability cannot be read from
    ``nereus.__version__`` (upstream did not bump it), so a stale
    environment changes the numbers without failing.  Recording the probe
    result per run makes that attributable.
    """
    try:
        from feather.diag.base import DiagnosticBase

        return {"regrid": {
            "nereus_conservative_available": DiagnosticBase._conservative_available(),
            "conservative_fluxes": config.nereus.get("conservative_fluxes", True),
            "conservative_max_points": config.get_conservative_max_points(),
            "method": config.nereus.get("method", "nearest"),
        }}
    except Exception as exc:  # provenance must never fail the run
        return {"regrid": {"error": f"{type(exc).__name__}: {exc}"}}


def _pipeline(
    config: FeatherConfig,
    *,
    steps: str | list[str],
    diagnostics: list[str] | None,
    variables: list[str] | None,
    experiment: str,
    period: tuple[str, str],
    api_key: str | None,
    openai_api_key: str | None,
    skip_existing: bool,
    compile_pdf: bool,
    cmip6_individual: bool,
    added_value_regions: bool | list[str] | None,
    benchmarks: list[str] | None,
    save_netcdf: bool,
    individual_netcdf_only: bool,
    ensemble_only: bool,
    replot_from_netcdf: bool,
    no_llm: bool,
) -> dict[str, Any]:
    """Body of :func:`run_pipeline`, run inside its provenance scope."""
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
            cmip6_individual=cmip6_individual,
            added_value_regions=added_value_regions,
            benchmarks=benchmarks,
            save_netcdf=save_netcdf,
            individual_netcdf_only=individual_netcdf_only,
            ensemble_only=ensemble_only,
            replot_from_netcdf=replot_from_netcdf,
            skip_existing=skip_existing,
        )

    # ── Step 2: LLM analysis ────────────────────────────────────────
    if "analyze" in steps and not no_llm:
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

        gen = SiteGenerator(config, no_llm=no_llm)
        summary["site_dir"] = gen.build()

    return summary


def _run_diagnostics(
    config: FeatherConfig,
    *,
    diagnostics: list[str] | None = None,
    variables: list[str] | None = None,
    experiment: str = "baseline_hist",
    period: tuple[str, str] = ("1990", "2014"),
    cmip6_individual: bool = False,
    added_value_regions: bool | list[str] | None = False,
    benchmarks: list[str] | None = None,
    save_netcdf: bool = False,
    individual_netcdf_only: bool = False,
    ensemble_only: bool = False,
    replot_from_netcdf: bool = False,
    skip_existing: bool = True,
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

    # Build benchmark loaders (CMIP6, HighResMIP, …).  The first benchmark
    # doubles as the legacy ``cmip6_loader`` so diagnostics that have not yet
    # been generalised still render the primary benchmark's MMM.
    benchmark_loaders = []
    benchmark_cfgs = config.get_benchmarks()
    if benchmark_cfgs and benchmarks:
        # Restrict to the user-selected benchmark names (case-insensitive,
        # matched against the benchmark ``name`` or ``label``).
        wanted = {b.lower() for b in benchmarks}
        selected = [
            bc for bc in benchmark_cfgs
            if str(bc.get("name", "")).lower() in wanted
            or str(bc.get("label", "")).lower() in wanted
        ]
        missing = wanted - {
            str(bc.get("name", "")).lower() for bc in benchmark_cfgs
        } - {str(bc.get("label", "")).lower() for bc in benchmark_cfgs}
        if missing:
            logger.warning(
                "Requested benchmark(s) not found in config: %s (available: %s)",
                sorted(missing),
                [bc.get("name") for bc in benchmark_cfgs],
            )
        benchmark_cfgs = selected
    if benchmark_cfgs:
        from feather.data.cmip6 import CMIP6Loader
        for bcfg in benchmark_cfgs:
            benchmark_loaders.append(CMIP6Loader(config, cmip6_cfg=bcfg))
        logger.info("Benchmarks enabled: %s",
                    [b.label for b in benchmark_loaders])
    cmip6_loader = benchmark_loaders[0] if benchmark_loaders else None

    total_figures = 0
    for name in names:
        cls = get_diagnostic(name)
        import inspect
        sig = inspect.signature(cls.__init__)
        # Build constructor kwargs — intersect user variables with diagnostic's
        kwargs: dict[str, Any] = {
            "cmip6_loader": cmip6_loader,
            "experiment": experiment,
            "period": period,
        }
        # Pass the full benchmark list only to diagnostics that accept it
        # (those generalised for multiple benchmarks).
        if benchmark_loaders and "benchmarks" in sig.parameters:
            kwargs["benchmarks"] = benchmark_loaders
        # NetCDF export, only for diagnostics that support it.
        if save_netcdf and "save_netcdf" in sig.parameters:
            kwargs["save_netcdf"] = True
        # Individual-member NetCDF-only mode (no figures), for the bias-map
        # diagnostics that support it.
        if individual_netcdf_only and (
            "individual_netcdf_only" in sig.parameters
        ):
            kwargs["individual_netcdf_only"] = True
        # The time-series diagnostic may extend beyond the analysis period
        # (e.g. to show each model's full projection continuation).
        if name == "timeseries":
            kwargs["period"] = config.get_timeseries_period()
        if cmip6_individual and "cmip6_individual" in sig.parameters:
            kwargs["cmip6_individual"] = True
        # Per-region Added Value output is opt-in.  ``--added-value-regions``
        # passed bare yields an empty list from argparse, which means "the
        # default set" rather than "no sets" — so test against None, not
        # truthiness, or the bare flag would silently do nothing.
        if added_value_regions is not None and added_value_regions is not False \
                and "regions" in sig.parameters:
            kwargs["regions"] = added_value_regions or True
        # Ensemble-only mode (plot just the ensemble bias figures).
        if ensemble_only and "ensemble_only" in sig.parameters:
            kwargs["ensemble_only"] = True
        # In NetCDF-only mode, skip diagnostics that cannot honour it — they
        # would otherwise render figures the user explicitly opted out of.
        if individual_netcdf_only and (
            "individual_netcdf_only" not in sig.parameters
        ):
            logger.info(
                "Skipping %s: does not support --individual-netcdf-only",
                name,
            )
            continue
        # In ensemble-only mode, skip diagnostics that cannot honour it.
        if ensemble_only and "ensemble_only" not in sig.parameters:
            logger.info(
                "Skipping %s: does not support --ensemble-only", name,
            )
            continue
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
            # Only pass the selection to diagnostics that accept it; others
            # derive their variables internally (still filtered/skipped above).
            if "variables" in sig.parameters:
                kwargs["variables"] = overlap
        # Replot-from-NetCDF mode: skip diagnostics that cannot honour it.
        if replot_from_netcdf and not hasattr(cls, "replot_from_netcdf"):
            logger.info(
                "Skipping %s: does not support --replot-from-netcdf", name,
            )
            continue
        diag = cls(
            model_loader, obs_loader, config,
            **kwargs,
        )
        with provenance.diagnostic_scope(name):
            try:
                if replot_from_netcdf:
                    saved = diag.replot_from_netcdf(skip_existing=skip_existing)
                else:
                    saved = diag.run(skip_existing=skip_existing)
                total_figures += len(saved)
                provenance.emit("diagnostic", name=name, status="ok",
                                n_figures=len(saved))
            except Exception as exc:
                logger.exception("Diagnostic %s failed", name)
                provenance.emit("diagnostic", name=name, status="failed",
                                error=f"{type(exc).__name__}: {exc}")

    return total_figures


def _create_model_loader(config: FeatherConfig):
    """Create a model data loader from config.

    Dispatches based on ``config.data_source.type``.  When models use
    different source types, returns a :class:`CompositeModelLoader`.
    """
    if config.is_multi_source():
        from feather.data.composite_loader import CompositeModelLoader
        return CompositeModelLoader(config)

    if config.get_data_source_type() == "cmor":
        from feather.data.cmor_loader import CMORLoader
        return CMORLoader(config)

    if config.get_data_source_type() == "netcdf_healpix":
        from feather.data.netcdf_loader import NetCDFLoader
        return NetCDFLoader(config)

    if config.get_data_source_type() == "grib_healpix":
        from feather.data.grib_loader import GRIBLoader
        return GRIBLoader(config)

    if config.get_data_source_type() == "kerchunk_parquet":
        from feather.data.kerchunk_loader import KerchunkParquetLoader
        return KerchunkParquetLoader(config)

    if config.get_data_source_type() == "icon_kerchunk":
        from feather.data.icon_kerchunk_loader import ICONKerchunkLoader
        return ICONKerchunkLoader(config)

    if config.get_data_source_type() == "cordex":
        from feather.data.cordex_loader import CORDEXLoader
        return CORDEXLoader(config)

    if config.get_data_source_type() == "cmip5":
        from feather.data.cmip5_loader import CMIP5Loader
        return CMIP5Loader(config)

    if config.get_data_source_type() == "cmip6_nc":
        from feather.data.cmip6_nc_loader import CMIP6NCLoader
        return CMIP6NCLoader(config)

    from feather.data.loader import DataLoader, MultiCatalogLoader

    catalogs = config.model_catalogs
    if not catalogs:
        logger.warning("No model_catalogs configured")
        return DataLoader()

    return MultiCatalogLoader(catalogs)
