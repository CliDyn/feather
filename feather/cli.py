"""Command-line interface for the feather pipeline.

Entry points::

    feather --steps report -v                     # installed console script
    python -m feather --steps analyze report -v   # module invocation
"""

import argparse
import logging

from feather.config import FeatherConfig
from feather.run import run_pipeline


def main(argv: list[str] | None = None):
    """Main CLI entry point for the feather pipeline."""
    parser = argparse.ArgumentParser(
        prog="feather",
        description="Feather — climate model evaluation pipeline "
                    "(diagnostics → analyze → report → website)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "diagnostics", "analyze", "report", "website"],
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=None,
        help="Only run these diagnostics (default: all registered)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Only run diagnostics using these variables",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Model experiment key (default: from config, or baseline_hist)",
    )
    parser.add_argument(
        "--period",
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Time period (default: from config, or 1990 2014)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Vertex AI API key for the analyze step",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for the report step",
    )
    parser.add_argument(
        "--cmip6-individual",
        action="store_true",
        help="Plot individual CMIP6 model biases (plus MMM) instead of MMM only",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run steps even if output already exists",
    )
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="Compile LaTeX report to PDF (requires pdflatex)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args(argv)

    if args.verbose >= 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    elif args.verbose >= 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # Allow comma-separated values (e.g. --diagnostics ocean_sst,ocean_en4)
    def _split_csv(items):
        if items is None:
            return None
        out = []
        for item in items:
            out.extend(item.split(","))
        return out

    args.diagnostics = _split_csv(args.diagnostics)
    args.variables = _split_csv(args.variables)
    args.steps = [s for item in args.steps for s in item.split(",")]

    cfg = FeatherConfig.from_yaml(args.config)
    if args.output:
        cfg.output_dir = args.output

    # Derive experiment and period from config if not set on CLI
    experiment = args.experiment or cfg.get_experiment()
    period = tuple(args.period) if args.period else cfg.get_period()

    project_name = cfg.project.get("name", "DestinE")
    print(f"Project:     {project_name}")
    print(f"Config:      {args.config}")
    print(f"Output:      {cfg.output_dir}")
    print(f"Steps:       {args.steps}")
    print(f"Diagnostics: {args.diagnostics or 'all'}")
    print()

    result = run_pipeline(
        cfg,
        steps=args.steps,
        diagnostics=args.diagnostics,
        variables=args.variables,
        experiment=experiment,
        period=period,
        api_key=args.api_key,
        openai_api_key=args.openai_api_key,
        skip_existing=not args.no_skip_existing,
        compile_pdf=args.compile_pdf,
        cmip6_individual=args.cmip6_individual,
    )

    print()
    print("Pipeline complete:")
    print(f"  Figures:   {result['figures']}")
    print(f"  Analyses:  {result['analyses']}")
    print(f"  Syntheses: {result['syntheses']}")
    if result["report"]:
        print(f"  Report:    {result['report']}")
    if result["site_dir"]:
        print(f"  Site:      {result['site_dir']}")
