#!/usr/bin/env python
"""Run LLM analysis on existing diagnostic figures.

Scans {output_dir}/figures/ for PNG+JSON pairs, sends each to Gemini
for scientific interpretation, and saves structured analysis to
{output_dir}/analysis/.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --config configs/default.yaml -v
    python scripts/run_analysis.py --diagnostics global_biases timeseries
    python scripts/run_analysis.py --api-key YOUR_KEY --model gemini-2.5-flash
"""

import argparse
import logging

from feather.config import FeatherConfig
from feather.llm.analyzer import FigureAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM analysis on feather diagnostic figures"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name (overrides config)",
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=None,
        help="Only analyse these diagnostics (default: all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-analyse figures even if analysis already exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

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

    cfg = FeatherConfig.from_yaml(args.config)
    if args.output:
        cfg.output_dir = args.output
    if args.model:
        cfg.llm.setdefault("figure_analysis", {})["model"] = args.model

    print(f"Config:      {args.config}")
    print(f"Output:      {cfg.output_dir}")
    print(f"Diagnostics: {args.diagnostics or 'all'}")
    print()

    analyzer = FigureAnalyzer(cfg, api_key=args.api_key)

    skip_existing = not args.no_skip_existing
    result = analyzer.run(
        skip_existing=skip_existing,
        diagnostics=args.diagnostics,
    )

    print()
    print(f"Done — {result['figure_analyses']} figure(s) analysed, "
          f"{result['syntheses']} synthesis(es) written")


if __name__ == "__main__":
    main()
