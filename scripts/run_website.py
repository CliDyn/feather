#!/usr/bin/env python
"""Generate static website from diagnostic figures and analysis.

Scans {output_dir}/figures/ and {output_dir}/analysis/, renders
Jinja2 templates, and writes a self-contained HTML site to
{output_dir}/site/.

Usage:
    python scripts/run_website.py
    python scripts/run_website.py --config configs/default.yaml -v
    python scripts/run_website.py --output ./my_output
"""

import argparse
import logging

from feather.config import FeatherConfig
from feather.website.generator import SiteGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate static website from feather diagnostic output"
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

    print(f"Config: {args.config}")
    print(f"Output: {cfg.output_dir}")
    print()

    gen = SiteGenerator(cfg)
    site_dir = gen.build()

    print()
    print(f"Site generated: {site_dir}")
    print(f"Open {site_dir / 'index.html'} in a browser to view.")


if __name__ == "__main__":
    main()
