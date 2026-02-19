#!/usr/bin/env python
"""Run the global biases diagnostic.

Usage:
    python scripts/run_global_biases.py
    python scripts/run_global_biases.py --variables avg_2t avg_tprate avg_msl
    python scripts/run_global_biases.py --models ifs-fesom
    python scripts/run_global_biases.py --output ./my_output
"""

import argparse

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.obs import ObsLoader
from feather.diag.global_biases import GlobalBiases


def main():
    parser = argparse.ArgumentParser(description="Global biases diagnostic")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--variables", nargs="+", default=["avg_2t"],
                        help="Model variables to evaluate")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to include (default: all from config)")
    parser.add_argument("--experiment", default="baseline_hist",
                        help="Experiment key")
    parser.add_argument("--period", nargs=2, default=["1990", "2014"],
                        metavar=("START", "END"),
                        help="Time period for climatologies")
    parser.add_argument("--output", default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    cfg = FeatherConfig.from_yaml(args.config)
    if args.output:
        cfg.output_dir = args.output
    if args.models:
        cfg.models = args.models

    print(f"Config:     {args.config}")
    print(f"Variables:  {args.variables}")
    print(f"Models:     {cfg.models}")
    print(f"Experiment: {args.experiment}")
    print(f"Period:     {args.period[0]}–{args.period[1]}")
    print(f"Output:     {cfg.output_dir}")
    print()

    model_loader = DataLoader.from_catalog(cfg.model_catalogs["2d"])
    obs_loader = ObsLoader(cfg)

    diag = GlobalBiases(
        model_loader, obs_loader, cfg,
        variables=args.variables,
        experiment=args.experiment,
        period=tuple(args.period),
    )

    print("Running global biases diagnostic...")
    saved = diag.run()

    print(f"\nDone — {len(saved)} figure(s) saved:")
    for png, json_path in saved:
        print(f"  {png}")


if __name__ == "__main__":
    main()
