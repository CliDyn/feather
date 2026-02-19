#!/usr/bin/env python
"""Run all diagnostics (global biases, time series, seasonal cycle).

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --variables avg_2t avg_tprate avg_msl
    python scripts/run_all.py --models ifs-fesom ifs-nemo
    python scripts/run_all.py --output ./my_output
"""

import argparse

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.obs import ObsLoader
from feather.diag.global_biases import GlobalBiases
from feather.diag.timeseries import TimeseriesDiag
from feather.diag.seasonal_cycle import SeasonalCycleDiag


def main():
    parser = argparse.ArgumentParser(description="Run all feather diagnostics")
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
    parser.add_argument("--diagnostics", nargs="+",
                        default=["global_biases", "timeseries", "seasonal_cycle"],
                        choices=["global_biases", "timeseries", "seasonal_cycle"],
                        help="Which diagnostics to run (default: all)")
    args = parser.parse_args()

    cfg = FeatherConfig.from_yaml(args.config)
    if args.output:
        cfg.output_dir = args.output
    if args.models:
        cfg.models = args.models

    print(f"Config:      {args.config}")
    print(f"Variables:   {args.variables}")
    print(f"Models:      {cfg.models}")
    print(f"Experiment:  {args.experiment}")
    print(f"Period:      {args.period[0]}–{args.period[1]}")
    print(f"Output:      {cfg.output_dir}")
    print(f"Diagnostics: {args.diagnostics}")
    print()

    model_loader = DataLoader.from_catalog(cfg.model_catalogs["2d"])
    obs_loader = ObsLoader(cfg)

    diag_classes = {
        "global_biases": GlobalBiases,
        "timeseries": TimeseriesDiag,
        "seasonal_cycle": SeasonalCycleDiag,
    }

    all_saved = []
    for diag_name in args.diagnostics:
        cls = diag_classes[diag_name]
        print(f"{'='*60}")
        print(f"Running {diag_name}...")
        print(f"{'='*60}")

        diag = cls(
            model_loader, obs_loader, cfg,
            variables=args.variables,
            experiment=args.experiment,
            period=tuple(args.period),
        )

        saved = diag.run()
        all_saved.extend(saved)

        print(f"  → {len(saved)} figure(s) saved")
        print()

    print(f"{'='*60}")
    print(f"All done — {len(all_saved)} total figure(s) saved:")
    for png, json_path in all_saved:
        print(f"  {png}")


if __name__ == "__main__":
    main()
