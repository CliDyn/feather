#!/usr/bin/env python
"""Legacy wrapper — use ``feather --steps report`` instead."""

from feather.cli import main

if __name__ == "__main__":
    main(["--steps", "report"])
