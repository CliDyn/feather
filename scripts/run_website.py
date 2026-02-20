#!/usr/bin/env python
"""Legacy wrapper — use ``feather --steps website`` instead."""

from feather.cli import main

if __name__ == "__main__":
    main(["--steps", "website"])
