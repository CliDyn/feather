#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 feather contributors
# SPDX-License-Identifier: BSD-3-Clause
"""Legacy wrapper — use ``feather --steps website`` instead."""

from feather.cli import main

if __name__ == "__main__":
    main(["--steps", "website"])
