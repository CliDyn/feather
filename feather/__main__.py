# SPDX-FileCopyrightText: 2026 feather contributors
# SPDX-License-Identifier: BSD-3-Clause
"""Allow running feather as ``python -m feather``."""

from feather.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
