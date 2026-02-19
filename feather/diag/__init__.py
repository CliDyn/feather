"""Diagnostics modules.

Core infrastructure:
- :mod:`~feather.diag.figure_meta` — figure metadata sidecar system
- :mod:`~feather.diag.base` — :class:`DiagnosticBase` abstract base class
- :mod:`~feather.diag.registry` — ``@register`` decorator and discovery
"""

from feather.diag.base import DiagnosticBase
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata
from feather.diag.registry import get_diagnostic, list_diagnostics, register
