"""Diagnostics modules.

Core infrastructure:
- :mod:`~feather.diag.figure_meta` — figure metadata sidecar system
- :mod:`~feather.diag.base` — :class:`DiagnosticBase` abstract base class
- :mod:`~feather.diag.registry` — ``@register`` decorator and discovery

Diagnostics:
- :mod:`~feather.diag.global_biases` — climatology bias maps
- :mod:`~feather.diag.timeseries` — global-mean time series
- :mod:`~feather.diag.seasonal_cycle` — monthly climatological cycle
"""

from feather.diag.base import DiagnosticBase
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata
from feather.diag.registry import get_diagnostic, list_diagnostics, register

# Import diagnostics so @register decorators execute.
import feather.diag.global_biases  # noqa: F401
import feather.diag.timeseries  # noqa: F401
import feather.diag.seasonal_cycle  # noqa: F401
