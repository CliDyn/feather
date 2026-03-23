"""Diagnostics modules.

Core infrastructure:
- :mod:`~feather.diag.figure_meta` — figure metadata sidecar system
- :mod:`~feather.diag.base` — :class:`DiagnosticBase` abstract base class
- :mod:`~feather.diag.registry` — ``@register`` decorator and discovery

Diagnostics (12 registered):
- :mod:`~feather.diag.global_biases` — climatology bias maps (18 variables)
- :mod:`~feather.diag.timeseries` — global-mean time series
- :mod:`~feather.diag.seasonal_cycle` — monthly climatological cycle
- :mod:`~feather.diag.global_trends` — per-grid-point linear trends
- :mod:`~feather.diag.climate_variability` — STD of deseasonalised, detrended fields
- :mod:`~feather.diag.radiation_budget` — radiation budget analysis vs CERES EBAF
- :mod:`~feather.diag.sea_ice` — sea ice evaluation vs OSI-SAF + PIOMAS/GIOMAS
- :mod:`~feather.diag.ocean_sst` — ocean SST evaluation vs ESA-CCI
- :mod:`~feather.diag.ocean_en4` — ocean 3D T/S evaluation vs EN4 v4.2.2
- :mod:`~feather.diag.precipitation_mswep` — precipitation evaluation vs MSWEP v2.8
- :mod:`~feather.diag.temperature_berkeley` — temperature evaluation vs Berkeley Earth
- :mod:`~feather.diag.teleconnections` — climate variability modes (ENSO, NAO, SAM, AO, IOD, PDO, QBO)
"""

from feather.diag.base import DiagnosticBase
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata
from feather.diag.registry import get_diagnostic, list_diagnostics, register

# Import diagnostics so @register decorators execute.
import feather.diag.global_biases  # noqa: F401
import feather.diag.timeseries  # noqa: F401
import feather.diag.seasonal_cycle  # noqa: F401
import feather.diag.radiation_budget  # noqa: F401
import feather.diag.sea_ice  # noqa: F401
import feather.diag.ocean_sst  # noqa: F401
import feather.diag.ocean_en4  # noqa: F401
import feather.diag.global_trends  # noqa: F401
import feather.diag.climate_variability  # noqa: F401
import feather.diag.precipitation_mswep  # noqa: F401
import feather.diag.temperature_berkeley  # noqa: F401
import feather.diag.teleconnections  # noqa: F401
