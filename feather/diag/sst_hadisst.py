"""Sea-surface temperature evaluation against HadISST.

A second SST observational reference alongside ESA-CCI.  HadISST is coarser
(1°) but spans the full 1980–2014 analysis window (ESA-CCI covers only
1990–2014), so it enables a full-period SST evaluation and Added Value.

The diagnostic reuses the :class:`~feather.diag.ocean_sst.OceanSST` figure
machinery (bias maps, global-mean time series, seasonal cycle, zonal-mean
profile) with HadISST as the reference, and — like ``ocean_sst`` — writes the
benchmark/model/ensemble bias NetCDF (via :mod:`feather.diag.ocean_bias`) so
the ocean Added Value diagnostic can use HadISST as a second ``tos`` reference
(``added_value._OBS_NETCDF_SOURCE["HADISST"] == "sst_hadisst"``).
"""

import logging

from feather.diag.ocean_sst import OceanSST, _to_celsius
from feather.diag.registry import register
from feather.util.temporal import monthly_climatology

logger = logging.getLogger(__name__)


@register
class SSTHadISST(OceanSST):
    """SST evaluation against HadISST (1°, full 1980–2014 record).

    Same figures as :class:`OceanSST` but referenced to HadISST, plus the
    ``tos``-vs-HadISST benchmark-bias NetCDF used by Added Value.
    """

    name = "sst_hadisst"
    title = "Sea Surface Temperature (HadISST)"
    domain = "o2d"
    variables = ["tos"]
    group = "ocean_surface"

    _obs_label = "HadISST"
    _obs_dataset_name = "HadISST"
    _ocean_bias_obs = "HADISST"

    # -- Obs loading overrides (HadISST instead of ESA-CCI) -----------------

    def _load_obs_monthly(self):
        """Full monthly HadISST SST series in °C over the analysis period."""
        da = self.obs_loader.load_hadisst(period=self.period)
        return _to_celsius(da)

    def _load_obs_timemean(self):
        """Annual-mean HadISST SST in °C (period mean of the monthly series)."""
        return self._load_obs_monthly().mean("time")

    def _load_obs_ymonmean(self):
        """Monthly-climatology HadISST SST in °C (``month`` dim 1–12)."""
        return monthly_climatology(self._load_obs_monthly())
