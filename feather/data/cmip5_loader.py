"""Load CMIP5 GCM data from the ESGF CMIP5 DRS tree.

Directory layout (``output1``)::

    {root}/{institute}/{GCM}/{experiment}/mon/atmos/Amon/{variant}/
        {version}/{var}/{var}_Amon_{GCM}_{experiment}_{variant}_{period}.nc

Only the first realisation (``r1i1p1``) is used.  The reference period is
assembled by stitching experiments (``historical`` 1981–2005 + ``rcp85``
2006–2010); the latest available version directory is selected.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = "/pool/data/CMIP5/data/cmip5/output1"
# Bounds/aux variables (often object/cftime dtype) that break dask auto-chunking.
_BOUNDS_VARS = ["time_bnds", "lat_bnds", "lon_bnds", "bnds", "time_bounds"]


class CMIP5Loader:
    """Load CMIP5 monthly atmosphere data with experiment stitching."""

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("cmip5_root", _DEFAULT_ROOT))
        self._cache: dict[tuple, xr.DataArray] = {}

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,   # unused (always Amon); API parity
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        cache_key = (model, variable)
        if cache_key in self._cache:
            da = self._cache[cache_key]
        else:
            da = self._open_member(model, variable)
            self._cache[cache_key] = da

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(self, model: str, variable: str):
        da = self.load_var(model, variable)
        return np.asarray(da["lon"].values), np.asarray(da["lat"].values)

    # ── Private ────────────────────────────────────────────────────────

    def _open_member(self, model: str, variable: str) -> xr.DataArray:
        mc = self._config.model_configs[model]
        experiments = list(mc.experiments) if mc.experiments else [mc.experiment]
        experiments = [e for e in experiments if e]
        if not experiments:
            raise ValueError(f"No experiment(s) configured for CMIP5 model {model!r}")

        segments: list[xr.DataArray] = []
        for exp in experiments:
            d = self._var_dir(mc, exp, variable)
            if d is None:
                continue
            files = sorted(d.glob("*.nc"))
            if not files:
                continue
            logger.info("CMIP5 %s/%s: opening %d file(s)", mc.gcm, exp, len(files))
            ds = xr.open_mfdataset(
                files, chunks={}, combine="by_coords",
                decode_timedelta=False, drop_variables=_BOUNDS_VARS, use_cftime=True,
            )
            segments.append(ds[variable])

        if not segments:
            raise FileNotFoundError(
                f"No CMIP5 data for {model!r}/{variable!r} (experiments={experiments})"
            )
        if len(segments) == 1:
            da = segments[0]
        else:
            da = xr.concat(segments, dim="time").sortby("time")
        da.name = variable
        return da

    def _var_dir(self, mc, experiment: str, variable: str) -> Path | None:
        """Resolve ``.../Amon/{variant}/{version}/{var}/`` (latest version)."""
        variant = mc.variant or "r1i1p1"
        # Prefer the configured institute (CMIP5 GCM names are not unique to
        # one institute, e.g. HadGEM2-ES under both MOHC and INPE); else glob.
        if mc.institution:
            institutes = [mc.institution]
        else:
            institutes = [p.parent.name for p in self._root.glob(f"*/{mc.gcm}")]

        for inst in institutes:
            base = (self._root / inst / mc.gcm / experiment
                    / "mon" / "atmos" / "Amon" / variant)
            if not base.exists():
                continue
            versions = sorted(base.glob("v*"))
            if not versions:
                versions = [base]
            for v in reversed(versions):
                var_dir = v / variable
                if var_dir.exists() and list(var_dir.glob("*.nc")):
                    return var_dir
        return None
