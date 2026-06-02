"""Load CMIP6 GCM data from the ESGF CMIP6 DRS NetCDF tree.

Directory layout::

    {root}/{activity}/{institute}/{model}/{experiment}/{variant}/{table}/
        {var}/{grid}/{version}/{var}_{table}_{model}_{experiment}_..._{period}.nc

``activity`` is inferred from the experiment (``historical`` → ``CMIP``,
``ssp*`` → ``ScenarioMIP``).  The institute and grid label are auto-detected
unless given in the config.  Experiments may be stitched (e.g. historical +
ssp585); the latest version directory is selected.

Distinct from the existing zarr-based :class:`CMIP6Loader`, which only serves
the pre-staged historical MMM.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = "/work/ik1017/CMIP6/data/CMIP6"
# Bounds/aux variables (often object/cftime dtype) that break dask auto-chunking.
_BOUNDS_VARS = ["time_bnds", "lat_bnds", "lon_bnds", "bnds", "time_bounds"]


def _activity(experiment: str) -> str:
    return "ScenarioMIP" if experiment.startswith("ssp") else "CMIP"


class CMIP6NCLoader:
    """Load CMIP6 monthly atmosphere data from the CMOR NetCDF tree."""

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("cmip6_root", _DEFAULT_ROOT))
        self._cache: dict[tuple, xr.DataArray] = {}

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        cache_key = (model, variable)
        if cache_key in self._cache:
            da = self._cache[cache_key]
        else:
            da = self._open_member(model, variable, table or "Amon")
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

    def _open_member(self, model: str, variable: str, table: str) -> xr.DataArray:
        mc = self._config.model_configs[model]
        experiments = list(mc.experiments) if mc.experiments else [mc.experiment]
        experiments = [e for e in experiments if e]
        if not experiments:
            raise ValueError(f"No experiment(s) configured for CMIP6 model {model!r}")

        segments: list[xr.DataArray] = []
        for exp in experiments:
            d = self._var_dir(mc, exp, variable, table)
            if d is None:
                continue
            files = sorted(d.glob("*.nc"))
            if not files:
                continue
            logger.info("CMIP6 %s/%s: opening %d file(s)", model, exp, len(files))
            ds = xr.open_mfdataset(
                files, chunks={}, combine="by_coords",
                decode_timedelta=False, drop_variables=_BOUNDS_VARS, use_cftime=True,
            )
            segments.append(ds[variable])

        if not segments:
            raise FileNotFoundError(
                f"No CMIP6 data for {model!r}/{variable!r} (experiments={experiments})"
            )
        if len(segments) == 1:
            da = segments[0]
        else:
            da = xr.concat(segments, dim="time").sortby("time")
        da.name = variable
        return da

    def _var_dir(self, mc, experiment: str, variable: str, table: str) -> Path | None:
        """Resolve ``.../{var}/{grid}/{version}/`` (latest version)."""
        variant = mc.variant or "r1i1p1f1"
        activity = self._root / _activity(experiment)

        # Institute: configured, else glob (CMIP6 model names are unique).
        if mc.institution:
            institutes = [mc.institution]
        else:
            institutes = [p.parent.name for p in activity.glob(f"*/{mc.gcm or mc.name}")]

        model_dir_name = mc.gcm or mc.name
        for inst in institutes:
            var_base = (activity / inst / model_dir_name / experiment
                        / variant / table / variable)
            if not var_base.exists():
                continue
            # Grid label (gn/gr/gr1…): configured or first available.
            grids = [mc.grid_dir] if mc.grid_dir else [
                d.name for d in sorted(var_base.iterdir()) if d.is_dir()
            ]
            for grid in grids:
                grid_dir = var_base / grid
                if not grid_dir.exists():
                    continue
                versions = sorted(grid_dir.glob("v*"))
                if not versions:
                    versions = [grid_dir]
                for v in reversed(versions):
                    if list(v.glob("*.nc")):
                        return v
        return None
