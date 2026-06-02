"""Load CORDEX-CORE regional climate model data (e.g. AFR-22).

Directory tree (ESGF CORDEX DRS)::

    {root}/{institution}/{gcm}/{experiment}/{variant}/{rcm}/{rcm_version}/
        {freq}/{var}/{data_version}/{var}_{domain}_{gcm}_{experiment}_..._{period}.nc

Each CORDEX member is identified by (institution, gcm, rcm, variant) and is
configured as one entry in ``models``.  The reference period is assembled by
**stitching** experiments (e.g. ``historical`` 1981–2005 + ``rcp85`` 2006–2010)
listed in ``experiments``; the loader concatenates them along time and slices
to the requested period.

Monthly data is preferred.  When a variable has no ``mon`` output (true for
several AFR-22 RCMs — only GERICS-REMO2015 publishes monthly), the loader
falls back to ``day`` frequency and resamples to monthly means, so downstream
diagnostics always receive monthly data.

The native grid is rotated-pole: dims ``(time, rlat, rlon)`` with 2-D ``lat``
and ``lon`` coordinate arrays, which are preserved for regridding.
"""

import logging
from pathlib import Path

import xarray as xr

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)


class CORDEXLoader:
    """Load CORDEX-CORE regional model data with experiment stitching."""

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("root", ""))
        self._cache: dict[tuple, xr.DataArray] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,   # accepted for API parity, unused
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load *variable* for a CORDEX member, stitching its experiments."""
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
        """Return (lon, lat) coordinate arrays (2-D for rotated grids)."""
        import numpy as np
        da = self.load_var(model, variable)
        return np.asarray(da["lon"].values), np.asarray(da["lat"].values)

    # ── Private helpers ────────────────────────────────────────────────

    def _open_member(self, model: str, variable: str) -> xr.DataArray:
        """Open and time-concatenate a variable across the member's experiments."""
        mc = self._config.model_configs[model]
        experiments = list(mc.experiments) if mc.experiments else [mc.experiment]
        experiments = [e for e in experiments if e]
        if not experiments:
            raise ValueError(f"No experiment(s) configured for CORDEX model {model!r}")

        segments: list[xr.DataArray] = []
        for exp in experiments:
            seg = self._open_experiment(mc, exp, variable)
            if seg is not None:
                segments.append(seg)
        if not segments:
            raise FileNotFoundError(
                f"No CORDEX data found for {model!r}/{variable!r} "
                f"(experiments={experiments})"
            )

        if len(segments) == 1:
            da = segments[0]
        else:
            da = xr.concat(segments, dim="time").sortby("time")
            # Drop any duplicate timestamps at experiment boundaries
            _, index = _unique_index(da["time"].values)
            da = da.isel(time=index)

        da.name = variable
        return da

    def _open_experiment(
        self, mc, experiment: str, variable: str,
    ) -> xr.DataArray | None:
        """Open one experiment (mon preferred, day resampled to monthly)."""
        # Try monthly first, then daily fallback.
        for freq in ("mon", "day"):
            data_dir = self._var_dir(mc, experiment, freq, variable)
            if data_dir is None:
                continue
            nc_files = sorted(data_dir.glob("*.nc"))
            if not nc_files:
                continue
            logger.info(
                "CORDEX %s/%s/%s: opening %d %s file(s)",
                mc.gcm, mc.rcm, experiment, len(nc_files), freq,
            )
            ds = xr.open_mfdataset(
                nc_files, chunks="auto", combine="by_coords",
                decode_timedelta=False, use_cftime=True,
            )
            if variable not in ds:
                raise KeyError(
                    f"{variable!r} not in CORDEX dataset {data_dir} "
                    f"(have {list(ds.data_vars)})"
                )
            da = ds[variable]
            if freq == "day":
                # Resample daily → monthly mean so pr/tas are monthly rates.
                da = da.resample(time="1MS").mean()
            return da
        return None

    def _var_dir(self, mc, experiment: str, freq: str, variable: str) -> Path | None:
        """Resolve ``.../{rcm}/{rcm_version}/{freq}/{var}/{data_version}/``.

        Returns the latest data-version directory, or None if absent.
        """
        variant = mc.variant or "r1i1p1"
        member_dir = (
            self._root / mc.institution / mc.gcm / experiment / variant / mc.rcm
        )
        if not member_dir.exists():
            return None

        # RCM version dir (e.g. v1/v0/r2) — configured or auto-detected.
        if mc.rcm_version:
            rcm_ver_dir = member_dir / mc.rcm_version
        else:
            subdirs = sorted(d for d in member_dir.iterdir() if d.is_dir())
            if not subdirs:
                return None
            rcm_ver_dir = subdirs[-1]

        var_dir = rcm_ver_dir / freq / variable
        if not var_dir.exists():
            return None

        versions = sorted(var_dir.glob("v*"))
        if versions:
            return versions[-1]
        if list(var_dir.glob("*.nc")):
            return var_dir
        return None


def _unique_index(values):
    """Return (unique_values, indices-of-first-occurrence) preserving order."""
    import numpy as np
    seen = set()
    idx = []
    for i, v in enumerate(values):
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen.add(key)
            idx.append(i)
    return np.asarray(values)[idx], np.asarray(idx)
