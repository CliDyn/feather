"""Load model data from per-year NetCDF files on a HEALPix grid.

Supports directory structures like::

    {root}/{dir_name}/{dir_name}_{year}.nc

where *dir_name* encodes the domain and variable (e.g. ``sfc_mean2t``,
``o2d_avg_tos``, ``o3d_avg_thetao``).  The data dimension is ``gsize``
(HEALPix 1-D), which is renamed to ``values`` on load.  Singleton
dimensions (``height``, ``depth``) are squeezed.

HEALPix pixel coordinates (``longitude``, ``latitude``) are derived
from ``nside`` using :mod:`healpy` and attached to the output DataArray.
"""

import logging
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var

logger = logging.getLogger(__name__)

# Singleton dimensions that should be squeezed automatically
_SQUEEZE_DIMS = {"height", "depth", "lev"}


class NetCDFLoader:
    """Load model data from per-year NetCDF files on a HEALPix grid.

    Parameters
    ----------
    config : FeatherConfig
        Pipeline config with ``data_source``, ``model_configs``, etc.
    """

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("root", ""))
        self._nside_override = config.data_source.get("nside")
        self._cache: dict[tuple, xr.DataArray] = {}

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load a variable for a model from per-year NetCDF files.

        Parameters
        ----------
        model : str
            Model name (must match ``config.model_configs`` key).
        variable : str
            CMOR variable name (e.g. ``"tas"``).
        period : tuple of str, optional
            (start, end) for time slicing.
        time_mean : bool
            Compute time mean before returning.

        Returns
        -------
        xr.DataArray
        """
        cache_key = (model, variable)
        if cache_key in self._cache:
            da = self._cache[cache_key]
        else:
            da = self._open_variable(model, variable)
            self._cache[cache_key] = da

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")

        return da

    # ── Private helpers ────────────────────────────────────────────────

    def _get_dir_name(self, model: str, variable: str) -> str:
        """Resolve the on-disk directory name for *variable*.

        Lookup order:
        1. Per-model ``variable_aliases`` (e.g. ``"tas"`` → ``"sfc_mean2t"``)
        2. ``destine_variable`` from ``VARIABLE_REGISTRY`` (secondary fallback)
        3. Canonical CMOR name (last resort)
        """
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.variable_aliases:
            alias = mcfg.variable_aliases.get(variable)
            if alias:
                return alias

        # Fallback: destine_variable from registry
        try:
            vinfo = get_var(variable)
            if vinfo.destine_variable:
                return vinfo.destine_variable
        except KeyError:
            pass

        return variable

    def _get_scale_factor(self, model: str, variable: str) -> float:
        """Return a post-load scale factor (default 1.0)."""
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.scale_factors:
            return mcfg.scale_factors.get(variable, 1.0)
        return 1.0

    def _open_variable(self, model: str, variable: str) -> xr.DataArray:
        """Open and return a DataArray for a single variable."""
        dir_name = self._get_dir_name(model, variable)
        var_dir = self._root / dir_name

        if not var_dir.exists():
            raise FileNotFoundError(
                f"Variable directory not found: {var_dir}"
            )

        nc_files = sorted(var_dir.glob(f"{dir_name}_*.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF files matching {dir_name}_*.nc in {var_dir}"
            )

        logger.info(
            "Opening %s/%s (%d files) from %s",
            model, variable, len(nc_files), var_dir,
        )

        ds = xr.open_mfdataset(nc_files, chunks="auto", combine="by_coords")

        # Correct IFS time offset: monthly means are timestamped at the
        # start of the *next* month (Jan 1990 mean → 1990-02-01).
        # Shift back by one month so seasonal grouping works correctly.
        if "time" in ds.dims:
            ds = self._correct_time_offset(ds)

        # Find the data variable in the dataset
        nc_var = self._find_data_var(ds, dir_name, variable)
        da = ds[nc_var]

        # Rename gsize → values
        if "gsize" in da.dims:
            da = da.rename({"gsize": "values"})

        # Squeeze singleton dimensions (height, depth, lev when size=1)
        for dim in list(da.dims):
            if dim in _SQUEEZE_DIMS and da.sizes[dim] == 1:
                da = da.squeeze(dim, drop=True)

        # Attach HEALPix coordinates if not present (NEST ordering)
        if "values" in da.dims and "longitude" not in da.coords:
            n_pix = da.sizes["values"]
            nside = self._nside_override or int(np.sqrt(n_pix / 12))
            lon, lat = hp.pix2ang(nside, np.arange(n_pix), nest=True, lonlat=True)
            da = da.assign_coords(
                longitude=("values", lon),
                latitude=("values", lat),
            )

        # Apply per-model scale factor
        scale = self._get_scale_factor(model, variable)
        if scale != 1.0:
            da = da * scale

        return da

    @staticmethod
    def _correct_time_offset(ds: xr.Dataset) -> xr.Dataset:
        """Shift IFS time stamps back by one month.

        IFS monthly means are stamped at the start of the *next* month
        (e.g. the January 1990 mean has timestamp 1990-02-01).  This
        causes ``time.month`` grouping to be off by one, breaking
        seasonal cycle diagnostics and March/September sea-ice selection.

        We shift each timestamp back by one month so that the January
        mean is labelled 1990-01-01, etc.
        """
        raw = pd.DatetimeIndex(ds["time"].values)
        corrected = raw - pd.DateOffset(months=1)
        ds["time"] = ("time", corrected.values)
        return ds

    @staticmethod
    def _find_data_var(
        ds: xr.Dataset, dir_name: str, variable: str,
    ) -> str:
        """Find the data variable name in the dataset.

        Tries:
        1. Strip domain prefix from dir_name (e.g. ``sfc_mean2t`` → ``mean2t``)
        2. Canonical CMOR name
        3. First non-coordinate variable
        """
        # Strip domain prefix: "sfc_mean2t" → "mean2t", "o3d_avg_thetao" → "avg_thetao"
        parts = dir_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ("sfc", "o2d", "o3d", "pl"):
            stripped = parts[1]
            if stripped in ds:
                return stripped

        if variable in ds:
            return variable

        # Last resort: first non-coordinate data variable
        data_vars = [v for v in ds.data_vars if v not in ds.coords]
        if data_vars:
            return data_vars[0]

        raise KeyError(
            f"No data variable found in dataset for {variable!r} "
            f"(dir_name={dir_name!r}). Available: {list(ds.data_vars)}"
        )
