"""Load model data from a standard CMOR directory tree.

Supports directory structures like EERIE HighResMIP::

    {root}/{institution}/{model}/{experiment}/{variant}/{table}/{variable}/gr/v*/

Each variable is stored as one-file-per-year NetCDF files following the
CMIP6/CMOR naming convention.
"""

import logging
from pathlib import Path

import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var

logger = logging.getLogger(__name__)

# Map feather domain names → typical CMOR table prefixes
_DOMAIN_TABLE_MAP = {
    "sfc": "Amon",
    "o2d": "Omon",
    "o3d": "Omon",
    "pl": "Amon",
}


class CMORLoader:
    """Load model data from a CMOR directory tree.

    Parameters
    ----------
    config : FeatherConfig
        Pipeline config with ``data_source``, ``model_configs``, etc.
    """

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("root", ""))
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
        """Load a variable for a model from CMOR files.

        Parameters
        ----------
        model : str
            Model name (must match ``config.model_configs`` key).
        variable : str
            CMOR variable name (e.g. ``"tas"``).
        table : str, optional
            CMOR table (e.g. ``"Amon"``).  Inferred from variable registry
            if not provided.
        period : tuple of str, optional
            (start, end) for time slicing.
        time_mean : bool
            Compute time mean before returning.

        Returns
        -------
        xr.DataArray
        """
        if table is None:
            try:
                vinfo = get_var(variable)
                table = vinfo.cmip6_table or _DOMAIN_TABLE_MAP.get(vinfo.domain, "Amon")
            except KeyError:
                table = "Amon"

        cache_key = (model, variable, table)
        if cache_key in self._cache:
            da = self._cache[cache_key]
        else:
            da = self._open_variable(model, variable, table)
            self._cache[cache_key] = da

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")

        return da

    def load_dataset(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,
    ) -> xr.Dataset:
        """Load the full dataset for a variable (includes coords/bounds).

        Useful when you need ``lat_bnds``/``lon_bnds`` or other metadata.
        """
        if table is None:
            try:
                vinfo = get_var(variable)
                table = vinfo.cmip6_table or _DOMAIN_TABLE_MAP.get(vinfo.domain, "Amon")
            except KeyError:
                table = "Amon"

        data_dir = self._find_version_dir(model, variable, table)
        nc_files = sorted(data_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF files in {data_dir}"
            )
        return xr.open_mfdataset(nc_files, chunks="auto", combine="by_coords")

    def available_variables(self, model: str, table: str = "Amon") -> list[str]:
        """List available variables for a model/table."""
        mcfg = self._config.model_configs.get(model)
        if mcfg is None:
            return []
        table_dir = self._table_dir(model, table)
        if not table_dir.exists():
            return []
        return sorted(d.name for d in table_dir.iterdir() if d.is_dir())

    # ── Private helpers ────────────────────────────────────────────────

    def _open_variable(
        self, model: str, variable: str, table: str,
    ) -> xr.DataArray:
        """Open and return a DataArray for a single variable."""
        data_dir = self._find_version_dir(model, variable, table)
        nc_files = sorted(data_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF files in {data_dir}"
            )

        logger.info(
            "Opening %s/%s/%s (%d files)",
            model, table, variable, len(nc_files),
        )

        ds = xr.open_mfdataset(nc_files, chunks="auto", combine="by_coords")
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        return ds[variable]

    def _table_dir(self, model: str, table: str) -> Path:
        """Build path to the table directory."""
        mcfg = self._config.model_configs[model]
        return (
            self._root
            / mcfg.institution
            / model
            / (mcfg.experiment or self._config.get_experiment())
            / (mcfg.variant or "r1i1p1f1")
            / table
        )

    def _find_version_dir(
        self, model: str, variable: str, table: str,
    ) -> Path:
        """Find the latest version directory for a variable.

        Path: ``{root}/{institution}/{model}/{experiment}/{variant}/{table}/{variable}/gr/v*/``
        """
        var_dir = self._table_dir(model, table) / variable / "gr"
        if not var_dir.exists():
            raise FileNotFoundError(
                f"Variable directory not found: {var_dir}"
            )

        # Take the latest version directory
        versions = sorted(var_dir.glob("v*"))
        if not versions:
            raise FileNotFoundError(
                f"No version directories in {var_dir}"
            )

        return versions[-1]
