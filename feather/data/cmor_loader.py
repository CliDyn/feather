"""Load model data from a standard CMOR directory tree.

Supports directory structures like EERIE Ensemble::

    {root}/{institution}/{model}/{experiment}/{variant}/{table}/{variable}/gr/v*/

Also supports per-model overrides for non-standard layouts (e.g. HadGEM3)::

    {data_root}/{experiment}/{variant}/{table}/{variable}/{grid_label}/v*/

Each variable is stored as one-file-per-year (or per-month) NetCDF files
following the CMIP6/CMOR naming convention.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
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

    def _get_alias(self, model: str, variable: str) -> str:
        """Return the on-disk variable name for *variable*.

        Checks per-model ``variable_aliases`` first, otherwise returns
        the canonical name unchanged.
        """
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.variable_aliases:
            return mcfg.variable_aliases.get(variable, variable)
        return variable

    def _get_scale_factor(self, model: str, variable: str) -> float:
        """Return a post-load scale factor (default 1.0)."""
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.scale_factors:
            return mcfg.scale_factors.get(variable, 1.0)
        return 1.0

    @staticmethod
    def _decode_time_manually(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
        """Decode a raw numeric time coordinate, recovering fill values.

        Some HadGEM3 files have fill values (9.97e+36) in the time
        coordinate that prevent normal CF time decoding.  This method
        reads the ``units`` and ``calendar`` attributes and converts
        valid values using ``cftime.num2date``.  For fill values, it
        falls back to ``time_bounds`` (midpoint) if available, otherwise
        sets NaT.
        """
        import cftime as cf

        raw = ds[time_dim]
        units = raw.attrs.get("units", "seconds since 1850-01-01 00:00:00")
        calendar = raw.attrs.get("calendar", "gregorian")

        vals = raw.values.astype(np.float64)
        # Mask fill values (typically ~1e+20 or ~1e+36)
        valid = np.abs(vals) < 1e15

        # Try to recover fill-value times from time_bounds midpoints
        bounds_name = raw.attrs.get("bounds", "time_bounds")
        if not valid.all() and bounds_name in ds:
            bounds = ds[bounds_name].values.astype(np.float64)
            midpoints = bounds.mean(axis=-1)
            vals[~valid] = midpoints[~valid]
            valid[:] = True
            logger.info(
                "Recovered %d fill-value time(s) from %s midpoints",
                int((~valid).sum()) if not valid.all() else int(vals.shape[0]),
                bounds_name,
            )

        dates = np.full(vals.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        if valid.any():
            decoded = cf.num2date(
                vals[valid], units, calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            dates[valid] = np.array(decoded, dtype="datetime64[ns]")

        ds[time_dim] = (time_dim, dates)
        return ds

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

        alias = self._get_alias(model, variable)

        logger.info(
            "Opening %s/%s/%s (%d files)%s",
            model, table, variable, len(nc_files),
            f" [alias={alias}]" if alias != variable else "",
        )

        # Suppress SerializationWarning about multiple fill values
        # (e.g. HadGEM3 siconc has both float32 and float64 _FillValue)
        # and FutureWarning about data_vars default change.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="variable.*multiple fill values")
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message=".*data_vars.*")

            # Detect time dimension name from first file to set concat_dim.
            # Use decode_times=False to avoid crashes on files with fill
            # values in the time coordinate (e.g. HadGEM3 SImon).
            ds0 = xr.open_dataset(
                nc_files[0], chunks="auto",
                decode_timedelta=False, decode_times=False,
            )
            time_dim = "time_counter" if "time_counter" in ds0.dims else "time"
            ds0.close()

            try:
                ds = xr.open_mfdataset(
                    nc_files, chunks="auto",
                    combine="nested", concat_dim=time_dim,
                    decode_timedelta=False,
                )
            except (ValueError, OverflowError):
                # Some files have fill values (9.97e+36) in the time
                # coordinate that cause time decoding to fail.  Open
                # without decoding and reconstruct time manually.
                logger.warning(
                    "Time decoding failed for %s/%s/%s — retrying with "
                    "decode_times=False",
                    model, table, variable,
                )
                ds = xr.open_mfdataset(
                    nc_files, chunks="auto",
                    combine="nested", concat_dim=time_dim,
                    decode_timedelta=False, decode_times=False,
                )
                ds = self._decode_time_manually(ds, time_dim)

        # Normalise time dimension: some datasets use "time_counter"
        if "time_counter" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"time_counter": "time"})

        # Find the variable in the dataset.  Try canonical name first,
        # then the alias with underscores (dir names use hyphens, NetCDF
        # variable names use underscores).
        nc_var = None
        for candidate in [variable, alias, alias.replace("-", "_")]:
            if candidate in ds:
                nc_var = candidate
                break

        if nc_var is None:
            raise KeyError(
                f"Variable {variable!r} (alias={alias!r}) not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )

        da = ds[nc_var]

        # Apply per-model scale factor (e.g. clt fraction→percentage)
        scale = self._get_scale_factor(model, variable)
        if scale != 1.0:
            da = da * scale

        return da

    def _table_dir(self, model: str, table: str) -> Path:
        """Build path to the table directory.

        Uses per-model ``data_root`` if set, otherwise constructs from
        the global root + institution + model name.
        """
        mcfg = self._config.model_configs[model]
        if mcfg.data_root:
            # Per-model override: {data_root}/{experiment}/{variant}/{table}
            return (
                Path(mcfg.data_root)
                / (mcfg.experiment or self._config.get_experiment())
                / (mcfg.variant or "r1i1p1f1")
                / table
            )
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
        """Find the directory containing NetCDF files for a variable.

        Tries paths in order:

        1. ``{table_dir}/{alias}/{grid_label}/v*/``  (standard CMOR)
        2. ``{table_dir}/{alias}/``  (flat layout, no grid_label/version)

        The grid label defaults to ``"gr"`` but can be overridden per model
        via ``ModelConfig.grid_label``.
        """
        alias = self._get_alias(model, variable)
        mcfg = self._config.model_configs.get(model)
        grid_label = (mcfg.grid_label if mcfg and mcfg.grid_label else "gr")

        base = self._table_dir(model, table)

        # Try 1: standard layout with grid_label and version dirs
        var_dir = base / alias / grid_label
        if var_dir.exists():
            versions = sorted(var_dir.glob("v*"))
            if versions:
                return versions[-1]
            # grid_label dir exists but no version subdirs — use it directly
            if list(var_dir.glob("*.nc")):
                return var_dir

        # Try 2: flat layout — files directly under the variable directory
        flat_dir = base / alias
        if flat_dir.exists() and list(flat_dir.glob("*.nc")):
            logger.debug(
                "Using flat layout for %s/%s/%s (no %s/v* subdirs)",
                model, table, alias, grid_label,
            )
            return flat_dir

        raise FileNotFoundError(
            f"Variable directory not found: tried {base / alias / grid_label} "
            f"and {flat_dir}"
        )
