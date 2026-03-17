"""Load model data from per-month GRIB files on a HEALPix grid.

Supports directory structures like::

    {root}/{year}/{dir_name}/{file_prefix}_{YYYYMM}.grib

where *dir_name* encodes the domain and variable (e.g. ``sfc_mean2t``,
``o2d_avg_tos``, ``o3d_avg_thetao``) and *file_prefix* is the directory
name with the domain prefix stripped (``mean2t``, ``avg_tos``,
``avg_thetao``).

The data dimension is ``values`` (HEALPix 1-D) with ``latitude`` and
``longitude`` coordinates provided natively by cfgrib.  Singleton
dimensions (``heightAboveGround``, ``step``, ``valid_time``, etc.) are
squeezed automatically.

Ocean 3D variables have an ``oceanModelLayer`` dimension with integer
indices that are remapped to real depth values via config.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var

logger = logging.getLogger(__name__)

# Singleton dimensions/coordinates that should be squeezed automatically
_SQUEEZE_DIMS = {
    "heightAboveGround", "step", "valid_time", "surface",
    "entireAtmosphere", "meanSea", "iceTopOnWater", "iceLayerOnWater",
    "oceanSurface",
}


class GRIBLoader:
    """Load model data from per-month GRIB files on a HEALPix grid.

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
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load a variable for a model from per-month GRIB files.

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

    def _get_data_root(self, model: str) -> Path:
        """Return data root, using per-model override if set."""
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.data_root:
            return Path(mcfg.data_root)
        return self._root

    def _get_dir_name(self, model: str, variable: str) -> str:
        """Resolve the on-disk directory name for *variable*.

        Lookup order:
        1. Per-model ``variable_aliases`` (e.g. ``"tas"`` → ``"sfc_mean2t"``)
        2. ``destine_variable`` from ``VARIABLE_REGISTRY``
        3. Canonical CMOR name
        """
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.variable_aliases:
            alias = mcfg.variable_aliases.get(variable)
            if alias:
                return alias

        try:
            vinfo = get_var(variable)
            if vinfo.destine_variable:
                return vinfo.destine_variable
        except KeyError:
            pass

        return variable

    @staticmethod
    def _get_file_prefix(dir_name: str) -> str:
        """Derive file prefix by stripping domain prefix from dir_name.

        ``"sfc_mean2t"`` → ``"mean2t"``
        ``"o3d_avg_thetao"`` → ``"avg_thetao"``
        ``"o2d_avg_tos"`` → ``"avg_tos"``
        """
        parts = dir_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ("sfc", "o2d", "o3d", "pl",
                                              "sol", "misc"):
            return parts[1]
        return dir_name

    def _get_scale_factor(self, model: str, variable: str) -> float:
        """Return a post-load scale factor (default 1.0)."""
        mcfg = self._config.model_configs.get(model)
        if mcfg and mcfg.scale_factors:
            return mcfg.scale_factors.get(variable, 1.0)
        return 1.0

    def _get_depth_levels(self) -> list[float] | None:
        """Return ocean 3D depth levels from config, if any."""
        return self._config.ocean_3d.get("depth_levels")

    def _discover_grib_files(
        self, data_root: Path, dir_name: str, file_prefix: str,
    ) -> list[Path]:
        """Find all GRIB files matching the expected pattern.

        Searches ``{data_root}/{year}/{dir_name}/{file_prefix}_*.grib``
        across all year subdirectories.
        """
        files = []
        if not data_root.exists():
            return files

        for year_dir in sorted(data_root.iterdir()):
            if not year_dir.is_dir():
                continue
            # Only pick up numeric year directories
            if not year_dir.name.isdigit():
                continue
            var_dir = year_dir / dir_name
            if var_dir.exists():
                files.extend(sorted(var_dir.glob(f"{file_prefix}_*.grib")))

        return files

    def _open_variable(self, model: str, variable: str) -> xr.DataArray:
        """Open and return a DataArray for a single variable."""
        dir_name = self._get_dir_name(model, variable)
        file_prefix = self._get_file_prefix(dir_name)
        data_root = self._get_data_root(model)

        grib_files = self._discover_grib_files(data_root, dir_name,
                                                file_prefix)
        if not grib_files:
            raise FileNotFoundError(
                f"No GRIB files for {model}/{variable} "
                f"(dir={dir_name}, prefix={file_prefix}) under {data_root}"
            )

        logger.info(
            "Opening %s/%s (%d files) from %s",
            model, variable, len(grib_files), data_root,
        )

        ds = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            combine="nested",
            concat_dim="time",
            backend_kwargs={"indexpath": ""},
            chunks="auto",
        )

        # Find the data variable
        nc_var = self._find_data_var(ds, dir_name, variable)
        da = ds[nc_var]

        # Squeeze singleton dimensions
        for dim in list(da.dims):
            if dim in _SQUEEZE_DIMS and da.sizes.get(dim, 0) == 1:
                da = da.squeeze(dim, drop=True)
        # Also drop scalar coords from the squeeze set
        for coord in list(da.coords):
            if coord in _SQUEEZE_DIMS and coord not in da.dims:
                da = da.drop_vars(coord)

        # Remap ocean model layers to real depth values
        if "oceanModelLayer" in da.dims:
            da = self._remap_ocean_depth(da)

        # Apply per-model scale factor
        scale = self._get_scale_factor(model, variable)
        if scale != 1.0:
            da = da * scale

        return da

    def _remap_ocean_depth(self, da: xr.DataArray) -> xr.DataArray:
        """Replace ``oceanModelLayer`` indices with real depth values.

        If ``ocean_3d.depth_levels`` is configured, maps the integer
        GRIB layer indices (2, 4, ..., 56) to corresponding depth
        values.  Otherwise, just renames the dimension to ``lev``.
        """
        depth_levels = self._get_depth_levels()
        n_layers = da.sizes["oceanModelLayer"]

        if depth_levels and len(depth_levels) == n_layers:
            da = da.assign_coords(
                oceanModelLayer=np.array(depth_levels, dtype=np.float64),
            )
        da = da.rename({"oceanModelLayer": "lev"})
        return da

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
        parts = dir_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ("sfc", "o2d", "o3d", "pl",
                                              "sol", "misc"):
            stripped = parts[1]
            if stripped in ds:
                return stripped

        if variable in ds:
            return variable

        data_vars = [v for v in ds.data_vars if v not in ds.coords]
        if data_vars:
            return data_vars[0]

        raise KeyError(
            f"No data variable found in dataset for {variable!r} "
            f"(dir_name={dir_name!r}). Available: {list(ds.data_vars)}"
        )
