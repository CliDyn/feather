"""Observation data access."""

from pathlib import Path

import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import VARIABLE_REGISTRY


class ObsLoader:
    """Load observation datasets from configured paths."""

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._cache: dict[str, xr.Dataset] = {}

    def load(self, dataset: str, variable: str,
             period: tuple[str, str] = None) -> xr.DataArray:
        """Load an observation variable, optionally sliced to a period.

        Parameters
        ----------
        dataset : str
            Dataset name, e.g. "ERA5", "CERES_EBAF".
        variable : str
            Variable key as defined in config, e.g. "t2m", "toa_sw_all_mon".
        period : tuple of str, optional
            (start, end) for time slicing, e.g. ("1990", "2014").
        """
        ds_cfg = self._config.obs_datasets.get(dataset)
        if ds_cfg is None:
            raise KeyError(
                f"Dataset {dataset!r} not in config. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        base_path = Path(ds_cfg["path"])
        variables = ds_cfg.get("variables", {})

        if variable in variables:
            filepath = base_path / variables[variable]
        else:
            # Try to find the file by glob
            candidates = sorted(base_path.glob(f"*{variable}*"))
            if not candidates:
                raise FileNotFoundError(
                    f"Variable {variable!r} not found in {dataset} config "
                    f"and no matching file in {base_path}"
                )
            filepath = candidates[0]

        cache_key = f"{dataset}/{variable}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]

        # Find the data variable in the dataset
        da = self._find_variable(ds, variable)

        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))

        return da

    def load_for_model_var(self, model_var: str,
                           period: tuple[str, str] = None) -> xr.DataArray:
        """Load the matching observation for a model variable.

        Uses VARIABLE_REGISTRY to find obs_dataset + obs_variable.
        """
        if model_var not in VARIABLE_REGISTRY:
            raise KeyError(f"No registry entry for model variable {model_var!r}")

        var_info = VARIABLE_REGISTRY[model_var]
        da = self.load(var_info.obs_dataset, var_info.obs_variable, period=period)

        # Apply unit conversion if needed
        if var_info.obs_unit_factor != 1.0:
            da = da * var_info.obs_unit_factor
        if var_info.obs_unit_offset != 0.0:
            da = da + var_info.obs_unit_offset

        return da

    def load_ceres(self, ceres_var: str, period=None, file_key="toa"):
        """Load a specific variable from a CERES EBAF file.

        CERES files contain many variables in a single file, so this
        provides direct access by variable name rather than going
        through VARIABLE_REGISTRY.

        Parameters
        ----------
        ceres_var : str
            Variable name inside the CERES file (e.g., 'toa_net_all_mon').
        period : tuple of str, optional
            (start, end) for time slicing.
        file_key : str
            Config key under CERES_EBAF variables: 'toa' or 'surface'.
        """
        ds_cfg = self._config.obs_datasets.get("CERES_EBAF")
        if ds_cfg is None:
            raise KeyError(
                "CERES_EBAF not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        base_path = Path(ds_cfg["path"])
        variables = ds_cfg.get("variables", {})
        if file_key not in variables:
            raise FileNotFoundError(
                f"File key {file_key!r} not in CERES_EBAF config. "
                f"Available: {list(variables.keys())}"
            )

        filepath = base_path / variables[file_key]
        cache_key = f"CERES_EBAF/{file_key}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        if ceres_var not in ds.data_vars:
            raise KeyError(
                f"Variable {ceres_var!r} not in CERES {file_key} file. "
                f"Available: {list(ds.data_vars)}"
            )

        da = ds[ceres_var]
        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def list_datasets(self) -> list[str]:
        """List configured observation datasets."""
        return list(self._config.obs_datasets.keys())

    def list_variables(self, dataset: str) -> list[str]:
        """List configured variables for a dataset."""
        ds_cfg = self._config.obs_datasets.get(dataset, {})
        return list(ds_cfg.get("variables", {}).keys())

    @staticmethod
    def _find_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
        """Find a data variable in a dataset by name or best match."""
        if variable in ds.data_vars:
            return ds[variable]

        # Try case-insensitive match
        for name in ds.data_vars:
            if name.lower() == variable.lower():
                return ds[name]

        # Return the first non-coordinate data variable
        data_vars = [v for v in ds.data_vars
                     if v not in ds.coords and v not in ("lat", "lon",
                                                          "latitude", "longitude",
                                                          "time", "depth")]
        if data_vars:
            return ds[data_vars[0]]

        raise KeyError(
            f"Variable {variable!r} not found in dataset. "
            f"Available: {list(ds.data_vars)}"
        )
