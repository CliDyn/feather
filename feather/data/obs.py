"""Observation data access."""

from pathlib import Path

import numpy as np
import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var


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
        var_info = get_var(model_var)
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

    def load_esa_cci(self, product: str = "analysed_sst", period=None):
        """Load ESA-CCI SST product.

        Parameters
        ----------
        product : str
            Config key: ``"analysed_sst"`` (monthly), ``"timemean"``,
            or ``"ymonmean"``.
        period : tuple of str, optional
            Time slicing (relevant for monthly data).

        Returns
        -------
        xr.DataArray
            SST data (Kelvin, dims vary by product).
        """
        cfg = self._config.obs_datasets.get("ESA_CCI")
        if cfg is None:
            raise KeyError(
                "ESA_CCI not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        path = Path(cfg["path"])
        variables = cfg.get("variables", {})
        if product not in variables:
            raise FileNotFoundError(
                f"Product {product!r} not in ESA_CCI config. "
                f"Available: {list(variables.keys())}"
            )

        filepath = path / variables[product]
        cache_key = f"ESA_CCI/{product}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        da = ds["analysed_sst"]
        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_osisaf(self, hemisphere: str, period=None):
        """Load OSI-SAF sea ice concentration dataset.

        Parameters
        ----------
        hemisphere : str
            ``"nh"`` or ``"sh"``.
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.Dataset
            Raw dataset on EASE2 grid with ``ice_conc`` variable.
        """
        ds_cfg = self._config.obs_datasets.get("OSI_SAF")
        if ds_cfg is None:
            raise KeyError(
                "OSI_SAF not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        base_path = Path(ds_cfg["path"])
        variables = ds_cfg.get("variables", {})
        if hemisphere not in variables:
            raise FileNotFoundError(
                f"Hemisphere {hemisphere!r} not in OSI_SAF config. "
                f"Available: {list(variables.keys())}"
            )

        filepath = base_path / variables[hemisphere]
        cache_key = f"OSI_SAF/{hemisphere}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        if period is not None and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds

    def load_psc(self, product: str, period=None):
        """Load PSC (PIOMAS/GIOMAS) sea ice thickness dataset.

        Parameters
        ----------
        product : str
            ``"piomas"`` (NH) or ``"giomas"`` (SH).
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.Dataset
            Raw dataset with ``sithick``, ``areacello``, ``latitude``.
        """
        ds_cfg = self._config.obs_datasets.get("PSC")
        if ds_cfg is None:
            raise KeyError(
                "PSC not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        base_path = Path(ds_cfg["path"])
        variables = ds_cfg.get("variables", {})
        if product not in variables:
            raise FileNotFoundError(
                f"Product {product!r} not in PSC config. "
                f"Available: {list(variables.keys())}"
            )

        filepath = base_path / variables[product]
        cache_key = f"PSC/{product}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        if period is not None and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds

    def load_en4(self, variable: str = "thetao", period=None):
        """Load EN4 v4.2.2 ocean variable.

        Parameters
        ----------
        variable : str
            ``"thetao"`` (temperature) or ``"so"`` (salinity).
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.DataArray
            Data with dims ``(time, lev, lat, lon)``.
        """
        ds = self.load_en4_dataset(variable, period=period)
        da = self._find_variable(ds, variable)
        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_en4_dataset(self, variable: str = "thetao", period=None):
        """Load EN4 v4.2.2 as a full Dataset (includes ``lev_bnds``).

        Parameters
        ----------
        variable : str
            ``"thetao"`` or ``"so"`` — selects the file to open.
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.Dataset
            Full dataset with ``lev``, ``lev_bnds``, etc.
        """
        ds_cfg = self._config.obs_datasets.get("EN4")
        if ds_cfg is None:
            raise KeyError(
                "EN4 not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        base_path = Path(ds_cfg["path"])
        variables = ds_cfg.get("variables", {})
        if variable not in variables:
            raise FileNotFoundError(
                f"Variable {variable!r} not in EN4 config. "
                f"Available: {list(variables.keys())}"
            )

        filepath = base_path / variables[variable]
        cache_key = f"EN4/{variable}"
        if cache_key not in self._cache:
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        if period is not None and "time" in ds.dims:
            ds = ds.sel(time=slice(period[0], period[1]))
        return ds

    def load_mswep(self, period=None) -> xr.DataArray:
        """Load MSWEP v2.8 monthly precipitation (converted to kg/m²/s).

        MSWEP data is stored as mm/month.  The conversion is time-varying
        because months have different numbers of days.  Longitudes are
        shifted from -180..180 to 0..360 to match the framework convention.

        Parameters
        ----------
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.DataArray
            Precipitation in kg/m²/s on a 0.1° global grid.
        """
        ds_cfg = self._config.obs_datasets.get("MSWEP")
        if ds_cfg is None:
            raise KeyError(
                "MSWEP not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        cache_key = "MSWEP/precipitation"
        if cache_key not in self._cache:
            base_path = Path(ds_cfg["path"])
            variables = ds_cfg.get("variables", {})
            filepath = base_path / variables.get(
                "pr", "zarr/mswep280.past-nrt.monthly.zarr"
            )
            self._cache[cache_key] = xr.open_zarr(str(filepath))

        da = self._cache[cache_key]["precipitation"]

        # Convert mm/month -> kg/m²/s  (1 mm = 1 kg/m²)
        seconds_in_month = da.time.dt.days_in_month * 86400
        da = da / seconds_in_month

        # Convert lons from -180..180 -> 0..360 (match ERA5/model convention)
        lon = da.lon.values
        lon_360 = np.where(lon < 0, lon + 360, lon)
        sort_idx = np.argsort(lon_360)
        da = da.isel(lon=sort_idx).assign_coords(lon=lon_360[sort_idx])

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
