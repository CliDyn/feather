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

    def load_cru(self, variable: str, period=None) -> xr.DataArray:
        """Load a CRU TS v4.09 monthly land variable.

        CRU TS is a 0.5° **land-only** gridded dataset (ocean cells are NaN)
        distributed as per-decade NetCDF files.  This method opens all decade
        files for the requested variable, concatenates them along time, and
        returns the field in framework-canonical units:

        ============  =========================  ====================
        CRU variable  meaning                    returned units
        ============  =========================  ====================
        ``pre``       precipitation              kg/m²/s
        ``tmp``       mean near-surface temp.    K
        ``tmn``       min near-surface temp.     K
        ``tmx``       max near-surface temp.     K
        ``cld``       cloud cover                % (unchanged)
        ============  =========================  ====================

        Longitudes are shifted from −180..180 to 0..360 to match the
        framework convention.  Auxiliary variables (``stn``/``mae``/``maea``)
        are dropped.

        Parameters
        ----------
        variable : str
            One of ``pre``, ``tmp``, ``tmn``, ``tmx``, ``cld``.
        period : tuple of str, optional
            (start, end) for time slicing.
        """
        ds_cfg = self._config.obs_datasets.get("CRU")
        if ds_cfg is None:
            raise KeyError(
                "CRU not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        cache_key = f"CRU/{variable}"
        if cache_key not in self._cache:
            base_path = Path(ds_cfg["path"])
            variables = ds_cfg.get("variables", {})
            # Config may give an explicit glob pattern per variable; otherwise
            # fall back to the standard CRU TS file-naming convention.
            pattern = variables.get(variable, f"cru_ts4.09.*.{variable}.dat.nc")
            files = sorted(base_path.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No CRU files matching {pattern!r} in {base_path}"
                )
            ds = xr.open_mfdataset(
                files, combine="by_coords", chunks="auto",
                data_vars="minimal", coords="minimal", compat="override",
            )
            self._cache[cache_key] = ds

        da = self._cache[cache_key][variable]

        # Shift lons −180..180 → 0..360
        if "lon" in da.coords and float(da.lon.min()) < 0:
            lon = da.lon.values
            lon_360 = np.where(lon < 0, lon + 360, lon)
            sort_idx = np.argsort(lon_360)
            da = da.isel(lon=sort_idx).assign_coords(lon=lon_360[sort_idx])

        # Unit conversion to framework-canonical units
        if variable == "pre":
            seconds_in_month = da.time.dt.days_in_month * 86400
            da = da / seconds_in_month  # mm/month → kg/m²/s
        elif variable in ("tmp", "tmn", "tmx"):
            da = da + 273.15  # °C → K
        # cld stays in %

        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_chirps(self, period=None) -> xr.DataArray:
        """Load CHIRPS v3.0 monthly precipitation (converted to kg/m²/s).

        CHIRPS is a 0.05° quasi-global (60°N–60°S) **land-only** satellite +
        gauge precipitation product stored as mm/month.  The conversion to
        kg/m²/s is time-varying (months differ in length).  Longitudes are
        shifted from −180..180 to 0..360.

        Parameters
        ----------
        period : tuple of str, optional
            (start, end) for time slicing.
        """
        ds_cfg = self._config.obs_datasets.get("CHIRPS")
        if ds_cfg is None:
            raise KeyError(
                "CHIRPS not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        cache_key = "CHIRPS/precip"
        if cache_key not in self._cache:
            base_path = Path(ds_cfg["path"])
            variables = ds_cfg.get("variables", {})
            filepath = base_path / variables.get(
                "precip", "chirps-v3.0.monthly.nc"
            )
            self._cache[cache_key] = xr.open_dataset(filepath, chunks="auto")

        ds = self._cache[cache_key]
        da = ds["precip"]

        # Rename dims latitude/longitude → lat/lon
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)

        # Mask the −9999 fill value if it survived decoding
        da = da.where(da > -9000)

        # Convert mm/month → kg/m²/s
        seconds_in_month = da.time.dt.days_in_month * 86400
        da = da / seconds_in_month

        # Shift lons −180..180 → 0..360
        if float(da.lon.min()) < 0:
            lon = da.lon.values
            lon_360 = np.where(lon < 0, lon + 360, lon)
            sort_idx = np.argsort(lon_360)
            da = da.isel(lon=sort_idx).assign_coords(lon=lon_360[sort_idx])

        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        return da

    def load_berkeley_hr(self, dataset_key: str, period=None) -> xr.DataArray:
        """Load a Berkeley Earth high-resolution (0.25°) gridded field in K.

        Handles the Berkeley Earth gridded format used by both the global
        TAVG product and the land TMAX/TMIN products.  Files store monthly
        *anomalies* (°C, re: 1951–1980) plus a 12-month ``climatology`` array;
        the absolute temperature is reconstructed as
        ``anomaly + climatology[month_of_year]``.  Time is encoded as decimal
        years and converted to a proper ``DatetimeIndex``.

        Parameters
        ----------
        dataset_key : str
            obs_datasets key, e.g. ``BERKELEY_EARTH_HR``,
            ``BERKELEY_EARTH_LAND_TMAX``, ``BERKELEY_EARTH_LAND_TMIN``.
        period : tuple of str, optional
            (start, end) for time slicing.

        Returns
        -------
        xr.DataArray
            Absolute temperature in K on a 0.25° grid, dims ``lat``/``lon``,
            lons in 0..360.
        """
        import pandas as pd

        ds_cfg = self._config.obs_datasets.get(dataset_key)
        if ds_cfg is None:
            raise KeyError(
                f"{dataset_key} not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        cache_key = f"{dataset_key}/temperature"
        if cache_key not in self._cache:
            base_path = Path(ds_cfg["path"])
            variables = ds_cfg.get("variables", {})
            filename = variables.get("temperature")
            if filename is None:
                # single-entry datasets: take the only configured file
                filename = next(iter(variables.values()))
            self._cache[cache_key] = xr.open_dataset(
                base_path / filename, chunks="auto",
            )

        ds_full = self._cache[cache_key]

        # Decimal-year time → DatetimeIndex (e.g. 1981.125 → Feb 1981)
        dec_years = ds_full["time"].values
        years = dec_years.astype(int)
        months = np.floor((dec_years - years) * 12).astype(int) + 1
        months = np.clip(months, 1, 12)
        datetimes = pd.to_datetime(
            [f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)]
        )
        ds_full = ds_full.assign_coords(time=datetimes)

        anom = ds_full["temperature"]
        if period is not None:
            anom = anom.sel(time=slice(period[0], period[1]))
        clim = ds_full["climatology"]  # (month_number, lat, lon)

        month_idx = anom.time.dt.month.values - 1
        clim_matched = clim.values[month_idx]
        abs_temp = anom + xr.DataArray(
            clim_matched, dims=anom.dims, coords=anom.coords,
        )

        rename = {}
        if "latitude" in abs_temp.dims:
            rename["latitude"] = "lat"
        if "longitude" in abs_temp.dims:
            rename["longitude"] = "lon"
        if rename:
            abs_temp = abs_temp.rename(rename)

        if float(abs_temp.lon.min()) < 0:
            abs_temp = abs_temp.assign_coords(
                lon=((abs_temp.lon + 360) % 360),
            ).sortby("lon")

        return abs_temp + 273.15  # °C → K

    def load_hadisst(self, period=None) -> xr.DataArray:
        """Load HadISST monthly sea-surface temperature in K.

        HadISST is a 1° global SST reconstruction covering 1870–present (so,
        unlike ESA-CCI's 1990–2014, it spans the full 1980–2014 analysis
        window).  Land and ice-covered cells are stored as a large negative
        fill (~-1000 °C); these are masked to NaN.  Values are returned in
        Kelvin (the canonical ``tos`` unit) with dims ``lat``/``lon`` and
        longitudes in 0..360.

        Parameters
        ----------
        period : tuple of str, optional
            (start, end) for time slicing.
        """
        ds_cfg = self._config.obs_datasets.get("HADISST")
        if ds_cfg is None:
            raise KeyError(
                "HADISST not configured in obs_datasets. "
                f"Available: {list(self._config.obs_datasets.keys())}"
            )

        cache_key = "HADISST/sst"
        if cache_key not in self._cache:
            base_path = Path(ds_cfg["path"])
            variables = ds_cfg.get("variables", {})
            filename = variables.get("sst") or next(iter(variables.values()))
            self._cache[cache_key] = xr.open_dataset(
                base_path / filename, chunks="auto",
            )

        ds = self._cache[cache_key]
        var = "sst" if "sst" in ds.data_vars else next(iter(ds.data_vars))
        da = ds[var]

        # Mask land/ice fill cells (~-1000 °C); keep real SST only.
        da = da.where(da > -100.0)

        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)

        # Shift lons −180..180 → 0..360 to match the framework convention.
        if float(da.lon.min()) < 0:
            da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")

        if period is not None and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))

        return da + 273.15  # °C → K (canonical tos unit)

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
