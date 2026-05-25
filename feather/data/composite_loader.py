"""Composite model loader for multi-data-source configurations.

When models in a single config use different backends (e.g. DestinE
intake catalogs + GRIB files), :class:`CompositeModelLoader` wraps
one backend per source type and routes ``load_var()`` / ``load_coords()``
calls to the correct one based on each model's ``data_source_type``.
"""

import logging

import numpy as np
import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var

logger = logging.getLogger(__name__)


class _DestinECatalogAdapter:
    """Wrap a :class:`MultiCatalogLoader` with the unified load API.

    Translates ``load_var(model, variable, ...)`` calls into the
    DestinE-style catalog key + DestinE variable name lookup that the
    catalog loader expects.
    """

    def __init__(self, config: FeatherConfig, catalog_loader):
        self._config = config
        self._catalog = catalog_loader

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        from feather.data.loader import DataLoader

        vinfo = get_var(variable)
        destine_var = vinfo.destine_variable or variable

        mc = self._config.model_configs.get(model)
        exp = (mc.experiment if mc and mc.experiment
               else self._config.get_experiment())
        cat_key = mc.catalog_key if mc and mc.catalog_key else model

        key = DataLoader.make_key(exp, cat_key, vinfo.domain)
        da = self._catalog.load_var(key, destine_var)

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(
        self, model: str, variable: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        from feather.data.loader import DataLoader

        vinfo = get_var(variable)
        mc = self._config.model_configs.get(model)
        exp = (mc.experiment if mc and mc.experiment
               else self._config.get_experiment())
        cat_key = mc.catalog_key if mc and mc.catalog_key else model

        key = DataLoader.make_key(exp, cat_key, vinfo.domain)
        ds = self._catalog.load(key)
        return np.asarray(ds["longitude"]), np.asarray(ds["latitude"])


class CompositeModelLoader:
    """Route ``load_var()`` to the correct backend per model.

    Parameters
    ----------
    config : FeatherConfig
        Pipeline config with per-model ``data_source_type`` overrides.
    """

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._backends: dict[str, object] = {}  # model → backend
        self._init_backends()

    def _init_backends(self):
        """Create one backend per unique source type, map models."""
        # Group models by source type
        type_to_models: dict[str, list[str]] = {}
        for model in self._config.models:
            src = self._config.get_model_data_source_type(model)
            type_to_models.setdefault(src, []).append(model)

        # Cache of instantiated backends (one per type)
        backend_cache: dict[str, object] = {}

        for src_type, models in type_to_models.items():
            if src_type not in backend_cache:
                backend_cache[src_type] = self._create_backend(src_type)
            backend = backend_cache[src_type]
            for model in models:
                self._backends[model] = backend

        types_used = list(type_to_models.keys())
        logger.info(
            "CompositeModelLoader: %d models across %d backends (%s)",
            len(self._backends), len(backend_cache), types_used,
        )

    def _create_backend(self, src_type: str):
        """Instantiate a backend for the given source type."""
        if src_type == "grib_healpix":
            from feather.data.grib_loader import GRIBLoader
            return GRIBLoader(self._config)

        if src_type == "cmor":
            from feather.data.cmor_loader import CMORLoader
            return CMORLoader(self._config)

        if src_type == "netcdf_healpix":
            from feather.data.netcdf_loader import NetCDFLoader
            return NetCDFLoader(self._config)

        if src_type == "kerchunk_parquet":
            from feather.data.kerchunk_loader import KerchunkParquetLoader
            return KerchunkParquetLoader(self._config)

        if src_type == "destine_catalog":
            from feather.data.loader import MultiCatalogLoader
            catalogs = self._config.model_catalogs
            if not catalogs:
                raise ValueError(
                    "destine_catalog backend requires model_catalogs in config"
                )
            cat_loader = MultiCatalogLoader(catalogs)
            return _DestinECatalogAdapter(self._config, cat_loader)

        raise ValueError(f"Unknown data source type: {src_type!r}")

    def _get_backend(self, model: str):
        """Return the backend for *model*, raising if unknown."""
        backend = self._backends.get(model)
        if backend is None:
            raise KeyError(
                f"Model {model!r} not configured in CompositeModelLoader. "
                f"Known models: {list(self._backends)}"
            )
        return backend

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load a variable, routing to the correct backend.

        Parameters
        ----------
        table : str, optional
            CMOR table (e.g. ``"day"``).  Forwarded to backends that
            accept it (CMORLoader); silently ignored by others.
        """
        import inspect

        backend = self._get_backend(model)
        sig = inspect.signature(backend.load_var)
        kwargs: dict = {"period": period, "time_mean": time_mean}
        if "table" in sig.parameters and table is not None:
            kwargs["table"] = table
        return backend.load_var(model, variable, **kwargs)

    def load_coords(
        self, model: str, variable: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load (lon, lat) coordinates for a model variable."""
        backend = self._get_backend(model)

        # DestinECatalogAdapter has its own load_coords
        if hasattr(backend, "load_coords"):
            return backend.load_coords(model, variable)

        # GRIBLoader / NetCDFLoader: load var and extract coords
        da = backend.load_var(model, variable)
        src = self._config.get_model_data_source_type(model)
        if src == "cmor":
            return np.asarray(da.lon), np.asarray(da.lat)
        # HEALPix-style (grib_healpix, netcdf_healpix)
        return np.asarray(da["longitude"]), np.asarray(da["latitude"])
