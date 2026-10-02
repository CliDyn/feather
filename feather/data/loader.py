"""Unified model data access."""

from pathlib import Path

import xarray as xr

from feather import provenance


def _entry_urlpath(entry) -> str | None:
    """Store location of an intake catalog entry (best effort)."""
    try:
        url = getattr(entry, "urlpath", None) or entry.describe()["args"]["urlpath"]
        return str(url) if url else None
    except Exception:
        return None


def _store_fingerprint_paths(url: str | None) -> list[str]:
    """Files fingerprinted for a store.

    A DestinE zarr store holds hundreds of thousands of chunks, so only its
    consolidated metadata is fingerprinted: a republished store rewrites it.
    """
    if not url:
        return []
    path = Path(url)
    if path.is_dir():
        for name in (".zmetadata", "zarr.json", ".zgroup"):
            if (path / name).exists():
                return [str(path / name)]
        return []
    return [url] if path.exists() else []


def _record_catalog_read(prov: dict, key: str, variable: str, ds, url: str | None,
                         catalog: str | None = None) -> None:
    from feather.data.variables import get_var

    try:
        canonical = get_var(variable).name
    except KeyError:
        canonical = variable
    provenance.record_read(
        prov, (key, variable),
        role="model", backend="intake", variable=canonical,
        paths=lambda: _store_fingerprint_paths(url), data=ds[variable],
        catalog_key=key, catalog=catalog, store=url,
        store_variable=variable if variable != canonical else None,
    )


class DataLoader:
    """Unified model data access via intake catalogs or explicit paths."""

    def __init__(self, catalog=None, paths=None):
        self._catalog = catalog
        self._paths = paths or {}
        self._cache: dict[str, xr.Dataset] = {}
        self._prov: dict[tuple, tuple] = {}

    @classmethod
    def from_catalog(cls, catalog_path: str, **kwargs) -> "DataLoader":
        """Open an intake catalog."""
        import intake

        cat = intake.open_catalog(catalog_path)
        return cls(catalog=cat)

    @classmethod
    def from_paths(cls, paths: dict[str, str], **kwargs) -> "DataLoader":
        """Open explicit file paths.

        Parameters
        ----------
        paths : dict
            Maps entry names to file paths (nc/zarr).
        """
        return cls(paths=paths)

    def load(self, key: str) -> xr.Dataset:
        """Load a catalog entry as a lazy xr.Dataset."""
        if key in self._cache:
            return self._cache[key]

        if self._catalog is not None and key in self._catalog:
            ds = self._catalog[key].to_dask()
        elif key in self._paths:
            path = self._paths[key]
            if path.endswith(".zarr") or Path(path).is_dir():
                ds = xr.open_zarr(path)
            else:
                ds = xr.open_dataset(path, chunks="auto")
        else:
            available = self.list_entries()
            raise KeyError(
                f"Entry {key!r} not found. Available: {available[:10]}"
                + (f"... ({len(available)} total)" if len(available) > 10 else "")
            )

        self._cache[key] = ds
        return ds

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        """Load a single variable from a catalog entry."""
        ds = self.load(key)
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        if provenance.current_run() is not None:
            if self._catalog is not None and key in self._catalog:
                url = _entry_urlpath(self._catalog[key])
            else:
                url = self._paths.get(key)
            _record_catalog_read(self._prov, key, variable, ds, url)
        return ds[variable]

    def get_mesh(self) -> xr.Dataset:
        """Get HEALPix mesh (lon, lat, area) from the first loaded dataset.

        Falls back to nereus.healpix.load_mesh if no datasets loaded.
        """
        if self._cache:
            ds = next(iter(self._cache.values()))
            return ds[["longitude", "latitude"]].copy()
        raise RuntimeError("No datasets loaded yet. Call load() first.")

    def list_entries(self) -> list[str]:
        """List available catalog entries or path keys."""
        entries = []
        if self._catalog is not None:
            entries.extend(list(self._catalog))
        if self._paths:
            entries.extend(list(self._paths.keys()))
        return entries

    @staticmethod
    def make_key(experiment: str, model: str, domain: str,
                 member: int = 1) -> str:
        """Build catalog key from components.

        Convention:
            2D (sfc, o2d): {exp}_2_{model}_{member}_0001_clmn_high_{domain}
            3D (o3d, pl):  {exp}_2_{model}_{member}_0001_clmn_standard_{domain}
        """
        resolution = "standard" if domain in ("o3d", "pl") else "high"
        return f"{experiment}_2_{model}_{member}_0001_clmn_{resolution}_{domain}"


class MultiCatalogLoader:
    """DataLoader that searches across multiple intake catalogs."""

    def __init__(self, catalog_paths: dict[str, str]):
        import intake

        self._catalogs = {}
        self._cache: dict[str, xr.Dataset] = {}
        self._prov: dict[tuple, tuple] = {}
        for label, path in catalog_paths.items():
            self._catalogs[label] = intake.open_catalog(path)

    def load(self, key: str) -> xr.Dataset:
        if key in self._cache:
            return self._cache[key]

        for label, cat in self._catalogs.items():
            if key in cat:
                ds = cat[key].to_dask()
                self._cache[key] = ds
                return ds

        available = self.list_entries()[:10]
        raise KeyError(
            f"Entry {key!r} not found in any catalog. "
            f"First entries: {available}"
        )

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        ds = self.load(key)
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        if provenance.current_run() is not None:
            label = next((lb for lb, cat in self._catalogs.items() if key in cat), None)
            url = _entry_urlpath(self._catalogs[label][key]) if label else None
            _record_catalog_read(self._prov, key, variable, ds, url, catalog=label)
        return ds[variable]

    def list_entries(self) -> list[str]:
        entries = []
        for cat in self._catalogs.values():
            entries.extend(list(cat))
        return entries

    @staticmethod
    def make_key(experiment: str, model: str, domain: str,
                 member: int = 1) -> str:
        return DataLoader.make_key(experiment, model, domain, member)
