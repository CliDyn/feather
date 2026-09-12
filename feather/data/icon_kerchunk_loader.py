"""Load ICON-ESM-ER ensemble members from Parquet-format kerchunk stores.

The EERIE CMOR tree publishes only ``r1i1p1f1`` for ICON-ESM-ER.  The
additional hist-1950 members (``r2i1p1f1``, ``r3i1p1f1``) exist solely as
intake catalogues pointing at kerchunk reference stores::

    /work/bm1344/DKRZ/intake/disk/phase2-model-output/icon-esm-er/hist-1950/
        r2i1p1f1/{atmos,ocean,land}/gr025/main.yaml
            → /work/bm1344/DKRZ/kerchunks_batched/ICON/phase2/hist-1950/
                  v20240618/{2,3}/erc2023_{realm}_native_*_remap025.parq

This loader reads the ``2d_monthly_mean`` gr025 stores directly (bypassing
intake) and presents them through the same ``load_var``/``load_coords`` API
as :class:`~feather.data.cmor_loader.CMORLoader`, so the members drop into a
multi-source config next to the CMOR-published r1.

Why a dedicated loader
----------------------
:class:`~feather.data.kerchunk_loader.KerchunkParquetLoader` handles the
IFS-FESOM2 r2/r3 stores, which are a different animal: flat ``value``
dimension, GRIB short names, ``9999.0`` fill.  The ICON stores are already
rectilinear ``(time, lat, lon)`` on the standard 0.25° grid with near-CMOR
variable names, so almost nothing is shared.

Store conventions handled here
------------------------------
* Grid is standard and needs no reordering: ``lat`` −90→90 (721),
  ``lon`` 0→359.75 (1440).  Time is monthly, 1975-02 → 2014-12.
* Singleton ``height`` / ``height_2`` / ``height_3`` / ``lev`` / ``depth``
  dimensions are squeezed away.
* Ocean fields mark land with ``missing_value = -9e33``.  The zarr metadata
  also declares ``fill_value: 0.0``, so letting xarray mask automatically
  would turn every genuine zero (open-water ``conc``, ``hi``) into NaN.
  Stores are therefore opened with ``mask_and_scale=False`` and masked here
  on the sentinel alone.
* Sign/unit conversions to the CMOR conventions used elsewhere in feather
  are listed in the variable tables below.

Deliberately *not* provided
---------------------------
* Net radiation (``rss``/``rls``/``rst``/``rlt`` and clear-sky variants).
  The store carries the full up/down component set and the nets could be
  derived, but :class:`CMORLoader` does not derive them for ICON r1 either.
  Deriving here would put r2/r3 into radiation panels that r1 is absent
  from and silently skew per-model ensemble statistics.  Keep the three
  members interchangeable; deriving nets is a separate change that belongs
  in ``CMORLoader`` so it lands for r1 at the same time.
* 3-D ocean (``thetao``/``so``/``uo``/``vo``) and 3-D atmosphere
  (``ta``/``ua``/``va``/``zg``).  r2 publishes no model-level ocean store at
  all and r3's covers only 90 months, so neither supports a 1980–2014
  climatology.  Diagnostics that need them (``ocean_en4``) skip these
  members via the usual ``KeyError`` path.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from feather.config import FeatherConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable maps: CMOR canonical name → (store variable, scale, offset)
#
# result = raw * scale + offset
# ---------------------------------------------------------------------------

#: Atmosphere 2-D monthly-mean fields.
#:
#: Names are already CMOR apart from the cases noted.  Two conversions:
#:
#: * ``clt`` is a 0-1 fraction in the store (units ``m2 m-2``) but a
#:   percentage in the CMOR tree (ICON r1 global mean ≈ 65 %) → ×100.
#: * ``hfls``/``hfss`` are positive-*downward* in native ICON output
#:   (global means ≈ −87 / −18 W m⁻²) but positive-*upward* in the CMOR tree
#:   (≈ +87 / +19 W m⁻²) → ×(−1).
#:
#: Radiation components are already CMOR-signed (upwelling positive:
#: ``rlut`` ≈ +239, ``rsut`` ≈ +101 W m⁻²) and pass through unchanged.
_ATMOS2D: dict[str, tuple[str, float, float]] = {
    # Temperature
    "tas":     ("tas",     1.0, 0.0),   # K
    "ts":      ("ts",      1.0, 0.0),   # K
    "tasmin":  ("tasmin",  1.0, 0.0),   # K, monthly mean of daily minima
    "tasmax":  ("tasmax",  1.0, 0.0),   # K, monthly mean of daily maxima
    # Pressure
    "psl":     ("psl",     1.0, 0.0),   # Pa
    "ps":      ("ps",      1.0, 0.0),   # Pa
    # Wind
    "uas":     ("uas",     1.0, 0.0),   # m/s
    "vas":     ("vas",     1.0, 0.0),   # m/s
    "sfcWind": ("sfcwind", 1.0, 0.0),   # m/s (store spells it lower-case)
    # Humidity
    "hurs":    ("hur",     1.0, 0.0),   # % (store's singleton-level field)
    "huss":    ("hus2m",   1.0, 0.0),   # kg/kg
    # Moisture fluxes
    "pr":      ("pr",      1.0, 0.0),   # kg/m²/s
    "evspsbl": ("evspsbl", 1.0, 0.0),   # kg/m²/s
    "prw":     ("prw",     1.0, 0.0),   # kg/m²
    # Clouds
    "clt":     ("clt",   100.0, 0.0),   # fraction → %
    "clivi":   ("clivi",   1.0, 0.0),   # kg/m²
    # Surface turbulent heat fluxes — ICON down-positive → CMOR up-positive
    "hfss":    ("hfss",   -1.0, 0.0),   # W/m²
    "hfls":    ("hfls",   -1.0, 0.0),   # W/m²
    # Surface radiation components (W/m², CMOR-signed)
    "rsds":    ("rsds",    1.0, 0.0),
    "rsdscs":  ("rsdscs",  1.0, 0.0),
    "rsus":    ("rsus",    1.0, 0.0),
    "rsuscs":  ("rsuscs",  1.0, 0.0),
    "rlds":    ("rlds",    1.0, 0.0),
    "rldscs":  ("rldscs",  1.0, 0.0),
    "rlus":    ("rlus",    1.0, 0.0),
    # TOA radiation components (W/m², CMOR-signed)
    "rsdt":    ("rsdt",    1.0, 0.0),
    "rsut":    ("rsut",    1.0, 0.0),
    "rsutcs":  ("rsutcs",  1.0, 0.0),
    "rlut":    ("rlut",    1.0, 0.0),
    "rlutcs":  ("rlutcs",  1.0, 0.0),
    # Wind stress (N/m²)
    "tauu":    ("tauu",    1.0, 0.0),
    "tauv":    ("tauv",    1.0, 0.0),
}

#: Ocean / sea-ice 2-D monthly-mean fields.
#:
#: ``to``/``so`` carry a singleton ``depth`` (1 m) and ``conc``/``hi``/``hs``
#: a singleton ``lev`` (ice class); both are squeezed.  ``to`` is already °C
#: and ``so`` already practical salinity (PSU) — ICON is *not* a TEOS-10
#: model, so no absolute-salinity conversion applies.  ``conc`` is a 0-1
#: fraction in the store but a percentage in the CMOR tree → ×100.
_OCEAN2D: dict[str, tuple[str, float, float]] = {
    "tos":       ("to",      1.0, 0.0),   # °C
    "sos":       ("so",      1.0, 0.0),   # PSU
    "zos":       ("ssh",     1.0, 0.0),   # m
    "mlotst":    ("mlotst",  1.0, 0.0),   # m (sigma-t criterion)
    "siconc":    ("conc",  100.0, 0.0),   # fraction → %
    "sithick":   ("hi",      1.0, 0.0),   # m
    "sisnthick": ("hs",      1.0, 0.0),   # m
}

#: Singleton dimensions to drop when present on a store variable.
_SQUEEZE_DIMS = ("height", "height_2", "height_3", "lev", "depth")

#: Land/missing sentinel used by the ocean stores (``missing_value``).
#: Anything at or below this is land; genuine fields never approach it.
_FILL_THRESHOLD = -1e30

#: Store file names under ``{root}/{member}/``.
_STORE_FILES = {
    "atmos2d": "erc2023_atmos_native_2d_monthly_mean_remap025.parq",
    "ocean2d": "erc2023_ocean_native_2d_monthly_mean_remap025.parq",
}


class ICONKerchunkLoader:
    """Load ICON-ESM-ER members from gr025 kerchunk reference stores.

    Parameters
    ----------
    config : FeatherConfig
        Pipeline config.  Each model handled by this loader needs a
        ``data_root`` pointing at the directory that holds the per-member
        sub-directories (``.../v20240618``), plus a ``member`` index naming
        the sub-directory (``2`` for r2i1p1f1, ``3`` for r3i1p1f1).  A global
        ``data_source.root`` is used when a model sets no ``data_root``.
    """

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("root", ""))
        self._store_cache: dict[tuple[str, str], xr.Dataset] = {}

    # ------------------------------------------------------------------
    # Public API (matches CMORLoader / KerchunkParquetLoader)
    # ------------------------------------------------------------------

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,   # accepted for API parity; ignored
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load *variable* for *model* as ``(time, lat, lon)``.

        Raises
        ------
        KeyError
            If *variable* is not published by these stores.  Diagnostics
            treat this the same as a missing CMOR file and skip the model.
        """
        da = self._load_raw(model, variable)

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(
        self, model: str, variable: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lon, lat)`` 1-D coordinate arrays for *model*.

        Both realms share the same 0.25° grid, so the atmosphere store
        answers for every variable.
        """
        ds = self._open_store(model, "atmos2d")
        return np.asarray(ds["lon"]), np.asarray(ds["lat"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_raw(self, model: str, variable: str) -> xr.DataArray:
        if variable in _ATMOS2D:
            store, (name, scale, offset) = "atmos2d", _ATMOS2D[variable]
        elif variable in _OCEAN2D:
            store, (name, scale, offset) = "ocean2d", _OCEAN2D[variable]
        else:
            raise KeyError(
                f"Variable {variable!r} not published by the ICON gr025 "
                f"monthly stores. Known: "
                f"{sorted(_ATMOS2D) + sorted(_OCEAN2D)}"
            )

        ds = self._open_store(model, store)
        if name not in ds:
            raise KeyError(
                f"Store variable {name!r} (for {variable!r}) missing from the "
                f"{store} store of {model!r}"
            )

        raw = ds[name]
        # Stay lazy: the caller may only want a few months of a 40-year field.
        data = raw.data.astype(np.float32)
        data = np.where(data <= _FILL_THRESHOLD, np.float32(np.nan), data)
        if scale != 1.0:
            data = data * np.float32(scale)
        if offset != 0.0:
            data = data + np.float32(offset)

        da = xr.DataArray(
            data,
            dims=raw.dims,
            coords={d: ds[d] for d in raw.dims if d in ds.coords},
            name=variable,
            attrs={"units": raw.attrs.get("units", ""), "long_name": variable},
        )

        for dim in _SQUEEZE_DIMS:
            if dim in da.dims and da.sizes[dim] == 1:
                da = da.squeeze(dim, drop=True)

        return self._apply_model_scale(model, variable, da)

    def _apply_model_scale(
        self, model: str, variable: str, da: xr.DataArray,
    ) -> xr.DataArray:
        """Apply any per-model ``scale_factors`` override from the config."""
        mc = self._config.model_configs.get(model)
        if mc and mc.scale_factors:
            factor = mc.scale_factors.get(variable, 1.0)
            if factor != 1.0:
                da = da * factor
        return da

    def _open_store(self, model: str, store: str) -> xr.Dataset:
        """Open (and cache) the reference store for *model*/*store*."""
        key = (model, store)
        if key in self._store_cache:
            return self._store_cache[key]

        import fsspec

        path = self._store_path(model, store)
        logger.debug("Opening ICON kerchunk store %s for %s", path, model)
        fs = fsspec.filesystem(
            "reference", fo=str(path), remote_protocol="file", lazy=True,
        )
        # mask_and_scale=False: the stores declare a zarr-level
        # ``fill_value: 0.0`` alongside the real ``missing_value``, and
        # automatic masking would erase every genuine zero.  Masking is done
        # explicitly on the sentinel in ``_load_raw``.
        ds = xr.open_dataset(
            fs.get_mapper(""),
            engine="zarr",
            consolidated=False,
            chunks={},
            mask_and_scale=False,
        )
        self._store_cache[key] = ds
        return ds

    def _store_path(self, model: str, store: str) -> Path:
        """Resolve the parquet store path for *model*/*store*."""
        mc = self._config.model_configs.get(model)
        root = Path(mc.data_root) if (mc and mc.data_root) else self._root
        member = str(mc.member) if mc else "1"

        path = root / member / _STORE_FILES[store]
        if not path.exists():
            raise FileNotFoundError(
                f"ICON kerchunk store not found: {path}"
            )
        return path
