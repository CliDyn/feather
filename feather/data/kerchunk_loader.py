"""Load model data from Parquet-format kerchunk reference stores.

Handles IFS-FESOM2 hist-1950 ensemble members (e.g. r2i1p1f1, r3i1p1f1)
whose data is stored as kerchunk parquet references to raw GRIB/FESOM files.

Store layout (auto-detected under ``{root}/{variant}/``)::

    atmos/gr025/2D_monthly_0.25deg_atmos_avg.parq   # 2-D sfc monthly
    atmos/gr025/3D_monthly_0.25deg_atmos_avg.parq   # 3-D pl monthly
    ocean/gr025/2D_daily_avg_*.parq                  # ocean 2-D daily

Grid conventions
----------------
* Atmos: flat ``(time, value=1038240)`` with 1-D ``lat``/``lon`` coordinate
  arrays (lat 90→-90 north-first, lon 0→179.75 then -180→-0.25).
  Reshaped to ``(time, lat=721, lon=1440)`` and sorted to standard
  ascending lat (-90→90) and 0→360 lon ordering.
* Ocean: proper ``(time, depth=1, lat=721, lon=1440)``.  The singleton
  depth dimension is squeezed and the daily time series is resampled to
  monthly means.

Requires ``kerchunk``, ``fastparquet``, ``fsspec``, and ``gribscan`` in the
Python environment.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from feather.config import FeatherConfig
from feather.data.variables import get_var

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable mapping: CMOR canonical name → (store, kerchunk_name, scale)
#
# store: "atmos2d" | "atmos3d" | "ocean2d"
# scale: multiply the raw values by this factor; use negative to flip sign
# ---------------------------------------------------------------------------

# Atmos 2-D (monthly) variables
_ATMOS2D: dict[str, tuple[str, float]] = {
    # Temperature
    "tas":     ("mean2t",   1.0),     # K
    "ts":      ("mskt",     1.0),     # K (skin temperature)
    # Pressure
    "psl":     ("mmsl",     1.0),     # Pa
    "ps":      ("msp",      1.0),     # Pa
    # Wind
    "uas":     ("m10u",     1.0),     # m/s
    "vas":     ("m10v",     1.0),     # m/s
    "sfcWind": ("mean10ws", 1.0),     # m/s
    # Precipitation/evaporation (m/s → kg/m²/s = ×1000)
    "pr":      ("tprate",   1000.0),
    "prc":     ("cprate",   1000.0),
    "evspsbl": ("erate",    1000.0),  # m of water equiv./s → kg/m²/s
    # Cloud cover (0-1 fraction → %)
    "clt":     ("meantcc",  100.0),
    # Surface heat fluxes — IFS positive-downward → CMOR positive-upward → ×(-1)
    "hfss":    ("msshf",   -1.0),
    "hfls":    ("mslhf",   -1.0),
    # Surface downwelling radiation (W/m², positive downward)
    "rsds":    ("msdwswrf", 1.0),
    "rlds":    ("msdwlwrf", 1.0),
    # Surface net radiation (W/m², positive downward = feather convention)
    "rss":     ("msnswrf",  1.0),     # feather name for surface net SW
    "rls":     ("msnlwrf",  1.0),     # feather name for surface net LW
    "rsscs":   ("msnswrfcs", 1.0),
    "rlscs":   ("msnlwrfcs", 1.0),
    # TOA radiation (W/m²)
    "rsdt":    ("mtdwswrf", 1.0),     # TOA incident SW
    "rst":     ("mtnswrf",  1.0),     # feather name for TOA net SW
    "rlt":     ("mtnlwrf",  1.0),     # feather name for TOA net LW
    "rstcs":   ("mtnswrfcs", 1.0),
    "rltcs":   ("mtnlwrfcs", 1.0),
    # Total-column water / clouds
    "prw":     ("mtcwv",    1.0),     # total column water vapour (kg/m²)
    "clwvi":   ("mtclw",    1.0),     # total column liquid water (kg/m²)
    "clivi":   ("mtciw",    1.0),     # total column ice water (kg/m²)
    # Wind stress (N/m²)
    "tauu":    ("metss",    1.0),
    "tauv":    ("mntss",    1.0),
}

# Radiation variables that must be derived from two atmos-2D fields.
# Each entry: CMOR name → (field_a, scale_a, field_b, scale_b)
# Result = scale_a * a  +  scale_b * b
_DERIVED: dict[str, tuple[str, float, str, float]] = {
    # rsut = rsdt - (net TOA SW) = mtdwswrf - mtnswrf
    "rsut":  ("rsdt",  1.0, "rst",  -1.0),
    # rlut = OLR = -(net TOA LW) = -mtnlwrf
    "rlut":  ("rlt",  -1.0, "rlt",   0.0),   # rlut = -rlt (b term zeroed)
    # rsus = rsds - rss = msdwswrf - msnswrf
    "rsus":  ("rsds",  1.0, "rss",  -1.0),
    # rlus = rlds - rls = msdwlwrf - msnlwrf
    "rlus":  ("rlds",  1.0, "rls",  -1.0),
}

# Atmos 3-D (pressure-level, monthly)
_ATMOS3D: dict[str, tuple[str, float]] = {
    "ta":   ("mt", 1.0),
    "hus":  ("mq", 1.0),
    "ua":   ("mu", 1.0),
    "va":   ("mv", 1.0),
    "zg":   ("mz", 1.0),
    "wap":  ("mw", 1.0),
}

# Ocean 2-D variables (daily → resampled to monthly in loader)
# Raw tos is in K; subtract 273.15 to match CMORLoader (°C).
_OCEAN2D: dict[str, tuple[str, float, float]] = {
    # (kerchunk_name, scale, offset)  → result = raw * scale + offset
    "tos":      ("avg_tos",      1.0, -273.15),   # K → °C
    "siconc":   ("avg_siconc",   1.0,  0.0),      # fraction 0-1
    "sithick":  ("avg_sithick",  1.0,  0.0),      # m
    "sisnthick":("avg_sisnthick",1.0,  0.0),      # m
    "sos":      ("avg_sos",      1.0,  0.0),      # g/kg ≈ PSU
    "zos":      ("avg_zos",      1.0,  0.0),      # m
    "mlotst":   ("avg_mlotst125",1.0,  0.0),      # m (125 m threshold)
}

# Ocean fill value sentinel (large negative float)
_OCEAN_FILL_THRESHOLD = -1e30
# Atmos fill value
_ATMOS_FILL_VALUE = 9999.0


class KerchunkParquetLoader:
    """Load model data from parquet-format kerchunk reference stores.

    Parameters
    ----------
    config : FeatherConfig
        Pipeline config.  Must provide ``data_source.root`` pointing at the
        base directory that contains ``{variant}/atmos/`` and
        ``{variant}/ocean/`` sub-trees.
    """

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("root", ""))
        # Cache opened xarray datasets keyed by (model, store_type)
        self._store_cache: dict[tuple, xr.Dataset] = {}
        # Cache loaded lat/lon grids keyed by model
        self._grid_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API (matches CMORLoader)
    # ------------------------------------------------------------------

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        """Load *variable* for *model*.

        Parameters
        ----------
        model, variable : str
            Model name and CMOR variable name.
        period : (start, end), optional
            Year strings for time slicing.
        time_mean : bool
            Return temporal mean if True.
        """
        da = self._load_raw(model, variable)

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(self, model: str, variable: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lon, lat)`` 1-D coordinate arrays for *model*."""
        try:
            vinfo = get_var(variable)
            use_ocean = vinfo.domain in ("o2d", "o3d")
        except KeyError:
            use_ocean = False

        if use_ocean:
            ds = self._open_store(model, "ocean2d")
            return np.asarray(ds["lon"]), np.asarray(ds["lat"])

        lon, lat = self._get_atmos_grid(model)
        return lon, lat

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _load_raw(self, model: str, variable: str) -> xr.DataArray:
        """Load without period/time_mean filtering."""
        if variable in _OCEAN2D:
            return self._load_ocean_var(model, variable)

        if variable in _ATMOS3D:
            return self._load_atmos3d_var(model, variable)

        if variable in _ATMOS2D:
            return self._load_atmos2d_var(model, variable)

        if variable in _DERIVED:
            return self._load_derived_var(model, variable)

        raise KeyError(
            f"Variable {variable!r} not supported by KerchunkParquetLoader. "
            f"Known: {sorted(_ATMOS2D) + sorted(_OCEAN2D) + sorted(_DERIVED)}"
        )

    # ------------------------------------------------------------------
    # Atmos 2-D
    # ------------------------------------------------------------------

    def _load_atmos2d_var(self, model: str, variable: str) -> xr.DataArray:
        kname, scale = _ATMOS2D[variable]
        ds = self._open_store(model, "atmos2d")
        raw = ds[kname]

        # Replace GRIB fill value (exact equality: 9999.0 is the GRIB sentinel,
        # never a real value even for Pa-unit fields like psl ~100 000 Pa)
        data = raw.values.copy().astype(np.float32)
        data[data == _ATMOS_FILL_VALUE] = np.nan
        if scale != 1.0:
            data *= np.float32(scale)

        time = ds["time"].values
        lat, lon, data_3d = self._reshape_atmos_flat(data, ds)
        da = xr.DataArray(
            data_3d,
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": lat, "lon": lon},
            name=variable,
            attrs={"units": raw.attrs.get("units", ""), "long_name": variable},
        )
        return da

    def _load_derived_var(self, model: str, variable: str) -> xr.DataArray:
        """Compute a CMOR radiation component from two base fields."""
        a_name, scale_a, b_name, scale_b = _DERIVED[variable]
        a = self._load_raw(model, a_name)
        if scale_b == 0.0:
            da = a * scale_a
        else:
            b = self._load_raw(model, b_name)
            da = a * scale_a + b * scale_b
        return da.rename(variable)

    # ------------------------------------------------------------------
    # Atmos 3-D
    # ------------------------------------------------------------------

    def _load_atmos3d_var(self, model: str, variable: str) -> xr.DataArray:
        kname, scale = _ATMOS3D[variable]
        ds = self._open_store(model, "atmos3d")
        raw = ds[kname]

        data = raw.values.copy().astype(np.float32)
        data[data >= _ATMOS_FILL_VALUE] = np.nan
        if scale != 1.0:
            data *= np.float32(scale)

        time = ds["time"].values
        levels = ds["level"].values
        lat, lon, data_4d = self._reshape_atmos_flat(data, ds, has_level=True)
        da = xr.DataArray(
            data_4d,
            dims=["time", "level", "lat", "lon"],
            coords={"time": time, "level": levels, "lat": lat, "lon": lon},
            name=variable,
            attrs={"units": raw.attrs.get("units", ""), "long_name": variable},
        )
        return da

    # ------------------------------------------------------------------
    # Ocean 2-D (daily → monthly)
    # ------------------------------------------------------------------

    def _load_ocean_var(self, model: str, variable: str) -> xr.DataArray:
        kname, scale, offset = _OCEAN2D[variable]
        ds = self._open_store(model, "ocean2d")
        raw = ds[kname]

        # Squeeze singleton depth/lev dimension
        data = raw.squeeze(drop=True)

        # Mask fill values
        fill = raw.attrs.get("missing_value", None)
        if fill is not None and np.isfinite(fill):
            data = data.where(data > _OCEAN_FILL_THRESHOLD)
        else:
            data = data.where(data > _OCEAN_FILL_THRESHOLD)

        if scale != 1.0:
            data = data * scale
        if offset != 0.0:
            # Guard against double-conversion when xarray's zarr backend has
            # already applied add_offset via CF decoding.  The netcdf backend
            # moves the attribute to raw.encoding after decode, but the zarr
            # backend does NOT — so we cannot rely on raw.encoding.  Instead
            # we use a value-based heuristic: for the K→°C offset (-273.15)
            # a single-timestep mean that is already < 100 means the data is
            # in °C (zarr decoded it), so we skip the explicit subtraction.
            already_decoded = raw.encoding.get("add_offset", None) == offset
            if not already_decoded and offset == -273.15:
                try:
                    sample = float(data.isel(time=0, drop=True).mean().compute().values)
                    if np.isfinite(sample) and sample < 100.0:
                        already_decoded = True
                        logger.debug(
                            "K→°C for %s skipped (value-based): sample=%.2f already °C",
                            variable, sample,
                        )
                except Exception as exc:
                    logger.debug(
                        "K→°C sample check failed for %s (%s); applying offset", variable, exc
                    )
            if already_decoded:
                logger.debug(
                    "Skipping explicit offset %.4f for %s — already applied by zarr decode",
                    offset, variable,
                )
            else:
                data = data + offset

        # Resample daily → monthly mean (label on month start)
        data = (
            data
            .resample(time="MS")
            .mean(skipna=True)
        )
        data = data.rename(variable)
        # After the K→°C offset is applied (either explicitly or via CF
        # decode), set units="degC" so _needs_celsius_conversion returns False
        # and avoids a spurious extra subtraction of 273.15 downstream.
        out_units = "degC" if offset == -273.15 else raw.attrs.get("units", "")
        data.attrs.update(
            units=out_units,
            long_name=raw.attrs.get("long_name", variable),
        )
        return data

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    def _open_store(self, model: str, store_type: str) -> xr.Dataset:
        """Open (and cache) the xarray dataset for *model*/*store_type*."""
        key = (model, store_type)
        if key in self._store_cache:
            return self._store_cache[key]

        import fsspec

        path = self._store_path(model, store_type)
        logger.debug("Opening kerchunk store %s for %s/%s", path, model, store_type)
        fs = fsspec.filesystem("reference", fo=str(path), remote_protocol="file", lazy=True)
        ds = xr.open_dataset(
            fs.get_mapper(""),
            engine="zarr",
            consolidated=False,
            chunks={},
            mask_and_scale=False,  # raw values only; we apply offsets explicitly
        )
        self._store_cache[key] = ds
        return ds

    def _store_path(self, model: str, store_type: str) -> Path:
        """Resolve the parquet file path for *model* and *store_type*."""
        mc = self._config.model_configs.get(model)
        variant = mc.variant if mc and mc.variant else "r1i1p1f1"
        # Per-model data_root overrides global root (needed in composite configs
        # where the global root is the CMOR tree, not the kerchunk base).
        root = Path(mc.data_root) if (mc and mc.data_root) else self._root
        base = root / variant

        if store_type == "atmos2d":
            p = base / "atmos" / "gr025" / "2D_monthly_0.25deg_atmos_avg.parq"
        elif store_type == "atmos3d":
            p = base / "atmos" / "gr025" / "3D_monthly_0.25deg_atmos_avg.parq"
        elif store_type == "ocean2d":
            ocean_dir = base / "ocean" / "gr025"
            matches = sorted(ocean_dir.glob("2D_daily_avg*.parq"))
            if not matches:
                raise FileNotFoundError(
                    f"No ocean daily parquet found in {ocean_dir}"
                )
            p = matches[0]
        else:
            raise ValueError(f"Unknown store_type: {store_type!r}")

        if not p.exists():
            raise FileNotFoundError(f"Kerchunk store not found: {p}")
        return p

    # ------------------------------------------------------------------
    # Grid reshaping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_grid_shape(lat_flat: np.ndarray) -> tuple[int, int]:
        """Infer (n_lat, n_lon) from the flat latitude coordinate array.

        The IFS atmos grid is stored row-major (north-first): the same
        latitude value repeats n_lon times before the next row begins.
        We detect n_lon as the first index where the lat value changes,
        then derive n_lat from total cell count.
        """
        n_total = len(lat_flat)
        first_lat = lat_flat[0]
        change_idx = np.argmax(lat_flat != first_lat)
        # argmax returns 0 if no change found (all same lat) — guard against that
        n_lon = int(change_idx) if change_idx > 0 else n_total
        n_lat = n_total // n_lon
        return n_lat, n_lon

    def _get_atmos_grid(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon_1d, lat_1d) for the atmos grid of *model*."""
        if model in self._grid_cache:
            return self._grid_cache[model]

        ds = self._open_store(model, "atmos2d")
        lat_flat = ds["lat"].values
        lon_flat = ds["lon"].values

        n_lat, n_lon = self._detect_grid_shape(lat_flat)
        lat_2d = lat_flat.reshape(n_lat, n_lon)
        lon_2d = lon_flat.reshape(n_lat, n_lon)
        lat_1d = lat_2d[:, 0]
        lon_1d = lon_2d[0, :]

        # Normalise lon to 0–360
        lon_1d = np.where(lon_1d < 0, lon_1d + 360.0, lon_1d)

        # Sort lat ascending (-90 → 90)
        lat_sort = np.argsort(lat_1d)
        lat_1d = lat_1d[lat_sort]
        # Sort lon ascending (0 → 359.75)
        lon_sort = np.argsort(lon_1d)
        lon_1d = lon_1d[lon_sort]

        self._grid_cache[model] = (lon_1d, lat_1d)
        return lon_1d, lat_1d

    def _reshape_atmos_flat(
        self,
        data: np.ndarray,
        ds: xr.Dataset,
        has_level: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape flat spatial dim to proper (lat, lon) grid.

        Returns ``(lat_1d, lon_1d, data_nd)`` where the spatial axes are
        sorted to ascending lat (-90→90) and lon (0→360).

        Parameters
        ----------
        data : np.ndarray
            Raw array from the store.
            Shape ``(time, value)`` for 2-D or ``(time, level, value)`` for 3-D.
        has_level : bool
            True for 3-D pressure-level data.
        """
        lat_flat = ds["lat"].values
        lon_flat = ds["lon"].values
        n_lat, n_lon = self._detect_grid_shape(lat_flat)

        lat_2d = lat_flat.reshape(n_lat, n_lon)
        lon_2d = lon_flat.reshape(n_lat, n_lon)
        lat_1d = lat_2d[:, 0]
        lon_1d = lon_2d[0, :]
        lon_1d = np.where(lon_1d < 0, lon_1d + 360.0, lon_1d)

        # Argsort for ascending lat and lon
        lat_sort = np.argsort(lat_1d)
        lon_sort = np.argsort(lon_1d)
        lat_1d = lat_1d[lat_sort]
        lon_1d = lon_1d[lon_sort]

        if has_level:
            n_time, n_lev, _ = data.shape
            data = data.reshape(n_time, n_lev, n_lat, n_lon)
            data = data[:, :, lat_sort, :][:, :, :, lon_sort]
        else:
            n_time, _ = data.shape
            data = data.reshape(n_time, n_lat, n_lon)
            data = data[:, lat_sort, :][:, :, lon_sort]

        return lat_1d, lon_1d, data
