"""Shared Berkeley Earth high-resolution loading and land-masking helpers.

The Berkeley Earth gridded products (Global TAVG HR, Land TMAX/TMIN) store
monthly *anomalies* in °C relative to 1951–1980 plus a separate 12-month
``climatology`` field, so the absolute temperature is reconstructed as
``anomaly + climatology[month_of_year]``.  Time is encoded as decimal years.

Crucially, the Land+Ocean product reports **sea-surface temperature over the
ocean**, not 2 m air temperature.  Comparing it cell-by-cell against model
``tas`` is therefore not like-for-like — the mismatch is largest over
sea ice and western boundary currents.  The gridded files carry a
``land_mask`` variable (land area fraction, 0–1); the diagnostics use it to
restrict the comparison to land, which is the domain the Berkeley Earth
station network actually constrains.

Both :mod:`feather.diag.temperature_berkeley` and
:mod:`feather.diag.added_value` share these helpers so the obs field, the
land mask and the masking threshold stay identical between the two.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Default land-area-fraction threshold above which a cell counts as land.
DEFAULT_LAND_THRESHOLD = 0.5

#: ``sftlf`` is stored as a percentage; a cell is land above this value.
_SFTLF_LAND_PERCENT = 50.0


# ── Config knobs ──────────────────────────────────────────────────────


def land_only_enabled(config: Any) -> bool:
    """Whether Berkeley Earth comparisons are restricted to land.

    Controlled by ``project.berkeley_land_only`` and defaulting to *True*:
    the Land+Ocean product's ocean values are SST, so a global comparison
    against model ``tas`` is not like-for-like.
    """
    return bool(getattr(config, "project", {}).get("berkeley_land_only", True))


def land_threshold(config: Any) -> float:
    """Land-area-fraction cut-off (``project.berkeley_land_threshold``)."""
    return float(
        getattr(config, "project", {}).get(
            "berkeley_land_threshold", DEFAULT_LAND_THRESHOLD,
        )
    )


# ── File access ───────────────────────────────────────────────────────


def _dataset_path(config: Any, dataset_key: str) -> Path:
    """Resolve the on-disk path of a configured Berkeley Earth file."""
    ds_cfg = config.obs_datasets[dataset_key]
    variables = ds_cfg.get("variables", {})
    filename = variables.get("temperature")
    if filename is None:
        filename = next(iter(variables.values()))
    return Path(ds_cfg["path"]) / filename


def _decimal_year_to_datetime(dec_years: np.ndarray):
    """Convert Berkeley decimal-year time values to month-start datetimes."""
    import pandas as pd

    years = np.asarray(dec_years).astype(int)
    months = np.floor((np.asarray(dec_years) - years) * 12).astype(int) + 1
    months = np.clip(months, 1, 12)
    return pd.to_datetime(
        [f"{y:04d}-{m:02d}-01" for y, m in zip(years, months)]
    )


def _to_lat_lon_0_360(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude → lat/lon and shift lons to 0..360."""
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    if "lon" in da.coords and float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")
    return da


def load_berkeley_hr(
    config: Any,
    dataset_key: str = "BERKELEY_EARTH_HR",
    period: tuple[str, str] | None = None,
    *,
    land_only: bool = False,
    threshold: float = DEFAULT_LAND_THRESHOLD,
) -> xr.DataArray:
    """Load a Berkeley Earth gridded field as absolute temperature in K.

    Parameters
    ----------
    config : FeatherConfig
        Provides ``obs_datasets``.
    dataset_key : str
        Key in ``obs_datasets`` (e.g. ``BERKELEY_EARTH_HR``).
    period : tuple of str, optional
        ``(start, end)`` time slice.
    land_only : bool
        Mask cells whose land area fraction is below *threshold*.  Requires
        a ``land_mask`` variable in the file; a warning is logged and the
        field returned unmasked when it is absent.
    threshold : float
        Land-area-fraction cut-off.

    Returns
    -------
    xr.DataArray
        Absolute temperature in K, dims ``lat``/``lon``, lons in 0..360.
    """
    ds_full = xr.open_dataset(_dataset_path(config, dataset_key), chunks="auto")
    ds_full = ds_full.assign_coords(
        time=_decimal_year_to_datetime(ds_full["time"].values),
    )

    anom = ds_full["temperature"]
    if period is not None:
        anom = anom.sel(time=slice(period[0], period[1]))
    clim = ds_full["climatology"]  # (month_number, lat, lon)

    month_idx = anom.time.dt.month.values - 1
    clim_matched = clim.values[month_idx]
    abs_temp = anom + xr.DataArray(
        clim_matched, dims=anom.dims, coords=anom.coords,
    )

    if land_only:
        if "land_mask" in ds_full:
            abs_temp = abs_temp.where(ds_full["land_mask"] >= threshold)
        else:
            logger.warning(
                "%s has no land_mask variable -- returning unmasked field",
                dataset_key,
            )

    abs_temp = _to_lat_lon_0_360(abs_temp)
    return abs_temp + 273.15  # °C → K


def load_land_fraction(
    config: Any, dataset_key: str = "BERKELEY_EARTH_HR",
) -> xr.DataArray | None:
    """Berkeley Earth land area fraction (lat/lon, lons 0..360), or None.

    Returns *None* — rather than raising — when the dataset is not
    configured or carries no ``land_mask``, so callers can degrade to an
    unmasked comparison.
    """
    if dataset_key not in getattr(config, "obs_datasets", {}):
        return None
    try:
        ds = xr.open_dataset(_dataset_path(config, dataset_key))
    except (OSError, KeyError, StopIteration) as exc:
        logger.warning("Could not open %s for land mask: %s", dataset_key, exc)
        return None
    if "land_mask" not in ds:
        return None
    return _to_lat_lon_0_360(ds["land_mask"])


# ── Land masks on arbitrary grids ─────────────────────────────────────


def _spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Spatial dims of *da* (everything except time-like dims)."""
    return tuple(d for d in da.dims if d not in ("time", "month", "season"))


def land_fraction_on_grid(
    frac: xr.DataArray,
    lon: np.ndarray,
    lat: np.ndarray,
    grid_type: str,
) -> np.ndarray:
    """Sample a lat/lon land fraction onto an arbitrary target grid.

    HEALPix targets are sampled point-wise (``lon``/``lat`` are per-cell
    arrays); lat/lon targets are sampled on the axis outer product.  Nearest
    neighbour throughout — a land mask must stay binary-ish, and bilinear
    would smear coastlines.
    """
    lon_t = np.asarray(lon).ravel() % 360.0
    lat_t = np.asarray(lat).ravel()

    if grid_type == "healpix":
        sampled = frac.sel(
            lat=xr.DataArray(lat_t, dims="points"),
            lon=xr.DataArray(lon_t, dims="points"),
            method="nearest",
        )
        return np.asarray(sampled.values)

    sampled = frac.interp(
        lat=xr.DataArray(lat_t, dims="lat"),
        lon=xr.DataArray(lon_t, dims="lon"),
        method="nearest",
        kwargs={"fill_value": 0.0},
    )
    return np.asarray(sampled.values)


def _model_sftlf_mask(
    model_loader: Any,
    model: str,
    da: xr.DataArray,
) -> xr.DataArray | None:
    """Boolean land mask from the model's own ``sftlf`` (fx, %), or None.

    Returns *None* whenever ``sftlf`` is unavailable or does not line up
    with *da*'s spatial grid, so the caller can fall back to the Berkeley
    land mask.
    """
    try:
        sftlf = model_loader.load_var(model, "sftlf", table="fx")
    except (KeyError, FileNotFoundError, OSError, AttributeError,
            ValueError, TypeError):
        return None
    if sftlf is None:
        return None

    try:
        sftlf = sftlf.squeeze(drop=True)
        for d in ("time", "height", "depth"):
            if d in sftlf.dims:
                sftlf = sftlf.isel({d: 0})
        sftlf = _to_lat_lon_0_360(sftlf)

        dims = _spatial_dims(da)
        if len(dims) == 2 and {"lat", "lon"} <= set(sftlf.dims):
            # Regular lat/lon model: sample sftlf onto the model's own axes.
            lat_name, lon_name = dims
            sftlf = sftlf.interp(
                lat=xr.DataArray(
                    np.asarray(da[lat_name].values), dims="lat"),
                lon=xr.DataArray(
                    np.asarray(da[lon_name].values) % 360.0, dims="lon"),
                method="nearest", kwargs={"fill_value": 0.0},
            )
        # Unstructured (HEALPix) grids: sftlf must already share the grid.
        if sftlf.shape != tuple(da.sizes[d] for d in dims):
            return None
        return xr.DataArray(
            np.asarray(sftlf.values) > _SFTLF_LAND_PERCENT,
            dims=dims,
            coords={d: da[d] for d in dims if d in da.coords},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("  %s: could not use sftlf land mask: %s", model, exc)
        return None


def model_land_mask(
    model_loader: Any,
    model: str,
    da: xr.DataArray,
    lon: np.ndarray,
    lat: np.ndarray,
    grid_type: str,
    frac: xr.DataArray | None,
    threshold: float = DEFAULT_LAND_THRESHOLD,
) -> xr.DataArray | None:
    """Boolean land mask (True = land) on *da*'s native spatial grid.

    Prefers the model's own ``sftlf`` (land area fraction, %) so each model
    is masked by its own land-sea definition; falls back to the Berkeley
    Earth ``land_mask`` sampled onto the model grid when ``sftlf`` is not
    published (the usual case for the EERIE members).  Returns *None* when
    neither source is available.
    """
    mask = _model_sftlf_mask(model_loader, model, da)
    if mask is not None:
        return mask
    if frac is None:
        return None

    dims = _spatial_dims(da)
    try:
        sampled = land_fraction_on_grid(frac, lon, lat, grid_type)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("  %s: could not sample Berkeley land mask: %s", model, exc)
        return None

    expected = tuple(da.sizes[d] for d in dims)
    if sampled.shape != expected:
        # lat/lon sampling returns (nlat, nlon); reshape only if compatible.
        if sampled.size != int(np.prod(expected)):
            logger.warning(
                "  %s: land mask shape %s does not match grid %s -- skipping",
                model, sampled.shape, expected,
            )
            return None
        sampled = sampled.reshape(expected)

    return xr.DataArray(
        sampled >= threshold,
        dims=dims,
        coords={d: da[d] for d in dims if d in da.coords},
    )


def mask_latlon_to_land(
    da: xr.DataArray,
    frac: xr.DataArray | None,
    threshold: float = DEFAULT_LAND_THRESHOLD,
) -> xr.DataArray:
    """Mask a regular lat/lon field (e.g. ERA5) to land, if a mask exists."""
    if frac is None:
        return da
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    if lat_name not in da.dims or lon_name not in da.dims:
        return da
    try:
        frac_i = frac.interp(
            lat=xr.DataArray(np.asarray(da[lat_name].values), dims="lat"),
            lon=xr.DataArray(
                np.asarray(da[lon_name].values) % 360.0, dims="lon"),
            method="nearest", kwargs={"fill_value": 0.0},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not interpolate land mask: %s", exc)
        return da
    mask = xr.DataArray(
        np.asarray(frac_i.values) >= threshold,
        dims=(lat_name, lon_name),
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
    )
    return da.where(mask)
