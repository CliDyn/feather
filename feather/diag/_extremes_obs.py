"""Configurable observation reference for extreme-index diagnostics.

The tropical-nights and heatwave diagnostics (and their climate-change
variants) compare model daily extremes against a monthly-mean observational
reference.  By default that reference is Berkeley Earth Land TMIN/TMAX.  When
the config sets ``project.extremes_obs_reference: "ERA5"`` and an
``ERA5_TMINMAX`` dataset is present in ``obs_datasets`` (the derived ERA5
monthly tasmin/tasmax produced by ``scripts/era5_derive_tasminmax.sh``), this
module supplies the ERA5 monthly-mean field instead.

ERA5 monthly tasmin/tasmax are absolute temperatures (K) on a regular 0.25°
lat/lon grid, so the period mean is a plain time average — no anomaly +
climatology reconstruction is needed (unlike the Berkeley gridded format).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: obs_datasets key holding the derived ERA5 monthly tasmin/tasmax files.
ERA5_DATASET = "ERA5_TMINMAX"


def use_era5_obs(config) -> bool:
    """True when the config requests ERA5 as the extremes obs reference."""
    ref = str(config.project.get("extremes_obs_reference", "")).upper()
    return ref == "ERA5" and ERA5_DATASET in config.obs_datasets


def obs_ref_label(config, var: str) -> str:
    """Human-readable label for the active obs reference.

    ``var`` is the canonical variable (``"tasmin"`` or ``"tasmax"``).
    """
    if use_era5_obs(config):
        return "ERA5"
    return "Berkeley Earth Land " + ("TMIN" if var == "tasmin" else "TMAX")


def obs_ref_model_name(config) -> str | None:
    """Name of the model that doubles as the obs reference, or ``None``.

    When ERA5 is the configured extremes obs reference it is also added as a
    flat "model" (reading its derived daily CMOR tree) so it participates in
    the climatology/time-series figures.  In the mean-Tmin/Tmax **bias** maps
    that model must be excluded: ``ERA5(model) − ERA5(obs)`` is ~zero
    everywhere by construction and only adds a misleading blank panel.  Returns
    ``None`` for the Berkeley reference (which is never itself a model).
    """
    return "ERA5" if use_era5_obs(config) else None


def era5_mean_available(config, var: str) -> bool:
    """True when an ERA5 monthly file for ``var`` is configured and exists."""
    if not use_era5_obs(config):
        return False
    ds_cfg = config.obs_datasets.get(ERA5_DATASET, {})
    fname = ds_cfg.get("variables", {}).get(var)
    if fname is None:
        return False
    return (Path(ds_cfg["path"]) / fname).exists()


def load_era5_mean(
    config, var: str, period, model_lat: np.ndarray, model_lon: np.ndarray,
) -> xr.DataArray | None:
    """Period-mean ERA5 monthly ``var`` (K), interpolated to the model grid.

    Returns ``None`` if the dataset/file is unavailable so callers can fall
    back to the Berkeley reference.
    """
    ds_cfg = config.obs_datasets.get(ERA5_DATASET)
    if ds_cfg is None:
        return None
    fname = ds_cfg.get("variables", {}).get(var)
    if fname is None:
        return None
    path = Path(ds_cfg["path"]) / fname
    try:
        ds = xr.open_dataset(path, chunks="auto")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("  Could not open ERA5 %s obs (%s): %s", var, path, exc)
        return None

    da = ds[var]
    if "time" in da.dims:
        da = da.sel(time=slice(period[0], period[1])).mean("time")
    if hasattr(da, "compute"):
        da = da.compute()

    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")

    return da.interp(
        lat=xr.DataArray(model_lat, dims="lat"),
        lon=xr.DataArray(model_lon, dims="lon"),
        method="linear", kwargs={"fill_value": np.nan},
    )


#: obs_datasets key for the derived ERA5 land-sea mask (sftlf, %).
ERA5_SFTLF = "ERA5_SFTLF"


def _open_era5_sftlf(config) -> xr.DataArray | None:
    """Open the configured ERA5 sftlf field (land area %, lat/lon), or None."""
    ds_cfg = config.obs_datasets.get(ERA5_SFTLF)
    if ds_cfg is None:
        return None
    variables = ds_cfg.get("variables", {})
    fname = variables.get("sftlf") or next(iter(variables.values()), None)
    if fname is None:
        return None
    try:
        ds = xr.open_dataset(Path(ds_cfg["path"]) / fname)
    except (FileNotFoundError, OSError):
        return None
    da = ds["sftlf"].squeeze(drop=True)
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")
    return da


def era5_approx_exceedance_series(
    config, var: str, start: str, end: str, threshold_c: float,
) -> xr.DataArray | None:
    """Approximate annual count of exceedance days from ERA5 monthly means.

    Mirrors the Berkeley "approximate TN from monthly" reference: for each
    month, if the monthly-mean value exceeds ``threshold_c`` (°C) the whole
    month contributes its days; summed per calendar year and reduced to an
    area-weighted **land** mean.  Land masking uses the configured ERA5
    ``sftlf`` (``ERA5_SFTLF``); without it the series is omitted (returns
    ``None``) rather than mixing in ocean points.
    """
    from feather.util.spatial import latlon_global_mean

    ds_cfg = config.obs_datasets.get(ERA5_DATASET)
    if ds_cfg is None:
        return None
    fname = ds_cfg.get("variables", {}).get(var)
    if fname is None:
        return None
    try:
        ds = xr.open_dataset(Path(ds_cfg["path"]) / fname, chunks="auto")
    except (FileNotFoundError, OSError):
        return None

    da = ds[var]
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")

    da = da.sel(time=slice(start, end)) - 273.15  # K → °C
    days = xr.DataArray(da.time.dt.days_in_month.values, dims=["time"],
                        coords={"time": da.time})
    monthly = xr.where(da > threshold_c, days, 0.0)

    mask = _open_era5_sftlf(config)
    if mask is None:
        logger.warning(
            "ERA5_SFTLF not configured — skipping ERA5 approximate %s series",
            var)
        return None
    mask = mask.interp(lat=da.lat, lon=da.lon, method="nearest",
                       kwargs={"fill_value": 0.0}) > 50.0
    monthly = monthly.where(mask)

    annual = monthly.groupby("time.year").sum("time")
    return latlon_global_mean(annual).compute()
