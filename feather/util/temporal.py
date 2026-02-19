"""Temporal utilities: climatologies, anomalies, seasonal grouping."""

import xarray as xr


def climatology(da: xr.DataArray,
                period: tuple[str, str] = None) -> xr.DataArray:
    """Time-mean climatology, optionally restricted to a period.

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.
    period : tuple of str, optional
        (start, end) for time slicing, e.g. ("1990", "2014").

    Returns
    -------
    xr.DataArray
        Time-mean field.
    """
    if period is not None:
        da = da.sel(time=slice(period[0], period[1]))
    return da.mean(dim="time")


def seasonal_climatology(da: xr.DataArray,
                         period: tuple[str, str] = None) -> xr.Dataset:
    """Seasonal climatologies (DJF, MAM, JJA, SON).

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.
    period : tuple of str, optional
        (start, end) for time slicing.

    Returns
    -------
    xr.Dataset
        Dataset with one DataArray per season.
    """
    if period is not None:
        da = da.sel(time=slice(period[0], period[1]))

    seasonal = da.groupby("time.season").mean(dim="time")
    result = xr.Dataset()
    for season in ["DJF", "MAM", "JJA", "SON"]:
        if season in seasonal.season.values:
            result[season] = seasonal.sel(season=season)
    return result


def monthly_climatology(da: xr.DataArray,
                        period: tuple[str, str] = None) -> xr.DataArray:
    """Monthly climatological cycle (Jan-Dec means).

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.
    period : tuple of str, optional
        (start, end) for time slicing.

    Returns
    -------
    xr.DataArray
        Climatological cycle with ``month`` coordinate (1-12).
    """
    if period is not None:
        da = da.sel(time=slice(period[0], period[1]))
    return da.groupby("time.month").mean(dim="time")


def anomaly(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Compute anomaly = da - climatology.

    Parameters
    ----------
    da : xr.DataArray
        Full time series.
    clim : xr.DataArray
        Climatological mean (no time dimension) or monthly climatology
        (with ``month`` coordinate).

    Returns
    -------
    xr.DataArray
        Anomaly field.
    """
    if "month" in clim.dims:
        return da.groupby("time.month") - clim
    return da - clim


def annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Annual mean via resample.

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.

    Returns
    -------
    xr.DataArray
        Annual means.
    """
    return da.resample(time="YE").mean()
