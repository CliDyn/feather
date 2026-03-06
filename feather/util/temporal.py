"""Temporal utilities: climatologies, anomalies, seasonal grouping, trends."""

import numpy as np
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


def linear_trend(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Per-grid-point linear trend in original units per year.

    Computes the OLS slope at each grid point by regressing values
    against time expressed in decimal years.

    Parameters
    ----------
    da : xr.DataArray
        Data with a time (datetime64) or numeric dimension.
        Must be materialized (not dask-backed) — call ``.compute()`` first.
    dim : str
        Name of the dimension to regress along.

    Returns
    -------
    xr.DataArray
        Linear trend (slope) at each grid point, in units per year.
        Multiply by 10 for per-decade trends.
    """
    coord = da[dim]

    # Convert time coordinate to decimal years
    if np.issubdtype(coord.dtype, np.datetime64):
        t0 = coord.values[0]
        x = (coord.values - t0) / np.timedelta64(1, "D") / 365.25
    else:
        # Numeric coordinate (e.g. year integers from groupby)
        x = coord.values.astype(np.float64)
        x = x - x[0]

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        # Cannot compute trend with fewer than 2 points
        other_dims = [d for d in da.dims if d != dim]
        other_coords = {k: da.coords[k] for k in da.coords if k != dim}
        shape = [da.sizes[d] for d in other_dims]
        return xr.DataArray(
            np.full(shape, np.nan) if shape else np.nan,
            dims=other_dims or None,
            coords=other_coords,
        )

    # Stack all non-dim dimensions into a flat array for vectorized regression
    other_dims = [d for d in da.dims if d != dim]
    if other_dims:
        stacked = da.stack(flat=other_dims)  # (dim, flat)
        y = np.asarray(stacked.values, dtype=np.float64)  # (n, m)
    else:
        y = np.asarray(da.values, dtype=np.float64).reshape(n, 1)

    # Vectorized OLS: slope = (N*sum(x*y) - sum(x)*sum(y)) / (N*sum(x²) - sum(x)²)
    sx = x.sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx

    # einsum for (n,) × (n, m) → (m,)
    sxy = np.einsum("i,ij->j", x, y)
    sy = y.sum(axis=0)

    slope = (n * sxy - sx * sy) / denom

    if other_dims:
        # Unstack back to original shape
        result = stacked.isel({dim: 0}).drop_vars(dim, errors="ignore").copy(
            data=slope,
        )
        return result.unstack("flat")
    else:
        return xr.DataArray(float(slope[0]))


def deseason(da: xr.DataArray,
             period: tuple[str, str] = None) -> xr.DataArray:
    """Remove monthly seasonal cycle, returning anomalies.

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.
    period : tuple of str, optional
        (start, end) for climatology computation.

    Returns
    -------
    xr.DataArray
        Deseasonalised anomalies.
    """
    clim = monthly_climatology(da, period=period)
    return anomaly(da, clim)


def detrend(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Remove per-grid-point linear trend, preserving variability.

    Parameters
    ----------
    da : xr.DataArray
        Data with a time or numeric dimension.
        Must be materialised (not dask-backed) — call ``.compute()`` first.
    dim : str
        Name of the dimension to detrend along.

    Returns
    -------
    xr.DataArray
        Data with the linear trend removed.
    """
    slope = linear_trend(da, dim=dim)
    coord = da[dim]
    if np.issubdtype(coord.dtype, np.datetime64):
        t0 = coord.values[0]
        t = (coord.values - t0) / np.timedelta64(1, "D") / 365.25
    else:
        t = coord.values.astype(np.float64)
        t = t - t[0]
    t_da = xr.DataArray(t, dims=[dim], coords={dim: da[dim]})
    return da - slope * t_da


_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def seasonal_annual_mean(
    da: xr.DataArray,
    season: str,
    period: tuple[str, str] | None = None,
) -> xr.DataArray:
    """Annual means for a specific season (DJF, MAM, JJA, SON).

    Filters to the season's months, groups by year, and computes
    per-year means.  For DJF, December is assigned to the following
    year (e.g. Dec 1990 → year 1991).

    Parameters
    ----------
    da : xr.DataArray
        Data with a ``time`` dimension.
    season : str
        One of ``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``.
    period : tuple of str, optional
        (start, end) for time slicing before season selection.

    Returns
    -------
    xr.DataArray
        Seasonal annual means with ``year`` coordinate (integer years).
    """
    if period is not None:
        da = da.sel(time=slice(period[0], period[1]))

    months = _SEASON_MONTHS[season.upper()]

    # Filter to season months
    month_vals = da["time.month"]
    mask = month_vals.isin(months)
    da_season = da.sel(time=mask)

    if len(da_season.time) == 0:
        return da_season

    # Assign year: for DJF, shift December to the following year
    years = da_season["time.year"].values.copy()
    if season.upper() == "DJF":
        month_arr = da_season["time.month"].values
        years[month_arr == 12] += 1

    # Group by year and mean
    year_coord = xr.DataArray(years, dims="time")
    grouped = da_season.groupby(year_coord).mean(dim="time")
    return grouped.rename({"group": "year"})
