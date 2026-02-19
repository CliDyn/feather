"""Spatial utilities: zonal means, global/regional means, HEALPix mesh."""

import numpy as np
import xarray as xr


def zonal_mean(da: xr.DataArray, lat: xr.DataArray,
               lat_bins: np.ndarray = None,
               weights: xr.DataArray = None) -> xr.DataArray:
    """Zonal mean by latitude-band binning on native HEALPix grid.

    Bins pixels into latitude bands and computes (weighted) mean per band.

    Parameters
    ----------
    da : xr.DataArray
        Data on the HEALPix grid (dim: ``values`` or last dim).
    lat : xr.DataArray
        Latitude of each HEALPix pixel (same length as spatial dim).
    lat_bins : np.ndarray, optional
        Bin edges in degrees. Default: 1-degree bins from -90 to 90.
    weights : xr.DataArray, optional
        Per-pixel weights (e.g., cell area). If None, uniform weights.

    Returns
    -------
    xr.DataArray
        Zonal mean with coordinate ``lat`` (bin centres).
    """
    if lat_bins is None:
        lat_bins = np.arange(-90, 91, 1.0)

    bin_centres = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lat_vals = np.asarray(lat)

    # Determine which spatial dimension to bin over
    spatial_dim = _get_spatial_dim(da)

    # Assign each pixel to a bin
    bin_idx = np.digitize(lat_vals, lat_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(bin_centres))

    # Build output shape: replace spatial dim with lat bins
    other_dims = [d for d in da.dims if d != spatial_dim]
    other_coords = {d: da.coords[d] for d in other_dims if d in da.coords}

    result_shape = [da.sizes[d] for d in other_dims] + [len(bin_centres)]

    # Compute binned means
    data = np.asarray(da)
    spatial_axis = list(da.dims).index(spatial_dim)

    # Move spatial axis to last position for easier indexing
    data = np.moveaxis(data, spatial_axis, -1)

    if weights is not None:
        w = np.asarray(weights)
    else:
        w = np.ones(data.shape[-1])

    out = np.full(data.shape[:-1] + (len(bin_centres),), np.nan)

    for i in range(len(bin_centres)):
        mask = valid & (bin_idx == i)
        if mask.any():
            masked_data = data[..., mask]
            masked_w = w[mask]
            # Weighted mean along last axis
            wsum = np.nansum(masked_w)
            if wsum > 0:
                out[..., i] = np.nansum(masked_data * masked_w, axis=-1) / wsum

    coords = {**other_coords, "lat": ("lat", bin_centres)}
    dims = other_dims + ["lat"]
    return xr.DataArray(out, dims=dims, coords=coords, name=da.name)


def global_mean(da: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """Area-weighted global mean.

    Parameters
    ----------
    da : xr.DataArray
        Data array with a spatial dimension.
    area : xr.DataArray
        Cell areas with the same spatial dimension.

    Returns
    -------
    xr.DataArray
        Global mean (spatial dimension reduced).
    """
    spatial_dim = _get_spatial_dim(da)
    weighted = da.weighted(area)
    return weighted.mean(dim=spatial_dim)


def regional_mean(da: xr.DataArray, area: xr.DataArray,
                  region: str = None,
                  bbox: tuple = None) -> xr.DataArray:
    """Area-weighted regional mean.

    Parameters
    ----------
    da : xr.DataArray
        Data array.
    area : xr.DataArray
        Cell areas.
    region : str, optional
        Named region (for future use with nereus.get_region_mask).
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) bounding box.

    Returns
    -------
    xr.DataArray
        Regional mean.
    """
    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        # Assume lat/lon are coordinates or can be extracted
        lat = da.latitude if "latitude" in da.coords else da.lat
        lon = da.longitude if "longitude" in da.coords else da.lon
        mask = ((lat >= lat_min) & (lat <= lat_max) &
                (lon >= lon_min) & (lon <= lon_max))
        da = da.where(mask)
        area = area.where(mask)

    return global_mean(da, area)


def healpix_mesh(ncells: int) -> xr.Dataset:
    """Get HEALPix mesh via nereus.

    Parameters
    ----------
    ncells : int
        Number of HEALPix cells (e.g. 12582912 for nside=1024).

    Returns
    -------
    xr.Dataset
        Dataset with longitude, latitude, area variables.
    """
    import nereus as nr

    return nr.healpix.load_mesh(ncells)


def _get_spatial_dim(da: xr.DataArray) -> str:
    """Identify the spatial dimension name in a DataArray."""
    for candidate in ("values", "ncells", "cell", "npix"):
        if candidate in da.dims:
            return candidate
    # Fall back to last dim if it's not time or depth
    for dim in reversed(da.dims):
        if dim not in ("time", "depth", "level", "month", "lat", "lon"):
            return dim
    raise ValueError(f"Cannot identify spatial dimension in {da.dims}")
