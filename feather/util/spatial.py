"""Spatial utilities: zonal means, global/regional means, regridding, HEALPix mesh."""

import logging
from functools import lru_cache

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


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
            # Weighted mean along last axis, excluding NaN data points
            not_nan = ~np.isnan(masked_data)
            # Zero out weights where data is NaN so denominator is correct
            effective_w = masked_w * not_nan  # broadcast: (n,) * (..., n)
            wsum = effective_w.sum(axis=-1)
            nonzero = wsum > 0
            if np.any(nonzero):
                result = np.nansum(masked_data * masked_w, axis=-1)
                # Avoid division by zero; leave NaN where entire band is NaN
                out[..., i] = np.where(nonzero, result / wsum, np.nan)

    coords = {**other_coords, "lat": ("lat", bin_centres)}
    dims = other_dims + ["lat"]
    return xr.DataArray(out, dims=dims, coords=coords, name=da.name)


def global_mean(da: xr.DataArray,
                area: xr.DataArray = None) -> xr.DataArray:
    """Area-weighted global mean for unstructured grids (e.g. HEALPix).

    For HEALPix data (equal-area cells), pass ``area=None`` to compute
    a simple mean — no area weighting is needed.

    Parameters
    ----------
    da : xr.DataArray
        Data array with a spatial dimension.
    area : xr.DataArray, optional
        Cell areas.  If *None*, computes a simple (unweighted) mean,
        which is correct for equal-area grids like HEALPix.

    Returns
    -------
    xr.DataArray
        Global mean (spatial dimension reduced).
    """
    spatial_dim = _get_spatial_dim(da)
    if area is not None:
        return da.weighted(area).mean(dim=spatial_dim)
    return da.mean(dim=spatial_dim)


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


def latlon_global_mean(da: xr.DataArray,
                       area: xr.DataArray | np.ndarray = None) -> xr.DataArray:
    """Area-weighted global mean for regular lat/lon grids.

    Uses proper cell areas computed via ``nereus.mesh_from_arrays()``
    (or pre-computed areas if provided).  Falls back to cosine-latitude
    weighting when nereus is not available.

    Parameters
    ----------
    da : xr.DataArray
        Data on a regular lat/lon grid.
    area : xr.DataArray or np.ndarray, optional
        Pre-computed 2-D cell areas with shape ``(nlat, nlon)``.
        Pass this when you already have areas (e.g. CMIP6 ``areacella``
        / ``areacello``).  If *None*, areas are computed from the grid
        coordinates.

    Returns
    -------
    xr.DataArray
        Global mean (spatial dimensions reduced).
    """
    lat_name, lon_name = _find_latlon_dims(da)

    if area is None:
        area = compute_latlon_areas(
            da[lat_name].values, da[lon_name].values,
        )

    # Wrap numpy array in a DataArray aligned to da's spatial dims
    if not isinstance(area, xr.DataArray):
        area = xr.DataArray(
            np.asarray(area),
            dims=(lat_name, lon_name),
            coords={
                lat_name: da[lat_name].values,
                lon_name: da[lon_name].values,
            },
        )

    return da.weighted(area).mean(dim=[lat_name, lon_name])


def compute_latlon_areas(
    lat: np.ndarray, lon: np.ndarray,
) -> np.ndarray:
    """Compute cell areas (m²) for a regular lat/lon grid.

    Uses ``nereus.mesh_from_arrays()`` for accurate area computation.
    Results are cached per unique grid shape + bounds so repeated calls
    with the same grid are free.

    Parameters
    ----------
    lat : array-like, 1-D
        Latitude values in degrees.
    lon : array-like, 1-D
        Longitude values in degrees.

    Returns
    -------
    np.ndarray, shape (nlat, nlon)
        Cell areas in m².
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Cache key: grid geometry (length + endpoints)
    cache_key = (
        len(lat), len(lon),
        float(lat[0]), float(lat[-1]),
        float(lon[0]), float(lon[-1]),
    )
    return _cached_latlon_areas(cache_key, tuple(lat), tuple(lon))


@lru_cache(maxsize=16)
def _cached_latlon_areas(cache_key, lat_tuple, lon_tuple):
    """LRU-cached helper — nereus mesh creation + reshape."""
    import warnings

    import nereus as nr

    lat = np.array(lat_tuple)
    lon = np.array(lon_tuple)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # meshgrid info
        mesh = nr.mesh_from_arrays(lon, lat)

    nlat, nlon = len(lat), len(lon)
    return mesh.area.values.reshape(nlat, nlon)


def _find_latlon_dims(da: xr.DataArray) -> tuple[str, str]:
    """Identify latitude and longitude dimension names in a DataArray.

    Checks common naming conventions and raises a clear error
    if neither can be found.

    Returns
    -------
    (lat_name, lon_name) : tuple[str, str]
    """
    lat_name = lon_name = None

    lat_candidates = {"lat", "latitude", "nav_lat", "y", "rlat", "nlat"}
    lon_candidates = {"lon", "longitude", "nav_lon", "x", "rlon", "nlon"}

    for dim in da.dims:
        low = dim.lower()
        if low in lat_candidates and lat_name is None:
            lat_name = dim
        elif low in lon_candidates and lon_name is None:
            lon_name = dim

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot identify lat/lon dimensions in {da.dims}. "
            f"Expected names like 'lat'/'lon' or 'latitude'/'longitude'."
        )

    return lat_name, lon_name


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


def spatial_ttest(field_a, field_b, weights=None):
    """Paired t-test across grid points between two 2D fields.

    Treats each grid point as a paired sample. Tests whether the
    mean difference (field_a - field_b) is significantly different from zero.

    When *weights* is provided (e.g., cell areas), uses a weighted paired
    t-test with Kish's effective sample size.  This makes the test
    consistent with area-weighted global mean bias and avoids
    overrepresenting polar grid cells on regular lat/lon grids.

    Parameters
    ----------
    field_a, field_b : array-like
        Two fields of the same shape (e.g., model and obs climatologies).
    weights : array-like, optional
        Per-grid-point weights (e.g., cell areas).  Must broadcast to
        the same shape as the fields.  If *None*, all grid points are
        weighted equally (falls back to ``scipy.stats.ttest_rel``).

    Returns
    -------
    (t_statistic, p_value) : tuple[float, float]
    """
    from scipy import stats

    a = np.asarray(field_a).ravel()
    b = np.asarray(field_b).ravel()

    if weights is not None:
        w = np.asarray(weights).ravel()
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    else:
        valid = np.isfinite(a) & np.isfinite(b)

    if valid.sum() < 3:
        return float("nan"), float("nan")

    av, bv = a[valid], b[valid]
    d = av - bv

    # Guard against zero-variance differences (identical fields)
    if np.std(d) == 0.0:
        if np.mean(d) == 0.0:
            return 0.0, 1.0
        return float("inf") if np.mean(d) > 0 else float("-inf"), 0.0

    if weights is None:
        t_stat, p_val = stats.ttest_rel(av, bv)
        return float(t_stat), float(p_val)

    # Weighted paired t-test
    wv = w[valid]
    wv = wv / wv.sum()                    # normalise to sum to 1
    d_mean = np.sum(wv * d)               # weighted mean difference
    v2 = np.sum(wv ** 2)                  # sum of squared weights
    # Weighted variance (Bessel-corrected for reliability weights)
    d_var = np.sum(wv * (d - d_mean) ** 2) / (1.0 - v2)
    n_eff = 1.0 / v2                      # Kish's effective sample size
    se = np.sqrt(d_var / n_eff)
    t_stat = d_mean / se
    df = n_eff - 1.0
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return float(t_stat), float(p_val)


def spatial_anova(*fields):
    """One-way ANOVA F-test across multiple 2D spatial fields.

    Each field is a group; grid points are the samples within each group.
    Uses the intersection of valid (non-NaN) grid points across all fields.
    Tests whether group means differ significantly.

    Returns (f_statistic, p_value).
    """
    from scipy import stats

    arrays = [np.asarray(f).ravel() for f in fields]
    if len(arrays) < 2:
        return float("nan"), float("nan")
    # Use intersection of valid grid points
    valid = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        valid &= np.isfinite(a)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    groups = [a[valid] for a in arrays]
    f_stat, p_val = stats.f_oneway(*groups)
    return float(f_stat), float(p_val)


def spatial_variance_ratio(field_model, field_obs, weights=None):
    """Variance ratio F-test between model and obs spatial fields.

    Computes F = Var(model) / Var(obs) across grid points, testing
    whether the model reproduces the observed spatial variability.

    - F = 1: model has same spatial variance as observations
    - F > 1: model has more spatial variability (too noisy)
    - F < 1: model has less spatial variability (too smooth)

    Uses population variance (ddof=0) since the fields represent the
    entire spatial domain, not a sample.  When *weights* is provided
    (e.g., cell areas), uses area-weighted variances with Kish's
    effective sample size for the degrees of freedom.

    Parameters
    ----------
    field_model, field_obs : array-like
        Two fields of the same shape on the same grid.
    weights : array-like, optional
        Per-grid-point weights (e.g., cell areas).

    Returns
    -------
    (f_statistic, p_value) : tuple[float, float]
        Two-sided p-value testing H0: Var(model) == Var(obs).
    """
    from scipy import stats

    m = np.asarray(field_model).ravel()
    o = np.asarray(field_obs).ravel()

    if weights is not None:
        w = np.asarray(weights).ravel()
        valid = np.isfinite(m) & np.isfinite(o) & np.isfinite(w) & (w > 0)
    else:
        valid = np.isfinite(m) & np.isfinite(o)

    if valid.sum() < 3:
        return float("nan"), float("nan")

    mv, ov = m[valid], o[valid]

    if weights is not None:
        wv = w[valid]
        wv = wv / wv.sum()
        v2 = np.sum(wv ** 2)
        n_eff = 1.0 / v2
        # Weighted population variances (ddof=0)
        m_mean = np.sum(wv * mv)
        o_mean = np.sum(wv * ov)
        var_m = np.sum(wv * (mv - m_mean) ** 2)
        var_o = np.sum(wv * (ov - o_mean) ** 2)
        df = n_eff - 1.0
    else:
        n = len(mv)
        var_m = np.var(mv, ddof=0)
        var_o = np.var(ov, ddof=0)
        df = n - 1.0

    if var_o == 0.0:
        if var_m == 0.0:
            return 1.0, 1.0
        return float("inf"), 0.0

    f_stat = var_m / var_o
    # Two-sided p-value: probability of F this extreme or more
    p_val = 2.0 * min(
        stats.f.cdf(f_stat, df, df),
        stats.f.sf(f_stat, df, df),
    )
    return float(f_stat), float(p_val)


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
