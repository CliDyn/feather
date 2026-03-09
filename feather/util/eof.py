"""EOF (Empirical Orthogonal Function) computation via SVD.

Provides area-weighted EOF decomposition of 2D+time DataArrays
for climate variability mode analysis (NAO, SAM, AO, PDO, etc.).
"""

import numpy as np
import xarray as xr


def compute_eof(da, n_modes=3, weight="cos_lat"):
    """Compute leading EOFs of a 2D+time DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Anomalies with dims ``(time, lat, lon)`` — must be numpy-backed
        (call ``.compute()`` first if dask-backed). Should be
        deseasonalised and optionally detrended.
    n_modes : int
        Number of leading EOFs to return.
    weight : str
        Area weighting: ``"cos_lat"`` (sqrt cos-lat) or ``"none"``.

    Returns
    -------
    eofs : xr.DataArray
        Spatial patterns with dims ``(mode, lat, lon)``. Normalised so
        that the pattern represents the field anomaly (in original units)
        associated with one standard deviation of the PC.
    pcs : xr.DataArray
        Principal component time series with dims ``(time, mode)``.
        Normalised to unit variance.
    variance_explained : np.ndarray
        Fraction of total variance explained by each mode (length *n_modes*).
    """
    lat_name = _find_lat(da)
    lon_name = _find_lon(da)

    lats = da[lat_name].values
    nt = da.sizes["time"]
    nlat = len(lats)
    nlon = da.sizes[lon_name]
    nspace = nlat * nlon

    # Area weights (sqrt cos-lat)
    if weight == "cos_lat":
        cos_w = np.sqrt(np.maximum(np.cos(np.deg2rad(lats)), 0.0))
        # Broadcast to (lat, lon) then flatten
        w = np.repeat(cos_w, nlon)
    else:
        w = np.ones(nspace)

    # Reshape to (time, space)
    data = np.asarray(da.values).reshape(nt, nspace)

    # Handle NaNs: mask columns that have any NaN
    valid = np.all(np.isfinite(data), axis=0) & (w > 0)
    data_valid = data[:, valid]
    w_valid = w[valid]

    # Centre in time (subtract temporal mean)
    mean = data_valid.mean(axis=0)
    data_centred = data_valid - mean

    # Apply area weights
    data_weighted = data_centred * w_valid[np.newaxis, :]

    # SVD: data_weighted = U @ diag(S) @ Vt
    n_svd = min(n_modes, min(data_weighted.shape) - 1)
    if n_svd < 1:
        raise ValueError(
            f"Not enough data for EOF: shape {data_weighted.shape}, "
            f"requested {n_modes} modes"
        )

    U, S, Vt = np.linalg.svd(data_weighted, full_matrices=False)

    # Truncate to requested modes
    U = U[:, :n_svd]
    S = S[:n_svd]
    Vt = Vt[:n_svd, :]

    # Variance explained
    total_var = np.sum(data_weighted ** 2)
    var_explained = (S ** 2) / total_var

    # PCs: normalise to unit variance
    pcs_raw = U * S[np.newaxis, :]  # (time, mode) — unnormalised
    pc_std = pcs_raw.std(axis=0)
    pc_std[pc_std == 0] = 1.0
    pcs_norm = pcs_raw / pc_std  # unit variance

    # EOF patterns: un-weight and scale by PC std
    # Vt rows are weighted patterns; divide by weights to get physical units
    eof_patterns_valid = Vt / w_valid[np.newaxis, :]
    eof_patterns_valid = eof_patterns_valid * pc_std[:, np.newaxis]

    # Reconstruct full spatial array (NaN where masked)
    eof_full = np.full((n_svd, nspace), np.nan)
    eof_full[:, valid] = eof_patterns_valid

    eofs_3d = eof_full.reshape(n_svd, nlat, nlon)

    # Build xarray outputs
    mode_coord = np.arange(1, n_svd + 1)
    eofs_da = xr.DataArray(
        eofs_3d,
        dims=("mode", lat_name, lon_name),
        coords={
            "mode": mode_coord,
            lat_name: da[lat_name],
            lon_name: da[lon_name],
        },
    )

    pcs_da = xr.DataArray(
        pcs_norm,
        dims=("time", "mode"),
        coords={"time": da.time, "mode": mode_coord},
    )

    return eofs_da, pcs_da, var_explained[:n_svd]


def _find_lat(da):
    """Find latitude dimension name."""
    for name in ("lat", "latitude"):
        if name in da.dims:
            return name
    raise ValueError(f"No lat dimension found in {list(da.dims)}")


def _find_lon(da):
    """Find longitude dimension name."""
    for name in ("lon", "longitude"):
        if name in da.dims:
            return name
    raise ValueError(f"No lon dimension found in {list(da.dims)}")
