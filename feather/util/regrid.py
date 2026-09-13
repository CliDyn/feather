"""Regridding helpers that work around degenerate source grids.

``nereus`` builds conservative remapping weights from a spherical Voronoi
tessellation of the source points.  ``scipy.spatial.SphericalVoronoi``
rejects duplicate generators, and a regular lat/lon grid that **includes the
poles** supplies exactly that: at ``lat = ±90`` every longitude collapses to
the same physical point, so a 0.25° grid hands it 1440 identical generators
per pole and the build fails with::

    ValueError: Failed to build a spherical Voronoi tessellation from the
    source points for conservative remapping. ... Duplicate generators present.

Pole-inclusive grids are the norm, not the exception — the EERIE CMOR tree,
the ICON kerchunk stores and ERA5 are all 721×1440 spanning −90→90 — so
conservative remapping is unusable without handling this.  (MSWEP, at
−89.95→89.95, is one of the few that would work untouched.)

:func:`regrid` is a drop-in for ``nereus.regrid`` that collapses coincident
source points before building conservative weights, and averages the data
over each collapsed group.  This moves no data: it merges points that are
genuinely the same location.  For every other method it delegates unchanged,
so it is safe to use at any call site.

This is a workaround at the consumer end; the durable fix is for nereus to
deduplicate generators itself.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Rounding applied to unit-sphere coordinates when detecting coincident
#: points.  1e-9 of a unit radius is ~6 mm on Earth — far below any real
#: grid spacing, so only genuinely coincident points are merged.
_COORD_DECIMALS = 9


def dedupe_points(
    lon: np.ndarray, lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Collapse coincident ``(lon, lat)`` points.

    Points are compared as 3-D unit-sphere coordinates, so this catches both
    the pole collapse and any longitude wraparound duplicates (0° vs 360°).

    Returns
    -------
    lon_u, lat_u : ndarray
        Coordinates of the retained points.
    inverse : ndarray or None
        Index mapping each original point to its retained point, or ``None``
        when nothing was duplicated (the common case — callers can then skip
        the collapse entirely).
    """
    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    cos_lat = np.cos(lat_r)
    key = np.round(
        np.stack(
            [cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)],
            axis=1,
        ),
        decimals=_COORD_DECIMALS,
    )

    _, index, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True,
    )
    if index.size == lon.size:
        return lon, lat, None

    logger.info(
        "Collapsed %d coincident source points (%d → %d) for conservative "
        "remapping — a pole-inclusive grid duplicates every longitude at ±90°",
        lon.size - index.size, lon.size, index.size,
    )
    return lon[index], lat[index], inverse.ravel()


def collapse_values(data: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Average *data* over each group of coincident points.

    NaN-aware: a group averages only its finite members, and yields NaN when
    it has none (so a masked pole row stays masked rather than becoming 0).
    The spatial axis is the last one, matching ``nereus.regrid``.
    """
    data = np.asarray(data)
    n_groups = int(inverse.max()) + 1

    flat = data.reshape(-1, data.shape[-1])
    out = np.empty((flat.shape[0], n_groups), dtype=np.result_type(flat, np.float32))
    for i, row in enumerate(flat):
        valid = np.isfinite(row)
        num = np.bincount(inverse, weights=np.where(valid, row, 0.0),
                          minlength=n_groups)
        den = np.bincount(inverse, weights=valid.astype(float),
                          minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(den > 0, num / den, np.nan)

    return out.reshape(*data.shape[:-1], n_groups)


class DedupedInterpolator:
    """Wraps a nereus interpolator built on deduplicated source points.

    Collapses incoming data to the retained points on every call, so callers
    keep passing full-length arrays and nothing downstream changes.  Every
    other attribute (``target_lat``, ``target_lon``, ``method``, …) proxies
    to the wrapped interpolator.
    """

    def __init__(self, interpolator, inverse: np.ndarray):
        self._interpolator = interpolator
        self._inverse = inverse

    def __call__(self, data, *args, **kwargs):
        return self._interpolator(
            collapse_values(data, self._inverse), *args, **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._interpolator, name)


def regrid(data, lon=None, lat=None, *, method="nearest", **kwargs):
    """Drop-in for ``nereus.regrid`` that tolerates pole-inclusive grids.

    Delegates unchanged for every method other than ``"conservative"``.
    """
    import nereus as nr

    if method != "conservative" or lon is None or lat is None:
        return nr.regrid(data, lon=lon, lat=lat, method=method, **kwargs)

    lon_u, lat_u, inverse = dedupe_points(lon, lat)
    if inverse is None:
        return nr.regrid(data, lon=lon, lat=lat, method=method, **kwargs)

    regridded, interpolator = nr.regrid(
        collapse_values(np.asarray(data), inverse),
        lon=lon_u, lat=lat_u, method=method, **kwargs,
    )
    return regridded, DedupedInterpolator(interpolator, inverse)
