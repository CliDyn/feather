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

#: Merge radius on the unit sphere, in units of the sphere radius.
#:
#: This must match what ``scipy.spatial.SphericalVoronoi`` itself rejects.
#: Its constructor raises ``Duplicate generators present`` when
#: ``cKDTree(points).query_pairs(threshold * radius)`` finds anything, with
#: ``threshold`` defaulting to 1e-6 — so *near*-coincident points fail too,
#: not only exact duplicates.  Merging on exact equality is therefore not
#: enough: a curvilinear ocean grid can carry points a few metres apart that
#: scipy rejects but exact matching keeps.
#:
#: 1e-6 of a unit radius is ~6.4 m on Earth, orders of magnitude below any
#: climate grid spacing (0.25° ≈ 27 km), so this never merges cells that are
#: meaningfully distinct.
_MERGE_RADIUS = 1e-6


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
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    cos_lat = np.cos(lat_r)
    xyz = np.stack(
        [cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)],
        axis=1,
    )

    # Points within the merge radius of each other, grouped transitively:
    # connected components guarantee that no two *surviving* representatives
    # are within the radius, which is exactly scipy's acceptance condition.
    # Coordinates outside the valid range are fill values, not locations.
    # Several CMIP6 SImon grids publish lat/lon as 9.97e36; feeding those to
    # cKDTree yields hundreds of millions of "coincident" pairs and exhausts
    # memory. They cannot be deduplicated meaningfully, so leave the grid
    # alone and let the caller deal with it.
    finite = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90.0)
    if not finite.all():
        logger.warning(
            "%d of %d source points have invalid coordinates (fill values?) "
            "— skipping coincident-point merging for this grid",
            int((~finite).sum()), finite.size,
        )
        return lon, lat, None

    pairs = cKDTree(xyz).query_pairs(_MERGE_RADIUS, output_type="ndarray")
    if pairs.size == 0:
        return lon, lat, None

    n = lon.size
    graph = coo_matrix(
        (np.ones(pairs.shape[0], dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    n_groups, inverse = connected_components(graph, directed=False)
    if n_groups == n:
        return lon, lat, None

    # One representative per group: its first member.
    index = np.empty(n_groups, dtype=np.intp)
    order = np.argsort(inverse, kind="stable")
    first = order[np.concatenate(([True], np.diff(inverse[order]) != 0))]
    index[inverse[first]] = first

    logger.info(
        "Collapsed %d coincident source points (%d → %d) for conservative "
        "remapping — a pole-inclusive grid duplicates every longitude at ±90°",
        n - n_groups, n, n_groups,
    )
    return lon[index], lat[index], inverse


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

    *src_shape* is the source grid's spatial shape when the caller works in
    nereus' rectilinear convention (axis vectors plus ``(…, nlat, nlon)``
    data).  Those trailing dims are flattened to match the deduplicated point
    list, so a cached interpolator keeps accepting fields shaped the way the
    caller built it.
    """

    def __init__(self, interpolator, inverse: np.ndarray,
                 src_shape: tuple[int, ...] | None = None):
        self._interpolator = interpolator
        self._inverse = inverse
        self._src_shape = src_shape

    def __call__(self, data, *args, **kwargs):
        return self._interpolator(
            collapse_values(_flatten_spatial(data, self._src_shape),
                            self._inverse),
            *args, **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._interpolator, name)


def _flatten_spatial(data, src_shape: tuple[int, ...] | None):
    """Ravel *data*'s trailing *src_shape* dims into one point axis."""
    data = np.asarray(data)
    if src_shape is None:
        return data
    n = len(src_shape)
    if data.ndim >= n and data.shape[-n:] == tuple(src_shape):
        return data.reshape(*data.shape[:-n], -1)
    return data


def _mesh_if_rectilinear(data, lon, lat):
    """Expand nereus' rectilinear convention to an explicit point list.

    ``nereus.regrid`` accepts two source conventions: scattered points (1-D
    ``lon``/``lat`` of equal length, paired elementwise) and rectilinear axes
    (1-D ``lon``/``lat`` of *independent* lengths with ``(…, nlat, nlon)``
    data, which it meshgrids internally).  Deduplication needs paired points,
    so build the mesh here for the rectilinear case — otherwise a 721-lat,
    1440-lon grid reaches :func:`dedupe_points` as two arrays it tries to
    pair elementwise, and the coordinate maths fails to broadcast.

    Returns ``(lon_points, lat_points, src_shape)``, with *src_shape* None
    when the input was already a point list.
    """
    lon_a = np.asarray(lon)
    lat_a = np.asarray(lat)
    arr = np.asarray(data)
    rectilinear = (
        lon_a.ndim == 1 and lat_a.ndim == 1 and arr.ndim >= 2
        and arr.shape[-2:] == (lat_a.size, lon_a.size)
    )
    if not rectilinear:
        return lon_a, lat_a, None
    # meshgrid ordering matches raveling (…, nlat, nlon): index = ilat*nlon + ilon
    lon_m, lat_m = np.meshgrid(lon_a, lat_a)
    return lon_m.ravel(), lat_m.ravel(), (lat_a.size, lon_a.size)


def regrid(data, lon=None, lat=None, *, method="nearest", **kwargs):
    """Drop-in for ``nereus.regrid`` that tolerates pole-inclusive grids.

    Accepts either source convention nereus does — scattered points, or
    rectilinear axes with ``(…, nlat, nlon)`` data.  Delegates unchanged for
    every method other than ``"conservative"``.
    """
    import nereus as nr

    if method != "conservative" or lon is None or lat is None:
        return nr.regrid(data, lon=lon, lat=lat, method=method, **kwargs)

    lon_p, lat_p, src_shape = _mesh_if_rectilinear(data, lon, lat)
    if lon_p.size != lat_p.size:
        # Neither convention fits; let nereus report it in its own terms.
        return nr.regrid(data, lon=lon, lat=lat, method=method, **kwargs)

    lon_u, lat_u, inverse = dedupe_points(lon_p, lat_p)
    if inverse is None:
        return nr.regrid(data, lon=lon, lat=lat, method=method, **kwargs)

    regridded, interpolator = nr.regrid(
        collapse_values(_flatten_spatial(data, src_shape), inverse),
        lon=lon_u, lat=lat_u, method=method, **kwargs,
    )
    return regridded, DedupedInterpolator(interpolator, inverse, src_shape)
