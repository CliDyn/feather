"""CORDEX analysis regions (CMIP6 ``*-11`` domains).

This module provides a small, dependency-free catalogue of the CORDEX-CMIP6
``-11`` (0.11°) domains for use as **analysis regions** (not for loading model
output — that is :mod:`feather.data.cordex_loader`).  Each region is named by
its bare CORDEX domain prefix (``EUR``, ``SAM``, …).

Geometry
--------
CORDEX domains are defined on rotated-pole grids.  Following the agreed
convention we represent each mid-latitude domain by the **four geographic
corners** of its rotated bounding box (a quadrilateral polygon).  ``SEA`` has
no CMIP6 ``-11`` specification, so its ``SEA-22`` (0.22°) extent is used
instead.  The corner
coordinates were derived once, offline, from the authoritative WCRP-CORDEX
domain table (``rotated-latitude-longitude.csv``) via the standard
rotated-pole → geographic transform, and are baked in below so no runtime
dependency on ``cartopy``/``py-cordex`` is needed.

Caveat: a 4-corner quadrilateral slightly under-covers strongly rotated
domains near the bowed mid-edges (e.g. EUR reaches ~72°N at its top edge but
only ~66°N at the corners).  This is acceptable for area-weighted regional
statistics.

Polar caps
----------
The ANT and ARC domains enclose a pole, so a quadrilateral is meaningless for
them.  They are represented instead as **latitude caps** (everything south of
ANT's northern corner latitude, everything north of ARC's southern corner
latitude).

Public API
----------
``list_regions()``       ordered list of region names.
``get_region(name)``     :class:`CordexRegion` metadata.
``region_mask(lat, lon, name)``  boolean mask on a regular lat/lon grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CordexRegion", "list_regions", "get_region", "region_mask"]


@dataclass(frozen=True)
class CordexRegion:
    """A CORDEX-CMIP6 ``-11`` analysis region.

    Parameters
    ----------
    name : str
        Bare CORDEX domain prefix (e.g. ``"EUR"``).
    long_name : str
        Human-readable domain name.
    kind : str
        ``"polygon"`` for a 4-corner quadrilateral, ``"cap"`` for a polar cap.
    corners : tuple
        For ``kind="polygon"``: four ``(lon, lat)`` geographic corners in
        ``-180..180`` longitude, ordered counter-clockwise.  Empty for caps.
    cap_lat : float
        For ``kind="cap"``: the threshold latitude (degrees).
    cap_side : str
        For ``kind="cap"``: ``"north"`` (lat ≥ cap_lat) or ``"south"``
        (lat ≤ cap_lat).
    """

    name: str
    long_name: str
    kind: str
    corners: tuple = field(default=())
    cap_lat: float = float("nan")
    cap_side: str = ""


# ── Baked-in domain table (CMIP6 CORDEX *-11, geographic corners) ──────────
# Corners derived offline from WCRP-CORDEX/domain-tables
# (rotated-latitude-longitude.csv) using the rotated-pole transform.
_REGIONS: dict[str, CordexRegion] = {
    "SAM": CordexRegion("SAM", "South America", "polygon", (
        (-106.009, -52.548), (-16.691, -54.499),
        (-32.364, 17.303), (-86.839, 18.566))),
    "CAM": CordexRegion("CAM", "Central America", "polygon", (
        (-114.012, -19.530), (-30.413, -17.308),
        (-22.007, 31.349), (-124.480, 28.746))),
    "NAM": CordexRegion("NAM", "North America", "polygon", (
        (-127.219, 12.247), (-66.781, 12.247),
        (-22.861, 59.020), (-171.139, 59.020))),
    "EUR": CordexRegion("EUR", "Europe", "polygon", (
        (-10.064, 21.851), (36.414, 24.962),
        (64.964, 66.473), (-44.594, 59.978))),
    "AFR": CordexRegion("AFR", "Africa", "polygon", (
        (-24.805, -45.704), (60.445, -45.704),
        (60.445, 42.190), (-24.805, 42.190))),
    "WAS": CordexRegion("WAS", "South Asia", "polygon", (
        (26.027, -13.060), (106.546, -15.317),
        (115.747, 40.821), (19.651, 43.430))),
    "EAS": CordexRegion("EAS", "East Asia", "polygon", (
        (77.364, -17.310), (159.584, -18.297),
        (175.392, 53.453), (63.095, 54.707))),
    "CAS": CordexRegion("CAS", "Central Asia", "polygon", (
        (42.738, 18.190), (108.910, 18.883),
        (140.053, 55.983), (11.150, 54.831))),
    "AUS": CordexRegion("AUS", "Australasia", "polygon", (
        (88.958, -44.144), (-153.152, -39.119),
        (-177.913, 12.334), (110.104, 8.901))),
    "MED": CordexRegion("MED", "Mediterranean", "polygon", (
        (-6.135, 25.273), (38.442, 26.372),
        (51.206, 52.198), (-20.560, 50.492))),
    "MNA": CordexRegion("MNA", "Middle East and North Africa", "polygon", (
        (-26.565, -6.720), (75.405, -6.720),
        (75.405, 44.825), (-26.565, 44.825))),
    # SEA has no CORDEX-CMIP6 -11 spec, so the SEA-22 (0.22°) extent is used.
    # Its rotated pole is (180, 90) → effectively a regular lat/lon box.
    "SEA": CordexRegion("SEA", "South East Asia", "polygon", (
        (89.260, -15.082), (147.120, -15.082),
        (147.120, 27.117), (89.260, 27.117))),
    # Polar caps (enclose a pole → latitude threshold, not a quadrilateral).
    "ARC": CordexRegion("ARC", "Arctic", "cap",
                        cap_lat=48.613, cap_side="north"),
    "ANT": CordexRegion("ANT", "Antarctica", "cap",
                        cap_lat=-55.542, cap_side="south"),
}

#: Region order used by :func:`list_regions` (mid-latitude domains, then poles).
_ORDER = [
    "SAM", "CAM", "NAM", "EUR", "AFR", "WAS",
    "EAS", "CAS", "AUS", "MED", "MNA", "SEA", "ARC", "ANT",
]


def list_regions() -> list[str]:
    """Return the ordered list of CORDEX region names."""
    return list(_ORDER)


def get_region(name: str) -> CordexRegion:
    """Return the :class:`CordexRegion` for *name* (case-insensitive)."""
    try:
        return _REGIONS[name.upper()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown CORDEX region {name!r}; "
            f"available: {', '.join(_ORDER)}"
        ) from exc


def _wrap_to_frame(lons: np.ndarray, frame: str) -> np.ndarray:
    """Wrap longitudes into ``"0360"`` (0..360) or ``"-180"`` (-180..180)."""
    lons = np.asarray(lons, dtype=float)
    if frame == "0360":
        return lons % 360.0
    return ((lons + 180.0) % 360.0) - 180.0


def _choose_frame(corner_lons: np.ndarray) -> str:
    """Pick the longitude frame minimising the polygon's longitudinal span.

    A domain crossing the antimeridian (e.g. AUS) has a smaller span in the
    0..360 frame; one straddling the prime meridian (e.g. EUR) is smaller in
    -180..180.  Choosing the tighter frame avoids a polygon that wraps the
    globe and breaks the planar point-in-polygon test.
    """
    span = {}
    for frame in ("-180", "0360"):
        w = _wrap_to_frame(corner_lons, frame)
        span[frame] = float(w.max() - w.min())
    return min(span, key=span.get)


def region_mask(
    lat: np.ndarray, lon: np.ndarray, name: str,
) -> np.ndarray:
    """Boolean mask of grid points inside CORDEX region *name*.

    Parameters
    ----------
    lat, lon : array-like, 1-D
        Latitude and longitude axes of a regular grid (any longitude
        convention; 0..360 and -180..180 both accepted).
    name : str
        CORDEX region name (see :func:`list_regions`).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(len(lat), len(lon))``; ``True`` inside the
        region.
    """
    region = get_region(name)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    if region.kind == "cap":
        if region.cap_side == "north":
            return lat2d >= region.cap_lat
        return lat2d <= region.cap_lat

    # Polygon (4-corner quadrilateral). Choose a longitude frame in which the
    # polygon does not wrap the globe, then planar point-in-polygon.
    from matplotlib.path import Path

    corners = np.asarray(region.corners, dtype=float)
    frame = _choose_frame(corners[:, 0])
    poly_lon = _wrap_to_frame(corners[:, 0], frame)
    poly = np.column_stack([poly_lon, corners[:, 1]])
    path = Path(poly)

    pts_lon = _wrap_to_frame(lon2d.ravel(), frame)
    pts = np.column_stack([pts_lon, lat2d.ravel()])
    inside = path.contains_points(pts)
    return inside.reshape(lat2d.shape)
