"""IPCC AR6 WGI reference regions (Iturbide et al. 2020).

The 58 regions of `Iturbide et al. (2020)
<https://doi.org/10.5194/essd-12-2959-2020>`_ — 46 land and 15 ocean, three
of which (``CAR``, ``MED``, ``SEA``) belong to both sets, hence 58 rather
than 61.  Polygons come from :mod:`regionmask`
(``regionmask.defined_regions.ar6.all``), which vendors the reference
shapefiles published at `SantanderMetGroup/ATLAS
<https://github.com/SantanderMetGroup/ATLAS/tree/devel/reference-regions>`_.

Unlike the CORDEX catalogue in :mod:`feather.util.regions` — baked-in corner
tables with no runtime geometry dependency — the AR6 polygons are genuine
multi-vertex shapes, so ``regionmask`` is a hard dependency here.

Coverage
--------
The 58 regions **tile the globe**: on a 0.25° grid 99.998 % of cells fall in
exactly one region (the remainder are a few cells on the polygon seams).
There is no land/ocean sub-masking — an "ocean" region such as ``NAO``
covers its full polygon, and regional means are taken over every valid cell
inside it.

Public API
----------
``list_ar6_regions()``        ordered list of the 58 abbreviations.
``ar6_region_name(abbrev)``   human-readable name.
``is_ocean_region(abbrev)``   True for the 15 ocean regions.
``ar6_mask(lat, lon)``        integer region-number field on a lat/lon grid.
``ar6_region_masks(lat, lon)``  ``{abbrev: boolean mask}``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "list_ar6_regions",
    "ar6_region_name",
    "is_ocean_region",
    "ar6_mask",
    "ar6_region_masks",
    "ar6_regions",
    "N_AR6_REGIONS",
]

#: Number of AR6 reference regions (46 land + 15 ocean − 3 shared).
N_AR6_REGIONS = 58

#: Cache of computed masks, keyed by a grid signature.  The mask depends only
#: on the axes, so one entry serves every variable, period and member on that
#: grid — which matters because the point-in-polygon test over 58 polygons on
#: a 0.25° grid is far from free.
_MASK_CACHE: dict[tuple, np.ndarray] = {}


def ar6_regions():
    """Return the :mod:`regionmask` ``Regions`` object for the AR6 set.

    Raises
    ------
    ImportError
        With an actionable message when ``regionmask`` is not installed.
    """
    try:
        import regionmask
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The AR6 reference regions need `regionmask` (>=0.13). "
            "Install it with `pip install regionmask`."
        ) from exc

    regions = regionmask.defined_regions.ar6.all
    if len(regions) != N_AR6_REGIONS:  # pragma: no cover - upstream change
        logger.warning(
            "regionmask ar6.all has %d regions, expected %d — the region set "
            "may have changed upstream",
            len(regions), N_AR6_REGIONS,
        )
    return regions


def list_ar6_regions() -> list[str]:
    """Return the 58 AR6 abbreviations in regionmask's own order.

    The order is the AR6 numbering (``GIC``, ``NWN``, ``NEN``, …), i.e. land
    regions first, then ocean, which is what the reference figures use for
    their column order.
    """
    return [str(a) for a in ar6_regions().abbrevs]


def ar6_region_name(abbrev: str) -> str:
    """Return the human-readable name for *abbrev* (e.g. ``"Greenland/Iceland"``)."""
    regions = ar6_regions()
    lookup = {str(a): str(n) for a, n in zip(regions.abbrevs, regions.names)}
    try:
        return lookup[abbrev]
    except KeyError as exc:
        raise KeyError(
            f"Unknown AR6 region {abbrev!r}; available: "
            f"{', '.join(sorted(lookup))}"
        ) from exc


def is_ocean_region(abbrev: str) -> bool:
    """True when *abbrev* is one of the 15 AR6 ocean regions.

    ``CAR``, ``MED`` and ``SEA`` are in both the land and ocean sets and
    return True here.
    """
    import regionmask

    return abbrev in {str(a) for a in regionmask.defined_regions.ar6.ocean.abbrevs}


def _grid_key(lat: np.ndarray, lon: np.ndarray) -> tuple:
    """Signature identifying a lat/lon grid for the mask cache."""
    return (
        lat.size, lon.size,
        float(lat[0]), float(lat[-1]),
        float(lon[0]), float(lon[-1]),
    )


def ar6_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Region-number field on a regular lat/lon grid.

    Parameters
    ----------
    lat, lon : array-like, 1-D
        Grid axes.  Either longitude convention works — ``regionmask``
        wraps internally.

    Returns
    -------
    np.ndarray
        Float array of shape ``(len(lat), len(lon))`` holding the AR6 region
        number (0-57) of each cell, ``NaN`` where a cell falls in no region.
        Float rather than int precisely so the gaps can be NaN.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    key = _grid_key(lat, lon)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    mask = np.asarray(ar6_regions().mask(lon, lat).values, dtype=float)
    _MASK_CACHE[key] = mask
    return mask


def ar6_region_masks(
    lat: np.ndarray, lon: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return ``{abbrev: boolean mask}`` for all 58 regions on a grid."""
    field = ar6_mask(lat, lon)
    return {
        abbrev: (field == number)
        for number, abbrev in enumerate(list_ar6_regions())
    }
