"""Tests for CORDEX analysis regions (feather.util.regions)."""

import numpy as np
import pytest

from feather.util.regions import (
    CordexRegion,
    get_region,
    list_regions,
    region_mask,
)


@pytest.fixture
def grid():
    """A coarse 1° global grid (0..360 longitude)."""
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    return lat, lon


def _contains(lat, lon, name, plon, plat):
    """True if the grid cell nearest (plon, plat) is inside region *name*."""
    m = region_mask(lat, lon, name)
    i = int(np.argmin(np.abs(lat - plat)))
    j = int(np.argmin(np.abs((lon % 360) - (plon % 360))))
    return bool(m[i, j])


# ── Catalogue basics ───────────────────────────────────────────────────────

def test_list_regions_has_expected_domains():
    names = list_regions()
    assert len(names) == 14
    for expected in ("EUR", "SAM", "NAM", "AFR", "EAS", "AUS", "ARC", "ANT"):
        assert expected in names
    # SEA has no CMIP6 -11 spec → included via its SEA-22 extent instead.
    assert "SEA" in names


def test_get_region_case_insensitive():
    assert get_region("eur").name == "EUR"
    assert get_region("EUR").long_name == "Europe"


def test_get_region_unknown_raises():
    with pytest.raises(KeyError):
        get_region("NOPE")


def test_polygon_regions_have_four_corners():
    for name in list_regions():
        r = get_region(name)
        if r.kind == "polygon":
            assert len(r.corners) == 4
            for lon, lat in r.corners:
                assert -180.0 <= lon <= 180.0
                assert -90.0 <= lat <= 90.0


def test_polar_regions_are_caps():
    assert get_region("ANT").kind == "cap"
    assert get_region("ARC").kind == "cap"
    assert get_region("ANT").cap_side == "south"
    assert get_region("ARC").cap_side == "north"


# ── Mask geometry ──────────────────────────────────────────────────────────

def test_mask_shape_matches_grid(grid):
    lat, lon = grid
    m = region_mask(lat, lon, "EUR")
    assert m.shape == (len(lat), len(lon))
    assert m.dtype == bool
    assert m.any()


@pytest.mark.parametrize(
    "name,plon,plat,expected",
    [
        ("EUR", 2.35, 48.85, True),    # Paris
        ("EUR", -74.0, 40.7, False),   # New York
        ("NAM", -100.0, 40.0, True),   # US Great Plains
        ("NAM", 2.35, 48.85, False),   # Paris not in NAM
        ("AFR", 20.0, 0.0, True),      # equatorial Africa
        ("EAS", 116.0, 40.0, True),    # Beijing
        ("SAM", -60.0, -15.0, True),   # South America
        ("AUS", 134.0, -25.0, True),   # central Australia
        ("MED", 15.0, 40.0, True),     # Mediterranean
        ("SEA", 110.0, 5.0, True),     # South East Asia (SEA-22 extent)
        ("SEA", 0.0, 5.0, False),      # Atlantic not in SEA
    ],
)
def test_polygon_membership(grid, name, plon, plat, expected):
    lat, lon = grid
    assert _contains(lat, lon, name, plon, plat) is expected


def test_caps(grid):
    lat, lon = grid
    # Arctic cap: north of ~48.6°
    assert _contains(lat, lon, "ARC", 0.0, 80.0) is True
    assert _contains(lat, lon, "ARC", 0.0, 30.0) is False
    # Antarctic cap: south of ~-55.5°
    assert _contains(lat, lon, "ANT", 0.0, -80.0) is True
    assert _contains(lat, lon, "ANT", 0.0, -30.0) is False


def test_antimeridian_region_does_not_wrap_globe(grid):
    """AUS straddles the dateline; its mask must not cover the whole tropics."""
    lat, lon = grid
    m = region_mask(lat, lon, "AUS")
    frac = m.mean()
    # AUS covers ~11% of the globe; a wrapped polygon would cover ~half.
    assert 0.03 < frac < 0.25


def test_mask_accepts_both_longitude_conventions():
    lat = np.arange(-89.5, 90.0, 1.0)
    lon_0360 = np.arange(0.5, 360.0, 1.0)
    lon_pm180 = ((lon_0360 + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_pm180)
    m_a = region_mask(lat, lon_0360, "EUR")
    m_b = region_mask(lat, lon_pm180[order], "EUR")
    # Same set of cells flagged regardless of longitude convention.
    assert int(m_a.sum()) == int(m_b.sum())


def test_dataclass_frozen():
    r = get_region("EUR")
    assert isinstance(r, CordexRegion)
    with pytest.raises(Exception):
        r.name = "X"  # frozen
