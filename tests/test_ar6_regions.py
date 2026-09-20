"""Tests for the AR6 (Iturbide et al. 2020) reference-region catalogue."""

import numpy as np
import pytest

from feather.util import ar6_regions as ar6

# A coarse grid keeps the point-in-polygon test cheap; 2° still resolves
# every region well enough to land a probe point inside it.
LATS = np.arange(-89.0, 90.0, 2.0)
LONS = np.arange(1.0, 360.0, 2.0)


@pytest.fixture(autouse=True)
def _clear_mask_cache():
    """The module-level mask cache must not leak between tests."""
    ar6._MASK_CACHE.clear()
    yield
    ar6._MASK_CACHE.clear()


class TestCatalogue:
    def test_fifty_eight_regions(self):
        assert len(ar6.list_ar6_regions()) == ar6.N_AR6_REGIONS == 58

    def test_abbrevs_unique(self):
        names = ar6.list_ar6_regions()
        assert len(set(names)) == len(names)

    def test_numbering_starts_at_greenland(self):
        """AR6 numbering is fixed; GIC is region 0, so order matters."""
        assert ar6.list_ar6_regions()[0] == "GIC"

    def test_region_name_lookup(self):
        assert ar6.ar6_region_name("GIC") == "Greenland/Iceland"

    def test_unknown_region_name_raises(self):
        with pytest.raises(KeyError, match="Unknown AR6 region"):
            ar6.ar6_region_name("NOPE")

    @pytest.mark.parametrize("abbrev", ["NAO", "SOO", "ARO", "BOB"])
    def test_ocean_regions(self, abbrev):
        assert ar6.is_ocean_region(abbrev)

    @pytest.mark.parametrize("abbrev", ["WCE", "SAH", "TIB", "GIC"])
    def test_land_regions(self, abbrev):
        assert not ar6.is_ocean_region(abbrev)

    @pytest.mark.parametrize("abbrev", ["CAR", "MED", "SEA"])
    def test_shared_land_and_ocean_regions(self, abbrev):
        """Three regions are in both sets — why 46 + 15 gives 58, not 61."""
        assert ar6.is_ocean_region(abbrev)


class TestMask:
    def test_shape_matches_grid(self):
        mask = ar6.ar6_mask(LATS, LONS)
        assert mask.shape == (LATS.size, LONS.size)

    def test_tiles_the_globe(self):
        """Regions cover essentially every cell; only seams are unassigned."""
        mask = ar6.ar6_mask(LATS, LONS)
        assert np.isfinite(mask).mean() > 0.99

    def test_all_regions_present(self):
        mask = ar6.ar6_mask(LATS, LONS)
        assert len(np.unique(mask[np.isfinite(mask)])) == 58

    def test_numbers_in_range(self):
        mask = ar6.ar6_mask(LATS, LONS)
        finite = mask[np.isfinite(mask)]
        assert finite.min() >= 0 and finite.max() <= 57

    def test_cached_by_grid(self):
        first = ar6.ar6_mask(LATS, LONS)
        assert ar6.ar6_mask(LATS, LONS) is first

    def test_distinct_grids_not_confused(self):
        coarse = ar6.ar6_mask(LATS, LONS)
        finer = ar6.ar6_mask(np.arange(-89.5, 90, 1.0), np.arange(0.5, 360, 1.0))
        assert coarse.shape != finer.shape

    def test_accepts_negative_longitudes(self):
        """regionmask wraps internally, so -180..180 must work too."""
        mask = ar6.ar6_mask(LATS, np.arange(-179.0, 180.0, 2.0))
        assert np.isfinite(mask).mean() > 0.99

    def test_region_masks_are_disjoint(self):
        masks = ar6.ar6_region_masks(LATS, LONS)
        stacked = np.stack(list(masks.values()))
        assert stacked.sum(axis=0).max() <= 1

    def test_region_masks_keyed_by_abbrev(self):
        masks = ar6.ar6_region_masks(LATS, LONS)
        assert set(masks) == set(ar6.list_ar6_regions())

    @pytest.mark.parametrize(
        "abbrev,lat,lon",
        [
            ("WCE", 49.0, 11.0),      # central Europe
            ("SAH", 23.0, 11.0),      # Sahara
            ("TIB", 35.0, 89.0),      # Tibetan plateau
            ("NAO", 45.0, 320.0),     # mid North Atlantic
            ("SOO", -58.0, 41.0),     # Southern Ocean (further south is EAN)
            ("EAN", -70.0, 41.0),     # East Antarctica
        ],
    )
    def test_known_points_land_in_expected_region(self, abbrev, lat, lon):
        """Spot-check the geography rather than trusting the numbering."""
        masks = ar6.ar6_region_masks(LATS, LONS)
        i = int(np.argmin(np.abs(LATS - lat)))
        j = int(np.argmin(np.abs(LONS - lon)))
        assert masks[abbrev][i, j]
