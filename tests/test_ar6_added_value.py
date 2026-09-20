"""Tests for AR6 reference-region Added Value (feather.diag._ar6_added_value)."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from feather.diag import _ar6_added_value as ar6av
from feather.diag.added_value import _region_sets


def _config(**kwargs):
    """Minimal stand-in for FeatherConfig — only what the module reads."""
    base = dict(
        added_value={},
        models=["ModelA", "Model-B"],
        output_dir="/tmp/out",
        project={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


# ── Region-set selection (the widened --added-value-regions flag) ────────


class TestRegionSets:
    def test_false_means_none(self):
        assert _region_sets(False) == ()
        assert _region_sets(None) == ()

    def test_bare_flag_keeps_legacy_cordex_behaviour(self):
        """`--added-value-regions` used to be a bare bool meaning CORDEX."""
        assert _region_sets(True) == ("cordex14",)

    def test_named_sets(self):
        assert _region_sets(["ar6"]) == ("ar6",)
        assert _region_sets(["cordex14", "ar6"]) == ("cordex14", "ar6")

    def test_string_accepted(self):
        assert _region_sets("ar6") == ("ar6",)

    def test_case_insensitive(self):
        assert _region_sets(["AR6"]) == ("ar6",)

    def test_unknown_dropped_not_fatal(self):
        assert _region_sets(["ar6", "bogus"]) == ("ar6",)

    def test_duplicates_collapsed(self):
        assert _region_sets(["ar6", "ar6"]) == ("ar6",)


# ── Configuration ────────────────────────────────────────────────────────


class TestReferences:
    def test_tas_defaults_to_era5_only(self):
        """Berkeley is land-only, so it cannot score the 15 ocean regions."""
        assert ar6av.references_for("tas", _config()) == [
            ("ERA5", "global_biases"),
        ]

    def test_pr_defaults_to_era5_and_mswep(self):
        assert ar6av.references_for("pr", _config()) == [
            ("ERA5", "global_biases"),
            ("MSWEP", "precipitation_mswep"),
        ]

    def test_config_override(self):
        cfg = _config(added_value={"ar6_references": {"tas": ["ERA5", "BE"]}})
        assert ar6av.references_for("tas", cfg) == [
            ("ERA5", "global_biases"),
            ("BE", "temperature_berkeley"),
        ]

    def test_unknown_reference_dropped(self):
        cfg = _config(added_value={"ar6_references": {"tas": ["ERA5", "XX"]}})
        assert ar6av.references_for("tas", cfg) == [("ERA5", "global_biases")]

    def test_unlisted_variable_falls_back_to_era5(self):
        assert ar6av.references_for("clt", _config()) == [
            ("ERA5", "global_biases"),
        ]


class TestThresholds:
    def test_default_is_six_and_three(self):
        assert ar6av.region_thresholds(_config()) == (6, 3)

    def test_config_value(self):
        cfg = _config(added_value={"region_thresholds": [5, 2]})
        assert ar6av.region_thresholds(cfg) == (5, 2)

    @pytest.mark.parametrize("bad", [[], "nope", [1], None])
    def test_invalid_falls_back_to_default(self, bad):
        cfg = _config(added_value={"region_thresholds": bad})
        assert ar6av.region_thresholds(cfg) == (6, 3)


class TestBiasPath:
    def test_filename_matches_netcdf_export(self):
        path = ar6av.bias_nc_path(
            "/out", "global_biases", "tas", "annual", ("1980", "2014"),
        )
        assert path.as_posix() == (
            "/out/netcdf/global_biases/tas_annual_1980-2014.nc"
        )


# ── Regional averaging ───────────────────────────────────────────────────


class TestWeightedRegionMean:
    def test_uniform_field(self):
        field = np.full((4, 4), 0.5)
        mask = np.ones((4, 4), dtype=bool)
        weights = np.ones((4, 4))
        assert ar6av._weighted_region_mean(field, mask, weights) == 0.5

    def test_respects_mask(self):
        field = np.array([[1.0, -1.0], [1.0, -1.0]])
        mask = np.array([[True, False], [True, False]])
        weights = np.ones((2, 2))
        assert ar6av._weighted_region_mean(field, mask, weights) == 1.0

    def test_weights_applied(self):
        field = np.array([[1.0, 0.0]])
        mask = np.ones((1, 2), dtype=bool)
        weights = np.array([[3.0, 1.0]])
        assert ar6av._weighted_region_mean(field, mask, weights) == 0.75

    def test_nan_cells_ignored(self):
        """A land-only reference leaves NaN over ocean; it must not poison."""
        field = np.array([[1.0, np.nan]])
        mask = np.ones((1, 2), dtype=bool)
        weights = np.ones((1, 2))
        assert ar6av._weighted_region_mean(field, mask, weights) == 1.0

    def test_all_nan_returns_nan(self):
        field = np.full((2, 2), np.nan)
        assert np.isnan(ar6av._weighted_region_mean(
            field, np.ones((2, 2), dtype=bool), np.ones((2, 2)),
        ))

    def test_empty_mask_returns_nan(self):
        assert np.isnan(ar6av._weighted_region_mean(
            np.ones((2, 2)), np.zeros((2, 2), dtype=bool), np.ones((2, 2)),
        ))

    def test_zero_total_weight_returns_nan(self):
        assert np.isnan(ar6av._weighted_region_mean(
            np.ones((2, 2)), np.ones((2, 2), dtype=bool), np.zeros((2, 2)),
        ))


# ── Keep-sets ────────────────────────────────────────────────────────────


def _rows(spec, period="annual", reference="ERA5"):
    """Build rows from ``{region: [AV, ...]}`` with generated member names."""
    out = []
    for region, values in spec.items():
        for i, value in enumerate(values):
            out.append({
                "reference": reference, "period": period,
                "member": f"m{i}", "region": region, "AV": value,
            })
    return out


class TestKeepSet:
    def test_counts_positive_members(self):
        rows = _rows({"WCE": [0.1, 0.2, 0.3, -0.1]})
        assert ar6av.keep_set(rows, "annual", 3) == ["WCE"]
        assert ar6av.keep_set(rows, "annual", 4) == []

    def test_zero_is_not_positive(self):
        rows = _rows({"WCE": [0.0, 0.0, 0.1]})
        assert ar6av.keep_set(rows, "annual", 2) == []

    def test_nan_is_not_positive(self):
        rows = _rows({"WCE": [np.nan, np.nan, 0.1]})
        assert ar6av.keep_set(rows, "annual", 2) == []

    def test_ensemble_rows_excluded(self):
        """Mean/median are summaries of the members, not extra evidence."""
        rows = _rows({"WCE": [0.1]})
        rows += [
            {"reference": "ERA5", "period": "annual", "member": stat,
             "region": "WCE", "AV": 0.9}
            for stat in ("ens_mean", "ens_median")
        ]
        assert ar6av.keep_set(rows, "annual", 2) == []

    def test_period_filtered(self):
        rows = _rows({"WCE": [0.1, 0.2]}) + _rows({"WCE": [0.1]}, period="JJA")
        assert ar6av.keep_set(rows, "JJA", 2) == []
        assert ar6av.keep_set(rows, "annual", 2) == ["WCE"]

    def test_union_across_references(self):
        """A region qualifies if it passes under *either* reference."""
        rows = _rows({"WCE": [0.1, 0.2, 0.3]}, reference="ERA5")
        rows += _rows({"WCE": [-0.1, -0.2, -0.3]}, reference="MSWEP")
        assert ar6av.keep_set(rows, "annual", 3) == ["WCE"]

    def test_counts_are_not_pooled_across_references(self):
        """Two refs with 2 positives each is not the same as 4 positives."""
        rows = _rows({"WCE": [0.1, 0.2]}, reference="ERA5")
        rows += _rows({"WCE": [0.1, 0.2]}, reference="MSWEP")
        assert ar6av.keep_set(rows, "annual", 4) == []
        assert ar6av.keep_set(rows, "annual", 2) == ["WCE"]

    def test_single_reference_filter(self):
        rows = _rows({"WCE": [0.1, 0.2, 0.3]}, reference="ERA5")
        rows += _rows({"WCE": [-0.1, -0.2, -0.3]}, reference="MSWEP")
        assert ar6av.keep_set(rows, "annual", 3, reference="MSWEP") == []

    def test_output_follows_ar6_order(self):
        rows = _rows({"WCE": [0.1], "GIC": [0.1], "NWN": [0.1]})
        assert ar6av.keep_set(rows, "annual", 1) == ["GIC", "NWN", "WCE"]


class TestCsv:
    def test_roundtrip(self, tmp_path):
        rows = _rows({"WCE": [0.25, -0.5]})
        path = ar6av.rows_to_csv(rows, tmp_path / "sub" / "av.csv")
        assert path.exists()
        read = list(csv.DictReader(open(path)))
        assert len(read) == 2
        assert read[0]["region"] == "WCE"
        assert float(read[0]["AV"]) == 0.25


# ── End-to-end over a synthetic bias NetCDF ──────────────────────────────


@pytest.fixture
def bias_nc(tmp_path):
    """A 2° bias file shaped like global_biases' --save-netcdf output."""
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    shape = (lat.size, lon.size)

    def da(value):
        return xr.DataArray(
            np.full(shape, value, dtype=float),
            dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        )

    # Benchmark is wrong by 2 K everywhere; ModelA by 1 (better), Model-B by
    # 4 (worse).  So AV is positive for ModelA, negative for Model-B.
    ds = xr.Dataset({
        "CMIP6_MMM_bias": da(2.0),
        "ModelA_bias": da(1.0),
        "Model_B_bias": da(4.0),
        "ens_mean_bias": da(1.5),
        "ens_median_bias": da(1.5),
    })
    nc_dir = tmp_path / "netcdf" / "global_biases"
    nc_dir.mkdir(parents=True)
    ds.to_netcdf(nc_dir / "tas_annual_1980-2014.nc")
    return tmp_path


class _FakeDiag:
    """Just the attributes ``compute_region_av`` reaches for."""

    _bench_label = "CMIP6 MMM"

    def __init__(self, output_dir):
        self.config = _config(output_dir=str(output_dir))
        self.period = ("1980", "2014")


class TestComputeRegionAv:
    def test_row_count(self, bias_nc):
        rows = ar6av.compute_region_av(
            _FakeDiag(bias_nc), "tas", periods=("annual",),
        )
        # 4 members (2 models + mean + median) × 58 regions, one reference.
        assert len(rows) == 4 * 58

    def test_better_model_scores_positive(self, bias_nc):
        rows = ar6av.compute_region_av(
            _FakeDiag(bias_nc), "tas", periods=("annual",),
        )
        wce = {r["member"]: r["AV"] for r in rows if r["region"] == "WCE"}
        # AV = (2² − 1²)/max(4, 1) = 0.75
        assert wce["ModelA"] == pytest.approx(0.75)
        # AV = (2² − 4²)/max(4, 16) = −0.75
        assert wce["Model_B"] == pytest.approx(-0.75)

    def test_model_names_are_sanitized(self, bias_nc):
        """``Model-B`` is stored as ``Model_B_bias`` by netcdf_export."""
        rows = ar6av.compute_region_av(
            _FakeDiag(bias_nc), "tas", periods=("annual",),
        )
        assert {r["member"] for r in rows} == {
            "ModelA", "Model_B", "ens_mean", "ens_median",
        }

    def test_all_regions_covered(self, bias_nc):
        rows = ar6av.compute_region_av(
            _FakeDiag(bias_nc), "tas", periods=("annual",),
        )
        assert len({r["region"] for r in rows}) == 58

    def test_missing_file_yields_no_rows(self, tmp_path):
        rows = ar6av.compute_region_av(
            _FakeDiag(tmp_path), "tas", periods=("annual",),
        )
        assert rows == []

    def test_missing_period_skipped_not_fatal(self, bias_nc):
        """A season nobody exported must not kill the annual result."""
        rows = ar6av.compute_region_av(
            _FakeDiag(bias_nc), "tas", periods=("annual", "JJA"),
        )
        assert {r["period"] for r in rows} == {"annual"}

    def test_missing_benchmark_field_skipped(self, tmp_path):
        lat = np.arange(-89.0, 90.0, 2.0)
        lon = np.arange(1.0, 360.0, 2.0)
        ds = xr.Dataset({
            "ModelA_bias": xr.DataArray(
                np.ones((lat.size, lon.size)), dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            ),
        })
        nc_dir = tmp_path / "netcdf" / "global_biases"
        nc_dir.mkdir(parents=True)
        ds.to_netcdf(nc_dir / "tas_annual_1980-2014.nc")
        assert ar6av.compute_region_av(
            _FakeDiag(tmp_path), "tas", periods=("annual",),
        ) == []


class TestLoadAvFields:
    def test_returns_float32_fields(self, bias_nc):
        lat, lon, fields = ar6av.load_av_fields(
            _FakeDiag(bias_nc), "tas", "global_biases", "annual",
        )
        assert lat.size == 90 and lon.size == 180
        assert set(fields) == {"ens_mean", "ens_median", "ModelA", "Model_B"}
        assert all(f.dtype == np.float32 for f in fields.values())

    def test_ensemble_panels_come_first(self, bias_nc):
        """Map panel order puts the ensemble summaries before members."""
        _, _, fields = ar6av.load_av_fields(
            _FakeDiag(bias_nc), "tas", "global_biases", "annual",
        )
        assert list(fields)[:2] == ["ens_mean", "ens_median"]

    def test_missing_file_returns_none(self, tmp_path):
        assert ar6av.load_av_fields(
            _FakeDiag(tmp_path), "tas", "global_biases", "annual",
        ) is None


class TestMemberLabel:
    def test_ensemble_rows(self):
        cfg = _config()
        assert ar6av.member_label("ens_mean", cfg) == "Ensemble mean"
        assert ar6av.member_label("ens_median", cfg) == "Ensemble median"

    def test_sanitized_name_maps_back(self):
        assert ar6av.member_label("Model_B", _config()) == "Model-B"

    def test_unknown_passes_through(self):
        assert ar6av.member_label("mystery", _config()) == "mystery"
