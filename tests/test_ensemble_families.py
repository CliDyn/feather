"""Per-model-family summaries (``project.ensemble_mode: per_family``) beyond
global_biases: shared helpers, time series, Added Value (maps, bars, NetCDF,
ocean, AR6) and the Köppen–Trewartha ensemble map.

global_biases itself is covered by ``tests/test_global_biases_families.py``;
precipitation_mswep, temperature_berkeley and ocean_sst by a ``Family*``
class in their own test modules.
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.diag import _ar6_added_value as ar6av
from feather.diag import _families
from feather.diag.added_value import AddedValueDiag
from feather.diag.timeseries import TimeseriesDiag
from tests.conftest import MockCMIP6Loader
from tests.test_added_value import (
    MockCMORLoader,
    MockObsLoaderLatlon,
    _make_latlon,
    _write_bias_nc,
)


def _monthly(values, start="1990-01-01", calendar=None):
    n = len(values)
    if calendar is None:
        time = pd.date_range(start, periods=n, freq="MS") + pd.Timedelta(days=14)
    else:
        time = xr.date_range(start, periods=n, freq="MS", calendar=calendar,
                             use_cftime=True)
    return xr.DataArray(np.asarray(values, float), dims=("time",),
                        coords={"time": time})


# ── Shared helpers ───────────────────────────────────────────────────


class TestHelpers:
    FAMS = {"A": ["A", "A-r2"], "B": ["B"]}

    def test_label(self):
        assert _families.family_label("A", 3) == r"A mean $\mathbf{(3)}$"
        assert _families.family_label("B", 1, bold=False) == "B (1)"

    def test_mean_fields_skips_missing_members(self):
        fields = {"A": _make_latlon(1.0), "A-r2": _make_latlon(3.0), "B": None}
        out = _families.family_mean_fields(fields, self.FAMS)
        assert list(out) == ["A"]
        assert out["A"]["n_members"] == 2
        assert float(out["A"]["mean"].mean()) == pytest.approx(2.0)

    def test_compute_family_stats(self):
        obs = _make_latlon(0.0)
        mr = {
            m: {"annual_regrid": _make_latlon(v),
                "seasonal_regrids": {"DJF": _make_latlon(v + 1)}}
            for m, v in (("A", 1.0), ("A-r2", 3.0), ("B", -1.0))
        }
        out = _families.compute_family_stats(
            mr, self.FAMS, obs, {"DJF": obs}, None)
        assert set(out) == {"annual", "DJF"}
        assert out["annual"]["A"]["bias_gmean"] == pytest.approx(2.0)
        assert out["DJF"]["B"]["bias_gmean"] == pytest.approx(0.0)
        assert out["annual"]["A"]["rmse"] == pytest.approx(2.0)

    def test_mean_series_aligns_mixed_calendars(self):
        # Member r2 is on a 360-day calendar and starts a month later.
        ts = {
            "A": _monthly([1.0, 1.0, 1.0]),
            "A-r2": _monthly([3.0, 3.0], start="1990-02-01", calendar="360_day"),
            "B": _monthly([5.0, 5.0, 5.0]),
        }
        out = _families.family_mean_series(ts, self.FAMS)
        # B has one member → drawn as itself, no family line.
        assert list(out) == ["A"]
        assert out["A"]["ts"].sizes["time"] == 2
        np.testing.assert_allclose(out["A"]["ts"].values, 2.0)

    def test_lines_note(self):
        assert _families.family_lines_note(None) == ""
        assert _families.family_lines_note({"B": ["B"]}) == ""
        note = _families.family_lines_note(self.FAMS)
        assert "A (2 members)" in note and "not pooled" in note


# ── Time series ──────────────────────────────────────────────────────


@pytest.fixture
def family_ts_config(multi_model_config):
    return dataclasses.replace(
        multi_model_config,
        models=["ifs-fesom", "ifs-fesom-r2", "icon"],
        project={"ensemble_mode": "per_family"},
    )


class TestTimeseries:
    def test_no_pooled_stats(self, mock_multi_model_loader, mock_obs_loader,
                             family_ts_config):
        res = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, family_ts_config,
            variables=["tas"],
        ).compute()["tas"]
        assert res["ens_mean"] is None and res["ens_median"] is None

    def test_family_lines_in_all_three_figures(
        self, mock_multi_model_loader, mock_obs_loader, family_ts_config,
    ):
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, family_ts_config,
            variables=["tas"],
        )
        figs = diag.plot(diag.compute())
        assert len(figs) == 3
        for fig, meta in figs:
            labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
            assert "ifs-fesom mean (2)" in labels, meta["figure_id"]
            assert not any("ensemble mean" in lbl for lbl in labels)
            assert "not pooled" in meta["description"]
        plt.close("all")

    def test_pooled_mode_unchanged(self, mock_multi_model_loader,
                                   mock_obs_loader, multi_model_config):
        diag = TimeseriesDiag(
            mock_multi_model_loader, mock_obs_loader, multi_model_config,
            variables=["tas"],
        )
        figs = diag.plot(diag.compute())
        labels = [ln.get_label() for ln in figs[0][0].axes[0].get_lines()]
        assert any("ensemble mean" in lbl for lbl in labels)
        assert "not pooled" not in figs[0][1]["description"]
        plt.close("all")


# ── Added Value ──────────────────────────────────────────────────────


def _mc(name, family="", color="#1f77b4"):
    return ModelConfig(
        name=name, institution="INS", experiment="hist-1950",
        variant="r1i1p1f1",
        grids={"sfc": "latlon", "o2d": "latlon", "o3d": "latlon"},
        color=color, data_source_type="cmor", family=family,
    )


@pytest.fixture
def family_av_config(tmp_path):
    mcs = {
        "ModelA": _mc("ModelA", "ModelA", "#1f77b4"),
        "ModelA-r2": _mc("ModelA-r2", "ModelA", "#6baed6"),
        "ModelB": _mc("ModelB", "ModelB", "#ff7f0e"),
    }
    return FeatherConfig(
        model_catalogs={}, models=dict(mcs), model_configs=mcs,
        obs_root="", obs_datasets={}, cmip6={"enabled": True}, dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"), data_source={"type": "cmor"},
        project={"ensemble_mode": "per_family", "name": "EERIE"},
    )


def _av_diag(synth_obs, synth_cmip6, cfg, **kw):
    return AddedValueDiag(
        MockCMORLoader(synth_obs), MockObsLoaderLatlon(synth_obs), cfg,
        cmip6_loader=MockCMIP6Loader(synth_cmip6),
        variables=kw.pop("variables", ["psl"]), period=("1990", "1990"), **kw,
    )


def _seed_family_ncs(diag, var="psl"):
    """Bias NetCDFs with members only (no pooled ens fields): family A's
    mean bias is 2.0 (members 1 and 3), B's is -1; benchmark 4."""
    out = Path(diag.config.output_dir) / "netcdf" / "global_biases"
    for pk in ("annual", "DJF", "MAM", "JJA", "SON"):
        path = out / f"{var}_{pk}_1990-1990.nc"
        _write_bias_nc(
            path, models_bias={"ModelA": 1.0, "ModelA_r2": 3.0, "ModelB": -1.0},
            bench_bias=4.0, ens_mean_bias=0.0, ens_median_bias=0.0,
        )
        ds = xr.open_dataset(path).load()
        ds.close()
        ds.drop_vars(["ens_mean_bias", "ens_median_bias"]).to_netcdf(path)


class TestAddedValueFastPath:
    def test_family_av_from_member_biases(self, synth_obs, synth_cmip6,
                                          family_av_config):
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config)
        _seed_family_ncs(diag)
        res = diag._compute_variable_from_netcdf("psl")
        assert res is not None
        annual = res["av"]["annual"]
        assert annual["summaries"] == {
            "family_ModelA": "ModelA mean (2)", "family_ModelB": "ModelB (1)"}
        assert "ensemble_mean" not in annual
        # AV = (4² − 2²)/4² = 0.75 for family A; (16 − 1)/16 for B.
        assert annual["family_ModelA_domain_av"] == pytest.approx(0.75)
        assert annual["family_ModelB_domain_av"] == pytest.approx(15 / 16)
        stats = res["obs_stats"]["ERA5"]["annual"]
        assert set(stats["families"]) == {"ModelA", "ModelB"}
        assert "eerie_mean" not in stats and "cmip6_mean" not in stats

    def test_nc_checkpoint_roundtrip(self, synth_obs, synth_cmip6,
                                     family_av_config):
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config)
        _seed_family_ncs(diag)
        diag._compute_variable_from_netcdf("psl")
        assert diag._nc_path("psl", "annual", "family_ModelA").exists()
        assert diag._all_nc_exist("psl")
        loaded = diag._load_variable_from_nc("psl")
        assert loaded["av"]["annual"]["summaries"]["family_ModelA"] == (
            "ModelA mean (2)")

    def test_figures_and_bars(self, synth_obs, synth_cmip6, family_av_config):
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config)
        _seed_family_ncs(diag)
        vr = diag._compute_variable_from_netcdf("psl")
        figs = diag._plot_variable("psl", vr)
        fig1, meta1 = figs[0]
        assert meta1["figure_id"].endswith("_added_value")
        assert set(meta1["summary_statistics"]) - {"per_obs_stats"} == {
            "family_ModelA", "family_ModelB"}
        assert "model means" in meta1["title"]
        assert "not pooled" in meta1["description"]

        stats = {"psl": vr["obs_stats"]}
        (bfig, bmeta), = diag._plot_summary_bars_ensemble(stats, "annual")
        legend = [t.get_text() for t in bfig.axes[0].get_legend().get_texts()]
        assert legend[:2] == ["ModelA mean (2)", "ModelB (1)"]
        assert not any("CMIP6 mean" in t for t in legend)

        (mfig, _), = diag._plot_summary_bars_models(
            stats, "annual", show_cmip6_bar=False)
        legend = [t.get_text() for t in mfig.axes[0].get_legend().get_texts()]
        assert "ModelA mean (2)" in legend
        assert "Ensemble mean" not in legend
        (cfig, _), = diag._plot_summary_bars_models(stats, "annual")
        legend = [t.get_text() for t in cfig.axes[0].get_legend().get_texts()]
        assert not any("mean" in t for t in legend[:3])
        plt.close("all")

    def test_individual_benchmark_panels_skipped(self, synth_obs, synth_cmip6,
                                                 family_av_config):
        """CMIP6 members are scored against the pooled mean — not formed."""
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config)
        diag.cmip6_individual = True
        _seed_family_ncs(diag)
        _write_bias_nc(
            Path(diag.config.output_dir) / "netcdf" / "global_biases"
            / "psl_annual_individual_1990-1990.nc",
            models_bias={"CMIP6__ACCESS_CM2_r1i1p1f1": 3.0},
            bench_bias=4.0, ens_mean_bias=1.0, ens_median_bias=1.0,
        )
        res = diag._compute_variable_from_netcdf("psl")
        assert res["av"]["annual"]["per_cmip6_av"] == {}

    def test_ocean_blocks_without_pooled_fields(self, synth_obs, synth_cmip6,
                                                family_av_config):
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config)
        blk = {
            "bench_bias": _make_latlon(4.0),
            "ens_mean_bias": None, "ens_median_bias": None,
            "models": {"ModelA": _make_latlon(1.0),
                       "ModelA-r2": _make_latlon(3.0),
                       "ModelB": _make_latlon(-1.0)},
            "individual": {}, "area": None,
        }
        res = diag._ocean_av_from_blocks(
            "tos", {"blocks": {"annual": blk},
                    "eerie_models": list(blk["models"]), "bench_labels": []},
            "ESA_CCI",
        )
        annual = res["av"]["annual"]
        assert annual["family_ModelA_domain_av"] == pytest.approx(0.75)
        figs = diag._plot_ocean_variable("tos", res)
        assert "model means" in figs[0][1]["title"]
        plt.close("all")


class TestAddedValueRecompute:
    def test_recompute_stats_per_family(self, synth_obs, synth_cmip6,
                                        family_av_config):
        diag = _av_diag(synth_obs, synth_cmip6, family_av_config,
                        variables=["tas"], regions=True)
        res = diag.compute()["tas"]
        annual = res["av"]["annual"]
        assert set(annual["summaries"]) == {"family_ModelA", "family_ModelB"}
        stats = res["obs_stats"]["ERA5"]["annual"]
        assert set(stats["families"]) == {"ModelA", "ModelB"}
        assert "cmip6_mean" not in stats
        eur = stats["regions"]["EUR"]
        assert set(eur["families"]) == {"ModelA", "ModelB"}
        assert "eerie_mean" not in eur


# ── AR6 regions ──────────────────────────────────────────────────────


class _FakeAr6Diag:
    _bench_label = "CMIP6 MMM"

    def __init__(self, config):
        self.config = config
        self.period = ("1980", "2014")


@pytest.fixture
def ar6_family_diag(tmp_path):
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)

    def da(v):
        return xr.DataArray(np.full((lat.size, lon.size), v), dims=("lat", "lon"),
                            coords={"lat": lat, "lon": lon})

    nc_dir = tmp_path / "netcdf" / "global_biases"
    nc_dir.mkdir(parents=True)
    # Older file: members only, no family_*_mean_bias fields.
    xr.Dataset({
        "CMIP6_MMM_bias": da(2.0), "ModelA_bias": da(1.0),
        "ModelA_r2_bias": da(1.0), "ModelB_bias": da(4.0),
    }).to_netcdf(nc_dir / "tas_annual_1980-2014.nc")
    mcs = {m: _mc(m, f) for m, f in (
        ("ModelA", "ModelA"), ("ModelA-r2", "ModelA"), ("ModelB", "ModelB"))}
    cfg = FeatherConfig(
        model_catalogs={}, models=dict(mcs), model_configs=mcs,
        obs_root="", obs_datasets={}, cmip6={}, dask={}, nereus={},
        output_dir=str(tmp_path), project={"ensemble_mode": "per_family"},
    )
    return _FakeAr6Diag(cfg)


class TestAr6:
    def test_summary_rows(self, ar6_family_diag):
        assert ar6av.summary_rows(ar6_family_diag.config) == (
            "family_ModelA_mean", "family_ModelB_mean")
        assert ar6av.is_summary_row("family_ModelA_mean")
        assert not ar6av.is_summary_row("ModelA")

    def test_family_rows_from_member_biases(self, ar6_family_diag):
        rows = ar6av.compute_region_av(
            ar6_family_diag, "tas", periods=("annual",))
        wce = {r["member"]: r["AV"] for r in rows if r["region"] == "WCE"}
        assert wce["family_ModelA_mean"] == pytest.approx(0.75)
        assert wce["family_ModelB_mean"] == pytest.approx(-0.75)
        assert "ens_mean" not in wce

    def test_keep_set_counts_members_only(self, ar6_family_diag):
        rows = ar6av.compute_region_av(
            ar6_family_diag, "tas", periods=("annual",))
        # Two members positive everywhere; the family row must not add a third.
        assert len(ar6av.keep_set(rows, "annual", 2)) == 58
        assert ar6av.keep_set(rows, "annual", 3) == []

    def test_labels_and_maps(self, ar6_family_diag):
        cfg = ar6_family_diag.config
        assert ar6av.member_label("family_ModelA_mean", cfg) == "ModelA mean (2)"
        assert ar6av.member_label("family_ModelB_mean", cfg) == "ModelB (1)"
        _, _, fields = ar6av.load_av_fields(
            ar6_family_diag, "tas", "global_biases", "annual")
        assert list(fields)[:2] == ["family_ModelA_mean", "family_ModelB_mean"]


# ── Köppen–Trewartha ensemble grouping ───────────────────────────────


class TestClimateClassification:
    def _diag(self, cfg):
        from feather.diag.climate_classification import KTClimateClassification
        diag = KTClimateClassification.__new__(KTClimateClassification)
        diag.config = cfg
        return diag

    def test_groups_by_family_mean_only(self, family_av_config):
        for mc in family_av_config.model_configs.values():
            mc.ensemble = "EERIE"
        diag = self._diag(family_av_config)
        assert diag._ensemble_groups(family_av_config.models) == {
            "ModelA": ["ModelA", "ModelA-r2"], "ModelB": ["ModelB"]}
        assert diag._ensemble_names("ModelA") == {"mean": "ModelA Mean"}

    def test_pooled_groups_by_ensemble_label(self, family_av_config):
        for mc in family_av_config.model_configs.values():
            mc.ensemble = "EERIE"
        cfg = dataclasses.replace(family_av_config, project={})
        diag = self._diag(cfg)
        assert diag._ensemble_groups(cfg.models) == {
            "EERIE": ["ModelA", "ModelA-r2", "ModelB"]}
        assert set(diag._ensemble_names("EERIE")) == {"mean", "median"}
