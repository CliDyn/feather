"""Tests for per-model-family summaries in GlobalBiases (ensemble_mode)."""

import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from feather.config import FeatherConfig, ModelConfig
from feather.diag import _families
from feather.diag.global_biases import GlobalBiases


def _cfg(base, models, *, mode="per_family", families=None):
    cfg = dataclasses.replace(
        base, models=list(models),
        project={**(base.project or {}), "ensemble_mode": mode},
        model_configs={
            m: ModelConfig(name=m, family=(families or {}).get(m, ""))
            for m in models
        },
    )
    return cfg


# ── Config ───────────────────────────────────────────────────────────


class TestFamilies:
    def test_explicit_key_wins(self, minimal_config):
        cfg = _cfg(minimal_config, ["X", "Y"], families={"X": "Fam", "Y": "Fam"})
        assert cfg.get_model_families() == {"Fam": ["X", "Y"]}

    def test_suffix_fallback_and_order(self, minimal_config):
        cfg = _cfg(minimal_config, ["B", "A", "A-r2", "B-r3", "A-r10"])
        assert cfg.get_model_families() == {
            "B": ["B", "B-r3"], "A": ["A", "A-r2", "A-r10"]}

    def test_subset(self, minimal_config):
        cfg = _cfg(minimal_config, ["A", "A-r2", "B"])
        assert cfg.get_model_families(["A-r2", "B"]) == {"A": ["A-r2"], "B": ["B"]}

    def test_default_mode_is_pooled(self, minimal_config):
        assert minimal_config.get_ensemble_mode() == "pooled"

    def test_bad_mode_rejected(self, minimal_config):
        with pytest.raises(ValueError):
            _cfg(minimal_config, ["A"], mode="families").get_ensemble_mode()

    def test_eerie_config_families(self):
        cfg = FeatherConfig.from_yaml("configs/eerie_10_mems_cmip6.yaml")
        assert cfg.get_ensemble_mode() == "per_family"
        fams = cfg.get_model_families()
        assert {f: len(m) for f, m in fams.items()} == {
            "IFS-FESOM2-SR": 3, "IFS-NEMO-ER": 3, "ICON-ESM-ER": 3, "HadGEM3-GC5": 1}


# ── Computation ──────────────────────────────────────────────────────


class _OffsetLoader:
    """Mock loader returning the synthetic field plus a per-model offset."""

    def __init__(self, ds, offsets):
        self._ds, self._offsets = ds, offsets

    def load(self, key):
        return self._ds

    def load_var(self, key, variable):
        model = key.split("_2_", 1)[1].rsplit("_1_0001", 1)[0]
        return self._ds[variable] + self._offsets.get(model, 0.0)

    @staticmethod
    def make_key(experiment, model, domain, member=1):
        from feather.data.loader import DataLoader
        return DataLoader.make_key(experiment, model, domain, member)


@pytest.fixture
def fam_setup(synth_healpix, mock_obs_loader, minimal_config):
    models = ["A", "A-r2", "A-r3", "B"]
    loader = _OffsetLoader(synth_healpix, {"A": 1.0, "A-r2": 2.0, "A-r3": 3.0, "B": -1.0})
    cfg = _cfg(minimal_config, models)
    return loader, mock_obs_loader, cfg


class TestFamilyStats:
    def test_family_means(self, fam_setup):
        loader, obs, cfg = fam_setup
        res = GlobalBiases(loader, obs, cfg, variables=["tas"]).compute()["tas"]
        fam = res["family_data"]["annual"]
        assert list(fam) == ["A", "B"]
        assert fam["A"]["n_members"] == 3 and fam["A"]["members"] == ["A", "A-r2", "A-r3"]
        assert fam["B"]["n_members"] == 1
        models = res["models"]
        # Family mean equals the mean of its members on the common grid.
        expect = (models["A"]["annual_regrid"] + models["A-r2"]["annual_regrid"]
                  + models["A-r3"]["annual_regrid"]) / 3
        np.testing.assert_allclose(fam["A"]["mean"].values, expect.values, rtol=1e-6)
        np.testing.assert_allclose(
            fam["B"]["bias"].values, models["B"]["annual_bias"].values, rtol=1e-6)
        # The +2 K mean offset shows up between the two families' biases.
        assert fam["A"]["bias_gmean"] - fam["B"]["bias_gmean"] == pytest.approx(3.0, abs=0.05)
        assert set(res["family_data"]) == {"annual", "DJF", "MAM", "JJA", "SON"}
        assert res["ens_data"] == {}

    def test_missing_member_dropped_from_count(self, fam_setup):
        loader, obs, cfg = fam_setup
        diag = GlobalBiases(loader, obs, cfg, variables=["tas"])
        res = diag.compute()["tas"]
        mr = dict(res["models"])
        del mr["A-r3"]
        fams = cfg.get_model_families(list(mr))
        out = _families.compute_family_stats(
            mr, fams, res["obs"]["clim"], res["obs"]["seasonal_clim"], None)
        assert out["annual"]["A"]["n_members"] == 2

    def test_pooled_mode_unchanged(self, fam_setup):
        loader, obs, cfg = fam_setup
        cfg = dataclasses.replace(cfg, project={"ensemble_mode": "pooled"})
        res = GlobalBiases(loader, obs, cfg, variables=["tas"]).compute()["tas"]
        assert res["family_data"] == {}
        assert res["ens_data"]["annual"]["n_members"] == 4


# ── Figures and outputs ──────────────────────────────────────────────


class TestFamilyFigures:
    def test_figure_ids_and_panels(self, fam_setup, tmp_path):
        loader, obs, cfg = fam_setup
        saved = GlobalBiases(loader, obs, cfg, variables=["tas"]).run(skip_existing=False)
        names = {p.stem for p, _ in saved}
        for period in ("annual", "djf", "mam", "jja", "son"):
            assert f"tas_{period}_family_mean_bias_combined" in names
            assert f"tas_{period}_ens_bias_combined" not in names
            assert f"tas_{period}_bias_combined" in names
        side = next(j for p, j in saved if p.stem == "tas_annual_family_mean_bias_combined")
        meta = json.loads(side.read_text())
        labels = list(meta["summary_statistics"])
        assert labels == [r"A mean $\mathbf{(3)}$", r"B $\mathbf{(1)}$"]
        assert meta["summary_statistics"][labels[0]]["members"] == ["A", "A-r2", "A-r3"]
        assert meta["model_families"] == {"A": ["A", "A-r2", "A-r3"], "B": ["B"]}
        assert "not pooled" in meta["description"]

    def test_skip_existing_uses_family_ids(self, fam_setup):
        loader, obs, cfg = fam_setup
        diag = GlobalBiases(loader, obs, cfg, variables=["tas"])
        first = diag.run(skip_existing=False)
        again = GlobalBiases(loader, obs, cfg, variables=["tas"]).run(skip_existing=True)
        assert {p.stem for p, _ in again} == {p.stem for p, _ in first}

    def test_netcdf_family_fields(self, fam_setup):
        import xarray as xr

        loader, obs, cfg = fam_setup
        diag = GlobalBiases(loader, obs, cfg, variables=["tas"], save_netcdf=True)
        diag.run(skip_existing=False)
        files = sorted(diag._netcdf_dir.glob("tas_annual_*.nc"))
        assert files
        with xr.open_dataset(files[0]) as ds:
            assert "family_A_mean" in ds and "family_A_mean_bias" in ds
            assert "family_B_mean_bias" in ds
            assert "ens_mean" not in ds
