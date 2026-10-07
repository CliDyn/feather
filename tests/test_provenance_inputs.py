"""Tests for input (``read``) provenance records from the data loaders."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from feather import provenance
from feather.config import FeatherConfig, ModelConfig
from feather.data.cmor_loader import CMORLoader
from feather.data.obs import ObsLoader


def _write_nc(path: Path, var: str, start: str, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = np.arange(-80.0, 81.0, 40.0)
    lons = np.arange(0.0, 360.0, 90.0)
    xr.Dataset({var: xr.DataArray(
        np.ones((n, lats.size, lons.size)), dims=("time", "lat", "lon"),
        coords={"time": xr.date_range(start, periods=n, freq="MS"),
                "lat": lats, "lon": lons},
    )}).to_netcdf(path)


@pytest.fixture
def cmor_cfg(tmp_path):
    """One model declared ``hist-1950`` whose data actually begins in 1975."""
    root = tmp_path / "CMOR"
    vdir = root / "AWI" / "M" / "hist-1950" / "r2i1p1f1" / "Amon" / "tas" / "gr" / "v1"
    _write_nc(vdir / "tas_1975.nc", "tas", "1975-02", 11)
    _write_nc(vdir / "tas_1976.nc", "tas", "1976-01", 12)
    era = tmp_path / "ERA5"
    _write_nc(era / "ERA5_t2m.nc", "t2m", "1970-01", 24)
    return FeatherConfig(
        model_catalogs={}, models=["M"], obs_root=str(tmp_path),
        obs_datasets={"ERA5": {"path": str(era), "variables": {"t2m": "ERA5_t2m.nc"}}},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path / "out"),
        data_source={"type": "cmor", "root": str(root)},
        model_configs={"M": ModelConfig(
            name="M", institution="AWI", experiment="hist-1950",
            variant="r2i1p1f1", grids={"sfc": "latlon"}, color="#000000",
        )},
    )


def _reads(rec, scope):
    return [e for e in rec.events(scope) if e["step"] == "read"]


# ── Identity ─────────────────────────────────────────────────────────


class TestInputIdentity:
    def test_policies(self, tmp_path):
        f = tmp_path / "a.nc"
        f.write_bytes(b"abc")
        assert provenance.input_identity([str(f)], "none") == {
            "identity": None, "identity_method": "none"}
        stat = provenance.input_identity([str(f)], "stat")
        content = provenance.input_identity([str(f)], "content")
        assert stat["identity"].startswith("stat:")
        assert content["identity"].startswith("content:")
        assert stat["n_bytes"] == 3

    def test_stat_detects_replaced_file(self, tmp_path):
        f = tmp_path / "a.nc"
        f.write_bytes(b"abc")
        before = provenance.input_identity([str(f)], "stat")["identity"]
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert provenance.input_identity([str(f)], "stat")["identity"] != before

    def test_content_ignores_mtime_but_sees_bytes(self, tmp_path):
        f = tmp_path / "a.nc"
        f.write_bytes(b"abc")
        before = provenance.input_identity([str(f)], "content")["identity"]
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert provenance.input_identity([str(f)], "content")["identity"] == before
        f.write_bytes(b"abd")
        assert provenance.input_identity([str(f)], "content")["identity"] != before

    def test_order_independent_and_dirs_walked(self, tmp_path):
        d = tmp_path / "store.zarr"
        (d / "x").mkdir(parents=True)
        (d / "x" / "0").write_bytes(b"1")
        (d / ".zattrs").write_bytes(b"{}")
        a = provenance.input_identity([str(d), str(tmp_path / "missing")], "stat")
        b = provenance.input_identity([str(tmp_path / "missing"), str(d)], "stat")
        assert a == b
        assert a["missing_paths"] == 1

    def test_bad_policy_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            provenance.RunRecorder(tmp_path, {"run_id": "r"}, hash_policy="md5")


# ── CMOR loader ──────────────────────────────────────────────────────


class TestCMORReads:
    def test_read_record(self, cmor_cfg):
        loader = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            with provenance.diagnostic_scope("d"):
                loader.load_var("M", "tas", period=("1980", "2014"))
        (ev,) = _reads(rec, "d")
        assert ev["role"] == "model"
        assert ev["backend"] == "CMORLoader"
        assert ev["variable"] == "tas"
        assert ev["name"] == "M"
        assert ev["variant"] == "r2i1p1f1"
        assert ev["n_files"] == 2
        assert ev["first"].endswith("tas_1975.nc")
        assert ev["last"].endswith("tas_1976.nc")
        assert ev["identity"].startswith("stat:")
        assert ev["period_requested"] == ["1980", "2014"]
        assert ev["scale_factor"] == 1.0

    def test_declared_experiment_vs_actual_coverage(self, cmor_cfg):
        """The EERIE r2/r3 case: declared hist-1950, data from 1975."""
        loader = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            loader.load_var("M", "tas")
        (ev,) = _reads(rec, "_run")
        assert ev["declared_experiment"] == "hist-1950"
        assert ev["time_coverage"] == ["1975-02-01", "1976-12-01"]
        assert ev["n_times"] == 23

    def test_cache_hit_still_recorded_per_diagnostic(self, cmor_cfg):
        loader = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            with provenance.diagnostic_scope("a"):
                loader.load_var("M", "tas")
            with provenance.diagnostic_scope("b"):
                loader.load_var("M", "tas")  # cached
        assert _reads(rec, "a") and _reads(rec, "b")
        assert _reads(rec, "a")[0]["identity"] == _reads(rec, "b")[0]["identity"]

    def test_unchanged_outside_run(self, cmor_cfg):
        loader = CMORLoader(cmor_cfg)
        da = loader.load_var("M", "tas", period=("1976", "1976"))
        assert da.sizes["time"] == 12
        assert loader._prov == {}

    def test_data_identical_inside_run(self, cmor_cfg):
        out = CMORLoader(cmor_cfg).load_var("M", "tas").values
        with provenance.run_scope(cmor_cfg):
            inside = CMORLoader(cmor_cfg).load_var("M", "tas").values
        np.testing.assert_array_equal(out, inside)

    def test_hash_none(self, cmor_cfg):
        loader = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg, hash_policy="none") as rec:
            loader.load_var("M", "tas")
        ev = _reads(rec, "_run")[0]
        assert ev["identity"] is None and ev["identity_method"] == "none"


# ── Obs loader ───────────────────────────────────────────────────────


class TestObsReads:
    def test_tagged_with_model_variable(self, cmor_cfg):
        obs = ObsLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            obs.load_for_model_var("tas", period=("1970", "1970"))
        (ev,) = _reads(rec, "_run")
        assert ev["role"] == "obs"
        assert ev["variable"] == "tas"
        assert ev["obs_variable"] == "t2m"
        assert ev["dataset"] == "ERA5"
        assert ev["first"].endswith("ERA5_t2m.nc")
        assert ev["time_coverage"] == ["1970-01-01", "1971-12-01"]

    def test_obs_policy_separate(self, cmor_cfg):
        obs = ObsLoader(cmor_cfg)
        model = CMORLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg, hash_policy="stat",
                                  hash_policy_obs="content") as rec:
            obs.load("ERA5", "t2m")
            model.load_var("M", "tas")
        by_role = {e["role"]: e for e in _reads(rec, "_run")}
        assert by_role["obs"]["identity_method"] == "content"
        assert by_role["model"]["identity_method"] == "stat"
        record = json.loads(rec.record_path.read_text())
        assert record["hash_policy"] == {"default": "stat", "obs": "content"}

    def test_direct_load_untagged(self, cmor_cfg):
        """Direct ``load`` calls apply to every figure of the diagnostic."""
        obs = ObsLoader(cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            obs.load("ERA5", "t2m")
        assert "variable" not in _reads(rec, "_run")[0]

    def test_sign_flip_recorded(self, cmor_cfg, monkeypatch):
        from feather.diag.base import DiagnosticBase

        class _D(DiagnosticBase):
            name, title, domain, variables, group = "d", "D", "sfc", ["hfss"], "g"

            def compute(self):
                return {}

            def plot(self, results):
                return []

        class _Obs:
            def load_for_model_var(self, var, period=None):
                return xr.DataArray([1.0, 2.0])

        diag = _D(None, _Obs(), cmor_cfg)
        with provenance.run_scope(cmor_cfg) as rec:
            out = diag._load_obs_var("hfss")
        np.testing.assert_array_equal(out.values, [-1.0, -2.0])
        ev = rec.events("_run")[0]
        assert ev["step"] == "convert"
        assert ev["op"] == "cmor_sign_convention" and ev["factor"] == -1.0


# ── Kerchunk loaders ─────────────────────────────────────────────────


class TestKerchunkReads:
    def test_icon_store_recorded(self, tmp_path, monkeypatch):
        from tests.test_icon_kerchunk_loader import MODEL, _make_loader

        loader, _ = _make_loader(tmp_path, monkeypatch)
        store = tmp_path / "stores" / "2" / "erc2023_atmos_native_2d_monthly_mean_remap025.parq"
        store.mkdir(parents=True)
        (store / "refs.0.parq").write_bytes(b"x")
        cfg = loader._config
        with provenance.run_scope(cfg) as rec:
            loader.load_var(MODEL, "tas")
        (ev,) = _reads(rec, "_run")
        assert ev["backend"] == "ICONKerchunkLoader"
        assert ev["first"] == str(store)
        assert ev["declared_experiment"] == "hist-1950"
        assert ev["n_times"] == 6

    def test_missing_store_degrades_to_error_entry(self, tmp_path, monkeypatch):
        """A provenance failure must not fail the load."""
        from tests.test_icon_kerchunk_loader import MODEL, _make_loader

        loader, _ = _make_loader(tmp_path, monkeypatch)
        with provenance.run_scope(loader._config) as rec:
            da = loader.load_var(MODEL, "tas")
        assert da.sizes["time"] == 6
        (ev,) = _reads(rec, "_run")
        assert "FileNotFoundError" in ev["error"]


# ── Stitched experiments ─────────────────────────────────────────────


def test_stitched_experiments_record_every_segment(tmp_path):
    """Loaders that concatenate experiments must list all segments' files."""
    from tests.test_cmip5_cmip6_loaders import _cfg, _write
    from feather.data.cmip5_loader import CMIP5Loader

    root = tmp_path / "output1"
    gbase = root / "MPI-M" / "MPI-ESM-LR"
    d1 = gbase / "historical" / "mon" / "atmos" / "Amon" / "r1i1p1" / "v1" / "tas"
    _write(d1 / "tas_hist.nc", "tas",
           xr.date_range("2001-01-01", "2005-12-01", freq="MS", calendar="noleap"), 280.0)
    d2 = gbase / "rcp85" / "mon" / "atmos" / "Amon" / "r1i1p1" / "v1" / "tas"
    _write(d2 / "tas_rcp.nc", "tas",
           xr.date_range("2006-01-01", "2010-12-01", freq="MS", calendar="noleap"), 282.0)
    mc = {"G": ModelConfig(
        name="G", institution="MPI-M", gcm="MPI-ESM-LR", variant="r1i1p1",
        experiments=["historical", "rcp85"], data_source_type="cmip5",
    )}
    cfg = _cfg(tmp_path, mc, cmip5_root=str(root))
    loader = CMIP5Loader(cfg)
    with provenance.run_scope(cfg) as rec:
        loader.load_var("G", "tas", period=("2004", "2007"))
    (ev,) = _reads(rec, "_run")
    assert ev["n_files"] == 2
    assert ev["time_coverage"] == ["2001-01-01", "2010-12-01"]
    assert "missing_months" not in ev
