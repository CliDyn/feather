"""Tests for cross-experiment time stitching (DestinE + CMIP6).

The DestinE catalog splits each model across separate experiments
(e.g. ``baseline_hist`` 1990-2014 + ``projections_ssp3-7.0`` 2015+), and
the CMIP6 zarr store likewise splits ``historical`` + ``ssp370``. These
tests verify the segments are concatenated along time, de-duplicated and
sliced to the requested period, and that single-experiment configs are
unaffected.
"""

import numpy as np
import pytest
import xarray as xr

from feather.config import FeatherConfig
from feather.diag.base import DiagnosticBase


# ── Helpers ──────────────────────────────────────────────────────────


class _ConcreteDiag(DiagnosticBase):
    """Trivial concrete diagnostic to exercise base-class load helpers."""

    name = "stitch_probe"
    title = "Stitch Probe"
    domain = "sfc"
    variables = ["tas"]
    group = "evaluation"

    def compute(self):
        return {}

    def plot(self, results):
        return []


def _hp_segment(start: str, periods: int):
    """Tiny HEALPix-like segment: monthly ``tas`` on a ``values`` dim."""
    time = xr.date_range(start, periods=periods, freq="MS", calendar="standard")
    ncell = 12
    data = np.broadcast_to(
        np.arange(periods)[:, None] + 250.0, (periods, ncell)
    ).copy()
    return xr.Dataset(
        {
            "avg_2t": xr.DataArray(
                data, dims=("time", "values"), coords={"time": time}
            ),
            "longitude": xr.DataArray(np.linspace(0, 330, ncell), dims="values"),
            "latitude": xr.DataArray(np.linspace(-80, 80, ncell), dims="values"),
        }
    )


class _MultiExpLoader:
    """Mock DestinE loader returning a different segment per experiment key.

    ``segments`` maps an experiment token (substring of the catalog key)
    to a Dataset. Keys whose experiment token is absent raise ``KeyError``
    to emulate a model that did not run that experiment.
    """

    def __init__(self, segments: dict[str, xr.Dataset]):
        self._segments = segments

    def _match(self, key: str) -> xr.Dataset:
        for token, ds in self._segments.items():
            if key.startswith(token + "_") or f"{token}_2_" in key:
                return ds
        raise KeyError(key)

    def load(self, key: str) -> xr.Dataset:
        return self._match(key)

    def load_var(self, key: str, variable: str) -> xr.DataArray:
        return self._match(key)[variable]


def _stitch_config(experiments, tmp_path, period=("1990", "2025")):
    return FeatherConfig(
        project={"experiments": experiments, "period": list(period)},
        model_catalogs={"2d": "x"},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={"influence_radius": 1_000_000},
        output_dir=str(tmp_path / "output"),
    )


# ── Config helpers ───────────────────────────────────────────────────


def test_get_experiments_list(tmp_path):
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    assert cfg.get_experiments() == ["baseline_hist", "projections_ssp3-7.0"]


def test_get_experiments_fallback_single(tmp_path):
    cfg = FeatherConfig(
        project={"experiment": "baseline_cont"},
        model_catalogs={}, models=["ifs-fesom"], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path),
    )
    assert cfg.get_experiments() == ["baseline_cont"]


def test_timeseries_period_override(tmp_path):
    cfg = FeatherConfig(
        project={"period": ["1990", "2025"],
                 "timeseries_period": ["1990", "2050"]},
        model_catalogs={}, models=["ifs-fesom"], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path),
    )
    assert cfg.get_timeseries_period() == ("1990", "2050")
    assert cfg.get_period() == ("1990", "2025")


def test_timeseries_period_defaults_to_period(tmp_path):
    cfg = FeatherConfig(
        project={"period": ["1990", "2025"]},
        model_catalogs={}, models=["ifs-fesom"], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(tmp_path),
    )
    assert cfg.get_timeseries_period() == ("1990", "2025")


# ── DestinE stitching ────────────────────────────────────────────────


def _make_diag(loader, cfg):
    from tests.conftest import MockObsLoader

    obs = MockObsLoader(
        xr.Dataset({"t2m": xr.DataArray([0.0], dims="x")}), var_name="t2m"
    )
    return _ConcreteDiag(loader, obs, cfg)


def test_stitch_concatenates_two_experiments(tmp_path):
    loader = _MultiExpLoader({
        "baseline_hist": _hp_segment("1990-01", 300),          # 1990-2014
        "projections_ssp3-7.0": _hp_segment("2015-01", 420),   # 2015-2049
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas")  # no period -> full range
    assert da.sizes["time"] == 720
    years = da["time"].dt.year.values
    assert years.min() == 1990 and years.max() == 2049
    # strictly increasing, no duplicates
    assert (np.diff(da["time"].values.astype("datetime64[ns]")) > np.timedelta64(0)).all()


def test_stitch_slices_to_period(tmp_path):
    loader = _MultiExpLoader({
        "baseline_hist": _hp_segment("1990-01", 300),
        "projections_ssp3-7.0": _hp_segment("2015-01", 420),
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas", period=("1990", "2025"))
    years = da["time"].dt.year.values
    assert years.min() == 1990 and years.max() == 2025
    assert da.sizes["time"] == (2025 - 1990 + 1) * 12


def test_stitch_dedupes_overlap(tmp_path):
    # Overlapping month (2014-12 present in both segments) kept once.
    loader = _MultiExpLoader({
        "baseline_hist": _hp_segment("1990-01", 300),         # ...2014-12
        "projections_ssp3-7.0": _hp_segment("2014-12", 60),   # 2014-12...
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas")
    times = da["time"].values
    assert len(times) == len(np.unique(times))  # no duplicate timestamps


def test_stitch_skips_missing_experiment(tmp_path):
    # Model has no projection segment -> falls back to historical only.
    loader = _MultiExpLoader({
        "baseline_hist": _hp_segment("1990-01", 300),
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas")
    assert da["time"].dt.year.values.max() == 2014


def test_stitch_raises_when_all_missing(tmp_path):
    loader = _MultiExpLoader({"something_else": _hp_segment("1990-01", 12)})
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    with pytest.raises(KeyError):
        diag._load_model_var("ifs-fesom", "tas")


def test_single_experiment_no_concat(tmp_path):
    loader = _MultiExpLoader({"baseline_hist": _hp_segment("1990-01", 300)})
    cfg = _stitch_config(["baseline_hist"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas")
    assert da.sizes["time"] == 300


def test_explicit_experiment_arg_bypasses_stitch(tmp_path):
    loader = _MultiExpLoader({
        "baseline_hist": _hp_segment("1990-01", 300),
        "projections_ssp3-7.0": _hp_segment("2015-01", 420),
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    da = diag._load_model_var("ifs-fesom", "tas", experiment="baseline_hist")
    assert da.sizes["time"] == 300


def test_stitch_coords_use_first_available(tmp_path):
    loader = _MultiExpLoader({
        "projections_ssp3-7.0": _hp_segment("2015-01", 12),
    })
    cfg = _stitch_config(["baseline_hist", "projections_ssp3-7.0"], tmp_path)
    diag = _make_diag(loader, cfg)

    lon, lat = diag._load_model_coords("ifs-fesom", "avg_2t")
    assert lon.shape == (12,) and lat.shape == (12,)


# ── CMIP6 stitching ──────────────────────────────────────────────────


def test_cmip6_get_experiments(tmp_path):
    from feather.data.cmip6 import CMIP6Loader

    cfg = FeatherConfig(
        project={}, model_catalogs={}, models=[], obs_root="", obs_datasets={},
        cmip6={"enabled": True, "experiments": ["historical", "ssp370"],
               "catalog_path": "/nonexistent.yaml", "models": {}},
        dask={}, nereus={}, output_dir=str(tmp_path),
    )
    loader = CMIP6Loader(cfg)
    assert loader._get_experiments() == ["historical", "ssp370"]


def test_cmip6_get_experiments_default(tmp_path):
    from feather.data.cmip6 import CMIP6Loader

    cfg = FeatherConfig(
        project={}, model_catalogs={}, models=[], obs_root="", obs_datasets={},
        cmip6={"enabled": True, "catalog_path": "/nonexistent.yaml",
               "models": {}},
        dask={}, nereus={}, output_dir=str(tmp_path),
    )
    loader = CMIP6Loader(cfg)
    assert loader._get_experiments() == ["historical"]


def test_cmip6_zarr_path_experiment(tmp_path):
    from feather.data.cmip6 import CMIP6Loader

    cfg = FeatherConfig(
        project={}, model_catalogs={}, models=[], obs_root="", obs_datasets={},
        cmip6={"enabled": True, "catalog_path": "/nonexistent.yaml",
               "models": {}},
        dask={}, nereus={}, output_dir=str(tmp_path),
    )
    loader = CMIP6Loader(cfg)
    hist = loader._zarr_path("MRI-ESM2-0", "r1i1p1f1", "Amon", "tas")
    ssp = loader._zarr_path("MRI-ESM2-0", "r1i1p1f1", "Amon", "tas",
                            experiment="ssp370")
    assert hist.endswith("MRI-ESM2-0_historical_r1i1p1f1_Amon_tas.zarr")
    assert ssp.endswith("MRI-ESM2-0_ssp370_r1i1p1f1_Amon_tas.zarr")


def test_cmip6_open_stitched_concats(tmp_path, monkeypatch):
    """_open_stitched concatenates historical+ssp370 zarr segments."""
    from feather.data import cmip6 as cmip6_mod
    from feather.data.cmip6 import CMIP6Loader

    cfg = FeatherConfig(
        project={}, model_catalogs={}, models=[], obs_root="", obs_datasets={},
        cmip6={"enabled": True, "experiments": ["historical", "ssp370"],
               "catalog_path": "/nonexistent.yaml", "models": {}},
        dask={}, nereus={}, output_dir=str(tmp_path),
    )
    loader = CMIP6Loader(cfg)

    def _seg(start, periods):
        time = xr.date_range(start, periods=periods, freq="MS",
                             calendar="standard")
        return xr.Dataset({"tas": xr.DataArray(
            np.ones((periods, 2, 2)), dims=("time", "lat", "lon"),
            coords={"time": time, "lat": [0, 1], "lon": [0, 1]})})

    segs = {"historical": _seg("1990-01", 300),
            "ssp370": _seg("2015-01", 360)}

    monkeypatch.setattr(cmip6_mod.os.path, "exists", lambda p: True)

    def fake_open(path, **kw):
        return segs["ssp370"] if "ssp370" in path else segs["historical"]

    monkeypatch.setattr(cmip6_mod.xr, "open_zarr", fake_open)

    da = loader._open_stitched("tas", "MRI-ESM2-0", "r1i1p1f1", "Amon")
    assert da.sizes["time"] == 660
    assert da["time"].dt.year.values.min() == 1990
    assert da["time"].dt.year.values.max() == 2044


def test_cmip6_open_stitched_falls_back_to_historical(tmp_path, monkeypatch):
    """Missing ssp370 zarr -> historical-only (ends 2014)."""
    from feather.data import cmip6 as cmip6_mod
    from feather.data.cmip6 import CMIP6Loader

    cfg = FeatherConfig(
        project={}, model_catalogs={}, models=[], obs_root="", obs_datasets={},
        cmip6={"enabled": True, "experiments": ["historical", "ssp370"],
               "catalog_path": "/nonexistent.yaml", "models": {}},
        dask={}, nereus={}, output_dir=str(tmp_path),
    )
    loader = CMIP6Loader(cfg)

    time = xr.date_range("1990-01", periods=300, freq="MS", calendar="standard")
    hist = xr.Dataset({"tas": xr.DataArray(
        np.ones((300, 2, 2)), dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0, 1], "lon": [0, 1]})})

    monkeypatch.setattr(cmip6_mod.os.path, "exists",
                        lambda p: "ssp370" not in p)
    monkeypatch.setattr(cmip6_mod.xr, "open_zarr", lambda path, **kw: hist)

    da = loader._open_stitched("tas", "MRI-ESM2-0", "r1i1p1f1", "Amon")
    assert da.sizes["time"] == 300
