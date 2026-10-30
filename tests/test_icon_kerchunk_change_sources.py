"""``icon_kerchunk`` hist/future sources of the climate-change diagnostics.

ICON-ESM-ER r2/r3 have no CMOR output; their hist-1950 and ssp245 runs are
read from the gr025 kerchunk trees.  The loader factories of the three
``*_change`` diagnostics (``precip_extremes`` and ``temp_extremes_indices``
reuse ``TempExtremesChangeDiag._make_fut_loader``) must point an
``ICONKerchunkLoader`` at the configured tree and keep the member.
"""

from types import SimpleNamespace

import pytest

from feather.config import FeatherConfig, ModelConfig
from feather.data.icon_kerchunk_loader import (
    _STORE_FILES,
    ICONKerchunkLoader,
    loader_for_root,
)
from feather.diag.heatwave_change import HeatwaveChangeDiag
from feather.diag.temp_extremes_change import TempExtremesChangeDiag
from feather.diag.tropical_nights_change import TropicalNightsChangeDiag

MODEL = "ICON-ESM-ER-r3"
DIAGS = [TempExtremesChangeDiag, HeatwaveChangeDiag, TropicalNightsChangeDiag]


def _config(hist_root, fut_root=None):
    cfg = FeatherConfig(
        model_catalogs={}, models=[MODEL], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus={},
        output_dir=str(hist_root.parent / "output"),
        data_source={"type": "cmor", "root": "/nonexistent"},
        model_configs={
            MODEL: ModelConfig(name=MODEL, data_source_type="icon_kerchunk",
                               data_root=str(hist_root), member=3),
        },
    )
    cc = {"hist_data_source": "icon_kerchunk", "hist_data_root": str(hist_root)}
    if fut_root is not None:
        cc.update(future_data_source="icon_kerchunk",
                  future_data_root=str(fut_root), future_only_to="2030")
    return cfg, {MODEL: cc}


def _fake_diag(cfg, cc_models):
    return SimpleNamespace(config=cfg, _cc_models=cc_models,
                           fut_period=("2031", "2050"))


def _tree(root):
    (root / "3").mkdir(parents=True)
    (root / "3" / _STORE_FILES["atmos2d_day"]).touch()
    return root


def test_loader_for_root_keeps_member_and_original_config(tmp_path):
    hist, fut = tmp_path / "hist", _tree(tmp_path / "fut")
    cfg, _ = _config(hist)
    ldr = loader_for_root(cfg, MODEL, str(fut))
    assert isinstance(ldr, ICONKerchunkLoader)
    assert ldr._store_path(MODEL, "atmos2d_day").parent == fut / "3"
    assert cfg.model_configs[MODEL].data_root == str(hist)


@pytest.mark.parametrize("diag_cls", DIAGS)
def test_future_loader_reads_scenario_tree(diag_cls, tmp_path):
    hist, fut = _tree(tmp_path / "hist"), _tree(tmp_path / "fut")
    ldr = diag_cls._make_fut_loader(_fake_diag(*_config(hist, fut)), MODEL)
    assert isinstance(ldr, ICONKerchunkLoader)
    assert ldr._store_path(MODEL, "atmos2d_day").parent == fut / "3"


@pytest.mark.parametrize("diag_cls", DIAGS)
def test_hist_loader_reads_hist_tree(diag_cls, tmp_path):
    hist = _tree(tmp_path / "hist")
    ldr = diag_cls._make_hist_loader(_fake_diag(*_config(hist)), MODEL)
    assert isinstance(ldr, ICONKerchunkLoader)
    assert ldr._store_path(MODEL, "atmos2d_day").parent == hist / "3"


@pytest.mark.parametrize("diag_cls", DIAGS)
def test_future_without_root_gives_no_loader(diag_cls, tmp_path):
    cfg, cc = _config(tmp_path / "hist", tmp_path / "fut")
    del cc[MODEL]["future_data_root"]
    assert diag_cls._make_fut_loader(_fake_diag(cfg, cc), MODEL) is None
