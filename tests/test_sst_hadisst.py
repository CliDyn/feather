"""Structural tests for the sst_hadisst diagnostic (no real data).

End-to-end trend/Taylor computation needs real model + HadISST data and is
exercised on a compute node; here we check registration, the OceanSST
extension hook, and the obs retargeting.
"""

import pytest


def test_sst_hadisst_registered():
    import feather.diag  # noqa: F401  (triggers registration)
    from feather.diag.registry import list_diagnostics
    names = [d["name"] if isinstance(d, dict) else getattr(d, "name", str(d))
             for d in list_diagnostics()]
    assert "sst_hadisst" in names


def test_obs_retargeting_attrs():
    from feather.diag.sst_hadisst import SSTHadISST
    assert SSTHadISST._obs_label == "HadISST"
    assert SSTHadISST._obs_dataset_name == "HadISST"
    assert SSTHadISST._ocean_bias_obs == "HADISST"
    assert SSTHadISST.variables == ["tos"]
    assert SSTHadISST.domain == "o2d"


def test_extra_figure_ids():
    from feather.diag.sst_hadisst import SSTHadISST
    ids = SSTHadISST._extra_figure_ids(SSTHadISST)
    assert ids == [
        "sst_trend_combined", "sst_trend_arctic", "sst_trend_antarctic",
        "sst_taylor",
    ]


def test_trend_and_taylor_methods_present():
    from feather.diag.sst_hadisst import SSTHadISST
    for m in ("_compute_trends", "_plot_trends", "_plot_polar_trend",
              "_compute_taylor", "_plot_taylor", "_compute_cmip6_trends"):
        assert hasattr(SSTHadISST, m), m


def test_oceansst_hook_defaults_noop():
    # Base OceanSST must not add extra figures (backward compatible).
    from feather.diag.ocean_sst import OceanSST
    assert OceanSST._extra_figure_ids(OceanSST) == []
    # _extra_groups is a plain method returning [] for the base class.
    assert OceanSST._extra_groups(
        OceanSST, {}, {}, True) == []


def test_reuses_temperature_berkeley_helpers():
    import feather.diag.sst_hadisst as m
    from feather.diag.temperature_berkeley import TemperatureBerkeley
    assert m._grid_signature is TemperatureBerkeley._grid_signature
    assert m._pattern_correlation is TemperatureBerkeley._pattern_correlation
    assert m._std_ratio is TemperatureBerkeley._std_ratio
