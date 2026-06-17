"""Tests for the --benchmarks runtime selection in _run_diagnostics."""

from unittest.mock import MagicMock, patch

from feather.config import FeatherConfig


def _cfg(tmp_path):
    return FeatherConfig(
        model_catalogs={},
        models=["ifs-fesom"],
        obs_root="",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir=str(tmp_path / "out"),
        benchmarks=[
            {"name": "CMIP6", "label": "CMIP6 MMM",
             "zarr_dir": str(tmp_path / "c"), "experiment": "historical",
             "models": "auto"},
            {"name": "HighResMIP", "label": "HighResMIP MMM",
             "zarr_dir": str(tmp_path / "h"), "experiment": "hist-1950",
             "models": "auto"},
        ],
    )


def _run_with_benchmarks(cfg, benchmarks):
    """Run _run_diagnostics with everything mocked; capture loader cfgs."""
    made = []

    class _FakeLoader:
        def __init__(self, config, cmip6_cfg=None):
            made.append(cmip6_cfg)
            self.label = cmip6_cfg.get("label")

    mock_inst = MagicMock()
    mock_inst.run.return_value = []
    mock_cls = MagicMock(return_value=mock_inst)

    with patch("feather.data.obs.ObsLoader"), \
         patch("feather.data.cmip6.CMIP6Loader", _FakeLoader), \
         patch("feather.diag.registry.get_diagnostic", return_value=mock_cls), \
         patch("feather.diag.registry.registered_names",
               return_value=["global_biases"]), \
         patch("feather.run._create_model_loader"):
        from feather.run import _run_diagnostics
        _run_diagnostics(cfg, benchmarks=benchmarks)
    return [c.get("name") for c in made]


def test_select_single_benchmark(tmp_path):
    assert _run_with_benchmarks(_cfg(tmp_path), ["CMIP6"]) == ["CMIP6"]


def test_select_highresmip_case_insensitive(tmp_path):
    assert _run_with_benchmarks(_cfg(tmp_path), ["highresmip"]) == ["HighResMIP"]


def test_select_by_label(tmp_path):
    assert _run_with_benchmarks(_cfg(tmp_path), ["HighResMIP MMM"]) == [
        "HighResMIP"
    ]


def test_none_selects_all(tmp_path):
    assert _run_with_benchmarks(_cfg(tmp_path), None) == ["CMIP6", "HighResMIP"]


def test_unknown_benchmark_selects_none(tmp_path):
    assert _run_with_benchmarks(_cfg(tmp_path), ["NoSuch"]) == []
