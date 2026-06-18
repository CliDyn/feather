"""Tests for FeatherConfig.get_benchmarks() (multi-benchmark + legacy)."""

from feather.config import FeatherConfig


def _cfg(**overrides):
    base = dict(
        model_catalogs={},
        models=["m"],
        obs_root="/obs",
        obs_datasets={},
        cmip6={"enabled": False},
        dask={},
        nereus={},
        output_dir="/out",
    )
    base.update(overrides)
    return FeatherConfig(**base)


def test_no_benchmarks_returns_empty():
    assert _cfg().get_benchmarks() == []


def test_legacy_cmip6_block_wrapped():
    cfg = _cfg(cmip6={"enabled": True, "catalog_path": "/x.yaml",
                      "models": {"CanESM5": {"variant": "r1i1p1f1"}}})
    benches = cfg.get_benchmarks()
    assert len(benches) == 1
    assert benches[0]["label"] == "CMIP6 MMM"
    assert benches[0]["name"] == "CMIP6"


def test_benchmarks_list_takes_priority_and_filters_disabled():
    cfg = _cfg(
        cmip6={"enabled": True},  # ignored when benchmarks: present
        benchmarks=[
            {"name": "CMIP6", "label": "CMIP6 MMM",
             "zarr_dir": "/z/cmip6", "experiment": "historical",
             "models": "auto"},
            {"name": "HighResMIP", "zarr_dir": "/z/hr",
             "experiment": "hist-1950", "models": "auto"},
            {"name": "Disabled", "enabled": False, "zarr_dir": "/z/d"},
        ],
    )
    benches = cfg.get_benchmarks()
    assert [b["name"] for b in benches] == ["CMIP6", "HighResMIP"]
    # default label derived from name when omitted
    assert benches[1]["label"] == "HighResMIP MMM"


def test_from_yaml_parses_benchmarks(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump({
        "models": ["m"], "obs_root": "/obs", "obs_datasets": {},
        "dask": {}, "nereus": {}, "output_dir": "/out",
        "benchmarks": [
            {"name": "CMIP6", "zarr_dir": "/z", "experiment": "historical",
             "models": "auto"},
        ],
    }))
    cfg = FeatherConfig.from_yaml(str(p))
    assert len(cfg.get_benchmarks()) == 1
    assert cfg.get_benchmarks()[0]["name"] == "CMIP6"
