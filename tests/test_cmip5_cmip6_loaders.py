"""Tests for the CMIP5 and CMIP6 (NetCDF-tree) loaders (synthetic trees)."""

import numpy as np
import xarray as xr

from feather.config import FeatherConfig, ModelConfig
from feather.data.cmip5_loader import CMIP5Loader
from feather.data.cmip6_nc_loader import CMIP6NCLoader

_LAT = np.array([-2.0, 0.0, 2.0])
_LON = np.array([0.0, 2.0, 4.0])


def _write(path, var, times, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((len(times), len(_LAT), len(_LON)), value, dtype="float32")
    xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": times, "lat": _LAT, "lon": _LON}, name=var,
    ).to_dataset().to_netcdf(path)


def _cfg(tmp_path, model_configs, **ds_extra):
    ds = {"type": "cmip5"}
    ds.update(ds_extra)
    return FeatherConfig(
        model_catalogs={}, models=list(model_configs), obs_root="",
        obs_datasets={}, cmip6={}, dask={}, nereus={},
        output_dir=str(tmp_path), data_source=ds, model_configs=model_configs,
    )


def test_cmip5_stitch_and_latest_version(tmp_path):
    root = tmp_path / "output1"
    gbase = root / "MPI-M" / "MPI-ESM-LR"
    # historical: two versions — the latest must win
    for ver, val in [("v20110101", 270.0), ("v20120101", 280.0)]:
        d = gbase / "historical" / "mon" / "atmos" / "Amon" / "r1i1p1" / ver / "tas"
        t = xr.date_range("2001-01-01", "2005-12-01", freq="MS", calendar="noleap")
        _write(d / "tas_hist.nc", "tas", t, val)
    d2 = gbase / "rcp85" / "mon" / "atmos" / "Amon" / "r1i1p1" / "v20120101" / "tas"
    t2 = xr.date_range("2006-01-01", "2010-12-01", freq="MS", calendar="noleap")
    _write(d2 / "tas_rcp.nc", "tas", t2, 282.0)

    mc = {"G": ModelConfig(
        name="G", institution="MPI-M", gcm="MPI-ESM-LR", variant="r1i1p1",
        experiments=["historical", "rcp85"], data_source_type="cmip5",
    )}
    loader = CMIP5Loader(_cfg(tmp_path, mc, cmip5_root=str(root)))
    da = loader.load_var("G", "tas", period=("2004", "2007"))

    assert set(int(y) for y in da.time.dt.year.values) == {2004, 2005, 2006, 2007}
    assert da.sizes["time"] == 48
    # 2004-2005 from latest historical version (280), 2006-2007 from rcp85 (282)
    assert float(da.isel(time=0).mean()) == 280.0
    assert float(da.isel(time=-1).mean()) == 282.0


def test_cmip6_activity_and_grid(tmp_path):
    root = tmp_path / "CMIP6"
    # historical under CMIP, ssp585 under ScenarioMIP; grid label gn
    hbase = (root / "CMIP" / "CCCma" / "CanESM5" / "historical"
             / "r1i1p1f1" / "Amon" / "tas" / "gn" / "v20190429")
    th = xr.date_range("2001-01-01", "2005-12-01", freq="MS", calendar="noleap")
    _write(hbase / "tas_h.nc", "tas", th, 285.0)
    sbase = (root / "ScenarioMIP" / "CCCma" / "CanESM5" / "ssp585"
             / "r1i1p1f1" / "Amon" / "tas" / "gn" / "v20190429")
    ts = xr.date_range("2015-01-01", "2015-12-01", freq="MS", calendar="noleap")
    _write(sbase / "tas_s.nc", "tas", ts, 288.0)

    mc = {"C": ModelConfig(
        name="C", institution="CCCma", gcm="CanESM5", variant="r1i1p1f1",
        experiments=["historical", "ssp585"], data_source_type="cmip6_nc",
    )}
    loader = CMIP6NCLoader(_cfg(tmp_path, mc, type="cmip6_nc", cmip6_root=str(root)))

    hist = loader.load_var("C", "tas", period=("2001", "2005"))
    assert hist.sizes["time"] == 60
    assert float(hist.isel(time=0).mean()) == 285.0
    # stitched range spans historical + ssp585
    full = loader.load_var("C", "tas")
    assert set(int(y) for y in full.time.dt.year.values) == {2001, 2002, 2003, 2004, 2005, 2015}


def test_cmip6_grid_autodetect(tmp_path):
    """Grid label is auto-detected when not configured."""
    root = tmp_path / "CMIP6"
    base = (root / "CMIP" / "MPI-M" / "MPI-ESM1-2-LR" / "historical"
            / "r1i1p1f1" / "Amon" / "pr" / "gn" / "v20190710")
    t = xr.date_range("2001-01-01", "2001-12-01", freq="MS", calendar="noleap")
    _write(base / "pr.nc", "pr", t, 1.0e-5)
    mc = {"M": ModelConfig(
        name="M", institution="MPI-M", gcm="MPI-ESM1-2-LR", variant="r1i1p1f1",
        experiments=["historical"], data_source_type="cmip6_nc",
    )}
    loader = CMIP6NCLoader(_cfg(tmp_path, mc, type="cmip6_nc", cmip6_root=str(root)))
    da = loader.load_var("M", "pr")
    assert da.sizes["time"] == 12
