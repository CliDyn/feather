"""Tests for diagnostic NetCDF export (shared exporter + CLI wiring)."""

import numpy as np
import xarray as xr

from feather.diag import netcdf_export as nx


def _field(val=0.0):
    lat = np.linspace(-89, 89, 10)
    lon = np.linspace(0, 350, 12)
    return xr.DataArray(
        np.full((10, 12), val), dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


def _results():
    return {
        "obs": {
            "clim": _field(1.0),
            "seasonal_clim": {"DJF": _field(2.0)},
        },
        "models": {
            "IFS-FESOM2-SR": {
                "annual_regrid": _field(1.1), "annual_bias": _field(0.1),
                "seasonal_regrids": {"DJF": _field(2.1)},
                "seasonal_biases": {"DJF": _field(0.1)},
            },
        },
        "benchmark_data": {
            "CMIP6 MMM": {
                "annual": {"regrid": _field(1.2), "bias": _field(0.2)},
                "DJF": {"regrid": _field(2.2), "bias": _field(0.2)},
            },
            "HighResMIP MMM": {
                "annual": {"regrid": _field(1.3), "bias": _field(0.3)},
            },
        },
    }


def test_sanitize_name():
    assert nx.sanitize_name("IFS-FESOM2-SR") == "IFS_FESOM2_SR"
    assert nx.sanitize_name("CMIP6 MMM") == "CMIP6_MMM"


def test_filenames_contain_period(tmp_path):
    paths = nx.export_biasmap_netcdf(
        tmp_path, "tas", _results(), ("1980", "2014"), units="K",
    )
    names = sorted(p.name for p in paths)
    assert "tas_annual_1980-2014.nc" in names
    assert "tas_DJF_1980-2014.nc" in names


def test_export_contents(tmp_path):
    nx.export_biasmap_netcdf(
        tmp_path, "tas", _results(), ("1980", "2014"), units="K",
    )
    ds = xr.open_dataset(tmp_path / "tas_annual_1980-2014.nc")
    # Sources present (sanitized names)
    assert "obs" in ds
    assert "IFS_FESOM2_SR" in ds and "IFS_FESOM2_SR_bias" in ds
    assert "CMIP6_MMM" in ds and "CMIP6_MMM_bias" in ds
    assert "HighResMIP_MMM" in ds
    # Period recorded in attrs
    assert ds.attrs["period"] == "1980-2014"
    assert ds.attrs["period_start"] == "1980"
    assert ds.attrs["variable"] == "tas"
    ds.close()


def test_scale_applied(tmp_path):
    nx.export_biasmap_netcdf(
        tmp_path, "pr", _results(), ("1980", "2014"),
        units="mm/day", scale=10.0,
    )
    ds = xr.open_dataset(tmp_path / "pr_annual_1980-2014.nc")
    # obs was 1.0 → 10.0 after scale
    assert float(ds["obs"].mean()) == 10.0
    assert ds.attrs["units"] == "mm/day"
    ds.close()


def test_skip_existing(tmp_path):
    nx.export_biasmap_netcdf(tmp_path, "tas", _results(), ("1980", "2014"))
    p = tmp_path / "tas_annual_1980-2014.nc"
    mtime = p.stat().st_mtime_ns
    # Second call should not rewrite
    nx.export_biasmap_netcdf(tmp_path, "tas", _results(), ("1980", "2014"))
    assert p.stat().st_mtime_ns == mtime
