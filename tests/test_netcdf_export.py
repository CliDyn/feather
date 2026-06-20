"""Tests for diagnostic NetCDF export (shared exporter + CLI wiring)."""

import numpy as np
import pytest
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


def test_ensemble_fields_exported(tmp_path):
    """ens_data mean/median (and their bias) are written to the NetCDF."""
    res = _results()
    res["ens_data"] = {
        "annual": {
            "mean": _field(1.05), "median": _field(1.04),
            "mean_bias": _field(0.05), "median_bias": _field(0.04),
        },
        "DJF": {
            "mean": _field(2.05), "median": _field(2.04),
            "mean_bias": _field(0.05), "median_bias": _field(0.04),
        },
    }
    nx.export_biasmap_netcdf(tmp_path, "tas", res, ("1980", "2014"), units="K")
    ds = xr.open_dataset(tmp_path / "tas_annual_1980-2014.nc")
    for v in ("ens_mean", "ens_mean_bias", "ens_median", "ens_median_bias"):
        assert v in ds, v
    assert float(ds["ens_mean"].mean()) == pytest.approx(1.05)
    ds.close()


def test_no_ensemble_fields_when_absent(tmp_path):
    """No ens_data → no ens_* fields (backward compatible)."""
    nx.export_biasmap_netcdf(tmp_path, "tas", _results(), ("1980", "2014"))
    ds = xr.open_dataset(tmp_path / "tas_annual_1980-2014.nc")
    assert not any(str(v).startswith("ens_") for v in ds.data_vars)
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


# ── Individual benchmark-member export ────────────────────────────────


def _results_individual():
    res = _results()
    res["benchmark_individual_data"] = {
        "CMIP6 MMM": {
            "annual": {
                "ACCESS-CM2/r1i1p1f1": {
                    "regrid": _field(1.4), "bias": _field(0.4),
                },
                "MPI-ESM1-2-HR/r1i1p1f1": {
                    "regrid": _field(1.5), "bias": _field(0.5),
                },
            },
            "DJF": {
                "ACCESS-CM2/r1i1p1f1": {
                    "regrid": _field(2.4), "bias": _field(0.4),
                },
            },
        },
        "HighResMIP MMM": {
            "annual": {
                "ECMWF-IFS-HR/r1i1p1f1": {
                    "regrid": _field(1.6), "bias": _field(0.6),
                },
            },
        },
    }
    return res


def test_individual_filenames(tmp_path):
    paths = nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results_individual(), ("1980", "2014"), units="K",
    )
    names = sorted(p.name for p in paths)
    assert "tas_annual_individual_1980-2014.nc" in names
    assert "tas_DJF_individual_1980-2014.nc" in names


def test_individual_contents(tmp_path):
    nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results_individual(), ("1980", "2014"), units="K",
    )
    ds = xr.open_dataset(tmp_path / "tas_annual_individual_1980-2014.nc")
    # Members named {benchmark-prefix}__{member}; the trailing "MMM" dropped.
    assert "CMIP6__ACCESS_CM2_r1i1p1f1" in ds
    assert "CMIP6__ACCESS_CM2_r1i1p1f1_bias" in ds
    assert "CMIP6__MPI_ESM1_2_HR_r1i1p1f1" in ds
    assert "HighResMIP__ECMWF_IFS_HR_r1i1p1f1" in ds
    # No MMM/obs/model fields leak into the individual file.
    assert "obs" not in ds
    assert "CMIP6_MMM" not in ds
    assert ds.attrs["variable"] == "tas"
    ds.close()


def test_individual_djf_only_has_member_present(tmp_path):
    nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results_individual(), ("1980", "2014"),
    )
    ds = xr.open_dataset(tmp_path / "tas_DJF_individual_1980-2014.nc")
    assert "CMIP6__ACCESS_CM2_r1i1p1f1" in ds
    # HighResMIP had no DJF member → absent in that period file.
    assert "HighResMIP__ECMWF_IFS_HR_r1i1p1f1" not in ds
    ds.close()


def test_individual_scale_applied(tmp_path):
    nx.export_biasmap_individual_netcdf(
        tmp_path, "pr", _results_individual(), ("1980", "2014"),
        units="mm/day", scale=10.0,
    )
    ds = xr.open_dataset(tmp_path / "pr_annual_individual_1980-2014.nc")
    assert float(ds["CMIP6__ACCESS_CM2_r1i1p1f1"].mean()) == 14.0
    ds.close()


def test_individual_skip_existing(tmp_path):
    nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results_individual(), ("1980", "2014"),
    )
    p = tmp_path / "tas_annual_individual_1980-2014.nc"
    mtime = p.stat().st_mtime_ns
    nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results_individual(), ("1980", "2014"),
    )
    assert p.stat().st_mtime_ns == mtime


def test_individual_empty_when_no_data(tmp_path):
    # No "benchmark_individual_data" key → nothing written.
    written = nx.export_biasmap_individual_netcdf(
        tmp_path, "tas", _results(), ("1980", "2014"),
    )
    assert written == []


def test_individual_paths_helper(tmp_path):
    paths = nx.biasmap_individual_netcdf_paths(
        tmp_path, "tas", ("1980", "2014"),
    )
    names = [p.name for p in paths]
    assert names[0] == "tas_annual_individual_1980-2014.nc"
    assert "tas_SON_individual_1980-2014.nc" in names
