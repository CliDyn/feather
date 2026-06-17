"""Shared NetCDF export for climatology/bias map diagnostics.

Writes the per-source fields underlying the bias-map figures (observations,
each evaluated model, and each benchmark MMM — CMIP6, HighResMIP, …) to
NetCDF, one file per (variable, period).  The analysis period appears in both
the filename and the global attributes.

Diagnostics with the standard bias-map result structure
(``models`` → ``annual_regrid``/``annual_bias``/``seasonal_regrids``/
``seasonal_biases``; ``obs`` → ``clim``/``seasonal_clim``; ``benchmark_data``
→ ``{label: {period: {regrid, bias}}}``) can reuse :func:`export_biasmap_netcdf`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

_SEASONS = ("DJF", "MAM", "JJA", "SON")


def sanitize_name(name: str) -> str:
    """Make a NetCDF-friendly variable name (alnum + underscore)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")


def _nc_filename(var: str, period_key: str, period) -> str:
    return f"{var}_{period_key}_{period[0]}-{period[1]}.nc"


def biasmap_netcdf_paths(
    netcdf_dir: Path, var: str, period: tuple[str, str],
    *, seasons: bool = True,
) -> list[Path]:
    """Expected NetCDF paths for a variable (annual + seasonal)."""
    keys = ["annual"] + (list(_SEASONS) if seasons else [])
    return [netcdf_dir / _nc_filename(var, k, period) for k in keys]


def _period_fields(results: dict, period_key: str) -> dict[str, xr.DataArray]:
    """Collect obs/model/benchmark fields for one period from a result dict."""
    fields: dict[str, xr.DataArray] = {}

    # Observation climatology
    if period_key == "annual":
        obs = results.get("obs", {}).get("clim")
    else:
        obs = results.get("obs", {}).get("seasonal_clim", {}).get(period_key)
    if obs is not None:
        fields["obs"] = obs

    # Evaluated models (regridded field + bias vs obs)
    for model, mdata in results.get("models", {}).items():
        if period_key == "annual":
            regrid = mdata.get("annual_regrid")
            bias = mdata.get("annual_bias")
        else:
            regrid = mdata.get("seasonal_regrids", {}).get(period_key)
            bias = mdata.get("seasonal_biases", {}).get(period_key)
        key = sanitize_name(model)
        if regrid is not None:
            fields[key] = regrid
        if bias is not None:
            fields[f"{key}_bias"] = bias

    # Benchmark MMMs (CMIP6, HighResMIP, …)
    for label, bdata in results.get("benchmark_data", {}).items():
        period_data = bdata.get(period_key)
        if not period_data:
            continue
        key = sanitize_name(label)
        if period_data.get("regrid") is not None:
            fields[key] = period_data["regrid"]
        if period_data.get("bias") is not None:
            fields[f"{key}_bias"] = period_data["bias"]

    return fields


def export_biasmap_netcdf(
    netcdf_dir: Path,
    var: str,
    results: dict,
    period: tuple[str, str],
    *,
    units: str = "",
    scale: float = 1.0,
    skip_existing: bool = True,
    extra_attrs: dict | None = None,
) -> list[Path]:
    """Write one NetCDF per period for a bias-map diagnostic result.

    Each file holds the obs climatology, every evaluated model's regridded
    climatology and bias, and every benchmark MMM's climatology and bias, all
    on the common analysis grid.  Files already present are skipped when
    *skip_existing* is True.

    Returns the list of written (or already-present) paths.
    """
    netcdf_dir = Path(netcdf_dir)
    written: list[Path] = []

    period_keys = ["annual"]
    seasonal_clim = results.get("obs", {}).get("seasonal_clim", {})
    period_keys += [s for s in _SEASONS if s in seasonal_clim]

    for period_key in period_keys:
        path = netcdf_dir / _nc_filename(var, period_key, period)
        if skip_existing and path.exists():
            written.append(path)
            continue

        fields = _period_fields(results, period_key)
        if not fields:
            continue

        if scale != 1.0:
            fields = {k: v * scale for k, v in fields.items()}

        netcdf_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(fields)
        ds.attrs.update(
            variable=var,
            period_key=period_key,
            period_start=str(period[0]),
            period_end=str(period[1]),
            period=f"{period[0]}-{period[1]}",
            units=units,
            description=(
                "Per-source climatology and bias (model/benchmark minus obs) "
                "on the common analysis grid. Sources: obs, evaluated models, "
                "and benchmark MMMs (CMIP6, HighResMIP)."
            ),
        )
        if extra_attrs:
            ds.attrs.update(extra_attrs)
        ds.to_netcdf(path)
        logger.info("  Wrote NetCDF: %s", path.name)
        written.append(path)

    return written
