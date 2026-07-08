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
import numbers
import re
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_SEASONS = ("DJF", "MAM", "JJA", "SON")

#: Attribute value types that serialize cleanly to NetCDF (mirrors xarray's
#: own ``_validate_attrs`` check).
_VALID_ATTR_TYPES = (str, numbers.Number, np.ndarray, np.number, list, tuple, bytes)


#: Encoding keys that describe on-disk packing inherited from a source file
#: (e.g. HadISST stored as int16 with scale_factor/_FillValue). Derived export
#: fields are floats and may contain NaN, so re-encoding them into an integer
#: on-disk dtype raises "cannot convert float NaN to integer" — drop these and
#: let xarray write the in-memory float dtype instead.
_PACKING_ENCODING_KEYS = (
    "dtype", "_FillValue", "missing_value", "scale_factor", "add_offset",
)


def _sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Drop attrs whose values can't serialize to NetCDF (e.g. GRIB/earthkit
    dict attrs like ``_earthkit={'bitsPerValue': 24}``) and strip inherited
    integer-packing encoding that breaks float/NaN fields.

    Cleans the dataset's own attrs plus every variable's and coordinate's
    attrs and encoding. Mutates *ds* in place and returns it.
    """
    def clean(attrs: dict) -> None:
        for k in [k for k, v in attrs.items() if not isinstance(v, _VALID_ATTR_TYPES)]:
            del attrs[k]

    clean(ds.attrs)
    for var in ds.variables.values():
        clean(var.attrs)
        for k in _PACKING_ENCODING_KEYS:
            var.encoding.pop(k, None)
    return ds


def sanitize_name(name: str) -> str:
    """Make a NetCDF-friendly variable name (alnum + underscore)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")


def _nc_filename(var: str, period_key: str, period) -> str:
    return f"{var}_{period_key}_{period[0]}-{period[1]}.nc"


def _individual_nc_filename(var: str, period_key: str, period) -> str:
    return f"{var}_{period_key}_individual_{period[0]}-{period[1]}.nc"


def biasmap_netcdf_paths(
    netcdf_dir: Path, var: str, period: tuple[str, str],
    *, seasons: bool = True,
) -> list[Path]:
    """Expected NetCDF paths for a variable (annual + seasonal)."""
    keys = ["annual"] + (list(_SEASONS) if seasons else [])
    return [netcdf_dir / _nc_filename(var, k, period) for k in keys]


def biasmap_individual_netcdf_paths(
    netcdf_dir: Path, var: str, period: tuple[str, str],
    *, seasons: bool = True,
) -> list[Path]:
    """Expected individual-member NetCDF paths for a variable."""
    keys = ["annual"] + (list(_SEASONS) if seasons else [])
    return [netcdf_dir / _individual_nc_filename(var, k, period) for k in keys]


def _benchmark_member_prefix(label: str) -> str:
    """Field-name prefix for a benchmark's members (drops a trailing 'MMM')."""
    return sanitize_name(re.sub(r"\s*MMM\s*$", "", str(label)))


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

    # Evaluated-ensemble mean/median (and their bias vs obs), when present.
    edata = results.get("ens_data", {}).get(period_key)
    if edata:
        for stat in ("mean", "median"):
            if edata.get(stat) is not None:
                fields[f"ens_{stat}"] = edata[stat]
            if edata.get(f"{stat}_bias") is not None:
                fields[f"ens_{stat}_bias"] = edata[f"{stat}_bias"]

    return fields


def _individual_period_fields(
    results: dict, period_key: str,
) -> dict[str, xr.DataArray]:
    """Collect individual benchmark-member fields for one period.

    Reads ``results["benchmark_individual_data"]`` —
    ``{label: {period_key: {member_label: {regrid, bias}}}}`` — and names each
    field ``{benchmark}__{member}`` / ``{benchmark}__{member}_bias`` (e.g.
    ``CMIP6__ACCESS_CM2_r1i1p1f1_bias``, ``HighResMIP__ECMWF_IFS_HR_bias``).
    """
    fields: dict[str, xr.DataArray] = {}
    for label, perdict in results.get("benchmark_individual_data", {}).items():
        prefix = _benchmark_member_prefix(label)
        for member_label, mdata in perdict.get(period_key, {}).items():
            key = f"{prefix}__{sanitize_name(member_label)}"
            if mdata.get("regrid") is not None:
                fields[key] = mdata["regrid"]
            if mdata.get("bias") is not None:
                fields[f"{key}_bias"] = mdata["bias"]
    return fields


def _individual_period_keys(results: dict) -> list[str]:
    """Period keys ("annual" + seasons) present in the individual data."""
    keys: set[str] = set()
    for perdict in results.get("benchmark_individual_data", {}).values():
        keys.update(perdict.keys())
    ordered = ["annual"] + list(_SEASONS)
    return [k for k in ordered if k in keys]


#: Result-dict keys that hold metadata / non-source objects rather than
#: per-source fields — skipped by the generic collector.
_GENERIC_SKIP_KEYS = {
    "var_info", "colorbar_ranges", "colorbar_range", "stats", "statistics",
    "meta", "metadata",
}


def _is_exportable(da) -> bool:
    """True for a DataArray with at least one dimension (skip scalars)."""
    return isinstance(da, xr.DataArray) and da.ndim >= 1


def collect_dataarrays(
    obj, prefix: str, out: dict, *, depth: int = 0, max_depth: int = 6,
) -> None:
    """Recursively gather DataArrays from a nested result structure.

    Walks dicts/lists/tuples, building dotted-then-sanitised names from the
    key path (e.g. ``models_IFS_FESOM2_SR_annual_regrid``). Datasets are
    expanded into their data variables. Metadata keys (``var_info``,
    ``*_info``, …) and scalar/0-d arrays are skipped.
    """
    if depth > max_depth:
        return
    if isinstance(obj, xr.DataArray):
        if _is_exportable(obj):
            out[prefix or sanitize_name(str(obj.name) or "data")] = obj
        return
    if isinstance(obj, xr.Dataset):
        for v in obj.data_vars:
            name = f"{prefix}_{sanitize_name(str(v))}" if prefix else sanitize_name(str(v))
            if _is_exportable(obj[v]):
                out[name] = obj[v]
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = sanitize_name(str(k))
            if key in _GENERIC_SKIP_KEYS or key.endswith("_info"):
                continue
            collect_dataarrays(
                v, f"{prefix}_{key}" if prefix else key, out,
                depth=depth + 1, max_depth=max_depth,
            )
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            collect_dataarrays(
                v, f"{prefix}_{i}" if prefix else str(i), out,
                depth=depth + 1, max_depth=max_depth,
            )


def _data_span_period(fields: dict, fallback) -> tuple[str, str] | None:
    """Actual ``(start_year, end_year)`` from any 1-D datetime coord in *fields*.

    Reflects the real time span the data covers (e.g. a time series that ends
    where the model data ends, not the requested window).  Time-collapsed
    products (climatology/bias maps, month-indexed cycles) carry no datetime
    axis, so they fall back to *fallback* — the window used to compute them.
    """
    starts: list[str] = []
    ends: list[str] = []
    for da in fields.values():
        for coord in da.coords.values():
            if (coord.ndim == 1 and coord.size
                    and np.issubdtype(coord.dtype, np.datetime64)):
                vals = coord.values
                starts.append(str(np.datetime_as_string(np.min(vals), unit="Y")))
                ends.append(str(np.datetime_as_string(np.max(vals), unit="Y")))
    if starts:
        return (min(starts), max(ends))
    if fallback is not None:
        return (str(fallback[0]), str(fallback[1]))
    return None


def export_generic_netcdf(
    netcdf_dir: Path,
    token: str,
    results,
    period: tuple[str, str] | None,
    *,
    skip_existing: bool = True,
    extra_attrs: dict | None = None,
) -> list[Path]:
    """Write every per-source DataArray in *results* to a single NetCDF.

    Filename is ``{token}_{start}-{end}.nc`` (or ``{token}.nc`` when *period*
    is None and the data carries no time axis). The ``{start}-{end}`` reflects
    the *actual* data span (derived from the fields' datetime coordinate) when
    present, else the requested *period* (for time-collapsed maps/cycles).
    Fields are merged into one Dataset; any field whose dims clash with an
    already-added field (different grid/length) gets its dims renamed uniquely
    so no data is dropped. Existing files are skipped when *skip_existing*.
    """
    netcdf_dir = Path(netcdf_dir)
    token = sanitize_name(token)

    fields: dict = {}
    collect_dataarrays(results, "", fields)
    if not fields:
        return []

    eff_period = _data_span_period(fields, period)
    if eff_period is not None:
        fname = f"{token}_{eff_period[0]}-{eff_period[1]}.nc"
    else:
        fname = f"{token}.nc"
    path = netcdf_dir / fname
    if skip_existing and path.exists():
        return [path]

    ds = xr.Dataset()
    for name, da in fields.items():
        key = name
        # Strip name to avoid the DataArray's own .name shadowing the key.
        da = da.rename(key)

        # Isolate any dim that collides with an already-present dim whose
        # coordinate values differ.  Assigning such a field directly would let
        # xarray silently reindex it onto the existing axis, turning every
        # non-matching point into NaN — e.g. obs on first-of-month timestamps
        # merged against models on mid-month timestamps, or a coarse obs lat
        # grid merged against the finer common grid.  Dims that match exactly
        # (all models share the same time/lat/lon) stay shared so the file
        # remains compact and cross-comparable.
        isolate = {}
        for d in da.dims:
            if d not in ds.dims:
                continue
            if d in da.coords and d in ds.coords:
                same = (
                    ds[d].size == da[d].size
                    and np.array_equal(ds[d].values, da[d].values)
                )
            else:
                # No coordinate to compare on one side — shareable only when
                # the lengths line up; otherwise isolate to be safe.
                same = ds.sizes[d] == da.sizes[d]
            if not same:
                isolate[d] = f"{key}__{d}"
        if isolate:
            da = da.rename(isolate)

        try:
            ds = ds.assign({key: da})
            continue
        except Exception:  # noqa: BLE001 — dim/coord clash with prior field
            pass
        # Last resort: fully isolate this field's dims so incompatible
        # grids/lengths coexist.
        try:
            renamed = {d: f"{key}__{d}" for d in da.dims}
            iso = da.rename(renamed).reset_coords(drop=True)
            ds = ds.assign({key: iso})
        except Exception:  # noqa: BLE001
            logger.warning("  NetCDF: could not add field %s — skipping", key)

    if len(ds.data_vars) == 0:
        return []

    netcdf_dir.mkdir(parents=True, exist_ok=True)
    ds.attrs.update(
        token=token,
        period=f"{eff_period[0]}-{eff_period[1]}" if eff_period else "",
        period_start=eff_period[0] if eff_period else "",
        period_end=eff_period[1] if eff_period else "",
        description=(
            "Per-source diagnostic fields (obs, evaluated models, and "
            "benchmark MMMs: CMIP6, HighResMIP) on their analysis grids."
        ),
    )
    if extra_attrs:
        ds.attrs.update(extra_attrs)
    _sanitize_attrs(ds).to_netcdf(path)
    logger.info("  Wrote NetCDF: %s (%d fields)", path.name, len(ds.data_vars))
    return [path]


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
        _sanitize_attrs(ds).to_netcdf(path)
        logger.info("  Wrote NetCDF: %s", path.name)
        written.append(path)

    return written


def export_biasmap_individual_netcdf(
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
    """Write the individual benchmark members to their own NetCDF files.

    One file per period (``{var}_{period}_individual_{start}-{end}.nc``) holds
    every individual CMIP6 and HighResMIP member's regridded climatology and
    bias, named ``{benchmark}__{member}``. Kept separate from the obs/model/MMM
    file so each product skips/regenerates independently.

    Returns the list of written (or already-present) paths.
    """
    netcdf_dir = Path(netcdf_dir)
    written: list[Path] = []

    for period_key in _individual_period_keys(results):
        path = netcdf_dir / _individual_nc_filename(var, period_key, period)
        if skip_existing and path.exists():
            written.append(path)
            continue

        fields = _individual_period_fields(results, period_key)
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
                "Individual benchmark-member climatology and bias "
                "(member minus obs) on the common analysis grid. "
                "Sources: individual CMIP6 and HighResMIP models."
            ),
        )
        if extra_attrs:
            ds.attrs.update(extra_attrs)
        _sanitize_attrs(ds).to_netcdf(path)
        logger.info("  Wrote NetCDF: %s (%d members×fields)",
                    path.name, len(ds.data_vars))
        written.append(path)

    return written
