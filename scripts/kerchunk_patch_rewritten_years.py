#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 feather contributors
# SPDX-License-Identifier: BSD-3-Clause
"""Patch a kerchunk parquet reference store after source NetCDF files were rewritten.

The IFS-FESOM2 hist-1950 r2/r3 ocean daily stores (DKRZ, built 2026-02) index
yearly NetCDF files that the data owner later rewrote to fill missing days
(r2 2012: 7 days, r3 1999: 8 days; rewritten 2026-11). The stores then point at
stale byte ranges ("Zstd decompression error") and miss those days.

This script copies a store and, for every variable whose time dimension is fed
by the rewritten year's file, replaces that year's block of references with one
reference per chunk read from the file's HDF5 chunk index. Inline time
coordinates are re-spliced from the file's own time values, and shapes in
``.zmetadata`` are updated. Everything else is copied unchanged.

Needs h5py + pyarrow + numpy (not in the feather env; e.g. the ``aqua`` env):

    python scripts/kerchunk_patch_rewritten_years.py SRC.parq DST.parq --year 2012 [2013 ...]

Validate afterwards with ``--check`` (decodes every patched chunk).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ZSTD_HDF5 = 32015


def _read_refs(store: Path, var: str):
    return pq.read_table(store / var / "refs.0.parq")


def _write_refs(store: Path, var: str, table: pa.Table, like: pa.Table) -> None:
    table = table.cast(like.schema)
    pq.write_table(table, store / var / "refs.0.parq", compression="zstd")


def _year_block(paths: list, year: int) -> tuple[int, int, str]:
    """Contiguous [start, stop) of rows whose file is ``*_{year}.nc``."""
    pat = re.compile(rf"_{year}\.nc$")
    rows = [i for i, p in enumerate(paths) if isinstance(p, str) and pat.search(p)]
    if not rows:
        raise LookupError
    if rows != list(range(rows[0], rows[-1] + 1)):
        raise ValueError(f"rows of year {year} are not contiguous")
    files = {paths[i] for i in rows}
    if len(files) != 1:
        raise ValueError(f"year {year} spread over several files: {files}")
    return rows[0], rows[-1] + 1, files.pop()


def _chunk_refs(path: str, var: str, zarray: dict) -> list[tuple[int, int]]:
    """(offset, size) for each time step of *var* in the HDF5 file."""
    with h5py.File(path.replace("file://", ""), "r") as f:
        d = f[var]
        nd = d.ndim
        if list(zarray["chunks"][1:]) != list(d.shape[1:]):
            raise ValueError(f"{var}: store chunks {zarray['chunks']} vs file shape {d.shape}")
        plist = d.id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        want = [ZSTD_HDF5] if zarray.get("filters") else []
        if filters != want or zarray.get("compressor"):
            raise ValueError(f"{var}: file filters {filters} do not match store {zarray}")
        if d.chunks is None:  # contiguous: step through the block by hand
            step = int(np.prod(d.shape[1:])) * d.dtype.itemsize
            base = d.id.get_offset()
            return [(base + k * step, step) for k in range(d.shape[0])]
        if tuple(d.chunks) != (1, *d.shape[1:]):
            raise ValueError(f"{var}: file chunks {d.chunks} are not one step each")
        out = []
        for k in range(d.shape[0]):
            ci = d.id.get_chunk_info_by_coord((k,) + (0,) * (nd - 1))
            out.append((int(ci.byte_offset), int(ci.size)))
        return out


def _single_record_ref(path: str, var: str, zarray: dict) -> tuple[int, int] | None:
    """(offset, size) of a per-file constant record (shape (1, ...)), else None."""
    with h5py.File(path.replace("file://", ""), "r") as f:
        if var not in f:
            return None
        d = f[var]
        if d.ndim != len(zarray["shape"]) or d.shape[0] != 1:
            return None
        if list(d.shape[1:]) != list(zarray["chunks"][1:]):
            return None
        plist = d.id.get_create_plist()
        if plist.get_nfilters() or zarray.get("filters") or zarray.get("compressor"):
            return None  # only the uncompressed case is handled here
        if d.chunks is None:
            return int(d.id.get_offset()), int(d.size * d.dtype.itemsize)
        ci = d.id.get_chunk_info_by_coord((0,) * d.ndim)
        return int(ci.byte_offset), int(ci.size)


def _file_times(path: str, tname_candidates: list[str], units: str) -> np.ndarray:
    """Time values of the file, converted to the store's ``hours since`` units."""
    with h5py.File(path.replace("file://", ""), "r") as f:
        for tn in tname_candidates:
            if tn in f:
                t = f[tn]
                break
        else:
            raise KeyError(f"no time variable in {path}")
        fu = t.attrs["units"].decode() if isinstance(t.attrs["units"], bytes) else t.attrs["units"]
        vals = t[()].astype("f8")
    return _to_hours(vals, fu, units)


def _to_hours(vals, from_units: str, to_units: str) -> np.ndarray:
    """Convert ``<unit> since <date>`` values (unpadded dates allowed)."""
    scale = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}

    def parse(u):
        m = re.match(r"\s*(\w+) since (\d+)-(\d+)-(\d+)(?:[ T](\d+):(\d+)(?::(\d+))?)?", u)
        if not m:
            raise ValueError(f"cannot parse time units {u!r}")
        y, mo, d = (int(x) for x in m.group(2, 3, 4))
        hh, mm, ss = (int(x or 0) for x in m.group(5, 6, 7))
        ref = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}", "s")
        return m.group(1).lower(), ref

    fu, fref = parse(from_units)
    tu, tref = parse(to_units)
    secs = vals * scale[fu] + (fref - tref).astype("int64")
    return secs / scale[tu]


def patch(src: Path, dst: Path, years: list[int]) -> None:
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(src, dst)
    for year in years:
        print(f"year {year}")
        _patch_year(dst, year)


def _patch_year(dst: Path, year: int) -> None:
    """Rebuild *year*'s references in the store copy *dst* (in place)."""
    zm_path = dst / ".zmetadata"
    zm = json.loads(zm_path.read_text())
    meta = zm["metadata"]
    rs = zm.get("record_size")

    variables = sorted({k.split("/")[0] for k in meta if k.endswith("/.zarray")})
    dims = {v: meta.get(f"{v}/.zattrs", {}).get("_ARRAY_DIMENSIONS", []) for v in variables}
    time_dims = sorted({d[0] for d in dims.values() if d and d[0].startswith("time")})

    new_len: dict[str, int] = {}      # time dim -> new length
    year_file: dict[str, str] = {}    # time dim -> a file of the year using it
    old_block: dict[str, tuple[int, int]] = {}
    deferred: list[str] = []  # not a per-day field of the rewritten file

    for v in variables:
        d = dims[v]
        if v in time_dims or not d or not d[0].startswith("time"):
            continue
        za = meta[f"{v}/.zarray"]
        n_chunks_other = int(np.prod([-(-s // c) for s, c in zip(za["shape"][1:], za["chunks"][1:])]))
        if n_chunks_other != 1:
            raise ValueError(f"{v}: more than one chunk per time step")
        tab = _read_refs(dst, v)
        paths = tab.column("path").to_pylist()
        try:
            a, b, path = _year_block(paths, year)
        except LookupError:
            continue
        with h5py.File(path, "r") as f:
            in_file = v in f and f[v].ndim == len(za["shape"]) and f[v].shape[0] > 1
        if not in_file:
            deferred.append(v)
            continue
        refs = _chunk_refs(path, v, za)
        n_old = za["shape"][0]
        n = n_old - (b - a) + len(refs)

        cols = {c: tab.column(c).to_pylist()[:n_old] for c in ("path", "offset", "size", "raw")}
        new_rows = {
            "path": [path] * len(refs),
            "offset": [o for o, _ in refs],
            "size": [s for _, s in refs],
            "raw": [None] * len(refs),
        }
        for c in cols:
            cols[c] = cols[c][:a] + new_rows[c] + cols[c][b:]
        pad = max(len(tab) - n, 0) if rs else 0
        cols["path"] += [None] * pad
        cols["offset"] += [0] * pad
        cols["size"] += [0] * pad
        cols["raw"] += [None] * pad
        _write_refs(dst, v, pa.table(cols), tab)

        za["shape"][0] = n
        meta[f"{v}/.zarray"] = za
        td = d[0]
        if td in new_len and new_len[td] != n:
            raise ValueError(f"{td}: inconsistent new lengths {new_len[td]} vs {n} ({v})")
        new_len[td] = n
        year_file.setdefault(td, path)
        old_block.setdefault(td, (a, b))
        print(f"  {v:14s} {td:6s} rows {a}:{b} ({b - a}) -> {len(refs)}  shape {n_old} -> {n}")

    # Variables the rewritten file does not hold per day (lev_bnds: one
    # constant record per file that kerchunk tiled along time). Point every
    # row of the year at that record in the rewritten file; if the file lacks
    # the variable, repeat a neighbouring year's reference (untouched file).
    for v in deferred:
        za = meta[f"{v}/.zarray"]
        td = dims[v][0]
        tab = _read_refs(dst, v)
        a, b, path = _year_block(tab.column("path").to_pylist(), year)
        if td not in new_len:
            raise ValueError(f"{v}: no time length for year {year}")
        n_old = za["shape"][0]
        n = new_len[td]
        n_year = n - (n_old - (b - a))
        cols = {c: tab.column(c).to_pylist() for c in ("path", "offset", "size", "raw")}
        ref = _single_record_ref(path, v, za)
        if ref is not None:
            row = {"path": path, "offset": ref[0], "size": ref[1], "raw": None}
            how = "file's single record"
        elif a > 0 or b < n_old:
            j = a - 1 if a > 0 else b
            row = {c: cols[c][j] for c in cols}
            how = "neighbouring year's ref"
        else:
            raise ValueError(f"{v}: not in {path} and no other year to borrow from")
        for c in cols:
            tail = cols[c][b:n_old]
            pad = cols[c][n_old:][: max(len(tab) - n, 0)]
            cols[c] = cols[c][:a] + [row[c]] * n_year + tail + pad
        _write_refs(dst, v, pa.table(cols), tab)
        za["shape"][0] = n
        meta[f"{v}/.zarray"] = za
        print(f"  {v:14s} {td:6s} rows {a}:{b} ({b - a}) -> {n_year} ({how})  shape {n_old} -> {n}")

    for td in time_dims:
        if td not in new_len:
            continue
        za = meta[f"{td}/.zarray"]
        attrs = meta[f"{td}/.zattrs"]
        tab = _read_refs(dst, td)
        raw = tab.column("raw").to_pylist()[0]
        if za.get("compressor") or za.get("filters") or za["chunks"] != za["shape"]:
            raise ValueError(f"{td}: expected one inline uncompressed chunk, got {za}")
        old = np.frombuffer(raw, dtype=za["dtype"])
        a, b = old_block[td]
        new = _file_times(year_file[td], [td, "time"], attrs["units"])
        merged = np.concatenate([old[:a], new, old[b:]])
        if merged.size != new_len[td]:
            raise ValueError(f"{td}: {merged.size} values vs data length {new_len[td]}")
        if not np.all(np.diff(merged) > 0):
            raise ValueError(f"{td}: not strictly increasing after splice")
        cols = {c: tab.column(c).to_pylist() for c in ("path", "offset", "size", "raw")}
        cols["raw"][0] = merged.astype(za["dtype"]).tobytes()
        _write_refs(dst, td, pa.table(cols), tab)
        za["shape"] = za["chunks"] = [int(merged.size)]
        meta[f"{td}/.zarray"] = za
        print(f"  {td:14s} coordinate {old.size} -> {merged.size}")

    zm_path.write_text(json.dumps(zm))


def check(store: Path, year: int) -> None:
    """Decode every chunk of the patched year and compare with the file."""
    import numcodecs

    zm = json.loads((store / ".zmetadata").read_text())["metadata"]
    for v in sorted({k.split("/")[0] for k in zm if k.endswith("/.zarray")}):
        d = zm.get(f"{v}/.zattrs", {}).get("_ARRAY_DIMENSIONS", [])
        if not d or not d[0].startswith("time") or v.startswith("time"):
            continue
        za = zm[f"{v}/.zarray"]
        tab = _read_refs(store, v).to_pandas()
        try:
            a, b, path = _year_block(tab["path"].tolist(), year)
        except LookupError:
            continue
        codecs = [numcodecs.get_codec(f) for f in (za.get("filters") or [])]
        with h5py.File(path, "r") as f:
            if v not in f or f[v].shape[0] != b - a:
                print(f"  -- {v:14s} {b - a} refs (not a per-day field of {Path(path).name})")
                continue
            ds = f[v]
            with open(path, "rb") as fh:
                for k, r in enumerate(range(a, b)):
                    fh.seek(int(tab.offset[r]))
                    buf = fh.read(int(tab["size"][r]))
                    for c in reversed(codecs):
                        buf = c.decode(buf)
                    arr = np.frombuffer(buf, dtype=za["dtype"]).reshape(za["chunks"])
                    if k in (0, (b - a) // 2, b - a - 1):
                        ref = ds[k : k + 1]
                        if not np.array_equal(arr, ref, equal_nan=True):
                            raise AssertionError(f"{v} step {k}: decoded chunk differs from file")
        print(f"  ok {v:14s} {b - a} chunks decoded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--check", action="store_true", help="only validate DST")
    args = ap.parse_args()
    if not args.check:
        patch(args.src, args.dst, args.year)
    for year in args.year:
        check(args.dst, year)


if __name__ == "__main__":
    main()
