# SPDX-FileCopyrightText: 2026 feather contributors
# SPDX-License-Identifier: BSD-3-Clause
"""Stale kerchunk references: source files rewritten after the store was built."""
from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

import feather.data.kerchunk_loader as kl
from feather.data.kerchunk_loader import (
    stale_check_seconds,
    stale_reference_files,
    warn_stale_references,
)


def _make_store(tmp_path, n_files=4, age=1000.0):
    """A ``{store}/{var}/refs.0.parq`` store pointing at *n_files* sources.

    Sources are *age* seconds older than the reference files.
    """
    src = tmp_path / "src"
    src.mkdir()
    files = []
    for i in range(n_files):
        f = src / f"avg_tos_{2000 + i}.nc"
        f.write_bytes(b"x" * 16)
        files.append(f)
    store = tmp_path / "store.parq"
    for var in ("avg_tos", "time"):
        (store / var).mkdir(parents=True)
    pd.DataFrame({
        "path": [f"file://{f}" for f in files] + [None],
        "offset": [0] * n_files + [0],
        "size": [16] * n_files + [0],
        "raw": [None] * (n_files + 1),
    }).to_parquet(store / "avg_tos" / "refs.0.parq")
    pd.DataFrame({  # inline coordinate: no paths
        "path": [None], "offset": [0], "size": [0], "raw": [b"\0" * 8],
    }).to_parquet(store / "time" / "refs.0.parq")
    now = time.time()
    for f in files:
        os.utime(f, (now - age, now - age))
    return store, files


@pytest.fixture(autouse=True)
def _reset_checked():
    kl._STALE_CHECKED.clear()
    yield
    kl._STALE_CHECKED.clear()


def test_untouched_store_is_clean(tmp_path):
    store, _ = _make_store(tmp_path)
    assert stale_reference_files(store) == []


def test_rewritten_source_is_reported(tmp_path):
    store, files = _make_store(tmp_path)
    later = time.time() + 60
    os.utime(files[2], (later, later))
    stale = stale_reference_files(store)
    assert [p for p, _ in stale] == [str(files[2])]
    assert stale[0][1] == pytest.approx(later)


def test_missing_source_is_reported(tmp_path):
    store, files = _make_store(tmp_path)
    files[0].unlink()
    assert stale_reference_files(store) == [(str(files[0]), None)]


def test_non_directory_store_is_skipped(tmp_path):
    f = tmp_path / "refs.json"
    f.write_text("{}")
    assert stale_reference_files(f) == []
    assert stale_reference_files(tmp_path / "absent.parq") == []


def test_time_budget_stops_early(tmp_path):
    store, files = _make_store(tmp_path, n_files=6)
    for f in files:
        f.unlink()
    # zero budget: nothing read, nothing reported, no error
    assert stale_reference_files(store, time_budget=0.0) == []


def test_warning_logged_once(tmp_path, caplog):
    store, files = _make_store(tmp_path)
    later = time.time() + 60
    os.utime(files[1], (later, later))
    with caplog.at_level(logging.WARNING, logger=kl.__name__):
        warn_stale_references(store)
        warn_stale_references(store)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "changed after the store was built" in msgs[0]
    assert files[1].name in msgs[0]


def test_warning_disabled_by_zero_budget(tmp_path, caplog):
    store, files = _make_store(tmp_path)
    files[0].unlink()
    with caplog.at_level(logging.WARNING, logger=kl.__name__):
        warn_stale_references(store, time_budget=0)
    assert not caplog.records


def test_check_never_raises(tmp_path, monkeypatch, caplog):
    store, _ = _make_store(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(kl, "stale_reference_files", boom)
    with caplog.at_level(logging.WARNING, logger=kl.__name__):
        warn_stale_references(store)
    assert not caplog.records


@pytest.mark.parametrize("data_source, expected", [
    ({}, 5.0),
    ({"kerchunk_stale_check_seconds": 0}, 0.0),
    ({"kerchunk_stale_check_seconds": "2.5"}, 2.5),
    ({"kerchunk_stale_check_seconds": "soon"}, 5.0),
    (None, 5.0),
])
def test_stale_check_seconds(data_source, expected):
    assert stale_check_seconds(SimpleNamespace(data_source=data_source)) == expected


def test_open_store_runs_the_check(tmp_path, monkeypatch, caplog):
    """KerchunkParquetLoader._open_store warns before opening the store."""
    store, files = _make_store(tmp_path)
    files[3].unlink()
    loader = kl.KerchunkParquetLoader.__new__(kl.KerchunkParquetLoader)
    loader._config = SimpleNamespace(data_source={})
    loader._store_cache, loader._store_paths, loader._touched = {}, {}, []
    monkeypatch.setattr(loader, "_store_path", lambda m, s: store)
    monkeypatch.setattr(kl.xr, "open_dataset", lambda *a, **k: "opened")
    with caplog.at_level(logging.WARNING, logger=kl.__name__):
        assert loader._open_store("m", "ocean2d") == "opened"
    assert any(files[3].name in r.getMessage() for r in caplog.records)
