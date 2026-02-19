#!/usr/bin/env python
"""End-to-end test of the CMIP6Loader with real Levante data.

Run on a compute node (SLURM or Jupyter), NOT on the login node:

    conda activate nereus
    python scripts/test_cmip6_e2e.py
"""

import time

from feather.config import FeatherConfig
from feather.data.cmip6 import CMIP6Loader

config = FeatherConfig.from_yaml("configs/default.yaml")

# Override: enable CMIP6
config.cmip6["enabled"] = True

loader = CMIP6Loader(config)

print(f"Zarr directory: {loader.zarr_dir}")
print(f"Configured models: {list(loader.models.keys())}")
print()

# ── 1. Check data availability ──────────────────────────────────────
print("=" * 60)
print("1. Data availability")
print("=" * 60)

for var, table in [("tas", "Amon"), ("pr", "Amon"), ("tos", "Omon"), ("siconc", "SImon")]:
    models = loader.available_models(var, table)
    members = loader.available_members(var, table)
    print(f"  {var:8s} ({table}): {len(models)} models, {len(members)} members")
    for m in models:
        print(f"    - {m}")
print()

# ── 2. Load a single variable ───────────────────────────────────────
print("=" * 60)
print("2. Load single variable (MIROC6 tas, 1990-2010)")
print("=" * 60)

t0 = time.time()
da = loader.load_var(
    "tas", "MIROC6",
    variant="r1i1p1f1", table="Amon",
    period=("1990-01", "2010-12"),
)
dt = time.time() - t0

if da is not None:
    print(f"  Shape: {da.shape}, dims: {da.dims}")
    print(f"  Global mean: {float(da.mean()):.2f} K")
    print(f"  Min: {float(da.min()):.2f} K, Max: {float(da.max()):.2f} K")
    print(f"  Load time: {dt:.1f}s")
else:
    print("  MIROC6 tas not found — check zarr directory")
print()

# ── 3. Load via model variable mapping ──────────────────────────────
print("=" * 60)
print("3. Load via feather variable name (avg_2t → tas)")
print("=" * 60)

da2 = loader.load_var_for_model_var(
    "avg_2t", "MIROC6",
    variant="r1i1p1f1",
    period=("1990-01", "2010-12"),
)
if da2 is not None:
    print(f"  avg_2t → tas: global mean = {float(da2.mean()):.2f} K")
else:
    print("  Not found")
print()

# ── 4. Load area weights ────────────────────────────────────────────
print("=" * 60)
print("4. Area weights")
print("=" * 60)

area = loader.load_area("MIROC6", variant="r1i1p1f1", table="Amon")
if area is not None:
    print(f"  areacella shape: {area.shape}")
    print(f"  Total area: {float(area.sum()):.3e} m²")
else:
    print("  areacella not found for MIROC6")
print()

# ── 5. Multi-model mean (one_per_model) ─────────────────────────────
print("=" * 60)
print("5. Multi-model mean (one_per_model, tas, 1990-2010)")
print("=" * 60)

t0 = time.time()
mmm, info = loader.load_multi_model_mean(
    "tas", table="Amon",
    period=("1990-01", "2010-12"),
    ensemble_mode="one_per_model",
)
dt = time.time() - t0

if mmm is not None:
    print(f"  MMM shape: {mmm.shape}, dims: {mmm.dims}")
    print(f"  Global mean: {float(mmm.mean()):.2f} K")
    print(f"  Members used: {info['n_members']}")
    for m in info["models_used"]:
        print(f"    - {m}")
    if info["models_skipped"]:
        print(f"  Skipped: {info['models_skipped']}")
    print(f"  Time: {dt:.1f}s")
else:
    print(f"  No data. Skipped: {info['models_skipped']}")
print()

# ── 6. Multi-model mean (all_members) ───────────────────────────────
print("=" * 60)
print("6. Multi-model mean (all_members, tas, 1990-2010)")
print("=" * 60)

t0 = time.time()
mmm_all, info_all = loader.load_multi_model_mean(
    "tas", table="Amon",
    period=("1990-01", "2010-12"),
    ensemble_mode="all_members",
)
dt = time.time() - t0

if mmm_all is not None:
    print(f"  MMM shape: {mmm_all.shape}")
    print(f"  Global mean: {float(mmm_all.mean()):.2f} K")
    print(f"  Members used: {info_all['n_members']}")
    print(f"  Time: {dt:.1f}s")
else:
    print(f"  No data. Skipped: {info_all['models_skipped']}")
print()

# ── 7. Convenience wrapper ──────────────────────────────────────────
print("=" * 60)
print("7. load_mmm_for_model_var (avg_2t)")
print("=" * 60)

mmm_var, info_var = loader.load_mmm_for_model_var(
    "avg_2t",
    period=("1990-01", "2010-12"),
    ensemble_mode="one_per_model",
)
if mmm_var is not None:
    print(f"  avg_2t MMM global mean: {float(mmm_var.mean()):.2f} K")
    print(f"  Members: {info_var['n_members']}")
print()

# ── 8. Ocean variable ───────────────────────────────────────────────
print("=" * 60)
print("8. Ocean variable (tos, Omon)")
print("=" * 60)

mmm_tos, info_tos = loader.load_multi_model_mean(
    "tos", table="Omon",
    period=("1990-01", "2010-12"),
    ensemble_mode="one_per_model",
)
if mmm_tos is not None:
    print(f"  SST MMM global mean: {float(mmm_tos.mean()):.2f} K")
    print(f"  Members: {info_tos['n_members']}")
else:
    print(f"  No tos data. Skipped: {info_tos['models_skipped']}")
print()

print("=" * 60)
print("Done!")
print("=" * 60)
