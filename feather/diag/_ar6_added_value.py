"""Added Value over the 58 AR6 (Iturbide et al. 2020) reference regions.

Companion to :mod:`feather.diag.added_value`, reached with
``--added-value-regions ar6``.  Where the CORDEX-region code in
``added_value.py`` re-derives Added Value from source data, this module
reads the **bias NetCDFs that the bias-map diagnostics already wrote**::

    {output_dir}/netcdf/global_biases/{var}_{period}_{p0}-{p1}.nc      ERA5 ref
    {output_dir}/netcdf/precipitation_mswep/{var}_{period}_...nc       MSWEP ref
    {output_dir}/netcdf/temperature_berkeley/{var}_{period}_...nc      Berkeley ref

each carrying ``{member}_bias`` for every evaluated member plus
``{benchmark}_MMM_bias``.  So the regional analysis costs a few seconds once
those exist, and `--save-netcdf` on the relevant bias diagnostic is the
prerequisite.  A missing file skips that variable/reference with a warning
rather than falling back to a silent recompute — the recompute is hours of
work and should be an explicit decision.

Method
------
Added Value is computed **per grid cell and then area-averaged per region**,
not from region-mean biases:

* ``AV = (bias_bench² − bias_member²) / max(bias_bench², bias_member²)``,
  clipped to [-1, 1] — :meth:`AddedValueDiag._av_from_biases`.
* Regional value = cos(lat)-weighted, NaN-aware mean of that field over the
  region's mask.

Averaging the *field* keeps cancelling biases honest: a member whose regional
mean bias is near zero only because a warm half offsets a cold half does not
score as skilful.

Keep-sets
---------
A region "qualifies" when at least *N* of the individual members (ensemble
mean and median excluded — they are summaries, not evidence) have a positive
regional AV.  Two thresholds are drawn, from
``added_value.region_thresholds`` (default ``[6, 3]``): a majority set and a
looser "helps at least a subset" set.  With more than one reference the
keep-set is the **union** across references, so region outlines and labels
are identical in every panel and match the tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from feather.diag.netcdf_export import sanitize_name
from feather.util.ar6_regions import ar6_region_masks, list_ar6_regions

logger = logging.getLogger(__name__)

#: Default reference datasets per variable: ``(label, diagnostic directory)``.
#: The diagnostic directory names the subdirectory of ``{output}/netcdf/``
#: whose bias fields use that reference.  Berkeley Earth is deliberately not
#: a default for ``tas`` — it is land-only, so it cannot score the 15 ocean
#: regions — but ``added_value.ar6_references`` can add it back.
DEFAULT_REFERENCES: dict[str, list[tuple[str, str]]] = {
    "tas": [("ERA5", "global_biases")],
    "pr": [("ERA5", "global_biases"), ("MSWEP", "precipitation_mswep")],
}

#: Reference label → diagnostic directory, for config-supplied overrides.
_REFERENCE_DIRS = {
    "ERA5": "global_biases",
    "MSWEP": "precipitation_mswep",
    "BE": "temperature_berkeley",
    "BERKELEY": "temperature_berkeley",
}

#: Period keys as they appear in the NetCDF filenames.
ALL_PERIODS = ("annual", "DJF", "MAM", "JJA", "SON")

#: Summary rows that are tabulated but excluded from keep-set counting.
ENSEMBLE_ROWS = ("ens_mean", "ens_median")


def references_for(var: str, config) -> list[tuple[str, str]]:
    """Resolve the ``(label, diagnostic dir)`` references for *var*.

    ``added_value.ar6_references`` overrides the defaults, e.g.::

        added_value:
          ar6_references:
            tas: [ERA5, BE]
    """
    configured = (config.added_value or {}).get("ar6_references", {})
    labels = configured.get(var)
    if not labels:
        return list(DEFAULT_REFERENCES.get(var, [("ERA5", "global_biases")]))

    out: list[tuple[str, str]] = []
    for label in labels:
        key = str(label).upper()
        if key not in _REFERENCE_DIRS:
            logger.warning(
                "Unknown AR6 reference %r for %s — known: %s",
                label, var, ", ".join(sorted(_REFERENCE_DIRS)),
            )
            continue
        out.append((str(label), _REFERENCE_DIRS[key]))
    return out


def region_thresholds(config) -> tuple[int, int]:
    """Return the ``(majority, subset)`` keep-set thresholds from config.

    Defaults to ``(6, 3)`` — a majority of the 10-member EERIE ensemble, and
    the looser threshold used in the reference analysis.
    """
    raw = (config.added_value or {}).get("region_thresholds", [6, 3])
    try:
        majority, subset = int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError):
        logger.warning(
            "Invalid added_value.region_thresholds %r — using [6, 3]", raw,
        )
        return 6, 3
    return majority, subset


def bias_nc_path(
    output_dir: str | Path, diag_dir: str, var: str, period_key: str,
    period: tuple[str, str],
) -> Path:
    """Path of the bias NetCDF written by a bias-map diagnostic."""
    return (
        Path(output_dir) / "netcdf" / diag_dir
        / f"{var}_{period_key}_{period[0]}-{period[1]}.nc"
    )


def _weighted_region_mean(
    field: np.ndarray, mask: np.ndarray, weights: np.ndarray,
) -> float:
    """cos(lat)-weighted mean of *field* over *mask*, ignoring NaN."""
    vals = field[mask]
    wts = weights[mask]
    finite = np.isfinite(vals)
    if not finite.any():
        return float("nan")
    total = float(np.sum(wts[finite]))
    if total <= 0:
        return float("nan")
    return float(np.sum(wts[finite] * vals[finite]) / total)


def compute_region_av(
    diag,
    var: str,
    *,
    periods=ALL_PERIODS,
) -> "list[dict]":
    """Per-region Added Value rows for one variable.

    Returns a list of ``{reference, period, member, region, AV}`` dicts —
    one per (reference, period, member, region) combination that could be
    computed.  Empty when no bias NetCDF is available.
    """
    from feather.diag.added_value import AddedValueDiag

    bench_field = f"{sanitize_name(diag._bench_label)}_bias"
    members = [sanitize_name(m) for m in diag.config.models]
    rows: list[dict] = []

    for ref_label, diag_dir in references_for(var, diag.config):
        for period_key in periods:
            path = bias_nc_path(
                diag.config.output_dir, diag_dir, var, period_key, diag.period,
            )
            if not path.exists():
                logger.warning(
                    "  AR6 %s/%s/%s: no bias NetCDF at %s — run the %s "
                    "diagnostic with --save-netcdf first",
                    var, ref_label, period_key, path, diag_dir,
                )
                continue

            with xr.open_dataset(path) as ds:
                if bench_field not in ds:
                    logger.warning(
                        "  AR6 %s/%s/%s: %s missing from %s — skipping",
                        var, ref_label, period_key, bench_field, path.name,
                    )
                    continue

                lat = np.asarray(ds["lat"].values, dtype=float)
                lon = np.asarray(ds["lon"].values, dtype=float)
                masks = ar6_region_masks(lat, lon)
                weights = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)

                bench_bias = ds[bench_field]
                av_fields: dict[str, np.ndarray] = {}
                for member in [*members, *ENSEMBLE_ROWS]:
                    field = f"{member}_bias"
                    if field not in ds:
                        continue
                    av_fields[member] = AddedValueDiag._av_from_biases(
                        bench_bias, ds[field],
                    ).values

            if not av_fields:
                logger.warning(
                    "  AR6 %s/%s/%s: no member bias fields in %s",
                    var, ref_label, period_key, path.name,
                )
                continue

            for region, mask in masks.items():
                for member, av in av_fields.items():
                    rows.append({
                        "reference": ref_label,
                        "period": period_key,
                        "member": member,
                        "region": region,
                        "AV": _weighted_region_mean(av, mask, weights),
                    })

            logger.info(
                "  AR6 %s/%s/%s: %d members × 58 regions",
                var, ref_label, period_key, len(av_fields),
            )

    return rows


def keep_set(
    rows: "list[dict]", period_key: str, threshold: int,
    *, reference: str | None = None,
) -> list[str]:
    """Regions where at least *threshold* individual members have AV > 0.

    Ensemble mean/median rows are excluded from the count.  With
    *reference* ``None`` the rule is the union across references — a region
    qualifies if it passes under any of them — which is what keeps the map
    outlines identical to the table columns.
    """
    per_region: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["period"] != period_key or row["member"] in ENSEMBLE_ROWS:
            continue
        if reference is not None and row["reference"] != reference:
            continue
        if not np.isfinite(row["AV"]) or row["AV"] <= 0:
            continue
        counts = per_region.setdefault(row["region"], {})
        counts[row["reference"]] = counts.get(row["reference"], 0) + 1

    return [
        region for region in list_ar6_regions()
        if max(per_region.get(region, {}).values(), default=0) >= threshold
    ]


def rows_to_csv(rows: "list[dict]", path: Path) -> Path:
    """Write the per-member region table as CSV (the session's data artifact)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["reference", "period", "member", "region", "AV"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Gridded fields (for the map figures) ─────────────────────────────────


def load_av_fields(
    diag, var: str, diag_dir: str, period_key: str,
) -> "tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None":
    """Return ``(lat, lon, {member: AV field})`` for one reference/period.

    Fields are cast to float32 — a dozen members on a 0.25° grid is ~50 MB
    that way, against ~100 MB at float64, and the maps are rasterised at
    screen resolution anyway.  ``None`` when the NetCDF is absent.
    """
    from feather.diag.added_value import AddedValueDiag

    path = bias_nc_path(
        diag.config.output_dir, diag_dir, var, period_key, diag.period,
    )
    if not path.exists():
        return None

    bench_field = f"{sanitize_name(diag._bench_label)}_bias"
    members = [sanitize_name(m) for m in diag.config.models]

    with xr.open_dataset(path) as ds:
        if bench_field not in ds:
            return None
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        bench_bias = ds[bench_field]
        fields = {}
        for member in [*ENSEMBLE_ROWS, *members]:
            if f"{member}_bias" not in ds:
                continue
            fields[member] = AddedValueDiag._av_from_biases(
                bench_bias, ds[f"{member}_bias"],
            ).values.astype(np.float32)

    return (lat, lon, fields) if fields else None


def member_label(member: str, config) -> str:
    """Display label for a member key (``ens_mean`` → ``Ensemble mean``)."""
    if member == "ens_mean":
        return "Ensemble mean"
    if member == "ens_median":
        return "Ensemble median"
    for name in config.models:
        if sanitize_name(name) == member:
            return name
    return member


# ── Figures ──────────────────────────────────────────────────────────────


def _cbar_layout(fig_height: float) -> "tuple[float, float]":
    """``(gridspec bottom, colourbar y)`` as fractions of *fig_height*.

    Both are fixed distances in inches converted to fractions: the strip
    needs the same ~0.95 in of room whether the figure is a two-panel pair
    or a twelve-panel stack, and a constant fraction would either clip the
    caption on tall figures or waste half the short ones.
    """
    return 0.95 / fig_height, 0.42 / fig_height


def _robinson_axes(fig, spec):
    """Robinson-projection axes with coastlines, or plain axes if no cartopy."""
    try:
        import cartopy.crs as ccrs
        ax = fig.add_subplot(spec, projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.3, color="0.3")
        return ax, ccrs.PlateCarree()
    except ImportError:  # pragma: no cover - cartopy is a hard dependency
        return fig.add_subplot(spec), None


def _draw_regions(ax, keep: "list[str] | None" = None, label: bool = False):
    """Overlay AR6 region outlines, optionally only a keep-set."""
    from feather.util.ar6_regions import ar6_regions

    regions = ar6_regions()
    if keep is not None:
        numbers = [
            n for n, a in zip(regions.numbers, regions.abbrevs) if str(a) in keep
        ]
        if not numbers:
            return
        regions = regions[numbers]
    try:
        regions.plot_regions(
            # ``label="abbrev"``: the default is the AR6 *number*, which is
            # unreadable on a map — the tables are keyed by abbreviation.
            ax=ax, line_kws=dict(lw=0.4, color="k"),
            add_label=label, label="abbrev", label_multipolygon="all",
            text_kws=dict(fontsize=4.5, color="k",
                          bbox=dict(pad=0.15, color="w", alpha=0.6)),
        )
    except Exception as exc:  # pragma: no cover - plotting robustness
        logger.warning("  Could not draw AR6 region outlines: %s", exc)


def plot_member_maps(
    lat, lon, fields: "dict[str, np.ndarray]", *, config, title: str,
    subtitle: str, cmap, ncols: int = 2,
):
    """Gridded AV, one panel per member, with AR6 outlines on every panel."""
    import matplotlib.pyplot as plt

    keys = list(fields)
    nrows = int(np.ceil(len(keys) / ncols))
    height = 2.9 * nrows + 1.1
    fig = plt.figure(figsize=(6.2 * ncols, height))
    bottom, cbar_y = _cbar_layout(height)
    gs = fig.add_gridspec(
        nrows, ncols, hspace=0.10, wspace=0.05,
        top=1.0 - 0.55 / height, bottom=bottom,
    )

    mesh = None
    for i, key in enumerate(keys):
        ax, transform = _robinson_axes(fig, gs[i // ncols, i % ncols])
        kw = dict(transform=transform) if transform is not None else {}
        mesh = ax.pcolormesh(
            lon, lat, fields[key], cmap=cmap, vmin=-1.0, vmax=1.0,
            shading="auto", **kw,
        )
        _draw_regions(ax)
        ax.set_title(member_label(key, config), fontsize=8, loc="left")

    fig.suptitle(title, fontsize=11)
    if mesh is not None:
        cax = fig.add_axes([0.15, cbar_y, 0.7, 0.35 / height])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label(subtitle, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return fig


def plot_ensemble_maps(
    panels: "list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]", *,
    keep: "list[str]", title: str, subtitle: str, cmap,
):
    """Ensemble mean/median AV per reference, with the keep-set outlined.

    ``panels`` is ``[(panel title, lat, lon, field), …]`` — two entries for a
    single-reference variable, four when two references are in play.  Every
    panel carries the same outlines so the maps and the tables agree on
    which regions qualify.
    """
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    height = 3.5 * nrows + 1.2
    fig = plt.figure(figsize=(7.4 * ncols, height))
    bottom, cbar_y = _cbar_layout(height)
    gs = fig.add_gridspec(
        nrows, ncols, hspace=0.08, wspace=0.05,
        top=1.0 - 0.55 / height, bottom=bottom,
    )

    mesh = None
    for i, (ptitle, lat, lon, field) in enumerate(panels):
        ax, transform = _robinson_axes(fig, gs[i // ncols, i % ncols])
        kw = dict(transform=transform) if transform is not None else {}
        mesh = ax.pcolormesh(
            lon, lat, field, cmap=cmap, vmin=-1.0, vmax=1.0,
            shading="auto", **kw,
        )
        _draw_regions(ax, keep=keep, label=True)
        ax.set_title(ptitle, fontsize=9, loc="left")

    fig.suptitle(title, fontsize=11)
    if mesh is not None:
        cax = fig.add_axes([0.15, cbar_y, 0.7, 0.35 / height])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label(subtitle, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return fig


def _av_lookup(rows: "list[dict]", period_key: str) -> dict:
    """``{(reference, member, region): AV}`` for one period."""
    return {
        (r["reference"], r["member"], r["region"]): r["AV"]
        for r in rows if r["period"] == period_key
    }


def plot_region_table(
    rows: "list[dict]", period_key: str, *, references: "list[str]",
    members: "list[str]", regions: "list[str]", config, title: str,
    subtitle: str, cmap,
):
    """Region × member AV heatmap.

    Regions run along the x-axis and members down the y-axis, matching the
    reference figures.  With two references each cell is split on the
    diagonal — lower-left is the first reference, upper-right the second —
    so both can be read without doubling the figure.  A black box marks a
    positive value (the candidate beat the benchmark there).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon, Rectangle

    table = _av_lookup(rows, period_key)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    n_col, n_row = len(regions), len(members)
    split = len(references) > 1

    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.42 * n_col + 2.6), 0.42 * n_row + 2.2),
    )

    for j, member in enumerate(members):
        y = n_row - 1 - j
        for i, region in enumerate(regions):
            values = [table.get((ref, member, region)) for ref in references]
            if split:
                # Lower-left triangle = references[0], upper-right = [1].
                corners = (
                    [(i, y), (i + 1, y), (i, y + 1)],
                    [(i + 1, y), (i + 1, y + 1), (i, y + 1)],
                )
                for value, pts in zip(values, corners):
                    colour = "0.9" if value is None or not np.isfinite(value) \
                        else cmap(norm(value))
                    ax.add_patch(Polygon(pts, facecolor=colour, edgecolor="none"))
            else:
                value = values[0]
                colour = "0.9" if value is None or not np.isfinite(value) \
                    else cmap(norm(value))
                ax.add_patch(
                    Rectangle((i, y), 1, 1, facecolor=colour, edgecolor="none"),
                )

            finite = [v for v in values if v is not None and np.isfinite(v)]
            if finite and all(v > 0 for v in finite):
                ax.add_patch(Rectangle(
                    (i, y), 1, 1, facecolor="none", edgecolor="k", lw=0.8,
                ))
            if len(finite) == 1 and not split:
                ax.text(
                    i + 0.5, y + 0.5, f"{finite[0]:.2f}".lstrip("0"),
                    ha="center", va="center", fontsize=4.5,
                )

    ax.set_xlim(0, n_col)
    ax.set_ylim(0, n_row)
    ax.set_xticks(np.arange(n_col) + 0.5)
    ax.set_xticklabels(regions, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_row) + 0.5)
    ax.set_yticklabels(
        [member_label(m, config) for m in reversed(members)], fontsize=7,
    )
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(
        sm, ax=ax, orientation="horizontal", fraction=0.05,
        pad=0.28 / max(1.0, 0.42 * n_row + 2.2) + 0.12, aspect=40,
    )
    cb.set_label(subtitle, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if split:
        ax.text(
            1.005, -0.02,
            f"lower-left: {references[0]}   upper-right: {references[1]}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
        )
    fig.tight_layout()
    return fig
