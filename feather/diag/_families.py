"""Per-model-family ensemble summaries (``project.ensemble_mode: per_family``).

Some model sets must not be pooled into one ensemble: in EERIE, ICON-ESM-ER
runs without a parameterised convection scheme, so averaging its members with
the IFS and HadGEM runs would blend physically different configurations.  In
``per_family`` mode every diagnostic that would otherwise draw a pooled
ensemble mean/median instead summarises each model family separately — the
mean of that family's members — and never averages across families.

Families come from :meth:`feather.config.FeatherConfig.get_model_families`
(``ModelConfig.family``, else the model name minus a trailing ``-rN``).  This
module holds the pieces the diagnostics share: field and time-series family
means, panel labels, and the per-family bias-map statistics.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr

from feather.util.spatial import latlon_global_mean
from feather.util.temporal import normalize_monthly_time

logger = logging.getLogger(__name__)

#: Seasons a bias-map diagnostic may carry alongside ``"annual"``.
SEASONS = ("DJF", "MAM", "JJA", "SON")


def is_per_family(config) -> bool:
    """True when *config* summarises models per family instead of pooled."""
    return config.get_ensemble_mode() == "per_family"


def family_label(family: str, n: int, *, bold: bool = True) -> str:
    """Panel/legend label for a family summary.

    A multi-member family reads ``"<family> mean (n)"``; a single-member
    family is that member, so it reads ``"<family> (1)"``.  *bold* wraps the
    count in mathtext bold, matching the benchmark-MMM panel labels.
    """
    count = rf"$\mathbf{{({n})}}$" if bold else f"({n})"
    return f"{family} mean {count}" if n > 1 else f"{family} {count}"


def describe_families(fdata: dict[str, dict]) -> str:
    """``"A (3 members), B (1 member)"`` for figure descriptions."""
    return ", ".join(
        f"{f} ({d['n_members']} member{'s' if d['n_members'] != 1 else ''})"
        for f, d in fdata.items()
    )


def family_members(fdata: dict[str, dict]) -> dict[str, list[str]]:
    """``{family: [members used]}`` for sidecar metadata."""
    return {f: list(d["members"]) for f, d in fdata.items()}


def family_mean_fields(
    fields: dict[str, xr.DataArray], families: dict[str, list[str]],
) -> dict[str, dict]:
    """Mean of each family's member fields (all on one common grid).

    Members absent from *fields* are left out, so ``n_members`` is the count
    actually averaged; families with no member present are dropped.

    Returns
    -------
    dict
        Ordered ``{family: {"mean", "n_members", "members"}}``.
    """
    out: dict[str, dict] = {}
    for family, members in families.items():
        used = [m for m in members if fields.get(m) is not None]
        if not used:
            continue
        stack = [fields[m] for m in used]
        mean = (stack[0] if len(stack) == 1
                else xr.concat(stack, dim="member").mean("member"))
        out[family] = {"mean": mean, "n_members": len(used), "members": used}
    return out


def compute_family_stats(
    model_results: dict, families: dict[str, list[str]],
    obs_clim: xr.DataArray, obs_seasonal: dict, area,
    *, annual_key: str = "annual_regrid", seasonal_key: str = "seasonal_regrids",
) -> dict[str, dict]:
    """Mean field and bias per model family, per period.

    *model_results* is the per-model output of a bias-map diagnostic, each
    entry holding its annual common-grid climatology under *annual_key* and a
    ``{season: field}`` dict under *seasonal_key*.

    Returns
    -------
    dict
        ``{period: {family: {mean, bias, bias_gmean, rmse, n_members,
        members}}}`` for ``"annual"`` and each season with obs present.
    """
    periods = [("annual", obs_clim)] + [
        (s, obs_seasonal[s]) for s in SEASONS if s in (obs_seasonal or {})
    ]
    out: dict[str, dict] = {}
    for period, obs in periods:
        fields = {
            m: (mr.get(annual_key) if period == "annual"
                else (mr.get(seasonal_key) or {}).get(period))
            for m, mr in model_results.items()
        }
        fdata = family_mean_fields(fields, families)
        for d in fdata.values():
            bias = d["mean"] - obs
            d["bias"] = bias
            d["bias_gmean"] = float(latlon_global_mean(bias, area=area).values)
            d["rmse"] = float(np.sqrt(
                latlon_global_mean(bias ** 2, area=area).values))
        if fdata:
            out[period] = fdata
    return out


def family_bias_panels(
    fdata: dict[str, dict], scale: float = 1.0,
) -> tuple[dict[str, xr.DataArray], dict[str, dict]]:
    """``(bias_dict, summary_stats)`` panels for one period's family data.

    *scale* converts the stored biases to display units (e.g. 86400 for
    precipitation in mm/day).  ``bias_gmean``/``rmse`` are optional.
    """
    bias_dict: dict[str, xr.DataArray] = {}
    stats: dict[str, dict] = {}
    for family, d in fdata.items():
        lbl = family_label(family, d["n_members"])
        bias_dict[lbl] = d["bias"] * scale
        st: dict[str, Any] = {
            "n_members": d["n_members"], "members": list(d["members"]),
        }
        if d.get("bias_gmean") is not None:
            st["global_mean_bias"] = d["bias_gmean"] * scale
        if d.get("rmse") is not None:
            st["rmse"] = d["rmse"] * scale
        stats[lbl] = st
    return bias_dict, stats


def family_mean_series(
    model_ts: dict[str, xr.DataArray], families: dict[str, list[str]],
    *, min_members: int = 2,
) -> dict[str, dict]:
    """Per-family mean of the members' time series.

    Members are put on first-of-month timestamps (mixed calendars still
    align) and reduced to the months they share before averaging.  Families
    with fewer than *min_members* series are skipped: a single-member family
    is already drawn as that member.

    Returns
    -------
    dict
        Ordered ``{family: {"ts", "n_members", "members"}}``.
    """
    out: dict[str, dict] = {}
    for family, members in families.items():
        used, series = [], []
        for m in members:
            ts = model_ts.get(m)
            if ts is None:
                continue
            ts = normalize_monthly_time(ts.reset_coords(drop=True))
            if ts is not None:
                used.append(m)
                series.append(ts)
        if len(series) < max(min_members, 1):
            continue
        if len(series) == 1:
            mean = series[0]
        else:
            aligned = xr.align(*series, join="inner")
            if aligned[0].sizes.get("time", 0) == 0:
                logger.warning(
                    "Members of %s share no common month — no family mean",
                    family)
                continue
            mean = xr.concat(list(aligned), dim="member").mean("member")
        out[family] = {"ts": mean, "n_members": len(used), "members": used}
    return out


def plot_family_lines(ax, family_ts: dict[str, dict], color_fn, *,
                      transform=None, lw: float = 3.0) -> None:
    """Draw each family's annual-mean series as a thick line.

    *color_fn(model)* gives the colour of the family's first member, so a
    family line matches its members' hue.  *transform(ts)* maps a series to
    display units (default identity).
    """
    from feather.diag._ts_panel import _to_plot_time
    from feather.util.temporal import annual_mean

    for family, d in family_ts.items():
        ts = d["ts"] if transform is None else transform(d["ts"])
        a = annual_mean(ts)
        ax.plot(_to_plot_time(a.time.values), a.values,
                color=color_fn(d["members"][0]), lw=lw, zorder=5,
                label=family_label(family, d["n_members"], bold=False))


def benchmark_panels(
    benchmark_data: dict, benchmark_info: dict | None, period_key: str,
) -> dict[str, dict]:
    """Benchmark MMM panel inputs for one period, from a bias-map result.

    Reads the common ``{label: {period: {"bias", "bias_gmean"?, "rmse"?}}}``
    layout with member counts from *benchmark_info*.
    """
    out: dict[str, dict] = {}
    for label, b_data in (benchmark_data or {}).items():
        c = b_data.get(period_key)
        if c is None:
            continue
        out[label] = {
            "bias": c["bias"],
            "bias_gmean": c.get("bias_gmean"),
            "rmse": c.get("rmse"),
            "n_members": (benchmark_info or {}).get(label, {}).get(
                "n_members", 0),
        }
    return out


def plot_family_bias_figure(
    diag, *, var: str, period_key: str, period_label: str,
    fdata: dict[str, dict], benchmarks: dict[str, dict],
    obs_plot, obs_title: str, long_name: str,
    scale: float = 1.0, cmap="RdBu_r", bias_cmap="RdBu_r",
    vmin=None, vmax=None, bias_vmax=None, units: str = "",
    land: bool = False, obs_name: str | None = None,
    **meta_kwargs,
):
    """Obs + one bias panel per model family + benchmark MMM panels.

    The per-family replacement for a diagnostic's pooled
    ``{var}_{period}_ens_bias_combined`` figure; its figure id is
    ``{var}_{period}_family_mean_bias_combined``.  Families are summarised
    separately (never pooled): a multi-member family is shown as the mean of
    its members, a single-member family as that member.

    *benchmarks* is ``{label: {"bias", "bias_gmean"?, "rmse"?,
    "n_members"}}`` for this period (see :func:`benchmark_panels`).  Stored
    values are multiplied by *scale* for display; *obs_plot* is already in
    display units.  *units* labels both the colourbars and the sidecar, as
    the statistics are in display units.  *obs_name* is the reference named
    in the description (default *obs_title*).  Remaining keywords go to
    ``diag._build_metadata``.
    """
    from feather.plot.maps import plot_combined_bias_map

    bias_dict, stats = family_bias_panels(fdata, scale)
    for label, b in benchmarks.items():
        m = b.get("n_members", 0)
        lbl = rf"{label} $\mathbf{{({m})}}$" if m else label
        bias_dict[lbl] = b["bias"] * scale
        st: dict[str, Any] = {"n_members": m}
        if b.get("bias_gmean") is not None:
            st["global_mean_bias"] = b["bias_gmean"] * scale
        if b.get("rmse") is not None:
            st["rmse"] = b["rmse"] * scale
        stats[lbl] = st

    fig, _ = plot_combined_bias_map(
        obs_plot, bias_dict,
        title=f"{long_name} {period_label} — Model means",
        obs_title=obs_title,
        cmap=cmap, bias_cmap=bias_cmap,
        vmin=vmin, vmax=vmax, bias_vmax=bias_vmax,
        units=units, land=land, method=diag._regrid_method,
    )
    meta = diag._build_metadata(
        title=f"{long_name} {period_label} Bias — Model Means",
        figure_id=f"{var}_{period_key.lower()}_family_mean_bias_combined",
        models=list(diag.config.models),
        description=(
            f"{period_label} climatology bias maps for {long_name} "
            f"(model - {obs_name or obs_title}): one panel per model family, "
            f"each the mean of its ensemble members "
            f"({describe_families(fdata)}), plus the benchmark multi-model "
            f"means. Model families are compared separately and are not "
            f"pooled into one ensemble."
        ),
        computation_notes=(
            "Each member is regridded to the common grid; members of a "
            "family are averaged there and the observed climatology is "
            "subtracted. Panel labels give the number of members (or "
            "benchmark models) averaged."
        ),
        plot_type="combined_bias_map",
        units=units,
        summary_statistics=stats,
        extra={"model_families": family_members(fdata)},
        **meta_kwargs,
    )
    return fig, meta


def family_summary_ids(var: str, periods=("annual", "djf", "mam", "jja", "son"),
                       ) -> list[str]:
    """Figure ids of the per-family bias-map summaries for *var*."""
    return [f"{var}_{p}_family_mean_bias_combined" for p in periods]


def family_lines_note(families: dict[str, list[str]] | None) -> str:
    """Description sentence for time-series figures with family-mean lines."""
    if not families:
        return ""
    multi = {f: m for f, m in families.items() if len(m) > 1}
    if not multi:
        return ""
    fams = ", ".join(f"{f} ({len(m)} members)" for f, m in multi.items())
    return (
        f" Thick lines are per-family means of the members ({fams}); model "
        f"families are summarised separately and not pooled into one ensemble."
    )
