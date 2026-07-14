"""Line and profile plots using matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

from feather.plot.styles import CMIP6_COLOR, MODEL_COLORS, OBS_COLOR


def plot_timeseries(time, model_values, obs_values=None, *,
                    title="", ylabel="", model_label="Model",
                    obs_label="Reference", ax=None, **kwargs):
    """Line plot of a time series, optionally with obs overlay.

    Parameters
    ----------
    time : array-like
        Time coordinates.
    model_values : array-like
        Model time series.
    obs_values : array-like, optional
        Observation time series.
    title, ylabel : str
        Plot labels.
    model_label, obs_label : str
        Legend labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    **kwargs
        Extra keyword arguments passed to plot().

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(time, model_values, label=model_label, **kwargs)
    if obs_values is not None:
        ax.plot(time, obs_values, label=obs_label, color=OBS_COLOR,
                linewidth=2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_seasonal_cycle(model_monthly, obs_monthly=None, *,
                        title="", ylabel="", model_label="Model",
                        obs_label="Reference", ax=None, **kwargs):
    """12-month cycle (Jan-Dec) line plot.

    Parameters
    ----------
    model_monthly : array-like
        Model monthly climatology (length 12).
    obs_monthly : array-like, optional
        Observation monthly climatology (length 12).
    title, ylabel : str
        Plot labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    **kwargs
        Extra keyword arguments passed to plot().

    Returns
    -------
    fig, ax
    """
    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J",
                    "J", "A", "S", "O", "N", "D"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(months, model_monthly, marker="o", label=model_label, **kwargs)
    if obs_monthly is not None:
        ax.plot(months, obs_monthly, marker="s", label=obs_label,
                color=OBS_COLOR, linewidth=2)

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_zonal_profile(lats, model_values, obs_values=None, *,
                       title="", xlabel="", model_label="Model",
                       obs_label="Reference", ax=None, **kwargs):
    """Zonal mean profile: latitude on y-axis, variable on x-axis.

    Parameters
    ----------
    lats : array-like
        Latitude values.
    model_values : array-like
        Model zonal means.
    obs_values : array-like, optional
        Observation zonal means.
    title, xlabel : str
        Plot labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    **kwargs
        Extra keyword arguments passed to plot().

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    ax.plot(model_values, lats, label=model_label, **kwargs)
    if obs_values is not None:
        ax.plot(obs_values, lats, label=obs_label, color=OBS_COLOR,
                linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Latitude")
    ax.set_ylim(-90, 90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_budget_bars(budget_data, *, title="Radiation Budget",
                     ylabel="W/m\u00b2", ax=None, colors=None):
    """Grouped bar chart of radiation budget components.

    Parameters
    ----------
    budget_data : dict[str, dict[str, float]]
        Outer keys are component names (e.g. "TOA SW", "TOA LW", ...),
        inner keys are source labels (model names, "Obs", "CMIP6 MMM"),
        values are global-mean flux values in W/m².
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    colors : dict[str, str], optional
        Explicit source-label → colour map (e.g. per-model config colours).
        Falls back to :func:`_budget_bar_color` for any source not present.

    Returns
    -------
    fig, ax
    """
    colors = colors or {}
    if not budget_data:
        fig, ax_ = plt.subplots(figsize=(12, 6))
        ax_.set_title(title)
        return fig, ax_

    components = list(budget_data.keys())
    # Collect all source labels across components (preserving order)
    all_sources = []
    seen = set()
    for comp_data in budget_data.values():
        for src in comp_data:
            if src not in seen:
                all_sources.append(src)
                seen.add(src)

    n_groups = len(components)
    n_bars = len(all_sources)
    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.5), 6))
    else:
        fig = ax.figure

    for i, src in enumerate(all_sources):
        values = [budget_data[comp].get(src, 0.0) for comp in components]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        color = colors.get(src) or _budget_bar_color(src)
        ax.bar(x + offset, values, bar_width, label=src, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()

    return fig, ax


def plot_gregory(scatter_data, *, title="Gregory Plot",
                 xlabel="Global Mean T2m (K)",
                 ylabel="Net TOA Radiation (W/m\u00b2)",
                 ax=None):
    """Gregory scatter plot: global-mean T2m vs net TOA radiation.

    Parameters
    ----------
    scatter_data : list[dict]
        Each dict has keys:
        - ``label``: str
        - ``t2m_monthly``: array of monthly global-mean T2m
        - ``toa_monthly``: array of monthly net TOA
        - ``t2m_annual``: array of annual-mean T2m (optional)
        - ``toa_annual``: array of annual-mean net TOA (optional)
        - ``color``: str (optional)
        - ``alpha``: float (optional, for scatter points)
        - ``show_regression``: bool (optional, default True)
    title, xlabel, ylabel : str
        Plot labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for entry in scatter_data:
        label = entry["label"]
        t2m_m = np.asarray(entry["t2m_monthly"])
        toa_m = np.asarray(entry["toa_monthly"])
        color = entry.get("color")
        alpha = entry.get("alpha", 0.4)
        show_reg = entry.get("show_regression", True)

        # Monthly scatter (small, semi-transparent)
        ax.scatter(t2m_m, toa_m, s=10, alpha=alpha, color=color,
                   label=f"{label} (monthly)")

        # Annual means as larger markers
        if "t2m_annual" in entry and "toa_annual" in entry:
            t2m_a = np.asarray(entry["t2m_annual"])
            toa_a = np.asarray(entry["toa_annual"])
            ax.scatter(t2m_a, toa_a, s=60, marker="D", color=color,
                       edgecolors="k", linewidth=0.5, zorder=5,
                       label=f"{label} (annual)")

        # Linear regression
        if show_reg and len(t2m_m) > 1:
            mask = np.isfinite(t2m_m) & np.isfinite(toa_m)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(t2m_m[mask], toa_m[mask], 1)
                x_fit = np.array([t2m_m[mask].min(), t2m_m[mask].max()])
                ax.plot(x_fit, slope * x_fit + intercept, color=color,
                        linewidth=1.5, linestyle=entry.get("linestyle", "-"),
                        label=f"{label} ({slope:.2f} W/m\u00b2/K)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_taylor_diagram(
    model_stats,
    *,
    title="Taylor Diagram",
    obs_label="Obs",
    cmip6_stats=None,
    cmip6_individual_stats=None,
    season_markers=None,
    model_colors=None,
    figsize=(8, 8),
):
    """Taylor diagram: pattern correlation vs normalised standard deviation.

    Points are plotted on a polar axes where ``theta = arccos(correlation)``
    and ``r = normalised_std`` (model STD / obs STD).  The reference
    observation sits at (r=1, theta=0).  CRMS contours are centred on
    the reference point.

    Parameters
    ----------
    model_stats : dict[str, dict[str, dict]]
        ``{model: {season: {"corr": R, "std_ratio": sigma_m/sigma_o}}}``.
    title : str
        Figure title.
    obs_label : str
        Label for the reference marker.
    cmip6_stats : dict, optional
        Same structure as *model_stats* for CMIP6 MMM.
    cmip6_individual_stats : dict, optional
        Same structure for individual CMIP6 models.
    season_markers : dict, optional
        ``{season: marker_char}``.  Defaults to ``{"ANN": "o", "DJF": "v", "JJA": "^"}``.
    model_colors : dict, optional
        ``{model: hex_color}``.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax
    """
    if season_markers is None:
        season_markers = {"ANN": "o", "DJF": "v", "JJA": "^"}
    if model_colors is None:
        model_colors = {}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Restrict to first quadrant (corr >= 0 → theta 0..pi/2)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # 0° at top

    # Reference point (perfect model)
    ax.plot(0.0, 1.0, marker="*", markersize=15, color=OBS_COLOR,
            label=obs_label, zorder=10)

    # CRMS contours centred on (r=1, theta=0)
    max_r = 2.0
    theta_grid = np.linspace(0, np.pi / 2, 100)
    for crms_val in [0.25, 0.5, 0.75, 1.0, 1.5]:
        r_contour = []
        for th in theta_grid:
            # CRMS² = 1 + r² - 2*r*cos(theta) → solve for r
            # quadratic: r² - 2*cos(theta)*r + (1 - CRMS²) = 0
            a_coeff = 1.0
            b_coeff = -2.0 * np.cos(th)
            c_coeff = 1.0 - crms_val ** 2
            disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
            if disc < 0:
                r_contour.append(np.nan)
                continue
            r1 = (-b_coeff + np.sqrt(disc)) / 2.0
            r2 = (-b_coeff - np.sqrt(disc)) / 2.0
            r_pos = r1 if r1 >= 0 else r2
            if r_pos < 0 or r_pos > max_r:
                r_contour.append(np.nan)
            else:
                r_contour.append(r_pos)
        ax.plot(theta_grid, r_contour, color="gray", linewidth=0.5,
                linestyle="--", alpha=0.5)

    # Plot CMIP6 individual (background)
    if cmip6_individual_stats:
        first = True
        for _model, seasons in cmip6_individual_stats.items():
            for season, stats in seasons.items():
                corr = stats["corr"]
                std_r = stats["std_ratio"]
                theta = np.arccos(np.clip(corr, 0.0, 1.0))
                marker = season_markers.get(season, "o")
                label = "CMIP6 members" if first else "_nolegend_"
                ax.plot(theta, std_r, marker=marker, color=CMIP6_COLOR,
                        alpha=0.3, markersize=5, linestyle="none",
                        label=label)
                first = False

    # Plot CMIP6 MMM
    if cmip6_stats:
        for season, stats in cmip6_stats.items():
            corr = stats["corr"]
            std_r = stats["std_ratio"]
            theta = np.arccos(np.clip(corr, 0.0, 1.0))
            marker = season_markers.get(season, "o")
            ax.plot(theta, std_r, marker=marker, color=CMIP6_COLOR,
                    markersize=8, linestyle="none",
                    label=f"CMIP6 MMM ({season})")

    # Plot evaluated models
    legend_seasons_done = set()
    for model, seasons in model_stats.items():
        color = model_colors.get(model, None)
        for season, stats in seasons.items():
            corr = stats["corr"]
            std_r = stats["std_ratio"]
            theta = np.arccos(np.clip(corr, 0.0, 1.0))
            marker = season_markers.get(season, "o")
            label = f"{model} ({season})"
            ax.plot(theta, std_r, marker=marker, color=color,
                    markersize=8, linestyle="none", label=label)

    # Axis labels
    ax.set_rlabel_position(0)
    ax.set_ylabel("Normalised Standard Deviation", labelpad=30)

    # Correlation labels on theta axis
    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ax.set_thetagrids(
        [np.degrees(np.arccos(c)) for c in corr_ticks],
        labels=[str(c) for c in corr_ticks],
    )

    ax.set_rmax(max_r)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)

    plt.tight_layout()
    return fig, ax


def _budget_bar_color(source: str) -> str:
    """Pick bar color for a radiation budget source label."""
    if source in MODEL_COLORS:
        return MODEL_COLORS[source]
    if source.lower() in ("obs", "ceres", "ceres + era5"):
        return OBS_COLOR
    if "cmip6" in source.lower():
        return CMIP6_COLOR
    return "#666666"
