"""Line and profile plots using matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

from feather.plot.styles import MODEL_COLORS, OBS_COLOR


def plot_timeseries(time, model_values, obs_values=None, *,
                    title="", ylabel="", model_label="Model",
                    obs_label="Obs", ax=None, **kwargs):
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
                        obs_label="Obs", ax=None, **kwargs):
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
                       obs_label="Obs", ax=None, **kwargs):
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
