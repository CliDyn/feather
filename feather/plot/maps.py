"""Map plotting wrappers using nereus."""

import matplotlib.pyplot as plt
import numpy as np


def plot_bias_map(model_data, obs_data, lon, lat, *,
                  title="", projection="rob", resolution=0.25,
                  interpolator=None, cmap="RdBu_r", **kwargs):
    """Three-panel figure: model | observation | bias (model - obs).

    Uses nereus.plot() for each panel.

    Parameters
    ----------
    model_data : array-like
        Model field (1D HEALPix or 2D).
    obs_data : array-like
        Observation field (on regular grid).
    lon, lat : array-like
        Coordinates for model data.
    title : str
        Figure title.
    projection : str
        Map projection for nereus.
    resolution : float
        Regrid resolution for nereus.
    interpolator : optional
        Reusable nereus interpolator.
    cmap : str
        Colormap for model/obs panels.
    **kwargs
        Extra keyword arguments passed to nereus.plot().

    Returns
    -------
    fig, axes, interpolator
    """
    import nereus as nr

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={"projection": nr.projection(projection)})

    # Model panel
    _, interpolator = nr.plot(
        np.asarray(model_data), np.asarray(lon), np.asarray(lat),
        ax=axes[0], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, **kwargs,
    )
    axes[0].set_title("Model")

    # Obs panel — on its own grid, use pcolormesh
    if hasattr(obs_data, "lon") and hasattr(obs_data, "lat"):
        obs_data.plot(ax=axes[1], transform=nr.projection("plate"),
                      cmap=cmap, add_colorbar=True)
    else:
        axes[1].text(0.5, 0.5, "Obs (regrid needed)", transform=axes[1].transAxes,
                     ha="center")
    axes[1].set_title("Observation")

    # Bias panel — model - obs (requires regridded obs)
    axes[2].text(0.5, 0.5, "Bias = Model - Obs\n(compute after regridding)",
                 transform=axes[2].transAxes, ha="center", va="center")
    axes[2].set_title("Bias")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig, axes, interpolator


def plot_single_map(data, lon, lat, *, title="", projection="rob",
                    resolution=0.25, interpolator=None, cmap="viridis",
                    **kwargs):
    """Single map panel via nereus.plot().

    Parameters
    ----------
    data : array-like
        Field to plot (1D HEALPix).
    lon, lat : array-like
        Coordinates.
    title : str
        Axes title.
    projection : str
        Map projection.
    resolution : float
        Regrid resolution.
    interpolator : optional
        Reusable nereus interpolator.
    cmap : str
        Colormap.
    **kwargs
        Extra keyword arguments passed to nereus.plot().

    Returns
    -------
    fig, ax, interpolator
    """
    import nereus as nr

    fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                           subplot_kw={"projection": nr.projection(projection)})

    _, interpolator = nr.plot(
        np.asarray(data), np.asarray(lon), np.asarray(lat),
        ax=ax, projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, **kwargs,
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax, interpolator
