"""Map plotting wrappers using nereus."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_bias_map(model_data, obs_data, lon, lat, *,
                  bias_data=None,
                  title="", model_title="Model", obs_title="Observation",
                  bias_title="Bias (Model \u2212 Obs)",
                  projection="rob", resolution=0.25,
                  interpolator=None, cmap="RdBu_r", bias_cmap="RdBu_r",
                  vmin=None, vmax=None, **kwargs):
    """Three-panel figure: model | observation | bias (model - obs).

    Uses nereus.plot() for the model panel (HEALPix data) and standard
    cartopy/xarray plotting for the observation and bias panels (regular
    lat/lon grids).

    Parameters
    ----------
    model_data : array-like
        Model field (1D HEALPix).
    obs_data : xr.DataArray
        Observation field on a regular lat/lon grid.
    lon, lat : array-like
        Coordinates for model data.
    bias_data : xr.DataArray, optional
        Pre-computed bias on a regular lat/lon grid.  If *None*, the bias
        panel shows a placeholder.
    title : str
        Figure super-title.
    model_title, obs_title, bias_title : str
        Panel titles.
    projection : str
        Map projection for nereus.
    resolution : float
        Regrid resolution for nereus.
    interpolator : optional
        Reusable nereus interpolator.
    cmap : str
        Colormap for model/obs panels.
    bias_cmap : str
        Colormap for bias panel (default diverging).
    vmin, vmax : float, optional
        Colorbar limits for model/obs panels.
    **kwargs
        Extra keyword arguments passed to nereus.plot().

    Returns
    -------
    fig, axes, interpolator
    """
    import cartopy.crs as ccrs
    import nereus as nr
    from nereus.plotting import get_projection

    fig, axes = plt.subplots(
        1, 3, figsize=(20, 5),
        subplot_kw={"projection": get_projection(projection)},
    )

    # --- Panel 1: Model (HEALPix via nereus) ---
    _, _, interpolator = nr.plot(
        np.asarray(model_data), np.asarray(lon), np.asarray(lat),
        ax=axes[0], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        **kwargs,
    )
    axes[0].set_title(model_title)

    # --- Panel 2: Observation (regular lat/lon grid) ---
    if isinstance(obs_data, xr.DataArray):
        obs_data.plot(
            ax=axes[1], transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=True,
        )
        axes[1].coastlines()
        axes[1].set_global()
    axes[1].set_title(obs_title)

    # --- Panel 3: Bias ---
    if bias_data is not None and isinstance(bias_data, xr.DataArray):
        abs_max = float(np.nanpercentile(np.abs(bias_data.values), 98))
        if abs_max == 0:
            abs_max = 1.0
        bias_data.plot(
            ax=axes[2], transform=ccrs.PlateCarree(),
            cmap=bias_cmap, vmin=-abs_max, vmax=abs_max,
            add_colorbar=True,
        )
        axes[2].coastlines()
        axes[2].set_global()
    else:
        axes[2].text(
            0.5, 0.5, "Bias (pre-compute required)",
            transform=axes[2].transAxes, ha="center", va="center",
        )
    axes[2].set_title(bias_title)

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
    from nereus.plotting import get_projection

    fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                           subplot_kw={"projection": get_projection(projection)})

    _, _, interpolator = nr.plot(
        np.asarray(data), np.asarray(lon), np.asarray(lat),
        ax=ax, projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, **kwargs,
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax, interpolator
