"""Map plotting wrappers using nereus."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_bias_map(model_data, obs_data, *,
                  bias_data=None,
                  title="", model_title="Model", obs_title="Observation",
                  bias_title="Bias (Model \u2212 Obs)",
                  projection="rob", resolution=0.25,
                  cmap="RdBu_r", bias_cmap="RdBu_r",
                  vmin=None, vmax=None, units="",
                  figsize_per_panel=(7, 5)):
    """Three-panel figure: model | observation | bias (model - obs).

    Uses ``nereus.plot()`` for all three panels to ensure consistent
    sizing and colorbar placement.  All data inputs should be 2-D
    ``xr.DataArray`` objects on a regular lat/lon grid (with ``lat``
    and ``lon`` coordinates).

    Parameters
    ----------
    model_data : xr.DataArray
        Model climatology regridded to the observation grid (2-D).
    obs_data : xr.DataArray
        Observation climatology on a regular lat/lon grid (2-D).
    bias_data : xr.DataArray, optional
        Pre-computed bias on the same grid.  If *None*, the bias panel
        shows a placeholder.
    title : str
        Figure super-title.
    model_title, obs_title, bias_title : str
        Panel titles.
    projection : str
        Map projection name (default ``'rob'`` — Robinson).
    resolution : float
        Nereus plotting resolution in degrees.
    cmap : str
        Colormap for model/obs panels.
    bias_cmap : str
        Colormap for bias panel (default diverging).
    vmin, vmax : float, optional
        Colorbar limits for model/obs panels.  If *None*, computed from
        the 2nd / 98th percentile of both model and obs data.
    units : str
        Colorbar label (e.g. ``'K'``).
    figsize_per_panel : tuple
        ``(width, height)`` per panel in inches.

    Returns
    -------
    fig, axes
    """
    import nereus as nr
    from nereus.plotting import get_projection

    proj = get_projection(projection)
    ncols = 3
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1]),
        subplot_kw={"projection": proj},
    )

    # Auto-compute shared vmin/vmax for model and obs panels
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            v[np.isfinite(v)] for v in [
                np.asarray(model_data).ravel(),
                np.asarray(obs_data).ravel(),
            ]
        ])
        if vmin is None:
            vmin = float(np.percentile(all_vals, 2))
        if vmax is None:
            vmax = float(np.percentile(all_vals, 98))

    # Symmetric bounds for bias panel
    bias_abs_max = 1.0
    if bias_data is not None:
        bv = np.asarray(bias_data).ravel()
        bv = bv[np.isfinite(bv)]
        if len(bv) > 0:
            bias_abs_max = float(np.percentile(np.abs(bv), 98)) or 1.0

    interpolator = None  # shared across panels on the same grid

    # --- Panel 1: Model ---
    vals, lons, lats = _flatten_latlon(model_data)
    _, _, interpolator = nr.plot(
        vals, lons, lats,
        ax=axes[0], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=True, colorbar_label=units, title=model_title,
    )

    # --- Panel 2: Observation (same grid → reuse interpolator) ---
    vals, lons, lats = _flatten_latlon(obs_data)
    _, _, interpolator = nr.plot(
        vals, lons, lats,
        ax=axes[1], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=True, colorbar_label=units, title=obs_title,
    )

    # --- Panel 3: Bias ---
    if bias_data is not None:
        vals, lons, lats = _flatten_latlon(bias_data)
        _, _, _ = nr.plot(
            vals, lons, lats,
            ax=axes[2], projection=projection, resolution=resolution,
            interpolator=interpolator, cmap=bias_cmap,
            vmin=-bias_abs_max, vmax=bias_abs_max,
            colorbar=True, colorbar_label=units, title=bias_title,
        )
    else:
        axes[2].text(
            0.5, 0.5, "Bias (pre-compute required)",
            transform=axes[2].transAxes, ha="center", va="center",
        )
        axes[2].set_title(bias_title)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


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


def _flatten_latlon(da: xr.DataArray):
    """Flatten a 2-D lat/lon DataArray into 1-D arrays for nereus.plot().

    Returns
    -------
    (values_1d, lon_1d, lat_1d) : tuple of np.ndarray
    """
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    lons_2d, lats_2d = np.meshgrid(
        da[lon_name].values, da[lat_name].values,
    )
    return da.values.ravel(), lons_2d.ravel(), lats_2d.ravel()
