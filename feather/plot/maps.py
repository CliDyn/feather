"""Map plotting wrappers using nereus."""

import cmocean  # noqa: F401  — registers cmo.* colormaps with matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_bias_map(model_data, obs_data, *,
                  bias_data=None,
                  title="", model_title="Model", obs_title="Reference",
                  bias_title="Bias (Model \u2212 Ref)",
                  projection="rob", resolution=0.25,
                  cmap="RdBu_r", bias_cmap="RdBu_r",
                  vmin=None, vmax=None, bias_vmax=None,
                  units="",
                  land=False,
                  method="nearest",
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
    bias_vmax : float, optional
        Symmetric colorbar limit for the bias panel (``-bias_vmax`` to
        ``+bias_vmax``).  If *None*, computed from the 98th percentile
        of ``|bias_data|``.  Pass a shared value across models to enable
        cross-model comparison.
    units : str
        Colorbar label (e.g. ``'K'``).
    method : str
        Interpolation method for ``nereus.plot()`` (default ``"nearest"``).
        Use ``"linear"`` for smoother rendering of coarse grids.
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
    if bias_vmax is not None:
        bias_abs_max = bias_vmax
    elif bias_data is not None:
        bv = np.asarray(bias_data).ravel()
        bv = bv[np.isfinite(bv)]
        bias_abs_max = float(np.percentile(np.abs(bv), 98)) or 1.0 if len(bv) > 0 else 1.0
    else:
        bias_abs_max = 1.0

    interpolator = None  # shared across panels on the same grid

    # --- Panel 1: Model ---
    vals, lons, lats = _flatten_latlon(model_data)
    _, _, interpolator = nr.plot(
        vals, lons, lats,
        ax=axes[0], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=True, colorbar_label=units, title=model_title,
        land=land, method=method,
    )

    # --- Panel 2: Observation (same grid → reuse interpolator) ---
    vals, lons, lats = _flatten_latlon(obs_data)
    _, _, interpolator = nr.plot(
        vals, lons, lats,
        ax=axes[1], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=True, colorbar_label=units, title=obs_title,
        land=land, method=method,
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
            land=land, method=method,
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


def plot_combined_bias_map(
    obs_data, bias_dict, *,
    title="", obs_title="Reference",
    cmap="RdBu_r", bias_cmap="RdBu_r",
    vmin=None, vmax=None, bias_vmax=None,
    units="",
    projection="rob", resolution=0.25,
    max_cols=3,
    land=False,
    method="nearest",
    figsize_per_panel=(7, 5),
    bias_title_prefix="Bias",
):
    """Combined multi-panel figure: obs climatology + bias maps.

    Produces one figure with an observation panel followed by one bias
    panel per entry in *bias_dict*.  Layout wraps to multiple rows when
    more than *max_cols* panels are needed.

    Parameters
    ----------
    obs_data : xr.DataArray
        Observation climatology on a regular lat/lon grid (2-D).
    bias_dict : dict[str, xr.DataArray]
        Ordered mapping of ``label -> bias_field``.  Each entry becomes
        one bias panel titled with the label.
    title : str
        Figure super-title.
    obs_title : str
        Title for the observation panel.
    cmap : str
        Colormap for the observation (field) panel.
    bias_cmap : str
        Diverging colormap for bias panels.
    vmin, vmax : float, optional
        Colorbar limits for the obs panel.  If *None*, computed from the
        2nd / 98th percentile of obs data.
    bias_vmax : float, optional
        Symmetric colorbar limit for bias panels (``-bias_vmax`` to
        ``+bias_vmax``).  If *None*, computed from all bias fields.
    units : str
        Colorbar label.
    projection : str
        Map projection name (default Robinson).
    resolution : float
        Nereus plotting resolution in degrees.
    max_cols : int
        Maximum columns before wrapping to a new row.
    method : str
        Interpolation method for ``nereus.plot()`` (default ``"nearest"``).
        Use ``"linear"`` for smoother rendering of coarse grids.
    figsize_per_panel : tuple
        ``(width, height)`` per panel in inches.

    Returns
    -------
    fig, axes
    """
    import math

    import nereus as nr
    from nereus.plotting import get_projection

    n_panels = 1 + len(bias_dict)
    ncols = min(n_panels, max_cols)
    nrows = math.ceil(n_panels / ncols)

    proj = get_projection(projection)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        subplot_kw={"projection": proj},
    )

    # Flatten axes to 1-D array for uniform indexing
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.asarray(axes).ravel().tolist()

    # Auto-compute vmin/vmax for obs panel
    if vmin is None or vmax is None:
        obs_vals = np.asarray(obs_data).ravel()
        obs_vals = obs_vals[np.isfinite(obs_vals)]
        if vmin is None:
            vmin = float(np.percentile(obs_vals, 2))
        if vmax is None:
            vmax = float(np.percentile(obs_vals, 98))

    # Auto-compute symmetric bias range from all bias fields
    if bias_vmax is None and bias_dict:
        all_bias = np.concatenate([
            np.asarray(b).ravel()[np.isfinite(np.asarray(b).ravel())]
            for b in bias_dict.values()
        ])
        bias_vmax = float(np.percentile(np.abs(all_bias), 98)) if len(all_bias) > 0 else 1.0
        bias_vmax = bias_vmax or 1.0

    interpolator = None  # shared across all panels (same grid)

    # --- Panel 0: Observation ---
    vals, lons, lats = _flatten_latlon(obs_data)
    _, _, interpolator = nr.plot(
        vals, lons, lats,
        ax=axes_flat[0], projection=projection, resolution=resolution,
        interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=True, colorbar_label=units, title=obs_title,
        land=land, method=method,
    )

    # --- Bias panels ---
    for i, (label, bias_field) in enumerate(bias_dict.items(), start=1):
        vals, lons, lats = _flatten_latlon(bias_field)
        _, _, interpolator = nr.plot(
            vals, lons, lats,
            ax=axes_flat[i], projection=projection, resolution=resolution,
            interpolator=interpolator, cmap=bias_cmap,
            vmin=-bias_vmax, vmax=bias_vmax,
            colorbar=True, colorbar_label=units,
            title=f"{bias_title_prefix}: {label}",
            land=land, method=method,
        )

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes_flat[:n_panels]


def plot_combined_map(
    data_dict, *,
    title="", cmap="YlOrRd",
    vmin=None, vmax=None, units="",
    projection="rob", resolution=0.25,
    max_cols=3, land=False, method="nearest",
    figsize_per_panel=(7, 5),
):
    """Combined multi-panel figure with shared colormap and range.

    All panels use the same colormap and colorbar limits.  Useful for
    fields that are always positive (e.g. standard deviation maps).

    Parameters
    ----------
    data_dict : dict[str, xr.DataArray]
        Ordered mapping of ``label -> field``.  Each entry becomes
        one panel titled with the label.
    title : str
        Figure super-title.
    cmap : str
        Colormap for all panels.
    vmin, vmax : float, optional
        Shared colorbar limits.  If *None*, computed from the
        2nd / 98th percentile across all fields.
    units : str
        Colorbar label.
    projection : str
        Map projection name (default Robinson).
    resolution : float
        Nereus plotting resolution in degrees.
    max_cols : int
        Maximum columns before wrapping to a new row.
    land : bool
        Whether to show land overlay.
    method : str
        Interpolation method for ``nereus.plot()``.
    figsize_per_panel : tuple
        ``(width, height)`` per panel in inches.

    Returns
    -------
    fig, axes
    """
    import math

    import nereus as nr
    from nereus.plotting import get_projection

    n_panels = len(data_dict)
    ncols = min(n_panels, max_cols)
    nrows = math.ceil(n_panels / ncols)

    proj = get_projection(projection)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        subplot_kw={"projection": proj},
    )

    # Flatten axes to 1-D array for uniform indexing
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.asarray(axes).ravel().tolist()

    # Auto-compute shared vmin/vmax from all fields
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            np.asarray(d).ravel()[np.isfinite(np.asarray(d).ravel())]
            for d in data_dict.values()
        ])
        if vmin is None:
            vmin = float(np.percentile(all_vals, 2))
        if vmax is None:
            vmax = float(np.percentile(all_vals, 98))

    interpolator = None  # shared across all panels (same grid)

    for i, (label, field) in enumerate(data_dict.items()):
        vals, lons, lats = _flatten_latlon(field)
        _, _, interpolator = nr.plot(
            vals, lons, lats,
            ax=axes_flat[i], projection=projection, resolution=resolution,
            interpolator=interpolator, cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar=True, colorbar_label=units, title=label,
            land=land, method=method,
        )

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes_flat[:n_panels]


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
