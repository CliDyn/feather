"""Base class for all diagnostics in Feather.

Every diagnostic inherits from :class:`DiagnosticBase` and implements
``compute()`` and ``plot()`` methods. The ``run()`` method orchestrates
both steps, saves figures with metadata sidecars, and returns the list
of generated file paths.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from feather.config import FeatherConfig
from feather.data.loader import DataLoader
from feather.data.obs import ObsLoader
from feather.diag.figure_meta import build_metadata, save_figure_with_metadata

logger = logging.getLogger(__name__)


class DiagnosticBase(ABC):
    """Abstract base class for climate diagnostics.

    Subclasses must define class-level attributes and implement
    ``compute()`` and ``plot()``.

    Class attributes
    ----------------
    name : str
        Machine-readable identifier (used in filenames, CLI).
    title : str
        Human-readable title for plot suptitles.
    domain : str
        Data domain — ``'sfc'``, ``'o2d'``, ``'pl'``, or ``'o3d'``.
    variables : list[str]
        Model variable names used by the diagnostic.
    group : str
        Thematic group for dashboard navigation (e.g. ``'temperature'``).

    Parameters
    ----------
    model_loader : DataLoader
        Initialised model data loader.
    obs_loader : ObsLoader
        Observation data loader.
    config : FeatherConfig
        Pipeline configuration.
    cmip6_loader : optional
        CMIP6 data loader (Phase 4+).
    """

    name: str = ""
    title: str = ""
    domain: str = "sfc"
    variables: list[str] = []
    group: str = ""

    def __init__(
        self,
        model_loader: DataLoader,
        obs_loader: ObsLoader,
        config: FeatherConfig,
        *,
        cmip6_loader: Any = None,
    ):
        self.model_loader = model_loader
        self.obs_loader = obs_loader
        self.config = config
        self.cmip6_loader = cmip6_loader

    # ── Properties ────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Output directory for this diagnostic's figures."""
        return Path(self.config.output_dir) / "figures" / self.name

    @property
    def cmip6_enabled(self) -> bool:
        """True when CMIP6 data is available and enabled in config."""
        return (
            self.cmip6_loader is not None
            and self.config.cmip6.get("enabled", False)
        )

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Run the computation and return named results.

        Returns
        -------
        dict[str, Any]
            Arbitrary results dict consumed by ``plot()``.
        """

    @abstractmethod
    def plot(self, results: dict[str, Any]) -> list[tuple[plt.Figure, dict]]:
        """Generate figures from computation results.

        Parameters
        ----------
        results : dict
            Output of ``compute()``.

        Returns
        -------
        list of (fig, metadata)
            Each element is a matplotlib Figure paired with its
            metadata dict (built via :func:`build_metadata`).
        """

    # ── Orchestration ─────────────────────────────────────────────────

    def run(self, skip_existing: bool = True) -> list[tuple[Path, Path]]:
        """Execute the full diagnostic: compute → plot → save.

        Parameters
        ----------
        skip_existing : bool
            When True, subclass overrides may skip variables whose
            output figures already exist on disk.  The default base
            implementation does not skip (subclasses opt in).

        Returns
        -------
        list of (png_path, json_path)
            All generated figure/metadata pairs.
        """
        logger.info("Running diagnostic: %s", self.name)
        results = self.compute()
        figure_pairs = self.plot(results)

        saved = []
        for fig, meta in figure_pairs:
            png_path, json_path = save_figure_with_metadata(
                fig, meta, self.output_dir, meta["figure_id"],
            )
            saved.append((png_path, json_path))

        logger.info(
            "Diagnostic %s complete — %d figure(s)", self.name, len(saved),
        )
        return saved

    # ── Helpers for subclasses ────────────────────────────────────────

    def _build_metadata(
        self,
        title: str,
        figure_id: str,
        models: list[str],
        *,
        description: str = "",
        computation_notes: str = "",
        period: tuple[str, str] | None = None,
        obs_dataset: str = "",
        obs_variable: str = "",
        plot_type: str = "",
        spatial_extent: str = "global",
        summary_statistics: dict[str, Any] | None = None,
        variables: list[str] | None = None,
        cmip6_info: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper around :func:`build_metadata`."""
        return build_metadata(
            diagnostic_name=self.name,
            title=title,
            figure_id=figure_id,
            variables_used=variables or self.variables,
            models=models,
            description=description,
            computation_notes=computation_notes,
            period=period,
            obs_dataset=obs_dataset,
            obs_variable=obs_variable,
            plot_type=plot_type,
            spatial_extent=spatial_extent,
            summary_statistics=summary_statistics,
            cmip6_info=cmip6_info,
            extra=extra,
        )

    def _save(
        self,
        fig: plt.Figure,
        metadata: dict[str, Any],
        filename: str,
    ) -> tuple[Path, Path]:
        """Save figure + JSON sidecar to this diagnostic's output dir."""
        return save_figure_with_metadata(
            fig, metadata, self.output_dir, filename,
        )

    def _figure_exists(self, figure_id: str) -> bool:
        """Check if both PNG and JSON sidecar exist for *figure_id*."""
        d = self.output_dir
        return (d / f"{figure_id}.png").exists() and (d / f"{figure_id}.json").exists()

    def _load_model_var(
        self,
        model: str,
        variable: str,
        *,
        experiment: str = "",
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> "xr.DataArray":
        """Load a model variable, dispatching to the correct loader.

        For DestinE (HEALPix/catalog), uses ``self.model_loader`` with
        ``DataLoader.make_key()``.  For CMOR (lat/lon), uses
        ``self.model_loader.load_var()`` directly.

        Parameters
        ----------
        model : str
            Model name.
        variable : str
            CMOR variable name (e.g. ``"tas"``).
        experiment : str
            Experiment key (DestinE only, defaults to config).
        period : tuple of str, optional
            (start, end) for time slicing.
        time_mean : bool
            Whether to compute time mean.

        Returns
        -------
        xr.DataArray
        """
        from feather.data.variables import get_var

        if self.config.get_data_source_type() in ("cmor", "netcdf_healpix"):
            return self.model_loader.load_var(
                model, variable, period=period, time_mean=time_mean,
            )

        # DestinE path: build catalog key, look up DestinE variable name
        vinfo = get_var(variable)
        destine_var = vinfo.destine_variable or variable
        exp = experiment or self.config.get_experiment()
        key = DataLoader.make_key(exp, model, vinfo.domain)
        da = self.model_loader.load_var(key, destine_var)
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def _load_model_coords(
        self, model: str, variable: str, *, experiment: str = "",
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Load lon/lat coordinates for a model variable.

        Returns
        -------
        (lon, lat) : tuple of np.ndarray
            1D longitude and latitude arrays.
        """
        import numpy as np

        from feather.data.variables import get_var

        src = self.config.get_data_source_type()
        if src == "cmor":
            da = self.model_loader.load_var(model, variable)
            return np.asarray(da.lon), np.asarray(da.lat)

        if src == "netcdf_healpix":
            da = self.model_loader.load_var(model, variable)
            return np.asarray(da["longitude"]), np.asarray(da["latitude"])

        # DestinE: coords are in the Dataset
        vinfo = get_var(variable)
        exp = experiment or self.config.get_experiment()
        key = DataLoader.make_key(exp, model, vinfo.domain)
        ds = self.model_loader.load(key)
        return np.asarray(ds["longitude"]), np.asarray(ds["latitude"])

    def _model_global_mean(
        self, da: "xr.DataArray", model: str,
    ) -> "xr.DataArray":
        """Compute area-weighted global mean, dispatching by grid type.

        HEALPix: simple ``.mean("values")`` (equal-area cells).
        Lat/lon: ``latlon_global_mean()`` with cos-lat weighting.

        Parameters
        ----------
        da : xr.DataArray
            Model data array.
        model : str
            Model name (for grid type lookup).

        Returns
        -------
        xr.DataArray
            Scalar or time series of global means.
        """
        grid_type = self.config.get_grid_type(model, self.domain)
        if grid_type == "healpix":
            from feather.util.spatial import global_mean
            return global_mean(da)

        from feather.util.spatial import latlon_global_mean
        return latlon_global_mean(da)

    def _load_obs_var(
        self, variable: str, period: tuple[str, str] | None = None,
    ) -> "xr.DataArray":
        """Load observation data with sign correction for CMOR sources.

        When the model data source is CMOR (e.g. EERIE), variables like
        ``hfss`` and ``hfls`` need their ERA5 obs values negated to match
        the CMOR sign convention (positive upward for surface fluxes).
        """
        from feather.data.variables import get_var

        da = self.obs_loader.load_for_model_var(variable, period)

        if self.config.get_data_source_type() == "cmor":
            vinfo = get_var(variable)
            if vinfo.cmor_obs_sign != 1.0:
                da = da * vinfo.cmor_obs_sign

        return da

    def _cmip6_global_mean_timeseries(
        self,
        var: str,
        period: tuple[str, str] | None = None,
        return_individual: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Compute CMIP6 ensemble-mean global-mean monthly time series.

        Loads raw monthly data (``time_mean=False``) for each CMIP6 model,
        computes area-weighted global mean per timestep, then averages
        across models.

        Parameters
        ----------
        var : str
            Feather variable name (e.g. ``"tas"``).
        period : tuple of str, optional
            (start, end) for time slicing.
        return_individual : bool, optional
            When True, ``info["individual_series"]`` contains a dict
            mapping model name → aligned individual time series.

        Returns
        -------
        (mmm_ts, info) : tuple
            mmm_ts: DataArray with ``time`` dim, or None.
            info: dict with ``n_members``, ``models_used``, and
            optionally ``individual_series``.
        """
        import numpy as np
        import xarray as xr

        from feather.data.variables import get_var
        from feather.util.spatial import latlon_global_mean

        if not self.cmip6_enabled:
            return None, {}

        vinfo = get_var(var)
        if not vinfo.cmip6_variable:
            return None, {}

        logger.info("  Computing CMIP6 global-mean time series for %s (%s)",
                     var, vinfo.cmip6_variable)

        member_series = []
        models_used = []

        for model in self.cmip6_loader.models:
            da = self.cmip6_loader.load_var(
                vinfo.cmip6_variable, model,
                table=vinfo.cmip6_table or None,
                period=period,
                time_mean=False,
            )
            if da is None:
                continue

            area = self.cmip6_loader.load_area(
                model, table=vinfo.cmip6_table or "Amon",
            )
            # Convert areacella to numpy so latlon_global_mean wraps it
            # with da's own coordinates — avoids misalignment when
            # areacella has different dim names or coordinate values.
            area = self._align_area(da, area)
            ts = latlon_global_mean(da, area=area)
            member_series.append(ts)
            models_used.append(model)

        if not member_series:
            logger.info("    No CMIP6 models available for %s", var)
            return None, {}

        # Align to common time axis, then ensemble mean
        aligned = xr.align(*member_series, join="inner")
        mmm_ts = sum(aligned) / len(aligned)
        info = {"n_members": len(models_used), "models_used": models_used}
        if return_individual:
            info["individual_series"] = dict(zip(models_used, aligned))
        logger.info("    CMIP6 MMM time series: %d models, %d timesteps",
                     len(models_used), len(mmm_ts.time))
        return mmm_ts, info

    @staticmethod
    def _align_area(da, area):
        """Convert area weights to numpy aligned with da's spatial grid.

        CMIP6 areacella may have different dimension names or slightly
        different coordinate values than the data variable.  Passing the
        raw xr.DataArray to ``da.weighted(area)`` causes silent
        misalignment.  Converting to numpy and letting
        ``latlon_global_mean`` re-wrap with da's own coords fixes this.

        Returns numpy array if shapes match, else None (fallback to
        nereus-computed areas).
        """
        if area is None:
            return None

        import numpy as np

        from feather.util.spatial import _find_latlon_dims

        area_np = np.asarray(area)
        try:
            lat_name, lon_name = _find_latlon_dims(da)
        except ValueError:
            return None

        expected_shape = (len(da[lat_name]), len(da[lon_name]))
        if area_np.shape == expected_shape:
            return area_np

        logger.warning(
            "areacella shape %s != data shape %s — falling back to "
            "nereus areas",
            area_np.shape, expected_shape,
        )
        return None
