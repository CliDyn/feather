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
        benchmarks: list | None = None,
        save_netcdf: bool = False,
    ):
        self.model_loader = model_loader
        self.obs_loader = obs_loader
        self.config = config
        # When True, diagnostics also write their per-source fields (obs,
        # evaluated models, benchmark MMMs) to NetCDF under
        # ``{output}/netcdf/{name}/`` via ``_maybe_export_netcdf``.
        self.save_netcdf = save_netcdf
        # ``benchmarks`` is the ordered list of benchmark loaders (CMIP6,
        # HighResMIP, …).  ``cmip6_loader`` is the primary (first) benchmark,
        # kept for diagnostics not yet generalised to multiple benchmarks.
        self._benchmarks_explicit = benchmarks is not None
        if benchmarks is not None:
            self.benchmarks = list(benchmarks)
        elif cmip6_loader is not None:
            self.benchmarks = [cmip6_loader]
        else:
            self.benchmarks = []
        if cmip6_loader is None and self.benchmarks:
            cmip6_loader = self.benchmarks[0]
        self.cmip6_loader = cmip6_loader

    # ── Properties ────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Output directory for this diagnostic's figures."""
        return Path(self.config.output_dir) / "figures" / self.name

    @property
    def cmip6_enabled(self) -> bool:
        """True when a (primary) benchmark loader is available and enabled.

        Honours both the legacy ``cmip6.enabled`` flag and the new
        ``benchmarks:`` list (presence of a benchmark loader).
        """
        return self.cmip6_loader is not None and (
            self.config.cmip6.get("enabled", False)
            or self._benchmarks_explicit
        )

    # ── NetCDF export ─────────────────────────────────────────────────

    @property
    def _netcdf_dir(self) -> Path:
        """Directory for this diagnostic's per-source NetCDF files."""
        return Path(self.config.output_dir) / "netcdf" / self.name

    def _maybe_export_netcdf(self, results, token: str) -> None:
        """Write *results*' per-source fields to NetCDF when requested.

        No-op unless ``self.save_netcdf`` is True. *token* names the file
        (typically a variable or mode); the analysis period is appended.
        Existing files are skipped. Failures are logged, never raised — the
        export must never break the diagnostic.
        """
        if not getattr(self, "save_netcdf", False):
            return
        from feather.diag import netcdf_export

        period = getattr(self, "period", None) or self.config.get_period()
        try:
            netcdf_export.export_generic_netcdf(
                self._netcdf_dir, token, results, period, skip_existing=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "NetCDF export failed for %s/%s", self.name, token,
                exc_info=True,
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
        benchmark_info: dict[str, Any] | None = None,
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
            benchmark_info=benchmark_info,
            extra=extra,
        )

    @staticmethod
    def _benchmark_meta_from_info(
        benchmark_info: dict | None,
    ) -> dict | None:
        """Normalize a ``{label: info}`` dict to per-benchmark member info.

        Returns ``{label: {n_members, models_used}}`` for the JSON sidecar,
        or ``None`` when no benchmark info is available.
        """
        if not benchmark_info:
            return None
        out = {}
        for label, info in benchmark_info.items():
            if not info:
                continue
            out[label] = {
                "n_members": info.get("n_members"),
                "models_used": info.get("models_used"),
            }
        return out or None

    @staticmethod
    def _benchmark_meta_from_list(benchmarks: list | None) -> dict | None:
        """Per-benchmark member info from a list of ``{label, info}`` entries."""
        if not benchmarks:
            return None
        out = {}
        for b in benchmarks:
            info = b.get("info") or {}
            out[b["label"]] = {
                "n_members": info.get("n_members"),
                "models_used": info.get("models_used"),
            }
        return out or None

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

    # ── Summary-statistics helpers (lat/lon fields & series) ───────────

    @staticmethod
    def _latlon_field_mean(field: "xr.DataArray") -> float:
        """Area-weighted (proper cell areas) mean of a 2-D lat/lon field.

        NaN cells (e.g. ocean on a land-only field) are skipped by
        ``xr.DataArray.weighted``.  Returns ``nan`` if the field is empty.
        """
        import numpy as np
        import xarray as xr

        from feather.util.spatial import compute_latlon_areas, latlon_global_mean

        areas = compute_latlon_areas(
            np.asarray(field["lat"]), np.asarray(field["lon"]),
        )
        areas_da = xr.DataArray(areas, dims=("lat", "lon"))
        return float(latlon_global_mean(field, areas_da).values)

    @staticmethod
    def _latlon_bias_stats(
        model_field: "xr.DataArray",
        obs_field: "xr.DataArray",
    ) -> dict[str, float]:
        """Area-weighted bias-map statistics for ``model_field`` vs ``obs_field``.

        Mirrors the stats reported by :class:`GlobalBiases`:
        ``global_mean_bias`` and ``rmse`` (area-weighted, NaN-safe), plus a
        paired t-test (mean bias ≠ 0) and a variance-ratio F-test, both
        area-weighted via :func:`spatial_ttest` / :func:`spatial_variance_ratio`.
        """
        import numpy as np
        import xarray as xr

        from feather.util.spatial import (
            compute_latlon_areas,
            latlon_global_mean,
            spatial_ttest,
            spatial_variance_ratio,
        )

        areas = compute_latlon_areas(
            np.asarray(model_field["lat"]), np.asarray(model_field["lon"]),
        )
        areas_da = xr.DataArray(areas, dims=("lat", "lon"))

        bias = model_field - obs_field
        gmean = float(latlon_global_mean(bias, areas_da).values)
        rmse = float(np.sqrt(latlon_global_mean(bias ** 2, areas_da).values))
        t_stat, t_pval = spatial_ttest(model_field, obs_field, weights=areas)
        f_stat, f_pval = spatial_variance_ratio(
            model_field, obs_field, weights=areas,
        )
        return {
            "global_mean_bias": gmean,
            "rmse": rmse,
            "t_test_statistic": t_stat,
            "t_test_p_value": t_pval,
            "variance_ratio": f_stat,
            "variance_ratio_p_value": f_pval,
        }

    @staticmethod
    def _series_stats(series: "xr.DataArray") -> dict[str, float]:
        """Mean, linear trend (per decade) and endpoints of a yearly series.

        ``series`` is a 1-D DataArray indexed by ``year``.  The trend is a
        least-squares slope in units-per-decade; ``start_value`` / ``end_value``
        are the first and last finite annual values.
        """
        import numpy as np

        years = np.asarray(series["year"], dtype=float)
        vals = np.asarray(series, dtype=float)
        finite = np.isfinite(years) & np.isfinite(vals)
        if finite.sum() < 2:
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            return {
                "mean": mean,
                "trend_per_decade": float("nan"),
                "start_value": float("nan"),
                "end_value": float("nan"),
            }
        yf, vf = years[finite], vals[finite]
        slope = float(np.polyfit(yf, vf, 1)[0])
        return {
            "mean": float(np.mean(vf)),
            "trend_per_decade": slope * 10.0,
            "start_value": float(vf[0]),
            "end_value": float(vf[-1]),
        }

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
        from feather.data.composite_loader import CompositeModelLoader
        from feather.data.variables import get_var

        if isinstance(self.model_loader, CompositeModelLoader):
            return self.model_loader.load_var(
                model, variable, period=period, time_mean=time_mean,
            )

        if self.config.get_data_source_type() in ("cmor", "netcdf_healpix",
                                                      "grib_healpix", "cordex",
                                                      "cmip5", "cmip6_nc"):
            return self.model_loader.load_var(
                model, variable, period=period, time_mean=time_mean,
            )

        # DestinE path: build catalog key, look up DestinE variable name
        vinfo = get_var(variable)
        destine_var = vinfo.destine_variable or variable
        mc = self.config.model_configs.get(model)
        member = mc.member if mc else 1

        # Experiments to stitch along time. An explicit ``experiment`` arg
        # forces a single entry; otherwise use the configured list (which
        # may concatenate e.g. baseline_hist + projections_ssp3-7.0).
        experiments = [experiment] if experiment else self.config.get_experiments()

        da = self._load_destine_stitched(
            model, destine_var, vinfo.domain, member, experiments,
        )
        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def _load_destine_stitched(
        self,
        model: str,
        destine_var: str,
        domain: str,
        member: int,
        experiments: list[str],
    ) -> "xr.DataArray":
        """Load a DestinE variable, concatenating across experiments in time.

        Each experiment maps to a separate catalog entry. Entries that are
        absent for this model (e.g. a projection a model did not run) are
        skipped. The surviving segments are concatenated along ``time``,
        sorted, and de-duplicated (overlapping months keep the first
        experiment in ``experiments`` order). A single experiment is loaded
        directly without concatenation.

        Raises ``KeyError`` if none of the experiments are available.
        """
        import numpy as np
        import xarray as xr

        segments: list[xr.DataArray] = []
        for exp in experiments:
            key = DataLoader.make_key(exp, model, domain, member=member)
            try:
                segments.append(self.model_loader.load_var(key, destine_var))
            except (KeyError, FileNotFoundError):
                logger.debug(
                    "Experiment %s unavailable for %s/%s — skipping segment",
                    exp, model, destine_var,
                )

        if not segments:
            raise KeyError(
                f"No data for {model!r}/{destine_var!r} in experiments "
                f"{experiments}"
            )
        if len(segments) == 1:
            return segments[0]

        combined = xr.concat(segments, dim="time")
        if "time" in combined.dims:
            combined = combined.sortby("time")
            # Drop duplicate timestamps from overlapping experiments
            _, keep = np.unique(combined["time"].values, return_index=True)
            if keep.size != combined.sizes["time"]:
                combined = combined.isel(time=np.sort(keep))
        return combined

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

        from feather.data.composite_loader import CompositeModelLoader
        from feather.data.variables import get_var

        if isinstance(self.model_loader, CompositeModelLoader):
            return self.model_loader.load_coords(model, variable)

        src = self.config.get_data_source_type()
        if src in ("cmor", "cordex", "cmip5", "cmip6_nc"):
            da = self.model_loader.load_var(model, variable)
            return np.asarray(da.lon), np.asarray(da.lat)

        if src in ("netcdf_healpix", "grib_healpix"):
            da = self.model_loader.load_var(model, variable)
            return np.asarray(da["longitude"]), np.asarray(da["latitude"])

        # DestinE: coords are in the Dataset and are time-invariant, so use
        # the first experiment available for this model.
        vinfo = get_var(variable)
        experiments = [experiment] if experiment else self.config.get_experiments()
        mc = self.config.model_configs.get(model)
        member = mc.member if mc else 1
        ds = None
        for exp in experiments:
            key = DataLoader.make_key(exp, model, vinfo.domain, member=member)
            try:
                ds = self.model_loader.load(key)
                break
            except (KeyError, FileNotFoundError):
                continue
        if ds is None:
            raise KeyError(
                f"No coordinates for {model!r} in experiments {experiments}"
            )
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
        loader: Any = None,
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

        loader = loader or self.cmip6_loader
        if loader is None or not self.cmip6_enabled:
            return None, {}

        vinfo = get_var(var)
        if not vinfo.cmip6_variable:
            return None, {}

        logger.info("  Computing benchmark global-mean time series for %s (%s)",
                     var, vinfo.cmip6_variable)

        member_series = []
        models_used = []

        for model in loader.models:
            da = loader.load_var(
                vinfo.cmip6_variable, model,
                table=vinfo.cmip6_table or None,
                period=period,
                time_mean=False,
            )
            if da is None:
                continue

            area = loader.load_area(
                model, table=vinfo.cmip6_table or "Amon",
            )
            # Convert areacella to numpy so latlon_global_mean wraps it
            # with da's own coordinates — avoids misalignment when
            # areacella has different dim names or coordinate values.
            area = self._align_area(da, area)
            try:
                ts = latlon_global_mean(da, area=area)
            except (ValueError, KeyError) as e:
                # Skip models on grids we cannot reduce to lat/lon
                # (e.g. unstructured ICON grids with dims like (time, i)).
                logger.warning(
                    "    Skipping %s for %s — cannot compute global mean: %s",
                    model, var, e,
                )
                continue
            # Drop non-dimension scalar coords (e.g. ``height`` on tas) that
            # some models carry and others don't — otherwise the cross-member
            # concat below raises on mismatched coords.
            member_series.append(ts.reset_coords(drop=True))
            models_used.append(model)

        if not member_series:
            logger.info("    No CMIP6 models available for %s", var)
            return None, {}

        # Align on the union of time steps (outer join) and average over the
        # members available at each step.  An inner join would truncate the
        # whole MMM to the shortest member's record (e.g. a HighResMIP member
        # that starts mid-period), which is not what we want.
        aligned = xr.align(*member_series, join="outer")
        mmm_ts = xr.concat(aligned, dim="member").mean("member", skipna=True)
        info = {"n_members": len(models_used), "models_used": models_used}
        if return_individual:
            info["individual_series"] = dict(zip(models_used, aligned))
        logger.info("    CMIP6 MMM time series: %d models, %d timesteps",
                     len(models_used), len(mmm_ts.time))
        return mmm_ts, info

    def _benchmark_timeseries(
        self,
        var: str,
        period: tuple[str, str] | None = None,
        return_individual: bool | None = None,
    ) -> list[dict]:
        """Per-benchmark global-mean MMM time series (CMIP6, HighResMIP, …).

        Loops the configured benchmark loaders and computes each one's
        ensemble-mean global-mean series via
        :meth:`_cmip6_global_mean_timeseries`.

        Returns
        -------
        list[dict]
            One entry per benchmark with data: ``label``, ``color``,
            ``ts`` (MMM series), ``info`` and ``individual`` (member series
            when ``return_individual``).  Empty if no benchmark has data.
        """
        from feather.plot.styles import benchmark_color

        if return_individual is None:
            return_individual = getattr(self, "cmip6_individual", False)
        if period is None:
            period = getattr(self, "period", None)

        out: list[dict] = []
        for i, bench in enumerate(self.benchmarks):
            ts, info = self._cmip6_global_mean_timeseries(
                var, period=period,
                return_individual=return_individual, loader=bench,
            )
            if ts is None:
                continue
            out.append({
                "label": getattr(bench, "label", "CMIP6 MMM"),
                "color": getattr(bench, "color", None) or benchmark_color(i),
                "ts": ts,
                "info": info,
                "individual": (
                    dict(info["individual_series"])
                    if return_individual and "individual_series" in info
                    else {}
                ),
            })
        return out

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
