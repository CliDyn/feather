"""CMIP6 data loader for multi-model mean comparisons.

Loads CMIP6 historical data from per-variable zarr files, regrids to a
common regular grid using nereus RegridInterpolator, and computes
multi-model mean (MMM) fields.

Design principles:
- Per-variable zarr loading (no intake at load time) — avoids staggered-grid conflicts
- Models missing variables are silently skipped (load_var returns None)
- Uses nereus RegridInterpolator (KDTree NN + influence radius) for MMM regridding
- Calendar normalization inside load_var before time slicing
- Two ensemble modes: "one_per_model" and "all_members"
"""

import logging
import os
from pathlib import Path

import nereus as nr
import numpy as np
import pandas as pd
import xarray as xr

from feather.data.variables import VARIABLE_REGISTRY, get_var

logger = logging.getLogger(__name__)


class CMIP6Loader:
    """Load CMIP6 data and compute multi-model mean on a common grid."""

    def __init__(self, config):
        """Initialize from a FeatherConfig.

        Parameters
        ----------
        config : FeatherConfig
            Must have a ``cmip6`` dict with ``catalog_path``, ``models``, etc.
        """
        self._cfg = config.cmip6
        self._zarr_dir = self._resolve_zarr_dir()
        self._area_cache: dict[str, xr.DataArray] = {}
        self._interp_cache: dict[str, nr.RegridInterpolator] = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def models(self) -> dict:
        """Configured CMIP6 models dict."""
        return self._cfg.get("models", {})

    @property
    def zarr_dir(self) -> str:
        """Directory containing per-variable zarr stores."""
        return self._zarr_dir

    # ── Public API ────────────────────────────────────────────────────

    def load_var(
        self,
        cmip6_var: str,
        model: str,
        *,
        variant: str | None = None,
        table: str | None = None,
        period: tuple[str, str] | None = None,
        season: str | None = None,
        time_mean: bool = True,
    ) -> xr.DataArray | None:
        """Load a single CMIP6 variable for one model+variant.

        Parameters
        ----------
        cmip6_var : str
            CMIP6 variable name (e.g. "tas").
        model : str
            CMIP6 model name (e.g. "MIROC6").
        variant : str, optional
            Variant label. If None, uses first configured variant.
        table : str, optional
            CMIP6 table (e.g. "Amon"). If None, inferred from registry.
        period : tuple of str, optional
            (start, end) date strings for time slicing (e.g. ("1990-01", "2010-12")).
        season : str, optional
            Season filter ("DJF", "MAM", "JJA", "SON").
        time_mean : bool
            If True (default), return time-mean 2D field. If False, return
            the full time series after period/season filtering.

        Returns
        -------
        xr.DataArray or None
            Time-mean 2D field (or full time series if ``time_mean=False``),
            or None if data not found.
        """
        if variant is None:
            model_cfg = self.models.get(model, {})
            variants = self._get_variants(model_cfg)
            if not variants:
                logger.warning("No variants configured for model %s", model)
                return None
            variant = variants[0]

        if table is None:
            table = self._infer_table(cmip6_var)

        zarr_path = self._zarr_path(model, variant, table, cmip6_var)
        if not os.path.exists(zarr_path):
            logger.debug("Zarr not found: %s", zarr_path)
            return None

        logger.info("Loading %s for %s/%s from zarr", cmip6_var, model, variant)
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception as e:
            logger.warning("Failed to open zarr %s: %s", zarr_path, e)
            return None

        if cmip6_var not in ds.data_vars:
            logger.warning("Variable %s not in %s", cmip6_var, zarr_path)
            return None

        da = ds[cmip6_var]

        # Normalize time coordinate
        if "time" in da.dims:
            da = self._normalize_time(da)
            if da is None:
                return None

            # Period filtering
            if period is not None:
                start, end = period
                da = da.sel(time=slice(start, end))

            # Season filtering
            if season is not None:
                da = da.sel(time=da["time.season"] == season)

            if da.sizes.get("time", 0) == 0:
                logger.debug("No timesteps after filtering for %s/%s", model, cmip6_var)
                return None

            # Normalize siconc
            if cmip6_var == "siconc":
                da = self._normalise_siconc(da)

            # Time mean (skip if caller wants the full time series)
            if time_mean:
                da = da.mean("time")
        else:
            # No time dimension — static field
            if cmip6_var == "siconc":
                da = self._normalise_siconc(da)

        return da.compute()

    def load_var_for_model_var(
        self,
        model_var: str,
        model: str,
        **kwargs,
    ) -> xr.DataArray | None:
        """Load CMIP6 data mapped from a feather model variable name.

        Maps e.g. "tas" (or legacy "avg_2t") -> CMIP6 "tas" / "Amon"
        via VARIABLE_REGISTRY.

        Parameters
        ----------
        model_var : str
            Feather variable name (CMOR or DestinE, e.g. "tas").
        model : str
            CMIP6 model name.
        **kwargs
            Passed to :meth:`load_var` (variant, period, season).

        Returns
        -------
        xr.DataArray or None
        """
        try:
            vinfo = get_var(model_var)
        except KeyError:
            return None
        if not vinfo.cmip6_variable:
            return None
        return self.load_var(
            vinfo.cmip6_variable,
            model,
            table=vinfo.cmip6_table or None,
            **kwargs,
        )

    def load_multi_model_mean(
        self,
        cmip6_var: str,
        *,
        table: str | None = None,
        period: tuple[str, str] | None = None,
        season: str | None = None,
        ensemble_mode: str | None = None,
    ) -> tuple[xr.DataArray | None, dict]:
        """Compute multi-model mean on a common regular grid.

        Each member is loaded on its native grid, regridded to a common
        lat/lon grid via ``nereus.regrid()`` (KDTree NN with influence
        radius masking), then averaged.

        Parameters
        ----------
        cmip6_var : str
            CMIP6 variable name.
        table : str, optional
            CMIP6 table. If None, inferred.
        period : tuple of str, optional
            (start, end) for time slicing.
        season : str, optional
            Season filter.
        ensemble_mode : str, optional
            "one_per_model" or "all_members". Overrides config default.

        Returns
        -------
        (mmm, info) : tuple
            mmm: DataArray on common grid, or None if no data.
            info: dict with n_members, models_used, models_skipped.
        """
        if table is None:
            table = self._infer_table(cmip6_var)

        member_pairs = self._get_member_pairs(ensemble_mode)
        logger.info(
            "Computing CMIP6 MMM for %s (%s) — %d members",
            cmip6_var, table, len(member_pairs),
        )

        resolution = self._cfg.get("regrid_resolution", 1.0)
        influence_radius = self._cfg.get("influence_radius", 80_000.0)

        regridded_fields = []
        models_used = []
        models_skipped = []

        for model, variant in member_pairs:
            member_label = f"{model}/{variant}"

            da = self.load_var(
                cmip6_var, model,
                variant=variant, table=table,
                period=period, season=season,
            )
            if da is None:
                models_skipped.append(member_label)
                continue

            # Find lat/lon
            try:
                lat, lon = self._find_lat_lon(da)
            except ValueError as e:
                logger.warning("Cannot find lat/lon for %s: %s", member_label, e)
                models_skipped.append(member_label)
                continue

            # For regular grids (1D lat + 1D lon), meshgrid to scattered
            if lat.ndim == 1 and lon.ndim == 1 and len(lat) != len(lon):
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                lon_flat = lon_2d.ravel()
                lat_flat = lat_2d.ravel()
            else:
                lon_flat = lon.ravel()
                lat_flat = lat.ravel()

            # Cache interpolator per model+grid (persists across calls)
            cache_key = f"{model}_{table}_{resolution}_{len(lon_flat)}"
            if cache_key not in self._interp_cache:
                logger.info("  Regridding %s (building interpolator)", member_label)
                regridded, interpolator = nr.regrid(
                    da.values.ravel(),
                    lon=lon_flat, lat=lat_flat,
                    resolution=resolution,
                    influence_radius=influence_radius,
                    lon_bounds=(0.0, 360.0),
                    as_xarray=True,
                )
                self._interp_cache[cache_key] = interpolator
            else:
                logger.info("  Regridding %s (cached interpolator)", member_label)
                interpolator = self._interp_cache[cache_key]
                regridded_np = interpolator(da.values.ravel())
                regridded = xr.DataArray(
                    regridded_np, dims=("lat", "lon"),
                    coords={
                        "lat": interpolator.target_lat[:, 0],
                        "lon": interpolator.target_lon[0, :],
                    },
                )
            regridded_fields.append(regridded)
            models_used.append(member_label)

        info = {
            "n_members": len(models_used),
            "models_used": models_used,
            "models_skipped": models_skipped,
        }

        if not regridded_fields:
            logger.info("  No CMIP6 data available for %s", cmip6_var)
            return None, info

        stacked = xr.concat(regridded_fields, dim="member")
        mmm = stacked.mean("member")
        logger.info(
            "  CMIP6 MMM computed: %d members used, %d skipped",
            len(models_used), len(models_skipped),
        )

        return mmm, info

    def load_mmm_for_model_var(
        self,
        model_var: str,
        **kwargs,
    ) -> tuple[xr.DataArray | None, dict]:
        """Convenience: load MMM mapped from a feather variable name.

        Parameters
        ----------
        model_var : str
            Feather variable name (CMOR or DestinE, e.g. "tas").
        **kwargs
            Passed to :meth:`load_multi_model_mean`.

        Returns
        -------
        (mmm, info) : tuple
        """
        try:
            vinfo = get_var(model_var)
        except KeyError:
            return None, {"n_members": 0, "models_used": [], "models_skipped": []}
        if not vinfo.cmip6_variable:
            return None, {"n_members": 0, "models_used": [], "models_skipped": []}
        return self.load_multi_model_mean(
            vinfo.cmip6_variable,
            table=vinfo.cmip6_table or None,
            **kwargs,
        )

    def load_area(
        self,
        model: str,
        variant: str | None = None,
        table: str = "Amon",
    ) -> xr.DataArray | None:
        """Load areacella / areacello weights for a model.

        Parameters
        ----------
        model : str
            CMIP6 model name.
        variant : str, optional
            Variant label. If None, uses first configured variant.
        table : str
            "Amon" (atmosphere) → areacella, "Omon"/"SImon" (ocean) → areacello.

        Returns
        -------
        xr.DataArray or None
        """
        if variant is None:
            model_cfg = self.models.get(model, {})
            variants = self._get_variants(model_cfg)
            if not variants:
                return None
            variant = variants[0]

        cache_key = f"{model}_{variant}_{table}"
        if cache_key in self._area_cache:
            return self._area_cache[cache_key]

        zarr_path = self._area_zarr_path(model, variant, table)
        if not os.path.exists(zarr_path):
            logger.debug("Area zarr not found: %s", zarr_path)
            self._area_cache[cache_key] = None
            return None

        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception as e:
            logger.warning("Failed to open area zarr %s: %s", zarr_path, e)
            self._area_cache[cache_key] = None
            return None

        # Find the area variable
        for var_name in ("areacella", "areacello"):
            if var_name in ds.data_vars:
                area = ds[var_name].compute()
                self._area_cache[cache_key] = area
                return area

        self._area_cache[cache_key] = None
        return None

    def get_member_pairs(
        self, ensemble_mode: str | None = None,
    ) -> list[tuple[str, str]]:
        """Public API: list of (model, variant) tuples.

        Parameters
        ----------
        ensemble_mode : str, optional
            "one_per_model" or "all_members". If None, uses config default.
        """
        return self._get_member_pairs(ensemble_mode)

    def available_models(
        self,
        cmip6_var: str,
        table: str | None = None,
    ) -> list[str]:
        """List models that have zarr data for a given variable.

        Parameters
        ----------
        cmip6_var : str
            CMIP6 variable name.
        table : str, optional
            CMIP6 table. If None, inferred.

        Returns
        -------
        list of str
            Model names with data on disk.
        """
        if table is None:
            table = self._infer_table(cmip6_var)

        result = []
        for model in self.models:
            model_cfg = self.models[model]
            variants = self._get_variants(model_cfg)
            if not variants:
                continue
            zarr_path = self._zarr_path(model, variants[0], table, cmip6_var)
            if os.path.exists(zarr_path):
                result.append(model)
        return result

    def available_members(
        self,
        cmip6_var: str,
        table: str | None = None,
    ) -> list[tuple[str, str]]:
        """List all (model, variant) pairs with data for a variable.

        Parameters
        ----------
        cmip6_var : str
            CMIP6 variable name.
        table : str, optional
            CMIP6 table. If None, inferred.

        Returns
        -------
        list of (model, variant) tuples
        """
        if table is None:
            table = self._infer_table(cmip6_var)

        result = []
        for model in self.models:
            model_cfg = self.models[model]
            for variant in self._get_variants(model_cfg):
                zarr_path = self._zarr_path(model, variant, table, cmip6_var)
                if os.path.exists(zarr_path):
                    result.append((model, variant))
        return result

    def available_models_for_model_var(self, model_var: str) -> list[str]:
        """Convenience: available models mapped from a feather variable name."""
        try:
            vinfo = get_var(model_var)
        except KeyError:
            return []
        if not vinfo.cmip6_variable:
            return []
        return self.available_models(
            vinfo.cmip6_variable,
            table=vinfo.cmip6_table or None,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _resolve_zarr_dir(self) -> str:
        """Derive zarr directory from catalog_path."""
        catalog_path = self._cfg.get("catalog_path", "")
        if not catalog_path:
            return ""

        # Try to extract from catalog YAML
        try:
            import yaml

            with open(catalog_path) as f:
                cat_data = yaml.safe_load(f)
            for source in cat_data.get("sources", {}).values():
                urlpath = source.get("args", {}).get("urlpath", "")
                if isinstance(urlpath, list):
                    urlpath = urlpath[0] if urlpath else ""
                if urlpath and urlpath.endswith(".zarr"):
                    return str(Path(urlpath).parent)
        except Exception:
            pass

        # Fallback: sibling "zarr" directory
        return str(Path(catalog_path).parent / "zarr")

    def _zarr_path(
        self, model: str, variant: str, table: str, var: str,
    ) -> str:
        """Build path to a per-variable zarr store.

        Format: {zarr_dir}/{Model}_historical_{variant}_{table}_{var}.zarr
        """
        return f"{self._zarr_dir}/{model}_historical_{variant}_{table}_{var}.zarr"

    def _area_zarr_path(
        self, model: str, variant: str, table: str,
    ) -> str:
        """Build path to an area-weight zarr store.

        Atmosphere tables → areacella (fx), ocean tables → areacello (Ofx).
        """
        if table in ("Omon", "SImon", "Ofx"):
            return f"{self._zarr_dir}/{model}_historical_{variant}_Ofx_areacello.zarr"
        return f"{self._zarr_dir}/{model}_historical_{variant}_fx_areacella.zarr"

    @staticmethod
    def _get_variants(model_cfg: dict) -> list[str]:
        """Get variant list from a model config entry (backward-compatible).

        Supports both ``variants: [list]`` and legacy ``variant: str`` format.
        """
        if "variants" in model_cfg:
            return list(model_cfg["variants"])
        if "variant" in model_cfg:
            return [model_cfg["variant"]]
        return []

    def _get_member_pairs(
        self, ensemble_mode: str | None = None,
    ) -> list[tuple[str, str]]:
        """Build flat list of (model, variant) pairs based on ensemble mode.

        Parameters
        ----------
        ensemble_mode : str, optional
            "one_per_model" or "all_members". If None, uses config default.
        """
        mode = ensemble_mode or self._cfg.get("ensemble_mode", "one_per_model")
        pairs = []
        for model, model_cfg in self.models.items():
            variants = self._get_variants(model_cfg)
            if mode == "one_per_model":
                if variants:
                    pairs.append((model, variants[0]))
            else:  # all_members
                for v in variants:
                    pairs.append((model, v))
        return pairs

    @staticmethod
    def _normalize_time(da: xr.DataArray) -> xr.DataArray | None:
        """Normalize time coordinate to first-of-month pandas timestamps.

        Different CMIP6 models use different calendars (360_day, noleap,
        standard) with different mid-month day conventions. This converts
        all to a common representation.
        """
        if "time" not in da.dims:
            return da

        try:
            times = da.time.values
            if len(times) == 0:
                return None

            t0 = times[0]
            if hasattr(t0, "year") and not isinstance(t0, (np.datetime64,)):
                # cftime object — extract year/month manually
                new_times = pd.to_datetime([
                    f"{t.year:04d}-{t.month:02d}-01" for t in times
                ])
            else:
                # numpy datetime64 — convert via pandas
                new_times = (
                    pd.to_datetime(times).to_period("M").to_timestamp()
                )

            return da.assign_coords(time=new_times)
        except Exception as e:
            logger.warning("Failed to normalize time coordinate: %s", e)
            return None

    @staticmethod
    def _find_lat_lon(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        """Find latitude and longitude arrays from a DataArray.

        Tries common coordinate naming conventions:
        lat/lon, latitude/longitude, nav_lat/nav_lon.
        """
        # Check coords first, then dims
        for lat_name, lon_name in [
            ("lat", "lon"),
            ("latitude", "longitude"),
            ("nav_lat", "nav_lon"),
        ]:
            if lat_name in da.coords and lon_name in da.coords:
                return (
                    np.asarray(da.coords[lat_name]),
                    np.asarray(da.coords[lon_name]),
                )

        raise ValueError(
            f"Cannot find lat/lon coordinates in {list(da.coords)}."
        )

    @staticmethod
    def _normalise_siconc(da: xr.DataArray) -> xr.DataArray:
        """Normalise sea-ice concentration to 0-1 fraction if stored as %."""
        max_val = float(da.max())
        if max_val > 10.0:
            return da / 100.0
        return da

    @staticmethod
    def _infer_table(cmip6_var: str) -> str:
        """Infer the CMIP6 table from the variable name.

        Checks VARIABLE_REGISTRY first, then uses heuristics.
        """
        # Check registry
        for vinfo in VARIABLE_REGISTRY.values():
            if vinfo.cmip6_variable == cmip6_var and vinfo.cmip6_table:
                return vinfo.cmip6_table

        # Heuristics
        ocean_vars = {"tos", "sos", "zos", "thetao", "so", "mlotst", "msftmz"}
        seaice_vars = {"siconc", "sithick", "sivol", "siarea"}
        if cmip6_var in ocean_vars:
            return "Omon"
        if cmip6_var in seaice_vars:
            return "SImon"
        return "Amon"
