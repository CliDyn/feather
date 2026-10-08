"""Load CMIP6 GCM data from the ESGF CMIP6 DRS NetCDF tree.

Directory layout::

    {root}/{activity}/{institute}/{model}/{experiment}/{variant}/{table}/
        {var}/{grid}/{version}/{var}_{table}_{model}_{experiment}_..._{period}.nc

``activity`` is inferred from the experiment (``historical`` → ``CMIP``,
``ssp*`` → ``ScenarioMIP``).  The institute and grid label are auto-detected
unless given in the config.  Experiments may be stitched (e.g. historical +
ssp585); the latest version directory is selected.

Distinct from the existing zarr-based :class:`CMIP6Loader`, which only serves
the pre-staged historical MMM.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from feather import provenance
from feather.config import FeatherConfig, ModelConfig
from feather.data._cftime import cftime_decode_kwargs
from feather.data import pool_discovery as _pd

logger = logging.getLogger(__name__)
_CFTIME = cftime_decode_kwargs()

_DEFAULT_ROOT = "/work/ik1017/CMIP6/data/CMIP6"
# Bounds/aux variables (often object/cftime dtype) that break dask auto-chunking.
_BOUNDS_VARS = ["time_bnds", "lat_bnds", "lon_bnds", "bnds", "time_bounds"]


def _activity(experiment: str) -> str:
    return "ScenarioMIP" if experiment.startswith("ssp") else "CMIP"


def discover_daily_models(
    root: str | Path,
    *,
    experiment: str,
    table: str,
    variable: str,
    member: str = "r1i1p1f1",
    exclude: tuple[str, ...] = (),
    max_models: int | None = None,
    require_also: tuple[str, ...] = (),
) -> list[ModelConfig]:
    """Discover CMIP6 models publishing ``{table}/{variable}`` for *experiment*.

    Walks the DRS tree ``{root}/{activity}/{inst}/{model}/{experiment}/...``
    (login-node safe — only ``glob``/``iterdir``) and returns a
    :class:`~feather.config.ModelConfig` for every model that has at least one
    NetCDF file for the requested variable, one member each (preferring
    *member*).  No data is read.

    Parameters
    ----------
    root : str or Path
        CMIP6 DRS root, e.g. ``/work/ik1017/CMIP6/data/CMIP6``.
    experiment, table, variable : str
        e.g. ``"historical"``, ``"day"``, ``"tasmax"``.
    member : str
        Preferred ensemble member (default ``r1i1p1f1``).
    exclude : tuple of str
        Model names to skip (case-sensitive ``source_id``).
    max_models : int, optional
        Keep at most this many models (sorted by name) — for staged runs.
    require_also : tuple of str
        Further experiments (e.g. ``("ssp245",)``) the member must also
        publish ``{table}/{variable}`` for.  The member is then chosen among
        those that have the files in *every* experiment — *member* first,
        else the lowest-numbered — instead of taking *member* (or the lowest)
        and dropping the model when that one lacks them.  Without it the
        original selection is kept unchanged.
    """
    activity_root = Path(root) / _activity(experiment)
    excl = set(exclude)
    found: list[ModelConfig] = []
    for model, exp_dir in _pd.iter_model_dirs(activity_root, experiment):
        if model in excl:
            continue
        if require_also:
            mem = _member_with(root, exp_dir, model, member, table, variable,
                               require_also)
        else:
            mem = _pd.select_member(exp_dir, prefer=member)
        if mem is None:
            continue
        files = _pd.variable_files(exp_dir, mem, table, variable)
        if not files:
            continue
        institution = exp_dir.parent.parent.name
        var_dir = exp_dir / mem / table / variable
        grid = _pd.select_grid(var_dir) or ""
        found.append(ModelConfig(
            name=model,
            institution=institution,
            experiment=experiment,
            variant=mem,
            grids={"sfc": "latlon"},
            experiments=[experiment],
            grid_dir=grid,
        ))
        if max_models is not None and len(found) >= max_models:
            break
    return found


def experiment_dirs(root, model: str, experiment: str,
                    institution: str = "") -> list[Path]:
    """``{root}/{activity}/{inst}/{model}/{experiment}`` for every institution.

    The same model can be filed under different institutions per experiment
    (MPI-ESM1-2-HR: historical under MPI-M, ssp245 under DKRZ), so the
    historical institution cannot be assumed.  *institution* is tried first.
    """
    activity = Path(root) / _activity(experiment)
    found = sorted(p for p in activity.glob(f"*/{model}/{experiment}") if p.is_dir())
    found.sort(key=lambda p: p.parent.parent.name != institution)
    return found


def _member_with(root, exp_dir: Path, model: str, prefer: str, table: str,
                 variable: str, others: tuple[str, ...]) -> str | None:
    """First member (*prefer* first) with the variable in every experiment."""
    members = sorted((d.name for d in exp_dir.iterdir() if d.is_dir()),
                     key=_pd._member_sort_key)
    if prefer in members:
        members.remove(prefer)
        members.insert(0, prefer)
    other_dirs = [experiment_dirs(root, model, e) for e in others]
    for mem in members:
        if not _pd.variable_files(exp_dir, mem, table, variable):
            continue
        if all(any(_pd.variable_files(d, mem, table, variable) for d in dirs)
               for dirs in other_dirs):
            return mem
    return None


class CMIP6NCLoader:
    """Load CMIP6 monthly atmosphere data from the CMOR NetCDF tree."""

    def __init__(self, config: FeatherConfig):
        self._config = config
        self._root = Path(config.data_source.get("cmip6_root", _DEFAULT_ROOT))
        self._cache: dict[tuple, xr.DataArray] = {}
        # Files behind each cached array, and their provenance records.
        self._inputs: dict[tuple, list] = {}
        self._prov: dict[tuple, tuple] = {}
        self._opened: list = []

    @classmethod
    def for_models(
        cls,
        base_config: FeatherConfig,
        model_configs: list[ModelConfig],
        *,
        root: str | Path = _DEFAULT_ROOT,
    ) -> "CMIP6NCLoader":
        """Build a loader serving an explicit set of CMIP6 models.

        Used to attach an auto-discovered daily CMIP6 ensemble to a diagnostic
        without disturbing the main config.  A shallow copy of *base_config* is
        made with its ``model_configs`` and CMIP6 root replaced, so the
        original config (and the EERIE model list) is untouched.
        """
        cfg = copy.copy(base_config)
        cfg.model_configs = {mc.name: mc for mc in model_configs}
        cfg.data_source = {**getattr(base_config, "data_source", {}),
                           "type": "cmip6_nc", "cmip6_root": str(root)}
        loader = cls(cfg)
        loader._model_names = [mc.name for mc in model_configs]
        return loader

    @property
    def model_names(self) -> list[str]:
        """Model names this loader was built for (factory path only)."""
        return list(getattr(self, "_model_names", list(self._config.model_configs)))

    def load_var(
        self,
        model: str,
        variable: str,
        *,
        table: str | None = None,
        period: tuple[str, str] | None = None,
        time_mean: bool = False,
    ) -> xr.DataArray:
        cache_key = (model, variable)
        if cache_key in self._cache:
            da = self._cache[cache_key]
        else:
            self._opened = []
            da = self._open_member(model, variable, table or "Amon")
            self._cache[cache_key] = da
            self._inputs[cache_key] = self._opened

        provenance.record_read(
            self._prov, cache_key, period=period,
            role="model", backend=type(self).__name__, variable=variable,
            paths=self._inputs.get(cache_key, ()), data=da,
            **provenance.model_fields(self._config, model),
        )

        if period and "time" in da.dims:
            da = da.sel(time=slice(period[0], period[1]))
        if time_mean and "time" in da.dims:
            da = da.mean("time")
        return da

    def load_coords(self, model: str, variable: str):
        da = self.load_var(model, variable)
        return np.asarray(da["lon"].values), np.asarray(da["lat"].values)

    # ── Private ────────────────────────────────────────────────────────

    def _open_member(self, model: str, variable: str, table: str) -> xr.DataArray:
        mc = self._config.model_configs[model]
        experiments = list(mc.experiments) if mc.experiments else [mc.experiment]
        experiments = [e for e in experiments if e]
        if not experiments:
            raise ValueError(f"No experiment(s) configured for CMIP6 model {model!r}")

        segments: list[xr.DataArray] = []
        for exp in experiments:
            d = self._var_dir(mc, exp, variable, table)
            if d is None:
                continue
            files = sorted(d.glob("*.nc"))
            if not files:
                continue
            logger.info("CMIP6 %s/%s: opening %d file(s)", model, exp, len(files))
            self._opened.extend(files)
            ds = xr.open_mfdataset(
                files, chunks={}, combine="by_coords", data_vars="minimal", coords="minimal", compat="override",
                decode_timedelta=False, drop_variables=_BOUNDS_VARS, **_CFTIME,
            )
            segments.append(ds[variable])

        if not segments:
            raise FileNotFoundError(
                f"No CMIP6 data for {model!r}/{variable!r} (experiments={experiments})"
            )
        if len(segments) == 1:
            da = segments[0]
        else:
            da = xr.concat(segments, dim="time").sortby("time")
        da.name = variable
        return da

    def _var_dir(self, mc, experiment: str, variable: str, table: str) -> Path | None:
        """Resolve ``.../{var}/{grid}/{version}/`` (latest version)."""
        variant = mc.variant or "r1i1p1f1"
        activity = self._root / _activity(experiment)

        # Institute: configured first, then any other that files the model
        # (CMIP6 model names are unique, but one model can sit under different
        # institutes per experiment, e.g. MPI-ESM1-2-HR ssp245 under DKRZ).
        model_dir_name = mc.gcm or mc.name
        globbed = [p.parent.name for p in activity.glob(f"*/{model_dir_name}")]
        institutes = ([mc.institution] if mc.institution else []) + [
            i for i in globbed if i != mc.institution]

        for inst in institutes:
            var_base = (activity / inst / model_dir_name / experiment
                        / variant / table / variable)
            if not var_base.exists():
                continue
            # Grid label (gn/gr/gr1…): configured or first available.
            grids = [mc.grid_dir] if mc.grid_dir else [
                d.name for d in sorted(var_base.iterdir()) if d.is_dir()
            ]
            for grid in grids:
                grid_dir = var_base / grid
                if not grid_dir.exists():
                    continue
                versions = sorted(grid_dir.glob("v*"))
                if not versions:
                    versions = [grid_dir]
                for v in reversed(versions):
                    if list(v.glob("*.nc")):
                        return v
        return None
