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

    def run(self) -> list[tuple[Path, Path]]:
        """Execute the full diagnostic: compute → plot → save.

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
