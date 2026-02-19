"""Figure metadata sidecar system.

Every diagnostic figure gets two files:
- ``{name}.png`` — the plot image
- ``{name}.json`` — structured metadata sidecar

The metadata JSON provides all context an LLM needs to write a
scientific interpretation without seeing the diagnostic code.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from feather.data.variables import get_var

logger = logging.getLogger(__name__)


def save_figure_with_metadata(
    fig: plt.Figure,
    metadata: dict[str, Any],
    output_dir: str | Path,
    filename: str,
    *,
    dpi: int = 150,
    close: bool = True,
) -> tuple[Path, Path]:
    """Save figure as PNG + write JSON metadata sidecar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    metadata : dict
        Metadata to write as JSON sidecar.
    output_dir : str or Path
        Directory for output files (created if needed).
    filename : str
        Base filename **without** extension (e.g. ``"t2m_annual_bias_ifs-fesom"``).
    dpi : int
        Resolution for the saved PNG.
    close : bool
        Whether to close the figure after saving.

    Returns
    -------
    (png_path, json_path) : tuple of Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename}.png"
    json_path = output_dir / f"{filename}.json"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)

    # Add generation timestamp
    metadata_out = {**metadata, "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(json_path, "w") as f:
        json.dump(metadata_out, f, indent=2, default=str)

    logger.info("Saved %s + %s", png_path.name, json_path.name)
    return png_path, json_path


def build_metadata(
    diagnostic_name: str,
    title: str,
    *,
    figure_id: str,
    variables_used: list[str],
    models: list[str],
    description: str = "",
    computation_notes: str = "",
    period: tuple[str, str] | None = None,
    obs_dataset: str = "",
    obs_variable: str = "",
    plot_type: str = "",
    spatial_extent: str = "global",
    summary_statistics: dict[str, Any] | None = None,
    cmip6_info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard metadata dict for the JSON sidecar.

    Pulls unit info, domain, group, and colormap from
    :data:`~feather.data.variables.VARIABLE_REGISTRY` for the first
    variable in *variables_used*.

    Parameters
    ----------
    diagnostic_name : str
        Machine-readable diagnostic identifier (e.g. ``"global_biases"``).
    title : str
        Human-readable plot title.
    figure_id : str
        Unique figure identifier used as the filename stem.
    variables_used : list of str
        Model variable names used (e.g. ``["avg_2t"]``).
    models : list of str
        Model names included in the figure.
    description : str
        Free-text description providing context for LLM analysis.
    computation_notes : str
        How the plotted quantity was computed.
    period : tuple of (start, end), optional
        Analysis period (e.g. ``("1990", "2014")``).
    obs_dataset : str
        Observation dataset name (e.g. ``"ERA5"``).
    obs_variable : str
        Variable name in the observation dataset.
    plot_type : str
        Type of plot (e.g. ``"bias_map"``, ``"timeseries"``).
    spatial_extent : str
        Spatial coverage (e.g. ``"global"``, ``"NH"``, ``"tropics"``).
    summary_statistics : dict, optional
        Computed statistics (e.g. global_mean_bias, RMSE).
    cmip6_info : dict, optional
        CMIP6 comparison metadata.
    extra : dict, optional
        Additional metadata fields.

    Returns
    -------
    dict
        Metadata dictionary ready for JSON serialization.
    """
    # Pull info from variable registry
    units = ""
    domain = ""
    group = ""
    cmap = ""
    if variables_used:
        try:
            vinfo = get_var(variables_used[0])
            units = vinfo.units
            domain = vinfo.domain
            group = vinfo.group
            cmap = vinfo.cmap
            # Auto-fill obs info from registry if not provided
            if not obs_dataset:
                obs_dataset = vinfo.obs_dataset
            if not obs_variable:
                obs_variable = vinfo.obs_variable
        except KeyError:
            pass

    meta: dict[str, Any] = {
        "diagnostic_name": diagnostic_name,
        "title": title,
        "figure_id": figure_id,
        "variables_used": variables_used,
        "models": models,
        "obs_dataset": obs_dataset,
        "obs_variable": obs_variable,
        "units": units,
        "period": list(period) if period else None,
        "description": description,
        "computation_notes": computation_notes,
        "domain": domain,
        "group": group,
        "spatial_extent": spatial_extent,
        "plot_type": plot_type,
        "colormap": cmap,
        "cmip6_info": cmip6_info,
        "summary_statistics": summary_statistics or {},
    }

    if extra:
        meta.update(extra)

    return meta
