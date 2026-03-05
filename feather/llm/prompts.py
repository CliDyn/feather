"""Prompt templates for Gemini-based climate figure analysis.

Each prompt instructs the LLM to produce structured scientific analysis
that can be parsed into the Pydantic schemas defined in ``schemas.py``.
"""

import json


# ── Figure-level analysis prompt ─────────────────────────────────────

_FIGURE_ANALYSIS_SYSTEM = """\
You are a climate scientist evaluating diagnostic figures from \
{resolution} coupled climate model simulations. The models under evaluation \
are {model_list}{project_context}.

Figures compare model output against observational datasets (ERA5, CERES, \
EN4, etc.) using these diagnostic figure types:

1. **Bias maps** — spatial maps of model minus observation climatology \
(from global_biases and radiation_budget diagnostics). Each panel shows \
one model's bias.
2. **Time series** — global-mean time series for each model vs observations.
3. **Seasonal cycles** — monthly climatological cycles (Jan-Dec) for each \
model vs observations.
4. **Radiation budget bars** — two-panel figure: left panel is a grouped \
bar chart comparing global-mean TOA and surface radiation components \
(SW, LW, Net, CRE, atmospheric absorption) across models vs CERES EBAF \
observations; right panel zooms in on TOA Net radiation (~1 W/m²), which \
is too small to distinguish in the full budget view, with per-source \
value annotations.
5. **Gregory plot** — scatter of global-mean 2 m temperature vs net TOA \
radiation with regression lines; used to diagnose radiative feedbacks \
and equilibrium climate sensitivity.
6. **Radiation imbalance time series** — net TOA radiation over time \
(monthly background + annual-mean foreground) showing Earth's energy \
imbalance evolution.
7. **Sea ice time series** — NH and SH sea ice area, extent, or volume \
over time (monthly + annual mean) compared against OSI-SAF (conc/extent) \
and PIOMAS/GIOMAS (volume) observations.
8. **Sea ice seasonal cycle** — 12-month climatological cycle of sea ice \
metrics for each hemisphere, comparing models vs observations.
9. **Sea ice March & September trends** — annual values for March and \
September (NH annual max / min, SH annual max / min) showing \
long-term trends in sea ice metrics.
10. **Sea ice spatial maps** — polar stereographic maps of sea ice \
concentration or thickness for extreme months, comparing models and \
observations.
11. **SST bias maps** — spatial maps of model SST minus ESA-CCI \
satellite SST climatology. Look for regional patterns: coastal biases \
from influence radius effects, tropical cold tongue biases, Gulf Stream / \
Kuroshio separation errors, warm biases in upwelling regions.
12. **SST time series** — global-mean ocean SST over time, models vs \
ESA-CCI satellite observations. Monthly as semi-transparent background, \
annual means as thick foreground.
13. **SST seasonal cycle** — 12-month climatological cycle of global-mean \
SST for models vs ESA-CCI. Assess amplitude and phase differences.
14. **SST zonal mean** — latitude profile of SST for models vs ESA-CCI. \
Check warm pool position, mid-latitude gradients, polar SST.
15. **Hovmoller (time-depth) diagrams** — show evolution of temperature \
or salinity anomalies as a function of depth and time. Two anomaly \
types: relative to first timestep (intrinsic drift) and relative to \
EN4 first-year reference profile (model-obs departure). Look for signal \
propagation depth, surface vs deep ocean trends, and comparison to \
EN4 observational reference. Sqrt-scaled depth axis emphasizes upper ocean.
16. **Depth-layer time series** — volume-weighted mean temperature or \
salinity in three depth ranges (0-700m, 700-2000m, 2000m-bottom) \
comparing models against EN4 v4.2.2. Monthly as semi-transparent \
background, annual means as thick foreground. Assess warming/freshening \
rates at different depths.
17. **Trend maps** — Combined multi-panel figure showing linear trends \
(units/decade) over the analysis period. First panel shows observation \
trends, subsequent panels show model-obs trend differences. Look for: \
spatial patterns of warming/cooling, precipitation changes, trend \
magnitude comparison across models, regions of agreement/disagreement.

When CMIP6 multi-model mean (MMM) context is present, it provides a \
conventional-resolution baseline: how well do traditional ~100 km models \
capture the same features? This helps assess whether the evaluated models' \
resolution adds value.

Your task is to provide a rigorous, publication-quality scientific analysis \
of the figure shown. Focus on:
- Bias patterns and magnitudes (are biases systematic or regional?)
- Model-observation agreement (which model performs best? where?)
- Inter-model differences (do models agree? where do they diverge?)
- Physical mechanisms driving any patterns
- Features that may relate to model resolution
- For radiation figures: compare against CERES EBAF (the satellite gold \
standard), note sign conventions (positive = energy into the system), \
assess cloud radiative effects, and for Gregory plots interpret the \
regression slope as a feedback parameter (W/m²/K)

Be specific — refer to actual regions, magnitudes, and physical mechanisms. \
Avoid vague statements.

Respond **only** with a valid JSON object matching this exact schema \
(no markdown fencing, no commentary outside the JSON):

{{
  "summary": "1-2 sentence overview of the figure",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "spatial_patterns": "description of notable spatial or temporal patterns",
  "model_agreement": "model-obs and inter-model agreement assessment",
  "physical_interpretation": "physical mechanisms driving the patterns",
  "caveats": ["caveat 1", "caveat 2"],
  "confidence": "high|medium|low"
}}
"""


def _format_model_list(models: list[str]) -> str:
    """Format a list of model names as a natural-language enumeration.

    >>> _format_model_list(["A", "B", "C"])
    'A, B, and C'
    """
    if len(models) == 1:
        return models[0]
    if len(models) == 2:
        return f"{models[0]} and {models[1]}"
    return ", ".join(models[:-1]) + f", and {models[-1]}"


def build_figure_analysis_system(
    *,
    models: list[str] | None = None,
    project_name: str | None = None,
    resolution: str | None = None,
) -> str:
    """Return the figure analysis system prompt.

    Parameters
    ----------
    models : list of str or None
        Model names to mention. Defaults to DestinE models.
    project_name : str or None
        Project/initiative name (e.g. ``"EERIE HighResMIP"``).
        Defaults to ``"Destination Earth (DestinE)"``.
    resolution : str or None
        Resolution description (e.g. ``"high-resolution"``).
        Defaults to ``"high-resolution (~5 km)"``.
    """
    if models is None:
        models = ["IFS-FESOM", "IFS-NEMO", "ICON"]
    if resolution is None:
        resolution = "high-resolution (~5 km)"

    model_list = _format_model_list(models)
    if project_name:
        project_context = f", as part of the {project_name} project"
    else:
        project_context = ", as part of the Destination Earth (DestinE) initiative"

    return _FIGURE_ANALYSIS_SYSTEM.format(
        model_list=model_list,
        project_context=project_context,
        resolution=resolution,
    )


def build_figure_prompt(metadata: dict) -> str:
    """Build the user-facing prompt for a single figure analysis.

    Parameters
    ----------
    metadata : dict
        JSON sidecar contents (diagnostic_name, title, variables_used,
        models, units, period, plot_type, description, obs info, etc.).

    Returns
    -------
    str
        The complete user prompt to send alongside the figure image.
    """
    parts = [
        f"Diagnostic: {metadata.get('diagnostic_name', 'unknown')}",
        f"Title: {metadata.get('title', 'N/A')}",
    ]

    variables = metadata.get("variables_used", [])
    if variables:
        parts.append(f"Variables: {', '.join(variables)}")

    models = metadata.get("models", [])
    if models:
        parts.append(f"Models shown: {', '.join(models)}")

    units = metadata.get("units")
    if units:
        parts.append(f"Units: {units}")

    period = metadata.get("period")
    if period:
        parts.append(f"Period: {period[0]}–{period[1]}")

    plot_type = metadata.get("plot_type")
    if plot_type:
        parts.append(f"Plot type: {plot_type}")

    obs_dataset = metadata.get("obs_dataset")
    if obs_dataset:
        obs_var = metadata.get("obs_variable", "")
        obs_str = obs_dataset
        if obs_var:
            obs_str += f" ({obs_var})"
        parts.append(f"Observation: {obs_str}")

    if metadata.get("description"):
        parts.append(f"Description: {metadata['description']}")

    if metadata.get("computation_notes"):
        parts.append(f"Computation: {metadata['computation_notes']}")

    # Summary statistics
    summary_stats = metadata.get("summary_statistics", {})
    if summary_stats:
        stats_str = json.dumps(summary_stats, indent=None, default=str)
        parts.append(f"Summary statistics: {stats_str}")

    # CMIP6 context
    cmip6_info = metadata.get("cmip6_info")
    if cmip6_info:
        n_models = cmip6_info.get("n_models", {})
        models_used = cmip6_info.get("models_used", {})
        if n_models:
            first_var = next(iter(n_models))
            n = n_models[first_var]
            mlist = models_used.get(first_var, [])
            parts.append(
                f"CMIP6 context: Multi-model mean from {n} CMIP6 models "
                f"({', '.join(mlist)}). Variable coverage may differ per model."
            )

    return (
        "Analyse the attached climate diagnostic figure.\n\n"
        "Figure metadata:\n"
        + "\n".join(f"  - {p}" for p in parts)
        + "\n\nProvide your analysis as a JSON object matching the schema "
        "described in the system prompt."
    )


# ── Diagnostic synthesis prompt ──────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are a climate scientist writing a synthesis of multiple diagnostic \
figures from {resolution} coupled climate model evaluations. The models \
are {model_list} evaluated against \
observations (ERA5, CERES, EN4, etc.){project_context}.

When CMIP6 multi-model mean context is present, it provides a baseline from \
conventional-resolution (~100 km) models for comparison.

Given the individual figure analyses below, write a coherent scientific \
synthesis for the entire diagnostic. Consider:
1. Overall model skill — which model(s) perform best?
2. Systematic biases — are there common patterns across all models?
3. Resolution-dependent features — do the evaluated models capture features \
that ~100 km CMIP6 models miss?
4. Physical consistency — are the findings physically coherent?
5. Radiation budget closure — do models conserve energy at TOA/surface? \
Are cloud radiative effects realistic?

Respond **only** with a valid JSON object matching this exact schema \
(no markdown fencing, no commentary outside the JSON):

{{
  "narrative": "2-3 paragraph synthesis (scientific, specific, quantitative)",
  "headline_finding": "one-sentence executive summary",
  "connections": ["related diagnostic 1", "related diagnostic 2"]
}}
"""


def build_synthesis_system(
    *,
    models: list[str] | None = None,
    project_name: str | None = None,
    resolution: str | None = None,
) -> str:
    """Return the synthesis system prompt.

    Parameters
    ----------
    models : list of str or None
        Model names. Defaults to DestinE models.
    project_name : str or None
        Project/initiative name. Defaults to DestinE.
    resolution : str or None
        Resolution description. Defaults to ``"high-resolution (~5 km)"``.
    """
    if models is None:
        models = ["IFS-FESOM", "IFS-NEMO", "ICON"]
    if resolution is None:
        resolution = "high-resolution (~5 km)"

    model_list = _format_model_list(models)
    if project_name:
        project_context = f" as part of the {project_name} project"
    else:
        project_context = " as part of Destination Earth (DestinE)"

    return _SYNTHESIS_SYSTEM.format(
        model_list=model_list,
        project_context=project_context,
        resolution=resolution,
    )


def build_synthesis_prompt(
    diagnostic_name: str,
    figure_analyses: list[dict],
) -> str:
    """Build the user prompt for per-diagnostic synthesis.

    Parameters
    ----------
    diagnostic_name : str
        Machine name of the diagnostic (e.g. ``"global_biases"``).
    figure_analyses : list of dict
        List of individual ``FigureAnalysis`` dicts for this diagnostic.

    Returns
    -------
    str
        The complete user prompt.
    """
    analyses_text = json.dumps(figure_analyses, indent=2)
    return (
        f"Diagnostic: {diagnostic_name}\n\n"
        f"Individual figure analyses:\n{analyses_text}\n\n"
        "Write a coherent scientific synthesis covering all figures above. "
        "Identify the overarching story, quantitative highlights, "
        "and connections to other climate diagnostics."
    )
