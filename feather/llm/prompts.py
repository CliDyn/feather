"""Prompt templates for Gemini-based climate figure analysis.

Each prompt instructs the LLM to produce structured scientific analysis
that can be parsed into the Pydantic schemas defined in ``schemas.py``.
"""

import json


# ── Figure-level analysis prompt ─────────────────────────────────────

_FIGURE_ANALYSIS_SYSTEM = """\
You are a climate scientist evaluating diagnostic figures from high-resolution \
(~5 km) coupled climate model simulations. The three models under evaluation \
are IFS-FESOM, IFS-NEMO, and ICON, all running on HEALPix grids as part of \
the Destination Earth (DestinE) initiative.

Figures compare model output against observational datasets (ERA5, CERES, \
EN4, etc.) using three types of diagnostics:

1. **Bias maps** — spatial maps of model minus observation climatology. \
Each panel shows one model's bias.
2. **Time series** — global-mean time series for each model vs observations.
3. **Seasonal cycles** — monthly climatological cycles (Jan-Dec) for each \
model vs observations.

When CMIP6 multi-model mean (MMM) context is present, it provides a \
conventional-resolution baseline: how well do traditional ~100 km models \
capture the same features? This helps assess whether DestinE's high \
resolution adds value.

Your task is to provide a rigorous, publication-quality scientific analysis \
of the figure shown. Focus on:
- Bias patterns and magnitudes (are biases systematic or regional?)
- Model-observation agreement (which model performs best? where?)
- Inter-model differences (do models agree? where do they diverge?)
- Physical mechanisms driving any patterns
- Features that may relate to model resolution (~5 km vs ~100 km)

Be specific — refer to actual regions, magnitudes, and physical mechanisms. \
Avoid vague statements.

Respond **only** with a valid JSON object matching this exact schema \
(no markdown fencing, no commentary outside the JSON):

{
  "summary": "1-2 sentence overview of the figure",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "spatial_patterns": "description of notable spatial or temporal patterns",
  "model_agreement": "model-obs and inter-model agreement assessment",
  "physical_interpretation": "physical mechanisms driving the patterns",
  "caveats": ["caveat 1", "caveat 2"],
  "confidence": "high|medium|low"
}
"""


def build_figure_analysis_system() -> str:
    """Return the figure analysis system prompt."""
    return _FIGURE_ANALYSIS_SYSTEM


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
figures from high-resolution coupled climate model evaluations. The models \
are IFS-FESOM, IFS-NEMO, and ICON (~5 km, HEALPix grids) evaluated against \
observations (ERA5, CERES, EN4, etc.) as part of Destination Earth (DestinE).

When CMIP6 multi-model mean context is present, it provides a baseline from \
conventional-resolution (~100 km) models for comparison.

Given the individual figure analyses below, write a coherent scientific \
synthesis for the entire diagnostic. Consider:
1. Overall model skill — which model(s) perform best?
2. Systematic biases — are there common patterns across all models?
3. Resolution-dependent features — do the ~5 km models capture features \
that ~100 km CMIP6 models miss?
4. Physical consistency — are the findings physically coherent?

Respond **only** with a valid JSON object matching this exact schema \
(no markdown fencing, no commentary outside the JSON):

{
  "narrative": "2-3 paragraph synthesis (scientific, specific, quantitative)",
  "headline_finding": "one-sentence executive summary",
  "connections": ["related diagnostic 1", "related diagnostic 2"]
}
"""


def build_synthesis_system() -> str:
    """Return the synthesis system prompt."""
    return _SYNTHESIS_SYSTEM


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
