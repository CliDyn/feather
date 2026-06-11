"""Prompt templates for Gemini-based climate figure analysis.

Each prompt instructs the LLM to produce structured scientific analysis
that can be parsed into the Pydantic schemas defined in ``schemas.py``.

Prompts adapt to different comparison types via ``comparison_type``:

- ``"multi_model"`` (default) — independent models compared against each
  other and CMIP6.
- ``"resolution_sensitivity"`` — same model family at different resolutions;
  focus on how skill scales with resolution.
- ``"single_model"`` — one model evaluated against observations only.
- ``"baseline_evaluation"`` — baseline simulations assessed as a reference
  for future component improvements (e.g. TerraDT).
"""

import json


# ── Shared figure-type descriptions ─────────────────────────────────

_FIGURE_TYPES = """\
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
18. **Climate variability STD maps** — Combined multi-panel figure showing \
the standard deviation of deseasonalised and detrended monthly data. All \
panels use the same colormap. Compare spatial patterns of variability \
across models and ERA5: mid-latitude storm tracks, tropical variability, \
polar amplification of variability.
19. **Climate variability STD difference maps** — Combined multi-panel \
figure showing ERA5 STD (first panel) and model-ERA5 STD differences \
(subsequent panels). Positive differences = model more variable than obs. \
Look for systematic over/under-estimation of variability by region.
20. **Precipitation bias maps** — multi-panel showing MSWEP v2.8 observations \
and model biases in precipitation rate. Look for biases in the ITCZ position, \
monsoon regions, storm tracks, and orographic precipitation. Note any \
systematic wet/dry biases.
21. **Relative precipitation bias maps** — percentage deviation from MSWEP. \
Highlights where models over/underestimate precipitation relative to observed \
amounts. Masked in arid regions where small absolute values cause extreme \
percentages.
22. **Precipitation intensity distribution** — area-weighted PDF of grid-cell \
climatological mean precipitation rates. Shows whether models produce too \
much drizzle (excess low-intensity cells) or underestimate heavy \
precipitation regions. Log-scale axes.
23. **Precipitation zonal mean** — latitude profile of annual-mean \
precipitation for models vs MSWEP v2.8. Shows ITCZ position (tropical \
maximum), subtropical dry zones, and extratropical storm track precipitation. \
The most informative single view for precipitation evaluation.
24. **Temperature bias maps (Berkeley Earth)** — spatial maps of model 2m \
temperature minus Berkeley Earth Land+Ocean climatology. Berkeley Earth is an \
independent station-based dataset, providing a complementary reference to \
ERA5 reanalysis. Look for cold/warm biases in polar regions, continents, and \
systematic differences.
25. **Temperature warming trend maps** — linear trends in 2m temperature \
(K/decade) over the analysis period. Global Robinson projection shows \
observation trends and model-obs trend differences; polar stereographic \
projections (>50N, <50S) highlight Arctic amplification and Antarctic \
warming patterns. Compare warming rates across models and observations.
26. **Taylor diagram** — polar plot comparing spatial pattern correlation \
(angular axis) vs normalised standard deviation (radial axis) of model \
temperature fields against Berkeley Earth. Multiple seasons (ANN, DJF, JJA) \
shown with different markers. Reference point at (1,0) represents perfect \
match. CRMS contour circles indicate centred RMS error. Assess overall \
spatial skill and seasonal dependence.
27. **Teleconnection index time series** — monthly (semi-transparent) and \
annual-mean (thick) time series of climate variability mode indices (ENSO, \
NAO, SAM, AO, IOD, PDO, QBO). Compares phase, amplitude, and timing of \
modes across models, ERA5, and optionally CMIP6. For ENSO, assess El Nino \
and La Nina event timing; for NAO/AO, assess winter dominance.
28. **Teleconnection spatial patterns** — EOF loading maps (for EOF-based \
modes: NAO, SAM, AO, PDO) or regression maps (for box-index modes: ENSO, \
IOD) showing the spatial footprint of each mode. Multi-panel: obs + models. \
Compare pattern structure, amplitude, and spatial extent across models.
29. **Teleconnection power spectra** — Welch periodogram showing power \
spectral density vs period (years) for each mode index. Assess whether \
models reproduce the observed spectral peak (e.g. 3-7 year ENSO band). \
Log-log axes. Shaded band indicates typical period range.
30. **Teleconnection seasonal variance** — grouped bar chart showing \
monthly standard deviation of each mode index. Reveals whether models \
capture the seasonal locking of variability (e.g. ENSO peaks in DJF, \
IOD in SON). Compare peak month and amplitude across models and obs."""


# ── Comparison-type-dependent analysis blocks ───────────────────────

_ANALYSIS_FOCUS = {
    "multi_model": """\
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
regression slope as a feedback parameter (W/m2/K)""",

    "resolution_sensitivity": """\
These models represent the SAME model system at different atmospheric \
resolutions. The primary scientific question is: how does model performance \
scale with increasing resolution? When CMIP6 multi-model mean (MMM) context \
is present, it provides a conventional-resolution (~100 km) baseline.

Your task is to provide a rigorous, publication-quality scientific analysis \
of the figure shown. Focus on:
- Resolution scaling: do biases decrease monotonically with finer resolution, \
or are there diminishing returns?
- Resolution-sensitive features: which regional features (e.g. Western \
Boundary Currents, orographic precipitation, tropical convection, sea ice \
edge) improve most at higher resolution?
- Resolution-insensitive features: which biases persist regardless of \
resolution, suggesting they stem from parameterisation choices rather than \
grid spacing?
- Physical mechanisms: explain WHY certain features respond to resolution \
(e.g. resolved vs parameterised processes, topographic representation)
- For radiation figures: compare against CERES EBAF (the satellite gold \
standard), note sign conventions (positive = energy into the system), \
assess cloud radiative effects, and for Gregory plots interpret the \
regression slope as a feedback parameter (W/m2/K)""",

    "single_model": """\
When CMIP6 multi-model mean (MMM) context is present, it provides a \
conventional-resolution baseline from ~100 km models for comparison.

Your task is to provide a rigorous, publication-quality scientific analysis \
of the figure shown. Focus on:
- Bias patterns and magnitudes (are biases systematic or regional?)
- Model-observation agreement (where does the model perform well/poorly?)
- Temporal behaviour: trends, drift, variability compared to observations
- Physical mechanisms driving any bias patterns
- Where the model outperforms or underperforms the CMIP6 ensemble mean
- For radiation figures: compare against CERES EBAF (the satellite gold \
standard), note sign conventions (positive = energy into the system), \
assess cloud radiative effects, and for Gregory plots interpret the \
regression slope as a feedback parameter (W/m2/K)""",

    "baseline_evaluation": """\
These models are baseline simulations for the TerraDT project, which aims \
to enhance DestinE climate Digital Twins for cryosphere, land surface, and \
related interactions. The evaluation establishes a reference performance \
level before new Digital Twin Components (land ice, sea ice, aerosols, \
vegetation) are integrated.

When CMIP6 multi-model mean (MMM) context is present, it provides a \
conventional-resolution (~100 km) baseline for comparison.

Your task is to provide a rigorous, publication-quality scientific analysis \
of the figure shown. Focus on:
- Bias patterns and magnitudes — which biases are most relevant for \
downstream impact studies (sea level, ice extent, carbon cycle)?
- Model-observation agreement — where does each model perform adequately \
as a baseline, and where are improvements most needed?
- Inter-model spread — do the three model systems agree on bias patterns, \
or do structural differences dominate?
- Cryosphere and land surface features — pay special attention to sea ice \
extent/thickness, polar temperature biases, precipitation over ice sheets, \
and land-atmosphere coupling
- Resolution effects — at ~10 km atmospheric resolution, which features \
are already well-captured vs requiring improved process representation?
- For radiation figures: compare against CERES EBAF, note sign conventions, \
assess cloud radiative effects, and for Gregory plots interpret the \
regression slope as a feedback parameter (W/m2/K)""",
}

_ANALYSIS_TAIL = """\

CRITICAL: You MUST discuss EVERY evaluated model ({model_list}) by name \
when it appears in the figure. Do not omit any of them from your analysis. \
CMIP6 models may be summarised collectively as "CMIP6 MMM" or "CMIP6 \
ensemble" — you do NOT need to name individual CMIP6 members.

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
}}"""


# ── Comparison-type-dependent synthesis blocks ──────────────────────

_SYNTHESIS_FOCUS = {
    "multi_model": """\
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
Are cloud radiative effects realistic?""",

    "resolution_sensitivity": """\
These models represent the same model system at different resolutions. \
When CMIP6 multi-model mean context is present, it provides a ~100 km \
baseline for comparison.

Given the individual figure analyses below, write a coherent scientific \
synthesis for the entire diagnostic. Consider:
1. Resolution scaling — how does model skill improve with increasing \
resolution? Are the improvements monotonic?
2. High-resolution added value — which features are only captured at the \
finest resolution? Where do medium resolutions already perform adequately?
3. Resolution-insensitive biases — which systematic errors persist across \
all resolutions, pointing to structural model issues?
4. Physical consistency — are the resolution-dependent improvements \
physically coherent (e.g. better-resolved topography reducing \
precipitation biases)?
5. Cost-benefit — given the computational cost of higher resolution, \
where is the added value most/least justified?""",

    "single_model": """\
When CMIP6 multi-model mean context is present, it provides a baseline from \
conventional-resolution (~100 km) models for comparison.

Given the individual figure analyses below, write a coherent scientific \
synthesis for the entire diagnostic. Consider:
1. Overall model skill — where does the model perform well or poorly?
2. Systematic biases — are there persistent spatial or temporal patterns?
3. Comparison to CMIP6 — how does the model compare to the ~100 km ensemble?
4. Physical consistency — are the findings physically coherent?
5. Key strengths and weaknesses — what are the model's standout features \
and primary limitations?""",

    "baseline_evaluation": """\
These are baseline simulations for TerraDT — a reference against which \
future improvements in land ice, sea ice, aerosol, and land surface \
components will be measured. When CMIP6 multi-model mean context is \
present, it provides a ~100 km baseline for comparison.

Given the individual figure analyses below, write a coherent scientific \
synthesis for the entire diagnostic. Consider:
1. Baseline adequacy — is the current model performance sufficient as a \
starting point for TerraDT Digital Twin development?
2. Priority improvement areas — which biases are most critical for the \
downstream impact studies (cryosphere, land surface, carbon cycle)?
3. Inter-model consistency — do all three models share similar weaknesses, \
or are some biases model-specific?
4. Cryosphere performance — how well do models represent sea ice, polar \
temperatures, and ice-sheet-adjacent processes?
5. Physical consistency — are the findings physically coherent across \
different diagnostic types?""",
}

_SYNTHESIS_TAIL = """\

CRITICAL: You MUST mention and discuss ALL evaluated models ({model_list}) \
by name. Do not omit any model from the synthesis.

Respond **only** with a valid JSON object matching this exact schema \
(no markdown fencing, no commentary outside the JSON):

{{
  "narrative": "2-3 paragraph synthesis (scientific, specific, quantitative)",
  "headline_finding": "one-sentence executive summary",
  "connections": ["related diagnostic 1", "related diagnostic 2"]
}}"""


# ── Builder functions ───────────────────────────────────────────────

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
    comparison_type: str = "multi_model",
    comparison_description: str = "",
) -> str:
    """Return the figure analysis system prompt.

    Parameters
    ----------
    models : list of str or None
        Model names to mention. Defaults to DestinE models.
    project_name : str or None
        Project/initiative name (e.g. ``"EERIE Ensemble"``).
        Defaults to ``"Destination Earth (DestinE)"``.
    resolution : str or None
        Resolution description (e.g. ``"high-resolution"``).
        Defaults to ``"high-resolution (~5 km)"``.
    comparison_type : str
        One of ``"multi_model"``, ``"resolution_sensitivity"``,
        ``"single_model"``.
    comparison_description : str
        Optional free-text context appended after the intro paragraph.
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

    # Intro paragraph
    intro = (
        f"You are a climate scientist evaluating diagnostic figures from "
        f"{resolution} coupled climate model simulations. The models under "
        f"evaluation are {model_list}{project_context}."
    )

    # Optional comparison description
    desc_block = ""
    if comparison_description:
        desc_block = f"\n\n{comparison_description}"

    # Figure types (shared)
    types_block = (
        "\n\nFigures compare model output against observational datasets "
        "(ERA5, CERES, EN4, etc.) using these diagnostic figure types:\n\n"
        + _FIGURE_TYPES
    )

    # Comparison-type-dependent analysis focus
    focus = _ANALYSIS_FOCUS.get(comparison_type, _ANALYSIS_FOCUS["multi_model"])

    # Tail (shared, needs format)
    tail = _ANALYSIS_TAIL.format(model_list=model_list)

    return intro + desc_block + types_block + "\n\n" + focus + tail


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
        parts.append(f"Period: {period[0]}\u2013{period[1]}")

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
        + "\n\nIMPORTANT: You must discuss every evaluated model shown in "
        "this figure by name. Do not omit any of them. CMIP6 models can be "
        "summarised collectively.\n\n"
        "Provide your analysis as a JSON object matching the schema "
        "described in the system prompt."
    )


# ── Diagnostic synthesis prompt ──────────────────────────────────────

def build_synthesis_system(
    *,
    models: list[str] | None = None,
    project_name: str | None = None,
    resolution: str | None = None,
    comparison_type: str = "multi_model",
    comparison_description: str = "",
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
    comparison_type : str
        One of ``"multi_model"``, ``"resolution_sensitivity"``,
        ``"single_model"``.
    comparison_description : str
        Optional free-text context appended after the intro paragraph.
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

    intro = (
        f"You are a climate scientist writing a synthesis of multiple "
        f"diagnostic figures from {resolution} coupled climate model "
        f"evaluations. The models are {model_list} evaluated against "
        f"observations (ERA5, CERES, EN4, etc.){project_context}."
    )

    desc_block = ""
    if comparison_description:
        desc_block = f"\n\n{comparison_description}"

    focus = _SYNTHESIS_FOCUS.get(comparison_type, _SYNTHESIS_FOCUS["multi_model"])
    tail = _SYNTHESIS_TAIL.format(model_list=model_list)

    return intro + desc_block + "\n\n" + focus + tail


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
