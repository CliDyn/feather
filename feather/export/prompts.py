"""Prompt templates for the report generator.

Stage 1: Editorial curation — select figures and plan report structure.
Stage 2: Section writing — write scientific prose for each section.

Prompts adapt to different comparison types via ``comparison_type``:

- ``"multi_model"`` (default) — independent models compared against each
  other and CMIP6.
- ``"resolution_sensitivity"`` — same model family at different resolutions.
- ``"single_model"`` — one model evaluated against observations only.
- ``"baseline_evaluation"`` — baseline simulations assessed as a reference
  for future component improvements (e.g. TerraDT).
"""

import json


def _format_model_list(models: list[str]) -> str:
    """Format a list of model names as a natural-language enumeration."""
    if len(models) == 1:
        return models[0]
    if len(models) == 2:
        return f"{models[0]} and {models[1]}"
    return ", ".join(models[:-1]) + f", and {models[-1]}"


# ── Comparison-type-dependent curation blocks ───────────────────────

_CURATION_CRITERIA = {
    "multi_model": """\
Selection criteria:
1. Scientific importance — biases that matter for applications
2. Clear, visually striking results
3. Thematic coherence within sections
4. Balance across climate system components (atmosphere, ocean, ice)
5. Stories where the models agree or disagree with each other \
and with CMIP6""",

    "resolution_sensitivity": """\
Selection criteria:
1. Resolution sensitivity — figures that clearly show how skill changes \
with resolution (monotonic improvement, saturation, or non-monotonic \
behaviour)
2. High-resolution added value — features only captured at the finest \
resolution (e.g. WBCs, orographic precipitation, mesoscale eddies)
3. Resolution-insensitive biases — systematic errors shared across all \
resolutions, indicating structural model limitations
4. Balance across climate system components (atmosphere, ocean, ice)
5. Clear, visually striking results that illustrate the resolution story""",

    "single_model": """\
Selection criteria:
1. Scientific importance — biases that matter for applications
2. Clear, visually striking results
3. Thematic coherence within sections
4. Balance across climate system components (atmosphere, ocean, ice)
5. Where the model outperforms or underperforms the CMIP6 ensemble""",

    "baseline_evaluation": """\
Selection criteria:
1. Baseline performance — figures that clearly show where models are \
adequate vs where improvements are needed
2. Cryosphere relevance — prioritise figures related to sea ice, polar \
regions, ice sheets, and land surface processes (key TerraDT targets)
3. Impact-relevant biases — biases that affect downstream applications \
(sea level projections, shipping routes, vegetation, urban planning)
4. Balance across climate system components (atmosphere, ocean, ice, land)
5. Inter-model agreement — stories where models agree or disagree on \
baseline performance""",
}


# ── Shared diagnostic descriptions ──────────────────────────────────

_DIAGNOSTIC_DESCRIPTIONS = """\
Available diagnostics:
- **Global biases**: Spatial bias maps (model minus obs climatology) for \
18 surface variables across annual, DJF, and JJA periods
- **Time series**: Global-mean time series for each variable, showing \
temporal evolution and trends
- **Seasonal cycle**: Monthly climatological cycles revealing seasonal \
bias structure
- **Radiation budget**: TOA and surface energy budget evaluation against \
CERES EBAF — includes budget bar charts, Gregory plots (feedback \
analysis), imbalance time series, and radiation bias maps
- **Sea ice**: Sea ice area, extent, and volume evaluation against \
OSI-SAF (concentration/extent) and PIOMAS/GIOMAS (volume) — includes \
time series, seasonal cycles, extreme month trends, and polar spatial maps
- **Ocean SST**: Sea surface temperature evaluation against ESA-CCI L4 \
v3.0.1 satellite observations — includes bias maps (annual/DJF/JJA), \
global-mean time series, seasonal cycle, and zonal mean profile
- **Ocean EN4**: 3D ocean temperature and salinity evaluation against \
EN4 v4.2.2 — includes surface bias maps, Hovmoller (time-depth) \
diagrams with two anomaly types, and depth-layer mean time series \
(0-700m, 700-2000m, 2000m-bottom)
- **Global trends**: Linear trend maps (units/decade) for 18 surface \
variables — shows observation trends and model-obs trend differences \
for annual, DJF, and JJA periods, revealing spatial patterns of \
warming/cooling and inter-model agreement on trend magnitudes
- **Climate variability**: Standard deviation of deseasonalised and \
detrended monthly fields — STD maps and STD difference maps for 18 \
surface variables, comparing models against ERA5
- **Precipitation (MSWEP)**: Dedicated precipitation evaluation against \
MSWEP v2.8 — includes absolute and relative bias maps (annual/DJF/JJA), \
global-mean time series, seasonal cycle, zonal mean profile (ITCZ, \
storm tracks), and precipitation intensity distribution (PDF)
- **Temperature (Berkeley Earth)**: 2m temperature evaluation against \
Berkeley Earth Land+Ocean (independent station-based dataset) — bias maps \
(annual/DJF/JJA), timeseries, seasonal cycle, zonal mean, warming trend \
maps (global + polar stereographic Arctic/Antarctic), and Taylor diagram \
(pattern correlation vs normalised standard deviation)
- **Teleconnections**: Large-scale climate variability modes — ENSO \
(Nino 3.4), NAO, SAM, AO, IOD, PDO, QBO. For each mode: index time \
series, spatial pattern (EOF/regression), power spectrum, and seasonal \
variance profile. Compares phase, amplitude, spectral characteristics, \
and seasonal locking across models, ERA5, and CMIP6"""

_CURATION_TAIL = """\

CRITICAL: Figure captions MUST name ALL evaluated models ({model_list}) \
that appear in that figure. Do not write captions that mention only a \
subset of the evaluated models when more are shown. Individual CMIP6 \
ensemble members do NOT need to be named — refer to them collectively as \
"CMIP6 MMM" or "CMIP6 ensemble". Some figures may have fewer evaluated \
models if a variable is unavailable — that is fine, just name all \
evaluated models that ARE present.

Respond ONLY with a valid JSON object (no markdown fencing) matching this schema:

{{
  "title": "Report title",
  "abstract": "2-3 paragraph abstract summarising key findings",
  "introduction": "1-2 paragraph introduction setting the context",
  "sections": [
    {{
      "section_id": "01_slug",
      "title": "Section Title",
      "narrative_hook": "1-2 sentence description of this section's story",
      "figure_ids": ["figure_stem_1", "figure_stem_2"],
      "diagnostics": ["diagnostic_directory_key_1", "diagnostic_directory_key_2"]
    }}
  ],
  "selected_figures": [
    {{
      "diagnostic": "diagnostic_directory_name (use the EXACT key from the section headers below, e.g. 'global_biases', 'radiation_budget', NOT the figure title)",
      "figure_id": "figure_stem",
      "caption": "Descriptive caption for this figure",
      "label": "fig:short_label"
    }}
  ],
  "conclusion": "1-2 paragraph conclusion"
}}"""


# ── Stage 1: Editorial Curation ──────────────────────────────────────

def build_curation_system(
    *,
    models: list[str] | None = None,
    project_name: str | None = None,
    resolution: str | None = None,
    period: tuple[str, str] | None = None,
    comparison_type: str = "multi_model",
    comparison_description: str = "",
) -> str:
    """Return the curation system prompt.

    Parameters
    ----------
    models : list of str or None
        Model names. Defaults to DestinE models.
    project_name : str or None
        Project/initiative name. Defaults to DestinE.
    resolution : str or None
        Resolution description. Defaults to ``"high-resolution (~5 km)"``.
    period : tuple of str or None
        Evaluation period. Defaults to ``("1990", "2014")``.
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
    if period is None:
        period = ("1990", "2014")

    model_list = _format_model_list(models)
    if project_name:
        project_context = f" from the {project_name} project"
    else:
        project_context = " from the DestinE initiative"

    intro = (
        f"You are a senior climate scientist and science editor preparing "
        f"a concise technical report evaluating {resolution} climate "
        f"models{project_context} against observations.\n\n"
        f"The report covers results from {len(models)} coupled models "
        f"\\u2014 {model_list}. These are compared against observational "
        f"datasets (ERA5, CERES, EN4) and optionally against a CMIP6 "
        f"multi-model mean ensemble. The evaluation period is "
        f"{period[0]}-{period[1]} (historical)."
    )

    desc_block = ""
    if comparison_description:
        desc_block = f"\n\n{comparison_description}"

    criteria = _CURATION_CRITERIA.get(
        comparison_type, _CURATION_CRITERIA["multi_model"],
    )

    tail = _CURATION_TAIL.format(model_list=model_list)

    return (
        intro + desc_block + "\n\n" + _DIAGNOSTIC_DESCRIPTIONS
        + "\n\nYour task: from the full set of diagnostic results, select "
        "the most compelling findings and organise them into 5-10 thematic "
        "sections for a publication-quality technical report.\n\n"
        + criteria + tail
    )


def build_curation_prompt(
    syntheses: dict[str, dict],
    figure_metadata: dict[str, list[dict]],
    figure_analyses: dict[str, list[dict]],
    n_highlights: int = 10,
) -> str:
    """Build the user prompt for Stage 1 editorial curation.

    Parameters
    ----------
    syntheses : dict
        diagnostic_name -> synthesis JSON dict.
    figure_metadata : dict
        diagnostic_name -> list of figure metadata dicts.
    figure_analyses : dict
        diagnostic_name -> list of figure analysis dicts.
    n_highlights : int
        Target number of figures to select.
    """
    parts = [
        f"Please select approximately {n_highlights} figures for the report.\n",
        "=" * 60,
        "DIAGNOSTIC SYNTHESES",
        "=" * 60,
    ]

    for diag_name, synth in sorted(syntheses.items()):
        parts.append(f"\n--- {diag_name} ---")
        parts.append(json.dumps(synth, indent=2))

    parts.append("\n" + "=" * 60)
    parts.append("AVAILABLE FIGURES (with metadata and analyses)")
    parts.append("=" * 60)

    for diag_name in sorted(figure_metadata.keys()):
        parts.append(f"\n--- {diag_name} ---")
        metas = figure_metadata[diag_name]
        analyses = figure_analyses.get(diag_name, [])

        for meta in metas:
            fig_id = meta.get("figure_id", "unknown")
            parts.append(f"\n  Figure: {fig_id}")
            parts.append(f"    Title: {meta.get('title', 'N/A')}")
            parts.append(f"    Variables: {', '.join(meta.get('variables_used', []))}")
            parts.append(f"    Models: {', '.join(meta.get('models', []))}")
            parts.append(f"    Description: {meta.get('description', 'N/A')}")

            # Find matching analysis
            matching = [a for a in analyses if a.get("figure_id") == fig_id]
            if matching:
                a = matching[0]
                parts.append(f"    Summary: {a.get('summary', 'N/A')}")
                parts.append(f"    Spatial patterns: {a.get('spatial_patterns', 'N/A')}")
                parts.append(f"    Confidence: {a.get('confidence', 'N/A')}")
                findings = a.get("key_findings", [])
                for f in findings:
                    parts.append(f"      - {f}")

    parts.append("\n\nSelect the most interesting findings and organise "
                 "them into a coherent report structure.")

    return "\n".join(parts)


# ── Comparison-type-dependent section blocks ────────────────────────

_SECTION_FOCUS = {
    "multi_model": """\
Write in an IPCC-like style:
- Factual, quantitative, cite specific magnitudes and regions
- No speculation beyond what the data shows
- Reference figures by their labels (e.g. "Figure~\\ref{{fig:label}}")
- Plain text only — NO LaTeX commands (except figure references as above)
- NO markdown formatting
- CRITICAL: You MUST discuss ALL evaluated models ({model_list}) by name \
in the section text. Do not omit any model — each one deserves individual \
assessment of its performance

IMPORTANT REQUIREMENTS for depth and quality:
- Write 2-4 substantial paragraphs per section
- Dedicate at least one full paragraph to EACH figure in the section, \
describing in detail what it shows: spatial patterns, regional hotspots, \
inter-model differences, magnitudes, and physical interpretation
- Include quantitative values wherever the analyses provide them \
(e.g. bias magnitudes in K, percentage changes)
- Discuss where models agree or disagree with each other and with \
observations (and optionally with CMIP6 MMM), and explain WHY
- Connect the findings to physical mechanisms (feedbacks, circulation \
changes, thermodynamic constraints)
- For radiation budget figures: discuss energy balance closure, cloud \
radiative effects, and compare against CERES EBAF (the satellite reference \
standard). For Gregory plots, interpret the regression slope as a feedback \
parameter and relate to equilibrium climate sensitivity.
- End the section with a brief synthesis tying the figures together""",

    "resolution_sensitivity": """\
Write in an IPCC-like style:
- Factual, quantitative, cite specific magnitudes and regions
- No speculation beyond what the data shows
- Reference figures by their labels (e.g. "Figure~\\ref{{fig:label}}")
- Plain text only — NO LaTeX commands (except figure references as above)
- NO markdown formatting
- CRITICAL: You MUST discuss ALL evaluated models ({model_list}) by name \
in the section text, explicitly comparing across resolutions

IMPORTANT REQUIREMENTS for depth and quality:
- Write 2-4 substantial paragraphs per section
- Dedicate at least one full paragraph to EACH figure in the section, \
describing in detail what it shows
- Frame the discussion around resolution scaling: does the highest-resolution \
configuration outperform the coarser ones? By how much? Is the improvement \
monotonic?
- Highlight resolution-sensitive features (e.g. Western Boundary Currents, \
orographic precipitation, tropical convection, mesoscale ocean eddies) and \
explain WHY they respond to resolution
- Identify biases that persist across all resolutions — these indicate \
structural model limitations independent of grid spacing
- Include quantitative values wherever the analyses provide them
- Connect the findings to physical mechanisms (resolved vs parameterised \
processes, topographic representation, eddy-resolving thresholds)
- For radiation budget figures: discuss energy balance closure, cloud \
radiative effects, and compare against CERES EBAF
- End the section with a brief synthesis on the cost-benefit of \
increased resolution for the features discussed""",

    "single_model": """\
Write in an IPCC-like style:
- Factual, quantitative, cite specific magnitudes and regions
- No speculation beyond what the data shows
- Reference figures by their labels (e.g. "Figure~\\ref{{fig:label}}")
- Plain text only — NO LaTeX commands (except figure references as above)
- NO markdown formatting
- CRITICAL: You MUST discuss the evaluated model ({model_list}) in detail

IMPORTANT REQUIREMENTS for depth and quality:
- Write 2-4 substantial paragraphs per section
- Dedicate at least one full paragraph to EACH figure in the section, \
describing in detail what it shows: spatial patterns, regional hotspots, \
bias magnitudes, and physical interpretation
- Include quantitative values wherever the analyses provide them
- Discuss where the model outperforms or underperforms the CMIP6 \
ensemble and explain WHY
- Connect the findings to physical mechanisms
- For radiation budget figures: discuss energy balance closure, cloud \
radiative effects, and compare against CERES EBAF
- End the section with a brief synthesis tying the figures together""",

    "baseline_evaluation": """\
Write in an IPCC-like style:
- Factual, quantitative, cite specific magnitudes and regions
- No speculation beyond what the data shows
- Reference figures by their labels (e.g. "Figure~\\ref{{fig:label}}")
- Plain text only — NO LaTeX commands (except figure references as above)
- NO markdown formatting
- CRITICAL: You MUST discuss ALL evaluated models ({model_list}) by name \
in the section text. Do not omit any model

IMPORTANT REQUIREMENTS for depth and quality:
- Write 2-4 substantial paragraphs per section
- Dedicate at least one full paragraph to EACH figure in the section
- Frame the discussion as a baseline assessment: the current model \
configurations will serve as references for improvements in land ice, \
sea ice, aerosol, and land surface representations within TerraDT
- Highlight biases most relevant for cryosphere and land surface \
applications (sea level rise, glacier retreat, shipping routes, \
vegetation/carbon sequestration, urban climate extremes)
- Include quantitative values wherever the analyses provide them
- Discuss inter-model agreement and relate to structural model differences
- Connect the findings to physical mechanisms
- For radiation budget figures: discuss energy balance closure, cloud \
radiative effects, and compare against CERES EBAF
- End the section with a brief synthesis identifying priority areas \
for model improvement""",
}


# ── Stage 2: Section Writing ─────────────────────────────────────────

def build_section_system(
    *,
    models: list[str] | None = None,
    resolution: str | None = None,
    period: tuple[str, str] | None = None,
    comparison_type: str = "multi_model",
    comparison_description: str = "",
) -> str:
    """Return the section writing system prompt.

    Parameters
    ----------
    models : list of str or None
        Model names. Defaults to DestinE models.
    resolution : str or None
        Resolution description. Defaults to ``"high-resolution (~5 km)"``.
    period : tuple of str or None
        Evaluation period. Defaults to ``("1990", "2014")``.
    comparison_type : str
        One of ``"multi_model"``, ``"resolution_sensitivity"``,
        ``"single_model"``.
    comparison_description : str
        Optional free-text context appended after the intro line.
    """
    if models is None:
        models = ["IFS-FESOM", "IFS-NEMO", "ICON"]
    if resolution is None:
        resolution = "high-resolution (~5 km)"
    if period is None:
        period = ("1990", "2014")

    model_list = _format_model_list(models)

    intro = (
        f"You are a climate scientist writing a section of a technical "
        f"report evaluating {resolution} models ({model_list}) against "
        f"observations (ERA5, CERES, EN4) for the period "
        f"{period[0]}-{period[1]}."
    )

    desc_block = ""
    if comparison_description:
        desc_block = f"\n\n{comparison_description}"

    focus = _SECTION_FOCUS.get(comparison_type, _SECTION_FOCUS["multi_model"])
    focus = focus.format(model_list=model_list)

    tail = """

Respond ONLY with a valid JSON object (no markdown fencing):

{{
  "section_id": "USE THE EXACT section_id PROVIDED IN THE USER PROMPT",
  "title": "Section Title",
  "body": "2-4 paragraphs of detailed scientific prose..."
}}"""

    return intro + desc_block + "\n\n" + focus + tail


def build_section_prompt(
    section: dict,
    selected_figures: list[dict],
    figure_analyses: dict[str, list[dict]],
    syntheses: dict[str, dict],
) -> str:
    """Build the user prompt for writing a single section.

    Parameters
    ----------
    section : dict
        ReportSection dict from Stage 1.
    selected_figures : list of dict
        SelectedFigure dicts for figures in this section.
    figure_analyses : dict
        diagnostic_name -> list of analysis dicts.
    syntheses : dict
        diagnostic_name -> synthesis dict.
    """
    parts = [
        f"SECTION ID (use this exactly in your response): {section['section_id']}",
        f"Section title: {section['title']}",
        f"Narrative hook: {section['narrative_hook']}",
        f"\n{'='*60}",
        "FIGURES IN THIS SECTION — discuss EACH one in depth",
        f"{'='*60}",
    ]

    for fig in selected_figures:
        sep = "\u2500" * 40
        parts.append(f"\n{sep}")
        parts.append(f"Figure: {fig['figure_id']} (from diagnostic: {fig['diagnostic']})")
        parts.append(f"Caption: {fig['caption']}")
        parts.append(f"LaTeX label: {fig['label']}")

        # Add full analysis data for this figure
        diag_analyses = figure_analyses.get(fig["diagnostic"], [])
        matching = [a for a in diag_analyses
                    if a.get("figure_id") == fig["figure_id"]]
        if matching:
            a = matching[0]
            parts.append(f"\nFull analysis for this figure:")
            parts.append(f"  Summary: {a.get('summary', '')}")
            parts.append(f"  Spatial patterns: {a.get('spatial_patterns', '')}")
            parts.append(f"  Physical interpretation: "
                         f"{a.get('physical_interpretation', '')}")
            parts.append(f"  Model agreement: {a.get('model_agreement', '')}")
            parts.append(f"  Confidence: {a.get('confidence', '')}")
            findings = a.get("key_findings", [])
            if findings:
                parts.append("  Key findings:")
                for f in findings:
                    parts.append(f"    - {f}")
            caveats = a.get("caveats", [])
            if caveats:
                parts.append("  Caveats:")
                for c in caveats:
                    parts.append(f"    - {c}")

    # Add full syntheses for related diagnostics
    parts.append(f"\n{'='*60}")
    parts.append("DIAGNOSTIC SYNTHESES for context")
    parts.append(f"{'='*60}")
    for diag_name in section.get("diagnostics", []):
        if diag_name in syntheses:
            synth = syntheses[diag_name]
            parts.append(f"\n--- {diag_name} ---")
            parts.append(f"Headline: {synth.get('headline_finding', '')}")
            parts.append(f"Full narrative:\n{synth.get('narrative', '')}")
            connections = synth.get("connections", [])
            if connections:
                parts.append(f"Connections: {', '.join(connections)}")

    parts.append(f"\n{'='*60}")
    parts.append("INSTRUCTIONS")
    parts.append(f"{'='*60}")
    parts.append(
        f"Write the section '{section['title']}' (section_id: "
        f"{section['section_id']}).\n"
        "- Dedicate at least one full paragraph to EACH figure listed above\n"
        "- Describe what the figure shows in detail: spatial patterns, "
        "magnitudes, regional hotspots\n"
        "- Compare models against each other and against observations explicitly\n"
        "- Explain physical mechanisms driving the patterns\n"
        "- Use quantitative values from the analyses\n"
        "- Reference each figure as Figure~\\ref{label}\n"
        "- End with a synthesis paragraph connecting the figures"
    )

    return "\n".join(parts)
