"""Prompt templates for the report generator.

Stage 1: Editorial curation — select figures and plan report structure.
Stage 2: Section writing — write scientific prose for each section.
"""

import json


def _format_model_list(models: list[str]) -> str:
    """Format a list of model names as a natural-language enumeration."""
    if len(models) == 1:
        return models[0]
    if len(models) == 2:
        return f"{models[0]} and {models[1]}"
    return ", ".join(models[:-1]) + f", and {models[-1]}"


# ── Stage 1: Editorial Curation ──────────────────────────────────────

_CURATION_SYSTEM = """\
You are a senior climate scientist and science editor preparing a concise \
technical report evaluating {resolution} climate models{project_context} \
against observations.

The report covers results from {n_models} coupled models — {model_list}. \
These are compared against observational datasets (ERA5, CERES, EN4) and \
optionally against a CMIP6 multi-model mean ensemble. The evaluation \
period is {period} (historical).

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

Your task: from the full set of diagnostic results, select the most \
compelling findings and organise them into 5-10 thematic sections for a \
publication-quality technical report.

Selection criteria:
1. Scientific importance — biases that matter for applications
2. Clear, visually striking results
3. Thematic coherence within sections
4. Balance across climate system components (atmosphere, ocean, ice)
5. Stories where the models agree or disagree with each other \
and with CMIP6

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
}}
"""


def build_curation_system(
    *,
    models: list[str] | None = None,
    project_name: str | None = None,
    resolution: str | None = None,
    period: tuple[str, str] | None = None,
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

    return _CURATION_SYSTEM.format(
        model_list=model_list,
        n_models=len(models),
        project_context=project_context,
        resolution=resolution,
        period=f"{period[0]}-{period[1]}",
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


# ── Stage 2: Section Writing ─────────────────────────────────────────

_SECTION_SYSTEM = """\
You are a climate scientist writing a section of a technical report \
evaluating {resolution} models ({model_list}) against observations \
(ERA5, CERES, EN4) for the period {period}.

Write in an IPCC-like style:
- Factual, quantitative, cite specific magnitudes and regions
- No speculation beyond what the data shows
- Reference figures by their labels (e.g. "Figure~\\ref{{fig:label}}")
- Plain text only — NO LaTeX commands (except figure references as above)
- NO markdown formatting

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
- End the section with a brief synthesis tying the figures together

Respond ONLY with a valid JSON object (no markdown fencing):

{{
  "section_id": "USE THE EXACT section_id PROVIDED IN THE USER PROMPT",
  "title": "Section Title",
  "body": "2-4 paragraphs of detailed scientific prose..."
}}
"""


def build_section_system(
    *,
    models: list[str] | None = None,
    resolution: str | None = None,
    period: tuple[str, str] | None = None,
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
    """
    if models is None:
        models = ["IFS-FESOM", "IFS-NEMO", "ICON"]
    if resolution is None:
        resolution = "high-resolution (~5 km)"
    if period is None:
        period = ("1990", "2014")

    model_list = _format_model_list(models)

    return _SECTION_SYSTEM.format(
        model_list=model_list,
        resolution=resolution,
        period=f"{period[0]}-{period[1]}",
    )


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
        parts.append(f"\n{'─'*40}")
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
