"""Pydantic schemas for the report generator.

Defines the data structures for editorial curation (Stage 1) and
section writing (Stage 2) outputs from the OpenAI API.
"""

from pydantic import BaseModel, Field


class SelectedFigure(BaseModel):
    """A figure selected for inclusion in the report."""

    diagnostic: str = Field(
        ..., description="Diagnostic directory name, e.g. 'global_biases'."
    )
    figure_id: str = Field(
        ..., description="Figure stem name, e.g. 'global_biases_avg_2t'."
    )
    caption: str = Field(
        ..., description="LaTeX-ready caption (plain text, no commands)."
    )
    label: str = Field(
        ..., description="LaTeX label for cross-referencing, e.g. 'fig:t2m_bias'."
    )


class ReportSection(BaseModel):
    """A thematic section proposed by the LLM."""

    section_id: str = Field(
        ..., description="Short slug, e.g. '01_surface_temperature'."
    )
    title: str = Field(
        ..., description="Section title for the report."
    )
    narrative_hook: str = Field(
        ..., description="1-2 sentence hook describing the section's story."
    )
    figure_ids: list[str] = Field(
        ..., description="List of figure stems to include in this section."
    )
    diagnostics: list[str] = Field(
        ..., description="Diagnostic names relevant to this section."
    )


class ReportStructure(BaseModel):
    """Stage 1 output: editorial plan for the full report."""

    title: str = Field(
        ..., description="Report title."
    )
    abstract: str = Field(
        ..., description="2-3 paragraph abstract (plain text)."
    )
    introduction: str = Field(
        ..., description="1-2 paragraph introduction (plain text)."
    )
    sections: list[ReportSection] = Field(
        ..., min_length=2, max_length=10, description="2-10 thematic sections."
    )
    selected_figures: list[SelectedFigure] = Field(
        ..., min_length=4,
        description="Selected figures for the report (typically 4-10)."
    )
    conclusion: str = Field(
        ..., description="1-2 paragraph conclusion (plain text)."
    )


class WrittenSection(BaseModel):
    """Stage 2 output: LLM-written prose for one section."""

    section_id: str = Field(
        ..., description="Must match the ReportSection.section_id."
    )
    title: str = Field(
        ..., description="Section title (may be refined from Stage 1)."
    )
    body: str = Field(
        ..., description=(
            "2-4 paragraphs of scientific prose. Plain text only — "
            "no LaTeX commands, no markdown."
        ),
    )
