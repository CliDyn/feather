"""Pydantic schemas for structured LLM analysis output.

These models define the expected shape of Gemini's analysis responses,
enabling validation and consistent downstream consumption (website, reports).
"""

from pydantic import BaseModel, Field


class FigureAnalysis(BaseModel):
    """Analysis of a single diagnostic figure by the LLM."""

    summary: str = Field(
        ...,
        description="1-2 sentence overview of what the figure shows.",
    )
    key_findings: list[str] = Field(
        ...,
        min_length=2,
        max_length=7,
        description="3-5 bullet-point findings (scientific, specific).",
    )
    spatial_patterns: str = Field(
        ...,
        description="Notable spatial or temporal patterns visible in the figure.",
    )
    model_agreement: str = Field(
        ...,
        description=(
            "Model-observation and inter-model agreement: where models "
            "agree/disagree and possible reasons."
        ),
    )
    physical_interpretation: str = Field(
        ...,
        description="Physical mechanism explanation for the observed patterns.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Limitations, caveats, or things to watch out for.",
    )
    confidence: str = Field(
        ...,
        description="Overall confidence: high / medium / low.",
        pattern=r"^(high|medium|low)$",
    )


class DiagnosticSynthesis(BaseModel):
    """Synthesis across all figures for a single diagnostic."""

    narrative: str = Field(
        ...,
        description="2-3 paragraph scientific synthesis of the diagnostic.",
    )
    headline_finding: str = Field(
        ...,
        description="One-sentence executive summary / headline.",
    )
    connections: list[str] = Field(
        default_factory=list,
        description=(
            "Links to other diagnostics that would complement or "
            "extend this analysis."
        ),
    )
