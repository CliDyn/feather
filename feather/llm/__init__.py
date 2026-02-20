"""LLM analysis sub-package for Feather.

Uses Gemini to provide scientific analysis of generated diagnostic figures.
"""

from feather.llm.analyzer import FigureAnalyzer
from feather.llm.schemas import DiagnosticSynthesis, FigureAnalysis

__all__ = [
    "FigureAnalyzer",
    "FigureAnalysis",
    "DiagnosticSynthesis",
]
