"""Gemini-based analysis of climate diagnostic figures.

The :class:`FigureAnalyzer` scans ``{output_dir}/figures/`` for PNG files
with matching JSON metadata sidecars, sends each figure to Gemini for
scientific interpretation, and saves structured analysis results to
``{output_dir}/analysis/``.

No Dask or data access required — works entirely on already-generated figures.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from feather.config import FeatherConfig
from feather.llm.prompts import (
    build_figure_analysis_system,
    build_figure_prompt,
    build_synthesis_prompt,
    build_synthesis_system,
)
from feather.llm.schemas import DiagnosticSynthesis, FigureAnalysis

logger = logging.getLogger(__name__)


class FigureAnalyzer:
    """Analyse diagnostic figures using Gemini via Vertex AI Express.

    Parameters
    ----------
    config : FeatherConfig
        Feather configuration (for output paths and LLM settings).
    api_key : str or None
        Vertex AI API key. Falls back to the env var specified in config
        (``llm.figure_analysis.api_key_env``, default ``VERTEX_API_KEY``).
    """

    def __init__(
        self,
        config: FeatherConfig,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self.figures_dir = Path(config.output_dir) / "figures"
        self.analysis_dir = Path(config.output_dir) / "analysis"

        # Read LLM config for figure analysis
        fa_config = config.llm.get("figure_analysis", {})
        self.model_name = fa_config.get("model", "gemini-2.5-flash")
        self.max_retries = fa_config.get("max_retries", 3)
        self.retry_delay = fa_config.get("retry_delay", 10)
        self.skip_existing_default = fa_config.get("skip_existing", True)
        self.thinking_budget = fa_config.get("thinking_budget", 0)

        # Configure Vertex AI client
        api_key_env = fa_config.get("api_key_env", "VERTEX_API_KEY")
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise RuntimeError(
                f"Vertex AI API key not found. Set the '{api_key_env}' "
                "environment variable or pass api_key= to FigureAnalyzer."
            )
        self.client = genai.Client(api_key=key)
        logger.info("Using Gemini model: %s (Gemini Developer API)", self.model_name)

        # Extract prompt context from config for templated system prompts
        self._prompt_models = config.models
        self._prompt_project = config.project.get("name") if config.project else None
        self._prompt_resolution = config.project.get(
            "resolution"
        ) if config.project else None
        self._comparison_type = config.get_comparison_type()
        self._comparison_description = config.get_comparison_description()

    # ── Public API ───────────────────────────────────────────────────

    def run(
        self,
        *,
        skip_existing: bool | None = None,
        diagnostics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyse all figures and produce per-diagnostic syntheses.

        Parameters
        ----------
        skip_existing : bool or None
            If True, skip figures that already have analysis JSON.
            Defaults to the config value.
        diagnostics : list of str or None
            If provided, only analyse these diagnostic names.

        Returns
        -------
        dict
            ``{"figure_analyses": int, "syntheses": int}``
        """
        if skip_existing is None:
            skip_existing = self.skip_existing_default

        figure_pairs = self._discover_figures()
        logger.info(
            "Discovered %d figure(s) across %d diagnostic(s)",
            sum(len(v) for v in figure_pairs.values()),
            len(figure_pairs),
        )

        # Filter diagnostics if requested
        if diagnostics:
            figure_pairs = {
                k: v for k, v in figure_pairs.items() if k in diagnostics
            }

        n_figures = 0
        n_syntheses = 0

        for diag_name, pairs in sorted(figure_pairs.items()):
            logger.info("── Diagnostic: %s (%d figures)", diag_name, len(pairs))
            analyses = []

            for png_path, json_path in pairs:
                analysis_path = self._analysis_path(diag_name, png_path.stem)
                if skip_existing and analysis_path.exists():
                    logger.info("  Skipping (exists): %s", png_path.stem)
                    with open(analysis_path) as f:
                        analyses.append(json.load(f))
                    continue

                try:
                    analysis = self.analyze_figure(png_path, json_path)
                    analysis_dict = analysis.model_dump()
                    self._save_json(analysis_path, analysis_dict)
                    analyses.append(analysis_dict)
                    n_figures += 1
                    logger.info("  Analysed: %s", png_path.stem)
                except Exception:
                    logger.exception("  Failed: %s", png_path.stem)

            # Per-diagnostic synthesis
            if analyses:
                synthesis_path = self._synthesis_path(diag_name)
                if skip_existing and synthesis_path.exists():
                    logger.info("  Synthesis exists, skipping")
                    continue
                try:
                    synthesis = self.synthesize_diagnostic(diag_name, analyses)
                    self._save_json(synthesis_path, synthesis.model_dump())
                    n_syntheses += 1
                    logger.info("  Synthesis complete: %s", diag_name)
                except Exception:
                    logger.exception("  Synthesis failed: %s", diag_name)

        logger.info(
            "Analysis complete — %d figure(s), %d synthesis(es)",
            n_figures,
            n_syntheses,
        )
        return {"figure_analyses": n_figures, "syntheses": n_syntheses}

    def analyze_figure(
        self,
        png_path: Path,
        json_path: Path,
    ) -> FigureAnalysis:
        """Analyse a single figure by sending it to Gemini.

        Parameters
        ----------
        png_path : Path
            Path to the PNG figure.
        json_path : Path
            Path to the JSON metadata sidecar.

        Returns
        -------
        FigureAnalysis
            Structured analysis result.
        """
        with open(json_path) as f:
            metadata = json.load(f)

        user_prompt = build_figure_prompt(metadata)
        image_part = types.Part.from_bytes(
            data=png_path.read_bytes(), mime_type="image/png"
        )

        response_text = self._call_gemini(
            system_instruction=build_figure_analysis_system(
                models=self._prompt_models,
                project_name=self._prompt_project,
                resolution=self._prompt_resolution,
                comparison_type=self._comparison_type,
                comparison_description=self._comparison_description,
            ),
            contents=[image_part, user_prompt],
        )

        analysis_data = self._parse_json_response(response_text)
        return FigureAnalysis(**analysis_data)

    def synthesize_diagnostic(
        self,
        diagnostic_name: str,
        figure_analyses: list[dict],
    ) -> DiagnosticSynthesis:
        """Produce a synthesis across all figures for a diagnostic.

        Parameters
        ----------
        diagnostic_name : str
            Machine name of the diagnostic.
        figure_analyses : list of dict
            Individual FigureAnalysis dicts.

        Returns
        -------
        DiagnosticSynthesis
        """
        user_prompt = build_synthesis_prompt(diagnostic_name, figure_analyses)

        response_text = self._call_gemini(
            system_instruction=build_synthesis_system(
                models=self._prompt_models,
                project_name=self._prompt_project,
                resolution=self._prompt_resolution,
                comparison_type=self._comparison_type,
                comparison_description=self._comparison_description,
            ),
            contents=[user_prompt],
        )

        synthesis_data = self._parse_json_response(response_text)
        return DiagnosticSynthesis(**synthesis_data)

    # ── Private helpers ──────────────────────────────────────────────

    def _discover_figures(self) -> dict[str, list[tuple[Path, Path]]]:
        """Scan figures/ for PNG+JSON pairs grouped by diagnostic subdir."""
        result: dict[str, list[tuple[Path, Path]]] = {}

        if not self.figures_dir.exists():
            logger.warning("Figures directory not found: %s", self.figures_dir)
            return result

        for diag_dir in sorted(self.figures_dir.iterdir()):
            if not diag_dir.is_dir():
                continue
            pairs = []
            for png in sorted(diag_dir.glob("*.png")):
                json_file = png.with_suffix(".json")
                if json_file.exists():
                    pairs.append((png, json_file))
                else:
                    logger.warning(
                        "No JSON sidecar for %s — skipping", png.name
                    )
            if pairs:
                result[diag_dir.name] = pairs

        return result

    def _call_gemini(
        self,
        system_instruction: str,
        contents: list,
    ) -> str:
        """Call Gemini via Vertex AI Express with retry on transient errors."""
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
        }
        if self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return response.text
            except Exception as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s — "
                        "retrying in %ds",
                        attempt,
                        self.max_retries,
                        exc,
                        self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise

        raise RuntimeError("Gemini call failed after all retries")  # pragma: no cover

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse a JSON response, stripping markdown fencing and fixing
        invalid LaTeX escape sequences."""
        cleaned = text.strip()

        # Strip ```json ... ``` wrapper
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Try parsing as-is first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fix invalid escape sequences (e.g. \Delta, \theta from LaTeX)
        # \u not followed by 4 hex digits (e.g. \units)
        cleaned = re.sub(
            r"\\u(?![0-9a-fA-F]{4})",
            r"\\\\u",
            cleaned,
        )
        # All other invalid escapes
        cleaned = re.sub(
            r'\\(?!["\\/bfnrtu])',
            r"\\\\",
            cleaned,
        )

        return json.loads(cleaned)

    def _analysis_path(self, diagnostic_name: str, figure_stem: str) -> Path:
        """Path for a figure-level analysis JSON."""
        return self.analysis_dir / diagnostic_name / f"{figure_stem}_analysis.json"

    def _synthesis_path(self, diagnostic_name: str) -> Path:
        """Path for a diagnostic-level synthesis JSON."""
        return self.analysis_dir / diagnostic_name / "synthesis.json"

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        """Save a dict as formatted JSON, creating directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
