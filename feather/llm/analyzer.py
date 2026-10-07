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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from feather import provenance
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
        # Cap on response length. For gemini-2.5 models the thinking tokens
        # count toward this budget, so when thinking is enabled the JSON
        # response can be truncated (→ "Unterminated string" on parse) unless
        # the cap leaves room beyond the thinking budget.
        self.max_output_tokens = fa_config.get("max_output_tokens", 8192)

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
                        existing = json.load(f)
                    self._warn_if_stale(existing, json_path, png_path)
                    analyses.append(existing)
                    continue

                try:
                    analysis, prov = self.analyze_figure_with_provenance(
                        png_path, json_path,
                    )
                    analysis_dict = analysis.model_dump()
                    analysis_dict["provenance"] = prov
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
                    synthesis, prov = self.synthesize_diagnostic_with_provenance(
                        diag_name, analyses,
                    )
                    synthesis_dict = synthesis.model_dump()
                    synthesis_dict["provenance"] = prov
                    self._save_json(synthesis_path, synthesis_dict)
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
        return self.analyze_figure_with_provenance(png_path, json_path)[0]

    def analyze_figure_with_provenance(
        self,
        png_path: Path,
        json_path: Path,
    ) -> tuple[FigureAnalysis, dict[str, Any]]:
        """:meth:`analyze_figure`, plus the interpretation provenance record.

        The record binds the analysis to the exact sidecar and image it was
        generated from (``sidecar_sha256``, ``figure_sha256``), so a later
        re-run that changes the figure leaves a detectably stale analysis.
        """
        with open(json_path) as f:
            metadata = json.load(f)

        user_prompt = build_figure_prompt(metadata)
        png_bytes = png_path.read_bytes()
        image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        system = build_figure_analysis_system(
            models=self._prompt_models,
            project_name=self._prompt_project,
            resolution=self._prompt_resolution,
            comparison_type=self._comparison_type,
            comparison_description=self._comparison_description,
        )

        response_text = self._call_gemini(
            system_instruction=system,
            contents=[image_part, user_prompt],
            response_schema=FigureAnalysis,
        )

        analysis_data, repairs = _parse_json_with_repairs(response_text)
        analysis = FigureAnalysis(**analysis_data)
        prov = self._interpretation_record(
            FigureAnalysis, system, user_prompt, repairs,
            sidecar_sha256=provenance.sidecar_digest(metadata),
            figure_sha256=provenance.bytes_digest(png_bytes),
            figure_run_id=metadata.get("run_id"),
        )
        return analysis, prov

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
        return self.synthesize_diagnostic_with_provenance(
            diagnostic_name, figure_analyses,
        )[0]

    def synthesize_diagnostic_with_provenance(
        self,
        diagnostic_name: str,
        figure_analyses: list[dict],
    ) -> tuple[DiagnosticSynthesis, dict[str, Any]]:
        """:meth:`synthesize_diagnostic`, plus its provenance record."""
        # Provenance blocks are bookkeeping, not evidence: keep them out of
        # the prompt (they would only cost tokens).
        clean = [{k: v for k, v in a.items() if k != "provenance"}
                 for a in figure_analyses]
        user_prompt = build_synthesis_prompt(diagnostic_name, clean)
        system = build_synthesis_system(
            models=self._prompt_models,
            project_name=self._prompt_project,
            resolution=self._prompt_resolution,
            comparison_type=self._comparison_type,
            comparison_description=self._comparison_description,
        )

        response_text = self._call_gemini(
            system_instruction=system,
            contents=[user_prompt],
            response_schema=DiagnosticSynthesis,
        )

        synthesis_data, repairs = _parse_json_with_repairs(response_text)
        synthesis = DiagnosticSynthesis(**synthesis_data)
        prov = self._interpretation_record(
            DiagnosticSynthesis, system, user_prompt, repairs,
            n_analyses=len(figure_analyses),
            analysis_sidecar_sha256=sorted(
                (a.get("provenance") or {}).get("sidecar_sha256") or ""
                for a in figure_analyses
            ),
        )
        return synthesis, prov

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
        response_schema: Any = None,
    ) -> str:
        """Call Gemini via Vertex AI Express with retry on transient errors.

        When ``response_schema`` is provided, the model is run in structured
        JSON mode so the response is guaranteed to be schema-conformant JSON
        (avoids unparseable free-form text with unescaped quotes/newlines).
        """
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        # Thinking tokens count toward max_output_tokens on gemini-2.5, so the
        # output cap must exceed the thinking budget or the JSON gets truncated.
        max_output = self.max_output_tokens
        if self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )
            max_output = max(max_output, self.thinking_budget + 4096)
        config_kwargs["max_output_tokens"] = max_output

        started = time.monotonic()
        failures: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                self._raise_if_truncated(response)
                self._last_call = {
                    "attempts": attempt,
                    "failed_attempts": failures,
                    "finish_reason": _finish_reason(response),
                    "tokens": _usage(response),
                    "max_output_tokens": max_output,
                    "wall_s": round(time.monotonic() - started, 2),
                }
                return response.text
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}"[:300])
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
    def _raise_if_truncated(response: Any) -> None:
        """Raise if the model stopped at the token cap (truncated JSON).

        A ``MAX_TOKENS`` finish reason means the response was cut off, so the
        JSON is incomplete and would fail to parse with a misleading
        "Unterminated string" error. Raising here lets the retry loop run and
        produces a clear, diagnosable message if it persists.
        """
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            reason = getattr(cand, "finish_reason", None)
            if reason is not None and str(reason).rsplit(".", 1)[-1] == "MAX_TOKENS":
                raise RuntimeError(
                    "Gemini response truncated at the output-token cap "
                    "(finish_reason=MAX_TOKENS); increase "
                    "llm.figure_analysis.max_output_tokens or lower "
                    "thinking_budget."
                )

    def _interpretation_record(
        self, schema: type, system: str, user_prompt: str,
        repairs: list[str], **fields: Any,
    ) -> dict[str, Any]:
        """Provenance of one LLM call: who said it, from what, at what cost."""
        call = getattr(self, "_last_call", None) or {}
        return {
            "schema_version": provenance.SCHEMA_VERSION,
            "run_id": provenance.current_run_id(),
            "provider": "vertex",
            "model": self.model_name,
            "schema": schema.__name__,
            "thinking_budget": self.thinking_budget,
            "prompt_sha256": {
                "system": provenance.text_digest(system),
                "user": provenance.text_digest(user_prompt),
            },
            **call,
            "repairs": repairs,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }

    @staticmethod
    def _warn_if_stale(analysis: dict, json_path: Path, png_path: Path) -> None:
        """Warn when a cached analysis was written against a different figure."""
        prov = analysis.get("provenance") or {}
        recorded = prov.get("sidecar_sha256")
        if not recorded:
            return
        try:
            with open(json_path) as f:
                current = provenance.sidecar_digest(json.load(f))
        except (OSError, ValueError):
            return
        if current != recorded:
            logger.warning(
                "  Stale analysis for %s: its figure metadata changed since it "
                "was written (rerun with --no-skip-existing to refresh)",
                png_path.stem,
            )

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse a JSON response, tolerating common LLM deviations.

        Handles markdown fencing, invalid LaTeX escape sequences, literal
        control characters inside strings (``strict=False``), and leading or
        trailing prose around the JSON object.
        """
        return _parse_json_with_repairs(text)[0]

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


# ── Module helpers ───────────────────────────────────────────────────


def _parse_json_with_repairs(text: str) -> tuple[dict, list[str]]:
    """Parse an LLM JSON response; also return which repairs were needed.

    The repair list is part of the interpretation provenance: an analysis
    that only parsed after escape fixing or prose stripping deserves a
    second look.
    """
    repairs: list[str] = []
    cleaned = text.strip()

    # Strip ```json ... ``` wrapper
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
        repairs.append("strip_markdown_fence")

    # strict=False permits literal control characters (e.g. unescaped
    # newlines/tabs) inside string values, which Gemini occasionally emits.
    try:
        return json.loads(cleaned, strict=False), repairs
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
    repairs.append("fix_invalid_escapes")

    try:
        return json.loads(cleaned, strict=False), repairs
    except json.JSONDecodeError:
        # Last resort: extract the outermost {...} object and retry,
        # discarding any leading/trailing prose the model added.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            repairs.append("extract_outer_object")
            return json.loads(cleaned[start:end + 1], strict=False), repairs
        raise


def _finish_reason(response: Any) -> str | None:
    """Finish reason of the first candidate, as a plain name."""
    for cand in getattr(response, "candidates", None) or []:
        reason = getattr(cand, "finish_reason", None)
        if reason is not None:
            return str(getattr(reason, "name", reason)).rsplit(".", 1)[-1]
    return None


def _usage(response: Any) -> dict[str, int | None]:
    """Token usage reported by the API (``None`` where not reported)."""
    um = getattr(response, "usage_metadata", None)

    def _get(name: str) -> int | None:
        v = getattr(um, name, None) if um is not None else None
        return v if isinstance(v, int) else None

    return {
        "input": _get("prompt_token_count"),
        "output": _get("candidates_token_count"),
        "thinking": _get("thoughts_token_count"),
        "total": _get("total_token_count"),
    }
