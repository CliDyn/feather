"""Report generator — main orchestrator.

Collects diagnostic outputs, runs 3-stage LLM workflow, and produces
a LaTeX technical report with selected figures.

Stage 1: Editorial curation (1 API call) -> ReportStructure
Stage 2: Section writing (1 call per section) -> WrittenSection per section
Stage 3: LaTeX assembly (pure Python) -> report.tex
"""

import json
import logging
from pathlib import Path

from feather.config import FeatherConfig
from feather.export.latex_builder import (
    build_document,
    compile_pdf,
    copy_figures,
)
from feather.export.openai_client import OpenAIClient
from feather.export.prompts import (
    build_curation_prompt,
    build_curation_system,
    build_section_prompt,
    build_section_system,
)
from feather.export.schemas import (
    ReportStructure,
    WrittenSection,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate a LaTeX technical report from diagnostic outputs.

    Parameters
    ----------
    config : FeatherConfig
        Feather configuration.
    compile_pdf : bool
        Whether to run pdflatex after generating .tex.
    api_key : str or None
        OpenAI API key. Falls back to env var from config.
    """

    def __init__(
        self,
        config: FeatherConfig,
        *,
        compile_pdf: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self.compile_pdf_flag = compile_pdf
        self.output_dir = Path(config.output_dir)
        self.pub_dir = self.output_dir / "publication"
        self.figures_dir = self.output_dir / "figures"
        self.analysis_dir = self.output_dir / "analysis"

        report_cfg = config.report if config.report else {}
        self.n_highlights = report_cfg.get("n_highlights", 10)
        self.client = OpenAIClient(report_cfg, api_key=api_key)

    def run(self, *, skip_existing: bool = True) -> Path:
        """Run the full report pipeline.

        Parameters
        ----------
        skip_existing : bool
            If True, skip stages whose cached outputs already exist.

        Returns
        -------
        Path
            Path to the generated report.tex.
        """
        self.pub_dir.mkdir(parents=True, exist_ok=True)

        # Collect all inputs
        syntheses = self._load_syntheses()
        figure_metadata = self._load_figure_metadata()
        figure_analyses = self._load_figure_analyses()

        logger.info(
            "Collected %d syntheses, %d diagnostics with figures, "
            "%d diagnostics with analyses",
            len(syntheses), len(figure_metadata), len(figure_analyses),
        )

        # Stage 1: Editorial curation
        structure = self._stage1_curation(
            syntheses, figure_metadata, figure_analyses,
            skip_existing=skip_existing,
        )
        n_figs = len(structure["selected_figures"])
        n_secs = len(structure["sections"])
        logger.info(
            "Stage 1 complete: %d sections, %d figures selected",
            n_secs, n_figs,
        )

        # Stage 2: Section writing
        written_sections = self._stage2_sections(
            structure, figure_analyses, syntheses,
            skip_existing=skip_existing,
        )
        logger.info("Stage 2 complete: %d sections written", len(written_sections))

        # Stage 3: LaTeX assembly
        tex_path = self._stage3_assembly(structure, written_sections)
        logger.info("Stage 3 complete: %s", tex_path)

        # Optional PDF compilation
        if self.compile_pdf_flag:
            compile_pdf(tex_path)

        return tex_path

    # ── Stage 1: Editorial Curation ──────────────────────────────────

    def _stage1_curation(
        self,
        syntheses: dict[str, dict],
        figure_metadata: dict[str, list[dict]],
        figure_analyses: dict[str, list[dict]],
        *,
        skip_existing: bool = True,
    ) -> dict:
        """Run Stage 1 or load from cache."""
        cache_path = self.pub_dir / "structure.json"
        if skip_existing and cache_path.exists():
            logger.info("Stage 1: loading cached structure.json")
            with open(cache_path) as f:
                result = json.load(f)
        else:
            logger.info("Stage 1: calling OpenAI for editorial curation...")
            user_prompt = build_curation_prompt(
                syntheses, figure_metadata, figure_analyses,
                n_highlights=self.n_highlights,
            )
            data = self.client.chat_json(
                system=build_curation_system(), user=user_prompt,
            )

            # Validate with Pydantic
            structure = ReportStructure(**data)
            result = structure.model_dump()

        # Fix diagnostic names — the LLM sometimes returns figure titles
        # instead of directory names (e.g. "2 m temperature annual mean
        # bias (ERA5)" instead of "global_biases").
        fixed = self._fix_diagnostic_names(result, figure_metadata)

        self._save_json(cache_path, result)
        if fixed:
            logger.info("Fixed %d diagnostic name(s) in curation output", fixed)

        return result

    @staticmethod
    def _fix_diagnostic_names(
        result: dict, figure_metadata: dict[str, list[dict]],
    ) -> int:
        """Resolve ``diagnostic`` fields to actual directory names.

        The LLM sometimes returns human-readable titles (e.g. "2 m
        temperature annual mean bias (ERA5)") instead of the directory
        key (e.g. "global_biases").  Build a figure_id → directory lookup
        from *figure_metadata* and correct any mismatches in-place.

        Also fixes the ``diagnostics`` list in each section by deriving
        it from the corrected ``selected_figures``.

        Returns the number of corrections made.
        """
        # Build figure_id -> diagnostic directory lookup
        fig_id_to_diag: dict[str, str] = {}
        for diag_name, metas in figure_metadata.items():
            for meta in metas:
                fid = meta.get("figure_id", "")
                if fid:
                    fig_id_to_diag[fid] = diag_name

        # Fix selected_figures
        n_fixed = 0
        for fig in result.get("selected_figures", []):
            fig_id = fig.get("figure_id", "")
            correct = fig_id_to_diag.get(fig_id)
            if correct and fig.get("diagnostic") != correct:
                logger.debug(
                    "Corrected diagnostic for %s: '%s' -> '%s'",
                    fig_id, fig["diagnostic"], correct,
                )
                fig["diagnostic"] = correct
                n_fixed += 1

        # Rebuild sections[].diagnostics from the corrected figures
        fig_diag_map = {
            f["figure_id"]: f["diagnostic"]
            for f in result.get("selected_figures", [])
        }
        for sec in result.get("sections", []):
            sec["diagnostics"] = list(dict.fromkeys(
                fig_diag_map[fid]
                for fid in sec.get("figure_ids", [])
                if fid in fig_diag_map
            ))

        return n_fixed

    # ── Stage 2: Section Writing ─────────────────────────────────────

    def _stage2_sections(
        self,
        structure: dict,
        figure_analyses: dict[str, list[dict]],
        syntheses: dict[str, dict],
        *,
        skip_existing: bool = True,
    ) -> list[dict]:
        """Run Stage 2 for each section, with per-section caching."""
        sections_dir = self.pub_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)

        # Build a lookup for selected figures
        fig_by_id = {
            fig["figure_id"]: fig
            for fig in structure["selected_figures"]
        }

        written = []
        for sec in structure["sections"]:
            sid = sec["section_id"]
            cache_path = sections_dir / f"{sid}.json"

            if skip_existing and cache_path.exists():
                logger.info("Stage 2: loading cached %s", sid)
                with open(cache_path) as f:
                    written.append(json.load(f))
                continue

            logger.info("Stage 2: writing section '%s'...", sec["title"])

            # Gather figures for this section
            sec_figures = [
                fig_by_id[fid] for fid in sec["figure_ids"]
                if fid in fig_by_id
            ]

            user_prompt = build_section_prompt(
                sec, sec_figures, figure_analyses, syntheses,
            )
            data = self.client.chat_json(
                system=build_section_system(), user=user_prompt,
            )

            # Validate
            ws = WrittenSection(**data)
            result = ws.model_dump()

            self._save_json(cache_path, result)
            written.append(result)

        return written

    # ── Stage 3: LaTeX Assembly ──────────────────────────────────────

    def _stage3_assembly(
        self,
        structure: dict,
        written_sections: list[dict],
    ) -> Path:
        """Copy figures and render the LaTeX document."""
        # Copy selected figures
        pub_figures_dir = self.pub_dir / "figures"
        figure_paths = copy_figures(
            structure["selected_figures"],
            self.figures_dir,
            pub_figures_dir,
        )

        # Render LaTeX
        tex_source = build_document(structure, written_sections, figure_paths)
        tex_path = self.pub_dir / "report.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        return tex_path

    # ── Data collection helpers ──────────────────────────────────────

    def _load_syntheses(self) -> dict[str, dict]:
        """Load all synthesis.json files from output/analysis/."""
        result = {}
        if not self.analysis_dir.exists():
            return result

        for synth_path in sorted(self.analysis_dir.glob("*/synthesis.json")):
            diag_name = synth_path.parent.name
            with open(synth_path) as f:
                result[diag_name] = json.load(f)

        return result

    def _load_figure_metadata(self) -> dict[str, list[dict]]:
        """Load all figure JSON sidecars from output/figures/."""
        result: dict[str, list[dict]] = {}
        if not self.figures_dir.exists():
            return result

        for diag_dir in sorted(self.figures_dir.iterdir()):
            if not diag_dir.is_dir():
                continue
            metas = []
            for json_path in sorted(diag_dir.glob("*.json")):
                with open(json_path) as f:
                    meta = json.load(f)
                meta["figure_id"] = json_path.stem
                metas.append(meta)
            if metas:
                result[diag_dir.name] = metas

        return result

    def _load_figure_analyses(self) -> dict[str, list[dict]]:
        """Load all per-figure analysis JSONs from output/analysis/."""
        result: dict[str, list[dict]] = {}
        if not self.analysis_dir.exists():
            return result

        for diag_dir in sorted(self.analysis_dir.iterdir()):
            if not diag_dir.is_dir():
                continue
            analyses = []
            for json_path in sorted(diag_dir.glob("*_analysis.json")):
                with open(json_path) as f:
                    data = json.load(f)
                fig_id = json_path.stem.replace("_analysis", "")
                data["figure_id"] = fig_id
                analyses.append(data)
            if analyses:
                result[diag_dir.name] = analyses

        return result

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        """Save dict as formatted JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
