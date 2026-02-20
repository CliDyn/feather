"""LaTeX document builder — assembles the final report from LLM outputs.

Handles:
- ``escape_latex()`` for sanitising all LLM-generated text
- Figure copying to publication directory
- Jinja2 template rendering
- Optional ``pdflatex`` compilation
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Directory containing the Jinja2 templates
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text.

    Handles the standard LaTeX specials (& % $ # _ { } ~ ^ \\) plus
    common Unicode characters that appear in LLM output.

    Figure references like ``Figure~\\ref{fig:label}`` are preserved.
    """
    # Preserve figure/table references before escaping
    ref_pattern = r'(?:Figures?~)?\\ref\{[^}]+\}'
    refs: list[str] = []

    def _save_ref(m: re.Match) -> str:
        refs.append(m.group(0))
        return f"XREFPLACEHOLDER{len(refs) - 1}XREFEND"

    text = re.sub(ref_pattern, _save_ref, text)

    # Escape backslashes first
    text = text.replace("\\", "\\textbackslash{}")

    # Standard LaTeX specials
    for old, new in [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]:
        text = text.replace(old, new)

    # Common Unicode -> LaTeX
    for char, replacement in {
        "\u00b0": "$^{\\circ}$",       # degree symbol
        "\u2013": "--",                  # en-dash
        "\u2014": "---",                 # em-dash
        "\u2018": "`",                   # left single quote
        "\u2019": "'",                   # right single quote
        "\u201c": "``",                  # left double quote
        "\u201d": "''",                  # right double quote
        "\u2026": "\\ldots{}",           # ellipsis
        "\u00b1": "$\\pm$",             # plus-minus
        "\u2264": "$\\leq$",            # less-than-or-equal
        "\u2265": "$\\geq$",            # greater-than-or-equal
        "\u00d7": "$\\times$",          # multiplication sign
        "\u2192": "$\\rightarrow$",     # right arrow
    }.items():
        text = text.replace(char, replacement)

    # Restore figure references (these are already valid LaTeX)
    for i, ref in enumerate(refs):
        text = text.replace(f"XREFPLACEHOLDER{i}XREFEND", ref)

    return text


def copy_figures(
    selected_figures: list[dict],
    figures_src_dir: Path,
    pub_figures_dir: Path,
) -> dict[str, Path]:
    """Copy selected PNG figures to the publication figures directory.

    Parameters
    ----------
    selected_figures : list of dict
        SelectedFigure dicts with 'diagnostic' and 'figure_id' keys.
    figures_src_dir : Path
        Source directory (e.g. ``output/figures/``).
    pub_figures_dir : Path
        Destination directory (e.g. ``output/publication/figures/``).

    Returns
    -------
    dict[str, Path]
        Mapping of figure_id -> relative path from publication dir.
    """
    pub_figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for fig in selected_figures:
        diag = fig["diagnostic"]
        fig_id = fig["figure_id"]
        src = figures_src_dir / diag / f"{fig_id}.png"

        if not src.exists():
            logger.warning("Figure not found: %s — skipping", src)
            continue

        dest_dir = pub_figures_dir / diag
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{fig_id}.png"
        shutil.copy2(src, dest)

        # Relative path from publication dir (for LaTeX \includegraphics)
        paths[fig_id] = Path("figures") / diag / f"{fig_id}.png"
        logger.info("Copied: %s", paths[fig_id])

    return paths


def build_document(
    structure: dict,
    written_sections: list[dict],
    figure_paths: dict[str, Path],
) -> str:
    """Render the LaTeX document from the Jinja2 template.

    Parameters
    ----------
    structure : dict
        ReportStructure dict from Stage 1.
    written_sections : list of dict
        WrittenSection dicts from Stage 2.
    figure_paths : dict
        figure_id -> relative Path for ``\\includegraphics``.

    Returns
    -------
    str
        Complete LaTeX document source.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    template = env.get_template("report.tex.jinja2")

    # Build figure lookup: figure_id -> {caption, label, path}
    fig_lookup = {}
    for fig in structure["selected_figures"]:
        fig_id = fig["figure_id"]
        if fig_id in figure_paths:
            fig_lookup[fig_id] = {
                "caption": escape_latex(fig["caption"]),
                "label": fig["label"],
                "path": str(figure_paths[fig_id]),
            }

    # Build section data with escaped text and associated figures.
    section_map = {ws["section_id"]: ws for ws in written_sections}
    sections_data = []
    for i, sec in enumerate(structure["sections"]):
        sid = sec["section_id"]
        written = section_map.get(sid)
        if written is None and i < len(written_sections):
            written = written_sections[i]
        written = written or {}
        sec_figures = [
            fig_lookup[fid] for fid in sec["figure_ids"]
            if fid in fig_lookup
        ]
        sections_data.append({
            "title": escape_latex(written.get("title", sec["title"])),
            "body": escape_latex(written.get("body", "")),
            "figures": sec_figures,
        })

    return template.render(
        title=escape_latex(structure["title"]),
        abstract=escape_latex(structure["abstract"]),
        introduction=escape_latex(structure["introduction"]),
        sections=sections_data,
        conclusion=escape_latex(structure["conclusion"]),
    )


def compile_pdf(tex_path: Path) -> bool:
    """Run ``pdflatex`` twice (for TOC) to compile the report.

    Returns True on success, False on failure (logged, not raised).
    """
    if not shutil.which("pdflatex"):
        logger.warning("pdflatex not found on PATH — skipping PDF compilation")
        return False

    pub_dir = tex_path.parent.resolve()
    tex_abs = tex_path.resolve()
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory", str(pub_dir),
        str(tex_abs),
    ]

    for run in (1, 2):
        logger.info("pdflatex pass %d/2...", run)
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(pub_dir),
            timeout=120, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            if run == 2:
                logger.warning("pdflatex returned non-zero on pass 2 "
                               "(may be warnings)")
            else:
                logger.info("pdflatex pass 1 had warnings (expected, "
                            "will resolve on pass 2)")

    pdf_path = tex_abs.with_suffix(".pdf")
    if pdf_path.exists():
        logger.info("PDF compiled: %s", pdf_path)
        return True

    logger.error("PDF not found after compilation: %s", pdf_path)
    return False
