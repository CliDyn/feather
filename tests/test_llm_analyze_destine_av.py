"""Integration test for the `analyze` step on the DestinE Added-Value config.

Exercises :class:`feather.llm.analyzer.FigureAnalyzer` end-to-end with a
*live* Vertex AI / Gemini call across **every registered diagnostic**, using
the settings declared in ``configs/destine_added_value.yaml``.

The set of diagnostics is discovered dynamically from the registry
(:func:`feather.diag.registry.list_diagnostics`), so new diagnostics are
covered automatically without editing this test.

Running it
----------
Marked ``integration`` and skipped automatically unless ``VERTEX_API_KEY``
is set (the key never lives in the repo)::

    export VERTEX_API_KEY=<your-gemini-key>
    conda activate feather
    pytest tests/test_llm_analyze_destine_av.py -v -m integration

Note: this makes ~2 live calls per registered diagnostic (one figure
analysis + one synthesis), so the full run takes a few minutes. The real
config's ``output_dir`` is overridden to a pytest ``tmp_path`` so the live
run never writes into the production output tree.
"""

import json
import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import feather.diag  # noqa: F401  (registers diagnostics via @register)  # noqa: E402
from feather.config import FeatherConfig  # noqa: E402
from feather.diag.registry import list_diagnostics  # noqa: E402
from feather.llm.analyzer import FigureAnalyzer  # noqa: E402
from feather.llm.schemas import DiagnosticSynthesis, FigureAnalysis  # noqa: E402

pytestmark = pytest.mark.integration

# Repo root → configs/destine_added_value.yaml
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "destine_added_value.yaml"
)


def _config_or_skip(tmp_path: Path) -> FeatherConfig:
    """Load the real DestinE AV config with output_dir redirected to tmp."""
    if not _CONFIG_PATH.exists():
        pytest.skip(f"config not found: {_CONFIG_PATH}")
    if not os.getenv("VERTEX_API_KEY"):
        pytest.skip("VERTEX_API_KEY not set — skipping live Gemini analyze test")

    config = FeatherConfig.from_yaml(str(_CONFIG_PATH))
    # Never write into the production output tree.
    config.output_dir = str(tmp_path)

    # The config's declared model can drift / be retired by the provider
    # (e.g. ``gemini-3-pro-preview`` returns 404 "no longer available").
    # Pin the test to a known-good current model so it validates the
    # pipeline rather than the config's model lifecycle. Override with
    # FEATHER_TEST_GEMINI_MODEL if needed.
    fa = config.llm.setdefault("figure_analysis", {})
    fa["model"] = os.getenv("FEATHER_TEST_GEMINI_MODEL", "gemini-2.5-flash")
    fa["max_retries"] = 4
    fa["retry_delay"] = 5
    # Keep this a *small* test: the figures are synthetic placeholders, so
    # reasoning budget buys nothing — disable it to keep each of the ~2 calls
    # per diagnostic fast and cheap.
    fa["thinking_budget"] = 0
    return config


def _write_figure(figures_root: Path, info: dict, models: list[str],
                  period: tuple[str, str]) -> None:
    """Render a generic PNG + JSON sidecar for one diagnostic."""
    name = info["name"]
    variables = list(info.get("variables") or [])
    var0 = variables[0] if variables else "tas"
    figure_id = f"{name}_fig"

    diag_dir = figures_root / name
    diag_dir.mkdir(parents=True, exist_ok=True)

    # A real (if simple) figure so Gemini receives genuine image content.
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title(info["title"])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.imshow(
        [[i + j for j in range(8)] for i in range(4)],
        cmap="RdBu_r",
        aspect="auto",
        extent=(-180, 180, -90, 90),
    )
    fig.savefig(diag_dir / f"{figure_id}.png", dpi=60)
    plt.close(fig)

    meta = {
        "diagnostic_name": name,
        "title": info["title"],
        "figure_id": figure_id,
        "variables_used": variables or [var0],
        "models": models,
        "obs_dataset": "ERA5",
        "obs_variable": var0,
        "units": "",
        "period": list(period),
        "description": (
            f"Synthetic placeholder figure for the {info['title']} "
            f"diagnostic, used to exercise the analyze step."
        ),
        "domain": info["domain"],
        "group": info["group"],
        "spatial_extent": "global",
        "plot_type": "map",
        "colormap": "RdBu_r",
    }
    (diag_dir / f"{figure_id}.json").write_text(json.dumps(meta))


def test_analyze_step_live_on_all_diagnostics(tmp_path):
    """End-to-end analyze step (real Gemini) across every registered diagnostic.

    Lays down one figure per registered diagnostic, runs the analyze step
    with no diagnostics filter (so all are processed), and validates that
    each produced analysis/synthesis conforms to the Pydantic schemas.
    """
    config = _config_or_skip(tmp_path)

    diagnostics = list_diagnostics()
    assert diagnostics, "no diagnostics registered"
    all_names = {d["name"] for d in diagnostics}
    period = config.get_period()

    figures_root = tmp_path / "figures"
    for info in diagnostics:
        _write_figure(figures_root, info, config.models, period)

    analyzer = FigureAnalyzer(config)  # key picked up from VERTEX_API_KEY
    result = analyzer.run(skip_existing=False)  # no filter → all diagnostics

    n = len(diagnostics)
    # A live LLM is non-deterministic and the provider can return transient
    # errors that ``run()`` swallows per-figure. Require a strong majority to
    # succeed rather than an exact count, but validate every produced artifact.
    assert result["figure_analyses"] >= n - 1, (
        f"only {result['figure_analyses']}/{n} figures analysed"
    )
    assert result["syntheses"] == result["figure_analyses"]

    analysis_root = tmp_path / "analysis"
    produced = {p.name for p in analysis_root.iterdir()} if analysis_root.exists() else set()
    # Only registered diagnostics may appear, and (modulo at most one transient
    # failure) essentially all of them should.
    assert produced.issubset(all_names), f"unexpected diagnostics: {produced - all_names}"
    assert len(produced) >= n - 1

    # Every produced analysis/synthesis must validate against the schemas.
    for name in produced:
        analysis_file = analysis_root / name / f"{name}_fig_analysis.json"
        synthesis_file = analysis_root / name / "synthesis.json"
        assert analysis_file.exists(), f"missing analysis for {name}"
        assert synthesis_file.exists(), f"missing synthesis for {name}"
        FigureAnalysis(**json.loads(analysis_file.read_text()))
        DiagnosticSynthesis(**json.loads(synthesis_file.read_text()))
