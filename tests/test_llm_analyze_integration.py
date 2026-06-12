"""Integration test for the `analyze` step against the real Gemini API.

Unlike ``tests/test_llm.py`` (fully mocked), this test exercises the
``FigureAnalyzer`` end-to-end with a *live* Vertex AI / Gemini call, using
the settings declared in ``configs/eerie_all_members_indices.yaml``.

It is restricted to the four climate-extremes diagnostics:
``heatwave``, ``heatwave_change``, ``tropical_nights`` and
``tropical_nights_change``.

Running it
----------
The test is marked ``integration`` and is skipped automatically unless the
``VERTEX_API_KEY`` environment variable is set (so the key never lives in
the repository)::

    export VERTEX_API_KEY=<your-gemini-key>
    conda activate feather
    pytest tests/test_llm_analyze_integration.py -v -m integration

The real config's ``output_dir`` is overridden to a pytest ``tmp_path`` so
the live run never writes into the production output tree.
"""

import json
import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from feather.config import FeatherConfig  # noqa: E402
from feather.llm.analyzer import FigureAnalyzer  # noqa: E402
from feather.llm.schemas import DiagnosticSynthesis, FigureAnalysis  # noqa: E402

pytestmark = pytest.mark.integration

# Repo root → configs/eerie_all_members_indices.yaml — the config that wires
# the tropical-nights / heatwave indices diagnostics (and their `_change`
# variants via its project.climate_change block).
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eerie_all_members_indices.yaml"
)

# The four climate-extremes diagnostics under test (registry ``name`` values).
DIAGNOSTICS = [
    "heatwave",
    "heatwave_change",
    "tropical_nights",
    "tropical_nights_change",
]

# Per-diagnostic figure metadata mirroring what each diagnostic actually
# emits (figure_id / variables / plot_type / group), so the live prompt is
# representative of a real run.
_FIGURE_SPECS = {
    "heatwave": {
        "figure_id": "heatwave_0_map",
        "title": "Heatwave Days (HWFI) Climatology",
        "variables": ["tasmax"],
        "plot_type": "map",
        "description": (
            "Mean annual count of heatwave days per model over 1980-2014, "
            "with the ERA5-derived reference for comparison."
        ),
    },
    "heatwave_change": {
        "figure_id": "heatwave_change_map",
        "title": "Heatwave Days Projected Change",
        "variables": ["tasmax"],
        "plot_type": "map",
        "description": (
            "Change in annual heatwave days between the historical baseline "
            "and the future scenario for each model."
        ),
    },
    "tropical_nights": {
        "figure_id": "tropical_nights_climatology",
        "title": "Tropical Nights (TR) Climatology",
        "variables": ["tasmin"],
        "plot_type": "map",
        "description": (
            "Mean annual count of tropical nights (Tmin > 20 degC) per model "
            "over 1980-2014 versus the observational reference."
        ),
    },
    "tropical_nights_change": {
        "figure_id": "tropical_nights_change_map",
        "title": "Tropical Nights Projected Change",
        "variables": ["tasmin"],
        "plot_type": "map",
        "description": (
            "Change in annual tropical-night count between the historical "
            "baseline and the future scenario for each model."
        ),
    },
}


def _config_or_skip(tmp_path: Path) -> FeatherConfig:
    """Load the real EERIE config with output_dir redirected to tmp_path."""
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
    # Give transient provider errors room to recover so the live test is not
    # flaky, while staying reasonably fast.
    fa["max_retries"] = 4
    fa["retry_delay"] = 5
    return config


def _write_figure(figures_root: Path, diag_name: str, models: list[str]) -> None:
    """Render a genuine PNG + JSON sidecar for one diagnostic."""
    spec = _FIGURE_SPECS[diag_name]
    diag_dir = figures_root / diag_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    # A real (if simple) figure so Gemini receives genuine image content.
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title(spec["title"])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.imshow(
        [[i + j for j in range(8)] for i in range(4)],
        cmap="YlOrRd",
        aspect="auto",
        extent=(-180, 180, -90, 90),
    )
    png_path = diag_dir / f"{spec['figure_id']}.png"
    fig.savefig(png_path, dpi=60)
    plt.close(fig)

    meta = {
        "diagnostic_name": diag_name,
        "title": spec["title"],
        "figure_id": spec["figure_id"],
        "variables_used": spec["variables"],
        "models": models,
        "obs_dataset": "ERA5",
        "obs_variable": spec["variables"][0],
        "units": "days/year",
        "period": ["1980", "2014"],
        "description": spec["description"],
        "domain": "sfc",
        "group": "extremes",
        "spatial_extent": "global",
        "plot_type": spec["plot_type"],
        "colormap": "YlOrRd",
    }
    (diag_dir / f"{spec['figure_id']}.json").write_text(json.dumps(meta))


def test_analyze_step_live_on_extremes_diagnostics(tmp_path):
    """End-to-end analyze step (real Gemini) on the four extremes diagnostics.

    Asserts that the live run produces one schema-valid figure analysis and
    one synthesis per diagnostic, written to ``{output_dir}/analysis/``.
    """
    config = _config_or_skip(tmp_path)

    figures_root = tmp_path / "figures"
    for diag in DIAGNOSTICS:
        _write_figure(figures_root, diag, config.models)

    analyzer = FigureAnalyzer(config)  # key picked up from VERTEX_API_KEY
    result = analyzer.run(skip_existing=False, diagnostics=DIAGNOSTICS)

    # A live LLM is non-deterministic and the provider can return transient
    # errors that ``run()`` swallows per-figure. Require a strong majority to
    # succeed rather than an exact count, but validate every artifact that was
    # produced. ``syntheses`` is produced only for diagnostics that yielded an
    # analysis, so the two counts move together.
    assert result["figure_analyses"] >= len(DIAGNOSTICS) - 1, (
        f"only {result['figure_analyses']}/{len(DIAGNOSTICS)} figures analysed"
    )
    assert result["syntheses"] == result["figure_analyses"]

    # Only the requested diagnostics may be analysed (filter honoured).
    analysis_root = tmp_path / "analysis"
    produced = {p.name for p in analysis_root.iterdir()} if analysis_root.exists() else set()
    assert produced.issubset(set(DIAGNOSTICS)), f"unexpected diagnostics: {produced}"

    # Every produced analysis/synthesis must validate against the schemas.
    for diag in produced:
        spec = _FIGURE_SPECS[diag]
        analysis_file = analysis_root / diag / f"{spec['figure_id']}_analysis.json"
        synthesis_file = analysis_root / diag / "synthesis.json"
        assert analysis_file.exists(), f"missing analysis for {diag}"
        assert synthesis_file.exists(), f"missing synthesis for {diag}"
        FigureAnalysis(**json.loads(analysis_file.read_text()))
        DiagnosticSynthesis(**json.loads(synthesis_file.read_text()))


def test_analyze_step_skips_other_diagnostics(tmp_path):
    """The diagnostics filter must not analyse figures outside the list."""
    config = _config_or_skip(tmp_path)

    figures_root = tmp_path / "figures"
    # One in-scope diagnostic and one out-of-scope diagnostic.
    _write_figure(figures_root, "tropical_nights", config.models)
    other = figures_root / "global_biases"
    other.mkdir(parents=True, exist_ok=True)
    (other / "tas_annual_bias.png").write_bytes(
        (figures_root / "tropical_nights"
         / f"{_FIGURE_SPECS['tropical_nights']['figure_id']}.png").read_bytes()
    )
    (other / "tas_annual_bias.json").write_text(
        json.dumps({"diagnostic_name": "global_biases", "group": "evaluation"})
    )

    analyzer = FigureAnalyzer(config)
    result = analyzer.run(skip_existing=False, diagnostics=["tropical_nights"])

    assert result["figure_analyses"] == 1
    assert result["syntheses"] == 1
    assert (tmp_path / "analysis" / "tropical_nights" / "synthesis.json").exists()
    assert not (tmp_path / "analysis" / "global_biases").exists()
