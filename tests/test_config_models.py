"""Guard against shipping configs that reference a retired LLM model.

Providers retire model IDs (e.g. ``gemini-3-pro-preview`` started returning
404 NOT_FOUND), which breaks the ``analyze`` / ``report`` steps for every
config still pointing at them. This test fails loudly if any config under
``configs/`` references a known-retired model ID, so the swap is caught in
CI rather than at run time.
"""

from pathlib import Path

import pytest

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# Model IDs the provider has retired. Add to this set as models are sunset.
RETIRED_MODEL_IDS = frozenset({
    "gemini-3-pro-preview",
})

_CONFIG_FILES = sorted(_CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", _CONFIG_FILES, ids=lambda p: p.name)
def test_config_has_no_retired_model(config_path):
    text = config_path.read_text()
    offenders = [m for m in RETIRED_MODEL_IDS if m in text]
    assert not offenders, (
        f"{config_path.name} references retired model(s) {offenders}; "
        "update to a current model ID."
    )


def test_configs_dir_is_not_empty():
    """Sanity check that the parametrization actually scanned files."""
    assert _CONFIG_FILES, f"no YAML configs found under {_CONFIGS_DIR}"
