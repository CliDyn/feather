"""Shared matplotlib defaults and model colors."""

import matplotlib.pyplot as plt

MODEL_COLORS = {
    "ifs-fesom": "#1f77b4",
    "ifs-nemo": "#ff7f0e",
    "icon": "#2ca02c",
}

OBS_COLOR = "black"
CMIP6_COLOR = "#888888"
ENS_COLOR = "#2c3e50"   # EERIE ensemble mean/median lines (dark slate)

# Fallback palette for benchmark MMM lines (CMIP6, HighResMIP, …) when a
# benchmark config does not set an explicit colour.  Index 0 keeps the
# historic CMIP6 grey.
BENCHMARK_FALLBACK_COLORS = [CMIP6_COLOR, "#9467bd", "#8c564b", "#17becf"]


def benchmark_color(index: int) -> str:
    """Return a fallback colour for the *index*-th benchmark."""
    return BENCHMARK_FALLBACK_COLORS[index % len(BENCHMARK_FALLBACK_COLORS)]


def apply_style():
    """Set shared matplotlib defaults."""
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })
