"""Single source of truth for figure style. Import this before plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# One colour per model, used in every figure. Okabe-Ito, colourblind safe.
MODEL_COLORS = {
    "mean": "#999999",
    "additive": "#E69F00",
    "cpa_no_adv": "#56B4E9",
    "cpa": "#009E73",
    "gears_like": "#CC79A7",
}

CONTINUOUS_CMAP = "viridis"


def set_style() -> None:
    """Apply the project-wide matplotlib settings."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "image.cmap": CONTINUOUS_CMAP,
        }
    )


def save(fig: plt.Figure, name: str, outdir: str | Path = "figures") -> None:
    """Save a figure as both PNG (for the README) and SVG (for reuse)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}")
