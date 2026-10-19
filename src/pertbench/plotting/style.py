from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures")

# Perturbation types (Plot 1)
CATEGORY_COLORS = {
    "control": "#444444",
    "single": "#56B4E9",
    "double": "#D55E00",
}

# One color per model, used in every later figure
MODEL_COLORS = {
    "mean": "#999999",
    "additive": "#0072B2",
    "cpa": "#D55E00",
    "cpa_small_kl": "#E69F00",
    "cpa_adv": "#CC79A7",
    "gears": "#009E73",
}


def apply_style() -> None:
    """Set global matplotlib settings shared by all figures."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG (300 DPI) and SVG in the figures folder."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.svg")
    print(f"Saved {FIGURES_DIR / name}.png and .svg")
