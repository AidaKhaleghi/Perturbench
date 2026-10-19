import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pertbench.plotting.style import MODEL_COLORS, apply_style

# metric, axis label with units, log scale, higher is better
PANELS = (
    ("pearson", "Pearson correlation (unitless)", False, True),
    ("mse", "Delta MSE (log-expr.², log scale)", True, False),
)
TITLES = {
    "pearson": "Pearson correlation (higher is better)",
    "mse": "Delta MSE (lower is better)",
}


def paired_table(
    metrics: pd.DataFrame,
    dataset: str,
    model: str,
    metric: str,
    baseline: str = "additive",
    exclude_own_genes: bool = False,
) -> pd.DataFrame:
    """Return a table with the baseline and model values for each perturbation."""
    part = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["exclude_own_genes"] == exclude_own_genes)
        & (metrics["metric"] == metric)
    ]
    base = part[part["model"] == baseline].groupby("perturbation")["value"].mean()
    mod = part[part["model"] == model].groupby("perturbation")["value"].mean()
    table = pd.DataFrame({"baseline": base, "model": mod}).dropna()
    table.index.name = "perturbation"
    return table.reset_index()


def most_extreme(table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the n rows with the largest absolute difference from the baseline."""
    gap = (table["model"] - table["baseline"]).abs()
    return table.loc[gap.sort_values(ascending=False).index[:n]]


def label_offset(fx: float, fy: float, k: int) -> tuple[int, int, str]:
    """Return an offset for a label, given its position in the axis range."""
    dx = 14 if fx < 0.6 else -14
    if fy < 0.12:
        dy = 18
    elif fy > 0.88:
        dy = -18
    else:
        dy = 14 if k % 2 == 0 else -14
    return dx, dy, "left" if dx > 0 else "right"


def plot_paired(
    metrics: pd.DataFrame,
    dataset: str,
    model: str,
    baseline: str = "additive",
    n_labels: int = 5,
    exclude_own_genes: bool = False,
) -> plt.Figure:
    """Return a figure with two panels, one for each metric."""
    apply_style()
    color = MODEL_COLORS[model]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))

    for ax, (metric, label, log, higher_better) in zip(axes, PANELS):
        t = paired_table(metrics, dataset, model, metric, baseline, exclude_own_genes)
        x, y = t["baseline"].to_numpy(), t["model"].to_numpy()

        values = np.concatenate([x, y])
        lo, hi = values.min(), values.max()
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = lo / 1.25, hi * 1.25
        else:
            pad = 0.04 * (hi - lo)
            lo, hi = lo - pad, hi + pad

        def fraction(value: float) -> float:
            """Position of a value inside the axis range, from 0 to 1."""
            if log:
                return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
            return (value - lo) / (hi - lo)

        ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.2, zorder=1)
        ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        for k, (_, row) in enumerate(most_extreme(t, n_labels).iterrows()):
            dx, dy, ha = label_offset(
                fraction(row["baseline"]), fraction(row["model"]), k
            )
            ax.annotate(
                row["perturbation"],
                (row["baseline"], row["model"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8},
                zorder=6,
            )

        upper = model if higher_better else baseline
        ax.set_xlabel(f"{baseline}: {label}")
        ax.set_ylabel(f"{model}: {label}")
        ax.set_title(
            f"{TITLES[metric]}\n{len(t)} test perturbations. "
            f"Above the dashed line: {upper} is better"
        )

    fig.tight_layout()
    return fig
