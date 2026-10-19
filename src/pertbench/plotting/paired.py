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
LABEL_OFFSETS = ((12, 10), (12, -14), (-12, 10), (-12, -14), (14, 0))


def paired_table(
    metrics: pd.DataFrame,
    dataset: str,
    model: str,
    metric: str,
    baseline: str = "additive",
    exclude_own_genes: bool = False,
) -> pd.DataFrame:
    """Return a table of baseline vs model values for each perturbation."""
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


def most_extreme(table: pd.DataFrame, n: int = 5, log: bool = False) -> pd.DataFrame:
    """Return the n rows farthest from the diagonal.

    The distance is the absolute difference, or the absolute log ratio if log is True.
    """
    if log:
        gap = np.log(table["model"] / table["baseline"]).abs()
    else:
        gap = (table["model"] - table["baseline"]).abs()
    return table.loc[gap.sort_values(ascending=False).index[:n]]


def plot_paired(
    metrics: pd.DataFrame,
    dataset: str,
    model: str,
    baseline: str = "additive",
    n_labels: int = 5,
    exclude_own_genes: bool = False,
) -> plt.Figure:
    """Scatter plot of baseline vs model values for each perturbation."""
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

        ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.2, zorder=1)
        ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        upper, lower = (model, baseline) if higher_better else (baseline, model)
        ax.text(0.03, 0.97, f"{upper} better", transform=ax.transAxes,
                va="top", ha="left", color="0.3")
        ax.text(0.97, 0.03, f"{lower} better", transform=ax.transAxes,
                va="bottom", ha="right", color="0.3")

        for k, (_, row) in enumerate(most_extreme(t, n_labels, log).iterrows()):
            dx, dy = LABEL_OFFSETS[k % len(LABEL_OFFSETS)]
            ax.annotate(
                row["perturbation"],
                (row["baseline"], row["model"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8},
            )

        ax.set_xlabel(f"{baseline}: {label}")
        ax.set_ylabel(f"{model}: {label}")
        ax.set_title(f"{TITLES[metric]}\none point per test perturbation (n = {len(t)})")

    fig.tight_layout()
    return fig
