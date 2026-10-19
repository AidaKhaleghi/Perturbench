import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pertbench.plotting.style import MODEL_COLORS, apply_style

MODEL_ORDER = ("mean", "additive", "cpa_small_kl", "cpa_adv", "cpa")

# metric name, title, direction hint, y-axis label with units
PANELS = (
    ("mse", "Delta MSE", "lower is better", "Delta MSE (log-expr.²)"),
    ("pearson", "Pearson correlation", "higher is better", "Correlation (unitless)"),
    ("direction_acc", "Direction accuracy", "higher is better", "Fraction of genes (0 to 1)"),
    ("slope", "Shift slope", "1 = correct size", "Slope (unitless)"),
)


def per_seed_means(
    metrics: pd.DataFrame,
    dataset: str,
    exclude_own_genes: bool = False,
) -> pd.DataFrame:
    """Mean over test perturbations for each model, seed, and metric."""
    part = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["exclude_own_genes"] == exclude_own_genes)
    ]
    return part.groupby(["model", "seed", "metric"])["value"].mean().reset_index()


def plot_main_results(
    metrics: pd.DataFrame,
    dataset: str,
    exclude_own_genes: bool = False,
) -> plt.Figure:
    """Bar plot of mean metric values per model, with error bars for std over seeds."""
    apply_style()
    seeds = per_seed_means(metrics, dataset, exclude_own_genes)
    models = [m for m in MODEL_ORDER if m in set(seeds["model"])]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 5.5))
    for ax, (metric, title, hint, ylabel) in zip(axes, PANELS):
        part = seeds[seeds["metric"] == metric]
        groups = [part.loc[part["model"] == m, "value"].to_numpy() for m in models]
        means = np.array([g.mean() for g in groups])

        ax.bar(x, means, color=[MODEL_COLORS[m] for m in models], width=0.7)
        for i, g in enumerate(groups):
            if len(g) < 2:
                continue
            ax.errorbar(
                i, g.mean(), yerr=g.std(ddof=1),
                fmt="none", ecolor="black", elinewidth=1.5, capsize=4, zorder=4,
            )
            offsets = np.linspace(-0.2, 0.2, len(g))
            ax.scatter(
                i + offsets, g, s=18, color="black",
                edgecolor="white", linewidth=0.5, alpha=0.8, zorder=5,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        if metric == "slope":
            ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n({hint})")

    fig.tight_layout()
    return fig
