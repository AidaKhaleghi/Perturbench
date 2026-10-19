from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pertbench.analysis import line_with_band
from pertbench.plotting.style import MODEL_COLORS, apply_style

PLOT5_MODELS = ("mean", "additive", "cpa", "cpa_small_kl", "cpa_adv")
CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]


def _panel(ax, table: pd.DataFrame, xcol: str, ycol: str, models) -> None:
    """Scatter plus trend line with band for each model."""
    x_all = table[xcol].to_numpy()
    grid = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 50)
    for model in models:
        part = table[table["model"] == model].dropna(subset=[xcol, ycol])
        x, y = part[xcol].to_numpy(), part[ycol].to_numpy()
        color = MODEL_COLORS[model]
        ax.scatter(x, y, s=14, color=color, alpha=0.35, linewidths=0)
        line, low, high = line_with_band(x, y, grid)
        ax.plot(grid, line, color=color, linewidth=2)
        ax.fill_between(grid, low, high, color=color, alpha=0.15, linewidth=0)


def plot_generalization(table: pd.DataFrame, models=PLOT5_MODELS) -> plt.Figure:
    """Three panels of generalization metrics vs similarity or interaction strength."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _panel(axes[0], table, "sim_novel", "pearson", models)
    axes[0].set_xlabel("Max cosine similarity to training deltas (unitless)")
    axes[0].set_ylabel("Pearson correlation of delta (unitless)")
    axes[0].set_title("a  Correlation vs similarity")

    _panel(axes[1], table, "sim_novel", "mse", models)
    axes[1].set_xlabel("Max cosine similarity to training deltas (unitless)")
    axes[1].set_ylabel("MSE on top-20 genes (log1p units squared)")
    axes[1].set_title("b  Error vs similarity")

    _panel(axes[2], table, "interaction", "mse", models)
    axes[2].set_xlabel("Interaction strength, ||true - additive|| / ||true|| (unitless)")
    axes[2].set_ylabel("MSE on top-20 genes (log1p units squared)")
    axes[2].set_title("c  Error vs genetic interaction")

    handles = [Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2, label=m) for m in models]
    axes[0].legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_gene_level(panels: list[dict]) -> plt.Figure:
    """One panel per perturbation."""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        true = panel["true"]
        lim = 1.1 * max(np.abs(true).max(), max(np.abs(p).max() for p in panel["preds"].values()))
        ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1, linestyle="--")
        ax.axhline(0, color="#CCCCCC", linewidth=0.8)
        ax.axvline(0, color="#CCCCCC", linewidth=0.8)
        for model, pred in panel["preds"].items():
            ax.scatter(true, pred, s=26, color=MODEL_COLORS[model], alpha=0.85,
                       linewidths=0, label=model)
        main = panel["label_model"]
        err = np.abs(panel["preds"][main] - true)
        for i in np.argsort(err)[::-1][: panel["label_n"]]:
            ax.annotate(panel["genes"][i], (true[i], panel["preds"][main][i]),
                        xytext=(6, 6), textcoords="offset points", fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True delta from control (log1p units)")
        ax.set_ylabel("Predicted delta from control (log1p units)")
        ax.set_title(panel["title"])
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_embeddings(
    coords: np.ndarray,
    names: list[str],
    cluster: np.ndarray,
    color_value: np.ndarray,
    color_label: str,
    label_names: list[str],
) -> plt.Figure:
    """UMAP of perturbation embeddings in two panels."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for k in np.unique(cluster):
        sel = cluster == k
        axes[0].scatter(coords[sel, 0], coords[sel, 1], s=45,
                        color=CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)],
                        label=f"cluster {int(k) + 1}", linewidths=0)
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title("a  Colored by similarity of expression deltas")
    sc_ = axes[1].scatter(coords[:, 0], coords[:, 1], s=45, c=color_value,
                          cmap="viridis", linewidths=0)
    fig.colorbar(sc_, ax=axes[1], label=color_label)
    axes[1].set_title("b  Colored by one expression shift")
    for ax in axes:
        for i, n in enumerate(names):
            if n in label_names:
                ax.annotate(n, coords[i], xytext=(5, 5), textcoords="offset points", fontsize=11)
        ax.set_xlabel("UMAP 1 (arbitrary units)")
        ax.set_ylabel("UMAP 2 (arbitrary units)")
    fig.tight_layout()
    return fig
