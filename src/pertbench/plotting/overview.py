import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D

from pertbench.plotting.style import CATEGORY_COLORS, apply_style, save_figure

TYPE_NAMES = {0: "control", 1: "single", 2: "double"}


def perturbation_type(name: str) -> str:
    """Return control, single, or double for a perturbation name."""
    if name == "control":
        return "control"
    return "double" if "_" in name else "single"


def plot_overview(
    adata: ad.AnnData,
    n_umap_cells: int = 20_000,
    n_labels: int = 5,
    seed: int = 0,
) -> plt.Figure:
    """Two-panel overview: cells per perturbation and a UMAP by type."""
    apply_style()
    fig, (ax_bar, ax_umap) = plt.subplots(1, 2, figsize=(14, 7))

    # Left panel: sorted horizontal bars, largest at the top
    # Control is excluded because it is much larger than all others
    all_counts = adata.obs["perturbation"].astype(str).value_counts()
    n_control = int(all_counts.get("control", 0))
    counts = all_counts.drop("control").sort_values()
    n_bars = len(counts)

    colors = [CATEGORY_COLORS[perturbation_type(p)] for p in counts.index]
    ax_bar.barh(np.arange(n_bars), counts.values, color=colors, height=0.85)
    ax_bar.set_yticks([])
    x_max = float(counts.max())
    ax_bar.set_xlim(0, x_max * 1.9)
    ax_bar.set_xlabel("Cells per perturbation (number of cells)")
    ax_bar.set_ylabel(f"Perturbations (n = {n_bars}, sorted)")
    ax_bar.set_title(f"Cell counts (control: {n_control:,} cells, not shown)")

    # Label the largest bars. Labels are spread apart so they do not overlap
    spacing = 10  # distance between labels, in bar positions
    for k in range(n_labels):
        i = n_bars - 1 - k
        ax_bar.annotate(
            counts.index[i],
            xy=(counts.values[i], i),
            xytext=(x_max * 1.15, n_bars - 1 - k * spacing),
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8},
        )

    # Right panel: UMAP of a random subsample
    rng = np.random.default_rng(seed)
    n = min(n_umap_cells, adata.n_obs)
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    sub = adata[idx].copy()
    sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=50, random_state=seed)
    sc.pp.neighbors(sub, n_neighbors=15, random_state=seed)
    sc.tl.umap(sub, random_state=seed)

    types = sub.obs["nperts"].astype(int).map(TYPE_NAMES).to_numpy()
    coords = sub.obsm["X_umap"]
    # Draw doubles first and control last, so control stays visible
    for kind in ("double", "single", "control"):
        mask = types == kind
        ax_umap.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2,
            alpha=0.5,
            color=CATEGORY_COLORS[kind],
            rasterized=True,
        )
    ax_umap.set_xlabel("UMAP 1 (arbitrary units)")
    ax_umap.set_ylabel("UMAP 2 (arbitrary units)")
    ax_umap.set_title(f"UMAP of {n:,} random cells")
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=c, label=k)
        for k, c in CATEGORY_COLORS.items()
    ]
    ax_umap.legend(handles=handles, loc="best", frameon=False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    fig = plot_overview(adata)
    save_figure(fig, "plot1_dataset_overview")
