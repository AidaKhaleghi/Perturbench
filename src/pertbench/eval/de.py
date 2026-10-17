import numpy as np
import pandas as pd
import anndata as ad

from pertbench.config import EvalConfig
from pertbench.data.utils import mean_by_perturbation
from pertbench.data.split import make_split

CONTROL_LABEL = "control"


def top_de_genes(
    true_mean: np.ndarray,
    control_mean: np.ndarray,
    n: int = 20,
    exclude_idx: list[int] | None = None,
) -> np.ndarray:
    """Return column indices of the n genes with the largest absolute delta."""
    score = np.abs(true_mean - control_mean).astype(float)
    if exclude_idx:
        score[exclude_idx] = -np.inf
    return np.argsort(-score, kind="stable")[:n]


def de_genes_for_perturbations(
    true_means: pd.DataFrame,
    control_mean: np.ndarray,
    cfg: EvalConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute the top DE gene indices for every perturbation."""
    cfg = cfg or EvalConfig()
    column_index = {gene: i for i, gene in enumerate(true_means.columns)}

    result: dict[str, np.ndarray] = {}
    for pert in true_means.index:
        if pert == CONTROL_LABEL:
            continue
        exclude = None
        if cfg.exclude_perturbed_genes:
            exclude = [column_index[g] for g in pert.split("_") if g in column_index]
        result[pert] = top_de_genes(
            true_means.loc[pert].to_numpy(),
            control_mean,
            n=cfg.n_de_genes,
            exclude_idx=exclude,
        )
    return result


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))

    train = adata[adata.obs["perturbation"].isin(split.train)]
    control_mean = mean_by_perturbation(train).loc["control"].to_numpy()

    test = adata[adata.obs["perturbation"].isin(split.test)]
    true_means = mean_by_perturbation(test)
    var_names = np.asarray(adata.var_names)

    for exclude in (False, True):
        cfg = EvalConfig(exclude_perturbed_genes=exclude)
        de = de_genes_for_perturbations(true_means, control_mean, cfg)
        print(f"\n=== exclude_perturbed_genes = {exclude} ===")
        print("perturbations:", len(de), "| genes per list:", {len(v) for v in de.values()})

        own_counts = [
            sum(var_names[i] in pert.split("_") for i in idx) for pert, idx in de.items()
        ]
        print("mean number of own target genes in top 20:", round(float(np.mean(own_counts)), 2))
        print("perturbations with at least one own gene:", int(np.sum(np.array(own_counts) > 0)))
        print("distinct genes across all lists:", len(set(np.concatenate(list(de.values())))))

        for pert in split.test[:3]:
            idx = de[pert]
            delta = true_means.loc[pert].to_numpy() - control_mean
            print(pert, "->", ", ".join(f"{var_names[i]} ({delta[i]:+.2f})" for i in idx[:6]), "...")
