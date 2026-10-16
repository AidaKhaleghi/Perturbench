from pathlib import Path

import anndata as ad
import scanpy as sc
import numpy as np
from scipy import sparse

from pertbench.config import PreprocessConfig
from pertbench.data.download import RAW_DIR

PROCESSED_DIR = Path("data/processed")
CONTROL_LABEL = "control"


def perturbed_genes(adata: ad.AnnData) -> set[str]:
    """Return the set of all gene names that are perturbed in the dataset."""
    return {
        gene
        for pert in adata.obs["perturbation"].unique()
        if pert != CONTROL_LABEL
        for gene in pert.split("_")
    }


def drop_unmeasured_perturbations(adata: ad.AnnData) -> ad.AnnData:
    """Remove cells whose perturbed genes are not in adata.var_names."""
    measured = set(adata.var_names)
    labels = adata.obs["perturbation"].astype(str)
    unmeasured = {
        pert
        for pert in labels.unique()
        if pert != CONTROL_LABEL and not set(pert.split("_")) <= measured
    }
    keep = ~labels.isin(unmeasured)
    print(
        f"Dropping {len(unmeasured)} perturbations "
        f"({int((~keep).sum())} cells): {sorted(unmeasured)}"
    )
    adata = adata[keep.to_numpy()].copy()
    # Remove the dropped names from the list of allowed categories
    adata.obs["perturbation"] = (
        adata.obs["perturbation"].astype("category").cat.remove_unused_categories()
    )
    return adata


def preprocess(
    name: str,
    cfg: PreprocessConfig | None = None,
    force: bool = False,
) -> Path:
    """Normalize, log-transform, select HVGs(Highly Variable Genes), and save to data/processed/."""
    cfg = cfg or PreprocessConfig()

    target = PROCESSED_DIR / f"{name}.h5ad"
    if target.exists() and not force:
        print(f"{target} already exists, skipping preprocessing.")
        return target
    
    adata = ad.read_h5ad(RAW_DIR / f"{name}.h5ad")
    adata.var_names_make_unique()
    adata = drop_unmeasured_perturbations(adata)

    # Use a sparse float matrix. Normalization needs floats, not integers.
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)
    
    sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].to_numpy()
    pert_mask = adata.var_names.isin(perturbed_genes(adata))
    adata.var["is_perturbed_gene"] = pert_mask

    keep = hvg_mask | (pert_mask if cfg.keep_perturbed_genes else np.zeros_like(pert_mask))
    adata = adata[:, keep].copy()

    adata.obs["is_control"] = adata.obs["perturbation"] == CONTROL_LABEL
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(target, compression="gzip")
    print(f"Saved {adata.n_obs} cells x {adata.n_vars} genes to {target}.")
    return target


if __name__ == "__main__":
    target_path = preprocess("norman2019")

    a = ad.read_h5ad(target_path)
    print(a)
    print("first var names:", list(a.var_names[:5]))
    print("max of X:", a.X.max())
    print("max of raw counts:", a.layers["counts"].max())

    perts = [p for p in a.obs["perturbation"].unique() if p != "control"]
    genes = {g for p in perts for g in p.split("_")}
    print("perturbed genes:", len(genes))
    print("of those in the final gene set:", len(genes & set(a.var_names)))
    print("singles:", sum("_" not in p for p in perts))
    print("doubles:", sum("_" in p for p in perts))
    print("empty categories:", int((a.obs["perturbation"].value_counts() == 0).sum()))
