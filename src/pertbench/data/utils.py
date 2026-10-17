import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from pertbench.data.split import make_split


def mean_by_perturbation(adata: ad.AnnData) -> pd.DataFrame:
    """Return the mean expression of each perturbation."""
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    names, inverse = np.unique(labels, return_inverse=True)
    n_cells = len(labels)

    # Sparse indicator matrix: entry (i, j) is 1 if cell j has perturbation i
    indicator = sparse.csr_matrix(
        (np.ones(n_cells), (inverse, np.arange(n_cells))),
        shape=(len(names), n_cells),
    )
    sums = indicator @ adata.X
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    counts = np.bincount(inverse, minlength=len(names))
    return pd.DataFrame(sums / counts[:, None], index=names, columns=adata.var_names)


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))

    train = adata[adata.obs["perturbation"].isin(split.train)].copy()
    means = mean_by_perturbation(train)
    print("means shape:", means.shape)

    # The control row must equal a direct mean over the control cells
    control = train[train.obs["is_control"].to_numpy()]
    direct = np.asarray(control.X.mean(axis=0)).ravel()
    print("control row matches direct mean:", np.allclose(means.loc["control"], direct, atol=1e-5))
    print("test perturbations in means:", means.index.isin(split.test).sum())
