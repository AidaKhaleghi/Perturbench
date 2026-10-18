from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from pertbench.data.split import make_split

CONTROL_LABEL = "control"


class PerturbationVocab:
    """Maps perturbed gene names to positions in a multi-hot vector."""

    def __init__(self, genes: list[str]) -> None:
        self.genes = sorted(set(genes))
        self.index = {g: i for i, g in enumerate(self.genes)}

    def __len__(self) -> int:
        return len(self.genes)

    @classmethod
    def from_perturbations(cls, perturbations: list[str]) -> "PerturbationVocab":
        """Build the vocabulary from perturbation names such as "KLF1_BAK1"."""
        genes = {
            g
            for p in perturbations
            if p != CONTROL_LABEL
            for g in p.split("_")
        }
        return cls(sorted(genes))

    def encode(self, perturbation: str) -> np.ndarray:
        """Return a multi-hot vector. Control gives all zeros."""
        vec = np.zeros(len(self.genes), dtype=np.float32)
        if perturbation == CONTROL_LABEL:
            return vec
        for gene in perturbation.split("_"):
            if gene not in self.index:
                raise KeyError(f"Gene '{gene}' is not in the vocabulary.")
            vec[self.index[gene]] = 1.0
        return vec


@dataclass
class CellTensors:
    """Cells as tensors.

    x: expression, shape (n_cells, n_genes).
    pert: multi-hot perturbation vectors, shape (n_cells, n_vocab).
    is_control: boolean mask, shape (n_cells,).
    labels: perturbation name of each cell.
    """

    x: torch.Tensor
    pert: torch.Tensor
    is_control: torch.Tensor
    labels: np.ndarray


def to_tensors(
    adata: ad.AnnData,
    vocab: PerturbationVocab,
    device: str = "cpu",
) -> CellTensors:
    """Convert an AnnData object to dense tensors on the given device."""
    x = adata.X
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x_t = torch.from_numpy(x.astype(np.float32, copy=False))

    labels = adata.obs["perturbation"].astype(str).to_numpy()
    names, inverse = np.unique(labels, return_inverse=True)
    table = np.stack([vocab.encode(n) for n in names])
    pert_t = torch.from_numpy(table[inverse])

    is_control = torch.from_numpy(np.array(adata.obs["is_control"], dtype=bool))
    return CellTensors(
        x=x_t.to(device),
        pert=pert_t.to(device),
        is_control=is_control.to(device),
        labels=labels,
    )


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))

    train = adata[adata.obs["perturbation"].isin(split.train)].copy()
    vocab = PerturbationVocab.from_perturbations(split.train)
    print("vocabulary size:", len(vocab))

    t = to_tensors(train, vocab)
    size_mb = t.x.element_size() * t.x.nelement() / 1e6
    print("x:", tuple(t.x.shape), f"{size_mb:.0f} MB")
    print("pert:", tuple(t.pert.shape))

    row_sums = t.pert.sum(dim=1)
    for k in (0, 1, 2):
        print(f"cells with {k} perturbed genes:", int((row_sums == k).sum()))
    print("control cells:", int(t.is_control.sum()))
    print("zero rows match control:", bool(((row_sums == 0) == t.is_control).all()))

    # Held-out perturbations must be encodable with the training vocabulary
    for name in split.val + split.test:
        vocab.encode(name)
    print("val and test names encode without error: True")
    print("gpu available:", torch.cuda.is_available())
