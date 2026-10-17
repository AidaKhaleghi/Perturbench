import anndata as ad
import numpy as np
import pandas as pd

from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.models.base import PerturbationModel


CONTROL_LABEL = "control"


class MeanBaseline(PerturbationModel):
    """Predict the average expression of all perturbed training cells.

    The perturbation name is ignored. Control cells are excluded.
    The average is taken over cells, so perturbations with many cells
    have more weight than perturbations with few cells.
    """

    name = "mean"

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.var_names_: np.ndarray | None = None

    def fit(self, train_data: ad.AnnData) -> "MeanBaseline":
        """Compute the mean expression over all perturbed training cells."""
        is_control = train_data.obs["is_control"].to_numpy()
        if is_control.all():
            raise ValueError("Training data has no perturbed cells.")

        perturbed_x = train_data.X[~is_control]
        self.mean_ = np.asarray(perturbed_x.mean(axis=0)).ravel()
        self.var_names_ = np.asarray(train_data.var_names)
        self.n_cells_ = int((~is_control).sum())
        return self

    def _check_fitted(self) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.mean_

    def predict(self, perturbation: str, n_cells: int) -> np.ndarray:
        """Return the same mean profile for n_cells cells."""
        mean = self._check_fitted()
        return np.tile(mean, (n_cells, 1))

    def predict_mean(self, perturbation: str, n_cells: int = 100) -> np.ndarray:
        """Return the mean profile directly, without building a matrix."""
        return self._check_fitted().copy()


class AdditiveBaseline(PerturbationModel):
    """Predict control mean + shift of gene A + shift of gene B.

    Each shift is the mean of the cells with that single perturbation
    minus the control mean. Only single perturbations define shifts.
    """

    name = "additive"

    def __init__(self) -> None:
        self.control_mean_: np.ndarray | None = None
        self.deltas_: pd.DataFrame | None = None
        self.var_names_: np.ndarray | None = None

    def fit(self, train_data: ad.AnnData) -> "AdditiveBaseline":
        """Learn the control mean and one shift per single perturbation."""
        means = mean_by_perturbation(train_data)
        if CONTROL_LABEL not in means.index:
            raise ValueError("Training data has no control cells.")

        self.control_mean_ = means.loc[CONTROL_LABEL].to_numpy()
        single_names = [
            p for p in means.index if p != CONTROL_LABEL and "_" not in p
        ]
        if not single_names:
            raise ValueError("Training data has no single perturbations.")

        self.deltas_ = means.loc[single_names] - self.control_mean_
        self.var_names_ = np.asarray(train_data.var_names)
        return self

    def predict_mean(self, perturbation: str, n_cells: int = 100) -> np.ndarray:
        """Return control mean plus the summed shifts of the perturbed genes."""
        if self.control_mean_ is None or self.deltas_ is None:
            raise RuntimeError("Call fit() before predict().")
        if perturbation == CONTROL_LABEL:
            return self.control_mean_.copy()

        genes = perturbation.split("_")
        missing = [g for g in genes if g not in self.deltas_.index]
        if missing:
            raise KeyError(
                f"No single perturbation in training for genes: {missing}"
            )
        shift = self.deltas_.loc[genes].to_numpy().sum(axis=0)
        return self.control_mean_ + shift

    def predict(self, perturbation: str, n_cells: int) -> np.ndarray:
        """Return the same predicted profile for n_cells cells."""
        return np.tile(self.predict_mean(perturbation), (n_cells, 1))


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))
    train = adata[adata.obs["perturbation"].isin(split.train)].copy()

    # Test the mean baseline
    mean_model = MeanBaseline().fit(train)
    print("cells used for the mean:", mean_model.n_cells_)

    pred = mean_model.predict("BPGM_ZBTB1", n_cells=10)
    print("predict shape:", pred.shape)
    print("rows identical:", np.allclose(pred, pred[0]))
    print(
        "predict_mean matches predict:",
        np.allclose(mean_model.predict_mean("BPGM_ZBTB1"), pred.mean(axis=0)),
    )
    print(
        "same output for two perturbations:",
        np.allclose(mean_model.predict_mean("BPGM_ZBTB1"), mean_model.predict_mean("KLF1")),
    )

    # Delta of the mean baseline: the shift of its prediction from control
    control = train[train.obs["is_control"].to_numpy()]
    control_mean = np.asarray(control.X.mean(axis=0)).ravel()
    delta = mean_model.mean_ - control_mean
    top = np.argsort(-np.abs(delta))[:5]
    print("top 5 genes by absolute delta:")
    for i in top:
        print(f"  {mean_model.var_names_[i]}: {delta[i]:+.3f}")

    # Test the additive baseline
    additive = AdditiveBaseline().fit(train)
    print("single shifts learned:", additive.deltas_.shape)

    # Part 1: correctness checks
    name = split.test[0]
    gene_a, gene_b = name.split("_")
    manual = (
        additive.control_mean_
        + additive.deltas_.loc[gene_a].to_numpy()
        + additive.deltas_.loc[gene_b].to_numpy()
    )
    print(name, "matches manual sum:", np.allclose(additive.predict_mean(name), manual))
    print("predict shape:", additive.predict(name, 10).shape)
    print(
        "differs from mean baseline:",
        not np.allclose(additive.predict_mean(name), mean_model.predict_mean(name)),
    )
    try:
        additive.predict_mean("NOTAGENE_KLF1")
        print("unseen gene raises KeyError: False")
    except KeyError:
        print("unseen gene raises KeyError: True")

    # Part 2: rough comparison on the test set, all genes, delta from control
    test = adata[adata.obs["perturbation"].isin(split.test)]
    true_means = mean_by_perturbation(test)
    control_mean = additive.control_mean_

    r_additive, r_mean = [], []
    for pert in true_means.index:
        true_delta = true_means.loc[pert].to_numpy() - control_mean
        pred_add = additive.predict_mean(pert) - control_mean
        pred_mean = mean_model.predict_mean(pert) - control_mean
        r_additive.append(np.corrcoef(pred_add, true_delta)[0, 1])
        r_mean.append(np.corrcoef(pred_mean, true_delta)[0, 1])

    print(f"test perturbations: {len(r_additive)}")
    print(f"mean Pearson on delta, additive: {np.mean(r_additive):.3f}")
    print(f"mean Pearson on delta, mean baseline: {np.mean(r_mean):.3f}")
