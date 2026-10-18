import anndata as ad
import numpy as np
import torch

from pertbench.config import EvalConfig, ModelConfig
from pertbench.data.split import make_split
from pertbench.data.tensors import PerturbationVocab, to_tensors
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import evaluate_model
from pertbench.models.base import PerturbationModel
from pertbench.models.cpa import CPAModule


class CPA(PerturbationModel):
    """CPA-style conditional VAE with the shared PerturbationModel interface."""

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        seed: int = 0,
        device: str | None = None,
        name: str = "cpa",
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

        self.module_: CPAModule | None = None
        self.vocab_: PerturbationVocab | None = None
        self.var_names_: np.ndarray | None = None
        self.train_ = None
        self.control_x_: torch.Tensor | None = None

    def fit(self, train_data: ad.AnnData) -> "CPA":
        """Prepare data and build the network."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        names = list(train_data.obs["perturbation"].astype(str).unique())
        self.vocab_ = PerturbationVocab.from_perturbations(names)
        self.var_names_ = np.asarray(train_data.var_names)
        self.train_ = to_tensors(train_data, self.vocab_, self.device)

        # Reference control cells, shuffled once with the model seed
        control_idx = torch.nonzero(self.train_.is_control).squeeze(1).cpu().numpy()
        if len(control_idx) == 0:
            raise ValueError("Training data has no control cells.")
        order = torch.from_numpy(rng.permutation(control_idx)).to(self.device)
        self.control_x_ = self.train_.x[order]

        self.module_ = CPAModule(
            n_genes=train_data.n_vars,
            n_perts=len(self.vocab_),
            latent_dim=self.cfg.latent_dim,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        # Step 4 adds the training loop here.
        return self

    def _require_fitted(self) -> CPAModule:
        if self.module_ is None or self.vocab_ is None or self.control_x_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.module_

    def _decode(self, perturbation: str, n_cells: int) -> torch.Tensor:
        module = self._require_fitted()
        if n_cells < 1:
            raise ValueError("n_cells must be at least 1.")

        # Use the fixed control cells
        n_available = self.control_x_.shape[0]
        idx = torch.arange(n_cells, device=self.device) % n_available
        x_control = self.control_x_[idx]

        pert = torch.from_numpy(self.vocab_.encode(perturbation)).to(self.device)
        pert = pert.unsqueeze(0).expand(n_cells, -1)
        return module.predict_from_control(x_control, pert)

    def predict(self, perturbation: str, n_cells: int) -> np.ndarray:
        """Predict n_cells cells under the perturbation, shape (n_cells, n_genes)."""
        return self._decode(perturbation, n_cells).cpu().numpy()

    def predict_mean(self, perturbation: str, n_cells: int = 1000) -> np.ndarray:
        """Predict the mean expression over n_cells decoded cells."""
        return self._decode(perturbation, n_cells).mean(dim=0).cpu().numpy()


if __name__ == "__main__":
    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))
    train = adata[adata.obs["perturbation"].isin(split.train)].copy()
    test = adata[adata.obs["perturbation"].isin(split.test)]

    model = CPA(seed=0).fit(train)
    print("device:", model.device)
    print("reference control cells:", model.control_x_.shape[0])

    name = split.test[0]
    pred = model.predict(name, 10)
    print("predict shape:", pred.shape)
    print("predict_mean shape:", model.predict_mean(name).shape)
    print("deterministic:", np.allclose(model.predict_mean(name), model.predict_mean(name)))
    print(
        "predict_mean matches predict:",
        np.allclose(model.predict_mean(name, 10), pred.mean(axis=0), atol=1e-6),
    )
    print(
        "perturbations differ:",
        not np.allclose(model.predict_mean(split.test[0]), model.predict_mean(split.test[1])),
    )
    try:
        model.predict_mean("NOTAGENE_KLF1")
        print("unseen gene raises KeyError: False")
    except KeyError:
        print("unseen gene raises KeyError: True")

    control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
    true_means = mean_by_perturbation(test)
    de = de_genes_for_perturbations(true_means, control_mean, EvalConfig())
    result = evaluate_model(model, true_means, control_mean, de, exclude_own_genes=False)

    print("\nUntrained model, mean over 39 test perturbations:")
    print(result.groupby("metric")["value"].mean().round(3).to_string())
    print("NaN values:", int(result["value"].isna().sum()))
