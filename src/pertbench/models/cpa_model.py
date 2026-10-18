import time
from pathlib import Path
import gc

import anndata as ad
import numpy as np
import pandas as pd
import torch

from pertbench.config import EvalConfig, ModelConfig, TrainConfig
from pertbench.data.split import make_split
from pertbench.data.tensors import PerturbationVocab, to_tensors
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import delta_metrics
from pertbench.models.base import PerturbationModel
from pertbench.models.cpa import CPAModule, cpa_loss


class CPA(PerturbationModel):
    """CPA-style conditional VAE with the shared PerturbationModel interface."""

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        train_cfg: TrainConfig | None = None,
        seed: int = 0,
        device: str | None = None,
        name: str = "cpa",
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

        self.module_: CPAModule | None = None
        self.vocab_: PerturbationVocab | None = None
        self.var_names_: np.ndarray | None = None
        self.train_ = None
        self.control_x_: torch.Tensor | None = None
        self.history_: list[dict] = []
        self.best_epoch_: int | None = None
        self._val: dict | None = None

    def fit(self, train_data: ad.AnnData, val_data: ad.AnnData | None = None) -> "CPA":
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

        self._val = self._prepare_validation(val_data) if val_data is not None else None
        self._train_loop()
        return self

    def _prepare_validation(self, val_data: ad.AnnData) -> dict:
        """Prepare validation tensors, true means, and top DE genes."""
        if not np.array_equal(np.asarray(val_data.var_names), self.var_names_):
            raise ValueError("Validation genes differ from training genes.")
        control_mean = (
            self.train_.x[self.train_.is_control].mean(dim=0).cpu().numpy()
        )
        true_means = mean_by_perturbation(val_data)
        return {
            "tensors": to_tensors(val_data, self.vocab_, self.device),
            "control_mean": control_mean,
            "true_means": true_means,
            "de": de_genes_for_perturbations(true_means, control_mean, EvalConfig()),
        }

    @torch.no_grad()
    def _val_loss(self) -> dict[str, float]:
        """Loss components on the validation cells, averaged over cells."""
        module = self.module_
        tensors = self._val["tensors"]
        module.eval()
        n = tensors.x.shape[0]
        sums = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        for start in range(0, n, 2048):
            x = tensors.x[start : start + 2048]
            p = tensors.pert[start : start + 2048]
            x_hat, mu, logvar = module(x, p)
            loss = cpa_loss(x_hat, x, mu, logvar, self.cfg.kl_weight)
            for key in sums:
                sums[key] += float(loss[key]) * x.shape[0]
        return {key: value / n for key, value in sums.items()}

    def val_de_mse(self) -> float:
        """Mean top-20 delta MSE over the validation perturbations."""
        if self._val is None:
            raise RuntimeError("No validation data was given to fit().")
        info = self._val
        scores = []
        for pert, idx in info["de"].items():
            control = info["control_mean"][idx]
            true_delta = info["true_means"].loc[pert].to_numpy()[idx] - control
            pred_delta = self.predict_mean(pert)[idx] - control
            scores.append(delta_metrics(pred_delta, true_delta)["mse"])
        return float(np.mean(scores))

    def _checkpoint_dir(self) -> Path | None:
        if self.train_cfg.checkpoint_dir is None:
            return None
        path = Path(self.train_cfg.checkpoint_dir) / f"{self.name}_seed{self.seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _train_loop(self) -> None:
        tc = self.train_cfg
        module = self.module_
        x_all, p_all = self.train_.x, self.train_.pert
        n = x_all.shape[0]

        optimizer = torch.optim.Adam(module.parameters(), lr=tc.lr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        ckpt_dir = self._checkpoint_dir()

        self.history_ = []
        best_score = float("inf")
        best_state = None
        bad_epochs = 0
        start_time = time.time()

        for epoch in range(1, tc.max_epochs + 1):
            module.train()
            order = torch.randperm(n, generator=generator, device=self.device)
            sums = {k: torch.zeros((), device=self.device) for k in ("total", "recon", "kl")}

            for start in range(0, n, tc.batch_size):
                idx = order[start : start + tc.batch_size]
                x, p = x_all[idx], p_all[idx]
                x_hat, mu, logvar = module(x, p)
                loss = cpa_loss(x_hat, x, mu, logvar, self.cfg.kl_weight)
                optimizer.zero_grad(set_to_none=True)
                loss["total"].backward()
                optimizer.step()
                for key in sums:
                    sums[key] += loss[key].detach() * len(idx)

            row = {"epoch": epoch}
            row.update({f"train_{k}": float(v) / n for k, v in sums.items()})

            improved = False
            if self._val is not None:
                row.update({f"val_{k}": v for k, v in self._val_loss().items()})
                row["val_de_mse"] = self.val_de_mse()
                if row["val_de_mse"] < best_score:
                    improved = True
                    best_score = row["val_de_mse"]
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in module.state_dict().items()
                    }
                    self.best_epoch_ = epoch
                    bad_epochs = 0
                    if ckpt_dir is not None:
                        torch.save({"model": best_state, "epoch": epoch}, ckpt_dir / "best.pt")
                else:
                    bad_epochs += 1
            row["is_best"] = improved
            self.history_.append(row)

            if ckpt_dir is not None:
                torch.save({"model": module.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
                pd.DataFrame(self.history_).to_csv(ckpt_dir / "history.csv", index=False)

            if epoch == 1 or epoch % tc.log_every == 0 or epoch == tc.max_epochs:
                message = f"epoch {epoch:4d} | train {row['train_total']:9.2f}"
                if self._val is not None:
                    message += (
                        f" | val {row['val_total']:9.2f}"
                        f" | val_de_mse {row['val_de_mse']:.4f}"
                    )
                print(f"{message} | {time.time() - start_time:6.0f} s", flush=True)

            if tc.patience is not None and self._val is not None and bad_epochs >= tc.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch_}.")
                break

        if best_state is not None and tc.restore_best:
            module.load_state_dict(best_state)
        elif best_state is None:
            self.best_epoch_ = len(self.history_)

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
    val = adata[adata.obs["perturbation"].isin(split.val)].copy()

    # Run 1: three epochs, no early stopping
    ckpt = "results/checkpoints_check"
    cfg = TrainConfig(max_epochs=3, patience=None, checkpoint_dir=ckpt)
    model = CPA(train_cfg=cfg, seed=0).fit(train, val)

    history = pd.DataFrame(model.history_)
    print()
    print(history.round(3).to_string(index=False))
    print("train recon decreased:", history.train_recon.iloc[-1] < history.train_recon.iloc[0])
    print("best epoch:", model.best_epoch_)
    print(
        "restored weights reproduce the best score:",
        np.isclose(model.val_de_mse(), history.val_de_mse.min(), atol=1e-6),
    )
    print("checkpoint files:", sorted(p.name for p in (Path(ckpt) / "cpa_seed0").iterdir()))
    del model
    gc.collect()

    # Run 2: learning rate 0, so the score cannot improve after epoch 1
    cfg = TrainConfig(lr=0.0, max_epochs=20, patience=2, checkpoint_dir=None)
    model = CPA(train_cfg=cfg, seed=0).fit(train, val)
    print("\nlr = 0: epochs run:", len(model.history_), "| best epoch:", model.best_epoch_)
