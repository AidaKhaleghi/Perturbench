import anndata as ad
import numpy as np
import pandas as pd
import torch

from pertbench.config import TrainConfig
from pertbench.data.split import make_split
from pertbench.models.cpa_model import CPA

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()))
train = adata[adata.obs["perturbation"].isin(split.train)]

# 10 control cells and 40 perturbed cells, chosen with a fixed seed
rng = np.random.default_rng(0)
is_control = train.obs["is_control"].to_numpy(dtype=bool)
idx = np.concatenate(
    [
        rng.choice(np.flatnonzero(is_control), size=10, replace=False),
        rng.choice(np.flatnonzero(~is_control), size=40, replace=False),
    ]
)
small = train[np.sort(idx)].copy()
print("cells:", small.n_obs, "| perturbations:", small.obs["perturbation"].nunique())

cfg = TrainConfig(
    lr=1e-3,
    batch_size=50,
    max_epochs=2000,
    patience=None,
    checkpoint_dir=None,
    log_every=250,
)
model = CPA(train_cfg=cfg, seed=0).fit(small)
print("device:", model.device)

history = pd.DataFrame(model.history_)
first, last = history.train_recon.iloc[0], history.train_recon.iloc[-1]
print(f"\ntrain recon (training mode): {first:.2f} -> {last:.4f}")
print(f"final value is {100 * last / first:.3f}% of the start")

# Evaluation mode: no sampling noise
module = model.module_
module.eval()
with torch.no_grad():
    x, p = model.train_.x, model.train_.pert
    x_hat, _, _ = module(x, p)
    err = ((x_hat - x) ** 2).sum(dim=1)
ctrl = model.train_.is_control
print(f"eval-mode recon, all cells:       {float(err.mean()):.4f}")
print(f"eval-mode recon, control cells:   {float(err[ctrl].mean()):.4f}")
print(f"eval-mode recon, perturbed cells: {float(err[~ctrl].mean()):.4f}")
print("pass (eval-mode recon below 1% of start):", float(err.mean()) < 0.01 * first)
