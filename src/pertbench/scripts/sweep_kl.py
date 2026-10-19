import gc

import anndata as ad
import numpy as np
import pandas as pd
import torch

from pertbench.config import EvalConfig, ModelConfig, TrainConfig
from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.models.cpa_model import CPA

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()))
train = adata[adata.obs["perturbation"].isin(split.train)].copy()
val = adata[adata.obs["perturbation"].isin(split.val)].copy()

control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
true_means = mean_by_perturbation(val)
de = de_genes_for_perturbations(true_means, control_mean, EvalConfig())


def shift_slope(model) -> float:
    """Mean of dot(pred, true) / dot(true, true) over validation perturbations."""
    slopes = []
    for pert, idx in de.items():
        c = control_mean[idx]
        true_delta = true_means.loc[pert].to_numpy()[idx] - c
        pred_delta = model.predict_mean(pert)[idx] - c
        slopes.append(pred_delta @ true_delta / (true_delta @ true_delta))
    return float(np.mean(slopes))


rows = []
for kl in (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0):
    cfg = TrainConfig(
        max_epochs=50,
        patience=None,
        restore_best=False,
        checkpoint_dir=None,
        log_every=50,
    )
    model = CPA(
        cfg=ModelConfig(kl_weight=kl), train_cfg=cfg, seed=0, name=f"cpa_kl{kl}"
    ).fit(train, val)
    h = pd.DataFrame(model.history_)
    rows.append(
        {
            "kl_weight": kl,
            "train_recon": h.train_recon.iloc[-1],
            "val_recon": h.val_recon.iloc[-1],
            "val_kl": h.val_kl.iloc[-1],
            "val_de_mse_final": h.val_de_mse.iloc[-1],
            "val_de_mse_min": h.val_de_mse.min(),
            "best_epoch": model.best_epoch_,
            "slope_final": shift_slope(model),
        }
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

table = pd.DataFrame(rows)
print("\nReference: zero shift 0.659, mean baseline 0.518, additive 0.052 (validation MSE)")
print("Reference slopes: mean 0.164, additive 0.939")
print(table.round(4).to_string(index=False))
table.to_csv("results/sweep_kl.csv", index=False)
