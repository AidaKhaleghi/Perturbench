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

ADV_WEIGHTS = (0.0, 1.0, 10.0, 100.0, 300.0, 1000.0)
SEEDS = (0, 1)

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
for adv, seed in [(a, s) for a in ADV_WEIGHTS for s in SEEDS]:
    cfg = TrainConfig(
        lr=3e-4,
        max_epochs=100,
        patience=None,
        restore_best=False,
        checkpoint_dir=None,
        log_every=100,
    )
    model = CPA(
        cfg=ModelConfig(latent_dim=64, kl_weight=1e-3, adv_weight=adv),
        train_cfg=cfg,
        seed=seed,
    ).fit(train, val)

    h = pd.DataFrame(model.history_)
    adv_ratio = (
        float(h.train_adv_d.iloc[-1] / model.adv_prior_loss_) if adv > 0 else np.nan
    )
    rows.append(
        {
            "adv_weight": adv,
            "seed": seed,
            "mse_last10": float(h.val_de_mse.iloc[-10:].mean()),
            "slope": shift_slope(model),
            "val_recon": float(h.val_recon.iloc[-1]),
            "val_kl": float(h.val_kl.iloc[-1]),
            "adv_d_over_prior": adv_ratio,
            "mse_noise": float(h.val_de_mse.iloc[-10:].std()),
        }
    )
    pd.DataFrame(rows).to_csv("results/sweep_adv_raw.csv", index=False)
    print(f"done: adv={adv} seed={seed} mse_last10={rows[-1]['mse_last10']:.4f}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

raw = pd.DataFrame(rows)
summary = raw.groupby("adv_weight").agg(
    mse_last10=("mse_last10", "mean"),
    mse_std=("mse_last10", "std"),
    slope=("slope", "mean"),
    val_recon=("val_recon", "mean"),
    val_kl=("val_kl", "mean"),
    adv_d_over_prior=("adv_d_over_prior", "mean"),
    mse_noise=("mse_noise", "mean"),
)
print("\nReference validation MSE: zero shift 0.659, mean baseline 0.518, additive 0.052")
print("Collapsed CPA (kl 30, latent 64, lr 3e-4): mse_last10 0.0569, slope 0.735, val_recon about 78")
print(summary.round(4).to_string())
summary.to_csv("results/sweep_adv.csv")
