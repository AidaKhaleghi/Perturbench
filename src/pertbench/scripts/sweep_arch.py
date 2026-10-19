import gc
import itertools

import anndata as ad
import numpy as np
import pandas as pd
import torch

from pertbench.config import EvalConfig, ModelConfig, TrainConfig
from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import evaluate_model
from pertbench.models.cpa_model import CPA

KL_WEIGHT = 30.0
LATENT_DIMS = (16, 64, 256)
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
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
for latent, lr, seed in itertools.product(LATENT_DIMS, LEARNING_RATES, SEEDS):
    cfg = TrainConfig(
        lr=lr,
        max_epochs=100,
        patience=None,
        restore_best=False,
        checkpoint_dir=None,
        log_every=100,
    )
    model = CPA(
        cfg=ModelConfig(latent_dim=latent, kl_weight=KL_WEIGHT),
        train_cfg=cfg,
        seed=seed,
        name="cpa",
    ).fit(train, val)

    h = pd.DataFrame(model.history_)
    result = evaluate_model(model, true_means, control_mean, de, False, seed=seed)
    m = result.groupby("metric")["value"].mean()
    rows.append(
        {
            "latent_dim": latent,
            "lr": lr,
            "seed": seed,
            "mse_last10": float(h.val_de_mse.iloc[-10:].mean()),
            "mse_final": float(m["mse"]),
            "pearson": float(m["pearson"]),
            "slope": shift_slope(model),
            "val_kl": float(h.val_kl.iloc[-1]),
        }
    )
    pd.DataFrame(rows).to_csv("results/sweep_arch_raw.csv", index=False)
    print(
        f"done: latent={latent} lr={lr} seed={seed} "
        f"mse_last10={rows[-1]['mse_last10']:.4f}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

raw = pd.DataFrame(rows)
summary = (
    raw.groupby(["latent_dim", "lr"])
    .agg(
        mse_last10=("mse_last10", "mean"),
        mse_last10_std=("mse_last10", "std"),
        mse_final=("mse_final", "mean"),
        pearson=("pearson", "mean"),
        slope=("slope", "mean"),
        val_kl=("val_kl", "mean"),
    )
    .sort_values("mse_last10")
)
print("\nReference validation MSE: zero shift 0.659, mean baseline 0.518, additive 0.052")
print(summary.round(4).to_string())
summary.to_csv("results/sweep_arch.csv")
