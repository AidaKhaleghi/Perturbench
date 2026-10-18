import anndata as ad
import numpy as np
import pandas as pd

from pertbench.config import EvalConfig, TrainConfig
from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import evaluate_model
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline
from pertbench.models.cpa_model import CPA

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()))
train = adata[adata.obs["perturbation"].isin(split.train)].copy()
val = adata[adata.obs["perturbation"].isin(split.val)].copy()

cfg = TrainConfig(max_epochs=100, patience=None, log_every=5)
model = CPA(train_cfg=cfg, seed=0).fit(train, val)
print("\ndevice:", model.device)
print("best epoch:", model.best_epoch_)

history = pd.DataFrame(model.history_)
columns = ["epoch", "train_recon", "val_recon", "val_kl", "val_de_mse"]
print(history[columns].iloc[[0, 1, 2, 4, 9, 19, 29, 49, 74, 99]].round(3).to_string(index=False))
print("lowest val_de_mse:", round(history.val_de_mse.min(), 4))

# Validation comparison with the baselines
control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
true_means = mean_by_perturbation(val)
de = de_genes_for_perturbations(true_means, control_mean, EvalConfig())
models = [MeanBaseline().fit(train), AdditiveBaseline().fit(train), model]

tables = [evaluate_model(m, true_means, control_mean, de, False) for m in models]
summary = (
    pd.concat(tables)
    .groupby(["model", "metric"])["value"]
    .mean()
    .unstack("metric")
    .round(3)
)
print("\nValidation, mean over 13 perturbations:")
print(summary.to_string())


def shift_slope(m) -> float:
    """Mean of dot(pred, true) / dot(true, true) over validation perturbations."""
    slopes = []
    for pert, idx in de.items():
        c = control_mean[idx]
        true_delta = true_means.loc[pert].to_numpy()[idx] - c
        pred_delta = m.predict_mean(pert)[idx] - c
        slopes.append(pred_delta @ true_delta / (true_delta @ true_delta))
    return float(np.mean(slopes))


print("\nShift slope (1 = right size, 0 = no shift):")
for m in models:
    print(f"  {m.name}: {shift_slope(m):.3f}")

offset = model.predict_mean("control") - control_mean
print("\nCPA control reconstruction, mean squared error per gene:", round(float(np.mean(offset**2)), 5))
print("CPA control reconstruction, error on the 20 genes of the first val perturbation:",
      round(float(np.mean(offset[next(iter(de.values()))] ** 2)), 5))

history.to_csv("results/cpa_seed0_history.csv", index=False)
