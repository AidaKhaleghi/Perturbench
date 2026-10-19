import gc
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from pertbench.config import EvalConfig
from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import evaluate_model
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline
from pertbench.models.variants import VARIANT_MODEL_CONFIGS, build_variant

SEED = 0

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()))
train = adata[adata.obs["perturbation"].isin(split.train)].copy()
val = adata[adata.obs["perturbation"].isin(split.val)].copy()

# Baseline validation scores, used as reference lines in plotthing
control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
true_means = mean_by_perturbation(val)
de = de_genes_for_perturbations(true_means, control_mean, EvalConfig())
refs = {}
for baseline in (MeanBaseline().fit(train), AdditiveBaseline().fit(train)):
    result = evaluate_model(baseline, true_means, control_mean, de, False)
    refs[baseline.name] = float(result.loc[result["metric"] == "mse", "value"].mean())
Path("results").mkdir(exist_ok=True)
pd.Series(refs).to_csv("results/val_references.csv", header=["val_mse"])
print("baseline validation MSE:", {k: round(v, 4) for k, v in refs.items()})

for name in VARIANT_MODEL_CONFIGS:
    print(f"\n=== {name} ===", flush=True)
    model = build_variant(name, seed=SEED).fit(train, val)
    h = pd.DataFrame(model.history_)
    print(
        f"{name}: mse_last10 {h.val_de_mse.iloc[-10:].mean():.4f} | "
        f"val_recon {h.val_recon.iloc[-1]:.1f} | val_kl {h.val_kl.iloc[-1]:.2f}"
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
