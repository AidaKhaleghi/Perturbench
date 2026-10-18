import anndata as ad
import numpy as np

from pertbench.config import EvalConfig
from pertbench.data.split import make_split
from pertbench.data.utils import mean_by_perturbation
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.eval.metrics import delta_metrics
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()))
train = adata[adata.obs["perturbation"].isin(split.train)].copy()
val = adata[adata.obs["perturbation"].isin(split.val)]

control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
true_means = mean_by_perturbation(val)
de = de_genes_for_perturbations(true_means, control_mean, EvalConfig())


def val_mse(predict_mean) -> float:
    """Mean top-20 delta MSE over the validation perturbations."""
    scores = []
    for pert, idx in de.items():
        true_delta = true_means.loc[pert].to_numpy()[idx] - control_mean[idx]
        pred_delta = predict_mean(pert)[idx] - control_mean[idx]
        scores.append(delta_metrics(pred_delta, true_delta)["mse"])
    return float(np.mean(scores))


mean_model = MeanBaseline().fit(train)
additive = AdditiveBaseline().fit(train)
print("validation perturbations:", len(de))
print(f"zero shift (control mean): {val_mse(lambda p: control_mean):.4f}")
print(f"mean baseline:             {val_mse(mean_model.predict_mean):.4f}")
print(f"additive baseline:         {val_mse(additive.predict_mean):.4f}")
