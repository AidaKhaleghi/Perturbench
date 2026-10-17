import numpy as np
import pandas as pd
from pathlib import Path
import anndata as ad

from pertbench.models.base import PerturbationModel
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline
from pertbench.config import EvalConfig
from pertbench.eval.de import de_genes_for_perturbations
from pertbench.data.utils import mean_by_perturbation
from pertbench.data.split import make_split

METRIC_NAMES = ("mse", "pearson", "direction_acc")


def delta_metrics(pred_delta: np.ndarray, true_delta: np.ndarray) -> dict[str, float]:
    """Compute all metrics for one perturbation."""
    mse = float(np.mean((pred_delta - true_delta) ** 2))
    if np.std(pred_delta) == 0 or np.std(true_delta) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred_delta, true_delta)[0, 1])
    direction = float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))
    return {"mse": mse, "pearson": pearson, "direction_acc": direction}


def evaluate_model(
    model: PerturbationModel,
    true_means: pd.DataFrame,
    control_mean: np.ndarray,
    de_genes: dict[str, np.ndarray],
    exclude_own_genes: bool,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate one fitted model on all test perturbations."""
    rows = []
    for pert, idx in de_genes.items():
        true_delta = true_means.loc[pert].to_numpy()[idx] - control_mean[idx]
        pred_delta = model.predict_mean(pert)[idx] - control_mean[idx]
        for metric, value in delta_metrics(pred_delta, true_delta).items():
            rows.append(
                {
                    "model": model.name,
                    "seed": seed,
                    "exclude_own_genes": exclude_own_genes,
                    "perturbation": pert,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RESULTS_PATH = Path("results/metrics.csv")


    class OracleModel(PerturbationModel):
        """Returns the true test means. Used only to test the metric code."""

        name = "oracle"

        def __init__(self, true_means: pd.DataFrame) -> None:
            self.true_means = true_means

        def fit(self, train_data: ad.AnnData) -> "OracleModel":
            return self

        def predict_mean(self, perturbation: str, n_cells: int = 100) -> np.ndarray:
            return self.true_means.loc[perturbation].to_numpy()

        def predict(self, perturbation: str, n_cells: int) -> np.ndarray:
            return np.tile(self.predict_mean(perturbation), (n_cells, 1))


    adata = ad.read_h5ad("data/processed/norman2019.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))

    train = adata[adata.obs["perturbation"].isin(split.train)].copy()
    test = adata[adata.obs["perturbation"].isin(split.test)]
    control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
    true_means = mean_by_perturbation(test)

    models = [MeanBaseline().fit(train), AdditiveBaseline().fit(train)]

    tables = []
    for exclude in (False, True):
        cfg = EvalConfig(exclude_perturbed_genes=exclude)
        de = de_genes_for_perturbations(true_means, control_mean, cfg)
        for model in models:
            tables.append(evaluate_model(model, true_means, control_mean, de, exclude))
        if not exclude:
            oracle = evaluate_model(
                OracleModel(true_means), true_means, control_mean, de, exclude
            )

    print("Oracle check, mean over perturbations (expect mse 0, pearson 1, direction 1):")
    print(oracle.groupby("metric")["value"].mean().round(3).to_string())

    metrics = pd.concat(tables, ignore_index=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {len(metrics)} rows to {RESULTS_PATH}")

    summary = (
        metrics.groupby(["exclude_own_genes", "model", "metric"])["value"]
        .mean()
        .unstack("metric")
        .round(3)
    )
    print("\nMean over test perturbations:")
    print(summary.to_string())
    print("\nNaN values:", int(metrics["value"].isna().sum()))
