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
from pertbench.models.base import PerturbationModel
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline
from pertbench.models.variants import VARIANT_MODEL_CONFIGS, build_variant

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RUNS_DIR = RESULTS_DIR / "runs"
SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEED = 0
BASELINES = ("mean", "additive")


def _fit(name: str, seed: int, train: ad.AnnData, val: ad.AnnData) -> PerturbationModel:
    """Fit one model. Only train and val cells are used here, never test cells."""
    if name == "mean":
        return MeanBaseline().fit(train)
    if name == "additive":
        return AdditiveBaseline().fit(train)
    return build_variant(name, seed=seed).fit(train, val)


def collect_results() -> pd.DataFrame:
    """Combine all per-run files into results/benchmark.csv and return the table."""
    files = sorted(RUNS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No run files in {RUNS_DIR}.")
    metrics = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    return metrics


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean over perturbations per seed, then mean and std over seeds."""
    per_seed = (
        metrics.groupby(["dataset", "exclude_own_genes", "model", "seed", "metric"])["value"]
        .mean()
        .reset_index()
    )
    summary = (
        per_seed.groupby(["dataset", "exclude_own_genes", "model", "metric"])["value"]
        .agg(["mean", "std"])
        .unstack("metric")
    )
    return summary.swaplevel(axis=1).sort_index(axis=1)


def run_benchmark(dataset: str, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """Fit every model on the training perturbations and score it on the test set.

    Each run is saved to its own file. A restarted benchmark skips finished runs.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(PROCESSED_DIR / f"{dataset}.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()), seed=SPLIT_SEED)
    train = adata[adata.obs["perturbation"].isin(split.train)].copy()
    val = adata[adata.obs["perturbation"].isin(split.val)].copy()
    test = adata[adata.obs["perturbation"].isin(split.test)]
    print(f"train {train.n_obs} cells | val {val.n_obs} | test {test.n_obs}")

    control_mean = mean_by_perturbation(train).loc["control"].to_numpy()
    true_means = mean_by_perturbation(test)
    de_by_flag = {
        exclude: de_genes_for_perturbations(
            true_means, control_mean, EvalConfig(exclude_perturbed_genes=exclude)
        )
        for exclude in (False, True)
    }

    jobs = [(name, 0) for name in BASELINES]
    jobs += [(name, seed) for name in VARIANT_MODEL_CONFIGS for seed in seeds]

    for name, seed in jobs:
        path = RUNS_DIR / f"{dataset}_{name}_seed{seed}.csv"
        if path.exists():
            print(f"skip {path.name}: already done", flush=True)
            continue

        model = _fit(name, seed, train, val)
        tables = [
            evaluate_model(model, true_means, control_mean, de, exclude, seed=seed)
            for exclude, de in de_by_flag.items()
        ]
        table = pd.concat(tables, ignore_index=True)
        table.insert(0, "dataset", dataset)
        table.to_csv(path, index=False)

        mask = (table["metric"] == "mse") & (~table["exclude_own_genes"])
        print(f"done {name} seed {seed}: test mse {table.loc[mask, 'value'].mean():.4f}", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return collect_results()


if __name__ == "__main__":
    results = run_benchmark("norman2019")
    summary = summarize(results)
    print("\nTest set. Mean over perturbations, then mean and std over seeds.")
    for exclude, part in summary.groupby(level="exclude_own_genes"):
        print(f"\nexclude_own_genes = {exclude}")
        print(part.droplevel(["dataset", "exclude_own_genes"]).round(3).to_string())


    # Check benchmark
    metrics = pd.read_csv("results/benchmark.csv")
    keys = ["dataset", "model", "seed", "exclude_own_genes", "perturbation", "metric"]

    print("rows:", len(metrics), "| expected: 5304")
    print("duplicate rows:", int(metrics.duplicated(subset=keys).sum()))
    print("NaN values:", int(metrics["value"].isna().sum()))
    print("models:", sorted(metrics["model"].unique()))
    print("seeds per model:", metrics.groupby("model")["seed"].nunique().to_dict())
    print("test perturbations:", metrics["perturbation"].nunique())

    baselines = metrics[
        (~metrics["exclude_own_genes"]) & metrics["model"].isin(["mean", "additive"])
    ]
    print(baselines.groupby(["model", "metric"])["value"].mean().unstack("metric").round(3))
    print("Week 2: additive mse 0.045 pearson 0.960 direction 0.996")
    print("Week 2: mean     mse 0.289 pearson 0.538 direction 0.862")
