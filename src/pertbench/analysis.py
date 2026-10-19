from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

if TYPE_CHECKING:
    import anndata as ad

    from pertbench.data.split import Split

CONTROL_LABEL = "control"
CPA_VARIANTS = ("cpa_small_kl", "cpa_adv", "cpa")
SEEDS = (0, 1, 2, 3, 4)
PREDICTIONS_DIR = Path("results/predictions")


@dataclass
class Context:
    split: Split
    var_names: np.ndarray
    train: ad.AnnData
    train_means: pd.DataFrame
    test_means: pd.DataFrame
    control_mean: np.ndarray
    de: dict[str, np.ndarray]


def load_context(dataset: str = "norman2019") -> Context:
    """Load the processed data and compute the split, means, and top DE genes."""
    import anndata as ad

    from pertbench.config import EvalConfig
    from pertbench.data.split import make_split
    from pertbench.data.utils import mean_by_perturbation
    from pertbench.eval.de import de_genes_for_perturbations

    adata = ad.read_h5ad(f"data/processed/{dataset}.h5ad")
    split = make_split(list(adata.obs["perturbation"].unique()))
    labels = adata.obs["perturbation"].astype(str)
    train = adata[labels.isin(split.train).to_numpy()].copy()
    test = adata[labels.isin(split.test).to_numpy()]

    train_means = mean_by_perturbation(train)
    test_means = mean_by_perturbation(test)
    control_mean = train_means.loc[CONTROL_LABEL].to_numpy()
    de = de_genes_for_perturbations(test_means, control_mean, EvalConfig())
    return Context(
        split=split,
        var_names=np.asarray(adata.var_names),
        train=train,
        train_means=train_means,
        test_means=test_means,
        control_mean=control_mean,
        de=de,
    )


def load_predictions(
    models: tuple[str, ...],
    seeds: tuple[int, ...] = SEEDS,
    directory: Path = PREDICTIONS_DIR,
) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    """Load saved test predictions."""
    perts: list[str] | None = None
    out: dict[str, list[np.ndarray]] = {}
    for model in models:
        arrays = []
        for seed in seeds:
            path = directory / f"{model}_seed{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. Run scripts/export_predictions.py first."
                )
            data = np.load(path, allow_pickle=False)
            names = [str(p) for p in data["perts"]]
            if perts is None:
                perts = names
            elif names != perts:
                raise ValueError(f"{path} has a different perturbation order.")
            arrays.append(data["preds"])
        out[model] = arrays
    return perts or [], out


def cosine_to_rows(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between vec and every row of matrix."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (matrix @ vec) / norms
    cos[norms == 0] = np.nan
    return cos


def residual_norm_ratio(true_delta: np.ndarray, add_delta: np.ndarray) -> float:
    """Size of the non-additive part relative to the size of the true delta."""
    denom = float(np.linalg.norm(true_delta))
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(true_delta - add_delta) / denom)


def residual_correlation(
    pred_delta: np.ndarray, add_delta: np.ndarray, true_delta: np.ndarray
) -> float:
    """Correlation between the predicted and the true deviation from additive."""
    a = pred_delta - add_delta
    b = true_delta - add_delta
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_shrink(true_deltas: list[np.ndarray], add_deltas: list[np.ndarray]) -> float:
    """Least-squares factor lam so that lam * additive best matches the true delta."""
    num = sum(float(t @ a) for t, a in zip(true_deltas, add_deltas))
    den = sum(float(a @ a) for a in add_deltas)
    return num / den


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval. NaN values are dropped."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    low, high = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return float(v.mean()), float(low), float(high)


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Spearman correlation with a 95% bootstrap interval over pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        rhos[i] = spearmanr(x[idx], y[idx])[0]
    low, high = np.nanpercentile(rhos, [2.5, 97.5])
    return rho, float(low), float(high)


def line_with_band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear fit of y on x with a 95% bootstrap band."""
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    preds = np.full((n_boot, len(grid)), np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) == 0:
            continue
        s, b = np.polyfit(x[idx], y[idx], 1)
        preds[i] = s * grid + b
    low, high = np.nanpercentile(preds, [2.5, 97.5], axis=0)
    return slope * grid + intercept, low, high


def _cosine_matrix(m: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between the rows of m."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    unit = m / np.maximum(norms, 1e-12)
    return unit @ unit.T


def pair_similarity_correlation(
    emb: np.ndarray, delta: np.ndarray, n_perm: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Do genes with similar embeddings have similar single-perturbation deltas?"""
    n = emb.shape[0]
    iu = np.triu_indices(n, k=1)
    rank_delta = rankdata(_cosine_matrix(delta)[iu])
    sim_emb = _cosine_matrix(emb)

    def corr(sim_vec: np.ndarray) -> float:
        return float(np.corrcoef(rankdata(sim_vec), rank_delta)[0, 1])

    rho = corr(sim_emb[iu])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = corr(sim_emb[np.ix_(p, p)][iu])
    p_value = (1 + int(np.sum(null >= rho))) / (1 + n_perm)
    return rho, float(p_value)


def top_gene_metrics(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    control_mean: np.ndarray,
    idx: np.ndarray,
) -> tuple[float, float]:
    """MSE and Pearson correlation of the delta from control on the genes idx."""
    pred = (pred_mean - control_mean)[idx]
    true = (true_mean - control_mean)[idx]
    mse = float(np.mean((pred - true) ** 2))
    pearson = float("nan") if np.std(pred) == 0 else float(np.corrcoef(pred, true)[0, 1])
    return mse, pearson


def single_delta_matrix(ctx: Context) -> tuple[list[str], np.ndarray]:
    """Mean delta from control of each training single perturbation."""
    genes = [p for p in ctx.train_means.index if p != CONTROL_LABEL and "_" not in p]
    delta = ctx.train_means.loc[genes].to_numpy() - ctx.control_mean
    return genes, delta


def additive_delta(pert: str, ctx: Context) -> np.ndarray | None:
    """Sum of the two training single deltas of a pair, or None if one is missing."""
    parts = pert.split("_")
    if any(p not in ctx.train_means.index for p in parts):
        return None
    return sum(ctx.train_means.loc[p].to_numpy() - ctx.control_mean for p in parts)


def similarity_to_training(pert: str, ctx: Context) -> tuple[float, float]:
    """Maximum cosine similarity between the delta of a test pair and training deltas."""
    names = [p for p in ctx.train_means.index if p != CONTROL_LABEL]
    matrix = ctx.train_means.loc[names].to_numpy() - ctx.control_mean
    true_delta = ctx.test_means.loc[pert].to_numpy() - ctx.control_mean
    cos = cosine_to_rows(true_delta, matrix)
    own = np.array([n in pert.split("_") for n in names])
    return float(np.nanmax(cos)), float(np.nanmax(cos[~own]))
