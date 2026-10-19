import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pertbench.plotting.paired import paired_table, plot_paired
from pertbench.plotting.style import save_figure

DATASET = "norman2019"
VARIANTS = ("cpa", "cpa_adv", "cpa_small_kl")
rng = np.random.default_rng(0)


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000) -> tuple[float, float]:
    """95% interval of the mean, from resampling perturbations with replacement."""
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


metrics = pd.read_csv("results/benchmark.csv")

# Labels only for cpa. The other variants lie on one side of the diagonal.
for model in VARIANTS:
    fig = plot_paired(metrics, DATASET, model, n_labels=5 if model == "cpa" else 0)
    save_figure(fig, f"plot_per_perturbation_{model}")
    plt.close(fig)

print("\nPaired comparison with additive (model minus additive)")
for model in VARIANTS:
    for metric, higher_better in (("pearson", True), ("mse", False)):
        t = paired_table(metrics, DATASET, model, metric)
        diff = (t["model"] - t["baseline"]).to_numpy()
        n_better = int((diff > 0).sum() if higher_better else (diff < 0).sum())
        low, high = bootstrap_ci(diff)
        print(
            f"{model:13s} {metric:8s} better on {n_better:2d} of {len(diff)} | "
            f"mean difference {diff.mean():+.4f} | 95% interval [{low:+.4f}, {high:+.4f}]"
        )

# Which perturbations cause the MSE gap of cpa?
t = paired_table(metrics, DATASET, "cpa", "mse")
t["diff"] = t["model"] - t["baseline"]
t = t.sort_values("diff", ascending=False).reset_index(drop=True)
print("\ncpa minus additive, MSE. Five largest and three smallest differences:")
print(pd.concat([t.head(5), t.tail(3)]).round(4).to_string(index=False))
print("sum of all 39 differences:", round(float(t["diff"].sum()), 4))
print("median difference:", round(float(t["diff"].median()), 4))
print("share of the sum from the 2 largest:", round(float(t["diff"].iloc[:2].sum() / t["diff"].sum()), 2))
print("mean difference without the 2 largest:", round(float(t["diff"].iloc[2:].mean()), 4))
