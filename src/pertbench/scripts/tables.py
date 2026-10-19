from pathlib import Path

import numpy as np
import pandas as pd

from pertbench.analysis import (
    CPA_VARIANTS,
    bootstrap_mean_ci,
    fit_shrink,
    load_context,
    load_predictions,
    residual_correlation,
    residual_norm_ratio,
    similarity_to_training,
    spearman_ci,
    top_gene_metrics,
    additive_delta,
)

OUT = Path("results/table")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("mean", "additive") + CPA_VARIANTS

ctx = load_context()
perts, base = load_predictions(("mean", "additive"), seeds=(0,))
_, cpa = load_predictions(CPA_VARIANTS)
preds = {**base, **cpa}
control = ctx.control_mean

# Scaled additive: one factor fitted on training doubles only
doubles = [p for p in ctx.split.train if "_" in p and additive_delta(p, ctx) is not None]
lam = fit_shrink(
    [ctx.train_means.loc[p].to_numpy() - control for p in doubles],
    [additive_delta(p, ctx) for p in doubles],
)
print(f"scaled additive factor (fit on {len(doubles)} training doubles): {lam:.3f}")
preds["additive_scaled"] = [control + lam * (base["additive"][0] - control)]

rows = []
for i, p in enumerate(perts):
    idx = ctx.de[p]
    true_mean = ctx.test_means.loc[p].to_numpy()
    true_d = (true_mean - control)[idx]
    add_d = (base["additive"][0][i] - control)[idx]
    sim_all, sim_novel = similarity_to_training(p, ctx)
    interaction = residual_norm_ratio(true_d, add_d)
    for model, arrays in preds.items():
        per_seed = []
        for arr in arrays:
            mse, pear = top_gene_metrics(arr[i], true_mean, control, idx)
            rc = residual_correlation((arr[i] - control)[idx], add_d, true_d)
            per_seed.append((mse, pear, rc))
        mse, pear, rc = np.nanmean(np.array(per_seed), axis=0)
        rows.append(dict(model=model, perturbation=p, mse=mse, pearson=pear, resid_corr=rc,
                         sim_all=sim_all, sim_novel=sim_novel, interaction=interaction))
table = pd.DataFrame(rows)
table.to_csv(OUT / "per_perturbation.csv", index=False)

one = table[table["model"] == "additive"].set_index("perturbation")
print("\nCheck against the benchmark (mean over perturbations, seeds averaged first):")
print(table.groupby("model")[["mse", "pearson"]].mean().round(3).to_string())

print("\nSimilarity to training deltas (test perturbations):")
print(one[["sim_all", "sim_novel", "interaction"]].describe().round(3).to_string())

print("\nSpearman correlation across the test perturbations, with 95% bootstrap interval:")
for xcol in ("sim_novel", "interaction"):
    for ycol in ("pearson", "mse"):
        for model in MODELS:
            part = table[table["model"] == model]
            r, lo, hi = spearman_ci(part[xcol].to_numpy(), part[ycol].to_numpy())
            print(f"  {xcol:11s} vs {ycol:7s} {model:15s} rho {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")

print("\nMean MSE in the weakest-interaction half and the strongest-interaction half:")
median = one["interaction"].median()
low_set = set(one.index[one["interaction"] <= median])
for model in list(MODELS) + ["additive_scaled"]:
    part = table[table["model"] == model].set_index("perturbation")["mse"]
    a = part[part.index.isin(low_set)].mean()
    b = part[~part.index.isin(low_set)].mean()
    print(f"  {model:15s} weak {a:.3f} | strong {b:.3f}")

print("\nResidual correlation corr(pred - additive, true - additive), mean and 95% interval:")
for model in MODELS:
    part = table[table["model"] == model]["resid_corr"].to_numpy()
    m, lo, hi = bootstrap_mean_ci(part)
    print(f"  {model:15s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\nFive test perturbations with the strongest interaction:")
print(one.sort_values("interaction", ascending=False).head(5)[["interaction", "sim_novel"]]
      .round(3).to_string())

print("\nThree worst perturbations per model by MSE:")
for model in MODELS:
    part = table[table["model"] == model].sort_values("mse", ascending=False).head(3)
    print(f"  {model}: " + ", ".join(f"{r.perturbation} ({r.mse:.3f})" for r in part.itertuples()))
print("\nThree lowest Pearson per model:")
for model in MODELS:
    part = table[table["model"] == model].sort_values("pearson").head(3)
    print(f"  {model}: " + ", ".join(f"{r.perturbation} ({r.pearson:.3f})" for r in part.itertuples()))
