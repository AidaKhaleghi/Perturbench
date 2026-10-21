import numpy as np
import pandas as pd

from pertbench.analysis import (
    additive_delta,
    bootstrap_mean_ci,
    fit_shrink,
    load_context,
    load_predictions,
    residual_correlation,
)

ctx = load_context()
control = ctx.control_mean
perts, base = load_predictions(("additive",), seeds=(0,))
_, cpa = load_predictions(("cpa",))

doubles = [p for p in ctx.split.train if "_" in p and additive_delta(p, ctx) is not None]
lam = fit_shrink(
    [ctx.train_means.loc[p].to_numpy() - control for p in doubles],
    [additive_delta(p, ctx) for p in doubles],
)

print("1. Residual correlation of cpa (mean over seeds, 95% interval over perturbations)")
for label, factor in (("additive", 1.0), (f"scaled additive ({lam:.3f})", lam)):
    values = []
    for i, p in enumerate(perts):
        idx = ctx.de[p]
        true_d = (ctx.test_means.loc[p].to_numpy() - control)[idx]
        ref = factor * (base["additive"][0][i] - control)[idx]
        per_seed = [residual_correlation((a[i] - control)[idx], ref, true_d) for a in cpa["cpa"]]
        values.append(np.nanmean(per_seed))
    m, lo, hi = bootstrap_mean_ci(np.array(values))
    print(f"   against {label:24s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

table = pd.read_csv("results\\table\\per_perturbation.csv")
wide = table.pivot(index="perturbation", columns="model", values="mse")
inter = table[table["model"] == "additive"].set_index("perturbation")["interaction"]
strong = inter > inter.median()
rng = np.random.default_rng(0)
print("\n2. MSE difference cpa - additive (negative means cpa is better)")
for name, mask in (("weak interaction", ~strong), ("strong interaction", strong)):
    diff = (wide["cpa"] - wide["additive"])[mask.reindex(wide.index)].to_numpy()
    m, lo, hi = bootstrap_mean_ci(diff)
    print(f"   {name:19s} n={len(diff)} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n3. Interaction strength and its rank among the 39 test pairs (1 = strongest)")
rank = inter.rank(ascending=False).astype(int)
for p in ("ZBTB10_DLX2", "ZBTB10_ELMSAN1", "KLF1_COL2A1", "KIF18B_KIF2C", "KLF1_CLDN6"):
    if p in inter.index:
        print(f"   {p:16s} {inter[p]:.3f}  rank {rank[p]}")

print("\n4. Training doubles that contain the gene")
train_doubles = [p for p in ctx.split.train if "_" in p]
counts = pd.Series([g for p in train_doubles for g in p.split("_")]).value_counts()
for g in ("ZBTB10", "DLX2", "ELMSAN1", "KLF1", "COL2A1", "KIF18B", "KIF2C", "CLDN6"):
    print(f"   {g:8s} {int(counts.get(g, 0))}")
print(f"   median over genes in training doubles: {counts.median():.0f}")
