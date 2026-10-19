import numpy as np
import pandas as pd

from pertbench.analysis import load_context, load_predictions
from pertbench.plotting.diagnostics import plot_gene_level
from pertbench.plotting.style import save_figure

N_GENES = 50
ctx = load_context()
perts, base = load_predictions(("additive",), seeds=(0,))
_, cpa = load_predictions(("cpa",))
table = pd.read_csv("results/table/per_perturbation.csv")
cpa_rows = table[table["model"] == "cpa"].sort_values("mse")
good, bad = cpa_rows.iloc[0]["perturbation"], cpa_rows.iloc[-1]["perturbation"]
print("well predicted:", good, "| badly predicted:", bad)

panels = []
for pert in (good, bad):
    i = perts.index(pert)
    true = ctx.test_means.loc[pert].to_numpy() - ctx.control_mean
    idx = np.argsort(np.abs(true))[::-1][:N_GENES]
    cpa_pred = np.mean([a[i] for a in cpa["cpa"]], axis=0) - ctx.control_mean
    add_pred = base["additive"][0][i] - ctx.control_mean
    r_cpa = np.corrcoef(cpa_pred[idx], true[idx])[0, 1]
    r_add = np.corrcoef(add_pred[idx], true[idx])[0, 1]
    print(f"{pert}: top-{N_GENES} Pearson cpa {r_cpa:.3f}, additive {r_add:.3f}")
    errors = np.abs(cpa_pred[idx] - true[idx])
    for j in np.argsort(errors)[::-1][:5]:
        g = idx[j]
        print(f"   largest cpa error: {ctx.var_names[g]} true {true[g]:+.2f} cpa {cpa_pred[g]:+.2f}"
              f" additive {add_pred[g]:+.2f}")
    panels.append(dict(
        title=f"{pert} (CPA r = {r_cpa:.2f}, additive r = {r_add:.2f})",
        true=true[idx],
        preds={"additive": add_pred[idx], "cpa": cpa_pred[idx]},
        genes=ctx.var_names[idx], label_model="cpa", label_n=4,
    ))
save_figure(plot_gene_level(panels), "plot_gene_level")
