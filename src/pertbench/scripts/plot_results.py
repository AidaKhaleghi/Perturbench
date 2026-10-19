import matplotlib.pyplot as plt
import pandas as pd

from pertbench.plotting.results import per_seed_means, plot_main_results
from pertbench.plotting.style import save_figure

metrics = pd.read_csv("results/benchmark.csv")

for dataset in metrics["dataset"].unique():
    seeds = per_seed_means(metrics, dataset)
    mse = seeds[seeds["metric"] == "mse"].pivot(index="seed", columns="model", values="value")
    print(f"{dataset}: test MSE for each seed")
    print(mse[["cpa_small_kl", "cpa_adv", "cpa"]].round(4).to_string())

    fig = plot_main_results(metrics, dataset)
    save_figure(fig, f"plot_main_results_{dataset}")
    plt.close(fig)
