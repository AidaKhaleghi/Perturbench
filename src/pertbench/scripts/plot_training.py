import matplotlib.pyplot as plt
import pandas as pd

from pertbench.models.variants import VARIANT_MODEL_CONFIGS
from pertbench.plotting.style import save_figure
from pertbench.plotting.training import lowest_smoothed_epoch, plot_training_curves

refs = pd.read_csv("results/val_references.csv", index_col=0)["val_mse"].to_dict()

for name in VARIANT_MODEL_CONFIGS:
    history = pd.read_csv(f"results/checkpoints/{name}_seed0/history.csv")
    fig = plot_training_curves(history, name, refs)
    save_figure(fig, f"plot2_training_{name}")
    plt.close(fig)
    print(name, "lowest trailing-average epoch:", lowest_smoothed_epoch(history))
