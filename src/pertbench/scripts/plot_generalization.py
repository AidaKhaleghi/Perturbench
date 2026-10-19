import pandas as pd

from pertbench.plotting.diagnostics import plot_generalization
from pertbench.plotting.style import save_figure

table = pd.read_csv("results/table/per_perturbation.csv")
save_figure(plot_generalization(table), "plot_generalization")
