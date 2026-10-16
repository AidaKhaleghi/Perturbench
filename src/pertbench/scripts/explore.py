import anndata as ad
from pathlib import Path

adata = ad.read_h5ad(Path("data/raw/norman2019.h5ad"))

# 1. How are perturbations named?
print("Unique perturbations:", adata.obs["perturbation"].unique())
print("Perturbation counts:")
print(adata.obs["perturbation"].value_counts().head(15))

# 2. How many cells have 0, 1, or 2 perturbed genes?
print("\nCell counts by perturbed gene count:")
print(adata.obs["nperts"].value_counts())

# 3. What is the control label?
print("\nControl label:")
print(adata.obs.groupby("nperts")["perturbation"].unique())

# 4. How many distinct singles and doubles?
counts = adata.obs["perturbation"].value_counts()
print("\ndistinct perturbations:", len(counts))
print("fewest cells in one perturbation:", counts.min())
print("most cells in one perturbation:", counts.max())

# 5. Is .X raw counts (integers)?
x = adata.X[:5, :].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :]
print("\nmax value in first 5 cells:", x.max())
print("Are values integers?", (x == x.round()).all())

# 6. Number of single and double perturbations
obs = adata.obs
singles = obs.loc[obs["nperts"] == 1, "perturbation"].unique()
doubles = obs.loc[obs["nperts"] == 2, "perturbation"].unique()
print("\nSingles:", len(singles), "Doubles:", len(doubles))

# 7. Which doubles have both genes available as singles?
single_set = set(singles)
both = [d for d in doubles if all(g in single_set for g in d.split("_"))]
print("\nNumber of doubles with both singles present:", len(both))
