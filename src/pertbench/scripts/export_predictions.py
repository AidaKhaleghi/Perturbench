from pathlib import Path

import anndata as ad
import numpy as np

from pertbench.analysis import CPA_VARIANTS, PREDICTIONS_DIR, SEEDS
from pertbench.data.split import make_split
from pertbench.models.baselines import AdditiveBaseline, MeanBaseline
from pertbench.models.variants import build_variant

adata = ad.read_h5ad("data/processed/norman2019.h5ad")
split = make_split(list(adata.obs["perturbation"].unique()), seed=0)
train = adata[adata.obs["perturbation"].isin(split.train)].copy()
perts = np.array(split.test, dtype=str)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def save(name: str, seed: int, model) -> None:
    preds = np.stack([model.predict_mean(p, 1000) for p in perts]).astype(np.float32)
    np.savez(PREDICTIONS_DIR / f"{name}_seed{seed}.npz", perts=perts, preds=preds)
    print(f"saved {name} seed {seed}: {preds.shape}", flush=True)


save("mean", 0, MeanBaseline().fit(train))
save("additive", 0, AdditiveBaseline().fit(train))

for name in CPA_VARIANTS:
    for seed in SEEDS:
        checkpoint = Path(f"results/checkpoints/{name}_seed{seed}/last.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} is missing. Copy results/checkpoints first.")
        model = build_variant(name, seed=seed).load(train, checkpoint)
        save(name, seed, model)
