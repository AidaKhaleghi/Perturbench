import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import umap

from pertbench.analysis import load_context, pair_similarity_correlation, single_delta_matrix
from pertbench.data.tensors import PerturbationVocab
from pertbench.plotting.diagnostics import plot_embeddings
from pertbench.plotting.style import save_figure

N_CLUSTERS = 5
ctx = load_context()
vocab = PerturbationVocab.from_perturbations(list(ctx.train_means.index))
genes, delta = single_delta_matrix(ctx)

for variant in ("cpa", "cpa_small_kl"):
    rhos = []
    for seed in range(5):
        state = torch.load(f"results/checkpoints/{variant}_seed{seed}/last.pt", map_location="cpu")
        emb_all = state["model"]["pert_embedding.weight"].numpy().T  # (n_vocab, latent)
        emb = emb_all[[vocab.index[g] for g in genes]]
        rho, p = pair_similarity_correlation(emb, delta)
        rhos.append(rho)
        print(f"{variant} seed {seed}: Spearman(embedding similarity, delta similarity) "
              f"{rho:+.3f}, permutation p {p:.4f}", flush=True)
        if variant == "cpa" and seed == 0:
            emb_plot = emb
    print(f"{variant}: mean rho {np.mean(rhos):+.3f}\n")

# Data-derived groups: hierarchical clustering of single deltas (cosine distance)
cluster = fcluster(linkage(pdist(delta, "cosine"), "average"), N_CLUSTERS, "maxclust") - 1
for k in range(N_CLUSTERS):
    members = [g for g, c in zip(genes, cluster) if c == k]
    print(f"cluster {k + 1} ({len(members)} genes): {', '.join(members)}")

# One expression gene with the largest variance across single deltas
key = int(np.argmax(delta.var(axis=0)))
key_name = str(ctx.var_names[key])
print("expression gene used for the color scale:", key_name)

coords = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=0).fit_transform(emb_plot)
extreme = np.argsort(np.abs(delta[:, key]))[::-1][:8]
fig = plot_embeddings(
    coords, genes, cluster, delta[:, key], f"Shift of {key_name} (log1p units)",
    [genes[i] for i in extreme],
)
save_figure(fig, "plot_embeddings")
