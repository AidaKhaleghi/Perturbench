from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    target_sum: float = 1e4
    n_top_genes: int = 2000
    keep_perturbed_genes: bool = True

@dataclass
class EvalConfig:
    n_de_genes: int = 20
    exclude_perturbed_genes: bool = False

@dataclass
class ModelConfig:
    latent_dim: int = 64
    hidden_dim:int = 512
    kl_weight: float = 1e-3
