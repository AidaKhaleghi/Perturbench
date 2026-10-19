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
    adv_weight: float = 0
    disc_hidden_dim: int = 128

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 100
    patience: int | None = 10
    checkpoint_dir: str | None = "results/checkpoints"
    log_every: int = 1
    restore_best: bool = True
