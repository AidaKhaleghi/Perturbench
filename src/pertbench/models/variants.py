from pertbench.config import ModelConfig, TrainConfig
from pertbench.models.cpa_model import CPA

# Settings chosen on the validation set only (Week 3 sweeps).
VARIANT_MODEL_CONFIGS = {
    "cpa": ModelConfig(latent_dim=64, kl_weight=30.0),
    "cpa_small_kl": ModelConfig(latent_dim=64, kl_weight=1e-3),
    "cpa_adv": ModelConfig(latent_dim=64, kl_weight=1e-3, adv_weight=100.0),
}


def make_train_config(**overrides) -> TrainConfig:
    """Shared training settings: lr 3e-4, 100 epochs, no early stopping."""
    base = dict(
        lr=3e-4,
        max_epochs=100,
        patience=None,
        restore_best=False,
        log_every=10,
    )
    base.update(overrides)
    return TrainConfig(**base)


def build_variant(name: str, seed: int = 0, **train_overrides) -> CPA:
    """Create an untrained CPA variant with the frozen settings."""
    return CPA(
        cfg=VARIANT_MODEL_CONFIGS[name],
        train_cfg=make_train_config(**train_overrides),
        seed=seed,
        name=name,
    )
