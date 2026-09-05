from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    target_sum: float = 1e4
    n_top_genes: int = 2000

@dataclass
class EvalConfig:
    n_de_genes: int = 20
