from abc import ABC, abstractmethod

import anndata as ad
import numpy as np


class PerturbationModel(ABC):
    """Common interface for all perturbation prediction models."""

    name: str = "model"

    @abstractmethod
    def fit(self, train_data: ad.AnnData) -> "PerturbationModel":
        """Learn from the training cells. Returns self."""

    @abstractmethod
    def predict(self, perturbation: str, n_cells: int) -> np.ndarray:
        """Predict expression of n_cells cells under a perturbation."""

    def predict_mean(self, perturbation: str, n_cells: int = 100) -> np.ndarray:
        """Predict the mean expression over cells, shape (n_genes,)."""
        return self.predict(perturbation, n_cells).mean(axis=0)
