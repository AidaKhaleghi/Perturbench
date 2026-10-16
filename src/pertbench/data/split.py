"""Perturbation-level train/val/test splits."""

from __future__ import annotations

import anndata as ad


def make_split(
    adata: ad.AnnData,
    seed: int = 0,
    frac_doubles_train: float = 0.5,
    frac_val: float = 0.1,
) -> ad.AnnData:
    """Assign every cell to train, val, or test by its perturbation.

    Test conditions are double perturbations whose two genes each appear as a
    single perturbation in train, but whose pair does not appear in train.
    Control cells go to every split.
    """
    raise NotImplementedError
