from dataclasses import dataclass

import numpy as np
import anndata as ad

CONTROL_LABEL = "control"


@dataclass
class Split:
    """Perturbation names assigned to each set."""
    train: list[str]
    val: list[str]
    test: list[str]


def pair_key(perturbation: str) -> frozenset[str]:
    """Return an order-free key, so A_B and B_A are the same pair."""
    return frozenset(perturbation.split("_"))


def make_split(
    perturbations: list[str],
    train_frac: float = 0.6,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Split:
    """Split perturbation names into train, validation, and test.

    All singles and the control go to train. Doubles are shuffled
    and divided by the given fractions. The rest of the doubles are test.
    """
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to < 1")

    names = sorted({str(p) for p in perturbations})
    singles = [p for p in names if p != CONTROL_LABEL and "_" not in p]
    doubles = [p for p in names if "_" in p]

    groups : dict[frozenset[str], list[str]] = {}
    for name in doubles:
        groups.setdefault(pair_key(name), []).append(name)
    keys = sorted(groups, key=lambda k: tuple(sorted(k)))
    
    rng = np.random.default_rng(seed)
    shuffled = [keys[i] for i in rng.permutation(len(keys))]
    
    n_train = round(train_frac * len(shuffled))
    n_val = round(val_frac * len(shuffled))
    train_keys = shuffled[:n_train]
    val_keys = shuffled[n_train:n_train + n_val]
    test_keys = shuffled[n_train + n_val:]

    def expand(selected: list[frozenset[str]]) -> list[str]:
        return sorted(name for key in selected for name in groups[key])

    
    train = [CONTROL_LABEL] + singles + expand(train_keys)
    val = expand(val_keys)
    test = expand(test_keys)

    train_singles = set(singles)
    for set_name, members in (("val", val), ("test", test)):
        for name in members:
            unseen = set(name.split("_")) - train_singles
            if unseen:
                raise ValueError(f"{set_name} perturbation {name} has unseen singles {sorted(unseen)}")

    if len(set(train) | set(val) | set(test)) != len(train) + len(val) + len(test):
        raise ValueError("A perturbation appears in more than one split")

    return Split(train=train, val=val, test=test)


if __name__ == "__main__":
    a = ad.read_h5ad("data/processed/norman2019.h5ad", backed="r")
    split = make_split(list(a.obs["perturbation"].unique()), train_frac=0.6, val_frac=0.1, seed=0)

    labels = a.obs["perturbation"].astype(str)
    for name in ("train", "val", "test"):
        members = getattr(split, name)
        n_doubles = sum("_" in p for p in members)
        n_cells = int(labels.isin(members).sum())
        print(f"{name}: {len(members)} perturbations, {n_doubles} doubles, {n_cells} cells")

    print("first 5 test perturbations:", split.test[:5])
