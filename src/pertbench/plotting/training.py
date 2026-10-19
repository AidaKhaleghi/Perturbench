import matplotlib.pyplot as plt
import pandas as pd

from pertbench.plotting.style import MODEL_COLORS, apply_style

SMOOTH_WINDOW = 5


def lowest_smoothed_epoch(history: pd.DataFrame, window: int = SMOOTH_WINDOW) -> int:
    """Return the epoch where the trailing average of val_de_mse is lowest."""
    smooth = history["val_de_mse"].rolling(window).mean()
    return int(history.loc[smooth.idxmin(), "epoch"])


def plot_training_curves(
    history: pd.DataFrame,
    name: str,
    references: dict[str, float] | None = None,
) -> plt.Figure:
    """Plot loss components and validation delta MSE against epoch."""
    apply_style()
    color = MODEL_COLORS[name]
    stop = lowest_smoothed_epoch(history)
    epoch = history["epoch"]
    has_adv = "train_adv_d" in history.columns

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.ravel()

    # Panels with a train curve and a validation curve
    pairs = [
        (flat[0], "total", "Total loss (per cell)"),
        (flat[1], "recon", "Reconstruction (squared error per cell)"),
        (flat[2], "kl", "KL divergence (nats per cell)"),
    ]
    for ax, key, ylabel in pairs:
        ax.plot(epoch, history[f"train_{key}"], color=color, label="train")
        ax.plot(
            epoch,
            history[f"val_{key}"],
            color=color,
            linestyle="--",
            label="validation",
        )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    # Validation delta MSE, raw and smoothed, on a log axis
    ax = flat[3]
    ax.plot(
        epoch,
        history["val_de_mse"],
        color=color,
        alpha=0.35,
        label="each epoch",
    )
    ax.plot(
        epoch,
        history["val_de_mse"].rolling(SMOOTH_WINDOW).mean(),
        color=color,
        linewidth=2.5,
        label=f"{SMOOTH_WINDOW}-epoch trailing average",
    )
    for ref_name, value in (references or {}).items():
        ax.axhline(
            value,
            color=MODEL_COLORS[ref_name],
            linestyle=":",
            linewidth=2,
            label=f"{ref_name} baseline",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Validation delta MSE (log-expr.², log scale)")
    ax.legend(frameon=False)

    # Adversary losses, only for variants with a discriminator
    if has_adv:
        ax = flat[4]
        ax.plot(epoch, history["train_adv_d"], color=color, label="discriminator loss")
        ax.plot(
            epoch,
            history["train_adv_conf"],
            color=color,
            linestyle="--",
            label="confusion loss",
        )
        ax.set_ylabel("Adversary loss (nats per cell)")
        ax.legend(frameon=False)

    n_visible = 5 if has_adv else 4
    for i, ax in enumerate(flat):
        if i >= n_visible:
            ax.axis("off")
            continue
        ax.axvline(
            stop,
            color="0.4",
            linestyle="-.",
            linewidth=1,
            label=f"lowest {SMOOTH_WINDOW}-epoch average (epoch {stop})" if i == 0 else None,
        )
        ax.set_xlabel("Epoch (number)")
    flat[0].legend(frameon=False)

    fig.suptitle(f"{name}: training curves, seed 0")
    fig.tight_layout()
    return fig
