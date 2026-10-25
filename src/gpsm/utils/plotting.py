"""Plot a training history (loss / threshold-accuracy) for a paper figure.

Takes the ``history`` dict returned by ``MotionMambaTrainer.train()`` /
``SMTrainer.train()`` directly -- no separate logging needed.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    out_path: str = "training_history.png",
    title: str = "MotionMamba training",
    acc_threshold: Optional[float] = None,
) -> str:
    """Save a two-panel figure: loss on top, threshold-accuracy below.

    Args:
        history:       The dict ``train()`` returns -- ``train_loss`` and,
                        if present, ``val_loss``/``train_acc``/``val_acc``.
        out_path:       Where to save the PNG.
        title:          Figure title.
        acc_threshold:  The tolerance the accuracy curve was computed with
                         (``acc_threshold`` passed to ``train()``) -- shown
                         in the axis label so the figure is self-describing
                         rather than needing the threshold explained
                         separately in the paper text.

    Returns:
        ``out_path``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_acc = "train_acc" in history and len(history["train_acc"]) > 0
    fig, axes = plt.subplots(2 if has_acc else 1, 1, figsize=(7, 7 if has_acc else 4), sharex=True)
    if not has_acc:
        axes = [axes]

    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train loss", color="#4C6EF5", linewidth=1.8)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="val loss", color="#E03131", linewidth=1.8)
    ax.set_ylabel("MSE loss")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    if has_acc:
        ax = axes[1]
        ax.plot(epochs, history["train_acc"], label="train acc", color="#4C6EF5", linewidth=1.8)
        if "val_acc" in history and history["val_acc"]:
            ax.plot(epochs, history["val_acc"], label="val acc", color="#E03131", linewidth=1.8)
        label = "Threshold accuracy"
        if acc_threshold is not None:
            label += f"  (τ={acc_threshold}, normalized L2)"
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
