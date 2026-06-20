"""Publication-oriented plots for StatsPAI neural causal estimators."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .exports import neural_effects_frame, neural_training_frame


def neural_causal_plot(
    result: Any,
    type: str = "cate",
    *,
    ax: Any = None,
    figsize: Any = (8, 5),
    bins: int = 30,
    color: str = "#2563EB",
    treated_color: str = "#DC2626",
    control_color: str = "#64748B",
    alpha: float = 0.72,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Plot diagnostics for TARNet/CFRNet/DragonNet results.

    Parameters
    ----------
    result : CausalResult
        Neural causal result.
    type : {"cate", "effects", "propensity", "loss"}
        ``"cate"`` draws a CATE histogram, ``"effects"`` draws sorted
        unit-level CATEs, ``"propensity"`` draws DragonNet overlap, and
        ``"loss"`` draws per-epoch training/validation loss.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.tarnet(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                 epochs=50)  # doctest: +SKIP
    >>> fig, ax = sp.neural_causal_plot(res, type='cate')  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for neural causal plots. "
            "Install: pip install statspai[plotting]"
        ) from exc

    typ = type.lower()
    if typ in {"ite", "cates", "distribution"}:
        typ = "cate"
    if typ in {"sorted", "ranked"}:
        typ = "effects"
    if typ in {"overlap", "prop"}:
        typ = "propensity"
    if typ not in {"cate", "effects", "propensity", "loss"}:
        raise ValueError("type must be one of: cate, effects, propensity, loss")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    df = neural_effects_frame(result)
    architecture = result.model_info.get("architecture", result.method)

    if typ == "cate":
        cate = df["cate"].to_numpy()
        ax.hist(cate, bins=bins, color=color, alpha=alpha, edgecolor="white")
        ax.axvline(np.mean(cate), color="#111827", linewidth=1.5, label="Mean")
        ax.axvline(
            np.median(cate),
            color="#F59E0B",
            linestyle="--",
            linewidth=1.2,
            label="Median",
        )
        ax.axvline(0, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xlabel("Estimated individual treatment effect")
        ax.set_ylabel("Count")
        ax.set_title(title or f"{architecture} CATE Distribution")
        ax.legend(frameon=False)

    elif typ == "effects":
        ordered = np.sort(df["cate"].to_numpy())
        x = np.arange(1, len(ordered) + 1)
        ax.plot(x, ordered, color=color, linewidth=1.7)
        ax.axhline(np.mean(ordered), color="#111827", linewidth=1.2, label="Mean CATE")
        ax.axhline(0, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.fill_between(x, ordered, np.mean(ordered), color=color, alpha=0.12)
        ax.set_xlabel("Units sorted by estimated effect")
        ax.set_ylabel("Estimated individual treatment effect")
        ax.set_title(title or f"{architecture} Sorted Unit Effects")
        ax.legend(frameon=False)

    elif typ == "propensity":
        if "propensity" not in df or "treatment" not in df:
            raise ValueError(
                "Propensity plot requires a result with propensity scores "
                "(DragonNet)."
            )
        treated = df.loc[df["treatment"] == 1, "propensity"]
        control = df.loc[df["treatment"] == 0, "propensity"]
        ax.hist(
            control,
            bins=bins,
            alpha=0.55,
            color=control_color,
            label="Control",
            density=True,
        )
        ax.hist(
            treated,
            bins=bins,
            alpha=0.55,
            color=treated_color,
            label="Treated",
            density=True,
        )
        ax.axvline(0.05, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.axvline(0.95, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xlabel("Estimated propensity score")
        ax.set_ylabel("Density")
        ax.set_title(title or f"{architecture} Propensity Overlap")
        ax.legend(frameon=False)

    else:
        hist = neural_training_frame(result)
        if hist.empty:
            raise ValueError("No training history recorded on this result.")
        ax.plot(
            hist["epoch"], hist["train_loss"], color=color, linewidth=1.6, label="Train"
        )
        if "val_loss" in hist and hist["val_loss"].notna().any():
            ax.plot(
                hist["epoch"],
                hist["val_loss"],
                color="#F59E0B",
                linewidth=1.6,
                label="Validation",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title or f"{architecture} Training Loss")
        ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


__all__ = ["neural_causal_plot"]
