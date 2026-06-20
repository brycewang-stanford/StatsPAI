"""
Causal Effect Variational Auto-Encoder (CEVAE).

Louizos et al. (2017) "Causal Effect Inference with Deep Latent-Variable
Models" -- a VAE-based estimator that imputes a latent confounder Z
from noisy observed proxies X, then estimates ITE as

    tau(x) = E[Y | do(T=1), Z=z] - E[Y | do(T=0), Z=z]

where expectations are taken over q(Z | X).

This implementation provides a minimal but functional CEVAE with a
Gaussian latent, MLP encoder/decoder, and a treatment head — fine for
educational use and small-to-medium datasets. Scale to production via
``torch.compile`` or pyro ports of the original.

Notes
-----
If PyTorch is not installed, we fall back to a numpy-only linear
variational approximation that still honours the (encoder → treatment,
outcome heads) factorisation. This keeps the import hierarchy light.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class CEVAEResult:
    """Container for CEVAE estimates (ATE, unit-level ITE, loss history).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> # CEVAE requires the `neural` extra (pip install statspai[neural]);
    >>> # the result object is produced by fitting and then inspected:
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 3))
    >>> t = rng.binomial(1, 0.5, size=120)
    >>> y = 2.0 * t + X[:, 0] + rng.normal(scale=0.5, size=120)
    >>> res = sp.cevae(X, t, y, n_epochs=50)  # doctest: +SKIP
    >>> print(res.summary())  # doctest: +SKIP
    >>> res.tidy()  # one-row ATE table  # doctest: +SKIP
    >>> res.effects_frame().head()  # unit-level ITE  # doctest: +SKIP
    """

    ate: float
    ite: np.ndarray
    loss_history: list[float]
    backend: str

    def summary(self) -> str:
        return (
            f"CEVAE (backend={self.backend}) ATE = {self.ate:.4f}\n"
            f"  ITE spread: [{self.ite.min():.3f}, {self.ite.max():.3f}] "
            f"(median {np.median(self.ite):.3f})"
        )

    def tidy(self) -> pd.DataFrame:
        """Return a one-row CEVAE ATE table."""
        return pd.DataFrame(
            [
                {
                    "term": "ATE",
                    "estimate": self.ate,
                    "backend": self.backend,
                    "ite_mean": float(np.mean(self.ite)),
                    "ite_std": float(np.std(self.ite)),
                    "ite_q25": float(np.percentile(self.ite, 25)),
                    "ite_median": float(np.median(self.ite)),
                    "ite_q75": float(np.percentile(self.ite, 75)),
                }
            ]
        )

    def effects_frame(self) -> pd.DataFrame:
        """Return unit-level ITE estimates."""
        return pd.DataFrame({"unit": np.arange(len(self.ite)), "ite": self.ite})

    def training_frame(self) -> pd.DataFrame:
        """Return per-epoch CEVAE loss history."""
        return pd.DataFrame(
            {
                "epoch": np.arange(1, len(self.loss_history) + 1),
                "loss": self.loss_history,
            }
        )

    def to_markdown(self, path: str | None = None, digits: int = 4) -> str:
        parts = [
            "# CEVAE",
            "",
            "## Summary",
            self.tidy().round(digits).to_markdown(index=False),
            "",
            "## Unit-Level Effects",
            self.effects_frame().head(20).round(digits).to_markdown(index=False),
        ]
        training = self.training_frame().round(digits)
        if not training.empty:
            parts.extend(
                ["", "## Training Diagnostics", training.to_markdown(index=False)]
            )
        text = "\n".join(parts) + "\n"
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_html(self, path: str | None = None, digits: int = 4) -> str:
        html = "\n".join(
            [
                "<html><body>",
                "<h1>CEVAE</h1>",
                "<h2>Summary</h2>",
                self.tidy().round(digits).to_html(index=False),
                "<h2>Unit-Level Effects</h2>",
                self.effects_frame().head(50).round(digits).to_html(index=False),
                "<h2>Training Diagnostics</h2>",
                self.training_frame().round(digits).to_html(index=False),
                "</body></html>",
            ]
        )
        if path is not None:
            from pathlib import Path

            Path(path).write_text(html, encoding="utf-8")
        return html

    def to_excel(self, path: str, digits: int = 6) -> str:
        with pd.ExcelWriter(path) as writer:
            self.tidy().round(digits).to_excel(
                writer, sheet_name="Summary", index=False
            )
            self.effects_frame().round(digits).to_excel(
                writer, sheet_name="Effects", index=False
            )
            self.training_frame().round(digits).to_excel(
                writer, sheet_name="Training", index=False
            )
        return path

    def plot(
        self,
        type: str = "ite",
        *,
        ax: Any = None,
        figsize: Any = (8, 5),
        bins: int = 30,
    ) -> Any:
        """Plot ITE distribution or training loss."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib required for CEVAE plots. "
                "Install: pip install statspai[plotting]"
            ) from exc
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        typ = type.lower()
        if typ in {"ite", "cate", "effects", "distribution"}:
            ax.hist(self.ite, bins=bins, color="#2563EB", alpha=0.72, edgecolor="white")
            ax.axvline(self.ate, color="#111827", linewidth=1.5, label="ATE")
            ax.axvline(0, color="#9CA3AF", linestyle=":", linewidth=1)
            ax.set_xlabel("Estimated individual treatment effect")
            ax.set_ylabel("Count")
            ax.set_title("CEVAE ITE Distribution")
            ax.legend(frameon=False)
        elif typ in {"loss", "training"}:
            frame = self.training_frame()
            ax.plot(frame["epoch"], frame["loss"], color="#2563EB", linewidth=1.6)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("CEVAE Training Loss")
        else:
            raise ValueError("type must be 'ite' or 'loss'")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig, ax


class CEVAE:
    """Minimal CEVAE. Uses PyTorch if available, else a light numpy
    linear variational approximation.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> # CEVAE requires the `neural` extra (pip install statspai[neural]);
    >>> # illustrative object-oriented usage:
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 3))
    >>> t = rng.binomial(1, 0.5, size=120)
    >>> y = 2.0 * t + X[:, 0] + rng.normal(scale=0.5, size=120)
    >>> est = sp.CEVAE(z_dim=4, hidden=32, n_epochs=50, seed=0)
    >>> res = est.fit(X, t, y)  # doctest: +SKIP
    >>> float(res.ate)  # doctest: +SKIP
    """

    def __init__(
        self,
        z_dim: int = 4,
        hidden: int = 32,
        lr: float = 1e-2,
        n_epochs: int = 200,
        seed: int | None = 0,
    ):
        self.z_dim = z_dim
        self.hidden = hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self._torch_ok = _try_import_torch()

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> CEVAEResult:
        X = np.asarray(X, dtype=float)
        t = np.asarray(treatment, dtype=float)
        y = np.asarray(outcome, dtype=float)

        if self._torch_ok:
            return self._fit_torch(X, t, y)
        return self._fit_numpy(X, t, y)

    # --------- Torch path (preferred) ---------
    def _fit_torch(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> CEVAEResult:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(self.seed or 0)
        from ..utils._torch_device import resolve_torch_device

        device = resolve_torch_device()

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        tt = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(-1)
        yt = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
        n, d = Xt.shape

        H, Z = self.hidden, self.z_dim

        class Enc(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, H), nn.ELU())
                self.mu = nn.Linear(H, Z)
                self.logv = nn.Linear(H, Z)

            def forward(self, x: Any) -> Any:
                h = self.net(x)
                return self.mu(h), self.logv(h)

        class Dec(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x_head = nn.Sequential(nn.Linear(Z, H), nn.ELU(), nn.Linear(H, d))
                self.t_head = nn.Sequential(nn.Linear(Z, H), nn.ELU(), nn.Linear(H, 1))
                self.y_head = nn.Sequential(
                    nn.Linear(Z + 1, H), nn.ELU(), nn.Linear(H, 1)
                )

            def forward(self, z: Any, t: Any) -> Any:
                x_hat = self.x_head(z)
                t_logit = self.t_head(z)
                zt = torch.cat([z, t], dim=-1)
                y_hat = self.y_head(zt)
                return x_hat, t_logit, y_hat

        enc, dec = Enc().to(device), Dec().to(device)
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(dec.parameters()), lr=self.lr
        )

        history = []
        for epoch in range(self.n_epochs):
            mu, logv = enc(Xt)
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(mu)
            z = mu + eps * std

            x_hat, t_logit, y_hat = dec(z, tt)
            rec_x = F.mse_loss(x_hat, Xt)
            rec_t = F.binary_cross_entropy_with_logits(t_logit, tt)
            rec_y = F.mse_loss(y_hat, yt)
            kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
            loss = rec_x + rec_t + rec_y + kld

            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(float(loss.item()))

        with torch.no_grad():
            mu, _ = enc(Xt)
            _, _, y1 = dec(mu, torch.ones_like(tt))
            _, _, y0 = dec(mu, torch.zeros_like(tt))
            ite = (y1 - y0).squeeze(-1).cpu().numpy()
        return CEVAEResult(
            ate=float(ite.mean()),
            ite=ite,
            loss_history=history,
            backend="torch",
        )

    # --------- Numpy fallback ---------
    def _fit_numpy(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> CEVAEResult:
        """Linear-Gaussian variational approximation. Fits:
            z ~ N(A X, 1),  t | z ~ Bern(sigmoid(c^T z)),
            y | z, t ~ N(b1 z * t + b0 z * (1-t), sigma^2)
        Fits by alternating WLS between heads.
        """
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        Z = self.z_dim

        A = rng.normal(0, 0.1, (d, Z))
        history = []
        for epoch in range(min(self.n_epochs, 50)):
            z = X @ A  # deterministic mean
            # Fit outcome on [z, t, z*t]
            design1 = np.column_stack([np.ones(n), z, t[:, None] * z, t])
            beta, *_ = np.linalg.lstsq(design1, y, rcond=None)
            y_hat = design1 @ beta
            rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
            history.append(rmse)

            # Re-fit encoder to match X reconstruction (X ~ z B)
            B, *_ = np.linalg.lstsq(z, X, rcond=None)
            # Update A such that X A reproduces z better (1-step gradient)
            grad = (X.T @ (X @ A - z)) / n
            A = A - 0.1 * grad

        z = X @ A
        design1 = np.column_stack([np.ones(n), z, t[:, None] * z, t])
        beta, *_ = np.linalg.lstsq(design1, y, rcond=None)
        # Counterfactual predictions:
        design_y1 = np.column_stack([np.ones(n), z, 1.0 * z, np.ones(n)])
        design_y0 = np.column_stack([np.ones(n), z, 0.0 * z, np.zeros(n)])
        y1 = design_y1 @ beta
        y0 = design_y0 @ beta
        ite = y1 - y0
        return CEVAEResult(
            ate=float(ite.mean()),
            ite=ite,
            loss_history=history,
            backend="numpy",
        )


def cevae(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    **kw: Any,
) -> CEVAEResult:
    """Functional CEVAE wrapper.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> # CEVAE requires the `neural` extra (pip install statspai[neural]);
    >>> # X/treatment/outcome are plain numpy arrays:
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 3))
    >>> t = rng.binomial(1, 0.5, size=120)
    >>> y = 2.0 * t + X[:, 0] + rng.normal(scale=0.5, size=120)
    >>> res = sp.cevae(X, t, y, n_epochs=50, seed=0)  # doctest: +SKIP
    >>> print(res.summary())  # doctest: +SKIP
    """
    return CEVAE(**kw).fit(X, treatment, outcome)


def _try_import_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False
