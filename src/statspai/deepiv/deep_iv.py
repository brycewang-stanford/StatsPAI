"""
DeepIV: Deep Learning Instrumental Variables (Hartford et al. 2017).

Two-stage procedure:
  Stage 1 — Mixture Density Network (MDN) learns P(T | Z, X)
  Stage 2 — Response network h(T, X) trained with counterfactual loss
             using MC samples from the Stage-1 distribution.

The approach allows arbitrary non-linear treatment effects and handles
continuous endogenous regressors without parametric assumptions on the
first stage.

Implementation notes
--------------------
This implementation follows the same defaults as Microsoft's EconML
``DeepIVEstimator``: the Stage-2 gradient is computed from a single set
of MC samples (``n_gradient_samples=0``). Per Hartford et al. Section
3.2, the resulting gradient is a biased estimator of the true loss
gradient, because the treatment sample appears in both the residual and
the gradient path. As discussed in the EconML docs, the single-sample
version *"is not guaranteed to lead to consistent estimates, but has the
advantage of requiring only a single set of samples from the
distribution, and can be interpreted as regularizing the loss with a
variance penalty"*.

To enable the **unbiased paired-sample gradient** described in the
paper, set ``n_gradient_samples > 0``. This draws two independent sets
of MC samples per gradient step and uses the identity
``E[(y - h(p, x))(y - h(p', x))] = (y - E[h(p, x)])^2`` for ``p ⊥ p'``.

When to use DeepIV
------------------
- Low-to-moderate dimensional X, continuous treatment, reasonably
  strong instruments.
- When you want a flexible non-parametric first stage but don't need
  theoretical convergence guarantees.

When NOT to use DeepIV
----------------------
- Weak instruments: the two-stage procedure compounds instability.
- High-dimensional X with complex h(T, X): consider DeepGMM
  (Bennett et al. 2019), DFIV (Xu et al. 2021) or DualIV (Muandet et
  al. 2020), which are generally more stable in these settings.
- When you need rigorous model selection / convergence guarantees.
  See RegDeepIV (Dikkala et al. 2024) for a theoretically grounded
  variant.

Requires PyTorch (optional dependency):
    pip install statspai[deepiv]

References
----------
Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017).
    "Deep IV: A Flexible Approach for Counterfactual Prediction."
    Proceedings of the 34th International Conference on Machine Learning.

Microsoft EconML. ``DeepIVEstimator`` — reference implementation whose
    ``n_gradient_samples`` default of 0 this package follows.
    https://github.com/py-why/EconML

Bennett, A., Kallus, N., & Schnabel, T. (2019).
    "Deep Generalized Method of Moments for Instrumental Variable
    Analysis." NeurIPS 2019.

Xu, L., Chen, Y., Srinivasan, S., de Freitas, N., Doucet, A., & Gretton,
    A. (2021). "Learning Deep Features in Instrumental Variable
    Regression." ICLR 2021.

Muandet, K., Mehrjou, A., Lee, S. K., & Raj, A. (2020).
    "Dual Instrumental Variable Regression." NeurIPS 2020.
"""

from typing import Optional, List, Dict, Any, Tuple
import math
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deepiv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instruments: List[str],
    covariates: List[str],
    n_components: int = 10,
    hidden_layers: Tuple[int, ...] = (128, 64),
    first_stage_epochs: int = 100,
    second_stage_epochs: int = 100,
    n_samples: int = 1,
    n_gradient_samples: int = 0,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    alpha: float = 0.05,
    random_state: int = 42,
    verbose: bool = False,
) -> CausalResult:
    """
    Estimate causal effects using Deep Instrumental Variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treat : str
        Endogenous treatment variable name (continuous).
    instruments : list of str
        Excluded instrument variable names.
    covariates : list of str
        Exogenous control variable names.
    n_components : int, default 10
        Number of Gaussian mixture components in the MDN (stage 1).
    hidden_layers : tuple of int, default (128, 64)
        Hidden layer sizes for both networks.
    first_stage_epochs : int, default 100
        Training epochs for the treatment model (MDN).
    second_stage_epochs : int, default 100
        Training epochs for the response model.
    n_samples : int, default 1
        Number of MC samples per observation used to form the Stage-2
        residual ``(y - h(t, x))``.
    n_gradient_samples : int, default 0
        Number of **independent** additional samples used for the
        gradient path. When ``0`` (default), a single set of samples is
        used, matching Microsoft EconML's default behaviour and
        producing a biased but variance-regularized gradient estimator.
        When ``> 0``, two independent sample sets are drawn and the
        unbiased paired-sample estimator
        ``mean((y - h(p, x)) * (y - h(p', x)))`` from Hartford et al.
        Section 3.2 is used instead. Set to ``1`` or higher if you
        specifically need unbiased gradients (e.g. for asymptotic
        consistency arguments).
    batch_size : int, default 256
        Mini-batch size.
    learning_rate : float, default 1e-3
        Adam learning rate.
    alpha : float, default 0.05
        Significance level for confidence interval.
    random_state : int, default 42
        Random seed for reproducibility.
    verbose : bool, default False
        Print training progress.

    Returns
    -------
    CausalResult

    Notes
    -----
    Use the **single-sample default** (``n_gradient_samples=0``) for
    most applications — it matches EconML, trains roughly 2x faster, and
    the implicit variance regularization often helps in small samples.

    Use ``n_gradient_samples >= 1`` when you need the unbiased gradient
    (e.g. for theoretical guarantees or when training with very large
    batches where the bias dominates).

    For high-dimensional covariates or weak instruments, consider
    DeepGMM / DFIV / DualIV instead — see the module docstring for
    references.

    Examples
    --------
    >>> result = sp.deepiv(
    ...     df, y='lwage', treat='educ',
    ...     instruments=['nearc4'],
    ...     covariates=['exper', 'expersq'],
    ... )
    >>> print(result.summary())

    >>> # Custom architecture with unbiased gradient
    >>> result = sp.deepiv(
    ...     df, y='sales', treat='price',
    ...     instruments=['cost_shifter'],
    ...     covariates=['demand_controls'],
    ...     n_components=20,
    ...     hidden_layers=(256, 128, 64),
    ...     first_stage_epochs=200,
    ...     n_gradient_samples=1,   # paired-sample unbiased gradient
    ... )
    """
    estimator = DeepIV(
        data=data, y=y, treat=treat,
        instruments=instruments, covariates=covariates,
        n_components=n_components, hidden_layers=hidden_layers,
        first_stage_epochs=first_stage_epochs,
        second_stage_epochs=second_stage_epochs,
        n_samples=n_samples,
        n_gradient_samples=n_gradient_samples,
        batch_size=batch_size,
        learning_rate=learning_rate, alpha=alpha,
        random_state=random_state, verbose=verbose,
    )
    return estimator.fit()


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class DeepIV:
    """
    Deep Instrumental Variables estimator (Hartford et al. 2017).

    Follows the same defaults as Microsoft EconML's ``DeepIVEstimator``.
    See the module docstring for a discussion of the biased vs unbiased
    gradient trade-off and when to prefer modern alternatives.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Endogenous treatment variable (continuous).
    instruments : list of str
        Excluded instruments.
    covariates : list of str
        Exogenous controls.
    n_components : int
        Gaussian mixture components in the MDN.
    hidden_layers : tuple of int
        Hidden layer sizes.
    first_stage_epochs : int
        Stage 1 training epochs.
    second_stage_epochs : int
        Stage 2 training epochs.
    n_samples : int
        MC samples per observation used to form the Stage-2 residual.
    n_gradient_samples : int, default 0
        Independent additional samples for the gradient path. ``0``
        reproduces EconML's default (biased but variance-regularized);
        ``>= 1`` activates Hartford et al.'s paired-sample unbiased
        gradient estimator.
    batch_size : int
    learning_rate : float
    alpha : float
        Significance level.
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        instruments: List[str],
        covariates: List[str],
        n_components: int = 10,
        hidden_layers: Tuple[int, ...] = (128, 64),
        first_stage_epochs: int = 100,
        second_stage_epochs: int = 100,
        n_samples: int = 1,
        n_gradient_samples: int = 0,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        alpha: float = 0.05,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.instruments = instruments
        self.covariates = covariates
        self.n_components = n_components
        self.hidden_layers = hidden_layers
        self.first_stage_epochs = first_stage_epochs
        self.second_stage_epochs = second_stage_epochs
        self.n_samples = n_samples
        self.n_gradient_samples = n_gradient_samples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        self._validate()

    def _validate(self):
        all_cols = [self.y, self.treat] + self.instruments + self.covariates
        missing = [c for c in all_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")
        if len(self.instruments) == 0:
            raise ValueError("At least one instrument is required")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.n_gradient_samples < 0:
            raise ValueError("n_gradient_samples must be >= 0")

    def fit(self) -> CausalResult:
        """
        Fit the DeepIV model and return causal effect estimates.

        Returns
        -------
        CausalResult
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for DeepIV. "
                "Install: pip install statspai[deepiv]  "
                "or: pip install torch"
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Prepare data
        cols = [self.y, self.treat] + self.instruments + self.covariates
        clean = self.data[cols].dropna()
        n = len(clean)

        Y = clean[self.y].values.astype(np.float32)
        T = clean[self.treat].values.astype(np.float32)
        Z = clean[self.instruments].values.astype(np.float32)
        X = clean[self.covariates].values.astype(np.float32)

        # Standardise for training stability
        self._y_mean, self._y_std = Y.mean(), Y.std() + 1e-8
        self._t_mean, self._t_std = T.mean(), T.std() + 1e-8
        Y_s = (Y - self._y_mean) / self._y_std
        T_s = (T - self._t_mean) / self._t_std

        Z_means, Z_stds = Z.mean(0), Z.std(0) + 1e-8
        X_means, X_stds = X.mean(0), X.std(0) + 1e-8
        Z_s = (Z - Z_means) / Z_stds
        X_s = (X - X_means) / X_stds

        device = torch.device('cpu')

        # ---------------------------------------------------------------
        # Stage 1: Mixture Density Network for P(T | Z, X)
        # ---------------------------------------------------------------
        zx_dim = Z_s.shape[1] + X_s.shape[1]
        mdn = _build_mdn(zx_dim, self.n_components, self.hidden_layers).to(device)

        ZX = np.column_stack([Z_s, X_s])
        ZX_t = torch.tensor(ZX, dtype=torch.float32, device=device)
        T_t = torch.tensor(T_s, dtype=torch.float32, device=device)

        dataset1 = TensorDataset(ZX_t, T_t)
        loader1 = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True)

        opt1 = optim.Adam(mdn.parameters(), lr=self.learning_rate)

        for epoch in range(self.first_stage_epochs):
            epoch_loss = 0.0
            mdn.train()
            for zx_batch, t_batch in loader1:
                opt1.zero_grad()
                pi, mu, sigma = mdn(zx_batch)
                loss = _mdn_loss(pi, mu, sigma, t_batch)
                loss.backward()
                opt1.step()
                epoch_loss += loss.item() * len(t_batch)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Stage 1 epoch {epoch+1}/{self.first_stage_epochs}, "
                      f"loss={epoch_loss/n:.4f}")

        mdn.eval()

        # ---------------------------------------------------------------
        # Stage 2: Response network h(T, X) with counterfactual loss
        # ---------------------------------------------------------------
        tx_dim = 1 + X_s.shape[1]
        response_net = _build_response_net(tx_dim, self.hidden_layers).to(device)

        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_s, dtype=torch.float32, device=device)

        dataset2 = TensorDataset(ZX_t, X_t, Y_t)
        loader2 = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True)

        opt2 = optim.Adam(response_net.parameters(), lr=self.learning_rate)

        # Whether to use the unbiased paired-sample gradient from
        # Hartford et al. (2017) Section 3.2. When False, we fall back
        # to the single-sample biased (but variance-regularized) loss
        # that matches EconML's default (``n_gradient_samples=0``).
        use_unbiased_grad = self.n_gradient_samples > 0
        n_pairs = max(self.n_samples, self.n_gradient_samples)

        for epoch in range(self.second_stage_epochs):
            epoch_loss = 0.0
            response_net.train()
            for zx_batch, x_batch, y_batch in loader2:
                opt2.zero_grad()

                with torch.no_grad():
                    pi, mu, sigma = mdn(zx_batch)

                if use_unbiased_grad:
                    # Two INDEPENDENT sample sets from P(T | Z, X). The
                    # paired identity
                    #     E_{p,p'~F}[(y - h(p,x)) * (y - h(p',x))]
                    #   = (y - E_{p~F}[h(p,x)])^2
                    # gives an unbiased estimator of both the loss AND
                    # its gradient w.r.t. the network parameters.
                    with torch.no_grad():
                        t_samples_a = _sample_mdn(pi, mu, sigma, n_pairs)
                        t_samples_b = _sample_mdn(pi, mu, sigma, n_pairs)

                    loss = torch.tensor(0.0, device=device)
                    for s in range(n_pairs):
                        t_a = t_samples_a[:, s].unsqueeze(1)
                        t_b = t_samples_b[:, s].unsqueeze(1)
                        tx_a = torch.cat([t_a, x_batch], dim=1)
                        tx_b = torch.cat([t_b, x_batch], dim=1)
                        h_a = response_net(tx_a).squeeze()
                        h_b = response_net(tx_b).squeeze()
                        loss = loss + torch.mean(
                            (y_batch - h_a) * (y_batch - h_b)
                        )
                    loss = loss / n_pairs
                else:
                    # Single-sample biased estimator (EconML default).
                    # Same treatment sample is used in both the residual
                    # and the gradient path; this introduces a bias but
                    # trains ~2x faster and acts as implicit variance
                    # regularization.
                    with torch.no_grad():
                        t_samples = _sample_mdn(
                            pi, mu, sigma, self.n_samples
                        )

                    loss = torch.tensor(0.0, device=device)
                    for s in range(self.n_samples):
                        t_s = t_samples[:, s].unsqueeze(1)
                        tx = torch.cat([t_s, x_batch], dim=1)
                        y_pred = response_net(tx).squeeze()
                        loss = loss + torch.mean((y_batch - y_pred) ** 2)
                    loss = loss / self.n_samples

                loss.backward()
                opt2.step()
                epoch_loss += loss.item() * len(y_batch)

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Stage 2 epoch {epoch+1}/{self.second_stage_epochs}, "
                      f"loss={epoch_loss/n:.4f}")

        response_net.eval()

        # ---------------------------------------------------------------
        # Estimate treatment effect: E[h(t1, X) - h(t0, X)]
        # Default: effect of 1-SD increase from mean
        # ---------------------------------------------------------------
        t0_raw = self._t_mean
        t1_raw = self._t_mean + self._t_std
        t0_s = 0.0  # (t0_raw - t_mean) / t_std
        t1_s = 1.0  # (t1_raw - t_mean) / t_std

        with torch.no_grad():
            t0_vec = torch.full((n, 1), t0_s, dtype=torch.float32, device=device)
            t1_vec = torch.full((n, 1), t1_s, dtype=torch.float32, device=device)

            tx0 = torch.cat([t0_vec, X_t], dim=1)
            tx1 = torch.cat([t1_vec, X_t], dim=1)

            y0 = response_net(tx0).squeeze().numpy()
            y1 = response_net(tx1).squeeze().numpy()

        # Rescale back to original Y scale
        effects = (y1 - y0) * self._y_std
        ate = float(np.mean(effects))

        # Bootstrap SE for inference
        n_boot = 500
        boot_ates = np.empty(n_boot)
        rng = np.random.RandomState(self.random_state)
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            boot_ates[b] = np.mean(effects[idx])

        se = float(np.std(boot_ates, ddof=1))
        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (ate - z_crit * se, ate + z_crit * se)
        z_stat = ate / se if se > 0 else 0.0
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))

        # Marginal effects at different treatment levels
        detail_rows = []
        for delta_sd in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            t_base = delta_sd
            t_up = delta_sd + 1.0
            with torch.no_grad():
                tv0 = torch.full((n, 1), t_base, dtype=torch.float32, device=device)
                tv1 = torch.full((n, 1), t_up, dtype=torch.float32, device=device)
                txv0 = torch.cat([tv0, X_t], dim=1)
                txv1 = torch.cat([tv1, X_t], dim=1)
                eff = (response_net(txv1).squeeze().numpy()
                       - response_net(txv0).squeeze().numpy()) * self._y_std
            t_level = self._t_mean + delta_sd * self._t_std
            detail_rows.append({
                'treatment_level': round(float(t_level), 4),
                'marginal_effect': round(float(np.mean(eff)), 6),
                'std_dev': round(float(np.std(eff, ddof=1)), 6),
            })

        detail = pd.DataFrame(detail_rows)

        model_info = {
            'n_components': self.n_components,
            'hidden_layers': self.hidden_layers,
            'first_stage_epochs': self.first_stage_epochs,
            'second_stage_epochs': self.second_stage_epochs,
            'n_instruments': len(self.instruments),
            'n_covariates': len(self.covariates),
            'n_mc_samples': self.n_samples,
            'n_gradient_samples': self.n_gradient_samples,
            'gradient_estimator': (
                'unbiased (paired)' if self.n_gradient_samples > 0
                else 'single-sample (EconML default)'
            ),
            'treatment_baseline': round(float(t0_raw), 4),
            'treatment_shift': round(float(self._t_std), 4),
        }

        self._mdn = mdn
        self._response_net = response_net
        self._effects = effects
        self._x_means = X_means
        self._x_stds = X_stds

        return CausalResult(
            method='DeepIV (Hartford et al. 2017)',
            estimand='LATE',
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='deepiv',
        )

    def effect(self, t0: float, t1: float, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate E[Y(t1) - Y(t0) | X] for given treatment levels.

        Parameters
        ----------
        t0 : float
            Baseline treatment value (original scale).
        t1 : float
            Counterfactual treatment value (original scale).
        X : np.ndarray, optional
            Covariates. If None, uses training data.

        Returns
        -------
        np.ndarray
            Individual-level treatment effects.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")

        if not hasattr(self, '_response_net'):
            raise ValueError("Model must be fitted first. Call .fit()")

        if X is None:
            cols = self.covariates
            clean = self.data[[self.y, self.treat] + self.instruments + cols].dropna()
            X_raw = clean[cols].values.astype(np.float32)
        else:
            X_raw = np.asarray(X, dtype=np.float32)

        X_s = (X_raw - self._x_means) / self._x_stds

        t0_s = (t0 - self._t_mean) / self._t_std
        t1_s = (t1 - self._t_mean) / self._t_std
        n = len(X_s)

        device = torch.device('cpu')
        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            tv0 = torch.full((n, 1), t0_s, dtype=torch.float32, device=device)
            tv1 = torch.full((n, 1), t1_s, dtype=torch.float32, device=device)
            y0 = self._response_net(torch.cat([tv0, X_t], dim=1)).squeeze().numpy()
            y1 = self._response_net(torch.cat([tv1, X_t], dim=1)).squeeze().numpy()

        return (y1 - y0) * self._y_std


# ---------------------------------------------------------------------------
# Neural network builders (PyTorch)
# ---------------------------------------------------------------------------

def _build_hidden_layers(input_dim: int, hidden_layers: Tuple[int, ...]):
    """Build a sequence of hidden layers with ReLU activations."""
    import torch.nn as nn

    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(h))
        prev = h
    return nn.Sequential(*layers), prev


def _build_mdn(input_dim: int, n_components: int, hidden_layers: Tuple[int, ...]):
    """Build a Gaussian Mixture Density Network for P(T | Z, X)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    hidden_seq, h_dim = _build_hidden_layers(input_dim, hidden_layers)

    class MDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden_seq
            self.pi_head = nn.Linear(h_dim, n_components)
            self.mu_head = nn.Linear(h_dim, n_components)
            self.log_sigma_head = nn.Linear(h_dim, n_components)

        def forward(self, x):
            h = self.hidden(x)
            pi = F.softmax(self.pi_head(h), dim=-1)
            mu = self.mu_head(h)
            sigma = torch.exp(self.log_sigma_head(h)).clamp(min=1e-4)
            return pi, mu, sigma

    return MDN()


def _build_response_net(input_dim: int, hidden_layers: Tuple[int, ...]):
    """Build a feed-forward network h(T, X) -> Y."""
    import torch.nn as nn

    hidden_seq, h_dim = _build_hidden_layers(input_dim, hidden_layers)

    class ResponseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden_seq
            self.output_head = nn.Linear(h_dim, 1)

        def forward(self, x):
            h = self.hidden(x)
            return self.output_head(h)

    return ResponseNet()


def _mdn_loss(pi, mu, sigma, target):
    """Negative log-likelihood for Gaussian MDN."""
    import torch

    target = target.unsqueeze(1)  # (batch, 1)
    log_probs = (
        -0.5 * ((target - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * math.log(2 * math.pi)
    )
    log_pi = torch.log(pi + 1e-8)
    log_sum = torch.logsumexp(log_pi + log_probs, dim=1)
    return -torch.mean(log_sum)


def _sample_mdn(pi, mu, sigma, n_samples: int):
    """Sample from a Gaussian mixture density network."""
    import torch

    # Select which mixture component
    comp_idx = torch.multinomial(pi, n_samples, replacement=True)  # (batch, n_samples)

    # Gather means and sigmas for selected components
    mu_exp = mu.unsqueeze(2).expand(-1, -1, n_samples)
    sigma_exp = sigma.unsqueeze(2).expand(-1, -1, n_samples)
    comp_exp = comp_idx.unsqueeze(1)  # (batch, 1, n_samples)

    mu_sel = torch.gather(mu_exp, 1, comp_exp).squeeze(1)
    sigma_sel = torch.gather(sigma_exp, 1, comp_exp).squeeze(1)

    # Sample from selected Gaussian
    eps = torch.randn_like(mu_sel)
    return mu_sel + sigma_sel * eps


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

CausalResult._CITATIONS['deepiv'] = (
    "@inproceedings{hartford2017deep,\n"
    "  title={Deep IV: A Flexible Approach for Counterfactual Prediction},\n"
    "  author={Hartford, Jason and Lewis, Greg and Leyton-Brown, Kevin "
    "and Taddy, Matt},\n"
    "  booktitle={Proceedings of the 34th International Conference on "
    "Machine Learning},\n"
    "  pages={1414--1423},\n"
    "  year={2017},\n"
    "  organization={PMLR}\n"
    "}"
)
