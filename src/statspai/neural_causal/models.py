"""
Neural Causal Inference Models: TARNet, CFRNet, DragonNet.

Architecture overview
---------------------

**TARNet** (Treatment-Agnostic Representation Network):
    X -> shared_repr(X) = Phi(X) -> { head_0(Phi) if D=0
                                     { head_1(Phi) if D=1
    Loss = factual outcome MSE (observed arm only).
    CATE(x) = head_1(Phi(x)) - head_0(Phi(x)).

**CFRNet** (Counterfactual Regression Network):
    Same architecture as TARNet, but adds an IPM (Integral Probability
    Metric) penalty that forces the learned representation Phi to be
    balanced across treatment arms:
        Loss = MSE + alpha * IPM(Phi_treated, Phi_control)
    where IPM is approximated by MMD with RBF kernel.

**DragonNet** (Targeted Regularisation):
    X -> shared_repr(X) = Phi(X) -> { head_0(Phi) -> mu_0
                                     { head_1(Phi) -> mu_1
                                     { epsilon_head(Phi) -> propensity e(x)
    Loss = MSE + beta * CrossEntropy(e, D) + gamma * targeted_reg
    The propensity head acts as a targeted regulariser, ensuring the
    representation captures treatment-relevant information.
    Targeted regularisation term (Shi et al. 2019):
        t_reg = E[ (Y - D*mu_1 - (1-D)*mu_0) * (D/e - (1-D)/(1-e)) ]^2

Requires PyTorch (optional dependency):
    pip install statspai[neural]  or  pip install torch

References
----------
Shalit, U., Johansson, F. D., & Sontag, D. (2017).
"Estimating individual treatment effect: generalization bounds and algorithms."
Proceedings of the 34th International Conference on Machine Learning. [@shalit2017estimating]

Shi, C., Blei, D. M., & Veitch, V. (2019).
"Adapting neural networks for the estimation of treatment effects."
Advances in Neural Information Processing Systems, 32. [@shi2019adapting]
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# Public API - functional interface
# ======================================================================

def tarnet(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    repr_layers: Tuple[int, ...] = (200, 200),
    head_layers: Tuple[int, ...] = (100,),
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
    verbose: bool = False,
) -> CausalResult:
    """
    Estimate treatment effects using TARNet (Shalit et al. 2017).

    TARNet learns a shared representation of covariates and uses
    separate outcome heads for treated and control groups. It handles
    complex, non-linear relationships between covariates and outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment variable name (0/1).
    covariates : list of str
        Covariate variable names.
    repr_layers : tuple of int, default (200, 200)
        Hidden layer sizes for the shared representation network.
    head_layers : tuple of int, default (100,)
        Hidden layer sizes for each treatment-specific outcome head.
    epochs : int, default 300
        Training epochs.
    batch_size : int, default 256
        Mini-batch size.
    learning_rate : float, default 1e-3
        Adam learning rate.
    weight_decay : float, default 1e-4
        L2 regularisation strength.
    dropout : float, default 0.1
        Dropout rate for representation layers.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 500
        Bootstrap iterations for standard error estimation.
    random_state : int, default 42
        Random seed.
    verbose : bool, default False
        Print training progress.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.tarnet(df, y='outcome', treat='treatment',
    ...                    covariates=['x1', 'x2', 'x3'])
    >>> print(result.summary())

    >>> # Access individual CATE predictions
    >>> cate = result.model_info['cate']
    """
    est = TARNet(
        data=data, y=y, treat=treat, covariates=covariates,
        repr_layers=repr_layers, head_layers=head_layers,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        dropout=dropout, alpha=alpha, n_bootstrap=n_bootstrap,
        random_state=random_state, verbose=verbose,
    )
    return est.fit()


def cfrnet(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    repr_layers: Tuple[int, ...] = (200, 200),
    head_layers: Tuple[int, ...] = (100,),
    ipm_weight: float = 1.0,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
    verbose: bool = False,
) -> CausalResult:
    """
    Estimate treatment effects using CFRNet (Shalit et al. 2017).

    Extends TARNet with an Integral Probability Metric (IPM) penalty
    that encourages the learned representation to be balanced across
    treatment arms. Uses MMD with RBF kernel as the IPM approximation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment variable name (0/1).
    covariates : list of str
        Covariate variable names.
    repr_layers : tuple of int, default (200, 200)
        Hidden layer sizes for the shared representation network.
    head_layers : tuple of int, default (100,)
        Hidden layer sizes for each treatment-specific outcome head.
    ipm_weight : float, default 1.0
        Weight for the IPM regularisation term (alpha in the paper).
        Higher values enforce stronger distributional balance.
    epochs : int, default 300
        Training epochs.
    batch_size : int, default 256
        Mini-batch size.
    learning_rate : float, default 1e-3
        Adam learning rate.
    weight_decay : float, default 1e-4
        L2 regularisation strength.
    dropout : float, default 0.1
        Dropout rate for representation layers.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 500
        Bootstrap iterations for standard error estimation.
    random_state : int, default 42
        Random seed.
    verbose : bool, default False
        Print training progress.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.cfrnet(df, y='outcome', treat='treatment',
    ...                    covariates=['x1', 'x2', 'x3'],
    ...                    ipm_weight=0.5)
    >>> print(result.summary())
    """
    est = CFRNet(
        data=data, y=y, treat=treat, covariates=covariates,
        repr_layers=repr_layers, head_layers=head_layers,
        ipm_weight=ipm_weight,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        dropout=dropout, alpha=alpha, n_bootstrap=n_bootstrap,
        random_state=random_state, verbose=verbose,
    )
    return est.fit()


def dragonnet(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    repr_layers: Tuple[int, ...] = (200, 200),
    head_layers: Tuple[int, ...] = (100,),
    propensity_weight: float = 1.0,
    targeted_reg_weight: float = 1.0,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
    verbose: bool = False,
) -> CausalResult:
    """
    Estimate treatment effects using DragonNet (Shi et al. 2019).

    DragonNet extends TARNet with a propensity score head and optional
    targeted regularisation. The propensity head ensures the shared
    representation captures information relevant for treatment
    assignment, while targeted regularisation improves finite-sample
    efficiency via a TMLE-inspired augmentation term.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment variable name (0/1).
    covariates : list of str
        Covariate variable names.
    repr_layers : tuple of int, default (200, 200)
        Hidden layer sizes for the shared representation network.
    head_layers : tuple of int, default (100,)
        Hidden layer sizes for each outcome head.
    propensity_weight : float, default 1.0
        Weight for the propensity score cross-entropy loss (beta).
    targeted_reg_weight : float, default 1.0
        Weight for the targeted regularisation term (gamma).
        Set to 0.0 to disable targeted regularisation.
    epochs : int, default 300
        Training epochs.
    batch_size : int, default 256
        Mini-batch size.
    learning_rate : float, default 1e-3
        Adam learning rate.
    weight_decay : float, default 1e-4
        L2 regularisation strength.
    dropout : float, default 0.1
        Dropout rate for representation layers.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 500
        Bootstrap iterations for standard error estimation.
    random_state : int, default 42
        Random seed.
    verbose : bool, default False
        Print training progress.

    Returns
    -------
    CausalResult
        Includes AIPW-adjusted ATE using the learned propensity scores
        and outcome predictions.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.dragonnet(df, y='outcome', treat='treatment',
    ...                       covariates=['x1', 'x2', 'x3'])
    >>> print(result.summary())

    >>> # DragonNet without targeted regularisation
    >>> result = sp.dragonnet(df, y='outcome', treat='treatment',
    ...                       covariates=['x1', 'x2'],
    ...                       targeted_reg_weight=0.0)
    """
    est = DragonNet(
        data=data, y=y, treat=treat, covariates=covariates,
        repr_layers=repr_layers, head_layers=head_layers,
        propensity_weight=propensity_weight,
        targeted_reg_weight=targeted_reg_weight,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        dropout=dropout, alpha=alpha, n_bootstrap=n_bootstrap,
        random_state=random_state, verbose=verbose,
    )
    return est.fit()


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data(data, y, treat, covariates):
    """Validate and extract arrays from DataFrame."""
    cols = [y, treat] + covariates
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    clean = data[cols].dropna()
    Y = clean[y].values.astype(np.float32)
    D = clean[treat].values.astype(np.float32)
    X = clean[covariates].values.astype(np.float32)

    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d.astype(int)) == {0, 1}):
        raise ValueError(
            f"Treatment must be binary (0/1), got unique values: {unique_d}"
        )

    return Y, D, X, len(Y)


def _standardise(arr):
    """Standardise array columns to zero mean, unit variance."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return (arr - mean) / std, mean, std


# ======================================================================
# PyTorch network components
# ======================================================================

def _check_torch():
    """Import and return torch, raising clear error if unavailable."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for neural causal models. "
            "Install: pip install statspai[neural]  or: pip install torch"
        )


def _build_repr_net(input_dim, hidden_layers, dropout):
    """Build shared representation network Phi(X)."""
    torch, nn = _check_torch()

    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers), prev


def _build_head(input_dim, hidden_layers):
    """Build a treatment-arm outcome head."""
    torch, nn = _check_torch()

    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ======================================================================
# TARNet
# ======================================================================

class TARNet:
    """
    Treatment-Agnostic Representation Network (Shalit et al. 2017).

    Learns shared representations Phi(X) and separate outcome
    prediction heads for treated (D=1) and control (D=0) groups.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates.
    repr_layers : tuple of int
        Representation network hidden layer sizes.
    head_layers : tuple of int
        Outcome head hidden layer sizes.
    epochs : int
        Training epochs.
    batch_size : int
    learning_rate : float
    weight_decay : float
        L2 regularisation.
    dropout : float
    alpha : float
        Significance level.
    n_bootstrap : int
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        repr_layers: Tuple[int, ...] = (200, 200),
        head_layers: Tuple[int, ...] = (100,),
        epochs: int = 300,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.repr_layers = repr_layers
        self.head_layers = head_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.verbose = verbose

    def fit(self) -> CausalResult:
        """Fit TARNet and return treatment effect estimates."""
        torch, nn = _check_torch()
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Y, D, X, n = _prepare_data(self.data, self.y, self.treat, self.covariates)
        X_s, self._x_mean, self._x_std = _standardise(X)
        Y_s, self._y_mean, self._y_std = _standardise(Y.reshape(-1, 1))
        Y_s = Y_s.ravel()

        device = torch.device('cpu')
        x_dim = X_s.shape[1]

        # Build networks
        repr_net, repr_dim = _build_repr_net(x_dim, self.repr_layers, self.dropout)
        head_0 = _build_head(repr_dim, self.head_layers)
        head_1 = _build_head(repr_dim, self.head_layers)

        repr_net = repr_net.to(device)
        head_0 = head_0.to(device)
        head_1 = head_1.to(device)

        all_params = (list(repr_net.parameters()) +
                      list(head_0.parameters()) +
                      list(head_1.parameters()))
        optimiser = torch.optim.Adam(all_params, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_s, dtype=torch.float32, device=device)
        D_t = torch.tensor(D, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_t, Y_t, D_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        # Training loop
        for epoch in range(self.epochs):
            repr_net.train()
            head_0.train()
            head_1.train()
            epoch_loss = 0.0

            for x_b, y_b, d_b in loader:
                optimiser.zero_grad()
                phi = repr_net(x_b)

                y0_pred = head_0(phi).squeeze()
                y1_pred = head_1(phi).squeeze()

                # Factual loss: use only observed outcome
                y_pred = d_b * y1_pred + (1 - d_b) * y0_pred
                loss = torch.mean((y_b - y_pred) ** 2)

                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(y_b)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  TARNet epoch {epoch+1}/{self.epochs}, "
                      f"loss={epoch_loss/n:.6f}")

        # Predict CATE
        repr_net.eval()
        head_0.eval()
        head_1.eval()

        with torch.no_grad():
            phi = repr_net(X_t)
            mu0 = head_0(phi).squeeze().numpy()
            mu1 = head_1(phi).squeeze().numpy()

        # Rescale to original Y scale
        cate = (mu1 - mu0) * self._y_std[0]
        ate = float(np.mean(cate))

        se = _bootstrap_ate_se(cate, self.n_bootstrap, self.random_state)
        pvalue, ci = _inference(ate, se, self.alpha)

        self._repr_net = repr_net
        self._head_0 = head_0
        self._head_1 = head_1
        self._cate = cate

        model_info = self._build_model_info(cate, D, n)

        return CausalResult(
            method='TARNet (Shalit et al. 2017)',
            estimand='ATE',
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='tarnet',
        )

    def effect(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict CATE for new observations.

        Parameters
        ----------
        X_new : np.ndarray, optional
            New covariates. If None, uses training data.

        Returns
        -------
        np.ndarray
            Individual-level CATE estimates.
        """
        torch, _ = _check_torch()

        if not hasattr(self, '_repr_net'):
            raise ValueError("Model must be fitted first. Call .fit()")

        if X_new is None:
            return self._cate.copy()

        X_new = np.asarray(X_new, dtype=np.float32)
        X_s = (X_new - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)

        self._repr_net.eval()
        self._head_0.eval()
        self._head_1.eval()

        with torch.no_grad():
            phi = self._repr_net(X_t)
            mu0 = self._head_0(phi).squeeze().numpy()
            mu1 = self._head_1(phi).squeeze().numpy()

        return (mu1 - mu0) * self._y_std[0]

    def _build_model_info(self, cate, D, n):
        return {
            'architecture': 'TARNet',
            'repr_layers': self.repr_layers,
            'head_layers': self.head_layers,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'n_covariates': len(self.covariates),
            'covariates': self.covariates,
            'cate': cate,
            'cate_mean': float(np.mean(cate)),
            'cate_median': float(np.median(cate)),
            'cate_std': float(np.std(cate)),
            'cate_q25': float(np.percentile(cate, 25)),
            'cate_q75': float(np.percentile(cate, 75)),
            'n_treated': int(np.sum(D == 1)),
            'n_control': int(np.sum(D == 0)),
        }


# ======================================================================
# CFRNet
# ======================================================================

class CFRNet:
    """
    Counterfactual Regression Network (Shalit et al. 2017).

    Extends TARNet with an IPM (Integral Probability Metric) penalty
    that enforces distributional balance of the learned representation
    across treatment arms. Uses MMD with RBF kernel as the IPM.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
    repr_layers : tuple of int
    head_layers : tuple of int
    ipm_weight : float
        Weight for the IPM regularisation term.
    epochs : int
    batch_size : int
    learning_rate : float
    weight_decay : float
    dropout : float
    alpha : float
    n_bootstrap : int
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        repr_layers: Tuple[int, ...] = (200, 200),
        head_layers: Tuple[int, ...] = (100,),
        ipm_weight: float = 1.0,
        epochs: int = 300,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.repr_layers = repr_layers
        self.head_layers = head_layers
        self.ipm_weight = ipm_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.verbose = verbose

    def fit(self) -> CausalResult:
        """Fit CFRNet and return treatment effect estimates."""
        torch, nn = _check_torch()
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Y, D, X, n = _prepare_data(self.data, self.y, self.treat, self.covariates)
        X_s, self._x_mean, self._x_std = _standardise(X)
        Y_s, self._y_mean, self._y_std = _standardise(Y.reshape(-1, 1))
        Y_s = Y_s.ravel()

        device = torch.device('cpu')
        x_dim = X_s.shape[1]

        repr_net, repr_dim = _build_repr_net(x_dim, self.repr_layers, self.dropout)
        head_0 = _build_head(repr_dim, self.head_layers)
        head_1 = _build_head(repr_dim, self.head_layers)

        repr_net = repr_net.to(device)
        head_0 = head_0.to(device)
        head_1 = head_1.to(device)

        all_params = (list(repr_net.parameters()) +
                      list(head_0.parameters()) +
                      list(head_1.parameters()))
        optimiser = torch.optim.Adam(all_params, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_s, dtype=torch.float32, device=device)
        D_t = torch.tensor(D, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_t, Y_t, D_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        for epoch in range(self.epochs):
            repr_net.train()
            head_0.train()
            head_1.train()
            epoch_loss = 0.0
            epoch_ipm = 0.0

            for x_b, y_b, d_b in loader:
                optimiser.zero_grad()
                phi = repr_net(x_b)

                y0_pred = head_0(phi).squeeze()
                y1_pred = head_1(phi).squeeze()

                y_pred = d_b * y1_pred + (1 - d_b) * y0_pred
                mse_loss = torch.mean((y_b - y_pred) ** 2)

                # IPM: MMD with RBF kernel
                mask1 = d_b == 1
                mask0 = d_b == 0
                if mask1.sum() > 1 and mask0.sum() > 1:
                    ipm = _mmd_rbf(phi[mask1], phi[mask0])
                else:
                    ipm = torch.tensor(0.0, device=device)

                loss = mse_loss + self.ipm_weight * ipm
                loss.backward()
                optimiser.step()
                epoch_loss += mse_loss.item() * len(y_b)
                epoch_ipm += ipm.item() * len(y_b)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  CFRNet epoch {epoch+1}/{self.epochs}, "
                      f"MSE={epoch_loss/n:.6f}, "
                      f"IPM={epoch_ipm/n:.6f}")

        # Predict CATE
        repr_net.eval()
        head_0.eval()
        head_1.eval()

        with torch.no_grad():
            phi = repr_net(X_t)
            mu0 = head_0(phi).squeeze().numpy()
            mu1 = head_1(phi).squeeze().numpy()

        cate = (mu1 - mu0) * self._y_std[0]
        ate = float(np.mean(cate))

        se = _bootstrap_ate_se(cate, self.n_bootstrap, self.random_state)
        pvalue, ci = _inference(ate, se, self.alpha)

        self._repr_net = repr_net
        self._head_0 = head_0
        self._head_1 = head_1
        self._cate = cate

        model_info = {
            'architecture': 'CFRNet',
            'repr_layers': self.repr_layers,
            'head_layers': self.head_layers,
            'ipm_weight': self.ipm_weight,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'n_covariates': len(self.covariates),
            'covariates': self.covariates,
            'cate': cate,
            'cate_mean': float(np.mean(cate)),
            'cate_median': float(np.median(cate)),
            'cate_std': float(np.std(cate)),
            'cate_q25': float(np.percentile(cate, 25)),
            'cate_q75': float(np.percentile(cate, 75)),
            'n_treated': int(np.sum(D == 1)),
            'n_control': int(np.sum(D == 0)),
        }

        return CausalResult(
            method='CFRNet (Shalit et al. 2017)',
            estimand='ATE',
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='cfrnet',
        )

    def effect(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict CATE for new observations."""
        torch, _ = _check_torch()

        if not hasattr(self, '_repr_net'):
            raise ValueError("Model must be fitted first. Call .fit()")

        if X_new is None:
            return self._cate.copy()

        X_new = np.asarray(X_new, dtype=np.float32)
        X_s = (X_new - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)

        self._repr_net.eval()
        self._head_0.eval()
        self._head_1.eval()

        with torch.no_grad():
            phi = self._repr_net(X_t)
            mu0 = self._head_0(phi).squeeze().numpy()
            mu1 = self._head_1(phi).squeeze().numpy()

        return (mu1 - mu0) * self._y_std[0]


# ======================================================================
# DragonNet
# ======================================================================

class DragonNet:
    """
    DragonNet: Targeted Regularisation (Shi et al. 2019).

    Three-headed architecture with shared representation:
    - Head 0: control outcome prediction mu_0(X)
    - Head 1: treated outcome prediction mu_1(X)
    - Propensity head: treatment probability e(X) = P(D=1|X)

    The propensity head acts as a targeted regulariser, ensuring the
    shared representation captures treatment-assignment-relevant
    information. An optional targeted regularisation term further
    improves finite-sample efficiency.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
    repr_layers : tuple of int
    head_layers : tuple of int
    propensity_weight : float
        Weight for propensity cross-entropy loss (beta).
    targeted_reg_weight : float
        Weight for targeted regularisation term (gamma).
    epochs : int
    batch_size : int
    learning_rate : float
    weight_decay : float
    dropout : float
    alpha : float
    n_bootstrap : int
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        repr_layers: Tuple[int, ...] = (200, 200),
        head_layers: Tuple[int, ...] = (100,),
        propensity_weight: float = 1.0,
        targeted_reg_weight: float = 1.0,
        epochs: int = 300,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.repr_layers = repr_layers
        self.head_layers = head_layers
        self.propensity_weight = propensity_weight
        self.targeted_reg_weight = targeted_reg_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.verbose = verbose

    def fit(self) -> CausalResult:
        """Fit DragonNet and return treatment effect estimates."""
        torch, nn = _check_torch()
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Y, D, X, n = _prepare_data(self.data, self.y, self.treat, self.covariates)
        X_s, self._x_mean, self._x_std = _standardise(X)
        Y_s, self._y_mean, self._y_std = _standardise(Y.reshape(-1, 1))
        Y_s = Y_s.ravel()

        device = torch.device('cpu')
        x_dim = X_s.shape[1]

        # Build three-headed network
        repr_net, repr_dim = _build_repr_net(x_dim, self.repr_layers, self.dropout)
        head_0 = _build_head(repr_dim, self.head_layers)
        head_1 = _build_head(repr_dim, self.head_layers)

        # Propensity head: representation -> P(D=1|X)
        prop_layers = []
        prev = repr_dim
        for h in self.head_layers:
            prop_layers.append(nn.Linear(prev, h))
            prop_layers.append(nn.ELU())
            prev = h
        prop_layers.append(nn.Linear(prev, 1))
        prop_layers.append(nn.Sigmoid())
        prop_head = nn.Sequential(*prop_layers)

        repr_net = repr_net.to(device)
        head_0 = head_0.to(device)
        head_1 = head_1.to(device)
        prop_head = prop_head.to(device)

        # Epsilon for targeted regularisation (learnable scalar)
        epsilon = nn.Parameter(torch.zeros(1, device=device))

        all_params = (list(repr_net.parameters()) +
                      list(head_0.parameters()) +
                      list(head_1.parameters()) +
                      list(prop_head.parameters()) +
                      [epsilon])
        optimiser = torch.optim.Adam(all_params, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_s, dtype=torch.float32, device=device)
        D_t = torch.tensor(D, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_t, Y_t, D_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        bce = nn.BCELoss()

        for epoch in range(self.epochs):
            repr_net.train()
            head_0.train()
            head_1.train()
            prop_head.train()
            epoch_mse = 0.0
            epoch_prop = 0.0
            epoch_treg = 0.0

            for x_b, y_b, d_b in loader:
                optimiser.zero_grad()
                phi = repr_net(x_b)

                mu0 = head_0(phi).squeeze()
                mu1 = head_1(phi).squeeze()
                e = prop_head(phi).squeeze().clamp(0.01, 0.99)

                # Factual outcome loss
                y_pred = d_b * mu1 + (1 - d_b) * mu0
                mse_loss = torch.mean((y_b - y_pred) ** 2)

                # Propensity loss
                prop_loss = bce(e, d_b)

                # Targeted regularisation (Shi et al. 2019, Eq. 4)
                if self.targeted_reg_weight > 0:
                    clever_cov = d_b / e - (1 - d_b) / (1 - e)
                    y_targeted = y_pred + epsilon * clever_cov
                    t_reg = torch.mean((y_b - y_targeted) ** 2)
                else:
                    t_reg = torch.tensor(0.0, device=device)

                loss = (mse_loss +
                        self.propensity_weight * prop_loss +
                        self.targeted_reg_weight * t_reg)

                loss.backward()
                optimiser.step()
                epoch_mse += mse_loss.item() * len(y_b)
                epoch_prop += prop_loss.item() * len(y_b)
                epoch_treg += t_reg.item() * len(y_b)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  DragonNet epoch {epoch+1}/{self.epochs}, "
                      f"MSE={epoch_mse/n:.6f}, "
                      f"Prop={epoch_prop/n:.6f}, "
                      f"TReg={epoch_treg/n:.6f}")

        # Predict
        repr_net.eval()
        head_0.eval()
        head_1.eval()
        prop_head.eval()

        with torch.no_grad():
            phi = repr_net(X_t)
            mu0 = head_0(phi).squeeze().numpy()
            mu1 = head_1(phi).squeeze().numpy()
            e_hat = prop_head(phi).squeeze().numpy()
            e_hat = np.clip(e_hat, 0.01, 0.99)

        # Rescale outcomes to original Y scale
        mu0_orig = mu0 * self._y_std[0] + self._y_mean[0]
        mu1_orig = mu1 * self._y_std[0] + self._y_mean[0]

        # AIPW estimator using DragonNet predictions (doubly robust)
        Y_orig = Y
        aipw_scores = (
            (mu1_orig - mu0_orig)
            + D * (Y_orig - mu1_orig) / e_hat
            - (1 - D) * (Y_orig - mu0_orig) / (1 - e_hat)
        )
        ate_aipw = float(np.mean(aipw_scores))

        # Plug-in CATE for individual predictions
        cate = (mu1 - mu0) * self._y_std[0]

        # Standard error from AIPW influence function
        se = float(np.std(aipw_scores, ddof=1) / np.sqrt(n))
        pvalue, ci = _inference(ate_aipw, se, self.alpha)

        self._repr_net = repr_net
        self._head_0 = head_0
        self._head_1 = head_1
        self._prop_head = prop_head
        self._cate = cate
        self._e_hat = e_hat

        model_info = {
            'architecture': 'DragonNet',
            'repr_layers': self.repr_layers,
            'head_layers': self.head_layers,
            'propensity_weight': self.propensity_weight,
            'targeted_reg_weight': self.targeted_reg_weight,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'n_covariates': len(self.covariates),
            'covariates': self.covariates,
            'se_method': 'AIPW_influence_function',
            'ate_plugin': float(np.mean(cate)),
            'ate_aipw': ate_aipw,
            'propensity_mean': float(np.mean(e_hat)),
            'propensity_std': float(np.std(e_hat)),
            'cate': cate,
            'cate_mean': float(np.mean(cate)),
            'cate_median': float(np.median(cate)),
            'cate_std': float(np.std(cate)),
            'cate_q25': float(np.percentile(cate, 25)),
            'cate_q75': float(np.percentile(cate, 75)),
            'n_treated': int(np.sum(D == 1)),
            'n_control': int(np.sum(D == 0)),
        }

        return CausalResult(
            method='DragonNet (Shi et al. 2019)',
            estimand='ATE',
            estimate=ate_aipw,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='dragonnet',
        )

    def effect(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict CATE for new observations."""
        torch, _ = _check_torch()

        if not hasattr(self, '_repr_net'):
            raise ValueError("Model must be fitted first. Call .fit()")

        if X_new is None:
            return self._cate.copy()

        X_new = np.asarray(X_new, dtype=np.float32)
        X_s = (X_new - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)

        self._repr_net.eval()
        self._head_0.eval()
        self._head_1.eval()

        with torch.no_grad():
            phi = self._repr_net(X_t)
            mu0 = self._head_0(phi).squeeze().numpy()
            mu1 = self._head_1(phi).squeeze().numpy()

        return (mu1 - mu0) * self._y_std[0]

    def propensity(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict propensity scores P(D=1|X).

        Parameters
        ----------
        X_new : np.ndarray, optional
            New covariates. If None, uses training data.

        Returns
        -------
        np.ndarray
            Propensity scores.
        """
        torch, _ = _check_torch()

        if not hasattr(self, '_prop_head'):
            raise ValueError("Model must be fitted first. Call .fit()")

        if X_new is None:
            return self._e_hat.copy()

        X_new = np.asarray(X_new, dtype=np.float32)
        X_s = (X_new - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)

        self._repr_net.eval()
        self._prop_head.eval()

        with torch.no_grad():
            phi = self._repr_net(X_t)
            e = self._prop_head(phi).squeeze().numpy()

        return np.clip(e, 0.01, 0.99)


# ======================================================================
# IPM: MMD with RBF kernel
# ======================================================================

def _mmd_rbf(X, Y, bandwidth=None):
    """
    Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    Approximates the IPM between distributions of treated and control
    representations. Uses the median heuristic for bandwidth selection
    when not specified.

    Parameters
    ----------
    X : torch.Tensor
        Representation vectors from group 1 (treated).
    Y : torch.Tensor
        Representation vectors from group 0 (control).
    bandwidth : float, optional
        RBF kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    torch.Tensor
        Scalar MMD^2 estimate.
    """
    import torch

    nx, ny = X.shape[0], Y.shape[0]

    # Bandwidth selection via median heuristic
    if bandwidth is None:
        with torch.no_grad():
            XY = torch.cat([X, Y], dim=0)
            dists = torch.cdist(XY, XY)
            # Use median of non-zero distances
            nonzero_dists = dists[dists > 0]
            if len(nonzero_dists) > 0:
                median_dist = torch.median(nonzero_dists)
                bandwidth = float(median_dist.item()) ** 2
            else:
                bandwidth = 1.0
            bandwidth = max(bandwidth, 1e-6)

    def rbf(A, B):
        dist_sq = torch.cdist(A, B).pow(2)
        return torch.exp(-dist_sq / (2 * bandwidth))

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    # Unbiased MMD^2 estimator
    mmd2 = (Kxx.sum() / max(nx * (nx - 1), 1)
            + Kyy.sum() / max(ny * (ny - 1), 1)
            - 2 * Kxy.sum() / (nx * ny))

    return mmd2


# ======================================================================
# Shared inference helpers
# ======================================================================

def _bootstrap_ate_se(cate, n_bootstrap, random_state):
    """Bootstrap standard error of ATE from individual CATE values."""
    rng = np.random.RandomState(random_state)
    n = len(cate)
    boot_means = np.array([
        rng.choice(cate, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.std(boot_means, ddof=1))


def _inference(estimate, se, alpha):
    """Compute p-value and confidence interval."""
    if se > 0:
        z_stat = estimate / se
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
    else:
        pvalue = 0.0
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    return pvalue, ci


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS['tarnet'] = (
    "@inproceedings{shalit2017estimating,\n"
    "  title={Estimating Individual Treatment Effect: Generalization Bounds "
    "and Algorithms},\n"
    "  author={Shalit, Uri and Johansson, Fredrik D and Sontag, David},\n"
    "  booktitle={Proceedings of the 34th International Conference on "
    "Machine Learning},\n"
    "  pages={3076--3085},\n"
    "  year={2017},\n"
    "  organization={PMLR}\n"
    "}"
)

CausalResult._CITATIONS['cfrnet'] = (
    "@inproceedings{shalit2017estimating,\n"
    "  title={Estimating Individual Treatment Effect: Generalization Bounds "
    "and Algorithms},\n"
    "  author={Shalit, Uri and Johansson, Fredrik D and Sontag, David},\n"
    "  booktitle={Proceedings of the 34th International Conference on "
    "Machine Learning},\n"
    "  pages={3076--3085},\n"
    "  year={2017},\n"
    "  organization={PMLR}\n"
    "}"
)

CausalResult._CITATIONS['dragonnet'] = (
    "@inproceedings{shi2019adapting,\n"
    "  title={Adapting Neural Networks for the Estimation of Treatment Effects},\n"
    "  author={Shi, Claudia and Blei, David M and Veitch, Victor},\n"
    "  booktitle={Advances in Neural Information Processing Systems},\n"
    "  volume={32},\n"
    "  year={2019}\n"
    "}"
)
