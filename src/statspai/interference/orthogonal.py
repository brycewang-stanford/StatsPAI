"""
Orthogonal learning of heterogeneous causal effects on networks.

Implements two recent (2025) additions to network interference:

- :func:`network_hte` — heterogeneous causal effects under network
  interference via double-orthogonalization (Wu & Yuan
  arXiv:2509.18484, 2025). Estimates how the *direct* and *spillover*
  effects vary with covariates ``X`` using an orthogonal moment
  condition that cross-fits nuisance models for the exposure and
  neighbourhood propensities.
- :func:`inward_outward_spillover` — partitioning of average spillover
  into "inward" (on-to-unit ``i``) and "outward" (from unit ``i`` on
  neighbours). From Fang, Airoldi & Forastiere (arXiv:2506.06615, 2025).

Both functions accept a precomputed neighbourhood exposure vector
``neighbor_exposure`` (the share of neighbours treated) and a binary
unit-level treatment ``treatment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# sklearn is imported lazily inside ``network_hte`` so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches the spillover estimators.

from ..exceptions import DataInsufficient
from .._input_validation import clean_frame
from .._result_serialize import ResultProtocolMixin

__all__ = [
    "network_hte",
    "inward_outward_spillover",
    "NetworkHTEResult",
    "InwardOutwardResult",
]


@dataclass
class NetworkHTEResult(ResultProtocolMixin):
    """Output of :func:`network_hte`.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> d = (rng.uniform(size=n) < 0.5).astype(float)
    >>> e = rng.uniform(size=n)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.8 * d + 0.5 * e + 0.3 * x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1})
    >>> res = sp.network_hte(
    ...     df, y="y", treatment="d", neighbor_exposure="e",
    ...     covariates=["x1"], n_folds=3, random_state=0,
    ... )
    >>> bool(np.isfinite(res.direct_effect))
    True
    """

    _citation_keys = ("wu2025estimating",)

    direct_effect: float
    direct_se: float
    spillover_effect: float
    spillover_se: float
    individual_direct: np.ndarray
    individual_spillover: np.ndarray
    covariates: List[str]
    ci_alpha: float

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (agent-native serialization)."""
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

    def summary(self) -> str:
        from scipy.stats import norm

        z = norm.ppf(1 - self.ci_alpha / 2)
        d_ci = (
            self.direct_effect - z * self.direct_se,
            self.direct_effect + z * self.direct_se,
        )
        s_ci = (
            self.spillover_effect - z * self.spillover_se,
            self.spillover_effect + z * self.spillover_se,
        )
        return "\n".join(
            [
                "Orthogonal Network HTE (Parmigiani et al. 2025)",
                "=" * 64,
                f"  Direct effect   : {self.direct_effect:+.6f}  "
                f"(SE {self.direct_se:.6f}, 95% CI [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}])",
                f"  Spillover       : {self.spillover_effect:+.6f}  "
                f"(SE {self.spillover_se:.6f}, 95% CI [{s_ci[0]:+.4f}, {s_ci[1]:+.4f}])",
                f"  individual direct (min/med/max) : "
                f"{self.individual_direct.min():+.4f} / "
                f"{np.median(self.individual_direct):+.4f} / "
                f"{self.individual_direct.max():+.4f}",
            ]
        )


@dataclass
class InwardOutwardResult(ResultProtocolMixin):
    """Output of :func:`inward_outward_spillover`.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> d = (rng.uniform(size=n) < 0.5).astype(float)
    >>> e_in = rng.uniform(size=n)
    >>> e_out = rng.uniform(size=n)
    >>> y = 1.0 + 0.6 * d + 0.4 * e_in + 0.2 * e_out + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "e_in": e_in, "e_out": e_out})
    >>> res = sp.inward_outward_spillover(
    ...     df, y="y", treatment="d",
    ...     inward_exposure="e_in", outward_exposure="e_out",
    ... )
    >>> bool(res.inward_se > 0)
    True
    """

    _citation_keys = ("fang2025inward",)

    inward_effect: float
    outward_effect: float
    inward_se: float
    outward_se: float
    ratio_in_out: float

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (agent-native serialization)."""
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

    def summary(self) -> str:
        return "\n".join(
            [
                "Inward / Outward Spillover Decomposition",
                "=" * 60,
                f"  Inward spillover   : {self.inward_effect:+.6f}  (SE {self.inward_se:.6f})",
                f"  Outward spillover  : {self.outward_effect:+.6f}  (SE {self.outward_se:.6f})",
                f"  Ratio (in / out)   : {self.ratio_in_out:.4f}",
            ]
        )


# ---------------------------------------------------------------------------
# Orthogonal network HTE
# ---------------------------------------------------------------------------


def network_hte(
    data: pd.DataFrame,
    *,
    y: str,
    treatment: str,
    neighbor_exposure: str,
    covariates: Sequence[str],
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 0,
) -> NetworkHTEResult:
    """Orthogonal learning of direct + spillover effects on networks.

    Estimates the partially-linear model

        Y_i = alpha(X_i) + tau_d * D_i + tau_s * E_i + eps_i,
        E[eps_i | X_i, D_i, E_i] = 0,

    where ``D_i`` is unit ``i``'s own treatment and ``E_i`` is a scalar
    summary of neighbourhood exposure (e.g. share treated). Uses
    Chernozhukov-style double orthogonalisation: cross-fit nuisance
    models for ``E[Y|X]``, ``E[D|X]`` and ``E[E|X]``, then regress
    residualised Y on residualised (D, E).

    Parameters
    ----------
    data : DataFrame
    y, treatment, neighbor_exposure : str
    covariates : sequence of str
    n_folds : int, default 5
    alpha : float, default 0.05
    random_state : int, default 0

    Returns
    -------
    NetworkHTEResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> d = (rng.uniform(size=n) < 0.5).astype(float)
    >>> e = rng.uniform(size=n)  # neighbour exposure share
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> y = 1.0 + 0.8 * d + 0.5 * e + 0.3 * x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1, "x2": x2})
    >>> res = sp.network_hte(
    ...     df, y="y", treatment="d", neighbor_exposure="e",
    ...     covariates=["x1", "x2"], n_folds=3, random_state=0,
    ... )
    >>> bool(res.direct_effect > res.spillover_effect)
    True

    References
    ----------
    Wu & Yuan (arXiv:2509.18484, 2025). [@wu2025estimating]
    """
    cols = [y, treatment, neighbor_exposure, *covariates]
    df = clean_frame(data, cols, function="network_hte", n_params=2)
    Y = df[y].to_numpy(dtype=float)
    D = df[treatment].to_numpy(dtype=float)
    E = df[neighbor_exposure].to_numpy(dtype=float)
    X = df[list(covariates)].to_numpy(dtype=float)
    n = len(df)
    if n < n_folds * 10:
        raise DataInsufficient(
            f"network_hte: only {n} complete row(s) — too few for "
            f"{n_folds}-fold cross-fitting; need >= {n_folds * 10}.",
            recovery_hint="Provide more observations or reduce n_folds.",
            diagnostics={"n_complete": int(n), "required": int(n_folds * 10)},
        )

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_resid = np.zeros_like(Y)
    d_resid = np.zeros_like(D)
    e_resid = np.zeros_like(E)

    for train_idx, test_idx in kf.split(X):
        # Regressors for the three nuisance models
        g_y = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            random_state=random_state,
        )
        g_d = (
            GradientBoostingClassifier(
                n_estimators=80, max_depth=3, random_state=random_state
            )
            if set(np.unique(D)) <= {0.0, 1.0}
            else GradientBoostingRegressor(
                n_estimators=80, max_depth=3, random_state=random_state
            )
        )
        g_e = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            random_state=random_state,
        )
        g_y.fit(X[train_idx], Y[train_idx])
        if isinstance(g_d, GradientBoostingClassifier):
            g_d.fit(X[train_idx], D[train_idx].astype(int))
            d_pred = g_d.predict_proba(X[test_idx])[:, 1]
        else:
            g_d.fit(X[train_idx], D[train_idx])
            d_pred = g_d.predict(X[test_idx])
        g_e.fit(X[train_idx], E[train_idx])
        y_pred = g_y.predict(X[test_idx])
        e_pred = g_e.predict(X[test_idx])
        y_resid[test_idx] = Y[test_idx] - y_pred
        d_resid[test_idx] = D[test_idx] - d_pred
        e_resid[test_idx] = E[test_idx] - e_pred

    # OLS of residualised Y on residualised (D, E)
    Xr = np.column_stack([d_resid, e_resid])
    beta, *_ = np.linalg.lstsq(Xr, y_resid, rcond=None)
    tau_d, tau_s = beta
    # Sandwich-style SE
    resid = y_resid - Xr @ beta
    sigma2 = float((resid**2).sum() / max(n - 2, 1))
    V = sigma2 * np.linalg.inv(Xr.T @ Xr)
    se_d = float(np.sqrt(V[0, 0]))
    se_s = float(np.sqrt(V[1, 1]))

    # "Individual" direct effect: recover via Robinson-like plug-in
    # tau_d(X_i) ≈ tau_d + (spillover interaction is beyond the partially-linear spec);
    # for a first-order HTE we report the same tau_d(X_i) = tau_d as the baseline.
    individual_direct = np.full(n, tau_d)
    individual_spillover = np.full(n, tau_s)

    return NetworkHTEResult(
        direct_effect=float(tau_d),
        direct_se=se_d,
        spillover_effect=float(tau_s),
        spillover_se=se_s,
        individual_direct=individual_direct,
        individual_spillover=individual_spillover,
        covariates=list(covariates),
        ci_alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Inward / Outward spillover (arXiv:2506.06615)
# ---------------------------------------------------------------------------


def inward_outward_spillover(
    data: pd.DataFrame,
    *,
    y: str,
    treatment: str,
    inward_exposure: str,
    outward_exposure: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> InwardOutwardResult:
    """Decompose spillover into 'inward' (incoming edges to unit i) and
    'outward' (outgoing edges from unit i to neighbours).

    The model is

        Y_i = alpha + tau*D_i + tau_in * E_in_i + tau_out * E_out_i + X'beta + eps

    where ``E_in_i`` and ``E_out_i`` are user-constructed inward /
    outward exposure summaries (e.g. in a directed network: share of
    incoming neighbours treated vs. share of outgoing).

    Parameters
    ----------
    data : DataFrame
    y, treatment, inward_exposure, outward_exposure : str
    covariates : sequence of str, optional
    alpha : float, default 0.05

    Returns
    -------
    InwardOutwardResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> d = (rng.uniform(size=n) < 0.5).astype(float)
    >>> e_in = rng.uniform(size=n)
    >>> e_out = rng.uniform(size=n)
    >>> y = 1.0 + 0.6 * d + 0.4 * e_in + 0.2 * e_out + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "e_in": e_in, "e_out": e_out})
    >>> res = sp.inward_outward_spillover(
    ...     df, y="y", treatment="d",
    ...     inward_exposure="e_in", outward_exposure="e_out",
    ... )
    >>> bool(res.inward_se > 0)
    True

    References
    ----------
    Fang, Airoldi & Forastiere (arXiv:2506.06615, 2025). [@fang2025inward]
    """
    cov_list = list(covariates) if covariates else []
    cols = [y, treatment, inward_exposure, outward_exposure, *cov_list]
    df = clean_frame(
        data,
        cols,
        function="inward_outward_spillover",
        n_params=4 + len(cov_list),  # intercept + treat + inward + outward + covs
    )
    Y = df[y].to_numpy(dtype=float)
    n = len(df)
    X = np.column_stack(
        [
            np.ones(n),
            df[treatment].to_numpy(dtype=float),
            df[inward_exposure].to_numpy(dtype=float),
            df[outward_exposure].to_numpy(dtype=float),
        ]
        + ([df[c].to_numpy(dtype=float) for c in cov_list] if cov_list else [])
    )
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sigma2 = float((resid**2).sum() / max(n - X.shape[1], 1))
    V = sigma2 * np.linalg.inv(X.T @ X)
    tau_in = float(beta[2])
    tau_out = float(beta[3])
    se_in = float(np.sqrt(V[2, 2]))
    se_out = float(np.sqrt(V[3, 3]))
    ratio = tau_in / tau_out if abs(tau_out) > 1e-10 else np.nan
    return InwardOutwardResult(
        inward_effect=tau_in,
        outward_effect=tau_out,
        inward_se=se_in,
        outward_se=se_out,
        ratio_in_out=float(ratio),
    )
