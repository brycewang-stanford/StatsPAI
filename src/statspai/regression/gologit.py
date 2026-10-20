"""Generalized ordered logit / probit (the model of Stata's ``gologit2``).

For an ordinal outcome with categories ``c_1 < ... < c_M``,

    Pr(Y > c_j | x) = F(a_j + x'b_j),   j = 1, ..., M - 1,

so ``Pr(Y = c_1) = 1 - F_1``, ``Pr(Y = c_j) = F_{j-1} - F_j`` and
``Pr(Y = c_M) = F_{M-1}``. With every ``b_j`` equal this is the ordered logit
(``a_j = -cut_j``); with none equal it is the unconstrained generalized model;
in between (``pl=`` / ``npl=``) the partial proportional odds model. The
unconstrained model can predict negative probabilities; the fit then warns,
as gologit2 does.

Parameters are estimated by maximum likelihood over the free coefficients
(BFGS, then Newton on complex-step scores); the reported coefficients repeat
a constrained variable's coefficient in every equation and the covariance is
mapped accordingly, which is gologit2's ``e(b)`` / ``e(V)`` layout.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core._vcov import ml_vcov
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning
from ._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    require_two_clusters,
    robust_convergence,
    se_from_vcov,
)
from .multinomial import outcome_categories

__all__ = ["gologit"]


def _cdf(link: str, z: Any) -> Any:
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-z))
    return special.ndtr(z)


@records_call("gologit")
@accepts_aliases(robust="vce", covariates="x")
def gologit(
    data: pd.DataFrame,
    y: Optional[str] = None,
    x: Union[str, Sequence[str], None] = None,
    pl: Union[bool, str, Sequence[str], None] = None,
    npl: Union[bool, str, Sequence[str], None] = None,
    link: str = "logit",
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    *,
    formula: Optional[str] = None,
) -> EconometricResults:
    """
    Generalized ordered logit (or probit), with partial proportional odds.

    Equivalent to Stata's user-written ``gologit2 y x, pl(...) npl(...)
    link()``: ``Pr(Y > c_j) = F(a_j + x'b_j)`` with a separate coefficient
    vector for each cut of the outcome, or a common one for the variables
    that meet the parallel-lines assumption.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Ordinal outcome (at least three categories, ordered by value).
    x : str or list of str
        Regressors (a constant per equation is added).
    pl : bool or list of str, optional
        ``True`` constrains every variable to parallel lines (the ordered
        model); a list constrains those variables only.
    npl : bool or list of str, optional
        ``True`` (the default when neither is given) frees every variable; a
        list frees those variables and constrains the others.
    link : {'logit', 'probit'}, default 'logit'
    vce : {None, 'oim', 'robust', 'cluster'}, optional
        Observed information (default), sandwich or cluster sandwich.
    cluster : str, optional
    alpha : float, default 0.05
    formula : str, optional
        ``'y ~ x1 + C(g)'`` instead of ``y`` / ``x``.

    Returns
    -------
    EconometricResults
        ``params`` named ``'<category>:<variable>'`` and
        ``'<category>:_cons'`` for every category but the last (equation
        ``j`` models ``Pr(Y > c_j)``), constrained coefficients repeated.
        ``model_info`` lists the parallel and free variables, the
        log-likelihood and the number of rows with a negative predicted
        probability. ``sp.margins`` reports ``Pr(Y = c)`` for every outcome.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=600), "w": rng.normal(size=600)})
    >>> latent = df["x"] - 0.5 * df["w"] + rng.logistic(size=600)
    >>> df["y"] = np.digitize(latent, [-1, 0, 1])
    >>> fit = sp.gologit(df, y="y", x=["x", "w"], npl=["x"])
    >>> sorted(fit.model_info["parallel"])
    ['w']
    """
    link = str(link).lower()
    if link not in ("logit", "probit"):
        raise MethodIncompatibility(
            f"gologit: link must be 'logit' or 'probit'; got {link!r}.",
        )
    from ..core._vcov_spec import parse_se_request

    req = parse_se_request(
        vce, cluster, function="gologit", supported=("nonrobust", "robust", "cluster")
    )
    se_kind, cluster = req.kind, req.cluster
    if formula is not None:
        if y is not None or x is not None:
            raise MethodIncompatibility(
                "gologit: pass a formula or y= and x=, not both."
            )
        from ..core.utils import formula_to_columns

        data, y, x = formula_to_columns(formula, data)
    if y is None or x is None:
        raise MethodIncompatibility(
            "gologit: the outcome and the regressors are needed.",
            recovery_hint="Pass formula='y ~ x1 + x2', or y= and x=.",
        )
    xs = [x] if isinstance(x, str) else list(x)
    if pl is not None and npl is not None:
        raise MethodIncompatibility("gologit: pass pl= or npl=, not both.")

    def _names(v: Any) -> List[str]:
        return [v] if isinstance(v, str) else list(v)

    if pl is True:
        parallel = list(xs)
    elif pl not in (None, False):
        parallel = _names(pl)
    elif npl not in (None, True, False):
        free = _names(npl)
        parallel = [v for v in xs if v not in free]
    else:
        parallel = []
    unknown = [v for v in parallel if v not in xs]
    if unknown or (npl not in (None, True, False) and set(_names(npl)) - set(xs)):
        raise MethodIncompatibility(
            "gologit: pl= / npl= must name regressors of the model.",
            diagnostics={"unknown": unknown or sorted(set(_names(npl)) - set(xs))},
        )
    cols = [y] + xs + ([cluster] if cluster else [])
    missing = [c for c in cols if c not in data.columns]
    if missing:
        from ..exceptions import ColumnNotFound

        raise ColumnNotFound.from_missing(
            missing, list(data.columns), context="gologit"
        )
    df = data.loc[data[cols].notna().all(axis=1)]
    yv = df[y].to_numpy()
    cats = outcome_categories(df[y])
    M = len(cats)
    if M < 3:
        raise DataInsufficient(
            f"gologit: the outcome needs at least three categories; got {M}.",
            recovery_hint="With two categories use sp.logit / sp.probit.",
        )
    yi = pd.Index(cats).get_indexer(yv)
    n = len(df)
    X = (
        np.column_stack([df[v].to_numpy(dtype=float) for v in xs])
        if xs
        else np.zeros((n, 0))
    )
    p = X.shape[1]
    J = M - 1
    par_idx = [xs.index(v) for v in parallel]
    free_idx = [j for j in range(p) if j not in par_idx]
    # theta = [a_1..a_J, b_parallel, b_free(eq 1), ..., b_free(eq J)]
    n_theta = J + len(par_idx) + J * len(free_idx)
    # A maps theta to the reported (J x (p + 1)) layout, row-major by equation
    A = np.zeros((J * (p + 1), n_theta))
    for j in range(J):
        base = j * (p + 1)
        for q, col in enumerate(par_idx):
            A[base + col, J + q] = 1.0
        for q, col in enumerate(free_idx):
            A[base + col, J + len(par_idx) + j * len(free_idx) + q] = 1.0
        A[base + p, j] = 1.0
    onehot = np.zeros((n, M))
    onehot[np.arange(n), yi] = 1.0

    def probs(theta: Any) -> Any:
        coef = (A @ theta).reshape(J, p + 1)
        Fj = _cdf(link, X @ coef[:, :p].T + coef[:, p][None, :])  # (n, J)
        one = np.ones((n, 1), dtype=Fj.dtype)
        zero = np.zeros((n, 1), dtype=Fj.dtype)
        cum = np.concatenate([one, Fj, zero], axis=1)
        return cum[:, :-1] - cum[:, 1:]

    def obs_loglik(theta: Any) -> Any:
        P = probs(theta)
        return np.log(np.sum(P * onehot, axis=1))

    def neg_total(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = -float(np.sum(obs_loglik(theta)))
        return v if np.isfinite(v) else 1e300

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        g: np.ndarray = -complex_step_scores(obs_loglik, theta).sum(axis=0)
        return g

    share = np.array([(yi > j).mean() for j in range(J)])
    share = np.clip(share, 1e-3, 1 - 1e-3)
    a0 = np.log(share / (1 - share)) if link == "logit" else stats.norm.ppf(share)
    theta0 = np.r_[a0, np.zeros(n_theta - J)]
    opt = optimize.minimize(
        neg_total, theta0, jac=neg_grad, method="BFGS", options={"maxiter": 3000}
    )
    converged, _ = robust_convergence(opt)
    theta, scores, H, _ = ml_newton_polish(obs_loglik, opt.x)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    clusters = df[cluster].to_numpy() if se_kind == "cluster" else None
    if clusters is not None:
        require_two_clusters(clusters, "gologit")
    V_t = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    b = A @ theta
    V = A @ V_t @ A.T
    names = [f"{_label(cats[j])}:{v}" for j in range(J) for v in xs + ["_cons"]]
    P_hat = probs(theta)
    n_neg = int(np.sum(np.any(P_hat < 0, axis=1)))
    if n_neg:
        warnings.warn(
            StatsPAIWarning(
                f"gologit: {n_neg} row(s) have a negative predicted "
                "probability; the unconstrained model does not keep the "
                "cumulative probabilities ordered (gologit2 warns too).",
                recovery_hint="Constrain variables to parallel lines (pl= / "
                "npl=), or check for sparse outcome categories.",
                diagnostics={"n_negative": n_neg},
            ),
            stacklevel=2,
        )
    info: Dict[str, Any] = {
        "model_type": f"Generalized ordered {link}",
        "link": link,
        "categories": [c.item() if hasattr(c, "item") else c for c in cats],
        "parallel": parallel,
        "free": [xs[j] for j in free_idx],
        "log_likelihood": float(np.sum(obs_loglik(theta))),
        "n_negative_probabilities": n_neg,
        "converged": bool(converged or grad_norm < 1e-6),
        "gradient_norm": grad_norm,
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "alpha": alpha,
    }
    res = EconometricResults(
        params=pd.Series(b, index=names),
        std_errors=pd.Series(se_from_vcov(V), index=names),
        model_info=info,
        data_info={
            "nobs": n,
            "n_obs": n,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "df_resid": n - n_theta,
            "inference": "z",
        },
    )
    res._ml_state = {
        "model": "gologit",
        "theta": np.asarray(b, dtype=float),
        "vcov": np.asarray(V, dtype=float),
        "terms": list(xs) + ["_cons"],
        "formula": formula,
        "categories": info["categories"],
        "link": link,
        "y": y,
        "nobs": n,
    }
    return res


def _label(c: Any) -> str:
    if isinstance(c, (float, np.floating)) and float(c).is_integer():
        return str(int(c))
    return str(c)
