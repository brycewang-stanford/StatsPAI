"""
Nested logit, in the form that is consistent with random utility.

Alternatives are grouped into nests. With ``V_j`` the systematic utility
of alternative ``j`` in nest ``k`` and ``lambda_k`` the nest's
dissimilarity parameter,

    I_k      = log sum_{l in k} exp(V_l / lambda_k)
    P(j | k) = exp(V_j / lambda_k) / exp(I_k)
    P(k)     = exp(lambda_k I_k) / sum_m exp(lambda_m I_m)

``lambda_k = 1`` for every nest is the conditional logit; a value between
0 and 1 means the unobserved utilities within the nest are correlated
(``1 - lambda_k^2`` is roughly that correlation), which relaxes the
independence of irrelevant alternatives across nests.

References
----------
[@heiss2002structural]
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._aliases import accepts_aliases
from ..core._vcov import ml_vcov
from ..core.results import CausalResult, EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

__all__ = ["nlogit"]


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@accepts_aliases(robust="vce", covariates="x")
def nlogit(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    chid: Optional[str] = None,
    alt: Optional[str] = None,
    nests: Optional[Mapping[str, Sequence[Any]]] = None,
    constants: bool = True,
    common_lambda: bool = False,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Nested logit (random-utility consistent), by maximum likelihood.

    Equivalent to Stata's ``nlogit`` with a two-level tree and to R's
    ``mlogit::mlogit(nests = ...)``.

    Parameters
    ----------
    data : pd.DataFrame
        Long format: one row per choice situation and alternative.
    y : str
        1 for the chosen alternative, 0 otherwise; exactly one 1 per
        choice situation.
    x : str or list of str, optional
        Variables that vary across alternatives, with one coefficient
        each.
    chid : str
        Choice-situation identifier.
    alt : str
        Alternative identifier. Every choice situation must offer every
        alternative.
    nests : dict
        ``{nest name: [alternatives]}``. Every alternative belongs to
        exactly one nest. A nest with a single alternative has no
        dissimilarity parameter (it is fixed at 1).
    constants : bool, default True
        Alternative-specific constants, the first alternative (sorted)
        as base.
    common_lambda : bool, default False
        One dissimilarity parameter for all nests.
    vce : {None, 'robust', 'cluster'}, optional
        Observed information (default), sandwich, or cluster sandwich.
        Scores are summed within a choice situation first.
    cluster : str, optional
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params``: the coefficients of ``x``, the constants
        (``_cons:<alt>``) and the dissimilarity parameters
        (``lambda:<nest>``). ``model_info`` carries the log-likelihood,
        the conditional-logit log-likelihood and the likelihood-ratio
        test that every ``lambda`` equals 1 (``lr_iia_chi2``,
        ``lr_iia_pvalue``).

    Notes
    -----
    A dissimilarity parameter above 1 is not consistent with random
    utility maximisation for all values of the regressors; the estimate
    is returned and a warning says so.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, modes = 400, ["air", "bus", "car", "train"]
    >>> df = pd.DataFrame({"trip": np.repeat(np.arange(n), 4), "mode": modes * n})
    >>> df["cost"] = rng.normal(size=len(df))
    >>> utility = -0.8 * df["cost"].to_numpy() + rng.gumbel(size=len(df))
    >>> best = utility.reshape(n, 4).argmax(axis=1)
    >>> df["chosen"] = (np.tile(np.arange(4), n) == np.repeat(best, 4)).astype(int)
    >>> res = sp.nlogit(df, y="chosen", x=["cost"], chid="trip", alt="mode",
    ...                 nests={"ground": ["bus", "car", "train"], "air": ["air"]})
    >>> res.model_info["lr_iia_pvalue"]  # doctest: +SKIP

    References
    ----------
    [@heiss2002structural]
    """
    from ..core._vcov_spec import parse_se_request

    req = parse_se_request(
        vce, cluster, function="nlogit", supported=("nonrobust", "robust", "cluster")
    )
    se_kind, cl = req.kind, req.cluster
    xs = _as_list(x)
    if chid is None or alt is None or not nests:
        raise MethodIncompatibility("nlogit: chid=, alt= and nests= are required.")
    if not xs and not constants:
        raise MethodIncompatibility(
            "nlogit: nothing to estimate; give x= or keep constants=True."
        )
    extra = [cl] if isinstance(cl, str) else []
    cols = [y, chid, alt] + xs + extra
    missing = [c for c in cols if c not in data]
    if missing:
        raise MethodIncompatibility(
            f"nlogit: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    df = data[list(dict.fromkeys(cols))].dropna().sort_values([chid, alt])
    alts = sorted(pd.unique(df[alt]).tolist())
    J = len(alts)
    member: Dict[Any, str] = {}
    for name, items in nests.items():
        for a in items:
            if a in member:
                raise MethodIncompatibility(
                    f"nlogit: alternative {a!r} is listed in two nests."
                )
            member[a] = str(name)
    unknown = [a for a in member if a not in alts]
    orphan = [a for a in alts if a not in member]
    if unknown or orphan:
        raise MethodIncompatibility(
            "nlogit: nests= and the alternatives in the data do not match "
            f"(not in data: {unknown}; in no nest: {orphan}).",
            diagnostics={"unknown": unknown, "orphan": orphan},
        )
    sizes = df.groupby(chid, sort=False)[alt].agg(["size", "nunique"])
    if not (sizes["size"].eq(J).all() and sizes["nunique"].eq(J).all()):
        raise MethodIncompatibility(
            "nlogit: every choice situation must list each of the "
            f"{J} alternatives exactly once.",
            diagnostics={"n_incomplete": int((sizes["size"] != J).sum())},
        )
    n = len(sizes)
    Y = df[y].to_numpy(dtype=float).reshape(n, J)
    if not (np.all(np.isin(Y, (0.0, 1.0))) and np.all(Y.sum(axis=1) == 1.0)):
        raise MethodIncompatibility(
            f"nlogit: y={y!r} must mark exactly one chosen alternative per "
            "choice situation."
        )
    X = (
        df[xs].to_numpy(dtype=float).reshape(n, J, len(xs))
        if xs
        else np.zeros((n, J, 0))
    )
    kx = len(xs)
    kc = J - 1 if constants else 0
    nest_names = [str(k) for k in nests]
    nest_of = np.array([nest_names.index(member[a]) for a in alts])
    free = [k for k, name in enumerate(nest_names) if np.sum(nest_of == k) > 1]
    if not free:
        raise MethodIncompatibility(
            "nlogit: every nest has a single alternative, which is the "
            "conditional logit (sp.clogit)."
        )
    k_lam = 1 if common_lambda else len(free)
    k_total = kx + kc + k_lam
    if n <= k_total:
        raise DataInsufficient(
            f"nlogit: {n} choice situations for {k_total} parameters."
        )
    chosen = np.argmax(Y, axis=1)
    rows = np.arange(n)
    onehot = np.stack([(nest_of == k).astype(float) for k in range(len(nest_names))])

    def utilities(theta: np.ndarray) -> np.ndarray:
        V = X @ theta[:kx] if kx else np.zeros((n, J), dtype=theta.dtype)
        if kc:
            V = V + np.concatenate(
                [np.zeros(1, dtype=theta.dtype), theta[kx : kx + kc]]
            )
        return V

    def lambdas(theta: np.ndarray) -> np.ndarray:
        lam = np.ones(len(nest_names), dtype=theta.dtype)
        tail = theta[kx + kc :]
        for i, k in enumerate(free):
            lam[k] = tail[0] if common_lambda else tail[i]
        return lam

    def obs(theta: np.ndarray, nested: bool = True) -> np.ndarray:
        theta = np.asarray(theta)
        V = utilities(theta)
        if not nested:
            top = np.max(np.real(V), axis=1, keepdims=True)
            return np.asarray(
                V[rows, chosen] - top[:, 0] - np.log(np.sum(np.exp(V - top), axis=1))
            )
        lam = lambdas(theta)
        S = V / lam[nest_of][None, :]
        top = np.max(np.real(S), axis=1, keepdims=True)
        E = np.exp(S - top)
        # Inclusive values, one per nest (the shared shift is added back).
        IV = np.log(E @ onehot.T) + top
        W = lam[None, :] * IV
        wtop = np.max(np.real(W), axis=1, keepdims=True)
        log_den = wtop[:, 0] + np.log(np.sum(np.exp(W - wtop), axis=1))
        kj = nest_of[chosen]
        return np.asarray(
            S[rows, chosen] - IV[rows, kj] + lam[kj] * IV[rows, kj] - log_den
        )

    def maximise(fn: Any, start: np.ndarray) -> Any:
        def neg(t: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                v = float(np.sum(np.real(fn(t))))
            return -v if np.isfinite(v) else 1e300

        res = optimize.minimize(neg, start, method="BFGS", options={"gtol": 1e-6})
        return ml_newton_polish(fn, np.asarray(res.x, dtype=float))

    # Conditional logit first: the start, and the null of the IIA test.
    k_cl = kx + kc
    th_cl, _, _, _ = maximise(lambda t: obs(t, nested=False), np.zeros(k_cl))
    ll_cl = float(np.sum(np.real(obs(th_cl, nested=False))))
    start = np.concatenate([th_cl, np.ones(k_lam)])
    theta, scores, H, _ = maximise(obs, start)
    ll = float(np.sum(np.real(obs(theta))))
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    clusters = df[cl].to_numpy().reshape(n, J)[:, 0] if se_kind == "cluster" else None
    V = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se = se_from_vcov(V)
    lam_names = (
        ["lambda"] if common_lambda else [f"lambda:{nest_names[k]}" for k in free]
    )
    names = xs + [f"_cons:{a}" for a in alts[1:]] * bool(kc) + lam_names
    lam_hat = theta[kx + kc :]
    if np.any(lam_hat > 1.0) or np.any(lam_hat <= 0.0):
        import warnings

        from ..exceptions import AssumptionWarning

        warnings.warn(
            "nlogit: a dissimilarity parameter is outside (0, 1] "
            f"({dict(zip(lam_names, np.round(lam_hat, 3)))}); the fitted "
            "model is not consistent with random utility maximisation for "
            "all values of the regressors. The nesting structure may be "
            "wrong.",
            AssumptionWarning,
            stacklevel=2,
        )
    lr = 2.0 * (ll - ll_cl)
    info: Dict[str, Any] = {
        "alpha": alpha,
        "model_type": "Nested logit",
        "citation_key": "nlogit",
        "method": "Maximum likelihood (RUM-consistent)",
        "vce": se_kind,
        "cluster": cl if se_kind == "cluster" else None,
        "nests": {str(k): list(v) for k, v in nests.items()},
        "alternatives": alts,
        "common_lambda": bool(common_lambda),
        "ll": ll,
        "log_likelihood": ll,
        "ll_clogit": ll_cl,
        "lr_iia_chi2": float(lr),
        "lr_iia_df": k_lam,
        "lr_iia_pvalue": float(stats.chi2.sf(lr, k_lam)),
        "aic": -2.0 * ll + 2.0 * k_total,
        "bic": -2.0 * ll + np.log(n) * k_total,
        "gradient_norm": grad_norm,
        "converged": bool(grad_norm < 1e-5 * max(1.0, n**0.5)),
    }
    if clusters is not None:
        info["n_clusters"] = int(pd.unique(clusters).size)
    return EconometricResults(
        params=pd.Series(theta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": n,
            "n_alternatives": J,
            "df_model": k_total,
            "df_resid": n - k_total,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "inference": "z",
            "y": chosen,
            "llobs": np.real(obs(theta)),
        },
        diagnostics={
            "Log-Likelihood": ll,
            "Log-Lik. (conditional logit)": ll_cl,
            "LR test of IIA (chi2)": float(lr),
            "Prob > chi2": info["lr_iia_pvalue"],
        },
    )


# Citation: the random-utility-consistent parametrisation used here, set
# against the "non-normalised" one. Mirrors paper.bib.
CausalResult._CITATIONS["nlogit"] = (
    "@article{heiss2002structural,\n"
    "  title={Structural Choice Analysis with Nested Logit Models},\n"
    "  author={Heiss, Florian},\n"
    "  journal={The Stata Journal},\n"
    "  volume={2},\n"
    "  number={3},\n"
    "  pages={227--252},\n"
    "  year={2002},\n"
    "  doi={10.1177/1536867X0200200301}\n"
    "}"
)
