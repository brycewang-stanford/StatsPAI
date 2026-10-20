"""Path-specific effects with causally ordered mediators.

With mediators ``M_1 -> M_2 -> ... -> M_K`` (each may affect the later
ones) the total effect of a binary treatment splits into a direct effect
and one effect through each mediator. Writing ``M~_j`` for the first ``j``
mediators and ``psi(a, j, a') = E[Y(a, M~_j(a'))]`` (treatment ``a``
everywhere except that the first ``j`` mediators take their values under
``a'``; later mediators respond to ``a`` and to those values):

* Type I: ``direct = psi(1, K, 0) - E[Y(0)]``,
  ``via M_j = psi(1, j - 1, 0) - psi(1, j, 0)`` (``psi(1, 0, 0) = E[Y(1)]``);
* Type II: the same with the roles of 1 and 0 exchanged,
  ``direct = E[Y(1)] - psi(0, K, 1)``,
  ``via M_j = psi(0, j, 1) - psi(0, j - 1, 1)``.

For one mediator these are the natural direct and indirect effects (pure
direct / total indirect in Type I). ``psi`` is identified from the
outcome regression on the treatment, the covariates and the first ``j``
mediators, ``mu_j(a, x, m~_j)``: it is the mean of ``mu_j(a, X, M~_j)``
over the units with treatment ``a'``, reweighted to the covariate
distribution of the whole sample. The *pure imputation* estimator does the
reweighting by regressing the imputed values on the covariates among those
units (linear, or a fractional logit for a logistic outcome model) and
averaging the predictions over everyone; the
*imputation-based weighting* estimator uses inverse propensity weights
instead (``ps=True``). Both are the estimators of R ``paths`` (Zhou and
Yamamoto, [@zhou2023tracing]); the point estimates agree to 1e-12 with linear
outcome models and to R ``glm``'s convergence with logistic ones
(``tests/reference_parity/test_mediate_paths_parity.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["mediate_paths"]


def _ols(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    if w is None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def _logit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logit coefficients for a design whose first column is the constant."""
    from ..matching._binary_fit import fit_binary_index  # shared logit fit

    return fit_binary_index(X[:, 1:], y, "logit")["beta"]


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _estimate(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    Ms: List[np.ndarray],
    interaction: bool,
    outcome_model: str,
    ps: bool,
) -> Dict[str, Any]:
    """Point estimates of every ``psi`` and both decompositions."""
    n = len(Y)
    K = len(Ms)

    def design(a: np.ndarray, j: int) -> np.ndarray:
        cols = [np.ones(n), a, X]
        for k in range(j):
            cols.append(Ms[k])
            if interaction:
                cols.append(a[:, None] * Ms[k])
        return np.column_stack(cols)

    fits = []
    for j in range(K + 1):
        D = design(A, j)
        if np.linalg.matrix_rank(D) < D.shape[1]:
            raise MethodIncompatibility(
                f"mediate_paths: outcome model {j} (treatment, covariates and "
                f"the first {j} mediator blocks) is rank deficient.",
                recovery_hint="Drop collinear covariates or mediator columns.",
            )
        fits.append(_logit(D, Y) if outcome_model == "logistic" else _ols(D, Y))

    def predict(j: int, a: float) -> np.ndarray:
        eta = design(np.full(n, float(a)), j) @ fits[j]
        return _expit(eta) if outcome_model == "logistic" else eta

    Xc = np.column_stack([np.ones(n), X])
    e = None
    if ps:
        e = _expit(Xc @ _logit(Xc, A))

    def psi(a: float, j: int, a_med: float) -> float:
        if j == 0:
            return float(predict(0, a).mean())
        yhat = predict(j, a)
        g = A == a_med
        if ps:
            w = 1.0 / (e[g] if a_med == 1 else 1.0 - e[g])
            return float(np.sum(w * yhat[g]) / np.sum(w))
        if outcome_model == "logistic":
            # fractional logit of the imputed probabilities, as R paths
            # refits the outcome model's family
            from ._four_way_engine import fit_glm

            b, _ = fit_glm(Xc[g], yhat[g], "logistic")
            return float(_expit(Xc @ b).mean())
        b = _ols(Xc[g], yhat[g])
        return float((Xc @ b).mean())

    ey1, ey0 = psi(1, 0, 1), psi(0, 0, 0)
    p1 = [ey1] + [psi(1, j, 0) for j in range(1, K + 1)]
    p0 = [ey0] + [psi(0, j, 1) for j in range(1, K + 1)]
    type1 = {"direct": p1[K] - ey0}
    type2 = {"direct": ey1 - p0[K]}
    for j in range(K, 0, -1):
        type1[f"via_M{j}"] = p1[j - 1] - p1[j]
        type2[f"via_M{j}"] = p0[j] - p0[j - 1]
    return {
        "type1": type1,
        "type2": type2,
        "total": ey1 - ey0,
        "ey1": ey1,
        "ey0": ey0,
    }


def mediate_paths(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediators: Sequence[Union[str, Sequence[str]]],
    covariates: Optional[List[str]] = None,
    interaction: bool = True,
    outcome_model: str = "linear",
    ps: bool = False,
    decomposition: str = "type1",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> CausalResult:
    """Path-specific effects of a binary treatment through ordered mediators.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome.
    treat : str
        Binary treatment (0/1).
    mediators : sequence
        The mediators in causal order, ``M_1`` first. Each entry is a
        column name or a list of columns that form one mediator block (R
        ``paths``' ``m = list(...)``). A mediator may affect every later
        one; a later mediator may not affect an earlier one.
    covariates : list of str, optional
        Pre-treatment confounders, entered linearly in every model.
    interaction : bool, default True
        Include treatment x mediator interactions in the outcome models.
    outcome_model : {'linear', 'logistic'}, default 'linear'
        Model for ``E[Y | A, X, M_1..M_j]``; ``'logistic'`` for a 0/1
        outcome (effects on the risk-difference scale).
    ps : bool, default False
        Reweight with a logistic propensity score of the treatment on the
        covariates (the imputation-based weighting estimator) instead of
        the pure imputation estimator.
    decomposition : {'type1', 'type2'}, default 'type1'
        Which decomposition is the headline (both are in ``detail``).
        Type I switches the mediators from the treated to the control
        value one at a time starting from ``M_1``; for one mediator it is
        pure direct + total indirect effect.
    n_boot : int, default 500
        Nonparametric bootstrap replications (units resampled, every model
        refitted) for the standard errors and percentile intervals.
    alpha : float, default 0.05
    seed : int, default 42

    Returns
    -------
    CausalResult
        ``estimate`` is the total effect. ``detail`` has one row per
        decomposition and component (``direct``, ``via_M1`` ... ``via_MK``,
        ``total``) with bootstrap SE, percentile interval and the
        bootstrap sign p-value. ``model_info['type1']`` /
        ``['type2']`` hold the point estimates.

    Notes
    -----
    Identification needs no unmeasured confounding of treatment-outcome,
    treatment-mediator and mediator-outcome relations given the
    covariates and earlier mediators, and no other post-treatment
    confounder of the mediators (an exposure-induced confounder has to be
    entered as an earlier mediator). The direct and path-specific effects
    always sum to the total effect.

    References
    ----------
    [@zhou2023tracing]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> a = rng.integers(0, 2, n)
    >>> m1 = 0.6 * a + rng.normal(size=n)
    >>> m2 = 0.5 * a + 0.7 * m1 + rng.normal(size=n)
    >>> y = 0.4 * a + 0.5 * m1 + 0.8 * m2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "a": a, "m1": m1, "m2": m2})
    >>> r = sp.mediate_paths(df, y="y", treat="a", mediators=["m1", "m2"],
    ...                      interaction=False, n_boot=50)
    >>> sorted(r.model_info["type1"])
    ['direct', 'via_M1', 'via_M2']
    """
    if outcome_model not in ("linear", "logistic"):
        raise MethodIncompatibility(
            f"outcome_model must be 'linear' or 'logistic', got {outcome_model!r}",
            recovery_hint="Use 'logistic' for a 0/1 outcome.",
        )
    if decomposition not in ("type1", "type2"):
        raise MethodIncompatibility(
            f"decomposition must be 'type1' or 'type2', got {decomposition!r}",
            recovery_hint="Both are reported in result.detail.",
        )
    if not mediators:
        raise MethodIncompatibility(
            "mediate_paths needs at least one mediator.",
            recovery_hint="Pass mediators=['m1', 'm2', ...] in causal order.",
        )
    blocks = [[m] if isinstance(m, str) else list(m) for m in mediators]
    med_cols = [c for b in blocks for c in b]
    if len(set(med_cols)) != len(med_cols):
        raise MethodIncompatibility(
            "mediate_paths: a column appears in two mediator blocks.",
            recovery_hint="Each mediator column belongs to one block.",
        )
    cov = list(covariates or [])
    check_columns(
        data,
        {"y": y, "treat": treat, "mediators": med_cols, "covariates": cov or None},
        function="sp.mediate_paths",
    )
    df = data[[y, treat] + med_cols + cov].dropna()
    if len(df) < len(data):
        import warnings

        warnings.warn(
            f"mediate_paths: dropped {len(data) - len(df)} row(s) with missing "
            "values.",
            UserWarning,
            stacklevel=2,
        )
    from ..core._validate import treatment_as_float

    A = treatment_as_float(df[treat], function="sp.mediate_paths")
    if not np.isin(A, (0.0, 1.0)).all() or len(np.unique(A)) < 2:
        raise MethodIncompatibility(
            f"mediate_paths: treatment {treat!r} must be 0/1 with both values.",
            recovery_hint="Recode the treatment to 0/1.",
        )
    Y = df[y].to_numpy(dtype=float)
    if outcome_model == "logistic" and not np.isin(Y, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"mediate_paths: outcome_model='logistic' needs a 0/1 {y!r}.",
            recovery_hint="Use outcome_model='linear'.",
        )
    X = df[cov].to_numpy(dtype=float) if cov else np.zeros((len(df), 0))
    Ms = [df[b].to_numpy(dtype=float) for b in blocks]
    n = len(df)
    n_par = 2 + X.shape[1] + sum(m.shape[1] for m in Ms) * (2 if interaction else 1)
    if n <= n_par + 1:
        raise DataInsufficient(
            f"mediate_paths: {n} rows for an outcome model with {n_par} terms.",
            recovery_hint="Use fewer covariates or mediators.",
        )

    point = _estimate(Y, A, X, Ms, interaction, outcome_model, ps)
    K = len(Ms)
    keys = ["direct"] + [f"via_M{j}" for j in range(K, 0, -1)]

    rng = np.random.default_rng(seed)
    boot: Dict[str, List[float]] = {
        f"{t}:{k}": [] for t in ("type1", "type2") for k in keys + ["total"]
    }
    n_failed = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(A[idx])) < 2:
            n_failed += 1
            continue
        try:
            b = _estimate(
                Y[idx],
                A[idx],
                X[idx],
                [m[idx] for m in Ms],
                interaction,
                outcome_model,
                ps,
            )
        except (MethodIncompatibility, np.linalg.LinAlgError, ValueError):
            n_failed += 1
            continue
        for t in ("type1", "type2"):
            for k in keys:
                boot[f"{t}:{k}"].append(b[t][k])
            boot[f"{t}:total"].append(b["total"])

    rows = []
    for t in ("type1", "type2"):
        for k in keys + ["total"]:
            est = point["total"] if k == "total" else point[t][k]
            draws = np.asarray(boot[f"{t}:{k}"], dtype=float)
            if draws.size >= 2:
                se = float(np.std(draws, ddof=1))
                lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
                side = draws <= 0 if est >= 0 else draws >= 0
                pv = float(np.clip(2 * side.mean(), 1 / draws.size, 1.0))
            else:
                se = lo = hi = pv = float("nan")
            rows.append(
                {
                    "decomposition": t,
                    "effect": k,
                    "estimate": float(est),
                    "se": se,
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "pvalue": pv,
                }
            )
    detail = pd.DataFrame(rows)
    head = detail[(detail.decomposition == decomposition) & (detail.effect == "total")]
    model_info: Dict[str, Any] = {
        "estimator": "imputation-based weighting" if ps else "pure imputation",
        "type1": point["type1"],
        "type2": point["type2"],
        "total_effect": point["total"],
        "mediators": blocks,
        "decomposition": decomposition,
        "interaction": interaction,
        "outcome_model": outcome_model,
        "n_boot": int(n_boot),
        "n_boot_failed": n_failed,
    }
    result = CausalResult(
        method="Path-specific effects (ordered mediators)",
        estimand="Total effect (path decomposition in detail)",
        estimate=float(point["total"]),
        se=float(head["se"].iloc[0]),
        pvalue=float(head["pvalue"].iloc[0]),
        ci=(float(head["ci_lower"].iloc[0]), float(head["ci_upper"].iloc[0])),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="mediate_paths",
    )
    return result


CausalResult._CITATIONS["mediate_paths"] = (
    "@article{zhou2023tracing,\n"
    "  title={Tracing Causal Paths from Experimental and Observational "
    "Data},\n"
    "  author={Zhou, Xiang and Yamamoto, Teppei},\n"
    "  journal={The Journal of Politics},\n"
    "  volume={85},\n"
    "  number={1},\n"
    "  pages={250--265},\n"
    "  year={2023},\n"
    "  doi={10.1086/720310}\n"
    "}"
)
