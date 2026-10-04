"""Regression adjustment for randomized experiments, with interactions.

Lin (2013) [@lin2013agnostic] showed that the regression of the outcome on
treatment, centred covariates and their interactions with treatment gives
an estimate of the average treatment effect that is consistent whatever
the true outcome model, and asymptotically no less precise than the
difference in means. Adjustment without the interactions has neither
guarantee when the arms differ in size and effects are heterogeneous.

The numerical reference is R ``estimatr::lm_lin``. Its variance treats the
covariate means as known, which is right for the average effect in the
sample at hand. For the average effect in the population the covariates
were drawn from, estimating those means adds ``gamma' Var(X) gamma / n``,
with ``gamma`` the interaction coefficients; ``superpopulation=True`` adds
it, as the experiments chapter of Chernozhukov et al. does
[@chernozhukov2024applied].
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["lm_lin"]

# Verbatim from paper.bib (lin2013agnostic).
CausalResult._CITATIONS["lm_lin"] = (
    "@article{lin2013agnostic,\n"
    "  title={Agnostic notes on regression adjustments to experimental "
    "data: Reexamining Freedman's critique},\n"
    "  author={Lin, Winston},\n"
    "  journal={The Annals of Applied Statistics},\n"
    "  volume={7},\n"
    "  number={1},\n"
    "  pages={295--318},\n"
    "  year={2013},\n"
    "  doi={10.1214/12-AOAS583}\n"
    "}"
)

_VCE_INDEPENDENT = ("classical", "hc0", "hc1", "hc2", "hc3")
_VCE_CLUSTER = ("cr2", "stata")


def _covariate_columns(frame: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """Numeric columns as they are, other columns as indicators."""
    parts: List[pd.DataFrame] = []
    for name in covariates:
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            parts.append(col.astype(float).to_frame(name))
        else:
            dummies = pd.get_dummies(col.astype("category"), drop_first=True)
            dummies.columns = [f"{name}[{level}]" for level in dummies.columns]
            parts.append(dummies.astype(float))
    return pd.concat(parts, axis=1)


def _independent(design: np.ndarray) -> np.ndarray:
    """Mask of columns that are not linear combinations of earlier ones."""
    keep = np.zeros(design.shape[1], dtype=bool)
    basis = np.empty((design.shape[0], 0))
    for j in range(design.shape[1]):
        col = design[:, j]
        resid = col
        if basis.shape[1]:
            resid = col - basis @ (basis.T @ col)
        norm = float(np.linalg.norm(resid))
        if norm > 1e-9 * max(float(np.linalg.norm(col)), 1.0):
            keep[j] = True
            basis = np.column_stack([basis, resid / norm])
    return keep


def lm_lin(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    cluster: Optional[str] = None,
    vce: Optional[str] = None,
    superpopulation: bool = False,
    alpha: float = 0.05,
) -> CausalResult:
    """Average treatment effect by regression with treatment-covariate
    interactions (Lin 2013).

    Fits ``y ~ treat + Xc + treat:Xc`` with ``Xc`` the covariates minus
    their sample means, and reports the coefficient on ``treat``. Use it to
    analyse a randomized experiment more precisely than
    :func:`difference_in_means` without relying on the regression being
    correctly specified.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    treat : str
        Treatment, with exactly two values. The estimate contrasts the
        larger value with the smaller one.
    covariates : sequence of str
        Pre-treatment covariates. String and categorical columns enter as
        indicators with the first level omitted. Columns that are linear
        combinations of earlier ones are dropped.
    cluster : str, optional
        Clusters that were assigned to treatment together.
    vce : str, optional
        Without ``cluster``: ``'hc2'`` (default), ``'hc0'``, ``'hc1'``,
        ``'hc3'`` or ``'classical'``; inference uses t with ``n - k``
        degrees of freedom. With ``cluster``: ``'cr2'`` (default, with
        Bell-McCaffrey degrees of freedom) or ``'stata'`` (CR1, ``G - 1``).
        These are the ``se_type`` choices and defaults of R
        ``estimatr::lm_lin``.
    superpopulation : bool, default False
        ``False``: the variance is that of the regression coefficient,
        which takes the covariate means as fixed (``estimatr``). ``True``:
        adds ``gamma' Var(X) gamma / n`` for the estimation of those means,
        which is needed when the target is the average effect in the
        population the sample was drawn from.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    CausalResult
        ``estimate``, ``se``, ``pvalue`` and ``ci`` for the average
        treatment effect. ``detail`` has every coefficient of the
        regression; the rows named ``treat:<covariate>`` say how the
        effect varies with that covariate.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = rng.normal(size=n)
    >>> d = rng.binomial(1, 0.3, size=n)
    >>> y = d * (1 + x) + 2 * x + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x": x})
    >>> fit = sp.lm_lin(df, "y", "d", ["x"])
    >>> bool(fit.se < sp.difference_in_means(df, "y", "d").se)
    True

    References
    ----------
    [@lin2013agnostic]
    """
    context = "lm_lin"
    covariates = list(covariates)
    if not covariates:
        raise MethodIncompatibility(
            f"{context}: no covariates given.",
            recovery_hint="Without covariates use sp.difference_in_means.",
        )
    columns = [y, treat, *covariates] + ([cluster] if cluster else [])
    missing = [c for c in columns if c not in data.columns]
    if missing:
        from ..exceptions import ColumnNotFound

        raise ColumnNotFound(
            f"{context}: columns not in data: {missing}",
            diagnostics={"missing_columns": missing},
        )
    frame = data[columns].dropna()
    levels = np.sort(frame[treat].unique())
    if len(levels) != 2:
        raise MethodIncompatibility(
            f"{context}: {treat!r} must take exactly two values, found "
            f"{len(levels)}.",
            recovery_hint="Compare two arms at a time.",
        )
    if cluster is None:
        vce_key = "hc2" if vce is None else str(vce).lower()
        allowed = _VCE_INDEPENDENT
    else:
        vce_key = "cr2" if vce is None else str(vce).lower()
        allowed = _VCE_CLUSTER
    if vce_key not in allowed:
        raise MethodIncompatibility(
            f"{context}: vce={vce!r} is not available "
            f"{'with' if cluster else 'without'} cluster=.",
            recovery_hint=f"Choose one of {list(allowed)}.",
        )

    n = len(frame)
    d = (frame[treat].to_numpy() == levels[1]).astype(float)
    X = _covariate_columns(frame, covariates)
    Xc = X - X.mean()
    inter = Xc.mul(d, axis=0)
    inter.columns = [f"{treat}:{c}" for c in Xc.columns]
    names = ["Intercept", treat, *Xc.columns, *inter.columns]
    design = np.column_stack([np.ones(n), d, Xc.to_numpy(), inter.to_numpy()])
    keep = _independent(design)
    if not keep[1]:
        raise DataInsufficient(
            f"{context}: the treatment is collinear with the intercept."
        )
    n_cov = int(keep[2 : 2 + Xc.shape[1]].sum())
    smaller_arm = int(min(d.sum(), n - d.sum()))
    if smaller_arm <= n_cov:
        raise DataInsufficient(
            f"{context}: the smaller arm has {smaller_arm} complete rows for "
            f"{n_cov} covariate columns; the interacted regression fits each "
            "arm separately and needs more rows than covariates in both.",
            recovery_hint="Use fewer covariates, or sp.rlasso_effect to select.",
        )
    dropped = [nm for nm, k in zip(names, keep) if not k]
    design, names = design[:, keep], [nm for nm, k in zip(names, keep) if k]
    k = design.shape[1]
    if n <= k:
        raise DataInsufficient(
            f"{context}: {n} complete rows for {k} coefficients.",
            recovery_hint="Use fewer covariates, or sp.rlasso_effect to select.",
        )

    from ..regression.ols import regress

    safe = [f"v{j}" for j in range(1, k)]
    work = pd.DataFrame(design[:, 1:], columns=safe)
    work["_y"] = frame[y].to_numpy(dtype=float)
    formula = "_y ~ " + " + ".join(safe)
    if cluster is None:
        robust = "nonrobust" if vce_key == "classical" else vce_key
        fit = regress(formula, data=work, robust=robust)
        df = float(n - k)
    else:
        cl = frame[cluster].to_numpy()
        work["_cl"] = pd.factorize(cl)[0]
        n_clusters = int(work["_cl"].max()) + 1
        if vce_key == "cr2":
            fit = regress(formula, data=work, cluster="_cl", vce="cr2")
            from .jackknife import _satterthwaite_dof

            codes = work["_cl"].to_numpy()
            xtx_inv = np.linalg.inv(design.T @ design)
            df = float(
                _satterthwaite_dof(design, codes, np.unique(codes), xtx_inv, k)[1]
            )
        else:
            fit = regress(formula, data=work, cluster="_cl")
            df = float(n_clusters - 1)
    coef = np.asarray(fit.params, dtype=float)
    ses = np.asarray(fit.std_errors, dtype=float)
    est, var = float(coef[1]), float(ses[1] ** 2)

    model_info: Dict[str, object] = {
        "vce": vce_key,
        "n_coefficients": int(k),
        "treatment_levels": [levels[0], levels[1]],
        "n_treated": int(d.sum()),
        "n_control": int(n - d.sum()),
        "dropped_collinear": dropped,
        "superpopulation": bool(superpopulation),
        "se_regression": float(ses[1]),
    }
    if cluster is not None:
        model_info["cluster"] = cluster
        model_info["n_clusters"] = n_clusters
    if superpopulation:
        gamma = np.array(
            [coef[names.index(c)] if c in names else 0.0 for c in inter.columns]
        )
        cate = Xc.to_numpy() @ gamma  # mean zero by construction
        if cluster is None:
            extra = float(np.sum(cate**2) / (n - 1) / n)
        else:
            totals = np.bincount(work["_cl"].to_numpy(), weights=cate)
            extra = float(np.sum(totals**2) * n_clusters / (n_clusters - 1) / n**2)
        var += extra
        model_info["var_covariate_means"] = extra

    se = float(np.sqrt(var))
    tstat = est / se if se > 0 else float("nan")
    pvalue = float(2 * stats.t.sf(abs(tstat), df)) if se > 0 else float("nan")
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    model_info.update({"df": df, "df_inference": df, "statistic": tstat})
    detail = pd.DataFrame({"term": names, "coef": coef, "se": ses})
    return CausalResult(
        method="Regression adjustment with treatment interactions (Lin 2013)",
        estimand="ATE",
        estimate=est,
        se=se,
        pvalue=pvalue,
        ci=(float(est - crit * se), float(est + crit * se)),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="lm_lin",
    )
