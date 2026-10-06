"""Hausman-Taylor estimator for panels with time-invariant regressors.

Fixed effects cannot estimate the coefficient of a regressor that does not
vary within unit; random effects can, but only if every regressor is
uncorrelated with the unit effect. Hausman and Taylor's estimator sits
between the two. Some regressors are declared correlated with the unit
effect (*endogenous* in that sense; all are still uncorrelated with the
idiosyncratic error). The within variation of every time-varying regressor
and the unit means of the exogenous time-varying ones serve as instruments,
so the coefficients of time-invariant regressors are identified as long as
there are at least as many exogenous time-varying regressors as endogenous
time-invariant ones.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr

from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xthtaylor"]


def _iv(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> tuple:
    """2SLS coefficients of ``y`` on ``X`` with instruments ``W`` and the
    projected regressors."""
    # The instrument set can be rank deficient (with few distinct panel
    # lengths 1 - theta_i is a combination of the unit means of period
    # dummies), and an unpivoted factorisation of such a matrix does not
    # span its column space. Pivot, and keep the columns that count.
    Q, R, _ = qr(W, mode="economic", pivoting=True)
    size = np.abs(np.diag(R))
    Q = Q[:, size > 1e-10 * size[0]]
    PX = Q @ (Q.T @ X)
    return np.linalg.lstsq(PX, y, rcond=None)[0], PX


def xthtaylor(
    formula: str,
    data: pd.DataFrame,
    *,
    id: str,
    endog: Sequence[str],
    method: str = "ht",
    time: Optional[str] = None,
    vce: str = "conventional",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """Hausman-Taylor instrumental-variable estimator for panel data.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 + z1 + z2"`` with time-varying and time-invariant
        regressors together; which is which is read from the data.
    data : pandas.DataFrame
        Long panel, one row per unit and period. Rows with a missing value
        in a model variable are dropped.
    id : str
        Unit identifier.
    endog : list of str
        Regressors that may be correlated with the unit effect. A name
        covers the columns built from it (``"industry"`` covers every
        level of ``C(industry)``).
    method : {'ht', 'amacurdy'}, default 'ht'
        ``'amacurdy'`` is the Amemiya-MaCurdy estimator: every period's
        value of an exogenous time-varying regressor is an instrument, in
        place of its unit mean. It needs a balanced panel and ``time=``,
        and it is more efficient when those regressors are uncorrelated
        with the unit effect period by period, a stronger assumption than
        Hausman and Taylor's.
    time : str, optional
        Period identifier, needed by ``method='amacurdy'``.
    vce : {'conventional', 'robust'}, default 'conventional'
        ``'robust'`` clusters on the unit.
    cluster : str, optional
        Another variable to cluster on (units must be nested in it).
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        Coefficients of every regressor and the constant, with z
        inference. ``model_info`` holds ``sigma_u``, ``sigma_e``, ``rho``,
        the range of ``theta`` and the four groups of regressors
        (``tv_exogenous``, ``tv_endogenous``, ``ti_exogenous``,
        ``ti_endogenous``).

    Notes
    -----
    The steps are those of Stata's ``xthtaylor``, which this reproduces:

    1. the within estimator of the time-varying coefficients, and
       ``sigma_e^2`` as its residual sum of squares over ``N - G``;
    2. the unit means of the within residuals regressed on the
       time-invariant regressors by 2SLS, with the exogenous time-varying
       and exogenous time-invariant regressors as instruments;
    3. ``sigma_u^2`` from the unit means of the resulting residuals (the
       harmonic mean of the panel lengths when the panel is unbalanced);
    4. 2SLS on the quasi-demeaned data, ``w - theta_i * wbar_i``, with the
       within deviations of all time-varying regressors, the unit means of
       the exogenous time-varying ones and the exogenous time-invariant
       regressors as instruments.

    The standard errors are those of the last step with ``N - K`` degrees
    of freedom; the clustered ones carry the factor
    ``G / (G - 1) * (N - 1) / (N - K)``.

    One departure from Stata. With period dummies among the exogenous
    time-varying regressors of an unbalanced panel, Stata's instrument set
    holds the unit means of the dummies it kept, so its estimates change
    with the period it happens to omit. Here the constant is added to the
    instruments in that case, which is the same as using the means of all
    the dummies, and the fit does not depend on the base period. Without
    such a factor, and in balanced panels, the two programs agree to
    rounding.

    The estimator is consistent only if the regressors declared exogenous
    really are uncorrelated with the unit effect. A fixed-effects fit of
    the time-varying coefficients does not rely on that; compare the two
    (:func:`statspai.hausman`).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> units, periods = 200, 5
    >>> a = rng.normal(size=units)
    >>> df = pd.DataFrame({"id": np.repeat(np.arange(units), periods)})
    >>> df["x1"] = rng.normal(size=len(df))
    >>> df["x2"] = rng.normal(size=len(df)) + a[df.id]
    >>> df["z1"] = rng.normal(size=units)[df.id]
    >>> df["z2"] = (rng.normal(size=units) + a)[df.id]
    >>> noise = rng.normal(size=len(df))
    >>> df["y"] = 1 + df.x1 + df.x2 + df.z1 + df.z2 + a[df.id] + noise
    >>> fit = sp.xthtaylor("y ~ x1 + x2 + z1 + z2", df, id="id", endog=["x2", "z2"])
    >>> fit.model_info["ti_endogenous"]
    ['z2']
    >>> bool(abs(fit.params["x1"] - 1) < 0.2)
    True

    References
    ----------
    hausman1981panel
    """
    from ..regression.ols import regress

    vce = str(vce).lower()
    if vce not in ("conventional", "robust"):
        raise MethodIncompatibility(
            f"sp.xthtaylor: vce={vce!r} is not 'conventional' or 'robust'.",
            recovery_hint="Use vce='robust' to cluster on the unit.",
        )
    for name, value in (("id", id), ("cluster", cluster)):
        if value is not None and value not in data.columns:
            raise MethodIncompatibility(
                f"sp.xthtaylor: {name}={value!r} is not a column.",
                recovery_hint="Pass the name of the variable.",
            )
    method = str(method).lower()
    if method not in ("ht", "amacurdy"):
        raise MethodIncompatibility(
            f"sp.xthtaylor: method={method!r} is not available.",
            recovery_hint="Use 'ht' or 'amacurdy'.",
        )
    if method == "amacurdy" and (time is None or time not in data.columns):
        raise MethodIncompatibility(
            "sp.xthtaylor: method='amacurdy' needs time=, the period " "identifier.",
            recovery_hint="Pass time='year' (the panel has to be balanced).",
        )
    endog = [str(e) for e in endog]
    if not endog:
        raise MethodIncompatibility(
            "sp.xthtaylor: endog= is empty. With no regressor correlated "
            "with the unit effect the model is random effects.",
            recovery_hint="Use sp.panel(method='re'), or name the "
            "regressors that may be correlated with the unit effect.",
        )

    ols = regress(formula, data=data)
    info = ols.data_info
    frame = data.loc[info["sample_index"]]
    if frame[id].isna().any():
        raise MethodIncompatibility(
            "sp.xthtaylor: the unit identifier is missing on rows of the "
            "estimation sample.",
            recovery_hint="Drop those rows first.",
        )
    design = np.asarray(info["X"], dtype=float)
    names = [str(v) for v in info["var_names"]]
    y = np.asarray(info["y"], dtype=float)
    const = [j for j, nm in enumerate(names) if nm in ("Intercept", "const", "_cons")]
    if not const:
        raise MethodIncompatibility(
            "sp.xthtaylor: the model needs a constant.",
            recovery_hint="Remove '- 1' from the formula.",
        )
    cols = [j for j in range(len(names)) if j not in const]

    codes = pd.factorize(frame[id])[0]
    n_units, n_obs = int(codes.max()) + 1, len(y)
    sizes = np.bincount(codes).astype(float)
    T = sizes[codes]

    def unit_mean(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return np.bincount(codes, weights=a)[codes] / T
        out = [np.bincount(codes, weights=a[:, j]) / sizes for j in range(a.shape[1])]
        return np.column_stack(out)[codes] if out else np.zeros((n_obs, 0))

    X_all = design[:, cols]
    means = unit_mean(X_all)
    scale = np.sqrt((X_all**2).sum(axis=0))
    scale[scale == 0] = 1.0
    varying = np.sqrt(((X_all - means) ** 2).sum(axis=0)) > 1e-10 * scale

    def is_endog(col: str) -> bool:
        return any(
            col == e or col.startswith((f"C({e})", f"{e}[", f"{e}:")) for e in endog
        )

    labels = [names[j] for j in cols]
    unknown = [e for e in endog if not any(is_endog(c) and e in c for c in labels)]
    if unknown:
        raise MethodIncompatibility(
            f"sp.xthtaylor: endog={unknown} are not regressors of the model.",
            recovery_hint="Check the names; a factor is named by its variable.",
        )
    groups: Dict[str, List[int]] = {
        "tv_exogenous": [],
        "tv_endogenous": [],
        "ti_exogenous": [],
        "ti_endogenous": [],
    }
    for j, col in enumerate(labels):
        key = ("tv_" if varying[j] else "ti_") + (
            "endogenous" if is_endog(col) else "exogenous"
        )
        groups[key].append(j)
    k1, g2 = len(groups["tv_exogenous"]), len(groups["ti_endogenous"])
    if not groups["tv_exogenous"] + groups["tv_endogenous"]:
        raise MethodIncompatibility(
            "sp.xthtaylor: no regressor varies within unit.",
            recovery_hint="The estimator needs time-varying regressors.",
        )
    if not groups["ti_exogenous"] + groups["ti_endogenous"]:
        raise MethodIncompatibility(
            "sp.xthtaylor: no regressor is constant within unit; there is "
            "nothing for the estimator to add to fixed effects.",
            recovery_hint="Use sp.panel(method='fe').",
        )
    if k1 < g2:
        raise MethodIncompatibility(
            f"sp.xthtaylor: the model is under-identified: {g2} "
            f"time-invariant endogenous regressor(s) but only {k1} "
            "time-varying exogenous one(s) to instrument them.",
            recovery_hint="Declare fewer regressors endogenous, or add "
            "exogenous time-varying regressors.",
        )
    X1, X2 = X_all[:, groups["tv_exogenous"]], X_all[:, groups["tv_endogenous"]]
    Z1, Z2 = X_all[:, groups["ti_exogenous"]], X_all[:, groups["ti_endogenous"]]
    X = np.column_stack([X1, X2])
    Z = np.column_stack([Z1, Z2])
    k = X.shape[1] + Z.shape[1] + 1
    if n_obs - n_units <= X.shape[1] or n_obs <= k:
        raise DataInsufficient(
            "sp.xthtaylor: not enough observations.",
            recovery_hint="Use fewer regressors.",
        )
    one = np.ones((n_obs, 1))

    # 1. within estimator and sigma_e
    y_mean, X_mean = unit_mean(y), unit_mean(X)
    X_dev = X - X_mean
    b_within = np.linalg.lstsq(X_dev, y - y_mean, rcond=None)[0]
    e_within = (y - y_mean) - X_dev @ b_within
    sigma2_e = float(e_within @ e_within) / (n_obs - n_units)

    # 2. time-invariant coefficients from the unit means of the residuals
    Zc = np.column_stack([Z, one])
    gamma, _ = _iv(y_mean - X_mean @ b_within, Zc, np.column_stack([X1, Z1, one]))

    # 3. sigma_u
    resid_mean = unit_mean(y - X @ b_within - Zc @ gamma)
    sigma2_1 = float((resid_mean**2).sum()) / n_units
    t_bar = n_units / float((1.0 / sizes).sum())
    sigma2_u = max((sigma2_1 - sigma2_e) / t_bar, 0.0)
    theta = 1.0 - np.sqrt(sigma2_e / (T * sigma2_u + sigma2_e))

    # 4. 2SLS on the quasi-demeaned data
    th = theta[:, None]
    X1_mean = unit_mean(X1)
    regressors = np.column_stack(
        [
            X1 - th * X1_mean,
            X2 - th * unit_mean(X2),
            Z1 * (1 - th),
            Z2 * (1 - th),
            1 - th,
        ]
    )
    between = X1_mean
    if method == "amacurdy":
        periods, when = np.unique(frame[time].to_numpy(), return_inverse=True)
        cell = np.zeros((n_units, periods.size), dtype=int)
        np.add.at(cell, (codes, when), 1)
        if not np.all(cell == 1):
            raise MethodIncompatibility(
                "sp.xthtaylor: method='amacurdy' needs a balanced panel, "
                "every unit observed once in each period.",
                recovery_hint="Use method='ht', or keep the units observed "
                "in every period.",
            )
        # each exogenous time-varying regressor in every period, as a
        # characteristic of the unit
        wide = np.zeros((n_units, periods.size, X1.shape[1]))
        wide[codes, when, :] = X1
        between = wide.reshape(n_units, -1)[codes]
    instruments = [X1 - X1_mean, X2 - unit_mean(X2), between, Z1, 1 - th]
    if np.ptp(theta) > 0 and any("[T." in labels[j] for j in groups["tv_exogenous"]):
        # An exogenous time-varying factor (period dummies) enters without
        # its base level. The unit mean of the base level's indicator is an
        # instrument like the others, and it is one minus the sum of theirs:
        # adding the constant makes the fit the same whichever level is the
        # base. In a balanced panel theta is one number and the constant is
        # in the span already.
        instruments.append(one)
    instruments = np.column_stack(instruments)
    beta, projected = _iv(y - theta * y_mean, regressors, instruments)
    resid = (y - theta * y_mean) - regressors @ beta
    try:
        bread = np.linalg.inv(projected.T @ projected)
    except np.linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "sp.xthtaylor: the instruments do not identify the model "
            "(singular projected design).",
            recovery_hint="Check for collinear regressors.",
        ) from exc
    clusters = None
    if cluster is not None:
        clusters = pd.factorize(frame[cluster])[0]
    elif vce == "robust":
        clusters = codes
    if clusters is None:
        cov = bread * float(resid @ resid) / (n_obs - k)
    else:
        n_clusters = int(clusters.max()) + 1
        scores = np.zeros((n_clusters, k))
        np.add.at(scores, clusters, projected * resid[:, None])
        factor = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - k)
        cov = factor * bread @ (scores.T @ scores) @ bread

    order = (
        groups["tv_exogenous"] + groups["tv_endogenous"]
        + groups["ti_exogenous"] + groups["ti_endogenous"]
    )  # fmt: skip
    out_names = [labels[j] for j in order] + ["_cons"]
    slopes = list(range(k - 1))
    b_s = beta[slopes]
    wald = float(b_s @ np.linalg.solve(cov[np.ix_(slopes, slopes)], b_s))
    title = "Amemiya-MaCurdy" if method == "amacurdy" else "Hausman-Taylor"
    model_info: Dict[str, Any] = {
        "model_type": title,
        "estimator": method,
        "method": "xthtaylor",
        "sigma_u": float(np.sqrt(sigma2_u)),
        "sigma_e": float(np.sqrt(sigma2_e)),
        "rho": sigma2_u / (sigma2_u + sigma2_e),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "n_groups": n_units,
        "robust": "nonrobust" if clusters is None else "cluster",
        "cluster": cluster or (id if clusters is not None else None),
        "alpha": alpha,
        **{key: [labels[j] for j in idx] for key, idx in groups.items()},
    }
    if clusters is not None:
        model_info["n_clusters"] = int(clusters.max()) + 1
    result = EconometricResults(
        params=pd.Series(beta, index=out_names),
        std_errors=pd.Series(np.sqrt(np.diag(cov)), index=out_names),
        model_info=model_info,
        data_info={
            "nobs": n_obs,
            "df_model": k - 1,
            "df_resid": n_obs - k,
            "dependent_var": info.get("dependent_var"),
            "var_names": out_names,
            "var_cov": cov,
            "inference": "z",
            "sample_index": frame.index,
        },
        diagnostics={
            "Wald chi2": wald,
            "Prob > chi2": float(stats.chi2.sf(wald, k - 1)),
            "sigma_u": float(np.sqrt(sigma2_u)),
            "sigma_e": float(np.sqrt(sigma2_e)),
            "rho": sigma2_u / (sigma2_u + sigma2_e),
        },
    )
    result.alpha = alpha
    result._compute_statistics()
    return result
