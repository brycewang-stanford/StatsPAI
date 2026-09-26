"""Joint Wald tests after multiple imputation (Stata ``mi test``).

Two tests of ``H0: Q_j = 0`` for a set of ``k`` pooled coefficients, both
built from the pooled within / between covariances ``U`` and ``B`` over
``m`` imputations, ``T = U + (1 + 1/m) B`` and the average relative
increase in variance ``r = (1 + 1/m) tr(B U^{-1}) / k``:

* ``method="equal_fmi"`` (Stata's default) assumes ``B`` proportional to
  ``U``: ``F = Q' U^{-1} Q / (k (1 + r))``.
* ``method="unrestricted"`` (Stata ``ufmitest``): ``F = Q' T^{-1} Q / k``.
  ``B`` is estimated from ``m`` draws, so this is unreliable unless ``m``
  is large relative to ``k``.

Denominator df ``nu`` (``t = k (m - 1)``, ``nu_c`` the complete-data df,
``nu_c* = nu_c (nu_c + 1) / (nu_c + 3)``, ``g = (1 + 1/m) tr(B T^{-1}) / k``
the average fraction of missing information, ``nu_obs = (1 - g) nu_c*``):

========================  ==========================  ==========================
case                      large sample (``nosmall``)  small sample (default)
========================  ==========================  ==========================
``k = 1`` (either test)   ``nu_L``                    ``[1/nu_L + 1/nu_obs]^-1``
unrestricted              ``nu_L``                    ``[1/nu_L + 1/nu_obs]^-1``
equal FMI, ``t > 4``      Li et al. (1991)            Reiter (2007)
equal FMI, ``t <= 4``     ``(k+1)/2 * nu_L``          ``(k+1)/2`` times the above
========================  ==========================  ==========================

with ``nu_L = (m-1)(1+1/r)^2``. For ``k = 1`` the two tests coincide,
``g = r/(1+r)`` and the small-sample df is Barnard & Rubin (1999). For
``k > 1`` it is Marchenko & Reiter's (2009) proposal as Stata computes it,
which differs from the paper's formulas in two ways, both confirmed against
Stata: the large-sample part stays ``nu_L`` (built on ``r``, i.e. on
``U``) where the paper writes ``(m-1)/g^2`` (built on ``T``), with only
``nu_obs`` using ``g``; and the equal-FMI switch is on ``t = k (m - 1)``
(Stata's manual), not the paper's ``m (k - 1)``.
``tests/reference_parity/test_mi_test_parity.py`` pins every
cell against Stata 18 on 326 test configurations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from scipy import stats

from ..exceptions import AssumptionWarning, MethodIncompatibility


def _reiter_df(t: int, r: float, dfcom_star: float) -> float:
    """Reiter's (2007) small-sample df for the equal-FMI test, ``t > 4``."""
    a = r * t / (t - 2.0)
    c0 = 1.0 / (t - 4.0)
    c1 = dfcom_star - 2.0 * (1.0 + a)
    c2 = dfcom_star - 4.0 * (1.0 + a)
    a2 = a * a
    z = 1.0 / c2 + c0 * (
        a2 * c1 / ((1.0 + a) ** 2 * c2)
        + 8.0 * a2 * c1 / ((1.0 + a) * c2**2)
        + 4.0 * a2 / ((1.0 + a) * c2)
        + 4.0 * a2 / (c2 * c1)
        + 16.0 * a2 * c1 / c2**3
        + 8.0 * a2 / c2**2
    )
    return 4.0 + 1.0 / z


def mi_test(
    pooled: Dict[str, Any],
    terms: Union[str, Sequence[str]],
    *,
    method: str = "equal_fmi",
    small: bool = True,
) -> Dict[str, Any]:
    """
    Joint Wald test of pooled multiple-imputation coefficients.

    Parameters
    ----------
    pooled : dict
        Output of :func:`sp.mi_estimate` (it carries the within / between
        covariances ``ubar_matrix`` / ``b_matrix`` and the complete-data df
        ``dfcom``).
    terms : str or list of str
        Coefficient names tested jointly equal to zero (``var_names``).
    method : {"equal_fmi", "unrestricted"}, default "equal_fmi"
        ``"equal_fmi"`` is Stata ``mi test``'s default test (equal fractions
        of missing information across the tested coefficients);
        ``"unrestricted"`` is ``mi test, ufmitest``.
    small : bool, default True
        Small-sample denominator df, Stata's default: Reiter (2007) for the
        equal-FMI test and Barnard & Rubin (1999) / Marchenko & Reiter
        (2009) otherwise. Applied only when the pooled result carries a
        finite complete-data df (``dfcom``, e.g. the residual df of
        ``sp.regress``); without one the large-sample df is used, as Stata
        does when ``e(df_r)`` is not set. ``small=False`` is
        ``mi test, nosmall``.

    Returns
    -------
    dict
        ``F``, ``df1``, ``df2``, ``pvalue``, ``rvi`` (average relative
        increase in variance), ``method``, ``terms``, ``n_imputations``,
        ``dfcom`` and ``df_adjustment`` (``"small"`` or ``"large"``, the df
        actually used).

    Warns
    -----
    AssumptionWarning
        When Reiter's df is used outside its range
        (``nu_c* <= 4 (1 + a)``, a tiny complete-data df relative to the
        missing information): the value is Stata's, but it can fall below
        4 and should not be trusted. Use ``method="unrestricted"`` or more
        data.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    >>> df["y"] = 1 + df["x1"] + 0.5 * df["x2"] + rng.normal(size=200)
    >>> df.loc[rng.choice(200, 30, replace=False), "x2"] = np.nan
    >>> pooled = sp.mi_estimate(sp.mice(df, m=5, seed=0), sp.regress,
    ...                         formula="y ~ x1 + x2")
    >>> res = sp.mi_test(pooled, ["x1", "x2"])
    >>> res["df1"], res["df_adjustment"]
    (2, 'small')

    References
    ----------
    [@reiter2007small] [@barnard1999small] [@marchenko2009improved];
    Stata 18 [MI] manual, ``mi estimate`` methods and formulas.
    """
    method = str(method).lower()
    if method not in ("equal_fmi", "unrestricted"):
        raise MethodIncompatibility(
            f"mi_test: method must be 'equal_fmi' or 'unrestricted', got {method!r}.",
            recovery_hint="Use method='equal_fmi' (Stata default) or 'unrestricted'.",
        )
    for key in ("ubar_matrix", "b_matrix", "params", "var_names", "n_imputations"):
        if key not in pooled:
            raise MethodIncompatibility(
                f"mi_test: the pooled result has no {key!r}; pass the output "
                "of sp.mi_estimate.",
                recovery_hint="Pool with sp.mi_estimate first.",
            )
    names: List[str] = [terms] if isinstance(terms, str) else list(terms)
    var_names = list(pooled["var_names"])
    missing = [t for t in names if t not in var_names]
    if not names or missing or len(set(names)) != len(names):
        raise MethodIncompatibility(
            (
                f"mi_test: terms must be distinct pooled coefficient names; "
                f"unknown: {missing}"
                if missing
                else "mi_test: terms must be a non-empty list of distinct names."
            ),
            recovery_hint=f"Choose from {var_names}.",
        )
    idx = [var_names.index(t) for t in names]
    m = int(pooled["n_imputations"])
    q = np.asarray(pooled["params"], dtype=float)[idx]
    U = np.asarray(pooled["ubar_matrix"], dtype=float)[np.ix_(idx, idx)]
    B = np.asarray(pooled["b_matrix"], dtype=float)[np.ix_(idx, idx)]
    k = len(idx)
    if np.linalg.matrix_rank(U) < k:
        raise MethodIncompatibility(
            "mi_test: the within-imputation covariance of the tested terms is "
            "singular; the restrictions are not jointly testable.",
            recovery_hint="Drop collinear terms from the test.",
        )
    dfcom = float(pooled.get("dfcom", np.inf))
    use_small = bool(small) and np.isfinite(dfcom)
    Uinv = np.linalg.inv(U)
    T = U + (1.0 + 1.0 / m) * B
    r = (1.0 + 1.0 / m) * float(np.trace(B @ Uinv)) / k
    t = k * (m - 1)

    if method == "equal_fmi":
        F = float(q @ Uinv @ q) / (k * (1.0 + r))
    else:
        F = float(q @ np.linalg.solve(T, q)) / k

    with np.errstate(divide="ignore"):
        # Large-sample df of the univariate / equal-FMI t <= 4 cases.
        inv_nu_large = 0.0 if r <= 0 else 1.0 / ((m - 1) * (1.0 + 1.0 / r) ** 2)
        if use_small:
            dfcom_star = dfcom * (dfcom + 1.0) / (dfcom + 3.0)
            g = (1.0 + 1.0 / m) * float(np.trace(np.linalg.solve(T, B))) / k
            inv_nu_obs = 1.0 / ((1.0 - g) * dfcom_star)
            if k == 1 or method == "unrestricted":
                df2 = 1.0 / (inv_nu_large + inv_nu_obs)
            elif t > 4:
                a = r * t / (t - 2.0)
                if dfcom_star <= 4.0 * (1.0 + a):
                    warnings.warn(
                        f"mi_test: Reiter's (2007) small-sample df is outside "
                        f"its range (complete-data df {dfcom:g} is too small "
                        f"for the missing information, r = {r:.3g}); the df "
                        "matches Stata but is not reliable.",
                        AssumptionWarning,
                        stacklevel=2,
                    )
                df2 = _reiter_df(t, r, dfcom_star)
            else:
                df2 = (k + 1.0) / 2.0 / (inv_nu_large + inv_nu_obs)
        elif r <= 0:
            df2 = np.inf
        elif k == 1:
            df2 = (m - 1) * (1.0 + 1.0 / r) ** 2
        elif method == "unrestricted":
            df2 = (m - 1) * (1.0 + 1.0 / r) ** 2
        elif t > 4:
            df2 = 4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / r) ** 2
        else:
            df2 = t * (1.0 + 1.0 / k) * (1.0 + 1.0 / r) ** 2 / 2.0
    df2 = float(df2)
    pvalue = (
        float(stats.chi2.sf(F * k, k))
        if not np.isfinite(df2)
        else float(stats.f.sf(F, k, df2))
    )
    return {
        "F": F,
        "df1": k,
        "df2": df2,
        "pvalue": pvalue,
        "rvi": r,
        "method": method,
        "terms": names,
        "n_imputations": m,
        "dfcom": dfcom,
        "df_adjustment": "small" if use_small else "large",
    }


__all__ = ["mi_test"]
