"""Joint Wald tests after multiple imputation (Stata ``mi test``).

Two tests of ``H0: Q_j = 0`` for a set of ``k`` pooled coefficients, both
built from the pooled within / between covariances ``U`` and ``B`` over
``m`` imputations and the average relative increase in variance
``r = (1 + 1/m) tr(B U^{-1}) / k``:

* ``method="equal_fmi"`` (Stata's default):
  assumes ``B`` proportional to ``U`` and uses
  ``F = Q' U^{-1} Q / (k (1 + r))`` on ``(k, nu)`` df with, for
  ``t = k (m - 1) > 4``, ``nu = 4 + (t - 4) [1 + (1 - 2/t) / r]^2`` and
  otherwise ``nu = t (1 + 1/k) (1 + 1/r)^2 / 2``.
* ``method="unrestricted"`` (Stata ``ufmitest``): ``F = Q' T^{-1} Q / k``
  with ``T = U + (1 + 1/m) B`` and ``nu = (m - 1)(1 + 1/r)^2``. ``B`` is
  estimated from ``m`` draws, so this is unreliable unless ``m`` is large
  relative to ``k``.

These are the large-sample degrees of freedom, i.e. Stata ``mi test,
nosmall`` (``tests/reference_parity/test_mi_test_parity.py`` pins both
methods against Stata 18 to 1e-9). Stata's default additionally applies
the small-sample df adjustment of Reiter (2007) when the complete-data df
is finite; that adjustment is not implemented, and ``small=True`` raises
rather than silently returning the large-sample df.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np
from scipy import stats

from ..exceptions import MethodIncompatibility


def mi_test(
    pooled: Dict[str, Any],
    terms: Union[str, Sequence[str]],
    *,
    method: str = "equal_fmi",
    small: bool = False,
) -> Dict[str, Any]:
    """
    Joint Wald test of pooled multiple-imputation coefficients.

    Parameters
    ----------
    pooled : dict
        Output of :func:`sp.mi_estimate` (it carries the within / between
        covariances ``ubar_matrix`` / ``b_matrix``).
    terms : str or list of str
        Coefficient names tested jointly equal to zero (``var_names``).
    method : {"equal_fmi", "unrestricted"}, default "equal_fmi"
        ``"equal_fmi"`` is Stata ``mi test``'s default test (equal fractions
        of missing information across the tested coefficients);
        ``"unrestricted"`` is ``mi test, ufmitest``.
    small : bool, default False
        Stata's small-sample (Reiter 2007) denominator df. Not implemented:
        ``True`` raises. The default reproduces ``mi test, nosmall``.

    Returns
    -------
    dict
        ``F``, ``df1``, ``df2``, ``pvalue``, ``rvi`` (average relative
        increase in variance), ``method``, ``terms`` and ``n_imputations``.

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
    >>> res["df1"]
    2

    References
    ----------
    Stata 18 [MI] manual, ``mi estimate`` methods and formulas, equations
    (5)-(6) and the references given there.
    """
    if small:
        raise MethodIncompatibility(
            "mi_test: the small-sample (Reiter 2007) denominator df is not "
            "implemented; the large-sample df (Stata `mi test, nosmall`) is.",
            recovery_hint="Call with small=False and report the test as "
            "large-sample, or compute the small-sample df in Stata.",
        )
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
    Uinv = np.linalg.inv(U)
    r = (1.0 + 1.0 / m) * float(np.trace(B @ Uinv)) / k
    with np.errstate(divide="ignore"):
        if method == "equal_fmi":
            F = float(q @ Uinv @ q) / (k * (1.0 + r))
            t = k * (m - 1)
            if r <= 0:
                df2 = np.inf
            elif t > 4:
                df2 = 4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / r) ** 2
            else:
                df2 = t * (1.0 + 1.0 / k) * (1.0 + 1.0 / r) ** 2 / 2.0
        else:
            T = U + (1.0 + 1.0 / m) * B
            F = float(q @ np.linalg.solve(T, q)) / k
            df2 = np.inf if r <= 0 else (m - 1) * (1.0 + 1.0 / r) ** 2
    pvalue = (
        float(stats.chi2.sf(F * k, k))
        if not np.isfinite(df2)
        else float(stats.f.sf(F, k, df2))
    )
    return {
        "F": F,
        "df1": k,
        "df2": float(df2),
        "pvalue": pvalue,
        "rvi": r,
        "method": method,
        "terms": names,
        "n_imputations": m,
    }


__all__ = ["mi_test"]
