"""
Hausman (1978) specification test.

Tests whether the fixed effects (FE) or random effects (RE) estimator
is appropriate for panel data. Under H0 (RE is consistent), both FE
and RE are consistent but RE is more efficient. Under H1, only FE
is consistent.

References
----------
Hausman, J.A. (1978).
"Specification Tests in Econometrics."
*Econometrica*, 46(6), 1251-1271. [@hausman1978specification]
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import AssumptionWarning


def hausman_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str,
    time: str,
    alpha: float = 0.05,
    sigmamore: bool = False,
) -> Dict[str, Any]:
    """
    Hausman test for FE vs RE in panel data.

    Equivalent to Stata's ``hausman fe re``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Dependent variable.
    x : list of str
        Independent variables (time-varying).
    id : str
        Unit identifier.
    time : str
        Time period identifier.
    alpha : float, default 0.05
    sigmamore : bool, default False
        Stata's ``hausman fe re, sigmamore``: both covariance matrices use the
        RE disturbance variance. Use it when the classical statistic is
        negative (``recommendation == "inconclusive"``).

    Returns
    -------
    dict
        ``'statistic'``: Hausman chi² statistic
        ``'df'``: degrees of freedom
        ``'pvalue'``: p-value (``nan`` when the statistic is negative)
        ``'recommendation'``: 'FE', 'RE', or 'inconclusive' when
        ``V_FE - V_RE`` is not positive semi-definite and the statistic is
        negative (an ``AssumptionWarning`` is raised)
        ``'psd_violation'``: whether ``V_FE - V_RE`` fails to be positive
        definite
        ``'beta_fe'``, ``'beta_re'``: coefficient vectors

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_panel(n_units=40, n_periods=8, seed=0)
    >>> result = sp.hausman_test(df, y='y', x=['x'], id='unit', time='time')
    >>> bool(result['recommendation'] in ('FE', 'RE'))
    True
    >>> bool(0.0 <= result['pvalue'] <= 1.0)
    True

    Notes
    -----
    The test statistic is:

    .. math::
        H = (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})'
        [V(\\hat{\\beta}_{FE}) - V(\\hat{\\beta}_{RE})]^{-1}
        (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})

    Under H0, H ~ χ²(k).

    - Reject H0 (p < 0.05) → Use Fixed Effects
    - Fail to reject → Use Random Effects (more efficient)

    See Hausman (1978, *Econometrica*).
    """
    # One implementation, pinned to Stata (xtreg fe / xtreg re / hausman) in
    # tests/reference_parity/test_hausman_stata_parity.py. The within / GLS
    # estimators this function used to carry mis-estimated the RE variance
    # (statistic 223 where Stata reports 4.4 on the same panel).
    from ..panel.panel_diagnostics import _hausman_from_data

    return _hausman_from_data(data, y, x, id, time, alpha, sigmamore=sigmamore)


def _hausman_decision(
    b_diff: np.ndarray,
    V_diff: np.ndarray,
    k: int,
    alpha: float,
) -> Dict[str, Any]:
    """Statistic, p-value and FE / RE recommendation from the FE-RE contrast.

    The chi2(k) reference needs ``V_FE - V_RE`` positive definite. When it is
    not, the quadratic form can be negative; the old code clamped it to 0,
    which reported p = 1 and recommended RE -- an unwarranted conclusion from a
    test whose assumptions had failed. A negative statistic is now reported as
    is, with ``pvalue = nan`` and ``recommendation = "inconclusive"``; a
    non-negative one from a non-positive-definite difference keeps its p-value
    but warns.
    """
    try:
        H = float(b_diff @ np.linalg.inv(V_diff) @ b_diff)
    except np.linalg.LinAlgError:
        H = float(b_diff @ np.linalg.pinv(V_diff) @ b_diff)
    min_eig = float(np.linalg.eigvalsh((V_diff + V_diff.T) / 2).min())

    if H < 0:
        warnings.warn(
            f"Hausman statistic is negative (chi2({k}) = {H:.4f}): "
            "V_FE - V_RE is not positive semi-definite on these data, so the "
            "chi2 reference distribution does not apply and the test cannot "
            "choose between FE and RE. Rerun with sigmamore=True (Stata's "
            "`hausman, sigmamore`), or use the cluster-robust Mundlak test: "
            "sp.panel(..., method='mundlak', cluster='entity') and "
            "sp.test(result, '_mean_x1 _mean_x2 ...').",
            AssumptionWarning,
            stacklevel=3,
        )
        return {
            "statistic": H,
            "df": k,
            "pvalue": float("nan"),
            "recommendation": "inconclusive",
            "psd_violation": True,
            "interpretation": (
                f"chi2({k}) = {H:.4f} < 0: the FE-RE variance difference is "
                "not positive semi-definite, so the test is inconclusive."
            ),
        }
    if min_eig <= 0:
        warnings.warn(
            "V_FE - V_RE is not positive definite (smallest eigenvalue "
            f"{min_eig:.3g}); the chi2({k}) p-value of the Hausman test is "
            "unreliable.",
            AssumptionWarning,
            stacklevel=3,
        )
    pvalue = float(stats.chi2.sf(H, k))
    reject = pvalue <= alpha
    detail = (
        "Reject H0: use Fixed Effects."
        if reject
        else "Cannot reject H0: Random Effects is more efficient."
    )
    return {
        "statistic": H,
        "df": k,
        "pvalue": pvalue,
        "recommendation": "FE" if reject else "RE",
        "psd_violation": min_eig <= 0,
        "interpretation": f"chi2({k}) = {H:.4f}, p = {pvalue:.4f}. {detail}",
    }


CausalResult._CITATIONS["hausman"] = (
    "@article{hausman1978specification,\n"
    "  title={Specification Tests in Econometrics},\n"
    "  author={Hausman, Jerry A.},\n"
    "  journal={Econometrica},\n"
    "  volume={46},\n"
    "  number={6},\n"
    "  pages={1251--1271},\n"
    "  year={1978},\n"
    "  publisher={Wiley}\n"
    "}"
)
