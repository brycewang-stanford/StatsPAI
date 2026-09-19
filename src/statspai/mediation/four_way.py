"""
VanderWeele (2014) four-way decomposition of the total effect.

The total effect (TE) of a binary exposure :math:`A` on outcome
:math:`Y` (for a unit with covariates :math:`C`) decomposes into four
non-overlapping components:

.. math::

    TE = CDE(0) + INT_{ref} + INT_{med} + PIE.

* **Controlled Direct Effect (CDE(0))** — direct effect at :math:`M = 0`.
* **Reference Interaction (INT_ref)** — requires only mediator-exposure
  interaction.
* **Mediated Interaction (INT_med)** — requires both mediation AND
  interaction.
* **Pure Indirect Effect (PIE)** — pure mediation with no interaction.

Under no unmeasured confounding of A-Y, A-M, M-Y and no
exposure-induced mediator-outcome confounder, all four are identified
and estimated from simple regressions of ``M`` and ``Y`` on ``A, M,
A*M, C``. This is the standard parametric VanderWeele (2014)
four-way decomposition.

References
----------
VanderWeele, T. J. (2014). "A unification of mediation and
interaction: a four-way decomposition." *Epidemiology*, 25(5),
749-761. [@vanderweele2014effect]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .._input_validation import clean_frame
from .._result_serialize import ResultProtocolMixin

# sklearn is imported lazily inside ``four_way_decomposition`` so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches the four-way mediation decomposition.


@dataclass
class FourWayResult(ResultProtocolMixin):
    """Result container for :func:`four_way_decomposition`.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> treat = np.tile([0.0, 1.0], 50)
    >>> mediator = 0.5 + 0.4 * treat + np.repeat(np.linspace(-1, 1, 50), 2)
    >>> y = 1 + 2 * treat + 3 * mediator + 4 * treat * mediator
    >>> df = pd.DataFrame({"y": y, "a": treat, "m": mediator})
    >>> res = sp.four_way_decomposition(df, y="y", treat="a", mediator="m")
    >>> type(res).__name__
    'FourWayResult'
    >>> round(float(res.total_effect), 1)
    6.8
    """

    _citation_keys = ("vanderweele2014effect",)

    cde: float
    int_ref: float
    int_med: float
    pie: float
    total_effect: float
    proportions: Dict[str, float]
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)
    se: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (agent-native serialization)."""
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

    def summary(self) -> str:  # pragma: no cover
        prop = self.proportions
        return (
            "VanderWeele (2014) Four-Way Decomposition\n"
            "-----------------------------------------\n"
            f"  N               : {self.n_obs}\n"
            f"  Total Effect    : {self.total_effect:+.4f}\n"
            f"  CDE(0)          : {self.cde:+.4f}  ({prop.get('cde', 0):.1%})\n"
            f"  INT_ref         : {self.int_ref:+.4f}  ({prop.get('int_ref', 0):.1%})\n"
            f"  INT_med         : {self.int_med:+.4f}  ({prop.get('int_med', 0):.1%})\n"
            f"  PIE             : {self.pie:+.4f}  ({prop.get('pie', 0):.1%})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"FourWayResult(TE={self.total_effect:+.4f})"


def four_way_decomposition(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[Sequence[str]] = None,
    a0: float = 0.0,
    a1: float = 1.0,
    m0: float = 0.0,
    vce: str = "ols",
) -> FourWayResult:
    """
    Parametric four-way decomposition of TE = CDE + INT_ref + INT_med + PIE.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, mediator : str
    covariates : sequence of str, optional
    a0, a1 : float
        Reference and comparison levels of the treatment (default 0, 1).
    m0 : float
        Mediator reference level at which CDE is evaluated.
    vce : {'ols', 'ml'}, default 'ols'
        Covariance of the two regressions fed to the delta-method
        standard errors in ``result.se`` (covariates are held fixed at
        their sample means, as in the point estimates). ``'ols'`` uses
        ``s^2 (X'X)^{-1}`` with ``s^2 = RSS / (n - p)`` -- R
        ``CMAverse::cmest(estimation = "paramfunc", inference = "delta")``;
        ``'ml'`` uses ``RSS / n`` -- Stata ``med4way``, which fits both
        linear models by maximum likelihood.

    Returns
    -------
    FourWayResult
        ``se`` holds delta-method standard errors for ``total_effect``,
        ``cde``, ``int_ref``, ``int_med`` and ``pie``.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> treat = np.tile([0.0, 1.0], 50)
    >>> mediator = 0.5 + 0.4 * treat + np.repeat(np.linspace(-1, 1, 50), 2)
    >>> y = 1 + 2 * treat + 3 * mediator + 4 * treat * mediator
    >>> df = pd.DataFrame({"y": y, "a": treat, "m": mediator})
    >>> res = sp.four_way_decomposition(df, y="y", treat="a", mediator="m")
    >>> round(float(res.cde), 1)
    2.0
    >>> round(float(res.pie), 1)
    1.2
    """
    cov = list(covariates or [])
    if vce not in ("ols", "ml"):
        raise ValueError(f"vce must be 'ols' or 'ml', got {vce!r}")
    from sklearn.linear_model import LinearRegression

    df = clean_frame(
        data,
        [y, treat, mediator] + cov,
        function="four_way_decomposition",
        n_params=4 + len(cov),  # outcome model: 1 + A + M + A*M + covariates
    )
    n = len(df)

    A = df[treat].to_numpy(dtype=float)
    M = df[mediator].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    Xc = df[cov].to_numpy(dtype=float) if cov else np.zeros((n, 0))

    # Outcome model: Y ~ A + M + A*M + C
    X_out = np.column_stack([np.ones(n), A, M, A * M, Xc])
    lr = LinearRegression(fit_intercept=False).fit(X_out, Y)
    theta0 = float(lr.coef_[0])
    theta1 = float(lr.coef_[1])  # coeff on A
    theta2 = float(lr.coef_[2])  # coeff on M
    theta3 = float(lr.coef_[3])  # coeff on A*M

    # Mediator model: M ~ A + C
    X_med = np.column_stack([np.ones(n), A, Xc])
    mlm = LinearRegression(fit_intercept=False).fit(X_med, M)
    beta0 = float(mlm.coef_[0])
    beta1 = float(mlm.coef_[1])

    # E[M | A=a, C=Cbar]
    Cbar = Xc.mean(axis=0) if Xc.size else np.array([])
    EM_a0 = (
        beta0 + beta1 * a0 + (Cbar @ mlm.coef_[2:]) if Xc.size else beta0 + beta1 * a0
    )

    # Closed-form from VanderWeele (2014) Table 1:
    cde = (theta1 + theta3 * m0) * (a1 - a0)
    int_ref = theta3 * (EM_a0 - m0) * (a1 - a0)
    int_med = theta3 * beta1 * (a1 - a0) ** 2
    pie = (theta2 + theta3 * a0) * beta1 * (a1 - a0)

    te = cde + int_ref + int_med + pie

    # Delta-method standard errors. The outcome and mediator regressions
    # are fitted separately, so their coefficient covariances are
    # block-diagonal.
    def _vcov(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
        dof = X.shape[0] if vce == "ml" else X.shape[0] - X.shape[1]
        s2 = float(resid @ resid) / dof
        return s2 * np.linalg.inv(X.T @ X)

    V_t = _vcov(X_out, Y - X_out @ lr.coef_)
    V_b = _vcov(X_med, M - X_med @ mlm.coef_)
    dA = a1 - a0
    EM_minus = EM_a0 - m0
    beta_c = np.asarray(mlm.coef_[2:], dtype=float)
    Cfix = Cbar if Xc.size else np.zeros(0)

    def _grad(name: str):
        gt = np.zeros(X_out.shape[1])
        gb = np.zeros(X_med.shape[1])
        if name in ("cde", "te"):
            gt[1] += dA
            gt[3] += m0 * dA
        if name in ("int_ref", "te"):
            gt[3] += EM_minus * dA
            gb[0] += theta3 * dA
            gb[1] += theta3 * a0 * dA
            gb[2:] += theta3 * Cfix * dA
        if name in ("int_med", "te"):
            gt[3] += beta1 * dA**2
            gb[1] += theta3 * dA**2
        if name in ("pie", "te"):
            gt[2] += beta1 * dA
            gt[3] += a0 * beta1 * dA
            gb[1] += (theta2 + theta3 * a0) * dA
        return gt, gb

    se = {}
    for name, key in (
        ("te", "total_effect"),
        ("cde", "cde"),
        ("int_ref", "int_ref"),
        ("int_med", "int_med"),
        ("pie", "pie"),
    ):
        gt, gb = _grad(name)
        se[key] = float(np.sqrt(gt @ V_t @ gt + gb @ V_b @ gb))
    del beta_c
    if abs(te) > 1e-10:
        prop = {
            "cde": cde / te,
            "int_ref": int_ref / te,
            "int_med": int_med / te,
            "pie": pie / te,
        }
    else:
        prop = {"cde": 0.0, "int_ref": 0.0, "int_med": 0.0, "pie": 0.0}

    _result = FourWayResult(
        cde=float(cde),
        int_ref=float(int_ref),
        int_med=float(int_med),
        pie=float(pie),
        total_effect=float(te),
        proportions=prop,
        n_obs=n,
        detail={
            "theta": [theta0, theta1, theta2, theta3],
            "beta": [beta0, beta1],
            "vce": vce,
        },
        se=se,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.mediation.four_way_decomposition",
            params={
                "y": y,
                "treat": treat,
                "mediator": mediator,
                "covariates": list(covariates) if covariates else None,
                "a0": a0,
                "a1": a1,
                "m0": m0,
                "vce": vce,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["four_way_decomposition", "FourWayResult"]
