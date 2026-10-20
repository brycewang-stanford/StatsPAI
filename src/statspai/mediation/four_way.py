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
and estimated from regressions of ``M`` on ``A, C`` and ``Y`` on ``A, M,
A*M, C``: the parametric VanderWeele (2014) four-way decomposition, with
a linear, logistic or Poisson outcome and a linear or logistic mediator
(Stata ``med4way``). With a logistic or Poisson outcome the components
are on the excess-relative-risk scale.

References
----------
VanderWeele, T. J. (2014). "A unification of mediation and
interaction: a four-way decomposition." *Epidemiology*, 25(5),
749-761. [@vanderweele2014effect]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .._input_validation import clean_frame
from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..exceptions import MethodIncompatibility


@dataclass
class FourWayResult(ResultProtocolMixin):
    """Result container for :func:`four_way_decomposition`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
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
        scale = self.detail.get("scale", "difference")
        return (
            "VanderWeele (2014) Four-Way Decomposition\n"
            "-----------------------------------------\n"
            f"  N               : {self.n_obs}\n"
            f"  Scale           : {scale}\n"
            f"  Total Effect    : {self.total_effect:+.4f}\n"
            f"  CDE(0)          : {self.cde:+.4f}  ({prop.get('cde', 0):.1%})\n"
            f"  INT_ref         : {self.int_ref:+.4f}  ({prop.get('int_ref', 0):.1%})\n"
            f"  INT_med         : {self.int_med:+.4f}  ({prop.get('int_med', 0):.1%})\n"
            f"  PIE             : {self.pie:+.4f}  ({prop.get('pie', 0):.1%})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"FourWayResult(TE={self.total_effect:+.4f})"


@records_call("four_way_decomposition")
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
    yreg: str = "linear",
    mreg: str = "linear",
    covariate_values: Optional[Mapping[str, float]] = None,
) -> FourWayResult:
    """
    Parametric four-way decomposition of TE = CDE + INT_ref + INT_med + PIE.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, mediator : str
    covariates : sequence of str, optional
        Entered linearly in both the outcome and the mediator model.
    a0, a1 : float
        Reference and comparison levels of the treatment (default 0, 1).
    m0 : float
        Mediator reference level at which CDE is evaluated.
    vce : {'ols', 'ml'}, default 'ols'
        Variance used for linear models (a linear outcome, a normal
        mediator). ``'ols'`` uses ``s^2 = RSS / (n - p)`` -- R
        ``CMAverse::cmest(estimation = "paramfunc")``; ``'ml'`` uses
        ``RSS / n`` -- Stata ``med4way``, which fits linear models by
        maximum likelihood. With a normal mediator and a log-linear
        outcome the mediator variance enters the components, so the two
        choices also differ in the point estimates; its sampling variance
        ``2 sigma^4 / dof`` enters the standard errors.
    yreg : {'linear', 'logistic', 'poisson'}, default 'linear'
        Outcome model, with the exposure-mediator interaction. For
        ``'logistic'`` and ``'poisson'`` the components are on the
        excess-relative-risk scale (Stata ``med4way``'s ``ereri_*``): they
        sum to ``RR - 1``. A logistic outcome is read as a risk, which needs
        a rare outcome (odds ratio close to the risk ratio), as in
        ``med4way`` and ``CMAverse``.
    mreg : {'linear', 'logistic'}, default 'linear'
        Mediator model: normal linear regression, or logistic regression
        for a 0/1 mediator.
    covariate_values : mapping, optional
        Values of the covariates at which the decomposition is evaluated
        (``med4way``'s ``c()``); covariates not named are held at their
        sample means, the default for all of them.

    Returns
    -------
    FourWayResult
        ``se`` holds delta-method standard errors for ``total_effect``,
        ``cde``, ``int_ref``, ``int_med``, ``pie`` and the proportions
        (``prop_cde`` ... ``prop_pie``, ``prop_mediated``,
        ``prop_interaction``, ``prop_eliminated`` -- ``med4way``'s
        ``p_*``, ``op_m``, ``op_ati``, ``op_e``). ``detail['scale']`` is
        ``'difference'`` or ``'excess_relative_risk'``; on the latter
        ``detail['total_rr']`` is the ratio itself.

    Notes
    -----
    Every component is a closed-form function of the two regressions
    (``mediation/_four_way_engine.py``); the Jacobian for the delta method
    is computed by complex-step differentiation, so it carries no
    truncation error. Against Stata ``med4way`` 2.3.2 (``vce='ml'``) the
    estimates agree to 1e-8 and the standard errors to 1e-7 for every
    outcome / mediator model (``tests/reference_parity/
    test_four_way_nonlinear_parity.py``). With ``vce='ols'`` the estimates
    equal ``CMAverse``'s; its standard errors leave out the mediator
    variance's sampling error.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
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
    from . import _four_way_engine as eng

    cov = list(covariates or [])
    if vce not in ("ols", "ml"):
        raise MethodIncompatibility(
            f"vce must be 'ols' or 'ml', got {vce!r}",
            recovery_hint="Use vce='ols' (CMAverse) or vce='ml' (med4way).",
        )
    if yreg not in ("linear", "logistic", "poisson"):
        raise MethodIncompatibility(
            f"yreg must be 'linear', 'logistic' or 'poisson', got {yreg!r}",
            recovery_hint="Pick the outcome model matching the outcome's type.",
        )
    if mreg not in ("linear", "logistic"):
        raise MethodIncompatibility(
            f"mreg must be 'linear' or 'logistic', got {mreg!r}",
            recovery_hint="Use mreg='logistic' for a 0/1 mediator.",
        )
    unknown = sorted(set(covariate_values or {}) - set(cov))
    if unknown:
        raise MethodIncompatibility(
            f"covariate_values names {unknown}, which are not in covariates.",
            recovery_hint="Give values only for columns listed in covariates=.",
        )

    df = clean_frame(
        data,
        [y, treat, mediator] + cov,
        function="four_way_decomposition",
        n_params=4 + len(cov),  # outcome model: 1 + A + M + A*M + covariates
    )
    n = len(df)

    from ..core._validate import treatment_as_float

    A = treatment_as_float(df[treat], function="sp.four_way_decomposition")
    M = df[mediator].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    Xc = df[cov].to_numpy(dtype=float) if cov else np.zeros((n, 0))
    for col, vals, family in ((y, Y, yreg), (mediator, M, mreg)):
        if family == "logistic" and not np.isin(vals, (0.0, 1.0)).all():
            raise MethodIncompatibility(
                f"four_way_decomposition: logistic model for {col!r} needs a "
                "0/1 variable.",
                recovery_hint="Recode it to 0/1 or use the linear model.",
            )
        if family == "poisson" and (vals < 0).any():
            raise MethodIncompatibility(
                f"four_way_decomposition: Poisson model for {col!r} needs a "
                "non-negative outcome.",
                recovery_hint="Use yreg='linear' for outcomes that can be negative.",
            )

    X_out = np.column_stack([np.ones(n), A, M, A * M, Xc])
    X_med = np.column_stack([np.ones(n), A, Xc])

    def _linear(X: np.ndarray, v: np.ndarray):
        coef, *_ = np.linalg.lstsq(X, v, rcond=None)
        resid = v - X @ coef
        dof = X.shape[0] if vce == "ml" else X.shape[0] - X.shape[1]
        s2 = float(resid @ resid) / dof
        return coef, s2 * np.linalg.inv(X.T @ X), s2, dof

    if yreg == "linear":
        theta, V_t, _, _ = _linear(X_out, Y)
    else:
        theta, V_t = eng.fit_glm(X_out, Y, yreg)
    blocks = [V_t]
    params = [theta]
    if mreg == "linear":
        beta, V_b, s2_m, dof_m = _linear(X_med, M)
        blocks.append(V_b)
        params.append(beta)
        if yreg != "linear":
            # The normal mediator's variance enters exp-scale components.
            params.append(np.array([s2_m]))
            blocks.append(np.array([[2.0 * s2_m**2 / dof_m]]))
        else:
            params.append(np.array([s2_m]))
            blocks.append(np.zeros((1, 1)))  # no role on the linear scale
    else:
        beta, V_b = eng.fit_glm(X_med, M, "logistic")
        blocks.append(V_b)
        params.append(beta)

    cbar = Xc.mean(axis=0) if cov else np.zeros(0)
    for j, name in enumerate(cov):
        if covariate_values and name in covariate_values:
            cbar[j] = float(covariate_values[name])

    from scipy.linalg import block_diag

    p_all = np.concatenate(params)
    V_all = block_diag(*blocks)
    f = eng.components(yreg, mreg, X_out.shape[1], X_med.shape[1], cbar, a0, a1, m0)
    with np.errstate(divide="ignore", invalid="ignore"):
        est, se_all = eng.delta_method(f, p_all, V_all)
    vals = eng.as_dict(est)
    ses = eng.as_dict(se_all)
    cde, int_ref = vals["cde"], vals["int_ref"]
    int_med, pie, te = vals["int_med"], vals["pie"], vals["total_effect"]

    # Degenerate total effect: zero up to rounding on the outcome's scale
    # (its SD on the difference scale; the ERR scale is unit-free).
    unit = float(np.std(Y)) if yreg == "linear" else 1.0
    prop_keys = [k for k in eng.NAMES if k.startswith("prop_")]
    if abs(te) <= 1e-10 * max(unit, 1e-300):
        warnings.warn(
            "four_way_decomposition: the total effect is zero up to rounding, "
            "so the proportions are undefined (NaN).",
            UserWarning,
            stacklevel=2,
        )
        for k in prop_keys:
            vals[k] = ses[k] = float("nan")
    prop = {
        "cde": vals["prop_cde"],
        "int_ref": vals["prop_int_ref"],
        "int_med": vals["prop_int_med"],
        "pie": vals["prop_pie"],
        "mediated": vals["prop_mediated"],
        "interaction": vals["prop_interaction"],
        "eliminated": vals["prop_eliminated"],
    }
    detail: Dict[str, Any] = {
        "theta": [float(v) for v in theta[:4]],
        "beta": [float(v) for v in beta[:2]],
        "vce": vce,
        "yreg": yreg,
        "mreg": mreg,
        "scale": "difference" if yreg == "linear" else "excess_relative_risk",
        "covariate_values": dict(zip(cov, (float(v) for v in cbar))),
    }
    if yreg != "linear":
        detail["total_rr"] = float(te + 1.0)
    if mreg == "linear":
        detail["mediator_variance"] = float(params[2][0])

    _result = FourWayResult(
        cde=float(cde),
        int_ref=float(int_ref),
        int_med=float(int_med),
        pie=float(pie),
        total_effect=float(te),
        proportions=prop,
        n_obs=n,
        detail=detail,
        se={k: ses[k] for k in eng.NAMES},
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
                "yreg": yreg,
                "mreg": mreg,
                "covariate_values": (
                    dict(covariate_values) if covariate_values else None
                ),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["four_way_decomposition", "FourWayResult"]
