"""
Prediction intervals for synthetic control methods.

``sp.scdata`` / ``sp.scest`` / ``sp.scpi`` implement the synthetic control
prediction and the prediction intervals of Cattaneo, Feng and Titiunik
[@cattaneo2021prediction] in the form documented for the ``scpi`` software
[@cattaneo2025scpi]: one treated unit, the outcome as the only feature, no
covariate adjustment and an identity weighting matrix.

* weights: ``simplex`` (``w >= 0, sum(w) = 1``), ``lasso``
  (``||w||_1 <= Q``), ``ridge`` (``||w||_2 <= Q`` with a data-driven Q),
  ``L1-L2`` (simplex and ``||w||_2 <= Q2``) and ``ols``;
* in-sample uncertainty: simulated bounds on ``p_t'(w_hat - w_0)``;
* out-of-sample uncertainty: sub-Gaussian, location-scale or
  quantile-regression bounds on the post-treatment error;
* simultaneous bounds over the post-treatment periods.

Clean-room implementation: written from the papers and the R ``scpi`` help
pages, with R ``scpi`` (GPL-3) run only as a black box to settle
conventions; no reference source was read. The specification and the
black-box evidence are in ``docs/dev/cleanroom/scpi.md``. Several treated
units (``sp.scest(treat=)``) are handled in ``_scest_multi``.

References
----------
[@cattaneo2021prediction],
[@cattaneo2025scpi]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _scpi_solvers as _sv
from ._scpi_inference import scpi_inference

__all__ = ["scdata", "scest", "scpi"]

_W_CONSTR = ("simplex", "lasso", "ridge", "ols", "L1-L2")
_W_ALIASES = {
    "ls": "ols",
    "LS": "ols",
    "l1-l2": "L1-L2",
    "L1L2": "L1-L2",
    "l1l2": "L1-L2",
}
_PI_TYPES = ("in_sample", "out_of_sample", "both")
_E_METHODS = {"gaussian": "subgaussian", "ls": "ls", "qreg": "qreg"}
_RADIUS_FLOOR = 0.5  # smallest data-driven ridge radius
_MIN_T0_RADIUS = 5  # below this many pre-periods the ridge lambda is not reported


def _sort_key(v: Any) -> Any:
    return (str(type(v)), v)


# ====================================================================== #
#  Data preparation
# ====================================================================== #


@accepts_aliases(id="unit", y="outcome")
def scdata(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
) -> Dict[str, Any]:
    """Prepare data matrices for synthetic control estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data (one row per unit-period).
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : Any
        Identifier of the treated unit.
    treatment_time : Any
        First treatment period (periods ``< treatment_time`` are
        pre-treatment).

    Returns
    -------
    dict
        ``A`` (pre-treatment outcome of the treated unit, length T0),
        ``B`` (donors' pre-treatment outcomes, T0 x J), ``C`` (``None``: no
        covariate adjustment), ``P`` (donors' post-treatment outcomes,
        T1 x J), ``J``, ``KM`` (0), ``T0``, ``T1``, the outcome series
        ``Y_pre`` / ``Y_post`` / ``Y_donors_pre`` / ``Y_donors_post``,
        ``donor_names`` (sorted), ``pre_times`` / ``post_times`` / ``times``,
        ``treated_unit`` and ``treatment_time``.

    Notes
    -----
    Duplicate (unit, time) rows and missing outcomes of the treated unit
    are refused. Donors with a missing pre-treatment outcome are dropped
    with a warning; a donor missing a post-treatment outcome is refused,
    because the synthetic control would be undefined in that period.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> d = sp.scdata(df, outcome="packspercapita", unit="state",
    ...               time="year", treated_unit="California",
    ...               treatment_time=1989)
    >>> d["B"].shape == (d["T0"], d["J"])
    True

    References
    ----------
    [@cattaneo2025scpi]
    """
    check_columns(
        data, {"outcome": outcome, "unit": unit, "time": time}, function="sp.scdata"
    )
    df = data[[unit, time, outcome]]
    if df.duplicated([unit, time]).any():
        raise MethodIncompatibility(
            "Duplicate (unit, time) rows in data.",
            recovery_hint="Aggregate or drop the repeated unit-period rows first.",
            diagnostics={"n_duplicates": int(df.duplicated([unit, time]).sum())},
        )
    if treated_unit not in set(df[unit]):
        raise MethodIncompatibility(
            f"Treated unit {treated_unit!r} not found in data.",
            recovery_hint=f"Pass one of the values of the {unit!r} column.",
        )
    wide = df.pivot(index=time, columns=unit, values=outcome).sort_index()
    times = wide.index.to_numpy()
    pre_mask = times < treatment_time
    T0, T1 = int(pre_mask.sum()), int((~pre_mask).sum())
    if T0 < 2:
        raise DataInsufficient(
            "Need at least 2 pre-treatment periods.",
            recovery_hint="Use a later treatment_time or a longer panel.",
            diagnostics={"n_pre_periods": T0},
        )
    if T1 < 1:
        raise DataInsufficient(
            "Need at least 1 post-treatment period.",
            recovery_hint="treatment_time must not be after the last period.",
            diagnostics={"n_post_periods": T1},
        )
    y_tr = wide[treated_unit].to_numpy(dtype=float)
    if np.isnan(y_tr).any():
        raise MethodIncompatibility(
            "The treated unit has missing outcome values.",
            recovery_hint="Impute or drop the periods where the treated "
            "unit's outcome is missing.",
            diagnostics={"missing_times": times[np.isnan(y_tr)].tolist()},
        )
    donors = sorted((u for u in wide.columns if u != treated_unit), key=_sort_key)
    Yd = wide[donors]
    bad_pre = [
        u for u in donors if np.isnan(Yd[u].to_numpy(dtype=float)[pre_mask]).any()
    ]
    if bad_pre:
        warnings.warn(
            f"Dropping donors with missing pre-treatment outcomes: {bad_pre}",
            UserWarning,
            stacklevel=2,
        )
        donors = [u for u in donors if u not in bad_pre]
        Yd = wide[donors]
    if not donors:
        raise DataInsufficient(
            "All donor units have missing pre-treatment data.",
            recovery_hint="Supply donors observed in every pre-treatment period.",
        )
    Ydv = Yd.to_numpy(dtype=float)
    bad_post = [u for j, u in enumerate(donors) if np.isnan(Ydv[~pre_mask, j]).any()]
    if bad_post:
        raise MethodIncompatibility(
            f"Donors {bad_post} have outcomes missing in the post-treatment period.",
            recovery_hint="Drop those donors or impute their post-treatment outcomes.",
            diagnostics={"donors": bad_post},
        )
    B = Ydv[pre_mask]
    P = Ydv[~pre_mask]
    return {
        "A": y_tr[pre_mask].copy(),
        "B": B,
        "C": None,
        "P": P,
        "J": len(donors),
        "KM": 0,
        "T0": T0,
        "T1": T1,
        "Y_pre": y_tr[pre_mask].copy(),
        "Y_post": y_tr[~pre_mask].copy(),
        "Y_donors_pre": B.copy(),
        "Y_donors_post": P.copy(),
        "donor_names": list(donors),
        "pre_times": times[pre_mask],
        "post_times": times[~pre_mask],
        "times": times,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
    }


# ====================================================================== #
#  Weight constraints and estimation
# ====================================================================== #


def _warn_deprecated_lambdas(lasso_lambda: Any, ridge_lambda: Any) -> None:
    if lasso_lambda is not None or ridge_lambda is not None:
        warnings.warn(
            "lasso_lambda / ridge_lambda are ignored: sp.scest and sp.scpi bound "
            "the norm of the weights instead of penalising it. Pass Q= (and "
            "Q2= for L1-L2) to set the bound.",
            DeprecationWarning,
            stacklevel=3,
        )


def _normalise_constr(w_constr: str) -> str:
    name = _W_ALIASES.get(w_constr, w_constr)
    if name not in _W_CONSTR:
        raise MethodIncompatibility(
            f"w_constr must be one of {_W_CONSTR} (or 'ls'), got {w_constr!r}.",
            recovery_hint="Use 'simplex' (default), 'lasso', 'ridge', "
            "'L1-L2' or 'ols'.",
            diagnostics={"w_constr": w_constr},
        )
    return name


def _positive(name: str, value: Any) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise MethodIncompatibility(
            f"{name} must be a positive number, got {value!r}.",
            recovery_hint=f"Pass {name} > 0 or leave it at the default.",
        )
    return v


def _ridge_radius(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """Data-driven ridge radius, floored at 0.5; lambda NaN for T0 < 5."""
    est = _sv.shrinkage_ridge(A, B, B.shape[1])
    Q = max(est["Q"], _RADIUS_FLOOR)
    lam = est["lambda"] if B.shape[0] >= _MIN_T0_RADIUS else np.nan
    return {"Q": float(Q), "lambda": float(lam)}


def _constraint_spec(
    A: np.ndarray,
    B: np.ndarray,
    name: str,
    Q: Optional[Any] = None,
    Q2: Optional[Any] = None,
) -> Dict[str, Any]:
    """Constraint description for the weights (outcome feature, V = I).

    Keys: ``name``, ``p`` (norm), ``dir`` (direction), ``lb`` (lower bound
    on each weight), ``Q`` (norm bound), ``Q2`` (L2 bound of ``L1-L2``) and
    ``lambda`` (shrinkage parameter of the data-driven ridge radius; ``None``
    when the radius is user-given or does not apply).
    """
    name = _normalise_constr(name)
    spec: Dict[str, Any] = {"name": name, "Q2": None, "lambda": None}
    if name == "simplex":
        spec.update(p="L1", dir="==", lb=0.0, Q=1.0 if Q is None else _positive("Q", Q))
    elif name == "lasso":
        spec.update(
            p="L1", dir="<=", lb=-np.inf, Q=1.0 if Q is None else _positive("Q", Q)
        )
    elif name == "ridge":
        spec.update(p="L2", dir="<=", lb=-np.inf)
        if Q is None:
            rr = _ridge_radius(np.asarray(A, float), np.asarray(B, float))
            spec.update(Q=rr["Q"], **{"lambda": rr["lambda"]})
        else:
            spec["Q"] = _positive("Q", Q)
    elif name == "L1-L2":
        spec.update(
            p="L1-L2", dir="==/<=", lb=0.0, Q=1.0 if Q is None else _positive("Q", Q)
        )
        if Q2 is None:
            rr = _ridge_radius(np.asarray(A, float), np.asarray(B, float))
            lam = rr["lambda"] if np.isfinite(rr["lambda"]) else 0.0
            spec.update(Q2=rr["Q"], **{"lambda": lam})
        else:
            spec["Q2"] = _positive("Q2", Q2)
    else:
        spec.update(p="no norm", dir=None, lb=-np.inf, Q=None)
    return spec


def _solve_weights(A: np.ndarray, B: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    name = spec["name"]
    if name == "simplex":
        return _sv.simplex_weights(B, A, spec["Q"])
    if name == "lasso":
        return _sv.lasso_ball(B, A, spec["Q"])
    if name == "ridge":
        return _sv.ridge_ball(B, A, spec["Q"])
    if name == "L1-L2":
        if spec["Q"] != 1.0:
            raise MethodIncompatibility(
                "L1-L2 weights sum to one; Q is fixed at 1.",
                recovery_hint="Drop Q= (use Q2= for the L2 bound).",
            )
        return _sv.simplex_l2(B, A, spec["Q2"])
    T0, J = B.shape
    if np.linalg.matrix_rank(B) < J:
        warnings.warn(
            f"OLS weights are not identified ({J} donors, rank "
            f"{np.linalg.matrix_rank(B)} "
            f"with T0 = {T0}); the minimum-norm least-squares weights are reported.",
            RuntimeWarning,
            stacklevel=3,
        )
    return np.asarray(np.linalg.lstsq(B, A, rcond=None)[0], dtype=float)


def _fit_weights(
    A: Any, B: Any, w_constr: Any, Q: Optional[Any] = None, Q2: Optional[Any] = None
) -> Any:
    """Weights and their constraint description: ``(w, spec)``."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    spec = _constraint_spec(A, B, w_constr, Q, Q2)
    return _solve_weights(A, B, spec), spec


@accepts_aliases(id="unit", y="outcome")
def scest(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    w_constr: str = "simplex",
    lasso_lambda: Optional[float] = None,
    ridge_lambda: Optional[float] = None,
    Q: Optional[float] = None,
    Q2: Optional[float] = None,
    treat: Optional[str] = None,
    effect: str = "unit-time",
    anticipation: int = 0,
) -> Dict[str, Any]:
    """Estimate synthetic control weights.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : Any, optional
        Identifier of the treated unit. With ``treat=``, optionally the
        subset of treated units to estimate (default all).
    treatment_time : Any, optional
        First treatment period of the single treated unit. Not used with
        ``treat=``, where each unit's adoption period is read from the
        treatment column.
    w_constr : str, default 'simplex'
        Constraint on the weights:

        - ``'simplex'``: ``w >= 0``, ``sum(w) = Q`` (Q = 1);
        - ``'lasso'``: ``||w||_1 <= Q`` (Q = 1), weights may be negative;
        - ``'ridge'``: ``||w||_2 <= Q``, by default
          ``Q = max(||b|| / (1 + lambda), 0.5)`` with ``b`` the least-squares
          weights (on a lasso-selected subset of donors when
          ``T0 - J <= 10``) and ``lambda = k sigma^2 / ||b||^2``;
        - ``'L1-L2'``: ``w >= 0``, ``sum(w) = 1`` and ``||w||_2 <= Q2``
          (Q2 as the ridge Q);
        - ``'ols'`` (alias ``'ls'``): unconstrained least squares.
    lasso_lambda, ridge_lambda : float, optional
        Deprecated and ignored (a ``DeprecationWarning`` is raised); the
        weights are norm-bounded, use ``Q`` / ``Q2``.
    Q : float, optional
        Norm bound for ``'simplex'`` / ``'lasso'`` / ``'ridge'``.
    Q2 : float, optional
        L2 bound for ``'L1-L2'``.
    treat : str, optional
        0/1 treatment column for several treated units with staggered
        adoption; each treated unit gets its own fit with the never-treated
        units as donors.
    effect : {'unit-time', 'unit', 'time'}, default 'unit-time'
        With ``treat=``: report every gap, each unit's average gap, or the
        average across units by horizon.
    anticipation : int, default 0
        With ``treat=``: number of periods before adoption excluded from
        the pre-treatment fit.

    Returns
    -------
    dict
        ``weights`` (array in ``donor_names`` order), ``w_constr``,
        ``w_constr_spec``, ``Y_synth_pre`` / ``Y_synth_post``,
        ``residuals_pre``, ``effects`` (post-treatment gaps), ``pre_rmspe``,
        ``donor_names`` and ``sc_data`` (the :func:`scdata` output). With
        ``treat=`` see :mod:`statspai.synth._scest_multi`.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> est = sp.scest(df, outcome="packspercapita", unit="state",
    ...                time="year", treated_unit="California",
    ...                treatment_time=1989)
    >>> round(float(est["weights"].sum()), 6)
    1.0

    References
    ----------
    [@cattaneo2025scpi]
    """
    _warn_deprecated_lambdas(lasso_lambda, ridge_lambda)
    if treat is not None:
        if treatment_time is not None:
            raise MethodIncompatibility(
                "sp.scest: with treat= each unit's adoption period comes from the "
                "treatment column; treatment_time= is not used.",
                recovery_hint="Drop treatment_time=, or drop treat= for one "
                "treated unit.",
            )
        from ._scest_multi import scest_multi

        return scest_multi(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            treat=treat,
            treated_unit=treated_unit,
            effect=effect,
            anticipation=anticipation,
            single=scest,
            fit_kw={"w_constr": w_constr, "Q": Q, "Q2": Q2},
        )
    if effect != "unit-time" or anticipation != 0:
        raise MethodIncompatibility(
            "sp.scest: effect= and anticipation= apply to several treated units "
            "(treat=); with one treated unit every post-treatment gap is "
            "reported and treatment_time= sets the window.",
            recovery_hint="Pass treat='<0/1 column>' or drop effect= / anticipation=.",
        )
    if treated_unit is None or treatment_time is None:
        raise MethodIncompatibility(
            "sp.scest: name the treated unit and its first treatment period "
            "(treated_unit=, treatment_time=), or pass a 0/1 treatment column "
            "as treat= for several treated units.",
            recovery_hint="e.g. treated_unit='California', treatment_time=1989.",
        )
    sc = scdata(data, outcome, unit, time, treated_unit, treatment_time)
    A, B, P = sc["A"], sc["B"], sc["P"]
    w, spec = _fit_weights(A, B, w_constr, Q, Q2)
    fit_pre, fit_post = B @ w, P @ w
    resid = A - fit_pre
    return {
        "weights": w,
        "w_constr": spec["name"],
        "w_constr_spec": spec,
        "Y_synth_pre": fit_pre,
        "Y_synth_post": fit_post,
        "residuals_pre": resid,
        "effects": sc["Y_post"] - fit_post,
        "pre_rmspe": float(np.sqrt(np.mean(resid**2))),
        "donor_names": sc["donor_names"],
        "sc_data": sc,
    }


# ====================================================================== #
#  Prediction intervals
# ====================================================================== #


def _bound_key(pi_type: str, e_method: str) -> str:
    if pi_type == "in_sample":
        return "insample"
    return _E_METHODS[e_method]


@accepts_aliases(id="unit", y="outcome")
def scpi(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    w_constr: str = "simplex",
    pi_type: str = "both",
    e_method: str = "gaussian",
    alpha: float = 0.05,
    cores: int = 1,
    seed: Optional[int] = None,
    lasso_lambda: Optional[float] = None,
    ridge_lambda: Optional[float] = None,
    Q: Optional[float] = None,
    Q2: Optional[float] = None,
    sims: int = 200,
    u_missp: bool = True,
    u_sigma: str = "HC1",
    u_order: int = 1,
    u_alpha: Optional[float] = None,
    e_order: int = 1,
    e_alpha: Optional[float] = None,
    rho: Union[str, float, None] = None,
    rho_max: float = 0.2,
    draws: Optional[np.ndarray] = None,
) -> CausalResult:
    """Prediction intervals for synthetic control methods.

    The synthetic control prediction error is bounded in two parts
    [@cattaneo2021prediction]: the in-sample error from estimating the
    weights (simulated: Gaussian draws of the score, and for each draw the
    extreme values of ``p_t'(w_hat - w)`` over the weights compatible with
    the simulated optimality condition and a locally relaxed constraint
    set), with level ``u_alpha``; and the out-of-sample error of the
    post-treatment shock, with level ``e_alpha``. The interval for the
    effect in period ``t`` has conditional coverage of at least
    ``1 - u_alpha - e_alpha``.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : Any
        Identifier of the treated unit.
    treatment_time : Any
        First treatment period.
    w_constr : str, default 'simplex'
        Weight constraint (``'simplex'``, ``'lasso'``, ``'ridge'``,
        ``'L1-L2'``, ``'ols'``); see :func:`scest`.
    pi_type : {'both', 'in_sample', 'out_of_sample'}, default 'both'
        Which interval is reported in ``period_results`` and ``ci``: the
        in-sample plus out-of-sample bounds, the in-sample bounds only, or
        the out-of-sample bounds only. All are kept in
        ``model_info['bounds']``.
    e_method : {'gaussian', 'ls', 'qreg'}, default 'gaussian'
        Out-of-sample bound: sub-Gaussian, location-scale or quantile
        regression.
    alpha : float, default 0.05
        Default for both ``u_alpha`` and ``e_alpha``; the reported interval
        has nominal coverage ``1 - u_alpha - e_alpha``.
    cores : int, default 1
        Accepted for compatibility; the simulation runs serially.
    seed : int, optional
        Seed of the standard normal draws.
    lasso_lambda, ridge_lambda : float, optional
        Deprecated and ignored; see :func:`scest`.
    Q : float, optional
        Norm bound of the weights for ``'simplex'`` / ``'lasso'`` /
        ``'ridge'``; see :func:`scest`.
    Q2 : float, optional
        L2 bound of the weights for ``'L1-L2'``; see :func:`scest`.
    sims : int, default 200
        Number of simulation draws (at least 10).
    u_missp : bool, default True
        Treat the weight model as misspecified: the conditional mean of the
        pseudo-residuals is estimated (by least squares on the active donors
        and a constant) instead of being set to zero.
    u_sigma : {'HC0', 'HC1', 'HC2', 'HC3', 'HC4'}, default 'HC1'
        Heteroskedasticity-robust variance of the pseudo-residuals.
    u_order : {0, 1}, default 1
        Residual design for the conditional mean: a constant (0) or the
        active donors and a constant (1). Falls back to 0 when
        ``T0 <= columns + 10``.
    u_alpha : float, optional
        Level of the in-sample bounds (default ``alpha``).
    e_order : {0, 1}, default 1
        Design for the out-of-sample moments, as ``u_order``.
    e_alpha : float, optional
        Level of the out-of-sample bounds (default ``alpha``).
    rho : {'type-2', 'type-1'} or float, optional
        Sparsity threshold that decides which weights are treated as zero
        (binding constraints); default ``'type-2'``.
    rho_max : float, default 0.2
        Upper bound on the data-driven rho.
    draws : np.ndarray, optional
        Standard normal draws, ``(n_donors, sims)``; overrides ``seed``.

    Returns
    -------
    CausalResult
        ``estimate`` is the average post-treatment effect and ``ci`` its
        interval (the same construction applied to the average over the
        post-treatment periods); ``se`` and ``pvalue`` are NaN (a
        prediction interval has no standard error). ``detail`` /
        ``model_info['period_results']`` hold the per-period effects with
        ``pi_lower`` / ``pi_upper``, the synthetic outcome with its band and
        the simultaneous bounds ``joint_lower`` / ``joint_upper``.
        ``model_info`` also carries every ingredient: ``bounds`` (in-sample,
        sub-Gaussian, location-scale, quantile-regression and joint bounds
        on the counterfactual minus the synthetic control), ``CI``,
        ``rho``, ``Sigma``, ``u_mean`` / ``u_var``, ``e_mean`` / ``e_var``,
        ``df``, ``vsig`` (per-draw in-sample values) and ``failed_sims``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> r = sp.scpi(df, outcome="packspercapita", unit="state", time="year",
    ...             treated_unit="California", treatment_time=1989,
    ...             sims=20, seed=0)
    >>> r.ci[0] < r.ci[1]
    True

    References
    ----------
    [@cattaneo2021prediction],
    [@cattaneo2025scpi]
    """
    _warn_deprecated_lambdas(lasso_lambda, ridge_lambda)
    if pi_type not in _PI_TYPES:
        raise MethodIncompatibility(
            "pi_type must be 'in_sample', 'out_of_sample', or 'both', got "
            f"{pi_type!r}.",
            recovery_hint="Use pi_type='both' (default).",
        )
    if e_method not in _E_METHODS:
        raise MethodIncompatibility(
            f"e_method must be 'gaussian', 'ls', or 'qreg', got {e_method!r}.",
            recovery_hint="Use e_method='gaussian' (default); all three are "
            "in model_info['bounds'].",
        )
    if int(sims) < 10:
        raise MethodIncompatibility(
            f"sims must be >= 10, got {sims}.",
            recovery_hint="Use sims=200 (default) or more.",
        )
    u_alpha = float(alpha) if u_alpha is None else float(u_alpha)
    e_alpha = float(alpha) if e_alpha is None else float(e_alpha)

    est = scest(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        w_constr=w_constr,
        Q=Q,
        Q2=Q2,
    )
    sc, w, spec = est["sc_data"], est["weights"], est["w_constr_spec"]
    A, B, P = sc["A"], sc["B"], sc["P"]
    T0 = sc["T0"]
    lam = spec.get("lambda")
    if spec["name"] == "ridge" and lam is not None and not np.isfinite(lam):
        raise DataInsufficient(
            f"sp.scpi: the data-driven ridge radius needs at least {_MIN_T0_RADIUS} "
            f"pre-treatment periods (T0 = {T0}), because its shrinkage parameter "
            "sets the degrees of freedom of the variance estimator.",
            recovery_hint="Pass the radius yourself, e.g. Q=1.0, or use "
            "w_constr='simplex'.",
            diagnostics={"n_pre_periods": T0},
        )

    rng = np.random.default_rng(seed)
    inf = scpi_inference(
        A,
        B,
        P,
        w,
        spec,
        sims=int(sims),
        draws=draws,
        rng=rng,
        u_missp=u_missp,
        u_sigma=u_sigma,
        u_order=u_order,
        u_alpha=u_alpha,
        e_order=e_order,
        e_alpha=e_alpha,
        rho=rho,
        rho_max=rho_max,
        aggregate=True,
    )
    return _assemble(sc, est, inf, pi_type, e_method, u_alpha, e_alpha)


def _assemble(
    sc: Dict[str, Any],
    est: Dict[str, Any],
    inf: Dict[str, Any],
    pi_type: str,
    e_method: str,
    u_alpha: float,
    e_alpha: float,
) -> CausalResult:
    """Package the inference output as a :class:`CausalResult`."""
    w = est["weights"]
    fit_post = est["Y_synth_post"]
    y_post = sc["Y_post"]
    effects = y_post - fit_post
    bounds = inf["bounds"]
    e_b = inf["e_bounds"]
    key = _bound_key(pi_type, e_method)
    if pi_type == "out_of_sample":
        band = e_b[_E_METHODS[e_method]]
    else:
        band = bounds[key]
    joint = bounds["joint"]
    # non-coverage of the reported interval: both parts, or the one shown
    level = {"both": u_alpha + e_alpha, "in_sample": u_alpha, "out_of_sample": e_alpha}[
        pi_type
    ]
    period = pd.DataFrame(
        {
            "time": sc["post_times"],
            "effect": effects,
            "pi_lower": y_post - fit_post - band[:, 1],
            "pi_upper": y_post - fit_post - band[:, 0],
            "synthetic": fit_post,
            "synthetic_lower": fit_post + band[:, 0],
            "synthetic_upper": fit_post + band[:, 1],
            "joint_lower": y_post - fit_post - joint[:, 1],
            "joint_upper": y_post - fit_post - joint[:, 0],
        }
    )
    agg = inf["aggregate"]
    agg_key = "e_" + _E_METHODS[e_method] if pi_type == "out_of_sample" else key
    a_lo, a_hi = agg[agg_key]
    y_bar, fit_bar = float(np.mean(y_post)), float(np.mean(fit_post))
    estimate = float(np.mean(effects))
    ci = (y_bar - (fit_bar + float(a_hi)), y_bar - (fit_bar + float(a_lo)))

    times = sc["times"]
    y_all = np.concatenate([sc["Y_pre"], y_post])
    synth_all = np.concatenate([est["Y_synth_pre"], fit_post])
    gap_table = pd.DataFrame(
        {
            "time": times,
            "treated": y_all,
            "synthetic": synth_all,
            "gap": y_all - synth_all,
        }
    )
    agg_out = {
        k: np.asarray(v, float) for k, v in agg.items() if k not in ("e_mean", "e_var")
    }
    model_info: Dict[str, Any] = {
        "period_results": period,
        "weights": dict(zip(sc["donor_names"], (float(x) for x in w))),
        "w_constr": est["w_constr"],
        "w_constr_spec": est["w_constr_spec"],
        "pi_type": pi_type,
        "e_method": e_method,
        "sigma_hat": float(np.std(est["residuals_pre"], ddof=1)),
        "treatment_time": sc["treatment_time"],
        "treated_unit": sc["treated_unit"],
        "gap_table": gap_table,
        "Y_synth": synth_all,
        "Y_treated": y_all,
        "times": times,
        "n_donors": sc["J"],
        "n_pre_periods": sc["T0"],
        "n_post_periods": sc["T1"],
        "pre_rmspe": est["pre_rmspe"],
        "u_alpha": u_alpha,
        "e_alpha": e_alpha,
        "nominal_coverage": 1.0 - level if pi_type == "both" else None,
        "aggregate_bounds": agg_out,
        "bounds": bounds,
        "CI": {k: fit_post[:, None] + v for k, v in bounds.items()},
    }
    for k in (
        "rho",
        "Q_star",
        "Q2_star",
        "lb",
        "df",
        "u_mean",
        "u_var",
        "u_T",
        "u_params",
        "u_order",
        "Sigma",
        "e_mean",
        "e_var",
        "e_T",
        "e_params",
        "e_order",
        "failed_sims",
        "n_slsqp_fallback",
        "sims",
        "vsig",
    ):
        model_info[k] = inf[k]
    return CausalResult(
        method="SCM with Prediction Intervals (Cattaneo et al. 2021)",
        estimand="ATT",
        estimate=estimate,
        se=float("nan"),
        pvalue=float("nan"),
        ci=ci,
        alpha=level,
        n_obs=int(sc["T0"] + sc["T1"]),
        detail=period,
        model_info=model_info,
        _citation_key="scpi",
    )


# ====================================================================== #
#  Citation registration
# ====================================================================== #

# verified against Crossref and doi.org (10.1080/01621459.2021.1979561)
CausalResult._CITATIONS["scpi"] = (
    "@article{cattaneo2021prediction,\n"
    "  title={Prediction Intervals for Synthetic Control Methods},\n"
    "  author={Cattaneo, Matias D. and Feng, Yingjie and Titiunik, Rocio},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1865--1880},\n"
    "  year={2021},\n"
    "  doi={10.1080/01621459.2021.1979561}\n"
    "}"
)
