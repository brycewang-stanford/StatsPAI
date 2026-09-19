"""
Prediction Intervals for Synthetic Control Methods (port of R ``scpi``).

``sp.scdata`` / ``sp.scest`` / ``sp.scpi`` reproduce the single-treated-unit,
outcome-only path of the R package ``scpi`` (Cattaneo, Feng, Palomba and
Titiunik; ``scdata`` -> ``scest`` -> ``scpi``) with ``V = "separate"``
(identity weighting matrix), no covariate adjustment and no constant:

* **weights** (``scest``): ``simplex`` (w >= 0, sum = Q, Q = 1), ``lasso``
  (||w||_1 <= Q, Q = 1), ``ridge`` (||w||_2 <= Q, Q from the shrinkage rule
  of ``shrinkage.EST``), ``L1-L2`` (simplex and ||w||_2 <= Q2) and ``ols``.
* **in-sample uncertainty**: the simulation of ``insampleUncertaintyGetDiag``
  -- conditional variance ``Sigma`` of the pseudo-residuals (``u.sigma``
  HC0-HC4, conditional mean from ``u.order``), the regularisation ``rho``
  (``type-1`` / ``type-2``), the locally relaxed constraint set of
  ``local.geom`` / ``local.geom.2step`` and, for each draw
  ``G = Sigma^{1/2} z``, the min / max of ``p_t'(b - beta)`` over
  ``{(b - beta)'Q(b - beta) - 2 G'(b - beta) <= 0}`` intersected with that set.
* **out-of-sample uncertainty** (``scpi.out``): ``gaussian`` (sub-Gaussian
  bound), ``ls`` (location-scale) and ``qreg`` (restricted regression
  quantiles, ``Qtools::rrq``), with the ``e.order`` design.
* **joint (simultaneous) bounds** of ``simultaneousPredGet``.

The optimisation problems that R hands to CVXR / ECOS / quantreg are solved
exactly here (active set, closed form, HiGHS LP; see ``_scpi_solvers``).

References
----------
[@cattaneo2021prediction]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from . import _scpi_solvers as _sv
from ._scpi_inference import scpi_inference

_W_CONSTR = ("simplex", "lasso", "ridge", "ols", "L1-L2")


# ====================================================================== #
#  Public API: scdata, scest, scpi
# ====================================================================== #


def scdata(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
) -> Dict[str, Any]:
    """
    Prepare data matrices for synthetic control estimation.

    Reshapes a long-format panel into the matrices of R ``scpi::scdata``
    (features = outcome only, no ``cov.adj``, ``constant = FALSE``):
    ``A`` (treated pre-treatment outcomes), ``B`` (donor pre-treatment
    outcomes, donors in sorted order as R's ``sort(B.names)``), ``C = None``
    and ``P`` (donor post-treatment outcomes).

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
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period (periods ``< treatment_time`` are pre-treatment).

    Returns
    -------
    dict
        Keys:

        - ``A`` / ``Y_pre``  : treated unit pre-treatment outcomes (T0,)
        - ``Y_post`` : treated unit post-treatment outcomes (T1,)
        - ``B`` / ``Y_donors_pre``  : donor pre-treatment matrix (T0, J)
        - ``P`` / ``Y_donors_post`` : donor post-treatment matrix (T1, J)
        - ``C`` : ``None`` (no covariate adjustment)
        - ``J``, ``KM`` (= 0), ``T0``, ``T1`` : dimensions as in R ``specs``
        - ``donor_names``   : list of donor unit labels (column order of B)
        - ``pre_times`` / ``post_times`` / ``times`` : time values
        - ``treated_unit`` / ``treatment_time`` : echoes

    Notes
    -----
    Donors with any missing pre-treatment outcome are dropped with a warning
    (R instead drops donors that are missing in *every* pre-treatment period
    and then the periods with any missing value); both agree on balanced
    panels.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()  # cols: state, year, packspercapita
    >>> prepared = sp.scdata(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> prepared['Y_pre'].shape  # 19 pre-treatment years (1970-1988)
    (19,)
    """
    if data.duplicated([unit, time]).any():
        raise ValueError(f"Duplicate ({unit}, {time}) rows in data.")
    pivot = data.pivot(index=time, columns=unit, values=outcome).sort_index()
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if post_mask.sum() < 1:
        raise ValueError("Need at least 1 post-treatment period.")  # pragma: no cover

    if treated_unit not in pivot.columns:
        raise ValueError(  # pragma: no cover
            f"Treated unit '{treated_unit}' not found in data."
        )

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    if np.isnan(Y_treated).any():
        raise ValueError("The treated unit has missing outcome values.")
    donor_cols = sorted((c for c in pivot.columns if c != treated_unit), key=str)

    if len(donor_cols) == 0:
        raise ValueError("No donor units found.")  # pragma: no cover

    Y_donors = pivot[donor_cols].values.astype(np.float64)

    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError(
            "All donor units have missing pre-treatment data."
        )  # pragma: no cover
    if not valid.all():
        dropped = [donor_cols[i] for i in range(len(donor_cols)) if not valid[i]]
        warnings.warn(
            f"Dropping donors with missing pre-treatment outcomes: {dropped}",
            UserWarning,
            stacklevel=2,
        )
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]
    if np.isnan(Y_donors[post_mask]).any():
        raise ValueError("Donor outcomes are missing in the post-treatment period.")

    A = Y_treated[pre_mask]
    B = Y_donors[pre_mask]
    P = Y_donors[post_mask]
    return {
        "A": A,
        "B": B,
        "C": None,
        "P": P,
        "J": B.shape[1],
        "KM": 0,
        "T0": int(pre_mask.sum()),
        "T1": int(post_mask.sum()),
        "Y_pre": A,
        "Y_post": Y_treated[post_mask],
        "Y_donors_pre": B,
        "Y_donors_post": P,
        "donor_names": donor_cols,
        "pre_times": times[pre_mask],
        "post_times": times[post_mask],
        "times": times,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
    }


def scest(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    w_constr: str = "simplex",
    lasso_lambda: Optional[float] = None,
    ridge_lambda: Optional[float] = None,
    Q: Optional[float] = None,
    Q2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Estimate synthetic control weights (R ``scpi::scest``).

    Solves min_w ||A - B w||^2 (``V`` = identity, R's ``V = "separate"``)
    over the constraint set named by ``w_constr``.

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
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period.
    w_constr : str, default 'simplex'
        Weight constraint, as in R ``w.constr = list(name = ...)``:

        - ``'simplex'`` : w >= 0, sum(w) = Q (Q = 1)
        - ``'lasso'``   : ||w||_1 <= Q (Q = 1), weights may be negative
        - ``'ridge'``   : ||w||_2 <= Q, Q from R's ``shrinkage.EST`` rule
          (``max(Q_hat, 0.5)`` with Q_hat = ||b_ols|| / (1 + lambda),
          lambda = sigma^2 J / ||b_ols||^2)
        - ``'L1-L2'``   : w >= 0, sum(w) = 1 and ||w||_2 <= Q2 (Q2 as ridge Q)
        - ``'ols'``     : unconstrained least squares (``'ls'`` is an alias)
    lasso_lambda, ridge_lambda : float, optional
        Deprecated and ignored (emit ``DeprecationWarning``).  Before 1.28.x
        these set L1 / L2 *penalties* of a different estimator; R ``scpi``
        constrains the norm of ``w`` instead -- use ``Q`` / ``Q2``.
    Q : float, optional
        Norm bound of the constraint (sum of weights for ``simplex``).
        Default: R's data-driven value.
    Q2 : float, optional
        L2 bound for ``'L1-L2'``.  Default: R's ridge rule.

    Returns
    -------
    dict
        Keys:

        - ``weights``       : np.ndarray (J,) of estimated donor weights
        - ``w_constr``      : constraint name
        - ``w_constr_spec`` : dict(name, p, dir, Q, Q2, lb, lambda) as in R
        - ``Y_synth_pre``   : synthetic pre-treatment outcomes (R ``Y.pre.fit``)
        - ``Y_synth_post``  : synthetic post-treatment outcomes (R ``Y.post.fit``)
        - ``residuals_pre`` : pre-treatment fit residuals
        - ``effects``       : post-treatment gaps (treated - synthetic)
        - ``pre_rmspe``     : root mean squared prediction error (pre)
        - ``donor_names``   : donor labels
        - ``sc_data``       : the prepared data dict from ``scdata``

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()  # cols: state, year, packspercapita
    >>> est = sp.scest(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> bool(est['pre_rmspe'] >= 0)  # pre-treatment fit RMSPE
    True
    >>> est['Y_synth_post'].shape  # 12 post-treatment years (1989-2000)
    (12,)

    References
    ----------
    [@cattaneo2021prediction]
    """
    _warn_deprecated_lambdas(lasso_lambda, ridge_lambda)
    sc = scdata(data, outcome, unit, time, treated_unit, treatment_time)
    w, spec = _fit_weights(sc["A"], sc["B"], w_constr, Q=Q, Q2=Q2)

    Y_synth_pre = sc["B"] @ w
    Y_synth_post = sc["P"] @ w
    residuals_pre = sc["A"] - Y_synth_pre
    effects = sc["Y_post"] - Y_synth_post
    pre_rmspe = float(np.sqrt(np.mean(residuals_pre**2)))

    return {
        "weights": w,
        "w_constr": spec["name"],
        "w_constr_spec": spec,
        "Y_synth_pre": Y_synth_pre,
        "Y_synth_post": Y_synth_post,
        "residuals_pre": residuals_pre,
        "effects": effects,
        "pre_rmspe": pre_rmspe,
        "donor_names": sc["donor_names"],
        "sc_data": sc,
    }


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
    """
    Prediction intervals for synthetic control methods (R ``scpi::scpi``).

    Port of the single-treated-unit path of R ``scpi`` (``effect =
    "unit-time"``): per post-period prediction intervals for the synthetic
    counterfactual combining in-sample (weight-estimation) uncertainty,
    quantified by simulation, and out-of-sample uncertainty, quantified by
    ``e_method``.  Interval for the treatment effect of period t is
    ``Y_t - [upper, lower]`` of the counterfactual interval.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome, unit, time : str
        Column names.
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period.
    w_constr : str, default 'simplex'
        Weight constraint; see :func:`scest`.
    pi_type : {'both', 'in_sample', 'out_of_sample'}, default 'both'
        Which interval is reported in ``period_results`` / ``ci``:
        ``'both'`` = R ``CI.all.<e_method>`` (in-sample + out-of-sample
        bounds), ``'in_sample'`` = R ``CI.in.sample``, ``'out_of_sample'`` =
        synthetic fit + out-of-sample bounds only (not an R table).
    e_method : {'gaussian', 'ls', 'qreg'}, default 'gaussian'
        Out-of-sample bound: sub-Gaussian, location-scale or quantile
        regression.  All three are always computed (``model_info``).
    alpha : float, default 0.05
        Default for both ``u_alpha`` and ``e_alpha``.  As in R, the
        combined interval has nominal coverage at least
        ``1 - (u_alpha + e_alpha)`` (0.90 with the defaults).
    cores : int, default 1
        Accepted for API compatibility; the simulation runs serially.
    seed : int, optional
        Seed of the ``numpy`` generator for the in-sample draws (R uses its
        own ``rnorm`` stream, so simulated bounds agree with R only up to
        Monte Carlo error unless ``draws`` is supplied).
    lasso_lambda, ridge_lambda : float, optional
        Deprecated and ignored; see :func:`scest`.
    Q, Q2 : float, optional
        Constraint bounds; see :func:`scest`.
    sims : int, default 200
        Number of in-sample simulation draws (R ``sims``; >= 10).
    u_missp : bool, default True
        Allow a misspecified conditional mean of the pseudo-residuals
        (R ``u.missp``).
    u_sigma : {'HC0','HC1','HC2','HC3','HC4'}, default 'HC1'
        Variance estimator of the pseudo-residuals.
    u_order : {0, 1}, default 1
        Order of the pseudo-residual mean model (R ``u.order``; ``u.lags``
        is fixed at 0).
    u_alpha, e_alpha : float, optional
        Levels of the in-sample / out-of-sample bounds (default ``alpha``).
    e_order : {0, 1}, default 1
        Order of the out-of-sample moment models (R ``e.order``;
        ``e.lags`` is fixed at 0).
    rho : {'type-1', 'type-2'} or float, optional
        Regularisation of the local geometry (default ``'type-2'``).
    rho_max : float, default 0.2
        Upper bound on ``rho``.
    draws : np.ndarray, optional
        ``(J, sims)`` matrix of standard-normal draws used for the in-sample
        simulation instead of generating them; passing the matrix R's
        ``rnorm`` produced reproduces R's simulated bounds.

    Returns
    -------
    CausalResult
        ``estimate`` = average post-treatment effect; ``ci`` = interval for
        that average (the R ``effect = "unit"`` construction: the averaged
        ``P`` row goes through the same simulation and out-of-sample
        models); ``se`` / ``pvalue`` are NaN (prediction intervals carry no
        standard error).  ``model_info`` holds ``period_results`` (per-period
        effect intervals), ``bounds`` (R ``inference.results$bounds``:
        insample / subgaussian / ls / qreg / joint), ``CI`` (the four
        synthetic-outcome tables), ``rho``, ``Q_star``, ``lb``, ``df``,
        ``u_mean``, ``u_var``, ``Sigma``, ``e_mean``, ``e_var``,
        ``failed_sims``, ``weights`` and the fit diagnostics.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()  # cols: state, year, packspercapita
    >>> # Subset donors to keep the simulation fast for this example
    >>> states = ['California', 'Alabama', 'Arkansas', 'Colorado',
    ...     'Connecticut', 'Delaware', 'Georgia', 'Illinois', 'Indiana']
    >>> df = df[df['state'].isin(states)]
    >>> result = sp.scpi(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989,
    ...     seed=42)
    >>> result.estimand
    'ATT'
    >>> bool(result.ci[0] < result.ci[1])  # interval for the average effect
    True

    >>> # In-sample (weight-estimation) uncertainty only
    >>> result_in = sp.scpi(df, outcome='packspercapita', unit='state',  # doctest: +SKIP
    ...     time='year', treated_unit='California', treatment_time=1989,
    ...     pi_type='in_sample', seed=42)

    >>> # Quantile-regression based out-of-sample component
    >>> result_qr = sp.scpi(df, outcome='packspercapita', unit='state',  # doctest: +SKIP
    ...     time='year', treated_unit='California', treatment_time=1989,
    ...     e_method='qreg', seed=42)

    References
    ----------
    [@cattaneo2021prediction]
    """
    if pi_type not in ("in_sample", "out_of_sample", "both"):
        raise ValueError(
            f"pi_type must be 'in_sample', 'out_of_sample', or 'both', "
            f"got '{pi_type}'."
        )
    if e_method not in ("gaussian", "ls", "qreg"):
        raise ValueError(
            f"e_method must be 'gaussian', 'ls', or 'qreg', " f"got '{e_method}'."
        )
    if sims < 10:
        raise ValueError("sims must be >= 10 (as in R scpi).")
    _warn_deprecated_lambdas(lasso_lambda, ridge_lambda)
    u_alpha = alpha if u_alpha is None else u_alpha
    e_alpha = alpha if e_alpha is None else e_alpha

    sc = scdata(data, outcome, unit, time, treated_unit, treatment_time)
    A, B, P = sc["A"], sc["B"], sc["P"]
    w, spec = _fit_weights(A, B, w_constr, Q=Q, Q2=Q2)

    rng = np.random.default_rng(seed)
    inf = scpi_inference(
        A,
        B,
        P,
        w,
        spec,
        sims=sims,
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

    Y_pre, Y_post = sc["Y_pre"], sc["Y_post"]
    fit_pre = B @ w
    fit_post = P @ w
    effects = Y_post - fit_post
    e_key = {"gaussian": "subgaussian", "ls": "ls", "qreg": "qreg"}[e_method]
    b = inf["bounds"]
    if pi_type == "in_sample":
        lo, hi = b["insample"][:, 0], b["insample"][:, 1]
    elif pi_type == "out_of_sample":
        lo, hi = inf["e_bounds"][e_key][:, 0], inf["e_bounds"][e_key][:, 1]
    else:
        lo, hi = b[e_key][:, 0], b[e_key][:, 1]
    sc_lo, sc_hi = fit_post + lo, fit_post + hi
    period_df = pd.DataFrame(
        {
            "time": sc["post_times"],
            "effect": effects,
            "pi_lower": Y_post - sc_hi,
            "pi_upper": Y_post - sc_lo,
            "synthetic": fit_post,
            "synthetic_lower": sc_lo,
            "synthetic_upper": sc_hi,
            "joint_lower": Y_post - (fit_post + b["joint"][:, 1]),
            "joint_upper": Y_post - (fit_post + b["joint"][:, 0]),
        }
    )

    agg = inf["aggregate"]
    att = float(np.mean(effects))
    if pi_type == "in_sample":
        a_lo, a_hi = agg["insample"]
    elif pi_type == "out_of_sample":
        a_lo, a_hi = agg["e_" + e_key]
    else:
        a_lo, a_hi = agg[e_key]
    ci = (att - a_hi, att - a_lo)

    Y_synth = np.concatenate([fit_pre, fit_post])
    Y_treated = np.concatenate([Y_pre, Y_post])
    e_pre = Y_pre - fit_pre
    gap_table = pd.DataFrame(
        {
            "time": sc["times"],
            "treated": Y_treated,
            "synthetic": Y_synth,
            "gap": Y_treated - Y_synth,
        }
    )
    model_info = {
        "period_results": period_df,
        "weights": dict(zip(sc["donor_names"], w)),
        "w_constr": spec["name"],
        "w_constr_spec": spec,
        "pi_type": pi_type,
        "e_method": e_method,
        "sigma_hat": float(np.std(e_pre, ddof=1)),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "gap_table": gap_table,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": sc["times"],
        "n_donors": B.shape[1],
        "n_pre_periods": sc["T0"],
        "n_post_periods": sc["T1"],
        "pre_rmspe": float(np.sqrt(np.mean(e_pre**2))),
        "u_alpha": u_alpha,
        "e_alpha": e_alpha,
        "nominal_coverage": 1.0 - (u_alpha + e_alpha) if pi_type == "both" else None,
        "aggregate_bounds": agg,
    }
    for k in (
        "bounds",
        "CI",
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
    model_info["CI"] = {k: fit_post[:, None] + v for k, v in inf["bounds"].items()}

    return CausalResult(
        method="SCM with Prediction Intervals (Cattaneo et al. 2021)",
        estimand="ATT",
        estimate=att,
        se=float("nan"),
        pvalue=float("nan"),
        ci=ci,
        alpha=(
            u_alpha + e_alpha
            if pi_type == "both"
            else (u_alpha if pi_type == "in_sample" else e_alpha)
        ),
        n_obs=len(Y_treated),
        detail=period_df,
        model_info=model_info,
        _citation_key="scpi",
    )


# ====================================================================== #
#  Weight estimation (w.constr.OBJ + b.est)
# ====================================================================== #


def _warn_deprecated_lambdas(lasso_lambda, ridge_lambda) -> None:
    if lasso_lambda is not None or ridge_lambda is not None:
        warnings.warn(
            "lasso_lambda / ridge_lambda are ignored: sp.scest / sp.scpi follow "
            "R scpi, which bounds ||w|| (pass Q / Q2) instead of penalising it.",
            DeprecationWarning,
            stacklevel=3,
        )


def _normalise_constr(w_constr: str) -> str:
    name = {"ls": "ols", "l1-l2": "L1-L2", "l1l2": "L1-L2"}.get(
        str(w_constr).lower(), str(w_constr)
    )
    if name not in _W_CONSTR:
        raise ValueError(
            f"w_constr must be one of {_W_CONSTR} (or 'ls'), got '{w_constr}'."
        )
    return name


def _constraint_spec(
    A: np.ndarray, B: np.ndarray, name: str, Q=None, Q2=None
) -> Dict[str, Any]:
    """R ``w.constr.OBJ`` for a single feature (the outcome), V = identity."""
    J = B.shape[1]
    spec: Dict[str, Any] = {"name": name, "Q2": None, "lambda": None}
    if name == "simplex":
        spec.update(p="L1", dir="==", lb=0.0, Q=1.0 if Q is None else float(Q))
    elif name == "ols":
        spec.update(p="no norm", dir=None, lb=-np.inf, Q=None)
    elif name == "lasso":
        spec.update(p="L1", dir="<=", lb=-np.inf, Q=1.0 if Q is None else float(Q))
    elif name in ("ridge", "L1-L2"):
        user = Q if name == "ridge" else Q2
        if user is None:
            if A.shape[0] >= 5:
                est = _sv.shrinkage_ridge(A, B, J)
                bound, lam = max(est["Q"], 0.5), est["lambda"]
            else:
                est = _sv.shrinkage_ridge(A, B, J)
                bound = max(est["Q"], 0.5)
                lam = np.nan if name == "ridge" else 0.0
        else:
            bound, lam = float(user), None
        if name == "ridge":
            spec.update(p="L2", dir="<=", lb=-np.inf, Q=bound, **{"lambda": lam})
        else:
            spec.update(
                p="L1-L2", dir="==/<=", lb=0.0, Q=1.0, Q2=bound, **{"lambda": lam}
            )
    return spec


def _solve_weights(A: np.ndarray, B: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    name = spec["name"]
    if name == "simplex":
        return _sv.qp_bounded_sum(B.T @ B, B.T @ A, np.zeros(B.shape[1]), spec["Q"])
    if name == "lasso":
        return _sv.lasso_ball(B, A, spec["Q"])
    if name == "ridge":
        return _sv.ridge_ball(B, A, spec["Q"])
    if name == "L1-L2":
        return _sv.simplex_l2(B, A, spec["Q2"])
    coef, rank = _sv.lm_coef(B, A)
    if rank < B.shape[1]:
        warnings.warn(
            "OLS weights are not identified (rank-deficient donor matrix); "
            "returning the minimum-norm least-squares solution.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.linalg.lstsq(B, A, rcond=None)[0]
    return coef


def _fit_weights(A, B, w_constr, Q=None, Q2=None):
    name = _normalise_constr(w_constr)
    spec = _constraint_spec(A, B, name, Q=Q, Q2=Q2)
    return _solve_weights(A, B, spec), spec


# ====================================================================== #
#  Citation registration
# ====================================================================== #

CausalResult._CITATIONS["scpi"] = (
    "@article{cattaneo2021prediction,\n"
    "  title={Prediction Intervals for Synthetic Control Methods},\n"
    "  author={Cattaneo, Matias D. and Feng, Yingjie "
    "and Titiunik, Rocio},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1865--1880},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
