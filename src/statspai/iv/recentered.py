"""
Recentered instruments for treatments built from exogenous shocks.

Many treatments and instruments combine as-good-as-random shocks ``g``
with non-random exposure ``w`` through a known formula,
``z_i = f_i(g, w)``: market access from new rail lines, simulated
eligibility from state policies, spillovers through a network. Even
when ``g`` is randomly assigned, units with high exposure get high
``z_i`` in every draw of the shocks, so ``z_i`` is correlated with
anything that drives exposure. Borusyak and Hull show that the bias
runs through one variable, the expected instrument::

    mu_i = E[ f_i(g, w) | w ]

which the researcher computes by recomputing ``z`` under counterfactual
shock draws that were as likely to occur as the realised one and
averaging them. Two estimators then identify ``beta``:

- ``method='recenter'``: instrument the treatment with
  ``z_i - mu_i`` (Table 1, column 2 of the paper);
- ``method='control'``: instrument with ``z_i`` and add ``mu_i`` as a
  control (column 3).

When the treatment is the exposure itself (``treat == exposure``) the
first is an IV regression of ``y`` on ``z`` instrumented by
``z - mu`` and the second is OLS of ``y`` on ``z`` and ``mu``.

The same draws give randomization inference. Under the sharp null
``beta = b`` the statistic ``T(b) = sum_i w_i (z_i - mu_i)(y_i - b x_i)``
(``y`` and ``x`` residualised on the controls) has a known distribution
over the counterfactual shocks; the recentered estimator is the
Hodges-Lehmann estimator that sets ``T`` to its randomization mean.
The confidence set collects every ``b`` that is not rejected and is
computed exactly, so it may be unbounded or a union of intervals when
the shocks move the treatment little.

References
----------
[@borusyak2023nonrandom]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import CausalResult
from ..exceptions import (
    AssumptionViolation,
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from ._recentered_extras import (
    check_conley,
    check_efficient,
    column_or_array,
    conley_refit,
    efficient_transform,
    singleton_keep_mask,
)
from ._recentered_ri import _balance_tests, _ri_confidence_set, _ri_pvalue

__all__ = ["recentered_iv"]

_VCE_TO_ROBUST = {
    "robust": "hc1",
    "hc0": "hc0",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "nonrobust": "nonrobust",
}


# --------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------


def _counterfactual_matrix(
    data: pd.DataFrame, counterfactual: Any
) -> Tuple[np.ndarray, List[str]]:
    """Return the (n, S) matrix of counterfactual exposures and its labels."""
    if isinstance(counterfactual, str):
        counterfactual = [counterfactual]
    if isinstance(counterfactual, pd.DataFrame):
        if len(counterfactual) != len(data):
            raise MethodIncompatibility(
                f"counterfactual has {len(counterfactual)} rows but data has "
                f"{len(data)}",
                recovery_hint=(
                    "Pass one row per observation of `data`, in the same order."
                ),
            )
        return (
            counterfactual.to_numpy(dtype=float),
            [str(c) for c in counterfactual.columns],
        )
    if isinstance(counterfactual, (list, tuple)) and all(
        isinstance(c, str) for c in counterfactual
    ):
        check_columns(
            data,
            {"counterfactual": list(counterfactual)},
            function="recentered_iv",
        )
        cols = list(counterfactual)
        return data[cols].to_numpy(dtype=float), cols
    arr = np.asarray(counterfactual, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != len(data):
        raise MethodIncompatibility(
            f"counterfactual must be an (n, S) array with n = {len(data)} rows; "
            f"got shape {arr.shape}",
            recovery_hint=(
                "Stack the exposures recomputed under each counterfactual "
                "shock draw as columns, one row per observation."
            ),
        )
    return arr, [f"draw_{s}" for s in range(arr.shape[1])]


def _wls_residualize(V: np.ndarray, C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Residuals of each column of ``V`` from a weighted projection on ``C``."""
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(C * sw, V * sw, rcond=None)
    return V - C @ coef


def _absorb_terms(absorb: Any) -> List[List[str]]:
    """``absorb=`` as a list of terms, each a list of columns (``a^b``)."""
    from ..regression.iv import _normalise_absorb

    return [
        [p.strip() for p in term.split("^") if p.strip()]
        for term in _normalise_absorb(absorb)
    ]


def _fe_codes(data: pd.DataFrame, idx: np.ndarray, parts: List[str]) -> np.ndarray:
    """Integer codes of one (possibly interacted) fixed effect on rows ``idx``."""
    sub = data.iloc[idx][parts].reset_index(drop=True)
    if len(parts) == 1:
        codes, _ = pd.factorize(sub[parts[0]], sort=True)
    else:
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(sub), sort=True)
    return codes.astype(np.int64)


def _fe_demean(V: np.ndarray, fe: pd.DataFrame) -> np.ndarray:
    """Within-transform the columns of ``V`` by the fixed effects ``fe``.

    Singletons are kept: their within value is zero, so they drop out of
    every inner product the randomization statistic is built from.
    """
    from ..fast.demean import demean

    V = np.asarray(V, dtype=float)
    out, info = demean(V, fe, drop_singletons=False, tol=1e-13, max_iter=100_000)
    if not all(info.converged):
        warnings.warn(
            "recentered_iv(absorb=...): the fixed-effect sweep for the "
            f"randomization statistic did not converge (max |dx| = "
            f"{max(info.max_dx):.2e}); the RI p-value and set are approximate.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return np.asarray(out, dtype=float).reshape(V.shape)


def _wvar(v: np.ndarray, w: np.ndarray) -> float:
    m = np.sum(w * v) / np.sum(w)
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------


@records_call("recentered_iv")
def recentered_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    exposure: str,
    counterfactual: Union[Sequence[str], np.ndarray, pd.DataFrame],
    covariates: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    vce: Literal[
        "robust", "hc0", "hc1", "hc2", "hc3", "nonrobust", "conley"
    ] = "robust",
    method: Literal["recenter", "control"] = "recenter",
    ri: bool = True,
    ri_null: float = 0.0,
    balance: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    absorb: Optional[Union[str, Sequence[str]]] = None,
    expected: Optional[Union[str, np.ndarray, pd.Series]] = None,
    resid_var: Optional[Union[str, np.ndarray, pd.Series]] = None,
    resid_mean: Optional[Union[str, np.ndarray, pd.Series]] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    conley_kernel: Literal["uniform", "bartlett"] = "uniform",
    conley_distance: Literal["haversine", "planar"] = "haversine",
) -> CausalResult:
    """Recentered IV for treatments built from exogenous shocks (Borusyak-Hull).

    Adjusts for non-random exposure to as-good-as-random shocks by
    subtracting (or controlling for) the expected instrument, the
    average of the instrument across counterfactual shock draws, and
    reports randomization inference from the same draws.

    Parameters
    ----------
    data : DataFrame
        One row per observation.
    y : str
        Outcome column.
    treat : str
        Treatment column. May equal ``exposure`` when the treatment is
        itself built from the shocks (the paper's OLS case).
    exposure : str
        Realised instrument ``z_i = f_i(g, w)``, computed from the
        realised shocks.
    counterfactual : list of str, (n, S) array or DataFrame
        The same instrument recomputed under ``S >= 2`` counterfactual
        shock draws (for example permutations of the shocks), one column
        per draw, rows aligned with ``data``. The expected instrument is
        their row mean. The realised draw is not added; include it as a
        column if the draws enumerate the whole permutation class.
    covariates : list of str, optional
        Predetermined controls, entered linearly with an intercept.
    weights : str, optional
        Analytic weights column.
    cluster : str, optional
        Cluster column for the asymptotic standard error.
    vce : str, default 'robust'
        One of 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'nonrobust',
        'conley'. Asymptotic variance when ``cluster`` is not given; ``'robust'``
        is HC1. ``'conley'`` is the Conley (1999) spatial HAC variance of
        ``sp.conley`` (the paper's application reports it), with the
        ``conley_*`` arguments; the unadjusted fit gets the same variance.
    method : {'recenter', 'control'}, default 'recenter'
        ``'recenter'`` instruments with ``exposure - mu``; ``'control'``
        instruments with ``exposure`` and controls for ``mu``.
    ri : bool, default True
        Compute the randomization p-value and confidence set.
    ri_null : float, default 0.0
        Value of ``beta`` under the null for the RI p-value.
    balance : list of str, optional
        Predetermined variables for RI falsification tests of
        ``E[(z - mu) r] = 0`` (one test per variable and a joint test).
    alpha : float, default 0.05
        Level of the confidence intervals.
    absorb : str or list of str, optional
        Fixed effects to absorb (``"f"``, ``["firm", "year"]``,
        ``"firm + year"``, or ``"prov^year"`` for an interaction). The
        point estimate and standard error come from ``sp.iv(absorb=...)``
        (ivreghdfe conventions: singletons dropped, fixed effects nested
        in the cluster not charged), and the randomization statistic is
        residualised on the same fixed effects (Frisch-Waugh-Lovell), so
        the Hodges-Lehmann identity still holds. Not combinable with
        ``weights``.
    expected : str or array of length n, optional
        The expected instrument ``mu`` when it is known in closed form
        (for example the stratum mean when the shocks are permuted
        within strata, or the mean of known assignment probabilities).
        It replaces the row mean of ``counterfactual`` in the estimator,
        and the randomization statistic of every draw is centred at it,
        ``(z_s - mu)' M (y - b x)``; because ``mu`` does not depend on
        the draws, the RI test stays exact for any ``S``. The draws are
        still needed for randomization inference;
        ``diagnostics['expected_draw_gap']`` reports the largest gap
        between ``mu`` and their row mean.
    resid_var : str or array of length n, optional
        ``Var[eps_i | w]``, the conditional variance of the structural
        residual given the exposure, one value per row (a diagonal
        ``Omega``). With it the instrument becomes the efficient
        recentered instrument of the working paper (eq. 9),
        ``E[eps eps' | w]^{-1} (z - mu)``, instead of ``z - mu``. Any
        positive values that depend on the exposure only keep the
        instrument valid (its mean over the shock draws stays zero); they
        give the efficiency gain when they are the true variances.
        ``method='recenter'`` only.
    resid_mean : str or array of length n, optional
        ``psi_i = E[eps_i | w]``, the part of the residual predictable
        from the exposure (requires ``resid_var``). The instrument is then
        partially residualised on it, ``Omega^{-1}(z - mu - nu rho psi)``
        (Lemma 3 of the working paper).
    conley_lat, conley_lon : str, optional
        Latitude and longitude columns in degrees, for ``vce='conley'``.
    conley_cutoff : float, optional
        Distance cutoff in km for ``vce='conley'``.
    conley_kernel : {'uniform', 'bartlett'}, default 'uniform'
        Spatial kernel; ``'bartlett'`` decays linearly to zero at the
        cutoff, as in the paper's Table 1.
    conley_distance : {'haversine', 'planar'}, default 'haversine'
        Great-circle distance, or Stata ``acreg``'s planar approximation.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` / ``ci`` are the IV (or OLS) estimate and
        its asymptotic inference. ``diagnostics`` holds the RI p-value
        (``ri_pvalue``), the RI confidence set (``ri_ci``, a list of
        intervals, possibly unbounded), the unadjusted estimate, the
        first-stage F, and how much of the variance of the instrument
        the recentering removes. ``detail`` compares the unadjusted and
        adjusted estimates.

    Notes
    -----
    The RI reference set is the realised draw together with the ``S``
    counterfactual ones, all centred at their common mean. With
    ``method='recenter'`` this centring only rescales the instrument by
    ``S / (S + 1)``, so the Hodges-Lehmann estimator of the test
    (``ri_hodges_lehmann``) equals the reported estimate for either
    method. With ``method='recenter'`` the RI test is exact for the
    sharp null of a constant effect whenever the realised and the
    counterfactual draws are exchangeable (for example independent
    random permutations of the shocks). With ``method='control'`` the
    control ``mu`` excludes the realised draw, so exactness holds only
    as ``S`` grows. With ``expected`` given, every draw (the realised
    one included) is centred at that fixed ``mu`` instead, which keeps the
    test exact for either method. With heterogeneous effects the IV
    estimand is a convex average of effects and the asymptotic standard
    error is the one to report.

    With ``absorb`` the expected instrument enters net of the fixed
    effects: a ``mu`` that is constant within them (for example the
    stratum mean when the shocks are permuted within the absorbed
    strata) is absorbed, and both methods then equal the unadjusted
    estimate.

    To simulate the counterfactual draws from a shock process instead of
    building them by hand, use ``sp.expected_instrument`` and pass its
    ``draws`` here.

    ``vce='conley'`` applies ``sp.conley`` to the fitted IV (or OLS)
    regression, so it equals ``sp.conley(sp.iv(...))`` on the
    recentered instrument; with ``absorb`` it runs on the rows that
    survive singleton dropping. The paper cautions that common exposure
    to the shocks can correlate the scores beyond any distance cutoff;
    the randomization inference does not rely on a cutoff.

    The efficient instrument follows Section 3.5 of the working paper
    (Borusyak and Hull, NBER WP 27845, revised 2021; the article refers
    to it for efficiency). Its first two steps, predicting the treatment
    from the shocks and recentering the prediction, are what
    ``exposure`` and ``counterfactual`` already express: pass the
    predicted treatment and its counterfactual draws there.
    ``resid_mean`` and ``resid_var`` are its steps 3 and 4. Randomization
    inference uses the same transformed instrument for every draw, so it
    stays exact; the balance tests keep the plain ``z - mu``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, S = 300, 200
    >>> expo = rng.uniform(0, 2, n)          # non-random exposure
    >>> def draw():
    ...     return expo * rng.binomial(1, 0.5, n)   # shocks hit exposed units
    >>> z = draw()
    >>> cf = np.column_stack([draw() for _ in range(S)])
    >>> y = 1.0 * z + 2.0 * expo + rng.normal(size=n)   # exposure confounds
    >>> df = pd.DataFrame({"y": y, "z": z})
    >>> res = sp.recentered_iv(df, "y", "z", "z", cf)
    >>> abs(res.estimate - 1.0) < 0.5
    True

    Two steps, with the draws simulated from the shock process by
    ``sp.expected_instrument`` (here: fair coins hitting exposed units):

    >>> units = pd.DataFrame({"expo": expo})
    >>> ei = sp.expected_instrument(
    ...     units,
    ...     instrument_fn=lambda d, g: d["expo"].to_numpy() * g,
    ...     shock_sampler={"kind": "bernoulli", "p": 0.5, "size": n},
    ...     n_draws=S, seed=1)
    >>> res2 = sp.recentered_iv(df, "y", "z", "z", counterfactual=ei.draws)
    >>> abs(res2.estimate - 1.0) < 0.5
    True

    References
    ----------
    [@borusyak2023nonrandom], [@borusyak2020nonrandom], [@conley1999estimation]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "data must be a pandas DataFrame",
            recovery_hint="Pass the data as a DataFrame.",
        )
    if method not in ("recenter", "control"):
        raise MethodIncompatibility(
            f"method must be 'recenter' or 'control', got {method!r}",
            recovery_hint="Use method='recenter' or method='control'.",
        )
    if vce not in _VCE_TO_ROBUST and vce != "conley":
        raise MethodIncompatibility(
            f"vce must be one of {sorted(_VCE_TO_ROBUST) + ['conley']}, got "
            f"{vce!r}",
            recovery_hint="Use vce='robust', or pass cluster= for clustering.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1), got {alpha!r}",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
        )
    check_conley(
        vce,
        conley_lat,
        conley_lon,
        conley_cutoff,
        conley_kernel,
        conley_distance,
        cluster,
        weights,
    )
    has_var, has_mean = check_efficient(method, resid_var, resid_mean)
    covariates = list(covariates or [])
    balance = list(balance or [])
    fe_terms = _absorb_terms(absorb)
    fe_cols = list(dict.fromkeys(c for term in fe_terms for c in term))
    if fe_terms and weights is not None:
        raise MethodIncompatibility(
            "recentered_iv(absorb=..., weights=...) is not supported: the "
            "absorbed IV path (sp.iv) has no weighted fixed-effect sweep.",
            recovery_hint=(
                "Drop weights=, or add the fixed effects as dummy columns in "
                "covariates= and keep weights=."
            ),
            diagnostics={"absorb": fe_cols, "weights": weights},
        )
    named: Dict[str, Any] = {"y": y, "treat": treat, "exposure": exposure}
    if covariates:
        named["covariates"] = covariates
    if fe_cols:
        named["absorb"] = fe_cols
    if balance:
        named["balance"] = balance
    if weights is not None:
        named["weights"] = weights
    if cluster is not None:
        named["cluster"] = cluster
    if vce == "conley":
        named["conley"] = [conley_lat, conley_lon]
    check_columns(data, named, function="recentered_iv")
    omega_all = column_or_array(data, resid_var, "resid_var") if has_var else None
    psi_all = column_or_array(data, resid_mean, "resid_mean") if has_mean else None

    Zcf, cf_names = _counterfactual_matrix(data, counterfactual)
    mu_given: Optional[np.ndarray] = None
    if expected is not None:
        if isinstance(expected, str):
            check_columns(data, {"expected": expected}, function="recentered_iv")
            raw = pd.to_numeric(data[expected], errors="coerce")
        else:
            raw = expected
        mu_given = np.asarray(raw, dtype=float).reshape(-1)
        if mu_given.size != len(data):
            raise MethodIncompatibility(
                f"expected has {mu_given.size} values but data has {len(data)} " "rows",
                recovery_hint="Pass one expected instrument per row of data.",
                diagnostics={"len_expected": int(mu_given.size)},
            )
    S = Zcf.shape[1]
    if S < 2:
        raise DataInsufficient(
            f"recentered_iv needs at least 2 counterfactual draws, got {S}",
            recovery_hint=(
                "Recompute the exposure under many counterfactual shock draws "
                "(hundreds of permutations is typical)."
            ),
        )

    numeric = [y, treat, exposure] + covariates + balance
    if vce == "conley":
        numeric += [conley_lat, conley_lon]
    if weights is not None:
        numeric.append(weights)
    frame = data[list(dict.fromkeys(numeric))].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
    keep &= np.isfinite(Zcf).all(axis=1)
    if mu_given is not None:
        keep &= np.isfinite(mu_given)
    for extra in (omega_all, psi_all):
        if extra is not None:
            keep &= np.isfinite(extra)
    if cluster is not None:
        keep &= data[cluster].notna().to_numpy()
    if fe_cols:
        keep &= data[fe_cols].notna().all(axis=1).to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(
            f"recentered_iv dropped {n_dropped} rows with missing values in "
            "the outcome, treatment, exposure, controls"
            + (", fixed effects" if fe_cols else "")
            + " or counterfactual draws.",
            AssumptionWarning,
            stacklevel=2,
        )
    idx = np.flatnonzero(keep)
    n = idx.size
    if n < 3:
        raise DataInsufficient(
            f"only {n} complete rows remain",
            recovery_hint="Check the missing values in the inputs.",
        )

    yv = frame[y].to_numpy(dtype=float)[idx]
    xv = frame[treat].to_numpy(dtype=float)[idx]
    zv = frame[exposure].to_numpy(dtype=float)[idx]
    Zc = Zcf[idx]
    wv = (
        frame[weights].to_numpy(dtype=float)[idx] if weights is not None else np.ones(n)
    )
    if np.any(wv < 0) or not np.any(wv > 0):
        raise MethodIncompatibility(
            "weights must be non-negative and not all zero",
            recovery_hint="Check the weights column.",
        )
    Xc = (
        frame[covariates].to_numpy(dtype=float)[idx] if covariates else np.empty((n, 0))
    )

    mu = Zc.mean(axis=1) if mu_given is None else mu_given[idx]
    zt = zv - mu
    zscale = max(float(np.max(np.abs(zv))), float(np.max(np.abs(Zc))), 1e-300)
    if float(np.max(np.abs(zt))) <= 1e-12 * zscale:
        raise AssumptionViolation(
            "the realised exposure equals its expectation for every unit, so "
            "the shocks leave no variation to use",
            recovery_hint=(
                "Check that the counterfactual draws re-randomise the shocks "
                "rather than copy the realised ones."
            ),
        )
    mu_sd = float(np.sqrt(_wvar(mu, wv)))
    mu_constant = mu_sd <= 1e-10 * zscale
    if mu_constant:
        warnings.warn(
            "The expected exposure is the same for every unit, so the "
            "intercept already absorbs it and the adjusted estimate equals "
            "the unadjusted one.",
            AssumptionWarning,
            stacklevel=2,
        )
    fe_frame: Optional[pd.DataFrame] = None
    if fe_terms:
        fe_frame = pd.DataFrame(
            {
                f"_rc_fe{k}": _fe_codes(data, idx, term)
                for k, term in enumerate(fe_terms)
            }
        )
        within = _fe_demean(np.column_stack([zt, mu]), fe_frame)
        if float(np.max(np.abs(within[:, 0]))) <= 1e-10 * zscale:
            raise AssumptionViolation(
                "the recentered exposure z - mu does not vary within the "
                "absorbed fixed effects, so no first stage is left",
                recovery_hint=(
                    "Absorb coarser fixed effects, or check that the shocks "
                    "vary within them."
                ),
                diagnostics={"absorb": [("^".join(t)) for t in fe_terms]},
            )
        if not mu_constant and float(np.max(np.abs(within[:, 1]))) <= (1e-10 * zscale):
            mu_constant = True
            warnings.warn(
                "The expected exposure is constant within the absorbed fixed "
                "effects, so they already absorb it and the adjusted "
                "estimate equals the unadjusted one.",
                AssumptionWarning,
                stacklevel=2,
            )

    eff_info: Dict[str, float] = {}
    omega = psi = None
    zt_inst = zt
    if has_var:
        omega = omega_all[idx]  # type: ignore[index]
        psi = psi_all[idx] if psi_all is not None else None
        if np.any(omega <= 0):
            raise MethodIncompatibility(
                "resid_var must be positive: it is the conditional variance "
                "of the residual",
                recovery_hint="Check the resid_var values (no zeros or negatives).",
                diagnostics={"min_resid_var": float(np.min(omega))},
            )
        zt_inst, eff_info = efficient_transform(zt, omega, psi)

    same = treat == exposure
    # Working frame with safe names so any column name works in a formula.
    work = pd.DataFrame({"_rc_y": yv, "_rc_x": xv, "_rc_z": zv, "_rc_zt": zt_inst})
    work["_rc_mu"] = mu
    cov_names = [f"_rc_c{j}" for j in range(Xc.shape[1])]
    for j, nm in enumerate(cov_names):
        work[nm] = Xc[:, j]
    fit_kw: Dict[str, Any] = {"robust": _VCE_TO_ROBUST.get(vce, "hc1")}
    if weights is not None:
        work["_rc_w"] = wv
        fit_kw["weights"] = "_rc_w"
    if cluster is not None:
        work["_rc_g"] = data[cluster].to_numpy()[idx]
        fit_kw["cluster"] = "_rc_g"
    if fe_frame is not None:
        for nm in fe_frame.columns:
            work[nm] = fe_frame[nm].to_numpy()
        fit_kw["absorb"] = list(fe_frame.columns)

    from ..regression.iv import iv as _iv_api
    from ..regression.ols import regress as _regress

    ctrl = list(cov_names)
    use_mu = method == "control" and not mu_constant
    if use_mu:
        ctrl.append("_rc_mu")
    rhs_ctrl = "".join(f" + {c}" for c in ctrl)

    def _fit_iv(instr: str, controls: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _iv_api(
                f"_rc_y ~ (_rc_x ~ {instr}){controls}",
                data=work,
                alpha=alpha,
                **fit_kw,
            )

    def _fit_ols(controls: str) -> Any:
        if fe_frame is not None:
            # sp.regress has no absorb=; 2SLS with the regressor as its own
            # instrument is OLS (same estimate, sandwich and DOF charge).
            work["_rc_xi"] = work["_rc_x"]
            return _fit_iv("_rc_xi", controls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _regress(f"_rc_y ~ _rc_x{controls}", data=work, **fit_kw)

    if method == "recenter":
        if same:
            work["_rc_x"] = zv
        main = _fit_iv("_rc_zt", rhs_ctrl)
    elif same:
        main = _fit_ols(rhs_ctrl)
    else:
        main = _fit_iv("_rc_z", rhs_ctrl)
    naive = (
        _fit_ols("".join(f" + {c}" for c in cov_names))
        if same
        else _fit_iv("_rc_z", "".join(f" + {c}" for c in cov_names))
    )
    if vce == "conley":
        coords = pd.DataFrame(
            {
                "_rc_lat": frame[conley_lat].to_numpy(dtype=float)[idx],
                "_rc_lon": frame[conley_lon].to_numpy(dtype=float)[idx],
            }
        )
        if fe_frame is not None:
            coords = coords[singleton_keep_mask(fe_frame)]
        c_args = (coords, conley_cutoff, conley_kernel, conley_distance, alpha)
        main = conley_refit(main, *c_args)
        naive = conley_refit(naive, *c_args)

    est = float(main.params["_rc_x"])
    se = float(main.std_errors["_rc_x"])
    pval = float(main.pvalues["_rc_x"])
    ci_df = main.conf_int(alpha=alpha)
    ci = (float(ci_df.loc["_rc_x"].iloc[0]), float(ci_df.loc["_rc_x"].iloc[1]))

    model_info: Dict[str, Any] = {
        "method": method,
        "n_draws": S,
        "n_dropped": n_dropped,
        "treatment_is_exposure": bool(same),
        "vce": "cluster" if cluster is not None else vce,
        "unadjusted_estimate": float(naive.params["_rc_x"]),
        "unadjusted_se": float(naive.std_errors["_rc_x"]),
        "expected_exposure_mean": float(np.average(mu, weights=wv)),
        "expected_exposure_sd": mu_sd,
        "expected_exposure_constant": bool(mu_constant),
        "var_ratio_recentered": _wvar(zt, wv) / max(_wvar(zv, wv), 1e-300),
        "r2_exposure_on_expected": (
            0.0
            if mu_constant
            else float(np.corrcoef(np.sqrt(wv) * zv, np.sqrt(wv) * mu)[0, 1] ** 2)
        ),
        "inference_rule": main.inference_rule(),
    }
    if vce == "conley":
        model_info["conley"] = {
            "cutoff_km": float(conley_cutoff),  # type: ignore[arg-type]
            "kernel": conley_kernel,
            "distance": conley_distance,
        }
    if has_var:
        model_info["efficient_instrument"] = {
            "resid_var": True,
            "resid_mean": bool(has_mean),
            **eff_info,
        }
    fs_f = (
        main.model_info.get("first_stage_f")
        if not (method == "control" and same)
        else None
    )
    model_info["first_stage_f"] = None if fs_f is None else float(fs_f)
    if mu_given is not None:
        model_info["expected_given"] = True
        model_info["expected_draw_gap"] = float(np.max(np.abs(Zc.mean(axis=1) - mu)))
    if fe_frame is not None:
        n_single = int(main.model_info.get("n_dropped_singletons", 0) or 0)
        model_info["absorb"] = ["^".join(t) for t in fe_terms]
        model_info["fe_cardinality"] = list(main.model_info.get("fe_cardinality", []))
        model_info["n_dropped_singletons"] = n_single
    if use_mu:
        model_info["mu_coefficient"] = float(main.params["_rc_mu"])
        model_info["mu_se"] = float(main.std_errors["_rc_mu"])

    detail = pd.DataFrame(
        [
            {
                "estimator": "unadjusted",
                "estimate": model_info["unadjusted_estimate"],
                "se": model_info["unadjusted_se"],
            },
            {"estimator": method, "estimate": est, "se": se},
        ]
    )

    # Randomization inference treats the realised draw as one of S + 1
    # equally likely draws and centres all of them at their common mean,
    # so the reference set does not depend on which draw was realised.
    # A given (closed-form) mu does not depend on the draws, so centring at
    # it keeps the test exact without the common-mean construction.
    mu_all = (zv + Zc.sum(axis=1)) / (S + 1.0) if mu_given is None else mu
    zt_all = zv - mu_all
    if ri:
        if fe_frame is None:
            C = np.column_stack([np.ones(n), Xc] + ([mu] if use_mu else []))
            resid = _wls_residualize(np.column_stack([yv, xv]), C, wv)
        else:
            # FWL: sweep the fixed effects (they span the constant) out of
            # y, x and the controls, then project on the swept controls.
            Cfe = np.column_stack([Xc] + ([mu] if use_mu else []))
            swept = _fe_demean(np.column_stack([yv, xv, Cfe]), fe_frame)
            resid = swept[:, :2]
            if Cfe.shape[1]:
                resid = _wls_residualize(resid, swept[:, 2:], wv)
        y_r, x_r = resid[:, 0], resid[:, 1]
        Zt_s = Zc - mu_all[:, None]
        zt_ri = zt_all
        if omega is not None:
            # The efficient map is linear and fixed given w, so every draw
            # goes through it and the test stays exact.
            zt_ri, _ = efficient_transform(zt_all, omega, psi)
            Zt_s, _ = efficient_transform(Zt_s, omega, psi)
        a = float(np.sum(wv * zt_ri * y_r))
        c = float(np.sum(wv * zt_ri * x_r))
        a_s = (wv * y_r) @ Zt_s
        c_s = (wv * x_r) @ Zt_s
        sets = _ri_confidence_set(a, c, a_s, c_s, alpha)
        model_info["ri_null"] = float(ri_null)
        model_info["ri_pvalue"] = _ri_pvalue(a, c, a_s, c_s, ri_null)
        model_info["ri_ci"] = sets
        model_info["ri_ci_bounded"] = bool(
            sets and np.isfinite(sets[0][0]) and np.isfinite(sets[-1][1])
        )
        model_info["ri_hodges_lehmann"] = a / c if abs(c) > 0 else float("nan")
        model_info["ri_statistic"] = a - float(ri_null) * c
        if not model_info["ri_ci_bounded"]:
            warnings.warn(
                "The randomization confidence set is unbounded: the "
                "counterfactual shocks move the treatment too little to "
                "bound the effect.",
                AssumptionWarning,
                stacklevel=2,
            )

    if balance:
        Rb = frame[balance].to_numpy(dtype=float)[idx]
        model_info["balance"] = _balance_tests(
            zt_all, Zc - mu_all[:, None], Rb, balance, wv
        ).to_dict(orient="records")

    return CausalResult(
        method=(
            "Recentered IV (Borusyak-Hull)"
            if method == "recenter"
            else "Expected-instrument control (Borusyak-Hull)"
        ),
        estimand="beta",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=float(alpha),
        n_obs=int(n - model_info.get("n_dropped_singletons", 0)),
        detail=detail,
        model_info=model_info,
        _citation_key="borusyak2023nonrandom",
    )


# Kept in sync with paper.bib (key borusyak2023nonrandom).
CausalResult._CITATIONS["borusyak2023nonrandom"] = (
    "@article{borusyak2023nonrandom,\n"
    "  title={Nonrandom Exposure to Exogenous Shocks},\n"
    "  author={Borusyak, Kirill and Hull, Peter},\n"
    "  journal={Econometrica},\n"
    "  volume={91},\n"
    "  number={6},\n"
    "  pages={2155--2185},\n"
    "  year={2023},\n"
    "  doi={10.3982/ecta19367}\n"
    "}"
)
