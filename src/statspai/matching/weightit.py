"""Balancing weights for binary, multi-category and continuous treatments.

:func:`weightit` is the one entry point: it reads the treatment type from
the column (or ``treat_type=``) and returns weights from a generalized
propensity-score model (``method='glm'``), the just-identified covariate
balancing propensity score (``'cbps'``), entropy balancing (``'ebal'``) or
energy balancing (``'energy'``, categorical treatments). :func:`sp.bal_tab` reports the balance the weights achieve.

References
----------
[@greifer2026weightit], [@imai2014covariate], [@hainmueller2012entropy],
[@robins2000marginal], [@hirano2004propensity], [@huling2024energy]
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..core._covariates import expand_categorical_covariates
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from ..exceptions import warn as _sp_warn
from . import _weightit_core as _core

_METHODS = ("glm", "cbps", "ebal", "energy")
_TREAT_TYPES = ("binary", "multinomial", "continuous")
_ESTIMANDS = {
    ("glm", "binary"): ("ATE", "ATT", "ATC", "ATO", "ATM"),
    ("glm", "multinomial"): ("ATE", "ATT", "ATO", "ATM"),
    ("cbps", "binary"): ("ATE", "ATT", "ATC"),
    ("cbps", "multinomial"): ("ATE", "ATT"),
    ("ebal", "binary"): ("ATE", "ATT", "ATC"),
    ("ebal", "multinomial"): ("ATE", "ATT"),
    ("energy", "binary"): ("ATE", "ATT", "ATC"),
    ("energy", "multinomial"): ("ATE", "ATT"),
}


class WeightItResult(ResultProtocolMixin):
    """Balancing weights returned by :func:`sp.weightit`.

    Attributes
    ----------
    weights : pd.Series
        One weight per row of the input data (``NaN`` for rows dropped for
        missing values).
    ps : pd.Series or pd.DataFrame or None
        Binary: ``P(treat = treated level | X)``. Multi-category: one column
        of generalized propensity scores per level. Continuous ``'glm'``:
        the fitted conditional mean of the treatment. ``None`` for
        entropy balancing.
    treat_type, method, estimand, focal : str
        What was estimated.
    diagnostics : dict
        Weight range, coefficient of variation, effective sample sizes
        (overall and per group), number of rows dropped.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=300)
    >>> g = np.where(x + rng.normal(size=300) > 0.5, 'b', 'a')
    >>> df = pd.DataFrame({'g': g, 'x': x})
    >>> W = sp.weightit(df, treat='g', covariates=['x'], method='ebal')
    >>> bool(abs(W.balance().table.loc['x', 'diff_adj']) < 1e-8)
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("greifer2026weightit",)

    def __init__(
        self,
        *,
        weights: pd.Series,
        ps: Any,
        treat: str,
        covariates: List[str],
        treat_type: str,
        method: str,
        estimand: str,
        focal: Any,
        levels: Optional[List[Any]],
        diagnostics: Dict[str, Any],
        data: pd.DataFrame,
    ) -> None:
        self.weights = weights
        self.ps = ps
        self.treat = treat
        self.covariates = covariates
        self.treat_type = treat_type
        self.method = method
        self.estimand = estimand
        self.focal = focal
        self.levels = levels
        self.diagnostics = diagnostics
        self.model_info = {"diagnostics": diagnostics}
        self._data = data

    @property
    def n_obs(self) -> int:
        return int(self.weights.notna().sum())

    def balance(self, **kwargs: Any) -> Any:
        """Balance table of these weights (:func:`sp.bal_tab`)."""
        from .bal_tab import bal_tab

        return bal_tab(self._data, self.treat, self.covariates, weights=self, **kwargs)

    def summary(self) -> str:
        d = self.diagnostics
        lines = [
            f"Balancing weights: method={self.method!r}, "
            f"treatment={self.treat_type}, estimand={self.estimand}"
            + (f", focal={self.focal!r}" if self.focal is not None else ""),
            "=" * 70,
            f"Observations: {self.n_obs}"
            + (
                f" ({d['n_dropped']} dropped for missing values)"
                if d.get("n_dropped")
                else ""
            ),
            f"Weight range: [{d['min_weight']:.4g}, {d['max_weight']:.4g}]   "
            f"coef. of variation: {d['coef_of_variation']:.3f}",
            f"Effective sample size: {d['ess']:.1f}",
        ]
        if "ess_by_group" in d:
            for lab, e in d["ess_by_group"].items():
                lines.append(f"  {lab}: n = {d['n_by_group'][lab]}, ESS = {e:.1f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _infer_treat_type(t: pd.Series, treat: str) -> str:
    if isinstance(t.dtype, pd.CategoricalDtype) or not (
        pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)
    ):
        return "binary" if t.nunique() == 2 else "multinomial"
    n_levels = t.nunique()
    if n_levels == 2:
        return "binary"
    vals = t.to_numpy(dtype=float)
    if n_levels <= 10 and np.allclose(vals, np.round(vals)):
        raise MethodIncompatibility(
            f"weightit: treatment {treat!r} is numeric with {n_levels} integer "
            "values; it could be a multi-category treatment or a dose.",
            recovery_hint="Pass treat_type='multinomial' for categories or "
            "treat_type='continuous' for a dose.",
            diagnostics={"levels": sorted(np.unique(vals).tolist())},
        )
    return "continuous"


def _levels_of(t: pd.Series) -> List[Any]:
    if isinstance(t.dtype, pd.CategoricalDtype):
        present = set(t.unique())
        return [lv for lv in t.cat.categories if lv in present]
    try:
        return sorted(t.unique().tolist())
    except TypeError:
        return sorted(t.unique().tolist(), key=repr)


def weightit(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    method: str = "glm",
    estimand: str = "ATE",
    *,
    focal: Any = None,
    treat_type: Optional[str] = None,
    moments: Optional[int] = None,
    interactions: bool = False,
    d_moments: Optional[int] = None,
    improved: bool = True,
    lam: float = 1e-4,
    dist: str = "scaled_euclidean",
    min_w: float = 1e-8,
) -> WeightItResult:
    """Balancing weights for a binary, multi-category or continuous treatment.

    The R ``WeightIt`` workflow in one call: estimate weights, then check
    them with :meth:`WeightItResult.balance` (:func:`sp.bal_tab`).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treat : str
        Treatment column. Text, categorical or two-valued columns are
        categorical treatments; other numeric columns are continuous.
    covariates : sequence of str
        Covariates; categorical ones are expanded into indicators.
    method : {'glm', 'cbps', 'ebal', 'energy'}, default 'glm'
        ``'glm'``: inverse generalized propensity scores from a
        (multinomial) logit, or for a continuous treatment the ratio of
        normal densities ``phi(z_A) / phi(r / sigma)`` from a linear model
        (the stabilized weight ``f(A) / f(A | X)`` up to a constant).
        ``'cbps'``: just-identified covariate balancing propensity score
        (categorical treatments): the logit coefficients that make the
        weighted sums of ``(1, X)`` equal across groups. ``'ebal'``:
        entropy balancing; for a continuous treatment it zeroes the
        weighted treatment-covariate covariances while keeping the means.
        ``'energy'`` (binary and multi-category treatments): energy
        balancing, the non-negative weights that minimise the energy
        distance between each weighted group and the target distribution
        (the full sample for the ATE, the focal group for the ATT), a
        distance between whole multivariate distributions rather than
        between chosen moments.
    estimand : str, default 'ATE'
        ``'ATE'`` or ``'ATT'`` (with ``focal``) for every categorical
        method; ``'ATC'`` for a binary treatment; ``'ATO'`` (overlap) and
        ``'ATM'`` (matching) with ``'glm'``.
        Continuous treatments take ``'ATE'`` only.
    focal : optional
        The level whose covariate distribution is the target for
        ``estimand='ATT'``. A binary 0/1 treatment defaults to 1 (and to 0
        for ``'ATC'``); a multi-category one requires it.
    treat_type : {'binary', 'multinomial', 'continuous'}, optional
        Override the type read from the column. Required for a numeric
        column with 3 to 10 integer values, which could be either.
    moments : int, optional
        For ``'cbps'`` / ``'ebal'``: balance powers up to this order of
        each non-binary covariate (default 1). For ``'energy'``: add exact
        balance constraints on the means of these powers (default 0, no
        constraint beyond the energy distance).
    interactions : bool, default False
        For ``'cbps'`` / ``'ebal'`` / ``'energy'`` (with ``moments >= 1``):
        also balance pairwise products.
    d_moments : int, optional
        Continuous ``'ebal'``: also keep the central moments 2..``d_moments``
        of the treatment and of each non-binary covariate at their sample
        values (without it only the means are held, and the balanced
        covariate powers of ``moments``). Default ``None`` (= 1).
    improved : bool, default True
        ``'energy'`` with ``estimand='ATE'``: add the energy distances
        between every pair of weighted groups to those between each group
        and the full sample (the improved energy balancing weights of
        Huling and Mak). ``False`` is the standard version.
    lam : float, default 1e-4
        ``'energy'``: penalty on the squared weights, ``lam / 2`` times the
        sum of squares of each group's normalized weights; larger values
        trade balance for a larger effective sample size.
    dist : {'scaled_euclidean', 'mahalanobis', 'euclidean'}
        ``'energy'``: distance between covariate vectors (columns divided
        by their standard deviations, whitened by the sample covariance,
        or as given).
    min_w : float, default 1e-8
        ``'energy'``: smallest allowed weight.

    Returns
    -------
    WeightItResult
        ``.weights`` (aligned to ``data``), ``.ps``, ``.diagnostics`` and
        ``.balance()``.

    Notes
    -----
    Conventions follow R ``WeightIt`` 1.7.0, checked on shared fixtures
    for every supported method / treatment type / estimand
    (``tests/reference_parity/test_weightit_cobalt_parity.py``,
    ``tests/reference_parity/test_weightit_energy_parity.py``). Weights
    are normalized as WeightIt normalizes them: ATE inverse-probability
    weights are ``1 / P(A = a | X)``; ATT weights equal 1 in the focal
    group; entropy-balancing weights sum to ``n`` per group (ATE), to
    ``n_focal`` per group (ATT) or to ``n`` (continuous). Rows with a
    missing treatment or covariate are dropped (``NaN`` weight).

    The continuous CBPS of WeightIt minimizes a GMM criterion over the
    treatment model and the marginal moments and does not reach exact
    balance; it is not offered here. ``method='ebal'`` reaches exact
    covariance balance.

    Energy balancing solves its quadratic programme exactly (active-set
    iteration on the optimality conditions). Its objective is flat in some
    directions -- with ``lam`` near zero, weights several hundredths apart
    can differ in the objective only in the eleventh digit -- so where
    WeightIt's iterative solver (OSQP) stops short, the two weight vectors
    differ by more than their objective values do. Energy balancing of a
    continuous treatment (independence weights) is not implemented.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 500
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> dose = 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'dose': dose, 'x1': x1, 'x2': x2})
    >>> W = sp.weightit(df, 'dose', ['x1', 'x2'], method='ebal')
    >>> bool(W.balance().table['corr_adj'].abs().max() < 1e-8)
    True

    References
    ----------
    [@greifer2026weightit], [@imai2014covariate], [@hainmueller2012entropy]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("weightit: data must be a pandas DataFrame.")
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"weightit: method must be one of {_METHODS}; got {method!r}.",
            recovery_hint="Use 'glm', 'cbps' or 'ebal'.",
        )
    if treat not in data.columns:
        raise MethodIncompatibility(f"weightit: treatment column {treat!r} not found.")
    covariates = list(covariates)
    missing = [c for c in covariates if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"weightit: covariates {missing} not found in data.",
            diagnostics={"missing": missing},
        )
    if not covariates:
        raise MethodIncompatibility("weightit: at least one covariate is required.")
    if moments is None:
        moments = 0 if method == "energy" else 1
    low = 0 if method == "energy" else 1
    if int(moments) < low or int(moments) > 3:
        raise MethodIncompatibility(
            f"weightit: moments must be between {low} and 3 for method="
            f"{method!r}; got {moments}.",
            recovery_hint="Use moments=1 (means), 2 or 3.",
        )
    if method == "energy":
        from ._energy import DISTANCES

        if dist not in DISTANCES:
            raise MethodIncompatibility(
                f"weightit: dist must be one of {DISTANCES}; got {dist!r}.",
                recovery_hint="Use 'scaled_euclidean' (the default).",
            )
        if not (np.isfinite(lam) and lam >= 0):
            raise MethodIncompatibility(
                f"weightit: lam must be a non-negative number; got {lam!r}.",
                recovery_hint="Use the default 1e-4, or a larger value for "
                "less variable weights.",
            )
        if not (np.isfinite(min_w) and min_w < 1):
            raise MethodIncompatibility(
                f"weightit: min_w must be a number below 1; got {min_w!r}.",
                recovery_hint="Use the default 1e-8.",
            )
    estimand = str(estimand).upper()
    if estimand == "ATU":
        estimand = "ATC"

    work, cov_cols, _info = expand_categorical_covariates(
        data, covariates, function="weightit"
    )
    complete = work[[treat] + cov_cols].notna().all(axis=1)
    n_dropped = int((~complete).sum())
    df = work.loc[complete]
    if n_dropped:
        _sp_warn(
            AssumptionWarning,
            f"weightit: {n_dropped} rows with a missing treatment or covariate "
            "were dropped (their weight is NaN).",
            recovery_hint="Impute the covariates first if those rows belong "
            "in the analysis.",
            stacklevel=2,
        )
    if len(df) < len(cov_cols) + 3:
        raise DataInsufficient(
            f"weightit: {len(df)} complete rows for {len(cov_cols)} covariates.",
            recovery_hint="Use fewer covariates or more data.",
        )
    t = df[treat]
    if treat_type is None:
        treat_type = _infer_treat_type(t, treat)
    elif treat_type not in _TREAT_TYPES:
        raise MethodIncompatibility(
            f"weightit: treat_type must be one of {_TREAT_TYPES}; got {treat_type!r}."
        )
    X = df[cov_cols].to_numpy(dtype=float)
    Xt = _core._standardize(X)

    if treat_type == "continuous":
        if not pd.api.types.is_numeric_dtype(t):
            raise MethodIncompatibility(
                "weightit: a continuous treatment must be numeric.",
                recovery_hint="Use treat_type='multinomial' for categories.",
            )
        if estimand != "ATE":
            raise MethodIncompatibility(
                "weightit: a continuous treatment has no treated group; only "
                "estimand='ATE' is defined.",
                recovery_hint="Use estimand='ATE'.",
            )
        a = t.to_numpy(dtype=float)
        ps: Any = None
        if method == "glm":
            w, mu, sigma = _core.continuous_glm(a, Xt)
            ps = pd.Series(mu, index=df.index, name="treat_mean")
            extra = {"sigma": sigma}
        elif method == "ebal":
            M, _ = _core.balance_terms(X, cov_cols, moments, interactions)
            dm = 1 if d_moments is None else int(d_moments)
            if dm < 1 or dm > 3:
                raise MethodIncompatibility(
                    f"weightit: d_moments must be 1, 2 or 3; got {d_moments}."
                )
            w = _core.continuous_ebal(a, X, M, dm)
            extra = {"d_moments": dm}
        else:
            raise MethodIncompatibility(
                f"weightit: method={method!r} is implemented for binary and "
                "multi-category treatments only.",
                recovery_hint="Use method='ebal', which balances the "
                "treatment-covariate covariances exactly, or method='glm'.",
            )
        diag = _core.describe_weights(w, None, [])
        diag.update(extra)
        diag["n_dropped"] = n_dropped
        return _finish(
            data,
            complete,
            w,
            ps,
            treat,
            covariates,
            treat_type,
            method,
            estimand,
            None,
            None,
            diag,
        )

    levels = _levels_of(t)
    K = len(levels)
    if K < 2:
        raise DataInsufficient("weightit: the treatment has a single level.")
    if treat_type == "binary" and K != 2:
        raise MethodIncompatibility(
            f"weightit: treat_type='binary' but the treatment has {K} levels.",
            recovery_hint="Use treat_type='multinomial'.",
        )
    if treat_type == "multinomial" and K == 2:
        treat_type = "binary"
    allowed = _ESTIMANDS[(method, treat_type)]
    if estimand not in allowed:
        raise MethodIncompatibility(
            f"weightit: estimand={estimand!r} is not available for method="
            f"{method!r} with a {treat_type} treatment.",
            recovery_hint=f"Use one of {allowed}.",
        )
    code = {lv: k for k, lv in enumerate(levels)}
    G = np.array([code[v] for v in t.tolist()])
    focal_code: Optional[int] = None
    if estimand in ("ATT", "ATC"):
        if focal is None:
            if treat_type == "binary" and set(levels) == {0, 1}:
                focal = 1 if estimand == "ATT" else 0
            elif treat_type == "binary" and set(levels) == {False, True}:
                focal = estimand == "ATT"
            else:
                raise MethodIncompatibility(
                    f"weightit: estimand={estimand!r} needs focal=, the level "
                    "whose covariate distribution is the target.",
                    recovery_hint=f"Pass focal= one of {levels}.",
                )
        if focal not in code:
            raise MethodIncompatibility(
                f"weightit: focal={focal!r} is not a treatment level.",
                recovery_hint=f"Use one of {levels}.",
            )
        focal_code = code[focal]
        estimand_core = "ATT"  # ATC = ATT with the control level as focal
    else:
        if focal is not None:
            raise MethodIncompatibility(
                f"weightit: focal= applies to estimand='ATT' / 'ATC', not {estimand!r}."
            )
        estimand_core = estimand

    for k, lv in enumerate(levels):
        if int((G == k).sum()) < 2:
            raise DataInsufficient(
                f"weightit: treatment level {lv!r} has fewer than 2 rows.",
                recovery_hint="Merge or drop the sparse level.",
            )

    ps = None
    if method == "glm":
        beta = _core.mlogit_fit(Xt, G, K)
        P, _ = _core._probs(Xt, beta, K)
        w = _core.group_weights(P, G, estimand_core, focal_code)
    elif method == "cbps":
        M, _ = _core.balance_terms(X, cov_cols, moments, interactions)
        Mt = _core._standardize(M)
        beta0 = (
            _core.mlogit_fit(Mt, G, K)
            if estimand_core == "ATE"
            else np.zeros((K - 1) * Mt.shape[1])
        )
        beta = _core.cbps_fit(Mt, G, K, estimand_core, focal_code, beta0)
        P, _ = _core._probs(Mt, beta, K)
        w = _core.group_weights(P, G, estimand_core, focal_code)
    elif method == "energy":
        from ._energy import energy_weights

        M = (
            _core.balance_terms(X, cov_cols, moments, interactions)[0]
            if moments >= 1
            else None
        )
        w = energy_weights(
            X,
            G,
            K,
            estimand_core,
            focal_code,
            improved=improved,
            lam=float(lam),
            dist=dist,
            min_w=float(min_w),
            M=M,
        )
        P = None
    else:
        M, _ = _core.balance_terms(X, cov_cols, moments, interactions)
        w = _core.ebal_groups(M, G, K, estimand_core, focal_code)
        P = None
    if P is not None:
        if treat_type == "binary":
            ps = pd.Series(P[:, 1], index=df.index, name=f"P({treat}={levels[1]!r})")
        else:
            ps = pd.DataFrame(P, index=df.index, columns=[str(v) for v in levels])
        pmin = float(P[np.arange(len(G)), G].min())
        if pmin < 1e-3:
            _sp_warn(
                AssumptionWarning,
                f"weightit: a generalized propensity score of {pmin:.1e}; "
                "positivity is close to failing and the largest weights "
                "dominate.",
                recovery_hint="Inspect result.diagnostics['ess_by_group'] and "
                "consider estimand='ATO' or trimming the sample.",
                stacklevel=2,
            )
    diag = _core.describe_weights(w, G, levels)
    diag["n_dropped"] = n_dropped
    return _finish(
        data,
        complete,
        w,
        ps,
        treat,
        covariates,
        treat_type,
        method,
        estimand,
        focal,
        levels,
        diag,
    )


def _finish(
    data: pd.DataFrame,
    complete: pd.Series,
    w: np.ndarray,
    ps: Any,
    treat: str,
    covariates: List[str],
    treat_type: str,
    method: str,
    estimand: str,
    focal: Any,
    levels: Optional[List[Any]],
    diag: Dict[str, Any],
) -> WeightItResult:
    full = np.full(len(data), np.nan)
    full[np.asarray(complete)] = w
    weights = pd.Series(full, index=data.index, name="weights")
    return WeightItResult(
        weights=weights,
        ps=ps,
        treat=treat,
        covariates=list(covariates),
        treat_type=treat_type,
        method=method,
        estimand=estimand,
        focal=focal,
        levels=levels,
        diagnostics=diag,
        data=data,
    )
