"""
Refutation tests: rerun an estimator on data altered so that the answer
is known, and see whether the estimator gives it.

An effect estimate cannot be validated against the truth, which is not
observed. It can be checked against things that are known by
construction. A treatment column shuffled across units has no effect on
anything, so the estimator should return zero. So should the real
treatment on an outcome that was generated without it. A covariate drawn
from a random number generator confounds nothing, and a random four
fifths of the sample is the same population, so neither should move the
estimate. An estimator that fails one of these is not estimating what it
is taken to estimate on these data, whatever the reason.

Passing is a necessary condition only. None of the checks can detect an
unobserved confounder: for that see :func:`statspai.sensemakr` and
:func:`statspai.evalue`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

__all__ = ["refute", "RefutationResult"]

_METHODS = (
    "placebo_treatment",
    "dummy_outcome",
    "random_common_cause",
    "data_subset",
)

#: What the estimate should be on the altered data.
_EXPECTED = {
    "placebo_treatment": "zero",
    "dummy_outcome": "zero",
    "random_common_cause": "estimate",
    "data_subset": "estimate",
}

_DESCRIPTION = {
    "placebo_treatment": "treatment shuffled across units",
    "dummy_outcome": "outcome replaced by one the treatment does not enter",
    "random_common_cause": "an independent random covariate added",
    "data_subset": "random subsets of the sample",
}


@dataclass
class RefutationResult(ResultProtocolMixin):
    """Outcome of one refutation test.

    Attributes
    ----------
    method : str
        Which test was run.
    estimate : float
        The estimate on the data as they are.
    new_effect : float
        Mean of the estimates on the altered data.
    expected : float
        What the estimator should return on the altered data: zero for
        ``'placebo_treatment'`` and ``'dummy_outcome'``, the original
        estimate for the other two.
    p_value : float
        Two-sided Monte Carlo p-value that the altered-data estimates are
        centred on ``expected``: twice the smaller of the shares at or
        below and at or above it, with one added to numerator and
        denominator. Small means the estimates sit to one side of where
        they should be.
    refuted : bool
        ``p_value < alpha``.
    interval : tuple of float
        Central ``1 - alpha`` range of the altered-data estimates.
    permutation_pvalue : float or None
        ``'placebo_treatment'`` only: the share of placebo estimates at
        least as large in absolute value as the original one. A
        randomisation p-value for the effect itself, not a refutation.
    simulations : numpy.ndarray
        The altered-data estimates.
    n_simulations, n_failed : int
        Reruns that returned an estimate, and reruns that raised.
    details : dict
        ``first_error`` when a rerun failed; ``subset_fraction`` for
        ``'data_subset'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=400)
    >>> d = (x + rng.normal(size=400) > 0).astype(int)
    >>> df = pd.DataFrame({'y': 2 * d + x + rng.normal(size=400), 'd': d, 'x': x})
    >>> res = sp.refute(sp.aipw, df, y='y', treat='d', covariates=['x'],
    ...                 method='placebo_treatment', n_simulations=40, seed=0)
    >>> isinstance(res, sp.RefutationResult), res.refuted
    (True, False)
    """

    method: str
    estimate: float
    new_effect: float
    expected: float
    p_value: float
    refuted: bool
    alpha: float
    interval: Tuple[float, float]
    permutation_pvalue: Optional[float]
    simulations: np.ndarray = field(repr=False)
    n_simulations: int = 0
    n_failed: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """A formatted account of the test.

        Examples
        --------
        >>> import numpy as np, pandas as pd, statspai as sp
        >>> rng = np.random.default_rng(1)
        >>> d = rng.integers(0, 2, 300)
        >>> df = pd.DataFrame({'y': d + rng.normal(size=300), 'd': d,
        ...                    'x': rng.normal(size=300)})
        >>> res = sp.refute(sp.aipw, df, y='y', treat='d', covariates=['x'],
        ...                 method='data_subset', n_simulations=40, seed=0)
        >>> print(res.summary().splitlines()[1])
          Refutation: data_subset (random subsets of the sample)
        """
        width = 70
        target = "zero" if _EXPECTED[self.method] == "zero" else "the original estimate"
        verdict = (
            "REFUTED: the estimates are not centred where they should be"
            if self.refuted
            else "not refuted"
        )
        lines = [
            "=" * width,
            f"  Refutation: {self.method} ({_DESCRIPTION[self.method]})",
            "=" * width,
            f"  Original estimate        : {self.estimate: .6g}",
            f"  Mean on altered data     : {self.new_effect: .6g}"
            f"   (should be {target})",
            f"  Central {1 - self.alpha:.0%} of reruns   : "
            f"[{self.interval[0]:.6g}, {self.interval[1]:.6g}]",
            f"  p-value                  : {self.p_value:.4g}",
            f"  Verdict (alpha = {self.alpha:g})   : {verdict}",
        ]
        if self.permutation_pvalue is not None:
            lines.append(
                "  Original vs placebo draws: p = "
                f"{self.permutation_pvalue:.4g} (randomisation test of the effect)"
            )
        lines.append(
            f"  Reruns                   : {self.n_simulations}"
            + (f" ({self.n_failed} failed)" if self.n_failed else "")
        )
        lines.append("=" * width)
        lines.append(
            "  Passing does not rule out an unobserved confounder; see " "sp.sensemakr."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RefutationResult({self.method}: estimate={self.estimate:.6g}, "
            f"new_effect={self.new_effect:.6g}, p_value={self.p_value:.4g}, "
            f"refuted={self.refuted})"
        )


def _effect(res: Any, treat: str) -> float:
    """The scalar effect in whatever an estimator returned."""
    if isinstance(res, (int, float, np.integer, np.floating)):
        return float(res)
    est = getattr(res, "estimate", None)
    if est is not None and np.ndim(est) == 0:
        return float(est)
    params = getattr(res, "params", None)
    if params is not None:
        try:
            return float(params[treat])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    raise MethodIncompatibility(
        "refute: cannot read a scalar effect from the estimator's return "
        f"value ({type(res).__name__}).",
        recovery_hint=(
            "Wrap the call so it returns a number: "
            "lambda df, **kw: sp.regress('y ~ d + x', data=df).params['d']."
        ),
    )


def refute(
    estimator: Callable[..., Any],
    data: pd.DataFrame,
    *,
    y: str,
    treat: str,
    covariates: Optional[Sequence[str]] = None,
    method: str = "placebo_treatment",
    n_simulations: int = 100,
    subset_fraction: float = 0.8,
    outcome_function: Optional[Callable[[pd.DataFrame], Any]] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    **estimator_kwargs: Any,
) -> RefutationResult:
    """
    Rerun an estimator on altered data whose answer is known.

    Parameters
    ----------
    estimator : callable
        Called as ``estimator(data, y=y, treat=treat,
        covariates=covariates, **estimator_kwargs)``; ``covariates`` is
        left out when it is ``None``. It returns a number, a result with a
        scalar ``.estimate``, or a regression result whose ``.params`` has
        an entry named ``treat``. Any ``sp`` estimator with the house
        signature fits (``sp.aipw``, ``sp.ipw``, ``sp.dml``,
        ``sp.g_computation``, ``sp.match``, ...); wrap the others in a
        ``lambda``.
    data : pd.DataFrame
    y, treat : str
        Outcome and treatment columns.
    covariates : sequence of str, optional
        Adjustment covariates passed on to the estimator.
    method : str, default 'placebo_treatment'
        - ``'placebo_treatment'``: the treatment column is shuffled
          across units, which leaves it with no effect and no relation to
          the covariates. The estimates should be centred on zero.
        - ``'dummy_outcome'``: the outcome is replaced by one the
          treatment does not enter, so the effect is zero by construction.
          By default that is the observed outcome shuffled across units.
          With ``outcome_function`` it is that function of the data plus
          noise, which leaves the treatment *associated* with the new
          outcome through the covariates: the adjustment has to remove an
          association that is known to be spurious.
        - ``'random_common_cause'``: a standard normal column is added to
          the covariates. It confounds nothing, so the estimates should be
          centred on the original one. Needs ``covariates``.
        - ``'data_subset'``: the estimator is rerun on random subsets of
          ``subset_fraction`` of the rows, drawn without replacement. The
          estimates should be centred on the original one.
    n_simulations : int, default 100
        Number of reruns. The smallest p-value attainable is
        ``2 / (n_simulations + 1)``, so at least ``2 / alpha - 1`` are
        required (39 at the default level).
    subset_fraction : float, default 0.8
        Share of rows kept by ``'data_subset'``.
    outcome_function : callable, optional
        ``'dummy_outcome'`` only. Maps the data frame to one value per
        row; it must not use the treatment column. Normal noise with the
        standard deviation of those values (one, if they are constant) is
        added in every rerun.
    alpha : float, default 0.05
        Level at which the test is called refuted.
    seed : int, optional
        Seed for the alterations. An estimator with randomness of its own
        takes its seed through ``estimator_kwargs``.
    **estimator_kwargs
        Passed to the estimator on every call.

    Returns
    -------
    RefutationResult

    Notes
    -----
    The p-value asks whether the reruns are *centred* where they should
    be, by counting how many fall on each side. It does not ask whether
    they are close: read ``new_effect`` and ``interval`` against the size
    of the original estimate for that.

    Shuffling the treatment breaks its relation to the covariates as well
    as to the outcome, so ``'placebo_treatment'`` exercises the estimator
    on unconfounded data. ``'dummy_outcome'`` with an ``outcome_function``
    of the confounders is the check that keeps the confounding in place.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = rng.normal(size=n)
    >>> d = (x + rng.normal(size=n) > 0).astype(int)
    >>> y = 2.0 * d + 3.0 * x + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x': x})

    Adjusting for the confounder passes a confounded dummy outcome:

    >>> ok = sp.refute(sp.aipw, df, y='y', treat='d', covariates=['x'],
    ...                method='dummy_outcome', n_simulations=40, seed=0,
    ...                outcome_function=lambda f: 3.0 * f['x'])
    >>> ok.refuted
    False

    A comparison of means, which ignores the confounder, does not:

    >>> naive = lambda f, y, treat, **kw: (
    ...     f.loc[f[treat] == 1, y].mean() - f.loc[f[treat] == 0, y].mean())
    >>> bad = sp.refute(naive, df, y='y', treat='d', method='dummy_outcome',
    ...                 n_simulations=40, seed=0,
    ...                 outcome_function=lambda f: 3.0 * f['x'])
    >>> bad.refuted
    True
    """
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"refute: method={method!r} is not one of {_METHODS}.",
            recovery_hint=(
                "For sensitivity to an unobserved confounder use sp.sensemakr "
                "or sp.evalue."
            ),
        )
    if not callable(estimator):
        raise MethodIncompatibility(
            "refute: estimator must be callable, e.g. sp.aipw.",
            recovery_hint="Pass the function itself, not its result.",
        )
    cols = [y, treat, *(covariates or [])]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"refute: columns not in data: {missing}",
            diagnostics={"missing_columns": missing},
        )
    needed = int(np.ceil(2.0 / alpha - 1.0 - 1e-9))
    if n_simulations < needed:
        raise MethodIncompatibility(
            f"refute: with n_simulations={n_simulations} the smallest "
            f"p-value is {2.0 / (n_simulations + 1):.3g}, so nothing could "
            f"be refuted at alpha={alpha:g}.",
            recovery_hint=f"Use n_simulations >= {needed}.",
            diagnostics={"n_simulations": n_simulations, "minimum": needed},
        )
    if method == "random_common_cause" and covariates is None:
        raise MethodIncompatibility(
            "refute: method='random_common_cause' adds a column to "
            "covariates=, which was not given.",
            recovery_hint="Pass covariates=[...] (an empty list is accepted).",
        )
    if method == "data_subset" and not 0 < subset_fraction < 1:
        raise MethodIncompatibility(
            f"refute: subset_fraction must lie strictly between 0 and 1, "
            f"got {subset_fraction}."
        )
    if outcome_function is not None and method != "dummy_outcome":
        raise MethodIncompatibility(
            "refute: outcome_function= applies to method='dummy_outcome' only."
        )

    rng = np.random.default_rng(seed)
    n = len(data)
    base_covariates = None if covariates is None else list(covariates)

    def run(frame: pd.DataFrame, covs: Optional[List[str]]) -> float:
        kwargs = dict(estimator_kwargs)
        if covs is not None:
            kwargs["covariates"] = covs
        return _effect(estimator(frame, y=y, treat=treat, **kwargs), treat)

    original = run(data, base_covariates)

    noise_col = "_random_cause"
    while noise_col in data.columns:
        noise_col += "_"

    signal: Optional[np.ndarray] = None
    noise_sd = 1.0
    if outcome_function is not None:
        signal = np.asarray(outcome_function(data), dtype=float).reshape(-1)
        if signal.shape[0] != n:
            raise MethodIncompatibility(
                f"refute: outcome_function returned {signal.shape[0]} values "
                f"for {n} rows."
            )
        spread = float(np.std(signal))
        noise_sd = spread if spread > 0 else 1.0

    sims: List[float] = []
    n_failed = 0
    first_error: Optional[str] = None
    for _ in range(n_simulations):
        covs = base_covariates
        if method == "placebo_treatment":
            frame = data.copy()
            frame[treat] = rng.permutation(data[treat].to_numpy())
        elif method == "dummy_outcome":
            frame = data.copy()
            if signal is None:
                frame[y] = rng.permutation(data[y].to_numpy())
            else:
                frame[y] = signal + rng.normal(0.0, noise_sd, size=n)
        elif method == "random_common_cause":
            frame = data.copy()
            frame[noise_col] = rng.normal(size=n)
            covs = [*(base_covariates or []), noise_col]
        else:  # data_subset
            keep = rng.choice(
                n, size=max(int(round(subset_fraction * n)), 2), replace=False
            )
            frame = data.iloc[np.sort(keep)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sims.append(run(frame, covs))
        except Exception as exc:  # a rerun that cannot be fitted is counted
            n_failed += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    draws = np.asarray(sims, dtype=float)
    draws = draws[np.isfinite(draws)]
    n_failed += len(sims) - len(draws)
    if len(draws) < max(10, n_simulations // 2):
        raise DataInsufficient(
            f"refute: only {len(draws)} of {n_simulations} reruns returned an "
            f"estimate. First error: {first_error}.",
            recovery_hint=(
                "Check that the estimator runs on a shuffled or subsetted "
                "copy of the data."
            ),
            diagnostics={"n_failed": n_failed, "first_error": first_error},
        )
    if n_failed:
        warnings.warn(
            f"refute: {n_failed} of {n_simulations} reruns failed and are left "
            f"out. First error: {first_error}.",
            RuntimeWarning,
            stacklevel=2,
        )

    expected = 0.0 if _EXPECTED[method] == "zero" else float(original)
    b = len(draws)
    below = int(np.sum(draws <= expected))
    above = int(np.sum(draws >= expected))
    p_value = min(1.0, 2.0 * (min(below, above) + 1) / (b + 1))
    interval = (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )
    perm_p: Optional[float] = None
    if method == "placebo_treatment":
        perm_p = float((np.sum(np.abs(draws) >= abs(original)) + 1) / (b + 1))

    details: Dict[str, Any] = {}
    if first_error is not None:
        details["first_error"] = first_error
    if method == "data_subset":
        details["subset_fraction"] = subset_fraction
    if method == "dummy_outcome":
        details["outcome"] = "permuted" if signal is None else "outcome_function"

    return RefutationResult(
        method=method,
        estimate=float(original),
        new_effect=float(np.mean(draws)),
        expected=expected,
        p_value=float(p_value),
        refuted=bool(p_value < alpha),
        alpha=alpha,
        interval=interval,
        permutation_pvalue=perm_p,
        simulations=draws,
        n_simulations=b,
        n_failed=n_failed,
        details=details,
    )
