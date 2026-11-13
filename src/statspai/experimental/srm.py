"""Sample-ratio-mismatch (SRM) check for randomised experiments.

An experiment that was designed to send, say, half of the units to each arm
and ends up with 50.9% in one of them has a problem upstream of any
outcome analysis: assignment, logging or filtering treated the arms
differently, and the comparison of outcomes is no longer a comparison of
randomised groups. The check is Pearson's chi-square goodness-of-fit test
of the observed arm counts against the designed shares.

The statistic, its degrees of freedom and p-value are those of R
``chisq.test(x, p = expected)`` (``tests/reference_parity/
test_gaps_ml_parity.py``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as _stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
)

__all__ = ["srm_test", "SRMResult"]


@dataclass
class SRMResult(ResultProtocolMixin):
    """Result of :func:`srm_test`.

    Attributes
    ----------
    statistic : float
        Pearson chi-square statistic.
    df : int
        Degrees of freedom, the number of arms minus one.
    p_value : float
        Upper-tail chi-square probability of ``statistic``.
    mismatch : bool
        ``p_value < alpha``.
    table : pandas.DataFrame
        One row per arm: ``observed`` count, ``expected`` count,
        ``expected_share``, ``observed_share``, the Pearson ``residual``
        ``(O - E) / sqrt(E)`` and the standardised residual ``stdres``
        ``(O - E) / sqrt(E (1 - share))``.
    n : int
        Number of units counted.
    n_missing : int
        Rows whose arm was missing and that were left out.
    alpha : float
        Level of the alarm.
    """

    statistic: float
    df: int
    p_value: float
    mismatch: bool
    table: pd.DataFrame
    n: int
    n_missing: int = 0
    alpha: float = 0.001
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method: str = "Sample ratio mismatch (chi-square goodness of fit)"

    def summary(self) -> str:
        """Plain-text report."""
        verdict = (
            "MISMATCH: the arm sizes are not what the design implies"
            if self.mismatch
            else "no alarm at this level"
        )
        lines = [
            self.method,
            "-" * 62,
            self.table.to_string(float_format=lambda v: f"{v:.4f}"),
            "",
            f"  chi2({self.df}) = {self.statistic:.4f}, p = {self.p_value:.4g}, "
            f"N = {self.n}",
            f"  alarm level alpha = {self.alpha:g}: {verdict}",
        ]
        if self.n_missing:
            lines.append(f"  {self.n_missing} rows with a missing arm were left out")
        if self.diagnostics.get("small_expected"):
            lines.append(
                "  expected count below 5 in at least one arm: the chi-square "
                "approximation may be poor"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SRMResult(statistic={self.statistic:.4f}, df={self.df}, "
            f"p_value={self.p_value:.4g}, mismatch={self.mismatch})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dictionary."""
        return {
            "method": self.method,
            "statistic": float(self.statistic),
            "df": int(self.df),
            "p_value": float(self.p_value),
            "mismatch": bool(self.mismatch),
            "alpha": float(self.alpha),
            "n": int(self.n),
            "n_missing": int(self.n_missing),
            "arms": [
                {"arm": _plain(arm), **{k: _plain(v) for k, v in row.items()}}
                for arm, row in self.table.iterrows()
            ],
            "diagnostics": dict(self.diagnostics),
        }


def _plain(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _observed_counts(data: Any, treat: Optional[str]) -> "tuple[pd.Series, int]":
    """Arm counts and the number of rows left out for a missing arm."""
    if isinstance(data, pd.DataFrame):
        if treat is None:
            raise MethodIncompatibility(
                "srm_test: pass treat=, the column that holds the arm of each row.",
                recovery_hint=(
                    "sp.srm_test(df, 'arm'), or pass the arm counts themselves "
                    "as a mapping: sp.srm_test({'control': 5000, 'treated': 5100})."
                ),
            )
        if treat not in data.columns:
            raise ColumnNotFound.from_missing(
                {"treat": treat}, list(data.columns), context="srm_test"
            )
        col = data[treat]
        n_missing = int(col.isna().sum())
        counts = col.dropna().value_counts(sort=False)
        try:
            counts = counts.sort_index()
        except TypeError:  # mixed labels: keep the order of first appearance
            pass
        return counts.astype(float), n_missing
    if isinstance(data, pd.Series):
        counts = data
    elif isinstance(data, Mapping):
        counts = pd.Series(dict(data))
    else:
        raise MethodIncompatibility(
            "srm_test: data must be a DataFrame (with treat=) or a mapping / "
            f"Series of arm -> count, got {type(data).__name__}.",
            recovery_hint=(
                "sp.srm_test(df, 'arm') or sp.srm_test({'a': 510, 'b': 490})."
            ),
        )
    if treat is not None:
        raise MethodIncompatibility(
            "srm_test: treat= names a column of a DataFrame; it has no meaning "
            "when the arm counts are passed directly.",
            recovery_hint="Drop treat=, or pass the row-level DataFrame.",
        )
    values = pd.to_numeric(counts, errors="coerce")
    if values.isna().any() or not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise DataInsufficient(
            "srm_test: every arm count must be a finite number.",
            recovery_hint="Pass non-negative integer counts per arm.",
        )
    arr = values.to_numpy(dtype=float)
    if np.any(arr < 0) or np.any(arr != np.round(arr)):
        raise DataInsufficient(
            "srm_test: arm counts must be non-negative whole numbers.",
            recovery_hint="Pass the number of units observed in each arm.",
        )
    return values.astype(float), 0


def _expected_shares(
    expected: Any, arms: List[Hashable], observed: pd.Series
) -> "tuple[List[Hashable], np.ndarray]":
    """Arms (possibly extended by designed arms nobody landed in) and shares."""
    if expected is None:
        k = len(arms)
        return arms, np.full(k, 1.0 / k)
    if isinstance(expected, (Mapping, pd.Series)):
        exp = dict(expected)
        unknown = [a for a in arms if a not in exp]
        if unknown:
            raise MethodIncompatibility(
                f"srm_test: expected= has no entry for the observed arm(s) "
                f"{unknown!r}. Its keys are {list(exp)!r}.",
                recovery_hint=(
                    "Give every arm a designed share, e.g. "
                    "expected={0: 0.5, 1: 0.5}. An arm outside the design is "
                    "itself a problem to resolve before testing."
                ),
            )
        # A designed arm that received nobody still counts, with zero observed.
        all_arms = arms + [a for a in exp if a not in arms]
        raw = np.array([exp[a] for a in all_arms], dtype=float)
    else:
        raw = np.asarray(expected, dtype=float).ravel()
        if raw.size != len(arms):
            raise MethodIncompatibility(
                f"srm_test: expected= has {raw.size} entries for {len(arms)} "
                f"arms ({arms!r}).",
                recovery_hint=(
                    "Pass one share per arm in sorted arm order, or a mapping "
                    "arm -> share."
                ),
            )
        all_arms = arms
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or raw.sum() <= 0:
        raise MethodIncompatibility(
            "srm_test: expected shares must be finite, non-negative and not "
            "all zero.",
            recovery_hint="Pass shares or ratios such as [0.5, 0.5] or [2, 1].",
        )
    shares = raw / raw.sum()
    obs = np.array([float(observed.get(a, 0.0)) for a in all_arms])
    impossible = [a for a, s, o in zip(all_arms, shares, obs) if s == 0 and o > 0]
    if impossible:
        raise MethodIncompatibility(
            f"srm_test: arm(s) {impossible!r} have units but an expected share "
            "of zero, which no sampling variation can produce.",
            recovery_hint=(
                "Give the arm its designed share, or remove its rows if they "
                "do not belong to the experiment."
            ),
        )
    keep = shares > 0
    return [a for a, k in zip(all_arms, keep) if k], shares[keep]


def srm_test(
    data: Union[pd.DataFrame, Mapping[Hashable, float], pd.Series],
    treat: Optional[str] = None,
    expected: Union[None, Sequence[float], Mapping[Hashable, float]] = None,
    *,
    alpha: float = 0.001,
) -> SRMResult:
    """Sample-ratio-mismatch test: are the arm sizes what the design implies?

    Pearson's chi-square goodness-of-fit test of the observed number of
    units per arm against the designed allocation, for any number of arms.
    A small p-value says that assignment, logging or filtering treated the
    arms differently, so the outcome comparison should not be trusted until
    the cause is found.

    Parameters
    ----------
    data : pandas.DataFrame, mapping or pandas.Series
        Row-level data with the assignment in column ``treat``, or the arm
        counts themselves as ``{arm: count}``.
    treat : str, optional
        Assignment column of ``data``. Required with a DataFrame.
    expected : sequence of float or mapping, optional
        Designed allocation: one share (or ratio, such as ``[2, 1]``) per
        arm in sorted arm order, or a mapping ``{arm: share}``. Values are
        rescaled to sum to one. A mapping may name an arm that received no
        units; it enters with an observed count of zero. Default: equal
        shares.
    alpha : float, default 0.001
        Level below which ``mismatch`` is set. The check is run on every
        experiment and a false alarm stops the analysis, so the level is
        stricter than for an outcome test.

    Returns
    -------
    SRMResult
        ``statistic``, ``df``, ``p_value``, ``mismatch`` and a per-arm
        ``table`` of observed and expected counts with residuals.

    Raises
    ------
    DataInsufficient
        Fewer than two arms, no units, or counts that are not whole numbers.
    MethodIncompatibility
        ``expected`` does not match the arms, is negative, or gives a share
        of zero to an arm that has units.

    Notes
    -----
    The statistic is :math:`\\sum_k (O_k - E_k)^2 / E_k` with
    :math:`E_k = N p_k`, referred to a chi-square distribution with
    :math:`K - 1` degrees of freedom. It equals R ``chisq.test(x, p = p)``.
    A warning is issued when an expected count is below 5, where the
    approximation is poor.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.srm_test({"control": 5000, "treated": 5300})
    >>> round(r.statistic, 4), r.df, r.mismatch
    (8.7379, 1, False)
    >>> r = sp.srm_test({"a": 3400, "b": 3300, "c": 3600}, expected=[1, 1, 1])
    >>> round(r.p_value, 6)
    0.001118
    """
    if not (isinstance(alpha, (int, float)) and 0.0 < float(alpha) < 1.0):
        raise MethodIncompatibility(
            f"srm_test: alpha must lie strictly between 0 and 1, got {alpha!r}.",
            recovery_hint="Use a small level such as alpha=0.001.",
        )
    observed, n_missing = _observed_counts(data, treat)
    arms: List[Hashable] = list(observed.index)
    all_arms, shares = _expected_shares(expected, arms, observed)
    obs = np.array([float(observed.get(a, 0.0)) for a in all_arms])
    n = float(obs.sum())
    if len(all_arms) < 2:
        raise DataInsufficient(
            f"srm_test: found {len(all_arms)} arm; a sample ratio needs at "
            "least two.",
            recovery_hint=(
                "Check the assignment column. If one designed arm received no "
                "units, name it in expected={arm: share, ...} so that it is "
                "counted with zero."
            ),
        )
    if n <= 0:
        raise DataInsufficient(
            "srm_test: there are no units to count.",
            recovery_hint="Pass data with at least one assigned unit.",
        )
    exp_counts = n * shares
    statistic = float(np.sum((obs - exp_counts) ** 2 / exp_counts))
    df = len(all_arms) - 1
    p_value = float(_stats.chi2.sf(statistic, df))
    table = pd.DataFrame(
        {
            "observed": obs,
            "expected": exp_counts,
            "observed_share": obs / n,
            "expected_share": shares,
            "residual": (obs - exp_counts) / np.sqrt(exp_counts),
            "stdres": (obs - exp_counts) / np.sqrt(exp_counts * (1.0 - shares)),
        },
        index=pd.Index(all_arms, name=treat if treat is not None else "arm"),
    )
    diagnostics: Dict[str, Any] = {}
    small = [a for a, e in zip(all_arms, exp_counts) if e < 5]
    if small:
        diagnostics["small_expected"] = [_plain(a) for a in small]
        warnings.warn(
            f"srm_test: expected count below 5 in arm(s) {small!r}; the "
            "chi-square approximation may be poor.",
            AssumptionWarning,
            stacklevel=2,
        )
    return SRMResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        mismatch=bool(p_value < float(alpha)),
        table=table,
        n=int(round(n)),
        n_missing=n_missing,
        alpha=float(alpha),
        diagnostics=diagnostics,
    )
