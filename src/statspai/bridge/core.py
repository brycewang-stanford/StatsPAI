"""
Bridge dispatcher and result class.

A bridging theorem identifies the same causal estimand via two paths
that rest on different identifying assumptions. The empirical signature
of agreement is that path A and path B should give numerically close
point estimates. Disagreement is a red flag: at least one of the two
identifying assumptions fails.

This module ships a unified ``sp.bridge(kind=..., ...)`` dispatcher
that runs both paths, performs an agreement test on the difference,
and reports a doubly-robust point estimate (a precision-weighted
average of the two paths when they agree; a warning otherwise).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats


# Registered bridge implementations. New bridges register themselves at
# import time by inserting (kind, callable) into _BRIDGES.
_BRIDGES: Dict[str, Callable[..., 'BridgeResult']] = {}


def _register(kind: str):
    """Decorator: register a bridge implementation under ``kind``."""
    def deco(fn: Callable[..., 'BridgeResult']):
        _BRIDGES[kind] = fn
        return fn
    return deco


@dataclass
class BridgeResult:
    """
    Result of a bridging-theorem comparison.

    Attributes
    ----------
    kind : str
        Which bridge was applied (``did_sc`` / ``ewm_cate`` / ...).
    path_a_name, path_b_name : str
        Human-readable names of the two estimation paths.
    estimate_a, estimate_b : float
        Point estimates from each path.
    se_a, se_b : float
        Standard errors from each path.
    diff, diff_se, diff_p : float
        Difference, its SE (assuming independence; conservative when
        the two paths share data), and the two-sided p-value for
        H0: paths agree.
    estimate_dr : float
        Doubly-robust combined estimate (precision-weighted average
        when agreement_p > 0.05; falls back to ``estimate_a`` with a
        warning otherwise).
    se_dr : float
        SE of ``estimate_dr``.
    n_obs : int
    detail : dict
        Extra path-specific metadata.
    reference : str
    """

    kind: str
    path_a_name: str
    path_b_name: str
    estimate_a: float
    estimate_b: float
    se_a: float
    se_b: float
    diff: float
    diff_se: float
    diff_p: float
    estimate_dr: float
    se_dr: float
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)
    reference: str = ""

    @property
    def agreement(self) -> bool:
        """True iff paths fail to reject equality at 5%."""
        return self.diff_p > 0.05

    def summary(self) -> str:
        flag = "AGREE" if self.agreement else "DISAGREE"
        rows = [
            "=" * 72,
            f"Bridging theorem: {self.kind}",
            f"Reference: {self.reference}" if self.reference else "",
            "-" * 72,
            f"  Path A ({self.path_a_name}):"
            f" {self.estimate_a:+.4f}  (SE {self.se_a:.4f})",
            f"  Path B ({self.path_b_name}):"
            f" {self.estimate_b:+.4f}  (SE {self.se_b:.4f})",
            "-" * 72,
            f"  Δ = A − B = {self.diff:+.4f}"
            f"  (SE {self.diff_se:.4f}, p = {self.diff_p:.3f})",
            f"  Verdict: {flag} on the bridging implication",
            f"  DR estimate: {self.estimate_dr:+.4f}  (SE {self.se_dr:.4f})",
            f"  N = {self.n_obs}",
            "=" * 72,
        ]
        return "\n".join(r for r in rows if r)

    def __repr__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


def _agreement_test(
    est_a: float, se_a: float,
    est_b: float, se_b: float,
) -> Tuple[float, float, float]:
    """Wald test for H0: est_a == est_b, assuming independence.

    Returns (diff, diff_se, diff_p). Conservative when the two paths
    share data (covariance ignored, inflating the SE).
    """
    diff = float(est_a - est_b)
    diff_se = float(np.sqrt(max(se_a, 0.0) ** 2 + max(se_b, 0.0) ** 2))
    if diff_se <= 0:
        return diff, 0.0, 1.0
    z = diff / diff_se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return diff, diff_se, p


def _dr_combine(
    est_a: float, se_a: float,
    est_b: float, se_b: float,
    diff_p: float,
) -> Tuple[float, float]:
    """Precision-weighted combined estimate.

    If paths agree (diff_p > 0.05), return inverse-variance weighted
    average. Otherwise warn and return path A.
    """
    if diff_p <= 0.05:
        warnings.warn(
            f"Bridge paths disagree (p = {diff_p:.3f} ≤ 0.05). "
            f"Returning path A only; investigate which assumption "
            f"is failing.",
            RuntimeWarning, stacklevel=3,
        )
        return float(est_a), float(se_a)
    var_a = max(se_a, 1e-12) ** 2
    var_b = max(se_b, 1e-12) ** 2
    w_a = 1.0 / var_a
    w_b = 1.0 / var_b
    est = (w_a * est_a + w_b * est_b) / (w_a + w_b)
    se = float(np.sqrt(1.0 / (w_a + w_b)))
    return float(est), se


def bridge(kind: str, **kwargs) -> BridgeResult:
    """
    Run a bridging-theorem comparison.

    Parameters
    ----------
    kind : {'did_sc', 'ewm_cate', 'cb_ipw', 'kink_rdd',
            'dr_calib', 'surrogate_pci'}
        Which bridge to run.
    **kwargs
        Bridge-specific keyword arguments. See the per-bridge
        implementations in :mod:`statspai.bridge` for details.

    Returns
    -------
    BridgeResult
        Two path estimates + agreement test + DR combined estimate.

    Examples
    --------
    DID ≡ Synthetic Control (Shi-Athey 2025) on a panel where one
    unit gets treated at time T:

    >>> result = sp.bridge(
    ...     kind="did_sc", data=df,
    ...     y='gdp', unit='state', time='year',
    ...     treated_unit='CA', treatment_time=1989,
    ... )
    >>> print(result.summary())
    """
    # Lazy import: each bridge module registers itself on import.
    from . import (
        did_sc as _,         # noqa: F401
        ewm_cate as _,       # noqa: F401
        cb_ipw as _,         # noqa: F401
        kink_rdd as _,       # noqa: F401
        dr_calib as _,       # noqa: F401
        surrogate_pci as _,  # noqa: F401
    )
    if kind not in _BRIDGES:
        raise ValueError(
            f"Unknown bridge kind={kind!r}. Available: "
            f"{sorted(_BRIDGES.keys())}"
        )
    return _BRIDGES[kind](**kwargs)
