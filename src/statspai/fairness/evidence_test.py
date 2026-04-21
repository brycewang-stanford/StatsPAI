"""
Evidence-Without-Injustice counterfactual-fairness test.

A post-hoc fairness diagnostic that bridges the Kusner-style counterfactual
criterion with legal/philosophical notions of admissible evidence: the
algorithm is allowed to use features correlated with the protected
attribute *provided* those features carry legitimate probative weight
("admissible evidence").  The algorithm is "unjust" if, holding those
admissible features fixed at their factual values, its predictions still
change when the protected attribute is intervened on.

Implements the procedure in

    Kwak & Pleasants (arXiv:2510.12822, 2025).
    "Evidence Without Injustice: A New Counterfactual Test for Fair
    Algorithms."

Formally, given
  - A predictor  f: X -> [0, 1]
  - A protected attribute A with factual value a_i
  - A user-supplied SCM intervention  do(A = a')
  - A set E of *admissible-evidence* features that are held at their
    observed values across interventions,

the EWI test statistic is

    T = E_i [ | f( X_i^{do(A=a'), E=e_i} ) - f(X_i) | ]

and the algorithm passes when T is within a tolerance and the bootstrap
(1 - α) CI excludes the user's threshold.

Compared to Kusner-Loftus-Russell-Silva (2018) counterfactual fairness,
this test is **strictly weaker**: freezing the admissible features means
the algorithm can legitimately depend on, say, credit score even if credit
score is correlated with race — as long as the model does not ADD race-
specific variation on top of the admissible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .core import FairnessResult


__all__ = [
    "evidence_without_injustice",
    "EvidenceWithoutInjusticeResult",
]


@dataclass
class EvidenceWithoutInjusticeResult(FairnessResult):
    """Extended fairness result with bootstrap CI and per-alt statistics.

    Inherits the standard ``metric/value/per_group/threshold/passes/notes``
    fields from :class:`FairnessResult` and adds bootstrap-inference
    artefacts specific to Kwak-Pleasants 2025.
    """

    ci: Optional[tuple] = None
    pvalue: Optional[float] = None
    alpha: float = 0.05
    n_boot: int = 0
    admissible_features: List[str] = field(default_factory=list)

    def summary(self) -> str:  # type: ignore[override]
        parent = super().summary()
        extra = []
        if self.ci is not None:
            extra.append(
                f"  {(1 - self.alpha) * 100:.0f}% CI  : "
                f"({self.ci[0]:.6f}, {self.ci[1]:.6f})"
            )
        if self.pvalue is not None:
            extra.append(f"  p-value   : {self.pvalue:.4g}")
        if self.admissible_features:
            extra.append(
                f"  admissible: {self.admissible_features}"
            )
        return "\n".join([parent, *extra])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evidence_without_injustice(
    data: pd.DataFrame,
    predictor: Callable[[pd.DataFrame], np.ndarray],
    *,
    protected: str,
    admissible_features: Sequence[str],
    scm_intervention: Callable[[pd.DataFrame, Any], pd.DataFrame],
    alternative_values: Optional[Sequence[Any]] = None,
    threshold: float = 0.05,
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: Optional[int] = None,
) -> EvidenceWithoutInjusticeResult:
    """Kwak-Pleasants (2025) evidence-without-injustice fairness test.

    Parameters
    ----------
    data : DataFrame
        Observed covariates — must include ``protected`` and all
        ``admissible_features``.
    predictor : Callable(DataFrame) -> ndarray
        Deployed model.  Called with (a) the original data and (b) the
        counterfactual data after intervention + admissibility freezing.
    protected : str
        Name of the protected-attribute column.
    admissible_features : sequence of str
        Feature columns whose *factual* values are preserved in the
        counterfactual world — these encode the "admissible evidence" that
        the algorithm may legitimately use even if correlated with ``A``.
        Pass ``[]`` to recover classical counterfactual fairness.
    scm_intervention : Callable(DataFrame, value) -> DataFrame
        User-supplied SCM.  Returns a DataFrame of the same shape with
        ``protected`` (and all non-admissible descendants) updated under
        ``do(A=value)``.  The admissibility freeze is enforced *after*
        calling this function — admissible columns are overwritten with
        their factual values.
    alternative_values : sequence, optional
        Values of ``A`` to intervene on.  Defaults to all levels other
        than the unit's factual value.
    threshold : float, default 0.05
        Practical-significance threshold on ``T``.  ``passes = True``
        iff ``ci[1] < threshold``.
    alpha : float, default 0.05
        Significance level for the bootstrap CI.
    n_boot : int, default 500
    random_state : int, optional

    Returns
    -------
    EvidenceWithoutInjusticeResult
        Standard :class:`FairnessResult` fields plus ``ci``, ``pvalue``,
        ``n_boot``, and ``admissible_features``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> A = rng.integers(0, 2, n)
    >>> credit = 600 + 100 * A + rng.normal(0, 30, n)  # correlated w/ A
    >>> race_noise = 0.0 * A  # zero direct effect -> fair
    >>> df = pd.DataFrame({'A': A, 'credit': credit, 'noise': rng.normal(size=n)})
    >>> def predictor(d): return 1 / (1 + np.exp(-(d['credit'] / 100 - 6)))
    >>> def intervene(d, a_alt):
    ...     out = d.copy(); out['A'] = a_alt
    ...     out['credit'] = 600 + 100 * a_alt + (d['credit'] - (600 + 100 * d['A']))
    ...     return out
    >>> res = sp.fairness.evidence_without_injustice(
    ...     df, predictor, protected='A',
    ...     admissible_features=['credit'],
    ...     scm_intervention=intervene,
    ...     n_boot=200, random_state=0,
    ... )
    >>> bool(res.passes)
    True

    Notes
    -----
    The bootstrap is a paired nonparametric bootstrap over the units: for
    each replicate we resample rows with replacement and recompute the
    test statistic.  This accounts for predictor-randomness only via
    plug-in (the predictor is treated as fixed).  For full accounting of
    predictor uncertainty, pass in a Bayesian posterior predictor or wrap
    the predictor in its own bootstrap at a higher level.
    """
    # --- Validation --------------------------------------------------------
    if protected not in data.columns:
        raise ValueError(f"`protected` column {protected!r} not in data.")
    missing = [c for c in admissible_features if c not in data.columns]
    if missing:
        raise ValueError(f"admissible_features not in data: {missing}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0; got {threshold}")
    if n_boot < 99:
        raise ValueError(f"n_boot must be >= 99 for a stable CI; got {n_boot}")

    # --- Predictor on factual ---------------------------------------------
    y_obs = np.asarray(predictor(data), dtype=float)
    if y_obs.shape[0] != len(data):
        raise ValueError(
            "predictor returned wrong length: "
            f"{y_obs.shape[0]} vs {len(data)}."
        )
    observed_a = data[protected].to_numpy()
    if alternative_values is None:
        alternative_values = list(pd.unique(observed_a))
        if len(alternative_values) < 2:
            raise ValueError(
                f"Protected attribute {protected!r} has only one level; "
                "EWI test is undefined."
            )

    # --- Compute counterfactual predictions under each alternative --------
    def _ewi_statistic(sample_df: pd.DataFrame, y_base: np.ndarray) -> float:
        """Compute T = E_i max_{a'} |f(X_i^{do(A=a'), E=e_i}) - f(X_i)|."""
        a_obs = sample_df[protected].to_numpy()
        max_abs = np.zeros(len(sample_df), dtype=float)
        for a_alt in alternative_values:
            df_cf = scm_intervention(sample_df, a_alt)
            if not isinstance(df_cf, pd.DataFrame):
                raise TypeError(
                    "scm_intervention must return a DataFrame, got "
                    f"{type(df_cf).__name__}."
                )
            if len(df_cf) != len(sample_df):
                raise ValueError(
                    "scm_intervention length mismatch "
                    f"({len(df_cf)} vs {len(sample_df)})."
                )
            # Freeze admissible evidence at factual values.
            for col in admissible_features:
                df_cf = df_cf.copy() if not df_cf is sample_df else df_cf
                df_cf[col] = sample_df[col].to_numpy()
            y_cf = np.asarray(predictor(df_cf), dtype=float)
            differs = a_obs != a_alt
            diff = np.abs(y_cf - y_base)
            max_abs = np.where(differs, np.maximum(max_abs, diff), max_abs)
        return float(max_abs.mean())

    stat_obs = _ewi_statistic(data, y_obs)

    # --- Paired nonparametric bootstrap -----------------------------------
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot)
    n_ok = 0
    n = len(data)
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        d_b = data.iloc[ix].reset_index(drop=True)
        y_b = y_obs[ix]
        try:
            boots[n_ok] = _ewi_statistic(d_b, y_b)
            n_ok += 1
        except Exception:
            continue
    if n_ok < max(99, int(0.7 * n_boot)):
        raise RuntimeError(
            f"EWI bootstrap only produced {n_ok}/{n_boot} valid replicates."
        )
    boots = boots[:n_ok]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))

    # One-sided p-value: H0 : T = 0, H1 : T > 0.
    se_boot = float(boots.std(ddof=1))
    if se_boot > 0:
        z = stat_obs / se_boot
        pvalue = float(1 - stats.norm.cdf(z))
    else:
        pvalue = float("nan")

    # Per-alternative breakdown
    per_alt: Dict[Any, float] = {}
    for a_alt in alternative_values:
        df_cf = scm_intervention(data, a_alt)
        for col in admissible_features:
            df_cf[col] = data[col].to_numpy()
        y_cf = np.asarray(predictor(df_cf), dtype=float)
        differs = observed_a != a_alt
        if differs.any():
            per_alt[a_alt] = float(np.abs(y_cf - y_obs)[differs].mean())
        else:
            per_alt[a_alt] = 0.0

    return EvidenceWithoutInjusticeResult(
        metric="evidence_without_injustice",
        value=float(stat_obs),
        per_group=per_alt,
        threshold=float(threshold),
        passes=bool(hi < threshold),
        notes=(
            f"EWI(T) with |E|={len(admissible_features)} admissible features. "
            f"Passes iff CI-upper < threshold."
        ),
        ci=(lo, hi),
        pvalue=pvalue,
        alpha=float(alpha),
        n_boot=int(n_ok),
        admissible_features=list(admissible_features),
    )
