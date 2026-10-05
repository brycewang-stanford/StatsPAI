"""Joint event-study covariance and uniform (sup-t) bands for any DiD fit.

Two practitioner requirements of the modern DiD literature need the *joint*
covariance of an event study rather than its per-horizon standard errors:

* **Simultaneous confidence bands.** A pointwise 95% interval at each of
  ``k`` horizons covers the whole path with probability well below 95%.
  The sup-t band replaces ``z_{1-alpha/2}`` by the ``1 - alpha`` quantile of
  ``max_j |Z_j|`` with ``Z ~ N(0, R)`` and ``R`` the correlation matrix of
  the coefficients (Montiel Olea and Plagborg-Moller 2019,
  [@olea2019simultaneous]; the ``cband`` of Callaway and Sant'Anna 2021,
  [@callaway2021difference]). Baker et al. (2026) report pointwise and
  simultaneous intervals side by side in their event-study figures and
  note that the bands are produced by default *by the Callaway--Sant'Anna
  packages* [@baker2026difference, §5.2.3 and fn. 20] -- which is the
  problem this module solves: every other estimator in the family had the
  covariance and exposed no band.
* **HonestDiD.** The Rambachan-Roth (2023) fixed-length confidence interval
  is a function of the event-study vector *and its covariance*
  [@rambachan2023more].

Before this module only Callaway-Sant'Anna / ``aggte`` results exposed the
joint covariance in a form those tools could read, so ``honest_did`` fell
back to a worst-case-bias approximation for every other estimator.
:func:`event_study_vcov` reads it off every event-study estimator in
``sp.did``; :func:`uniform_bands` builds the sup-t band on top of it.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility

__all__ = ["EventStudyVcov", "event_study_vcov", "uniform_bands"]


class EventStudyVcov(NamedTuple):
    """Event-study coefficients with their covariance.

    ``times`` are the relative event times (reference period excluded),
    ``beta`` the coefficients and ``vcov`` their covariance, all aligned.
    ``joint`` is ``False`` when the off-diagonal blocks are not available
    from the estimator (``vcov`` is then diagonal, or block diagonal, and
    ``note`` says which); ``source`` names where the matrix was read.
    """

    times: np.ndarray
    beta: np.ndarray
    vcov: np.ndarray
    joint: bool
    source: str
    note: str = ""

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.vcov, index=self.times, columns=self.times)


_GARDNER_LABEL = re.compile(r"^D_k([+-]\d+)$")


def _es_frame(result: Any) -> Optional[pd.DataFrame]:
    """The event-study table as ``relative_time`` / ``att`` / ``se``."""
    mi = getattr(result, "model_info", None) or {}
    es = mi.get("event_study")
    if isinstance(es, dict) and {"horizon", "coef", "se"} <= set(es):
        # sp.gardner_did stores a dict keyed by 'D_k+h' labels.
        rows = []
        for lab in es["horizon"]:
            m = _GARDNER_LABEL.match(str(lab))
            if m is None:
                continue
            rows.append(
                {
                    "relative_time": int(m.group(1)),
                    "att": float(es["coef"][lab]),
                    "se": float(es["se"][lab]),
                    "_label": lab,
                }
            )
        return pd.DataFrame(rows)
    if isinstance(es, pd.DataFrame) and {"relative_time", "att", "se"} <= set(
        es.columns
    ):
        return es
    det = getattr(result, "detail", None)
    if isinstance(det, pd.DataFrame) and {"relative_time", "att", "se"} <= set(
        det.columns
    ):
        return det
    return None


def _usable(es: pd.DataFrame) -> pd.DataFrame:
    """Drop reference / unestimated rows (SE zero or not finite)."""
    keep = np.isfinite(es["att"].to_numpy(dtype=float)) & (
        es["se"].to_numpy(dtype=float) > 0
    )
    if "is_reference" in es.columns:
        keep &= ~es["is_reference"].fillna(False).to_numpy(dtype=bool)
    return es.loc[keep]


def _from_labelled(
    es: pd.DataFrame, V: pd.DataFrame, source: str, joint: bool, note: str = ""
) -> EventStudyVcov:
    es = _usable(es)
    times = [int(t) for t in es["relative_time"] if int(t) in set(V.index)]
    if not times:
        raise MethodIncompatibility(
            "The event-study covariance shares no event time with the table.",
            recovery_hint="Refit the estimator with an event-study window.",
            diagnostics={"source": source},
        )
    sub = es.set_index(es["relative_time"].astype(int)).loc[times]
    return EventStudyVcov(
        times=np.asarray(times, dtype=int),
        beta=sub["att"].to_numpy(dtype=float),
        vcov=V.loc[times, times].to_numpy(dtype=float),
        joint=joint,
        source=source,
        note=note,
    )


def event_study_vcov(result: Any, *, allow_diagonal: bool = True) -> EventStudyVcov:
    """Event-study coefficients and their joint covariance, from any DiD fit.

    .. versionadded:: 1.30.0

    Reads the covariance each estimator produces:

    ========================================  ==================================
    estimator                                 source
    ========================================  ==================================
    ``callaway_santanna`` (raw fit)           ``aggte(type='dynamic')``
    ``aggte(type='dynamic')``                 ``model_info['vcov']``
    ``event_study`` (dynamic TWFE)            ``model_info['vcov']``
    ``sun_abraham``                           ``model_info['vcov_event_time']``
    ``gardner_did(event_study=True)``         ``model_info['event_study']['vcov']``
    ``did_imputation(horizon=...)``           ``model_info['event_study_vcov']``
    ``stacked_did``, ``lp_did``,              ``model_info['event_study_vcov']``
    ``did_multiplegt_dyn``
    ``etwfe`` (raw fit)                       ``etwfe_emfx(type='event')``
    ``etwfe_emfx(type='event')``              ``model_info['event_study_vcov']``
    ========================================  ==================================

    Reference rows (fixed at zero) are excluded.

    Parameters
    ----------
    result : CausalResult
        A fitted event-study estimator.
    allow_diagonal : bool, default True
        If no joint covariance is available, return ``diag(se**2)`` with
        ``joint=False`` and a warning instead of raising.

    Returns
    -------
    EventStudyVcov
        ``times``, ``beta``, ``vcov``, ``joint``, ``source``, ``note``.

    Notes
    -----
    ``did_imputation`` with ``pretrend_method='bjs'`` estimates its leads
    in an auxiliary regression on the untreated sample and its horizons by
    imputation; neither the construction nor Stata ``did_imputation``
    provides their cross-covariance, so the matrix is block diagonal and
    ``joint`` is ``False`` (the post-period block is exact).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> fit = sp.sun_abraham(df, y="lemp", g="first_treat", t="year",
    ...                      i="countyreal")
    >>> es = sp.event_study_vcov(fit)
    >>> es.vcov.shape == (len(es.times), len(es.times))
    True
    """
    mi = getattr(result, "model_info", None) or {}

    # 1. Estimators that store a labelled event-study covariance.
    V = mi.get("event_study_vcov")
    es = _es_frame(result)
    if isinstance(V, pd.DataFrame) and es is not None:
        block = bool(V.attrs.get("block_diagonal", False))
        return _from_labelled(
            es,
            V,
            source="model_info['event_study_vcov']",
            joint=not block,
            note=(
                "block diagonal: leads and horizons come from separate "
                "regressions whose cross-covariance is not produced"
                if block
                else ""
            ),
        )

    # 2. aggte(type='dynamic') and raw Callaway-Sant'Anna fits.
    if (
        mi.get("aggregation") == "dynamic"
        or getattr(result, "_influence_funcs", None) is not None
    ):
        from ._flci import event_study_moments

        mom = event_study_moments(result)
        if mom is not None:
            b, S, t = mom
            return EventStudyVcov(
                times=np.asarray(t, dtype=int),
                beta=np.asarray(b, dtype=float),
                vcov=np.asarray(S, dtype=float),
                joint=True,
                source="aggte(type='dynamic') influence functions",
            )

    # 3. Sun-Abraham interaction-weighted event study.
    if mi.get("vcov_event_time") is not None and es is not None:
        times = [int(t) for t in mi.get("event_times", [])]
        Vdf = pd.DataFrame(np.asarray(mi["vcov_event_time"], dtype=float), times, times)
        return _from_labelled(es, Vdf, "model_info['vcov_event_time']", True)

    # 4. Dynamic TWFE event study.
    if mi.get("vcov") is not None and mi.get("vcov_event_times") is not None:
        times = [int(t) for t in mi["vcov_event_times"]]
        Vdf = pd.DataFrame(np.asarray(mi["vcov"], dtype=float), times, times)
        if es is not None:
            return _from_labelled(es, Vdf, "model_info['vcov']", True)

    # 5. Gardner two-stage event study.
    es_raw = mi.get("event_study")
    if isinstance(es_raw, dict) and isinstance(es_raw.get("vcov"), pd.DataFrame):
        lab = es_raw["vcov"]
        rt = []
        for x in lab.index:
            m = _GARDNER_LABEL.match(str(x))
            rt.append(int(m.group(1)) if m else None)
        if all(r is not None for r in rt) and es is not None:
            Vdf = pd.DataFrame(lab.to_numpy(dtype=float), rt, rt)
            return _from_labelled(es, Vdf, "model_info['event_study']['vcov']", True)

    # 6. Raw ETWFE fit: aggregate its cells to event time.
    if mi.get("event_vcov") is not None and "cohorts" in mi:
        from .wooldridge_did import etwfe_emfx

        return event_study_vcov(
            etwfe_emfx(result, type="event", include_leads=True),
            allow_diagonal=allow_diagonal,
        )

    if es is None:
        raise MethodIncompatibility(
            "Result does not expose an event study.",
            recovery_hint=(
                "Fit an event-study estimator (e.g. sp.sun_abraham, "
                "sp.did_imputation(horizon=...), sp.aggte(type='dynamic'))."
            ),
            diagnostics={"result_type": type(result).__name__},
        )
    if not allow_diagonal:
        raise MethodIncompatibility(
            "This result carries per-horizon standard errors but no joint "
            "covariance.",
            recovery_hint="Pass allow_diagonal=True to use diag(se^2).",
            diagnostics={"method": getattr(result, "method", None)},
        )
    es = _usable(es)
    warnings.warn(
        "event_study_vcov: no joint event-study covariance on this result; "
        "using diag(se^2). Uniform bands built from it are conservative "
        "(Sidak) and HonestDiD intervals ignore cross-horizon correlation.",
        UserWarning,
        stacklevel=2,
    )
    se = es["se"].to_numpy(dtype=float)
    return EventStudyVcov(
        times=es["relative_time"].to_numpy(dtype=int),
        beta=es["att"].to_numpy(dtype=float),
        vcov=np.diag(se**2),
        joint=False,
        source="diag(se^2)",
        note="no joint covariance available",
    )


def _sup_t_critical(
    corr: np.ndarray, alpha: float, n_draws: int, rng: np.random.Generator
) -> float:
    """``1 - alpha`` quantile of ``max_j |Z_j|``, ``Z ~ N(0, corr)``."""
    k = corr.shape[0]
    # Symmetric square root via eigen-decomposition: robust to the
    # rank-deficient correlation matrices event studies routinely produce.
    w, U = np.linalg.eigh(0.5 * (corr + corr.T))
    root = U * np.sqrt(np.clip(w, 0.0, None))
    z = rng.standard_normal((n_draws, k)) @ root.T
    return float(np.quantile(np.max(np.abs(z), axis=1), 1.0 - alpha))


def _sup_t_draws(corr: np.ndarray, n_draws: int, rng: np.random.Generator) -> Any:
    """Draws of ``max_j |Z_j|``, ``Z ~ N(0, corr)`` (same stream as above)."""
    w, U = np.linalg.eigh(0.5 * (corr + corr.T))
    root = U * np.sqrt(np.clip(w, 0.0, None))
    z = rng.standard_normal((n_draws, corr.shape[0])) @ root.T
    return np.max(np.abs(z), axis=1)


def uniform_bands(
    result: Any,
    *,
    alpha: float = 0.05,
    which: str = "all",
    window: Optional[Sequence[int]] = None,
    n_draws: int = 100_000,
    seed: Optional[int] = 0,
    terms: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Sup-t simultaneous confidence band and test for an event study.

    .. versionadded:: 1.30.0

    Parameters
    ----------
    result : CausalResult
        Any event-study fit accepted by :func:`event_study_vcov`. With
        ``terms=``, any fit that has ``params`` and their full covariance
        (``sp.regress``, ``sp.feols``, ...).
    terms : sequence of str, optional
        Names of the coefficients to cover, for an event study written by
        hand as a regression on lead and lag dummies:
        ``sp.uniform_bands(fit, terms=["lead3", "lead2", "lead1"])``. The
        rows are then labelled by ``term`` and ``which`` / ``window`` do
        not apply.

        .. versionadded:: 1.39.0
    alpha : float, default 0.05
        One minus the simultaneous coverage.
    which : {'all', 'post', 'pre'}, default 'all'
        Which event times the band covers jointly. ``'post'`` is the usual
        choice for the treatment-effect path, ``'pre'`` for a visual
        pre-trend check; ``'all'`` covers both.
    window : (int, int), optional
        Restrict the covered event times to ``lo <= k <= hi`` (applied
        together with ``which``), e.g. the window a figure displays. The
        critical value depends on how many coefficients are covered, so a
        band must be computed on the window it is drawn on.
    n_draws : int, default 100_000
        Gaussian draws for the critical value. At the default the
        seed-to-seed SD of the quantile is about 5e-3 (measured 0.004 to
        0.006 on the mpdta event studies), roughly 0.2 percent of the
        critical value; raise ``n_draws`` for a steadier band.
    seed : int or None, default 0
        Seed for the draws.

    Returns
    -------
    pandas.DataFrame
        One row per event time: ``relative_time``, ``att``, ``se``,
        pointwise ``ci_lower`` / ``ci_upper`` and simultaneous
        ``cband_lower`` / ``cband_upper``. ``.attrs`` carries
        ``crit_pointwise``, ``crit_uniform``, ``joint`` (whether the full
        covariance was used), ``source`` and ``alpha``, and the sup-t test
        of the hypothesis that every covered coefficient is zero:
        ``max_t`` (the largest absolute t statistic) and ``supt_pvalue``
        (the share of draws of ``max_j |Z_j|`` at or above it). With many
        correlated pre-trend coefficients this test stays reliable where
        the Wald test's covariance is close to singular.

    Notes
    -----
    The band is ``beta_j +/- c * se_j`` where ``c`` is the ``1 - alpha``
    quantile of ``max_j |Z_j|`` for ``Z ~ N(0, R)``, ``R`` the correlation
    matrix of the covered coefficients [@olea2019simultaneous]. When the
    estimator exposes no joint covariance, ``R`` is the identity and ``c``
    is the Sidak value, which is conservative for Gaussian vectors with any
    correlation (Sidak's inequality); ``attrs['joint']`` is then ``False``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> fit = sp.sun_abraham(df, y="lemp", g="first_treat", t="year",
    ...                      i="countyreal")
    >>> band = sp.uniform_bands(fit, which="post")
    >>> bool(band.attrs["crit_uniform"] > band.attrs["crit_pointwise"])
    True
    """
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1); got {alpha!r}.",
            recovery_hint="Pass e.g. alpha=0.05.",
            diagnostics={"alpha": alpha},
        )
    if which not in ("all", "post", "pre"):
        raise MethodIncompatibility(
            f"which must be 'all', 'post' or 'pre'; got {which!r}.",
            recovery_hint="Use which='post' for the effect path.",
            diagnostics={"which": which},
        )
    if int(n_draws) < 1000:
        raise MethodIncompatibility(
            f"n_draws={n_draws} is too few for a tail quantile.",
            recovery_hint="Use n_draws >= 1000 (default 100000).",
            diagnostics={"n_draws": n_draws},
        )
    if terms is not None:
        if which != "all" or window is not None:
            raise MethodIncompatibility(
                "terms= names the coefficients to cover; which= and window= "
                "select event times and do not apply with it.",
                diagnostics={"which": which, "window": repr(window)},
            )
        return _uniform_bands_terms(
            result, list(terms), float(alpha), int(n_draws), seed
        )
    es = event_study_vcov(result)
    sel: np.ndarray = np.ones(es.times.size, dtype=bool)
    if which == "post":
        sel = np.asarray(es.times >= 0)
    elif which == "pre":
        sel = np.asarray(es.times < 0)
    if window is not None:
        try:
            lo, hi = (int(window[0]), int(window[1]))
        except (TypeError, ValueError, IndexError):
            raise MethodIncompatibility(
                f"window must be a pair of integers; got {window!r}.",
                recovery_hint="Pass e.g. window=(0, 4).",
                diagnostics={"window": repr(window)},
            ) from None
        sel &= (es.times >= lo) & (es.times <= hi)
    if not sel.any():
        raise MethodIncompatibility(
            f"No {which!r} event times to cover.",
            recovery_hint="Choose which='all' or refit with a wider window.",
            diagnostics={"times": es.times.tolist()},
        )
    times, beta = es.times[sel], es.beta[sel]
    V = es.vcov[np.ix_(sel, sel)]
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    if np.any(se <= 0):
        raise MethodIncompatibility(
            "A covered event-time coefficient has zero variance.",
            recovery_hint="Exclude reference periods from the band.",
            diagnostics={"times": times.tolist()},
        )
    corr = V / np.outer(se, se)
    rng = np.random.default_rng(seed)
    sup = _sup_t_draws(corr, int(n_draws), rng)
    crit_u = float(np.quantile(sup, 1.0 - float(alpha)))
    max_t = float(np.max(np.abs(beta / se)))
    crit_p = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    out = pd.DataFrame(
        {
            "relative_time": times,
            "att": beta,
            "se": se,
            "ci_lower": beta - crit_p * se,
            "ci_upper": beta + crit_p * se,
            "cband_lower": beta - crit_u * se,
            "cband_upper": beta + crit_u * se,
        }
    )
    out.attrs.update(
        {
            "crit_pointwise": crit_p,
            "crit_uniform": crit_u,
            "max_t": max_t,
            "supt_pvalue": float(np.mean(sup >= max_t)),
            "joint": bool(es.joint),
            "source": es.source,
            "note": es.note,
            "alpha": float(alpha),
            "which": which,
            "window": None if window is None else (int(window[0]), int(window[1])),
            "n_draws": int(n_draws),
        }
    )
    return out


def _uniform_bands_terms(
    result: Any, terms: list, alpha: float, n_draws: int, seed: Optional[int]
) -> pd.DataFrame:
    """Sup-t band and test for named coefficients of any fit."""
    from ..postestimation._covariance import coefficient_covariance

    params = getattr(result, "params", None)
    if not isinstance(params, pd.Series) or not terms:
        raise MethodIncompatibility(
            "terms= needs a fit with named coefficients (result.params) and "
            "at least one name.",
            recovery_hint="Pass e.g. an sp.regress fit and its lead dummies.",
        )
    missing = [t for t in terms if t not in params.index]
    if missing or len(set(terms)) != len(terms):
        raise MethodIncompatibility(
            f"terms not among the coefficients, or repeated: {missing or terms}.",
            recovery_hint=f"Coefficients: {list(params.index)[:20]}",
            diagnostics={"missing": missing},
        )
    V_full, _ = coefficient_covariance(result)
    if V_full is None:
        raise MethodIncompatibility(
            "This fit carries standard errors but not the covariance of its "
            "coefficients, which a simultaneous band needs.",
            recovery_hint="Refit with an estimator that stores the covariance "
            "(sp.regress, sp.feols).",
        )
    pos = [list(params.index).index(t) for t in terms]
    beta = params.to_numpy(dtype=float)[pos]
    V = np.asarray(V_full, dtype=float)[np.ix_(pos, pos)]
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    if np.any(se <= 0) or not np.all(np.isfinite(se)):
        raise MethodIncompatibility(
            "A covered coefficient has zero or undefined variance.",
            recovery_hint="Leave omitted (collinear) terms out of terms=.",
            diagnostics={"terms": terms},
        )
    corr = V / np.outer(se, se)
    sup = _sup_t_draws(corr, n_draws, np.random.default_rng(seed))
    crit_u = float(np.quantile(sup, 1.0 - alpha))
    crit_p = float(stats.norm.ppf(1.0 - alpha / 2.0))
    max_t = float(np.max(np.abs(beta / se)))
    out = pd.DataFrame(
        {
            "term": terms,
            "estimate": beta,
            "se": se,
            "ci_lower": beta - crit_p * se,
            "ci_upper": beta + crit_p * se,
            "cband_lower": beta - crit_u * se,
            "cband_upper": beta + crit_u * se,
        }
    )
    out.attrs.update(
        {
            "crit_pointwise": crit_p,
            "crit_uniform": crit_u,
            "max_t": max_t,
            "supt_pvalue": float(np.mean(sup >= max_t)),
            "joint": True,
            "source": "coefficient covariance of the fit",
            "note": "",
            "alpha": alpha,
            "n_draws": n_draws,
        }
    )
    return out
