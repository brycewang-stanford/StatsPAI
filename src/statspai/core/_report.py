"""The interval to report, given the diagnostics a fit already holds.

A fitted result carries the conventional estimate and interval, and next to
them the diagnostics that say when that interval cannot be trusted (a weak
first stage, a rejected pre-trend test, a density break at an RD cutoff).
Reading the two together is left to the caller, and a caller in a hurry
reports the conventional interval. :func:`recommended_report` does the
reading: it returns the conventional interval when no diagnostic fires, the
robust alternative when one does, and a refusal when no defensible interval
exists. The conventional numbers are never changed; they are repeated under
``conventional`` so both can be shown.

Rules (each follows the cited paper, not a tuning exercise):

* IV, one endogenous regressor, one instrument. The Wald interval with the
  1.96 critical value is valid only when the first-stage F exceeds 104.7
  [@lee2022valid]. Below that the report is the Anderson-Rubin set, which
  is valid at any instrument strength and efficient in the just-identified
  model [@anderson1949estimation; @andrews2019weak]; when the arrays needed
  to invert the test are not available (weights, clustering) it is the tF
  interval of [@lee2022valid].
* IV, several instruments. When the effective F [@olea2013robust] is below
  its 10% worst-case-bias critical value, the report is the LIML estimate
  with the Anderson-Rubin set [@andrews2019weak].
* An unbounded robust set means the data do not bound the effect at the
  stated level: the report is a refusal that says so.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["recommended_report", "report_data"]

# The estimation data, when the caller has it: some reports (the wild
# cluster bootstrap) cannot be computed from the fitted result alone.
_DATA: "contextvars.ContextVar[Optional[pd.DataFrame]]" = contextvars.ContextVar(
    "statspai_report_data", default=None
)


@contextlib.contextmanager
def report_data(data: Optional[pd.DataFrame]) -> Iterator[None]:
    """Make ``data`` available to every ``report()`` computed inside."""
    token = _DATA.set(data)
    try:
        yield
    finally:
        _DATA.reset(token)


# First-stage F above which the just-identified Wald interval has correct
# size with the usual critical value (Lee, McCrary, Moreira and Porter 2022).
_TF_STRONG_F = 104.7
# Worst-case bias fraction for the Olea-Pflueger critical value.
_MOP_TAU = 0.10


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _blank(conventional: Dict[str, Any], level: float) -> Dict[str, Any]:
    return {
        "status": "conventional",
        "estimate": conventional.get("estimate"),
        "ci": conventional.get("ci"),
        "level": level,
        "ci_method": conventional.get("ci_method", "wald"),
        "basis": [],
        "refuse": None,
        "conventional": conventional,
        "alternatives": {},
    }


def _refuse(rep: Dict[str, Any], reason: str) -> Dict[str, Any]:
    rep.update(status="refuse", estimate=None, ci=None, ci_method=None, refuse=reason)
    return rep


# --------------------------------------------------------------------- IV


def _iv_frame(iv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rebuild a named frame from the arrays an IV fit stores."""
    n_exog, n_endog = int(iv["n_exog"]), int(iv["n_endog"])
    X, W, y = np.asarray(iv["X"]), np.asarray(iv["W"]), np.asarray(iv["y"])
    if n_endog != 1 or W.shape[1] <= n_exog:
        return None
    exog, Z, d = X[:, :n_exog], W[:, n_exog:], X[:, n_exog]
    # the weak-IV routines add their own constant
    scale = np.maximum(np.abs(exog).max(axis=0), 1e-300) if n_exog else np.ones(0)
    keep = [j for j in range(n_exog) if np.ptp(exog[:, j]) > 1e-12 * scale[j]]
    has_const = len(keep) < n_exog
    cols: Dict[str, np.ndarray] = {"__y": y, "__d": d}
    xn = [f"__x{j}" for j in keep]
    zn = [f"__z{j}" for j in range(Z.shape[1])]
    cols.update({name: exog[:, j] for name, j in zip(xn, keep)})
    cols.update({name: Z[:, j] for j, name in enumerate(zn)})
    return {
        "data": pd.DataFrame(cols),
        "exog": xn,
        "instruments": zn,
        "has_const": has_const,
        "arrays": (y, exog, d.reshape(-1, 1), Z),
    }


def _iv_report(result: Any) -> Optional[Dict[str, Any]]:
    mi = getattr(result, "model_info", None) or {}
    di = getattr(result, "data_info", None) or {}
    iv = di.get("iv") if isinstance(di, dict) else None
    if not isinstance(iv, dict) or int(iv.get("n_endog", 0)) != 1:
        return None
    endog = iv["endog_names"][0]
    level = 1.0 - float(mi.get("alpha", 0.05) or 0.05)
    try:
        est = _f(result.params[endog])
        se = _f(result.std_errors[endog])
        lo = _f(result.conf_int_lower[endog])
        hi = _f(result.conf_int_upper[endog])
    except Exception:
        return None
    conventional = {
        "estimate": est,
        "se": se,
        "ci": [lo, hi],
        "ci_method": "wald",
        "estimator": str(mi.get("method", "2sls")),
    }
    rep = _blank(conventional, level)
    rep["term"] = endog

    diag = getattr(result, "diagnostics", None) or {}
    f_eff = _f(diag.get("Olea-Pflueger effective F")) or _f(mi.get("first_stage_f"))
    k = int(np.asarray(iv["W"]).shape[1]) - int(iv["n_exog"])
    if f_eff is None or est is None or se is None:
        return rep
    rep["first_stage_F"] = f_eff
    rep["n_instruments"] = k

    weighted = bool(iv.get("weighted"))
    clustered = mi.get("cluster") not in (None, False, "")
    robust = str(mi.get("robust") or "nonrobust").lower()
    ar_vcov = (
        "classic" if robust in ("nonrobust", "none", "false", "classic") else "HC1"
    )
    arrays_ok = not weighted and not clustered and abs(level - 0.95) < 1e-9
    frame = _iv_frame(iv) if arrays_ok else None
    if frame is not None and not frame["has_const"]:
        frame = None  # the weak-IV routines assume an intercept

    def _ar() -> Optional[Dict[str, Any]]:
        if frame is None:
            return None
        from ..diagnostics.weak_iv import anderson_rubin_test

        out = anderson_rubin_test(
            frame["data"],
            y="__y",
            endog="__d",
            instruments=frame["instruments"],
            exog=frame["exog"] or None,
            ar_vcov=ar_vcov,
        )
        a, b = out["ar_ci"]
        return {
            "ci": [float(a), float(b)],
            "bounded": bool(np.isfinite(a) and np.isfinite(b)),
            "disjoint": bool(out.get("ar_ci_disjoint", False)),
            "vcov": out.get("ar_vcov"),
        }

    if k == 1:
        if f_eff >= _TF_STRONG_F:
            return rep
        from ..diagnostics.weak_iv import tF_critical_value

        crit = float(tF_critical_value(f_eff)) if abs(level - 0.95) < 1e-9 else None
        rep["basis"].append(
            f"first-stage F = {f_eff:.1f} is below {_TF_STRONG_F}: the Wald "
            "interval with the 1.96 critical value under-covers "
            "(Lee, McCrary, Moreira and Porter 2022)."
        )
        if crit is not None:
            tf_ci = (
                [est - crit * se, est + crit * se]
                if math.isfinite(crit)
                else [-math.inf, math.inf]
            )
            rep["alternatives"]["tF"] = {"critical_value": crit, "ci": tf_ci}
        ar = _ar()
        if ar is not None:
            rep["alternatives"]["anderson_rubin"] = ar
        chosen, name = None, None
        if ar is not None and not ar["disjoint"]:
            chosen, name = ar["ci"], "anderson_rubin"
        elif crit is not None:
            chosen, name = rep["alternatives"]["tF"]["ci"], "tF"
        if chosen is None:
            rep["basis"].append(
                "no weak-instrument-robust interval is available for this "
                "fit (level other than 95%); treat the Wald interval as a "
                "lower bound on the uncertainty."
            )
            rep["status"] = "caution"
            return rep
        if not (math.isfinite(chosen[0]) and math.isfinite(chosen[1])):
            return _refuse(
                rep,
                f"the instrument is too weak (first-stage F = {f_eff:.1f}) to "
                f"bound the effect: the {level:.0%} weak-instrument-robust "
                "set is unbounded.",
            )
        rep.update(status="adjusted", ci=chosen, ci_method=name)
        return rep

    # several instruments
    crit10 = None
    if frame is not None:
        from ..diagnostics.weak_iv import effective_f_test

        eff = effective_f_test(
            frame["data"],
            endog="__d",
            instruments=frame["instruments"],
            exog=frame["exog"] or None,
            vcov="classic" if ar_vcov == "classic" else "HC1",
        )
        crit10 = _f(
            ((eff.get("critical_values") or {}).get("simplified") or {}).get(_MOP_TAU)
        )
    threshold = crit10 if crit10 is not None else 23.1
    rep["weak_threshold"] = threshold
    if f_eff >= threshold:
        return rep
    rep["basis"].append(
        f"effective F = {f_eff:.1f} with {k} instruments is below the "
        f"critical value {threshold:.1f} for a {_MOP_TAU:.0%} worst-case bias "
        "(Montiel Olea and Pflueger 2013): 2SLS is biased toward OLS and its "
        "Wald interval under-covers."
    )
    ar = _ar()
    if ar is None:
        rep["status"] = "caution"
        rep["basis"].append(
            "a weak-instrument-robust set could not be computed from this "
            "fit (weights or clustering); run sp.anderson_rubin_test with the "
            "same cluster variable."
        )
        return rep
    rep["alternatives"]["anderson_rubin"] = ar
    liml = None
    try:
        from ..regression.iv import _k_class_fit, _liml_kappa

        y, exog, d, Z = frame["arrays"]
        kappa = _liml_kappa(y, exog, d, Z)
        fit = _k_class_fit(
            y, exog, d, Z, kappa, robust="nonrobust" if ar_vcov == "classic" else "hc1"
        )
        liml = {
            "estimate": float(np.asarray(fit["params"]).ravel()[-1]),
            "se": float(np.asarray(fit["std_errors"]).ravel()[-1]),
            "kappa": float(kappa),
        }
        rep["alternatives"]["liml"] = liml
    except Exception as exc:  # the report must not fail the fit
        from ..workflow._degradation import record_degradation

        record_degradation(
            rep.setdefault("degradations", []), section="report.liml", exc=exc
        )
    if not ar["bounded"] or ar["disjoint"]:
        return _refuse(
            rep,
            f"the instruments are too weak (effective F = {f_eff:.1f}, {k} "
            f"instruments) to bound the effect: the {level:.0%} "
            "Anderson-Rubin set is unbounded or disjoint.",
        )
    rep.update(status="adjusted", ci=ar["ci"], ci_method="anderson_rubin")
    if liml is not None and ar["ci"][0] <= liml["estimate"] <= ar["ci"][1]:
        rep["estimate"] = liml["estimate"]
        rep["estimator"] = "liml"
    return rep


# -------------------------------------------------------------------- DiD

# Power a pre-trend test needs before "it did not reject" is evidence.
_PRETREND_POWER = 0.80


def _did_report(result: Any) -> Optional[Dict[str, Any]]:
    mi = getattr(result, "model_info", None) or {}
    es = mi.get("event_study") if isinstance(mi, dict) else None
    pt = mi.get("pretrend_test") if isinstance(mi, dict) else None
    if not isinstance(es, pd.DataFrame) or "relative_time" not in es:
        return None
    coef = "att" if "att" in es else ("estimate" if "estimate" in es else None)
    est, se = _f(getattr(result, "estimate", None)), _f(getattr(result, "se", None))
    if coef is None or est is None or se is None or se <= 0:
        return None
    alpha = float(getattr(result, "alpha", 0.05) or 0.05)
    from scipy import stats

    z = float(stats.norm.ppf(1 - alpha / 2))
    ci = getattr(result, "ci", None)
    conv_ci = (
        [float(ci[0]), float(ci[1])]
        if ci is not None and len(ci) == 2
        else [est - z * se, est + z * se]
    )
    rep = _blank(
        {"estimate": est, "se": se, "ci": conv_ci, "ci_method": "wald"}, 1 - alpha
    )

    rt = es["relative_time"].to_numpy(dtype=float)
    b = es[coef].to_numpy(dtype=float)
    ses = es["se"].to_numpy(dtype=float) if "se" in es else np.full(len(es), np.nan)
    post = rt >= 0
    pre = (rt < 0) & np.isfinite(ses) & (ses > 0)
    n_post, n_pre = int(post.sum()), int(pre.sum())
    p_pre = _f(pt.get("pvalue")) if isinstance(pt, dict) else None
    rejected = p_pre is not None and p_pre < alpha
    if p_pre is not None:
        rep["pretrend_pvalue"] = p_pre
    if n_post == 0 or n_pre == 0:
        return rep

    # The headline must be the plain average of the post-period event-study
    # coefficients for the sensitivity interval to be about the same number.
    same_target = abs(float(np.mean(b[post])) - est) <= 1e-6 * max(abs(est), 1.0)
    horizon = float(np.mean(rt[post] + 1.0))
    slope = z * se / horizon
    power = None
    try:
        from ..did.pretrends import pretrends_power

        pw = pretrends_power(result, slope=slope, alpha=alpha)
        power = _f(pw.get("power_joint")) or _f(pw.get("power"))
    except Exception as exc:
        from ..workflow._degradation import record_degradation

        record_degradation(
            rep.setdefault("degradations", []), section="report.power", exc=exc
        )
    if power is not None:
        rep["pretrend_power"] = power
        rep["pretrend_power_slope"] = slope
    underpowered = power is not None and power < _PRETREND_POWER
    if not rejected and not underpowered:
        return rep

    if rejected:
        rep["basis"].append(
            f"the pre-trend test rejects parallel trends (p = {p_pre:.3g}): the "
            "conventional estimate carries the bias of the differential trend."
        )
    else:
        rep["basis"].append(
            f"the pre-trend test has power {power:.2f} against a linear "
            f"differential trend of {slope:.3g} per period, the trend that "
            "would move the truth out of the conventional interval; not "
            "rejecting is weak evidence for parallel trends (Roth 2022)."
        )
    if not same_target or n_pre < 2:
        rep["status"] = "caution"
        rep["basis"].append(
            "a trend-robust interval for this aggregate is not computed here "
            "(the headline is not the plain average of the post-period "
            "event-study coefficients, or fewer than two pre-period "
            "coefficients are estimated); run sp.honest_did on the event "
            "study."
        )
        return rep
    try:
        from ..did.honest_did import honest_did

        lvec = np.full(n_post, 1.0 / n_post)
        sd = honest_did(
            result, l_vec=lvec, method="smoothness", m_grid=[0.0], alpha=alpha
        ).iloc[0]
        lo, hi = float(sd["ci_lower"]), float(sd["ci_upper"])
        rep["alternatives"]["linear_trend"] = {"ci": [lo, hi], "M": 0.0}
        rm = honest_did(
            result, l_vec=lvec, method="relative_magnitude", m_grid=[1.0], alpha=alpha
        ).iloc[0]
        rep["alternatives"]["relative_magnitude"] = {
            "ci": [float(rm["ci_lower"]), float(rm["ci_upper"])],
            "Mbar": 1.0,
        }
    except Exception as exc:
        from ..workflow._degradation import record_degradation

        record_degradation(
            rep.setdefault("degradations", []), section="report.honest_did", exc=exc
        )
        if "linear_trend" not in rep["alternatives"]:
            rep["status"] = "caution"
            return rep
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return _refuse(
            rep,
            "the design cannot bound the effect once a linear differential "
            "trend is allowed: the trend-robust interval is unbounded.",
        )
    rep["basis"].append(
        "the reported interval allows the pre-period differential trend to "
        "continue linearly after treatment (Rambachan and Roth 2023, "
        "smoothness restriction with M = 0)."
    )
    rep.update(status="adjusted", ci=[lo, hi], ci_method="honest_did_linear_trend")
    if rejected or not (lo <= est <= hi):
        rep["estimate"] = 0.5 * (lo + hi)
        rep["estimator"] = "linear_trend_adjusted"
    return rep


# --------------------------------------------------------------------- RD

# Density-test levels: below the first the design is flagged, below the
# second the sorting is treated as established.
_DENSITY_FLAG, _DENSITY_REFUSE = 0.05, 0.01


def _rd_report(result: Any) -> Optional[Dict[str, Any]]:
    mi = getattr(result, "model_info", None) or {}
    if not isinstance(mi, dict) or "rd_type" not in mi:
        return None
    est, se = _f(getattr(result, "estimate", None)), _f(getattr(result, "se", None))
    ci = getattr(result, "ci", None)
    if est is None or se is None or ci is None:
        return None
    alpha = float(getattr(result, "alpha", 0.05) or 0.05)
    rep = _blank(
        {
            "estimate": est,
            "se": se,
            "ci": [float(ci[0]), float(ci[1])],
            "ci_method": "robust_bias_corrected",
        },
        1 - alpha,
    )
    dens = mi.get("mccrary")
    p = _f(dens.get("pvalue")) if isinstance(dens, dict) else None
    if p is None:
        return rep
    rep["density_pvalue"] = p
    if p >= _DENSITY_FLAG:
        return rep
    rep["basis"].append(
        f"the density test rejects continuity of the running variable at the "
        f"cutoff (p = {p:.3g}): units appear to sort across it, so those just "
        "above and just below are not comparable (McCrary 2008; Cattaneo, "
        "Jansson and Ma 2020)."
    )
    if p < _DENSITY_REFUSE:
        return _refuse(
            rep,
            f"the running variable is manipulated at the cutoff (density test "
            f"p = {p:.3g}); the discontinuity does not identify the effect. A "
            "donut estimate (sp.rdrobust(..., donut=d)) identifies it only "
            "under the further assumption that sorting is confined to the "
            "excluded window.",
        )
    rep["status"] = "caution"
    return rep


# ---------------------------------------------------------- few clusters

_MAX_BOOT_TERMS = 10


def _cluster_report(result: Any) -> Optional[Dict[str, Any]]:
    mi = getattr(result, "model_info", None) or {}
    if not isinstance(mi, dict) or not hasattr(result, "params"):
        return None
    cluster = mi.get("cluster")
    g = _f(mi.get("n_clusters"))
    if not isinstance(cluster, str) or g is None:
        return None
    from ._agent_summary import _FEW_CLUSTERS_MIN

    level = 1.0 - float(mi.get("alpha", 0.05) or 0.05)
    focal = None
    finder = getattr(result, "_focal_term", None)
    if callable(finder):
        focal = finder()[0]
    try:
        terms = [str(t) for t in result.params.index]
    except Exception:
        return None
    slopes = [t for t in terms if t.lower() not in ("intercept", "const", "_cons")]

    def _conv(t: str) -> Dict[str, Any]:
        return {
            "estimate": _f(result.params[t]),
            "se": _f(result.std_errors[t]),
            "ci": [_f(result.conf_int_lower[t]), _f(result.conf_int_upper[t])],
            "ci_method": "cluster_robust",
        }

    if focal is not None:
        rep = _blank(_conv(focal), level)
        rep["term"] = focal
    else:
        rep = _blank({"ci_method": "cluster_robust"}, level)
        rep["terms"] = {t: _conv(t) for t in slopes[:_MAX_BOOT_TERMS]}
    rep["n_clusters"] = int(g)
    if g >= _FEW_CLUSTERS_MIN:
        return rep
    rep["basis"].append(
        f"{int(g)} clusters: cluster-robust standard errors are biased down "
        "and their t-tests over-reject with so few "
        "(Cameron, Gelbach and Miller 2008)."
    )
    data = _DATA.get()
    if data is None or cluster not in getattr(data, "columns", ()):
        rep["status"] = "caution"
        rep["basis"].append(
            "pass the estimation data (result.report(data=df)) to get the "
            "wild cluster bootstrap interval, or run sp.wild_cluster_boot."
        )
        return rep
    from ..inference.jackknife import wild_cluster_boot

    todo = [focal] if focal is not None else slopes[:_MAX_BOOT_TERMS]
    done: Dict[str, Any] = {}
    for t in todo:
        try:
            w = wild_cluster_boot(
                result,
                data,
                cluster,
                variable=t,
                confidence_set=True,
                seed=0,
                alpha=1 - level,
            )
            lo, hi = w.get("ci_inverted") or w["ci_boot"]
            done[t] = {
                "ci": [float(lo), float(hi)],
                "p_boot": _f(w.get("p_boot")),
                "n_boot": w.get("n_boot"),
            }
        except Exception as exc:
            from ..workflow._degradation import record_degradation

            record_degradation(
                rep.setdefault("degradations", []),
                section=f"report.wild_cluster_boot[{t}]",
                exc=exc,
            )
    if not done:
        rep["status"] = "caution"
        return rep
    rep["alternatives"]["wild_cluster_bootstrap"] = done
    rep["basis"].append(
        "the reported interval inverts the wild cluster restricted "
        "bootstrap test. It is not a remedy when only one or two clusters "
        "are treated."
    )
    rep.update(status="adjusted", ci_method="wild_cluster_bootstrap")
    if focal is not None and focal in done:
        rep["ci"] = done[focal]["ci"]
    else:
        for t, d in done.items():
            rep["terms"][t] = {**rep["terms"][t], "report_ci": d["ci"]}
    return rep


# ------------------------------------------------------------- dispatch

_BUILDERS: Sequence[Callable[[Any], Optional[Dict[str, Any]]]] = (
    _iv_report,
    _did_report,
    _rd_report,
    _cluster_report,
)


def recommended_report(
    result: Any, data: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """Return the estimate and interval to report for ``result``.

    Parameters
    ----------
    result : fitted StatsPAI result
    data : DataFrame, optional
        The estimation data. Needed only for reports that resample it
        (the wild cluster bootstrap under few clusters).

    Returns
    -------
    dict or None
        ``None`` when no rule covers the result. Otherwise a dict with

        * ``status``: ``'conventional'`` (no diagnostic fired),
          ``'adjusted'`` (a robust interval replaces the conventional one),
          ``'caution'`` (a diagnostic fired and no robust interval could be
          computed) or ``'refuse'`` (no defensible interval);
        * ``estimate``, ``ci``, ``level``, ``ci_method``;
        * ``basis``: the diagnostics behind the choice, as sentences;
        * ``refuse``: the reason when ``status == 'refuse'``;
        * ``conventional``: the fit's own estimate, standard error and
          interval, unchanged;
        * ``alternatives``: every robust quantity computed on the way.

    References
    ----------
    lee2022valid, andrews2019weak, anderson1949estimation, olea2013robust
    """
    if data is not None:
        with report_data(data):
            return recommended_report(result)
    for build in _BUILDERS:
        rep = build(result)
        if rep is not None:
            return rep
    return None


def report_lines(rep: Optional[Dict[str, Any]]) -> List[str]:
    """Render a report as a few plain lines for ``summary()``."""
    if not rep:
        return []
    if rep["status"] == "refuse":
        out = ["Report: no defensible interval. " + str(rep["refuse"])]
    else:
        lo, hi = rep["ci"]
        tag = {
            "conventional": "conventional interval; no diagnostic fired",
            "adjusted": f"{rep['ci_method']} interval replaces the conventional one",
            "caution": "conventional interval, but a diagnostic fired",
        }[rep["status"]]
        out = [
            f"Report: {rep['estimate']:.4g}, {rep['level']:.0%} CI "
            f"[{lo:.4g}, {hi:.4g}] ({tag})"
        ]
    out.extend("  - " + b for b in rep["basis"])
    return out


def summary_block(result: Any) -> List[str]:
    """Lines ``summary()`` appends when the report differs from the fit.

    Empty when no rule covers the result or no diagnostic fired, so the
    printed table of an untroubled fit is unchanged.
    """
    try:
        rep = recommended_report(result)
    except Exception:  # printing a fit must not fail on its report
        return []
    if not rep or rep.get("status") == "conventional":
        return []
    if rep.get("ci") is None and rep.get("status") != "refuse":
        return ["", "Report: " + " ".join(rep.get("basis") or [])]
    return [""] + report_lines(rep) + ["  (result.report() has the details)"]
