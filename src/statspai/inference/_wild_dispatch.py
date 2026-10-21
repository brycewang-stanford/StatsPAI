"""Which wild cluster bootstrap a fitted result needs.

``sp.wild_cluster_boot(result, ...)`` used to refit least squares on whatever
design the result stored: after ``sp.ivreg`` it bootstrapped the OLS
coefficient, after ``sp.logit`` a linear probability model, and it reported
those numbers as the fit's. The bootstrap has to be the one the estimator
defines, as Stata ``boottest`` chooses it after each command:

* least squares (``sp.regress``, ``sp.feols``, ``sp.hdfe_ols`` ...): the
  restricted wild cluster bootstrap (WCR) on the stored design, after
  checking that least squares on it reproduces the coefficients;
* 2SLS (``sp.ivreg``): the wild restricted efficient bootstrap (WRE,
  Davidson-MacKinnon 2010) of :func:`statspai.iv_wild_bootstrap`;
* logit / probit / Poisson (``sp.logit``, ``sp.probit``, ``sp.poisson``,
  ``sp.glm``): the score wild cluster bootstrap (Kline-Santos 2012) with
  ``boottest``'s studentisation;
* ``sp.did`` (2 x 2): the estimator's own ``vce='wild'``, replayed from the
  call record.

Anything else is refused with a typed error rather than bootstrapped as if
it were least squares.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

_SCORE_FAMILIES = {
    ("binomial", "logit"),
    ("binomial", "probit"),
    ("poisson", "log"),
}


def _family_link(result: Any) -> Optional[tuple]:
    mi = getattr(result, "model_info", None) or {}
    di = getattr(result, "data_info", None) or {}
    family = str(mi.get("family") or "").lower()
    link = mi.get("link")
    link_obj = di.get("link_obj")
    if link_obj is not None:
        link = getattr(link_obj, "name", link)
    link = str(link or "").lower()
    if not family and not link:
        return None
    return family, link


def _cluster_codes(
    result: Any, data: pd.DataFrame, cluster: str, n: int, who: str
) -> np.ndarray:
    if cluster not in data.columns:
        from ..exceptions import ColumnNotFound

        raise ColumnNotFound.from_missing({"cluster": cluster}, list(data.columns))
    col = data[cluster]
    kept = (getattr(result, "data_info", None) or {}).get("sample_index")
    if kept is not None and len(kept) == n and len(col) != n:
        try:
            col = col.loc[kept]
        except KeyError:
            pass
    if len(col) != n or col.isna().any():
        raise MethodIncompatibility(
            f"{who}: the cluster column must be row-aligned with the fitted "
            f"sample ({n} rows, no missing values); data has {len(col)} rows.",
            recovery_hint="Pass the DataFrame the model was fitted on, after "
            "dropping incomplete rows before fitting.",
        )
    codes: np.ndarray = pd.factorize(col.to_numpy())[0].astype(np.int64)
    return codes


def _require_plain(who: str, h0: float, confidence_set: bool, method: str) -> None:
    if float(h0) != 0.0 or confidence_set:
        raise MethodIncompatibility(
            f"{who}: the {method} is implemented for H0: beta = 0 without a "
            "test-inversion confidence set.",
            recovery_hint="Drop h0= / confidence_set=.",
            diagnostics={"h0": float(h0), "confidence_set": bool(confidence_set)},
        )


def _score(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    variable: str,
    *,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    h0: float,
    confidence_set: bool,
    family_link: tuple,
) -> Dict[str, Any]:
    import statsmodels.api as sm

    from .jackknife import glm_score_wild_boot

    who = "wild_cluster_boot"
    _require_plain(who, h0, confidence_set, "score wild bootstrap")
    di = result.data_info
    for key in ("weights", "analytic_weights", "offset", "exposure"):
        if di.get(key) is not None or (result.model_info or {}).get(key) is not None:
            raise MethodIncompatibility(
                f"{who}: the score bootstrap is implemented for unweighted "
                f"fits without an offset; this fit used {key}=.",
                recovery_hint="Refit without it, or use the fit's own vce=.",
            )
    X = np.asarray(di["X"], dtype=float)
    y = np.asarray(di["y"], dtype=float).ravel()
    names = [str(v) for v in di.get("var_names") or list(result.params.index)]
    if variable not in names:
        raise MethodIncompatibility(
            f"{who}: {variable!r} is not a coefficient of the fit.",
            recovery_hint=f"Use one of {names}.",
            diagnostics={"available": names},
        )
    family, link = family_link
    fam = (
        sm.families.Poisson()
        if family == "poisson"
        else sm.families.Binomial(
            link=(
                sm.families.links.Logit()
                if link == "logit"
                else sm.families.links.Probit()
            )
        )
    )
    codes = _cluster_codes(result, data, cluster, len(y), who)
    G = int(codes.max()) + 1
    if G < 2:
        raise MethodIncompatibility(
            f"{who}: cluster={cluster!r} has one cluster.",
            recovery_hint="Cluster on a variable with several groups.",
        )
    idx = names.index(variable)
    out = glm_score_wild_boot(
        X, y, fam, codes, idx, n_boot=n_boot, weight_type=weight_type, seed=seed
    )
    params = result.params
    se = getattr(result, "std_errors", None)
    return {
        "method": "score wild cluster bootstrap (Kline-Santos; boottest after "
        f"{'poisson' if family == 'poisson' else link})",
        "beta_hat": float(params[variable]) if variable in params.index else np.nan,
        "se_model": (
            float(se[variable]) if se is not None and variable in se else np.nan
        ),
        "t_stat": float(out["t_obs"]),
        "p_boot": float(out["p_boot"]),
        "n_clusters": G,
        "n_obs": int(len(y)),
        "n_boot": int(out["n_reps"]),
        "n_boot_requested": int(n_boot),
        "enumerated": bool(out["enumerated"]),
        "weight_type": weight_type,
        "h0": 0.0,
    }


def _iv(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    variable: str,
    *,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    alpha: float,
    h0: float,
    confidence_set: bool,
) -> Dict[str, Any]:
    from .iv_wild import iv_wild_bootstrap

    if confidence_set:
        raise MethodIncompatibility(
            "wild_cluster_boot: the test-inversion confidence set is "
            "implemented for least squares; after sp.ivreg use "
            "sp.ivreg(..., vce='wild') or sp.wild_cluster_ci_inv.",
            recovery_hint="Drop confidence_set=.",
        )
    kept = (getattr(result, "data_info", None) or {}).get("sample_index")
    n = len(result.data_info["iv"]["y"])
    if kept is not None and len(kept) == n and len(data) != n:
        data = data.loc[kept]  # rows the fit used (missing / singletons out)
    out = iv_wild_bootstrap(
        result,
        data,
        cluster,
        variable,
        n_boot=n_boot,
        weight_type=weight_type,
        seed=seed,
        alpha=alpha,
        beta0=h0,
    )
    out = dict(out)
    out.setdefault("method", "WRE wild restricted efficient bootstrap (2SLS)")
    out.setdefault("h0", float(h0))
    return out


def _did(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    variable: Optional[str],
    *,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    alpha: float,
    h0: float,
    confidence_set: bool,
    record: Any,
) -> Dict[str, Any]:
    import warnings

    from .vcov_refit import _public

    who = "wild_cluster_boot"
    _require_plain(who, h0, confidence_set, "DiD wild bootstrap")
    try:
        args, kwargs = record.restore(data)
    except LookupError as exc:
        raise MethodIncompatibility(
            f"{who}: {exc}", recovery_hint="Pass data=<the frame you fitted on>."
        ) from None
    fn = _public(record.function)
    import inspect

    params = list(inspect.signature(fn).parameters)
    kw = dict(zip(params, args))
    kw.update(kwargs)
    treat, time = kw.get("treat"), kw.get("time")
    accepted = {None, "att", "ATT", f"{treat}x{time}", f"{treat}:{time}"}
    accepted |= {f"{time}x{treat}", f"{time}:{treat}"}
    if variable not in accepted:
        raise MethodIncompatibility(
            f"{who}: after sp.did the bootstrap tests the ATT (the "
            f"{treat} x {time} interaction); got variable={variable!r}.",
            recovery_hint="Pass variable=None (or 'att').",
        )
    for k in ("vce", "cluster", "wild_reps", "wild_weight_type", "seed", "alpha"):
        kw.pop(k, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = fn(
            **kw,
            cluster=cluster,
            vce="wild",
            wild_reps=n_boot,
            wild_weight_type=weight_type,
            seed=seed,
            alpha=alpha,
        )
    est = float(getattr(result, "estimate", np.nan))
    if not np.isclose(float(boot.estimate), est, rtol=1e-10, atol=1e-12):
        raise MethodIncompatibility(
            f"{who}: replaying the sp.{record.function} call does not "
            "reproduce this result's estimate.",
            recovery_hint="Pass the frame the model was fitted on as data=, or "
            "refit with vce='wild', cluster=... directly.",
            diagnostics={"original": est, "replay": float(boot.estimate)},
        )
    if "wild" not in str((boot.model_info or {}).get("inference", "")):
        raise MethodIncompatibility(
            f"{who}: sp.{record.function} did not run the 2x2 wild bootstrap "
            "for this design (only the 2x2 regression has one).",
            recovery_hint="Use sp.did(..., method='2x2') or a regression.",
        )
    ci = tuple(float(c) for c in boot.ci)
    se = float(boot.se)
    return {
        "method": "WCR wild cluster bootstrap of the 2x2 DiD regression",
        "beta_hat": float(boot.estimate),
        "se_cluster": se,
        "t_stat": float(boot.estimate) / se if se > 0 else np.nan,
        "p_boot": float(boot.pvalue),
        "ci_boot": ci,
        "n_clusters": int(pd.Series(data[cluster]).nunique()),
        "n_boot_requested": int(n_boot),
        "weight_type": weight_type,
        "h0": 0.0,
    }


def dispatch_kind(result: Any) -> str:
    """'did', 'iv', 'glm' or 'least_squares': which bootstrap ``result`` takes."""
    from ..core._call_record import get_call_record

    record = get_call_record(result)
    if record is not None and record.function in ("did", "did_2x2"):
        return "did"
    di = getattr(result, "data_info", None) or {}
    if di.get("iv") is not None:
        return "iv"
    fl = _family_link(result)
    if fl is not None and fl[1] not in ("", "identity"):
        return "glm"
    return "least_squares"


def dispatch(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    variable: Optional[str],
    *,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    alpha: float,
    h0: float,
    confidence_set: bool,
) -> Optional[Dict[str, Any]]:
    """The bootstrap for a non-least-squares result, or None for least squares."""
    di = getattr(result, "data_info", None) or {}
    common: Dict[str, Any] = dict(n_boot=n_boot, weight_type=weight_type, seed=seed)
    # sp.did (a CausalResult): replay the estimator's own vce='wild'
    from ..core._call_record import get_call_record

    record = get_call_record(result)
    if record is not None and record.function in ("did", "did_2x2"):
        return _did(
            result,
            data,
            cluster,
            variable,
            alpha=alpha,
            h0=h0,
            confidence_set=confidence_set,
            record=record,
            **common,
        )
    if variable is None:
        raise MethodIncompatibility(
            "wild_cluster_boot: pass variable= (the coefficient to test).",
            recovery_hint="E.g. variable='x1'.",
        )
    if di.get("iv") is not None:
        return _iv(
            result,
            data,
            cluster,
            variable,
            alpha=alpha,
            h0=h0,
            confidence_set=confidence_set,
            **common,
        )
    fl = _family_link(result)
    if fl is not None and fl[1] not in ("", "identity"):
        if fl not in _SCORE_FAMILIES or di.get("X") is None or di.get("y") is None:
            raise MethodIncompatibility(
                "wild_cluster_boot: no wild bootstrap is implemented for this "
                f"model (family={fl[0]!r}, link={fl[1]!r}); refitting it by "
                "least squares would bootstrap a different estimator.",
                recovery_hint="Logit, probit and Poisson fits use the score "
                "bootstrap; otherwise see the estimator's own vce= options.",
                diagnostics={"family": fl[0], "link": fl[1]},
            )
        if (getattr(result, "model_info", None) or {}).get("fixed_effects") not in (
            None,
            "None",
            "",
        ):
            raise MethodIncompatibility(
                "wild_cluster_boot: after a GLM with absorbed fixed effects "
                "use the fit's own vce='wild' (sp.fepois / sp.feglm), which "
                "runs the score bootstrap on the absorbed design.",
                recovery_hint="Refit with vce='wild', cluster=...",
            )
        return _score(
            result,
            data,
            cluster,
            variable,
            h0=h0,
            confidence_set=confidence_set,
            family_link=fl,
            **common,
        )
    return None


def absorbed_fe_codes(result: Any, n: int) -> Optional[List[np.ndarray]]:
    """Codes of the fixed effects absorbed into the stored design, on its
    ``n`` rows, or None (no absorbed effects, or rows that do not match)."""
    from ..fixest._fixef import fe_state

    try:
        state = fe_state(result)
    except MethodIncompatibility:
        return None
    if state is None or not state.codes or len(state.codes[0]) != n:
        return None
    return [np.asarray(c, dtype=np.int64) for c in state.codes]


def require_least_squares(
    result: Any, X: np.ndarray, Y: np.ndarray, var_names: List[str]
) -> None:
    """Refuse a WCR bootstrap of a design that is not the fitted model's."""
    params = getattr(result, "params", None)
    if not isinstance(params, pd.Series):
        return  # an internal design handed over by an estimator
    from .jackknife import _match_varname

    matched = [_match_varname(vn, list(params.index)) for vn in var_names]
    ok = X.shape[1] == len(var_names) and None not in matched
    if ok:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        target = params[matched].to_numpy(dtype=float)
        scale = max(1.0, float(np.max(np.abs(target)))) if target.size else 1.0
        ok = bool(np.allclose(beta, target, rtol=0.0, atol=1e-6 * scale))
    if not ok:
        raise MethodIncompatibility(
            "wild_cluster_boot: least squares on the design stored for this "
            "result does not reproduce its coefficients, so the restricted "
            "wild bootstrap would test another estimator's coefficient.",
            recovery_hint="Use it after sp.regress / sp.feols / sp.hdfe_ols "
            "(least squares), sp.ivreg (WRE), or sp.logit / probit / poisson "
            "(score bootstrap); otherwise the estimator's own vce='wild'.",
            diagnostics={"design_columns": list(var_names)},
        )
