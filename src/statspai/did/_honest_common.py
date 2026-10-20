"""Shared pieces of :mod:`statspai.did.honest_did`.

Argument validation, the default ``M`` grids, the window / ``l_vec``
coercions, the provenance attributes :func:`sp.validation_scope` reads back,
the restriction resolver, the event-study extractor and the HonestDiD-order
moment reshuffle. Used by the native and R backends, by
:func:`breakdown_m` and by the moments interface.
"""

from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility


def _require_string(value: Any, *, argument: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodIncompatibility(
            f"`{argument}` must be a non-empty string.",
            recovery_hint=f"Pass a supported string value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    return value


def _require_alpha(alpha: Any) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    out = float(alpha)
    if not np.isfinite(out) or not (0.0 < out < 1.0):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    return out


def _require_int(value: Any, *, argument: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{argument}` must be an integer.",
            recovery_hint=f"Pass an integer relative-time value for `{argument}`.",
            diagnostics={"argument": argument, "value": value},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{argument}` must be an integer.",
            recovery_hint=f"Pass an integer relative-time value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        ) from exc


#: ``HonestDiD::createSensitivityResults_relativeMagnitudes`` default
#: (0.2.8): ``Mbarvec = seq(from = 0, to = 2, length.out = 10)``.
_RM_DEFAULT_GRID = tuple(float(m) for m in np.linspace(0.0, 2.0, 10))


def _coerce_m_grid(
    m_grid: Optional[Sequence[float]],
    *,
    se_hat: float,
    relative_magnitude: bool = False,
) -> List[float]:
    if m_grid is None:
        # M-bar under relative magnitudes is a unitless ratio to the
        # largest pre-period violation; a grid in SE units (0.01-0.02 on a
        # log outcome) would evaluate only near-zero ratios and report the
        # effect as robust everywhere.
        if relative_magnitude:
            return list(_RM_DEFAULT_GRID)
        sigma = float(se_hat)
        return [0.0, 0.5 * sigma, sigma, 1.5 * sigma, 2.0 * sigma, 3.0 * sigma]
    try:
        out = [float(m) for m in m_grid]
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "`m_grid` must be a sequence of finite non-negative numbers.",
            recovery_hint="Pass m_grid=[0, 0.5, 1.0] or leave it as None.",
            diagnostics={"argument": "m_grid", "type": type(m_grid).__name__},
        ) from exc
    if not out:
        raise DataInsufficient(
            "`m_grid` must contain at least one value.",
            recovery_hint="Pass at least one M value or leave m_grid as None.",
            diagnostics={"argument": "m_grid"},
        )
    if any((not np.isfinite(m)) or m < 0 for m in out):
        raise MethodIncompatibility(
            "`m_grid` must be a sequence of finite non-negative numbers.",
            recovery_hint="Remove NaN/Inf/negative M values from m_grid.",
            diagnostics={"argument": "m_grid", "values": out},
        )
    return out


def _window_filter(moments: Optional[tuple], window: Optional[tuple]):
    """Restrict ``(beta, sigma, times)`` to event times inside ``window``."""
    if moments is None or window is None:
        return moments
    lo, hi = window
    beta, sigma, times = moments
    keep = (times >= lo) & (times <= hi)
    if not keep.any():
        return None
    return beta[keep], sigma[np.ix_(keep, keep)], times[keep]


def _coerce_window(window: Any) -> Optional[tuple]:
    if window is None:
        return None
    try:
        lo, hi = (float(v) for v in window)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "window must be a (lo, hi) pair of event times.",
            recovery_hint="e.g. window=(-5, 5) as Stata estat event, window(-5 5).",
            diagnostics={"window": repr(window)},
        ) from exc
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= 0 or hi < 0:
        raise MethodIncompatibility(
            f"window={window!r} must contain both leads and horizons "
            "(lo < 0 <= hi).",
            recovery_hint="e.g. window=(-5, 5).",
            diagnostics={"window": repr(window)},
        )
    return lo, hi


def _coerce_l_vec(l_vec: Any, post_times: np.ndarray) -> np.ndarray:
    """Target weights over the post-treatment event times (HonestDiD
    ``l_vec``); ``'average'`` puts ``1 / n_post`` on each."""
    n_post = len(post_times)
    if isinstance(l_vec, str):
        if l_vec.strip().lower() in {"average", "mean", "avg"}:
            return np.full(n_post, 1.0 / n_post)
        raise MethodIncompatibility(
            f"l_vec={l_vec!r} is not recognised; pass 'average' or weights.",
            recovery_hint="e.g. l_vec='average' or l_vec=[0, 0, 1].",
            diagnostics={"l_vec": l_vec},
        )
    lv = np.asarray(l_vec, dtype=float).ravel()
    if lv.size != n_post or not np.all(np.isfinite(lv)) or not np.any(lv != 0):
        raise MethodIncompatibility(
            f"l_vec must hold {n_post} finite weights, not all zero, one per "
            f"post-treatment event time {[int(t) for t in post_times]}.",
            recovery_hint="Match the post-treatment periods of the (windowed) "
            "event study, or pass l_vec='average'.",
            diagnostics={"l_vec": lv.tolist(), "n_post": n_post},
        )
    return lv


_MOMENTS_METHOD = "Event study (user-supplied moments)"


def _target_kind(l_vec: Any) -> str:
    """``'single'`` / ``'average'`` / ``'weights'`` for a target as given."""
    if l_vec is None:
        return "single"
    if isinstance(l_vec, str):
        return "average"
    lv = np.asarray(l_vec, dtype=float).ravel()
    nonzero = np.flatnonzero(lv)
    if nonzero.size == 1 and lv[nonzero[0]] == 1.0:
        return "single"
    if np.allclose(lv, lv[0]) and np.isclose(lv.sum(), 1.0):
        return "average"
    return "weights"


#: (token in the result's method / estimator label, estimator) -- the event
#: studies sp.event_study_vcov reads, named as sp.validation_scope names them.
#: Kept here rather than imported from validation_scope so that editing the
#: evidence map does not touch the Track A estimation path of this module.
_SOURCE_TOKENS = (
    ("callaway and sant'anna", "callaway_santanna"),
    ("sun and abraham", "sun_abraham"),
    ("sun-abraham", "sun_abraham"),
    ("ols event study (twfe)", "event_study"),
    ("etwfe", "etwfe"),
    ("gardner", "gardner_did"),
    ("imputation estimator", "did_imputation"),
    ("bjs imputation", "did_imputation"),
)


def _vcov_source(result: Any) -> str:
    """The estimator whose event study (and covariance) is being analysed."""
    method = str(getattr(result, "method", None) or "")
    if method == _MOMENTS_METHOD:
        return "moments"
    info = getattr(result, "model_info", None) or {}
    label = " ".join(
        str(v or "").lower()
        for v in (method, info.get("estimator_label"), info.get("estimator"))
    )
    for token, name in _SOURCE_TOKENS:
        if token in label:
            return name
    return "supplied" if info.get("event_study_vcov") is not None else "other"


_INTERVAL_OF_METHOD = {
    "FLCI": "flci",
    "CONDITIONAL": "arp_conditional",
    "C-LF": "arp_c_lf",
    "CLF": "arp_c_lf",
    "C-F": "arp_c_f",
    "CF": "arp_c_f",
}


def _scope_attrs(
    out: pd.DataFrame,
    *,
    method: str,
    alpha: float,
    l_vec: Any,
    window: Optional[tuple],
    code_path: str,
    source: str,
    honestdid_method: Optional[str],
    custom_grid: bool,
    shape: str = "none",
) -> Dict[str, Any]:
    """The configuration of a sensitivity table, for ``sp.validation_scope``."""
    restriction = method
    interval = out.attrs.get("interval")
    if interval is None:  # backend='r': the method HonestDiD ran
        key = str(honestdid_method).upper().replace("_", "-")
        interval = _INTERVAL_OF_METHOD.get(key, key.lower())
    if interval in ("flci", "worst_case_bias"):
        rm_grid = "n/a"
    elif custom_grid:
        rm_grid = "user"
    elif out.attrs.get("grid_extended_at"):
        rm_grid = "extended"
    else:
        rm_grid = "honestdid"
    return {
        "function": "honest_did",
        "restriction": restriction,
        "shape": shape,
        "interval": interval,
        "code_path": code_path,
        "target": _target_kind(l_vec),
        "window": "full" if window is None else "restricted",
        "alpha": float(alpha),
        # 'zero_only' / 'zero' (with positive M too) / 'positive'
        "m_grid": (
            "zero_only"
            if bool((out["M"] == 0).all())
            else ("zero" if bool((out["M"] == 0).any()) else "positive")
        ),
        "rm_grid": rm_grid,
        "source": source,
    }


_FAMILY_OF_METHOD = {
    "smoothness": "smoothness",
    "sd": "smoothness",
    "relative_magnitude": "rm",
    "relative_magnitudes": "rm",
    "rm": "rm",
    "smoothness_relative_magnitude": "sdrm",
    "smoothness_relative_magnitudes": "sdrm",
    "sdrm": "sdrm",
}
_RESTRICTION_LABEL = {
    "smoothness": "smoothness",
    "rm": "relative_magnitude",
    "sdrm": "smoothness_relative_magnitude",
}
_HM_CANONICAL = {
    "c-lf": "C-LF",
    "clf": "C-LF",
    "lf": "C-LF",
    "conditional": "Conditional",
    "arp": "Conditional",
    "flci": "FLCI",
    "c-f": "C-F",
    "cf": "C-F",
}
_HYBRID_OF_HM = {"Conditional": "ARP", "C-LF": "LF", "C-F": "FLCI"}


def _resolve_restriction(
    method: str,
    honestdid_method: Optional[str],
    bias_direction: Optional[str],
    monotonicity_direction: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """``(family, honestdid_method, bias, mono)`` with HonestDiD's defaults.

    ``family`` is ``'smoothness'`` / ``'rm'`` / ``'sdrm'``; the method is the
    canonical HonestDiD spelling.
    """
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    family = _FAMILY_OF_METHOD.get(key)
    if family is None:
        raise MethodIncompatibility(
            "method must be 'smoothness', 'relative_magnitude' or "
            f"'smoothness_relative_magnitude', got '{method}'",
            recovery_hint="Use method='smoothness', 'relative_magnitude' or "
            "'smoothness_relative_magnitude' (HonestDiD's deviation from a "
            "linear trend).",
            diagnostics={"method": method},
        )
    bias = mono = None
    if bias_direction is not None:
        bias = _require_string(bias_direction, argument="bias_direction").lower()
        if bias not in {"positive", "negative"}:
            raise MethodIncompatibility(
                f"bias_direction must be 'positive' or 'negative', got "
                f"{bias_direction!r}.",
                recovery_hint="'positive' restricts delta_post >= 0, "
                "'negative' delta_post <= 0.",
                diagnostics={"bias_direction": bias_direction},
            )
    if monotonicity_direction is not None:
        mono = _require_string(
            monotonicity_direction, argument="monotonicity_direction"
        ).lower()
        if mono not in {"increasing", "decreasing"}:
            raise MethodIncompatibility(
                "monotonicity_direction must be 'increasing' or 'decreasing', "
                f"got {monotonicity_direction!r}.",
                recovery_hint="Pass monotonicity_direction='increasing' or "
                "'decreasing'.",
                diagnostics={"monotonicity_direction": monotonicity_direction},
            )
    if bias is not None and mono is not None:
        raise MethodIncompatibility(
            "Pass either a sign restriction (bias_direction) or a shape "
            "restriction (monotonicity_direction), not both.",
            recovery_hint="HonestDiD offers one of the two per sensitivity "
            "analysis; drop one of them.",
            diagnostics={"bias_direction": bias, "monotonicity_direction": mono},
        )
    if honestdid_method is None:
        if family == "smoothness":
            hm = "FLCI" if bias is None and mono is None else "C-F"
        else:
            hm = "C-LF"
    else:
        hm_key = _require_string(honestdid_method, argument="honestdid_method")
        hm = _HM_CANONICAL.get(hm_key.strip().lower().replace("_", "-"))
        if hm is None:
            raise MethodIncompatibility(
                "honestdid_method must be one of 'C-LF', 'Conditional', "
                "'FLCI', or 'C-F'.",
                recovery_hint="Use an HonestDiD inference method supported by "
                "the selected restriction.",
                diagnostics={"honestdid_method": honestdid_method},
            )
    if family != "smoothness" and hm not in {"C-LF", "Conditional"}:
        raise MethodIncompatibility(
            f"honestdid_method={hm!r} is not available for "
            f"method={method!r}; the relative-magnitude restrictions are a "
            "union of polyhedra, which the FLCI and its hybrid do not cover.",
            recovery_hint="Use honestdid_method='C-LF' (the default) or "
            "'Conditional'.",
            diagnostics={"method": method, "honestdid_method": hm},
        )
    return family, hm, bias, mono


def _rm_inputs(
    moments: Optional[tuple], e: int, l_vec: Optional[np.ndarray] = None
) -> Optional[tuple]:
    """(betahat, sigma, n_pre, n_post, l_post) in HonestDiD order, or None."""
    if moments is None:
        return None
    beta, sigma, times = moments
    post_mask = times >= 0
    n_pre = int((~post_mask).sum())
    post_times = times[post_mask]
    if n_pre < 1 or (l_vec is None and e not in set(post_times.tolist())):
        return None
    if l_vec is not None and len(l_vec) != int(post_mask.sum()):
        return None
    order = np.concatenate([np.where(~post_mask)[0], np.where(post_mask)[0]])
    return (
        beta[order],
        sigma[np.ix_(order, order)],
        n_pre,
        int(post_mask.sum()),
        (post_times == e).astype(float) if l_vec is None else np.asarray(l_vec),
    )


# ======================================================================
# Polymorphic event-study extractor
# ======================================================================


def _extract_event_study(result: CausalResult) -> pd.DataFrame:
    """Locate the event-study table inside a CausalResult.

    Accepts three flavours, all of which now surface the same shape:

    1. ``callaway_santanna()`` / ``sun_abraham()`` → event study lives in
       ``result.model_info['event_study']`` (legacy).
    2. ``aggte(cs, type='dynamic')`` → event study **is** the ``detail``
       frame (has ``relative_time`` + ``att`` + ``se``).
    3. ``did_multiplegt()`` → event study in ``model_info['event_study']``
       but placebos carry negative ``relative_time`` and dynamics
       non-negative ``relative_time`` — already compatible.

    Raises
    ------
    ValueError
        If no event study can be found.
    """
    info = result.model_info or {}
    es = cast(Optional[pd.DataFrame], info.get("event_study"))
    if isinstance(es, dict):
        # sp.gardner_did keeps its event study as a 'D_k+h'-keyed dict.
        from .es_inference import _es_frame

        es = _es_frame(result)
    if es is None and getattr(result, "detail", None) is not None:
        det = result.detail
        if isinstance(det, pd.DataFrame) and {"relative_time", "att", "se"}.issubset(
            det.columns
        ):
            es = det
    if es is None:
        raise MethodIncompatibility(
            "Result does not expose an event-study table.  Supported "
            "inputs: callaway_santanna(), sun_abraham(), did_multiplegt(), "
            "or aggte(result, type='dynamic').",
            recovery_hint=(
                "Pass a DID/event-study result with relative_time, att, and se "
                "columns."
            ),
            diagnostics={"result_type": type(result).__name__},
        )
    # Defensive copy — callers mutate / filter it.
    es = es.copy()
    # If aggte produced uniform-band columns but no explicit 'type', ignore.
    required = {"relative_time", "att", "se"}
    missing = required - set(es.columns)
    if missing:
        raise MethodIncompatibility(
            f"Event-study table is missing required columns: {missing}",
            recovery_hint=(
                "Provide an event-study table with relative_time, att, and se "
                "columns."
            ),
            diagnostics={"missing_columns": sorted(missing)},
        )
    return es
