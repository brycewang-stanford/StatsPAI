"""
statspai.iv — the unified Instrumental Variables namespace.

The goal of this subpackage is to be the single entry point for every
IV-flavoured workflow in StatsPAI, regardless of which sub-module the
underlying implementation lives in:

- Core point estimators (2SLS, LIML, Fuller, GMM, JIVE) live in
  :mod:`statspai.regression.iv` — re-exported as ``sp.iv.iv``,
  ``sp.iv.ivreg`` and ``sp.iv.IVRegression``.
- JIVE variants (UJIVE, IJIVE, RJIVE) live here in
  :mod:`statspai.iv.jive_variants`.
- Weak-identification diagnostics (Olea-Pflueger effective F, Lee-McCrary
  tF, Anderson-Rubin CI) live in :mod:`statspai.diagnostics.weak_iv`
  — re-exported as ``sp.iv.effective_f_test`` etc.
- New diagnostics introduced in this subpackage:
  ``kleibergen_paap_rk``, ``sanderson_windmeijer``, ``conditional_lr_test``.
- Plausibly exogenous sensitivity (Conley-Hansen-Rossi 2012):
  ``plausibly_exogenous_uci``, ``plausibly_exogenous_ltz``.
- Marginal Treatment Effects (Brinch-Mogstad-Wiswall 2017):
  ``mte``.
- Shift-share IV (Bartik + Adão-Kolesár-Morales correction) is re-exported
  as ``sp.iv.bartik`` / ``sp.iv.shift_share_se``.
- DeepIV (Hartford et al. 2017) is re-exported as ``sp.iv.deepiv``.

A thin :func:`fit` dispatcher ties everything together and auto-runs an
expanded diagnostic panel (first-stage F, MOP effective F, KP rk,
SW per-endog F, Hansen J, AR Wald).

Examples
--------
>>> import statspai as sp
>>> # Standard 2SLS with a rich diagnostic panel
>>> res = sp.iv.fit("y ~ (d ~ z1 + z2) + x1", data=df)
>>> print(res.summary())
>>> print(res.diagnostics)  # includes MOP F, KP rk, SW, AR CI

>>> # Sensitivity to exclusion-restriction violations
>>> chr = sp.iv.plausibly_exogenous_ltz(
...     y="y", endog="d", instruments=["z1", "z2"],
...     gamma_mean=0.0, gamma_var=0.01, data=df,
... )

>>> # Marginal treatment effects
>>> m = sp.iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"], data=df)
>>> m.mte_curve.plot(x="u", y="mte")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# ─── Core estimators (re-exports) ───────────────────────────────────────
from ..regression.iv import iv, ivreg, IVRegression
from ..regression.advanced_iv import liml, jive as jive_legacy, lasso_iv

# ─── Weak-identification diagnostics ────────────────────────────────────
from ..diagnostics.weak_iv import (
    anderson_rubin_test,
    effective_f_test,
    tF_critical_value,
)
from .weak_identification import (
    kleibergen_paap_rk,
    sanderson_windmeijer,
    conditional_lr_test,
    KleibergenPaapResult,
    SandersonWindmeijerResult,
    CLRResult,
)

# ─── Plausibly exogenous ────────────────────────────────────────────────
from .plausibly_exogenous import (
    plausibly_exogenous_uci,
    plausibly_exogenous_ltz,
    PlausiblyExogenousResult,
)

# ─── JIVE variants ──────────────────────────────────────────────────────
from .jive_variants import jive1, ujive, ijive, rjive, JIVEResult

# ─── Marginal Treatment Effects ─────────────────────────────────────────
from .mte import mte, MTEResult

# ─── MST sharp identified bounds (LP-based) ─────────────────────────────
from .ivmte_lp import ivmte_bounds, IVMTEBounds

# ─── Weak-IV-robust CIs by grid inversion ───────────────────────────────
from .weak_iv_ci import (
    anderson_rubin_ci,
    conditional_lr_ci,
    k_test_ci,
    WeakIVConfidenceSet,
)

# ─── Post-Lasso IV (Belloni-Chen-Chernozhukov-Hansen 2012) ──────────────
from .post_lasso import (
    bch_post_lasso_iv,
    bch_lambda,
    bch_selected,
    PostLassoResult,
)

# ─── Plot module (matplotlib imported lazily) ───────────────────────────
from . import plot  # noqa: F401

# ─── Bayesian IV (Chernozhukov-Hong 2003) ────────────────────────────────
from .bayesian_iv import bayesian_iv, BayesianIVResult

# ─── Non-parametric IV (Newey-Powell 2003) ───────────────────────────────
from .npiv import npiv, NPIVResult

# ─── Many-weak-instrument inference (Mikusheva-Sun 2024) ────────────────
from .many_weak import jive as jive_mw, many_weak_ar, ManyWeakIVResult

# ─── v0.10 IV frontier: Kernel IV / Continuous LATE / IVDML ─────────────
from .kernel_iv import kernel_iv, KernelIVResult
from .continuous_late import continuous_iv_late, ContinuousLATEResult
from .ivdml import ivdml, IVDMLResult

# ─── Shift-share / DeepIV re-exports ────────────────────────────────────
try:
    from ..bartik import bartik, shift_share_se, BartikIV, ssaggregate
except Exception:  # pragma: no cover
    bartik = shift_share_se = BartikIV = ssaggregate = None

try:
    from ..deepiv import deepiv, DeepIV
except Exception:  # pragma: no cover
    deepiv = DeepIV = None


# ═══════════════════════════════════════════════════════════════════════
#  Unified dispatcher: sp.iv.fit(...)
# ═══════════════════════════════════════════════════════════════════════

_METHOD_ALIASES = {
    "2sls": "2sls", "tsls": "2sls", "iv": "2sls",
    "liml": "liml", "fuller": "fuller",
    "gmm": "gmm",
    "jive": "jive", "jive1": "jive",
    "ujive": "ujive", "ijive": "ijive", "rjive": "rjive",
    "mte": "mte",
    "deepiv": "deepiv", "deep": "deepiv",
    "shift_share": "shift_share", "bartik": "shift_share",
}


def fit(
    formula=None,
    data=None,
    *,
    method: str = "2sls",
    y=None,
    endog=None,
    instruments=None,
    exog=None,
    robust: str = "nonrobust",
    cluster=None,
    augmented_diagnostics: bool = True,
    **kwargs,
):
    """
    Unified IV dispatcher.

    Parameters
    ----------
    formula : str, optional
        ``"y ~ (endog ~ z1 + z2) + x1 + x2"`` Patsy-style IV formula used
        by 2SLS/LIML/Fuller/GMM/JIVE paths.
    data : DataFrame, optional.
    method : str, default '2sls'
        One of 2sls, liml, fuller, gmm, jive, ujive, ijive, rjive,
        mte, deepiv, shift_share.
    y, endog, instruments, exog : arrays or column-name lists
        Alternative to ``formula`` — required for MTE / JIVE variants
        / ShiftShare / DeepIV which do not use the formula parser.
    robust : str, default 'nonrobust'
        Only applies to formula methods.
    cluster : optional cluster ID column name.
    augmented_diagnostics : bool, default True
        Attach Kleibergen-Paap rk, Sanderson-Windmeijer, Olea-Pflueger
        effective F, and Anderson-Rubin CI to the returned result's
        ``diagnostics`` dict when the method produces an EconometricResults.
    **kwargs
        Method-specific options (e.g. ``fuller_alpha``, ``poly_degree``).

    Returns
    -------
    EconometricResults | JIVEResult | MTEResult | ...
    """
    m = _METHOD_ALIASES.get(method.lower())
    if m is None:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"{sorted(set(_METHOD_ALIASES.values()))}"
        )

    if m in ("2sls", "liml", "fuller", "gmm", "jive"):
        if formula is None or data is None:
            raise ValueError(f"method='{method}' requires formula + data.")
        model = IVRegression(
            formula=formula, data=data, method=m,
            fuller_alpha=kwargs.get("fuller_alpha", 1.0),
        )
        result = model.fit(robust=robust, cluster=cluster)
        if augmented_diagnostics:
            _attach_augmented_diagnostics(model, result, kwargs)
        return result

    if m in ("ujive", "ijive", "rjive"):
        if formula is not None and data is not None:
            y_, endog_, instruments_, exog_ = _formula_to_parts(formula, data)
        else:
            y_, endog_, instruments_, exog_ = y, endog, instruments, exog
        fn = {"ujive": ujive, "ijive": ijive, "rjive": rjive}[m]
        return fn(y=y_, endog=endog_, instruments=instruments_, exog=exog_,
                  data=data, **kwargs)

    if m == "mte":
        if y is None or endog is None or instruments is None:
            raise ValueError("method='mte' requires y, endog, instruments.")
        return mte(
            y=y, treatment=endog, instruments=instruments, exog=exog, data=data,
            **kwargs,
        )

    if m == "deepiv":
        if deepiv is None:
            raise ImportError("DeepIV requires torch; install torch to use it.")
        return deepiv(
            y=y, treatment=endog, instruments=instruments, exog=exog,
            data=data, **kwargs,
        )

    if m == "shift_share":
        if bartik is None:
            raise ImportError("shift_share/bartik unavailable.")
        return bartik(y=y, shares=kwargs.pop("shares"),
                      shocks=kwargs.pop("shocks"), data=data, **kwargs)

    raise AssertionError(f"Unreachable: method={m}")  # pragma: no cover


def _formula_to_parts(formula: str, data):
    from ..core.utils import parse_formula
    parsed = parse_formula(formula)
    return (
        parsed["dependent"],
        parsed["endogenous"],
        parsed["instruments"],
        parsed.get("exogenous") or None,
    )


def _attach_augmented_diagnostics(model, result, opts: Dict[str, Any]):
    """Add KP rk, SW, MOP effective F to the EconometricResults diagnostics."""
    try:
        D = model.X_endog
        Z = model.Z
        W = model.X_exog

        kp = kleibergen_paap_rk(
            endog=D, instruments=Z, exog=W[:, 1:] if W.shape[1] > 1 else None,
            add_const=W.shape[1] >= 1 and np.allclose(W[:, 0], 1.0) if W.shape[1] else True,
            cov_type="robust",
        )
        result.diagnostics["KP rk LM"] = kp.rk_lm
        result.diagnostics["KP rk LM p-value"] = kp.rk_lm_pvalue
        result.diagnostics["KP rk Wald F"] = kp.rk_f

        if D.shape[1] >= 2:
            sw = sanderson_windmeijer(
                endog=D, instruments=Z,
                exog=W[:, 1:] if W.shape[1] > 1 else None,
                add_const=False,  # already handled above in W
                endog_names=getattr(model, "_endog_names", None),
            )
            for name, f in sw.sw_f.items():
                result.diagnostics[f"SW conditional F ({name})"] = f

        # Olea-Pflueger effective F (single endogenous variable case)
        if D.shape[1] == 1 and hasattr(model, "data") and model.data is not None:
            try:
                ep = effective_f_test(
                    data=getattr(model, "_clean_data", model.data),
                    endog=model._endog_names[0],
                    instruments=list(model._instrument_names),
                    exog=[e for e in model._exog_names if e != "Intercept"] or None,
                )
                if isinstance(ep, dict):
                    stat = ep.get("F_eff") or ep.get("statistic") or ep.get("effective_F")
                else:
                    stat = getattr(ep, "F_eff", None) or getattr(ep, "statistic", None)
                if stat is not None:
                    result.diagnostics["Olea-Pflueger effective F"] = float(stat)
            except Exception as e:
                result.diagnostics["OP effective F error"] = str(e)
    except Exception as e:  # pragma: no cover
        # Augmented diagnostics are optional; never crash the estimator.
        result.diagnostics["augmented_diagnostics_error"] = str(e)




__all__ = [
    # dispatcher
    "fit",
    # core estimators
    "iv", "ivreg", "IVRegression", "liml", "jive_legacy", "lasso_iv",
    # JIVE variants
    "jive1", "ujive", "ijive", "rjive", "JIVEResult",
    # weak-ID diagnostics
    "kleibergen_paap_rk", "sanderson_windmeijer", "conditional_lr_test",
    "anderson_rubin_test", "effective_f_test", "tF_critical_value",
    "KleibergenPaapResult", "SandersonWindmeijerResult", "CLRResult",
    # plausibly exogenous
    "plausibly_exogenous_uci", "plausibly_exogenous_ltz", "PlausiblyExogenousResult",
    # MTE
    "mte", "MTEResult",
    "ivmte_bounds", "IVMTEBounds",
    # Post-Lasso BCH
    "bch_post_lasso_iv", "bch_lambda", "bch_selected", "PostLassoResult",
    # Weak-IV-robust confidence sets
    "anderson_rubin_ci", "conditional_lr_ci", "k_test_ci",
    "WeakIVConfidenceSet",
    # Bayesian IV
    "bayesian_iv", "BayesianIVResult",
    # NPIV
    "npiv", "NPIVResult",
    # re-exports
    "bartik", "shift_share_se", "BartikIV", "ssaggregate",
    "deepiv", "DeepIV",
]
