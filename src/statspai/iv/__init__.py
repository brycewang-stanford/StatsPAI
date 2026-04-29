"""
statspai.iv — the unified Instrumental Variables namespace.

The goal of this subpackage is to be the single entry point for every
IV-flavoured workflow in StatsPAI, regardless of which sub-module the
underlying implementation lives in.

The subpackage itself is **callable**::

    sp.iv("y ~ (d ~ z) + x", data=df)                   # 2SLS (default)
    sp.iv("y ~ (d ~ z) + x", data=df, method="liml")    # LIML
    sp.iv(method="kernel", y=..., endog=..., instruments=..., data=df)
    sp.iv.fit(...)         # equivalent (fit = _dispatch alias)
    sp.iv.kernel_iv(...)   # individual estimators still reachable

That callable is provided by a tiny ``ModuleType`` subclass installed via
``sys.modules[__name__].__class__`` (the standard PEP 562-style trick).

Sub-method coverage (``method=`` keyword, all aliases lowercased):

- **K-class formula path** (``regression.iv``):
  ``2sls`` / ``tsls`` / ``iv``,  ``liml``,  ``fuller``,  ``gmm``,  ``jive``.
- **Modern JIVE variants** (``iv.jive_variants``):
  ``jive1``, ``ujive``, ``ijive``, ``rjive``.
- **Many-weak** (``iv.many_weak``):
  ``jive_mw``, ``many_weak_ar``.
- **Lasso / post-Lasso**:
  ``lasso`` (``regression.advanced_iv.lasso_iv``),
  ``post_lasso`` / ``bch`` (``iv.post_lasso.bch_post_lasso_iv``).
- **ML / nonparametric**:
  ``kernel`` (``iv.kernel_iv``), ``npiv`` (``iv.npiv``),
  ``ivdml`` (``iv.ivdml``), ``deepiv`` (``deepiv.deepiv``, optional).
- **Bayesian** (``iv.bayesian_iv``):  ``bayes`` / ``bayesian``.
- **LATE / MTE**:
  ``continuous_late`` (``iv.continuous_late``),
  ``mte`` (``iv.mte``),
  ``ivmte_bounds`` (``iv.ivmte_lp``).
- **Quantile IV** (``regression.iv_quantile``):  ``ivqreg`` / ``quantile``.
- **Plausibly exogenous sensitivity** (``iv.plausibly_exogenous``):
  ``plausibly_exog_uci`` / ``plausibly_exog_ltz``.
- **Shift-share** (``bartik``):  ``shift_share`` / ``bartik``.

Diagnostics (``anderson_rubin_test``, ``effective_f_test``,
``kleibergen_paap_rk``, ``sanderson_windmeijer``, ``conditional_lr_test``)
remain standalone — they are not estimators and intentionally do not show
up in the ``method=`` table.

Examples
--------
>>> import statspai as sp
>>> # Standard 2SLS with a rich diagnostic panel
>>> res = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df)
>>> print(res.summary())
>>> print(res.diagnostics)  # MOP F, KP rk, SW, AR CI

>>> # Sensitivity to exclusion-restriction violations
>>> chr = sp.iv(
...     method="plausibly_exog_ltz",
...     y="y", endog="d", instruments=["z1", "z2"],
...     gamma_mean=0.0, gamma_var=0.01, data=df,
... )

>>> # Marginal treatment effects
>>> m = sp.iv(method="mte",
...          y="y", endog="d", instruments=["z"], exog=["x"], data=df)
"""

from __future__ import annotations

import sys
from types import ModuleType
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
#  Unified dispatcher — sp.iv(..., method=...)
# ═══════════════════════════════════════════════════════════════════════

# Canonical method name → group it lives in.  Aliases collapse to the
# canonical name during normalisation.
_METHOD_ALIASES: Dict[str, str] = {
    # K-class formula path
    "2sls": "2sls", "tsls": "2sls", "iv": "2sls",
    "liml": "liml",
    "fuller": "fuller",
    "gmm": "gmm",
    "jive": "jive",  # AIK 1999 within K-class

    # Modern JIVE variants
    "jive1": "jive1",
    "ujive": "ujive",
    "ijive": "ijive",
    "rjive": "rjive",

    # Many-weak
    "jive_mw": "jive_mw", "ms_jive": "jive_mw",
    "many_weak_ar": "many_weak_ar", "many_weak": "many_weak_ar",

    # Lasso / post-Lasso
    "lasso": "lasso", "lasso_iv": "lasso",
    "post_lasso": "post_lasso", "bch": "post_lasso", "bch_lasso": "post_lasso",

    # ML / nonparametric
    "kernel": "kernel", "kernel_iv": "kernel",
    "npiv": "npiv", "newey_powell": "npiv",
    "ivdml": "ivdml", "dml": "ivdml",
    "deepiv": "deepiv", "deep": "deepiv",

    # Bayesian
    "bayes": "bayes", "bayesian": "bayes", "bayes_iv": "bayes",
    "bayesian_iv": "bayes",

    # LATE / MTE
    "continuous_late": "continuous_late", "continuous": "continuous_late",
    "mte": "mte",
    "ivmte_bounds": "ivmte_bounds", "ivmte": "ivmte_bounds",
    "mst_bounds": "ivmte_bounds",

    # Quantile IV
    "ivqreg": "ivqreg", "quantile": "ivqreg",

    # Plausibly exogenous
    "plausibly_exog_uci": "plausibly_exog_uci", "uci": "plausibly_exog_uci",
    "plausibly_exog_ltz": "plausibly_exog_ltz",
    "plausibly_exog": "plausibly_exog_ltz",
    "ltz": "plausibly_exog_ltz",

    # Shift-share
    "shift_share": "shift_share", "bartik": "shift_share",
}

# Methods that consume a Patsy-style ``"y ~ (endog ~ z) + x"`` formula.
_FORMULA_METHODS = frozenset({"2sls", "liml", "fuller", "gmm", "jive"})


def _dispatch(
    formula: Optional[str] = None,
    data: Any = None,
    *,
    method: str = "2sls",
    augmented_diagnostics: bool = True,
    **kwargs: Any,
):
    """
    Unified IV dispatcher.

    Parameters
    ----------
    formula : str, optional
        ``"y ~ (endog ~ z1 + z2) + x1"``.  Required for k-class methods
        (2sls/liml/fuller/gmm/jive) when ``y/endog/instruments/exog``
        kwargs are not given.  Ignored by methods that only accept the
        explicit-args style.
    data : DataFrame, optional.
    method : str, default ``'2sls'``
        See :data:`_METHOD_ALIASES` for the full table.  Aliases are
        case-insensitive and ``-``/``_`` are interchangeable.
    augmented_diagnostics : bool, default True
        When the chosen method produces an :class:`EconometricResults`,
        attach Kleibergen-Paap rk, Sanderson-Windmeijer per-endog F, and
        Olea-Pflueger effective F into ``result.diagnostics``.
    **kwargs
        Method-specific options (e.g. ``fuller_alpha=4`` for Fuller,
        ``bandwidth=`` for Kernel IV, ``n_folds=`` for IVDML).

    Returns
    -------
    Result object (type depends on ``method``).

    Raises
    ------
    ValueError
        Unknown ``method``.
    ImportError
        Optional dependency missing (e.g. torch for ``deepiv``).
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    key = method.lower().strip().replace("-", "_")
    canon = _METHOD_ALIASES.get(key)
    if canon is None:
        # NB: keep the prefix "Unknown method" — older tests grep for it.
        raise ValueError(
            f"Unknown method '{method}' for sp.iv. "
            f"Choose from: {sorted(set(_METHOD_ALIASES.values()))}"
        )

    # ── 1. K-class formula path ──────────────────────────────────────
    if canon in _FORMULA_METHODS:
        if formula is None or data is None:
            raise ValueError(
                f"method='{method}' requires `formula` and `data` "
                f"(e.g. sp.iv('y ~ (d ~ z) + x', data=df, method='{method}'))."
            )
        fuller_alpha = kwargs.pop("fuller_alpha", 1.0)
        robust = kwargs.pop("robust", "nonrobust")
        cluster = kwargs.pop("cluster", None)
        model = IVRegression(
            formula=formula, data=data, method=canon,
            fuller_alpha=fuller_alpha,
        )
        result = model.fit(robust=robust, cluster=cluster, **kwargs)
        if augmented_diagnostics:
            _attach_augmented_diagnostics(model, result)
        return result

    # ── 2. Modern JIVE variants (jive1/ujive/ijive/rjive) ────────────
    if canon in {"jive1", "ujive", "ijive", "rjive"}:
        y_, endog_, instruments_, exog_ = _resolve_iv_args(
            formula, data, kwargs, allow_formula=True,
        )
        fn = {"jive1": jive1, "ujive": ujive, "ijive": ijive, "rjive": rjive}[canon]
        return fn(
            y=y_, endog=endog_, instruments=instruments_, exog=exog_,
            data=data, **kwargs,
        )

    # ── 3. Many-weak inference ───────────────────────────────────────
    if canon == "jive_mw":
        return jive_mw(data=data, **kwargs)
    if canon == "many_weak_ar":
        return many_weak_ar(data=data, **kwargs)

    # ── 4. Lasso family ──────────────────────────────────────────────
    if canon == "lasso":
        if formula is not None and data is not None:
            kwargs.setdefault("formula", formula)
            kwargs.setdefault("data", data)
        elif data is not None:
            kwargs.setdefault("data", data)
        return lasso_iv(**kwargs)
    if canon == "post_lasso":
        return bch_post_lasso_iv(data=data, **kwargs)

    # ── 5. ML / nonparametric ────────────────────────────────────────
    if canon == "kernel":
        # kernel_iv uses singular ``treat``/``instrument`` — translate
        # and unwrap a singleton list of instruments.
        _rename(kwargs, {"endog": "treat", "treatment": "treat",
                         "instruments": "instrument"})
        _unwrap_singleton_str(kwargs, "instrument")
        return kernel_iv(data=data, **kwargs)
    if canon == "npiv":
        return npiv(data=data, **kwargs)
    if canon == "ivdml":
        # ivdml uses ``treat`` for endog and ``covariates`` for exog.
        _rename(kwargs, {"endog": "treat", "treatment": "treat",
                         "exog": "covariates"})
        return ivdml(data=data, **kwargs)
    if canon == "deepiv":
        if deepiv is None:
            raise ImportError(
                "method='deepiv' requires the optional deepiv extras. "
                "Install with: pip install 'statspai[deepiv]'."
            )
        return deepiv(data=data, **kwargs)

    # ── 6. Bayesian ──────────────────────────────────────────────────
    if canon == "bayes":
        return bayesian_iv(data=data, **kwargs)

    # ── 7. LATE / MTE ────────────────────────────────────────────────
    if canon == "continuous_late":
        _rename(kwargs, {"endog": "treat", "treatment": "treat",
                         "instruments": "instrument"})
        _unwrap_singleton_str(kwargs, "instrument")
        return continuous_iv_late(data=data, **kwargs)
    if canon == "mte":
        # mte's signature uses ``treatment`` for endog.
        _rename(kwargs, {"endog": "treatment"})
        return mte(data=data, **kwargs)
    if canon == "ivmte_bounds":
        _rename(kwargs, {"endog": "treatment"})
        return ivmte_bounds(data=data, **kwargs)

    # ── 8. Quantile IV ───────────────────────────────────────────────
    if canon == "ivqreg":
        from ..regression.iv_quantile import ivqreg as _ivqreg
        if formula is not None:
            kwargs.setdefault("formula", formula)
        return _ivqreg(data=data, **kwargs)

    # ── 9. Plausibly exogenous sensitivity ───────────────────────────
    if canon == "plausibly_exog_uci":
        return plausibly_exogenous_uci(data=data, **kwargs)
    if canon == "plausibly_exog_ltz":
        return plausibly_exogenous_ltz(data=data, **kwargs)

    # ── 10. Shift-share ──────────────────────────────────────────────
    if canon == "shift_share":
        if bartik is None:
            raise ImportError(
                "method='shift_share' requires sp.bartik. "
                "Check that the bartik subpackage is installed correctly."
            )
        return bartik(data=data, **kwargs)

    raise AssertionError(  # pragma: no cover
        f"Unreachable dispatcher branch: canonical='{canon}'."
    )


# Public alias — ``sp.iv.fit`` and ``sp.iv(...)`` route to the same
# underlying dispatcher.  ``fit`` predates the callable-module trick.
def fit(
    formula: Optional[str] = None,
    data: Any = None,
    *,
    method: str = "2sls",
    augmented_diagnostics: bool = True,
    **kwargs: Any,
):
    """Alias for :func:`_dispatch`.  See ``sp.iv.__doc__`` for usage."""
    return _dispatch(
        formula=formula, data=data, method=method,
        augmented_diagnostics=augmented_diagnostics, **kwargs,
    )


# ─── Helpers ────────────────────────────────────────────────────────────

def _rename(kwargs: Dict[str, Any], mapping: Dict[str, str]) -> None:
    """Translate alias kwargs to the underlying estimator's expected names.

    For each ``alias -> target`` pair: if ``alias`` is in ``kwargs`` and
    ``target`` is not, pop ``alias`` and set ``target``.  Raises if both
    are present (ambiguous).  Mutates ``kwargs`` in place.
    """
    for alias, target in mapping.items():
        if alias in kwargs:
            if target in kwargs:
                raise TypeError(
                    f"Got both '{alias}' and '{target}' — pick one. "
                    f"For this method '{target}' is the canonical name."
                )
            kwargs[target] = kwargs.pop(alias)


def _unwrap_singleton_str(kwargs: Dict[str, Any], key: str) -> None:
    """Unwrap a singleton list/tuple under ``key`` to a bare string.

    Some submethods (kernel_iv, continuous_iv_late) take a single
    ``instrument`` column name, not a list — but the dispatcher's unified
    contract uses ``instruments=[...]``.  Translate when there's exactly
    one element; raise a clear error if the user passed multiple to a
    method that only supports one.
    """
    val = kwargs.get(key)
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            kwargs[key] = val[0]
        elif len(val) > 1:
            raise ValueError(
                f"This method takes a single '{key}' column, got "
                f"{len(val)}: {list(val)}.  Pick one or use a different "
                f"method (e.g. method='2sls' supports multiple)."
            )


def _resolve_iv_args(formula, data, kwargs, *, allow_formula: bool):
    """Pop ``y/endog/instruments/exog`` from kwargs, falling back to the
    formula parser.  Used by methods that take explicit arrays/columns
    rather than a Patsy-style formula.

    Modifies ``kwargs`` in place to remove the consumed names.
    """
    y_ = kwargs.pop("y", None)
    endog_ = kwargs.pop("endog", None)
    instruments_ = kwargs.pop("instruments", None)
    exog_ = kwargs.pop("exog", None)
    if y_ is None or endog_ is None or instruments_ is None:
        if allow_formula and formula is not None and data is not None:
            y_, endog_, instruments_, exog_ = _formula_to_parts(formula, data)
        else:
            raise ValueError(
                "This IV method needs explicit `y`, `endog`, `instruments` "
                "(and optional `exog`) kwargs."
            )
    return y_, endog_, instruments_, exog_


def _formula_to_parts(formula: str, data):
    from ..core.utils import parse_formula
    parsed = parse_formula(formula)
    return (
        parsed["dependent"],
        parsed["endogenous"],
        parsed["instruments"],
        parsed.get("exogenous") or None,
    )


def _attach_augmented_diagnostics(model, result):
    """Attach KP rk, SW per-endog F, MOP effective F on top of the standard
    EconometricResults.diagnostics dict.  Failures are swallowed into a
    ``augmented_diagnostics_error`` key — augmenting is *additive* and must
    never break a successful fit.
    """
    try:
        D = model.X_endog
        Z = model.Z
        W = model.X_exog

        kp = kleibergen_paap_rk(
            endog=D,
            instruments=Z,
            exog=W[:, 1:] if W.shape[1] > 1 else None,
            add_const=(
                W.shape[1] >= 1 and np.allclose(W[:, 0], 1.0)
                if W.shape[1] else True
            ),
            cov_type="robust",
        )
        result.diagnostics["KP rk LM"] = kp.rk_lm
        result.diagnostics["KP rk LM p-value"] = kp.rk_lm_pvalue
        result.diagnostics["KP rk Wald F"] = kp.rk_f

        if D.shape[1] >= 2:
            sw = sanderson_windmeijer(
                endog=D,
                instruments=Z,
                exog=W[:, 1:] if W.shape[1] > 1 else None,
                add_const=False,  # constant already in W
                endog_names=getattr(model, "_endog_names", None),
            )
            for name, f in sw.sw_f.items():
                result.diagnostics[f"SW conditional F ({name})"] = f

        # Olea-Pflueger effective F (single-endogenous case only).
        if D.shape[1] == 1 and getattr(model, "data", None) is not None:
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
                    stat = (
                        getattr(ep, "F_eff", None)
                        or getattr(ep, "statistic", None)
                    )
                if stat is not None:
                    result.diagnostics["Olea-Pflueger effective F"] = float(stat)
            except Exception as e:
                result.diagnostics["OP effective F error"] = str(e)
    except Exception as e:  # pragma: no cover — never crash the estimator
        result.diagnostics["augmented_diagnostics_error"] = str(e)


# ═══════════════════════════════════════════════════════════════════════
#  Make the subpackage callable so ``sp.iv(...)`` works
# ═══════════════════════════════════════════════════════════════════════
#
# Without this, ``sp.iv`` resolves to a plain module — every callsite in
# the registry, agent summaries, MCP server docs, replication examples and
# `question/question.py:505` raises ``TypeError: 'module' object is not
# callable``.  PEP 562 lets us swap the module's class for one that defines
# ``__call__`` — Python's attribute lookup still finds submodules, members
# and ``__all__`` exactly as before.

class _CallableIVModule(ModuleType):
    """A ModuleType subclass that delegates calls to :func:`_dispatch`."""

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        return _dispatch(*args, **kwargs)


# Swap the live module's class.  Any reference to ``statspai.iv`` taken
# before this line still works — Python attribute lookups go through the
# instance's ``__class__`` at access time, not at import time.
sys.modules[__name__].__class__ = _CallableIVModule


__all__ = [
    # callable + alias
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
    "plausibly_exogenous_uci", "plausibly_exogenous_ltz",
    "PlausiblyExogenousResult",
    # MTE / IVMTE
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
    # Many-weak
    "jive_mw", "many_weak_ar", "ManyWeakIVResult",
    # Kernel IV with uniform inference (Lob et al. 2025)
    "kernel_iv", "KernelIVResult",
    # Continuous-instrument LATE (Xie et al. 2025)
    "continuous_iv_late", "ContinuousLATEResult",
    # IVDML
    "ivdml", "IVDMLResult",
    # re-exports
    "bartik", "shift_share_se", "BartikIV", "ssaggregate",
    "deepiv", "DeepIV",
]
