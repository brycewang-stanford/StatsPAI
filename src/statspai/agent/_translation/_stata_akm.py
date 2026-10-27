"""Saved fixed effects: ``reghdfe ..., absorb(name=var)`` / ``savefe`` and
``felsdvreg``.

``reghdfe y x, absorb(wfe=worker ffe=firm)`` fits the model and saves the
estimated effects as new variables. ``sp.hdfe_ols`` fits the same model and
its result's ``fixef(absorb_constant=False)`` returns the effects normalised
as reghdfe saves them (Stata 18, reghdfe 6.13: every effect after the first
averages zero within each connected set it forms with the first, the first
averages zero over the sample, the remainder is ``_cons``; agreement to
1e-7, the accuracy of reghdfe's own recovered effects, on
``tests/reference_parity/_fixtures/akm_twoway*``). ``savefe`` saves the
unnamed terms as ``__hdfe#__`` (``#`` = position in ``absorb()``), after
dropping existing ``__hdfe*__`` variables; a named variable that already
exists is an error (r(110)), and the effects are missing outside
``e(sample)``. ``sp.from_stata`` keeps ``python_code`` the single fit call
and adds ``python_script`` (fit, ``fixef``, the new columns); ``sp.stata``
creates the variables (:func:`saved_effects_after`).

``felsdvreg y x, ivar(i) jvar(j) feff() peff() xb() res() mover() mnum()
pobs() group() [normalize]`` (``cornelissen2008stata``, Stata Journal
st0143) is
``sp.akm(..., connected='all', variance='none')``: every observation, each
mobility group normalised on its own. The coefficients and their iid
standard errors agree with felsdvreg 1.8.0 to 1e-12 and the eight saved
variables to felsdvreg's float storage (2e-7) on the same fixture
(``AKMResult.felsdvreg_variables``). ``robust``, ``cluster()`` (felsdvreg
counts every fixed effect in the small-sample factor), ``cons``,
``takegroup``, ``grouponly``, ``hat()`` / ``orig()``, ``feffse()``,
``noadji`` / ``noadjj`` and ``cholsolve`` are not translated.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = [
    "HANDLERS",
    "felsdvreg_after",
    "saved_effects",
    "saved_effects_after",
    "saved_effects_code",
]

_NAME = r"[^\W\d]\w*"


def saved_effects(absorb: str) -> Tuple[Dict[int, str], bool]:
    """``{position: new variable}`` of the ``name=term`` entries of
    ``absorb()`` and whether ``savefe`` was given."""
    spec, _, sub = absorb.partition(",")
    named: Dict[int, str] = {}
    for k, raw in enumerate(spec.split()):
        if "=" in raw:
            named[k] = raw.split("=", 1)[0].strip()
    savefe = "savefe" in [w.strip().lower() for w in sub.split()]
    return named, savefe


def plan_saved_effects(
    absorb: str, fe_list: List[str]
) -> Tuple[Dict[str, str], Optional[str]]:
    """``{new variable: sp.hdfe_ols term}`` for the effects reghdfe saves,
    or an error when one cannot be recovered."""
    named, savefe = saved_effects(absorb)
    out: Dict[str, str] = {}
    for k, term in enumerate(fe_list):
        var = named.get(k)
        if var is None and savefe:
            var = f"__hdfe{k + 1}__"
        if var is None:
            continue
        if "c." in term:
            return {}, (
                f"absorb({var}=...) saves the slopes of {term!r}; sp.hdfe_ols "
                "recovers saved effects of factors only."
            )
        out[var] = term
    return out, None


def _term_key(term: str) -> List[str]:
    return term.split("^")


def saved_effects_code(payload: Dict[str, Any]) -> None:
    """Add ``python_script``: the call (``python_code``, which stays a single
    call as every translation's contract requires) assigned to ``fit``,
    then the lines that create the saved effects."""
    save = payload.get("save_fe") or {}
    if not save:
        return
    call = str(payload["python_code"])
    lines = [
        f"fit = {call}",
        "fe = fit.fixef(absorb_constant=False)  # reghdfe's saved effects; "
        "_cons is fe[...].attrs['constant']",
    ]
    for var, term in save.items():
        cols = _term_key(term)
        if len(cols) == 1:
            lines.append(f"df[{var!r}] = df[{cols[0]!r}].map(fe[{term!r}])")
        else:
            lines.append(
                f"df[{var!r}] = fe[{term!r}].reindex("
                f"pd.MultiIndex.from_frame(df[{cols!r}])).to_numpy()"
            )
    payload["python_script"] = "\n".join(lines)
    payload["notes"].append(
        "python_code is the fit; python_script also creates the saved "
        "variables in df."
    )


def saved_effects_after(session: Any, call: Dict[str, Any], data: pd.DataFrame) -> None:
    """Create the variables of ``absorb(name=var)`` / ``savefe`` in the
    session's data: the effect of each estimation-sample row, missing on
    the other rows (as reghdfe)."""
    from ._stata_expr import StataExprError

    save = call.get("save_fe") or {}
    steps = getattr(session, "_steps", None)
    if not save or steps is None:
        return
    result = session.last
    from ...panel._hdfe_fixef import row_effects

    sample_index, effects = row_effects(result)
    full = steps.data
    if not full.index.is_unique or not sample_index.isin(full.index).all():
        raise StataExprError(
            "the saved fixed effects cannot be placed on the rows of the data "
            "(the index is not unique)"
        )
    if any(str(v).startswith("__hdfe") for v in save):
        stale = [c for c in full.columns if re.fullmatch(r"__hdfe\d+__", str(c))]
        if stale:
            steps._own()
            steps.data = steps.data.drop(columns=stale)
    for var, term in save.items():
        column = pd.Series(np.nan, index=steps.data.index)
        column.loc[sample_index] = effects[term]
        steps.add_column(var, column.to_numpy(), double=True)


_FELSDVREG_VARS = ("peff", "feff", "xb", "res", "mover", "mnum", "pobs", "group")


def _h_felsdvreg(cmd: StataCommand) -> Dict[str, Any]:
    """``felsdvreg y x, ivar(i) jvar(j) feff() peff() xb() res() mover()
    mnum() pobs() group() [normalize]`` -> ``sp.akm(connected='all',
    variance='none')`` plus the saved variables."""
    opts = cmd.options
    if len(cmd.varlist) < 2:
        return _emit_error(
            "felsdvreg without regressors needs its `cons` option, which is "
            "not translated; list at least one regressor.",
            command="felsdvreg",
            suggestions=[],
        )
    y, xs = cmd.varlist[0], list(cmd.varlist[1:])
    bad = [x for x in xs if not re.fullmatch(_NAME, x)]
    if bad:
        return _emit_error(
            f"felsdvreg: regressor(s) {bad} are not plain variables (felsdvreg "
            "takes no factor variables; create the indicators).",
            command="felsdvreg",
            suggestions=[],
        )
    ivar = (opts.get("ivar") or "").strip()
    jvar = (opts.get("jvar") or "").strip()
    names: Dict[str, str] = {}
    missing = []
    for key in _FELSDVREG_VARS:
        val = (opts.get(key) or "").strip()
        if not val:
            missing.append(key)
        names[key] = val
    if not ivar or not jvar or missing:
        need = (["ivar"] if not ivar else []) + (["jvar"] if not jvar else []) + missing
        return _emit_error(
            "felsdvreg requires the options " + ", ".join(f"{k}()" for k in need),
            command="felsdvreg",
            suggestions=[],
        )
    normalize = "normalize" in opts
    args: Dict[str, Any] = {
        "y": y,
        "worker": ivar,
        "firm": jvar,
        "covariates": xs,
        "connected": "all",
        "variance": "none",
    }
    call = (
        f"sp.akm(df, y={y!r}, worker={ivar!r}, firm={jvar!r}, "
        f"covariates={xs!r}, connected='all', variance='none')"
    )
    rename = {k: names[k] for k in _FELSDVREG_VARS}
    script = "\n".join(
        [
            f"res = {call}",
            f"v = res.felsdvreg_variables(normalize={normalize!r})"
            f".rename(columns={rename!r})",
            "df = df.drop(columns=[c for c in v.columns if c in df.columns])"
            ".join(v)",
        ]
    )
    notes = [
        "felsdvreg -> sp.akm(connected='all', variance='none'): every "
        "observation, worker and firm effects normalised within each mobility "
        "group; the coefficients are res.coefficients, the saved variables "
        "res.felsdvreg_variables()"
        + (
            " (normalize: firm effects average zero in each group)."
            if normalize
            else " (the smallest firm of each group has effect zero, felsdvreg's "
            "reference)."
        ),
        "python_code is the fit; python_script also creates felsdvreg's "
        "eight variables in df.",
    ]
    out = _emit("akm", args, call, notes)
    out["python_script"] = script
    out["akm_generate"] = {"names": rename, "normalize": normalize}
    return out


def felsdvreg_after(session: Any, call: Dict[str, Any], data: pd.DataFrame) -> None:
    """Create felsdvreg's eight variables in the session's data (replacing
    them, as felsdvreg does), missing outside the estimation sample."""
    from ._stata_expr import StataExprError

    spec = call.get("akm_generate") or {}
    steps = getattr(session, "_steps", None)
    if not spec or steps is None:
        return
    v = session.last.felsdvreg_variables(normalize=bool(spec.get("normalize")))
    full = steps.data
    if not full.index.is_unique or not v.index.isin(full.index).all():
        raise StataExprError(
            "felsdvreg's variables cannot be placed on the rows of the data "
            "(the index is not unique)"
        )
    for key, var in spec["names"].items():
        if var in steps.data.columns:
            steps._own()
            steps.data = steps.data.drop(columns=[var])
        column = pd.Series(np.nan, index=steps.data.index)
        column.loc[v.index] = v[key].to_numpy(dtype=float)
        # felsdvreg compresses: its effects, xb and residual are floats
        steps.add_column(
            var, column.to_numpy(), double=key not in ("peff", "feff", "xb", "res")
        )


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "felsdvreg": _h_felsdvreg,
}
