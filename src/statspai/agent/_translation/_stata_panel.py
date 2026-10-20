"""Translations of Stata's ``xt`` commands beyond ``xtreg, fe``.

``xtreg`` without ``fe`` (random effects, the between estimator, Gaussian
maximum likelihood), ``xtsum``, ``xtserial``, ``xtregar``, ``xttobit`` and
``xtintreg``. All of them need the panel
declaration of ``xtset``, which lives on another line: the handlers accept
Stata's own ``i()`` / ``t()`` options, and ``sp.stata`` fills them in.

Also the quantiles-via-moments pair: ``mmqreg`` (SSC, Rios-Avila) ->
``sp.mmqreg``; ``xtqreg`` (SSC, Machado and Santos Silva) is refused with a
pointer to ``sp.mmqreg``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._stata import (
    _build_formula,
    _emit,
    _emit_error,
    _robust_kind,
    _split_varlist_y_x,
    _vce_cluster,
)
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "xtreg_random"]

_NEEDS_PANEL = (
    "Couldn't recover the panel declaration from this command alone "
    "(Stata's `xtset id time` lives in another line). Replace <panel_id> "
    "and <panel_time> with the actual columns."
)


def _panel_of(cmd: StataCommand) -> tuple:
    unit = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    time = cmd.options.get("t") or "<panel_time>"
    return str(unit).strip(), str(time).strip()


def xtreg_random(cmd: StataCommand, y: str, xs: List[str]) -> Dict[str, Any]:
    """``xtreg y x [, re | be | mle]`` -> ``sp.panel`` / ``sp.mixed``."""
    opts = cmd.options
    unit, time = _panel_of(cmd)
    notes: List[str] = []
    if "<" in unit or "<" in time:
        notes.append(_NEEDS_PANEL)
    formula = _build_formula(y, xs)
    if "mle" in opts:
        if not xs:
            return _emit_error(
                "xtreg, mle needs at least one regressor",
                command="xtreg",
                suggestions=[],
            )
        args: Dict[str, Any] = {
            "formula": formula,
            "entity": unit,
            "time": time,
            "method": "mle",
        }
        kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
        return _emit(
            "panel",
            args,
            f"sp.panel(df, {formula!r}, {kw})",
            notes,
            semantics=[
                "xtreg, mle is the Gaussian random-effects model by maximum "
                "likelihood, with standard errors from the observed "
                "information; sigma_u, sigma_e and rho are in "
                "result.model_info."
            ],
        )
    method = "be" if "be" in opts else "re"
    opts.get("re")
    args = {
        "formula": formula,
        "entity": unit,
        "time": time,
        "method": method,
        "ssc": "stata",
    }
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    elif _robust_kind(cmd) == "hc1":
        if method == "be":
            return _emit_error(
                "xtreg, be does not allow vce(robust)",
                command="xtreg",
                suggestions=[],
            )
        args["robust"] = "robust"
    opts.get("theta")  # prints theta; it is in result.model_info
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
    semantics = [
        "ssc='stata' asks sp.panel for Stata's small-sample conventions "
        "(z statistics; with vce(robust) the variance is clustered on the "
        "panel with the factor G/(G-1) * (N-1)/(N-K))."
    ]
    return _emit(
        "panel", args, f"sp.panel(df, {formula!r}, {kw})", notes, semantics=semantics
    )


def _h_xtsum(cmd: StataCommand) -> Dict[str, Any]:
    """``xtsum varlist`` -> ``sp.xtsum``."""
    unit, _ = _panel_of(cmd)
    cmd.options.get("t")
    args: Dict[str, Any] = {"id": unit}
    if cmd.varlist:
        args["variables"] = list(cmd.varlist)
    notes = [_NEEDS_PANEL] if "<" in unit else []
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("xtsum", args, f"sp.xtsum(df, {kw})", notes)


def _h_xtserial(cmd: StataCommand) -> Dict[str, Any]:
    """``xtserial y x1 x2 [, output]`` -> ``sp.xtserial``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "xtserial needs an outcome and at least one regressor",
            command="xtserial",
            suggestions=[],
        )
    unit, time = _panel_of(cmd)
    notes = [_NEEDS_PANEL] if "<" in unit or "<" in time else []
    args: Dict[str, Any] = {"y": y, "x": list(xs), "id": unit, "time": time}
    code = f"sp.xtserial(df, {y!r}, {list(xs)!r}, id={unit!r}, time={time!r})"
    return _emit(
        "xtserial",
        args,
        code,
        notes,
        semantics=[
            "`output` prints the first-difference regression: it is in the "
            "result's 'params' and 'std_errors'."
        ],
    )


def _plain_regressors(command: str, xs: List[str]) -> Optional[Dict[str, Any]]:
    """Refuse factor-variable terms the target takes only as columns."""
    bad = [x for x in xs if not str(x).isidentifier()]
    if not bad:
        return None
    return _emit_error(
        f"{command}: {', '.join(bad)} is not a plain variable; sp.{command} "
        "takes regressor columns only. Create the indicator / interaction "
        "columns first (sp.stata does this for `generate`).",
        command=command,
        suggestions=[],
    )


#: ``rhotype()`` values with the abbreviations of [XT] xtregar's syntax
#: diagram (``reg:ress``, ``tsc:orr``, ``th:eil``, ``nag:ar``, ``one:step``).
_RHOTYPES = (
    ("dw", 2),
    ("regress", 3),
    ("freg", 4),
    ("tscorr", 3),
    ("theil", 2),
    ("nagar", 3),
    ("onestep", 3),
)


def _h_xtregar(cmd: StataCommand) -> Dict[str, Any]:
    """``xtregar y x [, re | fe] [rhotype() rhof() twostep lbi]`` ->
    ``sp.xtregar``."""
    opts = cmd.options
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "xtregar needs an outcome and at least one regressor",
            command="xtregar",
            suggestions=[],
        )
    refused = _plain_regressors("xtregar", xs)
    if refused:
        return refused
    if "fe" in opts and "re" in opts:
        return _emit_error(
            "xtregar: fe and re may not be combined", command="xtregar", suggestions=[]
        )
    opts.get("re")  # the default, spelled out
    unit, time = _panel_of(cmd)
    notes = [_NEEDS_PANEL] if "<" in unit or "<" in time else []
    args: Dict[str, Any] = {
        "y": y,
        "x": list(xs),
        "id": unit,
        "time": time,
        "method": "fe" if "fe" in opts else "re",
    }
    raw = opts.get("rhotype")
    if raw is not None:
        word = raw.strip().lower()
        full = next(
            (f for f, n in _RHOTYPES if len(word) >= n and f.startswith(word)), None
        )
        if full is None:
            return _emit_error(
                f"xtregar: rhotype({raw}) is not one of dw, regress, freg, "
                "tscorr, theil, nagar, onestep",
                command="xtregar",
                suggestions=[],
            )
        if full != "dw":
            args["rhotype"] = full
    if opts.get("rhof") is not None:
        try:
            args["rhof"] = float(opts.get("rhof"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _emit_error(
                f"xtregar: rhof({opts.get('rhof')}) is not a number",
                command="xtregar",
                suggestions=[],
            )
    for flag in ("twostep", "lbi"):
        if flag in opts:
            args[flag] = True
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "xtregar",
        args,
        f"sp.xtregar(df, {kw})",
        notes,
        semantics=[
            "rho, sigma_u, sigma_e, the three R-squared and the LBI test "
            "(with lbi) are in result.model_info; Stata's e(rho_ar) is "
            "model_info['rho_ar']."
        ],
    )


def _limit(cmd: StataCommand, name: str) -> Any:
    """``ll(#)`` / ``ul(#)`` as a float, None when absent; an error payload
    when the limit is a variable or is not written out."""
    if name not in cmd.options:
        return None
    raw = cmd.options.get(name)
    if raw is not None and raw.strip() == ".":
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _emit_error(
            f"{cmd.command}: {name}"
            + (
                " without a value censors at the smallest (largest) observed "
                "outcome, which one translated line cannot see"
                if raw is None
                else f"({raw}) is not a number; a limit that varies by "
                "observation is not translated"
            )
            + f"; write {name}(#)",
            command=cmd.command,
            suggestions=[],
        )


def _quadrature(cmd: StataCommand, args: Dict[str, Any], notes: List[str]) -> Any:
    """``intpoints(#)`` -> ``intpoints=``; ``intmethod(ghermite)`` is kept
    out (returns the option name), Stata's adaptive default is read."""
    opts = cmd.options
    if opts.get("intpoints") is not None:
        try:
            args["intpoints"] = int(str(opts.get("intpoints")).strip())
        except ValueError:
            return _emit_error(
                f"{cmd.command}: intpoints({opts.get('intpoints')}) is not an "
                "integer",
                command=cmd.command,
                suggestions=[],
            )
    method = (opts.get("intmethod") or "").strip().lower()
    if method and not ("mvaghermite".startswith(method) and len(method) >= 2):
        notes.append(
            f"intmethod({method}): sp.{cmd.command} integrates with adaptive "
            "Gauss-Hermite quadrature centred at each panel's mode; Stata's "
            "non-adaptive rule approximates the same likelihood less "
            "accurately, so its numbers differ by its quadrature error."
        )
        return "intmethod"
    return None


_QUADRATURE_NOTE = (
    "Both programs approximate the same panel likelihood by adaptive "
    "Gauss-Hermite quadrature (sp centres each panel at its mode, Stata at "
    "the posterior mean); sp's default of 30 points is at least as accurate "
    "as Stata's default intpoints(12)."
)


def _h_xttobit(cmd: StataCommand) -> Dict[str, Any]:
    """``xttobit y x, ll(#) ul(#) [intpoints(#) tobit]`` -> ``sp.xttobit``."""
    opts = cmd.options
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "xttobit needs an outcome and at least one regressor",
            command="xttobit",
            suggestions=[],
        )
    refused = _plain_regressors("xttobit", xs)
    if refused:
        return refused
    unit, _ = _panel_of(cmd)
    notes = [_NEEDS_PANEL] if "<" in unit else []
    args: Dict[str, Any] = {"y": y, "x": list(xs), "id": unit}
    for name in ("ll", "ul"):
        value = _limit(cmd, name)
        if isinstance(value, dict):
            return value
        if value is not None:
            args[name] = value
    if "ll" not in args and "ul" not in args:
        return _emit_error(
            "xttobit without ll() or ul() censors nothing: it is the "
            "random-effects linear model by maximum likelihood",
            command="xttobit",
            suggestions=["sp.panel(df, 'y ~ x', entity=..., method='mle')"],
        )
    lost = _quadrature(cmd, args, notes)
    if isinstance(lost, dict):
        return lost
    semantics = [_QUADRATURE_NOTE]
    if "tobit" in opts:
        semantics.append(
            "tobit: the likelihood-ratio test of sigma_u = 0 against the "
            "pooled tobit is result.model_info['lr_sigma_u0'] (Stata's "
            "e(chi2_c)), with its p-value in 'lr_sigma_u0_pvalue'."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("xttobit", args, f"sp.xttobit(df, {kw})", notes, semantics=semantics)
    if lost:
        out["untranslated_options"] = [lost]
    if _oim_only(cmd):
        out["_vce_is_sp_default"] = True
    return out


def _oim_only(cmd: StataCommand) -> bool:
    """True when ``vce()`` is absent or Stata's default ``vce(oim)``."""
    vce = (cmd.options.get("vce") or "").split()
    return not vce or (len(vce) == 1 and vce[0].lower() == "oim")


def _h_xtintreg(cmd: StataCommand) -> Dict[str, Any]:
    """``xtintreg lo hi x [, intpoints(#) intreg]`` -> ``sp.xtintreg``."""
    opts = cmd.options
    if len(cmd.varlist) < 3:
        return _emit_error(
            "xtintreg needs depvar_lower, depvar_upper and at least one " "regressor",
            command="xtintreg",
            suggestions=[],
        )
    lo, hi, xs = cmd.varlist[0], cmd.varlist[1], list(cmd.varlist[2:])
    refused = _plain_regressors("xtintreg", xs)
    if refused:
        return refused
    unit, _ = _panel_of(cmd)
    notes = [_NEEDS_PANEL] if "<" in unit else []
    args: Dict[str, Any] = {"y_lower": lo, "y_upper": hi, "x": xs, "id": unit}
    lost = _quadrature(cmd, args, notes)
    if isinstance(lost, dict):
        return lost
    semantics = [
        _QUADRATURE_NOTE,
        "A missing depvar_lower (depvar_upper) is left- (right-) censored, "
        "as in Stata.",
    ]
    if "intreg" in opts:
        semantics.append(
            "intreg: the likelihood-ratio test of sigma_u = 0 against the "
            "pooled interval regression is result.model_info['lr_sigma_u0'] "
            "(Stata's e(chi2_c))."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("xtintreg", args, f"sp.xtintreg(df, {kw})", notes, semantics=semantics)
    if lost:
        out["untranslated_options"] = [lost]
    if _oim_only(cmd):
        out["_vce_is_sp_default"] = True
    return out


def _quantile_percent(raw: Optional[str], command: str) -> Any:
    """``quantile(#)`` as a fraction (one number; ``#`` >= 1 is a percent,
    as in qreg and mmqreg); an error payload otherwise."""
    if raw is None:
        return 0.5
    text = str(raw).strip()
    if len(text.replace(",", " ").split()) != 1 or "(" in text:
        return _emit_error(
            f"{command}: quantile({text}) lists several quantiles; sp.mmqreg "
            "fits one quantile per call (no joint covariance across "
            "quantiles). Write one line per quantile.",
            command=command,
            suggestions=[],
        )
    try:
        q = float(text)
    except ValueError:
        return _emit_error(
            f"{command}: quantile({text}) is not a number",
            command=command,
            suggestions=[],
        )
    if q >= 1:
        q = q / 100.0
    if not 0 < q < 1:
        return _emit_error(
            f"{command}: quantile({text}) is outside (0, 100)",
            command=command,
            suggestions=[],
        )
    return q


#: mmqreg options that change the estimator or its variance and have no
#: counterpart in sp.mmqreg.
_MMQREG_REFUSED = {
    "denopt": "denopt() changes the density estimator; sp.mmqreg uses "
    "mmqreg's default (fitted, Hall-Sheather)",
    "dfadj": "dfadj rescales the variance by N - k - absorbed coefficients",
    "jknife": "jknife applies the split-panel jackknife bias correction",
}


def _h_mmqreg(cmd: StataCommand) -> Dict[str, Any]:
    """``mmqreg y x [, quantile(#) absorb() robust cluster()]`` ->
    ``sp.mmqreg``."""
    opts = cmd.options
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "mmqreg needs an outcome and at least one regressor",
            command="mmqreg",
            suggestions=[],
        )
    refused = _plain_regressors("mmqreg", xs)
    if refused:
        return refused
    for name, why in _MMQREG_REFUSED.items():
        if name in opts:
            return _emit_error(
                f"mmqreg: {why}; not available in sp.mmqreg.",
                command="mmqreg",
                suggestions=[],
            )
    q = _quantile_percent(opts.get("quantile"), "mmqreg")
    if isinstance(q, dict):
        return q
    args: Dict[str, Any] = {"y": y, "x": list(xs)}
    absorb = (opts.get("absorb") or "").split()
    if absorb:
        args["absorb"] = absorb
    args["quantile"] = q
    vce_words = (opts.get("vce") or "").split()
    if vce_words and vce_words[0].lower() not in ("robust", "cluster"):
        return _emit_error(
            f"mmqreg: vce({opts.get('vce')}) is not translated; sp.mmqreg has "
            "vce='gls' (the default), 'robust' and 'cluster'",
            command="mmqreg",
            suggestions=[],
        )
    cluster_words = (opts.get("cluster") or "").split() or vce_words[1:]
    if len(cluster_words) > 1:
        return _emit_error(
            f"mmqreg: clustering on {' '.join(cluster_words)} (several "
            "variables); sp.mmqreg clusters on one",
            command="mmqreg",
            suggestions=[],
        )
    cluster = _vce_cluster(cmd)
    if cluster:
        args["vce"] = "cluster"
        args["cluster"] = cluster
    elif _robust_kind(cmd) == "hc1":
        args["vce"] = "robust"
    for display in ("nowarning", "nols"):
        opts.get(display)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "mmqreg",
        args,
        f"sp.mmqreg(df, {kw})",
        [],
        semantics=[
            "The default variance is mmqreg's ('gls'); robust and "
            "cluster() map to vce='robust' / vce='cluster'. The location and "
            "scale equations and the joint covariance are in "
            "result.model_info ('location', 'scale', 'vcov_all'); e(qval) "
            "is model_info['q'].",
            "sp.mmqreg drops observations alone in an absorbed group (with "
            "a warning); mmqreg keeps them, which can move q(tau) by one "
            "order statistic.",
        ],
    )


#: The call suggested in place of ``xtqreg``. Kept outside the handler so the
#: coverage table (which reads targets from handler source) does not list
#: ``xtqreg`` as translated.
_XTQREG_POINTER = (
    "sp.mmqreg(df, {y!r}, {x!r}, absorb={unit!r}, quantile={q}, "
    "vce='cluster', cluster={unit!r})"
)


def _h_xtqreg(cmd: StataCommand) -> Dict[str, Any]:
    """``xtqreg`` is refused: its quantile-coefficient standard errors use a
    GLS form sp does not reproduce. Points to ``sp.mmqreg``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    unit = cmd.options.get("i") or "<panel_id>"
    q = _quantile_percent(cmd.options.get("quantile"), "xtqreg")
    q_txt = repr(q) if not isinstance(q, dict) else "<quantile>"
    unit = str(unit).strip()
    call = _XTQREG_POINTER.format(
        y=y or "<y>", x=list(xs) if xs else "<x>", unit=unit, q=q_txt
    )
    return _emit_error(
        "xtqreg is not translated: its quantile-coefficient standard errors "
        "come from a GLS form (scale moment not centred) that sp does not "
        "reproduce, so a translated line would report different SEs under "
        "the same name. sp.mmqreg fits the same model: its coefficients, "
        "q(tau) and location / scale coefficients equal xtqreg's (1e-10 on "
        "the test fixture); for standard errors use vce='cluster' on the "
        "panel (xtqreg's location / scale covariances are those times "
        "G/(G-1)(N-1)/(N-K)) or vce='gls' (mmqreg's default).",
        command="xtqreg",
        suggestions=[call],
    )


HANDLERS = {
    "mmqreg": _h_mmqreg,
    "xtqreg": _h_xtqreg,
    "xtsum": _h_xtsum,
    "xtserial": _h_xtserial,
    "xtregar": _h_xtregar,
    "xttobit": _h_xttobit,
    "xtintreg": _h_xtintreg,
}
