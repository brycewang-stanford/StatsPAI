"""R command → StatsPAI tool-call translator.

Targets the most common R econometrics calls: ``feols`` (fixest),
``felm`` (lfe), ``lm`` (base), ``did`` (Callaway-Sant'Anna's R port),
``synth`` (Synth package). The R input is parsed with a small
regex-based scanner — we don't pull a full R parser dependency for
five common shapes.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


def _emit(tool: str, arguments: Dict[str, Any],
           python_code: str, notes: List[str] = None) -> Dict[str, Any]:
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "python_code": python_code,
        "notes": list(notes or []),
        "ok": True,
    }


def _emit_error(message: str, **extra) -> Dict[str, Any]:
    return {"tool": None, "ok": False, "error": message, **extra}


# ---------------------------------------------------------------------------
# Argument parser — handles ``foo(arg1 = "x", arg2 = c("a","b"))`` shapes
# ---------------------------------------------------------------------------

def _split_top_level(s: str, delim: str = ",") -> List[str]:
    """Split on ``delim`` outside any parens / quotes."""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    in_q: Optional[str] = None
    for ch in s:
        if in_q:
            buf.append(ch)
            if ch == in_q:
                in_q = None
            continue
        if ch in "\"'":
            in_q = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == delim and depth == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _parse_call(line: str) -> Optional[Tuple[str, List[str], Dict[str, str]]]:
    """Parse ``fn(arg1, arg2, key=val, ...)`` into name + positional + kwargs.

    Returns ``None`` if ``line`` doesn't match a function-call shape.
    """
    line = line.strip().rstrip(";").strip()
    m = re.match(r"^([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", line, flags=re.S)
    if not m:
        return None
    fn = m.group(1)
    body = m.group(2)
    parts = _split_top_level(body, ",")
    pos: List[str] = []
    kw: Dict[str, str] = {}
    for p in parts:
        # ``key = value`` (the space around ``=`` is normal R style)
        eq = re.match(r"^([A-Za-z_][\w.]*)\s*=\s*(.+)$", p, flags=re.S)
        if eq:
            kw[eq.group(1)] = eq.group(2).strip()
        else:
            pos.append(p)
    return fn, pos, kw


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _parse_c_vector(s: str) -> List[str]:
    """``c("a", "b", "c")`` or ``c(a, b)`` → list of stripped strings."""
    s = s.strip()
    m = re.match(r"^c\s*\((.*)\)\s*$", s, flags=re.S)
    if not m:
        return [_strip_quotes(s)]
    parts = _split_top_level(m.group(1), ",")
    return [_strip_quotes(p) for p in parts]


# ---------------------------------------------------------------------------
# fixest formula → (lhs, rhs, fe_terms, iv_lhs, iv_rhs)
# ---------------------------------------------------------------------------

def _parse_fixest_formula(formula: str) -> Dict[str, Any]:
    """Decompose a fixest ``y ~ x | id^year | (d ~ z) | id`` formula.

    Pipes split: outcome+covariates | fixed effects | IV part | clusters.
    Missing trailing pipes ⇒ those parts are empty.
    """
    # Strip wrapping quotes if the formula was passed as a string
    formula = _strip_quotes(formula).strip()
    parts = [p.strip() for p in formula.split("|")]
    main = parts[0] if parts else ""
    fe_part = parts[1].strip() if len(parts) >= 2 else ""
    iv_part = parts[2].strip() if len(parts) >= 3 else ""
    cluster_part = parts[3].strip() if len(parts) >= 4 else ""

    fe_terms: List[str] = []
    if fe_part:
        for term in re.split(r"\s*\+\s*", fe_part):
            # ``id^year`` → ``id^year`` (interaction); we keep verbatim.
            fe_terms.append(term.strip())

    iv_lhs = iv_rhs = ""
    if iv_part:
        m = re.match(r"^\(?\s*(.+?)\s*~\s*(.+?)\s*\)?\s*$", iv_part)
        if m:
            iv_lhs, iv_rhs = m.group(1).strip(), m.group(2).strip()

    return {
        "main": main,
        "fe_terms": fe_terms,
        "iv_lhs": iv_lhs,
        "iv_rhs": iv_rhs,
        "cluster_terms": [t.strip() for t in re.split(r"\s*\+\s*", cluster_part) if t.strip()],
    }


# ---------------------------------------------------------------------------
# Per-function handlers
# ---------------------------------------------------------------------------

def _h_feols(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    formula = pos[0] if pos else kw.get("fml") or kw.get("formula")
    if not formula:
        return _emit_error("feols requires a formula as the first argument")
    decomp = _parse_fixest_formula(formula)
    main = decomp["main"]
    fe_terms = decomp["fe_terms"]
    iv_lhs, iv_rhs = decomp["iv_lhs"], decomp["iv_rhs"]
    clusters = decomp["cluster_terms"]
    cluster_kw = kw.get("cluster")
    if cluster_kw and not clusters:
        clusters = _parse_c_vector(cluster_kw)

    # Recompose into a Wilkinson formula sp.fixest understands.
    if iv_lhs and iv_rhs:
        formula_out = f"{main} + ({iv_lhs} ~ {iv_rhs})"
    else:
        formula_out = main
    args: Dict[str, Any] = {
        "formula": formula_out,
        "fe": fe_terms,
    }
    if clusters:
        args["cluster"] = clusters[0] if len(clusters) == 1 else clusters
    notes: List[str] = []
    if "vcov" in kw:
        notes.append(f"R `vcov={kw['vcov']}` not auto-translated; check sp.fixest "
                      f"`robust=` / `cluster=` options.")
    if "weights" in kw:
        args["weights"] = _strip_quotes(kw["weights"])
    code_pairs = [f"data=df", f"fe={fe_terms!r}"]
    if "cluster" in args:
        code_pairs.append(f"cluster={args['cluster']!r}")
    python = f"sp.fixest({formula_out!r}, {', '.join(code_pairs)})"
    return _emit("fixest", args, python, notes)


def _h_felm(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """felm uses the same | structure as feols but is from the lfe package."""
    return _h_feols(pos, kw, _)  # delegate


def _h_lm(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("lm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    args: Dict[str, Any] = {"formula": formula}
    weights = kw.get("weights")
    if weights:
        args["weights"] = _strip_quotes(weights)
    code = f"sp.regress({formula!r}, data=df)"
    return _emit("regress", args, code)


def _h_did(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """Brantly Callaway's did::att_gt R API → sp.callaway_santanna."""
    yname = _strip_quotes(kw.get("yname", ""))
    gname = _strip_quotes(kw.get("gname", ""))
    tname = _strip_quotes(kw.get("tname", ""))
    idname = _strip_quotes(kw.get("idname", ""))
    missing = [n for n, v in (("yname", yname), ("gname", gname),
                                ("tname", tname), ("idname", idname)) if not v]
    if missing:
        return _emit_error(
            f"R `did::att_gt` translation needs {missing} kwargs.")
    args: Dict[str, Any] = {
        "y": yname, "g": gname, "t": tname, "i": idname,
    }
    est_method = _strip_quotes(kw.get("est_method", "dr"))
    if est_method in {"dr", "ipw", "reg"}:
        args["estimator"] = est_method
    python = (f"sp.callaway_santanna(data=df, y={yname!r}, g={gname!r}, "
               f"t={tname!r}, i={idname!r}, "
               f"estimator={args.get('estimator', 'dr')!r})")
    return _emit("callaway_santanna", args, python)


def _h_glm(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `glm(y ~ x, family=binomial, data=df)` → ``sp.glm`` (or sp.logit
    / sp.probit / sp.poisson when the family resolves to one of those).
    """
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("glm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    family = _strip_quotes(kw.get("family", "gaussian")) if kw.get("family") else "gaussian"
    # Recognise common family/link combos so we can route to the
    # specialised sp helper rather than the generic GLM.
    fam_lower = family.lower().strip()
    if "binomial" in fam_lower and ("logit" in fam_lower or fam_lower == "binomial"):
        return _emit("logit", {"formula": formula},
                      f"sp.logit({formula!r}, data=df)")
    if "binomial" in fam_lower and "probit" in fam_lower:
        return _emit("probit", {"formula": formula},
                      f"sp.probit({formula!r}, data=df)")
    if "poisson" in fam_lower:
        return _emit("poisson", {"formula": formula},
                      f"sp.poisson({formula!r}, data=df)")
    args: Dict[str, Any] = {"formula": formula, "family": family}
    python = f"sp.glm({formula!r}, data=df, family={family!r})"
    return _emit("glm", args, python)


def _h_glmer(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `lme4::glmer(y ~ x + (1|group), family=binomial, data=df)` →
    ``sp.glmer`` / multilevel GLM. The ``(1|group)`` random-effect
    syntax passes through verbatim — sp's multilevel module accepts
    the same shape."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("glmer requires a formula as the first argument")
    formula = _strip_quotes(formula)
    family = _strip_quotes(kw.get("family", "gaussian")) if kw.get("family") else "gaussian"
    args: Dict[str, Any] = {"formula": formula, "family": family}
    notes: List[str] = []
    code_pairs = ["data=df", f"family={family!r}"]
    python = f"sp.glmer({formula!r}, {', '.join(code_pairs)})"
    return _emit("glmer", args, python, notes)


def _h_lmer(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `lme4::lmer(y ~ x + (1|group), data=df)` → ``sp.multilevel`` /
    Gaussian random effects."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("lmer requires a formula as the first argument")
    formula = _strip_quotes(formula)
    args: Dict[str, Any] = {"formula": formula}
    python = f"sp.multilevel({formula!r}, data=df)"
    return _emit("multilevel", args, python)


def _h_plm(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `plm(y ~ x, data=df, model='within', index=c('id','t'))` →
    ``sp.panel`` with the chosen estimator."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("plm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    model = _strip_quotes(kw.get("model", "within")).lower() if kw.get("model") else "within"
    index = kw.get("index")
    panel_keys: List[str] = []
    if index:
        panel_keys = _parse_c_vector(index)
    args: Dict[str, Any] = {"formula": formula, "method": model}
    if panel_keys:
        args["id"] = panel_keys[0]
        if len(panel_keys) > 1:
            args["time"] = panel_keys[1]
    code_pairs = ["data=df", f"method={model!r}"]
    if "id" in args:
        code_pairs.append(f"id={args['id']!r}")
    if "time" in args:
        code_pairs.append(f"time={args['time']!r}")
    python = f"sp.panel({formula!r}, {', '.join(code_pairs)})"
    return _emit("panel", args, python)


def _h_matchit(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `MatchIt::matchit(treat ~ x1 + x2, data=df, method='nearest')` →
    ``sp.match``."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("matchit requires a formula as the first argument")
    formula = _strip_quotes(formula)
    method = _strip_quotes(kw.get("method", "nearest")).lower() if kw.get("method") else "nearest"
    # Map MatchIt method names → sp.match method names where they differ.
    method_alias = {
        "nearest": "nn", "exact": "exact", "cem": "cem",
        "subclass": "subclass", "optimal": "optimal", "full": "full",
        "genetic": "genmatch",
    }
    sp_method = method_alias.get(method, method)
    args: Dict[str, Any] = {"formula": formula, "method": sp_method}
    notes: List[str] = []
    if method != sp_method:
        notes.append(f"MatchIt method '{method}' mapped to sp.match "
                      f"method='{sp_method}'.")
    distance = kw.get("distance")
    if distance:
        args["distance"] = _strip_quotes(distance)
    python = f"sp.match({formula!r}, data=df, method={sp_method!r})"
    return _emit("match", args, python, notes)


def _h_synth(pos: List[str], kw: Dict[str, str], _) -> Dict[str, Any]:
    """R `Synth::synth(data.prep.obj=dataprep_out)` → translation note
    pointing at sp.synth's flat API.

    R's Synth requires a separate ``dataprep()`` call producing the
    object that ``synth()`` then consumes. We don't have access to
    that earlier call here, so emit a structured hint instead of a
    half-finished translation.
    """
    return _emit_error(
        "R `Synth::synth` translation needs explicit unit / time / treated / "
        "treatment_time mapping which the R API splits across "
        "`dataprep()` and `synth()`. Please call sp.synth() directly with "
        "outcome / unit / time / treated_unit / treatment_time kwargs. "
        "If the dataprep() call is in scope, the relevant fields are: "
        "predictors → predictors, dependent → outcome, unit.variable → "
        "unit, time.variable → time, treatment.identifier → "
        "treated_unit, time.predictors.prior[max] + 1 → treatment_time.",
        command="synth")


R_FUNCTION_MAP: Dict[str, Callable] = {
    "feols": _h_feols,
    "felm": _h_felm,
    "lm": _h_lm,
    "glm": _h_glm,
    "glmer": _h_glmer,
    "lmer": _h_lmer,
    "plm": _h_plm,
    "matchit": _h_matchit,
    "att_gt": _h_did,
    "did": _h_did,
    "synth": _h_synth,
}


def from_r(line: str) -> Dict[str, Any]:
    """Translate one R / fixest / felm call to a StatsPAI tool-call payload.

    Parameters
    ----------
    line : str
        Single R expression of the form ``fn(...)``. Multi-line
        scripts must be split by the caller.

    Returns
    -------
    dict
        Same shape as :func:`from_stata`.
    """
    parsed = _parse_call(line)
    if parsed is None:
        return _emit_error(
            "R input did not match `fn(...)` shape. Pass one R "
            "expression at a time (no assignment, no piping).")
    fn, pos, kw = parsed
    handler = R_FUNCTION_MAP.get(fn)
    if handler is None:
        from difflib import get_close_matches
        suggestions = get_close_matches(fn, list(R_FUNCTION_MAP.keys()),
                                          n=5, cutoff=0.55)
        return _emit_error(
            f"unsupported R function {fn!r}",
            command=fn, suggestions=suggestions)
    return handler(pos, kw, [])


__all__ = ["from_r", "R_FUNCTION_MAP"]
