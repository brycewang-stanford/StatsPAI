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


def _emit(
    tool: str,
    arguments: Dict[str, Any],
    python_code: str,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "python_code": python_code,
        "notes": list(notes or []),
        "ok": True,
    }


def _emit_error(message: str, **extra: Any) -> Dict[str, Any]:
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
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
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
        "cluster_terms": [
            t.strip() for t in re.split(r"\s*\+\s*", cluster_part) if t.strip()
        ],
    }


# ---------------------------------------------------------------------------
# Per-function handlers
# ---------------------------------------------------------------------------


def _pyfixest_fml(
    main: str,
    fe_terms: List[str],
    iv_lhs: Optional[str] = None,
    iv_rhs: Optional[str] = None,
) -> str:
    """Reassemble a pyfixest formula that :func:`sp.feols` accepts.

    ``sp.feols`` takes the fixed effects and IV inside the formula, not as
    separate arguments: ``depvar ~ exog | fe1 + fe2 | endog ~ instruments``
    (each ``|`` section optional). Note the IV section has NO parentheses —
    ``y ~ x | id | (d ~ z)`` is a pyfixest syntax error, ``y ~ x | id | d ~ z``
    is correct. Building the fml here (rather than emitting a non-existent
    ``sp.fixest(formula, fe=...)`` call) is what makes the translated payload
    actually runnable.
    """
    parts = [main.strip()]
    if fe_terms:
        parts.append(" + ".join(fe_terms))
    if iv_lhs and iv_rhs:
        parts.append(f"{iv_lhs.strip()} ~ {iv_rhs.strip()}")
    return " | ".join(parts)


def _clean_cluster_terms(terms: List[str]) -> List[str]:
    """Strip R one-sided-formula sugar (``~id``) and quotes so a cluster name
    matches a real column: ``~id`` / ``"id"`` → ``id``."""
    out = []
    for t in terms:
        if not t:
            continue
        t = _strip_quotes(t.strip()).lstrip("~").strip()
        if t:
            out.append(t)
    return out


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
    clusters = _clean_cluster_terms(clusters)

    # Target the real, registered ``sp.feols`` (there is no ``sp.fixest``
    # callable — it is a package). feols carries FE / IV inside the formula.
    fml = _pyfixest_fml(main, fe_terms, iv_lhs, iv_rhs)
    args: Dict[str, Any] = {"fml": fml}
    notes: List[str] = []
    unread_weights = False
    if clusters:
        args["cluster"] = clusters[0]
        if len(clusters) > 1:
            # feols takes a single ``cluster`` kwarg; surface the first and tell
            # the caller how to add the rest rather than silently dropping
            # clustering dimensions.
            joined = " + ".join(clusters)
            notes.append(
                f"Multiway clustering on {clusters}: sp.feols applies "
                f"cluster={clusters[0]!r}; for the full multiway VCOV pass "
                f"vcov={{'CRV1': {joined!r}}} explicitly."
            )
    if "vcov" in kw:
        notes.append(
            f"R `vcov={kw['vcov']}` not auto-translated; check sp.feols "
            f"`vcov=` / `cluster=` options."
        )
    if "weights" in kw:
        # fixest takes a one-sided formula (``~w``) or a vector; only a
        # column name can be carried over.
        w = _column(_strip_quotes(kw["weights"]).lstrip("~").strip())
        if w:
            args["weights"] = w
        else:
            notes.append(
                f"R `weights = {kw['weights']}` is an expression, not a "
                "column; compute it as a column and pass weights=<column>."
            )
            unread_weights = True
    code_pairs = [repr(fml), "data=df"]
    if "cluster" in args:
        code_pairs.append(f"cluster={args['cluster']!r}")
    if "weights" in args:
        code_pairs.append(f"weights={args['weights']!r}")
    python = f"sp.feols({', '.join(code_pairs)})"
    out = _emit("feols", args, python, notes)
    if unread_weights:
        out["untranslated_arguments"] = ["weights"]
    return out


def _h_felm(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """felm uses the same | structure as feols but is from the lfe package."""
    return _h_feols(pos, kw, _)  # delegate


class _Tracked(Dict[str, str]):
    """Keyword arguments of the R call that remember which were looked at."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.seen: set = set()

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        self.seen.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> str:
        self.seen.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.seen.add(key)
        return super().__contains__(key)


_R_HINTS = {
    "subset": "filter the data frame first (data=df[...])",
    "na.action": "StatsPAI drops incomplete rows, as na.omit does",
    "offset": "sp.glm takes offset=<column>",
    "weights": "compute the weights as a column and pass weights=<column>",
    "effect": "plm effect='twoways' adds time effects; the call above has "
    "entity effects only",
    "bstrap": "did::att_gt bootstraps by default; sp.callaway_santanna "
    "reports analytical standard errors unless bstrap=True",
}


def _note_untranslated(
    out: Dict[str, Any], kw: Dict[str, str], keys: List[str]
) -> None:
    listed = out.setdefault("untranslated_arguments", [])
    for k in keys:
        if k in listed:
            continue
        listed.append(k)
        hint = _R_HINTS.get(k)
        out["notes"].append(
            f"R argument `{k} = {dict.__getitem__(kw, k)}` was not translated"
            + (f"; {hint}." if hint else ".")
        )
    if not listed:
        del out["untranslated_arguments"]


def _unread(
    out: Dict[str, Any], kw: Dict[str, str], read: Tuple[str, ...]
) -> Dict[str, Any]:
    """Record R arguments the handler did not carry into the call.

    An argument that changes the estimate (``subset``, ``offset``,
    ``na.action``) must not vanish: it is listed in
    ``untranslated_arguments`` with a note, the way ``sp.from_stata`` lists
    ``untranslated_options``.
    """
    left = [k for k in kw if k not in read and k not in ("data", "__fn__")]
    _note_untranslated(out, kw, left)
    return out


def _column(expr: Optional[str]) -> Optional[str]:
    """A bare column name, or None for anything that is an R expression.

    ``weights = w`` names a column and translates; ``weights =
    as.numeric(w)`` or ``weights = 1 / p`` does not, and is then reported
    as untranslated instead of being passed on as a column called
    ``'1 / p'``.
    """
    if not expr:
        return None
    name = _strip_quotes(expr).strip()
    return name if re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", name) else None


def _h_lm(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("lm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    args: Dict[str, Any] = {"formula": formula}
    code = f"sp.regress({formula!r}, data=df"
    read = ["formula"]
    weights = _column(kw.get("weights"))
    if weights:
        read.append("weights")
        args["weights"] = weights
        code += f", weights={weights!r}"
    return _unread(_emit("regress", args, code + ")"), kw, tuple(read))


def _h_did(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """Brantly Callaway's did::att_gt R API → sp.callaway_santanna."""
    yname = _strip_quotes(kw.get("yname", ""))
    gname = _strip_quotes(kw.get("gname", ""))
    tname = _strip_quotes(kw.get("tname", ""))
    idname = _strip_quotes(kw.get("idname", ""))
    missing = [
        n
        for n, v in (
            ("yname", yname),
            ("gname", gname),
            ("tname", tname),
            ("idname", idname),
        )
        if not v
    ]
    if missing:
        return _emit_error(f"R `did::att_gt` translation needs {missing} kwargs.")
    args: Dict[str, Any] = {
        "y": yname,
        "g": gname,
        "t": tname,
        "i": idname,
    }
    notes: List[str] = []
    est_method = _strip_quotes(kw.get("est_method", "dr"))
    if est_method not in {"dr", "ipw", "reg"}:
        return _emit_error(
            f"did::att_gt est_method {est_method!r} has no counterpart",
            command="att_gt",
        )
    args["estimator"] = est_method
    # Arguments that change the estimate. R's defaults are written out
    # where StatsPAI's differ (base_period).
    control = _strip_quotes(kw.get("control_group", "nevertreated"))
    if control not in ("nevertreated", "notyettreated"):
        return _emit_error(
            f"did::att_gt control_group {control!r} is not one of "
            "'nevertreated', 'notyettreated'",
            command="att_gt",
        )
    args["control_group"] = control
    base = _strip_quotes(kw.get("base_period", "varying"))
    if base not in ("varying", "universal"):
        return _emit_error(
            f"did::att_gt base_period {base!r} is not 'varying' or 'universal'",
            command="att_gt",
        )
    args["base_period"] = base
    if "base_period" not in dict.keys(kw):
        notes.append(
            "did::att_gt defaults to base_period = 'varying'; it is written "
            "out because sp.callaway_santanna defaults to 'universal'."
        )
    anticipation = _strip_quotes(kw.get("anticipation", "0"))
    if anticipation.isdigit():
        if int(anticipation):
            args["anticipation"] = int(anticipation)
    else:
        return _emit_error(
            f"did::att_gt anticipation {anticipation!r} is not an integer",
            command="att_gt",
        )
    xformla = _strip_quotes(kw.get("xformla", "")).replace(" ", "")
    if xformla and xformla not in ("NULL", "~1"):
        terms = xformla.lstrip("~").split("+")
        if all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", t) for t in terms):
            args["x"] = terms
        else:
            return _emit_error(
                f"did::att_gt xformla {xformla!r} has terms that are not plain "
                "columns; build them as columns first",
                command="att_gt",
            )
    for r_name, sp_name in (("weightsname", "weights"), ("clustervars", "clustervars")):
        value = (
            _column(kw.get(r_name)) if kw.get(r_name) not in (None, "NULL") else None
        )
        if value:
            args[sp_name] = value
        elif kw.get(r_name) not in (None, "NULL"):
            return _emit_error(
                f"did::att_gt {r_name} = {kw[r_name]} is not a single column",
                command="att_gt",
            )
    for flag in ("panel", "allow_unbalanced_panel"):
        if flag in kw:
            args[flag] = _strip_quotes(kw[flag]).upper() in ("TRUE", "T")
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    python = f"sp.callaway_santanna(data=df, {pairs})"
    return _emit("callaway_santanna", args, python, notes)


_R_FAMILY = re.compile(
    r"""^(?:stats::)?(?P<name>[A-Za-z.]+)\s*
        (?:\(\s*(?:link\s*=\s*)?(?P<link>["']?[A-Za-z]*["']?)\s*\))?$""",
    re.VERBOSE,
)
_R_DEFAULT_LINK = {
    "binomial": "logit",
    "quasibinomial": "logit",
    "poisson": "log",
    "quasipoisson": "log",
    "gaussian": "identity",
    "gamma": "inverse",
    "inverse.gaussian": "1/mu^2",
}


def _h_glm(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """R ``glm(y ~ x, family = binomial(), data = df)`` → ``sp.logit`` /
    ``sp.probit`` / ``sp.poisson`` when the family and link resolve to one
    of those, ``sp.glm`` otherwise.

    The family is read the ways R accepts it: ``binomial``,
    ``binomial()``, ``"binomial"``, ``binomial("probit")``,
    ``binomial(link = "probit")``. ``quasibinomial`` and ``quasipoisson``
    give the same coefficients as their parents and differ only in the
    dispersion behind the standard errors, which a note says.
    """
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("glm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    raw = _strip_quotes(kw["family"]).strip() if kw.get("family") else "gaussian"
    m = _R_FAMILY.match(raw)
    if m is None:
        return _emit_error(
            f"glm family {raw!r} was not understood", command="glm", family=raw
        )
    name = m.group("name").lower()
    link = _strip_quotes(m.group("link") or "") or _R_DEFAULT_LINK.get(name)
    notes: List[str] = []
    quasi = name.startswith("quasi") and name != "quasi"
    if quasi:
        name = name[len("quasi") :]
        notes.append(
            f"quasi{name} has the coefficients of {name}; R scales its "
            "standard errors by the estimated dispersion. Add "
            "robust='hc1' for standard errors that do not assume the "
            "variance function."
        )
    weights = _column(kw.get("weights"))
    extra = f", weights={weights!r}" if weights else ""
    read = ("formula", "family") + (("weights",) if weights else ())

    def _out(tool: str, args: Dict[str, Any], code: str) -> Dict[str, Any]:
        if weights:
            args["weights"] = weights
        return _unread(_emit(tool, args, code, notes), kw, read)

    if name == "binomial" and link in ("logit", "probit"):
        return _out(
            link, {"formula": formula}, f"sp.{link}({formula!r}, data=df{extra})"
        )
    if name == "poisson" and link == "log":
        return _out(
            "poisson", {"formula": formula}, f"sp.poisson({formula!r}, data=df{extra})"
        )
    family = {"inverse.gaussian": "inverse_gaussian"}.get(name, name)
    if family not in (
        "gaussian",
        "binomial",
        "poisson",
        "gamma",
        "inverse_gaussian",
    ):
        return _emit_error(
            f"glm family {raw!r} has no sp.glm counterpart", command="glm", family=raw
        )
    args: Dict[str, Any] = {"formula": formula, "family": family}
    code = f"sp.glm({formula!r}, data=df, family={family!r}"
    if link and link != _R_DEFAULT_LINK.get(name):
        args["link"] = link
        code += f", link={link!r}"
    return _out("glm", args, code + extra + ")")


def _parse_lme4_formula(formula: str) -> Optional[Tuple[str, List[str], str]]:
    """Parse an lme4 mixed-model formula into ``(y, x_fixed, group)``.

    ``y ~ x1 + x2 + (1 | group)`` → ``("y", ["x1", "x2"], "group")``. Returns
    ``None`` if there is no ``~`` or no ``(... | group)`` random-effect term.
    Only the first grouping factor is captured (sp.mixed / sp.meglm take a
    single ``group``); a second one is surfaced as a note by the caller.
    """
    if "~" not in formula:
        return None
    lhs, rhs = formula.split("~", 1)
    y = lhs.strip()
    re_terms = re.findall(r"\(([^)]*\|[^)]*)\)", rhs)
    if not re_terms:
        return None
    # First random-effect grouping factor: right side of the first ``|``.
    group = re_terms[0].split("|", 1)[1].strip()
    # Fixed part = rhs with every (... | ...) term removed.
    fixed_rhs = re.sub(r"\([^)]*\|[^)]*\)", "", rhs)
    x_fixed = [
        t.strip()
        for t in fixed_rhs.split("+")
        if t.strip() and t.strip() not in ("1", "0", "")
    ]
    return y, x_fixed, group


def _h_glmer(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """R `lme4::glmer(y ~ x + (1|group), family=binomial, data=df)` →
    ``sp.meglm`` (or ``sp.melogit`` for a binomial family)."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("glmer requires a formula as the first argument")
    parsed = _parse_lme4_formula(_strip_quotes(formula))
    if parsed is None:
        return _emit_error(
            "glmer needs a mixed-model formula `y ~ x + (1|group)`",
            command="glmer",
        )
    y, x_fixed, group = parsed
    family = (
        _strip_quotes(kw.get("family", "gaussian")) if kw.get("family") else "gaussian"
    )
    # Binomial → the dedicated sp.melogit; any other family → the general
    # sp.meglm(family=...). Both take y / x_fixed / group explicitly.
    if "binomial" in family.lower():
        args: Dict[str, Any] = {"y": y, "x_fixed": x_fixed, "group": group}
        python = f"sp.melogit(data=df, y={y!r}, x_fixed={x_fixed!r}, group={group!r})"
        return _emit("melogit", args, python)
    args = {"y": y, "x_fixed": x_fixed, "group": group, "family": family}
    python = (
        f"sp.meglm(data=df, y={y!r}, x_fixed={x_fixed!r}, "
        f"group={group!r}, family={family!r})"
    )
    return _emit("meglm", args, python)


def _h_lmer(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """R `lme4::lmer(y ~ x + (1|group), data=df)` → ``sp.mixed`` (Gaussian
    linear mixed model)."""
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("lmer requires a formula as the first argument")
    parsed = _parse_lme4_formula(_strip_quotes(formula))
    if parsed is None:
        return _emit_error(
            "lmer needs a mixed-model formula `y ~ x + (1|group)`",
            command="lmer",
        )
    y, x_fixed, group = parsed
    args: Dict[str, Any] = {"y": y, "x_fixed": x_fixed, "group": group}
    python = f"sp.mixed(data=df, y={y!r}, x_fixed={x_fixed!r}, group={group!r}"
    reml = _strip_quotes(kw.get("REML", "TRUE")).upper()
    if reml in ("FALSE", "F"):
        args["method"] = "ml"
        python += ", method='ml'"
    return _emit("mixed", args, python + ")")


def _h_plm(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """R `plm(y ~ x, data=df, model='within', index=c('id','t'))` →
    ``sp.panel(data=df, formula=..., entity=..., time=..., method=...)``.

    sp.panel's signature is keyword-only for the model side — passing the
    formula positionally would collide with the implicit ``data`` parameter.
    The old code emitted the formula as a positional arg, which silently
    shadowed ``data`` and produced a dead on-ramp.
    """
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("plm requires a formula as the first argument")
    formula = _strip_quotes(formula)
    model = (
        _strip_quotes(kw.get("model", "within")).lower()
        if kw.get("model")
        else "within"
    )
    index = kw.get("index")
    panel_keys: List[str] = []
    if index:
        panel_keys = _parse_c_vector(index)
    if not panel_keys:
        return _emit_error(
            "plm needs `index=c(id_col)` (and optionally `t_col`) to identify "
            "the panel structure; sp.panel(entity=..., time=...) takes those "
            "column names directly.",
            command="plm",
        )
    args: Dict[str, Any] = {"formula": formula, "entity": panel_keys[0]}
    if len(panel_keys) > 1:
        args["time"] = panel_keys[1]
    args["method"] = model
    code_pairs = ["data=df", f"formula={formula!r}", f"entity={panel_keys[0]!r}"]
    if "time" in args:
        code_pairs.append(f"time={args['time']!r}")
    code_pairs.append(f"method={model!r}")
    python = f"sp.panel({', '.join(code_pairs)})"
    return _emit("panel", args, python)


def _h_matchit(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """R ``MatchIt::matchit(treat ~ x1 + x2, data = df)`` → ``sp.match``.

    The left-hand side of a ``matchit`` formula is the *treatment*;
    ``matchit`` only builds the matched sample and never sees an outcome.
    ``sp.match`` matches and estimates in one call, so it needs the outcome
    column as well: the translation leaves ``y`` out, lists it under
    ``missing_arguments`` and says so in a note. (Until 1.38.0 the
    treatment was passed as the outcome too, which asked for the effect of
    the treatment on itself.)

    ``matchit`` matches without replacement and targets the ATT by
    default; both are written out because ``sp.match`` matches with
    replacement by default.
    """
    formula = pos[0] if pos else kw.get("formula")
    if not formula:
        return _emit_error("matchit requires a formula as the first argument")
    parts = [s.strip() for s in _strip_quotes(formula).split("~", 1)]
    if len(parts) != 2:
        return _emit_error(
            "matchit expects a two-sided formula `treat ~ x1 + x2`", command="matchit"
        )
    treat, rhs = parts
    covariates = [c.strip() for c in rhs.split("+") if c.strip()]
    if not treat or not covariates:
        return _emit_error(
            "matchit formula must declare the treatment (`treat ~`) and at "
            "least one covariate; got " + formula,
            command="matchit",
        )
    method = (
        _strip_quotes(kw.get("method", "nearest")).lower()
        if kw.get("method")
        else "nearest"
    )
    # Map MatchIt method names → sp.match's actual method names.
    method_alias = {
        "nearest": "nearest",
        "exact": "subclass",  # sp.match's exact-style falls under subclass
        "cem": "cem",
        "subclass": "subclass",
        "optimal": "optimal",
        "full": "full",
        "genetic": "genetic",
        "mahalanobis": "mahalanobis",
        "cardinality": "cardinality",
        "cbps": "cbps",
    }
    sp_method = method_alias.get(method, method)
    args: Dict[str, Any] = {
        "treat": treat,
        "covariates": covariates,
        "method": sp_method,
    }
    notes: List[str] = [
        "matchit() only matches; sp.match also estimates the effect and "
        "needs the outcome column. Add y='<outcome>'."
    ]
    if method != sp_method:
        notes.append(
            f"MatchIt method '{method}' mapped to sp.match method='{sp_method}'."
        )
    read = ["formula", "method", "distance"]
    if sp_method == "nearest":
        read += ["replace", "ratio", "estimand"]
        replace = _strip_quotes(kw.get("replace", "FALSE")).upper() in ("TRUE", "T")
        args["replace"] = replace
        ratio = _strip_quotes(kw.get("ratio", "1"))
        if ratio.isdigit():
            args["n_matches"] = int(ratio)
        else:
            read.remove("ratio")
        estimand = _strip_quotes(kw.get("estimand", "ATT")).upper()
        if estimand in ("ATT", "ATC", "ATE"):
            args["estimand"] = estimand
        else:
            read.remove("estimand")
    distance = kw.get("distance")
    if distance:
        args["distance"] = _strip_quotes(distance)
        notes.append("sp.match('distance=...') may not apply to all methods.")
    code_pairs = ["data=df"] + [f"{k}={v!r}" for k, v in args.items()]
    out = _emit("match", args, f"sp.match({', '.join(code_pairs)})", notes)
    out["missing_arguments"] = ["y"]
    out = _unread(out, kw, tuple(read))
    if "caliper" in kw:
        out["notes"].append(
            "MatchIt's caliper is in standard deviations of the distance "
            "measure (std.caliper = TRUE); sp.match(caliper=) is on the raw "
            "scale unless caliper_scale= says otherwise."
        )
    return out


def _h_synth(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
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
        command="synth",
    )


def _h_staggered(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """Roth & Sant'Anna's staggered:: R API -> the design-based estimators.

    ``staggered`` names its columns i / t / g / y directly, and expresses the
    plug-in as ``beta = 1`` where StatsPAI says ``efficient=False``. The
    never-treated coding differs too: R demands ``g = Inf`` and silently
    misreads ``g = 0``, while StatsPAI accepts either — so a translated call
    is safe even when the R original would not have been.
    """
    fn = kw.pop("__fn__", "staggered")
    y = _strip_quotes(kw.get("y", ""))
    g = _strip_quotes(kw.get("g", ""))
    t = _strip_quotes(kw.get("t", ""))
    i = _strip_quotes(kw.get("i", ""))
    missing = [n for n, v in (("y", y), ("g", g), ("t", t), ("i", i)) if not v]
    if missing:
        return _emit_error(f"R `staggered::{fn}` translation needs {missing} kwargs.")

    target = {
        "staggered": "staggered_rollout",
        "staggered_cs": "staggered_cs",
        "staggered_sa": "staggered_sa",
    }[fn]
    args: Dict[str, Any] = {"y": y, "g": g, "t": t, "i": i}

    estimand = _strip_quotes(kw.get("estimand", "simple"))
    if estimand in {"simple", "cohort", "calendar", "eventstudy"}:
        args["estimand"] = estimand
    if estimand == "eventstudy" and "eventTime" in kw:
        args["event_time"] = kw["eventTime"]

    extra = ""
    if target == "staggered_rollout":
        # beta = 1 is the plug-in; beta absent (NULL) is the efficient default.
        if _strip_quotes(kw.get("beta", "")) == "1":
            args["efficient"] = False
            extra += ", efficient=False"
        if _strip_quotes(kw.get("use_last_treated_only", "")).upper() == "TRUE":
            args["use_last_treated_only"] = True
            extra += ", use_last_treated_only=True"
        if _strip_quotes(kw.get("use_DiD_A0", "")).upper() == "FALSE":
            args["use_did_a0"] = False
            extra += ", use_did_a0=False"
    if _strip_quotes(kw.get("compute_fisher", "")).upper() == "TRUE":
        args["fisher"] = True
        extra += ", fisher=True"

    python = (
        f"sp.{target}(data=df, y={y!r}, g={g!r}, t={t!r}, i={i!r}, "
        f"estimand={args.get('estimand', 'simple')!r}{extra})"
    )
    return _emit(target, args, python)


def _h_didff(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """Sant'Anna's didFF:: R API -> sp.functional_form_test / distributional_did."""
    fn = kw.pop("__fn__", "didFF")
    yname = _strip_quotes(kw.get("yname", ""))
    gname = _strip_quotes(kw.get("gname", ""))
    tname = _strip_quotes(kw.get("tname", ""))
    idname = _strip_quotes(kw.get("idname", ""))
    missing = [
        n
        for n, v in (
            ("yname", yname),
            ("gname", gname),
            ("tname", tname),
            ("idname", idname),
        )
        if not v
    ]
    if missing:
        return _emit_error(f"R `didFF::{fn}` translation needs {missing} kwargs.")

    distributional = (
        fn == "distDD" or _strip_quotes(kw.get("distDD", "")).upper() == "TRUE"
    )
    target = "distributional_did" if distributional else "functional_form_test"
    args: Dict[str, Any] = {"y": yname, "g": gname, "t": tname, "i": idname}

    extra = ""
    if "nbins" in kw:
        args["n_bins"] = kw["nbins"]
        extra += f", n_bins={kw['nbins']}"
    if "weightsname" in kw:
        args["weights"] = _strip_quotes(kw["weightsname"])
        extra += f", weights={args['weights']!r}"
    aggte_type = _strip_quotes(kw.get("aggte_type", ""))
    if aggte_type in {"simple", "group", "dynamic", "calendar"}:
        args["aggregation"] = aggte_type
        extra += f", aggregation={aggte_type!r}"

    python = (
        f"sp.{target}(data=df, y={yname!r}, g={gname!r}, "
        f"t={tname!r}, i={idname!r}{extra})"
    )
    return _emit(target, args, python)


R_FUNCTION_MAP: Dict[
    str, Callable[[List[str], Dict[str, str], List[str]], Dict[str, Any]]
] = {
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
    "staggered": _h_staggered,
    "staggered_cs": _h_staggered,
    "staggered_sa": _h_staggered,
    "didFF": _h_didff,
    "distDD": _h_didff,
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

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.from_r("feols(y ~ x | id, data = df)")
    >>> out["ok"]
    True
    >>> out["python_code"]
    "sp.feols('y ~ x | id', data=df)"
    """
    parsed = _parse_call(line)
    if parsed is None:
        return _emit_error(
            "R input did not match `fn(...)` shape. Pass one R "
            "expression at a time (no assignment, no piping)."
        )
    fn, pos, kw = parsed
    handler = R_FUNCTION_MAP.get(fn)
    if handler is None:
        from difflib import get_close_matches

        suggestions = get_close_matches(
            fn, list(R_FUNCTION_MAP.keys()), n=5, cutoff=0.55
        )
        return _emit_error(
            f"unsupported R function {fn!r}", command=fn, suggestions=suggestions
        )
    # A few handlers serve several R entry points that differ only in which
    # StatsPAI function they map to (staggered / staggered_cs / staggered_sa,
    # didFF / distDD), so they need to know which name was called.
    if handler in (_h_staggered, _h_didff):
        kw = {**kw, "__fn__": fn}
    tracked = _Tracked(kw)
    out = handler(pos, tracked, [])
    if out.get("ok"):
        # Whatever the handler never looked at cannot be in the call it
        # emitted. Say so instead of returning a clean "ok".
        never_read = [
            k for k in kw if k not in tracked.seen and k not in ("data", "__fn__")
        ]
        if never_read:
            _note_untranslated(out, kw, never_read)
    return out


__all__ = ["from_r", "R_FUNCTION_MAP"]
