"""``weakiv`` and ``acreg``: weak-IV-robust tests and spatial HAC.

``weakiv`` (Finlay, Magnusson and Schaffer, SSC) runs the Anderson-Rubin,
Kleibergen K, J, K-J and CLR tests and inverts them for confidence sets.
``sp.weakrobust`` computes the same statistics in weakiv's LM form when
``test_vcov`` or ``cluster`` is given (``diagnostics/_weakiv_lm.py``); the
translation therefore always writes ``test_vcov`` out. Two forms:

* ``weakiv ivreg2 y x (d = z1 z2), robust`` / ``weakiv ivregress 2sls ...``
  names the model; the estimation options (``robust``, ``cluster()``)
  pick the variance of the tests.
* bare ``weakiv [, null() level()]`` after ``ivreg2`` / ``ivregress``
  tests the model before it. ``sp.from_stata`` cannot see that model, so
  it reports ``model`` as not carried over; ``sp.stata`` fills ``y``,
  ``endog``, ``instruments``, ``exog`` and the variance from the IV
  regression before it and runs the tests on its estimation sample.

On the fixture of ``tests/reference_parity/test_ivreport_stata_parity.py``
the AR / K / J / CLR statistics and p-values agree with weakiv 2.4.07 to
1e-9 under i.i.d., robust and cluster variances. Confidence sets are exact
inversions of the reported tests; weakiv's are grid-interpolated (robust,
cluster) or the closed-form Wald-type interval (i.i.d.), so set endpoints
differ in the third or fourth digit (documented in docs/dev/modules/iv.md).

``acreg`` (Colella, Lalive, Sakalli and Thoenig) with ``spatial
latitude() longitude() dist()`` is Conley spatial HAC with acreg's planar
distance (111 km per degree), which ``sp.regress`` / ``sp.iv`` /
``sp.hdfe_ols`` reproduce to 1e-12. Mapped: OLS and 2SLS with the uniform
kernel; OLS with absorbed effects (``pfe1()`` / ``pfe2()``), the Bartlett
kernel, and the panel options ``id() time() lag() lagdist() hac``. Not
mapped (the error names the sp.conley call that computes them): 2SLS with
the Bartlett kernel, panel options or absorbed effects, and OLS with the
Bartlett kernel or panel options but no absorbed effect. Never mapped:
``network``, ``links_mat()``, ``dist_mat()``, ``cluster()``,
``weights()``, ``small``, ``ak0``, weights.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

from ._stata import _emit, _emit_error, _parse_iv_varlist
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "LITERAL", "fill_from_previous_iv"]


# ---------------------------------------------------------------------------
# weakiv
# ---------------------------------------------------------------------------

#: weakiv's grid / simulation controls: our sets invert the tests exactly
#: and the CLR p-value is the exact conditional integral, so they change
#: nothing here (they change weakiv's own numbers slightly)
_WEAKIV_GRID = (
    "gridpoints",
    "gridmin",
    "gridmax",
    "gridmult",
    "gridlimits",
    "usegrid",
    "grid",
    "gridlist",
    "clrsims",
)


def _iv_tokens(words: List[str]) -> List[str]:
    """Split ``(x``, ``=``, ``z)`` style words into ``(``/``=``/``)`` tokens."""
    text = " ".join(words)
    text = re.sub(r"([()=])", r" \1 ", text)
    return text.split()


def _numbers(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    try:
        return [float(t) for t in str(raw).replace(",", " ").split()]
    except ValueError:
        return None


def _variance(opts: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(test_vcov, cluster, error) from robust / cluster() / vce()."""
    vce = str(opts.get("vce") or "").split()
    cluster = opts.get("cluster")
    if vce:
        head = vce[0].lower()
        if head in ("cluster", "cl") and len(vce) >= 2:
            cluster = " ".join(vce[1:])
        elif head in ("robust", "r", "hc1"):
            return "robust", None, None
        elif head in ("unadjusted", "ols"):
            return "classic", None, None
        else:
            return None, None, f"vce({' '.join(vce)}) is not translated"
    if cluster:
        names = str(cluster).split()
        if len(names) != 1:
            return (
                None,
                None,
                (
                    "two-way clustering in weakiv is not translated; "
                    "sp.weakrobust clusters on one variable"
                ),
            )
        return "cluster", names[0], None
    if "robust" in opts:
        return "robust", None, None
    return "classic", None, None


def _h_weakiv(cmd: StataCommand) -> Dict[str, Any]:
    """``weakiv [ivreg2|ivregress 2sls ...] [, options]`` -> ``sp.weakrobust``."""
    tokens = _iv_tokens(list(cmd.varlist))
    if any(t.startswith("[") for t in tokens):
        return _emit_error(
            "weakiv with weights is not translated: sp.weakrobust has no " "weights.",
            command="weakiv",
            suggestions=[],
        )
    opts = cmd.options
    args: Dict[str, Any] = {}
    lost: List[str] = []
    notes: List[str] = []
    sem: List[str] = [
        "weakiv's statistics are the LM-form AR / K / J / K-J / CLR tests; "
        "sp.weakrobust gives them when test_vcov or cluster is set, which "
        "the call always does (agreement with weakiv 2.4.07 to 1e-9).",
        "Confidence sets invert the reported tests exactly; weakiv "
        "interpolates them on a grid (robust, cluster) or reports the "
        "closed-form Wald-type interval (i.i.d.), so endpoints can differ "
        "in the third or fourth digit. The CLR p-value is the exact "
        "conditional integral (weakiv: Simpson rule for one regressor, "
        "independent simulation of AR and K for several).",
    ]
    previous = not tokens
    if not previous:
        est = tokens[0].lower()
        rest = tokens[1:]
        if est == "ivregress":
            if not rest or rest[0].lower() != "2sls":
                return _emit_error(
                    "weakiv ivregress: only `ivregress 2sls` is translated "
                    "(the weak-IV-robust tests do not use the LIML / GMM "
                    "estimate, but their Wald rows would differ).",
                    command="weakiv",
                    suggestions=[],
                )
            rest = rest[1:]
        elif est != "ivreg2":
            return _emit_error(
                f"weakiv {tokens[0]}: only weakiv after ivreg2 / ivregress "
                "2sls is translated (sp.weakrobust is the linear IV model).",
                command="weakiv",
                suggestions=[],
            )
        parsed = _parse_iv_varlist(rest, "weakiv")
        if isinstance(parsed, dict):
            return parsed
        y, exog, endog, instruments = parsed
        args["y"] = y
        args["endog"] = endog[0] if len(endog) == 1 else list(endog)
        args["instruments"] = list(instruments)
        if exog:
            args["exog"] = list(exog)
        test_vcov, cluster, err = _variance(opts)
        if err is not None:
            return _emit_error(f"weakiv: {err}.", command="weakiv", suggestions=[])
        if cluster is not None:
            args["cluster"] = cluster
        else:
            args["test_vcov"] = test_vcov
    else:
        notes.append(
            "weakiv after an estimation command: y, endog, instruments, exog "
            "and the variance are those of the ivreg2 / ivregress 2sls fit "
            "before it; sp.stata fills them in and runs the tests on that "
            "fit's estimation sample."
        )
        lost.append("model")

    null = opts.get("null")
    if null is not None:
        vals = _numbers(null)
        if not vals:
            lost.append("null")
        else:
            args["h0"] = vals[0] if len(vals) == 1 else vals
    for key in ("levellist", "level"):
        raw = opts.get(key)
        if raw is None:
            continue
        vals = _numbers(raw)
        if vals is None or len(vals) != 1:
            lost.append(key)
            notes.append(f"{key}({raw}): one confidence level per call is translated.")
        else:
            args["alpha"] = round(1 - vals[0] / 100.0, 10)
    for key in ("subset", "project"):
        raw = opts.get(key)
        if raw is not None:
            names = str(raw).split()
            if names == ["_all"]:
                lost.append(key)
                notes.append(f"{key}(_all) is not translated; list the names.")
            else:
                args[key] = names[0] if len(names) == 1 else names
    if "lm" in opts:
        sem.append("lm is weakiv's default form of the tests.")
    for key in _WEAKIV_GRID:
        if key in opts:
            sem.append(
                f"{key}() only steers weakiv's grid / simulation; the sets "
                "here are exact inversions and the CLR p-value is exact."
            )
    # `small` and `md` change weakiv's statistics; they are left unread so
    # the option report lists them as not carried over.

    shown = []
    if previous:
        shown += ["y=<y>", "endog=<endog>", "instruments=<instruments>", "exog=<exog>"]
    shown += [f"{k}={v!r}" for k, v in args.items()]
    code = f"sp.weakrobust(df, {', '.join(shown)})"
    out = _emit("weakrobust", args, code, notes, semantics=sem)
    out["untranslated_options"] = lost
    if previous:
        out["from_previous_iv"] = True
    return out


def fill_from_previous_iv(
    last_call: Optional[Dict[str, Any]],
) -> Union[Dict[str, Any], str]:
    """``weakrobust`` arguments of the IV fit before a bare ``weakiv``.

    Returns the arguments, or a message saying why the fit before it cannot
    be tested this way.
    """
    if not last_call or last_call.get("tool") not in ("iv", "ivreg"):
        tool = (last_call or {}).get("tool")
        return (
            "weakiv needs an ivreg2 / ivregress 2sls fit before it "
            f"(the previous estimation command maps to {tool!r})"
        )
    a = dict(last_call.get("arguments") or {})
    method = str(a.get("method") or "2sls").lower()
    if method != "2sls" or a.get("absorb"):
        return "weakiv is translated after 2SLS without absorbed effects only"
    # the ivreg2 / ivregress translation writes small=False unless the
    # line had `small` (sp.iv's own default is small=True)
    if a.get("small", True) is not False:
        return (
            "the IV fit before it used `small`, which changes weakiv's "
            "statistics; sp.weakrobust has no such correction"
        )
    if a.get("weights") is not None:
        return "sp.weakrobust has no weights"
    m = re.fullmatch(
        r"\s*(\w+)\s*~\s*(.*?)\(\s*([\w\s+]+?)\s*~\s*([\w\s+]+?)\s*\)\s*(.*)",
        str(a.get("formula") or ""),
    )
    if m is None:
        return f"cannot read the IV formula {a.get('formula')!r}"
    y = m.group(1)
    exog = [t.strip() for t in (m.group(2) + " " + m.group(5)).split("+")]
    exog = [t for t in exog if t]
    endog = [t.strip() for t in m.group(3).split("+") if t.strip()]
    inst = [t.strip() for t in m.group(4).split("+") if t.strip()]
    if any(not re.fullmatch(r"\w+", t) for t in exog + endog + inst):
        return "factor or expression terms in the IV fit are not translated"
    out: Dict[str, Any] = {
        "y": y,
        "endog": endog[0] if len(endog) == 1 else endog,
        "instruments": inst,
    }
    if exog:
        out["exog"] = exog
    cluster = a.get("cluster")
    robust = str(a.get("robust") or a.get("vce") or "nonrobust").lower()
    if cluster is not None:
        if not isinstance(cluster, str):
            return "two-way clustering is not translated for weakiv"
        out["cluster"] = cluster
    elif robust in ("hc1", "robust", "hc0"):
        out["test_vcov"] = "robust"
    elif robust in ("nonrobust", "classic", "unadjusted"):
        out["test_vcov"] = "classic"
    else:
        return f"the IV fit's variance {robust!r} has no weakiv counterpart"
    return out


# ---------------------------------------------------------------------------
# acreg
# ---------------------------------------------------------------------------


def _acreg_refuse(why: str, code: str) -> Dict[str, Any]:
    return _emit_error(
        f"acreg: {why}. The same variance is "
        f"{code} (acreg's planar distance and kernels).",
        command="acreg",
        suggestions=["conley"],
    )


def _h_acreg(cmd: StataCommand) -> Dict[str, Any]:
    """``acreg y x [(d = z)], spatial latitude() longitude() dist() ...``."""
    opts = cmd.options
    if getattr(cmd, "weight", None) is not None:
        return _emit_error(
            "acreg with weights is not translated.", command="acreg", suggestions=[]
        )
    for name in ("network", "links_mat", "dist_mat", "cluster", "weights"):
        if name in opts:
            return _emit_error(
                f"acreg {name}(): only the spatial form (spatial latitude() "
                "longitude() dist()) is translated; arbitrary-cluster and "
                "network variances have no sp counterpart.",
                command="acreg",
                suggestions=[],
            )
    if "spatial" not in opts:
        return _emit_error(
            "acreg without `spatial` is not translated (network / cluster " "forms).",
            command="acreg",
            suggestions=[],
        )
    lat, lon = opts.get("latitude"), opts.get("longitude")
    raw_dist = opts.get("distcutoff")
    if not lat or not lon or raw_dist is None:
        return _emit_error(
            "acreg, spatial needs latitude(), longitude() and dist().",
            command="acreg",
            suggestions=[],
        )
    try:
        cutoff = float(str(raw_dist).strip())
    except ValueError:
        return _emit_error(f"acreg: dist({raw_dist}) is not a number.", command="acreg")
    if cutoff <= 0:
        return _emit_error("acreg: dist() must be positive.", command="acreg")
    lat, lon = str(lat).split()[0], str(lon).split()[0]
    kernel = "bartlett" if "bartlett" in opts else "uniform"
    unit, time = opts.get("id"), opts.get("time")
    raw_lag, raw_lagd = opts.get("lagcutoff"), opts.get("lagdistcutoff")
    hac = "hac" in opts
    pfe = [str(opts[k]).split()[0] for k in ("pfe1", "pfe2") if opts.get(k)]
    if opts.get("pfe2") and not opts.get("pfe1"):
        return _emit_error("acreg: pfe2() requires pfe1().", command="acreg")
    drop_singletons = "dropsingletons" in opts
    opts.get("nbclust")  # only used with cluster(), refused above

    tokens = list(cmd.varlist)
    is_iv = "=" in tokens
    if is_iv:
        parsed = _parse_iv_varlist(tokens, "acreg")
        if isinstance(parsed, dict):
            return parsed
        y, exog, endog, instruments = parsed
        formula = f"{y} ~ "
        if exog:
            formula += " + ".join(exog) + " + "
        formula += f"({' + '.join(endog)} ~ {' + '.join(instruments)})"
    else:
        if not tokens:
            return _emit_error("acreg requires an outcome", command="acreg")
        y, exog = tokens[0], tokens[1:]
        formula = f"{y} ~ " + (" + ".join(exog) if exog else "1")

    panel = unit is not None or time is not None or raw_lag is not None
    lag = lagd = None
    if panel:
        if not (unit and time and raw_lag is not None):
            return _emit_error(
                "acreg: the panel form needs id(), time() and lag() together.",
                command="acreg",
            )
        try:
            lag = int(str(raw_lag).strip())
            lagd = int(str(raw_lagd).strip()) if raw_lagd is not None else 0
        except ValueError:
            return _emit_error(
                "acreg: lag() / lagdist() must be integers.", command="acreg"
            )
        unit, time = str(unit).split()[0], str(time).split()[0]
    elif raw_lagd is not None or hac:
        return _emit_error(
            "acreg: lagdist() / hac need id(), time() and lag().", command="acreg"
        )
    time_kernel = "bartlett" if hac else "uniform"

    sem = [
        f"acreg's planar distance (111 km per degree of latitude, cos(lat) * "
        f"111 per degree of longitude) with a {kernel} kernel within "
        f"{cutoff:g} km.",
    ]
    conley_kw = (
        f"lat={lat!r}, lon={lon!r}, dist_cutoff={cutoff!r}, kernel={kernel!r}, "
        "distance='planar'"
        + (
            f", time={time!r}, lag_cutoff={lag!r}, unit={unit!r}, "
            f"lag_cutoff_cross={lagd!r}, time_kernel={time_kernel!r}"
            if panel
            else ""
        )
    )

    if is_iv:
        fit = f"sp.iv({formula!r}, data=df"
        if pfe:
            fit += f", absorb={pfe!r}"
        fit += ")"
        if pfe or panel or kernel != "uniform":
            return _acreg_refuse(
                "2SLS with the Bartlett kernel, panel options or absorbed "
                "effects is not a single sp call",
                f"sp.conley({fit}, data=df, {conley_kw})",
            )
        args: Dict[str, Any] = {
            "formula": formula,
            "vce": "conley",
            "conley_lat": lat,
            "conley_lon": lon,
            "conley_cutoff": cutoff,
        }
        code = (
            f"sp.iv({formula!r}, data=df, vce='conley', conley_lat={lat!r}, "
            f"conley_lon={lon!r}, conley_cutoff={cutoff!r})"
        )
        sem.append("Normal reference distribution, as acreg (z statistics).")
        out = _emit("iv", args, code, [], semantics=sem)
        return out

    if not pfe:
        if panel or kernel != "uniform":
            return _acreg_refuse(
                "OLS with the Bartlett kernel or panel options and no absorbed "
                "effect is not a single sp call",
                f"sp.conley(sp.regress({formula!r}, data=df), data=df, "
                f"{conley_kw})",
            )
        args = {
            "formula": formula,
            "vce": "conley",
            "conley_lat": lat,
            "conley_lon": lon,
            "conley_cutoff": cutoff,
        }
        code = (
            f"sp.regress({formula!r}, data=df, vce='conley', "
            f"conley_lat={lat!r}, conley_lon={lon!r}, conley_cutoff={cutoff!r})"
        )
        sem.append("Normal reference distribution, as acreg (z statistics).")
        return _emit("regress", args, code, [], semantics=sem)

    rhs = " + ".join(exog) if exog else "1"
    fml = f"{y} ~ {rhs} | {' + '.join(pfe)}"
    args = {
        "formula": fml,
        "vce": "conley",
        "conley_lat": lat,
        "conley_lon": lon,
        "conley_cutoff": cutoff,
        "drop_singletons": drop_singletons,
    }
    if kernel != "uniform":
        args["conley_kernel"] = kernel
    if panel:
        args.update(
            conley_time=time,
            conley_unit=unit,
            conley_lag=lag,
            conley_lag_cross=lagd,
            conley_time_kernel=time_kernel,
        )
    sem.append(
        "pfe1() / pfe2() are absorbed (sp.hdfe_ols); singleton groups are "
        + ("dropped (dropsingletons)." if drop_singletons else "kept, as acreg.")
    )
    sem.append(
        "Standard errors are acreg's; sp.hdfe_ols refers them to t(n - k) "
        "where acreg reports z, so p-values and intervals differ slightly."
    )
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
    code = f"sp.hdfe_ols({fml!r}, data=df, {shown})"
    return _emit("hdfe_ols", args, code, [], semantics=sem)


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "weakiv": _h_weakiv,
    "acreg": _h_acreg,
}

#: weakiv's varlist starts with the estimation command (or is empty), so the
#: generic varlist normalisation is skipped and the handler tokenises it.
LITERAL: FrozenSet[Callable[..., Dict[str, Any]]] = frozenset({_h_weakiv})
