"""Translations of the commands an introductory econometrics course runs
that the other handler modules do not cover.

Tests on a variance and z tests (``sdtest``, ``ztest`` and their immediate
forms), the endogenous-treatment model ``etregress``, the unit-root tests ``pperron`` and ``kpss``, and the
univariate time-series models ``arima`` and ``arch``.

Where Stata's default differs from the sp function's, Stata's is written
out: ``arima`` always has a constant and reports OPG standard errors,
``arch`` reports OPG standard errors, ``kpss`` tests trend stationarity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._stata import _emit, _emit_error, _opt_matches, _split_varlist_y_x, _vce_cluster
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS"]

_ROW_ORDER = (
    "Rows are taken in the DataFrame's order; sort by the time variable of "
    "`tsset` first (sp.stata does)."
)
_EQ = re.compile(r"([^\W\d]\w*)\s*={1,2}\s*(\S+)")


def _kw(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _bad(cmd: StataCommand, message: str) -> Dict[str, Any]:
    return _emit_error(f"{cmd.command}: {message}", command=cmd.command, suggestions=[])


def _number(text: Any) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _level(cmd: StataCommand, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = cmd.options.get("level")
    if raw is None:
        return None
    value = _number(raw)
    if value is None or not 10 <= value < 100:
        return _bad(cmd, f"level({raw}) is not a confidence level")
    args["alpha"] = round(1 - value / 100, 10)
    return None


# ------------------------------------------------------------ sdtest / ztest
def _h_sdtest(cmd: StataCommand) -> Dict[str, Any]:
    """``sdtest y == 5`` / ``sdtest y, by(g)`` / ``sdtest y == x`` ->
    ``sp.sdtest``."""
    text = " ".join(cmd.varlist).strip()
    args: Dict[str, Any] = {}
    eq = _EQ.fullmatch(text)
    if eq:
        args["y"] = eq.group(1)
        value = _number(eq.group(2))
        if value is None:
            args["other"] = eq.group(2)
        else:
            args["sd0"] = value
    elif len(cmd.varlist) == 1 and cmd.options.get("by"):
        args["y"] = cmd.varlist[0]
        args["by"] = str(cmd.options["by"]).split()[0]
    else:
        return _bad(
            cmd, "expected `sdtest y == #`, `sdtest y, by(group)` or `sdtest y == x`"
        )
    err = _level(cmd, args)
    if err:
        return err
    return _emit("sdtest", args, f"sp.sdtest(df, {_kw(args)})")


def _immediate(cmd: StataCommand) -> Any:
    """``#obs #mean #sd #val`` or ``#obs1 #mean1 #sd1 #obs2 #mean2 #sd2``."""
    raw = list(cmd.varlist)
    values = [None if tok == "." else _number(tok) for tok in raw]
    if len(raw) not in (4, 6) or any(
        v is None and t != "." for v, t in zip(values, raw)
    ):
        return _bad(
            cmd,
            "expected `#obs #mean #sd #val` or `#obs1 #mean1 #sd1 #obs2 #mean2 #sd2`",
        )
    return values


def _h_sdtesti(cmd: StataCommand) -> Dict[str, Any]:
    """``sdtesti 10 . 1.14 2`` -> ``sp.sdtest(n=10, sd=1.14, sd0=2)``."""
    values = _immediate(cmd)
    if isinstance(values, dict):
        return values
    args: Dict[str, Any]
    if len(values) == 4:
        n, mean, sd, null = values
        if n is None or sd is None or null is None:
            return _bad(cmd, "#obs, #sd and #val must be numbers")
        args = {"n": int(n), "sd": sd, "sd0": null}
        if mean is not None:
            args["mean"] = mean
    else:
        n1, m1, s1, n2, m2, s2 = values
        if None in (n1, s1, n2, s2):
            return _bad(cmd, "#obs and #sd must be numbers")
        args = {"n": (int(n1), int(n2)), "sd": (s1, s2)}
        if m1 is not None and m2 is not None:
            args["mean"] = (m1, m2)
    err = _level(cmd, args)
    if err:
        return err
    return _emit("sdtest", args, f"sp.sdtest({_kw(args)})")


def _known_sd(cmd: StataCommand, two: bool) -> Any:
    """``sd(#)``, or ``sd1(#) sd2(#)`` for two samples; Stata's default is 1."""
    opts = cmd.options
    single = _number(opts["sd"]) if opts.get("sd") is not None else None
    if not two:
        return 1.0 if opts.get("sd") is None else single
    first = _number(opts["sd1"]) if opts.get("sd1") is not None else None
    second = _number(opts["sd2"]) if opts.get("sd2") is not None else None
    if opts.get("sd1") is None and opts.get("sd2") is None:
        return 1.0 if opts.get("sd") is None else single
    if first is None or second is None:
        return None
    return (first, second)


def _h_ztest(cmd: StataCommand) -> Dict[str, Any]:
    """``ztest y == 20, sd(6)`` / ``ztest y, by(g) sd(6)`` -> ``sp.ztest``."""
    text = " ".join(cmd.varlist).strip()
    args: Dict[str, Any] = {}
    eq = _EQ.fullmatch(text)
    two = False
    if eq:
        args["y"] = eq.group(1)
        value = _number(eq.group(2))
        if value is not None:
            args["mu"] = value
        elif "unpaired" in cmd.options:
            args["other"], two = eq.group(2), True
        else:
            return _bad(
                cmd,
                "the paired z test (`ztest y == x` with corr()) is not "
                "translated; add `unpaired` for independent samples",
            )
    elif len(cmd.varlist) == 1 and cmd.options.get("by"):
        args["y"] = cmd.varlist[0]
        args["by"] = str(cmd.options["by"]).split()[0]
        two = True
        cmd.options.get("unpaired")  # by() groups are unpaired already
    else:
        return _bad(cmd, "expected `ztest y == #, sd(#)` or `ztest y, by(group) sd(#)`")
    sd = _known_sd(cmd, two)
    if sd is None:
        return _bad(cmd, "sd() / sd1() sd2() must be numbers")
    args["sd"] = sd
    err = _level(cmd, args)
    if err:
        return err
    return _emit("ztest", args, f"sp.ztest(df, {_kw(args)})")


def _h_ztesti(cmd: StataCommand) -> Dict[str, Any]:
    """``ztesti 10 88 0.71 85`` -> ``sp.ztest(n=10, mean=88, sd=0.71, mu=85)``."""
    values = _immediate(cmd)
    if isinstance(values, dict):
        return values
    if any(v is None for v in values):
        return _bad(cmd, "every argument must be a number")
    args: Dict[str, Any]
    if len(values) == 4:
        n, mean, sd, null = values
        args = {"n": int(n), "mean": mean, "sd": sd, "mu": null}
    else:
        n1, m1, s1, n2, m2, s2 = values
        args = {"n": (int(n1), int(n2)), "mean": (m1, m2), "sd": (s1, s2)}
    err = _level(cmd, args)
    if err:
        return err
    return _emit("ztest", args, f"sp.ztest({_kw(args)})")


def _h_prtest(cmd: StataCommand) -> Dict[str, Any]:
    """``prtest d == 0.4`` / ``prtest d, by(g)`` / ``prtest d == e`` ->
    ``sp.prtest``."""
    text = " ".join(cmd.varlist).strip()
    args: Dict[str, Any] = {}
    eq = _EQ.fullmatch(text)
    if eq:
        args["y"] = eq.group(1)
        value = _number(eq.group(2))
        if value is None:
            args["other"] = eq.group(2)
        else:
            args["p"] = value
    elif len(cmd.varlist) == 1 and cmd.options.get("by"):
        args["y"] = cmd.varlist[0]
        args["by"] = str(cmd.options["by"]).split()[0]
    else:
        return _bad(
            cmd, "expected `prtest d == #`, `prtest d, by(group)` or `prtest d == e`"
        )
    err = _level(cmd, args)
    if err:
        return err
    return _emit("prtest", args, f"sp.prtest(df, {_kw(args)})")


def _h_prtesti(cmd: StataCommand) -> Dict[str, Any]:
    """``prtesti 50 0.52 0.4`` -> ``sp.prtest(n=50, proportion=0.52, p=0.4)``;
    four numbers are two samples. ``count`` gives successes instead of
    proportions."""
    values = [_number(tok) for tok in cmd.varlist]
    if len(values) not in (3, 4) or any(v is None for v in values):
        return _bad(cmd, "expected `#obs #p #p0` or `#obs1 #p1 #obs2 #p2`")
    count = "count" in cmd.options
    args: Dict[str, Any]
    if len(values) == 3:
        n, p_hat, null = values
        args = {"n": int(n), "proportion": p_hat / n if count else p_hat, "p": null}
    else:
        n1, p1, n2, p2 = values
        args = {
            "n": (int(n1), int(n2)),
            "proportion": (p1 / n1, p2 / n2) if count else (p1, p2),
        }
    err = _level(cmd, args)
    if err:
        return err
    return _emit("prtest", args, f"sp.prtest({_kw(args)})")


def _h_normality(cmd: StataCommand) -> Dict[str, Any]:
    """``sktest x y [, noadjust]`` -> ``sp.sktest``; ``swilk x y`` ->
    ``sp.swilk``."""
    if not cmd.varlist:
        return _bad(cmd, "needs a varlist")
    args: Dict[str, Any] = {"variables": list(cmd.varlist)}
    tool = "swilk" if cmd.command == "swilk" else "sktest"
    if tool == "sktest" and "noadjust" in cmd.options:
        args["adjust"] = False
    return _emit(tool, args, f"sp.{tool}(df, {_kw(args)})")


def _h_ci(cmd: StataCommand) -> Dict[str, Any]:
    """``ci means x`` / ``ci variances x [, sd]`` / ``ci proportions d
    [, wilson ...]`` -> ``sp.ci``."""
    words = list(cmd.varlist)
    kinds = (("means", 4), ("variances", 3), ("proportions", 4))
    stat = None
    if words:
        head = words[0].lower()
        for full, shortest in kinds:
            if shortest <= len(head) <= len(full) and full.startswith(head):
                stat, words = full, words[1:]
                break
    if stat is None:
        return _bad(
            cmd,
            "expected `ci means`, `ci variances` or `ci proportions` followed "
            "by a varlist",
        )
    if not words:
        return _bad(cmd, "needs a varlist")
    args: Dict[str, Any] = {"variables": words, "stat": stat}
    if stat == "variances" and "sd" in cmd.options:
        args["stat"] = "sd"
    if stat == "proportions":
        chosen = [
            m for m in ("exact", "wald", "wilson", "agresti", "jeffreys")
            if m in cmd.options
        ]  # fmt: skip
        if len(chosen) > 1:
            return _bad(cmd, "more than one interval type was given")
        args["method"] = chosen[0] if chosen else "exact"
    err = _level(cmd, args)
    if err:
        return err
    return _emit("ci", args, f"sp.ci(df, {_kw(args)})")


def _h_ttesti(cmd: StataCommand) -> Dict[str, Any]:
    """``ttesti 10 88 1.1 85`` -> ``sp.ttest(n=10, mean=88, sd=1.1, mu=85)``;
    six numbers are two independent samples."""
    values = _immediate(cmd)
    if isinstance(values, dict):
        return values
    if any(v is None for v in values):
        return _bad(cmd, "every argument must be a number")
    args: Dict[str, Any]
    if len(values) == 4:
        n, mean, sd, null = values
        args = {"n": int(n), "mean": mean, "sd": sd, "mu": null}
    else:
        n1, m1, s1, n2, m2, s2 = values
        args = {"n": (int(n1), int(n2)), "mean": (m1, m2), "sd": (s1, s2)}
        if "welch" in cmd.options:
            args["welch"] = True
        elif "unequal" in cmd.options:
            args["unequal"] = True
    err = _level(cmd, args)
    if err:
        return err
    return _emit("ttest", args, f"sp.ttest({_kw(args)})")


# ---------------------------------------------- limited dependent variables
def _ml_vce(cmd: StataCommand, args: Dict[str, Any]) -> None:
    """``vce(robust)`` / ``vce(cluster c)`` onto ``robust=`` / ``cluster=``."""
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    elif "robust" in cmd.options or _opt_matches(cmd.options.get("vce"), "robust"):
        args["robust"] = "robust"


def _h_etregress(cmd: StataCommand) -> Dict[str, Any]:
    """``etregress y x, treat(d = z) [twostep]`` -> ``sp.etregress``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _bad(cmd, "requires an outcome variable")
    head, eq, tail = (cmd.options.get("treat") or "").partition("=")
    if not eq or len(head.split()) != 1 or not tail.split():
        return _bad(cmd, "treat() must be `treatvar = covariates`")
    args: Dict[str, Any] = {
        "y": y,
        "x": list(xs),
        "treatment": head.strip(),
        "z": tail.split(),
        "method": "twostep" if "twostep" in cmd.options else "mle",
    }
    if args["method"] == "mle":
        _ml_vce(cmd, args)
    return _emit("etregress", args, f"sp.etregress(data=df, {_kw(args)})")


# ------------------------------------------------------------- time series
def _h_pperron(cmd: StataCommand) -> Dict[str, Any]:
    """``pperron y [, lags(#) trend noconstant]`` ->
    ``sp.unitroot(df, 'y', test='pp')``."""
    if len(cmd.varlist) != 1:
        return _bad(cmd, "takes one variable")
    args: Dict[str, Any] = {"y": cmd.varlist[0], "test": "pp"}
    raw = cmd.options.get("lags")
    if raw is not None:
        value = _number(raw)
        if value is None or value != int(value):
            return _bad(cmd, f"lags({raw}) is not an integer")
        args["lags"] = int(value)
    if "trend" in cmd.options:
        args["trend"] = "ct"
    elif "noconstant" in cmd.options:
        args["trend"] = "n"
    semantics = [
        _ROW_ORDER,
        "The translated statistic is Z(t) (result.statistic); Z(rho) is "
        "result.z_rho. Critical values are MacKinnon's response surface, "
        "not the interpolated Fuller table pperron prints.",
    ]
    return _emit("unitroot", args, f"sp.unitroot(df, {_kw(args)})", semantics=semantics)


def _h_kpss(cmd: StataCommand) -> Dict[str, Any]:
    """``kpss y, maxlag(#) [notrend]`` -> ``sp.unitroot(df, 'y', test='kpss')``."""
    if len(cmd.varlist) != 1:
        return _bad(cmd, "takes one variable")
    raw = cmd.options.get("maxlag")
    value = _number(raw) if raw is not None else None
    if value is None or value != int(value):
        return _bad(
            cmd,
            "give maxlag(#): without it kpss picks the lag from the sample "
            "size by Schwert's rule, which one translated line cannot see",
        )
    args: Dict[str, Any] = {
        "y": cmd.varlist[0],
        "test": "kpss",
        "trend": "c" if "notrend" in cmd.options else "ct",
        "lags": int(value),
    }
    semantics = [
        _ROW_ORDER,
        "kpss prints the statistic at every lag order 0 .. maxlag; the "
        "call returns the one at maxlag. The null is stationarity.",
    ]
    return _emit("unitroot", args, f"sp.unitroot(df, {_kw(args)})", semantics=semantics)


def _order(raw: Optional[str], cmd: StataCommand, name: str) -> Any:
    """``ar(1/2)`` / ``ar(1 2)`` -> 2. A list that skips a lag (``ma(1 4)``)
    is a restricted model with no counterpart here."""
    if raw is None:
        return 0
    lags: List[int] = []
    for token in str(raw).replace(",", " ").split():
        lo, sep, hi = token.partition("/")
        try:
            a = int(lo)
            b = int(hi) if sep else a
        except ValueError:
            return _bad(cmd, f"{name}({raw}) is not a list of lags")
        lags.extend(range(a, b + 1))
    if not lags or sorted(set(lags)) != list(range(1, max(lags) + 1)):
        return _bad(
            cmd,
            f"{name}({raw}) skips a lag; only the unrestricted orders 1..p "
            "are translated",
        )
    return max(lags)


def _h_arima(cmd: StataCommand) -> Dict[str, Any]:
    """``arima y [x], arima(p,d,q)`` / ``arima y, ar(1/p) ma(1/q)`` ->
    ``sp.arima``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _bad(cmd, "requires a dependent variable")
    opts = cmd.options
    spec = opts.get("arima")
    if spec is not None:
        parts = [p for p in re.split(r"[,\s]+", str(spec).strip()) if p]
        try:
            p_, d_, q_ = (int(v) for v in parts)
        except ValueError:
            return _bad(cmd, f"arima({spec}) is not (#p,#d,#q)")
        if opts.get("ar") is not None or opts.get("ma") is not None:
            return _bad(cmd, "arima() cannot be combined with ar() / ma()")
    else:
        p_ = _order(opts.get("ar"), cmd, "ar")
        if isinstance(p_, dict):
            return p_
        q_ = _order(opts.get("ma"), cmd, "ma")
        if isinstance(q_, dict):
            return q_
        d_ = 0
    args: Dict[str, Any] = {"y": y, "order": (p_, d_, q_)}
    if xs:
        args["exog"] = list(xs)
    args["trend"] = "n" if "noconstant" in opts else "c"
    args["method"] = "innovations_mle"
    semantics = [
        _ROW_ORDER,
        "Stata's arima always estimates a constant (the mean of the series, "
        "or of its difference); sp.arima's default drops it once the series "
        "is differenced, so trend= is written out.",
        "Exact Gaussian maximum likelihood with OPG standard errors, as "
        "Stata's default. Stata stops at its own convergence tolerance, so "
        "its printed estimates agree to four or five digits, not all.",
    ]
    return _emit("arima", args, f"sp.arima(data=df, {_kw(args)})", semantics=semantics)


def _h_arch(cmd: StataCommand) -> Dict[str, Any]:
    """``arch y, arch(1/q) [garch(1/p)]`` -> ``sp.garch``."""
    if len(cmd.varlist) != 1:
        return _bad(
            cmd,
            "a mean equation with regressors is not translated; sp.garch "
            "models a constant mean",
        )
    opts = cmd.options
    model = "garch"
    if "earch" in opts or "egarch" in opts:
        # exponential GARCH: earch() carries both the signed shock and its
        # magnitude, egarch() the lagged log variances
        if any(k in opts for k in ("arch", "garch", "tarch")):
            return _bad(
                cmd,
                "earch() / egarch() mixed with arch(), garch() or tarch() "
                "is not translated; sp.garch fits one variance equation",
            )
        q_ = _order(opts.get("earch"), cmd, "earch")
        if isinstance(q_, dict):
            return q_
        p_ = _order(opts.get("egarch"), cmd, "egarch")
        if isinstance(p_, dict):
            return p_
        if q_ == 0:
            return _bad(cmd, "needs earch(), the lags of the standardised shock")
        model = "egarch"
    else:
        q_ = _order(opts.get("arch"), cmd, "arch")
        if isinstance(q_, dict):
            return q_
        p_ = _order(opts.get("garch"), cmd, "garch")
        if isinstance(p_, dict):
            return p_
        if q_ == 0:
            return _bad(cmd, "needs arch(), the lags of the squared innovations")
        if "tarch" in opts:
            t_ = _order(opts.get("tarch"), cmd, "tarch")
            if isinstance(t_, dict):
                return t_
            if t_ != q_:
                return _bad(
                    cmd,
                    f"tarch({opts.get('tarch')}) with arch({opts.get('arch')}): "
                    "sp.garch(model='gjr') has one threshold term per ARCH lag",
                )
            model = "gjr"
    args: Dict[str, Any] = {"y": cmd.varlist[0], "p": p_, "q": q_}
    if model != "garch":
        args["model"] = model
    if "noconstant" in opts:
        args["mean"] = False
    if "ar" in opts:
        ar_ = _order(opts.get("ar"), cmd, "ar")
        if isinstance(ar_, dict):
            return ar_
        if ar_:
            args["ar"] = ar_
    if "distribution" in opts:
        dist = str(opts.get("distribution") or "").strip().lower()
        if dist in ("t", "gaussian", "normal"):
            if dist == "t":
                args["dist"] = "t"
        else:
            return _bad(
                cmd,
                f"distribution({dist}) is not translated; sp.garch estimates "
                "the degrees of freedom of a t distribution or assumes "
                "normality",
            )
    vce = str(opts.get("vce") or "").strip().lower()
    if "robust" in opts or vce == "robust":
        args["vce"] = "robust"
    elif vce in ("", "opg"):
        args["vce"] = "opg"
    elif vce == "oim":
        args["vce"] = "oim"
    else:
        return _bad(cmd, f"vce({vce}) is not translated")
    semantics = [
        _ROW_ORDER,
        "sp.garch counts lagged variances in p and lagged squared "
        "innovations in q: arch(1) garch(1) is p=1, q=1; arch(1) alone is "
        "p=0, q=1.",
        "Stata's arch reports OPG standard errors by default, sp.garch the "
        "observed information; vce= is written out.",
    ]
    if model == "gjr":
        semantics.append(
            "Stata writes the threshold term on positive shocks, sp.garch on "
            "negative ones: gamma = -tarch and alpha = arch + tarch. The "
            "likelihood, the fitted variances and beta are the same."
        )
    if model == "egarch":
        semantics.append(
            "theta is Stata's earch (signed shock) and gamma its earch_a "
            "(magnitude of the shock)."
        )
    return _emit("garch", args, f"sp.garch(data=df, {_kw(args)})", semantics=semantics)


# ------------------------------------------------- comparing distributions
def _h_by_test(tool: str, stata: str) -> Any:
    """``<cmd> y, by(g)`` -> ``sp.<tool>(df, 'y', by='g')``."""

    def handler(cmd: StataCommand) -> Dict[str, Any]:
        by = cmd.options.get("by")
        if len(cmd.varlist) != 1 or not by:
            return _bad(cmd, f"expected `{stata} y, by(group)`")
        args: Dict[str, Any] = {"y": cmd.varlist[0], "by": str(by).split()[0]}
        return _emit(tool, args, f"sp.{tool}(df, {_kw(args)})")

    return handler


def _h_signrank(cmd: StataCommand) -> Dict[str, Any]:
    """``signrank y = x`` / ``signrank y = #`` -> ``sp.signrank``."""
    eq = _EQ.fullmatch(" ".join(cmd.varlist).strip())
    if not eq:
        return _bad(cmd, "expected `signrank y = x` or `signrank y = #`")
    args: Dict[str, Any] = {"y": eq.group(1)}
    value = _number(eq.group(2))
    if value is None:
        args["other"] = eq.group(2)
    else:
        args["value"] = value
    return _emit("signrank", args, f"sp.signrank(df, {_kw(args)})")


def _h_rank_correlation(cmd: StataCommand) -> Dict[str, Any]:
    """``spearman x y`` -> ``sp.spearman``; ``ktau x y`` -> ``sp.ktau``."""
    if len(cmd.varlist) != 2:
        return _bad(
            cmd,
            "two variables are translated; for a matrix of rank "
            "correlations loop over the pairs",
        )
    tool = "ktau" if cmd.command == "ktau" else "spearman"
    args: Dict[str, Any] = {"x": cmd.varlist[0], "y": cmd.varlist[1]}
    return _emit(tool, args, f"sp.{tool}(df, {_kw(args)})")


def _h_oneway(cmd: StataCommand) -> Dict[str, Any]:
    """``oneway y g [, bonferroni | sidak | scheffe]`` -> ``sp.oneway``."""
    if len(cmd.varlist) != 2:
        return _bad(cmd, "expected `oneway response factor`")
    args: Dict[str, Any] = {"y": cmd.varlist[0], "by": cmd.varlist[1]}
    chosen = [m for m in ("bonferroni", "sidak", "scheffe") if m in cmd.options]
    if len(chosen) > 1:
        return _bad(cmd, "one multiple-comparison adjustment per call is translated")
    if chosen:
        args["compare"] = chosen[0]
    return _emit("oneway", args, f"sp.oneway(df, {_kw(args)})")


HANDLERS = {
    "ranksum": _h_by_test("ranksum", "ranksum"),
    "kwallis": _h_by_test("kwallis", "kwallis"),
    "ksmirnov": _h_by_test("ksmirnov", "ksmirnov"),
    "median": _h_by_test("median_test", "median"),
    "robvar": _h_by_test("robvar", "robvar"),
    "signrank": _h_signrank,
    "spearman": _h_rank_correlation,
    "ktau": _h_rank_correlation,
    "oneway": _h_oneway,
    "sdtest": _h_sdtest,
    "sdtesti": _h_sdtesti,
    "ztest": _h_ztest,
    "ztesti": _h_ztesti,
    "ttesti": _h_ttesti,
    "prtest": _h_prtest,
    "prtesti": _h_prtesti,
    "sktest": _h_normality,
    "swilk": _h_normality,
    "ci": _h_ci,
    "etregress": _h_etregress,
    "pperron": _h_pperron,
    "kpss": _h_kpss,
    "arima": _h_arima,
    "arch": _h_arch,
}
