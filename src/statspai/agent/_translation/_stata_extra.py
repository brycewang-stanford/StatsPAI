"""``eventstudyinteract`` and ``power`` (two-mean designs).

``eventstudyinteract y rel_*, cohort(g) control_cohort(c) absorb(i.id
i.t)`` is Sun and Abraham's interaction-weighted estimator, which
``sp.sun_abraham`` computes with the cohort-share variance term that
``eventstudyinteract`` reports (Stata 18, 13 relative periods, with and
without ``covariates()``: every coefficient and SE agrees to 1e-9,
``tests/test_stata_translation_extra.py``). ``eventstudyinteract`` takes
the relative-period indicators the user built; ``sp.sun_abraham`` builds
one for every observed relative period except -1. The two agree only
when the varlist is exactly that set, which a single line cannot show:
``sp.from_stata`` reports the varlist as not carried over, and
``sp.stata`` checks it on the data (:func:`verify_relative_time`) and
runs the command only when it is.

``power twomeans | onemean | pairedmeans`` is ``sp.power_ttest``, which
solves the same t-test power equation (``n`` is the total sample size,
Stata's ``N``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._stata import _emit, _emit_error, _vce_cluster
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "LITERAL", "verify_relative_time"]

_NAME = r"[A-Za-z_]\w*"


def _h_eventstudyinteract(cmd: StataCommand) -> Dict[str, Any]:
    if len(cmd.varlist) < 2:
        return _emit_error(
            "eventstudyinteract needs an outcome and relative-period indicators",
            command="eventstudyinteract",
        )
    y, rel = cmd.varlist[0], list(cmd.varlist[1:])
    opts = cmd.options
    cohort = (opts.get("cohort") or "").strip()
    control = (opts.get("control_cohort") or "").strip()
    absorb = [re.sub(r"^i\.", "", t) for t in (opts.get("absorb") or "").split() if t]
    if not re.fullmatch(_NAME, cohort) or not re.fullmatch(_NAME, control):
        return _emit_error(
            "eventstudyinteract needs cohort(var) and control_cohort(var).",
            command="eventstudyinteract",
            suggestions=[],
        )
    if len(absorb) != 2 or not all(re.fullmatch(_NAME, a) for a in absorb):
        return _emit_error(
            "eventstudyinteract is translated for absorb(i.unit i.time) only "
            f"(got absorb({opts.get('absorb')})): sp.sun_abraham absorbs "
            "exactly unit and period effects.",
            command="eventstudyinteract",
            suggestions=["sun_abraham"],
        )
    if not all(re.fullmatch(_NAME, r) for r in rel):
        return _emit_error(
            "eventstudyinteract: the relative-period indicators must be "
            "variable names.",
            command="eventstudyinteract",
            suggestions=[],
        )
    unit, time = absorb
    args: Dict[str, Any] = {
        "y": y,
        "g": cohort,
        "t": time,
        "i": unit,
        "control_cohort": control,
    }
    covs = (opts.get("covariates") or "").split()
    if covs:
        args["covariates"] = covs
    untranslated: List[str] = ["varlist"]
    notes = [
        f"eventstudyinteract estimates one coefficient per indicator in its "
        f"varlist ({len(rel)} here); sp.sun_abraham builds an indicator for "
        "every observed relative period except -1. The two agree only when "
        "the varlist is exactly that set, which this line cannot show: "
        "sp.stata checks it on the data and runs the command only then.",
        f"absorb({' '.join(absorb)}) is read as unit {unit!r} and period " f"{time!r}.",
        "The coefficients are result.model_info['event_study'] (eventstudy"
        "interact's e(b_iw) / e(V_iw)).",
    ]
    cluster = _vce_cluster(cmd)
    vce = (opts.get("vce") or "").split()
    if cluster:
        if cluster != unit:
            args["cluster"] = cluster
    elif vce:
        untranslated.append("vce")
        notes.append(
            f"vce({' '.join(vce)}) is not carried over: sp.sun_abraham "
            "reports cluster-robust standard errors."
        )
    else:
        untranslated.append("vce")
        notes.append(
            "eventstudyinteract without vce() reports non-clustered "
            "standard errors; sp.sun_abraham clusters on the unit."
        )
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("sun_abraham", args, f"sp.sun_abraham(data=df, {pairs})", notes)
    out["untranslated_options"] = untranslated
    if cluster == unit:
        # vce(cluster unit) is what sp.sun_abraham computes with no argument
        out["_vce_is_sp_default"] = True
    out["verify_relative_time"] = {
        "dummies": rel,
        "g": cohort,
        "t": time,
        "control": control,
    }
    return out


def verify_relative_time(spec: Dict[str, Any], data: pd.DataFrame) -> Optional[str]:
    """None when ``spec['dummies']`` are the indicators of every observed
    relative period but -1 (what ``sp.sun_abraham`` builds); else why not."""
    cols = [spec["g"], spec["t"], spec["control"], *spec["dummies"]]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        return f"columns {missing} are not in the data"
    ctrl = pd.to_numeric(data[spec["control"]], errors="coerce").fillna(0) == 1
    g = pd.to_numeric(data[spec["g"]], errors="coerce")
    treated = ~ctrl & g.notna() & np.isfinite(g) & (g != 0)
    rel = pd.to_numeric(data[spec["t"]], errors="coerce") - g
    found: Dict[float, str] = {}
    for name in spec["dummies"]:
        d = pd.to_numeric(data[name], errors="coerce")
        if not d.dropna().isin([0, 1]).all():
            return f"{name!r} is not a 0/1 indicator"
        on = d == 1
        if (on & ~treated).any():
            return f"{name!r} is 1 for a control or never-treated unit"
        ks = pd.unique(rel[on])
        if len(ks) != 1:
            return f"{name!r} does not mark a single relative period"
        k = float(ks[0])
        if not on.equals(treated & (rel == k)):
            return f"{name!r} is not 1 exactly where {spec['t']} - {spec['g']} == {k:g}"
        found[k] = name
    observed = {float(k) for k in pd.unique(rel[treated])} - {-1.0}
    if set(found) != observed:
        left = sorted(observed - set(found))
        extra = sorted(set(found) - observed)
        return (
            f"the indicators cover relative periods {sorted(found)}; "
            f"sp.sun_abraham estimates every observed one but -1"
            + (f" (missing {left})" if left else "")
            + (f" (not observed {extra})" if extra else "")
        )
    return None


_POWER_TYPES = {
    "twomeans": "two-sample",
    "onemean": "one-sample",
    "pairedmeans": "paired",
}


def _number(text: Optional[str]) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _h_power(cmd: StataCommand) -> Dict[str, Any]:
    """``power twomeans m1 m2, sd() power() | n()`` -> ``sp.power_ttest``."""
    if not cmd.varlist or cmd.varlist[0] not in _POWER_TYPES:
        sub = cmd.varlist[0] if cmd.varlist else ""
        return _emit_error(
            f"power {sub}: only twomeans, onemean and pairedmeans are "
            "translated (sp.power_ttest); see sp.power for other designs.",
            command="power",
            suggestions=["power_ttest", "power"],
        )
    sub, values = cmd.varlist[0], [_number(v) for v in cmd.varlist[1:]]
    if len(values) != 2 or any(v is None for v in values):
        return _emit_error(
            f"power {sub} needs the two means as numbers (one value each).",
            command="power",
            suggestions=[],
        )
    opts = cmd.options
    for unsupported in (
        "sd1",
        "sd2",
        "n1",
        "n2",
        "knownsd",
        "knownsds",
        "diff",
        "corr",
        "sd0",
        "fpc",
        "cluster",
        "k1",
        "k2",
        "m1",
        "m2",
    ):
        if unsupported in opts:
            return _emit_error(
                f"power {sub}, {unsupported}() is not translated: "
                "sp.power_ttest takes one common sd and a total n.",
                command="power",
                suggestions=["power_ttest"],
            )
    sd_name = "sddiff" if sub == "pairedmeans" else "sd"
    sd = _number(opts.get(sd_name)) if opts.get(sd_name) is not None else 1.0
    power = _number(opts.get("power")) if opts.get("power") is not None else None
    n = _number(opts.get("n")) if opts.get("n") is not None else None
    alpha = _number(opts.get("alpha")) if opts.get("alpha") is not None else 0.05
    ratio = _number(opts.get("nratio")) if opts.get("nratio") is not None else 1.0
    if (
        sd is None
        or alpha is None
        or ratio is None
        or (opts.get("power") is not None and power is None)
        or (opts.get("n") is not None and n is None)
    ):
        return _emit_error(
            f"power {sub}: sd() / power() / n() / alpha() / nratio() must "
            "each be one number (lists of values are not translated).",
            command="power",
            suggestions=[],
        )
    if (power is None) == (n is None):
        if power is None:
            power = 0.8  # Stata's default when neither is given
        else:
            return _emit_error(
                f"power {sub}: give power() or n(), not both.",
                command="power",
                suggestions=[],
            )
    delta = values[1] - values[0]  # type: ignore[operator]
    args: Dict[str, Any] = {
        "delta": delta,
        "sd": sd,
        "alpha": alpha,
        "type": _POWER_TYPES[sub],
    }
    if power is not None:
        args["power"] = power
    else:
        args["n"] = n
    if "onesided" in opts:
        args["alternative"] = "greater" if delta > 0 else "less"
    if sub == "twomeans" and ratio != 1.0:
        args["ratio"] = ratio
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "power_ttest",
        args,
        f"sp.power_ttest({pairs})",
        [],
        semantics=[
            "n is the total sample size (Stata's N); Stata rounds a "
            "computed N up to an integer, as sp.power_ttest does."
        ],
    )


HANDLERS = {"eventstudyinteract": _h_eventstudyinteract, "power": _h_power}

#: Handlers whose arguments are numbers or keywords, not variables: the
#: varlist normalisation (factor notation, abbreviations) is skipped.
LITERAL = frozenset({_h_power})
