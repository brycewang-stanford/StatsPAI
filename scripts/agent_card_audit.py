#!/usr/bin/env python3
"""Semantic audit of the agent cards of the 30 most-used entry points.

The registry ratchets count *fields that are present*. This audit asks
whether what the fields say is true, by calling the function:

* **required** -- every argument the signature requires is in the
  schema's ``required`` list, and every extra argument the schema marks
  required really is: calling without it fails.
* **result class** -- the class the card names is the class a real call
  returns.
* **enums** -- every value a schema ``enum`` advertises is tried in a
  real call. ``ok`` means the call returned; ``precondition`` means it
  failed for a reason other than the value (it needs other arguments,
  other data, or an optional package), and the reason is recorded;
  ``rejected`` means the function refused the value the schema offers.
  A ``rejected`` value is a card defect.
* **alternatives** -- every function the card recommends instead exists.
* **provenance** -- how many card fields are curated, inherited from a
  family card, or inferred, so the three are not reported as one number.

Thirty functions is a deliberate limit: the ones an agent reaches first
(the curated MCP tools and the estimators with a validation-scope map),
each with a hand-written call on small data that this file owns. It is
not a sample of the registry and says nothing about the rest.

Outputs ``docs/dev/agent_card_audit.json`` and ``.md``.

Usage
-----
    python scripts/agent_card_audit.py            # run and write
    python scripts/agent_card_audit.py --check    # fail on a card defect
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import sys
import warnings
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
JSON_OUT = REPO_ROOT / "docs" / "dev" / "agent_card_audit.json"
MD_OUT = REPO_ROOT / "docs" / "dev" / "agent_card_audit.md"

#: Words in an error that mean "this option value is not accepted".
_REJECTION_WORDS = (
    "must be one of",
    "must be",
    "unknown",
    "not supported",
    "unsupported",
    "invalid",
    "not a valid",
    "not recognised",
    "not recognized",
    "expected one of",
    "choose from",
)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def _cross():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261003)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    dc = 0.8 * z + 0.5 * u + rng.normal(size=n)
    d = (0.6 * x1 + rng.normal(size=n) > 0).astype(int)
    m = 0.5 * d + 0.3 * x1 + rng.normal(size=n)
    y = 1 + 1.5 * d + 0.8 * dc + x1 - 0.5 * x2 + 0.7 * m + u + rng.normal(size=n)
    r = rng.uniform(-1, 1, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "dc": dc,
            "z": z,
            "x1": x1,
            "x2": x2,
            "m": m,
            "r": r,
            "yrd": 1 + 2 * (r >= 0) + r + rng.normal(scale=0.5, size=n),
            "yb": (y > np.median(y)).astype(int),
            "cnt": rng.poisson(np.exp(0.2 + 0.3 * x1)),
            "sel": (rng.uniform(size=n) < 0.7 + 0.1 * d).astype(int),
            "g": (rng.uniform(size=n) < 0.5).astype(int),
        }
    )


def _staggered():
    import statspai as sp

    df = sp.datasets.mpdta().copy()
    df["treated"] = (
        (df["first_treat"] > 0) & (df["year"] >= df["first_treat"])
    ).astype(int)
    df["g_nan"] = df["first_treat"].where(df["first_treat"] > 0)
    df["lemp_pos"] = (df["lemp"] > df["lemp"].median()).astype(int)
    return df


def _synth_panel():
    import statspai as sp

    return sp.datasets.california_prop99()


_DATA: Dict[str, Any] = {}


def _data(kind: str):
    if kind not in _DATA:
        _DATA[kind] = {"cross": _cross, "staggered": _staggered, "synth": _synth_panel}[
            kind
        ]()
    return _DATA[kind]


_CS = dict(y="lemp", g="first_treat", t="year", i="countyreal")
_BJS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
_X = ["x1", "x2"]

#: name -> (data kind, positional args after data / formula, keyword args).
#: ``"@"`` in the positional list marks where the DataFrame goes.
CALLS: Dict[str, Tuple[str, List[Any], Dict[str, Any]]] = {
    "regress": ("cross", ["y ~ d + x1 + x2", "@"], {}),
    "ivreg": ("cross", ["y ~ x1 + (dc ~ z)", "@"], {}),
    "iv": ("cross", ["y ~ x1 + (dc ~ z)", "@"], {}),
    "logit": ("cross", ["yb ~ d + x1", "@"], {}),
    "probit": ("cross", ["yb ~ d + x1", "@"], {}),
    "poisson": ("cross", ["cnt ~ d + x1", "@"], {}),
    "qreg": ("cross", ["@", "y ~ d + x1"], {}),
    "feols": ("staggered", ["lemp ~ treated | countyreal + year", "@"], {}),
    "panel": ("staggered", ["@", "lemp ~ treated", "countyreal", "year"], {}),
    "did": (
        "staggered",
        ["@"],
        dict(y="lemp", treat="first_treat", time="year", id="countyreal"),
    ),
    "callaway_santanna": ("staggered", ["@"], dict(_CS)),
    "sun_abraham": ("staggered", ["@"], dict(_CS)),
    "did_imputation": ("staggered", ["@"], dict(_BJS)),
    "gardner_did": ("staggered", ["@"], dict(_BJS)),
    "etwfe": ("staggered", ["@"], dict(_BJS)),
    "stacked_did": ("staggered", ["@"], dict(_BJS, window=(-2, 2))),
    "event_study": (
        "staggered",
        ["@"],
        dict(
            y="lemp", treat_time="g_nan", time="year", unit="countyreal", window=(-2, 2)
        ),
    ),
    "bacon_decomposition": (
        "staggered",
        ["@"],
        dict(y="lemp", treat="treated", time="year", id="countyreal"),
    ),
    "rdrobust": ("cross", ["@"], dict(y="yrd", x="r")),
    "rddensity": ("cross", ["@"], dict(x="r")),
    "synth": (
        "synth",
        ["@"],
        dict(
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        ),
    ),
    "sdid": (
        "synth",
        ["@"],
        dict(
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        ),
    ),
    "dml": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "metalearner": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "causal_forest": ("cross", ["y ~ d | x1 + x2", "@"], dict(n_estimators=100)),
    "psm": ("cross", ["@"], dict(y="y", d="d", X=_X)),
    "match": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "ipw": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "aipw": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "tmle": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "ebalance": ("cross", ["@"], dict(y="y", treat="d", covariates=_X)),
    "mediate": ("cross", ["@"], dict(y="y", treat="d", mediator="m", n_boot=30)),
    "oaxaca": ("cross", ["@"], dict(y="y", group="g", x=_X)),
    "lee_bounds": (
        "cross",
        ["@"],
        dict(y="y", treat="d", selection="sel", n_bootstrap=30),
    ),
}

#: The thirty audited. CALLS may hold a few more for convenience; this is
#: the list the document reports and the gate enforces.
TOP_30 = (
    "regress",
    "ivreg",
    "iv",
    "logit",
    "probit",
    "poisson",
    "qreg",
    "feols",
    "panel",
    "did",
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "gardner_did",
    "etwfe",
    "stacked_did",
    "event_study",
    "bacon_decomposition",
    "rdrobust",
    "rddensity",
    "synth",
    "sdid",
    "dml",
    "metalearner",
    "causal_forest",
    "psm",
    "match",
    "ipw",
    "aipw",
    "tmle",
)


#: An enum value that is valid only together with another argument, and
#: whose description says so. The audit retries with the context before
#: calling the value rejected.
ENUM_CONTEXT: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("dml", "score", "ATE"): {"model": "irm"},
    ("dml", "score", "ATTE"): {"model": "irm"},
}


def _invoke(name: str, overrides: Dict[str, Any] = None, drop: str = None) -> Any:
    import statspai as sp

    kind, positional, keywords = CALLS[name]
    frame = _data(kind)
    args = [frame if a == "@" else a for a in positional]
    kwargs = dict(keywords)
    kwargs.update(overrides or {})
    if drop is not None:
        if drop in kwargs:
            kwargs.pop(drop)
        else:
            # A required argument passed positionally: rebuild by name.
            params = list(inspect.signature(getattr(sp, name)).parameters)
            named = dict(zip(params, args))
            named.pop(drop, None)
            args, kwargs = [], {**named, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name)(*args, **kwargs)


def try_enum_value(name: str, arg: str, value: Any) -> Dict[str, Any]:
    """Call ``sp.<name>`` with ``arg=value`` and say what happened.

    ``{"status": "ok"}``, ``{"status": "ok", "with": {...}}`` when the
    value needs the companion arguments in :data:`ENUM_CONTEXT`,
    ``{"status": "precondition", "error": ...}`` when the call failed for
    a reason other than the value, or ``{"status": "rejected", ...}``.
    """
    try:
        _invoke(name, overrides={arg: value})
    except Exception as exc:  # noqa: BLE001 - classified below
        context = ENUM_CONTEXT.get((name, arg, str(value)))
        if context is not None:
            try:
                _invoke(name, overrides={arg: value, **context})
            except Exception as exc2:  # noqa: BLE001
                exc = exc2
            else:
                return {
                    "status": "ok",
                    "with": {k: str(v) for k, v in context.items()},
                }
        return {
            "status": _classify(exc, value),
            "error": f"{type(exc).__name__}: {str(exc)[:140]}",
        }
    return {"status": "ok"}


def _classify(exc: BaseException, value: Any) -> str:
    if isinstance(exc, ImportError):
        return "precondition"
    text = str(exc).lower()
    mentions = repr(value).lower() in text or str(value).lower() in text
    if mentions and any(word in text for word in _REJECTION_WORDS):
        return "rejected"
    return "precondition"


def _class_names(obj: Any) -> List[str]:
    return [c.__name__ for c in type(obj).__mro__]


def audit_function(name: str, enums: bool = True) -> Dict[str, Any]:
    import statspai as sp

    card = sp.describe_function(name)
    schema = sp.function_schema(name)["parameters"]
    fn = getattr(sp, name)
    params = inspect.signature(fn).parameters
    sig_required = [
        k
        for k, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    schema_required = list(schema.get("required") or [])
    out: Dict[str, Any] = {"function": name, "defects": []}

    # -- required ---------------------------------------------------------
    missing = [k for k in sig_required if k not in schema_required]
    if missing:
        out["defects"].append(
            f"signature requires {missing}, schema does not list them as required"
        )
    extra_checked = {}
    for arg in schema_required:
        if arg in sig_required:
            continue
        try:
            _invoke(name, drop=arg)
        except Exception as exc:  # noqa: BLE001 - any failure proves it is needed
            extra_checked[arg] = f"fails without it ({type(exc).__name__})"
        else:
            extra_checked[arg] = "call succeeds without it"
            out["defects"].append(
                f"schema marks {arg!r} required but the call succeeds without it"
            )
    out["required"] = {
        "signature": sig_required,
        "schema": schema_required,
        "schema_only": extra_checked,
    }

    # -- base call and result class --------------------------------------
    try:
        result = _invoke(name)
    except Exception as exc:  # noqa: BLE001 - reported, and a defect of the audit
        out["base_call"] = f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
        out["defects"].append("the audit's own base call fails; fix CALLS")
        return out
    out["base_call"] = "ok"
    declared = card.get("result_class") or card.get("returns")
    actual = type(result).__name__
    matches = bool(declared) and any(
        cls in str(declared) for cls in _class_names(result)
    )
    out["result_class"] = {"declared": declared, "actual": actual, "matches": matches}
    if not matches:
        out["defects"].append(
            f"card says the result is {declared!r}; the call returned {actual}"
        )

    # -- enums ------------------------------------------------------------
    enum_results: Dict[str, Dict[str, Any]] = {}
    for arg, spec in schema["properties"].items() if enums else ():
        values = spec.get("enum")
        if not values:
            continue
        per_value = {}
        for value in values:
            outcome = try_enum_value(name, arg, value)
            per_value[str(value)] = outcome
            if outcome["status"] == "rejected":
                out["defects"].append(
                    f"schema offers {arg}={value!r}; the function refuses it"
                )
        enum_results[arg] = per_value
    if enums:
        out["enums"] = enum_results

    # -- alternatives -----------------------------------------------------
    alternatives = []
    for alt in card.get("alternatives") or []:
        alt_name = (
            alt if isinstance(alt, str) else alt.get("function") or alt.get("name")
        )
        target = str(alt_name).replace("sp.", "").split("(")[0].strip()
        resolved = sp
        for part in target.split("."):
            resolved = getattr(resolved, part, None)
            if resolved is None:
                break
        alternatives.append({"name": target, "exists": resolved is not None})
        if resolved is None:
            out["defects"].append(f"alternative {target!r} is not a StatsPAI function")
    out["alternatives"] = alternatives

    # -- provenance of the card fields -----------------------------------
    prov = card.get("provenance") or {}
    counts: Dict[str, int] = {}
    if isinstance(prov, dict):
        for source in prov.values():
            counts[str(source)] = counts.get(str(source), 0) + 1
    out["provenance"] = counts
    for field in ("assumptions", "failure_modes"):
        if not card.get(field):
            out["defects"].append(f"card has no {field}")
    return out


def build(enums: bool = True) -> Dict[str, Any]:
    functions = [audit_function(name, enums=enums) for name in TOP_30]
    enum_counts = {"ok": 0, "precondition": 0, "rejected": 0}
    for f in functions:
        for per_value in (f.get("enums") or {}).values():
            for outcome in per_value.values():
                enum_counts[outcome["status"]] += 1
    return {
        "schema": 1,
        "generated_by": "scripts/agent_card_audit.py",
        "n_functions": len(functions),
        "n_with_defects": sum(1 for f in functions if f["defects"]),
        "enum_values": enum_counts,
        "functions": functions,
    }


def render(report: Dict[str, Any]) -> str:
    lines = [
        "# Agent card audit: the 30 most-used entry points",
        "",
        "Generated by `python scripts/agent_card_audit.py`; do not edit by hand.",
        "",
        "Each card is checked against a real call on small data: the required "
        "arguments, the result class, every value of every schema `enum`, and "
        "the functions the card recommends instead. A registry-wide field "
        "count cannot tell whether a field is true; this can, for these "
        "thirty. It says nothing about the rest of the registry.",
        "",
        f"- functions audited: **{report['n_functions']}**",
        f"- functions with a card defect: **{report['n_with_defects']}**",
        f"- enum values tried: **{sum(report['enum_values'].values())}** "
        f"({report['enum_values']['ok']} accepted, "
        f"{report['enum_values']['precondition']} need a precondition, "
        f"{report['enum_values']['rejected']} refused by the function)",
        "",
        "`precondition` means the call failed for a reason other than the "
        "value itself: it needs other arguments, other data or an optional "
        "package. Those are listed below so the card can say what the value "
        "needs. `rejected` means the schema advertises a value the function "
        "does not accept, which is a defect.",
        "",
        "| Function | Result class | Enum values ok / precondition / rejected | "
        "Card fields curated / inherited / other | Defects |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for f in report["functions"]:
        rc = f.get("result_class") or {}
        enums = f.get("enums") or {}
        tally = {"ok": 0, "precondition": 0, "rejected": 0}
        for per_value in enums.values():
            for outcome in per_value.values():
                tally[outcome["status"]] += 1
        prov = f.get("provenance") or {}
        curated = prov.get("curated", 0)
        inherited = sum(v for k, v in prov.items() if "inherit" in k)
        other = sum(prov.values()) - curated - inherited
        lines.append(
            f"| `{f['function']}` | {rc.get('actual', '--')} | "
            f"{tally['ok']} / {tally['precondition']} / {tally['rejected']} | "
            f"{curated} / {inherited} / {other} | "
            f"{'; '.join(f['defects']) or 'none'} |"
        )
    lines += ["", "## Enum values that need a precondition", ""]
    any_pre = False
    for f in report["functions"]:
        for arg, per_value in (f.get("enums") or {}).items():
            for value, outcome in per_value.items():
                if outcome["status"] == "precondition":
                    any_pre = True
                    lines.append(
                        f"- `sp.{f['function']}({arg}={value!r})`: {outcome['error']}"
                    )
    if not any_pre:
        lines.append("None.")
    lines += ["", "## Arguments the schema requires beyond the signature", ""]
    for f in report["functions"]:
        extra = (f.get("required") or {}).get("schema_only") or {}
        for arg, verdict in extra.items():
            lines.append(f"- `sp.{f['function']}`: `{arg}` -- {verdict}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on a card defect")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="with --check: skip the enum sweep (required arguments, result "
        "class and alternatives only; about a minute instead of a quarter hour)",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    report = build(enums=not (args.check and args.fast))
    if args.check:
        bad = {f["function"]: f["defects"] for f in report["functions"] if f["defects"]}
        if bad:
            for name, defects in bad.items():
                for defect in defects:
                    print(f"[agent_card_audit] sp.{name}: {defect}", file=sys.stderr)
            return 1
        print(f"[agent_card_audit] OK - {report['n_functions']} cards, no defect")
        return 0
    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(render(report), encoding="utf-8")
    print(
        f"[agent_card_audit] {report['n_functions']} functions, "
        f"{report['n_with_defects']} with defects, enums {report['enum_values']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
