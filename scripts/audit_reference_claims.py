#!/usr/bin/env python3
"""Object-level audit: what each estimator reports vs. what is pinned.

A parity archive answers ``does this function agree with its reference?``
The honest form of that question is narrower, and the difference matters.
A function reports several distinct objects --- a headline estimate, its
standard error, an event-study coefficient vector, the pre-treatment
subset of that vector, aggregations, diagnostics --- and a parity module
pins some of them.  Coverage of the function is not coverage of its
objects, and a documented claim of the form "matches Stata ``csdid``" is
routinely read as though it were.

That gap is not hypothetical.  ``sp.did_imputation`` documented Stata
``did_imputation, pretrends(k)`` while its *reported* pre-trend vector
used a different construction entirely, attenuated toward zero by the
untreated unit share.  Every pinned statistic passed throughout, because
the pinned statistic was the pooled ATT and the defect was in a vector
nobody had pinned.  This script exists so that the next instance is found
by running a report rather than by reading a new paper.

What it produces
----------------
For every probed estimator, one row per reported object:

``reported``
    The object exists in the result --- established by running the
    estimator, not by reading its docstring.
``pinned_sides``
    Which reference languages carry a value for *that object* in the
    committed parity result JSONs.

    ``pinned`` means the archive **has a reference number to compare
    against** -- not that the two agree. Agreement is the tolerance
    registry's job (``tests/r_parity/compare.py::TOLERANCES``), and the
    two questions must stay separate: module 84 pins
    ``did_imputation``'s horizon standard errors against both R and
    Stata *and* records that StatsPAI differs from them by 4.9-13%.
    Counting that row as unpinned would hide the evidence; counting it
    as agreement would misreport it. It is pinned, and it disagrees.
``claimed_reference``
    What the registry says the function is aligned with.
``verdict``
    ``pinned`` / ``unpinned`` / ``not-reported``.
``documents_reference``
    Whether the function's documentation names a runnable reference
    implementation anywhere, with the matched string carried alongside so
    the judgement can be checked rather than trusted.

``unpinned`` is the row class the exercise is for: the package reports an
object and has no pinned value for it.  Every defect found so far lived
there.  The ``documents_reference`` flag is reported separately and never
folded into the headline, because naming a reference in a docstring is
not the same as promising parity on the particular object in the row.

Usage
-----
    python scripts/audit_reference_claims.py                # markdown
    python scripts/audit_reference_claims.py --json out.json
    python scripts/audit_reference_claims.py --check        # nonzero exit
                                                            # on new gaps
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- #
# The object vocabulary.
#
# These are the things a DiD estimator hands back that a reader can act
# on.  Splitting the event study into its two halves is the whole point:
# the post-treatment half is what the estimator is for, and the
# pre-treatment half is what the identifying assumption is judged on, and
# they are different constructions that can be pinned independently.
# --------------------------------------------------------------------- #
OBJECTS: Tuple[str, ...] = (
    "headline_att",
    "headline_se",
    "event_study_att",
    "event_study_se",
    "pretrend_att",
    "pretrend_se",
    "pretrend_test",
)

# Known vocabulary gap, stated rather than hidden: the group and
# calendar aggregations have no slot here. Module 04 pins both vectors
# against R and Stata, and this audit does not count them -- they surface
# in ``unclassified_statistics`` instead. Adding them would change the
# denominator, so it is a deliberate follow-up rather than a quiet edit.
OBJECT_LABEL = {
    "headline_att": "headline ATT",
    "headline_se": "headline SE",
    "event_study_att": "event-study coefficients (post)",
    "event_study_se": "event-study SEs (post)",
    "pretrend_att": "pre-treatment leads",
    "pretrend_se": "pre-treatment lead SEs",
    "pretrend_test": "joint pre-trend test",
}


# --------------------------------------------------------------------- #
# Reading the committed parity archive
# --------------------------------------------------------------------- #
def _classify_statistic(name: str) -> Optional[Tuple[str, ...]]:
    """Map a pinned statistic name onto the object vocabulary.

    Deliberately conservative: a statistic that cannot be classified is
    reported as unclassified rather than being folded into whichever
    object looks closest, because the whole value of this audit is that
    it does not overstate coverage.
    """
    n = name.strip()
    low = n.lower()

    rel = re.search(r"rel(?:ative)?[_ ]?(-?\d+)", low)
    if rel is None:
        # ``tau`` must precede the bare ``t`` alternative: BJS-family
        # modules label horizons tau0..tauK, and a bare ``t`` does not
        # match "tau0" (the next character is a letter), so without this
        # the horizon rows fall through to the headline class and credit
        # coverage the archive does not have.
        rel = re.search(
            r"(?:^|_)(?:event|tau|lead|lag|horizon|pre|e|t|h)[_ ]?([+-]?\d+)",
            low,
        )
    is_se = low.startswith("se_") or low.endswith("_se") or "_se_" in low
    is_pre = low.startswith("pre") or low.startswith("placebo")
    if rel is not None:
        try:
            k = int(rel.group(1).lstrip("+"))
        except ValueError:  # pragma: no cover - regex guarantees digits
            k = 0
        if k < 0 or is_pre:
            return ("pretrend_se",) if is_se else ("pretrend_att",)
        return ("event_study_se",) if is_se else ("event_study_att",)
    if is_pre:
        if "test" in low or "joint" in low or low.endswith("_p") or "pval" in low:
            return ("pretrend_test",)
        return ("pretrend_se",) if is_se else ("pretrend_att",)
    if is_se:
        return ("headline_se",)
    if any(tag in low for tag in ("att", "estimate", "beta", "coef", "tau", "effect")):
        return ("headline_att",)
    return None


def read_parity_archive(root: str) -> Tuple[Dict[str, Dict[str, set]], Dict[str, set]]:
    """Return (module -> side -> objects) and (module -> unclassified stats)."""
    pinned: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    unclassified: Dict[str, set] = defaultdict(set)
    patterns = [
        os.path.join(root, "tests", "r_parity", "results", "*.json"),
        os.path.join(root, "tests", "stata_parity", "results", "*.json"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            module = payload.get("module")
            side = payload.get("side")
            if not module or not side:
                continue
            for row in payload.get("rows", []):
                stat = row.get("statistic")
                if not stat:
                    continue
                objects = _classify_statistic(stat)
                if objects is None:
                    unclassified[module].add(stat)
                    continue
                for obj in objects:
                    pinned[module][side].add(obj)
                # A row that carries its own standard error pins the SE
                # object too; the harness stores it beside the estimate
                # rather than as a separate statistic.
                if row.get("se") is not None:
                    for obj in objects:
                        if obj.endswith("_att"):
                            pinned[module][side].add(obj[:-4] + "_se")
    return pinned, unclassified


def map_modules_to_functions(root: str) -> Dict[str, List[str]]:
    """Which ``sp.<name>`` does each parity module's Python side call?"""
    import statspai as sp

    registered = {spec for spec in dir(sp) if not spec.startswith("_")}
    call = re.compile(r"\bsp\.([a-z_][a-z0-9_]*)\s*\(")
    out: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(root, "tests", "r_parity", "*.py"))):
        module = os.path.basename(path)[:-3]
        if module.startswith("_"):
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        names = [n for n in dict.fromkeys(call.findall(text)) if n in registered]
        if names:
            out[module] = names
    return out


# --------------------------------------------------------------------- #
# Probing what each estimator actually reports
# --------------------------------------------------------------------- #
def _frame_from_mapping(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Rebuild a frame from the ``{"horizon": [...], "coef": {...}}`` shape.

    ``sp.gardner_did`` returns its event study as a list of horizon labels
    plus dicts keyed by those labels, which ``pd.DataFrame`` refuses to
    combine. Reading it explicitly is the difference between reporting
    "no event study" and reporting the one that is there.
    """
    index = payload.get("horizon")
    if not isinstance(index, (list, tuple)) or not index:
        return None
    columns: Dict[str, List[Any]] = {"horizon": list(index)}
    for key, value in payload.items():
        if key == "horizon":
            continue
        if isinstance(value, dict):
            columns[key] = [value.get(k) for k in index]
        elif isinstance(value, (list, tuple)) and len(value) == len(index):
            columns[key] = list(value)
    return pd.DataFrame(columns)


def _event_frame_from_tidy(result: Any) -> Optional[pd.DataFrame]:
    """Some results expose the path through ``tidy()`` rather than model_info.

    ``sp.aggte(..., type="dynamic")`` is the case that matters here: its
    event study lives in the tidy frame under ``type == "event_study"``,
    and treating that as "no event study reported" would understate the
    package and overstate this audit.
    """
    tidy = getattr(result, "tidy", None)
    if not callable(tidy):
        return None
    try:
        frame = tidy()
    except Exception:  # noqa: BLE001 - absence is the answer
        return None
    if not isinstance(frame, pd.DataFrame) or "term" not in frame.columns:
        return None
    if "type" in frame.columns:
        frame = frame[frame["type"] == "event_study"]
    else:
        frame = frame[frame["term"].astype(str).str.startswith("event_")]
    if not len(frame):
        return None
    out = pd.DataFrame(
        {
            "_t": frame["term"].map(_coerce_relative_time),
            "_b": frame.get("estimate"),
            "_se": frame.get("std_error", pd.NA),
        }
    )
    return out.dropna(subset=["_t"])


def _event_frame(model_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
    es = model_info.get("event_study")
    if isinstance(es, pd.DataFrame):
        frame = es
    elif isinstance(es, dict):
        frame = _frame_from_mapping(es)
        if frame is None:
            return None
    else:
        return None
    time_col = next(
        (
            c
            for c in ("relative_time", "rel_time", "event_time", "horizon", "e")
            if c in frame.columns
        ),
        None,
    )
    est_col = next(
        (c for c in ("att", "estimate", "coef", "coefficient") if c in frame.columns),
        None,
    )
    if time_col is None or est_col is None:
        return None
    out = frame.rename(columns={time_col: "_t", est_col: "_b"})
    se_col = next((c for c in ("se", "std_error", "stderr") if c in out.columns), None)
    out["_se"] = out[se_col] if se_col else pd.NA
    keep = out[["_t", "_b", "_se"]].copy()
    # Horizon labels are sometimes strings such as "D_k-3".
    keep["_t"] = keep["_t"].map(_coerce_relative_time)
    return keep.dropna(subset=["_t"])


def _coerce_relative_time(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    m = re.search(r"(-?\d+)", str(value))
    return float(m.group(1)) if m else None


def probe_reported_objects(fn: Callable[[], Any]) -> Dict[str, bool]:
    """Run an estimator and record which objects it hands back."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn()
    model_info = getattr(result, "model_info", None) or {}
    reported = {obj: False for obj in OBJECTS}
    reported["headline_att"] = getattr(result, "estimate", None) is not None
    reported["headline_se"] = getattr(result, "se", None) is not None

    frame = _event_frame(model_info)
    if frame is None or not len(frame):
        frame = _event_frame_from_tidy(result)
    if frame is not None and len(frame):
        post = frame[frame["_t"] >= 0]
        pre = frame[frame["_t"] < 0]
        reported["event_study_att"] = bool(len(post))
        reported["event_study_se"] = bool(len(post) and post["_se"].notna().any())
        reported["pretrend_att"] = bool(len(pre))
        reported["pretrend_se"] = bool(len(pre) and pre["_se"].notna().any())
    reported["pretrend_test"] = bool(
        model_info.get("pretrend_test") or getattr(result, "pretrend_test", None)
    )
    return reported


def build_runners(sp_module: Any) -> Dict[str, Callable[[], Any]]:
    """Probe set: the staggered DiD family, on one canonical panel.

    The set is written out rather than discovered so that a function that
    cannot be probed is a visible omission with a reason, not a silent
    absence.  Every entry runs on ``sp.datasets.mpdta()``.
    """
    sp = sp_module
    df = sp.datasets.mpdta()
    y, unit, time, cohort, treat = "lemp", "countyreal", "year", "first_treat", "treat"
    horizon = [-3, -2, -1, 0, 1, 2, 3]

    return {
        "callaway_santanna": lambda: sp.aggte(
            sp.callaway_santanna(df, y=y, g=cohort, t=time, i=unit, estimator="reg"),
            type="dynamic",
        ),
        "sun_abraham": lambda: sp.sun_abraham(df, y=y, g=cohort, t=time, i=unit),
        "did_imputation": lambda: sp.did_imputation(
            df, y=y, group=unit, time=time, first_treat=cohort, horizon=horizon
        ),
        "gardner_did": lambda: sp.gardner_did(
            df,
            y=y,
            group=unit,
            time=time,
            first_treat=cohort,
            event_study=True,
            horizon=horizon,
        ),
        "wooldridge_did": lambda: sp.wooldridge_did(
            df, y=y, group=unit, time=time, first_treat=cohort
        ),
        "etwfe": lambda: sp.etwfe(df, y=y, group=unit, time=time, first_treat=cohort),
        "stacked_did": lambda: sp.stacked_did(
            df, y=y, group=unit, time=time, first_treat=cohort
        ),
        "lp_did": lambda: sp.lp_did(
            df, y=y, unit=unit, time=time, treatment=treat, horizons=(-3, 3)
        ),
        "event_study": lambda: sp.event_study(
            df,
            y=y,
            treat_time=cohort,
            time=time,
            unit=unit,
            window=(-3, 3),
            cluster=unit,
        ),
    }


# --------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------- #
# A reference field that cites a paper is not a parity claim.  Only a
# named software implementation is, because only that can be run and
# disagreed with.  Keeping these separate is what stops the audit from
# inflating its own headline number, so the rule is deliberately strict
# and the matched evidence is carried into the output: every
# "claimed-unpinned" verdict can be checked against the string that
# produced it.
#
# Bare English-ambiguous names ("did", "staggered", "fect", "package")
# are excluded on purpose. An earlier version matched them as substrings
# and scored "heterogeneous treatment effects" as naming the R package
# fect, which inflated the gap count.
_SOFTWARE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("Stata", r"\bStata\b"),
    ("R package", r"\bR\s+package\b"),
    ("CRAN", r"\bCRAN\b"),
    ("SSC", r"\bSSC\b"),
    ("pkg::fn", r"\b[A-Za-z][A-Za-z0-9.]{2,}::[A-Za-z_][A-Za-z0-9_.]*"),
    ("csdid", r"\bcsdid\b"),
    ("drdid", r"\bDRDID\b|\bdrdid\b"),
    ("jwdid", r"\bjwdid\b"),
    ("etwfe", r"\betwfe\b"),
    ("eventstudyinteract", r"\beventstudyinteract\b"),
    ("bacondecomp", r"\bbacondecomp\b"),
    ("honestdid", r"\bHonestDiD\b|\bhonestdid\b"),
    ("lpdid", r"\blpdid\b"),
    ("reghdfe", r"\breghdfe\b"),
    ("ppmlhdfe", r"\bppmlhdfe\b"),
    ("rdrobust", r"\brdrobust\b"),
    ("gsynth", r"\bgsynth\b"),
    ("MatchIt", r"\bMatchIt\b"),
    ("cobalt", r"\bcobalt\b"),
    ("didimputation", r"\bdidimputation\b"),
    ("did_imputation", r"\bdid_imputation\b"),
    ("did2s", r"\bdid2s\b"),
    ("synthdid", r"\bsynthdid\b"),
    ("pretrends", r"\bpretrends\b"),
    ("fixest", r"\bfixest\b"),
    ("sunab", r"\bsunab\b"),
)


def _names_software(text: str) -> Optional[Tuple[str, str]]:
    """Return (token, matched snippet) when the text names an implementation."""
    for token, pattern in _SOFTWARE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            lo = max(m.start() - 40, 0)
            snippet = " ".join(text[lo : m.end() + 40].split())
            return token, snippet
    return None


def claimed_reference(sp_module: Any, name: str) -> Tuple[str, str]:
    """The software implementation this function documents alignment with.

    Returns ``("", "")`` when the documented reference is a paper rather
    than a runnable implementation: a theory citation is not a parity
    claim and must not be scored as one.
    """
    texts: List[str] = []
    try:
        spec = sp_module.describe_function(name)
    except Exception:  # noqa: BLE001 - absence is the answer, not a failure
        spec = None
    if isinstance(spec, dict):
        texts.append(str(spec.get("reference") or ""))
        texts.append(str(spec.get("description") or ""))
    elif spec is not None:
        texts.append(str(getattr(spec, "reference", "") or ""))
        texts.append(str(getattr(spec, "description", "") or ""))
    fn = getattr(sp_module, name, None)
    texts.append(getattr(fn, "__doc__", "") or "")
    for text in texts:
        if not text:
            continue
        hit = _names_software(text)
        if hit:
            return hit
    return ("", "")


# --------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------- #
def run_audit(root: str) -> Dict[str, Any]:
    sys.path.insert(0, os.path.join(root, "src"))
    import statspai as sp

    pinned, unclassified = read_parity_archive(root)
    module_map = map_modules_to_functions(root)
    function_pins: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for module, functions in module_map.items():
        for function in functions:
            for side, objects in pinned.get(module, {}).items():
                function_pins[function][side] |= set(objects)

    runners = build_runners(sp)
    rows: List[Dict[str, Any]] = []
    probe_failures: Dict[str, str] = {}
    for name, runner in runners.items():
        try:
            reported = probe_reported_objects(runner)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            probe_failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        claim_token, claim_evidence = claimed_reference(sp, name)
        pins = function_pins.get(name, {})
        for obj in OBJECTS:
            sides = sorted(s for s, objs in pins.items() if obj in objs and s != "py")
            if not reported[obj]:
                verdict = "not-reported"
            elif sides:
                verdict = "pinned"
            else:
                verdict = "unpinned"
            rows.append(
                {
                    "function": name,
                    "object": obj,
                    "reported": reported[obj],
                    "pinned_sides": sides,
                    "documents_reference": bool(claim_token),
                    "reference_token": claim_token,
                    "reference_evidence": claim_evidence,
                    "verdict": verdict,
                }
            )

    unpinned = [r for r in rows if r["verdict"] == "unpinned"]
    documented = [r for r in unpinned if r["documents_reference"]]
    return {
        "generator": "scripts/audit_reference_claims.py",
        "statspai_version": getattr(sp, "__version__", None),
        "n_functions_probed": len(runners) - len(probe_failures),
        "probe_failures": probe_failures,
        "n_reported_objects": sum(1 for r in rows if r["reported"]),
        "n_pinned": sum(1 for r in rows if r["verdict"] == "pinned"),
        "n_unpinned": len(unpinned),
        "n_unpinned_in_documented_functions": len(documented),
        "unclassified_statistics": {k: sorted(v) for k, v in unclassified.items() if v},
        "rows": rows,
    }


def to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Reference-claim audit: reported objects vs. pinned evidence",
        "",
        f"StatsPAI {payload['statspai_version']}. "
        f"{payload['n_functions_probed']} estimators probed on "
        "`sp.datasets.mpdta()`.",
        "",
        f"- reported objects: **{payload['n_reported_objects']}**",
        f"- of those, carrying a reference value to compare against: "
        f"**{payload['n_pinned']}**  (\"pinned\" = the archive has a number "
        "for this object; whether it *agrees* is the tolerance registry's "
        "question, not this table's)",
        f"- unpinned: **{payload['n_unpinned']}**, of which "
        f"**{payload['n_unpinned_in_documented_functions']}** belong to a "
        "function whose documentation names a runnable reference somewhere",
        "",
        "The second count is the softer one and is reported as such: naming "
        "a reference in a docstring is not the same as promising parity on "
        "the particular object in the row. The first two counts need no such "
        "judgement, which is why the headline rests on them.",
        "",
        "| Function | Object | Pinned sides | Verdict | Docs name a reference |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        if not row["reported"]:
            continue
        sides = ", ".join(row["pinned_sides"]) or "--"
        doc = row["reference_token"] if row["documents_reference"] else "--"
        lines.append(
            f"| `{row['function']}` | {OBJECT_LABEL[row['object']]} | {sides} "
            f"| {row['verdict']} | {doc} |"
        )
    if payload["probe_failures"]:
        lines += ["", "## Not probed", ""]
        for name, reason in sorted(payload["probe_failures"].items()):
            lines.append(f"- `{name}`: {reason}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=REPO_ROOT)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--markdown", dest="md_out", default=None)
    parser.add_argument(
        "--check",
        type=int,
        default=None,
        metavar="MAX_GAPS",
        help="exit non-zero if unpinned reported objects exceed this budget",
    )
    args = parser.parse_args(argv)

    payload = run_audit(args.root)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.write("\n")
    markdown = to_markdown(payload)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(markdown)
    if not args.json_out and not args.md_out:
        print(markdown)

    print(
        f"[audit] probed={payload['n_functions_probed']} "
        f"reported={payload['n_reported_objects']} "
        f"pinned={payload['n_pinned']} "
        f"unpinned={payload['n_unpinned']}",
        file=sys.stderr,
    )
    if args.check is not None and payload["n_unpinned"] > args.check:
        print(
            f"FAIL: {payload['n_unpinned']} unpinned reported objects "
            f"exceeds the budget of {args.check}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
