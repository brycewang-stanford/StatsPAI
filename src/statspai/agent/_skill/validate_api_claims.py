#!/usr/bin/env python3
"""Validation gate for the packaged ``statspai-analysis`` skill.

SkillOpt (https://github.com/microsoft/SkillOpt) treats a skill document as a
trainable artifact whose edits are accepted only when they pass a validation
gate. This script is that gate for ``SKILL.md``: it turns the static
"verified against statspai X.Y.Z" stamp into a runnable check that any agent or
CI job can re-run against whatever ``statspai`` is installed.

It verifies three layers of perishable claims the skill makes:

  1. EXISTENCE   — every ``sp.<name>`` referenced in SKILL.md resolves.
  2. SIGNATURES  — the documented keyword/positional argument names exist.
  2b. CALLS      — every ``sp.<fn>(...)`` call written in a Python code block
                   of the skill is parsed and bound against the real
                   signature: an unknown keyword or too many positional
                   arguments fails. For a function taking ``**kwargs`` the
                   keywords are checked against its agent schema, the
                   family / design its first argument names (``sp.route``,
                   ``sp.power``) or a declared forwarding target.
  3. ATTRIBUTES  — the documented result-object attributes / return shapes hold
                   (the layer that drifts silently when the library adds a
                   convenience method, e.g. ``AFTResult.params`` in 1.19.0).

Usage
-----
    python validate_api_claims.py            # full gate; exits non-zero on drift
    python validate_api_claims.py --quick    # existence + signatures only (no fits)

Keep this green. When it goes red, either the library changed (update the skill
files + this file together) or a skill edit introduced a false claim. The skill
carries no hard-coded "verified against X.Y.Z" stamp: this gate, run against the
installed package, is the stamp.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import pathlib
import re
import textwrap
import types
from typing import Any

import statspai as sp

SKILL_DIR = pathlib.Path(__file__).parent
SKILL_MD = SKILL_DIR / "SKILL.md"
#: Every Markdown file of the skill is checked, not only SKILL.md: the
#: playbook lives in ``references/``.
SKILL_FILES = [SKILL_MD] + sorted((SKILL_DIR / "references").glob("*.md"))

# `sp.<name>` tokens that are illustrative placeholders, not real attributes.
_PLACEHOLDER_REFS = {"power_"}  # from the `sp.power_<design>` prose example

# Required argument names per documented call. We assert presence, not order,
# so additive library changes don't trip the gate.
_SIGNATURE_CLAIMS: dict[str, list[str]] = {
    "match": ["data", "y", "treat", "covariates"],  # y BEFORE treat
    "oster_delta": ["data", "y", "x_base", "x_controls", "r_max"],
    "oster_bounds": ["data", "y", "treat", "controls", "r_max"],
    "mediation": ["data", "y", "d", "m", "X"],
    "subgroup_analysis": ["data", "formula", "x", "by", "robust"],
    "callaway_santanna": ["data", "y", "g", "t", "i", "x"],  # x= not covariates=
    "sun_abraham": ["data", "y", "g", "t", "i"],
    "synth": ["data", "outcome", "unit", "time", "treatment_time"],
    "panel": ["data", "formula", "method"],
    "dml": ["data", "y", "model_y", "model_d"],
    "metalearner": [
        "data",
        "y",
        "treat",
        "covariates",
        "learner",
        "outcome_model",
        "propensity_model",
    ],
    "spec_curve": ["data", "y", "x", "controls", "se_types", "y_transforms"],
    "evalue": ["estimate", "ci", "measure"],
    "g_computation": ["data", "y", "treat", "covariates"],
    "causal_question": [
        "treatment",
        "outcome",
        "data",
        "estimand",
        "design",
        "time_structure",
        "time",
        "id",
        "covariates",
    ],
    "power": ["design", "power_target"],
    "bjs_pretrend_joint": ["result", "data", "y", "group", "time", "first_treat"],
    "honest_did": ["result", "e", "m_grid", "method"],
    "sensitivity_plot": ["sensitivity", "original_ci", "original_estimate"],
    "unified_sensitivity": ["result", "r2_treated", "r2_controlled", "include_oster"],
    "twoway_cluster": ["result", "data", "cluster1", "cluster2"],
    "conley": ["result", "data", "lat", "lon", "dist_cutoff"],
    "mean_comparison": ["data", "variables", "group", "test"],
    "sumstats": ["data", "vars", "by", "by_labels"],
    "cate_by_group": ["result", "data", "by", "n_groups"],
    "principal_strat": ["data", "y", "treat", "strata", "instrument"],
    "hal_tmle": ["data", "y", "treat", "covariates", "variant"],
    "aft": ["formula", "data", "family"],
    "offline_safe_policy": ["data", "state", "action", "reward", "cost"],
    "target_trial_emulate": [
        "protocol",
        "data",
        "outcome_col",
        "treatment_col",
        "time_zero_filter",
        "weights",
    ],
    "kaplan_meier": ["data", "duration", "event", "group"],
    "policy_tree": ["data", "y", "d", "X"],
    # references/modern-methods.md
    "did_imputation": ["data", "y", "group", "time", "first_treat", "horizon"],
    "gardner_did": ["data", "y", "group", "time", "first_treat", "event_study"],
    "etwfe": ["data", "y", "group", "time", "first_treat", "family"],
    "stacked_did": ["data", "y", "group", "time", "first_treat", "window"],
    "lp_did": ["data", "y", "unit", "time", "treatment", "horizons"],
    "did_multiplegt_dyn": [
        "data",
        "y",
        "group",
        "time",
        "treatment",
        "dynamic",
        "placebo",
        "seed",
    ],
    "compare_event_study_conventions": ["data", "y", "unit", "time", "first_treat"],
    "aggte": ["result", "type"],
    "uniform_bands": ["result", "alpha"],
    "did_few_treated": ["data", "y", "id", "time", "treat", "method"],
    "wild_cluster_bootstrap": ["data", "y", "x", "cluster", "test_var"],
    "cr2_se": ["result", "data", "cluster"],
    "iv_diag": [
        "data",
        "y",
        "endog",
        "instruments",
        "exog",
        "cluster",
        "include_clr_ci",
    ],
    "effective_f_test": ["data", "endog", "instruments", "exog"],
    "weakrobust": ["data", "y", "endog", "instruments", "exog"],
    "causal_forest": ["formula", "data", "fe", "id", "time", "random_state"],
    "best_linear_projection": ["forest", "A"],
    "forest_group_effects": ["forest", "by"],
    "rate": ["forest", "target"],
    "rate_split": ["forest", "target"],
    "forest_policy_tree": ["forest", "depth", "cost"],
    "iv_forest": ["data", "y", "treat", "instrument", "covariates"],
    "multi_arm_forest": ["data", "y", "treat", "covariates"],
    "causal_survival_forest": [
        "data",
        "time",
        "event",
        "treat",
        "covariates",
        "horizon",
    ],
    "dynamic_dml": ["data", "y", "treat", "id", "time", "covariates", "lags"],
    "fect": ["data", "y", "treat", "unit", "time", "method", "r"],
    "gsynth": ["data", "outcome", "unit", "time", "treated_unit", "treatment_time"],
    "rd_honest": ["data", "y", "x", "c"],
    "rdhte": ["data", "y", "x", "z", "c"],
    "validation_scope": ["result"],
}

# Modules referenced as ``sp.<mod>`` with the members the skill calls on them.
_MODULE_CLAIMS: dict[str, list[str]] = {
    "gformula": ["gformula_mc"],
    "bounds": ["manski_bounds", "lee_bounds"],
    "ope": ["ips", "direct_method", "doubly_robust", "snips", "switch_dr"],
    "conformal_causal": ["conformal_cate", "conformal_ite"],
    "fairness": ["fairness_audit"],
}

_PASS, _FAIL = "  ok  ", " DRIFT"


def _record(failures: list[str], ok: bool, label: str, detail: str = "") -> None:
    print(f"[{_PASS if ok else _FAIL}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


# ----------------------------------------------------------------- existence
def check_references(failures: list[str]) -> None:
    print("\n== 1. EXISTENCE: every sp.<name> in SKILL.md + references/ resolves ==")
    text = "\n".join(p.read_text(encoding="utf-8") for p in SKILL_FILES)
    refs = sorted(set(re.findall(r"\bsp\.([A-Za-z_][A-Za-z0-9_]*)", text)))
    refs = [r for r in refs if r not in _PLACEHOLDER_REFS]
    missing = [r for r in refs if not hasattr(sp, r)]
    _record(
        failures,
        not missing,
        f"{len(refs)} unique references",
        "" if not missing else f"missing: {missing}",
    )
    mods = sorted(r for r in refs if isinstance(getattr(sp, r, None), types.ModuleType))
    print(f"         (referenced modules: {mods})")


def check_modules(failures: list[str]) -> None:
    print("\n== 1b. MODULE MEMBERS ==")
    for mod, members in _MODULE_CLAIMS.items():
        m = getattr(sp, mod, None)
        is_mod = isinstance(m, types.ModuleType)
        absent = [x for x in members if not hasattr(m, x)] if m is not None else members
        _record(
            failures,
            is_mod and not absent,
            f"sp.{mod}",
            "" if (is_mod and not absent) else f"module={is_mod}, missing={absent}",
        )


# ---------------------------------------------------------------- signatures
def check_signatures(failures: list[str]) -> None:
    print("\n== 2. SIGNATURES: documented argument names exist ==")
    for name, required in _SIGNATURE_CLAIMS.items():
        obj = getattr(sp, name, None)
        if obj is None:
            _record(failures, False, f"sp.{name}", "function missing")
            continue
        try:
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError) as exc:
            _record(failures, False, f"sp.{name}", f"no signature ({exc})")
            continue
        absent = [p for p in required if p not in params]
        _record(
            failures,
            not absent,
            f"sp.{name}",
            "" if not absent else f"missing args: {absent}",
        )


# --------------------------------------------------------------------- calls
_FENCE_OPEN = re.compile(r"^```(?:python|py)\s*$")


def _python_blocks(path: pathlib.Path) -> Any:
    """Yield ``(first_line_number, source)`` for each fenced Python block.

    Fences inside a blockquote (``> ```python``) count, and a fence closes
    only on a line that is exactly the fence, so a literal `````` inside a
    string does not end the block early.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        quoted = re.match(r"^\s*(?:>\s?)+", raw)
        prefix = quoted.group(0) if quoted else ""
        if not _FENCE_OPEN.match(raw[len(prefix) :].strip()):
            i += 1
            continue
        start = i + 2
        body = []
        i += 1
        while i < len(lines):
            line = lines[i]
            if prefix and line.startswith(prefix.rstrip()):
                line = line[len(prefix) :] if line.startswith(prefix) else ""
            if line.strip() == "```":
                break
            body.append(line)
            i += 1
        i += 1
        yield start, textwrap.dedent("\n".join(body))


def _sp_call_target(node: ast.Call) -> Any:
    """``['did', 'callaway_santanna']`` for ``sp.did.callaway_santanna(...)``."""
    parts = []
    cur = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "sp" and parts:
        return list(reversed(parts))
    return None


#: ``sp.<fn>(**kwargs)`` forwards to this object's signature.
_FORWARD_TARGETS = {"causal_forest": "CausalForest"}


def _literal(node: ast.AST) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def _forwarded_keyword_problems(name: str, node: ast.Call, params: Any) -> list[str]:
    """Check the keywords of a call to a function that takes ``**kwargs``.

    The signature alone cannot say which keywords are accepted, so three
    further sources are used, in order: the function's agent schema (which
    lists the forwarded arguments of a dispatcher); for ``sp.route`` and
    ``sp.power`` the family / design named by the first argument, whose
    questions or wrapper signature define the keywords (and, for
    ``route``, the legal answers); and a declared forwarding target. A
    keyword none of them knows is reported: it may be accepted at run
    time, but nothing here can show that it is.
    """
    accepted = set(params)
    try:
        accepted |= set(sp.function_schema(name)["parameters"]["properties"])
    except (KeyError, TypeError, ValueError):
        # No schema for this name: only the resolvers below can vouch for
        # its keywords, and an unvouched keyword is reported.
        accepted |= set()
    first = _literal(node.args[0]) if node.args else None
    problems: list[str] = []
    if name == "route" and isinstance(first, str):
        try:
            questions = {
                q["key"]: set(q["options"])
                for q in sp.decision_guide(first)["questions"]
            }
        except Exception as exc:  # noqa: BLE001
            return [f"names an unknown family {first!r} ({type(exc).__name__})"]
        for kw in node.keywords:
            if not kw.arg or kw.arg in accepted:
                continue
            if kw.arg not in questions:
                problems.append(
                    f"asks {kw.arg!r}, not a question of the {first!r} guide"
                )
                continue
            answer = _literal(kw.value)
            if isinstance(answer, str) and answer not in questions[kw.arg]:
                problems.append(
                    f"answers {kw.arg}={answer!r}; the guide offers "
                    f"{sorted(questions[kw.arg])}"
                )
        return problems
    if name == "power" and isinstance(first, str):
        wrapper = getattr(sp, f"power_{first}", None)
        if wrapper is None:
            return [f"names an unknown design {first!r}"]
        accepted |= set(inspect.signature(wrapper).parameters)
    target = _FORWARD_TARGETS.get(name)
    if target and hasattr(sp, target):
        accepted |= set(inspect.signature(getattr(sp, target).__init__).parameters)
    unknown = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in accepted]
    if unknown:
        problems.append(
            f"passes {unknown}, which neither the signature, the schema nor a "
            "forwarding target lists"
        )
    return problems


def check_call_keywords(failures: list[str]) -> None:
    print(
        "\n== 2b. CALLS: every sp.<fn>(...) in a code block binds to the signature =="
    )
    n_blocks = n_calls = n_open = 0
    problems: list[str] = []
    for path in SKILL_FILES:
        for lineno, src in _python_blocks(path):
            n_blocks += 1
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                problems.append(
                    f"{path.name}:{lineno}: code block is not valid Python ({exc.msg})"
                )
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                parts = _sp_call_target(node)
                if parts is None:
                    continue
                label = f"{path.name}:{lineno + node.lineno - 1}: sp.{'.'.join(parts)}"
                obj = sp
                try:
                    for attr in parts:
                        obj = getattr(obj, attr)
                except AttributeError:
                    problems.append(f"{label} does not resolve")
                    continue
                try:
                    params = inspect.signature(obj).parameters
                except (TypeError, ValueError):
                    continue
                n_calls += 1
                kinds = {p.kind for p in params.values()}
                if inspect.Parameter.VAR_KEYWORD in kinds:
                    n_open += 1
                    problems.extend(
                        f"{label} {msg}"
                        for msg in _forwarded_keyword_problems(parts[-1], node, params)
                    )
                else:
                    unknown = [
                        k.arg for k in node.keywords if k.arg and k.arg not in params
                    ]
                    if unknown:
                        problems.append(f"{label} has no argument {unknown}")
                if inspect.Parameter.VAR_POSITIONAL not in kinds and not any(
                    isinstance(a, ast.Starred) for a in node.args
                ):
                    room = sum(
                        p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        for p in params.values()
                    )
                    if len(node.args) > room:
                        problems.append(
                            f"{label} is given {len(node.args)} positional "
                            f"arguments; it takes {room}"
                        )
    _record(
        failures,
        not problems,
        f"{n_calls} calls in {n_blocks} code blocks "
        f"({n_open} take **kwargs: keywords checked against the schema "
        "or the forwarding target)",
        "; ".join(problems),
    )


# ---------------------------------------------------------------- attributes
def _make_data() -> Any:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["t"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-df["x1"]))).astype(int)
    df["y"] = 2.0 * df["t"] + df["x1"] - 0.5 * df["x2"] + rng.normal(size=n)
    return df, rng


def check_attributes(failures: list[str]) -> None:
    print("\n== 3. ATTRIBUTES & RETURN SHAPES (smoke fits) ==")
    import numpy as np
    import pandas as pd

    df, rng = _make_data()

    # --- metalearner CATE + CausalResult attribute claims -----------------
    try:
        ml = sp.metalearner(
            df, y="y", treat="t", covariates=["x1", "x2"], learner="dr", n_bootstrap=10
        )
        ok = (
            isinstance(getattr(ml, "model_info", None), dict)
            and isinstance(ml.model_info.get("cate"), np.ndarray)
            and not hasattr(ml, "cate_estimates")  # claim: no such attr
            and hasattr(ml, "estimate")
            and hasattr(ml, "ci")
            and hasattr(ml, "n_obs")
            and hasattr(ml, "estimand")
            and not hasattr(ml, "point_estimate")  # claim: no such attr
            and list(ml.conf_int().index) == [ml.estimand]  # claim: one row
            and ml.nobs == ml.n_obs  # claim: nobs is an alias
            and list(ml.data_info) == ["nobs"]  # claim: key is "nobs"
        )
        _record(failures, ok, "metalearner → model_info['cate'] + CausalResult attrs")
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "metalearner", f"{type(exc).__name__}: {exc}")

    # --- IdentificationPlan attributes ------------------------------------
    try:
        q = sp.causal_question(
            treatment="t",
            outcome="y",
            data=df,
            estimand="ATE",
            design="auto",
            covariates=["x1", "x2"],
        )
        plan = q.identify()
        has = all(
            hasattr(plan, a)
            for a in (
                "assumptions",
                "estimand",
                "estimator",
                "fallback_estimators",
                "identification_story",
                "warnings",
                "summary",
            )
        )
        absent = not any(hasattr(plan, a) for a in ("equation", "threats"))
        qattrs = all(hasattr(q, a) for a in ("population", "treatment", "outcome"))
        _record(
            failures,
            has and absent and qattrs,
            "causal_question().identify() → plan/question attrs",
        )
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "IdentificationPlan", f"{type(exc).__name__}: {exc}")

    # --- regtable output enum rejects docx --------------------------------
    try:
        M = sp.regress("y ~ t + x1", df)
        raised = False
        try:
            sp.regtable(M, output="docx")
        except Exception:  # noqa: BLE001  (claim: docx is not a valid output enum)
            raised = True
        valid = all(
            _silent_ok(lambda v=v: sp.regtable(M, output=v))
            for v in ("text", "latex", "markdown", "html")
        )
        _record(
            failures,
            raised and valid,
            "regtable output enum (docx rejected; text/latex/md/html ok)",
        )
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "regtable enum", f"{type(exc).__name__}: {exc}")

    # --- AFTResult.params + regtable(aft) (the 1.19.0 fix) ----------------
    try:
        d = df.copy()
        d["dur"] = rng.exponential(scale=np.exp(0.3 * d["x1"]), size=len(d)) + 0.1
        d["event"] = (rng.uniform(size=len(d)) < 0.7).astype(int)
        aft = sp.aft("dur + event ~ x1 + x2", d, family="weibull")
        ok = (
            isinstance(aft.params, pd.Series)  # NEW in 1.19.0
            and hasattr(aft, "std_errors")
            and _silent_ok(lambda: sp.regtable(aft, output="text"))
            # Since the universal export protocol (v1.21.0), AFTResult
            # carries .to_word/.to_excel/... directly as well; regtable
            # stays the multi-model path.
            and hasattr(aft, "to_word")
            and all(
                hasattr(aft, a)
                for a in ("beta", "se", "var_names", "n", "n_events", "aic", "family")
            )
        )
        _record(failures, ok, "AFTResult.params + sp.regtable(aft) works")
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "AFTResult/regtable", f"{type(exc).__name__}: {exc}")

    # --- plot / model return shapes ---------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        M = sp.regress("y ~ t + x1 + x2", df)
        coef = sp.coefplot(M)
        binr = sp.binscatter(df, x="x1", y="y")
        cf = sp.causal_forest("y ~ t | x1 + x2", data=df, random_state=0)
        d = df.copy()
        d["dur"] = rng.exponential(size=len(d)) + 0.1
        d["event"] = (rng.uniform(size=len(d)) < 0.7).astype(int)
        km = sp.kaplan_meier(d, duration="dur", event="event")
        kmr = km.plot()
        pt = sp.policy_tree(df, y="y", d="t", X=["x1", "x2"], max_depth=2)
        ok = (
            isinstance(coef, tuple)
            and len(coef) == 2  # coefplot → (fig,ax)
            and isinstance(binr, tuple)
            and len(binr) == 3  # binscatter → (fig,ax,df)
            and hasattr(cf, "effect")
            and not hasattr(cf, "local_effects")
            and not isinstance(kmr, tuple)
            and hasattr(kmr, "figure")  # KM → bare Axes
            and hasattr(pt, "plot_tree")
            and not hasattr(pt, "plot")  # policy_tree.plot_tree()
        )
        _record(
            failures,
            ok,
            "plot/return shapes (coefplot/binscatter/forest/KM/policy_tree)",
        )
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "return shapes", f"{type(exc).__name__}: {exc}")

    # --- modern-methods.md: event-study inference, fe forest, fail-loud ---
    try:
        from statspai.exceptions import MethodIncompatibility

        rows = []
        for i in range(120):
            g = int(rng.choice([0, 4, 5]))
            a, x = rng.normal(), rng.normal()
            for t in range(1, 8):
                d = int(g > 0 and t >= g)
                rows.append(
                    dict(
                        id=i,
                        year=t,
                        g=g,
                        d=d,
                        x1=x,
                        y=a + 0.1 * t + (1 + x) * d + rng.normal(),
                    )
                )
        p = pd.DataFrame(rows)
        es = sp.aggte(
            sp.callaway_santanna(p, y="y", g="g", t="year", i="id"), type="dynamic"
        )
        V = sp.event_study_vcov(es)
        ub = sp.uniform_bands(es)
        bjs = sp.did_imputation(p, y="y", group="id", time="year", first_treat="g")
        fcf = sp.causal_forest(
            "y ~ d | x1",
            data=p,
            fe="twoway",
            id="id",
            time="year",
            n_estimators=200,
            random_state=0,
        )
        pooled = sp.causal_forest(
            "y ~ t | x1 + x2", data=df, n_estimators=100, random_state=0
        )
        fpt_rejects_pooled = False
        try:
            sp.forest_policy_tree(pooled, n_splits=2)
        except MethodIncompatibility:
            fpt_rejects_pooled = True
        cmp_rejects_staggered = False
        try:
            sp.compare_event_study_conventions(
                p, y="y", unit="id", time="year", first_treat="g"
            )
        except MethodIncompatibility:
            cmp_rejects_staggered = True
        ok = (
            all(hasattr(V, a) for a in ("beta", "vcov", "times", "as_frame"))
            and {"cband_lower", "cband_upper"} <= set(ub.columns)
            and isinstance(sp.pretrends_power(es), dict)
            and abs(float(fcf.att()) - bjs.estimate) < 1e-6  # fe forest ATT == BJS
            and "gain_over_treat_all" in sp.forest_policy_tree(fcf, depth=1, n_splits=2)
            and fpt_rejects_pooled
            and cmp_rejects_staggered
            and "split_stability"
            in sp.forest_policy_tree(fcf, depth=1, n_splits=2)["diagnostics"]
        )
        _record(failures, ok, "modern methods: ES vcov/bands, fe forest, fail-loud")
    except Exception as exc:  # noqa: BLE001
        _record(failures, False, "modern methods", f"{type(exc).__name__}: {exc}")


def _silent_ok(thunk: Any) -> bool:
    try:
        thunk()
        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="existence + signatures only (skip the smoke fits)",
    )
    args = ap.parse_args()

    print(
        f"statspai {sp.__version__}  ·  validating skill API claims "
        f"({len(SKILL_FILES)} files)"
    )
    failures: list[str] = []
    check_references(failures)
    check_modules(failures)
    check_signatures(failures)
    check_call_keywords(failures)
    if not args.quick:
        check_attributes(failures)

    print("\n" + "=" * 60)
    if failures:
        print(f"DRIFT DETECTED — {len(failures)} claim(s) no longer hold:")
        for f in failures:
            print(f"  • {f}")
        print("Update SKILL.md and this file together, then re-run.")
        return 1
    print("ALL CLAIMS HOLD — SKILL.md is consistent with the installed statspai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
