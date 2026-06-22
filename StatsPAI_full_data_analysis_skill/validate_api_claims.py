#!/usr/bin/env python3
"""Validation gate for the StatsPAI full-data-analysis skill.

SkillOpt (https://github.com/microsoft/SkillOpt) treats a skill document as a
trainable artifact whose edits are accepted only when they pass a validation
gate. This script is that gate for ``SKILL.md``: it turns the static
"verified against statspai X.Y.Z" stamp into a runnable check that any agent or
CI job can re-run against whatever ``statspai`` is installed.

It verifies three layers of perishable claims the skill makes:

  1. EXISTENCE   — every ``sp.<name>`` referenced in SKILL.md resolves.
  2. SIGNATURES  — the documented keyword/positional argument names exist.
  3. ATTRIBUTES  — the documented result-object attributes / return shapes hold
                   (the layer that drifts silently when the library adds a
                   convenience method, e.g. ``AFTResult.params`` in 1.19.0).

Usage
-----
    python validate_api_claims.py            # full gate; exits non-zero on drift
    python validate_api_claims.py --quick    # existence + signatures only (no fits)

Keep this green. When it goes red, either the library changed (update SKILL.md +
this file together) or a skill edit introduced a false claim.
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import re
import types

import statspai as sp

SKILL_MD = pathlib.Path(__file__).with_name("SKILL.md")

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
    print("\n== 1. EXISTENCE: every sp.<name> in SKILL.md resolves ==")
    text = SKILL_MD.read_text(encoding="utf-8")
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


# ---------------------------------------------------------------- attributes
def _make_data():
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
            and not hasattr(ml, "conf_int")  # claim: no such method
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
            and not hasattr(aft, "to_word")  # still goes via regtable
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


def _silent_ok(thunk) -> bool:
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

    print(f"statspai {sp.__version__}  ·  validating SKILL.md API claims")
    failures: list[str] = []
    check_references(failures)
    check_modules(failures)
    check_signatures(failures)
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
