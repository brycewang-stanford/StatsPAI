"""Diagnostic coverage — no estimator may silently skip a diagnostic.

A ``result.violations()`` check only fires if the estimator populated the
``model_info`` key it reads. When one IV entry point stores ``first_stage_f``
and another does not, the weak-instrument warning silently vanishes for the
second — exactly the kind of gap that erodes trust (it was real for ``sp.liml``
and ``sp.jive`` until fixed). This suite pins the whole IV family: every
estimator must record the first-stage strength and flag a weak instrument, and
none may cry wolf on a strong one. A new IV estimator that forgets the
diagnostic fails here instead of shipping a blind spot.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _iv_df(first_stage_coef: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = first_stage_coef * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


# (id, fitter) — every single-endogenous IV estimator StatsPAI exposes.
_IV_ESTIMATORS = [
    ("ivreg_2sls", lambda df: sp.ivreg("y ~ (d ~ z)", data=df)),
    ("iv_2sls", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="2sls")),
    ("iv_liml", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="liml")),
    ("iv_fuller", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="fuller")),
    ("iv_gmm", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="gmm")),
    ("liml", lambda df: sp.liml("y ~ (d ~ z)", data=df)),
    ("jive", lambda df: sp.jive(df, y="y", x_endog=["d"], z=["z"])),
]


@pytest.mark.parametrize("name,fit", _IV_ESTIMATORS, ids=[e[0] for e in _IV_ESTIMATORS])
def test_iv_estimator_records_first_stage_and_flags_weak(name, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = fit(_iv_df(0.03))
        strong = fit(_iv_df(2.0))

    assert weak.model_info.get("first_stage_f") is not None, (
        f"{name}: model_info['first_stage_f'] is missing — weak IV would be "
        "silently skipped by result.violations()"
    )
    assert "weak_instrument" in {
        v["test"] for v in weak.violations()
    }, f"{name}: a weak first stage did not surface in violations()"
    assert "weak_instrument" not in {
        v["test"] for v in strong.violations()
    }, f"{name}: false-positive weak-instrument flag on a strong first stage"


# --------------------------------------------------------------------------- #
#  Panel — every clustered method records n_clusters
# --------------------------------------------------------------------------- #


def _panel_df(n_units: int, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        a = rng.normal()
        for t in range(n_periods):
            x = rng.normal()
            rows.append({"id": i, "yr": t, "x": x, "y": x + a + rng.normal()})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["fe", "re", "twoway", "fd", "pooled"])
def test_panel_method_records_n_clusters_and_flags_few(method):
    def fit(df):
        return sp.panel(
            df, "y ~ x", entity="id", time="yr", method=method, cluster="entity"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        few = fit(_panel_df(12))
        many = fit(_panel_df(60))
    assert few.model_info.get("n_clusters") == 12, f"panel({method}) drops n_clusters"
    assert "few_clusters" in {v["test"] for v in few.violations()}
    assert "few_clusters" not in {v["test"] for v in many.violations()}


# --------------------------------------------------------------------------- #
#  Cluster-robust regressions — every ``cluster=`` estimator records n_clusters
# --------------------------------------------------------------------------- #


def _clustered_df(n_clusters: int, n: int = 1200, kind: str = "cont", seed: int = 1):
    """A clustered dataset with the requested number of clusters. ``kind``
    selects an outcome the estimator family accepts (continuous / binary /
    count)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_clusters, n)
    x = rng.normal(size=n)
    if kind == "bin":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(int)
    elif kind == "count":
        y = rng.poisson(np.exp(0.4 + 0.3 * x))
    else:
        y = x + rng.normal(size=n)
    return pd.DataFrame(
        {"y": y, "x": x, "g": g, "d": x + rng.normal(size=n), "z": rng.normal(size=n)}
    )


# (id, fitter, outcome-kind) — every regression entry point that accepts a
# cluster spec. Each must record n_clusters so few-cluster CRV inference is
# flagged; feols routes clusters through pyfixest's ``_G`` instead of a
# ``cluster=`` kwarg, but must land in the same diagnostic.
_CLUSTER_ESTIMATORS = [
    ("regress", lambda df: sp.regress("y ~ x", data=df, cluster="g"), "cont"),
    ("ivreg", lambda df: sp.ivreg("y ~ (d ~ z)", data=df, cluster="g"), "cont"),
    ("logit", lambda df: sp.logit("y ~ x", data=df, cluster="g"), "bin"),
    ("probit", lambda df: sp.probit("y ~ x", data=df, cluster="g"), "bin"),
    ("poisson", lambda df: sp.poisson("y ~ x", data=df, cluster="g"), "count"),
    ("nbreg", lambda df: sp.nbreg("y ~ x", data=df, cluster="g"), "count"),
    ("feols", lambda df: sp.feols("y ~ x", data=df, vcov={"CRV1": "g"}), "cont"),
]


@pytest.mark.parametrize(
    "name,fit,kind",
    _CLUSTER_ESTIMATORS,
    ids=[e[0] for e in _CLUSTER_ESTIMATORS],
)
def test_cluster_estimator_records_n_clusters_and_flags_few(name, fit, kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        few = fit(_clustered_df(12, kind=kind))
        many = fit(_clustered_df(60, kind=kind))
    assert few.model_info.get("n_clusters") == 12, (
        f"{name}: model_info['n_clusters'] missing — few-cluster CRV inference "
        "would be silently skipped by result.violations()"
    )
    assert "few_clusters" in {
        v["test"] for v in few.violations()
    }, f"{name}: 12 clusters did not surface as few_clusters in violations()"
    assert "few_clusters" not in {
        v["test"] for v in many.violations()
    }, f"{name}: false-positive few-cluster flag with 60 clusters"


# --------------------------------------------------------------------------- #
#  Matching — every method records the post-match balance table
# --------------------------------------------------------------------------- #


def _confounded(strength: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


@pytest.mark.parametrize("method", ["psm", "nearest", "mahalanobis", "cem"])
def test_match_method_records_balance_and_flags_imbalance(method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(
            _confounded(1.5), y="y", treat="d", covariates=["x1", "x2"], method=method
        )
    assert isinstance(
        r.model_info.get("balance"), pd.DataFrame
    ), f"match({method}) does not record a balance table"
    assert "balance" in {v["test"] for v in r.violations()}


def test_cbps_reports_residual_imbalance():
    """CBPS stores balance under std_mean_diff_after (not `balance`) and is not
    tagged 'matching' — residual imbalance must still surface."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.cbps(_confounded(1.5), y="y", treat="d", covariates=["x1", "x2"])
    assert isinstance(r.model_info.get("std_mean_diff_after"), dict)
    assert "balance" in {v["test"] for v in r.violations()}


# --------------------------------------------------------------------------- #
#  Count — Poisson flags over-dispersion and excess zeros
# --------------------------------------------------------------------------- #


def test_poisson_flags_overdispersion_and_excess_zeros():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    lam = np.exp(0.5 + 0.8 * x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        over = sp.poisson(
            "y ~ x",
            data=pd.DataFrame({"y": rng.poisson(lam * rng.gamma(1, 1, 600)), "x": x}),
        )
        zinfl = pd.DataFrame({"y": rng.poisson(lam), "x": x})
        zinfl.loc[rng.uniform(size=600) < 0.4, "y"] = 0
        zi = sp.poisson("y ~ x", data=zinfl)
        clean = sp.poisson("y ~ x", data=pd.DataFrame({"y": rng.poisson(lam), "x": x}))
    assert "overdispersion" in {v["test"] for v in over.violations()}
    assert "excess_zeros" in {v["test"] for v in zi.violations()}
    clean_tests = {v["test"] for v in clean.violations()}
    assert "overdispersion" not in clean_tests and "excess_zeros" not in clean_tests


# --------------------------------------------------------------------------- #
#  DML / IPW / TMLE — propensity overlap
# --------------------------------------------------------------------------- #


def _confounded_overlap(strength: float, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


_OVERLAP_ESTIMATORS = [
    (
        "dml_irm",
        lambda df: sp.dml(df, y="y", treat="d", covariates=["x1", "x2"], model="irm"),
    ),
    ("tmle", lambda df: sp.tmle(df, y="y", treat="d", covariates=["x1", "x2"])),
    ("ipw", lambda df: sp.ipw(df, y="y", treat="d", covariates=["x1", "x2"])),
]


@pytest.mark.parametrize(
    "name,fit", _OVERLAP_ESTIMATORS, ids=[e[0] for e in _OVERLAP_ESTIMATORS]
)
def test_propensity_estimator_flags_weak_overlap(name, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strong = fit(_confounded_overlap(4.0))
        good = fit(_confounded_overlap(0.6))
    assert "dml_overlap" in {
        v["test"] for v in strong.violations()
    }, f"{name}: weak overlap did not surface in violations()"
    assert "dml_overlap" not in {
        v["test"] for v in good.violations()
    }, f"{name}: false-positive overlap flag on a well-overlapped design"


# --------------------------------------------------------------------------- #
#  Synthetic control — pre-fit diagnostics are recorded
# --------------------------------------------------------------------------- #


def _synth_bad_fit() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for u in ["T"] + [f"D{i}" for i in range(8)]:
        base = 50 if u == "T" else rng.uniform(10, 30)
        slope = 5.0 if u == "T" else rng.uniform(-1, 1)
        for yr in range(1980, 1995):
            rows.append(
                {"u": u, "yr": yr, "y": base + slope * (yr - 1980) + rng.normal(0, 1)}
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "name,fit",
    [
        (
            "synth",
            lambda df: sp.synth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
        (
            "augsynth",
            lambda df: sp.augsynth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
        (
            "gsynth",
            lambda df: sp.gsynth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
    ],
)
def test_synth_records_prefit_diagnostics(name, fit):
    """Every SCM variant must record the pre-fit inputs so synth_prefit can be
    assessed (augsynth/gsynth fit the pre-period well, so they need not fire —
    but they must not be blind)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit(_synth_bad_fit())
    mi = r.model_info
    for key in ("pre_treatment_rmse", "Y_treated", "times", "treatment_time"):
        assert mi.get(key) is not None, f"{name}: model_info['{key}'] missing"


def test_plain_scm_flags_unmatchable_pre_trend():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.synth(
            _synth_bad_fit(),
            unit="u",
            time="yr",
            outcome="y",
            treated_unit="T",
            treatment_time=1990,
        )
    assert "synth_prefit" in {v["test"] for v in r.violations()}


# --------------------------------------------------------------------------- #
#  Limited-dependent / survival — assumption diagnostics
#
#  These estimators each carry a signature assumption whose violation quietly
#  invalidates the headline coefficient: Cox's proportional hazards, Tobit's
#  usable variation under censoring, Heckman's numerical identification, and a
#  logit's finiteness under separation. Each check only fires if the estimator
#  stored the statistic it reads (ph_test / censor_pct / rho / coefs) — pin the
#  storage AND the fire/clean behaviour so none silently regresses to a blind
#  spot the way the IV family once did.
# --------------------------------------------------------------------------- #


def test_cox_flags_nonproportional_hazards():
    """PH-violating data (covariate trends with failure time) must reject the
    proportional-hazards test; textbook proportional data must not."""
    rng = np.random.default_rng(11)
    n = 700
    t_bad = np.sort(rng.exponential(1.0, n)) + 0.01
    x_bad = np.linspace(-2, 2, n) + rng.normal(0, 0.3, n)  # x rises with time
    xg = np.random.default_rng(3).normal(size=n)
    t_good = -np.log(np.random.default_rng(3).uniform(size=n)) / np.exp(0.8 * xg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = sp.cox(
            data=pd.DataFrame({"t": t_bad, "d": np.ones(n, int), "x": x_bad}),
            duration="t",
            event="d",
            x=["x"],
        )
        good = sp.cox(
            data=pd.DataFrame({"t": t_good + 0.001, "d": np.ones(n, int), "x": xg}),
            duration="t",
            event="d",
            x=["x"],
        )
    assert bad.model_info.get("ph_test") is not None, (
        "cox: model_info['ph_test'] missing — the proportional-hazards check "
        "would be silently skipped by result.violations()"
    )
    assert "proportional_hazards" in {v["test"] for v in bad.violations()}
    assert "proportional_hazards" not in {v["test"] for v in good.violations()}


def test_tobit_flags_extreme_censoring():
    rng = np.random.default_rng(5)
    n = 800
    x = rng.normal(size=n)
    y_bad = np.maximum(-3.0 + 1.0 * x + rng.normal(size=n), 0.0)  # ~98% at floor
    y_good = np.maximum(1.0 + 2.0 * x + rng.normal(size=n), 0.0)  # ~33% censored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = sp.tobit(pd.DataFrame({"y": y_bad, "x": x}), y="y", x=["x"], ll=0)
        good = sp.tobit(pd.DataFrame({"y": y_good, "x": x}), y="y", x=["x"], ll=0)
    assert bad.model_info.get("censor_pct") is not None, (
        "tobit: model_info['censor_pct'] missing — the extreme-censoring check "
        "would be silently skipped by result.violations()"
    )
    assert "extreme_censoring" in {v["test"] for v in bad.violations()}
    assert "extreme_censoring" not in {v["test"] for v in good.violations()}


def test_heckman_flags_rho_boundary():
    """When the outcome error is (near-)perfectly correlated with the selection
    error, rho hits the ±1 boundary — a numerical red flag the check must
    surface; a well-identified moderate-rho fit must not."""
    rng_b = np.random.default_rng(9)
    n = 600
    z = rng_b.normal(size=n)
    x = rng_b.normal(size=n)
    u = rng_b.normal(size=n)
    sel = 0.5 + 0.8 * z + u > 0
    y = 1 + 2 * x + 3 * u  # outcome error == selection error => rho -> 1
    boundary_df = pd.DataFrame(
        {"y": np.where(sel, y, np.nan), "x": x, "z": z, "sel": sel.astype(int)}
    )

    rng_c = np.random.default_rng(5)
    m = 2000
    zc = rng_c.normal(size=m)
    xc = rng_c.normal(size=m)
    uc = rng_c.normal(size=m)
    epsc = 0.6 * uc + np.sqrt(1 - 0.6**2) * rng_c.normal(size=m)  # rho ~ 0.6
    selc = 0.3 + 1.0 * zc + 0.5 * xc + uc > 0
    yc = 1 + 2 * xc + 3 * epsc
    clean_df = pd.DataFrame(
        {"y": np.where(selc, yc, np.nan), "x": xc, "z": zc, "sel": selc.astype(int)}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boundary = sp.heckman(boundary_df, y="y", x=["x"], select="sel", z=["z"])
        clean = sp.heckman(clean_df, y="y", x=["x"], select="sel", z=["z"])
    assert boundary.model_info.get("rho") is not None, (
        "heckman: model_info['rho'] missing — the rho-boundary check would be "
        "silently skipped by result.violations()"
    )
    assert "heckman_rho_boundary" in {v["test"] for v in boundary.violations()}
    assert "heckman_rho_boundary" not in {v["test"] for v in clean.violations()}


def test_logit_flags_separation():
    rng = np.random.default_rng(0)
    n = 400
    xs = rng.normal(size=n)
    y_sep = (xs > 0).astype(int)  # outcome perfectly predicted by x
    xn = rng.normal(size=n)
    y_clean = (rng.uniform(size=n) < 1 / (1 + np.exp(-xn))).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sep = sp.logit("y ~ x", data=pd.DataFrame({"y": y_sep, "x": xs}))
        clean = sp.logit("y ~ x", data=pd.DataFrame({"y": y_clean, "x": xn}))
    assert "separation" in {v["test"] for v in sep.violations()}
    assert "separation" not in {v["test"] for v in clean.violations()}


# --------------------------------------------------------------------------- #
#  Actionability contract — every violation must say what to do
# --------------------------------------------------------------------------- #
#
# The suite above proves every estimator *flags* its diagnostic. This capstone
# proves every flag is *actionable*. ``sp.audit`` folds a live violation in as
# ``{"question": v["message"], "rationale": v["recovery_hint"],
#    "suggest_function": v["alternatives"][0] if alternatives else ""}``. A
# violation with an empty ``message`` or ``recovery_hint`` therefore surfaces in
# audit as a flagged failure with no guidance — fail-loud degenerating into
# "cries wolf". The static test below parses ``_agent_summary.py`` and asserts
# every violation dict literal carries a non-empty ``message`` and
# ``recovery_hint``, so a newly-added violation that forgets the recovery path
# fails here rather than shipping an actionless finding.


def _violation_string_value(node):
    """Best-effort evaluation of an AST node to its string content.

    Handles the four shapes the violation constructors actually use for
    ``message`` / ``recovery_hint``: a plain ``str`` constant, an f-string
    (``JoinedStr``), implicit/``+`` concatenation of those (``BinOp`` with
    ``Add``), and a conditional (``a if cond else b``). Returns the
    concatenated string, or ``None`` if the node is provably not a string
    expression (so the caller can flag "not a string").
    """
    import ast

    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        # An f-string interpolation part contributes non-empty content; only
        # the literal pieces carry known text, so stand in "X" for {expr}.
        return "".join(
            p.value if isinstance(p, ast.Constant) else "X" for p in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _violation_string_value(node.left)
        right = _violation_string_value(node.right)
        if left is None and right is None:
            return None
        return (left or "") + (right or "")
    if isinstance(node, ast.IfExp):
        return _violation_string_value(node.body) or _violation_string_value(
            node.orelse
        )
    return None


def _violation_dict_literals():
    """Every dict literal in ``_agent_summary.py`` that constructs a violation.

    A violation is identified structurally by a ``"test":`` key — the field
    ``violations()`` / ``audit`` branch on. Parsed from source (not by running
    the constructors) so the contract holds for *every* branch, including ones
    a fixture would find hard to trigger (e.g. NaN estimate, divergences).
    """
    import ast
    import pathlib

    import statspai.core._agent_summary as agg

    src = pathlib.Path(agg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "test" for k in node.keys
        ):
            fields = {
                k.value: v
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant)
            }
            out.append(fields)
    return out


def test_every_violation_is_actionable():
    """Static contract: every violation carries a non-empty message + recovery
    hint, so audit never flags a problem without a recovery path."""
    import ast

    literals = _violation_dict_literals()
    # Sanity: the parser found the population (guards against a refactor that
    # renames the field and silently empties this test).
    assert len(literals) >= 20, (
        f"expected >=20 violation dict literals, found {len(literals)} — did the "
        "'test' key get renamed?"
    )

    problems = []
    for fields in literals:
        test_node = fields.get("test")
        name = (
            test_node.value
            if isinstance(test_node, ast.Constant)
            else "<dynamic test name>"
        )
        for required in ("message", "recovery_hint"):
            if required not in fields:
                problems.append(f"{name}: missing '{required}'")
                continue
            value = _violation_string_value(fields[required])
            if value is None:
                problems.append(f"{name}: '{required}' is not a string expression")
            elif not value.strip():
                problems.append(f"{name}: '{required}' is empty")

    assert not problems, "non-actionable violations found:\n  " + "\n  ".join(problems)


def test_audit_folds_live_violation_with_actionable_content():
    """Runtime companion: a real fit that trips a violation must surface in
    ``sp.audit`` as a failed check carrying either a suggest_function or a
    rationale — the fold-in path, end to end."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.regress("y ~ x", data=_clustered_df(12), cluster="g")
    assert "few_clusters" in {v["test"] for v in fit.violations()}

    card = sp.audit(fit)
    failed = [c for c in card["checks"] if c["status"] == "failed"]
    assert failed, "expected at least one failed check on a 12-cluster fit"
    for c in failed:
        assert c.get("suggest_function") or c.get("rationale"), (
            f"audit check {c['name']!r} is a failed finding with neither a "
            "suggest_function nor a rationale — actionless"
        )


# --------------------------------------------------------------------------- #
#  Reachability contract — every recovery pointer must be a real function
# --------------------------------------------------------------------------- #
#
# ``test_every_violation_is_actionable`` proves the recovery text is non-empty.
# This goes one step further: the ``sp.xxx`` names an agent is told to run —
# violation ``alternatives`` and audit ``suggest_function`` — must resolve to a
# *registered, callable* function. A hint that says "run sp.mccrary" when the
# real name is ``sp.mccrary_test`` is a dead link: the agent follows the
# guidance straight into an AttributeError, which is worse than no guidance at
# all. Nine such dead links existed (sp.mccrary, sp.oster, sp.cinelli_hazlett,
# sp.hansen_j, sp.balance, sp.overlap_check, sp.rd_placebo,
# sp.rd_bandwidth_sensitivity, sp.synth_placebo) plus two in a violation
# (sp.rd_donut, sp.bounds); this test locks the repair so no future recovery
# pointer can regress into a dead link.


def _sp_refs_from_alternatives():
    """Every ``sp.xxx`` in a violation ``alternatives`` list (parsed from
    ``_agent_summary.py`` source, so all branches are covered)."""
    import ast
    import importlib
    import pathlib

    src = pathlib.Path(
        importlib.import_module("statspai.core._agent_summary").__file__
    ).read_text(encoding="utf-8")
    refs = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "test" for k in node.keys
        ):
            for k, v in zip(node.keys, node.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "alternatives"
                    and isinstance(v, ast.List)
                ):
                    for elt in v.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            refs.add(elt.value)
    return refs


def _sp_refs_from_audit_suggestions():
    """Every ``suggest_function`` string literal in ``audit.py``."""
    import ast
    import importlib
    import pathlib

    src = pathlib.Path(
        importlib.import_module("statspai.smart.audit").__file__
    ).read_text(encoding="utf-8")
    refs = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "suggest_function"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            refs.add(node.value.value)
    return refs


@pytest.mark.parametrize(
    "source",
    [_sp_refs_from_alternatives, _sp_refs_from_audit_suggestions],
    ids=["violation_alternatives", "audit_suggest_function"],
)
def test_recovery_pointers_resolve_to_registered_functions(source):
    """Every recovery pointer (``sp.xxx``) must name a registered function —
    a dead link sends an agent following the guidance into an error."""
    registered = set(sp.list_functions())
    dead = []
    for ref in sorted(source()):
        if not ref:  # empty string = "no suggestion", legitimately allowed
            continue
        name = ref[3:] if ref.startswith("sp.") else ref
        if name not in registered:
            dead.append(ref)
    assert not dead, (
        "recovery pointers name functions that are not registered — an agent "
        "following the hint hits AttributeError:\n  " + "\n  ".join(dead)
    )


# --------------------------------------------------------------------------- #
#  Prose reachability — sp.xxx mentioned inline in guidance text must resolve
# --------------------------------------------------------------------------- #
#
# The two tests above cover the *structured* recovery pointers (a violation's
# ``alternatives`` list, an audit check's ``suggest_function``). But guidance
# text itself — ``recovery_hint`` / ``rationale`` / ``message`` / ``question``
# — routinely names functions inline ("Report sp.wild_cluster_bootstrap …",
# "use sp.iv_bounds …"). An LLM agent reads that prose and calls what it names,
# so an inline ``sp.xxx`` that has been renamed or never existed is just as much
# a dead link as a broken list entry. This closes the loop: every ``sp.xxx`` an
# agent can encounter — structured OR prose — resolves to a callable.


def _prose_sp_pointers():
    """Every ``sp.xxx`` token appearing inside an agent-facing prose field
    (``recovery_hint`` / ``rationale`` / ``message`` / ``question``) across
    ``src/statspai``. Handles str / f-string / (+)-concat / conditional
    string expressions; strips trailing sentence punctuation."""
    import ast
    import importlib
    import pathlib
    import re

    prose_keys = {"recovery_hint", "rationale", "message", "question"}
    token = re.compile(r"\bsp\.[A-Za-z_][A-Za-z0-9_.]*")

    def _text(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(_text(p) or "" for p in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return (_text(node.left) or "") + (_text(node.right) or "")
        if isinstance(node, ast.IfExp):
            return (_text(node.body) or "") + (_text(node.orelse) or "")
        return ""

    pkg_root = pathlib.Path(importlib.import_module("statspai").__file__).parent
    refs: dict = {}

    def _scan(text, origin):
        for m in token.findall(text):
            refs.setdefault(m.rstrip("."), set()).add(origin)

    for path in pkg_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        origin = str(path.relative_to(pkg_root))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in prose_keys:
                _scan(_text(node.value), origin)
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value in prose_keys:
                        _scan(_text(v), origin)
    return refs


def test_prose_recovery_pointers_resolve():
    """Every ``sp.xxx`` named inline in guidance prose must resolve to a
    callable on the ``sp`` namespace — an LLM reading the hint calls what it
    names, so a renamed/absent inline pointer is a dead link too."""
    refs = _prose_sp_pointers()
    assert len(refs) >= 30, (
        f"expected >=30 inline prose pointers, found {len(refs)} — did a prose "
        "field get renamed, or the scanner break?"
    )

    dead = []
    for ref in sorted(refs):
        obj = sp
        for part in ref.split(".")[1:]:  # walk past the leading 'sp'
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if not callable(obj):
            dead.append(f"{ref} <- {sorted(refs[ref])}")

    assert not dead, (
        "guidance prose names sp.xxx pointers that do not resolve to a callable "
        "— an agent reading the hint hits AttributeError:\n  " + "\n  ".join(dead)
    )
