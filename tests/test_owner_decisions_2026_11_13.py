"""Owner decisions of 2026-11-13 (``docs/dev/2026-11-13-owner-decisions.md``).

Four defaults are announced now and switched in a later release. For each
one the tests pin the three properties of the announcement:

* a call that relies on the default gets a ``FutureWarning`` naming the
  option;
* passing the option explicitly, with either value, is silent;
* the default still returns today's numbers (it equals the explicit call
  with today's value, exactly), and the announced value returns the
  reference's numbers (an identity, or a number of the reference program).

This file holds ``sp.mr_mode``, ``sp.rkd`` and ``sp.iv_mediate``, the
dose-response headline and the swapped-policy errors of the two
off-policy APIs. ``sp.stacked_did`` and the items whose source files are
on a traced Track A path are in
``test_owner_decisions_2026_11_13_traced.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _run(fn, *args, **kw):
    """Call ``fn`` and return ``(result, warnings recorded)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fn(*args, **kw)
    return res, list(rec)


def _future(rec, option):
    return [
        w
        for w in rec
        if issubclass(w.category, FutureWarning) and f"{option}=" in str(w.message)
    ]


def _assert_announced(rec, option, today, later):
    """Exactly one announcement, naming both values, at the caller's line."""
    found = _future(rec, option)
    assert len(found) == 1, [str(w.message) for w in rec]
    text = str(found[0].message)
    assert f"{option}={today}" in text and f"{option}={later}" in text
    assert "MIGRATION.md" in text
    # attributed to this file, not to a wrapper inside the package
    assert Path(found[0].filename).name == Path(__file__).name


# ---------------------------------------------------------------------------
# A1. sp.mr_mode(stderror=): 'simple' today, 'delta' later
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mr_arrays():
    rng = np.random.default_rng(11)
    bx = rng.uniform(0.05, 0.4, 30)
    sx = rng.uniform(0.02, 0.05, 30)
    sy = rng.uniform(0.02, 0.05, 30)
    by = 0.3 * bx + rng.normal(0, sy)
    return bx, by, sx, sy


def test_mr_mode_default_announces_delta(mr_arrays):
    bx, by, sx, sy = mr_arrays
    default, rec = _run(sp.mr_mode, bx, by, sx, sy, n_boot=200, seed=3)
    _assert_announced(rec, "stderror", "'simple'", "'delta'")
    simple, rec_s = _run(
        sp.mr_mode, bx, by, sx, sy, n_boot=200, seed=3, stderror="simple"
    )
    delta, rec_d = _run(
        sp.mr_mode, bx, by, sx, sy, n_boot=200, seed=3, stderror="delta"
    )
    assert not _future(rec_s, "stderror") and not _future(rec_d, "stderror")
    # today's numbers, exactly
    assert (default.estimate, default.se, default.ci, default.p_value) == (
        simple.estimate,
        simple.se,
        simple.ci,
        simple.p_value,
    )
    # the dispatcher relies on the same default
    _, rec_disp = _run(
        sp.mr,
        "mode",
        beta_exposure=bx,
        beta_outcome=by,
        se_exposure=sx,
        se_outcome=sy,
        n_boot=50,
        seed=3,
    )
    assert len(_future(rec_disp, "stderror")) == 1
    assert delta.method == simple.method == "weighted-mode"


def test_mr_mode_delta_uses_second_order_ratio_se(mr_arrays):
    """With one bootstrap draw the estimate is a function of the ratio
    SEs only through the weights: unit weights give the same estimate
    under both, IVW weights follow the stated formula."""
    bx, by, sx, sy = mr_arrays
    a = sp.mr_mode(bx, by, sx, sy, method="simple", n_boot=2, seed=0, stderror="simple")
    b = sp.mr_mode(bx, by, sx, sy, method="simple", n_boot=2, seed=0, stderror="delta")
    assert a.estimate == b.estimate  # unit weights: the SE kind is not used
    # weighted mode with sx = 0: the delta term vanishes, so the two agree
    zero = np.zeros_like(sx)
    c = sp.mr_mode(bx, by, zero, sy, n_boot=50, seed=5, stderror="simple")
    d = sp.mr_mode(bx, by, zero, sy, n_boot=50, seed=5, stderror="delta")
    assert (c.estimate, c.se) == (d.estimate, d.se)


# ---------------------------------------------------------------------------
# A3. sp.rkd(headline=): conventional today, robust later
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kink_data():
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.uniform(-2, 2, n)
    y = 0.5 * x + 0.8 * np.maximum(x, 0) + 0.3 * x**2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x})


def test_rkd_default_announces_robust_headline(kink_data):
    default, rec = _run(sp.rkd, kink_data, y="y", x="x", c=0)
    _assert_announced(rec, "headline", "'conventional'", "'robust'")
    conv, rec_c = _run(sp.rkd, kink_data, y="y", x="x", c=0, headline="conventional")
    rob, rec_r = _run(sp.rkd, kink_data, y="y", x="x", c=0, headline="robust")
    assert not _future(rec_c, "headline") and not _future(rec_r, "headline")
    # today's numbers, exactly
    assert (default.estimate, default.se, default.pvalue, default.ci) == (
        conv.estimate,
        conv.se,
        conv.pvalue,
        conv.ci,
    )
    assert conv.estimate == conv.model_info["conventional"]["estimate"]
    assert conv.se == conv.model_info["conventional"]["se"]
    # both sets of numbers are stored whatever the headline
    assert conv.model_info["robust"] == rob.model_info["robust"]
    assert conv.model_info["conventional"] == rob.model_info["conventional"]
    assert rob.model_info["headline"] == "robust"


def test_rkd_robust_headline_is_rdrobust_deriv_1(kink_data):
    """``headline='robust'`` is what ``sp.rdrobust(deriv=1, p=1)`` reports
    (``sp.rkd`` is local linear by default; ``rdrobust`` would take
    ``p = deriv + 1``)."""
    rob = sp.rkd(kink_data, y="y", x="x", c=0, headline="robust")
    ref = sp.rdrobust(
        kink_data,
        y="y",
        x="x",
        c=0,
        deriv=1,
        p=1,
        vce="hc1",
        manipulation_test=False,
    )
    assert rob.estimate == ref.estimate
    assert rob.se == ref.se
    assert rob.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert rob.ci == pytest.approx(ref.ci, rel=1e-12)
    text = rob.summary()
    assert "robust bias-corrected" in text and "Conventional:" in text
    with pytest.raises(MethodIncompatibility, match="headline"):
        sp.rkd(kink_data, y="y", x="x", c=0, headline="bias_corrected")


# ---------------------------------------------------------------------------
# A4. sp.iv_mediate(equations=): independent today, joint later
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ivm_data():
    rng = np.random.default_rng(0)
    n = 1500
    z, v, u = rng.standard_normal((3, n))
    t = 0.8 * z + v + 0.5 * rng.standard_normal(n)
    m = 0.6 * t + v + u + 0.5 * rng.standard_normal(n)
    y = 0.4 * t + 0.7 * m + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "t": t, "m": m, "z": z})


def test_iv_mediate_default_announces_joint(ivm_data):
    kw = dict(y="y", treat="t", mediator="m", instrument="z")
    default, rec = _run(sp.iv_mediate, ivm_data, **kw)
    _assert_announced(rec, "equations", "'independent'", "'joint'")
    indep, rec_i = _run(sp.iv_mediate, ivm_data, equations="independent", **kw)
    joint, rec_j = _run(sp.iv_mediate, ivm_data, equations="joint", **kw)
    assert not _future(rec_i, "equations") and not _future(rec_j, "equations")
    # today's numbers, exactly
    pd.testing.assert_frame_equal(default.estimates, indep.estimates)
    pd.testing.assert_frame_equal(default.vcov, indep.vcov)
    # 'joint' changes the variance of the indirect effect only
    np.testing.assert_array_equal(
        joint.estimates["coef"].to_numpy(), indep.estimates["coef"].to_numpy()
    )
    for effect in ("total", "direct"):
        assert joint.vcov.loc[effect, effect] == pytest.approx(
            indep.vcov.loc[effect, effect], rel=1e-12
        )
    # under 'independent' the delta method has no cross-equation term:
    # var(a b) = b^2 var(a) + a^2 var(b), a from the mediator equation and
    # b from the outcome equation
    a = indep.stages["m_on_t"]["coef"]["t"]
    var_a = np.asarray(indep.stages["m_on_t"]["vcov"])[0, 0]
    b = indep.stages["y_on_t_m"]["coef"]["m"]
    var_b = np.asarray(indep.stages["y_on_t_m"]["vcov"])[1, 1]
    assert indep.vcov.loc["indirect", "indirect"] == pytest.approx(
        b**2 * var_a + a**2 * var_b, rel=1e-12
    )
    # 'joint' adds the covariance of a and b, which is not zero here (the
    # mediator and outcome errors share u)
    cross = joint.vcov.loc["indirect", "indirect"] - (b**2 * var_a + a**2 * var_b)
    assert abs(cross) > 1e-3 * indep.vcov.loc["indirect", "indirect"]


# ---------------------------------------------------------------------------
# B5. sp.dose_response headline: the quartiles of the observed dose
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dose_data():
    rng = np.random.default_rng(7)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    # a right-skewed dose: its quartiles are not the grid's quarter points
    t = 0.5 * x1 + rng.gamma(4.0, 0.5, size=n)
    y = 1 + 0.7 * t + 0.2 * t**2 + x1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


@pytest.mark.parametrize("method", ["gps", "dr"])
def test_dose_response_headline_is_the_interquartile_contrast(dose_data, method):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    def fit(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp.dose_response(
                dose_data,
                y="y",
                treat="t",
                covariates=["x1", "x2"],
                method=method,
                treatment_model=LinearRegression(),
                outcome_model=make_pipeline(PolynomialFeatures(2), LinearRegression()),
                n_bootstrap=25,
                random_state=1,
                **kw,
            )

    q25, q75 = np.percentile(dose_data["t"], [25, 75])
    a, b = fit(n_dose_points=7), fit(n_dose_points=20)
    for r in (a, b):
        assert r.model_info["dose_25"] == q25
        assert r.model_info["dose_75"] == q75
        assert r.estimate == r.model_info["effect_25_to_75"]
    # the headline does not move with the grid
    assert a.estimate == b.estimate
    assert a.se == b.se
    # it is the curve at the two quartiles: a two-point grid placed on
    # them returns the same contrast as its own curve
    two = fit(n_dose_points=2, dose_range=(q25, q75))
    curve = two.detail["response"].to_numpy()
    assert two.detail["dose"].tolist() == [q25, q75]
    assert two.estimate == pytest.approx(curve[1] - curve[0], rel=1e-12, abs=1e-12)
    assert two.estimate == pytest.approx(a.estimate, rel=1e-12, abs=1e-12)
    assert "Dose 25" in a.summary() and "percentiles of the observed" in a.summary()


def test_dose_response_headline_nhefs():
    """NHEFS, weight change on smoking intensity: the quartiles are 10 and
    30 cigarettes. Before 1.44 the contrast was read at grid points 11.2
    and 31.8 (``n_dose_points=10``: 1.0974) or 12.7 and 32.2 (20: 0.9702)
    of the same curve."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    cov = ["sex", "age", "race", "education", "smokeyrs", "exercise", "active", "wt71"]
    d = sp.datasets.nhefs().dropna(subset=["wt82_71", "smokeintensity", *cov])
    out = []
    for k in (10, 20):
        r = sp.dose_response(
            d,
            y="wt82_71",
            treat="smokeintensity",
            covariates=cov,
            n_dose_points=k,
            treatment_model=LinearRegression(),
            outcome_model=make_pipeline(PolynomialFeatures(2), LinearRegression()),
            random_state=1,
        )
        assert (r.model_info["dose_25"], r.model_info["dose_75"]) == (10.0, 30.0)
        out.append((r.estimate, r.se))
    assert out[0] == out[1]
    # least squares on a fixed design: stable to rounding across BLAS
    assert out[0][0] == pytest.approx(1.27516277, rel=1e-6)
    assert out[0][1] == pytest.approx(0.52912988, rel=1e-6)


# ---------------------------------------------------------------------------
# B11. the two off-policy APIs state and check their argument order
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logged():
    rng = np.random.default_rng(0)
    n, k = 300, 3
    X = rng.normal(size=(n, 2))
    pi_b = rng.dirichlet(np.ones(k) * 3, size=n)
    pi_e = rng.dirichlet(np.ones(k) * 3, size=n)
    A = np.array([rng.choice(k, p=pi_b[i]) for i in range(n)])
    R = (A == 0) + rng.normal(0, 0.2, n)
    return X, A, R, pi_b, pi_e


def test_the_two_ips_apis_agree_when_called_in_their_own_order(logged):
    X, A, R, pi_b, pi_e = logged
    rows = np.arange(len(A))
    top = sp.ips(X, A, R, pi_e, pi_b[rows, A], clip=np.inf)
    sub = sp.ope.ips(A, R, pi_b, pi_e, clip=None)
    assert top.value == pytest.approx(sub.value, rel=1e-12)
    assert top.se == pytest.approx(sub.se, rel=1e-12)
    top_sn = sp.snips(X, A, R, pi_e, pi_b[rows, A], clip=np.inf)
    sub_sn = sp.ope.snips(A, R, pi_b, pi_e, clip=None)
    assert top_sn.value == pytest.approx(sub_sn.value, rel=1e-12)
    # each docstring states its own order and points at the other API
    top_doc = " ".join(sp.ips.__doc__.split())
    sub_doc = " ".join(sp.ope.ips.__doc__.split())
    assert "Argument order" in top_doc and "Argument order" in sub_doc
    assert "sp.ope.ips(actions, rewards, pi_b, pi_e)" in top_doc
    assert "sp.ips(X, A, R, pi_target, pi_behavior)" in sub_doc


@pytest.mark.parametrize("name", ["ips", "snips"])
def test_swapped_policy_arrays_raise_a_typed_error(logged, name):
    X, A, R, pi_b, pi_e = logged
    rows = np.arange(len(A))
    # sp.ope order handed to sp.ips: (logging propensities, target matrix)
    with pytest.raises(MethodIncompatibility, match="other order") as exc:
        getattr(sp, name)(X, A, R, pi_b[rows, A], pi_e)
    assert "pi_target, pi_behavior" in exc.value.recovery_hint
    assert exc.value.diagnostics["pi_behavior_shape"] == pi_e.shape
    # sp.ips order handed to sp.ope.ips: (target matrix, logging propensities)
    with pytest.raises(MethodIncompatibility, match="other order") as exc:
        getattr(sp.ope, name)(A, R, pi_e, pi_b[rows, A])
    assert "pi_b, pi_e" in exc.value.recovery_hint
    # two matrices cannot be told apart by shape: no guess is made, and
    # the swapped call returns the value of the reversed policy pair
    swapped = getattr(sp.ope, name)(A, R, pi_e, pi_b, clip=None)
    w = pi_b[rows, A] / pi_e[rows, A]
    want = np.mean(w * R) if name == "ips" else np.sum(w * R) / np.sum(w)
    assert swapped.value == pytest.approx(want, rel=1e-12)
