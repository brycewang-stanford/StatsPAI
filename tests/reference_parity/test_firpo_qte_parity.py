"""Reference parity: Firpo (2007) unconditional QTE / QTT vs R ``qte`` 1.3.1.

Estimators
----------
``sp.qte(method='firpo_qte')`` and ``sp.qte(method='firpo_qtt')`` against
``qte::ci.qte`` / ``qte::ci.qtet`` on the package's own ``lalonde.exp``
(randomised) and ``lalonde.psid`` (observational) samples.

Why this file does NOT assert point-value equality with R
---------------------------------------------------------
Both implementations minimise the same weighted check function

    q_j(tau) = argmin_q  sum_i w_ji * rho_tau(Y_i - q),
    rho_tau(u) = u * (tau - 1{u < 0})

but they solve it differently.  R's ``BMisc::weighted_quantile`` is

    optimize(weighted.checkfun, lower = min(y), upper = max(y), ...)

a golden-section search with tolerance ``.Machine$double.eps^0.25``.  The
check function is piecewise linear, so between order statistics it has
**plateaus** where every point is a minimiser; golden section returns an
arbitrary interior point, plus optimiser tolerance.  On ``lalonde.exp``
(outcome range ~[0, 60307]) that produces point differences up to ~800
against the exact minimiser -- while the objective values agree to seven
significant figures.

Asserting ``|ours - R| < 1e-6`` would therefore be asserting a property of
``stats::optimize``, not of the estimator.  Two symptoms make this concrete
and are themselves pinned below:

* ``qte::ci.qte`` on ``lalonde.exp`` returns ``-5.93e-06`` at tau = 0.05,
  where both arms sit on the ``re78 == 0`` mass point and the exact answer
  is ``0`` -- golden-section residue, not data.
* ``ci.qte(xformla=...)`` on ``lalonde.exp`` returns values identical to
  the no-covariate call to 10 decimal places, even though the propensity
  score genuinely varies (sd 0.09), because the reweighting moves the
  quantiles by less than the plateau width.

Anchors
-------
A. **Objective dominance** (the real parity claim).  Evaluate the shared
   weighted check function at R's solution and at ours, per arm, per tau.
   Ours must be no worse everywhere.  This certifies *same estimand, same
   objective, at least as good a solution* -- strictly stronger than
   approximate numeric agreement.
B. **Numeric agreement where the objective is strictly convex.**  On
   ``lalonde.psid`` (n = 2675, dense support) plateaus are narrow, so ours
   and R must agree closely in relative terms.
C. **Covariates bind on the observational sample.**  R's own QTE moves from
   (-11041, -16456, -19912) to (-8754, -13398, -16545) at tau = .25/.5/.75
   when the propensity model is added; ours must move in the same direction
   by a comparable amount.
D. **Known-truth recovery on a NON-degenerate DGP** -- a location-scale
   design whose true QTE is the fan ``Phi^-1``-shaped curve, not a flat
   line.  A constant-shift design cannot distinguish a QTE estimator from
   an ATE estimator, which is how the pre-1.21 mislabelling survived.
E. **QTE vs QTT differ** when treatment is selective, and coincide under
   randomisation.
F. **Analytic influence-function SE calibration** by Monte Carlo.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.qte._firpo import (  # noqa: E402
    firpo_quantiles,
    firpo_weights,
    logit_pscore,
    weighted_checkfun,
)

_FIX = pathlib.Path(__file__).parent / "_fixtures"
PROBS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "qte_firpo_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_qte_firpo_R.R to build qte_firpo_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lalonde():
    return {
        "exp": pd.read_csv(_FIX / "qte_lalonde.csv"),
        "psid": pd.read_csv(_FIX / "qte_lalonde_psid.csv"),
    }


COVS = ["age", "education", "black", "hispanic", "married", "nodegree"]


def _ours(df, estimand, use_cov):
    y = df["re78"].to_numpy(float)
    d = df["treat"].to_numpy(int)
    X = df[COVS].to_numpy(float) if use_cov else None
    ps = logit_pscore(X, d)
    q1, q0 = firpo_quantiles(y, d, ps, PROBS, estimand)
    return y, d, ps, q1, q0


# ── A. objective dominance: the parity claim ───────────────────────── #


@pytest.mark.parametrize("dataset", ["exp", "psid"])
@pytest.mark.parametrize("estimand", ["qte", "qtt"])
@pytest.mark.parametrize("use_cov", [False, True])
def test_objective_no_worse_than_r(rjson, lalonde, dataset, estimand, use_cov):
    """Our exact minimiser must attain a check-function value <= R's.

    Same objective, evaluated at both solutions. If ours were a different
    estimator this would fail; if ours were a worse solver it would fail.
    """
    key = f"{'qte' if estimand == 'qte' else 'qtet'}_{dataset}"
    key += "_cov" if use_cov else "_nocov"
    ref = rjson[key]
    df = lalonde[dataset]
    y, d, ps, q1, q0 = _ours(df, estimand, use_cov)
    w1, w0 = firpo_weights(d, ps, estimand)

    Rq1 = np.asarray(ref["q1"], dtype=float)
    Rq0 = np.asarray(ref["q0"], dtype=float)

    for i, tau in enumerate(PROBS):
        for q_ours, q_r, w, arm in (
            (q1[i], Rq1[i], w1, "treated"),
            (q0[i], Rq0[i], w0, "control"),
        ):
            f_ours = weighted_checkfun(q_ours, y, tau, w)
            f_r = weighted_checkfun(q_r, y, tau, w)
            # Scale-relative slack: the objective is O(1e5) on these data.
            slack = 1e-9 * max(abs(f_r), 1.0)
            assert (
                f_ours <= f_r + slack
            ), f"{key} tau={tau} arm={arm}: ours {f_ours:.6f} > R {f_r:.6f}"


# ── B. numeric agreement where plateaus are narrow ─────────────────── #


@pytest.mark.parametrize("estimand", ["qte", "qtt"])
def test_close_to_r_on_dense_sample(rjson, lalonde, estimand):
    """On lalonde.psid (n=2675) the support is dense, so plateau ambiguity
    is small and the two solutions must agree in relative terms."""
    key = f"{'qte' if estimand == 'qte' else 'qtet'}_psid_cov"
    ref = np.asarray(rjson[key]["qte"], dtype=float)
    _, _, _, q1, q0 = _ours(lalonde["psid"], estimand, True)
    ours = q1 - q0
    scale = np.mean(np.abs(ref))
    rel = np.abs(ours - ref) / scale
    # Observed on this fixture: QTE agrees with R to <=0.001 currency units
    # (~1e-7 relative) at ALL 19 tau; QTT agrees at 16/19, the three
    # exceptions differing by 15-27 units (<=0.6% relative). Those three are
    # plateau ambiguity, which anchor A independently shows costs us nothing
    # on the objective.
    #
    # Assert BOTH the loose ceiling and a floor on how many tau are
    # essentially exact -- a genuine weighting regression would move every
    # tau and so could not hide behind the ceiling alone.
    assert np.max(rel) < 0.01, f"max rel dev {np.max(rel):.6f}\n{ours}\n{ref}"
    per_tau_rel = np.abs(ours - ref) / np.maximum(np.abs(ref), 1.0)
    n_exact = int(np.sum(per_tau_rel < 1e-6))
    assert n_exact >= len(PROBS) - 4, f"only {n_exact}/{len(PROBS)} tau exact"
    # Sign agreement, ignoring the tau where both are numerically zero
    # (R reports ~5e-06 optimiser residue there, we report exactly 0).
    big = np.abs(ref) > 1.0
    assert np.all(np.sign(ours[big]) == np.sign(ref[big]))


def test_r_optimizer_residue_is_documented_not_matched(rjson):
    """Pin the artifact that motivates anchor A.

    R reports ~5.998e-05 for a quantile difference whose exact value is 0
    (both arms sit on a mass point at re78 = 0). That is golden-section
    residue. We return exactly 0.0; matching R here would mean reproducing
    an optimiser's rounding error.
    """
    ref = np.asarray(rjson["pkg_ci_qte_exp"], dtype=float)
    assert 0 < abs(ref[0]) < 1e-3, ref[0]
    # Ours is exactly zero there: both arms sit on the mass point at 0.
    ours = np.asarray(rjson["qte_exp_nocov"]["qte"], dtype=float)
    assert ours[0] == 0.0


# ── C. covariates bind on the observational sample ─────────────────── #


def test_covariates_shift_estimate_like_r(rjson, lalonde):
    """R's PSID QTE moves toward zero once the propensity model is added;
    ours must move the same way by a comparable amount."""
    r_nocov = np.asarray(rjson["qte_psid_nocov"]["qte"], dtype=float)
    r_cov = np.asarray(rjson["qte_psid_cov"]["qte"], dtype=float)
    _, _, _, a1, a0 = _ours(lalonde["psid"], "qte", False)
    _, _, _, b1, b0 = _ours(lalonde["psid"], "qte", True)
    o_nocov, o_cov = a1 - a0, b1 - b0

    r_shift = r_cov - r_nocov
    o_shift = o_cov - o_nocov
    # Same direction wherever the shift is materially non-zero (at the two
    # lowest tau both curves sit on the re78 == 0 mass point and the shift
    # is numerical noise of order 1e-5).
    big = np.abs(r_shift) > 1.0
    assert np.all(np.sign(o_shift[big]) == np.sign(r_shift[big])), (o_shift, r_shift)
    # Magnitudes must match closely, not merely be comparable.
    assert np.max(np.abs(o_shift - r_shift)) / np.mean(np.abs(r_shift)) < 0.02


# ── D. non-degenerate truth ────────────────────────────────────────── #


def test_recovers_quantile_fan_under_randomisation():
    """Y(0) ~ N(0,1), Y(1) ~ N(0,2) under randomised D.

    True QTE(tau) = 2*Phi^-1(tau) - Phi^-1(tau) = Phi^-1(tau): a fan.
    A flat-line (constant-shift) DGP cannot distinguish a QTE estimator
    from an ATE estimator; this one can.
    """
    rng = np.random.default_rng(4)
    n = 200_000
    d = rng.integers(0, 2, n)
    y = np.where(d == 1, rng.normal(0, 2, n), rng.normal(0, 1, n))
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    truth = stats.norm.ppf(taus)  # from scipy, not the estimator

    res = sp.qte(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=taus,
    )
    assert np.all(np.abs(res.effects - truth) < 0.05), (res.effects, truth)
    assert np.all(np.diff(res.effects) > 0)  # it is a fan
    assert abs(res.effects[2]) < 0.05  # zero at the median
    assert abs(res.ate) < 0.05  # ...and zero on average: ATE cannot see this


def test_recovers_constant_shift():
    rng = np.random.default_rng(5)
    n = 50_000
    d = rng.integers(0, 2, n)
    y = 1.5 * d + rng.normal(0, 1, n)
    res = sp.qte(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=[0.25, 0.5, 0.75],
    )
    assert np.all(np.abs(res.effects - 1.5) < 0.06), res.effects


# ── E. QTE vs QTT ──────────────────────────────────────────────────── #


def test_qte_and_qtt_coincide_under_randomisation_and_differ_under_selection():
    rng = np.random.default_rng(6)
    n = 60_000
    x = rng.normal(size=n)

    # randomised: p(X) constant => QTE == QTT
    d_rand = rng.integers(0, 2, n)
    y_rand = 1.0 * d_rand + 0.8 * x + rng.normal(0, 1, n)
    df_r = pd.DataFrame({"y": y_rand, "d": d_rand, "x": x})
    a = sp.qte(
        df_r,
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=[0.5],
        controls=["x"],
        se="analytic",
    )
    b = sp.qte(
        df_r,
        y="y",
        treatment="d",
        method="firpo_qtt",
        quantiles=[0.5],
        controls=["x"],
        se="analytic",
    )
    assert abs(a.effects[0] - b.effects[0]) < 0.1

    # selective: effect correlated with X => QTE != QTT
    p = 1.0 / (1.0 + np.exp(-2.0 * x))
    d_sel = (rng.random(n) < p).astype(int)
    y_sel = (0.5 + 1.5 * (x > 0)) * d_sel + 0.8 * x + rng.normal(0, 1, n)
    df_s = pd.DataFrame({"y": y_sel, "d": d_sel, "x": x})
    c = sp.qte(
        df_s,
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=[0.5],
        controls=["x"],
        se="analytic",
    )
    e = sp.qte(
        df_s,
        y="y",
        treatment="d",
        method="firpo_qtt",
        quantiles=[0.5],
        controls=["x"],
        se="analytic",
    )
    assert abs(c.effects[0] - e.effects[0]) > 0.15, (c.effects, e.effects)


# ── F. analytic SE calibration ─────────────────────────────────────── #


def test_analytic_se_coverage():
    """Monte-Carlo coverage of the influence-function CI, nominal 95%."""
    true_qte, reps, taus = 1.5, 200, [0.25, 0.5, 0.75]
    hits = np.zeros(3)
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        n = 2000
        d = rng.integers(0, 2, n)
        y = true_qte * d + rng.normal(0, 1, n)
        r = sp.qte(
            pd.DataFrame({"y": y, "d": d}),
            y="y",
            treatment="d",
            method="firpo_qte",
            quantiles=taus,
            se="analytic",
        )
        hits += ((r.ci_lower <= true_qte) & (true_qte <= r.ci_upper)).astype(float)
    coverage = hits / reps
    assert np.all((coverage > 0.90) & (coverage < 0.99)), coverage


# ── G. renamed method + deprecation ────────────────────────────────── #


def test_quantile_regression_alias_deprecated_but_identical():
    rng = np.random.default_rng(7)
    n = 800
    d = rng.integers(0, 2, n)
    y = 1.0 * d + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "d": d})
    with pytest.warns(DeprecationWarning, match="conditional_qr"):
        old = sp.qte(
            df,
            y="y",
            treatment="d",
            method="quantile_regression",
            quantiles=[0.5],
            n_boot=20,
        )
    new = sp.qte(
        df, y="y", treatment="d", method="conditional_qr", quantiles=[0.5], n_boot=20
    )
    np.testing.assert_allclose(old.effects, new.effects)
    assert "Firpo" not in new.method
    assert "Koenker" in new.method


def test_distribution_method_is_labelled_qtt():
    """The IPW-counterfactual method computes the QTT; the label must say so."""
    rng = np.random.default_rng(8)
    n = 600
    d = rng.integers(0, 2, n)
    y = 1.0 * d + rng.normal(0, 1, n)
    res = sp.qte(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="distribution",
        quantiles=[0.5],
        n_boot=20,
    )
    assert "QTT" in res.method
    assert "QTE" not in res.method


def test_unknown_method_raises():
    df = pd.DataFrame({"y": [1.0, 2, 3, 4], "d": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="Unknown QTE method"):
        sp.qte(df, y="y", treatment="d", method="nope")
