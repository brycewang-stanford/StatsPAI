"""Pre-trend power and honest DiD from coefficients, against Stata.

Stata's ``pretrends`` and ``honestdid`` read ``e(b)`` and ``e(V)`` of
whatever event-study regression was run last::

    xtreg y lead4 lead3 lead2 lag0 lag1 lag2 lag3 i.year, fe vce(cluster id)
    matrix beta = e(b)[1, 1..7]
    matrix sigma = e(V)[1..7, 1..7]
    pretrends, numpre(3) b(beta) v(sigma) slope(.03)
    pretrends power 0.5, numpre(3) b(beta) v(sigma)
    honestdid, l_vec(l_vec) pre(1/3) post(4/7) mvec(...) delta(sd)

``sp.honest_did_from_moments`` took such input; ``sp.pretrends_power`` and
``sp.pretrends_slope_for_power`` only took a StatsPAI event-study result, so
a hand-built lead/lag regression had no way in.

The seven coefficients and their covariance below are the output of such a
regression (three leads, the period before treatment omitted, four lags;
108 clusters), printed by Stata 18 with 12 and 14 significant digits.

Evidence tier. The likelihood ratio is closed form and agrees to 1e-13. The
power is one minus a multivariate-normal rectangle probability, which both
sides integrate numerically, and the Bayes factor and the 50%-power slope
are functions of it: they agree to about 2e-4 (S: numerical integration,
not parity). The honest-DiD intervals are compared with the three decimals
Stata prints.
"""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

BETA = np.array(
    [
        0.060587209253,
        -0.073254011333,
        -0.108236333496,
        0.046864393581,
        0.250727397651,
        0.238045293620,
        0.218007526942,
    ]
)
_LOWER = [
    [1.9933590202550e-02],
    [1.0495878463412e-02, 1.0209262377861e-02],
    [3.1255680766513e-03, 3.5022239899622e-03, 4.8885623255168e-03],
    [
        1.0608672580379e-03,
        -9.3395958975401e-04,
        5.1108813376314e-04,
        6.4081009486629e-03,
    ],
    [
        -2.9938585123809e-04,
        9.0058950389958e-06,
        -7.4513938478459e-04,
        1.4924462682827e-03,
        1.6841623261848e-02,
    ],
    [
        2.2784075512751e-03,
        4.8846237270576e-04,
        1.9043951839426e-03,
        1.4056540338318e-03,
        9.0479070236985e-03,
        1.2230736393062e-02,
    ],
    [
        4.5609444988658e-03,
        1.7250692632890e-03,
        -2.2862041269388e-04,
        1.5937146540343e-03,
        4.5574182056991e-03,
        7.3174589161974e-03,
        1.4026350402586e-02,
    ],
]
SIGMA = np.zeros((7, 7))
for _i, _row in enumerate(_LOWER):
    for _j, _value in enumerate(_row):
        SIGMA[_i, _j] = SIGMA[_j, _i] = _value

# pretrends, numpre(3) b(beta) v(sigma) slope(.03)
STATA_POWER = 0.1900293668817012
STATA_BAYES = 0.9212164681285357
STATA_LR = 0.9993897914703789
# pretrends power 0.5, numpre(3) b(beta) v(sigma)
STATA_SLOPE_50 = 0.0744020441858424

# The power is a numerically integrated probability on both sides.
RTOL_INTEGRATED = 1e-3


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_power_bayes_factor_and_likelihood_ratio_match_stata():
    out = _quiet(sp.pretrends_power, BETA, slope=0.03, sigma=SIGMA, num_pre_periods=3)
    np.testing.assert_allclose(out["delta"], [-0.09, -0.06, -0.03], rtol=1e-12)
    assert out["likelihood_ratio"] == pytest.approx(STATA_LR, rel=1e-12)
    assert out["power"] == pytest.approx(STATA_POWER, rel=RTOL_INTEGRATED)
    assert out["bayes_factor"] == pytest.approx(STATA_BAYES, rel=RTOL_INTEGRATED)


def test_slope_for_half_power_matches_stata():
    out = _quiet(sp.pretrends_slope_for_power, BETA, sigma=SIGMA, num_pre_periods=3)
    assert out["slope"] == pytest.approx(STATA_SLOPE_50, rel=RTOL_INTEGRATED)
    np.testing.assert_allclose(out["times"], [-4.0, -3.0, -2.0])
    assert out["achieved_power"] == pytest.approx(0.5, abs=1e-4)


def test_slope_is_shorthand_for_the_linear_delta():
    by_slope = _quiet(
        sp.pretrends_power, BETA, slope=0.05, sigma=SIGMA, num_pre_periods=3
    )
    by_delta = _quiet(
        sp.pretrends_power,
        BETA,
        delta=0.05 * np.array([-3.0, -2.0, -1.0]),
        sigma=SIGMA,
        num_pre_periods=3,
    )
    # the rectangle probability is integrated by randomised quasi-Monte
    # Carlo, so two calls agree to its tolerance, not to the bit
    assert by_slope["power"] == pytest.approx(by_delta["power"], rel=1e-4)
    assert by_slope["noncentrality"] == by_delta["noncentrality"]
    np.testing.assert_array_equal(by_slope["delta"], by_delta["delta"])
    with pytest.raises(MethodIncompatibility, match="slope"):
        sp.pretrends_power(
            BETA, delta=[0.1, 0.1, 0.1], slope=0.05, sigma=SIGMA, num_pre_periods=3
        )


def test_explicit_event_times_equal_the_num_pre_layout():
    times = [-4, -3, -2, 0, 1, 2, 3]
    a = _quiet(sp.pretrends_power, BETA, slope=0.03, sigma=SIGMA, num_pre_periods=3)
    b = _quiet(sp.pretrends_power, BETA, slope=0.03, sigma=SIGMA, event_times=times)
    assert a["power"] == pytest.approx(b["power"], rel=1e-4)
    assert a["likelihood_ratio"] == b["likelihood_ratio"]
    assert a["noncentrality"] == b["noncentrality"]
    # the order the coefficients are given in does not matter
    perm = [3, 0, 6, 1, 5, 2, 4]
    c = _quiet(
        sp.pretrends_power,
        BETA[perm],
        slope=0.03,
        sigma=SIGMA[np.ix_(perm, perm)],
        event_times=[times[i] for i in perm],
    )
    assert c["likelihood_ratio"] == pytest.approx(a["likelihood_ratio"], rel=1e-12)
    assert c["noncentrality"] == pytest.approx(a["noncentrality"], rel=1e-12)


def test_the_full_covariance_is_used_not_its_diagonal():
    full = _quiet(sp.pretrends_power, BETA, slope=0.03, sigma=SIGMA, num_pre_periods=3)
    diag = _quiet(
        sp.pretrends_power,
        BETA,
        slope=0.03,
        sigma=np.diag(np.diag(SIGMA)),
        num_pre_periods=3,
    )
    assert abs(full["noncentrality"] - diag["noncentrality"]) > 0.1


def test_a_fitted_regression_goes_straight_in():
    import pandas as pd

    rng = np.random.default_rng(11)
    rows = []
    for unit in range(120):
        treated = unit < 60
        effect_time = 5
        for t in range(9):
            rel = t - effect_time
            y = 0.2 * t + (unit % 7) * 0.1 + rng.normal()
            if treated and rel >= 0:
                y += 1.0
            row = {"id": unit, "t": t, "y": y}
            for k in (-4, -3, -2, 0, 1, 2, 3):
                name = f"lead{-k}" if k < 0 else f"lag{k}"
                row[name] = float(treated and rel == k)
            rows.append(row)
    df = pd.DataFrame(rows)
    names = ["lead4", "lead3", "lead2", "lag0", "lag1", "lag2", "lag3"]
    fit = _quiet(
        sp.feols, "y ~ " + " + ".join(names) + " | id + t", df, vcov={"CRV1": "id"}
    )
    times = dict(zip(names, [-4, -3, -2, 0, 1, 2, 3]))
    from_fit = _quiet(sp.pretrends_power, fit, slope=0.1, event_times=times)
    beta = np.array([float(fit.params[nm]) for nm in names])
    cov = np.asarray(fit.vcov().loc[names, names])
    from_arrays = _quiet(
        sp.pretrends_power, beta, slope=0.1, sigma=cov, event_times=list(times.values())
    )
    assert from_fit["likelihood_ratio"] == pytest.approx(
        from_arrays["likelihood_ratio"], rel=1e-10
    )
    assert from_fit["noncentrality"] == pytest.approx(
        from_arrays["noncentrality"], rel=1e-10
    )


def test_moments_need_a_covariance():
    with pytest.raises(MethodIncompatibility, match="sigma"):
        sp.pretrends_power(BETA, slope=0.03, num_pre_periods=3)


# honestdid, l_vec(0 \ 1 \ 0 \ 0) pre(1/3) post(4/7) mvec(0(.1).5) alpha(.1) delta(sd)
STATA_FLCI = {
    0.0: (-0.041, 0.463),
    0.1: (-0.500, 0.569),
    0.2: (-0.800, 0.869),
    0.3: (-1.100, 1.169),
    0.4: (-1.400, 1.469),
    0.5: (-1.700, 1.769),
}


def test_smoothness_intervals_match_stata_honestdid():
    out = _quiet(
        sp.honest_did_from_moments,
        BETA,
        SIGMA,
        num_pre_periods=3,
        l_vec=[0, 1, 0, 0],
        m_grid=list(STATA_FLCI),
        method="smoothness",
        alpha=0.1,
    ).set_index("M")
    for m, (lo, hi) in STATA_FLCI.items():
        # Stata prints three decimals.
        assert out.loc[m, "ci_lower"] == pytest.approx(lo, abs=6e-4)
        assert out.loc[m, "ci_upper"] == pytest.approx(hi, abs=6e-4)


def test_event_study_from_moments_feeds_both_analyses():
    from statspai.did.honest_did import event_study_from_moments

    es = event_study_from_moments(BETA, SIGMA, num_pre_periods=3)
    assert list(es.detail["relative_time"]) == [-4, -3, -2, 0, 1, 2, 3]
    np.testing.assert_allclose(es.model_info["vcv_pre"], SIGMA[:3, :3])
    power = _quiet(sp.pretrends_power, es, slope=0.03)
    assert power["likelihood_ratio"] == pytest.approx(STATA_LR, rel=1e-12)
