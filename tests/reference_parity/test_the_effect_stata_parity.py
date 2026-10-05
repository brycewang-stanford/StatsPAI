"""Matching, weighting and sensitivity commands against Stata 18.

The commands are the ones Huntington-Klein's *The Effect* (2nd ed.) uses in
its matching, regression-discontinuity and partial-identification chapters:
``ebalance``, ``cem``, ``sensemakr`` and factor-variable regressions. They
are run on ``sp.datasets.nsw_dw()``, written to a .dta and read by Stata
18.0 MP, so both sides compute on the same numbers. The reference values
are the ``REF`` lines printed by ``_fixtures/_generate_the_effect_Stata.do``
with ``%20.12f``.

Two conventions were pinned down while writing these tests:

* Stata ``ebalance`` stops at ``tolerance(.015)`` by default, before the
  moments are balanced. The references use ``tolerance(1e-10)``, the value
  the default run is converging towards; a default run agrees with it to
  about three digits only. Higher moments are matched in Stata's scaling
  (the sample variance), which is ``dof_adjust=True`` here.
* Stata ``cem`` counts cut points, not bins: ``x(#k)`` is ``k - 1``
  intervals, and Sturges' rule gives one interval fewer than a histogram
  with that many bins.
"""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession

# Deterministic estimators on the same bytes. The loosest agreement is the
# entropy-balancing weights, which Stata iterates to 1e-10.
RTOL = 1e-8


@pytest.fixture(scope="module")
def nsw():
    return sp.datasets.nsw_dw()


def _session(data):
    return StataSession(data.copy())


def _run(session, *lines):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in lines:
            session.run(line)
    return session.output


E = ["age", "education", "black", "married", "nodegree"]

# targets(): (estimate, robust se) of `reg re78 treat [pw = w]`
EBALANCE = {
    "1": (1, -5688.581078092643, 1105.467912312509),
    "2 2 1 1 1": ([2, 2, 1, 1, 1], -5881.264785751378, 1162.899823357941),
    "3 2 1 1 1": ([3, 2, 1, 1, 1], -5732.185521118186, 1189.307286033362),
}


@pytest.mark.parametrize("targets", sorted(EBALANCE))
def test_ebalance_weights_match_stata(nsw, targets):
    moments, b, se = EBALANCE[targets]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.ebalance(
            nsw, y="re78", treat="treat", covariates=E, moments=moments, dof_adjust=True
        )
        data = nsw.assign(w=fit.model_info["weights_full"])
        reg = sp.regress("re78 ~ treat", data=data, weights="w", robust="hc1")
    assert not fit.model_info["weights_fallback"]
    np.testing.assert_allclose(fit.estimate, b, rtol=RTOL)
    np.testing.assert_allclose(reg.params["treat"], b, rtol=RTOL)
    np.testing.assert_allclose(reg.std_errors["treat"], se, rtol=RTOL)


@pytest.mark.parametrize("targets", sorted(EBALANCE))
def test_ebalance_command_in_a_session(nsw, targets):
    _, b, se = EBALANCE[targets]
    out = _run(
        _session(nsw),
        f"ebalance treat {' '.join(E)}, targets({targets}) g(w)",
        "reg re78 treat [pw = w]",
    )
    np.testing.assert_allclose(out.params["treat"], b, rtol=RTOL)
    np.testing.assert_allclose(out.std_errors["treat"], se, rtol=RTOL)


def test_ebalance_drops_a_moment_that_repeats_another(nsw):
    """The square of a 0/1 covariate is the covariate: asking for its
    variance used to make the dual singular and return uniform weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        every = sp.ebalance(nsw, y="re78", treat="treat", covariates=E, moments=2)
        some = sp.ebalance(
            nsw, y="re78", treat="treat", covariates=E, moments=[2, 2, 1, 1, 1]
        )
    assert every.model_info["redundant_moments"] == [
        "black^2",
        "married^2",
        "nodegree^2",
    ]
    assert not every.model_info["weights_fallback"]
    assert every.model_info["max_standardized_moment_gap"] < 1e-8
    np.testing.assert_allclose(every.estimate, some.estimate, rtol=1e-10)


def test_cem_command_matches_stata(nsw):
    session = _session(nsw)
    out = _run(
        session,
        "cem age education black(#2) re74(#6) re75(0 1000 5000 20000), tr(treat)",
        "reg re78 treat [iweight = cem_weights]",
    )
    held = session.data
    matched = held["cem_matched"] == 1
    assert int((matched & (held["treat"] == 1)).sum()) == 60
    assert int((matched & (held["treat"] == 0)).sum()) == 37
    np.testing.assert_allclose(out.params["treat"], -1341.415214452795, rtol=RTOL)
    np.testing.assert_allclose(out.std_errors["treat"], 1117.681230139595, rtol=RTOL)


def test_cem_default_cutpoints_match_stata(nsw):
    session = _session(nsw)
    out = _run(
        session,
        "cem age education, treatment(treat)",
        "reg re78 treat [iweight = cem_weights]",
    )
    assert int((session.data["cem_matched"] == 1).sum()) == 1318
    np.testing.assert_allclose(out.params["treat"], -7499.086654620090, rtol=RTOL)
    np.testing.assert_allclose(out.std_errors["treat"], 546.303940384469, rtol=RTOL)
    # the native call has the same default: Sturges' rule counts cut points
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = sp.match(
            nsw, y="re78", treat="treat", covariates=["age", "education"], method="cem"
        )
    assert native.model_info["n_bins"] == int(np.ceil(np.log2(len(nsw)) + 1)) - 1
    np.testing.assert_allclose(native.estimate, -7499.086654620090, rtol=RTOL)


def test_cem_bins_per_covariate(nsw):
    covariates = ["age", "education", "black"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        as_list = sp.match(
            nsw, y="re78", treat="treat", covariates=covariates,
            method="cem", n_bins=[4, 3, 2],
        )  # fmt: skip
        as_dict = sp.match(
            nsw, y="re78", treat="treat", covariates=covariates,
            method="cem", n_bins={"age": 4, "education": 3, "black": 2},
        )  # fmt: skip
        edges = sp.match(
            nsw, y="re78", treat="treat", covariates=covariates,
            method="cem", n_bins={"age": [25, 35], "education": 3, "black": 2},
        )  # fmt: skip
    assert as_list.estimate == as_dict.estimate
    assert as_dict.model_info["n_bins"] == {"age": 4, "education": 3, "black": 2}
    assert edges.model_info["n_bins"]["age"] == [25.0, 35.0]
    for bad in ([2, 2], {"nope": 2}, {"age": [35, 25]}, 0):
        with pytest.raises(sp.exceptions.MethodIncompatibility):
            sp.match(
                nsw, y="re78", treat="treat", covariates=covariates,
                method="cem", n_bins=bad,
            )  # fmt: skip


X = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]


def test_sensemakr_group_benchmark_matches_stata(nsw):
    out = _run(
        _session(nsw),
        f"sensemakr re78 treat {' '.join(X)}, treat(treat) "
        "gbenchmark(black hispanic) gname(race) kd(1 2)",
    )
    np.testing.assert_allclose(out["rv_q"], 0.07355805613617, rtol=1e-10)
    np.testing.assert_allclose(out["rv_qa"], 0.03770028801805, rtol=1e-10)
    np.testing.assert_allclose(out["partial_r2_yd"], 0.00580648362097, rtol=1e-10)
    table = out["benchmark_table"]
    assert list(table["variable"]) == ["race", "race"]
    reference = np.array(
        [
            [0.06824083182936, 0.00591165905836, 1689.87457421555951, 608.07636255705120],
            [0.13648166365872, 0.01194177432752, 1002.04527126330163, 629.72793019209735],
        ]
    )  # fmt: skip
    ours = table[["r2dz_x", "r2yz_dx", "adjusted_estimate", "adjusted_se"]]
    np.testing.assert_allclose(ours.to_numpy(dtype=float), reference, rtol=1e-9)


def test_sensemakr_refuses_an_unknown_benchmark(nsw):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not in controls"):
        sp.sensemakr(nsw, y="re78", treat="treat", controls=X, benchmark=["re47"])


def test_three_way_factorial_matches_stata(nsw):
    """black##c.age##c.age: every main effect, every two-way product and
    the three-way one, with `black` read as a factor."""
    session = _session(nsw)
    _run(session, "reg re78 black##c.age##c.age")
    np.testing.assert_allclose(
        session.value("_se[1.black#c.age#c.age]"), 2.790750472216, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_b[1.black#c.age]"), 455.319061748920, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_b[c.age#c.age]"), 0.861587440401, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_b[1.black#c.age#c.age]"), -5.378595409567, rtol=RTOL
    )


def test_unprefixed_variable_in_an_interaction_is_a_factor(nsw):
    session = _session(nsw)
    _run(
        session,
        "g byte agegrp = (age > 25) + (age > 35)",
        "reg re78 agegrp##c.education",
    )
    np.testing.assert_allclose(
        session.value("_b[2.agegrp#c.education]"), -292.703787339259, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_se[2.agegrp#c.education]"), 127.098189946805, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_b[1.agegrp]"), 4198.310133590404, rtol=RTOL
    )
    # the base level holds a zero, as in Stata
    assert session.value("_b[0.agegrp#c.education]") == 0.0


def test_reghdfe_factor_interaction_matches_stata(nsw):
    session = _session(nsw)
    _run(
        session,
        "g byte agegrp = (age > 25) + (age > 35)",
        "reghdfe re78 treat##ib1.agegrp, absorb(education) vce(robust)",
    )
    np.testing.assert_allclose(
        session.value("_b[1.treat#2.agegrp]"), -3472.595134911708, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_se[1.treat#2.agegrp]"), 1398.929706043425, rtol=RTOL
    )
    np.testing.assert_allclose(
        session.value("_b[1.treat#0.agegrp]"), 242.761652457853, rtol=RTOL
    )


def test_zero_weights_leave_the_sample(nsw):
    """`[aw = re74]`: the rows with re74 == 0 are not observations."""
    out = _run(_session(nsw), "reg re78 treat age if education > 8 [aw = re74]")
    assert int(out.nobs) == 1972
    np.testing.assert_allclose(out.params["treat"], -10167.760732737504, rtol=RTOL)
    np.testing.assert_allclose(out.std_errors["treat"], 3151.289225939832, rtol=RTOL)


# storage of the integers in Stata: (b[age], se[age]) of the regression on
# the collapsed data
COLLAPSE = {
    "long": ("int64", 84.065993492176, 106.276489070144),
    "byte": ("int8", 84.066047480052, 106.276470960660),
}


@pytest.mark.parametrize("storage", sorted(COLLAPSE))
def test_collapse_stores_means_as_stata_does(nsw, storage):
    """A mean of a byte, an int or a float is stored as a float, of a long
    or a double as a double; the regression afterwards sees the stored
    value, which moves the coefficient in the seventh digit."""
    dtype, b, se = COLLAPSE[storage]
    data = nsw.copy()
    for name in ("treat", "age", "education"):
        data[name] = data[name].astype(dtype)
    out = _run(
        _session(data),
        "collapse (mean) re78 age treat (sd) s = re75, by(education)",
        "reg re78 age treat",
    )
    np.testing.assert_allclose(out.params["age"], b, rtol=RTOL)
    np.testing.assert_allclose(out.std_errors["age"], se, rtol=RTOL)


# ------------------------------------------------------- Rosenbaum bounds
@pytest.fixture(scope="module")
def matched_differences(nsw):
    """re78 of each treated unit minus that of its nearest propensity-score
    match: the `diff` variable the Stata and R references were run on."""
    session = _session(nsw)
    _run(
        session,
        "psmatch2 treat age education black hispanic married nodegree re74 re75, "
        "outcome(re78) logit",
        "g double diff = re78 - _re78 if _treated==1 & _support==1",
    )
    return session


GAMMAS = [1.0, 1.25, 1.5, 1.75, 2.0]
# rbounds diff, gamma(1(.25)2): sig+ with the number of digits Stata
# printed for it, then t-hat+ and t-hat- (six digits)
RBOUNDS = [
    (1.1e-08, 2, 2297.86, 2297.86),
    (8.1e-06, 2, 1833.37, 2859.84),
    (0.000476, 3, 1375.25, 3289.24),
    (0.006701, 4, 957.246, 3623.36),
    (0.03879, 4, 667.566, 3884.08),
]
# DOS2::senWilcox: one-sided p-value, two-sided 95% bounds of the interval
SENWILCOX = [
    (1.0870662925377e-08, 1566.983245, 3148.231088),
    (8.0506561956906e-06, 946.522153, 3635.237924),
    (0.00047631719322172, 483.865041, 4060.503497),
    (0.0067009547344576, 194.468746, 4485.797963),
    (0.038789989090135, -114.326923, 4894.169254),
]


def test_rbounds_command_matches_stata(matched_differences):
    """Significance levels and Hodges-Lehmann bounds of Stata `rbounds`.

    Stata prints six significant digits, hence the tolerances. Where the
    signed-rank statistic sits on its target over a stretch of shifts
    (Gamma 1.5 and 2 here) the middle of the stretch is the estimate.
    """
    out = _run(matched_differences, "rbounds diff, gamma(1(.25)2)")
    table = out.detail
    assert out.n_pairs == 185
    np.testing.assert_allclose(table["Gamma"], GAMMAS)
    for row, (sig, digits, hl_low, hl_high) in zip(table.itertuples(), RBOUNDS):
        # half a unit of the last printed digit
        np.testing.assert_allclose(row.p_upper, sig, rtol=5.0 * 10.0**-digits)
        np.testing.assert_allclose(row.hl_lower, hl_low, rtol=5e-6)
        np.testing.assert_allclose(row.hl_upper, hl_high, rtol=5e-6)
    # alpha() of rbounds is the confidence level; at Gamma = 3 the lower
    # end of the estimate has crossed zero (Stata: -205.524 and 4951.32)
    wide = _run(matched_differences, "rbounds diff, gamma(1 1.5 3) alpha(.90)").detail
    np.testing.assert_allclose(wide["hl_lower"].iloc[2], -205.524, rtol=5e-6)
    np.testing.assert_allclose(wide["hl_upper"].iloc[2], 4951.32, rtol=5e-6)
    np.testing.assert_allclose(wide["p_upper"].iloc[2], 0.636705, rtol=5e-6)
    assert (wide["ci_lower"] <= wide["hl_lower"]).all()
    only = _run(matched_differences, "rbounds diff, gamma(1 2) sigonly").detail
    assert "hl_lower" not in only.columns


def test_rosenbaum_bounds_match_senwilcox(matched_differences):
    """p-values and confidence bounds of Rosenbaum's own `senWilcox`.

    The R run read the differences from a file with seven significant
    digits and finds each end with `uniroot`, so the bounds agree to about
    1e-3 in absolute terms; the p-values agree to eight digits. Stata
    `rbounds` uses the untied-data variance for these bounds and differs
    from both in the third digit (482.046 against 483.865 at Gamma 1.5).
    """
    diff = matched_differences.data["diff"].dropna().to_numpy()
    out = sp.rosenbaum_bounds(
        diff, np.zeros_like(diff), gamma_grid=GAMMAS, estimates=True
    )
    for row, (pval, low, high) in zip(out.detail.itertuples(), SENWILCOX):
        np.testing.assert_allclose(row.p_upper, pval, rtol=1e-6)
        np.testing.assert_allclose(row.ci_lower, low, atol=5e-3)
        np.testing.assert_allclose(row.ci_upper, high, atol=5e-3)


def test_rosenbaum_estimates_need_the_signed_rank_test():
    d = np.arange(1.0, 9.0)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="signed-rank"):
        sp.rosenbaum_bounds(d, np.zeros(8), method="sign", estimates=True)
    # at Gamma = 1 the bound is the Hodges-Lehmann estimate itself: the
    # median of the Walsh averages, 4.5 for 1..8
    out = sp.rosenbaum_bounds(d, np.zeros(8), gamma_grid=[1.0], estimates=True)
    np.testing.assert_allclose(out.detail["hl_lower"], 4.5)
    np.testing.assert_allclose(out.detail["hl_upper"], 4.5)
