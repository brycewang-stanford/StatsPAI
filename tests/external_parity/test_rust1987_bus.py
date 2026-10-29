"""Rust (1987) on Rust's bus data: Tables V, VI, VIII, IX and X reproduced.

``sp.datasets.rust_bus()`` discretises the raw odometer files as Rust's
NFXP pre-processor ``stordat.gpr`` does; ``sp.ddc`` then fits the partial
likelihood with ``beta`` fixed. Rust's published numbers are checked
against both.

Data
----
Fetched on demand (Zenodo record 3385050, pinned SHA-256). The tests run
when the archive is cached (``$STATSPAI_DATA_DIR``) or when
``STATSPAI_FETCH_DATASETS=1`` allows the download, and skip otherwise.

Sources of the reference numbers
--------------------------------
* Rust, J. (1987), Econometrica 55(5), 999-1033, read from the page images
  of the article (the scan on Rust's site,
  editorialexpress.com/jrust/crest_lectures/zurcher.pdf).
* Tables IX and X were checked a second time against the "original"
  columns of Ferrall's replication (ferrall.github.io/OODP/Rust1987.html,
  the online companion of Ferrall 2023, Journal of Applied Econometrics
  38(7), 1093-1100, doi:10.1002/jae.3001). They agree everywhere except
  group 4's ``RC`` at ``beta = 0`` in Table IX, which Ferrall lists as
  7.6538: the article prints 7.6358, and so do ``sp.ddc`` and ``sp.logit``
  (a static logit, so no transition convention is involved).
* Group 4's transition counts, Table V row and Table VIII entries were
  checked a second time against ruspy (OpenSourceEconomics, MIT),
  ``docs/source/tutorials/replication/replication.ipynb`` at commit
  9a336e8 and ``ruspy/test/resources/replication_test``.
* Tables V and VI for the other groups, and Table VIII for groups 1-3 and
  1-4, are in the article only. They are tied to the twice-checked
  numbers by Rust's own identities: each Table IX log-likelihood is the
  Table VI restricted log-likelihood plus the Table VIII partial one, and
  Table IX's heterogeneity statistic is twice the difference of its
  log-likelihoods. Both identities are asserted below.

Tolerances
----------
* Printed numbers: half a unit in the last printed decimal.
* Tables IX and X print Rust's third stage (full likelihood, transitions
  re-estimated jointly). The two-step tests below compare our second
  stage with them: Rust (p. 1019) reports the two "nearly identical";
  here they differ by at most 1.1e-4 in a parameter, so the bound is
  2e-4, and by at most 0.43 per cent in a standard error (group 1-3,
  ``beta = 0``, ``RC``: 1.0462 against 1.0417), so the bound is 5e-3
  relative. Rust's standard errors are BHHH, i.e. ``vce='opg'``. The
  third stage can only raise the full log-likelihood above our
  transition-plus-partial sum; the gap is at most 1.1e-3, so the bound on
  a Table IX / X log-likelihood is 1.5e-3 and on a myopia statistic
  (twice a difference of two of them) 3e-3.
* ``stage='full'`` (Rust's third stage) at ``beta = .9999``: every
  standard error (utility parameters and increment probabilities) and
  every log-likelihood of Tables IX and X lies within half a unit of the
  printed last digit, and so do the parameters in five of the six cells
  and the increment probabilities but one (Table X pooled ``theta_32``:
  .36216 against .3621; the frequency .36219 rounds to .3622 as well).
  The sixth, groups 1-3 in Table IX, has ``theta_11`` 4.82575 against
  4.8259 printed (our two-step value, 4.82587, lies between): the bound
  there is 2e-4, Rust's stopping point. Where the two-step estimates
  missed the printed digit (Table IX group 4 ``theta_11`` SE .6383 for
  .639, pooled ``RC`` 9.7557 for 9.7558), the third stage hits it. At
  ``beta = 0`` the choice probabilities do not depend on the transitions,
  so the third stage equals the second exactly; the 0.43 per cent ``RC``
  standard error above is a ``beta = 0`` cell and stays.
* The pooled 90-bin ``theta_30`` standard error prints as .0052 at both
  discount factors; the data give .00528, i.e. the multinomial
  sqrt(p (1 - p) / n) with p = .3489 and n = 8156. Truncated, like the
  pooled .348 of Table VI. Table X prints the pooled ``beta = 0``
  ``theta_33`` standard error as .0143 (the estimate itself, repeated);
  the ``beta = .9999`` column next to it prints .0013, as do the data.
* ``beta``: Rust (p. 1023) found that changing ``beta`` to .98 or .999999
  "produced negligible changes in the likelihood function and parameter
  estimates" and that a free ``beta`` was "driven to 1". The profile on
  the data rises monotonically to the upper end of the search range, the
  likelihood-ratio statistic between .98 and .9999 is below 3.84, and
  .999999 moves the log-likelihood by less than 0.01 and the parameters by
  less than 0.05 standard errors. The parameters at .98 are a different
  matter: ``RC`` stays within one standard error, but ``theta_11`` moves
  by one to two (group 4: 3.48 against 2.29, SE .639), so "negligible"
  holds for the likelihood, not for ``theta_11``. The profile
  likelihood-ratio interval for group 4 includes 0, matching Table IX's
  myopia test (marginal significance .0529), and excludes 0 for the
  pooled sample (.0035).
* Table VIII prints the second-stage log-likelihood itself; the bound is
  1e-3 (one value, -165.4585, prints as -165.458).

Three printed values do not match the data within those bounds, and the
tests record them:

* Table X gives group 4's ``RC`` as 10.896. The data give 10.0896 together
  with the printed log-likelihood -4495.135, and the text says ``RC`` is
  "nearly identical" across grid sizes (10.075 at 90 bins): a lost digit.
* Table IX gives groups 1-3 at ``beta = 0`` a log-likelihood of -2710.746,
  0.02 below the sum of the paper's own Table VI (-2575.98) and Table
  VIII (-134.747) entries for the same sample; a maximised full likelihood
  cannot lie below that sum. The data give -2710.7245, so the myopia
  statistic for groups 1-3 is 4.716 rather than the printed 4.760 (same
  conclusion at the 5 per cent level). The parameters in that cell match
  to 1e-4; its ``RC`` standard error is the 0.43 per cent outlier above.
* Table VI gives the pooled 1-4 ``theta_31`` as .348, where the data give
  .34882 (Table IX prints .3488 and .3489 for the same quantity): the
  printed digit is truncated rather than rounded.

The transition convention matters. ``stordat.gpr`` measures the step
after a replacement as the next reading's 1-based bin, one more than the
step from state 0; the ``increment`` column keeps that convention, and
every published transition estimate needs it. ``sp.ddc``'s own
``transition='increments'`` measures from state 0. On that convention the
estimates match Ferrall's corrected replication instead (last test).
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.datasets import _fetch, _rust_bus

pytestmark = pytest.mark.filterwarnings("ignore")

BETA = 0.9999


def _half(decimals):
    return 0.5 * 10.0 ** (-decimals) + 1e-12


@pytest.fixture(scope="module")
def bus():
    """All eight groups on Rust's two grids (90 and 175 bins)."""
    spec = _rust_bus.RUST_BUS
    if not _fetch.is_cached(spec) and not _fetch.fetch_allowed_in_tests():
        pytest.skip(
            f"{spec.filename} is not cached in {_fetch.cache_dir()} and "
            f"{_fetch.ENV_FETCH}=1 is not set (network fetch disabled)"
        )
    return {
        n: sp.datasets.rust_bus(groups="all", n_states=n) for n in (90, 175)
    }


def _sample(bus, groups, n=90):
    df = bus[n]
    return df[df["rust_sample"] & df["group"].isin(groups)]


def _increments(s):
    counts = np.bincount(s["increment"].astype(int))
    return counts, counts / counts.sum()


def _trans_ll(counts):
    c = counts[counts > 0]
    return float(np.sum(c * np.log(c / counts.sum())))


def _fit(s, beta, n=90, **kw):
    _, p = _increments(s)
    return sp.ddc(
        s, "replace", "state", beta=beta, n_states=n, transition=p, vce="opg", **kw
    )


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

#: Table I bus counts; Table IX / X observation counts.
BUSES = {1: 15, 2: 4, 3: 48, 4: 37, 5: 12, 6: 10, 7: 18, 8: 18}
OBS = {(1, 2, 3): 3864, (4,): 4292, (1, 2, 3, 4): 8156}


def test_sample_sizes(bus):
    df = bus[90]
    assert df.groupby("group")["bus"].nunique().to_dict() == BUSES
    # bus numbers identify buses across groups
    assert df.groupby("bus")["group"].nunique().max() == 1
    for groups, n in OBS.items():
        assert len(_sample(bus, groups)) == n
    # 60 engine replacements in groups 1-4 (groups 1 and 2 have none)
    s = _sample(bus, (1, 2, 3, 4))
    assert s.groupby("group")["replace"].sum().to_dict() == {1: 0, 2: 0, 3: 27, 4: 33}
    # mileage restarts at each replacement and never runs backwards
    assert (df["mileage"] >= 0).all()
    assert set(s["increment"].unique()) <= {0.0, 1.0, 2.0}


# Table V: within-group theta_31, theta_32, theta_33 (Pr of a 0, 1, 2 bin
# step) and the restricted log-likelihood.
TABLE_V = {
    1: ((0.197, 0.789, 0.014), -203.99),
    2: ((0.391, 0.599, 0.010), -138.57),
    3: ((0.307, 0.683, 0.010), -2219.58),
    4: ((0.392, 0.595, 0.013), -3140.57),
    5: ((0.489, 0.507, 0.005), -1079.18),
    6: ((0.618, 0.382, 0.000), -831.05),
    7: ((0.600, 0.397, 0.003), -1550.32),
    8: ((0.722, 0.278, 0.000), -1330.35),
}

# Table VI: pooled groups.
TABLE_VI = {
    (1, 2, 3): ((0.301, 0.688, 0.011), -2575.98),
    (1, 2, 3, 4): ((0.348, 0.639, 0.012), -5755.00),
    (4, 5): ((0.417, 0.572, 0.011), -4243.73),
    (6, 7): ((0.607, 0.392, 0.002), -2384.50),
    (6, 7, 8): ((0.652, 0.347, 0.001), -3757.76),
    (5, 6, 7, 8): ((0.618, 0.380, 0.002), -4904.41),
    (1, 2, 3, 4, 5, 6, 7, 8): ((0.475, 0.517, 0.007), -11237.68),
}


@pytest.mark.parametrize("group", sorted(TABLE_V))
def test_table_v_within_group_mileage(bus, group):
    probs, ll = TABLE_V[group]
    counts, p = _increments(_sample(bus, (group,)))
    p = np.pad(p, (0, 3 - p.size))
    np.testing.assert_allclose(p, probs, rtol=0, atol=_half(3))
    assert abs(_trans_ll(counts) - ll) <= _half(2)


@pytest.mark.parametrize("groups", sorted(TABLE_VI))
def test_table_vi_pooled_mileage(bus, groups):
    probs, ll = TABLE_VI[groups]
    counts, p = _increments(_sample(bus, groups))
    atol = np.full(3, _half(3))
    if groups == (1, 2, 3, 4):
        # printed .348 for 2845 / 8156 = .34882 (truncated, not rounded;
        # Table IX prints .3488 / .3489 for the same number)
        assert counts[0] == 2845
        atol[0] = 1e-3
    assert np.all(np.abs(p - probs) <= atol)
    assert abs(_trans_ll(counts) - ll) <= _half(2)


def test_group4_transitions_equal_ruspy(bus):
    """ruspy's replication: counts, frequencies and -log-likelihood."""
    counts, p = _increments(_sample(bus, (4,)))
    assert counts.tolist() == [1682, 2555, 55]
    np.testing.assert_allclose(
        p, [0.39189189, 0.59529357, 0.01281454], rtol=0, atol=5e-9
    )
    assert abs(-_trans_ll(counts) - 3140.5705570938244) <= 1e-9


# ---------------------------------------------------------------------------
# Table VIII: second-stage partial log-likelihoods
# ---------------------------------------------------------------------------

# (groups, cost) -> (beta = .9999, beta = 0); models 2, 3, 10, 11, 18, 19
TABLE_VIII = {
    ((1, 2, 3), "linear"): (-132.389, -134.747),
    ((4,), "linear"): (-163.584, -165.458),
    ((1, 2, 3, 4), "linear"): (-300.250, -306.641),
    ((1, 2, 3), "quadratic"): (-131.326, -131.534),
    ((4,), "quadratic"): (-163.402, -163.771),
    ((1, 2, 3, 4), "quadratic"): (-297.939, -299.328),
}


@pytest.mark.parametrize("key", list(TABLE_VIII), ids=lambda k: f"{k[0]}-{k[1]}")
def test_table_viii_partial_loglik(bus, key):
    groups, cost = key
    s = _sample(bus, groups)
    for beta, printed in zip((BETA, 0.0), TABLE_VIII[key]):
        fit = _fit(s, beta, cost=cost)
        assert fit.converged
        assert abs(fit.loglik - printed) <= 1e-3, (key, beta, fit.loglik)


def test_group4_equals_ruspy_notebook(bus):
    """ruspy's NFXP on the same data (estimagic BFGS, printed to 8-9 digits)."""
    fit = _fit(_sample(bus, (4,)), BETA)
    np.testing.assert_allclose(
        fit.params.to_numpy(), [10.0749422, 2.29309298], rtol=0, atol=1e-6
    )
    assert abs(-fit.loglik - 163.58428365680294) <= 1e-9


# ---------------------------------------------------------------------------
# Tables IX (90 bins) and X (175 bins): structural estimates
# ---------------------------------------------------------------------------

# groups -> beta -> (RC, se), (theta_11, se), theta_3 (first entries), LL
TABLE_IX = {
    (1, 2, 3): {
        BETA: ((11.7270, 2.602), (4.8259, 1.792), (0.3010, 0.6884), -2708.366),
        0.0: ((8.2985, 1.0417), (109.9031, 26.163), (0.3010, 0.6884), -2710.746),
    },
    (4,): {
        BETA: ((10.0750, 1.582), (2.2930, 0.639), (0.3919, 0.5953), -3304.155),
        0.0: ((7.6358, 0.7197), (71.5133, 13.778), (0.3919, 0.5953), -3306.028),
    },
    (1, 2, 3, 4): {
        BETA: ((9.7558, 1.227), (2.6275, 0.618), (0.3489, 0.6394), -6055.250),
        0.0: ((7.3055, 0.5067), (70.2769, 10.750), (0.3488, 0.6394), -6061.641),
    },
}

TABLE_X = {
    (1, 2, 3): {
        BETA: (
            (11.7257, 2.597),
            (2.4569, 0.9122),
            (0.0937, 0.4475, 0.4459, 0.0127),
            -3993.991,
        ),
        0.0: (
            (8.2969, 1.0477),
            (56.1656, 13.4205),
            (0.0937, 0.4475, 0.4459, 0.0127),
            -3996.353,
        ),
    },
    (4,): {
        # printed RC 10.896: a lost digit, see the module docstring
        BETA: (
            (10.0896, 1.581),
            (1.1732, 0.327),
            (0.1191, 0.5762, 0.2868, 0.0158),
            -4495.135,
        ),
        0.0: (
            (7.6423, 0.7204),
            (36.6692, 7.0675),
            (0.1191, 0.5762, 0.2868, 0.0158),
            -4496.997,
        ),
    },
    (1, 2, 3, 4): {
        BETA: (
            (9.7687, 1.226),
            (1.3428, 0.315),
            (0.1071, 0.5152, 0.3621, 0.0143),
            -8607.889,
        ),
        0.0: (
            (7.3113, 0.5073),
            (36.0175, 5.5145),
            (0.1070, 0.5152, 0.3622, 0.0143),
            -8614.238,
        ),
    },
}


def _check_structural(bus, groups, beta, row, n):
    (rc, rc_se), (th, th_se), theta3, ll = row
    s = _sample(bus, groups, n)
    counts, p = _increments(s)
    fit = _fit(s, beta, n=n)
    assert fit.converged and fit.n_obs == OBS[groups]
    np.testing.assert_allclose(fit.params.to_numpy(), [rc, th], rtol=0, atol=2e-4)
    np.testing.assert_allclose(fit.se.to_numpy(), [rc_se, th_se], rtol=5e-3)
    np.testing.assert_allclose(p[: len(theta3)], theta3, rtol=0, atol=2e-4)
    # Rust's LL is the transition part plus the partial likelihood
    full = _trans_ll(counts) + fit.loglik
    if (n, groups, beta) == (90, (1, 2, 3), 0.0):
        # printed -2710.746 is 0.02 below Table VI + Table VIII (docstring)
        assert abs(full - (-2575.98 - 134.747)) <= _half(2) + _half(3)
        assert abs(full - ll) > 0.02
    else:
        assert abs(full - ll) <= 1.5e-3
    return fit


@pytest.mark.parametrize("groups", list(TABLE_IX), ids=str)
def test_table_ix(bus, groups):
    rows = TABLE_IX[groups]
    fits = {b: _check_structural(bus, groups, b, rows[b], 90) for b in rows}
    # myopia test: LR = 2 (LL_.9999 - LL_0), printed 4.760 / 3.746 / 12.782;
    # for groups 1-3 the data give 4.716 (see the docstring)
    lr = 2 * (fits[BETA].loglik - fits[0.0].loglik)
    assert abs(lr - {3: 4.716, 1: 3.746, 4: 12.782}[len(groups)]) <= 3e-3


@pytest.mark.parametrize("groups", list(TABLE_X), ids=str)
def test_table_x(bus, groups):
    rows = TABLE_X[groups]
    fits = {b: _check_structural(bus, groups, b, rows[b], 175) for b in rows}
    lr = 2 * (fits[BETA].loglik - fits[0.0].loglik)
    assert abs(lr - {3: 4.724, 1: 3.724, 4: 12.698}[len(groups)]) <= 3e-3


def test_table_x_group4_rc_misprint(bus):
    """Table X prints 10.896; the maximiser of the printed LL is 10.0896."""
    fit = _fit(_sample(bus, (4,), 175), BETA, n=175)
    assert abs(fit.params["RC"] - 10.0896) <= 2e-4
    assert abs(fit.params["RC"] - 10.896) > 0.8
    # started at the printed value, NFXP climbs back to the same maximum
    again = _fit(_sample(bus, (4,), 175), BETA, n=175, theta0=[10.896, 1.1732])
    assert abs(again.params["RC"] - fit.params["RC"]) <= 1e-6


def test_heterogeneity_statistic(bus):
    """Table IX: LR = 2 (LL_123 + LL_4 - LL_1234) = 85.46."""
    ll = {}
    for groups in OBS:
        s = _sample(bus, groups)
        counts, _ = _increments(s)
        ll[groups] = _trans_ll(counts) + _fit(s, BETA).loglik
    lr = 2 * (ll[(1, 2, 3)] + ll[(4,)] - ll[(1, 2, 3, 4)])
    assert abs(lr - 85.46) <= _half(2)


def test_static_model_is_a_logit(bus):
    """beta = 0: RC and theta_11 are a logit of replace on 0.001 * state."""
    s = _sample(bus, (4,)).assign(x=lambda d: 0.001 * d["state"])
    params = sp.logit("replace ~ x", data=s).params
    assert abs(-params["Intercept"] - 7.6358) <= _half(4)
    assert abs(params["x"] - 71.5133) <= _half(4)


# ---------------------------------------------------------------------------
# Other estimators and conventions on the same data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", [(4,), (1, 2, 3, 4)], ids=str)
def test_npl_reaches_the_nfxp_estimate(bus, groups):
    s = _sample(bus, groups)
    nfxp = _fit(s, BETA)
    npl = _fit(s, BETA, method="npl")
    assert npl.converged and npl.diagnostics["npl_iterations"] <= 20
    np.testing.assert_allclose(npl.params, nfxp.params, rtol=1e-6)
    assert abs(npl.loglik - nfxp.loglik) <= 1e-8
    # The two-step CCP estimator uses frequency CCPs from sparse high-mileage
    # cells (few buses reach them); it lands within two standard errors
    # here (RC 9.13 against 10.07 for group 4), not on the MLE.
    ccp = _fit(s, BETA, method="ccp")
    z = np.abs((ccp.params - nfxp.params) / nfxp.se)
    assert np.all(z < 2) and np.any(z > 0.1)


# Ferrall's replication with the step after a replacement measured from 0
# (his Table 3 "Repl" columns, 175 bins, beta = .9999, all months).
FERRALL_175 = {
    (1, 2, 3): (11.7361, 2.4520),
    (4,): (10.0982, 1.1702),
    (1, 2, 3, 4): (9.7802, 1.3410),
}


@pytest.mark.parametrize("groups", list(FERRALL_175), ids=str)
def test_default_increments_match_ferrall_correction(bus, groups):
    df = bus[175]
    d = df[df["group"].isin(groups)]
    fit = sp.ddc(
        d, "replace", "state", id="bus", time="period", beta=BETA, n_states=175
    )
    assert fit.transition_params["mode"] == "increments"
    np.testing.assert_allclose(
        fit.params.to_numpy(), FERRALL_175[groups], rtol=0, atol=2e-4
    )


# ---------------------------------------------------------------------------
# Rust's third stage: stage='full' (Tables IX and X as printed)
# ---------------------------------------------------------------------------

# (bins, groups) -> printed beta = .9999 column as strings, so the number of
# printed decimals sets the tolerance: (RC, se), (theta_11, se), the free
# increment probabilities with their standard errors, LL.
STAGE3 = {
    (90, (1, 2, 3)): (
        ("11.7270", "2.602"),
        ("4.8259", "1.792"),
        (("0.3010", "0.0074"), ("0.6884", "0.0075")),
        "-2708.366",
    ),
    (90, (4,)): (
        ("10.0750", "1.582"),
        ("2.2930", "0.639"),
        (("0.3919", "0.0075"), ("0.5953", "0.0075")),
        "-3304.155",
    ),
    (90, (1, 2, 3, 4)): (
        ("9.7558", "1.227"),
        ("2.6275", "0.618"),
        (("0.3489", "0.0052"), ("0.6394", "0.0053")),
        "-6055.250",
    ),
    (175, (1, 2, 3)): (
        ("11.7257", "2.597"),
        ("2.4569", "0.9122"),
        (
            ("0.0937", "0.0047"),
            ("0.4475", "0.0080"),
            ("0.4459", "0.0080"),
            ("0.0127", "0.0018"),
        ),
        "-3993.991",
    ),
    (175, (4,)): (
        ("10.0896", "1.581"),  # printed 10.896 (lost digit, see above)
        ("1.1732", "0.327"),
        (
            ("0.1191", "0.0050"),
            ("0.5762", "0.0075"),
            ("0.2868", "0.0069"),
            ("0.0158", "0.0019"),
        ),
        "-4495.135",
    ),
    (175, (1, 2, 3, 4)): (
        ("9.7687", "1.226"),
        ("1.3428", "0.315"),
        (
            ("0.1071", "0.0034"),
            ("0.5152", "0.0055"),
            ("0.3621", "0.0053"),
            ("0.0143", "0.0013"),
        ),
        "-8607.889",
    ),
}


def _printed(value, text, slack=1e-9):
    """``value`` is within half a unit of the last printed decimal."""
    decimals = len(text.split(".")[1])
    return abs(value - float(text)) <= 0.5 * 10.0 ** (-decimals) + slack


def _full(bus, groups, beta, n):
    s = _sample(bus, groups, n)
    return sp.ddc(
        s,
        "replace",
        "state",
        beta=beta,
        n_states=n,
        increment="increment",
        stage="full",
        vce="opg",
    )


@pytest.mark.parametrize("key", list(STAGE3), ids=lambda k: f"{k[0]}-{k[1]}")
def test_stage3_matches_printed_digits(bus, key):
    n, groups = key
    (rc, rc_se), (th, th_se), incs, ll = STAGE3[key]
    fit = _full(bus, groups, BETA, n)
    assert fit.converged and fit.diagnostics["stage"] == "full"
    est, se = fit.params.to_numpy(), fit.se.to_numpy()
    assert _printed(est[0], rc)
    if key == (90, (1, 2, 3)):
        # 4.82575 against 4.8259; Rust's stopping point (module docstring)
        assert abs(est[1] - float(th)) <= 2e-4 and not _printed(est[1], th)
    else:
        assert _printed(est[1], th)
    assert _printed(se[0], rc_se) and _printed(se[1], th_se)
    p = fit.transition_params["increment_probs"]
    p_se = fit.transition_params["increment_se"]
    for d, (pt, st) in enumerate(incs):
        if key == (175, (1, 2, 3, 4)) and d == 2:
            # .36216 against .3621; the frequency .36219 rounds to .3622 too
            assert abs(p[d] - float(pt)) <= 1e-4 and not _printed(p[d], pt)
        else:
            assert _printed(p[d], pt), (d, p[d], pt)
        if key == (90, (1, 2, 3, 4)) and d == 0:
            # printed .0052 truncates .00528 (module docstring)
            assert np.floor(p_se[d] * 1e4) / 1e4 == float(st)
            assert not _printed(p_se[d], st)
        else:
            assert _printed(p_se[d], st), (d, p_se[d], st)
    assert _printed(fit.diagnostics["loglik_full"], ll)


@pytest.mark.parametrize("groups", list(OBS), ids=str)
def test_stage3_at_beta_zero_is_the_second_stage(bus, groups):
    full = _full(bus, groups, 0.0, 90)
    two = _fit(_sample(bus, groups), 0.0)
    np.testing.assert_allclose(full.params, two.params, rtol=1e-8)
    # BHHH: the sample cross-products of choice and increment scores have
    # mean zero but are not exactly zero, so the OPG blocks differ slightly
    # (observed 7e-5 relative; the Hessian version is exactly block diagonal)
    np.testing.assert_allclose(full.se, two.se, rtol=2e-4)


def test_stage3_moves_little_from_the_second_stage(bus):
    """Rust (p. 1019): the two stages are "nearly identical"."""
    full = _full(bus, (1, 2, 3, 4), BETA, 90)
    two = full.diagnostics["two_step_params"]
    for name in ("RC", "theta_1"):
        assert abs(full.params[name] - two[name]) < 0.01 * full.se[name]
    V = np.asarray(full.diagnostics["vcov_full"]["matrix"])
    corr = V / np.sqrt(np.outer(np.diag(V), np.diag(V)))
    # "almost perfectly block diagonal"
    assert np.max(np.abs(corr[:2, 2:])) < 0.05


# ---------------------------------------------------------------------------
# The discount factor (Rust 1987, p. 1023)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", list(OBS), ids=str)
def test_beta_profile_is_flat_and_rises_to_one(bus, groups):
    grid = [0.98, 0.999, 0.9999, 0.999999]
    fit = _full(bus, groups, grid, 90)
    prof = fit.beta_profile
    assert prof["converged"].all()
    ll = prof["loglik"].to_numpy()
    assert np.all(np.diff(ll) > 0) and fit.beta == grid[-1]
    # "negligible changes in the likelihood function" at .98 and .999999
    assert 2 * (prof.loc[0.9999, "loglik"] - prof.loc[0.98, "loglik"]) < 3.84
    assert abs(prof.loc[0.999999, "loglik"] - prof.loc[0.9999, "loglik"]) < 0.01
    # ... and in the parameters: true at .999999; at .98 RC stays within one
    # standard error but theta_11 moves one to two (module docstring)
    ref = _full(bus, groups, BETA, 90)
    se = ref.se.to_numpy()

    def z(b):
        # positional result: ndarray - Series is a labelled Series, and
        # pandas 3 no longer reads [0] as a position on it
        est = prof.loc[b, ["RC", "theta_1"]].to_numpy(float)
        return np.abs(est - ref.params.to_numpy(float)) / se

    assert np.all(z(0.999999) < 0.05)
    assert z(0.98)[0] < 1 and 0.9 < z(0.98)[1] < 2


@pytest.mark.parametrize(
    "groups, zero_inside", [((4,), True), ((1, 2, 3, 4), False)], ids=str
)
def test_beta_estimate_is_driven_to_one(bus, groups, zero_inside):
    with pytest.warns(sp.exceptions.AssumptionWarning, match="boundary"):
        fit = _full(bus, groups, "estimate", 90)
    d = fit.diagnostics
    assert d["beta_at_bound"] and fit.beta > 0.99999
    # the profile LR interval agrees with Table IX's myopia test
    assert (d["beta_ci"][0] == 0.0) is zero_inside
