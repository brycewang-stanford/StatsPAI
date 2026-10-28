"""``sp.recentered_iv`` against Stata ``ivreghdfe`` and ``ritest``.

Reference: Stata 18 MP, ivreghdfe 1.1.4 / ivreg2 4.1.12 / reghdfe 6.13.1,
ritest 1.1.7, on the fixtures written by
``_fixtures/_generate_recentered_iv_data.py`` and run through
``_fixtures/_generate_recentered_iv_Stata.do`` (values printed at 17
significant digits). All CSV values have 6 decimals, read with
``float_precision="round_trip"`` so both sides hold the same doubles. The
200 permutations are stored as ``recentered_iv_ri_perms.csv.gz`` (the plain
CSV is above the repository's 500 KB file limit); the do-file decompresses
it before ``import delimited``.

(d) 2SLS with the expected instrument known in closed form (a Bernoulli
share instrument, column ``mu``), one absorbed FE, clustered or HC1 SEs:
``ivreghdfe y x1 (d = zr), absorb(f)`` with ``zr = z - mu``
(``method='recenter'``), ``ivreghdfe y x1 mu (d = z), absorb(f)``
(``method='control'``) and the naive ``ivreghdfe y x1 (d = z), absorb(f)``.
StatsPAI gets ``mu`` two ways: as the row mean of counterfactual draws built
so that the mean is ``mu`` exactly (two copies of ``mu``; the mean of two
equal doubles is that double), and as ``expected="mu"``. Tolerance rel 1e-8
(observed <= 3e-15; one FE is demeaned exactly).

(c) Randomisation inference on 200 *fixed* within-stratum permutations of
the instrument, shared by both sides (ritest ``samplingsourcefile``), so the
comparison is deterministic, not a stochastic screen. Stata ranks
``T = zs' M (y - b0 d)`` (``M`` = residual maker of ``x`` and the stratum FE,
via ``reghdfe ..., resid tol(1e-14)``). Because ``M`` sweeps out the strata,
this equals the recentred statistic with the exact expected instrument
under within-stratum permutation, the stratum mean of ``zs``. StatsPAI gets
the permutations as ``counterfactual=``, the stratum mean as ``expected=``
and the strata as ``absorb=``. The count ``c = #{|T_s| >= |T_obs|}`` must
match exactly and the observed ``T`` to rel 1e-9 (observed 5.6e-15). ritest
counts with an absolute tolerance of 1e-7 and StatsPAI with a relative one
of 1e-12; no draw lies within either of the observed statistic, so the
counts are comparable.

Without ``expected=`` StatsPAI centres the draws at the mean of the realised
and the 200 permuted instruments, which is not constant within strata. The
statistic then shifts by ``mu_all' M (y - b0 d)`` (1.53 at ``b0 = 1.2``) and
the count at ``b0 = 1.2`` is 150, not 146: a different, equally exact test
(``test_default_centring_is_a_different_exact_test`` pins the identity).
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.iv.recentered import _fe_demean, _wls_residualize

FX = pathlib.Path(__file__).parent / "_fixtures"
HDFE = pd.read_csv(FX / "recentered_iv_hdfe.csv", float_precision="round_trip")
RI = pd.read_csv(FX / "recentered_iv_ri.csv", float_precision="round_trip")
PERMS = pd.read_csv(FX / "recentered_iv_ri_perms.csv.gz", float_precision="round_trip")
REL = 1e-8

# ivreghdfe, printed with %24.17g (_generate_recentered_iv_Stata.do).
STATA = {
    "recenter_cluster": (1.5910831104429131, 0.087675966038617187),
    "control_cluster": (1.5990554197786582, 0.078781504314487841),
    "naive_cluster": (1.8373777318146214, 0.066966266205030808),
    "recenter_robust": (1.5910831104429131, 0.093263586546852711),
}
# ritest on the 200 permutations in recentered_iv_ri_perms.csv.gz:
# observed statistic and count c.
RITEST = {0.0: (1092.7491932922692, 0), 1.2: (11.93870531662667, 146)}


def _mu_draws():
    mu = HDFE["mu"].to_numpy()
    draws = np.column_stack([mu, mu])
    assert np.array_equal(draws.mean(axis=1), mu)
    return draws


def _perm_draws():
    assert np.array_equal(PERMS["id"].to_numpy(), RI["id"].to_numpy())
    return PERMS[[f"zs{k}" for k in range(1, 201)]].to_numpy()


@pytest.mark.parametrize("source", ["draws", "expected"])
@pytest.mark.parametrize("method", ["recenter", "control"])
def test_given_mu_matches_ivreghdfe_cluster(method, source):
    extra = {"expected": "mu"} if source == "expected" else {}
    r = sp.recentered_iv(
        HDFE,
        "y",
        "d",
        "z",
        _mu_draws(),
        covariates=["x1"],
        cluster="c",
        method=method,
        absorb="f",
        ri=False,
        **extra,
    )
    b, se = STATA[f"{method}_cluster"]
    assert r.estimate == pytest.approx(b, rel=REL)
    assert r.se == pytest.approx(se, rel=REL)
    nb, nse = STATA["naive_cluster"]
    assert r.diagnostics["unadjusted_estimate"] == pytest.approx(nb, rel=REL)
    assert r.diagnostics["unadjusted_se"] == pytest.approx(nse, rel=REL)
    assert r.n_obs == 900
    assert r.diagnostics["absorb"] == ["f"]


def test_given_mu_matches_ivreghdfe_robust():
    r = sp.recentered_iv(
        HDFE, "y", "d", "z", _mu_draws(), covariates=["x1"], absorb="f", ri=False
    )
    b, se = STATA["recenter_robust"]
    assert r.estimate == pytest.approx(b, rel=REL)
    assert r.se == pytest.approx(se, rel=REL)


def test_recentring_moves_the_estimate_on_the_fixture():
    """The fixture's error loads on mu, so recentring matters: the naive and
    recentred estimates differ by far more than the recentred SE (Stata
    numbers above: 1.837 vs 1.591, SE 0.088)."""
    r = sp.recentered_iv(
        HDFE,
        "y",
        "d",
        "z",
        _mu_draws(),
        covariates=["x1"],
        absorb="f",
        cluster="c",
        ri=False,
    )
    assert r.diagnostics["unadjusted_estimate"] - r.estimate > 2.5 * r.se


@pytest.mark.parametrize("method", ["recenter", "control"])
@pytest.mark.parametrize("beta0", [0.0, 1.2])
def test_ri_counts_match_ritest_on_fixed_permutations(beta0, method):
    # Under within-stratum permutation E[z_i] is the stratum mean.
    mu = RI.groupby("g")["zs"].transform("mean").to_numpy()
    with pytest.warns(sp.AssumptionWarning, match="constant within the absorbed"):
        r = sp.recentered_iv(
            RI,
            "y",
            "d",
            "zs",
            _perm_draws(),
            covariates=["x"],
            absorb="g",
            expected=mu,
            method=method,
            ri_null=beta0,
        )
    t_obs, count = RITEST[beta0]
    assert r.diagnostics["ri_statistic"] == pytest.approx(t_obs, rel=1e-9)
    assert r.diagnostics["ri_pvalue"] == pytest.approx((1 + count) / 201, abs=1e-15)
    # The stratum mean is absorbed, so both methods are the unadjusted IV and
    # the Hodges-Lehmann root of the test is that estimate.
    # equal in exact arithmetic; the two paths round differently (1 ulp on
    # Linux numpy 2.2)
    assert r.estimate == pytest.approx(r.diagnostics["unadjusted_estimate"], rel=1e-12)
    assert r.diagnostics["ri_hodges_lehmann"] == pytest.approx(r.estimate, rel=1e-9)


def test_default_centring_is_a_different_exact_test():
    """Without ``expected=`` the draws are centred at the mean of all 201
    instruments. The statistic is then ritest's minus ``mu_all' M e``,
    an identity checked here against the Stata ``T`` (rel 1e-9)."""
    Z = _perm_draws()
    zs = RI["zs"].to_numpy()
    mu_all = (zs + Z.sum(axis=1)) / 201.0
    fe = pd.DataFrame({"g": RI["g"].to_numpy()})
    for beta0, (t_obs, _) in RITEST.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.recentered_iv(
                RI, "y", "d", "zs", Z, covariates=["x"], absorb="g", ri_null=beta0
            )
        sw = _fe_demean(
            np.column_stack([RI["y"] - beta0 * RI["d"], RI["x"]]).astype(float), fe
        )
        e = _wls_residualize(sw[:, :1], sw[:, 1:], np.ones(len(RI)))[:, 0]
        shift = float(mu_all @ e)
        assert r.diagnostics["ri_statistic"] == pytest.approx(t_obs - shift, rel=1e-9)
