"""Reference parity: weak-IV diagnostics and JIVE vs R ivDiag / ivmodel and Stata.

Functions pinned here, each against the implementation its own docstring
names, on identical CSV bytes:

* ``sp.tF_critical_value`` / ``sp.tF_adjustment`` -- R ``ivDiag::tF``
  (the LMMP 2022 5 % table, linear in sqrt(F)).
* ``sp.effective_f_test`` -- Stata ``weakivtest`` (Montiel Olea &
  Pflueger's own command) for one and three instruments, robust and
  clustered; R ``ivDiag::eff_F`` for one instrument.
* ``sp.iv_diag`` -- R ``ivDiag::ivDiag`` analytic block: 2SLS and OLS
  coefficients and SEs, classical first-stage F, effective F and the tF
  critical value / interval.
* ``sp.weakrobust`` -- R ``ivmodel::CLR`` (statistic, p-value, set) and
  Stata ``weakiv, md small`` (CLR, K, AR at H0).
* ``sp.jive`` -- Stata ``jive`` (Stata Journal st0108) ``ujive1`` /
  ``ujive2``, default and ``robust`` standard errors.
* ``sp.ivqreg`` (just-identified, scalar ``D``) -- the Chernozhukov-Hansen
  inverse-QR estimate as the root of the instrument's R ``quantreg::rq``
  coefficient. This is a definition-level check, not a package parity:
  the dedicated R package ``IVQR`` 0.1.0 stops with "the condition has
  length > 1" on R 4.5, and Stata ``ivqreg2`` is the Machado-Santos Silva
  estimator, not this one.

Fixtures
--------
``_generate_rd_iv_R.R`` writes ``rd_iv_ivw.csv`` (seeded synthetic design:
n = 800, 40 clusters, heteroskedastic, first-stage F about 4), copies
``ivDiag::rueda`` to ``rd_iv_rueda.csv`` and writes ``rd_iv_R.json``.
``_fixtures/_generate_rd_iv_stata.do`` reads the same two CSVs and writes
``rd_iv_Stata.json``.

Conventions each number depends on
----------------------------------
* ivDiag rounds every output to ``prec`` digits and calls ``eff_F`` /
  ``tF`` without forwarding ``prec``; the generator calls both with
  ``prec = 16`` and re-evaluates ``tF`` at the unrounded effective F.
* ivDiag's robust / cluster variances are ``lfe::felm``'s: HC1
  (``n / (n - k)``) and ``G / (G - 1) * (n - 1) / (n - k)``. StatsPAI's
  ``vcov='HC1'`` default and its cluster factor are the same.
* The tF critical value is indexed by the effective F (robust / cluster
  first-stage Wald F with one instrument), as ivDiag does, and is defined
  for one instrument only. Below F = 4 StatsPAI returns ``inf`` where
  ivDiag clamps to c(4) = 18.66 (documented; not compared).
* ``ivDiag::eff_F`` with more than one instrument uses the
  *un-partialled* instrument cross-product ``Z'Z`` (``felm``'s
  ``stage1$ivx``) in the trace; Montiel Olea & Pflueger's own
  ``weakivtest`` orthogonalises Z on the controls, and so does StatsPAI.
  With k = 3 ivDiag reports 2.84 where both others report 2.44 -- a
  reference problem (T4), asserted as a disagreement below.
* CLR is the homoskedastic Moreira statistic with
  ``Sigma = [y d]' M_[Z X] [y d] / (n - p - k)`` (ivmodel; weakiv ``md
  small``). Its conditional p-value is the one-dimensional integral
  ivmodel evaluates. ivmodel's CLR set solves for its critical value with
  ``uniroot``'s default tolerance (~1.2e-4), so the set is compared at
  5e-5 relative and additionally pinned by the identity p(endpoint) =
  alpha; weakiv stores the set only as a 6-7 digit display string.
* Stata ``weakiv, md small`` agrees with ivmodel / StatsPAI on CLR and K
  to 1.1e-7 and on AR to 2e-8; the residual was not bisected further
  (it is inside the 1e-6 budget and ivmodel matches StatsPAI to 1e-12).
* Stata ``jive``: the default V uses ``s^2`` from the *demeaned* residuals
  ``(r(Var)(N-1)/(N-k))``, StatsPAI ``e'e/(n-k)``. The two coincide
  because the jackknife instrument set contains the constant, so the IV
  normal equations force ``sum(e) = 0``; the test asserts that identity
  rather than assuming it.

Tolerances: deterministic closed forms -> rel 1e-9 unless stated; the
fixture JSONs carry 17 significant digits.
"""

from __future__ import annotations

import json
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


def _rel(a, b) -> float:
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


@pytest.fixture(scope="module")
def R():
    return json.loads((_FIX / "rd_iv_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ST():
    return json.loads((_FIX / "rd_iv_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ivw():
    return pd.read_csv(_FIX / "rd_iv_ivw.csv")


@pytest.fixture(scope="module")
def rueda():
    return pd.read_csv(_FIX / "rd_iv_rueda.csv")


_DESIGNS = {
    "rueda_hc": (
        "rueda",
        "e_vote_buying",
        "lm_pob_mesa",
        ["lz_pob_mesa_f"],
        ["lpopulation", "lpotencial"],
        None,
    ),
    "rueda_cl": (
        "rueda",
        "e_vote_buying",
        "lm_pob_mesa",
        ["lz_pob_mesa_f"],
        ["lpopulation", "lpotencial"],
        "muni_code",
    ),
    "ivw1_hc": ("ivw", "y", "d", ["z1"], ["x1", "x2"], None),
    "ivw1_cl": ("ivw", "y", "d", ["z1"], ["x1", "x2"], "cl"),
    "ivw3_hc": ("ivw", "y", "d", ["z1", "z2", "z3"], ["x1", "x2"], None),
    "ivw3_cl": ("ivw", "y", "d", ["z1", "z2", "z3"], ["x1", "x2"], "cl"),
}


def _design(key, ivw, rueda):
    src, y, d, z, x, cl = _DESIGNS[key]
    return (ivw if src == "ivw" else rueda), y, d, z, x, cl


# --------------------------------------------------------------------------- #
#  tF critical value -- ivDiag::tF
# --------------------------------------------------------------------------- #
def test_tf_critical_value_matches_ivdiag(R):
    grid = R["tF_grid"]
    ours = [sp.tF_critical_value(f) for f in grid["F"]]
    assert _rel(ours, grid["cF"]) < 1e-12
    alias = [sp.tF_adjustment(f) for f in grid["F"]]
    assert _rel(alias, grid["cF"]) < 1e-12


def test_tf_critical_value_known_points():
    # The two numbers the LMMP procedure is usually quoted by: F = 10
    # needs |t| > 3.44 rather than 1.96, and 1.96 is reached only at
    # F = 10.3**2. The pre-1.29 table gave 3.16 and reached 1.96 at F = 75.
    assert sp.tF_critical_value(10.0) == pytest.approx(3.4352668077979445, rel=1e-14)
    assert sp.tF_critical_value(100.0) > 1.96
    assert sp.tF_critical_value(10.3**2) == 1.96
    assert sp.tF_critical_value(3.99) == np.inf  # ivDiag clamps to 18.66
    fs = np.linspace(4.0, 120.0, 400)
    cs = np.array([sp.tF_critical_value(f) for f in fs])
    assert np.all(np.diff(cs) <= 1e-15)


# --------------------------------------------------------------------------- #
#  Effective F -- Stata weakivtest (all designs), ivDiag::eff_F (k = 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(_DESIGNS))
def test_effective_f_matches_weakivtest(key, ST, R, ivw, rueda):
    data, _y, d, z, x, cl = _design(key, ivw, rueda)
    res = sp.effective_f_test(data, endog=d, instruments=z, exog=x, cluster=cl)
    assert _rel(res["F_eff"], ST["weakivtest"][key]) < 1e-9
    if len(z) == 1:
        assert _rel(res["F_eff"], R["ivdiag"][key]["eff_F"]) < 1e-9
        # Identity: with one instrument F_eff is the robust / cluster
        # first-stage Wald F.
        wald = R["ivdiag"][key]["F_cluster" if cl else "F_robust"]
        assert _rel(res["F_eff"], wald) < 1e-9
    # Classical first-stage F (ivDiag F.standard)
    assert _rel(res["first_stage_F"], R["ivdiag"][key]["F_standard"]) < 1e-9


@pytest.mark.parametrize("key", ["ivw3_hc", "ivw3_cl"])
def test_ivdiag_eff_f_multi_instrument_is_the_outlier(key, ST, R):
    """T4: ivDiag's k > 1 effective F uses un-partialled Z'Z.

    weakivtest (MOP's own command) and StatsPAI agree to 1e-9 (previous
    test); ivDiag differs by 16 % / 13 % here. Asserting the gap keeps the
    reference problem visible if a later ivDiag release changes it.
    """
    gap = _rel(R["ivdiag"][key]["eff_F"], ST["weakivtest"][key])
    assert gap > 0.05


# --------------------------------------------------------------------------- #
#  iv_diag -- ivDiag::ivDiag analytic block
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(_DESIGNS))
def test_iv_diag_matches_ivdiag(key, R, ivw, rueda):
    data, y, d, z, x, cl = _design(key, ivw, rueda)
    ref = R["ivdiag"][key]
    r = sp.iv_diag(data, y=y, endog=d, instruments=z, exog=x, cluster=cl, n_boot=0)
    assert _rel(r.beta_2sls, ref["beta_2sls"]) < 1e-9
    assert _rel(r.se_2sls, ref["se_2sls"]) < 1e-9
    assert _rel(r.beta_ols, ref["beta_ols"]) < 1e-9
    # Through 1.28.0 se_ols was homoskedastic whatever vcov / cluster said.
    assert _rel(r.se_ols, ref["se_ols"]) < 1e-9
    assert _rel(r.first_stage_F, ref["F_standard"]) < 1e-9
    if len(z) == 1:
        assert _rel(r.effective_F, ref["eff_F"]) < 1e-9
        assert _rel(r.tF_critical_value, ref["tF_cF"]) < 1e-9
        assert _rel(r.tF_adjusted_ci, ref["tF_ci"]) < 1e-9
    else:
        # LMMP tF is a just-identified procedure; ivDiag omits it too.
        assert np.isnan(r.tF_critical_value)


# --------------------------------------------------------------------------- #
#  weakrobust -- ivmodel::CLR and Stata weakiv, md small
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel(ivw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.weakrobust(
            ivw, y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x1", "x2"]
        ).to_dict()


def test_weakrobust_clr_matches_ivmodel(panel, R):
    ref = R["ivmodel_ivw3"]
    # Through 1.28.0 the orthonormalised instruments were not orthonormal
    # for k >= 2 and this statistic was 4.1 % high.
    assert _rel(panel["clr_stat"], ref["clr_stat_b0"]) < 1e-9
    assert _rel(panel["clr_pvalue"], ref["clr_p_b0"]) < 1e-9
    # ivmodel solves its critical value with uniroot's default tolerance.
    assert _rel(panel["clr_ci"], ref["clr_ci"]) < 5e-5
    assert _rel(panel["ar_stat"], ref["ar_F_b0"]) < 1e-9


def test_weakrobust_clr_other_null_matches_ivmodel(ivw, R):
    from statspai.iv.weak_identification import conditional_lr_test

    r = conditional_lr_test(
        y="y",
        endog="d",
        instruments=["z1", "z2", "z3"],
        exog=["x1", "x2"],
        data=ivw,
        beta0=0.5,
    )
    assert _rel(r.statistic, R["ivmodel_ivw3"]["clr_stat_b05"]) < 1e-9
    assert _rel(r.pvalue, R["ivmodel_ivw3"]["clr_p_b05"]) < 1e-9


def _cset_hull(text: str):
    nums = [float(t) for t in re.findall(r"-?\d*\.\d+|-?\d+", text)]
    return min(nums), max(nums)


def test_weakrobust_matches_stata_weakiv(panel, ST):
    ref = ST["weakiv_md_small_ivw3"]
    assert _rel(panel["clr_stat"], ref["clr_stat"]) < 1e-6
    assert _rel(panel["k_stat"], ref["k_chi2"]) < 1e-6
    assert _rel(panel["k_pvalue"], ref["k_p"]) < 1e-6
    assert _rel(3.0 * panel["ar_stat"], ref["ar_chi2"]) < 1e-6
    # Display strings carry 6-7 significant digits.
    assert _rel(panel["clr_ci"], _cset_hull(ref["clr_cset"])) < 5e-6
    assert _rel(panel["k_ci"], _cset_hull(ref["k_cset"])) < 5e-6


def test_clr_set_endpoints_solve_the_exact_equation(panel, ivw):
    """Identity (no reference): p_CLR(endpoint) = alpha."""
    from statspai.iv.weak_identification import conditional_lr_test

    for b in panel["clr_ci"]:
        r = conditional_lr_test(
            y="y",
            endog="d",
            instruments=["z1", "z2", "z3"],
            exog=["x1", "x2"],
            data=ivw,
            beta0=b,
        )
        assert r.pvalue == pytest.approx(0.05, abs=1e-9)


def test_weakrobust_k_is_evaluated_at_h0(ivw):
    """Through 1.28.0 K 'at h0' was read off the grid point nearest h0.

    With h0 far outside the grid that returned K at the grid endpoint.
    """
    from statspai.iv.weak_iv_ci import k_test_ci

    kw = dict(y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x1", "x2"])
    for h0 in (0.0, 250.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = sp.weakrobust(ivw, h0=h0, include_clr=False, **kw).to_dict()
        exact = k_test_ci(data=ivw, beta_grid=np.array([h0]), **kw).statistic[0]
        assert _rel(p["k_stat"], exact) < 1e-12


# --------------------------------------------------------------------------- #
#  jive -- Stata jive (st0108) ujive1 / ujive2
# --------------------------------------------------------------------------- #
_ORDER = ["d", "x1", "x2", "_cons"]  # Stata e(b) order


@pytest.mark.parametrize("variant,skey", [("jive1", "ujive1"), ("jive2", "ujive2")])
def test_jive_matches_stata(variant, skey, ST, ivw):
    kw = dict(x_endog=["d"], x_exog=["x1", "x2"], z=["z1", "z2", "z3"], variant=variant)
    r = sp.jive(ivw, "y", **kw)
    ref = ST["jive"][skey]
    b = [float(r.params[k]) for k in _ORDER]
    assert _rel(b, ref["b"]) < 1e-9
    se = [float(r.std_errors[k]) for k in _ORDER]
    assert _rel(se, ref["se"]) < 1e-9
    # Why Stata's demeaned-residual s^2 is the same number: sum(e) = 0.
    X = np.column_stack([ivw["d"], ivw["x1"], ivw["x2"], np.ones(len(ivw))])
    e = ivw["y"].to_numpy() - X @ np.asarray(b)
    assert abs(e.mean()) < 1e-10 * np.sqrt(np.mean(e**2))
    assert ref["rmse"] ** 2 == pytest.approx(float(e @ e) / (len(ivw) - 4), rel=1e-9)

    rr = sp.jive(ivw, "y", robust="robust", **kw)
    ref = ST["jive"][skey + "_robust"]
    assert _rel([float(rr.std_errors[k]) for k in _ORDER], ref["se"]) < 1e-9


def test_jive1_instrument_is_leave_one_out_fit(ivw):
    """Identity: the JIVE1 instrument is the leave-one-out first-stage fit."""
    Zall = np.column_stack([np.ones(len(ivw)), ivw[["x1", "x2", "z1", "z2", "z3"]]])
    d = ivw["d"].to_numpy()
    r = sp.jive(ivw, "y", x_endog=["d"], x_exog=["x1", "x2"], z=["z1", "z2", "z3"])
    # Rebuild the IV estimate from brute-force leave-one-out fits.
    loo = np.empty(len(ivw))
    for i in range(len(ivw)):
        m = np.arange(len(ivw)) != i
        g = np.linalg.lstsq(Zall[m], d[m], rcond=None)[0]
        loo[i] = Zall[i] @ g
    Xh = np.column_stack([np.ones(len(ivw)), ivw[["x1", "x2"]], loo])
    X = np.column_stack([np.ones(len(ivw)), ivw[["x1", "x2"]], d])
    beta = np.linalg.solve(Xh.T @ X, Xh.T @ ivw["y"].to_numpy())
    assert _rel(float(r.params["d"]), beta[-1]) < 1e-9


# --------------------------------------------------------------------------- #
#  ivqreg -- root of the CH inverse-QR equation (R quantreg)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_ivqreg_is_the_inverse_qr_root(tau, R):
    ref = R["ivqr_root"][str(tau)]
    assert ref["n_sign_changes"] == 1  # the root is unique on the grid
    d = pd.read_csv(_FIX / "rd_iv_ivqr.csv")
    r = sp.ivqreg(d, y="y", endog="d", instruments="z", exog=["x1"], tau=tau)
    # Brent refinement stops at xtol 1e-6 in alpha; observed ~1e-13.
    assert _rel(float(r.params["d"]), ref["root"]) < 1e-6
