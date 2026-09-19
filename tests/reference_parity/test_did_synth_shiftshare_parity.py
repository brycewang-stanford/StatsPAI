"""Shift-share family (sp.ssaggregate / sp.shift_share_se / sp.bartik) vs R and Stata.

References, all on the same bytes (``_fixtures/did_synth_shiftshare_loc.csv``
and ``_shocks.csv``, written by ``_generate_did_synth_shiftshare_data.py``:
240 locations x 20 industries, shares that do NOT sum to one, an industry
component in the errors, endogenous ``x``):

* ``_fixtures/did_synth_shiftshare_R.json`` (``_generate_did_synth_shiftshare_R.R``):
  R ``ShiftShareSE::ivreg_ss`` / ``reg_ss`` 1.1.0 (Adao-Kolesar-Morales),
  ``ssaggregate`` (GitHub kylebutts/ssaggregate) + ``AER::ivreg`` /
  ``sandwich`` HC0, ``bartik.weight::bw`` (GitHub paulgp/bartik-weight),
  and ``AER::ivreg`` + ``sandwich`` for plain 2SLS.
* ``_fixtures/did_synth_shiftshare_stata.json``
  (``_fixtures/_generate_did_synth_shiftshare_stata.do``): Stata 18 SSC
  ``ivreg_ss`` / ``reg_ss`` (20241116), ``ssaggregate`` 1.2.2 +
  ``ivreg2 ..., robust``, ``bartik_weight`` (paulgp/bartik-weight@722ceb8),
  and ``ivregress 2sls``.

Conventions each number depends on (all the references' own):

* AKM: ``SE = sqrt(sum_k (hX_k s_k'e)^2) / RX`` with ``hX`` the coefficients
  of the control-residualised shift-share variable on the shares; controls
  always include the intercept. AKM0 inverts the null-imposed test; its
  "SE" is the CI half-width over ``z_{1-alpha/2}``; its p-value uses the
  null-imposed SE. All p-values / CIs are normal.
* IV rows: Homoscedastic uses ``RSS/n``, EHW = HC0, region cluster has no
  small-sample factor. OLS rows: ``RSS/(n-p)``, EHW x ``n/(n-p)``, cluster x
  ``G/(G-1) (n-1)/(n-p)``.
* BHJ ``ssaggregate``: residualise y, x on the controls, collapse with
  weights ``s_ln``; ``s_n`` sums to one. Shock-level IV with intercept,
  weights ``s_n``, HC0.
* ``sp.bartik`` default ``robust='hc1'`` = ``ivregress ..., vce(robust) small``
  = ``sandwich::vcovHC(type="HC1")``; ``'nonrobust'`` = ``ivregress ..., small``.
* Rotemberg: ``alpha_k = g_k Z_k'M_W x / G'Z'M_W x``,
  ``beta_k = Z_k'M_W y / Z_k'M_W x``.

Tolerance: 1e-9 relative everywhere (p-values additionally get an absolute
floor of 1e-15: the references compute ``2 * (1 - Phi(|t|))``, StatsPAI the
survival function, and the former's cancellation error is ~1e-16). Every quantity is a closed-form
least-squares / moment computation on identical bytes; observed agreement
is 1e-12 or better (the slack covers different QR / Cholesky routes).
In 1.28.0 and earlier ``sp.ssaggregate`` and ``sp.shift_share_se`` reported SEs 4x to
16x too small on this data (see ``test_akm_se_is_not_the_pre_fix_formula``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bartik._akm import _akm_fit

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "did_synth_shiftshare_R.json").read_text(encoding="utf-8"))
ST = json.loads((_FIX / "did_synth_shiftshare_stata.json").read_text(encoding="utf-8"))
LOC = pd.read_csv(_FIX / "did_synth_shiftshare_loc.csv")
SHK = pd.read_csv(_FIX / "did_synth_shiftshare_shocks.csv")
K = len(SHK)
SHCOLS = [f"sh{k + 1}" for k in range(K)]
S = LOC[SHCOLS].to_numpy()
G = SHK["g"].to_numpy()
CTRL = {"ctrl": ["c1", "ssum"], "noctrl": None}
RTOL = 1e-9
ROWS = ("Homoscedastic", "EHW", "Reg. cluster", "AKM", "AKM0")


def _close(ours, ref, rtol=RTOL):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float), np.asarray(ref, dtype=float), rtol=rtol
    )


def _close_p(ours, ref):
    # Both references compute p = 2 * (1 - Phi(|t|)), whose cancellation
    # error is ~1e-16 in absolute terms (so ~1e-4 relative at p ~ 1e-12);
    # StatsPAI uses the survival function. Hence an absolute floor of 1e-15.
    np.testing.assert_allclose(float(ours), float(ref), rtol=RTOL, atol=1e-15)


def _ssagg(spec, mode, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # incomplete-shares note
        if mode == "iv":
            return sp.ssaggregate(
                LOC,
                y="y",
                x="x",
                shares=S,
                shocks=G,
                controls=CTRL[spec],
                cluster="state",
                **kw,
            )
        return sp.ssaggregate(
            LOC, y="y", x="B", shares=S, controls=CTRL[spec], cluster="state", **kw
        )


def _bartik(spec, robust="hc1"):
    shares = pd.DataFrame(S, columns=SHCOLS)
    shocks = pd.Series(G, index=SHCOLS)
    return sp.bartik(
        LOC,
        y="y",
        endog="x",
        shares=shares,
        shocks=shocks,
        covariates=CTRL[spec],
        leave_one_out=False,
        robust=robust,
    )


# --------------------------------------------------------------------------
# AKM inference: sp.ssaggregate vs ShiftShareSE (R) and reg_ss / ivreg_ss (Stata)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["iv", "ols"])
@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_ssaggregate_all_rows_match_shiftsharese(mode, spec):
    ref = R[f"akm_{mode}_{spec}"]
    res = _ssagg(spec, mode)
    var = "x" if mode == "iv" else "B"
    d = res.diagnostics
    _close(res.params[var], ref["beta"])
    _close(res.std_errors[var], ref["se"]["AKM"])
    for row in ROWS:
        _close(d[f"SE ({row})"], ref["se"][row])
        _close_p(d[f"p-value ({row})"], ref["p"][row])
    for row in ("AKM", "AKM0"):
        _close(d[f"CI lower ({row}, 95%)"], ref["ci_l"][row])
        _close(d[f"CI upper ({row}, 95%)"], ref["ci_r"][row])


@pytest.mark.parametrize("mode", ["iv", "ols"])
@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_ssaggregate_akm_and_akm0_match_stata(mode, spec):
    ref = ST[f"akm_{mode}_{spec}"]
    res = _ssagg(spec, mode)
    d = res.diagnostics
    for row in ("AKM", "AKM0"):
        _close(res.params.iloc[-1], ref[row]["beta"])
        _close(d[f"SE ({row})"], ref[row]["se"])
        _close_p(d[f"p-value ({row})"], ref[row]["p"])
        _close(d[f"CI lower ({row}, 95%)"], ref[row]["ci_l"])
        _close(d[f"CI upper ({row}, 95%)"], ref[row]["ci_r"])


def test_iv_homoscedastic_and_ehw_rows_are_ivregress_unadjusted_and_robust():
    # ivreg_ss's own Homoscedastic / EHW rows are ivregress 2sls without and
    # with vce(robust), no `small`.
    d = _ssagg("ctrl", "iv").diagnostics
    _close(d["SE (Homoscedastic)"], ST["tsls_ctrl"]["unadjusted"]["se_x"])
    _close(d["SE (EHW)"], ST["tsls_ctrl"]["robust"]["se_x"])
    _close(d["SE (HC1)"], ST["tsls_ctrl"]["robust_small"]["se_x"])


def test_akm0_ci_at_alpha_010_and_null_p_value_with_beta0():
    ref_r = R["akm_iv_ctrl_a10_b0"]
    ref_s = ST["akm_iv_ctrl_a10_b0"]["AKM0"]
    d = _ssagg("ctrl", "iv", alpha=0.10).diagnostics
    for ref_l, ref_u in (
        (ref_r["ci_l"]["AKM0"], ref_r["ci_r"]["AKM0"]),
        (ref_s["ci_l"], ref_s["ci_r"]),
    ):
        _close(d["CI lower (AKM0, 90%)"], ref_l)
        _close(d["CI upper (AKM0, 90%)"], ref_u)
    _close(d["SE (AKM0)"], ref_r["se"]["AKM0"])
    _close(d["SE (AKM0)"], ref_s["se"])
    # beta0 enters only the null-imposed p-value; the public API tests
    # beta0 = 0, the shared kernel is checked at beta0 = 1.5 here.
    W = np.column_stack([np.ones(len(LOC)), LOC[["c1", "ssum"]].to_numpy()])
    k = _akm_fit(
        LOC["y"].to_numpy(),
        LOC["B"].to_numpy(),
        S,
        W,
        y2=LOC["x"].to_numpy(),
        beta0=1.5,
        alpha=0.10,
    )
    _close_p(k["p"]["AKM"], ref_r["p"]["AKM"])
    _close_p(k["p"]["AKM0"], ref_r["p"]["AKM0"])
    _close_p(k["p"]["AKM0"], ref_s["p"])


# --------------------------------------------------------------------------
# sp.shift_share_se on an sp.bartik fit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_shift_share_se_on_bartik_matches_ivreg_ss(spec):
    base = _bartik(spec)
    res = sp.shift_share_se(base, shares=S)
    ref = R[f"akm_iv_{spec}"]
    _close(res.params["x"], ref["beta"])
    _close(res.std_errors["x"], ref["se"]["AKM"])
    _close(res.std_errors["x"], ST[f"akm_iv_{spec}"]["AKM"]["se"])
    for row in ("Homoscedastic", "EHW", "AKM", "AKM0"):
        _close(res.diagnostics[f"SE ({row})"], ref["se"][row])
    # Point estimates untouched, controls keep their HC1 SEs.
    pd.testing.assert_series_equal(res.params, base.params)
    _close(res.std_errors.iloc[:-1], base.std_errors.iloc[:-1], rtol=0)
    assert res.diagnostics["SE (original)"] == float(base.std_errors["x"])


def test_shift_share_se_rejects_results_without_shift_share_inputs():
    fit = sp.regress("y ~ x + c1", data=LOC)
    with pytest.raises(ValueError, match="sp.bartik or sp.ssaggregate"):
        sp.shift_share_se(fit, shares=S)


def test_akm_se_is_not_the_pre_fix_formula():
    """Regression guard for the variance of 1.28.0 and earlier.

    The old code used u_k = sum_i s_ik Z_i e_i (instrument inside the sum,
    no residualised shock) -- 0.0665 here against AKM's 0.2904 -- and
    shift_share_se used the second-stage fitted values as the instrument
    (0.0176).
    """
    ref = R["akm_iv_ctrl"]["se"]["AKM"]
    assert float(_ssagg("ctrl", "iv").std_errors["x"]) > 0.9 * ref
    assert (
        float(sp.shift_share_se(_bartik("ctrl"), shares=S).std_errors["x"]) > 0.9 * ref
    )


def test_region_cluster_requires_existing_column():
    with pytest.raises(ValueError, match="cluster column"):
        sp.ssaggregate(LOC, y="y", x="x", shares=S, shocks=G, cluster="nope")


# --------------------------------------------------------------------------
# BHJ shock-level aggregation vs ssaggregate (R and Stata)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_shock_level_aggregation_matches_ssaggregate(spec):
    res = _ssagg(spec, "iv")
    agg = res.data_info["shock_data"]
    assert list(agg.columns) == ["shock", "s_n", "y", "x", "g"]
    for ref in (R[f"bhj_{spec}"], ST[f"bhj_{spec}"]):
        np.testing.assert_array_equal(np.arange(K), np.asarray(ref["n"]) - 1)
        _close(agg["s_n"], ref["s_n"])
        _close(agg["y"], ref["y"])
        _close(agg["x"], ref["x"])
        _close(res.diagnostics["beta (BHJ shock-level)"], ref["beta"])
        _close(res.diagnostics["SE (BHJ shock-level, HC0)"], ref["se_hc0"])
    _close(agg["s_n"].sum(), 1.0, rtol=1e-14)


def test_shock_level_iv_equals_location_iv_iff_sum_of_shares_controlled():
    # BHJ Prop. 1 identity (reference-free): with the sum-of-shares control
    # the shock-level coefficient IS the location-level shift-share IV.
    res = _ssagg("ctrl", "iv")
    _close(res.diagnostics["beta (BHJ shock-level)"], res.params["x"], rtol=1e-12)
    # Without it (incomplete shares) the two differ and both we and R warn.
    assert R["bhj_noctrl_warns"] is True
    with pytest.warns(UserWarning, match="sum of shares"):
        res0 = sp.ssaggregate(LOC, y="y", x="x", shares=S, shocks=G)
    assert abs(res0.diagnostics["beta (BHJ shock-level)"] - res0.params["x"]) > 1e-3


def test_complete_shares_no_controls_bhj_hc0_equals_akm():
    # Reference-free identity: with complete shares and only an intercept the
    # control-adjusted shocks are g_k - mean(B), the shock-level intercept is
    # zero, and the BHJ HC0 SE coincides with the AKM SE.
    Sc = S / S.sum(axis=1, keepdims=True)
    data = LOC.assign(B=Sc @ G)
    res = sp.ssaggregate(data, y="y", x="x", shares=Sc, shocks=G)
    d = res.diagnostics
    _close(d["SE (BHJ shock-level, HC0)"], d["SE (AKM)"], rtol=1e-10)
    _close(d["beta (BHJ shock-level)"], res.params["x"], rtol=1e-12)


# --------------------------------------------------------------------------
# sp.bartik: 2SLS and Rotemberg weights
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_bartik_2sls_hc1_and_nonrobust_match_r_and_stata(spec):
    ref = R[f"tsls_{spec}"]
    st = ST[f"tsls_{spec}"]
    hc1 = _bartik(spec)
    cl = _bartik(spec, robust="nonrobust")
    names = {"Intercept": "(Intercept)", "c1": "c1", "ssum": "ssum", "x": "x"}
    for ours, theirs in names.items():
        if ours not in hc1.params.index:
            continue
        _close(hc1.params[ours], ref["coef"][theirs])
        _close(hc1.std_errors[ours], ref["se_hc1"][theirs])
        _close(cl.std_errors[ours], ref["se_classical"][theirs])
    _close(hc1.params["x"], st["robust_small"]["b_x"])
    _close(hc1.std_errors["x"], st["robust_small"]["se_x"])
    _close(hc1.std_errors["Intercept"], st["robust_small"]["se_cons"])
    _close(cl.std_errors["x"], st["unadjusted_small"]["se_x"])
    _close(cl.std_errors["Intercept"], st["unadjusted_small"]["se_cons"])


def test_bartik_rejects_unsupported_robust():
    with pytest.raises(ValueError, match="not supported"):
        _bartik("ctrl", robust="hc3")


@pytest.mark.parametrize("spec", ["ctrl", "noctrl"])
def test_rotemberg_weights_match_bartik_weight(spec):
    res = _bartik(spec)
    rw = res.model_info["rotemberg_weights"].set_index("industry").loc[SHCOLS]
    for ref in (R[f"rotemberg_{spec}"], ST[f"rotemberg_{spec}"]):
        _close(rw["weight"], ref["alpha"])
        _close(rw["beta"], ref["beta"])
    # GPSS identity: the 2SLS estimate is the alpha-weighted sum of the
    # just-identified per-industry estimates, and the weights sum to one.
    _close((rw["weight"] * rw["beta"]).sum(), res.params["x"], rtol=1e-11)
    _close(rw["weight"].sum(), 1.0, rtol=1e-12)
