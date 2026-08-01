* ---------------------------------------------------------------------------
* Stata reference fixture for StatsPAI's dynamic-panel GMM family.
*
* Produces (in this directory):
*   dynpanel_abdata.csv     -- the Arellano-Bond (1991) UK employment panel
*                              (`webuse abdata`), 140 firms x 1976-1984,
*                              unbalanced with interior gaps.  Exported in
*                              %21.16e so the CSV round-trips at full double
*                              precision and Python reads the *same bytes*
*                              Stata estimated on.
*   dynpanel_stata_raw.csv  -- long-format reference values, one row per
*                              (spec, key, value).  Keys are
*                                coef:<term>  se:<term>  e:<scalar>  r:<scalar>
*
* Fold into JSON with:  python3 _fold_dynpanel_stata.py
*
* Requires: Stata 18, plus `ssc install xtabond2` and `ssc install xtdpdgmm`.
* Run:      stata -b do _generate_dynpanel_stata.do   (from this directory)
*
* Why these specs: they are the moment sets StatsPAI's dynamic-panel plan
* implements, in the order it implements them -- baseline difference GMM
* (validated today), lag-operator regressor lists, time dummies, a constant,
* capped instrument depth, predetermined/endogenous instrument classes,
* collapsed instruments, system GMM, and forward orthogonal deviations.
* Generating them in one pass means the Python side never has to guess a
* convention.
* ---------------------------------------------------------------------------
version 18
clear all
set more off

* xtabond2's cluster() option is only available in speed-favouring mode.
mata: mata set matafavor speed, perm

* ---------------------------------------------------------------------------
* dp_dump: append every coefficient, SE and e()-scalar of the last estimation
* to the long CSV under a spec label.  Generic over commands (xtabond,
* xtdpdsys, xtabond2, xtdpdgmm) because it reads `e(scalars)` rather than a
* hard-coded list.
* ---------------------------------------------------------------------------
capture program drop dp_dump
program define dp_dump
    syntax , spec(string)
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    local i = 0
    foreach nm of local names {
        local ++i
        local cv = `b'[1, `i']
        local sv = sqrt(`V'[`i', `i'])
        file write DP `"`spec',coef:`nm',`=string(`cv', "%21.16e")'"' _n
        file write DP `"`spec',se:`nm',`=string(`sv', "%21.16e")'"' _n
    }
    local escal : e(scalars)
    foreach s of local escal {
        local ev = e(`s')
        file write DP `"`spec',e:`s',`=string(`ev', "%21.16e")'"' _n
    }
end

* Note: Stata 18's collection framework clears r(chi2)/r(df) before an
* `estat sargan` call returns, so postestimation r()-scalars are not
* harvestable.  Everything the parity tests need is already in e():
*   e(arm1) / e(arm2)  Arellano-Bond AR(1)/AR(2) z statistics
*   e(sargan)          Sargan (one-step) or the Hansen J (after twostep)

* ---------------------------------------------------------------------------
* Data
* ---------------------------------------------------------------------------
webuse abdata, clear
xtset id year

* Full-precision export of the columns the parity tests use.
preserve
keep id year ind n w k ys yr1976-yr1984
format n w k ys %21.16e
export delimited using "dynpanel_abdata.csv", replace datafmt
restore

file open DP using "dynpanel_stata_raw.csv", write replace text
file write DP "spec,key,value" _n

* ---------------------------------------------------------------------------
* A. Difference GMM -- the paths StatsPAI already implements (regression
*    anchors), then the ones it is adding.
* ---------------------------------------------------------------------------

* A1 baseline: pure AR(1), no covariates, one-step robust.
xtabond n, lags(1) noconstant vce(robust)
dp_dump, spec(A1_ar1_1step_robust)

* A2 pure AR(1), one-step classical (Sargan is only defined here).
xtabond n, lags(1) noconstant
dp_dump, spec(A2_ar1_1step_classic)

* A3 AR(2), one-step robust.
xtabond n, lags(2) noconstant vce(robust)
dp_dump, spec(A3_ar2_1step_robust)

* A4 AR(1), two-step with Windmeijer-corrected robust SEs.
xtabond n, lags(1) noconstant twostep vce(robust)
dp_dump, spec(A4_ar1_2step_wc)

* A5 AR(1), two-step conventional (downward-biased) SEs + Sargan.
xtabond n, lags(1) noconstant twostep
dp_dump, spec(A5_ar1_2step_conv)

* A6 capped instrument depth (maxldep) -- StatsPAI's gmm_lags=(2, 4).
xtabond n, lags(1) noconstant maxldep(3) vce(robust)
dp_dump, spec(A6_ar1_maxldep3)

* A7 minimum lag shifted (StatsPAI gmm_lags=(3, None)).  `xtabond` has no
* option for this (only maxldep), so the reference comes from xtabond2:
* gmm(L.n, lag(2 .)) instruments the differenced equation with L3.n onwards.
xtabond2 n L.n, gmm(L.n, lag(2 .)) noleveleq noconstant robust
dp_dump, spec(A7_minlag3)

* ---------------------------------------------------------------------------
* B. Lag-operator regressor lists (the Arellano-Bond 1991 Table 4 shape).
*    B1 is the specification StatsPAI currently CANNOT reproduce because
*    listwise deletion of user-built lag columns amputates the instrument set.
* ---------------------------------------------------------------------------

* B1 AB(1991)-style: l(0/1).w and l(0/2).k as strictly exogenous regressors.
xtabond n l(0/1).w l(0/2).k, lags(2) noconstant vce(robust)
dp_dump, spec(B1_ab1991_1step_robust)

* B2 same, two-step Windmeijer.
xtabond n l(0/1).w l(0/2).k, lags(2) noconstant twostep vce(robust)
dp_dump, spec(B2_ab1991_2step_wc)

* B3 same as B1 plus year dummies.
xtabond n l(0/1).w l(0/2).k yr1979 yr1980 yr1981 yr1982 yr1983 yr1984, ///
    lags(2) noconstant vce(robust)
dp_dump, spec(B3_ab1991_yeardum)

* B4 with the constant Stata adds by default (level moment for _cons).
xtabond n l(0/1).w l(0/2).k, lags(2) vce(robust)
dp_dump, spec(B4_ab1991_cons)

* ---------------------------------------------------------------------------
* C. Instrument classes: predetermined and endogenous regressors.
* ---------------------------------------------------------------------------

* C1 w predetermined (own lags 1+ are valid instruments).
xtabond n l(0/2).k, lags(2) noconstant pre(w, lagstruct(1, .)) vce(robust)
dp_dump, spec(C1_w_predetermined)

* C2 w endogenous (own lags 2+ are valid instruments).
xtabond n l(0/2).k, lags(2) noconstant endogenous(w, lagstruct(1, .)) vce(robust)
dp_dump, spec(C2_w_endogenous)

* ---------------------------------------------------------------------------
* D. xtabond2 -- collapse, system GMM, forward orthogonal deviations.
*    xtabond2 is the reference the applied literature reports, so it is the
*    primary anchor for everything below.
* ---------------------------------------------------------------------------

* D1 difference GMM through xtabond2 (must equal A1 up to xtabond2's
*    small-sample conventions) -- pins the two references against each other.
xtabond2 n L.n, gmm(L.n, lag(1 .)) noleveleq noconstant robust
dp_dump, spec(D1_x2_diff_1step)

* D2 difference GMM, collapsed instruments.
xtabond2 n L.n, gmm(L.n, lag(1 .) collapse) noleveleq noconstant robust
dp_dump, spec(D2_x2_diff_collapse)

* D3 difference GMM with exogenous covariates, one-step robust.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k, equation(diff)) ///
    noleveleq noconstant robust
dp_dump, spec(D3_x2_diff_ivwk)

* D4 SYSTEM GMM, one-step robust (the headline missing capability).
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k) robust
dp_dump, spec(D4_x2_sys_1step)

* D5 system GMM, two-step Windmeijer.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k) twostep robust
dp_dump, spec(D5_x2_sys_2step_wc)

* D6 system GMM, collapsed.
xtabond2 n L.n w k, gmm(L.n, lag(1 .) collapse) iv(w k) twostep robust
dp_dump, spec(D6_x2_sys_collapse)

* D7 forward orthogonal deviations, difference GMM.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k, equation(diff)) ///
    noleveleq noconstant orthogonal robust
dp_dump, spec(D7_x2_fod_diff)

* D8 forward orthogonal deviations, system GMM two-step.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k) orthogonal twostep robust
dp_dump, spec(D8_x2_fod_sys_2step)

* D9 forward orthogonal deviations, system GMM one-step -- isolates the H
* cross-quadrant convention from the two-step weighting.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k) orthogonal robust
dp_dump, spec(D9_x2_fod_sys_1step)

* D10 forward orthogonal deviations, difference GMM two-step.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k, equation(diff)) ///
    noleveleq noconstant orthogonal twostep robust
dp_dump, spec(D10_x2_fod_diff_2step)

* ---------------------------------------------------------------------------
* F. Clustering on a coarser unit than the panel id (industry).  The moment
*    conditions are then independent across industries, not across firms, so
*    only the meat of the sandwich re-groups -- the one-step weight Z'HZ is a
*    within-firm object either way.
* ---------------------------------------------------------------------------
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k, equation(diff)) ///
    noleveleq noconstant cluster(ind)
dp_dump, spec(F1_x2_diff_cluster_ind)

xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k) cluster(ind)
dp_dump, spec(F2_x2_sys_cluster_ind)

* F4 sanity anchor: cluster(id) IS the default unit clustering, so this must
* reproduce the plain robust fit exactly.
xtabond2 n L.n w k, gmm(L.n, lag(1 .)) iv(w k, equation(diff)) ///
    noleveleq noconstant cluster(id)
dp_dump, spec(F4_x2_diff_cluster_id)

* ---------------------------------------------------------------------------
* G. Anderson-Hsiao (1981) simple IV: ONE pooled instrument for the
*    differenced lagged dependent variable rather than Arellano-Bond's
*    block-diagonal set.  Two classical variants:
*      levels       instrument L2.y    -> gmm(L.y, lag(1 1) collapse)
*      differences  instrument D.L2.y  -> iv(L2.y, equation(diff))
* ---------------------------------------------------------------------------
xtabond2 n L.n, gmm(L.n, lag(1 1) collapse) noleveleq noconstant robust
dp_dump, spec(G1_ah_levels)

xtabond2 n L.n, iv(L2.n, equation(diff)) noleveleq noconstant robust
dp_dump, spec(G2_ah_differences)

xtabond2 n L.n w k, gmm(L.n, lag(1 1) collapse) iv(w k, equation(diff)) ///
    noleveleq noconstant robust
dp_dump, spec(G3_ah_levels_wk)

xtabond2 n L.n w k, iv(L2.n w k, equation(diff)) noleveleq noconstant robust
dp_dump, spec(G4_ah_differences_wk)

* ---------------------------------------------------------------------------
* H. Bias-corrected LSDV (Bruno 2005, `xtlsdvc`).  The within estimator's
*    Nickell bias is estimated from a consistent initial estimator and
*    subtracted; the three initialisers and the three expansion orders are
*    the whole option surface.
*    NOTE: xtlsdvc's reported SEs are the LSDV ones (it says so); only the
*    coefficients are meaningful without bootstrap(), so only those are
*    compared.
* ---------------------------------------------------------------------------
xtlsdvc n w k, initial(ab) bias(2)
dp_dump, spec(H1_lsdvc_ab_b2)

xtlsdvc n w k, initial(ah) bias(2)
dp_dump, spec(H2_lsdvc_ah_b2)

xtlsdvc n w k, initial(bb) bias(2)
dp_dump, spec(H3_lsdvc_bb_b2)

xtlsdvc n w k, initial(ab) bias(1)
dp_dump, spec(H4_lsdvc_ab_b1)

xtlsdvc n w k, initial(ab) bias(3)
dp_dump, spec(H5_lsdvc_ab_b3)

xtlsdvc n, initial(ab) bias(2)
dp_dump, spec(H6_lsdvc_ar1_only)

* ---------------------------------------------------------------------------
* E. xtdpdsys -- Stata's built-in Blundell-Bond, a second system anchor.
* ---------------------------------------------------------------------------
xtdpdsys n, lags(1) vce(robust)
dp_dump, spec(E1_xtdpdsys_ar1_1step)

xtdpdsys n, lags(1) twostep vce(robust)
dp_dump, spec(E2_xtdpdsys_ar1_2step_wc)

xtdpdsys n w k, lags(1) vce(robust)
dp_dump, spec(E3_xtdpdsys_wk_1step)

file close DP
di "dynpanel fixture written."
