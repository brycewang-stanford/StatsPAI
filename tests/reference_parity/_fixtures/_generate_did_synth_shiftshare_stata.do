* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_did_synth_shiftshare_parity.py
* Requires Stata 18 and, installed into the PRIVATE ado dir
* _fixtures/_ado_did_synth/ (never PLUS; see the top of this file):
*   ssc: reg_ss, ivreg_ss (Adao-Kolesar-Morales-Zhang), ssaggregate
*        (Borusyak-Hull-Jaravel), ivreg2, ranktest, ftools, reghdfe
*   GitHub: bartik_weight.ado from paulgp/bartik-weight@722ceb8 (code/),
*        copied into _ado_did_synth/b/ (the repo has no stata.toc, so
*        `net install` cannot fetch it).
* Run _generate_did_synth_shiftshare_data.py first, then from _fixtures/:
*   stata-mp -b do _generate_did_synth_shiftshare_stata.do
* Writes did_synth_shiftshare_stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * ivreg_ss / reg_ss (akmtype 1 = AKM, 0 = AKM0): identical algebra to R
*   ShiftShareSE (Xddd = (L'L)^-1 L'Xdd is the residualised shock, the
*   variance is sum_k (e'L_k)^2 Xddd_k^2 / (Xdd'Gdd)^2). AKM0's reported
*   "se" is the CI half-width / z_{1-alpha/2}. The returned se / CI are read
*   from r() (full-precision scalars the Mata code posts), not from the
*   e() macros.
* * Homoskedastic / EHW rows: ivreg_ss calls ivregress 2sls (sigma^2 = RSS/N,
*   robust without small-sample factor); reproduced here directly.
*   ivregress ..., small is the n-k analogue.
* * ssaggregate: residualise y, x on the control set, collapse with
*   aweights s_ln to industry means; s_n normalised to sum to one. The
*   shock-level IV is ivreg2 y (x = g) [aw = s_n], robust (HC0), BHJ's
*   recommended command.
* * bartik_weight: alpha_k = g_k Z_k'M_W x / G'Z'M_W x, beta_k =
*   Z_k'M_W y / Z_k'M_W x, M_W annihilating [controls, 1]. The shock
*   vector is read from K wide variables g1..gK (bartik_weight collapses
*   them with (first)).
* ---------------------------------------------------------------------------
version 18
clear all
set more off
set type double
local A "`c(pwd)'/_ado_did_synth"
sysdir set PLUS "`A'"
adopath ++ "`A'"
adopath ++ "`A'/b"

import delimited using "did_synth_shiftshare_shocks.csv", clear asdouble
tempfile shocks
save `shocks'
local K = _N
forvalues k = 1/`K' {
    local g`k' = g[`k']
}

import delimited using "did_synth_shiftshare_loc.csv", clear asdouble
tempfile loc
save `loc'

tempname fh
file open `fh' using "did_synth_shiftshare_stata.json", write replace text
file write `fh' "{" _n

* ---- AKM / AKM0 --------------------------------------------------------------
* spec: name | command | controls
foreach spec in iv_ctrl iv_noctrl ols_ctrl ols_noctrl {
    local ctrl ""
    if strpos("`spec'", "_ctrl") local ctrl "control_varlist(c1 ssum)"
    file write `fh' `"  "akm_`spec'": {"'
    foreach t in 1 0 {
        use `loc', clear
        if substr("`spec'", 1, 2) == "iv" {
            quietly ivreg_ss y, endogenous_var(x) shiftshare_iv(b) ///
                share_varlist(sh1-sh`K') `ctrl' akmtype(`t')
        }
        else {
            quietly reg_ss y, shiftshare_var(b) share_varlist(sh1-sh`K') ///
                `ctrl' akmtype(`t')
        }
        local nm = cond(`t' == 1, "AKM", "AKM0")
        file write `fh' `""`nm'": {"beta": "' %23.16e (r(b)) `", "se": "' %23.16e (r(se)) ///
            `", "p": "' %23.16e (r(p)) `", "ci_l": "' %23.16e (r(CIl)) `", "ci_r": "' %23.16e (r(CIu)) "}"
        if `t' == 1 file write `fh' ", "
    }
    file write `fh' "}," _n
}
use `loc', clear
quietly ivreg_ss y, endogenous_var(x) shiftshare_iv(b) share_varlist(sh1-sh`K') ///
    control_varlist(c1 ssum) akmtype(0) alpha(0.10) beta0(1.5)
file write `fh' `"  "akm_iv_ctrl_a10_b0": {"AKM0": {"beta": "' %23.16e (r(b)) `", "se": "' %23.16e (r(se)) ///
    `", "p": "' %23.16e (r(p)) `", "ci_l": "' %23.16e (r(CIl)) `", "ci_r": "' %23.16e (r(CIu)) "}}," _n

* ---- 2SLS (ivregress) ----------------------------------------------------------
foreach spec in ctrl noctrl {
    local ctrl ""
    if "`spec'" == "ctrl" local ctrl "c1 ssum"
    file write `fh' `"  "tsls_`spec'": {"'
    local first = 1
    foreach v in "" "vce(robust)" "small" "vce(robust) small" {
        quietly ivregress 2sls y `ctrl' (x = b), `v'
        local nm = cond("`v'" == "", "unadjusted", cond("`v'" == "small", "unadjusted_small", ///
                   cond("`v'" == "vce(robust)", "robust", "robust_small")))
        if !`first' file write `fh' ", "
        local first = 0
        file write `fh' `""`nm'": {"b_x": "' %23.16e (_b[x]) `", "se_x": "' %23.16e (_se[x]) ///
            `", "b_cons": "' %23.16e (_b[_cons]) `", "se_cons": "' %23.16e (_se[_cons]) "}"
    }
    file write `fh' "}," _n
}

* ---- Rotemberg weights (bartik_weight) ---------------------------------------
foreach spec in ctrl noctrl {
    use `loc', clear
    forvalues k = 1/`K' {
        gen double g`k' = `g`k''
    }
    local ctrl ""
    if "`spec'" == "ctrl" local ctrl "controls(c1 ssum)"
    quietly bartik_weight, z(sh1-sh`K') weightstub(g1-g`K') y(y) x(x) `ctrl'
    matrix al = r(alpha)
    matrix be = r(beta)
    file write `fh' `"  "rotemberg_`spec'": {"alpha": ["'
    forvalues k = 1/`K' {
        file write `fh' %23.16e (al[`k', 1])
        if `k' < `K' file write `fh' ", "
    }
    file write `fh' `"], "beta": ["'
    forvalues k = 1/`K' {
        file write `fh' %23.16e (be[`k', 1])
        if `k' < `K' file write `fh' ", "
    }
    file write `fh' "]}," _n
}

* ---- BHJ ssaggregate + shock-level IV ------------------------------------------
foreach spec in ctrl noctrl {
    use `loc', clear
    if "`spec'" == "ctrl" {
        ssaggregate y x, n(n) s(sh) controls("c1 ssum")
    }
    else {
        ssaggregate y x, n(n) s(sh)
    }
    merge 1:1 n using `shocks', assert(3) nogen
    sort n
    quietly ivreg2 y (x = g) [aw = s_n], robust
    file write `fh' `"  "bhj_`spec'": {"beta": "' %23.16e (_b[x]) `", "se_hc0": "' %23.16e (_se[x])
    foreach v in n s_n y x {
        file write `fh' `", "`v'": ["'
        forvalues i = 1/`=_N' {
            file write `fh' %23.16e (`v'[`i'])
            if `i' < _N file write `fh' ", "
        }
        file write `fh' "]"
    }
    file write `fh' "}," _n
}

* ---- versions -------------------------------------------------------------------
file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "'
file write `fh' `""reg_ss": "SSC 20241116", "ivreg_ss": "SSC 20241116", "'
file write `fh' `""ssaggregate": "SSC 1.2.2 (20200826)", "ivreg2": "SSC", "'
file write `fh' `""bartik_weight": "GitHub paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770 code/bartik_weight.ado"}"' _n
file write `fh' "}" _n
file close `fh'
