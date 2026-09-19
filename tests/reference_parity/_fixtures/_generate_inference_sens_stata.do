* ---------------------------------------------------------------------------
* Stata references for tests/reference_parity/test_inference_sens_stata_parity.py
*
* Requires: Stata 18 + SSC boottest, ritest, rbounds, psacalc. They are
*           installed into a PRIVATE ado directory next to this file
*           (_ado_inference_sens/, not committed) -- never into the user's PLUS.
* Run:      stata -b do _generate_inference_sens_stata.do   (from this directory)
* Data:     inference_sens_*.csv, written by _generate_inference_sens_data.py.
* Output:   inference_sens_stata.json (every number at %25.17e).
* ---------------------------------------------------------------------------
version 18
clear all
set maxvar 10000
set type double
set more off
sysdir set PLUS "`c(pwd)'/_ado_inference_sens"
adopath ++ "`c(pwd)'/_ado_inference_sens"
foreach p in boottest ritest rbounds psacalc {
    capture which `p'
    if _rc {
        ssc install `p', replace
    }
}

capture file close fh
file open fh using "inference_sens_stata.json", write replace text
file write fh "{" _n

capture program drop _num
program define _num
    * _num <key> <expression> [last]
    args key value last
    local comma = cond("`last'" == "last", "", ",")
    file write fh `"    "`key'": "' %25.17e (`value') `"`comma'"' _n
end

capture program drop _open
program define _open
    args name
    file write fh `"  "`name'": {"' _n
end

capture program drop _close
program define _close
    file write fh `"  },"' _n
end

* ---------------------------------------------------------------------------
* 1. regress, vce(cluster) and vce(jackknife)
* ---------------------------------------------------------------------------
import delimited "inference_sens_reg.csv", clear asdouble

quietly regress y d x1 x2, vce(cluster s12)
_open regress_cluster_s12
foreach v in d x1 x2 _cons {
    _num b_`v' _b[`v']
    _num se_`v' _se[`v']
}
_num N e(N_clust) last
_close

* Delete-one-cluster jackknife centred at the replicate mean (default) ...
* The jackknife prefix stores its replicates in FLOAT unless told `double`
* (moves the SEs by ~7e-8 relative); `double` gives the full-precision
* number, and the float default is kept below to document the difference.
quietly regress y d x1 x2, vce(jackknife, cluster(s12))
_open jackknife_s12_float
_num se_d _se[d] last
_close
quietly regress y d x1 x2, vce(jackknife, cluster(s12) double)
matrix T = r(table)
_open jackknife_s12
foreach v in d x1 x2 _cons {
    _num b_`v' _b[`v']
    _num se_`v' _se[`v']
    local j = colnumb(T, "`v'")
    _num p_`v' T[4,`j']
    _num ll_`v' T[5,`j']
    _num ul_`v' T[6,`j']
}
_num df_r e(df_r) last
_close

* ... and centred at the full-sample estimate (mse).
quietly regress y d x1 x2, vce(jackknife, cluster(s12) mse double)
_open jackknife_mse_s12
foreach v in d x1 x2 _cons {
    _num se_`v' _se[`v']
}
_num df_r e(df_r) last
_close

* ---------------------------------------------------------------------------
* 2. boottest: restricted wild cluster bootstrap, Rademacher. G = 12, so
*    reps(9999) >= 2^12 and boottest enumerates all 4096 sign vectors: the
*    p-value is exact. ptolerance(1e-14) makes the CI search precise.
* ---------------------------------------------------------------------------
quietly regress y d x1 x2, vce(cluster s12)
boottest d, reps(9999) weighttype(rademacher) ptolerance(1e-14) nograph
matrix C = r(CI)
_open boottest_d_h0
_num p r(p)
_num t r(t)
_num reps r(reps)
_num ci_lo C[1,1]
_num ci_hi C[1,2] last
_close

boottest x1, reps(9999) weighttype(rademacher) ptolerance(1e-14) nograph
matrix C = r(CI)
_open boottest_x1_h0
_num p r(p)
_num t r(t)
_num reps r(reps)
_num ci_lo C[1,1]
_num ci_hi C[1,2] last
_close

boottest d = 0.2, reps(9999) weighttype(rademacher) ptolerance(1e-14) nograph
_open boottest_d_h02
_num p r(p)
_num t r(t)
_num reps r(reps) last
_close

* Subcluster bootstrap: CRVE clustered at g6 (6 clusters), signs flipped at
* s12 (12 bootstrap clusters nested in g6).
quietly regress y d x1 x2, vce(cluster g6)
boottest d, reps(9999) weighttype(rademacher) bootcluster(s12) ptolerance(1e-14) nograph
_open boottest_sub_d
_num p r(p)
_num t r(t)
_num reps r(reps) last
_close

* ---------------------------------------------------------------------------
* 3. ritest over the FULL assignment set of each design (samplingsourcefile:
*    one column d1..dN per assignment, written by _generate_inference_sens_data.py),
*    so c/N is the exact randomization p-value. ritest counts
*    |T| >= |T(obs)| - eps (eps = 1e-7) for the two-sided p and
*    T >= T(obs) - eps with option right.
* ---------------------------------------------------------------------------
capture program drop _ri
program define _ri
    * _ri <block> <design> <N> <exp> : <command>
    gettoken name 0 : 0
    gettoken design 0 : 0
    gettoken N 0 : 0
    gettoken exp 0 : 0, parse(":")
    gettoken colon cmd : 0, parse(":")
    import delimited "inference_sens_ri_`design'_perms.csv", clear
    tempfile perms
    save `perms'
    import delimited "inference_sens_ri_`design'.csv", clear asdouble
    gen id = _n
    quietly ritest d T=(`exp'), samplingsourcefile(`perms') samplingmatchvar(id) reps(`N') nodots: `cmd'
    matrix B = r(b)
    matrix P = r(p)
    matrix R = r(reps)
    scalar _obs = B[1,1]
    scalar _p2 = P[1,1]
    scalar _reps = R[1,1]
    quietly ritest d T=(`exp'), samplingsourcefile(`perms') samplingmatchvar(id) reps(`N') right nodots: `cmd'
    matrix P = r(p)
    _open `name'
    _num observed _obs
    _num p_two _p2
    _num p_upper P[1,1]
    _num reps _reps last
    _close
end

* ritest stores the estimates of the command it runs (est store), so the
* r-class statistics are posted through a small e-class wrapper. The signs of
* ttest's t and ranksum's z are flipped: both contrast group d = 0 minus d = 1.
capture program drop _estat
program define _estat, eclass
    args kind
    if "`kind'" == "t" {
        quietly ttest y, by(d) unequal
        local v = -r(t)
    }
    else if "`kind'" == "ks" {
        quietly ksmirnov y, by(d)
        local v = r(D)
    }
    else if "`kind'" == "ranksum" {
        quietly ranksum y, by(d)
        local v = -r(z)
    }
    tempname b
    matrix `b' = (`v')
    matrix colnames `b' = T
    ereturn post `b'
    ereturn local cmd "_estat"
end

_ri ri_simple_diff simple 924 _b[d] : regress y d
_ri ri_simple_t simple 924 _b[T] : _estat t
_ri ri_simple_ks simple 924 _b[T] : _estat ks
_ri ri_simple_ranksum simple 924 _b[T] : _estat ranksum
_ri ri_cluster_diff cluster 70 _b[d] : regress y d
_ri ri_strat_diff strat 4900 _b[d] : regress y d

* ---------------------------------------------------------------------------
* 4. rbounds (Gangl): Wilcoxon signed-rank Rosenbaum bounds on the paired
*    differences. sig+ / sig- are the upper / lower bound significance levels
*    (zeros ranked with the other pairs and weighted 0; no continuity
*    correction; 1 - normprob(sign(mean diff) * z)).
* ---------------------------------------------------------------------------
import delimited "inference_sens_pairs.csv", clear asdouble
quietly rbounds diff, gamma(1.5 2 2.5 3) sigonly
matrix O = r(outmat)
_open rbounds
forvalues i = 1/5 {
    _num gamma_`i' O[`i',1]
    _num sig_plus_`i' O[`i',2]
    local last = cond(`i' == 5, "last", "")
    _num sig_minus_`i' O[`i',3] `last'
}
_close

* ---------------------------------------------------------------------------
* 5. psacalc (Oster's own implementation of Oster 2019): exact delta for
*    beta = 0 and exact beta*(delta) (quadratic at delta = 1, cubic otherwise),
*    treatment t, controls x1 x2, on inference_sens_reg.csv. R_max is fixed
*    at 1.3 x the controlled R-squared (Oster's recommendation) and at 1
*    (psacalc's default).
* ---------------------------------------------------------------------------
import delimited "inference_sens_reg.csv", clear asdouble
quietly regress y t x1 x2
scalar _rt = e(r2)
scalar _rm13 = min(1, 1.3 * _rt)
_open psacalc
_num r_tilde _rt
_num rmax13 _rm13
quietly psacalc delta t, rmax(`=_rm13')
_num delta_rm13 r(delta)
quietly psacalc delta t, rmax(1)
_num delta_rm1 r(delta)
quietly psacalc delta t, rmax(`=_rm13') beta(0.3)
_num delta_rm13_beta03 r(delta)
quietly psacalc beta t, rmax(`=_rm13') delta(1)
_num beta_rm13_d1 r(beta)
_num beta_rm13_d1_alt1 r(altsol1)
quietly psacalc beta t, rmax(`=_rm13') delta(0.5)
_num beta_rm13_d05 r(beta)
quietly psacalc beta t, rmax(`=_rm13') delta(2)
_num beta_rm13_d2 r(beta)
quietly psacalc beta t, rmax(1) delta(1)
_num beta_rm1_d1 r(beta)
quietly psacalc delta t, rmax(`=_rm13') mcontrol(x2)
_num delta_rm13_mc_x2 r(delta)
quietly psacalc beta t, rmax(`=_rm13') delta(1) mcontrol(x2)
_num beta_rm13_d1_mc_x2 r(beta) last
_close

file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
foreach p in boottest ritest rbounds psacalc {
    quietly findfile `p'.ado
    file open vf using "`r(fn)'", read text
    file read vf line
    file close vf
    local line = subinstr(`"`line'"', `"""', "", .)
    file write fh `"    "`p'": "' `"""' `"`line'"' `"""' `","' _n
}
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
