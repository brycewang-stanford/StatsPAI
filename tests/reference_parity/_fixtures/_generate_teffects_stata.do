* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_teffects_R_parity.py
* (Stata-side block). Stata 18, `set type double`. User-written commands are
* installed into a PRIVATE ado directory next to this file -- never PLUS:
*
*   _ado_teffects/      leebounds (SSC, Tauchmann v1.5), tebounds (SJ15-2
*                       st0386), med4way (GitHub anddis/med4way),
*                       doseresponse + gpscore (SSC)
*   _ado_teffects_patched/  a copy of leebounds.ado written by this script
*                       (see block 4); created fresh on every run.
*
* The do-file installs the packages itself when they are missing. Run
* _generate_teffects_data.py first, then, from this directory:
*   stata-mp -b do _generate_teffects_stata.do
* Writes teffects_Stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * teffects aipw: logit (mlogit for a multivalued treatment) propensity,
*   per-arm linear outcome by ML; the robust SE is the stacked
*   M-estimation sandwich (divisor n).
* * leebounds: default vce(analytic); cieffect gives the Imbens-Manski
*   interval, whose critical value leebounds finds on a 0.001 grid.
*   leebounds stores each trimming threshold in a local macro
*   (`local uth = r(r1)`), which keeps ~15 significant digits. For a
*   continuous outcome the threshold then fails the `y == threshold` test
*   and the tie branch (fractional trimming) never runs; which side of the
*   rounded threshold the quantile observation lands on decides whether it
*   is kept. Block 4 re-runs leebounds with the thresholds held exactly
*   (%21x hex), which is the behaviour the code is written for.
* * tebounds: binary outcome, erates(0) (no misclassification); worst
*   case (j=2), MTS positive selection (j=3, "p"), MTS+MTR positive (j=5).
* * med4way: yreg(linear) mreg(linear), a0(0) a1(1) m(0), covariates at
*   their mean (stored by med4way with ~15 digits); delta-method V from
*   the ML fits.
* * doseresponse: normal GPS by ML (sigma^2 = RSS/N), outcome quadratic in
*   T and GPS with interaction; DRF at tpoints 0, .5, 1, 1.5, 2.
* * MSM: untruncated stabilised weights from two logits, cumulative product
*   within id; regress / logit with [pw], vce(cluster id). regress uses
*   G/(G-1) (N-1)/(N-k); logit uses G/(G-1) only.
* ---------------------------------------------------------------------------
version 18
clear all
set more off
set type double

local here "`c(pwd)'"
local ado "`here'/_ado_teffects"
local adop "`here'/_ado_teffects_patched"
capture mkdir "`ado'"
net set ado "`ado'"
adopath ++ "`ado'"
capture which leebounds
if _rc ssc install leebounds, replace
capture which tebounds
if _rc net install st0386, from(http://www.stata-journal.com/software/sj15-2) replace
capture which med4way
if _rc net install med4way, from("https://raw.githubusercontent.com/anddis/med4way/master/") replace
capture which doseresponse
if _rc ssc install doseresponse, replace

tempname fh
file open `fh' using "teffects_Stata.json", write replace text
file write `fh' "{" _n

* ---- 1. teffects aipw, binary treatment -----------------------------------
import delimited using "teffects_cs.csv", clear asdouble
quietly teffects aipw (y x1 x2 x3) (d x1 x2 x3), ate
local ate = _b[ATE:r1vs0.d]
local ate_se = _se[ATE:r1vs0.d]
quietly teffects aipw (y x1 x2 x3) (d x1 x2 x3), pomeans
file write `fh' `"  "aipw": {"ate": "' %23.16e (`ate') `", "ate_se": "' %23.16e (`ate_se') ", "
file write `fh' `""po0": "' %23.16e (_b[POmeans:0.d]) `", "po0_se": "' %23.16e (_se[POmeans:0.d]) ", "
file write `fh' `""po1": "' %23.16e (_b[POmeans:1.d]) `", "po1_se": "' %23.16e (_se[POmeans:1.d]) "}," _n

* ---- 2. teffects aipw, multivalued treatment ------------------------------
quietly teffects aipw (ymt x1 x2) (t3 x1 x2), ate
local a1 = _b[ATE:r1vs0.t3]
local s1 = _se[ATE:r1vs0.t3]
local a2 = _b[ATE:r2vs0.t3]
local s2 = _se[ATE:r2vs0.t3]
quietly teffects aipw (ymt x1 x2) (t3 x1 x2), pomeans
file write `fh' `"  "aipw_multi": {"ate1": "' %23.16e (`a1') `", "ate1_se": "' %23.16e (`s1') ", "
file write `fh' `""ate2": "' %23.16e (`a2') `", "ate2_se": "' %23.16e (`s2') ", "
file write `fh' `""po": ["' %23.16e (_b[POmeans:0.t3]) ", " %23.16e (_b[POmeans:1.t3]) ", " %23.16e (_b[POmeans:2.t3]) "], "
file write `fh' `""po_se": ["' %23.16e (_se[POmeans:0.t3]) ", " %23.16e (_se[POmeans:1.t3]) ", " %23.16e (_se[POmeans:2.t3]) "]}," _n

* ---- 3. leebounds as shipped ---------------------------------------------
gen dflip = 1 - d
foreach spec in d dflip {
    quietly leebounds ys `spec', select(s) cieffect
    matrix b = e(b)
    matrix V = e(V)
    file write `fh' `"  "leebounds_`spec'": {"lower": "' %23.16e (b[1,1]) `", "upper": "' %23.16e (b[1,2]) ", "
    file write `fh' `""V_lower": "' %23.16e (V[1,1]) `", "V_upper": "' %23.16e (V[2,2]) ", "
    file write `fh' `""ci_lower": "' %23.16e (e(cilower)) `", "ci_upper": "' %23.16e (e(ciupper)) ", "
    local trimmed_control = ("`e(trimmed)'" == "control")
    file write `fh' `""trim": "' %23.16e (e(trim)) `", "trimmed_control": `trimmed_control'}"' "," _n
}

* ---- 4. leebounds with the trimming thresholds held exactly --------------
* A copy of leebounds.ado whose four `local uth|lth = r(r1)` lines store
* the percentile as a %21x hex literal (exact double) instead of a
* ~15-digit decimal, so its `y == threshold` tie test sees the quantile.
capture mkdir "`adop'"
capture mkdir "`adop'/l"
tempfile p1
filefilter "`ado'/l/leebounds.ado" "`p1'", from("local uth = r(r1)") to("local uth : display %21x r(r1)") replace
filefilter "`p1'" "`adop'/l/leebounds.ado", from("local lth = r(r1)") to("local lth : display %21x r(r1)") replace
adopath - "`ado'"
adopath ++ "`adop'"
discard
quietly leebounds ys d, select(s)
matrix b = e(b)
matrix V = e(V)
file write `fh' `"  "leebounds_exact": {"lower": "' %23.16e (b[1,1]) `", "upper": "' %23.16e (b[1,2]) ", "
file write `fh' `""V_lower": "' %23.16e (V[1,1]) `", "V_upper": "' %23.16e (V[2,2]) "}," _n
adopath - "`adop'"
adopath ++ "`ado'"
discard

* ---- 5. tebounds (binary outcome, no misclassification) ------------------
quietly tebounds yb, treat(d) erates(0)
file write `fh' `"  "tebounds": {"worst": ["' %23.16e (e(lb21_ate_0)) ", " %23.16e (e(ub21_ate_0)) "], "
file write `fh' `""mts": ["' %23.16e (e(lb31p_ate_0)) ", " %23.16e (e(ub31p_ate_0)) "], "
file write `fh' `""mts_mtr": ["' %23.16e (e(lb51p_ate_0)) ", " %23.16e (e(ub51p_ate_0)) "]}," _n

* ---- 6. med4way (linear / linear) ----------------------------------------
quietly med4way ym d m x1, a0(0) a1(1) m(0) yreg(linear) mreg(linear)
matrix b = e(b)
matrix V = e(V)
file write `fh' `"  "med4way": {"b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) ", " %23.16e (b[1,3]) ", " %23.16e (b[1,4]) ", " %23.16e (b[1,5]) "], "
file write `fh' `""V_diag": ["' %23.16e (V[1,1]) ", " %23.16e (V[2,2]) ", " %23.16e (V[3,3]) ", " %23.16e (V[4,4]) ", " %23.16e (V[5,5]) "], "
file write `fh' `""order": "te cde intref intmed pie"},"' _n

* ---- 7. doseresponse (Hirano-Imbens GPS) ---------------------------------
xtile cut = tc, nq(3)
matrix tp = (0\0.5\1\1.5\2)
tempfile drout
quietly doseresponse x1 x2, outcome(ydose) t(tc) gpscore(gps) predict(hat) ///
    sigma(sd) cutpoints(cut) index(p50) nq_gps(5) dose_response(dr) ///
    tpoints(tp) reg_type_t(quadratic) reg_type_gps(quadratic) ///
    interaction(1) filename("`drout'")
matrix b = e(b)
quietly summarize sd
local sig = r(mean)
file write `fh' `"  "doseresponse": {"sigma": "' %23.16e (`sig') ", "
file write `fh' `""outcome_b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) ", " %23.16e (b[1,3]) ", " %23.16e (b[1,4]) ", " %23.16e (b[1,5]) ", " %23.16e (b[1,6]) "], "
file write `fh' `""outcome_b_order": "tc tc_sq gps gps_sq tc_gps _cons", "' 
preserve
use "`drout'", clear
sort treatment_level
file write `fh' `""doses": ["' %23.16e (treatment_level[1]) ", " %23.16e (treatment_level[2]) ", " %23.16e (treatment_level[3]) ", " %23.16e (treatment_level[4]) ", " %23.16e (treatment_level[5]) "], "
file write `fh' `""drf": ["' %23.16e (dr[1]) ", " %23.16e (dr[2]) ", " %23.16e (dr[3]) ", " %23.16e (dr[4]) ", " %23.16e (dr[5]) "]}," _n
restore

* ---- 8. MSM final stage (untruncated stabilised weights) -----------------
import delimited using "teffects_long.csv", clear asdouble
sort id t
by id: gen double a_lag = cond(_n == 1, 0, a[_n-1])
by id: gen double cum_a = sum(a)
quietly logit a a_lag v l
predict double pden, pr
quietly logit a a_lag v
predict double pnum, pr
gen double ratio = (a * pnum + (1 - a) * (1 - pnum)) / (a * pden + (1 - a) * (1 - pden))
by id: gen double sw = ratio if _n == 1
by id: replace sw = sw[_n-1] * ratio if _n > 1
quietly regress y cum_a v [pw = sw], vce(cluster id)
file write `fh' `"  "msm": {"coef": ["' %23.16e (_b[_cons]) ", " %23.16e (_b[cum_a]) ", " %23.16e (_b[v]) "], "
file write `fh' `""se": ["' %23.16e (_se[_cons]) ", " %23.16e (_se[cum_a]) ", " %23.16e (_se[v]) "], "
quietly logit yb cum_a v [pw = sw], vce(cluster id)
file write `fh' `""logit_coef": ["' %23.16e (_b[_cons]) ", " %23.16e (_b[cum_a]) ", " %23.16e (_b[v]) "], "
file write `fh' `""logit_se": ["' %23.16e (_se[_cons]) ", " %23.16e (_se[cum_a]) ", " %23.16e (_se[v]) "], "
file write `fh' `""N": "' %23.16e (e(N)) `", "N_clust": "' %23.16e (e(N_clust)) "}," _n

file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "leebounds": "1.5 (2013-07-17, Tauchmann)", "tebounds": "1.8 (SJ15-2 st0386)", "med4way": "GitHub anddis/med4way", "doseresponse": "SSC"}"' _n
file write `fh' "}" _n
file close `fh'
