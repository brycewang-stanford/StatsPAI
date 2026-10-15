* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_psm_options_parity.py:
* the option grid of sp.match (propensity-score and covariate matching).
*
* Run _generate_psm_options_data.py first (writes psm_options_data.csv),
* then, from this directory,
*   stata-mp -b do _generate_psm_options_stata.do
* Writes psm_options_Stata.json (%23.16e, 17 significant digits).
*
* User commands are installed into a PRIVATE ado tree next to this file
* (_ado_psm_options/, git-ignored by the _ado_*/ rule) -- never PLUS:
*   * psmatch2 (Leuven & Sianesi): ssc install psmatch2
*   * atts (Becker & Ichino, Stata Journal st0026_2):
*       net install st0026_2, from(http://www.stata-journal.com/software/sj5-3)
* teffects psmatch / nnmatch are official Stata 18.
*
* Every command is run on the same 500 rows; the data have no ties in the
* propensity score or in any covariate distance, so the comparisons are
* about the options and not about tie-breaking. Each entry records the
* command, the estimate and its standard error.
*
* psmatch2 always gets `logit` (its default is probit) and the outcome in
* outcome(); its estimate is r(att), its standard error r(seatt). teffects
* reports _b / _se of r1vs0.d.
* ---------------------------------------------------------------------------
version 18
set more off
set type double
local A "`c(pwd)'/_ado_psm_options"
capture mkdir "`A'"
sysdir set PLUS "`A'"
adopath ++ "`A'"
capture which psmatch2
if _rc ssc install psmatch2
capture which atts
if _rc net install st0026_2, from(http://www.stata-journal.com/software/sj5-3)

import delimited using "psm_options_data.csv", clear asdouble
local X "x1 x2 x3 x4"
sort id
tempfile base
save `base'

file open fh using "psm_options_Stata.json", write replace text
file write fh "{" _n

capture program drop put
program define put
    args key cmd est se
    file write fh `"  "`key'": {"cmd": "`cmd'", "est": "' %23.16e (`est') `", "se": "' %23.16e (`se') `"},"' _n
end

* ------------------------------------------------------------- psmatch2 ----
* nearest neighbour k = 1..10, Abadie-Imbens (2006) SE with ai(1)
forvalues k = 1/10 {
    use `base', clear
    quietly psmatch2 d `X', outcome(y) logit neighbor(`k') ai(1)
    put "psmatch2_nn`k'_ai1" "psmatch2 d X, outcome(y) logit neighbor(`k') ai(1)" r(att) r(seatt)
}
* nearest neighbour, ai(2)
use `base', clear
quietly psmatch2 d `X', outcome(y) logit neighbor(1) ai(2)
put "psmatch2_nn1_ai2" "psmatch2 d X, outcome(y) logit neighbor(1) ai(2)" r(att) r(seatt)
* nearest neighbour, psmatch2's own (homoskedastic) SE
use `base', clear
quietly psmatch2 d `X', outcome(y) logit neighbor(1)
put "psmatch2_nn1" "psmatch2 d X, outcome(y) logit neighbor(1)" r(att) r(seatt)

* caliper: five treated units have no control within 0.01
foreach c in 0.01 0.005 {
    local tag = subinstr("`c'", ".", "p", .)
    use `base', clear
    quietly psmatch2 d `X', outcome(y) logit neighbor(1) caliper(`c')
    put "psmatch2_nn1_cal`tag'" "psmatch2 d X, outcome(y) logit neighbor(1) caliper(`c')" r(att) r(seatt)
    quietly count if _treated == 1 & _support == 1
    file write fh `"  "psmatch2_nn1_cal`tag'_n_on_support": "' %12.0f (r(N)) "," _n
    use `base', clear
    quietly psmatch2 d `X', outcome(y) logit neighbor(1) caliper(`c') ai(1)
    put "psmatch2_nn1_cal`tag'_ai1" "psmatch2 d X, outcome(y) logit neighbor(1) caliper(`c') ai(1)" r(att) r(seatt)
}

* without replacement (psmatch2 matches the treated in ascending score order)
use `base', clear
quietly psmatch2 d `X', outcome(y) logit neighbor(1) noreplacement
put "psmatch2_nn1_norepl" "psmatch2 d X, outcome(y) logit neighbor(1) noreplacement" r(att) r(seatt)
use `base', clear
quietly psmatch2 d `X', outcome(y) logit neighbor(1) noreplacement ai(1)
put "psmatch2_nn1_norepl_ai1" "psmatch2 d X, outcome(y) logit neighbor(1) noreplacement ai(1)" r(att) r(seatt)

* Mahalanobis distance (pooled within-arm covariance)
foreach k in 1 3 {
    use `base', clear
    quietly psmatch2 d, mahalanobis(`X') outcome(y) neighbor(`k') ai(1)
    put "psmatch2_mahal_nn`k'_ai1" "psmatch2 d, mahalanobis(X) outcome(y) neighbor(`k') ai(1)" r(att) r(seatt)
}

* kernel matching, the five kernels at the default bandwidth and one other
foreach kt in epan normal biweight uniform tricube {
    use `base', clear
    quietly psmatch2 d `X', outcome(y) logit kernel kerneltype(`kt') bwidth(0.06)
    put "psmatch2_kernel_`kt'_bw0p06" "psmatch2 d X, outcome(y) logit kernel kerneltype(`kt') bwidth(0.06)" r(att) r(seatt)
}
use `base', clear
quietly psmatch2 d `X', outcome(y) logit kernel kerneltype(epan) bwidth(0.02)
put "psmatch2_kernel_epan_bw0p02" "psmatch2 d X, outcome(y) logit kernel kerneltype(epan) bwidth(0.02)" r(att) r(seatt)

* radius matching
foreach c in 0.02 0.05 {
    local tag = subinstr("`c'", ".", "p", .)
    use `base', clear
    quietly psmatch2 d `X', outcome(y) logit radius caliper(`c')
    put "psmatch2_radius_cal`tag'" "psmatch2 d X, outcome(y) logit radius caliper(`c')" r(att) r(seatt)
}

* ---------------------------------------------------- teffects psmatch ----
* Abadie-Imbens (2016) estimated-score SE, k = 1..10 matches
forvalues k = 1/10 {
    use `base', clear
    quietly teffects psmatch (y) (d `X', logit), atet nneighbor(`k')
    put "teffects_psmatch_atet_nn`k'" "teffects psmatch (y) (d X, logit), atet nneighbor(`k')" _b[r1vs0.d] _se[r1vs0.d]
}
use `base', clear
quietly teffects psmatch (y) (d `X', logit), ate nneighbor(3)
put "teffects_psmatch_ate_nn3" "teffects psmatch (y) (d X, logit), ate nneighbor(3)" _b[r1vs0.d] _se[r1vs0.d]
* caliper() is a condition here too (every observation needs its matches,
* and its same-arm nn(2) set, within it): 0.2 is met, 0.1 is not.
foreach est in atet ate {
    foreach c in 0.2 0.1 {
        local tag = subinstr("`c'", ".", "p", .)
        use `base', clear
        capture noisily teffects psmatch (y) (d `X', logit), `est' caliper(`c')
        if _rc == 0 {
            put "teffects_psmatch_`est'_nn1_cal`tag'" "teffects psmatch (y) (d X, logit), `est' caliper(`c')" _b[r1vs0.d] _se[r1vs0.d]
        }
        else {
            file write fh `"  "teffects_psmatch_`est'_nn1_cal`tag'_rc": "' %6.0f (_rc) "," _n
        }
    }
}

* ---------------------------------------------------- teffects nnmatch ----
* default metric (mahalanobis), vce(robust, nn(2)) unless stated
foreach k in 1 4 5 {
    use `base', clear
    quietly teffects nnmatch (y `X') (d), atet nneighbor(`k')
    put "teffects_nnmatch_atet_nn`k'" "teffects nnmatch (y X) (d), atet nneighbor(`k')" _b[r1vs0.d] _se[r1vs0.d]
}
foreach k in 4 5 {
    use `base', clear
    quietly teffects nnmatch (y `X') (d), ate nneighbor(`k')
    put "teffects_nnmatch_ate_nn`k'" "teffects nnmatch (y X) (d), ate nneighbor(`k')" _b[r1vs0.d] _se[r1vs0.d]
}
use `base', clear
quietly teffects nnmatch (y `X') (d), atet vce(robust, nn(3))
put "teffects_nnmatch_atet_vnn3" "teffects nnmatch (y X) (d), atet vce(robust, nn(3))" _b[r1vs0.d] _se[r1vs0.d]
use `base', clear
quietly teffects nnmatch (y `X') (d), ate nneighbor(4) vce(robust, nn(3))
put "teffects_nnmatch_ate_nn4_vnn3" "teffects nnmatch (y X) (d), ate nneighbor(4) vce(robust, nn(3))" _b[r1vs0.d] _se[r1vs0.d]
* teffects reads caliper() as a condition: every observation (either arm)
* needs nneighbor() matches within it, or the command stops (r(459)). So a
* caliper changes nothing or refuses. 3 is met by every unit; with 2, seven
* units have fewer than four matches and one control has none.
foreach c in 3 2 {
    use `base', clear
    capture noisily teffects nnmatch (y `X') (d), atet nneighbor(4) caliper(`c')
    if _rc == 0 {
        put "teffects_nnmatch_atet_nn4_cal`c'" "teffects nnmatch (y X) (d), atet nneighbor(4) caliper(`c')" _b[r1vs0.d] _se[r1vs0.d]
    }
    else {
        file write fh `"  "teffects_nnmatch_atet_nn4_cal`c'_rc": "' %6.0f (_rc) "," _n
    }
}
use `base', clear
quietly teffects nnmatch (y `X') (d), ate nneighbor(1) caliper(3)
put "teffects_nnmatch_ate_nn1_cal3" "teffects nnmatch (y X) (d), ate nneighbor(1) caliper(3)" _b[r1vs0.d] _se[r1vs0.d]

* ------------------------------------------------------------- atts ----
* Propensity-score stratification. The strata are sp.match's default:
* five quantile classes of the logit score over the whole sample, with
* numpy's default (linear-interpolation) quantiles and classes closed on
* the left. The score is psmatch2's logit score (identical to logit's).
use `base', clear
quietly logit d `X'
predict double ps, pr
mata:
p = st_data(., "ps")
s = sort(p, 1)
n = rows(s)
b = J(6, 1, .)
for (k = 0; k <= 5; k++) {
    h = (n - 1) * k / 5
    lo = floor(h)
    b[k + 1] = s[lo + 1] + (h - lo) * (lo + 1 < n ? s[lo + 2] - s[lo + 1] : 0)
}
b[1] = b[1] - 1e-6
b[6] = b[6] + 1e-6
st_store(., st_addvar("double", "block"), rowsum(J(n, 1, b') :<= p) :- 1)
end
* atts numbers its blocks 1, 2, ...
replace block = min(max(block, 0), 4) + 1
quietly atts y d, pscore(ps) blockid(block)
put "atts_quintiles" "atts y d, pscore(ps) blockid(block), five quantile classes" r(atts) r(seatts)

file write fh `"  "_meta": {"stata_version": "`c(stata_version)'", "precision": "double", "n": 500}"' _n
file write fh "}" _n
file close fh
