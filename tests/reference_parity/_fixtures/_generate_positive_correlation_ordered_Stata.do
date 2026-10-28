* Reference for sp.positive_correlation_test(method='residuals') with ordered
* outcomes: Stata 18 oprobit / probit on positive_correlation_ordered_data.csv
* (written by _generate_positive_correlation_ordered_data.py). The ordered
* probit generalized residual (normalden(l) - normalden(u)) /
* (normal(u) - normal(l)), with (l, u] = (cut_{j-1} - xb, cut_j - xb], is
* built from e(b); the score is sum e1 e2 and W = score^2 / sum (e1 e2)^2
* (cluster sums with g). Every fit runs at tight tolerances: at the default
* the probit stops 1e-7 (relative) short of the maximum, which moves the
* score variance by 2e-8. Three cases:
*   ord_bin    : oprobit cover x1 x2 ; probit claim x1 x3
*   ord_ord    : oprobit cover x1 x2 ; oprobit nclaim x1 x3
*   ord_ord_cl : as ord_ord, cluster g
* Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_positive_correlation_ordered_Stata.do
version 18
set type double
clear all
local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-14)"
import delimited "tests/reference_parity/_fixtures/positive_correlation_ordered_data.csv", clear asdouble

capture program drop _ores
program define _ores
    * generalized residual of the last oprobit, outcome coded 0..J-1
    args y out
    tempvar xb lo hi
    predict double `xb', xb
    local J = e(k_cat)
    gen double `lo' = .
    gen double `hi' = .
    forvalues j = 0/`=`J'-1' {
        if `j' > 0 {
            replace `lo' = _b[/cut`j'] - `xb' if `y' == `j'
        }
        if `j' < `J' - 1 {
            replace `hi' = _b[/cut`=`j'+1'] - `xb' if `y' == `j'
        }
    }
    gen double `out' = (cond(missing(`lo'), 0, normalden(`lo')) ///
        - cond(missing(`hi'), 0, normalden(`hi'))) ///
        / (cond(missing(`hi'), 1, normal(`hi')) - cond(missing(`lo'), 0, normal(`lo')))
end

oprobit cover x1 x2, `tight'
local ll_cover = e(ll)
local b_cover_x1 = _b[x1]
local cut2_cover = _b[/cut2]
_ores cover e_cover

probit claim x1 x3, `tight'
predict double xb_claim, xb
gen double e_claim = cond(claim == 1, normalden(xb_claim) / normal(xb_claim), ///
    -normalden(xb_claim) / normal(-xb_claim))

oprobit nclaim x1 x3, `tight'
local ll_nclaim = e(ll)
_ores nclaim e_nclaim

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/positive_correlation_ordered_Stata.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "oprobit_cover": {"ll": "' %24.17e (`ll_cover') `", "b_x1": "' %24.17e (`b_cover_x1') `", "cut2": "' %24.17e (`cut2_cover') `"},"' _n

foreach case in ord_bin ord_ord ord_ord_cl {
    local e2 = cond("`case'" == "ord_bin", "e_claim", "e_nclaim")
    capture drop s
    gen double s = e_cover * `e2'
    quietly summarize s
    local score = r(sum)
    if "`case'" == "ord_ord_cl" {
        preserve
        collapse (sum) s, by(g)
        gen double s2 = s^2
        quietly summarize s2
        local var = r(sum)
        restore
    }
    else {
        capture drop s2
        gen double s2 = s^2
        quietly summarize s2
        local var = r(sum)
    }
    local W = `score'^2 / `var'
    quietly correlate e_cover `e2'
    local rho = r(rho)
    local sep = cond("`case'" == "ord_ord_cl", "", ",")
    file write `fh' `"  "`case'": {"score": "' %24.17e (`score') `", "score_variance": "' %24.17e (`var') `", "W": "' %24.17e (`W') `", "corr": "' %24.17e (`rho') `", "N": "' (_N) `"}`sep'"' _n
}
file write `fh' "}" _n
file close `fh'
