* Stata 18 MP reference for tests/test_xttobit_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_xttobit.do
* then python ../_stata_tsv.py stata_xttobit.tsv stata_xttobit.json
set more off
set type double
import delimited using "retobit.csv", clear asdouble
xtset id t
file open fh using "stata_xttobit.tsv", write replace
file write fh "about" _tab "_meta" _tab "stata_version" _tab "`c(stata_version)'" _n
file write fh "about" _tab "_meta" _tab "born_date" _tab "`c(born_date)'" _n
file write fh "about" _tab "_meta" _tab "edition" _tab "`c(edition_real)'" _n
capture program drop dumpe
program dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
    foreach s in sigma_u sigma_e rho ll ll_c chi2_c N N_g {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
foreach q in 12 40 {
    xttobit y x1 x2, ll(0) ul(4) intpoints(`q')
    dumpe tobit_q`q'
    xtintreg lo hi x1 x2, intpoints(`q')
    dumpe intreg_q`q'
}
xttobit y x1 x2, ll(0) intpoints(40)
dumpe tobit_ll_q40
* xtintreg's analytic (d2) Hessian under adaptive quadrature disagrees with
* the curvature of its own likelihood (finite differences of e(ll) at
* from() iterate(0) reproduce our standard errors); with non-adaptive
* quadrature the reported standard errors equal that curvature
xtintreg lo hi x1 x2, intmethod(ghermite) intpoints(100)
dumpe intreg_gh100
* perfectly collinear regressors: Stata omits the later member of the
* dependent set ("note: x3 omitted because of collinearity") and reports
* the fit without it
gen double x3 = x1 + x2
gen double x4 = 2*x1
xttobit y x1 x2 x3, ll(0) ul(4) intpoints(40)
dumpe tobit_col_q40
xtintreg lo hi x1 x4 x2, intpoints(40)
dumpe intreg_col_q40
* added 2026-11-05: an upper limit only, and right-censored intervals
xttobit y x1 x2, ul(4) intpoints(40)
dumpe tobit_ul_q40
xtintreg lo hi2 x1 x2, intpoints(40)
dumpe intreg_rc_q40
* that default fit stops after four Newton steps, short of the maximum (its
* e(ll) is 4.8e-8 below the one of the next two fits); tightened, it is not
xtintreg lo hi2 x1 x2, intpoints(40) nrtolerance(1e-12) tolerance(1e-12) ltolerance(1e-14)
dumpe intreg_rc_q40_tight
xtintreg lo hi2 x1 x2, intmethod(ghermite) intpoints(100)
dumpe intreg_rc_gh100
file close fh
