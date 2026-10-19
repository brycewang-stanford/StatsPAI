* Stata 18 MP reference for tests/test_xttobit_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_xttobit.do
* (the tab-separated output is stored as stata_xttobit.json)
set more off
set type double
import delimited using "retobit.csv", clear asdouble
xtset id t
file open fh using "stata_xttobit.tsv", write replace
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
file close fh
