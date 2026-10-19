* Stata 18 MP reference for tests/test_xtpcse_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_xtpcse.do
* (the tab-separated output is stored as stata_xtpcse.json)
set more off
set type double
file open fh using "stata_xtpcse.tsv", write replace
capture program drop dumpe
program dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
    capture confirm matrix e(rho)
    if !_rc {
        tempname r
        matrix `r' = e(rho)
        forvalues j = 1/`=colsof(`r')' {
            file write fh "`tag'" _tab "rho`j'" _tab %24.17g (`r'[1,`j']) _tab "0" _n
        }
    }
    else {
        capture file write fh "`tag'" _tab "rho" _tab %24.17g (e(rho)) _tab "0" _n
    }
    file write fh "`tag'" _tab "N" _tab %24.17g (e(N)) _tab "0" _n
end
foreach f in tscs tscs_unbal {
    import delimited using "`f'.csv", clear asdouble
    xtset id t
    xtpcse y x1 x2
    dumpe `f'_default
    xtpcse y x1 x2, hetonly
    dumpe `f'_hetonly
    xtpcse y x1 x2, independent
    dumpe `f'_independent
    xtpcse y x1 x2, nmk
    dumpe `f'_nmk
    xtpcse y x1 x2, correlation(ar1)
    dumpe `f'_ar1
    xtpcse y x1 x2, correlation(psar1)
    dumpe `f'_psar1
    xtpcse y x1 x2, correlation(ar1) rhotype(dw)
    dumpe `f'_ar1_dw
    xtpcse y x1 x2, correlation(ar1) rhotype(tscorr)
    dumpe `f'_ar1_tscorr
    xtpcse y x1 x2, correlation(ar1) rhotype(freg)
    dumpe `f'_ar1_freg
    xtpcse y x1 x2, correlation(ar1) hetonly
    dumpe `f'_ar1_hetonly
}
xtpcse y x1 x2, pairwise
dumpe tscs_unbal_pairwise
xtpcse y x1 x2, correlation(ar1) np1
dumpe tscs_unbal_ar1_np1
xtpcse y x1 x2, correlation(ar1) pairwise
dumpe tscs_unbal_ar1_pairwise
file close fh
