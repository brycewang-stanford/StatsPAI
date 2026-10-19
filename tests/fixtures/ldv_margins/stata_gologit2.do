* Stata 18 MP + gologit2 3.2.8 (SSC, Richard Williams) reference for
* tests/test_gologit_stata.py, on the rows of _data.py.
* gologit2 is installed into a private directory, not the user's PLUS:
*   sysdir set PLUS "<scratch>/ado"
*   ssc install gologit2
* Then, from this directory: python _data.py . && stata-mp -b do stata_gologit2.do
set more off
set type double
capture program drop dump dumpe
program dump
    args tag
    tempname b V
    matrix `b' = r(b)
    matrix `V' = r(V)
    local names : colfullnames `b'
    forvalues j = 1/`=colsof(`b')' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
end
program dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    forvalues j = 1/`=colsof(`b')' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
end
import delimited using "ldv.csv", clear asdouble
file open fh using "stata_gologit2.tsv", write replace
gologit2 yo x1 x2 g2 g3
dumpe npl
gologit2 yo x1 x2 g2 g3, pl
dumpe pl
gologit2 yo x1 x2 g2 g3, npl(x1)
dumpe npl_x1
margins, dydx(*)
dump npl_x1_ame
gologit2 yo x1 x2 g2 g3, pl(x2 g2)
dumpe pl_x2g2
gologit2 yo x1 x2 g2 g3, link(p) npl(x1)
dumpe probit_npl_x1
gologit2 yo x1 x2 g2 g3, npl(x1) vce(robust)
dumpe npl_x1_robust
file close fh
