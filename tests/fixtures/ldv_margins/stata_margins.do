* Stata 18 MP reference for tests/test_ldv_margins_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_margins.do
* then python _to_json.py (writes stata_margins.json).
set more off
set type double
capture program drop dump
capture program drop dumpe
program dump
    args tag
    tempname b V
    matrix `b' = r(b)
    matrix `V' = r(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
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
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
end

import delimited using "ldv.csv", clear asdouble
replace g2 = g==2
replace g3 = g==3
file open fh using "stata_margins.tsv", write replace
mlogit ym x1 x2 i.g, base(0)
dumpe mlogit_e
margins, dydx(*)
dump mlogit_ame
margins, dydx(*) atmeans
dump mlogit_mem
ologit yo x1 x2 i.g
dumpe ologit_e
margins, dydx(*)
dump ologit_ame
oprobit yo x1 x2 i.g
dumpe oprobit_e
margins, dydx(*)
dump oprobit_ame
tobit yt x1 x2 i.g, ll(0)
dumpe tobit_e
margins, dydx(*)
dump tobit_xb
margins, dydx(*) predict(e(0,.))
dump tobit_e0
margins, dydx(*) predict(ystar(0,.))
dump tobit_ystar0
margins, dydx(*) predict(pr(0,.))
dump tobit_pr0
tobit yt x1 x2 i.g, ll(0) vce(robust)
margins, dydx(*) predict(e(0,.))
dump tobit_e0_robust
heckman yh x1 x2 g2 g3, select(s = z x1)
dumpe heckman_e
margins, dydx(*)
dump heckman_xb
margins, dydx(*) predict(ycond)
dump heckman_ycond
margins, dydx(*) predict(yexpected)
dump heckman_yexpected
margins, dydx(*) predict(psel)
dump heckman_psel
ivprobit yb x1 g2 g3 (xe = z)
dumpe ivprobit_e
margins, dydx(*)
dump ivprobit_ame
margins, dydx(*) predict(pr)
dump ivprobit_pr
margins, dydx(*) predict(xb)
dump ivprobit_xb
ologit yo x1 x2 i.g
margins, at(x1=(-1 0 1))
dump ologit_at
mlogit ym x1 x2 i.g, base(0)
margins g
dump mlogit_g
tobit yt x1 x2 i.g, ll(0)
margins, at(x1=(0 1)) predict(e(0,.))
dump tobit_at_e0
margins, at(x1=(0 1)) predict(e(0,.)) contrast(atcontrast(r) nowald)
dump tobit_at_e0_contrast
tobit yt x1 x2 i.g [pw=w], ll(0)
dumpe tobit_pw_e
margins, dydx(*) predict(e(0,.))
dump tobit_pw_e0
heckman yh x1 x2 g2 g3, select(s = z x1)
margins, at(x1=(0 1)) predict(ycond)
dump heckman_at_ycond
ivprobit yb x1 g2 g3 (xe = z)
margins, at(xe=(0 1)) predict(pr)
dump ivprobit_at_pr
margins, at(x1=(0 1)) predict(pr)
dump ivprobit_at_x1_pr
margins, dydx(*) predict(pr fixedasf)
dump ivprobit_fixedasf
heckprobit ybh x1 x2 g2 g3, select(s = z x1)
dumpe heckprobit_e
margins, dydx(*)
dump heckprobit_pmargin
margins, dydx(*) predict(pcond)
dump heckprobit_pcond
margins, dydx(*) predict(p11)
dump heckprobit_p11
margins, dydx(*) predict(p10)
dump heckprobit_p10
margins, dydx(*) predict(p01)
dump heckprobit_p01
margins, dydx(*) predict(p00)
dump heckprobit_p00
margins, dydx(*) predict(psel)
dump heckprobit_psel
heckprobit ybh x1 x2 g2 g3, select(s = z x1) vce(robust)
dumpe heckprobit_rob_e
intreg ilo ihi x1 x2 i.g
dumpe intreg_e
display e(N) " " e(N_unc) " " e(N_lc) " " e(N_rc) " " e(N_int) " " %20.15g e(ll)
margins, dydx(*)
dump intreg_xb
margins, dydx(*) predict(e(0,.))
dump intreg_e0
margins, dydx(*) predict(ystar(-1,3))
dump intreg_ystar
margins, dydx(*) predict(pr(0,2))
dump intreg_pr02
intreg ilo ihi x1 x2 i.g, vce(robust)
dumpe intreg_rob_e
intreg ilo ihi x1 x2 i.g [pw=w]
dumpe intreg_pw_e
margins, dydx(*) predict(e(0,.))
dump intreg_pw_e0
file close fh
