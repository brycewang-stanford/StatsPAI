* Stata 18 MP reference for tests/test_mmqreg.py.
* From this directory: python _data.py . && stata-mp -b do stata_mmqreg.do
* with mmqreg 2.6 (Rios-Avila, SSC; needs ftools and hdfe) and xtqreg 1.3
* (Machado and Santos Silva, SSC) on the adopath -- installed in a private
* PLUS directory, not the user's. The tab-separated stata_mmqreg.tsv is
* stored as stata_mmqreg.json ({tag: {matrix: {rows, cols, data}}}).
set more off
set type double
import delimited using "mmqr_panel.csv", clear asdouble
xtset id t
file open fh using "stata_mmqreg.tsv", write replace
capture program drop dumpm
program dumpm
    args tag mname
    tempname M
    matrix `M' = `mname'
    local cn : colfullnames `M'
    local rn : rowfullnames `M'
    forvalues r = 1/`=rowsof(`M')' {
        forvalues c = 1/`=colsof(`M')' {
            local a : word `r' of `rn'
            local b : word `c' of `cn'
            file write fh "`tag'" _tab "`mname'" _tab "`a'" _tab "`b'" _tab %24.17g (`M'[`r',`c']) _n
        }
    }
end
capture program drop dumps
program dumps
    args tag
    file write fh "`tag'" _tab "e(N)" _tab "r" _tab "c" _tab %24.17g (e(N)) _n
end
* Panel 61 has one observation: its location residual and fitted scale are
* both rounding noise (about 1e-16), and Stata keeps it, so its
* "standardized residual" is 0/0 noise that enters the sample quantile.
* sp.mmqreg drops singletons; the comparisons are on the data without it.
mmqreg y x1 x2 ax1, absorb(id) q(25)
dumpm singleton_kept e(qval)
dumps singleton_kept
drop if id == 61
foreach q in 25 50 90 {
    mmqreg y x1 x2 ax1, absorb(id) q(`q')
    dumpm fe1_gls_`q' e(b)
    dumpm fe1_gls_`q' e(V)
    dumpm fe1_gls_`q' e(fden)
    dumpm fe1_gls_`q' e(qval)
    dumps fe1_gls_`q'
    mmqreg y x1 x2 ax1, absorb(id) q(`q') robust
    dumpm fe1_robust_`q' e(V)
    mmqreg y x1 x2 ax1, absorb(id) q(`q') cluster(id)
    dumpm fe1_cluster_`q' e(V)
    dumps fe1_cluster_`q'
}
foreach v in gls robust cluster {
    local opt = cond("`v'" == "gls", "", cond("`v'" == "robust", "robust", "cluster(id)"))
    mmqreg y x1 x2 ax1, absorb(id t) q(50) `opt'
    dumpm fe2_`v'_50 e(b)
    dumpm fe2_`v'_50 e(V)
    dumps fe2_`v'_50
    mmqreg y x1 x2 ax1, q(50) `opt'
    dumpm nofe_`v'_50 e(b)
    dumpm nofe_`v'_50 e(V)
    dumps nofe_`v'_50
}
foreach q in 0.25 0.5 0.9 {
    local tg = round(`q'*100)
    xtqreg y x1 x2 ax1, i(id) quantile(`q') ls
    foreach m in e(b) e(V) e(q) e(b_location) e(V_location) e(b_scale) e(V_scale) {
        dumpm xtqreg_`tg' `m'
    }
    dumps xtqreg_`tg'
}
file close fh
