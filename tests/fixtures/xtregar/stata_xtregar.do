* Stata 18 MP reference for tests/test_xtregar_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_xtregar.do
* then python ../_stata_tsv.py stata_xtregar.tsv stata_xtregar.json
set more off
set type double
import delimited using "ar1panel.csv", clear asdouble
xtset id t
file open fh using "stata_xtregar.tsv", write replace
file write fh "about" _tab "_meta" _tab "stata_version" _tab "`c(stata_version)'" _n
file write fh "about" _tab "_meta" _tab "born_date" _tab "`c(born_date)'" _n
file write fh "about" _tab "_meta" _tab "edition" _tab "`c(edition_real)'" _n
capture program drop dumpe
program dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    forvalues j = 1/`=colsof(`b')' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
    foreach s in rho_ar sigma_u sigma_e rho_fov r2_w r2_b r2_o r2_a thta_min thta_5 thta_50 thta_95 thta_max N N_g g_min g_max Tbar Tcon corr F F_f df_r df_a df_b df_m chi2 rss mss rmse ll ll_0 d1 LBI N_LBI {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
xtregar y x1 x2, fe
dumpe fe
xtregar y x1 x2, re
dumpe re
foreach r in dw regress freg tscorr theil nagar onestep {
    xtregar y x1 x2, fe rhotype(`r')
    dumpe fe_`r'
    xtregar y x1 x2, re rhotype(`r')
    dumpe re_`r'
    if "`r'" != "onestep" {
        xtregar y x1 x2, fe rhotype(`r') twostep
        dumpe fe_`r'_2s
    }
}
xtregar y x1 x2, re twostep
dumpe re_dw_2s
xtregar y x1 x2, fe rhof(0.5)
dumpe fe_rhof
xtregar y x1 x2, re rhof(0.5)
dumpe re_rhof
xtregar y x1 x2, fe lbi
dumpe fe_lbi
xtregar y x1 x2 z, re
dumpe re_z
xtregar y x1 x2 if bal == 1, fe
dumpe fe_bal
xtregar y x1 x2 if bal == 1, re lbi
dumpe re_bal
* added 2026-11-05: twostep random effects for the other rhotypes, and the
* LBI statistic after random effects on the panel with gaps
foreach r in regress freg tscorr theil nagar {
    xtregar y x1 x2, re rhotype(`r') twostep
    dumpe re_`r'_2s
}
xtregar y x1 x2, re lbi
dumpe re_lbi
file close fh
