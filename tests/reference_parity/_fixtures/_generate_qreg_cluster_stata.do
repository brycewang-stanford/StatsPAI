* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_qreg_cluster_stata_parity.py
*
* Requires: Stata 18 + qreg2 (SSC; Machado, Parente and Santos Silva), the
*           authors' implementation of Parente and Santos Silva (2016).
*           Installed into ./_ado_qreg2 (gitignored), never into PLUS.
* Run:      stata -b do _generate_qreg_cluster_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* sp.qreg(vce="cluster <g>") -- the Parente-Santos Silva cluster-robust
* covariance -- and sp.qreg(vce="pss") -- the same estimator without
* clustering (qreg2's default heteroskedasticity-robust SE) -- at four
* quantiles, with the default MAD bandwidth scale and with silverman.
* Official qreg refuses vce(cluster), so qreg2 is the reference.
*
* Errors share a cluster component and are heteroskedastic in x1, so the
* clustered and unclustered variances differ materially.
* ---------------------------------------------------------------------------
version 18
clear all
set type double

local here "`c(pwd)'"
local ado "`here'/_ado_qreg2"
capture mkdir "`ado'"
capture which qreg2
if _rc {
    local plus_saved "`c(sysdir_plus)'"
    sysdir set PLUS "`ado'"
    ssc install qreg2, replace
    sysdir set PLUS "`plus_saved'"
}
adopath + "`ado'"

set seed 20260928
set obs 40
gen int g = _n
gen double c_g = rnormal()
expand 15
gen double x1 = rnormal() + 0.5*c_g
gen double x2 = runiform()
gen double y  = 1 + 0.5*x1 - 0.8*x2 + 0.9*c_g + (1 + 0.5*abs(x1))*rnormal()
drop c_g
gen long obs = _n
* string cluster ids (qreg2 encodes them) and a cluster id with holes
* (qreg2 drops those rows from the estimation sample)
gen str8 gs = "c" + string(g)
gen double g_miss = cond(mod(obs, 17) == 0, ., g)

format x1 x2 y %21.16e
export delimited obs g gs g_miss x1 x2 y using "qreg_cluster_data.csv", replace datafmt

capture program drop wr
program define wr
    args name
    matrix V = e(V)
    matrix T = r(table)
    local names : colnames e(b)
    file write fh `"  "`name'": {"' _n
    local i = 0
    foreach v of local names {
        local ++i
        local nm = cond("`v'" == "_cons", "cons", "`v'")
        file write fh `"    "b_`nm'": "' %21.16e (_b[`v']) `","' _n
        file write fh `"    "p_`nm'": "' %21.16e (T[4, `i']) `","' _n
        file write fh `"    "ll_`nm'": "' %21.16e (T[5, `i']) `","' _n
        file write fh `"    "ul_`nm'": "' %21.16e (T[6, `i']) `","' _n
        local j = 0
        foreach w of local names {
            local ++j
            local nw = cond("`w'" == "_cons", "cons", "`w'")
            file write fh `"    "V_`nm'_`nw'": "' %21.16e (V[`i', `j']) `","' _n
        }
    }
    file write fh `"    "zeros": "' %21.16e (e(zeros)) `","' _n
    file write fh `"    "f_r": "' %21.16e (e(f_r)) `","' _n
    file write fh `"    "N": "' %21.16e (e(N)) _n
    file write fh `"  },"' _n
end

file open fh using "qreg_cluster_stata.json", write replace text
file write fh "{" _n
foreach q in 25 50 75 90 {
    qreg2 y x1 x2, quantile(0.`q') cluster(g) notest
    wr cluster_q`q'
    qreg2 y x1 x2, quantile(0.`q') notest
    wr robust_q`q'
    qreg2 y x1 x2, quantile(0.`q') cluster(g) silverman notest
    wr cluster_silverman_q`q'
}
qreg2 y x1 x2, quantile(0.5) cluster(gs) notest
wr cluster_string_q50
qreg2 y x1 x2, quantile(0.5) cluster(g_miss) notest
wr cluster_missing_q50
qreg2 y x1 x2, quantile(0.5) silverman notest
wr robust_silverman_q50
file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write fh `"    "qreg2": "version 4.00 31 Aug 2020 (SSC; help file dated 3.9)","' _n
file write fh `"    "seed": 20260928,"' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote qreg_cluster_stata.json and qreg_cluster_data.csv"
