* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_synth2_stata_parity.py
*
* Reads textbook_rcm.csv and textbook_rcm_x.csv (written by
* _generate_textbook_methods_data.py) and
* writes textbook_synth2_Stata.csv: one "key,value" row per number, %23.16e.
*
* Run (from this directory, Stata 18):
*   stata-mp -b do _generate_synth2_stata.do
* Community commands: synth (Abadie, Diamond and Hainmueller), synth2 and rcm
* (Yan and Chen), installed from SSC into the private ado directory
* _ado_textbook_methods/ next to this file (gitignored).
* ---------------------------------------------------------------------------
clear all
set more off
capture mkdir "_ado_textbook_methods"
adopath ++ "_ado_textbook_methods"
net set ado "_ado_textbook_methods"
foreach pkg in synth synth2 rcm {
    capture which `pkg'
    if _rc ssc install `pkg'
}

capture program drop emit
program emit
    args key value
    file write fh "`key'," %23.16e (`value') _n
end
capture file close fh
file open fh using "textbook_synth2_Stata.csv", write replace
file write fh "key,value" _n

import delimited using "textbook_rcm.csv", clear asdouble
xtset unit t
local spec y y(2) y(5) y(8) y(11) y(14) y(17) y(1(1)15) y(16(1)30)

* --- in-space placebo with a cutoff, in-time placebo, leave-one-out
synth2 `spec', trunit(6) trperiod(31) placebo(unit cut(1.5) period(25)) loo nofigure frame(s2)
emit main.rmse e(rmse)
emit main.r2 e(r2)
emit main.att e(att)
matrix W = e(U_wt)
local names : rownames W
forvalues i = 1/`=rowsof(W)' {
    local u : word `i' of `names'
    emit weight.`u' W[`i',1]
}
matrix M = e(mspe)
local names : rownames M
forvalues i = 1/`=rowsof(M)' {
    local u : word `i' of `names'
    emit mspe.pre.`u' M[`i',1]
    emit mspe.post.`u' M[`i',2]
    emit mspe.ratio.`u' M[`i',3]
    emit mspe.relative.`u' M[`i',4]
}
matrix P = e(pval)
local names : rownames P
forvalues i = 1/`=rowsof(P)' {
    local t : word `i' of `names'
    emit effect.`t' P[`i',1]
    emit p_two.`t' P[`i',2]
    emit p_right.`t' P[`i',3]
    emit p_left.`t' P[`i',4]
}
frame s2 {
    forvalues t = 1/40 {
        quietly summarize pred·y if unit == 6 & t == `t'
        emit path.main.`t' r(mean)
        quietly summarize pred·y·loomin if unit == 6 & t == `t'
        emit path.loomin.`t' r(mean)
        quietly summarize pred·y·loomax if unit == 6 & t == `t'
        emit path.loomax.`t' r(mean)
        quietly summarize pred·y·25 if unit == 6 & t == `t'
        emit path.time25.`t' r(mean)
    }
}

* --- effect measured on some of the post-treatment periods only
synth2 `spec', trunit(6) trperiod(31) postperiod(31(1)35) placebo(unit) nofigure frame(s3)
emit post.att e(att)
matrix M = e(mspe)
emit post.mspe_post_treated M[1,2]
emit post.ratio_treated M[1,3]

* --- a list of pretend-treated units, and a shortened fitting period
synth2 `spec', trunit(6) trperiod(31) placebo(unit(1 2 3 4 5 7 8) cut(1.5)) nofigure frame(s4)
matrix P = e(pval)
forvalues i = 1/10 {
    emit some.p_two.`i' P[`i',2]
    emit some.p_right.`i' P[`i',3]
}
matrix M = e(mspe)
emit some.rows rowsof(M)
emit some.pre_last M[rowsof(M),1]
synth2 y y(12) y(15) y(18) y(21) y(24) y(27) y(11(1)20) y(21(1)30), trunit(6) trperiod(31) preperiod(11(1)30) nofigure frame(s5)
emit short.rmse e(rmse)
emit short.T0 e(T0)
matrix W = e(U_wt)
local names : rownames W
forvalues i = 1/`=rowsof(W)' {
    local u : word `i' of `names'
    emit short.weight.`u' W[`i',1]
}

* ============================================ rcm with covariates (Hsiao-Zhou)
import delimited using "textbook_rcm_x.csv", clear asdouble
xtset unit t
foreach m in forward best {
    rcm y x, trunit(1) trperiod(31) nofigure method(`m')
    emit rcmx.`m'.K e(K_preds_sel)
    emit rcmx.`m'.K_all e(K_preds_all)
    emit rcmx.`m'.att e(att)
    emit rcmx.`m'.rmse e(rmse)
    emit rcmx.`m'.r2 e(r2)
    matrix I = e(info)
    forvalues k = 1/23 {
        emit rcmx.`m'.aic.`k' I[`k',3]
        emit rcmx.`m'.bic.`k' I[`k',4]
        emit rcmx.`m'.r2.`k' I[`k',6]
    }
}
rcm y x, trunit(1) trperiod(31) nofigure method(forward) criterion(bic) placebo(unit cut(2) period(28))
emit rcmx.placebo.att e(att)
matrix M = e(mspe)
forvalues i = 1/12 {
    emit rcmx.placebo.pre.`i' M[`i',1]
    emit rcmx.placebo.post.`i' M[`i',2]
}
matrix P = e(pval)
forvalues i = 1/10 {
    emit rcmx.placebo.p_two.`i' P[`i',2]
    emit rcmx.placebo.p_right.`i' P[`i',3]
}
rcm y x, trunit(1) trperiod(31) nofigure method(forward) criterion(bic) placebo(unit(2 3 4 5) cut(2))
matrix P = e(pval)
forvalues i = 1/10 {
    emit rcmx.some.p_two.`i' P[`i',2]
    emit rcmx.some.p_left.`i' P[`i',4]
}
matrix M = e(mspe)
emit rcmx.some.rows rowsof(M)
* the pretend date as a fit of its own: same model, effects from period 28
rcm y x, trunit(1) trperiod(28) nofigure method(forward) criterion(bic)
emit rcmx.time28.K e(K_preds_sel)
emit rcmx.time28.att e(att)
* 24 pre-treatment periods and 23 candidates: the largest model has no
* residual degree of freedom and rcm selects it
rcm y x, trunit(1) trperiod(25) nofigure method(forward) criterion(bic)
emit rcmx.saturated.K e(K_preds_sel)
emit rcmx.saturated.r2 e(r2)
matrix I = e(info)
emit rcmx.saturated.bic.20 I[20,4]
emit rcmx.saturated.bic.22 I[22,4]
emit rcmx.saturated.bic.23 I[23,4]

file close fh
