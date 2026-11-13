* Stata reference for sp.meta_regression, including the model without an
* intercept (where metafor's residual-heterogeneity statistic is not the
* weighted residual sum of squares; see _generate_gaps_meta_R.R).
*
*   cd tests/reference_parity/_fixtures
*   stata-mp -b do _generate_gaps_meta_Stata.do
*
* Reads gaps_meta.csv, writes gaps_meta_Stata.csv (long: case, name, value).
* Stata 18 MP, official -meta regress-. The iterative estimators are run
* with tightened tolerances.
clear all
set type double
import delimited "gaps_meta.csv", clear
meta set yi sei

tempname fh
file open `fh' using "gaps_meta_Stata.csv", write replace
file write `fh' "case,name,value" _n

program define _store
    args fh case
    foreach s in tau2 Q_res df_Q_res p_Q_res I2_res H2_res R2 chi2 p F df_m df_r {
        capture confirm scalar e(`s')
        if !_rc {
            file write `fh' "`case',`s'," %24.16e (e(`s')) _n
        }
    }
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    local j = 0
    foreach nm of local names {
        local ++j
        file write `fh' "`case',b_`nm'," %24.16e (b[1, `j']) _n
        file write `fh' "`case',se_`nm'," %24.16e (sqrt(V[`j', `j'])) _n
    }
end

local tight "iterate(1000) tolerance(1e-14) nrtolerance(1e-14)"
meta regress dose blind, random(reml) `tight'
_store `fh' REML_two
meta regress dose blind, random(mle) `tight'
_store `fh' ML_two
meta regress dose blind, random(dlaird)
_store `fh' DL_two
meta regress dose blind, random(reml) se(khartung) `tight'
_store `fh' REML_two_knha
meta regress dose blind, fixed
_store `fh' fixed_two
meta regress dose blind, noconstant random(reml) `tight'
_store `fh' REML_noint
meta regress dose blind, noconstant random(dlaird)
_store `fh' DL_noint
meta regress dose blind, noconstant random(mle) `tight'
_store `fh' ML_noint
meta regress dose blind, noconstant random(dlaird) se(khartung)
_store `fh' DL_noint_knha
file close `fh'
