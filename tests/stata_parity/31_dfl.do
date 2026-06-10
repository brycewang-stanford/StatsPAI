* tests/stata_parity/31_dfl.do
*
* Module 31: DiNardo-Fortin-Lemieux reweighting decomposition.
*   StatsPAI:  sp.decompose("dfl", reference=1)
*   R:         ddecompose::dfl_decompose(reference_0=TRUE)
*   Stata:     audited Stata/Mata implementation of the same DFL bridge
*
* No canonical packaged Stata port is part of the portable parity baseline.
* This bridge implements the estimator directly: fit P(group==0 | X) by
* Newton-Raphson logit, reweight group 0 to group 1's covariate distribution,
* and decompose the mean gap into composition and structure components.

version 18
clear all

do _common.do
stata_parity_init, module(31_dfl)
stata_parity_open, module(31_dfl)

import delimited "${STATA_PARITY_DATA}/31_dfl.csv", clear case(preserve)

mata:
real colvector _sp_logit_p(real matrix X, real colvector y)
{
    real scalar iter
    real colvector beta, eta, p, W, grad, step, beta_new
    real matrix H

    beta = J(cols(X), 1, 0)
    for (iter = 1; iter <= 100; iter++) {
        eta = X * beta
        eta = eta :+ ((eta :< -30) :* (-30 :- eta))
        eta = eta :- ((eta :> 30) :* (eta :- 30))
        p = 1 :/ (1 :+ exp(-eta))
        W = p :* (1 :- p)
        grad = quadcross(X, y :- p)
        H = -quadcross(X, W, X)
        step = qrsolve(H, grad)
        beta_new = beta - step
        if (max(abs(beta_new :- beta)) < 1e-8) {
            beta = beta_new
            break
        }
        beta = beta_new
    }

    eta = X * beta
    eta = eta :+ ((eta :< -30) :* (-30 :- eta))
    eta = eta :- ((eta :> 30) :* (eta :- 30))
    return(1 :/ (1 :+ exp(-eta)))
}

void _sp_dfl_bridge()
{
    real matrix data, X
    real colvector y, g, treat_a, p_hat, p_a, psi, ya, yb, ia, ib
    real scalar pA, stat_a, stat_b, stat_cf, gap, composition, structure

    data = st_data(., ("log_wage", "educ", "exper", "female"))
    y = data[, 1]
    g = data[, 4]
    ia = selectindex(g :== 0)
    ib = selectindex(g :== 1)

    X = J(rows(data), 1, 1), data[, (2, 3)]
    treat_a = (g :== 0)
    p_hat = _sp_logit_p(X, treat_a)
    p_hat = p_hat :+ ((p_hat :< 0.001) :* (0.001 :- p_hat))
    p_hat = p_hat :- ((p_hat :> 0.999) :* (p_hat :- 0.999))

    ya = y[ia]
    yb = y[ib]
    p_a = p_hat[ia]
    pA = rows(ia) / rows(data)

    // StatsPAI reference=1: reweight group A (female==0) to group B's X.
    psi = ((1 :- p_a) :/ p_a) :* (pA / (1 - pA))

    stat_a = mean(ya)
    stat_b = mean(yb)
    stat_cf = quadcross(psi, ya) / sum(psi)
    gap = stat_a - stat_b
    composition = stat_a - stat_cf
    structure = stat_cf - stat_b

    st_numscalar("SP_gap", gap)
    st_numscalar("SP_composition", composition)
    st_numscalar("SP_structure", structure)
    st_numscalar("SP_stat_a", stat_a)
    st_numscalar("SP_stat_b", stat_b)
    st_numscalar("SP_stat_cf", stat_cf)
}
_sp_dfl_bridge()
end

local n = _N
stata_parity_row, statname("gap") estimate(`=SP_gap') nobs(`n')
stata_parity_row, statname("composition") estimate(`=SP_composition') nobs(`n')
stata_parity_row, statname("structure") estimate(`=SP_structure') nobs(`n')
stata_parity_row, statname("stat_a") estimate(`=SP_stat_a') nobs(`n')
stata_parity_row, statname("stat_b") estimate(`=SP_stat_b') nobs(`n')
stata_parity_row, statname("stat_cf") estimate(`=SP_stat_cf') nobs(`n')

stata_parity_extra, key(method) val("DFL mean reweighting")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge")
stata_parity_extra, key(stata_algorithm) val("Newton-Raphson logit P(group==0|educ,exper); reference=1 reweights group 0 to group 1 covariates")
stata_parity_extra, key(stata_reference_note) val("Portable algorithm bridge because no canonical packaged Stata DFL port is selected for the baseline.")
stata_parity_extra_num, key(reference) val(1)
stata_parity_extra, key(statistic) val("mean")

stata_parity_close, module(31_dfl)
