* tests/stata_parity/53_cr2.do
*
* Module 53: CR2 / CR3 cluster-robust SE.
*   StatsPAI:  sp.cr2_se and sp.fast.crve(type="cr3")
*   R:         clubSandwich::vcovCR(type="CR2"/"CR3")
*   Stata:     audited Stata/Mata implementation of the same covariance
*
* Stata's built-in vce(cluster) is CR1, and no canonical installed Stata
* command on the test machine exposes clubSandwich-style CR2/CR3. This bridge
* computes Bell-McCaffrey CR2 and analytic CR3 directly from cluster hat blocks
* and records that status in the result metadata.

version 18
clear all

do _common.do
stata_parity_init, module(53_cr2)
stata_parity_open, module(53_cr2)

import delimited "${STATA_PARITY_DATA}/53_cr2.csv", clear case(preserve)

mata:
void _sp_cr2cr3_bridge()
{
    real matrix data, X, bread, meat2, meat3, V2, V3
    real matrix Xg, H, M, Q, A2, A3
    real rowvector lambda
    real colvector y, cl, ids, idx, resid, eg, beta, uadj, score
    real scalar i, p

    data = st_data(., ("lemp", "treat", "year", "countyreal"))
    y = data[, 1]
    X = J(rows(data), 1, 1), data[, (2, 3)]
    cl = data[, 4]
    p = cols(X)

    bread = invsym(cross(X, X))
    beta = bread * quadcross(X, y)
    resid = y - X * beta

    ids = uniqrows(sort(cl, 1))
    meat2 = J(p, p, 0)
    meat3 = J(p, p, 0)

    for (i = 1; i <= rows(ids); i++) {
        idx = selectindex(cl :== ids[i])
        Xg = X[idx, .]
        eg = resid[idx]
        H = Xg * bread * Xg'
        M = I(rows(Xg)) - 0.5 :* (H + H')
        symeigensystem(M, Q=., lambda=.)
        lambda = lambda :+ ((lambda :< 1e-12) :* (1e-12 :- lambda))

        A2 = Q * diag(lambda:^(-0.5)) * Q'
        uadj = A2 * eg
        score = Xg' * uadj
        meat2 = meat2 + score * score'

        A3 = Q * diag(lambda:^(-1)) * Q'
        uadj = A3 * eg
        score = Xg' * uadj
        meat3 = meat3 + score * score'
    }

    V2 = bread * meat2 * bread
    V3 = bread * meat3 * bread

    st_numscalar("SP_b0", beta[1])
    st_numscalar("SP_btreat", beta[2])
    st_numscalar("SP_byear", beta[3])
    st_numscalar("SP_cr2_0", sqrt(V2[1, 1]))
    st_numscalar("SP_cr2_treat", sqrt(V2[2, 2]))
    st_numscalar("SP_cr2_year", sqrt(V2[3, 3]))
    st_numscalar("SP_cr3_0", sqrt(V3[1, 1]))
    st_numscalar("SP_cr3_treat", sqrt(V3[2, 2]))
    st_numscalar("SP_cr3_year", sqrt(V3[3, 3]))
}
_sp_cr2cr3_bridge()
end

local n = _N
stata_parity_row, statname("cr2_(Intercept)") estimate(`=SP_b0') stderr(`=SP_cr2_0') nobs(`n')
stata_parity_row, statname("cr3_(Intercept)") estimate(`=SP_b0') stderr(`=SP_cr3_0') nobs(`n')
stata_parity_row, statname("cr2_treat") estimate(`=SP_btreat') stderr(`=SP_cr2_treat') nobs(`n')
stata_parity_row, statname("cr3_treat") estimate(`=SP_btreat') stderr(`=SP_cr3_treat') nobs(`n')
stata_parity_row, statname("cr2_year") estimate(`=SP_byear') stderr(`=SP_cr2_year') nobs(`n')
stata_parity_row, statname("cr3_year") estimate(`=SP_byear') stderr(`=SP_cr3_year') nobs(`n')

stata_parity_extra, key(formula) val("lemp ~ treat + year")
stata_parity_extra, key(vcov) val("CR2 (Bell-McCaffrey) + CR3 (clubSandwich analytic)")
stata_parity_extra, key(cluster_var) val("countyreal")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge")
stata_parity_extra, key(stata_reference_note) val("Portable bridge because Stata built-in vce(cluster) is CR1; computes clubSandwich-style CR2/CR3 from cluster hat blocks.")
stata_parity_extra, key(stata_algorithm) val("CR2 uses (I-H_gg)^-1/2 adjusted scores; CR3 uses (I-H_gg)^-1 adjusted scores")

stata_parity_close, module(53_cr2)
