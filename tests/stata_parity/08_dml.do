* tests/stata_parity/08_dml.do
*
* Module 08: DML PLR with linear nuisance learners and explicit folds.
*   StatsPAI:  sp.dml(model="plr", model_y=LinearRegression(),
*                     model_d=LinearRegression(), fold_indices=fold_id)
*   R:         DoubleML::DoubleMLPLR + mlr3::regr.lm + set_sample_splitting()
*   Stata:     audited Stata/Mata implementation of the same DML2 score
*
* This is not a packaged Stata ddml reference. It is a portable algorithm
* bridge for the deterministic linear-nuisance PLR fixture: foldwise OLS
* residualisation, pooled DML2 theta, and the same orthogonal-score SE.

version 18
clear all

do _common.do
stata_parity_init, module(08_dml)
stata_parity_open, module(08_dml)

import delimited "${STATA_PARITY_DATA}/08_dml.csv", clear case(preserve)

mata:
void _sp_dml_plr_bridge()
{
    real matrix data, X, Xtr, Xte
    real colvector y, d, fold, y_resid, d_resid, train, test
    real colvector by, bd, psi_score
    real scalar f, n, theta, denom, J, sigma2, se

    data = st_data(., ("lwage", "educ", "exper", "expersq", "black", "south", "smsa", "fold_id"))
    y = data[, 1]
    d = data[, 2]
    X = J(rows(data), 1, 1), data[, (3, 4, 5, 6, 7)]
    fold = data[, 8]
    n = rows(data)
    y_resid = J(n, 1, 0)
    d_resid = J(n, 1, 0)

    for (f = 0; f <= 4; f++) {
        train = selectindex(fold :!= f)
        test = selectindex(fold :== f)
        Xtr = X[train, .]
        Xte = X[test, .]
        by = qrsolve(Xtr, y[train])
        bd = qrsolve(Xtr, d[train])
        y_resid[test] = y[test] - Xte * by
        d_resid[test] = d[test] - Xte * bd
    }

    denom = quadcross(d_resid, d_resid)
    theta = quadcross(d_resid, y_resid) / denom
    psi_score = (y_resid :- theta :* d_resid) :* d_resid
    J = -mean(d_resid:^2)
    sigma2 = mean(psi_score:^2)
    se = sqrt(sigma2 / (J^2 * n))

    st_numscalar("SP_theta", theta)
    st_numscalar("SP_se", se)
    st_numscalar("SP_n", n)
}
_sp_dml_plr_bridge()
end

local theta = SP_theta
local se = SP_se
local n = SP_n
local lo = `theta' - ${STATA_PARITY_Z95} * `se'
local hi = `theta' + ${STATA_PARITY_Z95} * `se'

stata_parity_row, statname("theta_DML_PLR") estimate(`theta') stderr(`se') cilo(`lo') cihi(`hi') nobs(`n')

stata_parity_extra, key(dml_model) val("PLR")
stata_parity_extra_num, key(n_folds) val(5)
stata_parity_extra, key(ml_g) val("foldwise OLS")
stata_parity_extra, key(ml_m) val("foldwise OLS")
stata_parity_extra, key(fold_source) val("user")
stata_parity_extra, key(fold_column) val("fold_id")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge")
stata_parity_extra, key(stata_algorithm) val("DML2 PLR: foldwise OLS residualization and pooled orthogonal score")
stata_parity_extra, key(stata_reference_note) val("Portable algorithm bridge for deterministic linear-nuisance PLR; not a packaged Stata ddml reference.")

stata_parity_close, module(08_dml)
