* tests/stata_parity/71_dml_family.do
*
* Module 71: the three DoubleML model classes module 08 does not cover.
*   StatsPAI:  sp.dml(model = "irm" | "pliv" | "iivm")
*   R:         DoubleML::DoubleMLIRM / DoubleMLPLIV / DoubleMLIIVM (1.0.2)
*   Stata:     ddml init interactive | iv | interactiveiv   <-- this file
*
* Shared-fold design. Cross-fitting noise is an artifact of the sample
* split, not of the estimator, so the CSV carries a deterministic
* `fold_id` column and all three engines consume it through their explicit
* sample-splitting API: `fold_indices=` in StatsPAI, `set_sample_splitting()`
* in DoubleML, and `ddml init ..., foldvar()` here. `ddml` numbers folds from
* 1, so this script passes `fold_id + 1`; the partition is otherwise
* identical row for row. With the split fixed by the data the remaining gap
* is the estimator.
*
* Learners are the closed-form pair used on the other two sides:
* `regress` for every conditional expectation of a continuous variable and
* `logit` (unpenalised MLE) for every binary one -- the Stata counterparts
* of LinearRegression / LogisticRegression(penalty=None) and of mlr3's
* regr.lm / classif.log_reg. Nothing is stacked or tuned, so the three
* implementations are solving the same estimating equations.
*
* Observed agreement against the Python side (2026-08-06):
*   theta_DML_IRM   rel 2.1e-12   -- score and propensity MLE agree
*   theta_DML_IIVM  rel ~1e-11    -- likewise, on the LATE score
*   theta_DML_PLIV  rel 6.9e-7    -- ddml fits the second-stage moment
*                                    condition with an intercept, which
*                                    DoubleML's PLIV score does not carry;
*                                    the residual is that intercept, not
*                                    cross-fitting noise.
*
* Tolerance (compare.py): rel_est < 1e-6 for PLIV, < 1e-4 for IRM / IIVM.

version 18
clear all

do _common.do
stata_parity_init, module(71_dml_family)
stata_parity_open, module(71_dml_family)

import delimited "${STATA_PARITY_DATA}/71_dml_family.csv", clear case(preserve)
count
local n = r(N)

* ddml folds are 1-based; the committed fold_id is 0-based.
gen int fold1 = fold_id + 1

* ------------------------------------------------------------------
* IRM -- binary treatment, ATE
* ------------------------------------------------------------------
qui ddml init interactive, foldvar(fold1)
qui ddml E[Y|X,D]: regress y_irm x1 x2 x3 x4 x5
qui ddml E[D|X]:   logit   d_bin x1 x2 x3 x4 x5
qui ddml crossfit
qui ddml estimate, robust

local bv = _b[d_bin]
local sv = _se[d_bin]
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
stata_parity_row, stat(theta_DML_IRM) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

* ------------------------------------------------------------------
* PLIV -- continuous endogenous treatment, continuous instrument
* ------------------------------------------------------------------
qui ddml init iv, foldvar(fold1)
qui ddml E[Y|X]: regress y_pliv x1 x2 x3 x4 x5
qui ddml E[D|X]: regress d_cont x1 x2 x3 x4 x5
qui ddml E[Z|X]: regress z_c x1 x2 x3 x4 x5
qui ddml crossfit
qui ddml estimate, robust

local bv = _b[d_cont]
local sv = _se[d_cont]
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
stata_parity_row, stat(theta_DML_PLIV) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

* ------------------------------------------------------------------
* IIVM -- binary treatment, binary instrument, LATE
* ------------------------------------------------------------------
qui ddml init interactiveiv, foldvar(fold1)
qui ddml E[Y|X,Z]: regress y_iivm x1 x2 x3 x4 x5
qui ddml E[D|X,Z]: logit   d_iv x1 x2 x3 x4 x5
qui ddml E[Z|X]:   logit   z_b x1 x2 x3 x4 x5
qui ddml crossfit
qui ddml estimate, robust

local bv = _b[d_iv]
local sv = _se[d_iv]
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
stata_parity_row, stat(theta_DML_IIVM) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(stata_command) val("ddml init interactive|iv|interactiveiv, foldvar(fold1); learners regress / logit; ddml crossfit; ddml estimate, robust")
stata_parity_extra, key(fold_source) val("user -- fold_id + 1 passed to ddml init foldvar(), the same partition DoubleML and sp.dml consume")
stata_parity_extra, key(learners) val("regress for continuous conditional expectations, logit (unpenalised MLE) for binary ones; no stacking, no tuning")
stata_parity_extra, key(pliv_intercept) val("ddml's PLIV second stage carries an intercept that DoubleML's score does not; this is the source of the 6.9e-7 PLIV gap")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18 and ddml from SSC")

stata_parity_close, module(71_dml_family)
