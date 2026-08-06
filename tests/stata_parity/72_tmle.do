* tests/stata_parity/72_tmle.do
*
* Module 72: the TMLE targeting step.
*   StatsPAI:  sp.tmle(fluctuation="per_arm")
*   R:         tmle::tmle (Gruber & van der Laan)
*   Stata:     audited Stata/Mata algorithm bridge        <-- this file
*
* Why a bridge and not a command. Stata ships no TMLE. The user-written
* `eltmle` is a wrapper that shells out to the same R package this module
* already pins, so a row built on it would re-measure the R reference
* through a shell rather than cross-validate it -- which is why the module's
* previous skip reason declined it, correctly. That argument rules out
* `eltmle`; it does not rule out implementing the published estimator.
*
* This follows the precedent set by modules 08 (DML PLR), 31 (DFL), 32
* (RIF), 53 (CR2) and 54/56 (multiway cluster): where no canonical Stata
* command implements the documented algorithm, the do-file implements the
* algorithm and labels itself an audited bridge.
*
* What makes this tractable is the module's own design. TMLE has two stages
* -- an initial fit of Q(a,W) and g(W), then a fluctuation of Q along the
* least-favourable submodel -- and only the second is the estimator. The
* Python side computes Q0, Q1 and g1W once and ships them in the CSV, so
* every engine runs the same targeting step on identical inputs. There is no
* learner to reproduce here, only arithmetic:
*
*   1. QAW  = A*Q1 + (1-A)*Q0                       (observed-arm initial fit)
*   2. H1   = A/g,  H0 = -(1-A)/(1-g)               (per-arm clever covariates)
*   3. eps  = coefficients from a no-constant logistic regression of Y on
*             (H1, H0) with offset logit(QAW)       -- the fluctuation
*   4. Q1*  = expit(logit(Q1) + eps1 * (1/g))
*      Q0*  = expit(logit(Q0) + eps0 * (-1/(1-g)))  (targeted arm fits)
*   5. psi  = mean(Q1* - Q0*)
*   6. SE   from the efficient influence curve
*             D = H(A,W)*(Y - QAW*) + (Q1* - Q0*) - psi,  se = sd(D)/sqrt(n)
*
* Step 3 uses Stata's own `glm ..., family(binomial) link(logit) offset()
* noconstant`, so the one iterative piece is a canonical Stata IRLS rather
* than hand-rolled Newton-Raphson.
*
* Fluctuation convention: `per_arm` is pinned deliberately. StatsPAI's
* documented default (`fluctuation='single'`) uses one clever covariate and
* a scalar epsilon; tmle::tmle uses two per-arm covariates. Both solve the
* efficient-influence-function equation and are asymptotically equivalent
* but differ at finite n -- on this fixture psi is 0.260765738 under
* 'single' against 0.260426124 under 'per_arm'. This script implements the
* two-covariate convention, the one the R reference uses.
*
* Tolerance: rel < 1e-6 on psi.

version 18
clear all

do _common.do
stata_parity_init, module(72_tmle)
stata_parity_open, module(72_tmle)

import delimited "${STATA_PARITY_DATA}/72_tmle.csv", clear case(preserve)
count
local n = r(N)

* ------------------------------------------------------------------
* 1-2. Observed-arm initial fit and the per-arm clever covariates.
* ------------------------------------------------------------------
gen double QAW = A * Q1 + (1 - A) * Q0
gen double H1  = A / g1W
gen double H0  = -(1 - A) / (1 - g1W)
gen double offs = log(QAW / (1 - QAW))

* ------------------------------------------------------------------
* 3. Fluctuation: no-constant logistic regression of Y on the two clever
*    covariates, offset at the initial fit.
* ------------------------------------------------------------------
qui glm Y H1 H0, family(binomial) link(logit) offset(offs) noconstant

local eps1 = _b[H1]
local eps0 = _b[H0]

* ------------------------------------------------------------------
* 4. Targeted arm-specific fits. For A=1 the clever covariate is 1/g and
*    H0 is zero; for A=0 it is -1/(1-g) and H1 is zero, so each arm is
*    fluctuated by its own epsilon.
* ------------------------------------------------------------------
gen double Q1s = invlogit(log(Q1 / (1 - Q1)) + `eps1' * (1 / g1W))
gen double Q0s = invlogit(log(Q0 / (1 - Q0)) + `eps0' * (-1 / (1 - g1W)))

* ------------------------------------------------------------------
* 5. Point estimate.
* ------------------------------------------------------------------
gen double blip = Q1s - Q0s
qui summarize blip, meanonly
local psi = r(mean)

* ------------------------------------------------------------------
* 6. Efficient-influence-curve standard error.
* ------------------------------------------------------------------
gen double QAWs = A * Q1s + (1 - A) * Q0s
gen double Hobs = A / g1W - (1 - A) / (1 - g1W)
gen double D    = Hobs * (Y - QAWs) + blip - `psi'
qui summarize D
* r(Var) is Stata's sample variance (n-1 divisor), which is the convention
* the Python and R sides use: se = sd(D)/sqrt(n). Using the population
* divisor instead shifts the SE by exactly sqrt(n/(n-1)) -- measured 4.2e-4
* on this fixture, which is above the module's registered rel_se budget and
* would read as a disagreement rather than as the arithmetic slip it is.
local se = sqrt(r(Var) / `n')

local lo = `psi' - ${STATA_PARITY_Z95} * `se'
local hi = `psi' + ${STATA_PARITY_Z95} * `se'
stata_parity_row, stat(psi_tmle_ate) est(`psi') std(`se') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra_num, key(epsilon_H1) val(`eps1')
stata_parity_extra_num, key(epsilon_H0) val(`eps0')
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge, materialized 2026-08-06 with licensed Stata 18")
stata_parity_extra, key(algorithm) val("per-arm TMLE fluctuation: glm Y H1 H0, family(binomial) link(logit) offset(logit(QAW)) noconstant; then expit-fluctuate each arm and average the blip")
stata_parity_extra, key(shared_nuisance) val("Q0, Q1 and g1W arrive in the CSV, computed once on the Python side, so only the targeting step differs between engines")
stata_parity_extra, key(why_not_eltmle) val("eltmle wraps the same R package this module pins, so a row built on it would re-measure the R reference through a shell rather than cross-validate it")
stata_parity_extra, key(se_convention) val("efficient influence curve D = H(A,W)*(Y - Q*(A,W)) + (Q1* - Q0*) - psi, se = sd(D)/sqrt(n) with the sample (n-1) variance divisor, matching both other sides")

stata_parity_close, module(72_tmle)
