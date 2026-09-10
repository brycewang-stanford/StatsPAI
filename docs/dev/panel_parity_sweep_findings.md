# Panel family — findings

Fourth family. Reference: Stata 18 MP (official `xtlogit`, `xtprobit`,
`xtgls`, `xtunitroot`, `hausman`) via the local runtime, plus `plm` 2.6.7.

Fixture: `scratchpad/panel_data.csv`, N=60, T=12, balanced, with `x1`
correlated with the unit effect.

## Measured gaps, like-for-like

| StatsPAI | Stata | worst rel | notes |
| --- | --- | ---: | --- |
| `panel_logit(method='re')` | `xtlogit, re` | 4.2e-03 | does NOT shrink with `n_quadrature` (12 → 30) |
| `panel_probit(method='re')` | `xtprobit, re` | 3.9e-03 | same |
| `panel_fgls(panels='heteroskedastic', corr='independent')` | `xtgls, panels(hetero)` | 2.8e-02 | defaults already match on both sides |

**The quadrature insensitivity was the whole clue**, and it is worth
keeping as a method note. A non-adaptive Gauss-Hermite rule that is merely
coarse converges as points are added. These did not move at all between 12
and 30 points, which rules out the integral and points at the likelihood
or the design matrix.

## Diagnosis (both closed in 1.27.0)

### P1 — `sp.panel_fgls` was running Stata's `igls` (⚠️ correctness)

Bisection took three lines. The textbook two-step FGLS — OLS, per-panel
sigma^2 from the OLS residuals, WLS with weight 1/sigma^2_i — reproduces
`xtgls, panels(hetero)` to **1.9e-15** on the coefficient and 7.0e-16 on
the standard error. So Stata was doing the obvious thing and StatsPAI was
not.

The loop in `panel/panel_fgls.py` re-estimated the variance parameters
from the *GLS* residuals and iterated to convergence. That is a different
estimator (iterated FGLS, which for this structure is ML) and Stata calls
it `igls` — an option, not the default. The docstring said "Equivalent to
Stata's `xtgls y x, panels(het) corr(ar1)`".

Fixed by making two-step the default and exposing `igls=True`, which
reproduces the old numbers against `xtgls, igls` to 4.7e-08. CLAUDE.md 5.1
step 2: follow the reference's default, expose the alternative, document
both.

### P2 — RE logit and RE probit had no intercept (⚠️ correctness)

An independently written RE-probit likelihood (Gauss-Hermite over
`u ~ N(0, sigma_u^2)`, `e ~ N(0,1)`) converges to Stata's answer at
**1.4e-08** with 60 points — so the model and its parameterisation are
right. The same likelihood *without a constant column* reproduces
`sp.panel_probit(method='re')` to **2.8e-08**.

That is the proof: the RE fitter builds its design from `x` alone. It
shares `_group_panel` with the conditional FE logit, where the constant is
correctly absent because it is differenced out; nothing put one back for
the RE path.

The bias is 0.39% here only because the fixture's regressors are centred.
With the intercept restored the RE models converge to Stata as the
quadrature is refined — 4.0e-08 at 60 points, log-likelihood 1.7e-09 —
and the test asserts that *convergence* rather than a fixed tolerance,
because Stata integrates adaptively and StatsPAI does not.

## Fixture caveats found on the way

* `hausman fe re` returns a **negative** chi2 (-96.3) on this fixture, so
  the difference of covariance matrices is not positive semi-definite.
  That is a property of the design, not a defect; a Hausman fixture needs
  `sigmamore`/`sigmaless` or a design where the two covariances nest
  cleanly. Do not pin `sp.hausman_test` against this fixture as it stands.
* `xtunitroot llc` / `ips` returned missing for the `r()` names used here;
  the correct return names need checking against `return list` before a
  fixture is built.

## Ready to certify once the above is settled

`sp.panel_unitroot` (Stata `xtunitroot llc | ips | fisher`),
`sp.xtdpdsys` (`xtdpdsys`), `sp.xtnbreg` (`xtnbreg`),
`sp.hausman_test` (`hausman` with a sound fixture).
