# Clean-room specification: CCT bandwidth selection and robust bias-corrected RD

Audit rows H6 (`rd/_cct_bandwidth.py`) of
`docs/dev/2026-10-20-license-provenance-audit.md`. Rewritten 2026-10-20 as
`rd/_cct_bandwidth.py` (selector), `rd/_cct_inference.py` (estimate and
variances for given bandwidths), `rd/_cct_side.py` (per-side fits) and
`rd/_nn_variance.py` (nearest-neighbour residuals, written by a separate
agent from the published definition).

Inputs used: the papers and supplements listed below, the R / Stata help
files of `rdrobust`, `rdbwselect`, and black-box runs of R `rdrobust` 4.0.0
(GPL-3) on committed or scratch data. No source of `rdrobust` (R, Stata,
Python) was read. The previous StatsPAI module was seen only through
`redact.py` (signatures and first docstring lines; its module docstring,
which the helper prints, sketched the three-stage cascade, which is also in
the supplements below). Statements marked **[BB]** were fixed by black-box
comparison and are not in the papers; every other statement cites a source.

Sources: CCT 2014 = Calonico, Cattaneo and Titiunik, *Econometrica* 82(6)
(`calonico2014robust`) and its Supplemental Appendix (SA); CCFT 2019 =
Calonico, Cattaneo, Farrell and Titiunik, *Review of Economics and
Statistics* 101(3), Supplemental Appendix Part IV; CCFT 2017 = the same
authors' *Stata Journal* 17(2) software article; CCF 2020 = Calonico,
Cattaneo and Farrell, *Econometrics Journal* 23(2) (`calonico2020optimal`).

## Estimator for given bandwidths (`cct_bias_corrected`)

Per side, `x` centred at the cutoff, columns `C = (Y, [T], Z...)`, weights
`w_i = K(x_i/h)/h` times the user weight (CCFT 2017 §2.1: user weights
multiply the kernel). Kernels: triangular `1-|u|`, Epanechnikov
`0.75(1-u^2)`, uniform `0.5`, on `|u| <= 1`.

* Conventional: `nu! e_nu' G_p^-1 P_p' W_h C`, `G_p = P_p' W_h P_p`, raw
  powers of `x` (CCT 2014 eq. 3.1).
* Bias-corrected (CCT 2014 §3, CCFT 2019 SA §12.4): subtract
  `nu! e_nu' G_p^-1 (P_p' W_h x^(p+1)) * beta_q(b)[p+1]`, where
  `beta_q(b)[p+1]` is the coefficient of `x^(p+1)` of the order-`q` fit at
  `b`. The result is linear in `C` with weights `a_bc`.
* Covariates (CCFT 2019 eq. (3)): common slopes `gamma` from the joint WLS
  of each core column on side-specific order-`p` polynomials at `h` and the
  covariates; the estimate is the combination `s = (1, -gamma)` of the
  per-column estimates. [BB: `gamma` equals R's `coef_covs` to 1e-14.]
* Fuzzy: `tau = tau_Y / tau_T` from covariate-adjusted jumps; delta-method
  weights `s = (1/tau_T, -tau_Y/tau_T^2, -(gamma_Y/tau_T - tau_Y gamma_T /
  tau_T^2))`; the bias-corrected estimate is the linearised
  `tau + s'(jump_bc - jump_cl)`, not the ratio of corrected jumps [BB].
* Variances: `sum_i (a_i s' rho_i)^2` per side, `rho_i` the residual
  vector. [BB] Residuals live on the union of the `h` and `b` windows:
  - `nn`: nearest-neighbour residuals (CCT 2014 §5; ties: CCFT 2017 §2.2)
    computed among the union window's observations, used for both
    variances;
  - plug-in (`hc0`..`hc3`, CCFT 2017 §2.2): residuals of the order-`p` fit
    at `h` (conventional) and of the order-`q` fit at `b` (robust); `hc1`
    scales by `sqrt(n/(n-k))` with `n` the union-window count and `k = p+1`
    or `q+1`; `hc2`, `hc3` by `(1-lev)^(-1/2)`, `(1-lev)^-1` with the
    leverage of the corresponding fit.
  - clusters: squared cluster sums times `G/(G-1) (n-1)/(n-k)` (CR1), with
    `n`, `G` counted on the union window of that side; `nn` and `hc1` with
    clusters use raw plug-in residuals [BB: equals R 4.0.0's `cr1` to
    1e-15]. `hc2` / `hc3` with clusters keep the leverage-scaled residuals
    with the CR1 factor (StatsPAI's existing behaviour, reproduced to
    1e-15). R 4.0.0 now maps these to its CR2 / CR3; the difference is
    0.07% to 0.6% of the SE on the fixtures tried (open item for the owner).

## Bandwidth selector (`cct_bandwidth`)

Pilot `c = C_K min(sd(x), IQR(x)/1.349) N^(-1/5)` (CCT 2014 SA S.2.6 Step 0;
CCFT 2019 SA §12.1), `C_K` = 2.576 / 1.843 (papers), 2.34 Epanechnikov
[BB]; `sd` with divisor `n-1` [BB]; IQR from the inverse-CDF sample
quantile (`numpy` `inverted_cdf`, R type 1/2) [BB].

A stage `(o, nu, o_B, hB)` on one side (CCFT 2019 SA §12.2-12.3):

* order-`o` fit at `c`: target `nu! e_nu' beta_o`, variance `V` (residuals
  of this fit's window, as above; hc1 `n` = window count, `k = o+1`),
  bias factor `F = nu! e_nu' G_o^-1 P_o' W_c x^(o+1)`;
* order-`o_B` fit at `hB`: `beta_oB[o+1]`; `B = F s' beta_oB[o+1]`,
  `R = F^2 Var(s' beta_oB[o+1])`;
* bandwidth `c [(1+2nu) V / (2(o+1-nu)(B^2 + 3 scale^2 R))]^(1/(2o+3))`.
  The factor 3 is the IK-type regularisation of CCT 2014 SA S.2.6 (Steps 1-2);
  `scale` (= `scaleregul`) enters squared [BB].
* Combination over sides: `rd` uses `V_l+V_r`, `(B_r - B_l)^2`, `R_l+R_r`;
  `sum` uses `(B_r + B_l)^2`; `two` keeps each side (CCFT 2017 §2.3).
* Covariates in the cascade [BB]: `gamma` is estimated **per side** from the
  stage's own order-`o` fit at `c` (not the joint fit); `s = (1, -gamma)`.
* Fuzzy in the cascade [BB]: per side, `s` is the delta-method weight of
  `Y_nu / T_nu`, the covariate-adjusted `nu`-th coefficients of the
  stage's own fit (levels when `nu = 0`). Constant treatment on either side
  falls back to the sharp selector (CCFT 2017 §3.2, `sharpbw`).

Cascade (CCFT 2019 SA §12.2-12.3; CCT 2014 SA Step 0-2):

1. `d`: `(o, nu, o_B) = (q+1, q+1, q+2)`, `hB` = that side's range of `x`
   [BB], no regularisation.
2. `b`: `(q, p+1, q+1)`, `hB = d`.
3. `h`: `(p, deriv, q)`, `hB = b`.

`bwrestrict` (help files: bandwidths within the range of `x`): `d`, `b`
and `h` are capped after each stage, a common bandwidth at the larger of
the two side ranges and a per-side one at its own range [BB].

Selectors (CCFT 2017 §2.3): `comb1 = min(rd, sum)`, `comb2 = median(two,
rd, sum)` elementwise; CER (CCF 2020; CCFT 2017 §2.3): `h * m^(-p /
((3+p)(3+2p)))`, `b` unchanged [BB], `m` the sample size or, with
clusters, the clusters counted on each side and added [BB, existing
fixture `test_rd_cluster_cer_parity.py`].

Mass points (help files: "the preliminary bandwidths ... contain a minimal
number of unique observations", default 10): under `masspoints='adjust'`
`N` in the pilot is the number of distinct `x` [BB]; when a side's tied
share `1 - unique/n` reaches 0.2 [BB: none at 0.19996, applied at 0.2002]
and `bwcheck` is not given, `bwcheck = 10`. A given `bwcheck` always
applies: `c = max(c, k-th smallest distinct |x| on each side + 1e-8)` with
`k = min(bwcheck, distinct count)` [BB, the `1e-8`]. Only the pilot is
floored [BB]. `check` and `off` use `N = n`.

Failure: a non-finite or non-positive plug-in ratio raises
`NumericalInstability` (StatsPAI convention; the reference returns a
number). A rank-deficient window in the estimator raises the same.

## Black-box evidence

Scratch harness (`R CMD` via `Rscript`, rdrobust 4.0.0) on: the senate
data (`rdsenate.csv`), a 2,000-row synthetic design, a 0.05-grid mass-point
design, asymmetric supports, and the committed fixtures. Agreement after the
rewrite: bandwidths 1e-11 or better on continuous data (mass-point design
at `p = 2`: 4e-9, the order-5 global fit on a grid is ill-conditioned in
both implementations); estimates and SEs for given bandwidths 1e-13 or
better for every `vce`, covariates, weights, clusters, fuzzy, `deriv`,
asymmetric bandwidths.
