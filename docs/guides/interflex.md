# Interaction effects with diagnostics: `sp.interflex`

`sp.interflex` is a native port of Hainmueller, Mummolo and Xu's
**interflex** (R and Stata): the conditional marginal effect of a
treatment `D` on an outcome `Y` across a moderator `X`, estimated three
ways, with the diagnostics the paper recommends before trusting a
multiplicative interaction term.

```python
import statspai as sp

# linear interaction model: ME(x) = b_D + b_DX * x, HC1 delta-method SEs
lin = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="linear")

# binning estimator (the default): one treatment effect per moderator bin
binned = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="binning", nbins=3)
binned.detail                      # bin, x (median), me, se, n, ci_lower, ci_upper
binned.model_info["tests"]         # x_lkurtosis, p_wald, p_lr

# kernel estimator: local linear regression with a density-adaptive Gaussian kernel
kern = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="kernel", bw=1.0,
                    vce="bootstrap", n_boot=200, seed=0)
sp.interflex_plot(kern)
```

`result.estimate` is the average treatment effect over treated
observations (binary `D`) or the average marginal effect over the sample
(continuous `D`); `result.detail` is the marginal-effect table on the
evaluation grid (`linear`, `kernel`) or at the bin medians (`binning`).

## What to look at

| Question | Where |
|---|---|
| Is the linear interaction adequate? | `model_info["tests"]["p_wald"]` / `["p_lr"]`: Wald and LR tests of the linear model against the fully interacted binning model. Small p-values say the marginal effect is not linear in `X`. |
| Is there common support? | `model_info["bins"]["counts"]` and the histogram in `sp.interflex_plot`; a bin with few treated observations extrapolates. |
| Is the moderator heavy-tailed? | `model_info["tests"]["x_lkurtosis"]`: the L-kurtosis of `X` (0.1226 for a normal distribution). |
| Does the picture change with the bandwidth? | Re-run the kernel estimator with a different `bw`; the interflex cross-validated bandwidth is not ported, so pass one explicitly. |

## Conventions and parity

Every convention follows the R package, so that Track A module 87 can
compare the two implementations on identical bytes (all 20 rows agree at
`6e-14`; the SSC Stata command agrees on the linear and binning rows at
`1e-14`):

- bins are cut at R's type-7 sample quantiles (right-closed, minimum in
  bin 1) unless `cutoffs=` is given, and centred at the bin medians;
- covariances are HC1 (`vce="robust"`, the default) or classical
  (`vce="homoscedastic"`); the kernel estimator also takes `vce="bootstrap"`;
- the kernel bandwidth adapts to the moderator's density exactly as the
  R package does, using a port of R's `stats::density()` (linear binning
  + FFT, `old.coords = FALSE`); `adaptive=False` gives the fixed Gaussian
  kernel of the Stata command;
- the Wald test interacts the covariates with the bins and uses the
  chi-square reference (R); `wald_full_moderate=False, wald_test="F"`
  reproduces Stata's `r(pwald)`.

Reference: Hainmueller, Mummolo and Xu (2019), *Political Analysis*
27(2), 163--192, `hainmueller2019much`.
