# Meta-analysis (evidence synthesis)

> **The output half of a systematic review.** Once you have extracted an
> effect size and its standard error from each study, `sp.meta_analysis`
> pools them — fixed-effect and random-effects — and reports the
> heterogeneity statistics, a prediction interval, Egger's test for
> small-study effects, and forest / funnel plots
> [@dersimonian1986meta; @higgins2002quantifying; @egger1997bias].

This is **summary-data** meta-analysis: you pass per-study effects and
SEs (log odds ratios, mean differences, log hazard ratios, …). It does
not fit individual-participant-data models.

---

## 1. One call

```python
import statspai as sp

# five studies' log odds ratios and their standard errors
effects = [0.10, 0.25, -0.05, 0.30, 0.15]
ses     = [0.05, 0.10,  0.08, 0.12, 0.06]

m = sp.meta_analysis(effects, ses, labels=["Trial A", "B", "C", "D", "E"])
print(m.summary())
```

The summary reports **both** models — you do not have to choose blind:

- **Fixed-effect (inverse-variance)** — assumes one common true effect.
- **Random-effects (DerSimonian-Laird)** — assumes the true effect varies
  across studies; wider CI, and the model StatsPAI reports by default
  (`method="DL"`).

Switch the headline model with `method="fixed"`, or estimate the
between-study variance by restricted maximum likelihood with
`method="REML"` (the default of R `metafor::rma` and Stata `meta`), by
maximum likelihood with `method="ML"`, or with the Paule-Mandel estimator
`method="PM"` (the value at which the generalised Q statistic equals its
degrees of freedom).

With few studies the normal interval of the random-effects estimate is
too narrow. `hksj=True` applies the Hartung-Knapp-Sidik-Jonkman
adjustment (`rma(test = "knha")`): a variance estimate from the weighted
residuals and a t distribution with `k - 1` degrees of freedom.

```python
r = sp.meta_analysis(effects, se, method="REML", hksj=True)
```

---

## 2. Heterogeneity — is pooling even sensible?

```python
m.q, m.q_pvalue     # Cochran's Q and its chi-square p-value
m.i2                # I^2 (fraction of variation due to heterogeneity)
m.tau2              # between-study variance estimate
m.prediction_interval   # where a *future* study's true effect is expected
```

A high `I^2` (say > 50%) or a small Q p-value warns that a single pooled
number hides real between-study differences — report the random-effects
estimate **and** the prediction interval, and consider a subgroup or
meta-regression analysis. The prediction interval is typically much wider
than the confidence interval of the pooled mean, which is the honest way
to convey heterogeneity.

---

## 3. Publication bias / small-study effects

```python
egg = m.egger_test()        # {'intercept', 'se', 't', 'p_value', 'df'}
print(egg["p_value"])

m.funnel_plot()             # visual asymmetry check
```

Egger's test regresses the standard normal deviate on precision; a
non-zero intercept flags funnel-plot asymmetry (often small-study or
publication bias). Treat it as a screen, not proof — asymmetry has many
possible causes.

### Trim and fill

`sp.trim_and_fill` (or `m.trim_and_fill()`) is the Duval-Tweedie
adjustment, R `metafor::trimfill`. It estimates from the ranks of the
centred effects how many studies are missing from one side of the funnel,
imputes them as mirror images of the most extreme studies on the other
side, and pools again.

```python
tf = sp.trim_and_fill(effects, ses)          # or m.trim_and_fill()
tf.k0, tf.side            # number of imputed studies and their side
tf.estimate, tf.ci        # pooled effect after filling
tf.original.estimate      # pooled effect of the observed studies
tf.funnel_plot()          # imputed studies as hollow markers
```

The side is chosen from the sign of the slope of the effect on its
standard error unless you pass `side="left"` or `side="right"`. The
estimator of the number of missing studies is `"L0"` by default; `"R0"`
and `"Q0"` are available. Read the filled estimate as a sensitivity
analysis: heterogeneity produces asymmetry too.

---

## 4. The forest plot

```python
m.forest_plot()             # per-study CIs + pooled diamond
```

---

## 5. Meta-regression

When heterogeneity is large, ask what explains it. `sp.meta_regression`
regresses the study effects on study-level moderators with a residual
between-study variance, R `metafor::rma(mods=)` and Stata `meta regress`.

```python
import pandas as pd

mods = pd.DataFrame({"dose": [2.0, 4.5, 1.0, 6.0, 3.5],
                     "blind": [1, 0, 1, 0, 1]})
mr = sp.meta_regression(effects, ses, mods["dose"], method="REML")
print(mr.summary())

mr.params, mr.std_errors      # intercept first
mr.tau2, mr.i2, mr.r2         # residual heterogeneity, share explained
mr.qm, mr.qm_pvalue           # omnibus test of the moderators
mr.qe, mr.qe_pvalue           # test of residual heterogeneity
mr.predict({"dose": [2, 5]})  # average effect at moderator values
```

`method` takes `"REML"` (default), `"DL"`, `"ML"`, `"PM"` or `"fixed"`;
`hksj=True` gives Knapp-Hartung t and F tests. Moderators must be numeric,
so code a categorical one as dummies. With few studies per moderator the
estimates are fragile; ten studies per moderator is a common rule of thumb.

---

## Notes & limitations

- DerSimonian-Laird is the classic random-effects estimator; for few
  studies prefer `method="REML"` with `hksj=True`.
- The iterative estimators (`"REML"`, `"ML"`, `"PM"`) are run to machine
  precision. `metafor` and Stata stop at a relative change of about 1e-5
  by default, so their printed `tau^2` agrees to about five digits unless
  you tighten their tolerances. `metafor::trimfill` always refits with the
  default tolerance.
- In a meta-regression without an intercept (`intercept=False`) the
  residual heterogeneity statistic and `tau^2` are those of Stata
  `meta regress, noconstant`. `metafor` 5.0.1 centres the effects before
  computing them, which gives different numbers in that case only.
- The prediction interval is `mu +/- t(k - 2) sqrt(se^2 + tau^2)`
  (Higgins, Thompson and Spiegelhalter); `metafor`'s default uses the
  normal quantile and is narrower. `I^2` and `H^2` are the Q-based ones
  under every `method`.
- Effect sizes must be supplied on an additive scale (log-transform ratio
  measures before pooling, then exponentiate the pooled estimate for
  reporting).

## Where to next

- [Power & sample size for epidemiological designs](power_epi.md)
- [Mendelian randomization](mendelian_family.md) — summary-data MR reuses
  the same inverse-variance machinery for genetic instruments.
