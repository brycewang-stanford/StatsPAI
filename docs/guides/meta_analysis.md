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

Switch the headline model with `method="fixed"`.

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

---

## 4. The forest plot

```python
m.forest_plot()             # per-study CIs + pooled diamond
```

---

## Notes & limitations

- DerSimonian-Laird is the classic random-effects estimator; for very few
  studies or rare events, REML / Paule-Mandel `tau^2` and Hartung-Knapp
  CIs are more conservative and are on the roadmap.
- Effect sizes must be supplied on an additive scale (log-transform ratio
  measures before pooling, then exponentiate the pooled estimate for
  reporting).

## Where to next

- [Power & sample size for epidemiological designs](power_epi.md)
- [Mendelian randomization](mendelian_family.md) — summary-data MR reuses
  the same inverse-variance machinery for genetic instruments.
