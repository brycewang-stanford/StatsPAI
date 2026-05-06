# Choosing an IV estimator

## 0. TL;DR flowchart

```
How strong is your instrument?
  Strong (F > 10)    -> 2SLS (sp.ivreg)
  Weak   (F < 10)    -> LIML / Fuller (more robust to weak instruments)
  Very weak          -> Anderson-Rubin / weak-IV CIs (sp.weak_iv_ci)

Do you have MANY instruments?
  Few (1-3)           -> 2SLS / LIML
  Many                -> JIVE / Post-Lasso IV (sp.post_lasso, sp.jive_variants)

Do you have concerns about INSTRUMENT EXOGENEITY?
  Tight exogeneity    -> 2SLS
  Plausibly exogenous -> sp.plausibly_exogenous (Conley-Hansen-Rossi)
  Untestable          -> sp.bounds (bounds analysis)
```

> **For an integrated post-2022 reporting bundle in a single call** —
> 2SLS + bootstrap SEs + Olea–Pflueger effective F + LMMP tF
> critical value + Anderson–Rubin / CLR / K confidence sets +
> Kleibergen–Paap rk + Conley–Hansen–Rossi LTZ sensitivity +
> the Blandhol-et-al. *TSLS-as-LATE* caveat — use `sp.iv.iv_diag(...)`.
> See §11 below.

## 1. The default: 2SLS with robust SE

```python
r = sp.ivreg('y ~ x1 + x2 + (d ~ z1 + z2)', data=df, robust='hc1')
```

Always report:
- First-stage F (rule of thumb: > 10; ideal > 30)
- Endogeneity test (Hausman / Durbin-Wu-Hausman)
- Overidentification (Sargan-Hansen J) if multiple instruments

```python
r.diagnostics  # contains First-stage F, Hausman p, Sargan J p
```

## 2. Weak instruments

If first-stage F < 10:

| Method                        | Call                                          |
|-------------------------------|-----------------------------------------------|
| LIML (less biased than 2SLS) | `sp.liml(df, y, x_endog=[...], z=[...])`      |
| Fuller (finite-sample adj.)   | `sp.liml(..., fuller=1)`                      |
| Anderson-Rubin test           | `sp.anderson_rubin_test(r)`                   |
| Weak-IV robust CI             | `sp.weak_iv_ci(r, method='ar')`               |
| tF critical values (Lee 2022) | `sp.tF_critical_value(first_stage_F=F)`       |
| Effective F (Montiel-Pflueger)| `sp.effective_f_test(r)`                      |

**Do not report the 2SLS point estimate if F < 10 without flagging
weak-IV.** Use the Anderson-Rubin CI, which is robust to any
first-stage strength.

## 3. Many instruments

With more than 5-10 instruments relative to sample size, 2SLS has
many-weak-instrument bias. Use:

```python
r = sp.post_lasso(df, y='y', d='d', z=['z1', 'z2', ..., 'z50'])
# Or JIVE (jackknife IV)
r = sp.jive_variants(df, y='y', d='d', z=[...], variant='jive2')
```

## 4. Plausibly exogenous instruments

If the exclusion restriction is not dead-certain, use Conley-Hansen-Rossi:

```python
r = sp.plausibly_exogenous(df, y='y', d='d', z='z',
                           gamma_range=(-0.1, 0.1))
# Shows the sensitivity of the causal conclusion to
# a direct Z -> Y channel of size gamma.
```

## 5. Marginal treatment effects (MTE)

If you want **more than the LATE** — e.g., ATE, ATT, PRTE — estimate
the MTE curve:

```python
r = sp.mte(df, y='y', d='d', z='z', covariates=[...])
r.plot()        # MTE curve over unobserved heterogeneity
r.summary()     # ATE, ATT, LATE all derived from the MTE
```

Or the Mogstad-Santos-Torgovitsky linear program:

```python
r = sp.ivmte_lp(df, y='y', d='d', z='z',
                target='ATE', bounds=True)
```

## 6. Fuzzy / discrete treatments

For a binary endogenous variable with a binary instrument, the Wald
ratio is the LATE for compliers:

```python
# Manually:
wald = ((df.loc[df.z==1, 'y'].mean() - df.loc[df.z==0, 'y'].mean()) /
        (df.loc[df.z==1, 'd'].mean() - df.loc[df.z==0, 'd'].mean()))
# Equivalent to:
sp.ivreg('y ~ (d ~ z)', data=df).params['d']
```

Validate the LATE interpretation:
```python
sp.kitagawa_test(df, y='y', d='d', z='z')  # Imbens-Rubin test
```

## 7. Shift-share / Bartik IV

```python
r = sp.bartik(df, y='y_growth', shares='industry_shares_t0',
              shocks='industry_shocks', unit='region', time='year')
```

With proper inference via Adao-Kolesar-Morales (2019) shift-share SE:
```python
r_se = sp.shift_share_se(r)
```

## 8. Reading the output

```python
r = sp.ivreg('y ~ x + (d ~ z)', data=df)
r.params          # Series with coefficients
r.std_errors      # Series with SEs
r.diagnostics     # {'First-stage F (d)': 45.2, 'Hausman p': 0.03, ...}
r.tidy()          # Long-format table including intercept + endogenous + exog
r.glance()        # nobs, method, log_likelihood, first-stage F if computed
r.predict(new_df) # Out-of-sample prediction
```

## 9. Non-parametric IV (v1.2)

For a **continuous treatment** where the linear `D ~ Z` first stage is
too restrictive, v1.2 ships two non-parametric alternatives:

| Goal                                                    | Call                                                             |
|---------------------------------------------------------|------------------------------------------------------------------|
| Structural function `h*(d)` + **uniform** bootstrap CI  | `sp.kernel_iv(df, y, treat, instrument, n_boot=200)`             |
| LATE on the **maximal complier class** (continuous Z)   | `sp.continuous_iv_late(df, y, treat, instrument, n_quantiles=5)` |

- `sp.kernel_iv` (Lob et al. 2025, arXiv:2511.21603) delivers a uniform
  confidence band over the whole `D`-grid, not just pointwise.
- `sp.continuous_iv_late` (Zeng et al. 2025, arXiv:2504.03063) identifies
  the LATE on the subpopulation most responsive to the instrument —
  the natural generalisation of Angrist-Imbens LATE beyond binary `Z`.

See [v1.2 frontier estimators](v1_2_frontier.md) for a detailed walkthrough.

## 10. Sanity checks across estimators (`sp.iv.iv_compare`)

Compare 2SLS to bias-corrected k-class and JIVE estimators in one call:

```python
table = sp.iv.iv_compare(
    "wage ~ (educ ~ nearc4 + nearc2) + exper + south",
    data=df,
    methods=("2sls", "liml", "fuller", "jive", "ujive"),
)
print(table)
sp.iv.plot.plot_iv_forest(table, reference=ols_estimate)
```

Wide divergence between LIML and 2SLS (especially with many instruments)
is a flag for many-weak-IV bias.

## 11. Modern reporting bundle (`sp.iv.iv_diag`)

The single highest-leverage IV addition in StatsPAI v1.14 is
`sp.iv.iv_diag` — a Python port of R `ivDiag`
[@lal2024much] that consolidates the post-2022
reporting standard into one call:

```python
r = sp.iv.iv_diag(
    df, y="wage", endog="educ",
    instruments=["nearc4", "nearc2"],
    exog=["exper", "south"],
    n_boot=1000,
    boot_methods=("pairs", "wild"),
    include_clr_ci=True,
    include_k_ci=True,
    ltz_gamma_sd=0.05,        # CHR (2012) plausibly-exogenous prior
    cluster="state",          # cluster-bootstrap + cluster-robust SE
    random_state=42,
)

print(r.summary())            # publication-ready text panel
r.to_frame()                  # tidy DataFrame, one row per CI method
r.plot("diagnostic")          # 2x2 panel: first-stage / AR / forest / leverage
r.plot("forest")              # forest plot of all CI methods
r.plot("weak_iv")             # AR / CLR / K / tF overlay
r.to_latex(caption="IV bundle", label="tab:iv")
r.to_excel("iv_bundle.xlsx")
r.to_word("iv_bundle.docx")
```

What you get back:

| Block          | Statistic                                                   | Reference                                                                                |
| -------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Strength       | First-stage F, Olea–Pflueger effective F, KP rk LM/F        | [@olea2013robust; @kleibergen2006generalized]                                            |
| Inference      | 2SLS analytic SE, pairs / wild bootstrap                    | [@young2022consistency; @cameron2008bootstrap]                                           |
| Weak-IV-robust | AR set, optional CLR / K, tF-corrected CI                   | [@anderson1949estimation; @moreira2003conditional; @kleibergen2002pivotal; @lee2022valid]|
| Sensitivity    | Conley–Hansen–Rossi LTZ CI                                  | [@conley2012plausibly]                                                                   |
| Caveats        | TSLS-as-LATE warning when exog is present + endog is binary | [@blandhol2025tsls; @sloczynski2024should]                                               |
| Comparator     | OLS for OLS-vs-IV gap (informative, not causal)             | [@young2022consistency]                                                                  |

Why bundle? The post-2022 picture (Keane & Neal 2024
[@keane2024practical]; Lal et al. 2024 [@lal2024much]; Young 2022
[@young2022consistency]) is unanimous: a single F is not enough.
A modern IV report needs the **F + AR + tF + bootstrap + LTZ + (when
applicable) MTE caveat** stack. `iv_diag` produces all of that, and
`r.to_frame()` is already in publication-table shape.

## 12. Sanity checks

Every IV paper should include these:

```python
r = sp.ivreg('y ~ (d ~ z)', data=df)

# 1. First-stage strength
assert r.diagnostics['First-stage F (d)'] > 10

# 2. Reduced-form visualisation — does Z predict Y at all?
sp.regress('y ~ z', data=df).tidy()  # sign should match expected IV direction

# 3. Overidentification (if multiple Zs)
# The Sargan / Hansen J p-value must be > 0.05 (fail to reject exogeneity)

# 4. Compare OLS to IV — large divergence can indicate selection
ols = sp.regress('y ~ d', data=df)
iv = sp.ivreg('y ~ (d ~ z)', data=df)
print(f"OLS: {ols.params['d']:.3f}  IV: {iv.params['d']:.3f}")
```

<!-- AGENT-BLOCK-START: iv -->

## For Agents

**Pre-conditions**
- formula includes the (endog ~ instruments) parenthesised block
- at least as many instruments as endogenous regressors (order condition)
- instruments are not themselves endogenous in the outcome equation

**Identifying assumptions**
- Relevance: instruments predict the endogenous regressor (first-stage F ≥ 10 rule of thumb)
- Exclusion: instruments affect outcome only through the endogenous regressor
- Monotonicity (for LATE interpretation under heterogeneous effects)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| First-stage F < 10 (Stock-Yogo 5% bias) | `AssumptionWarning` | Use weak-IV-robust inference (Anderson-Rubin) or LIML. | `sp.anderson_rubin_ci` |
| Over-identification test rejects (sp.estat 'overid') | `AssumptionViolation` | At least one instrument is invalid; drop instruments or switch to just-identified LIML. | `sp.iv` |
| Hausman endogeneity test fails to reject | `AssumptionWarning` | OLS may be consistent and more efficient; report both. | `sp.regress` |
| Many instruments (≥ 10) cause many-IV bias | `NumericalInstability` | Use LIML or JIVE which are robust to many weak instruments. | `sp.iv` |

**Alternatives (ranked)**
- `sp.deepiv`
- `sp.bartik`
- `sp.proximal`
- `sp.regress`

**Typical minimum N**: 100

<!-- AGENT-BLOCK-END -->
