# Decomposition methods in StatsPAI

`sp.decomposition` ships **19 estimators** under a single dispatcher
`sp.decompose(method=...)` — covering mean, distributional, inequality,
demographic, and causal decomposition. This guide is a quick map of
what's available, when to reach for each estimator, and what the v1.15
polish gives you (unified `to_excel`/`to_word`/`cite`/`confint`,
publication-quality plots, and the Yu--Elwert (2025) frontier method).

## Choosing an estimator

| Question | Method | One-liner |
| --- | --- | --- |
| Decompose mean gap into endowment + coefficient | `oaxaca` | Blinder (1973), Oaxaca (1973) |
| Sequential OVB attribution | `gelbach` | Gelbach (2016) |
| Logit / probit (binary outcome) | `fairlie`, `bauer_sinning` | Fairlie (2005), Bauer & Sinning (2008) |
| Quantile or Gini gap | `rif`, `ffl` | Firpo--Fortin--Lemieux (2009, 2018) |
| Reweight one group's X to match the other's | `dfl` | DiNardo, Fortin & Lemieux (1996) |
| Counterfactual *quantile* function | `machado_mata`, `melly` | MM (2005), Melly (2005) |
| Counterfactual *distribution* | `cfm` | Chernozhukov, Fernández-Val & Melly (2013) |
| Inequality between vs. within groups | `subgroup` | Shorrocks (1980), Cowell & Flachaire (2007) |
| Inequality contribution per regressor | `shapley_inequality` | Shorrocks (2013) |
| Inequality by income source | `gini_source` | Lerman & Yitzhaki (1985) |
| Aggregate rate gap (categorical) | `kitagawa`, `das_gupta` | Kitagawa (1955), Das Gupta (1993) |
| What gap *would close* under intervention | `gap_closing` | Lundberg (2022) |
| Direct vs. mediated | `mediation` | VanderWeele (2014) |
| Disparity due to mediator | `disparity` (`causal_jvw`) | Jackson & VanderWeele (2018) |
| **Causal disparity → baseline + prevalence + effect + selection** | **`yu_elwert`** | **Yu & Elwert (2025)** |

When in doubt: a *mean* gap → `oaxaca`; a *distributional* gap → `ffl`;
a *causal* gap → `yu_elwert` if you have a binary intermediate
treatment, otherwise `gap_closing` for an aggregate intervention story.

## Yu--Elwert (2025) — the frontier method

`sp.yu_elwert_decompose(...)` (alias `sp.decompose("yu_elwert", ...)` or
`sp.decompose("cdgd", ...)`) decomposes a group disparity
$D = E[Y \mid R{=}1] - E[Y \mid R{=}0]$
operating through a binary treatment $T$ into four mechanisms:

$$
D = \underbrace{(E[Y(0) \mid R{=}1] - E[Y(0) \mid R{=}0])}_{\text{baseline}}
   + \underbrace{E_0[\tau]\big(E[T \mid R{=}1] - E[T \mid R{=}0]\big)}_{\text{prevalence}}
   + \underbrace{E[T \mid R{=}1]\big(E_1[\tau] - E_0[\tau]\big)}_{\text{effect}}
   + \underbrace{\operatorname{Cov}_1(T, \tau) - \operatorname{Cov}_0(T, \tau)}_{\text{selection}}
$$

Identification requires only conditional ignorability of $T$ given
$(R, X)$ — *not* that $R$ itself is unconfounded — which is what makes
this approach distinct from causal-mediation methods. The "selection"
piece is the novel mechanism: it captures whether the right people
(those with the largest individual gain) end up treated within each
group.

Two estimators are built in:

- `method="plugin"` (default): within-cell OLS for $m_{rt}(X)$ and
  within-group logit for $p_r(X)$, then plug-in expectations. Fast,
  algebraic, and the residual `disparity − Σ components` is *exactly*
  zero by construction.
- `method="efficient"`: doubly-robust augmented moments with implicit
  cross-fitting. Recommended when the nuisance models are flexible /
  regularised; the small implied residual reflects the asymmetric
  augmentations rather than a bug.

```python
import statspai as sp

r = sp.yu_elwert_decompose(
    data=df, y="y", treatment="t", group="r",
    x=["age", "educ", "exp"],
    method="plugin",
    n_boot=499,
)
r.summary()
fig, ax = r.plot()           # mechanism bar chart with 95% CI whiskers
r.to_word("yu_elwert.docx")  # full Word report
```

## What the v1.15 polish ships

Every result object in `sp.decomposition` (Oaxaca, Gelbach, RIF, FFL,
DFL, Machado--Mata, Melly, CFM, Fairlie, Bauer--Sinning, Yun, Kitagawa,
Das Gupta, three inequality classes, GapClosing, Mediation, Disparity,
Yu--Elwert) inherits `DecompResultMixin`, exposing the same surface:

| Method | What it does |
| --- | --- |
| `result.summary()` | Pretty-printed text summary with significance stars and CIs. |
| `result.plot(...)` | Method-specific publication plot (forest, waterfall, mechanism, quantile process, etc.). |
| `result.confint(alpha=0.05)` | Normal-approx CIs from stored standard errors. |
| `result.cite()` | Numbered bibliography of canonical references. |
| `result.cite("bibtex_keys")` | The `paper.bib` keys. |
| `result.to_dict()` / `result.to_json()` | JSON-serialisable snapshot. |
| `result.to_latex()` | LaTeX `tabular` for the report. |
| `result.to_excel("out.xlsx")` | Multi-sheet workbook (Overall, Detailed, …). |
| `result.to_word("out.docx")` | One-shot Word report (requires `python-docx`). |

## Plotting

`statspai.decomposition.plots` exposes a small toolkit that every
result class draws on:

- `detailed_waterfall(df, ...)` — sign-coloured horizontal bars with
  optional 95% CI whiskers.
- `forest_plot(df, ...)` — point estimates with CI lines; greys out
  rows whose CI crosses zero.
- `quantile_process_plot(result, show_ci=True)` — gap, composition,
  structure as functions of $\tau$ with shaded CI bands when SEs are
  available on the grid.
- `counterfactual_cdf_plot(result)` — observed vs. counterfactual CDFs
  for CFM-style decompositions.
- `mediation_forest(result)` — NDE / NIE / total with CIs.
- `yu_elwert_mechanisms_plot(result)` — disparity, baseline,
  prevalence, effect, selection as a single bar chart.
- `rif_heatmap(grid_df)` — variable × quantile contributions heatmap.

All plots share a single palette (`DECOMP_PALETTE`) and a despined
minimal style (`apply_decomp_style`), so a panel of figures from
different methods looks like part of the same report.

## Inference

Every estimator that supports inference accepts:

- **Analytical** (delta-method) standard errors when the formula is
  closed-form (Oaxaca, Gelbach, RIF, Yun, Bauer--Sinning, Kitagawa).
- **Bootstrap** (cluster-aware, percentile / basic / normal CIs) when
  the formula is intractable (DFL, FFL, Machado--Mata, Melly, CFM,
  GapClosing, Mediation, Disparity, Yu--Elwert).
- **Wild bootstrap** for residual-based statistics
  (`statspai.decomposition._common.wild_bootstrap_stat`) with
  Rademacher or Mammen weights and optional cluster IDs.

`sp.oaxaca` holds the covariate means fixed by default (Stata
`oaxaca ..., vce(robust) fixed`). `means="stochastic"` adds their sampling
variance, the default of Stata `oaxaca ..., vce(robust)` [@jann2008blinder]:

```python
r = sp.oaxaca(df, y="lwage", group="black", x=X, means="stochastic")
r.overall["explained_se"], r.overall["unexplained_se"], r.overall["gap_se"]
```

Point estimates are the same under both. On the Card (1995) black-white
wage gap the explained part has SE 0.0080 with fixed means and 0.0115
with stochastic means.

## Reference parity

The plug-in Oaxaca, RIF, FFL, DFL, and Kitagawa estimators are
numerically aligned with Ben Jann's Stata `oaxaca`, Fernando
Rios-Avila's `rif`/`ddecompose` (Stata) and `ddecompose` (R), and the
Stata `kob` package introduced in Jann's 2024 UK Stata Conference
update. The Yu--Elwert estimator is aligned with the R `cdgd` package.

## References

The bibliography backing every method (DOI-verified) lives in
`paper.bib`; you can dump the keys for a result with
`result.cite("bibtex_keys")` and pull the formatted strings with
`result.cite("list")`.

Headline recent papers behind the v1.15 polish:

- **Yu, A. & Elwert, F. (2025).** Nonparametric causal decomposition
  of group disparities. *Annals of Applied Statistics*, 19(1),
  821–845. doi:10.1214/24-AOAS1990.
- **Oaxaca, R. L. & Sierminska, E. (2025).** Oaxaca-Blinder meets
  Kitagawa: What is the link? *PLOS ONE*, 20(5), e0321874.
  doi:10.1371/journal.pone.0321874.
- **Park, S., Kang, S., & Lee, C. (2024).** Choosing an Optimal
  Method for Causal Decomposition Analysis with Continuous Outcomes.
  *Sociological Methodology*, 54(1), 92–117.
  doi:10.1177/00811750231183711.
- **Ahrens, A., Hansen, C. B., Schaffer, M. E., & Wiemann, T. (2025).**
  Model averaging and double machine learning. *Journal of Applied
  Econometrics*, 40(3), 249–269. doi:10.1002/jae.3103.
- **Kröger, H. & Hartmann, J. (2021).** Extending the
  Kitagawa-Oaxaca-Blinder decomposition approach to panel data.
  *Stata Journal*, 21(2), 360–410. doi:10.1177/1536867X211025800.
- **Rios-Avila, F. (2020).** Recentered influence functions (RIFs) in
  Stata. *Stata Journal*, 20(1), 51–94.
  doi:10.1177/1536867X20909690.

For the full canonical list, run `result.cite()` on any result object.
