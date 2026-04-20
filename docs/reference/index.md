# API Reference — Overview

StatsPAI exposes 550+ top-level functions under a single `import statspai as sp`
namespace. Reference pages are grouped by methodological area:

| Area | Page | Flagship functions |
| --- | --- | --- |
| Difference-in-differences | [did](did.md) | `callaway_santanna`, `aggte`, `sun_abraham`, `bjs`, `dcdh`, `etwfe`, `cs_report`, `honest_did`, `breakdown_m` |
| Regression discontinuity | [rd](rd.md) | `rdrobust`, `rd2d`, `rkd`, `rdit`, `rdhonest`, `rdrandinf`, `rdpower`, `rd_forest`, `rdsummary` |
| Synthetic control | [synth](synth.md) | `synth`, `sdid`, `ascm`, `bayesian_synth`, `bsts_synth`, `penscm`, `synth_compare`, `synth_recommend`, `synth_report` |
| Decomposition | [decomposition](decomposition.md) | `decompose`, `oaxaca`, `gelbach`, `ffl_decompose`, `dfl_decompose`, `machado_mata`, `shapley_inequality`, `gap_closing` |
| Stochastic frontier | [frontier](frontier.md) | `frontier`, `xtfrontier`, `zisf`, `lcsf`, `malmquist`, `translog_design`, `te_summary` |
| Multilevel / mixed-effects | [multilevel](multilevel.md) | `mixed`, `melogit`, `mepoisson`, `meglm`, `megamma`, `menbreg`, `meologit`, `icc`, `lrtest` |
| Double / debiased ML | [dml](dml.md) | `dml` (PLR / IRM / PLIV), cross-fitting, influence-function SEs |
| Causal ML | [causal](causal.md) | `causal_forest`, `s_learner` … `dr_learner`, `tmle`, `tarnet`, `dragonnet`, `notears`, `policy_tree`, `bcf` |
| Sensitivity | [sensitivity](sensitivity.md) | `oster`, `sensemakr`, `e_value`, `rosenbaum_bounds`, `manski_bounds`, `spec_curve`, `robustness_report` |
| Smart workflow | [smart](smart.md) | `recommend`, `compare_estimators`, `assumption_audit`, `verify`, `verify_benchmark` |
| Spatial econometrics | [spatial](spatial.md) | `spatial_weights`, `moran_i`, `geary_c`, `sar`, `sem`, `sdm`, `gwr`, `mgwr`, `spatial_panel` |
| Time series | [timeseries](timeseries.md) | `arima`, `var`, `bvar`, `garch`, `cointegration`, `local_projections`, `structural_break` |
| Survival | [survival](survival.md) | `cox`, `aft`, `frailty`, `kaplan_meier`, `log_rank_test`, `competing_risks` |

Every result object follows the same contract:

```python
r = sp.someestimator(...)
r.summary()          # text table
r.plot()             # matplotlib figure
r.to_latex()         # LaTeX snippet
r.to_docx()          # Word paragraph
r.cite()             # BibTeX for the method's primary reference
r.to_markdown()      # Markdown table (most results)
```

Agent-native discovery:

```python
sp.list_functions(category='rd')         # list RD-family methods
sp.describe_function('rdrobust')         # natural-language docstring snippet
sp.function_schema('dml')                # JSON schema of arguments + returns
```
