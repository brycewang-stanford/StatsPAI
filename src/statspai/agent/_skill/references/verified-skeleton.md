# Verified skeleton

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (for a function that takes `**kwargs`, to its agent schema or forwarding target), and the result attributes on the gate's smoke-fit list exist.

## Verified skeleton (copy, then swap in your columns)

This minimal pipeline **runs start-to-finish against the installed StatsPAI** (every call below is executed by `validate_api_claims.py`). It is the golden path — adapt column names / design, keep the call shapes and the unpack-then-save figure idiom. The full playbook (§−1 → §8) expands each step.

```python
import numpy as np, pandas as pd, statspai as sp
# df has: wage, training(0/1), worker_id, firm_id, year, first_treat_year, age, edu, tenure, ...

# §1 Table 1 → Word/Excel/LaTeX
mc = sp.mean_comparison(df, ["age","edu","tenure"], group="training", test="ttest",
                        title="Table 1. Summary statistics")
mc.to_word("tables/table1.docx"); mc.to_excel("tables/table1.xlsx")

# §2 Estimand-first plan (freeze BEFORE estimating)
q = sp.causal_question(treatment="training", outcome="wage", data=df, estimand="ATT",
                       design="did", time_structure="panel", time="year", id="worker_id",
                       covariates=["age","edu","tenure"])
plan = q.identify(); print(plan.summary())

# §3 Identification figure — from a CS/SA result (NOT event_study()); plotters return (fig, ax)
cs = sp.callaway_santanna(df, y="wage", g="first_treat_year", t="year", i="worker_id", x=["age","edu"])
fig, ax = sp.enhanced_event_study_plot(cs, shade_pre=True); fig.savefig("figures/fig2a.png", dpi=300)

# §4 Main table — mix sp.regress (no FE) + sp.feols (HDFE, needs statspai[fixest]) in ONE regtable
M1 = sp.regress("wage ~ training", df, cluster="firm_id")
M2 = sp.feols("wage ~ training + age + edu + tenure | industry + year", df, vcov={"CRV1":"firm_id"})
rt = sp.regtable(M1, M2, template="aer", coef_labels={"training":"Job training"},
                 model_labels=["(1) OLS","(2) FE"], stats=["N","R2","Cluster","FE"],
                 title="Table 2. Effect of training on wages")
rt.to_word("tables/table2.docx"); rt.to_excel("tables/table2.xlsx")
open("tables/table2.tex","w").write(rt.to_latex())

# §5 Heterogeneity — per-row CATE at result.model_info["cate"] (there is NO .cate_estimates)
ml = sp.metalearner(df, y="wage", treat="training", covariates=["age","edu","tenure"], learner="dr")
fig, ax = sp.cate_plot(ml, kind="hist"); fig.savefig("figures/fig4.png", dpi=300)

# §7 Robustness — Oster + E-value + honest-DID sensitivity figure
sp.oster_bounds(data=df, y="wage", treat="training", controls=["age","edu","tenure"], r_max=1.3)
sp.evalue(estimate=M2.params["training"], ci=tuple(M2.conf_int().loc["training"]), measure="RR")
fig, ax = sp.sensitivity_plot(sp.honest_did(cs, method="smoothness"),
                              original_estimate=cs.estimate, original_ci=cs.ci)
fig.savefig("figures/fig6.png", dpi=300)

# §8 One-file replication bundle (Word/Excel/LaTeX/Markdown from one source)
c = sp.collect("Replication", template="aer")
c.add_summary(df, vars=["wage","age","edu","tenure"], stats=["mean","sd","n"], title="Table 1")
c.add_regression(M1, M2, model_labels=["(1)","(2)"], stats=["N","R2"], title="Table 2")
for ext in ("docx","xlsx","tex","md"): c.save(f"replication/paper.{ext}")
```

> Epi (§A) and ML-causal (§B) reuse this exact scaffolding — only the §4 estimator stack changes (TMLE/g-formula/MR for epi; DML/meta-learner/causal-forest for ML), and every estimator still returns a result that drops into `sp.regtable` / `sp.collect`.
