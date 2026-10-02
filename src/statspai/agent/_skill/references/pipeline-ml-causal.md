# Mode B: ML causal inference pipeline

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (for a function that takes `**kwargs`, to its agent schema or forwarding target), and the result attributes on the gate's smoke-fit list exist.

## §B. ML causal inference pipeline (Mode B)

> **Convention**: estimand-first, doubly-robust, ML-nuisance-learned, with **CATE distribution + policy value** as first-class outputs (not just a single ATE). The skill mirrors the AER skeleton but the Step-4 estimator stack is **DML + meta-learners + causal forest + neural-causal + BCF**, and Step-5 always reports a CATE distribution. Uncertainty is quantified by **conformal prediction** (`sp.conformal_causal`), not just normal-approximation SE.

Running example: a marketing uplift study — `treatment = personalized_offer`, `outcome = revenue_30d`, with 80+ covariates including text features (`prior_browsing_text`).

### B.0 Prep + nuisance super-learner

```python
import statspai as sp

# 0.1 Train/holdout split — DML uses cross-fitting internally, but holdout is for policy eval.
# statspai doesn't expose its own splitter; use sklearn directly.
from sklearn.model_selection import train_test_split
train, holdout = train_test_split(df, test_size=0.2, stratify=df["treatment"], random_state=42)

# 0.2 Nuisance learners. IMPORTANT: `sp.dml` / `sp.metalearner` do NOT accept a
# `sp.super_learner(...)` object — pass a scikit-learn estimator OBJECT, or (for `dml`
# only) a string alias from {'gbm','rf','lasso','ridge','linear','xgb','lgbm'}.
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                              RandomForestRegressor, RandomForestClassifier)
g_outcome = GradientBoostingRegressor()       # nuisance E[Y|X]  (a sklearn estimator)
g_treat   = GradientBoostingClassifier()      # nuisance E[D|X]  (propensity)

# `sp.super_learner` is a separate STANDALONE stacked predictor (it returns a fitted
# SuperLearner with .predict / .predict_proba) — use it for a reward model in OPE (B.4)
# or for your own predictions, NOT as the nuisance argument to dml/metalearner.
sl_reward = sp.super_learner(X=train[X_cols].values, y=train["revenue_30d"].values,
                             library=[LassoCV(), GradientBoostingRegressor(), RandomForestRegressor()],
                             n_folds=5, task="regression")
```

### B.1 Estimand & DAG learning (Step 2 + 2.5 in ML key)

```python
# estimand is an UPPERCASE enum: 'ATE'|'ATT'|'ATU'|'LATE'|'CATE'|'ITT'. The strategy
# is set via design=/estimand= on causal_question; q.identify() takes NO arguments.
q = sp.causal_question(treatment="treatment", outcome="revenue_30d", data=train,
                       population="marketed users", estimand="ATE",
                       design="selection_on_observables", covariates=X_cols)
plan = q.identify()

# DAG learning (when domain DAG isn't given)
proposed = sp.llm_dag_propose(variables=X_cols + ["treatment","revenue_30d"],
                              domain="e-commerce uplift")
constrained = sp.pc_algorithm(train[X_cols + ["treatment","revenue_30d"]],
                              variables=X_cols + ["treatment","revenue_30d"], alpha=0.05)
validated = sp.llm_dag_validate(proposed, train, alpha=0.05)   # (dag, data) positional
# Alternative learners: sp.notears(...), sp.causal_discovery(..., method="ges")
```

### B.2 Estimator stack — DML / meta-learner / GRF / neural / Bayesian

```python
# (1) DML — Chernozhukov double machine learning.
# Nuisance kwargs are `model_y` (outcome) and `model_d` (treatment) — NOT ml_g/ml_m.
# Each takes a sklearn estimator OR a string alias ('gbm'/'rf'/'lasso'/'xgb'/...).
dml = sp.dml(train, y="revenue_30d", d="treatment", X=X_cols,
             model="plr",                    # plr / irm / iv / pliv
             model_y=g_outcome, model_d=g_treat, n_folds=5)

# (2) Meta-learners — S / T / X / R / DR. outcome_model/propensity_model take sklearn
# estimator OBJECTS (not strings); omit them for sensible defaults.
ml_dr = sp.metalearner(train, y="revenue_30d", treat="treatment", covariates=X_cols,
                       learner="dr",         # 's' / 't' / 'x' / 'r' / 'dr'
                       outcome_model=GradientBoostingRegressor(),
                       propensity_model=GradientBoostingClassifier())

# (3) Causal forest (GRF / honest splits)
cf = sp.causal_forest("revenue_30d ~ treatment | " + " + ".join(X_cols),
                       train, n_estimators=2000, honest=True)

# (4) Neural causal — Dragonnet / TARNet / CEVAE. REQUIRES torch: pip install statspai[neural].
# Omit this block (and the neural columns below) if torch is not installed.
dn   = sp.dragonnet(train, y="revenue_30d", treat="treatment", covariates=X_cols,
                    repr_layers=(200,100), head_layers=(100,))
tar  = sp.tarnet  (train, y="revenue_30d", treat="treatment", covariates=X_cols)

# (5) Bayesian causal forest (full posterior over CATE)
bcf  = sp.bcf(train, y="revenue_30d", treat="treatment", covariates=X_cols,
              n_trees_mu=200, n_trees_tau=50)

# (6) Panel matrix completion (when units × periods)
mc   = sp.matrix_completion(panel_df, y="revenue", d="treatment", unit="user_id", time="week")

# Convergent evidence table — same regtable / collect stack. CausalResult AND CausalForest
# both flow into regtable; drop `dn` if you skipped the neural block.
rt = sp.regtable(dml, ml_dr, cf, dn, bcf,
                 model_labels=["(1) DML-PLR","(2) DR-Learner","(3) Causal forest",
                               "(4) Dragonnet","(5) BCF"],
                 stats=["N","ATE","CATE 5–95% range","Cross-fit folds","Nuisance R²"],
                 title="Table 2. ATE — ML estimator horse race")
rt.to_word ("tables/table2_ml.docx"); rt.to_excel("tables/table2_ml.xlsx")
```

### B.3 CATE distribution & subgroup view (the ML-causal headline)

```python
# 3.1 Per-row CATE. Plotters return (fig, ax). The raw per-row CATE vector lives at
# ml_dr.model_info["cate"] (an ndarray) — there is NO .cate_estimates attribute.
fig, ax = sp.cate_plot(ml_dr, kind="hist",
                       title="Figure B1. CATE distribution — DR-Learner")
fig.savefig("figures/figB1_cate_dist.png", dpi=300)

# 3.2 CATE by group (skill quartiles, gender, channel, …)
g = sp.cate_by_group(ml_dr, train, by="customer_value_quartile", n_groups=4)
fig, ax = sp.cate_group_plot(g, title="Figure B2. CATE by customer-value quartile")
fig.savefig("figures/figB2_cate_group.png", dpi=300)

# 3.3 Causal-forest local effects. CausalForest has no .local_effects(); get the per-row
# CATE vector with cf.effect(X) (ndarray) and plot it yourself.
import numpy as np, matplotlib.pyplot as plt
tau = cf.effect(train[X_cols].values)                # per-row CATE, length n
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(tau, bins=40); ax.set_xlabel("Causal-forest CATE"); ax.set_title("Figure B3. CF local effects")
fig.savefig("figures/figB3_local.png", dpi=300)
```

### B.4 Policy learning + off-policy evaluation

```python
import numpy as np

# 4.1 Learn an interpretable policy tree from CATE estimates. The result is dict-like
# with .plot_tree() (NOT .plot()), .summary(), .to_latex(), .to_excel(). plot_tree → (fig, ax).
pol_tree = sp.policy_tree(train, y="revenue_30d", d="treatment", X=X_cols, max_depth=3)
fig, ax = pol_tree.plot_tree()
fig.savefig("figures/figB4_policy.png", dpi=300)

# 4.2 Safe policy under a cost constraint. `state` and `action` must each be a SINGLE
# DISCRETE column name (not a list of feature columns). Encode the state into one
# discrete segment column first if you have many features.
train = train.assign(segment=train["customer_value_quartile"])      # one discrete state col
safe = sp.offline_safe_policy(train, state="segment", action="treatment",
                              reward="revenue_30d", cost="offer_cost", cost_threshold=2.50)

# 4.3 Off-policy evaluation on holdout — IPS / DR / SNIPS.
# sp.ope exposes ips / direct_method / doubly_robust / snips / switch_dr. CRITICAL shapes:
#   pi_b, pi_e are (n, K) probability matrices over K actions (one-hot for deterministic);
#   reward_model is a CALLABLE reward_model(X, a) -> length-n predicted reward.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
g_t = GradientBoostingClassifier().fit(train[X_cols], train["treatment"])
g_r = GradientBoostingRegressor().fit(
    np.column_stack([train[X_cols].values, train["treatment"].values]), train["revenue_30d"])

X_test = holdout[X_cols].values
A_test = holdout["treatment"].to_numpy(int)
R_test = holdout["revenue_30d"].to_numpy(float)
p1     = g_t.predict_proba(X_test)[:, 1]
pi_b   = np.column_stack([1 - p1, p1])                              # (n, 2) behavior policy
a_e    = (g_r.predict(np.column_stack([X_test, np.ones(len(X_test))]))    # treat-if-uplift>0
          > g_r.predict(np.column_stack([X_test, np.zeros(len(X_test))]))).astype(int)
pi_e   = np.column_stack([1 - a_e, a_e]).astype(float)             # (n, 2) one-hot eval policy
reward_model = lambda X, a: g_r.predict(np.column_stack([X, np.full(len(X), a)]))
opv = sp.ope.doubly_robust(X_test, A_test, R_test, pi_b=pi_b, pi_e=pi_e,
                           reward_model=reward_model)
print(f"Policy value (DR): {opv.value:.3f} ± {opv.se:.3f}")
# IPS/SNIPS need no reward model: sp.ope.snips(A_test, R_test, pi_b=pi_b, pi_e=pi_e)
```

### B.5 Uncertainty + fairness + robustness

```python
# 5.1 Conformal prediction intervals on CATE — distribution-free coverage.
# sp.conformal_causal exposes conformal_cate / conformal_ite / conformal_continuous /
# conformal_fair / conformal_interference and more — pick by estimand.
cp = sp.conformal_causal.conformal_cate(train, y="revenue_30d", treat="treatment",
                                         covariates=X_cols, alpha=0.10)   # 90% PI

# 5.2 Subgroup fairness audit — DP / EO gaps across protected attributes.
# fairness_audit audits a BINARY classifier: BOTH `predictions` and `labels` must be 0/1
# columns. (A meta-learner result has no .predict — score with your own classifier.)
holdout = holdout.assign(
    targeted  = (g_t.predict_proba(holdout[X_cols])[:, 1] > 0.5).astype(int),   # binary decision
    responded = (holdout["revenue_30d"] > holdout["revenue_30d"].median()).astype(int),  # binary label
)
fair = sp.fairness.fairness_audit(holdout, predictions="targeted",
                                  protected="gender", labels="responded",
                                  threshold=0.10)

# 5.3 Sensitivity dashboard — a TEXT/numeric dashboard (.summary() + numeric attrs),
# NOT a figure (no .plot()/.savefig()).
print(sp.sensitivity_dashboard(dml, train).summary())

# 5.4 (Reuse AER §7 robustness) Spec curve over nuisance/control choices.
# se_types ∈ {'nonrobust','hc1'/'robust','cluster'}; sc.plot() → (fig, ax).
sc = sp.spec_curve(train, y="revenue_30d", x="treatment",
                   controls=[X_cols[:1], X_cols[:3], X_cols],
                   se_types=["nonrobust", "robust"])
fig, ax = sc.plot()
fig.savefig("figures/figB6_spec_curve.png", dpi=300)
```

### B.6 Reporting checklist (ML-causal-specific footer)

When producing the Table-2 footer, include — in addition to the AER stars/SE language:

- **Nuisance learners** used (e.g., "outcome: SuperLearner[xgb, rf, lasso, nn]; treatment: same")
- **Cross-fitting**: number of folds, sample-splitting scheme
- **Overlap diagnostic**: PS distribution range, `% trimmed`
- **CATE summary**: mean / 5–95% range / share with CATE > 0
- **Policy value**: off-policy DR value vs. random / vs. always-treat baselines
- **Conformal coverage**: empirical coverage of nominal 1−α PI on holdout
- **Fairness audit**: subgroup CATE gaps vs. acceptable thresholds

> **Doubly-robust DML / DR-Learner / TMLE are preferred over single-robust S- or T-learner alone.** Report S- or T-learner only as a baseline in the horse race. Always check overlap before reporting any IPW-flavored estimator.

---
