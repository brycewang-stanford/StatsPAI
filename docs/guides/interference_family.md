# Interference & spillover — the full family

> **SUTVA is dead, long live SUTVA.** When a unit's treatment affects
> other units' outcomes — vaccines, networks, cluster-based rollouts,
> shared markets — the standard ATE is not point-identified. StatsPAI
> ships a tight set of 8 estimators covering the three main designs
> (partial interference, network interference, cluster RCT) plus two
> orthogonal / HTE variants from 2025.

The Stable Unit Treatment Value Assumption (SUTVA) says: my outcome
depends only on **my** treatment. Violate it and everything downstream
breaks — ATT is off, standard errors are off, event-study pretrends
are off. The literature splits the problem into three tractable sub-
problems, each with its own estimator family:

| Design | Key assumption | Estimator |
|---|---|---|
| **Partial interference** | Clusters interfere internally but not across clusters | `sp.spillover` |
| **Network interference** | Arbitrary spillover along a known graph | `sp.network_exposure`, `sp.peer_effects`, `sp.network_hte`, `sp.inward_outward_spillover` |
| **Cluster RCT** | Clusters randomized as units, with possible cross-cluster leakage | `sp.cluster_matched_pair`, `sp.cluster_cross_interference`, `sp.cluster_staggered_rollout`, `sp.dnc_gnn_did` |

Every function below is at top level: `sp.spillover`,
`sp.network_exposure`, `sp.peer_effects`, `sp.network_hte`,
`sp.inward_outward_spillover`, `sp.cluster_matched_pair`,
`sp.cluster_cross_interference`, `sp.cluster_staggered_rollout`,
`sp.dnc_gnn_did`.

---

## 1. Partial interference (clusters) — `sp.spillover`

The Hudgens-Halloran (2008) / Aronow-Samii (2017) framework in its
simplest form. Observations sit in **clusters** (households,
classrooms, villages); treatment interferes within a cluster but not
across clusters. The estimand decomposes into:

- **Direct effect**: own treatment effect, holding peer exposure fixed.
- **Spillover effect**: peer-exposure effect, holding own treatment fixed.
- **Total effect**: direct + spillover.

```python
r = sp.spillover(
    data=df, y="infected", treat="vaccinated",
    cluster="household",
    covariates=["age", "comorbidity"],
    exposure_fn="fraction",     # or "any" / "count"
    n_bootstrap=500,
)
print(r.model_info["direct_effect"])     # own-vaccine effect
print(r.model_info["spillover_effect"])  # herd-immunity effect
print(r.model_info["total_effect"])
```

**When to use it**: you have a natural cluster structure and partial
interference is plausible (the canonical vaccine / classroom / village
setup). Cluster bootstrap SEs are built in.

References: Hudgens & Halloran (2008), *JASA*; Aronow & Samii (2017),
*Annals of Applied Statistics*.

---

## 2. Network interference — Aronow-Samii exposure mapping

### `sp.network_exposure` — Horvitz-Thompson under arbitrary interference

The general-network generalization of `spillover`. You supply:
- Observed outcomes `Y` and treatment vector `Z`.
- An **adjacency** (matrix, DataFrame, or edge list).
- An **exposure mapping** that reduces the full `Z` vector to a
  low-dimensional categorical exposure per unit (e.g. the Aronow-Samii
  4-cell partition `c00 / c10 / c01 / c11`).

The estimator is Horvitz-Thompson with exposure probabilities computed
by Monte-Carlo simulation under the known (Bernoulli) design:

```python
r = sp.network_exposure(
    Y=y, Z=z, adjacency=A,
    mapping="as4",     # or "fraction" (by share of treated neighbors)
    p_treat=0.3,
    n_sim=3000,
    seed=0,
)
print(r.contrasts)     # direct / spillover / composite contrasts
print(r.estimates)     # HT mean per exposure level
```

**When to use it**: you have the full graph and know the randomization
design (Bernoulli with known `p_treat`). SEs use the conservative
Aronow-Samii Theorem 1 bound. Not suitable for observational data
without a known design — for that see `network_hte` below.

References: Aronow & Samii (2017), *AOAS* 11(4).

### `sp.peer_effects` — linear-in-means (Manski / Bramoullé)

Manski's "reflection problem" resolved via the Bramoullé-Djebbari-Fortin
(2009) exclusion restriction. Fits the model

```
y_i = α + β * (peer mean of y) + γ * (peer mean of X) + δ * X_i + ε_i
```

by 2SLS using powers of the adjacency matrix (`W^2 X`, `W^3 X`) as
instruments for `W y`.

```python
r = sp.peer_effects(
    data=df, y="gpa",
    covariates=["parent_education", "female"],
    W=adjacency_matrix,
    include_contextual=True,   # include γ (contextual peer effect)
)
print(r.endogenous_peer)       # β — the reflection coefficient
print(r.contextual_peer)       # γ — per-covariate contextual effect
```

**When to use it**: you want to decompose β (endogenous / social
multiplier) from γ (contextual peer effect). Identification requires
at least one pair of nodes at network distance 2 that are not directly
connected — check with `(A @ A > 0) & ~A.astype(bool)`. Fails silently
on complete graphs.

References: Manski (1993), *REStud*; Bramoullé-Djebbari-Fortin (2009),
*JoE*; Lee (2007), *JoE*.

### `sp.network_hte` — orthogonal learning (Parmigiani et al. 2025)

For when you want **heterogeneous** direct and spillover effects, with
ML nuisance models for `E[Y|X]`, `E[D|X]`, `E[E|X]` (`E` = neighbor
exposure summary). Uses Chernozhukov-style double orthogonalization
with cross-fitting:

```python
r = sp.network_hte(
    data=df, y="y", treatment="d",
    neighbor_exposure="share_treated_neighbors",
    covariates=["x1", "x2"],
    n_folds=5, random_state=0,
)
print(r.direct_effect, r.direct_se)
print(r.spillover_effect, r.spillover_se)
```

**When to use it**: observational data, many covariates, you want ML
robust estimates of direct and spillover effects that are immune to
first-order nuisance misspecification. Requires a **pre-computed
scalar neighbor exposure summary** — you control the mapping (share of
treated neighbors is the usual choice). Needs `n ≥ 10 * n_folds`.

Citation: Parmigiani et al. (arXiv:2509.18484, 2025).

### `sp.inward_outward_spillover` — directed networks

In a **directed** network (e.g. citations, referrals, follower graphs)
the spillover onto unit `i` from its incoming neighbors need not equal
the spillover from `i` onto its outgoing neighbors. Li-Ratkovic et al.
(2025) partition spillover into inward and outward components:

```python
r = sp.inward_outward_spillover(
    data=df, y="y", treatment="d",
    inward_exposure="share_incoming_treated",
    outward_exposure="share_outgoing_treated",
    covariates=["x1", "x2"],
)
print(r.inward_effect, r.outward_effect, r.ratio_in_out)
```

**When to use it**: your graph is directed and you want to test "does
influence flow the same way both directions?"  A `ratio_in_out` far
from 1 is the signal of directional asymmetry.

Citation: Li, Ratkovic et al. (arXiv:2506.06615, 2025).

---

## 3. Cluster RCTs with interference — four designs

### `sp.cluster_matched_pair` — Bai (2022)

You pair clusters on baseline covariates before randomizing one of
each pair to treatment. Matched-pair designs need the Bai SE, not
naive cluster SE — paired differences cancel design variance that
the pooled formula double-counts.

```python
r = sp.cluster_matched_pair(
    data=df, y="y", cluster="village",
    treat="treated_village", pair="pair_id",
)
print(r.estimate, r.se, r.ci)
```

**When to use it**: your design is literally matched pairs. Each pair
must contain exactly two clusters with opposite treatment status.

Reference: Bai (2022), arXiv:2211.14903.

### `sp.cluster_cross_interference` — Ding et al. (2025)

Cluster RCT where clusters **are not isolated** — neighboring clusters
share infrastructure, labor markets, etc. The estimator adds a control
for the share of treated neighbors per cluster:

```python
r = sp.cluster_cross_interference(
    data=df, y="y", cluster="village",
    treat="treated_village",
    neighbour_treat_share="pct_treated_neighbors",  # precomputed
)
print(r.direct_effect, r.direct_se)
print(r.spillover_effect, r.spillover_se)
```

**When to use it**: your clusters are geographically or socially
adjacent and leakage is plausible. The neighbor treatment share must
be pre-computed from user-supplied adjacency.

Reference: Ding et al. (2025), arXiv:2310.18836.

### `sp.cluster_staggered_rollout` — Zhou et al. (2025)

Cluster RCT with **staggered** treatment adoption — different clusters
turn on at different calendar times. Produces a dynamic event-study
ATT robust to the standard two-way-fixed-effect contamination
(Goodman-Bacon 2021 / Callaway-Sant'Anna 2021 style):

```python
r = sp.cluster_staggered_rollout(
    data=df, y="y", cluster="village",
    time="month",
    first_treat="first_treatment_month",   # 0 = never-treated
    leads=2, lags=4,
)
print(r.overall_att, r.overall_se)
print(r.event_study)      # rel_time, att, ci_low, ci_high
```

**When to use it**: the policy rolls out cluster-by-cluster over time.
For individual-level staggered DiD without clustering, use
`sp.callaway_santanna` instead.

Reference: Zhou et al. (2025), arXiv:2502.10939.

### `sp.dnc_gnn_did` — double negative controls + GNN (Zhao et al. 2026)

When you suspect **unmeasured** network confounding (e.g. latent
homophily), combine double negative controls with a GNN embedding of
each unit's network position. The embedding is optional — pass any
pre-computed `(n_units, k)` array, or omit for a simple lagged-outcome
feature.

```python
r = sp.dnc_gnn_did(
    data=df, y="outcome", treat="first_treat_period",
    time="year", id="user_id",
    nc_outcome=["clicks_before_launch"],    # should NOT be moved by treatment
    nc_exposure=["prior_tenure"],           # should NOT cause outcome
    embedding=my_gnn_embedding,             # optional
    n_boot=100,
)
```

**When to use it**: you believe there's **unmeasured** confounding
rooted in network structure (position, centrality, latent communities).
Requires at least one valid negative-control outcome and one valid
negative-control exposure — chosen from domain knowledge.

Reference: Zhao et al. (2026), arXiv:2601.00603.

---

## Decision guide

```
I have clusters that don't interact across cluster boundaries
  → sp.spillover

I have the full network + a known randomization design
  → sp.network_exposure (HT + Aronow-Samii mapping)

I want to decompose β_endogenous vs γ_contextual peer effects
  → sp.peer_effects

I have a network + lots of covariates + observational data
  → sp.network_hte  (orthogonal / cross-fit)

My network is directed and I want in vs out asymmetry
  → sp.inward_outward_spillover

---

I ran a cluster RCT with matched pairs
  → sp.cluster_matched_pair (Bai SE)

My clusters leak across boundaries
  → sp.cluster_cross_interference

Treatment rolled out cluster-by-cluster over time
  → sp.cluster_staggered_rollout

I suspect unmeasured network confounding (homophily, latent groups)
  → sp.dnc_gnn_did
```

---

## Diagnostics every interference analysis should report

1. **Exposure balance**. Tabulate the realized exposure levels
   (`c00`, `c10`, `c01`, `c11`) — if any cell has fewer than ~30 units
   the HT estimator will be unstable. Visible in
   `network_exposure(...).estimates["n_at_level"]`.
2. **Identification check for `peer_effects`**. Compute `W2 = W @ W`
   and confirm there exists at least one `(i, j)` with `W2[i,j] > 0`
   and `W[i,j] == 0`. No such pair → 2SLS is under-identified.
3. **Overlap for observational network HTE**. For `network_hte`,
   check `ps_hat` quantiles — the Chernozhukov DR residual-on-residual
   regression blows up if `ps_hat ∈ {0, 1}` anywhere.
4. **Parallel trends with interference**. For
   `cluster_staggered_rollout`, the pre-trend mean in the event-study
   table should be close to zero in absolute value — if not, the
   identifying assumption fails and the ATT is biased (same logic as
   vanilla CS2021, just at the cluster level).
5. **Sensitivity to exposure mapping**. For `spillover`, re-run with
   `exposure_fn="any"` and `"count"` and check that `direct_effect` is
   qualitatively stable. If flipping the exposure function flips the
   sign, the mapping is doing identification work that isn't warranted.

---

## How to read disagreement

If `sp.spillover` and `sp.network_exposure` give different direct
effects on the same dataset, the usual culprits are:

- **Cluster boundary != network boundary.** `spillover` assumes no
  across-cluster leakage; `network_exposure` allows it. A gap between
  them *is* a measurement of across-cluster leakage.
- **Median-split for `spillover`** vs **Bernoulli design for
  `network_exposure`**. The former is a nonparametric split on
  observed exposure; the latter is Horvitz-Thompson under a known
  design. Disagreement → your design assumption may be wrong.
- **Exposure function.** `fraction` (default) vs `any` can give very
  different direct effects in networks with heterogeneous degree. If
  your graph has a long-tailed degree distribution, `fraction` is
  usually the defensible choice.

For `peer_effects` vs `network_hte`: disagreement between
`endogenous_peer` (β) and `direct_effect` typically indicates that the
linear-in-means spec is too rigid — the ML residualization in
`network_hte` will catch nonlinear effects that 2SLS misses.

---

## Worked example: vaccine herd immunity

```python
import statspai as sp
import numpy as np
import pandas as pd

# Simulate: households of size 4, half randomly treated, herd effect
rng = np.random.default_rng(0)
n_households = 200
rows = []
for h in range(n_households):
    p_treat = rng.beta(2, 2)                     # household-level rate
    for _ in range(4):
        d = rng.binomial(1, p_treat)
        rows.append({"household": h, "d": d})
df = pd.DataFrame(rows)
# Herd immunity: outcome depends on own + peer vaccination
peer_share = df.groupby("household")["d"].transform(
    lambda s: (s.sum() - s) / 3
)
df["y"] = (
    1.0
    - 0.4 * df["d"]                              # direct effect
    - 0.6 * peer_share                           # spillover effect
    + rng.normal(0, 0.3, size=len(df))
)

r = sp.spillover(df, y="y", treat="d", cluster="household")
print(f"Direct    : {r.model_info['direct_effect']:+.3f}")
print(f"Spillover : {r.model_info['spillover_effect']:+.3f}")
print(f"Total     : {r.model_info['total_effect']:+.3f}")
```

On this DGP the direct effect is ≈ `-0.4`, the spillover ≈ `-0.6`, and
the total ≈ `-1.0` — with the spillover *larger than* the direct
effect, which is the canonical herd-immunity story. Running the same
data through a naive ATT (`sp.did`) would recover something between
`-0.4` and `-1.0` depending on the mix of treated peers, and would
confound the two pathways.

---

*This guide is current for StatsPAI ≥ 1.5.0. All functions listed are
stable and registered in `sp.list_functions()`; inspect any of them
with `sp.describe_function("spillover")`, etc. For dispatcher-style
access to the whole family, see `sp.interference(design=...)`.*
