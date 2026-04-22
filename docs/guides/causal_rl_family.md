# Causal reinforcement learning — the full family

> **When your decisions today change the data you learn from tomorrow**,
> classical causal methods break: propensity scores mutate, confounders
> drift, reward is delayed.  Causal RL bridges the gap.  StatsPAI v1.4
> ships bandit, MDP, offline-safe, confounding-robust deep-Q, and a
> benchmark suite — all under a consistent API.

This guide is for applied economists and decision-support engineers
deciding between classical causal inference and RL-style sequential
policy learning.  Every function below lives at top level:
`sp.causal_bandit`, `sp.causal_dqn`, `sp.counterfactual_policy_optimization`,
`sp.structural_mdp`, `sp.offline_safe_policy`, `sp.causal_rl_benchmark`.

---

## When to reach for causal RL (vs one-shot causal inference)

| Your problem looks like…                                              | Right tool                      |
|-----------------------------------------------------------------------|---------------------------------|
| "Did treatment X affect outcome Y once?" (one time period, no follow-up) | Classical causal (`sp.did`, `sp.iv`, …) |
| "Under which context should we assign A or B?" (contextual, one shot) | `sp.causal_bandit`              |
| "Best multi-step policy, offline data, no confounders"                | Standard RL (outside StatsPAI)  |
| "Multi-step policy + unobserved confounders"                          | `sp.causal_dqn`                 |
| "Multi-step policy + safety constraint"                               | `sp.offline_safe_policy`        |
| "Off-policy evaluation of a stochastic policy"                        | `sp.counterfactual_policy_optimization` |
| "General structural MDP with causal interpretation"                   | `sp.structural_mdp`             |

---

## Causal bandit — `sp.causal_bandit` (Bareinboim-Pearl 2015)

Contextual bandit that respects a causal DAG: certain arms are
causally linked to context variables, so pulling arm `A` with context
`X` reveals information about arm `A'` in that same context.  Standard
contextual bandits ignore this and waste samples.

```python
def reward_fn(arm, context):
    # user-supplied: returns noisy reward for this arm × context
    ...

r = sp.causal_bandit(
    arms=["A", "B", "C"],
    reward_fn=reward_fn,
    context={"age_group": "young"},
    n_samples=1000,
)
r.best_arm              # arm with highest posterior mean reward
r.regret_curve          # cumulative regret over samples
```

Reference: Bareinboim, Forney, Pearl (2015), *NeurIPS*.

---

## Causal DQN — `sp.causal_dqn` (Fu-Zhou 2025)

Deep Q-learning with a confounding-robust update rule.  Given offline
trajectories `(state, action, reward, next_state)` possibly collected
under an *unknown* logging policy, `causal_dqn` learns a Q-function
that is robust to bias from unobserved confounders in the logging
policy, bounded by `gamma_bound`.

```python
r = sp.causal_dqn(
    data=offline_trajectories,
    state="s_vec", action="a", reward="r", next_state="s_next_vec",
    gamma_bound=0.1,        # upper bound on confounding strength
    discount=0.95,
    n_iter=200,
    lr=0.05,
)
r.policy                     # learned greedy policy
r.q_values                   # Q̂(s, a)
r.confounding_sensitivity    # worst-case Q-value shift
```

**Why this over vanilla DQN**: vanilla DQN trained on confounded offline
data will happily converge to a Q-function that is biased in the direction
of the logging policy's mistakes.  `causal_dqn` explicitly caps how much
the confounding can distort Q and returns a sensitivity range.

Citation: Li, Zhang & Bareinboim (2025), arXiv:2510.21110.

---

## Offline-safe policy — `sp.offline_safe_policy` (Chemingui et al. 2025)

Same spirit as causal DQN, with an added **cost constraint**: the
learned policy must have expected cost below `cost_threshold` with
high probability.  Critical for RL in healthcare, robotics, pricing.

```python
r = sp.offline_safe_policy(
    data=offline_trajectories,
    state="s", action="a", reward="r", cost="c",
    cost_threshold=0.5,
    discount=0.95,
    n_iter=100,
)
r.safe_policy               # the safe, reward-maximising policy
r.cost_estimate             # estimated cost + its upper CI bound
r.reward_estimate
```

Internally uses a Lagrangian dual-ascent on `reward − λ · cost` with
the Lagrange multiplier driven to satisfy the constraint on the
validation split.

Citation: Chemingui et al. (2025), arXiv:2510.22027.

---

## Counterfactual policy optimisation — `sp.counterfactual_policy_optimization`

Evaluate a *candidate* policy on historical off-policy data, using
importance sampling + doubly-robust correction:

```python
r = sp.counterfactual_policy_optimization(
    data=historical,
    state="context",
    action="action_taken",
    reward="reward_observed",
    target_policy=lambda state: 0.8 if state > 0 else 0.2,   # π(a=1|s)
    noise_sd=1.0,
)
r.estimated_value            # V̂(π) on the target policy
r.se_value
```

**Use case**: "If we had always used policy π, what reward would we have
gotten?" — the evaluation half of OPE.  Pairs well with `sp.OPEResult`
which exposes IPS/SNIPS/DR variants.

---

## Structural MDP — `sp.structural_mdp`

Fits a general parametric Markov decision process from trajectories,
with explicit causal semantics: transitions, rewards, and policies are
all exposed as structural equations.  Useful when you want to *simulate*
counterfactual trajectories under a different policy.

```python
r = sp.structural_mdp(
    data=trajectories,
    state_cols=["health", "activity"],
    action_cols=["treatment"],
    reward="utility",
    next_state_cols=["health_next", "activity_next"],
    time="t", trajectory="patient_id",
)
r.transition_model           # P̂(s' | s, a)
r.reward_model               # R̂(s, a)
r.simulate(policy=my_policy) # counterfactual trajectories
```

---

## Benchmark: `sp.causal_rl_benchmark`

Reproducible synthetic DGPs to stress-test causal RL algorithms:

```python
bench = sp.causal_rl_benchmark(
    name="confounded_bandit",       # or "dosage" / "pricing" / "targeting"
    n_episodes=1000,
    confounding_strength=0.5,
    seed=0,
)
bench.regret_curves    # dict: algorithm → regret over time
bench.best             # algorithm with lowest cumulative regret
```

Reference: Cunha, Liu, French & Mian (2025), arXiv:2512.18135 — a
canonical unified causal-RL benchmark taxonomy.

---

## Decision guide

```
Single-step, observational:
  With context                → sp.causal_bandit
  Policy evaluation only      → sp.counterfactual_policy_optimization

Multi-step, offline data:
  No confounding              → (use a general RL library; StatsPAI is
                                 not trying to replace stable-baselines3)
  Confounding (bounded)       → sp.causal_dqn
  Safety constraint           → sp.offline_safe_policy
  Want to simulate counterfactuals → sp.structural_mdp

Benchmarking / compare algorithms:
  → sp.causal_rl_benchmark
```

---

## Sanity checks specific to causal RL

1. **Policy value calibration**: run `sp.counterfactual_policy_optimization`
   on the logging policy and check that the estimate lines up with the
   observed average reward.
2. **Confounding sensitivity** (`causal_dqn` only): plot `r.q_values` at
   `gamma_bound ∈ {0, 0.1, 0.5, 1.0}`.  Flat lines = robust; big swings =
   your answer depends on the confounding assumption.
3. **Safety constraint slack** (`offline_safe_policy` only): if the
   returned policy has `cost_estimate` right at `cost_threshold`, widen
   the threshold and check if reward jumps — you may be over-constrained.
4. **Benchmark sanity**: before trusting a custom pipeline on real data,
   run it against `sp.causal_rl_benchmark(name="confounded_bandit")` and
   verify regret scales as expected with `confounding_strength`.

---

*Current for StatsPAI ≥ 1.5.0.  Full list via `sp.list_functions()` —
all causal-RL functions are tagged `rl` and `causal`.*
