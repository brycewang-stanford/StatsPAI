# Mixtape Chapter 9 — Difference-in-Differences (StatsPAI replication)

This guide walks through Cunningham's *Causal Inference: The Mixtape*
Chapter 9 (Difference-in-Differences) using StatsPAI's causal toolbox.
It is designed to be the **template** other chapter replications
follow — if you want Chapter 8 (IV) or Chapter 6 (RD), fork this
structure and swap the estimators.

> **Why replicate a textbook?** Every serious empirical programme lives
> or dies on *identification assumptions* — not code.  Walking through
> Cunningham's Chapter 9 lets you see how StatsPAI maps one-for-one
> onto the pedagogical language (parallel trends, TWFE, staggered
> adoption, event studies) rather than the estimator zoo.

Running the code in this page end-to-end takes **about 30 seconds**.

---

## 0. Setup

```python
import numpy as np
import pandas as pd
import statspai as sp

# Reproducible throughout the chapter:
RNG_SEED = 2026
```

Mixtape uses the *Kansas* and *Cheng-Hoekstra castle-doctrine* data for
its worked examples.  Neither is bundled with StatsPAI, so we replicate
the *pedagogical pattern* — heterogeneous, staggered treatment on a
panel — using the canonical DGP:

```python
df = sp.dgp_did(
    n_units=200, n_periods=10,
    staggered=True, n_groups=4,
    effect=0.5, heterogeneous=True,
    seed=RNG_SEED,
)
df.head()
# columns: unit, time, y, treated, first_treat, group
```

`first_treat` is NaN for never-treated units and an integer period
otherwise — the standard Callaway-Sant'Anna convention.

---

## 1. The TWFE baseline (and why it's wrong under heterogeneity)

Mixtape first presents the classical two-way-fixed-effects estimator:

```python
twfe = sp.panel(
    df,
    "y ~ treated",
    entity="unit", time="time",
    method="fe",
    cluster="unit",
)
twfe.summary()
```

Under homogeneous effects this is the canonical DID.  Under
**staggered adoption with heterogeneous effects**, Goodman-Bacon (2021)
shows the TWFE coefficient is a weighted average of every 2×2 DID
comparison *including ones where already-treated units serve as the
control group* — so some weights can be negative.

Run the Bacon decomposition to see it:

```python
bacon = sp.bacon_decomposition(
    data=df, y="y", treat="treated",
    time="time", id="unit",
)
sp.bacon_plot(bacon)
```

Every bar labelled "Later vs. earlier (treated)" is the pathological
contribution Goodman-Bacon flagged.

---

## 2. The modern staggered-DID suite (CS / SA / BJS)

Since 2021 the canonical answer is to report **three estimators** in
parallel: Callaway-Sant'Anna (ATT decomposition), Sun-Abraham (weighted
event study), and Borusyak-Jaravel-Spiess (imputation).  StatsPAI ships
a one-liner for the race:

```python
race = sp.auto_did(
    df, y="y", g="first_treat", t="time", i="unit",
)
print(race.summary())
```

which prints:

```text
auto_did: staggered-DiD method race
============================================================
method  estimate  std_error  ci_lower  ci_upper  n_obs notes
    CS    0.5092     0.0538    0.4037    0.6146   2000    ok
    SA    0.4923     0.1091    0.2786    0.7061   2000    ok
   BJS    0.5039     0.0240    0.4569    0.5509   2000    ok

selected winner : bjs (rule=median)
```

All three land within 0.02 of the true effect (0.5).  The median-winner
rule (`select_by='median'`) is the default; pass `select_by='cs'` (etc.)
to force a specific headline estimate.

---

## 3. Event-study visualisation (Sun-Abraham)

Mixtape emphasises the event-study plot as the **primary evidence** for
parallel pre-trends.  StatsPAI:

```python
sa = sp.sun_abraham(
    data=df, y="y", g="first_treat", t="time", i="unit",
    event_window=(-3, 3),
)
sp.enhanced_event_study_plot(sa)
```

Pre-period coefficients should cluster around zero; post-period
coefficients should show the dynamic path of the treatment.

---

## 4. Honesty: Rambachan-Roth (2023) sensitivity

What if parallel trends doesn't quite hold?  Rambachan & Roth's
"Honest DID" machinery answers *how much* drift the result can absorb
before losing significance.

```python
cs = sp.callaway_santanna(
    data=df, y="y", g="first_treat", t="time", i="unit",
)
dyn = sp.aggte(cs, type="dynamic")

# Breakdown value: largest allowable pre-trend drift per period
m_star = sp.breakdown_m(dyn, e=0, method="smoothness")
print(f"Breakdown M* = {m_star:.4f}")

# Full sensitivity frame at several candidate M values
sens = sp.honest_did(dyn, e=0, method="smoothness",
                     m_grid=list(np.linspace(0, 0.5, 11)))
sens.tail()
```

A large `M*` means the effect survives substantial violations of
parallel trends — the result is robust.  A small `M*` (or zero!)
means the conclusion is fragile and should be flagged in the paper.

---

## 5. Multiverse / specification curve

Cunningham's Mixtape ends every DID chapter with a specification check.
StatsPAI ships **two** complementary multiverse tools — pick the one
that matches what you want to vary:

### 5a. Vary the *estimator* (CS vs SA vs BJS)

That's `sp.auto_did`, which we already used in Section 2.  The
leaderboard is itself the multiverse — three identification strategies,
one data set, three point estimates displayed side by side.

### 5b. Vary the *control set* — `sp.spec_curve`

```python
# Build a cross-sectional view of the last period (spec_curve takes a
# single-observation-per-unit frame, not the panel) for a covariate
# multiverse on a post-treatment outcome.
last = df[df["time"] == df["time"].max()]

sc = sp.spec_curve(
    data=last,
    y="y",
    x="treated",
    controls=[["group"], ["group", "unit"]],  # list-of-list: each inner list is one spec
)
sc.plot()
```

Every combination of a control set × an SE type × an (optional) sample
subset becomes one dot on the curve; stable estimates across the curve
imply the result is design-driven rather than specification-driven.

> **Note** — the ``spec_curve`` API is a covariate / SE multiverse,
> not an estimator multiverse.  If you want to race TWFE against CS
> against BJS, use ``auto_did`` above; if you want to see how the
> headline estimate moves when you change the control set, use this
> section.

---

## 6. What comes next

- **Chapter 6 replication** — sharp and fuzzy RD with `sp.rdrobust`
  and `sp.rd_honest` (McCrary manipulation + robust bias correction).
- **Chapter 8 replication** — IV with `sp.ivreg`, weak-IV-robust CIs
  via `sp.anderson_rubin_ci` / `sp.conditional_lr_ci`, and the
  Lee-McCrary (2022) `sp.tF_adjustment` critical value.
- **Chapter 10 replication** — synthetic control with `sp.synth`,
  `sp.sdid`, and the full donor-pool diagnostic suite.

Each follows the same pattern: **(1) classical baseline → (2) modern
race → (3) event-study visualisation → (4) sensitivity → (5) spec
curve**.

---

## Citation

If you use this guide or StatsPAI in academic work, please cite both:

```bibtex
@book{cunningham2021causal,
  title={Causal Inference: The Mixtape},
  author={Cunningham, Scott},
  year={2021},
  publisher={Yale University Press},
  url={https://mixtape.scunning.com/}
}

@software{statspai,
  title={StatsPAI: The AI-powered Statistics & Econometrics Toolkit for Python},
  author={Wang, Biaoyue},
  year={2026},
  url={https://github.com/brycewang-stanford/StatsPAI},
  version={0.9.10}
}
```
