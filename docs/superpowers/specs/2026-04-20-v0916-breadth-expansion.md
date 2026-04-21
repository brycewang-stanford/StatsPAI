# StatsPAI v0.9.16 — v1.0 breadth expansion (20 new modules)

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

Between the v0.9.15 tidy-multiterm spec (landed as db050de) and this
release, 20 new modules were authored to cover previously-unsupported
methods — most of them requested by users working on applied causal
projects where StatsPAI was already close but missing a specific
estimator. This spec catalogs the batch, flags gaps, and defines the
ship gate.

This is a **breadth pass**, analogous to the v1.0 upgrade recorded in
memory `project_v1_upgrade.md`, but scoped tighter: each module is a
standalone addition with no cross-cutting architectural change.

## 2. Inventory

20 new modules, ~4,800 LOC, organised by sub-package.

> **Note:** this inventory was captured at the start of review. The
> actual release grew to **27+ modules** and **3 dedicated test files**
> as the author added more coverage during the sign-off window (DAG
> do-calculus / SWIG / counterfactual, Balke-Pearl bounds, ICP causal
> discovery, four-way decomposition, peer effects, transport package,
> plus `tests/test_dag_scm.py` and `tests/test_icp.py`). The authoritative
> file list is the release commit diff.

### 2.1 Causal forests family (648 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `causal/iv_forest.py` | 200 | Forest-based LATE | Athey-Tibshirani-Wager 2019 |
| `causal/multi_arm_forest.py` | 176 | Multi-arm CATE | Tibshirani-Wager 2020 |
| `survival/causal_forest.py` | 272 | Survival CATE | Cui-Kosorok-Athey-Wager 2023 |

### 2.2 Causal discovery (726 LOC)
| Module | LOC | Output | Reference |
|---|---|---|---|
| `causal_discovery/fci.py` | 430 | PAG with latent-confounder edges | Spirtes-Meek-Richardson 1995, Zhang 2008 |
| `dag/identification.py` | 296 | Identifiability certificate on a user DAG | Pearl back-door/front-door + ID algorithm |

### 2.3 Dynamic treatment regimes (522 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `dtr/q_learning.py` | 185 | Q-function backward induction | Murphy 2003 |
| `dtr/a_learning.py` | 173 | Advantage / blip functions | Robins 2004, Moodie et al. 2007 |
| `dtr/snmm.py` | 164 | Structural nested mean model | Robins 1994 |

### 2.4 TMLE extensions (381 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `tmle/ltmle.py` | 381 | Longitudinal ATE under static regime | van der Laan-Gruber 2012 |

### 2.5 Proximal causal inference (505 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `proximal/negative_controls.py` | 312 | NCO/NCE calibration + double-NC ATE | Lipsitch 2010, Miao-Shi-Tchetgen 2018/2020 |
| `proximal/pci_regression.py` | 193 | Proximal 2-stage regression | Cui-Tchetgen 2020 |

### 2.6 Interference / spillover (323 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `interference/network_exposure.py` | 323 | HT estimator for exposure mappings | Aronow-Samii 2017 |

### 2.7 Dose-response / continuous treatment (253 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `dose_response/vcnet.py` | 253 | Varying-coefficient dose-response (+ SCIGAN) | Nie-Brunskill-Wager 2021 |

### 2.8 Matching (231 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `matching/genmatch.py` | 231 | Genetic-algorithm matching | Diamond-Sekhon 2013 |

### 2.9 Diagnostics / sensitivity (295 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `diagnostics/rosenbaum.py` | 295 | Rosenbaum Γ sensitivity bounds | Rosenbaum 2002 |

### 2.10 Spatial econometrics (376 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `spatial/did.py` | 208 | DID with spatial spillovers | Delgado-Florax 2015, Butts 2021 |
| `spatial/iv.py` | 168 | Spatial IV | Kelejian-Prucha 1998 |

### 2.11 Policy learning (222 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `policy_learning/ope.py` | 222 | Off-policy evaluation (IPS, DR, switch) | Dudík-Erhan-Langford-Li 2014 |

### 2.12 Time series causal (203 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `timeseries/its.py` | 203 | Interrupted time series | Wagner et al. 2002 |

### 2.13 Censoring / survival weights (266 LOC)
| Module | LOC | Estimand | Reference |
|---|---|---|---|
| `censoring/ipcw.py` | 260 | Inverse-probability-of-censoring weights | Robins-Finkelstein 2000 |

### 2.14 Target trial emulation (503 LOC, **only module WITH tests**)
| Module | LOC | Purpose |
|---|---|---|
| `target_trial/protocol.py` | 151 | Protocol specification primitives |
| `target_trial/emulate.py` | 125 | Emulation pipeline |
| `target_trial/ccw.py` + `ccw_internal.py` | 188 | Clone-censor-weight backbone |
| `target_trial/diagnostics.py` | 61 | Balance + positivity diagnostics |
| **`tests/test_target_trial.py`** | **188 LOC** | **sole dedicated test** |

## 3. Re-export surface

Main `src/statspai/__init__.py`: +35 / −8 lines, registers 13 new
top-level symbols:
`multi_arm_forest`, `iv_forest`, `genmatch`, `proximal_regression`,
`ltmle`, `vcnet`, `scigan`, `fci`, `network_exposure`, `q_learning`,
`a_learning`, `snmm`, `causal_survival_forest`.

Sub-package `__init__.py` diffs: 12 files, 1–4 lines each, all pure
re-exports.

`sp.*` smoke check: all 13 new symbols resolve on import.

## 4. Ship gate

### 4.1 Blockers identified

| # | Blocker | Severity |
|---|---|---|
| B1 | 22 of 23 new modules have **no dedicated tests** — only `target_trial` does | HIGH |
| B2 | Version string stays at `0.9.14` — not bumped to `0.9.16` | MED |
| B3 | `CHANGELOG.md` not updated | LOW |

### 4.2 Mitigation strategy — **WIP release**

This release ships as **`v0.9.16.dev` / "breadth expansion WIP"** with
a clear documentation disclaimer. Reasoning:

- Writing production tests for 22 modules upfront would gate the
  release by 2–4 weeks and leave finished code rotting in working
  tree.
- The import chain is healthy (`sp.*` all resolve), so users who
  discover a module and read its docstring + paper refs get a
  working-in-principle implementation.
- The known-gap signalling pattern matches `memory/project_v1_upgrade`
  and `project_synth_v090_review` precedent: ship, then harden with
  follow-up test commits per module.

### 4.3 Required pre-commit actions

Only what is strictly needed to avoid shipping a broken release:

1. Bump version `0.9.14 → 0.9.16` in `pyproject.toml` + `src/statspai/__init__.py`.
2. Run full test suite (no new tests added, but must confirm no
   regressions against the 2,319 existing tests).
3. Commit as **one release commit**: `feat: v0.9.16 — breadth expansion (20 new modules, test hardening follow-up)`.

### 4.4 Deferred (follow-up commits)

Per-module test files, one PR/commit per module, roughly prioritised by user-visibility:

| Priority | Module | Reason |
|---|---|---|
| P0 | `tmle/ltmle.py` | Most complex algorithm (recursive backward induction); highest failure surface |
| P0 | `causal_discovery/fci.py` | PAG output format is sticky to API changes |
| P1 | `dtr/{q,a,snmm}_learning.py` | 3-way API consistency |
| P1 | `proximal/negative_controls.py` | Paper-sensitive math |
| P1 | `interference/network_exposure.py` | MC exposure probs are numerically delicate |
| P2 | All others | Smaller and / or simpler |

## 5. Commit strategy

**Single commit**, following the project's historical release pattern
(`a546ead feat: v0.9.12 —`, `2fd88de feat: v0.9.13 —`, etc.):

```
feat: v0.9.16 — breadth expansion (20 new modules, test hardening follow-up)
```

The commit body explicitly declares the WIP test status so future
`git blame` readers see the gap.

## 6. Success criteria

1. `git push origin main` succeeds.
2. `pip install .` produces a 0.9.16 wheel.
3. `import statspai as sp` + 13 new symbol lookups still resolve after
   checkout.
4. Full existing test suite stays green (no regressions).
5. At least one follow-up commit per P0 module within 2 weeks.
