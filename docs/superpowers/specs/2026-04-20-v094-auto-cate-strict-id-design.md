# StatsPAI v0.9.4 — `sp.auto_cate()` + strict identification diagnostics

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design

## 1. Motivation

The 0.9.4 post-release retrospective (`社媒文档/4.20-升级说明/StatsPAI-0.9.3之后的一周...`) publicly committed to two concrete next steps after 0.9.3:

1. **"下一步打算加 `strict_mode=True`"** on `sp.check_identification` — making design blockers non-ignorable.
2. **"下一步：`sp.auto_cate()` 一键多学习器比赛"** — closing the *"ML CATE scheduling not as good as econml"* gap called out in Section 8.

This spec turns those promises into v0.9.4.

## 2. Scope

### In scope

- `sp.auto_cate()` — one-line CATE learner race across S/T/X/R/DR-Learners with honest cross-fitted scoring (R-loss) and BLP calibration.
- `strict_mode=True` on `check_identification` — raise `IdentificationError` when blockers are present.
- `IdentificationError` exception type exported at top level.
- IV-specific first-stage strength check inside `check_identification` (Stock–Yogo rule-of-thumb F < 10).
- Bump version `0.9.3 → 0.9.4`.
- Test coverage: ≥ 10 new tests for `auto_cate`, ≥ 4 new tests for `strict_mode` / IV strength.

### Out of scope (deferred to 0.9.5+)

- Optuna hyper-parameter search (we rely on sensible defaults; user can pass their own tuned estimators).
- `sp.bayes_did` / `sp.bayes_rd` (Bayesian causal — 0.9.5 preview only).
- Rust kernel.
- `'ate'` / deep `'rct'` design branches beyond what already exists.

## 3. `sp.auto_cate()` — design

### 3.1 Signature

```python
def auto_cate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: list[str],
    learners: Sequence[str] = ('s', 't', 'x', 'r', 'dr'),
    outcome_model=None,
    propensity_model=None,
    cate_model=None,
    n_folds: int = 5,
    score: str = 'r_loss',          # 'r_loss' | 'dr_mse' | 'tau_risk'
    alpha: float = 0.05,
    random_state: int = 42,
) -> AutoCATEResult
```

### 3.2 Algorithm

1. Validate inputs (binary treatment, columns present, no NA in relevant cols).
2. Cross-fit nuisance **once** — a single `m_hat = E[Y|X]` and `e_hat = P(D=1|X)` via K-fold — reused across learners for the R-loss honest score.
3. For each requested learner:
   a. Fit using existing `metalearner(...)` — gets ATE, SE, CI, in-sample CATE.
   b. Compute **honest CATE predictions** via out-of-fold re-fit (K folds; per-fold learner refit, predict on held-out fold).
   c. Compute **R-loss score** on held-out predictions: `R(τ̂) = E[((Y − m̂) − τ̂(X)·(D − ê))²]`. Lower is better (Nie & Wager 2021).
   d. Run **BLP calibration test** (β₁ ≈ 1, β₂ > 0 significant) using existing `blp_test`.
4. Assemble a leaderboard DataFrame.
5. Pick `best_learner` = the learner with lowest `r_loss` among those whose BLP β₁ is not significantly below 1 at `alpha` (fallback: pure argmin if none pass).
6. Return `AutoCATEResult`.

### 3.3 `AutoCATEResult`

A dataclass with:

| field | type | description |
|------|------|------|
| `leaderboard` | `pd.DataFrame` | 1 row per learner: `learner, ate, se, ci_lower, ci_upper, r_loss, blp_beta1, blp_beta1_pvalue, blp_beta2, blp_beta2_pvalue, cate_std, cate_iqr` |
| `best_learner` | `str` | The chosen winner (`'S-Learner'` / `'T-Learner'` / ...) |
| `best_result` | `CausalResult` | Full CausalResult for the winner (has `.tidy()`, `.glance()`, `.summary()`) |
| `results` | `dict[str, CausalResult]` | All fitted learners keyed by short name |
| `agreement` | `pd.DataFrame` | Pearson-ρ matrix of CATE vectors across learners (in-sample) — high agreement ⇒ stable heterogeneity, low ⇒ model-dependent |
| `selection_rule` | `str` | Human-readable explanation (`"lowest R-loss among BLP-calibrated learners"`) |
| `n_obs` | `int` | Sample size |

`AutoCATEResult.summary()` → pretty-printed leaderboard + agreement + recommendation.

### 3.4 Score: honest R-loss

Why R-loss: unlike ATE bootstrap SE (which is about variance of the mean), R-loss directly measures **CATE prediction quality** on held-out data. This is the econml / grf-aligned criterion. It works uniformly across S/T/X/R/DR because it only depends on held-out τ̂(X), m̂(X), ê(X).

Implementation: for each learner, run K-fold where the learner is fitted on train folds and τ̂ predicted on test fold. Use the **already-cross-fitted** m̂ and ê to compute `((Y − m̂) − τ̂·(D − ê))²` on held-out rows, averaged across folds.

Tie-break rule for learner selection:

```
candidates = learners where BLP β1 is not rejected as < 1 at alpha
if candidates nonempty:
    winner = argmin r_loss over candidates
else:
    winner = argmin r_loss over all learners
```

## 4. `strict_mode` on `check_identification`

### 4.1 API change

```python
def check_identification(..., strict: bool = False) -> IdentificationReport:
    ...
    if strict and report.verdict == 'BLOCKERS':
        raise IdentificationError(report)
    return report
```

Parameter name: `strict` (not `strict_mode` — follows scikit/statsmodels convention of short boolean flags).

### 4.2 `IdentificationError`

```python
class IdentificationError(ValueError):
    """Raised when `check_identification(strict=True)` finds BLOCKER findings."""
    def __init__(self, report: IdentificationReport):
        self.report = report
        blocker_msgs = [f.message for f in report.findings if f.severity == 'blocker']
        msg = (
            f"Design has {len(blocker_msgs)} identification blocker(s):\n  - "
            + "\n  - ".join(blocker_msgs)
            + "\n\nPass strict=False to receive a report instead of an exception."
        )
        super().__init__(msg)
```

Exported as `sp.IdentificationError`.

## 5. IV strength check

Add `_check_iv_strength(data, treatment, instrument, findings)` inside `identification.py`:

- Compute first-stage F on `treatment ~ instrument` (no covariates for the minimal check; if covariates given, partial out first).
- F < 10 → warning (Stock–Yogo rule of thumb).
- F < 5 → blocker (weak instrument, inference invalid).
- Emits `evidence={'first_stage_f': ...}`.

Invoked inside `check_identification` when `instrument is not None` OR `design == 'iv'`.

## 6. File plan

| File | Change |
|------|--------|
| `src/statspai/metalearners/auto_cate.py` | **NEW** — `auto_cate` + `AutoCATEResult` (~350 LOC) |
| `src/statspai/metalearners/__init__.py` | Export `auto_cate`, `AutoCATEResult` |
| `src/statspai/smart/identification.py` | Add `IdentificationError`, `strict` param, `_check_iv_strength` |
| `src/statspai/smart/__init__.py` | Export `IdentificationError` |
| `src/statspai/__init__.py` | Top-level exports: `auto_cate`, `AutoCATEResult`, `IdentificationError` |
| `tests/test_auto_cate.py` | **NEW** — ≥10 tests |
| `tests/test_check_identification.py` | ≥4 new tests (strict, IV strength) |
| `pyproject.toml` | `version = "0.9.4"` |
| `CHANGELOG.md` | Add 0.9.4 entry |

## 7. Test plan

### 7.1 `test_auto_cate.py`

1. `test_auto_cate_basic_api` — returns AutoCATEResult with expected attributes.
2. `test_auto_cate_leaderboard_shape` — leaderboard has one row per requested learner.
3. `test_auto_cate_recovers_ate_on_constant_effect_dgp` — best_result.estimate within 3σ of 3.0.
4. `test_auto_cate_all_learners_positive_on_positive_dgp` — every ATE positive.
5. `test_auto_cate_learner_subset` — passing `learners=['t','dr']` runs only those two.
6. `test_auto_cate_invalid_learner_raises` — `learners=['bogus']` raises.
7. `test_auto_cate_selection_rule_string_nonempty` — has a human explanation.
8. `test_auto_cate_agreement_matrix_shape` — NxN with ones on diagonal.
9. `test_auto_cate_best_result_is_causal_result` — `.tidy()` and `.glance()` work.
10. `test_auto_cate_custom_models_override_defaults` — pass RandomForest, still works.
11. `test_auto_cate_summary_string` — summary() output non-empty and mentions winner.
12. `test_auto_cate_available_at_sp_top_level` — `sp.auto_cate is ...`.

### 7.2 New `test_check_identification.py` tests

1. `test_strict_mode_raises_on_blocker` — perfect-separation DGP + `strict=True` → `IdentificationError`.
2. `test_strict_mode_allows_warnings` — warnings-only should NOT raise.
3. `test_strict_mode_default_is_non_strict` — default behavior unchanged.
4. `test_iv_weak_instrument_flagged` — `cor(z, d) ≈ 0` → first-stage F tiny → blocker/warning emitted.
5. `test_iv_strong_instrument_passes` — `cor(z, d) = 0.7` → no IV-strength warning.

## 8. Non-goals / trade-offs

- **No Optuna** — we want v0.9.4 shipped in hours, not days. `auto_cate` is still "auto" because it races learners and picks by R-loss; hyperparameter search can layer on later via `cate_model=` / `outcome_model=`.
- **R-loss > DR-loss as default** — DR-loss would need an explicit nuisance for variance of phi per learner, R-loss is symmetric and uses the shared nuisance.
- **Agreement via CATE vector correlation** — not a formal test; documented as a *sanity check*, not an inference object.
- **IV strength check is minimal** — no Kleibergen-Paap, no Montiel-Olea-Pflueger; those remain in `sp.iv.weakiv_tests` and documented in the IV guide. The diagnostic here is a "first-line trip wire" not a replacement for dedicated tests.

## 9. Success criteria

1. `pytest tests/test_auto_cate.py tests/test_check_identification.py tests/test_metalearners.py -q` → all green.
2. Full suite `pytest tests/ -q` shows **no new failures** compared to pre-change baseline.
3. Can execute the README-style snippet end-to-end:
   ```python
   import statspai as sp
   result = sp.auto_cate(df, y='y', treat='d', covariates=['x1','x2'])
   print(result.summary())
   ```
4. `sp.check_identification(df, ..., strict=True)` raises `sp.IdentificationError` on blocker DGP, does NOT raise on OK/WARNINGS DGP.
5. `sp.__version__ == '0.9.4'`.

## 10. Commit + release sequence

1. Implement + tests on `main` (per user preference: direct-push, no PR).
2. Run focused tests → run full suite → self-code-review pass → commit.
3. Update CHANGELOG + pyproject version bump → single commit.
4. Push to `main`.
