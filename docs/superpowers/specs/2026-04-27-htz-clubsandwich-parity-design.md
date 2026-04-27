# clubSandwich-equivalent HTZ Wald DOF — design spec

**Date**: 2026-04-27
**Status**: Draft → awaiting user review
**Target version**: v1.6.4 (or v1.7 minor bump if bundled with other follow-ups)
**Scope**: independent PR, standalone (no `crve` / `feols` / `event_study` wiring)
**Parity bar**: `rtol < 1e-8` vs R `clubSandwich::Wald_test(..., test="HTZ")`

---

## 1. Why this exists

### 1.1 Current gap

`statspai.fast.cluster_dof_wald_bm` (added v1.8.x in HDFE Phase 4) implements
the **BM 2002 §3 simplified** matrix Satterthwaite DOF for a multi-restriction
Wald test under CR2 cluster-robust inference:

```
ν_W = [Σ_g tr(E_g E_g^T)]² / Σ_g ||E_g E_g^T||_F²
```

That formula is one of the valid "Bell-McCaffrey-style" approximations, and
its current docstring (`src/statspai/fast/inference.py:743`) is honest:

> **Not equivalent to** `clubSandwich::Wald_test(..., test="HTZ")`. The HTZ
> test uses a Hotelling-T² approximation with the more elaborate
> Pustejovsky-Tipton 2018 §3.2 moment-matching DOF, which is typically much
> smaller (more conservative) than the BM 2002 §3 simplified [...] formula
> implemented here. On moderate panels the two can differ by **50-100%**.

For users who do paper replication / cross-language reproducibility against R
clubSandwich, that 50-100% drift is unacceptable. The entire point of `sp.fast`
parity claims is "drop-in replacement at machine precision".

### 1.2 What this PR ships

A **standalone** clubSandwich-equivalent HTZ Wald test under CR2 sandwich:

| Symbol | Layer | Returns |
|---|---|---|
| `cluster_dof_wald_htz(X, cluster, *, R, weights, bread)` | low-level DOF helper, mirrors `cluster_dof_wald_bm` | `float` η |
| `cluster_wald_htz(X, residuals, cluster, *, R, r, beta, weights, bread)` | full Wald test, mirrors R `Wald_test(test="HTZ")` | `WaldTestResult` |
| `WaldTestResult` | frozen dataclass | `(test, q, eta, F_stat, p_value, Q, R, r, V_R)` |

### 1.3 Non-goals (explicit YAGNI)

1. ❌ Working covariance Φ ≠ I (GLS / mixed-model). Defer to v2.
2. ❌ HTA / KZ / Naive / EDF / chi-sq variants from P-T 2018. Only HTZ.
3. ❌ Wiring HTZ into `sp.fast.feols` / `sp.fast.crve` / `sp.fast.event_study`.
   That is the next, separate PR.
4. ❌ Saddlepoint approximation for the small-G regime.
5. ❌ Wald test under sandwiches other than CR2 (no CR3, no jackknife).
6. ❌ Performance optimisation (numba/JAX). NumPy direct is sufficient since
   q is typically ≤ 5 and the per-cluster work is O(n_g · q) at worst.

---

## 2. API surface

### 2.1 `WaldTestResult`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class WaldTestResult:
    test: str            # "HTZ" (locked for v1; future: "HTA", "KZ", ...)
    q: int               # number of restrictions
    eta: float           # P-T 2018 §3.2 moment-matching DOF
    F_stat: float        # Hotelling-T² scaled F: (η - q + 1)/(η · q) · Q
    p_value: float       # 1 - F_{q, η-q+1}.cdf(F_stat)
    Q: float             # raw cluster-robust Wald: (Rβ̂-r)' V_R^{-1} (Rβ̂-r)
    R: np.ndarray        # (q, k) restriction matrix
    r: np.ndarray        # (q,)   null value
    V_R: np.ndarray      # (q, q) R · V^CR2 · R^T

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

### 2.2 `cluster_dof_wald_htz`

```python
def cluster_dof_wald_htz(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """clubSandwich-equivalent HTZ Wald DOF (Pustejovsky-Tipton 2018 §3.2).

    Independent of any specific fit — depends only on the design matrix X
    and cluster structure. For the full Wald statistic + p-value, use
    :func:`cluster_wald_htz`.
    """
```

**Mirror of `cluster_dof_wald_bm` signature** so users can swap one keyword.

### 2.3 `cluster_wald_htz`

```python
def cluster_wald_htz(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    r: Optional[np.ndarray] = None,        # default zeros(q)
    beta: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> WaldTestResult:
    """Cluster-robust HTZ Wald test (Pustejovsky-Tipton 2018, JBES 36(4)).

    Equivalent to R::clubSandwich::Wald_test(fit, constraints, vcov="CR2",
    test="HTZ") to ``rtol < 1e-8`` on the verified fixtures.
    """
```

### 2.4 Exports

```python
# src/statspai/fast/__init__.py
from .inference import (
    ...,
    cluster_dof_wald_htz,
    cluster_wald_htz,
    WaldTestResult,
)
__all__ = [..., "cluster_dof_wald_htz", "cluster_wald_htz", "WaldTestResult"]
```

Registered in `src/statspai/registry.py` per CLAUDE.md §4 (registry is hard
requirement for `sp.help` / `sp.list_functions` discoverability).

---

## 3. Algorithm (Pustejovsky-Tipton 2018 §3.2)

> **Implementation rule**: every formula below is to be **double-sourced** at
> implementation time — once from the P-T 2018 PDF (eq. 12 + Theorem 2), once
> from the R clubSandwich `Wald_test.R` source — and reconciled. Memory is
> **not** an acceptable source (CLAUDE.md §10 spirit). If the two sources
> disagree, stop and audit before writing code.

### 3.1 Notation

- `n` rows, `k` regressors, `G` clusters, `q` restrictions.
- `X` is the (already-FE-residualised) regressor matrix `(n × k)`.
- `W = diag(w)` are observation weights (default `w = 1`).
- `M_X = X^T W X`, `bread = M_X^{-1}`.
- For cluster `g`: `X_g (n_g × k)`, `w_g (n_g,)`, residuals `e_g (n_g,)`.
- `H_gg = X_g · bread · X_g^T · diag(w_g)` ((n_g × n_g)).
- `A_g = (I_g - H_gg)^{-1/2}` via symmetrised eigendecomposition with
  eigenvalue floor 1e-12 (same convention as `crve` / `cluster_dof_wald_bm`).
- `Φ = I` (working covariance, locked to identity in v1).
- `R (q × k)` restriction matrix, full row rank, `q ≤ k`.

### 3.2 Steps

```
Step 1  bread = (X^T W X)^{-1}                                      (k × k)

Step 2  per cluster g:
          H_gg = X_g · bread · X_g^T · diag(w_g)                    (n_g × n_g)
          Msym = I_g - 0.5·(H_gg + H_gg^T)                          symmetrise
          eigh(Msym) → A_g = (Msym)^{-1/2}, eig floor 1e-12
          G_g = R · bread · X_g^T · A_g · diag(√w_g)                (q × n_g)
                                              # Φ = I absorbed into A_g, w
          ẽ_g = A_g · diag(√w_g) · e_g                              (n_g,)
          v_g = G_g · ẽ_g                                           (q,)

Step 3  V_R = Σ_g v_g v_g^T                                         (q × q)
                                              # = R · V^CR2 · R^T
        Ω   = Σ_g G_g G_g^T                                         (q × q)
                                              # E[V_R] under H0 + iid working cov

Step 4  HTZ DOF (P-T 2018 Theorem 2 / eq. 12):
          eigh(Ω) → Ω^{-1/2} (eig floor 1e-12)
          B_g = Ω^{-1/2} · G_g                                      (q × n_g)
          for all g, h in 1..G (including g == h):
            P_{gh} = B_g · B_h^T                                    (q × q)
          η = (some moment-match function of {P_{gh}})
                                              # ⚠ exact form INTENTIONALLY NOT in this spec.
                                              # Reason: writing a formula here from memory
                                              # creates a false-confidence anchor. The
                                              # implementer MUST derive η from BOTH:
                                              #   (a) P-T 2018 PDF, eq. (12) + Theorem 2 proof,
                                              #   (b) clubSandwich::Wald_test.R source.
                                              # If (a) and (b) appear to disagree, stop and
                                              # write a reconciliation note BEFORE coding.
                                              # The R parity test in §4.2 + frozen fixture in
                                              # §4.3 are the ground truth that catches drift.

Step 5  Q = (R β̂ - r)^T · V_R^{-1} · (R β̂ - r)                    scalar
        F_stat = (η - q + 1)/(η · q) · Q
        p_value = 1 - scipy.stats.f(q, η - q + 1).cdf(F_stat)
```

### 3.3 Numerical conventions

- **Eigenvalue floors**: 1e-12 on both `Msym` (`A_g`) and `Ω` (`Ω^{-1/2}`),
  matching the existing `crve` / `cluster_dof_wald_bm` convention. This keeps
  the new code locked in step with the established CR2 path.
- **Symmetrisation**: `0.5 · (M + M^T)` everywhere we eig — guards against
  catastrophic cancellation in `H_gg`.
- **Cluster ordering**: `pd.factorize(cluster, sort=False)` (same as
  `cluster_dof_wald_bm`). Order-equivariance test in §4 locks this down.
- **Hotelling-T² scaling validity**: requires `η > q - 1`. If not, raise
  `ValueError` (see §5).

### 3.4 Self-consistency invariants (free cross-checks)

For `q = 1`, HTZ collapses to a t² test. **Critical**: the q=1 HTZ DOF is
**NOT** the same as `cluster_dof_bm` (which implements the BM 2002 *simplified*
formula `(Σλ)² / Σλ²` without cross-cluster terms). The q=1 HTZ DOF **is**
the same as `clubSandwich::coef_test(test="Satterthwaite")` — the
Pustejovsky-Tipton 2018 generalized form with cross-cluster terms over the
working covariance Φ.

The existing test `test_cluster_dof_bm_matches_r_clubsandwich` documents this
~5-10% drift on a canonical panel (G=15, q=1). The new HTZ implementation
should match clubSandwich Satterthwaite to `rtol < 1e-8`, which means it will
**deliberately drift** from `cluster_dof_bm` by the same ~5-10% on that panel.

Self-consistency invariants the new code must honor:

1. **q=1 ↔ clubSandwich Satterthwaite parity** (ground-truth):
   `cluster_dof_wald_htz(R=c.reshape(1,-1))` ≡
   `clubSandwich::coef_test(test="Satterthwaite", cluster=g)`'s `df_Satt`
   to `rtol < 1e-8`. Tested in §4.2.

2. **q=1 ↔ q=1 self-equivalence** (free, no R needed):
   `cluster_dof_wald_htz(X, g, R=c.reshape(1,-1))` ≡
   the q=1 path inside `cluster_wald_htz` produces the same η.
   Asserts the helper and full-test paths share Step 4. `< 1e-12`.

3. **Documented drift from BM simplified** (regression guard):
   For the canonical panel, `cluster_dof_wald_htz` − `cluster_dof_bm` is in
   the `[2%, 15%]` relative-difference band. If this band is broken, either
   the BM simplified formula was changed or the HTZ implementation regressed.
   Deliberately a `range` assertion, not an equality.

---

## 4. Tests

### 4.1 Unit tests (`tests/test_fast_htz.py`)

| Test | What it locks | Tolerance |
|---|---|---|
| `test_htz_q1_helper_matches_full` | q=1: `cluster_dof_wald_htz` η ≡ `cluster_wald_htz(...).eta` | `< 1e-12` |
| `test_htz_q1_documented_drift_from_bm` | q=1 HTZ vs `cluster_dof_bm`: relative diff ∈ `[2%, 15%]` (NOT equal — see §3.4) | range |
| `test_htz_eta_in_sane_range` | `q ≤ η ≤ G − q + 1` for balanced panels | sanity |
| `test_htz_validates_R_shape` | rank-deficient R, wrong cols, empty R, all `ValueError` | — |
| `test_htz_too_few_clusters_rejected` | `G ≤ q` ⇒ `ValueError` | — |
| `test_htz_F_p_value_consistency` | `p = 1 - F.cdf((η-q+1)/(ηq) · Q)` | `< 1e-10` |
| `test_htz_invariant_to_X_rescaling` | column-rescale invariance: η, p unchanged | `< 1e-10` |
| `test_htz_invariant_to_cluster_relabel` | permute cluster IDs ⇒ η, F, p unchanged | `< 1e-10` |
| `test_htz_independent_of_bread_arg` | passing precomputed bread = computing it | `< 1e-10` |
| `test_htz_eta_le_q_minus_1_rejects` | degenerate small-G ⇒ `ValueError` | — |
| `test_htz_singleton_cluster_warns` | `n_g = 1` cluster ⇒ `RuntimeWarning` but doesn't crash | — |
| `test_htz_zero_residuals_returns_p1` | e ≡ 0 ⇒ Q = 0 ⇒ p = 1 | exact |

### 4.2 R clubSandwich parity tests (`tests/test_fast_htz.py`, `skipif(no Rscript)`)

| Test | Panel | q | Tolerance |
|---|---|---|---|
| `test_htz_matches_r_clubsandwich_q1` | G=15, balanced, n_g=20 | 1 | `rtol < 1e-8` |
| `test_htz_matches_r_clubsandwich_q2` | G=25, balanced, n_g=15 | 2 | `rtol < 1e-8` |
| `test_htz_matches_r_clubsandwich_q3_unbal` | G=50, unbalanced (n_g ∈ [3, 50]) | 3 | `rtol < 1e-8` |

R script invoked via `subprocess.run(["Rscript", "-e", ...])`, output as JSON
(η, F_stat, p_value, q). Skip if `Rscript` not on PATH or `clubSandwich` not
installed (same convention as `test_cluster_dof_bm_matches_r_clubsandwich`).

### 4.3 Frozen-fixture parity (`tests/fixtures/htz_clubsandwich.json`)

The R parity tests are belt-and-suspenders. The **CI backbone** is a
frozen fixture:

```
tests/fixtures/
├── _gen_htz_fixture.R          # R generator (~100 LOC, not in CI)
├── _gen_htz_fixture.README.md  # how to regenerate (R version + clubSandwich version pinned)
├── htz_panel_q1.csv            # actual panel data (R wrote it — no RNG sync games)
├── htz_panel_q2.csv
├── htz_panel_q3_unbal.csv
└── htz_clubsandwich.json       # [{panel: "q1", G, q, R, beta, eta, F, p, V_R}, ...]
```

**Critical design decision**: the fixture stores the **actual panel CSV** that
R generated, not a seed. Reason: `numpy.random` and R's `set.seed()` produce
different streams; relying on seed sync is a known foot-gun. Instead, R
generates the data, writes it to CSV, and writes the test outputs (η, F, p,
V_R) to JSON. Python reads the same CSV, runs `cluster_wald_htz`, and
asserts `rtol < 1e-8` on the JSON values. Both sides see byte-identical input.

`test_htz_frozen_fixture` runs **without R installed** — it locks numerical
parity for every CI run.

### 4.4 Test totals

```
12 unit tests + 3 R-parity tests + 1 frozen-fixture (covers q1/q2/q3)
              = 16 test functions in tests/test_fast_htz.py
              + 4 fixture files in tests/fixtures/
```

Acceptance: `pytest tests/test_fast_htz.py -q` passes 13/16 without R, 16/16
with R + clubSandwich installed. The frozen-fixture test is the most
important — it runs in every CI environment and locks parity to 1e-8.

---

## 5. Error handling / boundaries

| Condition | Behavior |
|---|---|
| `G ≤ q` | `ValueError("HTZ Wald requires G > q; got G=..., q=...")` |
| `R` rank < q | `ValueError("R must have full row rank")` |
| `R.shape[1] != X.shape[1]` | `ValueError("R has ... cols but X has k=...")` |
| `R.shape[0] == 0` | `ValueError("R must have at least one row")` |
| `q > k` | `ValueError("R has q=... rows > k=...; over-determined")` |
| η ≤ q − 1 (Hotelling-T² scaling diverges) | `ValueError("HTZ DOF η=... ≤ q−1 ⇒ degenerate; use boottest_wald")` |
| Ω not positive definite (after eig floor) | warning + raise `LinAlgError` from up the stack |
| `weights` contains 0 / negative | `ValueError` (mirrors existing `boottest_wald` check) |
| singleton cluster (`n_g = 1`) | `RuntimeWarning("HTZ: singleton cluster g=...; A_g floored")` — doesn't crash |
| `r` shape mismatch | `ValueError("r has ... entries but R has q=... rows")` |

CLAUDE.md §3.7 "fail loudly" applies — no silent NaN returns.

---

## 6. File changes

### 6.1 Modified

```
src/statspai/fast/inference.py    +~250 LOC (Steps 1-5 + Wald_test wrapper + WaldTestResult)
src/statspai/fast/__init__.py     +3 imports/exports
src/statspai/registry.py          +3 rich-spec entries (CLAUDE.md §4 hard requirement)
src/statspai/__init__.py          (no change — sp.fast namespace already exposed)
CHANGELOG.md                      v1.6.4 entry: "Added: cluster_wald_htz / cluster_dof_wald_htz / WaldTestResult — clubSandwich-equivalent HTZ Wald (Pustejovsky-Tipton 2018, JBES 36(4))"
benchmarks/hdfe/SUMMARY.md        update §"What deliberately did NOT ship" — HTZ moves out of follow-up backlog
paper.bib                         add pustejovsky2018small (DOI: 10.1080/07350015.2016.1247004) — refs verified per CLAUDE.md §10
```

### 6.2 New

```
tests/test_fast_htz.py                   ~400 LOC  (16 tests)
tests/fixtures/htz_clubsandwich.json     ~3 KB     (frozen R output)
tests/fixtures/_gen_htz_fixture.R        ~80 LOC   (regeneration script)
tests/fixtures/_gen_htz_fixture.README.md ~30 LOC  (how-to-regen)
```

### 6.3 Untouched (out of scope)

- `crve()` — no HTZ wiring
- `feols()`, `fepois()`, `event_study()` — no HTZ wiring
- `cluster_dof_wald_bm` — kept; docstring gets a one-line cross-link
  to the new HTZ helper, no API change
- `pyfixest` adapter (`src/statspai/fixest/`) — completely untouched

---

## 7. Citations (CLAUDE.md §10 zero-hallucination)

To be added to `paper.bib` only after Crossref + DOI dual-source verification:

```bibtex
@article{pustejovsky2018small,
  author    = {Pustejovsky, James E. and Tipton, Elizabeth},
  title     = {Small-Sample Methods for Cluster-Robust Variance Estimation
               and Hypothesis Testing in Fixed Effects Models},
  journal   = {Journal of Business \& Economic Statistics},
  volume    = {36},
  number    = {4},
  pages     = {672--683},
  year      = {2018},
  doi       = {10.1080/07350015.2016.1247004}
}
```

**Verification at implementation time** (commit message must note this):
- ✅ Crossref API: `https://api.crossref.org/works/10.1080/07350015.2016.1247004`
- ✅ DOI resolves: `https://doi.org/10.1080/07350015.2016.1247004`
- ✅ Author / year / volume / pages match across both sources

If any of the four elements (authors / year / title / journal+DOI) cannot be
independently verified at the time of commit, the bibkey **does not enter
the bib file** — placeholder `[citation needed]` lives in docstrings until
verification clears.

---

## 8. Acceptance criteria

The PR is mergeable when **all** of the following hold:

1. `pytest tests/test_fast_htz.py -q` → 16/16 passed (or 13/16 if R missing)
2. `pytest -q` (full suite) → no regressions vs current `main` baseline
3. `python -c "import statspai as sp; assert hasattr(sp.fast, 'cluster_wald_htz')"` succeeds
4. `python -c "import statspai as sp; print(sp.list_functions())"` lists the
   3 new symbols (registry sanity)
5. `cluster_wald_htz` matches R `clubSandwich::Wald_test(test='HTZ')` to
   `rtol < 1e-8` on **at least three** distinct panels (q ∈ {1, 2, 3},
   balanced + unbalanced)
6. q=1 self-equivalence: `cluster_dof_wald_htz(X, g, R=c.reshape(1,-1))` ≡
   `cluster_wald_htz(...).eta` to `rtol < 1e-12` (helper ↔ full test consistency).
   **Note**: this is NOT equality with `cluster_dof_bm` — see §3.4 for why HTZ
   should drift from BM 2002 simplified by ~5-10% on the canonical panel.
7. CHANGELOG + MIGRATION + paper.bib all updated; no `[citation needed]`
   leaks past commit
8. No GPL/AGPL code copied. The implementation is a paper-faithful NumPy
   write-up; R clubSandwich is used **only** as a black-box reference for
   fixture generation and parity testing.

---

## 9. Risks / open questions

### 9.1 Highest-leverage risk

**Step 4 η formula transcription error.** The paper has multiple equivalent
forms (Theorem 2 has matrix and tensor variants); clubSandwich's R source
uses a specific computational form. If the formula is mis-transcribed, the
bug will surface in **at least three** places:

1. The R parity tests (§4.2) — directly compare to clubSandwich.
2. The frozen-fixture test (§4.3) — compares to R-generated JSON.
3. The "documented drift from BM simplified" range test (§4.1) — if the
   formula is wildly wrong, the relative-difference band will be violated.

**Mitigation**:

- Implementation-time **double-source verification**: P-T 2018 PDF +
  clubSandwich `Wald_test.R`, both open simultaneously. Reconciliation note
  written to commit message before any code is committed.
- The frozen-fixture test runs in every CI environment — even contributors
  without R get the parity check for free.
- Any drift from R → audit, fix, only then commit. Never ship "looks
  approximately right" — CLAUDE.md §3.6 evidence-first.

### 9.2 Working covariance Φ = I assumption

clubSandwich defaults `Φ = I` for OLS+CR2, but its API allows user-supplied
`target` matrices (for e.g. AR1 panels). v1 explicitly drops this.

**Risk**: if a user passes design where the OLS errors are obviously
heteroskedastic / autocorrelated and asks "why doesn't my HTZ match R when
I set `target=...` in R?", the answer is "v1 doesn't support that — file an
issue or wait for v2".

**Mitigation**: docstring + CHANGELOG explicit; raise `NotImplementedError`
if `target=` keyword ever sneaks in.

### 9.3 Edge cases under singletons / very few clusters

P-T 2018 doesn't characterise behavior under `n_g = 1` clusters cleanly.
Our implementation degrades gracefully (eig floor + warning) but the
numerical answer might differ from R clubSandwich slightly.

**Mitigation**: explicit unit test for the singleton case asserting "warns
+ doesn't crash"; not asserting parity to R for this pathological case.

---

## 10. Effort estimate

| Phase | Hours | Owner |
|---|---|---|
| Implement Step 1-3 (V_R, Ω, sandwich) | 1.5 | autonomous |
| Implement Step 4 (η + Hotelling-T² scaling) with double-source verification | 2.5 | autonomous, paper + R open |
| Implement Step 5 (Q, F, p-value) + WaldTestResult dataclass | 0.5 | autonomous |
| Unit tests (12) | 1.5 | autonomous |
| R parity tests (3) + R fixture generator + frozen JSON | 1.5 | autonomous (R must be on PATH) |
| Registry + exports + docstrings + cross-links | 0.5 | autonomous |
| CHANGELOG + paper.bib (with Crossref verification) + SUMMARY.md update | 0.5 | autonomous |
| Self-review pass + full pytest suite | 0.5 | autonomous |

**Total: ~9 hours** end-to-end (single working day, possible today).

If the η formula transcription requires multiple iterations against R
clubSandwich source, add 2-3 hours buffer. Honest worst case: ~12 hours.

---

## 11. References

- Bell, R. M., McCaffrey, D. F. (2002). Bias reduction in standard errors
  for linear regression with multi-stage samples. *Survey Methodology*
  28(2), 169-179. **(Existing in paper.bib pending verification.)**
- Imbens, G. W., Kolesár, M. (2016). Robust standard errors in small
  samples: some practical advice. *Review of Economics and Statistics*
  98(4), 701-712. **(Existing in paper.bib pending verification.)**
- Pustejovsky, J. E., Tipton, E. (2018). Small-sample methods for
  cluster-robust variance estimation and hypothesis testing in fixed
  effects models. *Journal of Business & Economic Statistics* 36(4),
  672-683. DOI: 10.1080/07350015.2016.1247004. **(To be added to
  paper.bib at commit time after Crossref verification — CLAUDE.md §10.)**
- R package: `clubSandwich` (Pustejovsky 2017+). License: GPL-2 — used
  **only** as a black-box reference for fixture generation, **not**
  copied or ported.
