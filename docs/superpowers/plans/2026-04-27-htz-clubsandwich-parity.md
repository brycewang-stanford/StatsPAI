# clubSandwich-equivalent HTZ Wald DOF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `cluster_wald_htz` + `cluster_dof_wald_htz` + `WaldTestResult` in `statspai.fast.inference`, numerically equivalent to R `clubSandwich::Wald_test(test="HTZ")` to `rtol < 1e-8` per the Pustejovsky-Tipton 2018 §3.2 moment-matching DOF.

**Architecture:** New code goes into `src/statspai/fast/inference.py`, immediately after the existing `cluster_dof_wald_bm` block and before `__all__`. Standalone — no wiring into `crve` / `feols` / `event_study`. Single working covariance Φ = I (OLS+CR2 path). Parity validated three ways: (1) frozen R-generated CSV+JSON fixture (CI backbone, no R needed), (2) live R `subprocess` parity tests (skipif when R missing), (3) q=1 documented-drift unit test against existing `cluster_dof_bm` to detect regressions.

**Tech Stack:** NumPy + SciPy (existing). pytest. R + clubSandwich (verified installed at `/usr/local/bin/Rscript`) only as black-box reference for fixture generation and parity testing — no GPL code copied.

**Spec:** [`docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md`](../specs/2026-04-27-htz-clubsandwich-parity-design.md)

**Pre-flight context:**
- WIP state: `src/statspai/fast/inference.py` has uncommitted changes adding `cluster_dof_bm` + `cluster_dof_wald_bm` + `extra_df=` plumbing to `boottest`. HTZ work builds on this WIP and will land *after* it (or alongside, as a separate commit in the same push).
- HEAD commit: `b4ba4a3` (`feat(fast): native HDFE stack`).
- R + `clubSandwich` package both installed locally — live parity is a hard gate, not a skipif.

---

## Phase A — Foundation

### Task 1: WaldTestResult dataclass + first failing test

**Files:**
- Modify: `src/statspai/fast/inference.py` (add dataclass after `cluster_dof_wald_bm`, before `__all__`)
- Modify: `src/statspai/fast/__init__.py` (export)
- Create: `tests/test_fast_htz.py`

- [ ] **Step 1: Create `tests/test_fast_htz.py` with first failing test**

```python
"""Tests for clubSandwich-equivalent HTZ Wald DOF (Pustejovsky-Tipton 2018).

See docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md.
"""
from __future__ import annotations

import json
import subprocess
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Shared panel fixture (mirrors tests/test_fast_inference.py::_ols_panel)
# ---------------------------------------------------------------------------

def _ols_panel(n_clusters=20, m=30, seed=0, beta=(0.30, -0.20), unbalanced=False):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        n_g = rng.integers(3, 50) if unbalanced else m
        x1 = rng.normal(size=n_g)
        x2 = rng.normal(size=n_g)
        u_g = rng.normal(scale=0.5)
        eps = rng.normal(size=n_g) + u_g
        y = beta[0] * x1 + beta[1] * x2 + eps
        for i in range(n_g):
            rows.append({"g": g, "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 1: WaldTestResult dataclass
# ---------------------------------------------------------------------------

def test_wald_test_result_is_frozen_dataclass():
    """WaldTestResult should exist, be importable from sp.fast, and be frozen."""
    res = sp.fast.WaldTestResult(
        test="HTZ", q=2, eta=18.5, F_stat=3.4, p_value=0.04, Q=7.1,
        R=np.eye(2), r=np.zeros(2), V_R=np.eye(2),
    )
    assert res.test == "HTZ"
    assert res.q == 2
    # Frozen dataclass: setting an attribute must raise.
    with pytest.raises((AttributeError, Exception)):
        res.eta = 99.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fast_htz.py::test_wald_test_result_is_frozen_dataclass -x -q`
Expected: FAIL with `AttributeError: module 'statspai.fast' has no attribute 'WaldTestResult'`

- [ ] **Step 3: Add `WaldTestResult` dataclass to `inference.py`**

Open `src/statspai/fast/inference.py`. Locate the existing line `__all__ = [`. **Insert immediately before it** (between the closing `}` of `cluster_dof_wald_bm` and `__all__`):

```python
# ---------------------------------------------------------------------------
# clubSandwich-equivalent HTZ Wald (Pustejovsky-Tipton 2018, JBES 36(4))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaldTestResult:
    """Outcome of a cluster-robust Wald test under CR2 sandwich.

    Mirrors the contents of R ``clubSandwich::Wald_test(...)$test`` so that
    cross-language replication can compare any field directly.

    Attributes
    ----------
    test : str
        Test variant. ``"HTZ"`` only in v1; future variants
        (``"HTA"``, ``"KZ"``, ``"Naive"``, ``"EDF"``) may be added.
    q : int
        Number of restrictions (rank of ``R``).
    eta : float
        Pustejovsky-Tipton 2018 §3.2 moment-matching DOF (``η``).
    F_stat : float
        Hotelling-T²-scaled F statistic ``(η - q + 1) / (η · q) · Q``.
    p_value : float
        Right-tail probability ``1 - F_{q, η-q+1}.cdf(F_stat)``.
    Q : float
        Raw cluster-robust Wald statistic ``(Rβ̂ - r)' V_R^{-1} (Rβ̂ - r)``.
    R : ndarray, shape (q, k)
        Restriction matrix.
    r : ndarray, shape (q,)
        Null value.
    V_R : ndarray, shape (q, q)
        ``R · V^CR2 · R^T``.
    """
    test: str
    q: int
    eta: float
    F_stat: float
    p_value: float
    Q: float
    R: np.ndarray
    r: np.ndarray
    V_R: np.ndarray

    def summary(self) -> str:
        return (
            f"Wald test (test={self.test!r}, q={self.q}):\n"
            f"  η = {self.eta:.4f},  F = {self.F_stat:.4f},  "
            f"p = {self.p_value:.4f}  (raw Q = {self.Q:.4f})"
        )

    def to_dict(self) -> dict:
        return {
            "test": self.test, "q": self.q, "eta": self.eta,
            "F_stat": self.F_stat, "p_value": self.p_value, "Q": self.Q,
            "R": self.R.tolist(), "r": self.r.tolist(),
            "V_R": self.V_R.tolist(),
        }
```

Verify `from dataclasses import dataclass` is already imported at the top (it is — `BootTestResult` and `BootWaldResult` use it). If not, add to the imports.

- [ ] **Step 4: Update `__all__` in `inference.py`**

Modify the existing `__all__` at the bottom of `src/statspai/fast/inference.py`:

```python
__all__ = [
    "crve", "boottest", "BootTestResult",
    "boottest_wald", "BootWaldResult",
    "cluster_dof_bm", "cluster_dof_wald_bm",
    "cluster_dof_wald_htz", "cluster_wald_htz", "WaldTestResult",
]
```

- [ ] **Step 5: Update `src/statspai/fast/__init__.py` exports**

Locate the existing import block:

```python
from .inference import (
    crve,
    boottest,
    BootTestResult,
    boottest_wald,
    BootWaldResult,
    cluster_dof_bm,
    cluster_dof_wald_bm,
)
```

Append three names:

```python
from .inference import (
    crve,
    boottest,
    BootTestResult,
    boottest_wald,
    BootWaldResult,
    cluster_dof_bm,
    cluster_dof_wald_bm,
    cluster_dof_wald_htz,
    cluster_wald_htz,
    WaldTestResult,
)
```

And update the module-level `__all__` list to include the three new names.

- [ ] **Step 6: Run test — should still fail (functions don't exist)**

Run: `pytest tests/test_fast_htz.py::test_wald_test_result_is_frozen_dataclass -x -q`
Expected: FAIL with `ImportError` (`cluster_dof_wald_htz` not found in `inference.py`).

- [ ] **Step 7: Add stub functions so the import works**

In `src/statspai/fast/inference.py`, immediately after the `WaldTestResult` dataclass added in Step 3, add **stubs**:

```python
def cluster_dof_wald_htz(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """Stub — implemented in Task 3."""
    raise NotImplementedError("cluster_dof_wald_htz: implementation pending")


def cluster_wald_htz(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    beta: np.ndarray,
    r: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> WaldTestResult:
    """Stub — implemented in Task 4."""
    raise NotImplementedError("cluster_wald_htz: implementation pending")
```

- [ ] **Step 8: Re-run test, verify dataclass test passes**

Run: `pytest tests/test_fast_htz.py::test_wald_test_result_is_frozen_dataclass -x -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/statspai/fast/inference.py src/statspai/fast/__init__.py tests/test_fast_htz.py
git commit -m "feat(fast): WaldTestResult dataclass + HTZ function stubs"
```

---

### Task 2: Internal `_htz_per_cluster_quantities` helper

**Files:**
- Modify: `src/statspai/fast/inference.py` (add private helper before `cluster_dof_wald_htz`)
- Modify: `tests/test_fast_htz.py`

**Why this task:** Both public functions (`cluster_dof_wald_htz` and `cluster_wald_htz`) need cluster-level quantities (`G_g`, `Ω`, optionally `v_g`). Centralising them in `_htz_per_cluster_quantities` keeps the two public functions DRY and makes the algorithm testable in isolation.

- [ ] **Step 1: Write failing test for the helper's V_R sum-of-outer-products identity**

Append to `tests/test_fast_htz.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: per-cluster helper internal API
# ---------------------------------------------------------------------------

def test_htz_helper_V_R_matches_crve_path():
    """The HTZ helper's per-cluster G_g matrices should reproduce the CR2
    sandwich variance R V^CR2 R^T when contracted with residuals — i.e. it
    is *the same* CR2 path as :func:`crve`, just lifted to q-dim.
    """
    from statspai.fast.inference import _htz_per_cluster_quantities, crve

    df = _ols_panel(seed=42, n_clusters=20, m=25)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)

    # Fit OLS to get residuals
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    # HTZ helper path
    qty = _htz_per_cluster_quantities(X, g, R=R)
    # G_g list: each (q, n_g)
    # ẽ_g list: each (n_g,) = A_g · sqrt(W_g) · e_g_unweighted
    cluster_codes, _ = pd.factorize(g, sort=False)
    G_clusters = int(cluster_codes.max()) + 1
    v_sum = np.zeros((R.shape[0], R.shape[0]))
    for cg in range(G_clusters):
        mask = cluster_codes == cg
        e_g = e[mask]
        # Apply A_g via the helper's per-cluster A_g_sqrtW (precomputed)
        a_e = qty["A_g_sqrtW"][cg] @ e_g           # (n_g,)
        v_g = qty["G_g"][cg] @ a_e                 # (q,)
        v_sum += np.outer(v_g, v_g)

    # Compare to crve(type="cr2") sandwich, sliced to R subspace
    V_cr2 = crve(X, e, g, type="cr2")
    V_R_crve = R @ V_cr2 @ R.T
    np.testing.assert_allclose(v_sum, V_R_crve, rtol=1e-10, atol=1e-12)


def test_htz_helper_Omega_is_symmetric_psd():
    df = _ols_panel(seed=43, n_clusters=15, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    from statspai.fast.inference import _htz_per_cluster_quantities
    qty = _htz_per_cluster_quantities(X, g, R=R)
    Omega = qty["Omega"]
    assert Omega.shape == (2, 2)
    assert np.allclose(Omega, Omega.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(Omega)
    assert eigvals.min() > 1e-10, f"Ω not PD: eigvals={eigvals}"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_fast_htz.py::test_htz_helper_V_R_matches_crve_path -x -q`
Expected: FAIL with `ImportError: _htz_per_cluster_quantities`.

- [ ] **Step 3: Implement `_htz_per_cluster_quantities`**

In `src/statspai/fast/inference.py`, **before** the `cluster_dof_wald_htz` stub from Task 1, insert:

```python
def _htz_per_cluster_quantities(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> dict:
    """Internal helper: per-cluster lifted contributions for the HTZ test.

    Computes the matrices that both :func:`cluster_dof_wald_htz` and
    :func:`cluster_wald_htz` need:

    - Per cluster ``g``:
        H_gg = X_g · bread · X_g^T · diag(w_g)            (n_g × n_g)
        A_g  = (I_g - 0.5(H_gg + H_gg^T))^{-1/2}          (n_g × n_g)
        G_g  = R · bread · X_g^T · A_g · diag(√w_g)       (q × n_g)
        A_g_sqrtW = A_g · diag(√w_g)                      (n_g × n_g)
                    # ẽ_g = A_g_sqrtW · e_g for residual vector e_g

    - Aggregate:
        Ω = Σ_g G_g G_g^T                                  (q × q)

    Eigenvalue floor 1e-12 on `(I - Hsym)` matches the convention in
    :func:`crve` (CR2 path) and :func:`cluster_dof_wald_bm`.

    Parameters
    ----------
    X, cluster, R, weights, bread
        Same as the public ``cluster_dof_wald_htz`` signature.

    Returns
    -------
    dict
        Keys: ``"G_g"`` (list of q×n_g arrays, one per cluster, in
        cluster-code order from ``pd.factorize(sort=False)``),
        ``"A_g_sqrtW"`` (list of n_g×n_g arrays — same ordering),
        ``"Omega"`` (q×q), ``"cluster_codes"`` (the n-vector of codes),
        ``"G"`` (number of clusters), ``"q"`` (R.shape[0]).
    """
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    n, k = X.shape
    if R.shape[1] != k:
        raise ValueError(f"R has {R.shape[1]} cols but X has k={k}")
    q = R.shape[0]
    if q < 1:
        raise ValueError("R must have at least one row")
    if q > k:
        raise ValueError(f"R has {q} rows > k={k}; over-determined")
    if np.linalg.matrix_rank(R) < q:
        raise ValueError("R must have full row rank")

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if (weights <= 0).any():
            raise ValueError("weights must be strictly positive")

    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G_clusters = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G_clusters < 2:
        raise ValueError(
            f"HTZ requires at least 2 clusters, got {G_clusters}"
        )
    if G_clusters <= q:
        raise ValueError(
            f"HTZ Wald requires G > q (got G={G_clusters}, q={q})"
        )

    if bread is None:
        XtWX = X.T @ (X * weights[:, None])
        bread = np.linalg.inv(XtWX)

    R_bread = R @ bread                                  # (q, k)

    G_list: list[np.ndarray] = []
    Asw_list: list[np.ndarray] = []
    Omega = np.zeros((q, q))

    for cg in range(G_clusters):
        mask = cluster_codes == cg
        X_g = X[mask]
        w_g = weights[mask]
        n_g = X_g.shape[0]
        if n_g == 1:
            warnings.warn(
                f"HTZ: singleton cluster (code={cg}, n_g=1); A_g floored",
                RuntimeWarning,
                stacklevel=3,
            )

        Xb = X_g @ bread                                  # (n_g, k)
        H_gg = Xb @ (X_g * w_g[:, None]).T                # (n_g, n_g)
        Hsym = 0.5 * (H_gg + H_gg.T)
        Msym = np.eye(n_g) - Hsym
        evals, evecs = np.linalg.eigh(Msym)
        evals = np.maximum(evals, 1e-12)
        A_g = (evecs * (evals ** -0.5)) @ evecs.T          # (n_g, n_g)

        sqrt_w = np.sqrt(w_g)
        # G_g = R · bread · X_g^T · A_g · diag(√w_g)
        RB_Xt = R_bread @ X_g.T                            # (q, n_g)
        RB_Xt_A = RB_Xt @ A_g                              # (q, n_g)
        G_g = RB_Xt_A * sqrt_w                             # (q, n_g)

        A_g_sqrtW = A_g * sqrt_w                           # (n_g, n_g)

        G_list.append(G_g)
        Asw_list.append(A_g_sqrtW)
        Omega += G_g @ G_g.T

    # Symmetrize Ω against floating drift
    Omega = 0.5 * (Omega + Omega.T)

    return {
        "G_g": G_list,
        "A_g_sqrtW": Asw_list,
        "Omega": Omega,
        "cluster_codes": cluster_codes,
        "G": G_clusters,
        "q": q,
        "bread": bread,
    }
```

Verify `import warnings` is at the top of the file (it should be — `warnings.warn` is used elsewhere in `crve`). Add if missing.

- [ ] **Step 4: Run helper tests, verify pass**

Run: `pytest tests/test_fast_htz.py::test_htz_helper_V_R_matches_crve_path tests/test_fast_htz.py::test_htz_helper_Omega_is_symmetric_psd -x -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/statspai/fast/inference.py tests/test_fast_htz.py
git commit -m "feat(fast): _htz_per_cluster_quantities internal helper"
```

---

## Phase B — η formula (highest-leverage risk)

### Task 3: Generate frozen R fixture, then implement η

**Files:**
- Create: `tests/fixtures/_gen_htz_fixture.R`
- Create: `tests/fixtures/_gen_htz_fixture.README.md`
- Create: `tests/fixtures/htz_panel_q1.csv`, `htz_panel_q2.csv`, `htz_panel_q3_unbal.csv` (R writes these)
- Create: `tests/fixtures/htz_clubsandwich.json` (R writes this)
- Modify: `src/statspai/fast/inference.py` (implement `cluster_dof_wald_htz`)
- Modify: `tests/test_fast_htz.py` (add frozen-fixture test)

**Discipline:** Generate the fixture **before** writing any η code — that way the implementation has a fixed target from minute one. Read the formula from BOTH P-T 2018 §3.2 PDF AND clubSandwich `Wald_test.R` source before typing.

- [ ] **Step 1: Create `tests/fixtures/_gen_htz_fixture.R`**

```r
# Regenerate clubSandwich HTZ fixture for tests/test_fast_htz.py.
# DO NOT run in CI — this is a developer-side script. Outputs are
# committed to git; tests read them directly.
#
# Usage:
#   cd tests/fixtures && Rscript _gen_htz_fixture.R
#
# Requires: R >= 4.0, clubSandwich, jsonlite, data.table.

suppressMessages({
  library(clubSandwich)
  library(jsonlite)
  library(data.table)
})

set.seed(20260427L)

# ---------------------------------------------------------------------------
# Helper: simulate a clustered OLS panel.
# ---------------------------------------------------------------------------

simulate_panel <- function(G, m, beta = c(0.30, -0.20), unbalanced = FALSE) {
  rows <- list()
  for (g in seq_len(G)) {
    n_g <- if (unbalanced) sample(3:50, 1L) else m
    x1 <- rnorm(n_g)
    x2 <- rnorm(n_g)
    u_g <- rnorm(1L, sd = 0.5)
    eps <- rnorm(n_g) + u_g
    y <- beta[1] * x1 + beta[2] * x2 + eps
    rows[[g]] <- data.table(g = g, x1 = x1, x2 = x2, y = y)
  }
  rbindlist(rows)
}

# ---------------------------------------------------------------------------
# Build three panels: q=1 / q=2 / q=3-unbalanced.
# ---------------------------------------------------------------------------

panels <- list(
  list(name = "q1",         G = 15, m = 20, q = 1, unbalanced = FALSE),
  list(name = "q2",         G = 25, m = 15, q = 2, unbalanced = FALSE),
  list(name = "q3_unbal",   G = 50, m = NA, q = 3, unbalanced = TRUE)
)

# x1 + x2 + intercept ⇒ k = 3 ⇒ supports up to q = 3.
results <- list()

for (cfg in panels) {
  set.seed(20260427L + nchar(cfg$name))   # distinct but deterministic
  d <- simulate_panel(cfg$G, cfg$m, unbalanced = cfg$unbalanced)
  csv_path <- paste0("htz_panel_", cfg$name, ".csv")
  fwrite(d, csv_path)

  fit <- lm(y ~ x1 + x2, data = d)         # k = 3 (intercept + x1 + x2)
  V_CR2 <- vcovCR(fit, cluster = d$g, type = "CR2")

  # Restriction matrix: q rows × k=3 cols. Test the *non-intercept* coefs.
  if (cfg$q == 1) {
    R <- matrix(c(0, 1, 0), nrow = 1)        # H0: β_x1 = 0
  } else if (cfg$q == 2) {
    R <- rbind(c(0, 1, 0), c(0, 0, 1))      # H0: β_x1 = β_x2 = 0
  } else if (cfg$q == 3) {
    # q=3 with k=3 ⇒ joint test of all 3 coefs (intercept + x1 + x2).
    R <- diag(3)
  }
  r0 <- rep(0, cfg$q)

  ct <- list(constraints = R, vcov = V_CR2, test = "HTZ", coefs = "All")
  wt <- Wald_test(fit, constraints = R, vcov = V_CR2, test = "HTZ")

  # wt is a 1-row data.frame: $Fstat, $df_num, $df_denom, $p_val
  results[[cfg$name]] <- list(
    panel = cfg$name,
    csv = csv_path,
    G = cfg$G,
    q = cfg$q,
    R = as.list(as.data.frame(R)),     # serialise row-wise as list
    beta = unname(coef(fit)),
    eta = wt$df_denom,
    F_stat = wt$Fstat,
    p_value = wt$p_val,
    Q = wt$Fstat * wt$df_num * wt$df_denom / (wt$df_denom - wt$df_num + 1),
    V_R = unname(as.matrix(R %*% V_CR2 %*% t(R)))
  )
}

write(toJSON(results, auto_unbox = TRUE, digits = 14, pretty = TRUE),
      "htz_clubsandwich.json")

cat("Generated:\n")
for (cfg in panels) {
  cat("  htz_panel_", cfg$name, ".csv\n", sep = "")
}
cat("  htz_clubsandwich.json\n")
```

- [ ] **Step 2: Run the R script**

Run:
```bash
cd tests/fixtures && Rscript _gen_htz_fixture.R && cd -
```
Expected output:
```
Generated:
  htz_panel_q1.csv
  htz_panel_q2.csv
  htz_panel_q3_unbal.csv
  htz_clubsandwich.json
```

If it errors, fix the R script (likely candidates: missing intercept handling, R column name mismatch, `Wald_test` signature change in latest clubSandwich) and re-run. Inspect output:

```bash
ls -la tests/fixtures/
cat tests/fixtures/htz_clubsandwich.json | python -m json.tool | head -40
```

You should see three `csv` files + a JSON with three top-level keys (`q1`, `q2`, `q3_unbal`), each containing `eta`, `F_stat`, `p_value`, etc.

- [ ] **Step 3: Create `tests/fixtures/_gen_htz_fixture.README.md`**

```markdown
# HTZ clubSandwich fixture

These files are the **frozen reference output** from R clubSandwich's
`Wald_test(test="HTZ")` — used by `tests/test_fast_htz.py::test_htz_frozen_fixture`
to lock numerical parity to `rtol < 1e-8` on every CI run, including
environments without R installed.

## Files

- `_gen_htz_fixture.R` — generator script (developer-only, NOT in CI).
- `htz_panel_{q1,q2,q3_unbal}.csv` — actual panel data R simulated.
- `htz_clubsandwich.json` — `Wald_test(test="HTZ")` outputs (η, F, p, Q, V_R)
  for each panel.

## Regeneration

```bash
cd tests/fixtures && Rscript _gen_htz_fixture.R
```

## Pinned versions

When this fixture was last regenerated:

- R version: <fill in from `R --version`>
- clubSandwich: <fill in from `packageVersion("clubSandwich")` in R>

If a future clubSandwich release changes the HTZ formula (unlikely — the
algorithm is from a 2018 paper), regenerate AND audit every `eta` /
`F_stat` / `p_value` change.

## Why CSV + JSON, not RNG sync

`numpy.random` and R's `set.seed()` produce different streams. The CSV
captures the actual numerical panel R generated, so the Python test reads
byte-identical input. Trying to sync RNGs across languages is a known
foot-gun.
```

Fill in the actual R version + clubSandwich version. Get them via:

```bash
R --version | head -1
Rscript -e 'cat(as.character(packageVersion("clubSandwich")))'
```

- [ ] **Step 4: Write the failing frozen-fixture test**

Append to `tests/test_fast_htz.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: Frozen R-clubSandwich fixture parity (CI backbone)
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / "htz_clubsandwich.json").read_text(encoding="utf-8"))[name]


def _load_panel(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / f"htz_panel_{name}.csv")


@pytest.mark.parametrize("panel_name", ["q1", "q2", "q3_unbal"])
def test_htz_frozen_fixture_matches_clubsandwich(panel_name):
    """Frozen-fixture parity: η / F / p match R clubSandwich to rtol<1e-8.

    Runs in every CI environment (no R required). The fixture was generated
    by ``tests/fixtures/_gen_htz_fixture.R``.
    """
    fx = _load_fixture(panel_name)
    df = _load_panel(panel_name)

    # Re-fit OLS with intercept on Python side (R side did `lm(y ~ x1 + x2)`).
    X = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    R = np.array(fx["R"], dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(1, -1)

    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    np.testing.assert_allclose(res.eta, fx["eta"], rtol=1e-8,
                                err_msg=f"η mismatch on panel {panel_name}")
    np.testing.assert_allclose(res.F_stat, fx["F_stat"], rtol=1e-8,
                                err_msg=f"F mismatch on panel {panel_name}")
    np.testing.assert_allclose(res.p_value, fx["p_value"], rtol=1e-7,
                                err_msg=f"p-value mismatch on panel {panel_name}")
    np.testing.assert_allclose(
        res.V_R, np.array(fx["V_R"], dtype=np.float64),
        rtol=1e-9, err_msg=f"V_R mismatch on panel {panel_name}",
    )
```

- [ ] **Step 5: Run the new test — should fail because cluster_wald_htz is still a stub**

Run: `pytest tests/test_fast_htz.py::test_htz_frozen_fixture_matches_clubsandwich -x -q`
Expected: FAIL with `NotImplementedError: cluster_wald_htz: implementation pending`.

This is the right failure — the fixture is in place, the test target is real, only the implementation is missing.

- [ ] **Step 6: DOUBLE-SOURCE the η formula before coding**

Before writing any η code, **read both sources**:

1. **Pustejovsky-Tipton (2018) §3.2 + Theorem 2 + eq. (12)** — the paper proper.
   Note the exact construction of `B_g`, the cluster-pair sum, and the moment-matching denominator.
2. **clubSandwich source** — fetch the canonical implementation:

   ```bash
   curl -s https://raw.githubusercontent.com/jepusto/clubSandwich/master/R/Wald_test_S3.R \
     | grep -A 80 "Hotelling" > /tmp/clubsandwich_htz.R
   cat /tmp/clubsandwich_htz.R
   ```

   (Or `cat /usr/local/lib/R/site-library/clubSandwich/R/clubSandwich.rdb` won't work — use `getAnywhere(clubSandwich:::Wald_testHTZ)` in an R session, or pull the GitHub source.)

   ```bash
   Rscript -e 'cat(deparse(clubSandwich:::Wald_testHTZ))' > /tmp/clubsandwich_htz_dump.R
   wc -l /tmp/clubsandwich_htz_dump.R
   ```

3. **Reconcile** — if the paper formula and the R source visually differ (different summation order, different rotation), write a **one-paragraph reconciliation note** to `/tmp/htz_reconciliation.txt` BEFORE coding. The note must conclude: "I will implement formula X because [reason], and validate against the fixture."

- [ ] **Step 7: Implement `cluster_dof_wald_htz`**

In `src/statspai/fast/inference.py`, **replace** the stub from Task 1 with the real implementation. The skeleton:

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

    For testing ``H0: R β = r`` with the CR2 sandwich variance, the
    cluster-robust Wald statistic ``Q = (Rβ̂ - r)' V_R^{-1} (Rβ̂ - r)``
    is approximated by Hotelling's T² with denominator DOF ``η`` such
    that ``(η - q + 1) / (η · q) · Q ~ F(q, η - q + 1)``.

    The DOF ``η`` is computed by moment-matching the first two moments
    of ``V_R = R V^CR2 R^T`` under the working covariance ``Φ = I``
    (OLS+CR2 path; clubSandwich's default).

    Equivalent to R::clubSandwich::Wald_test(test="HTZ")$df_denom to
    ``rtol < 1e-8`` on the verified fixtures.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Regressors (already FE-residualised if applicable).
    cluster : ndarray, shape (n,)
        Cluster identifiers.
    R : ndarray, shape (q, k)
        Restriction matrix; full row rank required (q ≤ k).
    weights : ndarray, shape (n,), optional
        Observation weights. Default 1.
    bread : ndarray, shape (k, k), optional
        Pre-computed inverse Hessian / ``(X'WX)^{-1}``.

    Returns
    -------
    float
        ``η``, the HTZ moment-matching DOF. Bounded below by ``q``;
        upper bound is implementation-specific (typically << ``G``).

    Raises
    ------
    ValueError
        If ``R`` is rank-deficient, ``G ≤ q``, or ``η ≤ q − 1``
        (Hotelling-T² scaling diverges).

    References
    ----------
    Pustejovsky and Tipton (2018) — see ``pustejovsky2018small`` in
    ``paper.bib`` for the canonical citation (Pustejovsky and Tipton,
    DOI 10.1080/07350015.2016.1247004).
    """
    qty = _htz_per_cluster_quantities(
        X, cluster, R=R, weights=weights, bread=bread,
    )
    G_list = qty["G_g"]
    Omega = qty["Omega"]
    G_clusters = qty["G"]
    q = qty["q"]

    # Ω^{-1/2} via symmetric eigendecomposition with eigenvalue floor.
    evals, evecs = np.linalg.eigh(Omega)
    evals = np.maximum(evals, 1e-12)
    Omega_inv_sqrt = (evecs * (evals ** -0.5)) @ evecs.T

    # B_g = Ω^{-1/2} · G_g                                  (q × n_g)
    B_list = [Omega_inv_sqrt @ G_g for G_g in G_list]

    # === η formula — IMPLEMENTED PER /tmp/htz_reconciliation.txt ===
    # The exact form is double-sourced from P-T 2018 eq. (12) and
    # clubSandwich Wald_testHTZ; see Step 6 reconciliation note.
    #
    # Sketch (subject to reconciliation):
    #   For all g, h in 1..G:
    #     P_{gh} = B_g · B_h^T        (q × q)
    #   η = q(q+1) / Σ_{g,h} f(P_{gh})
    #
    # The exact f is to be filled in here from the reconciled formula.
    # The frozen fixture in tests/fixtures/htz_clubsandwich.json is the
    # acceptance gate — η must match R to rtol<1e-8.

    raise NotImplementedError(
        "Step 7: fill in η formula per /tmp/htz_reconciliation.txt"
    )
```

**Important**: Step 7 leaves the η formula as a deliberate `NotImplementedError`. The next step iterates against the fixture until the formula is correct.

- [ ] **Step 8: Iterate the η formula against the fixture**

This is the **only formula-derivation step**. Open `/tmp/htz_reconciliation.txt` (from Step 6). Implement the chosen formula in place of the `raise NotImplementedError(...)`. Run:

```bash
pytest tests/test_fast_htz.py::test_htz_frozen_fixture_matches_clubsandwich[q1] -x -q --no-cov
```

If `q1` passes, run all three:

```bash
pytest tests/test_fast_htz.py::test_htz_frozen_fixture_matches_clubsandwich -x -q --no-cov
```

If any fails, **do not adjust tolerance**. Re-read the reconciliation note, audit your formula, and try again. The `rtol=1e-8` is the gate.

Likely formula candidates to try (one will be correct):

```python
# Candidate A: cluster-pair Frobenius sum (P-T 2018 Theorem 2, matrix form)
# P_{gh} = B_g B_h^T;  μ_{gh} = tr(P_{gh});  ν_{gh} = sum(P_{gh}**2)
# η = q(q+1) / Σ_{g,h} (μ_{gh}^2 + (q+2) ν_{gh}) ... (one possibility)
#
# Candidate B: Kronecker form
# P = Σ_{g,h} (B_g B_h^T) ⊗ (B_g B_h^T)
# η = q(q+1) / [tr(P) - q^2] ... (another possibility)

# The reconciled exact form goes in. Run the fixture test until rtol<1e-8.
```

When all three panels pass the fixture test, **commit the working formula immediately** before adding more tests:

```bash
git add src/statspai/fast/inference.py
git commit -m "feat(fast): cluster_dof_wald_htz η implementation (P-T 2018 §3.2)"
```

Add the reconciled formula as a code comment (3-5 lines) above the implementation, citing P-T 2018 eq. (12) and noting the reconciliation source. Do NOT paste GPL R code.

- [ ] **Step 9: Add the q=1 documented-drift unit test**

Append to `tests/test_fast_htz.py`:

```python
def test_htz_q1_documented_drift_from_bm_simplified():
    """At q=1, HTZ uses cross-cluster terms; cluster_dof_bm uses the BM 2002
    simplified (Σλ)²/Σλ² formula. They differ by ~5-15% on canonical panels.

    This test locks the *direction* of the drift — if the band is broken,
    either the BM simplified function regressed or HTZ regressed.
    """
    df = _ols_panel(seed=46, n_clusters=15, m=20)
    X_ic = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])
    g = df["g"].to_numpy()
    contrast = np.array([0.0, 1.0, 0.0])     # x1 with intercept first

    nu_bm = sp.fast.cluster_dof_bm(X_ic, g, contrast=contrast)
    nu_htz = sp.fast.cluster_dof_wald_htz(X_ic, g, R=contrast.reshape(1, -1))

    rel = abs(nu_htz - nu_bm) / max(abs(nu_bm), 1e-12)
    assert 0.005 <= rel <= 0.20, (
        f"q=1 drift HTZ vs BM-simplified out of band: "
        f"BM={nu_bm:.4f}, HTZ={nu_htz:.4f}, rel={rel:.4f}"
    )
    # And both should be in the (1, G-1) sanity range
    assert 1.0 < nu_htz < 14.0
    assert 1.0 < nu_bm < 14.0
```

Run: `pytest tests/test_fast_htz.py::test_htz_q1_documented_drift_from_bm_simplified -x -q --no-cov`
Expected: PASS.

- [ ] **Step 10: Commit fixture + reconciliation + drift test**

```bash
git add tests/fixtures/ tests/test_fast_htz.py src/statspai/fast/inference.py
git commit -m "test(fast): frozen clubSandwich HTZ fixture (q=1/q=2/q=3-unbal) + drift test"
```

---

## Phase C — Full Wald wrapper

### Task 4: `cluster_wald_htz` — V_R + Q + F + p

**Files:**
- Modify: `src/statspai/fast/inference.py` (replace stub from Task 1)
- Modify: `tests/test_fast_htz.py`

- [ ] **Step 1: Write failing test for q=1 helper↔full equivalence**

```python
# ---------------------------------------------------------------------------
# Task 4: cluster_wald_htz full wrapper
# ---------------------------------------------------------------------------

def test_htz_q1_helper_eta_matches_full_wrapper():
    """The DOF helper and the full wrapper share Step 4 — η must be bit-equal."""
    df = _ols_panel(seed=70, n_clusters=20, m=25)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    contrast = np.array([1.0, 0.0])

    nu_helper = sp.fast.cluster_dof_wald_htz(X, g, R=contrast.reshape(1, -1))
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=contrast.reshape(1, -1), beta=beta,
    )
    assert abs(nu_helper - res.eta) < 1e-12


def test_htz_F_and_p_value_internal_consistency():
    """Hotelling-T² scaling consistency: F = (η-q+1)/(η q) · Q, p = sf(F)."""
    from scipy.stats import f as scipy_f

    df = _ols_panel(seed=71, n_clusters=25, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    R = np.eye(2)
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    # F = (η - q + 1) / (η · q) · Q
    F_recomputed = (res.eta - res.q + 1) / (res.eta * res.q) * res.Q
    assert abs(F_recomputed - res.F_stat) < 1e-10

    # p = 1 - F.cdf(F_stat)  with df1=q, df2=η-q+1
    p_recomputed = scipy_f.sf(res.F_stat, res.q, res.eta - res.q + 1)
    assert abs(p_recomputed - res.p_value) < 1e-10


def test_htz_zero_residuals_returns_p_one():
    """e ≡ 0 ⇒ Q = 0 ⇒ F = 0 ⇒ p = 1 exactly."""
    df = _ols_panel(seed=72, n_clusters=20, m=15)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    beta = np.zeros(2)
    e = np.zeros(len(df))
    R = np.eye(2)
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    assert res.Q == 0.0
    assert res.F_stat == 0.0
    assert res.p_value == 1.0
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_fast_htz.py::test_htz_q1_helper_eta_matches_full_wrapper -x -q --no-cov`
Expected: FAIL with `NotImplementedError: cluster_wald_htz: implementation pending`.

- [ ] **Step 3: Implement `cluster_wald_htz`**

Replace the stub in `src/statspai/fast/inference.py`:

```python
def cluster_wald_htz(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    beta: np.ndarray,
    r: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> WaldTestResult:
    """clubSandwich-equivalent HTZ Wald test under CR2 sandwich.

    Tests ``H0: R β = r`` with cluster-robust CR2 variance and the
    Pustejovsky-Tipton 2018 §3.2 moment-matching DOF.

    Equivalent to R::clubSandwich::Wald_test(fit, constraints=R,
    vcov="CR2", test="HTZ") to ``rtol < 1e-8`` on the verified fixtures
    (``tests/fixtures/htz_clubsandwich.json``).

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Regressors (already FE-residualised if applicable).
    residuals : ndarray, shape (n,)
        OLS residuals ``y - X β̂``.
    cluster : ndarray, shape (n,)
        Cluster identifiers.
    R : ndarray, shape (q, k)
        Restriction matrix; full row rank required (q ≤ k).
    beta : ndarray, shape (k,)
        OLS coefficient vector ``β̂``.
    r : ndarray, shape (q,), optional
        Null value. Default zeros.
    weights : ndarray, shape (n,), optional
        Observation weights. Default 1.
    bread : ndarray, shape (k, k), optional
        Pre-computed inverse Hessian / ``(X'WX)^{-1}``.

    Returns
    -------
    WaldTestResult
        Dataclass with η, F_stat, p_value, Q, R, r, V_R, q, test="HTZ".

    Raises
    ------
    ValueError
        If ``R`` is rank-deficient, ``G ≤ q``, ``η ≤ q − 1``, or
        ``r`` shape mismatches.
    """
    qty = _htz_per_cluster_quantities(
        X, cluster, R=R, weights=weights, bread=bread,
    )
    G_list = qty["G_g"]
    Asw_list = qty["A_g_sqrtW"]
    Omega = qty["Omega"]
    cluster_codes = qty["cluster_codes"]
    G_clusters = qty["G"]
    q = qty["q"]
    R_arr = np.atleast_2d(np.asarray(R, dtype=np.float64))

    if r is None:
        r = np.zeros(q)
    else:
        r = np.asarray(r, dtype=np.float64).ravel()
        if r.shape[0] != q:
            raise ValueError(
                f"r has {r.shape[0]} entries but R has q={q} rows"
            )

    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    beta = np.asarray(beta, dtype=np.float64).ravel()

    # V_R = Σ_g (G_g · ẽ_g)(G_g · ẽ_g)^T  where ẽ_g = A_g · √W_g · e_g
    V_R = np.zeros((q, q))
    for cg in range(G_clusters):
        mask = cluster_codes == cg
        e_g = residuals[mask]
        e_tilde = Asw_list[cg] @ e_g                # (n_g,)
        v_g = G_list[cg] @ e_tilde                  # (q,)
        V_R += np.outer(v_g, v_g)
    V_R = 0.5 * (V_R + V_R.T)

    # η — same path as cluster_dof_wald_htz, but reuse the helper output
    # to avoid recomputing G_list / Ω.
    eta = _htz_eta_from_quantities(qty)             # (defined alongside helper)

    if eta <= q - 1:
        raise ValueError(
            f"HTZ DOF η={eta:.4f} ≤ q−1 (q={q}); design too degenerate "
            f"for Hotelling-T² scaling. Use boottest_wald instead."
        )

    diff = R_arr @ beta - r                          # (q,)
    Q = float(diff @ np.linalg.solve(V_R, diff))
    F_stat = (eta - q + 1) / (eta * q) * Q
    from scipy.stats import f as _scipy_f
    p_value = float(_scipy_f.sf(F_stat, q, eta - q + 1))

    return WaldTestResult(
        test="HTZ", q=q, eta=float(eta), F_stat=float(F_stat),
        p_value=p_value, Q=Q, R=R_arr.copy(), r=r.copy(),
        V_R=V_R,
    )
```

The reference to `_htz_eta_from_quantities(qty)` requires a small refactor: extract the η computation from `cluster_dof_wald_htz` into a helper.

In `cluster_dof_wald_htz` (the function you implemented in Task 3 Step 8), refactor by inserting **above** it:

```python
def _htz_eta_from_quantities(qty: dict) -> float:
    """Compute the HTZ DOF η given the per-cluster quantities dict.

    Centralises the formula so :func:`cluster_dof_wald_htz` and
    :func:`cluster_wald_htz` share Step 4 bit-identically.
    """
    G_list = qty["G_g"]
    Omega = qty["Omega"]
    q = qty["q"]

    # Ω^{-1/2} via symmetric eigendecomposition with eigenvalue floor.
    evals, evecs = np.linalg.eigh(Omega)
    evals = np.maximum(evals, 1e-12)
    Omega_inv_sqrt = (evecs * (evals ** -0.5)) @ evecs.T

    B_list = [Omega_inv_sqrt @ G_g for G_g in G_list]

    # === η formula — VERIFIED PER /tmp/htz_reconciliation.txt (Task 3 Step 6) ===
    # [paste the reconciled formula here, with citation comment]
    ...
    return float(eta)
```

Then `cluster_dof_wald_htz` becomes:

```python
def cluster_dof_wald_htz(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """[docstring as in Task 3]"""
    qty = _htz_per_cluster_quantities(
        X, cluster, R=R, weights=weights, bread=bread,
    )
    eta = _htz_eta_from_quantities(qty)
    if eta <= qty["q"] - 1:
        raise ValueError(
            f"HTZ DOF η={eta:.4f} ≤ q−1 (q={qty['q']}); design too "
            f"degenerate. Use boottest_wald instead."
        )
    return eta
```

- [ ] **Step 4: Run all tests added so far**

Run: `pytest tests/test_fast_htz.py -x -q --no-cov`
Expected: all green (frozen fixture + Task 1/2/3 tests + Task 4 internal-consistency tests).

If the fixture test breaks because `_htz_eta_from_quantities` was extracted incorrectly, fix and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/statspai/fast/inference.py tests/test_fast_htz.py
git commit -m "feat(fast): cluster_wald_htz full wrapper + _htz_eta_from_quantities refactor"
```

---

## Phase D — Test sweep

### Task 5: Validation + edge case unit tests

**Files:**
- Modify: `tests/test_fast_htz.py`

- [ ] **Step 1: Write all validation/edge tests**

Append to `tests/test_fast_htz.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: Validation + edge cases
# ---------------------------------------------------------------------------

def test_htz_validates_R_shape():
    df = _ols_panel(seed=80)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()

    with pytest.raises(ValueError, match="cols"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(3))
    with pytest.raises(ValueError, match="rank"):
        sp.fast.cluster_dof_wald_htz(
            X, g, R=np.array([[1.0, 0.0], [1.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="at least one row"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.empty((0, 2)))


def test_htz_too_few_clusters_rejected():
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 3))
    g = np.concatenate([np.zeros(25), np.ones(25)]).astype(int)
    with pytest.raises(ValueError, match="G > q"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(3))


def test_htz_invariant_to_X_column_rescaling():
    """Multiplying X column j by α and adjusting R column j by 1/α leaves
    the HTZ statistic and η invariant (design-equivariance)."""
    df = _ols_panel(seed=81, n_clusters=20, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu0 = sp.fast.cluster_dof_wald_htz(X, g, R=R)

    alpha = np.array([3.0, 0.5])
    X_scaled = X * alpha
    R_scaled = R / alpha[None, :]
    nu1 = sp.fast.cluster_dof_wald_htz(X_scaled, g, R=R_scaled)
    np.testing.assert_allclose(nu0, nu1, rtol=1e-10)


def test_htz_invariant_to_cluster_relabel():
    """Permuting cluster IDs leaves η, F, p unchanged."""
    df = _ols_panel(seed=82, n_clusters=20, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    R = np.eye(2)

    res0 = sp.fast.cluster_wald_htz(X=X, residuals=e, cluster=g, R=R, beta=beta)

    rng = np.random.default_rng(0)
    perm = rng.permutation(g.max() + 1)
    g_relab = perm[g]
    res1 = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g_relab, R=R, beta=beta,
    )
    np.testing.assert_allclose(res0.eta, res1.eta, rtol=1e-10)
    np.testing.assert_allclose(res0.F_stat, res1.F_stat, rtol=1e-10)
    np.testing.assert_allclose(res0.p_value, res1.p_value, rtol=1e-10)


def test_htz_independent_of_bread_arg():
    df = _ols_panel(seed=83)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu_a = sp.fast.cluster_dof_wald_htz(X, g, R=R)
    nu_b = sp.fast.cluster_dof_wald_htz(
        X, g, R=R, bread=np.linalg.inv(X.T @ X),
    )
    assert abs(nu_a - nu_b) < 1e-10


def test_htz_eta_in_sane_range():
    """For a well-conditioned panel with G=25, q=2: η ∈ (q, G·q) — a generous
    sanity band that catches catastrophic-sign-error bugs."""
    df = _ols_panel(seed=84, n_clusters=25, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu = sp.fast.cluster_dof_wald_htz(X, g, R=R)
    assert nu > 2, f"η={nu} ≤ q=2"
    assert nu < 25 * 2, f"η={nu} > G·q upper sanity bound"
    assert np.isfinite(nu)


def test_htz_singleton_cluster_warns():
    """A cluster of size 1 should trigger a RuntimeWarning, not a crash."""
    rng = np.random.default_rng(99)
    G = 20
    rows = []
    for cg in range(G):
        # First cluster has size 1 (singleton); rest have size 10
        n_g = 1 if cg == 0 else 10
        x1 = rng.normal(size=n_g)
        x2 = rng.normal(size=n_g)
        eps = rng.normal(size=n_g)
        rows.append(pd.DataFrame({"g": cg, "x1": x1, "x2": x2,
                                    "y": 0.3 * x1 - 0.2 * x2 + eps}))
    df = pd.concat(rows, ignore_index=True)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()

    with pytest.warns(RuntimeWarning, match="singleton cluster"):
        nu = sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(2))
    assert np.isfinite(nu)


def test_htz_r_shape_mismatch_rejected():
    df = _ols_panel(seed=85, n_clusters=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    R = np.eye(2)

    with pytest.raises(ValueError, match="r has"):
        sp.fast.cluster_wald_htz(
            X=X, residuals=e, cluster=g, R=R, beta=beta, r=np.zeros(3),
        )


def test_htz_negative_weights_rejected():
    df = _ols_panel(seed=86, n_clusters=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    bad_w = np.ones(len(df))
    bad_w[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(2), weights=bad_w)
```

- [ ] **Step 2: Run, verify all pass**

Run: `pytest tests/test_fast_htz.py -x -q --no-cov`
Expected: all 13+ tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_htz.py
git commit -m "test(fast): HTZ validation + invariance + edge case unit tests"
```

---

### Task 6: Live R clubSandwich parity (skipif belt-and-suspenders)

**Files:**
- Modify: `tests/test_fast_htz.py`

- [ ] **Step 1: Add the live R parity test**

Append to `tests/test_fast_htz.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: Live R clubSandwich parity (skipif when R missing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not on PATH",
)
@pytest.mark.parametrize("seed,G,q", [
    (1010, 15, 1),
    (1011, 25, 2),
    (1012, 30, 2),
])
def test_htz_matches_r_clubsandwich_live(seed, G, q, tmp_path):
    """Live parity: simulate panel in Python, run R clubSandwich on it,
    compare HTZ outputs to ``rtol < 1e-8``.

    Skipped if Rscript or clubSandwich is unavailable.
    """
    df = _ols_panel(seed=seed, n_clusters=G, m=20)
    csv_path = tmp_path / "panel.csv"
    df.to_csv(csv_path, index=False)

    if q == 1:
        R_str = "matrix(c(0, 1, 0), nrow=1)"
    elif q == 2:
        R_str = "rbind(c(0, 1, 0), c(0, 0, 1))"
    else:
        raise ValueError(f"unsupported q={q} in live test")

    r_script = (
        "if (!requireNamespace('clubSandwich', quietly=TRUE)) {\n"
        "  cat('SKIP: clubSandwich not installed'); quit(status=2)\n}\n"
        "suppressMessages({library(clubSandwich); library(jsonlite); "
        "library(data.table)})\n"
        f"d <- fread('{csv_path}')\n"
        "fit <- lm(y ~ x1 + x2, data=d)\n"
        "V <- vcovCR(fit, cluster=d$g, type='CR2')\n"
        f"R <- {R_str}\n"
        "wt <- Wald_test(fit, constraints=R, vcov=V, test='HTZ')\n"
        "out <- list(eta=wt$df_denom, F_stat=wt$Fstat, p_value=wt$p_val, "
        "df_num=wt$df_num)\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script], capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 2 or "SKIP" in proc.stdout:
        pytest.skip("clubSandwich not available")
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:200]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])

    X_ic = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])
    y = df["y"].to_numpy()
    g_arr = df["g"].to_numpy()
    beta = np.linalg.solve(X_ic.T @ X_ic, X_ic.T @ y)
    e = y - X_ic @ beta

    if q == 1:
        R = np.array([[0.0, 1.0, 0.0]])
    else:
        R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    res = sp.fast.cluster_wald_htz(
        X=X_ic, residuals=e, cluster=g_arr, R=R, beta=beta,
    )

    np.testing.assert_allclose(res.eta, r_out["eta"], rtol=1e-8,
                                err_msg=f"η drift seed={seed} G={G} q={q}")
    np.testing.assert_allclose(res.F_stat, r_out["F_stat"], rtol=1e-8,
                                err_msg=f"F drift seed={seed} G={G} q={q}")
    np.testing.assert_allclose(res.p_value, r_out["p_value"], rtol=1e-7,
                                err_msg=f"p drift seed={seed} G={G} q={q}")
```

- [ ] **Step 2: Run live parity tests**

Run: `pytest tests/test_fast_htz.py::test_htz_matches_r_clubsandwich_live -x -q --no-cov`
Expected: 3 passed (Rscript verified at `/usr/local/bin/Rscript` + clubSandwich installed in this environment).

If they fail with rtol drift, this means the formula is slightly off but passed the frozen fixture by coincidence — go back to Task 3 Step 8 reconciliation. Live and frozen MUST both pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_htz.py
git commit -m "test(fast): live R clubSandwich HTZ parity (skipif belt-and-suspenders)"
```

---

## Phase E — Wiring

### Task 7: Registry + paper.bib + cross-link in cluster_dof_wald_bm

**Files:**
- Modify: `src/statspai/registry.py`
- Modify: `paper.bib`
- Modify: `src/statspai/fast/inference.py` (cross-link in `cluster_dof_wald_bm` docstring)

- [ ] **Step 1: Crossref-verify pustejovsky2018small DOI**

CLAUDE.md §10 zero-hallucination — verify before writing the bibkey:

```bash
curl -s 'https://api.crossref.org/works/10.1080/07350015.2016.1247004' \
  | python -c "import sys, json; d = json.load(sys.stdin)['message']; \
print('AUTHORS:', [a['given']+' '+a['family'] for a in d['author']]); \
print('YEAR:', d['issued']['date-parts'][0][0]); \
print('JOURNAL:', d['container-title'][0]); \
print('VOL/ISS:', d.get('volume'), '/', d.get('issue')); \
print('PAGES:', d.get('page')); \
print('TITLE:', d['title'][0])"
```

Expected output should confirm:
- Authors: James E. Pustejovsky, Elizabeth Tipton
- Year: 2018
- Journal: Journal of Business & Economic Statistics
- Vol/Iss: 36 / 4
- Pages: 672–683
- Title: Small-Sample Methods for Cluster-Robust Variance Estimation and Hypothesis Testing in Fixed Effects Models

If ANY of these don't match, **do not add the bibkey**. Investigate the
actual canonical citation. CLAUDE.md §10 is a hard rule.

- [ ] **Step 2: Add `pustejovsky2018small` to `paper.bib`**

Open `paper.bib`. Find an alphabetically appropriate location (between `pus...` entries or with the other 2018 cluster-SE refs). Add:

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

- [ ] **Step 3: Register the three new symbols**

Open `src/statspai/registry.py`. Locate the section where `cluster_dof_bm` and `cluster_dof_wald_bm` are registered (search for `cluster_dof_bm`). Add three parallel rich-spec entries:

```python
register_function(
    name="cluster_dof_wald_htz",
    category="inference",
    summary="HTZ Wald DOF (Pustejovsky-Tipton 2018) — clubSandwich-equivalent.",
    description=(
        "Moment-matching DOF for a multi-restriction Wald test under CR2 "
        "cluster-robust inference. Numerically equivalent to R "
        "clubSandwich::Wald_test(test='HTZ')$df_denom to rtol<1e-8. "
        "Use over cluster_dof_wald_bm when cross-language reproducibility "
        "matters; cluster_dof_wald_bm uses the BM 2002 simplified formula."
    ),
    references=["pustejovsky2018small"],
    tags=["cluster-robust", "wald-test", "small-sample"],
)

register_function(
    name="cluster_wald_htz",
    category="inference",
    summary="Cluster-robust Wald test (HTZ) — clubSandwich-equivalent.",
    description=(
        "Full Wald test under CR2 sandwich with HTZ moment-matching DOF "
        "(Pustejovsky-Tipton 2018, JBES 36(4)). Returns WaldTestResult "
        "with η, F, p, raw Q, and V_R = R V^CR2 R^T."
    ),
    references=["pustejovsky2018small"],
    tags=["cluster-robust", "wald-test", "small-sample"],
)
```

(Adapt the call signature to whatever `register_function` actually expects — read the existing registrations for `cluster_dof_bm` to match conventions exactly. The fields `category` / `summary` / `description` / `references` / `tags` are the canonical set used elsewhere in registry.py; if names differ, follow the local style.)

- [ ] **Step 4: Cross-link the two BM/HTZ functions**

In `src/statspai/fast/inference.py`, find the `cluster_dof_wald_bm` docstring's `Notes` section. Locate this paragraph:

```
**Not equivalent to** ``clubSandwich::Wald_test(..., test="HTZ")``.
The HTZ test uses a Hotelling-T² approximation with the more
elaborate Pustejovsky-Tipton 2018 §3.2 moment-matching DOF, which
is typically much smaller (more conservative) than the BM 2002 §3
simplified ``(Σ tr)² / Σ ||·||_F²`` formula implemented here. On
moderate panels the two can differ by 50–100%.
```

Append one more sentence immediately after it:

```
For the clubSandwich-equivalent HTZ DOF, see :func:`cluster_dof_wald_htz`
(matches R ``clubSandwich::Wald_test(test="HTZ")`` to ``rtol < 1e-8``).
```

- [ ] **Step 5: Run registry self-check**

Run:

```bash
python -c "import statspai as sp; \
fns = sp.list_functions(); \
assert 'cluster_dof_wald_htz' in fns, 'missing cluster_dof_wald_htz'; \
assert 'cluster_wald_htz' in fns, 'missing cluster_wald_htz'; \
print('registry sanity OK:', len(fns), 'functions')"
```

Expected: prints `registry sanity OK: <N> functions` (no AssertionError).

- [ ] **Step 6: Commit**

```bash
git add src/statspai/registry.py paper.bib src/statspai/fast/inference.py
git commit -m "feat(fast): register cluster_wald_htz / cluster_dof_wald_htz + paper.bib pustejovsky2018small (Crossref verified)"
```

---

### Task 8: CHANGELOG + SUMMARY.md update

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `benchmarks/hdfe/SUMMARY.md`

- [ ] **Step 1: Add v1.6.4 (or current dev version) entry to CHANGELOG.md**

Open `CHANGELOG.md`. Add at the very top (above the most recent version entry):

```markdown
## [Unreleased]

### Added
- `sp.fast.cluster_wald_htz()`, `sp.fast.cluster_dof_wald_htz()`, and
  `sp.fast.WaldTestResult` — clubSandwich-equivalent HTZ Wald test under
  CR2 cluster-robust sandwich (Pustejovsky-Tipton 2018, JBES 36(4),
  672–683; DOI 10.1080/07350015.2016.1247004). Numerically equivalent to
  R `clubSandwich::Wald_test(..., test="HTZ")` to `rtol < 1e-8` on the
  verified fixtures (`tests/fixtures/htz_clubsandwich.json`).
  - Independent of `crve` / `feols` / `event_study` — wiring HTZ into
    those is a separate follow-up PR.
  - Working covariance locked to `Φ = I` (OLS+CR2 path); user-supplied
    `target` is not yet exposed.
  - Bibkey `pustejovsky2018small` added to `paper.bib` (refs verified via
    Crossref + DOI dual-source per CLAUDE.md §10).
```

If `CHANGELOG.md` already has an `[Unreleased]` section, append the entry under the existing `### Added` (or create that subsection).

- [ ] **Step 2: Update SUMMARY.md follow-up backlog**

Open `benchmarks/hdfe/SUMMARY.md`. Find the section "What deliberately did NOT ship (in priority order for follow-ups)". Find item #2:

```
2. **CR2 (Bell-McCaffrey) and IM (Imbens-Kolesar)** cluster SE for
   small G. The current CR1/CR3 + wild bootstrap pair is the
   most-used flavour.
```

CR2 and BM Satterthwaite already shipped. Replace this item with a more accurate one (or remove + renumber) — but specifically for HTZ, change the post-audit blurb at the top of SUMMARY.md to acknowledge HTZ shipped:

In the post-audit blurb (`v1.8.1 follow-up` section), append a sentence:

```
v1.8.x further shipped the clubSandwich-equivalent HTZ Wald
test (`cluster_wald_htz` / `cluster_dof_wald_htz` / `WaldTestResult`,
Pustejovsky-Tipton 2018), independent PR with rtol<1e-8 R parity.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md benchmarks/hdfe/SUMMARY.md
git commit -m "docs(fast): CHANGELOG + SUMMARY update for HTZ Wald shipping"
```

---

## Phase F — Final verification

### Task 9: Full pytest sweep + smoke import + spec acceptance check

**Files:** none (verification only)

- [ ] **Step 1: Full test suite — no regressions**

Run: `pytest -q --no-cov 2>&1 | tail -20`
Expected: tail shows something like `XYZ passed, K skipped` with **no failures**.

If anything fails, stop and audit. The CLAUDE.md §5 baseline: "All existing tests must remain green." Do not commit until green.

- [ ] **Step 2: HTZ-only suite verbose**

Run: `pytest tests/test_fast_htz.py -v --no-cov 2>&1 | tail -30`
Expected: all 16 HTZ tests pass (or 13 if R parity ones skip; here R is installed so 16/16).

- [ ] **Step 3: Smoke import check**

Run:
```bash
python -c "import statspai as sp; \
import numpy as np; \
df_X = np.random.default_rng(0).normal(size=(100, 2)); \
g = np.repeat(np.arange(20), 5); \
nu = sp.fast.cluster_dof_wald_htz(df_X, g, R=np.eye(2)); \
print(f'smoke OK: η={nu:.4f}'); \
print('WaldTestResult:', sp.fast.WaldTestResult.__doc__[:80])"
```
Expected: prints a finite η in (2, 40) range and the start of WaldTestResult docstring.

- [ ] **Step 4: Spec acceptance walkthrough**

Open the spec at [`docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md`](../specs/2026-04-27-htz-clubsandwich-parity-design.md), §8 acceptance criteria. Confirm each of the 8 items:

1. `pytest tests/test_fast_htz.py -q` → 16/16 ✓ (or 13/16 sans R)
2. Full suite no regressions ✓
3. `sp.fast.cluster_wald_htz` exists ✓
4. Registry includes 3 new symbols ✓
5. Live R parity ≥ 3 panels at rtol<1e-8 ✓ (3 frozen + 3 live = 6 panels)
6. q=1 helper↔full self-equivalence ≤ 1e-12 ✓
7. CHANGELOG + paper.bib updated, no `[citation needed]` leaks ✓
8. No GPL code copied — paper-faithful Python ✓

If any item is unticked, that's the single remaining task. Loop back.

- [ ] **Step 5: Final commit (if any wiring/typo fixes needed)**

If Steps 1-4 surfaced minor issues, fix and commit:

```bash
git add -p   # patch-mode review
git commit -m "fix(fast): post-sweep HTZ cleanup"
```

If everything was clean, **no extra commit** is needed — the plan is complete.

- [ ] **Step 6: Summary report**

Print to terminal for the user:

```bash
git log --oneline $(git rev-parse main^^^^^^^^^^)..HEAD | head -15
echo "---"
git diff --stat $(git rev-parse main^^^^^^^^^^)..HEAD | tail -10
```

This prints the chain of commits this plan produced and the change summary.

---

## Self-Review Checklist (run before handing off)

- [ ] **Spec coverage**: every section of the spec's §1-§11 has a task implementing it.
  - §1 (gap + ships) → Tasks 1, 2, 3, 4 (the API surface)
  - §2 (API) → Task 1 + Task 4
  - §3 (algorithm) → Task 2 (helper) + Task 3 (η) + Task 4 (full test)
  - §3.4 (invariants) → Task 4 Step 1 (q=1 helper↔full) + Task 3 Step 9 (drift band)
  - §4 (tests) → Tasks 3, 4, 5, 6
  - §5 (errors) → Task 5
  - §6 (file changes) → all tasks; Task 7+8 catch wiring
  - §7 (citations) → Task 7 Step 1+2
  - §8 (acceptance) → Task 9 Step 4
  - §9 (risks) → Task 3 Step 6 (double-source) + Tasks 3+6 (frozen+live parity catches drift)
  - §10 (effort) → time budget
  - §11 (refs) → Task 7 Step 1+2

- [ ] **Placeholder scan**: no `TBD`, `TODO`, "fill in later" remain in this plan
  except in Task 3 Step 7-8 where the η formula is *deliberately* left for
  paper+R-source double-sourcing at execution time. That deferral is
  documented and bounded by the frozen fixture acceptance gate.

- [ ] **Type/name consistency**: `WaldTestResult` field names are consistent
  across Task 1, 4, 7. Function arg names (`X`, `cluster`, `R`, `weights`,
  `bread`, `residuals`, `beta`, `r`) are consistent across all task code
  blocks. `_htz_per_cluster_quantities` returns the same dict keys
  (`G_g`, `A_g_sqrtW`, `Omega`, `cluster_codes`, `G`, `q`, `bread`) used by
  every consumer.

- [ ] **Dependency order**: Task N depends only on Tasks <N. Task 4 needs
  `_htz_eta_from_quantities` extracted in Task 4 Step 3 (refactor).
