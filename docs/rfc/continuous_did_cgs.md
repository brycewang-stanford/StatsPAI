# RFC — `sp.continuous_did(method='cgs')`: Callaway-Goodman-Bacon-Sant'Anna (2024) continuous DiD

> **Status**: draft 2026-04-23. No code written yet. All paper-specific formulas below carry `[待核验]` until two-source verification (Crossref DOI + paper PDF / NBER working-paper text) per CLAUDE.md §10.

## 1. 动机

`sp.continuous_did` 目前有三种 `method`：`twfe`（dose × post 交互的 TWFE OLS）、`att_gt`（dose 分位数分箱后逐箱 2×2 DID）、`dose_response`（ΔY 对 dose 的局部多项式）。这三种都是启发式近似，**不是** Callaway, Goodman-Bacon & Sant'Anna (2024, NBER WP 32117) 真正识别的 ATT(d|g,t) / ACRT(d|g,t)。

用户在使用 `sp.continuous_did` 时会默认引用 CGS (2024)——docstring 也这么写——但当前返回的数字与 CGS 的估计量在定义、假设强度、方差、和聚合权重上都不对齐。这是一个**预期 vs. 实现不匹配**的前沿缺口，也是今晚 `gap_audit.md` 识别的头等抓手。

## 2. 目标估计量 `[待核验]`

> CGS 2024 §3—§4，引用核验待接入 PDF。

设 `D_i` 为单元 `i` 的连续处理剂量，`G_i` 为其首次接受处理的时段（0 表示 never-treated），`T` 为时间。在 **strong parallel trends** 假设下：

- **Level ATT(d | g, t)**:
  [待核验 — CGS 2024 Definition 3.1]
  `ATT(d | g, t) = E[Y_t(d) − Y_t(0) | G=g, D=d]`
  on the support of `d` within cohort `g`.

- **Slope ACRT(d | g, t)**:
  [待核验 — CGS 2024 Definition 3.2]
  `ACRT(d | g, t) = ∂ ATT(d | g, t) / ∂ d`.

- **聚合**:
  [待核验 — CGS 2024 §4]
  - `ATT(d)` = weighted average over `(g, t)` with post-treatment `t ≥ g`, weights determined by the distribution of `(G, T)` among treated units at dose `d`.
  - `ACRT(d)` similarly aggregated.

### 2.1 识别假设

[待核验 — CGS 2024 Assumption 3.x]

1. **Strong parallel trends (SPT)**: for every dose `d` on the support,
   `E[Y_t(d) − Y_{t-1}(d) | G=g] = E[Y_t(d) − Y_{t-1}(d) | G=g']` for all `g, g'`.
   Stronger than standard PT (which only requires the above for `d=0`).
2. **No anticipation** of future dose.
3. **Overlap** in `(G, D)` — positive density of dose at each cohort.
4. **SUTVA**.

## 3. 估计器草案 `[待核验]`

> 以下所有"公式"都先写占位，落码前必须逐条对照 CGS 2024 paper。

### 3.1 Level ATT(d | g, t)

三选一（CGS 2024 提供 OR / IPW / DR 三种，推荐 DR）：

- **Outcome-regression (OR)**:
  [待核验 — CGS 2024 eq. (4.x)]
  fit `m_g(d, X) = E[Y_t − Y_{g-1} | G=g, D=d, X]` via flexible nonparametric (spline / local linear in `d`); `ATT_OR(d | g, t) = m_g(d, X) − m_0(d, X)` averaged over covariates.

- **IPW**:
  [待核验]
  weight by the dose propensity `π_g(d | X)` within cohort `g`.

- **Doubly-robust (DR)**:
  [待核验 — CGS 2024 §4.x]
  combines OR + IPW; consistent if either model is correct.

### 3.2 Slope ACRT(d | g, t)

[待核验 — CGS 2024 eq. (4.x)]
local-linear (or kernel) derivative of `ATT(d | g, t)` w.r.t. `d`.

### 3.3 聚合

- `ATT(d)` across `(g, t)`: [待核验 — CGS 2024 eq. (4.y)]
- `ACRT(d)` across `(g, t)`: [待核验]
- Overall summary scalar (e.g. mean across support): `E[ATT(d) | D=d, treated]`.

## 4. 推断 `[待核验]`

- **Analytical influence function**: CGS 2024 §5 provides IFs for each estimator; asymptotic variance = covariance of IF, clustered at unit level.
- **Multiplier bootstrap** over IFs for uniform confidence bands on `d ↦ ATT(d)` and `d ↦ ACRT(d)` [待核验 — §5.x].
- Simultaneous CIs for the dose-response curve, not just pointwise.

## 5. 拟议 API

```python
sp.continuous_did(
    data,
    y: str,
    dose: str,
    time: str,
    id: str,
    g: str | None = None,          # cohort column (first-treat period)
    method: str = "cgs",           # new default; old modes kept under deprecation
    estimator: str = "dr",         # 'dr' | 'or' | 'ipw'
    controls: list[str] | None = None,
    support: tuple[float, float] | None = None,  # (d_min, d_max) for curve
    n_grid: int = 50,              # dose-response grid resolution
    cluster: str | None = None,    # defaults to id
    n_boot: int = 999,             # multiplier bootstrap
    alpha: float = 0.05,
    bandwidth: float | str = "imse",  # for local-linear slope
    seed: int | None = None,
) -> CausalResult
```

### 5.1 返回 `CausalResult`

- `estimate`: overall `E[ACRT(d)]` or `E[ATT(d)]` depending on `estimand=` flag (default slope).
- `detail`: `pd.DataFrame` with columns `dose, att_d, acrt_d, se, ci_lower, ci_upper, n_treated_at_d`.
- `model_info`:
  - `method = 'CGS (2024) continuous DiD'`
  - `estimator = 'dr'|'or'|'ipw'`
  - `strong_pt_diagnostic`: pre-period test of SPT across cohorts.
  - `overlap_diagnostic`: density of dose per cohort; min/max.
  - `uniform_ci`: multiplier-bootstrap bands.

### 5.2 弃用路径

- `method='att_gt'` → keep working, emit `DeprecationWarning("method='att_gt' is the dose-quantile heuristic, not CGS (2024) ATT(d|g,t). Pass method='cgs' for the paper estimator; method='dose_bin' is the new non-deprecated name for the heuristic.")`.
- `method='twfe'` and `'dose_response'` remain non-deprecated as explicit heuristic modes.
- `MIGRATION.md` entry required.

## 6. 测试计划

### 6.1 Unit / analytic tests

- Constant `ATT(d) = c` DGP: estimator should recover `c` ± Monte-Carlo noise.
- Linear dose-response `ATT(d) = c·d`: `ACRT(d)` should be constant `= c`.
- Zero effect under SPT: CI should cover zero at nominal rate.
- SPT violation DGP: pre-period diagnostic should flag.

### 6.2 Reference parity

- Against **R `contdid` package** [待核验 — confirm package name and maintainer before running]. Fixture data: the canonical example from CGS 2024 replication (if the authors published replication files with the NBER WP).
- Tolerance `atol=1e-4` on `ATT(d)` at grid points; `atol=1e-3` on `ACRT(d)` (slope estimators are noisier).

### 6.3 Edge cases

- Single cohort (pure 2×2 continuous): should reduce to a canonical formula.
- No `D=0` units: should raise `DataInsufficient`.
- Dose measured with error: not handled (out of scope; cite `sp.did_misclassified` as the place for measurement-error DID).

### 6.4 Coverage

- Target ≥ 95% on `continuous_did._cgs_*` helpers.
- Integration test with `sp.did_analysis` / `sp.cs_report` workflow.

## 7. 实现建议

Create `src/statspai/did/continuous_did_cgs.py` (new) and let `continuous_did.py` dispatch `method='cgs'` to it. Keep `_continuous_did_att_gt`, `_continuous_did_twfe`, `_continuous_did_dose_response` unchanged under the `dose_bin`/`twfe`/`dose_response` names.

Share primitives with existing DiD code:

- **Propensity / outcome regression**: reuse what `wooldridge_did.py` DR-DiD and `callaway_santanna.py` DR-DiD already use. Do NOT re-implement.
- **Multiplier bootstrap**: whether an existing helper covers this needs verification — otherwise add to `did/_inference.py` as a new shared primitive (analogous to `rd/_core.py`, `decomposition/_common.py`).
- **Local-linear / kernel**: reuse `sp.nonparametric.lpoly` for slope estimation.

## 8. 风险 / 未决问题

1. **Paper version lock**: NBER WP 32117 has had revisions. Lock to the most recent arXiv / NBER date at implementation start; freeze that as the `paper.bib` citation target.
2. **Reference package availability**: the R `contdid` package may not exist at time of implementation — fallback is to implement from paper + authors' Stata replication files.
3. **SPT diagnostic**: CGS 2024 likely provides a specific pre-trend test for SPT [待核验]; if not, we'll need to invent a sensible analogue.
4. **Bandwidth selection** for the local-linear slope estimator — IMSE-optimal vs. cross-validation. Default to the same rule `sp.nonparametric.lpoly` uses.
5. **User-facing name collision**: current `sp.continuous_did(method='att_gt')` users will see the deprecation notice. Is the renamed-to-`dose_bin` the right new label? Alternatives: `quantile_did`, `binned_dose`, keep `att_gt` and flip default to `cgs`.

## 9. 建议的落地顺序

1. **Merge this RFC** (doc-only, no code risk).
2. User approves paper version lock.
3. Implement reference-parity fixtures (synthetic + CGS replication if available) → commit to `tests/reference_parity/`.
4. Implement `_continuous_did_cgs()` helper with OR estimator + unit tests; all `[待核验]` markers resolved to verified citations.
5. Add IPW + DR estimators; unit tests.
6. Multiplier bootstrap + uniform CI.
7. Deprecation wiring for old `method=` values; MIGRATION.md entry.
8. Update `docs/guides/choosing_did_estimator.md` + add `docs/guides/continuous_did.md`.
9. Update `CHANGELOG.md` under `### ⚠️ Correctness` if any user-visible behaviour changes beyond new method addition.
