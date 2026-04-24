# RFC — `sp.did_multiplegt_dyn`: de Chaisemartin-D'Haultfœuille (2024) intertemporal event-study DiD

> **Status**: draft 2026-04-23. No code written yet. All paper-specific formulas below carry `[待核验]` until two-source verification per CLAUDE.md §10. Verified anchors: `paper.bib` entry `dechaisemartin2024difference` (DOI 10.1162/rest_a_01414, *ReStat* 2024). ArXiv identifier for working-paper version left unstated until confirmed.

## 1. 动机

当前 `sp.did_multiplegt` 实现的是 **dCDH 2020 (AER) 的 DID_M 估计器**：consecutive-period switcher-vs-stayer 配对 DID 的样本加权平均。它做得已经不错——2024 joint placebo + avg cumulative overlay 已经落地。

但 Stata/R 社区里"did_multiplegt"这个名字的 muscle memory，今天指向的是 **dCDH 2024 ReStat**（arXiv:2007.04267）里的 `_dyn` 包 —— 一个 **intertemporal event-study** 估计器。它：

- 不是 pair 级别的 rollup，而是 first-switch 时段 `F` 的 long-difference 事件研究：对每个 horizon `l`，比较 `Y_{F+l} − Y_{F−1}` 在 switchers vs "not-yet-treated-at-horizon-l" 之间。
- 有**独立的影响函数方差**（非 bootstrap-only）。
- 处理 treatment reversal 的方式是 control group 在 horizon `l` 上仍保持未处理（稳定限制），不是简单的 sign-flip。

当前 `sp.did_multiplegt` 的 `dynamic=H` 参数虽然返回了 horizon-l 效应，但其内部路径是 pair rollup 的延展——**不等价于** dCDH 2024 的 `_dyn` 估计量。这是 `gap_audit.md` 标记的另一个头等抓手。

## 2. 目标估计量 `[待核验]`

> dCDH 2024 *ReStat* §2—§3（DOI `10.1162/rest_a_01414`，已在 `paper.bib` 核验），以及 R `DIDmultiplegtDYN` 包 reference manual。arXiv working-paper 版本号 [待核验]。所有公式核验前不得落到 docstring。

对每个 horizon `l ≥ 0`：

- **Switchers at F**: units first switching treatment at time `F`.
- **Not-yet-treated at F+l**: units whose treatment stays at its pre-F value from `F−1` through `F+l` inclusive.
  [待核验 — dCDH 2024 Definition 2.x]

**Dynamic effect at horizon `l`**:
[待核验 — dCDH 2024 eq. (2.x)]
`δ_l = E[ Y_{F+l} − Y_{F−1} | switch ] − E[ Y_{F+l} − Y_{F−1} | not-yet-treated-at-F+l ]`,
averaged over `F` with appropriate weights [待核验 — 2024 eq. (2.y)].

**Placebo effect at lag `l`**:
[待核验 — dCDH 2024 eq. (3.x)]
same structure but comparing `Y_{F−1−l} − Y_{F−1−l−1}` pre-treatment.

**Sign convention for off-switchers**: [待核验]. The current `sp.did_multiplegt` uses a sign flip; `_dyn` handles directionality via the stable-control restriction instead.

### 2.1 识别假设 `[待核验]`

1. **Parallel trends** between switchers at `F` and the not-yet-treated control at horizon `l`, for each `l`.
2. **No anticipation** prior to `F`.
3. **Stable controls**: unit remains at pre-`F` treatment value from `F−1` through `F+l`.
4. **SUTVA**.

## 3. 推断 `[待核验]`

dCDH 2024 provides:

- **Analytical influence function per horizon**: `IF_l(i)`, variance = `Var(IF_l) / N`, clustered at unit level.
- **Joint test** of all placebo lags: `W_P = δ̂_P' V_P⁻¹ δ̂_P ~ χ²(L_P)`.
- **Joint test** of all dynamic horizons + placebo: `W = δ̂' V⁻¹ δ̂ ~ χ²(L_P + L_D + 1)`.
- **Average cumulative effect**: `δ̄ = (1/(L+1)) Σ_l δ_l`, SE from the IF covariance across horizons.

Heteroskedastic-weights variant (dCDH 2023 survey, *The Econometrics Journal* 26(3):C1-C30, DOI 10.1093/ectj/utac017, bib key `dechaisemartin2022fixed` — venue verified 2026-04-24 via Crossref): alternative cell weighting less sensitive to group-size heterogeneity [待核验].

## 4. 拟议 API

```python
sp.did_multiplegt_dyn(
    data,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: list[str] | None = None,
    placebo: int = 0,                  # number of placebo lags
    dynamic: int = 3,                  # number of dynamic horizons
    weights: str = "plain",            # 'plain' | 'heteroskedastic' (dCDH 2022)
    cluster: str | None = None,
    inference: str = "analytical",     # 'analytical' (IF) | 'bootstrap' | 'both'
    n_boot: int = 999,
    alpha: float = 0.05,
    seed: int | None = None,
) -> CausalResult
```

### 4.1 返回 `CausalResult`

- `estimate`: `δ̄` = average cumulative effect over horizons `0..L_D`.
- `detail`:
  `pd.DataFrame` with columns `relative_time, type ('placebo'|'dynamic'), estimate, se, ci_lower, ci_upper, pvalue, n_switchers, n_controls`.
- `model_info`:
  - `method = "dCDH (2024) intertemporal event-study"`
  - `event_study`: the detail DataFrame, for plotting (matches `sp.did_multiplegt` shape so `sp.did_plot` works).
  - `joint_placebo_test`: `{statistic, df, pvalue}` — parallel-trends diagnostic.
  - `joint_overall_test`: `{statistic, df, pvalue}` — placebo + dynamic joint Wald.
  - `avg_cumulative_effect`: `{estimate, se, ci_lower, ci_upper, pvalue, n_horizons}` — per-IF covariance, NOT bootstrap.
  - `influence_functions`: optional `np.ndarray` of shape `(n_units, L_P + L_D + 1)` when `return_if=True`.
  - `weights`: echoes the chosen weighting scheme.

### 4.2 与现有 `sp.did_multiplegt` 的关系

- `sp.did_multiplegt` 保持不变：继续是 dCDH 2020 DID_M + 2024 joint placebo + avg cumulative overlay。
- `sp.did_multiplegt_dyn` 新增为独立函数。docstring 明确 trade-off：
  - "pair-rollup" vs "long-difference event-study"
  - 推荐场景各自为何
- `docs/guides/choosing_did_estimator.md` 新增决策分支：on-off switching + dynamic focus → `did_multiplegt_dyn`.

## 5. 测试计划

### 5.1 Unit / analytic tests

- Constant dynamic effect DGP `δ_l = c`: recover `c` at each horizon ± MC noise.
- Linearly growing dynamic effect `δ_l = c + a·l`: recover slope `a` via `avg_cumulative_effect` + per-horizon comparison.
- Zero-effect under parallel trends: both joint tests reject at nominal rate (size test).
- PT-violation DGP: joint placebo test rejects.
- Treatment-reversal DGP: estimator excludes reverted units from not-yet-treated control at appropriate horizons.

### 5.2 Reference parity

- Against **R `DIDmultiplegtDYN`** [待核验 — confirm current CRAN version + citation before running]. Fixtures: the canonical example from the dCDH 2024 replication bundle.
- Tolerance `atol=1e-4` on per-horizon `δ_l`; `atol=1e-4` on joint Wald statistic; `atol=1e-3` on analytical SE (to accommodate minor linear-algebra differences).

### 5.3 Edge cases

- All units switch at the same time: reduces to standard event-study.
- No units ever switch: raises `DataInsufficient`.
- Panel with holes: document current handling (drop or carry-forward? `[待核验]` against R package behaviour).
- Mismatched `placebo`/`dynamic` lengths vs. panel depth: should raise clear error.

### 5.4 Coverage

- Target ≥ 95% on `did_multiplegt_dyn.py`.
- Parity tests split between `tests/reference_parity/` (slow, network-free, uses saved fixtures) and `tests/test_did_multiplegt_dyn.py` (fast structural tests).

## 6. 实现建议

Create `src/statspai/did/did_multiplegt_dyn.py`. Possible shared primitives:

- **Influence-function plumbing**: BJS in `bjs_inference.py` already has IF-style analytical variance for imputation — consider a light factoring into `did/_if_utils.py` if the shapes match. Verify before refactoring.
- **Event-study DataFrame shape**: match `sp.did_multiplegt`'s `model_info['event_study']` exactly so `sp.did_plot` works without changes.
- **Joint Wald**: reuse the pattern in `did_multiplegt.py:_joint_placebo_test` — lift to `did/_if_utils.py`.

## 7. 风险 / 未决问题

1. **Paper version lock**: dCDH 2024 has multiple arXiv revisions; ReStat accepted version is the anchor, but the R package may track a more recent revision. Lock both.
2. **Naming**: `sp.did_multiplegt_dyn` matches Stata/R muscle memory. Alternative `sp.did_multiplegt(..., dyn=True)` is lighter but hides the estimator switch. RFC recommends the explicit function.
3. **Heteroskedastic weights variant**: add in the same PR or as a follow-up? Recommend follow-up — keep the first landing minimal and parity-verified.
4. **Interaction with `sp.honest_did`**: Rambachan-Roth sensitivity applies naturally to the dCDH_dyn event-study output. Expose via the same hook `sp.did_multiplegt` currently uses? Or a dedicated adapter?
5. **Influence-function export**: optional `return_if=True` surfaces the IF matrix for downstream use (sensitivity, stacking). Worth having from day one.

## 8. 建议的落地顺序

1. **Merge this RFC** (doc-only).
2. User approves paper version + R package version lock.
3. Build `tests/reference_parity/` fixtures from the R package's canonical example.
4. Implement the core long-difference estimator + IF variance + unit tests. All `[待核验]` markers resolved to verified citations before landing.
5. Add joint Wald tests (placebo + overall) + avg cumulative.
6. Add `weights='heteroskedastic'` variant (optional, follow-up).
7. Update `docs/guides/choosing_did_estimator.md` + add `docs/guides/multiplegt_dyn.md`.
8. Update `CHANGELOG.md` under `### Added`. No correctness-fix tag needed — this is a strictly additive new function.
9. Update `sp.did_plot` if the event-study DataFrame shape deviates (it shouldn't).

## 9. 反问自检

Before spending a sprint on this, confirm:

- **Is the parity target still `DIDmultiplegtDYN` (R) or has Stata `did_multiplegt_dyn` diverged?** Check both at implementation start.
- **Does the user base actually want `_dyn` semantics, or are they happy with the existing `sp.did_multiplegt(placebo=, dynamic=)` behaviour?** `gap_audit.md` §3 argues for `_dyn` based on external-package muscle memory; worth sanity-checking before the sprint.
