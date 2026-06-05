# StatsPAI Comprehensive Improvement Roadmap — 2026-06-04

> Branch: `improve/comprehensive-pass`. Produced by a focused multi-lane audit
> (correctness, performance, API consistency, docs). **Constraints honored:**
> `paper.md` / `paper.bib` left untouched to protect the JOSS #10604 review;
> the P1-WIP modules in the sibling agent's stash (`paper`, `llm_dag`,
> `causal_question`, `causal_text`, `question`) were not modified; all work is
> on a feature branch for review before merge.

This file tracks what was **done this session** and the **ranked backlog** of
verified-but-deferred improvements (each with reproduction evidence so the next
pass can pick it up cold).

---

## A. Completed this session (committed on branch)

| Commit | Lane | Summary | Verification |
|---|---|---|---|
| `ce8124f` | ⚠️ correctness | `sp.stabilized_weights`/`sp.msm` single-period IPTW silently collapsed to 1.0 (confounded; §7 violation). Drop zero-variance columns before the logit fit; warn on genuine failure. | matches textbook IPTW to `1.8e-15`; 3 new + 5 existing tests |
| `15d21d7` | performance | `conley` spatial-HAC sandwich vectorized (`Xe.T@Xe`; `M+M.T`) | ~140× on dense pairs; pinned to explicit-loop reference (2 tests) |
| `623eb20` | performance | `did._core.cluster_bootstrap_draw` pre-grouped fancy-index build | ~40× (6.6s→0.16s/200 draws); byte-identical to old loop (3 tests) |
| `142aceb` | agent-UX | 29 statically-broken registered examples repaired + permanent bind-guard test | 373 examples bind green; runtime-confirmed for the rebuilt ones |
| `118b551` (on `main`) | ⚠️ correctness | **B.1 done** — `structural_break` sup-F p-value → Andrews (1993) null (was naive `F(k,n-2k)`). | white-noise false-positive 33–37% → ~5%; power 1.00/0.88; 7 tests |
| `fd932c5` (branch `worktree-improve-correctness`) | ⚠️ correctness | **B.2 done** — `lpoly` SE now includes the `XᵀW²X` sandwich meat + exact weighted dof. | SE/MC-truth 0.75–0.96 → 0.99–1.03; 95% CI coverage restored; 4 tests |
| `ed72a8c` (branch) | ⚠️ correctness | **B.3 done** — `lee_bounds`/`manski_bounds` now use the Imbens-Manski (2004) `C_n` (was two-sided z on both ends → set CI, over-covered). | binding-endpoint coverage ~0.98 → ~0.95; `C_n∈[1.645,1.96]`; 4 tests |
| `be9f1aa` (branch) | ⚠️ correctness | **B.4 done** — `cusum_test` now uses the Brown-Durbin-Evans diverging linear boundary (was flat `1.358`). | H0 rejection ~32% → ~4-5%; power 1.00; 5 tests |

> **Section B (correctness) is now fully cleared (B.1–B.4).** All four were
> measured-as-broken, fixed, and validated (size/coverage + power).
>
> **Section C.1 DONE — `wild_cluster_bootstrap` ~25× faster, output unchanged.**
> The profile settled the earlier worry: the RNG draw is *negligible* (0.6 % of
> runtime); the cost is the two per-cluster Python loops. So the per-draw weight
> draw is kept byte-identical (no re-pin needed) and only the deterministic
> inner loops are vectorized (`Y*` gather + `S = Ind'·(X∘e)`, meat `= S'S`).
> Verified against the old code: `p_boot` exactly equal, t-dist to ~1e-15, on
> rademacher/webb/mammen.
>
> **Section C.2 DONE — `romano_wolf` ~3.8× (HC1) / ~19× (cluster).** Same
> pattern: pre-extract numpy (drop the per-draw `df.iloc` copy), share the QR /
> `(XᵀX)⁻¹` across outcomes, vectorize the first-coef SE (HC1 and Liang-Zeger),
> skip the unused p-value; per-draw resample unchanged. Verified the full
> `p_rw` table matches the old code **exactly** (diff 0.0) on HC1 + cluster.
>
> **Section C.3 DONE — `margins._compute_dydx` ~1000×.** The per-row
> `data.iloc[i]` central-difference loop is replaced by a whole-frame predictor
> evaluation (loop only over model terms). Bit-identical (`np.array_equal` on
> the full `margins()` table). Section C performance backlog is now cleared
> (C.1 wild bootstrap, C.2 romano_wolf, C.3 margins).

> **Working model (updated 2026-06-04):** the shared single working tree caused
> branch-switch churn (the `118b551` fix landed on `main` directly when the tree
> was switched under us). To stop that, this lane now runs in an **isolated git
> worktree** at `.claude/worktrees/improve-correctness` on branch
> `worktree-improve-correctness` (based on `origin/main`). Test edits and `paper`
> P1-WIP remain the sibling agent's lane on `main`. Merge this branch back after
> review. Rhythm: one verified correctness/perf fix per commit, each with a
> reproduction + post-fix validation before it lands.

---

## B. Deferred — correctness (verified, need careful fixes)

Ranked by whether a user would publish a wrong number.

1. ~~**[HIGH] `timeseries/structural_break.py` — sup-F / Bai–Perron naive F
   p-value.**~~ **DONE** in `118b551` (Andrews 1993 null via seeded simulation;
   white-noise false-positive 33–37% → ~5%, power retained; Bai-Perron now
   exposes `f_stats`/`p_values`). Kept for provenance.

2. ~~**[MED] `nonparametric/lpoly.py` — local-polynomial SE omits the kernel
   sandwich meat.**~~ **DONE** in `fd932c5`. Reported `σ²·(XᵀWX)⁻¹` (treats
   kernel weights as inverse-variance) instead of
   `(XᵀWX)⁻¹(XᵀW²X)(XᵀWX)⁻¹·σ²` (Fan & Gijbels
   1996). *Reproduced:* reported SE = **0.66×** Monte-Carlo truth ⇒ CIs ~34%
   too narrow (n=4000). *Fix:* swap in the sandwich; ⚠️ correctness note.

3. **[LOW/MED] `bounds/lee_manski.py:148` — "Imbens–Manski" CI is actually
   Horowitz–Manski (mislabeled, over-covers).** Applies the two-sided `z` to
   both endpoints rather than the one-sided `C_n` of Imbens & Manski (2004).
   Point bounds are correct. *Fix:* implement the `C_n` root-find or relabel.

4. **[MINOR] `timeseries/*cusum*` — CUSUM test uses a constant Brown–Durbin–
   Evans boundary `1.358` instead of the linear `a·[1+2(t−k)/(T−k)]`.** Not
   separately reproduced; verify before fixing.

> Modules **inspected and found correct** (numerically cross-checked, no bug):
> decomposition (Kitagawa/Das Gupta/source/GE indices/Shapley), mediation
> (VanderWeele four-way), multilevel (`mixed` vs statsmodels MixedLM to 5dp on
> betas *and* SEs), frontier (SFA likelihoods, Jondrow, Battese–Coelli,
> chi-bar LR), gmm (two-step/CUE variance), survival (Cox-Efron, KM/Greenwood,
> Weibull AFT, log-rank), qte (IRLS quantile vs statsmodels), spatial (Moran's
> I, SAR ML), dose_response (Hirano–Imbens GPS). One *labeling* note: the
> `qte(method='distribution')` path is a coherent QTT but is labeled "Firpo
> 2007" (which targets the unconditional QTE).

---

## C. Deferred — performance (measured wins, numerically identical)

All prototyped and verified to produce identical estimates/SEs; each needs the
same "pin to reference / equal-output test" treatment used for `conley`.

| Target | n / size | Baseline | Speedup | Notes |
|---|---|---|---|---|
| `inference/wild_bootstrap.py:186-208` wild cluster bootstrap inner loop | n=2000, G=50, 999 reps | 0.53s | **3.8×** | precompute per-cluster row blocks; cache `XtX_inv@Xᵀ`; reuse restricted fit. Flagship few-cluster inference. |
| `mht/romano_wolf.py:531-537` bootstrap | n=3000, S=10, 1000 reps | 2.17s | **3.4×** | pre-extract X/Y as numpy; share QR/`(XᵀX)⁻¹` across outcomes; drop per-draw `iloc+reset_index` copy. |
| `postestimation/margins.py:178-189` `_compute_dydx` | n=8000, 1 var | 0.12s | **~10–50×** (est.) | vectorize prediction over columns; closed-form dy/dx for linear models. Must reproduce eps-difference values exactly. |
| `matching/match.py:745` `replace=False` path | n=10000 | 0.8s | (large-n only) | `for u in used: d[u]=inf` is O(n_t²); `np.isin`/mask rewrite. Below bar unless large-n without-replacement matching becomes common. |

> **Already well-optimized (no action):** `panel/feols.py`+`panel/hdfe.py`
> (HDFE backend, numba kernels, sparse LSQR), `did/callaway_santanna.py`
> (analytic influence-function SEs on a wide numpy panel), `matching/match.py`
> `cdist` distance path.

---

## D. Deferred — API consistency & agent-native surface

> **D.1 ENGINE DONE** (`ExportMixin` in `core/results.py`, rolled out to
> `ARIMA/GARCH/VAR/BVAR/LocalProjections/Bootstrap` result classes; 11 tests).
> The mixin is *faithful by construction* — `tidy()` → coef-table-from-attrs →
> single-estimate row → scalar card; **never** flattens a coefficient matrix
> into a misleading table (verified on `BVARResult`'s 2-D `coef`); `cite()`
> never fabricates (§10). Remaining rollout is one line per class
> (`class XResult(ExportMixin, ...)`) **after** checking that class's
> `_export_frame()` is faithful (the per-class check is the whole point — a
> blanket attach to all 270 classes is *not* safe).
>
> **D.1 CAMPAIGN DONE — 54 of 277 result-like classes now exportable** (up from
> ~21). Rolled out + verified: 6 timeseries/inference standalones, the
> `EconometricResults` family (cite for Panel/Frontier/Production/Cox + all
> regression estimators), GenMatch/CardinalityMatch/ClusterCATE, and a 33-class
> high-value batch (coef-table: JIVE/FEOLS/SpatialPanel/PeerEffects; epi 2×2
> measures; DML/IV; MR frontier; +misc). The coef-table builder also learned
> the `t_stats`/`p_values`/`ci_*` name variants. Guarded by
> `tests/test_export_rollout.py` (per-class faithfulness + a coverage ratchet
> ≥50). **Phase 3 (the remaining ~181 scalar-card dataclasses + 42 non-dataclass)
> is DEFERRED**: they all fail synthetic construction (complex required fields /
> `__post_init__`), so their export cannot be verified faithful without
> instantiating each via its real estimator API — and they yield only thin
> 2-field cards. Not a safe/worthwhile blanket rollout; revisit per-class only
> when a specific thin-card result is actually wanted in a paper.

1. **[HIGH leverage] Generic `ExportMixin` for result objects.** 244/257
   result classes have `.summary()` but only ~11 have the full export quartet;
   `to_markdown` (8%), `cite` (11%), `to_word` (12%), `to_excel` (14%),
   `to_latex` (15%) are the most-missing. **82% (211/257) are `@dataclass`** and
   **88% expose ≥1 structured accessor** (`dataclass` / `tidy` / `to_dict`), so
   a single fallback mixin can light up the quartet without 116 hand-written
   implementations. Design sketch (in `core/results.py` next to the existing
   `SummaryText` mixin + `_to_jsonable`):
   - `_export_frame()` reads, in precedence order, `tidy()` → `to_dict()` →
     `dataclasses.fields()` → `vars()`; collapses the 4 observed result shapes
     (single-estimate row / coef-table / group-time panel / scalar+diagnostics)
     onto one `pd.DataFrame`.
   - `to_markdown/to_latex/to_excel/to_word` route that frame through the
     existing `output/` renderers (`regression_table.py`, `_excel_style.py`,
     `_format.py`).
   - **Subclass methods win** automatically when the mixin is the rightmost
     base (guard with `type(self).X is not ExportMixin.X`), so the 39 classes
     with bespoke `to_latex` keep theirs.
   - **`cite()` must NOT fabricate** (CLAUDE.md §10): only look up a class-level
     verified bib key against `paper.bib` via `output/_bibliography.py`; return
     empty/raise when none is attached.
   - Rollout: attach to the 211 dataclasses first; add `to_dict` to the 32
     accessor-less classes; wire `cite()` only where a verified key exists.

2. **[MED] 9 `*args/**kwargs` dispatchers expose empty `function_schema`
   params** (no agent guidance via introspection): `multi_cutoff_rd`,
   `geographic_rd`, `boundary_rd`, `multi_score_rd`, `anderson_rubin_ci`,
   `conditional_lr_ci`, `prevalence_ratio`, `diagnostic_test`, `etable`.
   Hand-author `params` in the registry for these.

> Schema health otherwise good: 0 schemas threw across 977 functions; 0
> all-untyped; the 21 schema "param drift" cases were the same root cause as
> the example breaks fixed in `142aceb` and are now resolved.

---

## E. Deferred — documentation (Examples gaps)

Add **doctest-backed** `Examples` sections (and *derive the registry example
from the doctest* so drift can't recur) to high-traffic functions that lack a
runnable `>>>` example. Tier 1 (headline `method=` dispatchers — most visible):
`rd` (`rd/__init__.py`), `synth` (`synth/scm.py:49`), `dml`
(`_article_aliases.py:608`), `ivreg`, `qte`, `sun_abraham`, `cic`,
`gardner_did`, `rdd`, `jive`, `synthdid_estimate`, `xlearner`, `psm`,
`propensity_score`. Tier 2 (report/plot helpers): `rdplot`, `did_report`,
`love_plot`, `marginsplot`, `etable`, `dml_diagnostics`, `rd_dashboard`.

> ⚠️ Do **not** mass-add `References` sections — 940 functions lack one, but
> CLAUDE.md §10 (zero-hallucination) makes auto-generating citations actively
> dangerous. References must be hand-verified against Crossref/DOI.

---

## F. Housekeeping / coordination notes

- **README LOC-drift** (`registry_stats.py --check` flags `269k LOC (core)` in
  `README.md`/`README_CN.md`): pre-existing whole-codebase line-count drift,
  independent of this branch. Refresh via `python scripts/registry_stats.py
  --table` at release time; left untouched here to avoid churn/conflict.
- **Merge coordination:** the sibling agent commits test-coverage to `main`
  (test files only). This branch edits *source* + adds *new* test files, so
  hard conflicts are unlikely; `registry.py`/`_baseline_cards.py`/`schemas/`
  are the files most likely to need a merge rebase.
