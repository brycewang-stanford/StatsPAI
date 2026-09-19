# Phase 3 integration ledger

Working tree: `.claude/worktrees/jss-submit` (branch `wt/jss-submit`, base
`origin/main` d127e7d8). Nothing committed; waits for maintainer authorisation
and for the pending 1.29.0 line (stata-grammar / panel-forest-v2) to land.

## Done in the integration tree

- Eight family patches applied (`pc-*` worktrees); one additive conflict in
  `test_track_a_alias_equivalence.py` resolved by keeping both sides.
- `.gitignore`: `tests/reference_parity/_fixtures/_ado_*/`, `_rlib_*/`.
- `registry.py`: ParamSpecs for 15 new parameters (descriptions copied from
  docstrings); `rdrobust` `p` default None; descriptions corrected for
  `causal_impact`, `harvest_did`, `xtnbreg` (+ model enum),
  `did_timevarying_covariates` (citation verified, `[待核验]` removed),
  `scpi` cost profile (cores= is serial).
- House style: new `four_way_decomposition` variance argument renamed
  `vcov` -> `vce`; `panel_unitroot(robust=)` registered as a false friend
  (Stata `xtunitroot hadri, robust`).
- `_parity_taxonomy.PYTHON_REFERENCE_ROWS`: `blp` (pyblp).
- `scripts/build_parity_index.py`: 132 phase-3 promotions; index, `docs/parity.md`
  regenerated. Estimator callables with a cross-language grade: 224 -> 362 of 773.
- Schemas regenerated.
- `iv_wild_bootstrap`: full Rademacher enumeration when 2**G <= n_boot and
  strict-inequality p (boottest rule); weak-IV fixture now exact vs boottest
  (22361 / 65536), test tightened from atol 4e-3.
- `did_timevarying_covariates` private-helper coverage tests rewritten to the
  new contract; `_article_aliases` doctests (`matrix_completion` 2.86 -> 2.1,
  truth 2.0; `rdd`, `psm` were already failing on main).
- `harvest_did` agent card / test docstring attribution; `xtnbreg` error text.
- Track A: python goldens of 12_sdid, 25_lmm, 26_glmm_logit, 27_glmm_aghq
  regenerated (drift 1.6e-2 / 1.4e-7 / 3.3e-6 / 1.1e-7, all inside budget);
  fixture lock rewritten; parity contract 42/42.
- Citations: 38 new author-year mentions checked against Crossref / RePEc / arXiv.
- Full suite before the index/registry work: 18,403 passed, 9 failed (all
  integrator items, now fixed).

## Blocked on 1.29.0 landing (conflict files)

- `twfe_decomposition`: wire `did/_twfe_weights.dcdh_fe_weights` into
  `did/wooldridge_did.py` (deliverable did_synth.md D4); the headline, Bacon
  weights and dCDH weights are currently wrong. Strict xfail pins it.
- Track A 42/44/45/49/61/62/63/64 python drift comes from d127e7d8; their new
  goldens live in the panel-forest-v2 line. Re-run
  `verify_reproduce_py.py` (full) after rebase.
- Pre-existing doctest failures (11) in `bounds/partial_id.py`,
  `multilevel/lmm.py`, `panel/hdfe.py`, `panel/panel_reg.py`, `synth/mc.py`,
  `tmle/tmle.py` (print / summary without expected output).
- Module 27 Stata headline 9.8e-7 vs 1e-6 budget: switch the do-file to
  `intmethod(mcaghermite)` as the reference quadrature (panel_glmm.md).
- Merge CHANGELOG / MIGRATION drafts (scratchpad `CHANGELOG_phase3.md`,
  `MIGRATION_phase3.md`) into `[Unreleased]`; add the integration items above.

## Paper (Paper-JSS) sync items

- 05-parity-compact.tex: "all but two ... econml / DoubleML" -> three, add
  `pyblp` for `sp.blp`.
- Coverage numbers, correctness-history table (new 1.30.0 row), Track A counts
  after rebase; `make submission-ready` after the release tag.

## Open (round 2 candidates)

rd2d / rd2d_bw / boundary_rd rebuild; malmquist; zisf; mgwr bandwidth search;
spatial_panel two-way vs xsmle; Fisher L* constant (5N+3 vs 5N+4); CUSUM
boundary constant (2.5e-6, grade B); sup-F p-value and sequential Bai-Perron;
gformula_ice point estimate; tmle ATT; margins_at / contrast / pwcompare Stata
fixtures; decisions for the maintainer (lonely PSU default, roc_curve /
direct_standardize / power_case_control defaults, cmprsk GPL-derived formulas
attribution).
