# Grammar + SE-menu convergence — week roadmap

Tracks the "consistent grammar + universal SE menu" workstream prompted by the
reviewer note on signature consistency. **Constraint (ratified 2026-06-30):**
additive + reversible only during JSS review — no estimator default or numerical
output changes; no hard deprecations until after acceptance. Canonical
spellings: `vce` (SE), `y` (outcome), `treat` (treatment).

## Done (this branch: `worktree-grammar-se-menu`)

- **D1 — signature house-style lint.** `src/statspai/_house_style.py` (canonical
  vocab + alias map + false-friend allowlist), `scripts/signature_house_style.py`
  (introspection lint, `--json/--check/--update-baseline` ratchet),
  `signature_house_style_baseline.json` (186 legacy sites, 41 acknowledged false
  friends), `tests/test_signature_house_style.py`.
- **D2 — alias plumbing.** `src/statspai/_aliases.py` (`@accepts_aliases`,
  silent additive forwarding, warns gated off, conflict→TypeError,
  signature-preserving). Applied to `sp.regress` (`vce=`→`robust=`, coef+SE
  bit-identical across nonrobust/hc1/hc3/cluster). `tests/test_aliases.py`.
- **D4a — SE-menu matrix gate.** `scripts/se_menu_matrix.py` (curated coverage
  matrix validated against the live API; drift guard; native-up/unsafe-down
  ratchet). Today: 34 native, 5 standalone, 5 unsafe (`ivreg`), 12 stranded.
  `tests/test_se_menu_matrix.py`.
- **Grammar doc.** `docs/guides/grammar.md` (canonical vocab + honest SE matrix
  + the two gates).
- **D3 — shared design extractor.** `inference/jackknife.py::_design_from_result`
  prefers stored `data_info['X'/'y'/'var_names']`, falls back to the formula
  re-parse only when rows can't be aligned (NA-filtered samples). `cr2_se` and
  `wild_cluster_boot` rerouted through it. Proven **byte-identical** to the
  re-parse path on `sp.regress` (forced-fallback test).
- **D4b (partial) — feols within-design.** `fixest/adapter.py` now stores the
  pyfixest within-transformed design (`_X`/`_Y`/`_coefnames`) on the result —
  purely additive, no feols numerics change. So `wild_cluster_boot` / `conley` /
  `twoway_cluster` / `cr2_se` now run on `sp.feols` (previously `KeyError: 'X'`).
  Matrix: feols `wild_cluster_boot` and `conley` flipped `na → standalone`.
  **Verified with no external tools:** (a) feols-no-FE ≡ regress exactly for
  cr2/wild/conley/twoway; (b) feols-FE wild bootstrap byte-identical to
  `regress` on manually FE-demeaned data (the textbook within wild bootstrap).
  `tests/test_se_menu_feols_wiring.py` (8 tests).
- **D4b-cont — native `feols(vce="wild")`.** `fixest/wrapper.py` now exposes a
  first-class `feols(..., vce="wild", cluster=...)` running the WCR wild cluster
  bootstrap on the within design (plus `vce=` as the canonical alias for `vcov=`
  and `cluster=` as one-way CRV1 shorthand, mirroring `sp.regress`). Matrix:
  feols `wild_cluster_boot` flipped `standalone → native` (native cells 34→35).
  **External parity vs Stata** `reghdfe`+`boottest` (18 MP, boottest 4.5.3):
  point estimate & CRV1 SE match `reghdfe` to ~1e-9, wild p-value matches
  `boottest`'s exact 2¹⁵ enumeration to MC error (0.0265 vs 0.0263).
  `tests/reference_parity/test_feols_wild_boottest_parity.py` (frozen Stata
  values, no R/Stata needed in CI).

## Remaining

### D3-cont — single vcov backend  *(deep, multi-session)*
- Wire `core/_vcov.py` canonical primitives (`sandwich_vcov`, `cluster_robust_vcov`,
  `hc_vcov`) as the single backend; retire per-estimator sandwich re-implementations.
- ⚠ Collision risk: `core/results.py` is edited by the parallel window — rebase/merge carefully.

- **D4b-cont — native `feols(vce="wild")`** (done; see commits on main).
- **IV-WRE — native `ivreg(vce="wild")`** (done). 2SLS structure stored on the
  result under `data_info['iv']`; `inference/iv_wild.py` implements the WRE
  (wild restricted efficient) bootstrap of Davidson-MacKinnon (2010), validated
  against Stata `ivreg2`+`boottest` across strong-IV (0.2016 vs 0.20155) and
  weak-IV (0.3415 vs 0.3412) regimes (the weak case confirms the *efficient*
  reduced form). Matrix: ivreg `wild_cluster_boot` flipped `unsafe → native`
  (native cells 35→36, unsafe 5→4). `tests/reference_parity/test_iv_wild_*`.

### D4b-cont — remaining SE-menu cells  *(touches estimators)*
Against the `se_menu_matrix` gate:
- **feols `cr2_cr3`**: held at `na`. The within-transform CR2 omits the FE
  projection from the leverage adjustment, so it is NOT guaranteed equal to
  `clubSandwich`'s absorbed-FE CR2. Flip only after an R `clubSandwich` parity
  test (needs the absorbed-FE hat-matrix correction).
- **Unify the two `feols` doors** (pyfixest vs panel) so their SE menus match.
- **IV two-way cluster** (done): native `ivreg(cluster=["a","b"])` via CGM 2011
  inclusion-exclusion on the projected regressors; matches `ivreg2, cluster(a b)
  small`. Matrix ivreg twoway `unsafe → native` (native 36→37, unsafe 4→3).
- **Multi-endogenous WRE** (done): `iv_wild_bootstrap` handles ≥1 endogenous
  regressor (the restricted estimation re-estimates the other endogenous by
  2SLS); validated vs boottest for the two-endogenous case.
- **IV CR2 / CR3** (done): native `ivreg(vce="CR2"/"CR3")` (== `"jackknife"`),
  Pustejovsky-Tipton (2018) adjustment on the projected 2SLS regressors; matches
  R `clubSandwich` to machine precision (strong + weak panels). Matrix cr2_cr3
  and jackknife `unsafe → native` (native 37→39, unsafe 3→1).
- **IV Conley** (done): native `ivreg(vce="conley", conley_lat=, conley_lon=,
  conley_cutoff=)`, spatial HAC on the projected 2SLS scores with acreg's planar
  distance; matches Stata `acreg ... spatial` to machine precision. Matrix
  conley `unsafe → native` (native 39→40, **unsafe → 0**).

**`ivreg`'s SE menu is complete: 8/8 native, all externally validated** (Stata
`boottest` / `ivreg2` / `acreg`, R `clubSandwich`). The whole coverage matrix
now has zero `standalone_unsafe` cells.

- **regress(vce=...) — 8/8 native (done).** cr_vcov_ols (CR1/CR2/CR3) + ols_conley_vcov
  (acreg planar) added to inference/jackknife.py + inference/conley.py.
  `regress(vce="CR2"/"CR3"/"jackknife"/"wild")`, `regress(cluster=[a,b])`
  two-way, and `regress(vce="conley", ...)` wired in ols.py. All five
  cells match Stata `reghdfe`/`acreg`/R `sandwich::vcovCL` to machine
  precision on the same 400-obs panel (frozen in
  `tests/reference_parity/test_ols_se_external_parity.py`). Matrix:
  regress twoway/cr2_cr3/jackknife/conley/wild_cluster_boot
  `standalone → native` (native 44→45, standalone 6→1). The whole
  coverage matrix now has only `conley` (for feols) and the in-progress
  row in `iv_wild` left as `standalone` — **no `standalone_unsafe` cells**.

### R3 — bias-reduced / spatial menu on the FE estimators
- **feols(vce=…) — 8/8 native (done).** `feols(vce="CR2"/"CR3"/"jackknife")` and
  `feols(vce="conley", conley_lat=/lon=/cutoff=)` on the FE-absorbed within
  design (`fixest/wrapper.py::_feols_bias_reduced`, reusing `cr_vcov_ols` /
  `conley_vcov_ols`). The within-transform's leverage adjustment reproduces R
  `clubSandwich::vcovCR(plm, model="within", type=…)` to machine precision
  (frozen `PLM_CR2_X=0.056095873`, `PLM_CR3_X=0.057956013`), resolving the D4b
  "held at na" concern — the absorbed-FE CR2 leverage *is* handled correctly by
  the within design. Matrix: feols cr2_cr3/jackknife `na → native`, conley
  `standalone → native` (native 44→48). `test_feols_bias_reduced_parity.py`.
- **panel(method="fe", vce=…) — full menu (done).** `sp.panel` FE now accepts
  `vce="CR2"/"CR3"/"jackknife"`, `vce="conley"` and two-way `cluster=[a,b]` via
  `panel_reg.py::_panel_bias_reduced` (entity within-transform → the same OLS
  helpers). OLS on the entity-demeaned design reproduces the linearmodels FE
  coefficients, so CR2/CR3 hit the *identical* clubSandwich anchor as feols, and
  Conley / two-way match `sp.regress` on the hand-demeaned data. Matrix: panel
  twoway/cr2_cr3/jackknife/conley `na → native` (native 48→52).
  `test_panel_bias_reduced_parity.py`.
- **fepois / feglm(vce="CR2"/"CR3"/"jackknife") — native (done).** GLM
  bias-reduced cluster SEs via `inference/jackknife.py::glm_cr_vcov` — the
  IRLS-weighted (`d=dμ/dη`, `V=Var(μ)`, `w=d²/V`) generalisation of the
  Pustejovsky-Tipton adjustment, dispatched from `fixest/wrapper.py`. Matches R
  `clubSandwich::vcovCR(glm, type=…)` for Poisson **and** logit to convergence
  precision. Key finding: unlike OLS, the weighted-projection FWL does **not**
  carry the CR2 leverage through FE absorption (the absorbed design differs by
  ~1%), so the SE is computed on the FE-as-dummies design and guarded against
  high-dimensional FE. Matrix: fepois/feglm cr2_cr3/jackknife `na → native`
  (native 52→56). `test_feglm_bias_reduced_parity.py`.
- **fepois / feglm(vce="wild") — native, BIT-EXACT vs boottest (done).** Restricted
  score wild cluster bootstrap (Kline-Santos 2012) via
  `inference/jackknife.py::glm_score_wild_boot`, the method Stata `boottest` runs
  after `poisson`/`logit`. Initially shipped as a ~2-decimal-consistent canonical
  version; then boottest's studentization was **reverse-engineered from its
  enumerated bootstrap distribution** (numerators + t statistics extracted via
  `svmat`, per-replication sign vectors recovered from the numerator multiset,
  the denominator solved as an exact quadratic form in the signs) and
  corroborated against `boottest.mata`. The convention: numerator = restricted
  one-step per-cluster contributions `q_g = (A s_g)[j]`; denominator =
  **cluster-share-centered** CRVE `D²(w) = Σ_g (w_g q_g − (n_g/N) N(w))²`
  (boottest.mata's `ClustShare` centering); observed statistic = the identity
  draw's t* (= boottest's `r(z)`); p counts **strict** exceedances over the
  enumerated 2^G grid (identity + its negation tie exactly and are excluded).
  Reproduces boottest's enumerated p **exactly** on four frozen references
  (Poisson x3 p=0.31640625 & z=-1.0999434, Poisson x2 p=0, logit x3
  p=0.95703125, logit x1 p=0.00390625); tie margins ≥ 4.6e-4 make the
  exact-equality tests stable. Matrix fepois/feglm wild_cluster_boot
  `na → native` (native 57→59). `test_feglm_wild_boottest_parity.py`.
- **hdfe_ols(vce=…) — 8/8 native (done).** The native-numpy HDFE estimator
  gains the canonical `vce=` menu on its absorber's within design:
  `robust`/`hc1`/`hc0` (HC with reghdfe's `N/(N-k-df_a)` factor — frozen Stata
  `reghdfe, vce(robust)` reference se_x=0.0431561136), `CR2`/`CR3`/`jackknife`
  (identical frozen clubSandwich plm anchor as feols/panel, via the new
  `cr_vcov_matrix` core extracted from `cr_vcov_ols`), `conley` (via
  `conley_vcov_matrix`; equals sp.regress on FE-demeaned data), and `wild` as
  shorthand for its native WCR path. Matrix hdfe_ols
  hc_robust/cr2_cr3/jackknife/conley `na → native` (native 59→63).
  `test_hdfe_ols_se_menu_parity.py`.
- **panel(method="fe", vce="wild") — native (done).** WCR wild cluster
  bootstrap on the entity-within design through the SAME
  `inference.jackknife.wild_cluster_boot` engine as `sp.regress(vce="wild")`:
  byte-identical p-values to regress on the hand-demeaned data with the same
  seed (discriminating null-covariate anchor). SEs stay CR1; p/CI from the
  bootstrap. Matrix panel wild_cluster_boot `na → native` (native 63→64).
  **Every regression-family estimator (regress/feols/ivreg/panel/hdfe_ols) is
  now 8/8 native.** `test_panel_bias_reduced_parity.py::test_panel_fe_wild_*`.
- **ppmlhdfe(cluster=[a,b]) — native two-way (done).** CGM-2011 inclusion-
  exclusion on the FE-residualised PPML design (`regression/count.py::
  _twoway_cluster_vcov`) with the single `G_min/(G_min-1)` factor — **byte-
  identical to Stata `ppmlhdfe ..., cluster(a b)`** (verified via Stata 18 MP;
  the one-way path already matched Stata exactly). Matrix ppmlhdfe twoway
  `na → native` (native 56→57). `test_ppmlhdfe_twoway_parity.py`. Note: GLM
  CR2/CR3 is *not* offered for ppmlhdfe — the reference-matching version needs
  the FE-as-dummies design (the absorbed CR2 differs ~1% with no published
  reference), which defeats ppmlhdfe's high-dimensional-FE purpose; use
  `cluster=` / two-way there.

### D5 — unify the result contract  *(collision risk: results.py)*
- One §3-true protocol: `summary`/`plot`/`to_latex`/`to_word`/`to_excel`/`cite`
  (today only 11/279 classes satisfy all six; `to_word` 6%, `to_excel` 7%).
- Reconcile naming: `postestimation_contract` advertises `to_docx`/`to_html`
  while §3 / canonical classes use `to_word`/`to_excel` — pick one, alias the other.
- Push `ResultProtocolMixin` (71 classes) to the full six.
- Convert the worst ~67 raw-`dict`-returning functions (`hausman_test`, `irf`,
  `evalue`, `sensemakr`, `wild_cluster_*`, …) to lightweight result objects.
- Extend `result_protocol_audit.py` to track the §3 six and gate it.

### D6 — family front doors  *(collision risk: recommend.py, detect_design)*
- BJS: `bjs` = `borusyak_jaravel_spiess` = `did_imputation` → one canonical +
  deprecated aliases. Same for `gardner_did` = `did_2stage`.
- Move/deprecate the misfiled `did_estimate` (a synthdid alias in the DiD namespace).
- `did(method=)` route to staggered estimators, or document the front door.
- Unify `detect_design` (3 designs) and `recommend._detect_design` (7) into
  shared code; add synthetic-control coverage (today zero in `recommend`).
- Coordinate with the parallel window already editing `recommend.py`.

### D7 — CI wiring + paper check
- Promote `signature_house_style --check` and `se_menu_matrix --check` to
  blocking gates in `.github/workflows/ci-cd.yml` (alongside `error_taxonomy_audit`).
- Add `result_protocol_audit` §3 gate once D5 lands.
- **Audit `paper.md` / JSS manuscript** for any "universal SE menu" wording that
  overstates current coverage — the matrix shows wild/CR2/Conley are OLS-path +
  FE-path-wild today, not universal. Reword to match reality before it reaches a reviewer.

## Merge note
This branch is isolated from a concurrent window editing `results.py`,
`recommend.py`, `iv.py`, `registry.py`, `README*`. New files here don't collide;
the only shared edit is `regression/ols.py` (one decorator line). Reconcile
`_house_style.py`/test refinements (e.g. the `rd_flex` `W` false-friend) when merging.
