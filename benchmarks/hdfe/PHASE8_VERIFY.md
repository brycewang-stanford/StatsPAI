# Phase 8 — Verification Report

**Scope**: ``sp.fast.etable`` — fixest::etable-style side-by-side
regression tables across multiple fitted models.

## Deliverables

* New module ``src/statspai/fast/etable.py``.
* New public API ``sp.fast.etable(*fits, names=, digits=, keep=, drop=,
  format=, se_format=, stars=)``.
* Output formats: ``DataFrame`` (default), ``latex``, ``html``,
  ``markdown``.
* Significance stars ( * /5%/10%/1%) computed from ``coef / se``
  z-statistics; toggleable.
* Cross-class compatibility: anything with ``coef()`` / ``se()`` /
  ``n_obs`` works (FePoisResult, EconometricResults, pyfixest objects,
  custom adapters).

## Test suite

```
tests/test_fast_etable.py
  test_etable_single_fepois ........................ PASSED
  test_etable_multi_models_aligned_by_var .......... PASSED
  test_etable_keep_filter .......................... PASSED
  test_etable_drop_filter .......................... PASSED
  test_etable_below_format_doubles_rows ............ PASSED
  test_etable_latex_output ......................... PASSED
  test_etable_html_output .......................... PASSED
  test_etable_stars_appear_when_significant ........ PASSED
  test_etable_stars_off ............................ PASSED
  test_etable_works_with_event_study ............... PASSED
  test_etable_unknown_format_rejected .............. PASSED
  test_etable_no_models_rejected ................... PASSED
12 passed in 2.94s
```

## Cumulative regression (Phases 1-8)

```
133 passed, 2 skipped
```

## Acceptance against original Phase 8 plan

| plan item                                                | status | note |
| -------------------------------------------------------- | ------ | ---- |
| ``sp.etable(*fits, format="latex"|"html"|"docx")``       | partial | LaTeX, HTML, Markdown ship; Word DOCX is pre-existing on each result type's ``to_docx`` and not yet wired to ``etable``. |
| Side-by-side comparison across models                    | ✅     | aligned by variable name |
| Significance stars                                       | ✅     | ``*`` p<0.10, ``**`` p<0.05, ``***`` p<0.01 |
| ``keep`` / ``drop`` filters                              | ✅     |  |
| Footer: N + R² + log-likelihood (when available)         | ✅     | optional, only when present on the fit object |
| ``sp.paper()`` auto-paper integration                    | ⏸     | ``sp.paper()`` already exists as a separate workflow tool; wiring HDFE-specific table output into it is a follow-up. |
| ``sp.causal_question`` auto-selecting HDFE backend       | ⏸     | Schema-introspection wiring; not part of Phase 8 critical path. |
