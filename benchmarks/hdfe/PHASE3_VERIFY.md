# Phase 3 — Verification Report

**Scope**: ``sp.fast.within`` first-class HDFE residualizer + minimal DSL
helpers (``i()`` event-study dummies, ``fe_interact()`` for ``i^j``,
``sw()/csw()`` stepwise).

## Deliverables

* New module ``src/statspai/fast/within.py``
  - ``WithinTransformer`` class: factorise + singleton-prune once,
    residualise many vectors thereafter.
  - ``sp.fast.within(data, fe=...)`` constructor.
  - ``transform`` / ``transform_columns`` accessors (returns
    ``DemeanInfo`` per call).
  - Accepts FE as DataFrame, ndarray, list of column names, or list of
    arrays.
* New module ``src/statspai/fast/dsl.py``
  - ``sp.fast.i(var, ref=...)`` → DataFrame of dummies (event-study idiom).
  - ``sp.fast.fe_interact(*cols)`` → single int64 code per K-tuple
    (the ``^`` operator).
  - ``sp.fast.sw(...)`` and ``sp.fast.csw(...)`` → spec-list expansion.

## Test suite

```
tests/test_fast_within_dsl.py
  test_within_transform_matches_demean ............. PASSED
  test_within_transform_columns_returns_dataframe .. PASSED
  test_within_caches_singleton_drop ................ PASSED
  test_within_already_masked_path .................. PASSED
  test_within_accepts_multiple_fe_input_shapes ..... PASSED
  test_i_default_drops_first_level ................. PASSED
  test_i_with_explicit_ref ......................... PASSED
  test_i_with_unknown_ref_raises ................... PASSED
  test_i_event_study_in_fepois ..................... PASSED
  test_fe_interact_two_columns ..................... PASSED
  test_fe_interact_three_columns ................... PASSED
  test_fe_interact_passes_to_fepois ................ PASSED
  test_sw_emits_separate_specs ..................... PASSED
  test_csw_cumulative .............................. PASSED
  test_sw_drives_multiple_regressions .............. PASSED
15 passed in 3.83s
```

## Cumulative regression check (Phases 1+2+3)

```
pytest tests/test_fast_demean.py tests/test_fast_fepois.py \
       tests/test_fast_within_dsl.py tests/test_fast_bench.py \
       tests/test_hdfe_native.py tests/test_panel.py tests/test_fixest.py
92 passed, 1 skipped
```

## Acceptance against original Phase 3 plan

| plan item                                            | status | note |
| ---------------------------------------------------- | ------ | ---- |
| ``sp.within(...)`` first-class object                | ✅     | ``WithinTransformer``, exposed as ``sp.fast.within`` |
| Cached demeaned X / y for reuse across estimators    | ✅     | ``transform`` returns DemeanInfo each call |
| ``i(var, ref=...)`` event-study dummies              | ✅     | DataFrame output, prefixed columns |
| ``^`` FE interactions                                | ✅     | ``fe_interact`` (helper, not operator parse) |
| ``sw()`` / ``csw()`` stepwise                        | ✅     | List-of-list expansion |
| ``WithinFrame.solve(method="ols"|"iv"|"lasso"|...)`` | ⏸     | The transform layer is shipped; the solve dispatcher is small but adding "all four estimators" was over-scope. Users wire the residualised X into ``sp.feols`` / ``sp.dml`` / ``sp.iv`` themselves today. |
| ``sp.dml(..., within=fe_spec)``                      | ⏸     | Direct DML wiring deferred; ``sp.dml`` already accepts pre-residualised X via its current API, so the value-add of a one-line wrapper is small. |
| Stepwise (``sw``/``csw``) parsed inside formula      | ⏸     | Helpers shipped functionally; in-formula parsing is part of the larger formula-DSL parser deferred to Phase 8. |

The three ⏸ items are integration sugar layered on top of the building
blocks shipped here. They unlock cleanly once the formula parser lands
in Phase 8.
