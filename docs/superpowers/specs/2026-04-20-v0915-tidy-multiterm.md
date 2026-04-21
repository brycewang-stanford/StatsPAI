# StatsPAI v0.9.15 — Multi-term `.tidy()` for BayesianMTEResult

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.13 added per-population ATT / ATU uncertainty fields on
`BayesianMTEResult`, and v0.9.14 wired them into `summary()`. But
`.tidy()` still returns a single row describing the ATE (primary
estimand). Users doing broom-style meta-analysis

```python
table = pd.concat([r.tidy() for r in results])
```

cannot access the ATT / ATU uncertainty through this pipeline
without falling back to attribute reads. v0.9.15 fills that gap:
`.tidy(terms=['ate', 'att', 'atu'])` returns a 3-row DataFrame with
the same broom schema.

Adding this requires one small upstream change: ATT / ATU need a
`prob_positive` scalar to match the schema. Currently
`_integrated_effect` discards the per-draw samples after computing
the posterior summary; we extend the 4-tuple return to a 5-tuple
`(mean, sd, hdi_lower, hdi_upper, prob_positive)` and store the
new scalars on the result.

## 2. Scope

### In scope (v0.9.15)

- **`BayesianMTEResult.tidy(terms=None | str | Sequence[str])`**:
  - `terms=None` (default): unchanged — single ATE row, back-compat.
  - `terms='ate'` / `'att'` / `'atu'`: single row of that term.
  - `terms=['ate', 'att']` etc.: multi-row DataFrame.
  - Invalid term names reject with a clear `ValueError`.

- **New `BayesianMTEResult` fields**:
  - `att_prob_positive: float = float('nan')`
  - `atu_prob_positive: float = float('nan')`

- **`_integrated_effect` returns 5-tuple** `(mean, sd, hdi_lower, hdi_upper, prob_positive)`. Caller unpacks into the new fields.

### Out of scope (deferred)

- `.tidy(terms=...)` on other Bayesian estimators (DID/RD/IV/fuzzy_rd/hte_iv) — they don't have ATT/ATU; their primary estimand row is already what they return.
- Full bivariate-normal HV.
- Rust Phase 2.

## 3. API

```python
def tidy(
    self,
    conf_level: Optional[float] = None,
    terms: Union[None, str, Sequence[str]] = None,
) -> pd.DataFrame:
    """Broom-style tidy summary.

    Parameters
    ----------
    conf_level : float, optional
        Unused for Bayesian output; HDI level set at fit time.
        Accepted for parity with ``CausalResult.tidy``.
    terms : None | str | sequence of str, default ``None``
        Which term(s) to include:

        - ``None`` or ``'ate'`` — single ATE row (back-compat).
        - ``'att'`` / ``'atu'`` — single row of that term.
        - list like ``['ate', 'att', 'atu']`` — multi-row.

        Unknown names raise ``ValueError``. When an ATT / ATU term
        is requested but the corresponding SD field is NaN (e.g.
        the fit had no untreated units, or the result was
        deserialised from an earlier snapshot), that row's
        uncertainty columns are filled with NaN but the row is still
        emitted.
    """
```

Returned schema (per row):

| column | value |
|---|---|
| `term` | `'ate'` / `'att'` / `'atu'` |
| `estimate` | posterior mean |
| `std_error` | posterior SD |
| `statistic` | estimate / std_error (NaN if SD == 0) |
| `p_value` | NaN (not frequentist) |
| `conf_low` | HDI lower |
| `conf_high` | HDI upper |
| `prob_positive` | posterior P(term > 0) |
| `hdi_prob` | fit's HDI coverage |

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/_base.py` | `BayesianMTEResult.tidy(terms=...)` override; two new fields; docstring |
| `src/statspai/bayes/mte.py` | `_integrated_effect` returns 5-tuple; caller wires `prob_positive` into fields |
| `tests/test_bayes_mte_tidy.py` | NEW — terms resolution, row shape, schema, invalid term validation |
| `pyproject.toml` | `version = "0.9.15"` |
| `CHANGELOG.md` | 0.9.15 entry |

## 5. Test plan

1. `test_tidy_none_back_compat` — default returns single ATE row, identical schema to v0.9.14.
2. `test_tidy_ate_single_row` — explicit `terms='ate'`.
3. `test_tidy_att_single_row` — `terms='att'` returns one row with `term='att'`, finite estimate/SD.
4. `test_tidy_all_three_terms` — `terms=['ate', 'att', 'atu']` returns 3 rows in order.
5. `test_tidy_invalid_term_raises` — `terms=['bogus']`.
6. `test_tidy_row_schema` — columns match spec-defined set.
7. `test_tidy_backward_compat_nan_prob_positive` — stub with NaN `att_prob_positive` → ATT row still emitted with NaN `prob_positive`.
8. `test_tidy_concat_workflow` — `pd.concat([r1.tidy(terms=['ate','att']), r2.tidy(terms=['ate','att'])])` produces a well-formed 4-row DataFrame.

## 6. Success criteria

1. Default-path `.tidy()` behaviour unchanged (back-compat).
2. `terms=['ate', 'att', 'atu']` returns 3 rows with correct schema.
3. Empty-subpopulation case produces a row with NaN uncertainty columns (never crashes).
4. Two rounds of review, zero ship-blockers.
5. Full Bayesian focused suite stays green.
