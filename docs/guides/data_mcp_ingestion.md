# From a data MCP to an estimate — the last mile

> `sp.from_worldbank` / `sp.from_fred` / `sp.from_sdmx` reshape what a data MCP
> already fetched into a tidy frame `sp.detect_design` → `sp.recommend` →
> `sp.cross_validate` can use.

## The 2026 workflow

The modern empirical loop is **data MCP → estimator**. An agent pulls a series
from an official data server — the World Bank `data360` MCP, `fred-mcp-server`,
the OECD / Eurostat SDMX MCPs — and then has to *estimate something*. StatsPAI
deliberately does **not** re-implement those connectors: they are official,
maintained, and excellent. StatsPAI's job is to be their best **consumer**.

The friction is shape. Each source returns its own JSON dialect — World Bank's
`[metadata, rows]`, FRED's `{"observations": [...]}`, the deeply nested
index-encoded SDMX-JSON of OECD/Eurostat. The ingestion helpers absorb that and
hand you a tidy long (or wide) panel.

These functions **never touch the network** — you pass them a payload the data
MCP already returned, so they are deterministic and offline-testable.

## World Bank

```python
import statspai as sp

# `payload` is whatever the World Bank MCP / REST returned.
panel = sp.from_worldbank(payload)            # tidy long: country, iso3,
                                              #   indicator, indicator_id,
                                              #   year, value
reg_frame = sp.from_worldbank(payload, wide=True)   # one column per indicator,
                                                     #   one row per (iso3, year)
```

`wide=True` is the form you want for a cross-country panel regression: pull
several indicators, get one column each, indexed by `(iso3, year)`.

## FRED

```python
# Single series:
ts = sp.from_fred(observations, series_id="cpi")        # date, cpi

# Several series merged on date in one call:
macro = sp.from_fred({"CPI": cpi_obs, "UNRATE": unrate_obs})   # date, CPI, UNRATE
```

FRED's `"."` missing-value sentinel becomes `NaN`; dates are parsed to
`datetime64` and the frame is sorted.

## OECD / Eurostat (SDMX-JSON)

```python
# SDMX encodes each cell by integer indices into per-dimension code lists;
# from_sdmx expands them back to readable dimension columns.
df = sp.from_sdmx(payload)     # e.g. LOCATION, SUBJECT, TIME_PERIOD, value
```

## End-to-end recipe

A conversational session ties the whole chain together — *no browser, no manual
reshaping*:

```python
import statspai as sp

# 1. (the agent calls the World Bank MCP and gets `payload`)
panel = sp.from_worldbank(payload, wide=True)

# 2. Let StatsPAI read the shape and suggest an estimator.
design = sp.detect_design(panel)
rec = sp.recommend(panel, y="life_expectancy", design=design.design)

# 3. Fit — and cross-validate the headline number across engines (see the
#    cross-engine validation guide).
cv = sp.cross_validate(
    panel, "feols",
    formula="life_expectancy ~ gdp_pc | iso3 + year", treatment="gdp_pc",
    engines=["statspai", "pyfixest", "R::fixest"],
)
print(cv.verdict)
```

That is the picture the econometrics community has been sketching for 2026 —
*"OECD MCP pulls the data, Jupyter MCP runs the code, you never open a
browser"* — with StatsPAI supplying the estimation-and-validation half of the
loop.

## Provenance

Every ingested frame stamps `df.attrs["source"]` (`"worldbank"` / `"fred"` /
`"sdmx"`) so downstream tooling and your own audit trail know where the numbers
came from. It also stamps `df.attrs["provenance"]`, a deterministic dictionary
that records the normalizer, payload kind, output shape, row count, columns and
source-specific labels such as World Bank indicator ids, FRED series ids, or
SDMX dimensions.

`sp.cross_validate` preserves that metadata under
`cv.provenance["data"]`. In an MCP run that loads a CSV/Parquet/Stata file via
`data_path`, the MCP response also includes a top-level `data_provenance`
record with the local file hash or a sanitized remote URL.

## Scope, on purpose

These are **normalisers, not connectors**. They will not fetch data, cache it,
or manage API keys — that is the data MCPs' job, and duplicating it would only
rot. If you need live fetching, call the relevant data MCP (or `wbgapi` /
`fredapi` / `pandasdmx`) and pass the result here.
