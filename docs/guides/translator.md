# Migrating Stata / R commands automatically

StatsPAI ships **live translators** that turn a Stata command or an R call into
the equivalent `sp.*` invocation — so moving an existing `.do` file or R script
across is mechanical, not a rewrite. The translators are available three ways:

- `sp.from_stata("...")` / `sp.from_r("...")` — directly from Python;
- as MCP tools (`from_stata` / `from_r`) an agent can call on a user's snippet;
- and their coverage is itself queryable via `sp.translation_coverage()`.

```python
import statspai as sp

sp.from_stata("reghdfe y x, absorb(id year) vce(cluster id)")
# → {'tool': 'hdfe_ols',
#    'python_code': "sp.hdfe_ols('y ~ x | id + year', data=df, cluster='id')",
#    'notes': [], 'untranslated_options': [], 'unapplied_sample': None, 'ok': True}

sp.from_r("feols(y ~ x | id, data = df)")
# → {'tool': 'feols', 'python_code': "sp.feols('y ~ x | id', data=df)", ...}
```

Each call returns one ready-to-run `python_code` string plus a `notes` list. The
design is deliberate:

- **Hand-curated, never a guess.** `vce(cluster id)` and `cluster(id)` mean
  different things in different Stata commands, so every command is mapped by
  hand to preserve its exact semantics.
- **No silent option loss.** Every option is either carried into the call,
  listed in `untranslated_options` (with a note), or listed in
  `ignored_display_options` when it only changes what Stata prints. An
  `if` / `in` qualifier is returned in `unapplied_sample`. A translation
  with an empty `untranslated_options` and no `unapplied_sample` is the
  same model; anything else is a partial translation that says so.
- **One command per call.** Multi-command `.do` files and multi-line R scripts
  must be split by the caller first.
- **Failure is non-fatal.** An unrecognized command returns
  `{"ok": False, "error": ..., "suggestions": [...]}` rather than raising.

## Stata grammar handled for every command

Real do-files rarely spell commands out in full. The following is handled
once, for all translated commands, before any command-specific mapping:

| Written as | Treated as |
| --- | --- |
| `reg y x, r` · `vce(r)` | `robust` |
| `cl(id)` · `vce(cl id)` | `cluster(id)` |
| `reghdfe ..., a(id year)` (also `ivreghdfe`, `ppmlhdfe`) | `absorb(id year)` |
| `nocons` | `noconstant` |
| `qui` / `cap` / `noi`, `eststo m1:`, `xi:` | peeled: they only change what is printed or stored |
| `level(90)` | `alpha=0.1` where the function takes `alpha` |

Three things are refused instead of guessed at:

- **Prefixes that change the estimate**: `by` / `bysort`, `bootstrap`,
  `jackknife`, `permute`, `svy`, `rolling`, `statsby`. Dropping them would
  translate a different model.
- **Macros** (`$controls`, `` `x' ``). Their contents are defined on another
  line of the do-file; write the variable list out.
- **`xtreg` without `fe`**. That is Stata's random-effects estimator; use
  `sp.panel(method='re')`.

### A do-file snippet, not just one line

`sp.stata(...)` takes the lines as they stand in a do-file. It resolves what
can be resolved by reading:

```python
sp.stata("""
    * baseline specification
    global ctrl "age educ"
    local fe id year
    xtset id year
    reghdfe y x $ctrl ///
        , a(`fe') cl(id)        // main
    xtreg y x $ctrl, fe r
""", data=df)
```

- `//`, `*` and `/* */` comments, `///` continuations and `#delimit ;`;
- `global` / `local` macros whose value is written out (`global ctrl "age
  educ"`, `local k = 3`), including macros defined from other macros;
- the panel declared by `xtset id year`, which supplies the `<panel_id>` of
  a later `xtreg, fe` or `xtabond`.

What only Stata can evaluate is refused: a macro set by an expression or an
extended function (`local n = _N`, `local k : word count ...`), a macro that
is never defined (Stata would expand it to nothing and run another model),
and `foreach` / `forvalues` / `program` blocks. Write the loop in Python
around `sp.stata`.

`sp.stata(...)` runs a translation only when nothing was lost: an entry in
`untranslated_options` raises `MethodIncompatibility` with the call to run
by hand.

### Qualifiers and data steps

`sp.stata` applies `if` and `in`, and runs the data steps a do-file usually
has between its estimation lines:

```python
sp.stata("""
    gen lwage = ln(wage)
    gen exp2 = exper^2
    keep if !missing(union)
    reg lwage educ exper exp2 if year >= 1985, r
    test exper exp2
""", data=df)
```

The caller's DataFrame is never modified. What is run:

| Stata | Note |
| --- | --- |
| `if exp`, `in f/l` | on estimation and descriptive commands |
| `generate`, `replace` | `x[_n-1]`, `_n`, `_N`; stores single precision unless `double`, as Stata |
| `keep` / `drop` (`if`, `in` or a variable list), `sort`, `preserve` / `restore` | |
| `mvdecode v, mv(#)`, `encode s, gen(v)` | |
| `predict v [, xb \| residuals]` | after `regress` / `ivreg` on plain columns |
| `scalar s = exp`, `display exp` | may use `_b[x]`, `_se[x]`, `e(N)`, `e(r2)`, `e(r2_a)`, and `r()` after `summarize`, `test`, `ttest` |

Expressions follow Stata's rules for missing values, which is where a
hand-written pandas filter goes wrong:

| Stata | Value where `x` is missing | pandas `df.x > 0` |
| --- | --- | --- |
| `x > 0` | true (missing is larger than any number) | false |
| `x < .` | false: the idiom for "x is observed" | |
| `x == .` | true | `NaN == NaN` is false |
| `if x` | true (missing is not zero) | |
| `x + 1`, `x / 0`, `ln(-1)` | missing | |

The functions available are `ln` `log` `log10` `exp` `sqrt` `abs` `floor`
`ceil` `int` `round` `mod` `min` `max` `sign` `cond` `missing` `mi`
`inlist` `inrange` `normal` `normalden` `invnormal` `chi2` `chi2tail`
`ttail` `invttail` `F` `Ftail`. Anything else is refused with the reason:
`e(sample)`, string functions, extended missing values (`.a`), `egen`,
`merge`, `reshape`, `collapse`. `use` is
refused as well, since it replaces the data; pass the DataFrame in. Settings
and output-only lines (`set more off`, `log using`, `label`, `describe`) are
skipped, and graph or export commands are skipped with a warning.

### Time-series operators

After `tsset time` or `xtset id time`, `L.x`, `L2.x`, `F.x`, `D.x`, `D2.x`,
`LD.x` and lag lists such as `L(1/4).x` are resolved against the time
variable, inside each panel, as Stata does:

```python
sp.stata("""
    tsset quarter
    newey growth L(1/2).growth L(1/2).spread, lag(4)
    test L.spread L2.spread
""", data=df)
```

A lag is missing where the earlier period is absent, which `shift(1)` on
the rows does not give you when a quarter is missing or the data are a
panel. The term `L2.spread` enters the model as a column named `spread_L2`,
and `test` / `lincom` accept either spelling. The time variable has to be a
count of periods or a pandas `Period`; a datetime column has no unit step
and is refused.

## What's covered — and how to check

The coverage is **introspectable**, so it can never drift from what the
translators actually do:

```python
cov = sp.translation_coverage()
cov["summary"]      # {'n_stata_commands': 51, 'n_r_functions': 16, ...}
cov["stata"]        # [{'command': 'reghdfe', 'targets': ['sp.hdfe_ols'], ...}, ...]
cov["limitations"]  # the documented gaps (see below)

print(sp.translation_coverage(fmt="markdown"))   # a ready-to-read table
```

Flagship Stata mappings (run `sp.translation_coverage()` for the authoritative,
always-current list):

| Stata | → StatsPAI |
| --- | --- |
| `regress` / `reg` | `sp.regress` |
| `reghdfe`, `ivreghdfe` | `sp.hdfe_ols` (reghdfe's singleton / dof / `t(G-1)` rules; `a#b` becomes `a^b`) |
| `xtreg, fe` | `sp.feols` |
| `summarize`, `sum2docx` | `sp.sumstats` |
| `correlate`, `pwcorr` | `sp.pwcorr` (casewise for `correlate`, pairwise for `pwcorr`) |
| `ttest` | `sp.ttest` |
| `ivreg2` / `ivregress` / `ivreg` | `sp.ivreg` (`robust` without `small` is `robust='hc0'`; the legacy `ivreg` is small-sample) |
| `probit`, `logit`, `poisson`, `nbreg`, `tobit` | `sp.probit` ... ; `vce(robust)` is `robust='robust'`, Stata's `N/(N-1)` |
| `newey y x, lag(m)` | `sp.regress(robust='hac', hac_lags=m, hac_small=True)` |
| `dfuller` | `sp.unitroot(test='adf')` |
| `csdid`, `didregress`, `did_imputation` | `sp.callaway_santanna` / `sp.did` / `sp.did_imputation` |
| `rdrobust`, `rdplot`, `rddensity` | `sp.rdrobust` / `sp.rdplot` / `sp.rddensity` |
| `synth` | `sp.synth` |
| `teffects` | `sp.ipw` / `sp.match` / `sp.aipw` |
| `psmatch2`, `ppmlhdfe`, `heckman`, `boottest` | `sp.psmatch2` / `sp.ppmlhdfe` / `sp.heckman` / `sp.wild_cluster_bootstrap` |

Flagship R mappings:

| R | → StatsPAI |
| --- | --- |
| `feols` / `felm` (fixest / lfe) | `sp.feols` |
| `lm` / `glm` | `sp.regress` / `sp.glm` |
| `plm`, `lmer` / `glmer` | `sp.panel` / `sp.mixed` / `sp.melogit` (or `sp.meglm`) |
| `att_gt` / `did` | `sp.callaway_santanna` |
| `matchit` (MatchIt) | `sp.match` |

## Known limitations

These are part of the queryable contract — `sp.translation_coverage()["limitations"]`:

- **Panel id/time.** `xtreg` / `xtabond` / `xtnbreg` emit a `<panel_id>`
  placeholder when the `xtset` / `tsset` declaration is on a different line; pass
  `id=` / `time=` explicitly to the resulting `sp.*` call.
- **Time series.** `arima` / `var` / `vec` / `granger` are not translated — call
  `sp.arima` / `sp.var` / `sp.johansen` directly. `dfgls` is answered with
  the matching `sp.unitroot` call. The seasonal operator `S.` is refused.
- **Estimation tables.** `esttab` / `eststo` / `outreg2` are not translated; use
  `sp.regtable` on the fitted results.
- **Dropped qualifiers are surfaced, not lost.** From `sp.from_stata`, which
  translates one line without data, a Stata `if`/`in` qualifier comes back
  in `unapplied_sample` and an unrecognized option in
  `untranslated_options`, each with a note. `sp.stata` applies the
  qualifier.
- **`xtreg, fe` has no `_cons`.** Stata prints the average fixed effect;
  `sp.feols` absorbs it.
- **`summarize, detail` percentiles** are interpolated between order
  statistics; Stata's rule picks an order statistic, so the two can differ
  within a gap between observations.
- **Macros and loops.** `sp.from_stata` translates one command and refuses a
  macro; `sp.stata` expands the macros defined by text in the same snippet.
  Loops and macros computed by Stata are not run.
- **`psmatch2` without `logit`.** Stata then fits a probit propensity score;
  `sp.psmatch2` fits a logit, so the translation lists `probit` in
  `untranslated_options`.

For the hand-written equivalence reference, see also
[Migrating from R to StatsPAI](migration-from-r.md).

## Checking a set of do-files

`python scripts/stata_corpus_scan.py <folder> [--csv out.csv]` runs every
estimation command of the do-files under a folder through the translator and
reports how many translate faithfully, how many say what they lost, and which
commands and options account for the rest, ranked by the number of projects
they appear in. It is the detector behind the grammar rules above; fixes go
into the translator against Stata's documented syntax, with synthetic tests.

That scan asks whether a command is translated. When the logs are at hand,
`python scripts/stata_log_replay.py <logs> --data <folder with the .dta files>`
asks whether it gives Stata's numbers: it runs every logged command through
one `sp.stata` session and compares each coefficient, standard error, test
statistic, p-value and displayed scalar with what Stata printed, to the
precision it printed. A translation that fits the right model with the wrong
small-sample factor passes the scan and fails the replay.
