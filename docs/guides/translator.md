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

`sp.stata(...)` runs a translation only when nothing was lost: an entry in
`untranslated_options` or an `if` / `in` qualifier raises
`MethodIncompatibility` with the call to run by hand. Filter the DataFrame
first for `if` (Stata treats missing as +infinity in comparisons; pandas does
not, so the filter is not applied automatically).

## What's covered — and how to check

The coverage is **introspectable**, so it can never drift from what the
translators actually do:

```python
cov = sp.translation_coverage()
cov["summary"]      # {'n_stata_commands': 38, 'n_r_functions': 11, ...}
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
| `ivreg2` / `ivregress` | `sp.ivreg` |
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
  `sp.arima` / `sp.var` / `sp.johansen` directly.
- **Estimation tables.** `esttab` / `eststo` / `outreg2` are not translated; use
  `sp.regtable` on the fitted results.
- **Dropped qualifiers are surfaced, not lost.** A Stata `if`/`in` qualifier
  comes back in `unapplied_sample` and an unrecognized option in
  `untranslated_options`, each with a note.
- **Macros and loops.** One command is translated at a time, so `$global` /
  `` `local' `` references and `foreach` bodies must be expanded first.

For the hand-written equivalence reference, see also
[Migrating from R to StatsPAI](migration-from-r.md).
