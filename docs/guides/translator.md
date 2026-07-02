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
# → {'tool': 'feols',
#    'python_code': "sp.feols('y ~ x | id + year', data=df, cluster='id')",
#    'notes': [], 'ok': True}

sp.from_r("feols(y ~ x | id, data = df)")
# → {'tool': 'feols', 'python_code': "sp.feols('y ~ x | id', data=df)", ...}
```

Each call returns one ready-to-run `python_code` string plus a `notes` list. The
design is deliberate:

- **Hand-curated, never a guess.** `vce(cluster id)` and `cluster(id)` mean
  different things in different Stata commands, so every command is mapped by
  hand to preserve its exact semantics.
- **No silent option loss.** When an option has no clean StatsPAI equivalent (a
  dropped `if`/`in` qualifier, an untranslated `vcov=`), it is surfaced in
  `notes` — never quietly discarded.
- **One command per call.** Multi-command `.do` files and multi-line R scripts
  must be split by the caller first.
- **Failure is non-fatal.** An unrecognized command returns
  `{"ok": False, "error": ..., "suggestions": [...]}` rather than raising.

## What's covered — and how to check

The coverage is **introspectable**, so it can never drift from what the
translators actually do:

```python
cov = sp.translation_coverage()
cov["summary"]      # {'n_stata_commands': 38, 'n_r_functions': 11, ...}
cov["stata"]        # [{'command': 'reghdfe', 'targets': ['sp.feols'], ...}, ...]
cov["limitations"]  # the documented gaps (see below)

print(sp.translation_coverage(fmt="markdown"))   # a ready-to-read table
```

Flagship Stata mappings (run `sp.translation_coverage()` for the authoritative,
always-current list):

| Stata | → StatsPAI |
| --- | --- |
| `regress` / `reg` | `sp.regress` |
| `reghdfe`, `xtreg`, `ivreghdfe` | `sp.feols` |
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
  `sp.arima` / `sp.var` / `sp.vecm` directly.
- **Estimation tables.** `esttab` / `eststo` / `outreg2` are not translated; use
  `sp.regtable` on the fitted results.
- **Dropped qualifiers are surfaced, not lost.** A Stata `if`/`in` qualifier or
  an unrecognized option appears in the per-command `notes`.

For the hand-written equivalence reference, see also
[Migrating from R to StatsPAI](migration-from-r.md).
