# Migrating from PyStataR to StatsPAI

`PyStataR` is deprecated. All of its functionality is now available in
[StatsPAI](https://github.com/brycewang-stanford/StatsPAI), under a
unified `sp.*` namespace.

```bash
pip install statspai
```

```python
import statspai as sp
```

## API mapping

| PyStataR | StatsPAI |
|---|---|
| `pdtab.tab1(df, 'x')` / `tab2(df, 'x', 'y')` | `sp.tab(df, 'x')` / `sp.tab(df, 'x', 'y')` |
| `pywinsor2.winsor2(df, ['x'], cuts=(1,99))` | `sp.winsor(df, ['x'], cuts=(1,99))` |
| `pywinsor2.outlier_indicator(df, ['x'])` | `sp.outlier_indicator(df, ['x'])` |
| `pyoutreg.outreg(models, 'out.xlsx')` | `sp.outreg2(models, filename='out.xlsx')` |
| `pyegen.rowmean(df, ['x1','x2'])` | `sp.rowmean(df, ['x1','x2'])` |
| `pyegen.rowtotal(df, ['x1','x2'])` | `sp.rowtotal(df, ['x1','x2'])` |
| `pyegen.rowmax/rowmin(df, [...])` | `sp.rowmax(df, [...])` / `sp.rowmin(df, [...])` |
| `pyegen.rowsd(df, [...])` | `sp.rowsd(df, [...])` |
| `pyegen.rownonmiss(df, [...])` | `sp.rowcount(df, [...])` |
| `pyegen.rank(df, 'x', by='g')` | `sp.rank(df, 'x', by='g')` |

## Why migrate

- **One package, one namespace.** `sp.*` covers everything PyStataR did,
  plus DID, RD, synthetic control, IV, matching, DML, causal forest,
  meta-learners, and more.
- **Actively maintained.** PyStataR is frozen; new features land only in
  StatsPAI.
- **Cleaner naming.** No "Stata" in the name — StatsPAI is Python-native.

## Questions

Open an issue on
[StatsPAI/issues](https://github.com/brycewang-stanford/StatsPAI/issues).
